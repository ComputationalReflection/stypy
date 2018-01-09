
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Module containing non-deprecated functions borrowed from Numeric.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: import types
7: import warnings
8: 
9: import numpy as np
10: from .. import VisibleDeprecationWarning
11: from . import multiarray as mu
12: from . import umath as um
13: from . import numerictypes as nt
14: from .numeric import asarray, array, asanyarray, concatenate
15: from . import _methods
16: 
17: 
18: _dt_ = nt.sctype2char
19: 
20: 
21: # functions that are methods
22: __all__ = [
23:     'alen', 'all', 'alltrue', 'amax', 'amin', 'any', 'argmax',
24:     'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip',
25:     'compress', 'cumprod', 'cumproduct', 'cumsum', 'diagonal', 'mean',
26:     'ndim', 'nonzero', 'partition', 'prod', 'product', 'ptp', 'put',
27:     'rank', 'ravel', 'repeat', 'reshape', 'resize', 'round_',
28:     'searchsorted', 'shape', 'size', 'sometrue', 'sort', 'squeeze',
29:     'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var',
30:     ]
31: 
32: 
33: try:
34:     _gentype = types.GeneratorType
35: except AttributeError:
36:     _gentype = type(None)
37: 
38: # save away Python sum
39: _sum_ = sum
40: 
41: 
42: # functions that are now methods
43: def _wrapit(obj, method, *args, **kwds):
44:     try:
45:         wrap = obj.__array_wrap__
46:     except AttributeError:
47:         wrap = None
48:     result = getattr(asarray(obj), method)(*args, **kwds)
49:     if wrap:
50:         if not isinstance(result, mu.ndarray):
51:             result = asarray(result)
52:         result = wrap(result)
53:     return result
54: 
55: 
56: def take(a, indices, axis=None, out=None, mode='raise'):
57:     '''
58:     Take elements from an array along an axis.
59: 
60:     This function does the same thing as "fancy" indexing (indexing arrays
61:     using arrays); however, it can be easier to use if you need elements
62:     along a given axis.
63: 
64:     Parameters
65:     ----------
66:     a : array_like
67:         The source array.
68:     indices : array_like
69:         The indices of the values to extract.
70: 
71:         .. versionadded:: 1.8.0
72: 
73:         Also allow scalars for indices.
74:     axis : int, optional
75:         The axis over which to select values. By default, the flattened
76:         input array is used.
77:     out : ndarray, optional
78:         If provided, the result will be placed in this array. It should
79:         be of the appropriate shape and dtype.
80:     mode : {'raise', 'wrap', 'clip'}, optional
81:         Specifies how out-of-bounds indices will behave.
82: 
83:         * 'raise' -- raise an error (default)
84:         * 'wrap' -- wrap around
85:         * 'clip' -- clip to the range
86: 
87:         'clip' mode means that all indices that are too large are replaced
88:         by the index that addresses the last element along that axis. Note
89:         that this disables indexing with negative numbers.
90: 
91:     Returns
92:     -------
93:     subarray : ndarray
94:         The returned array has the same type as `a`.
95: 
96:     See Also
97:     --------
98:     compress : Take elements using a boolean mask
99:     ndarray.take : equivalent method
100: 
101:     Examples
102:     --------
103:     >>> a = [4, 3, 5, 7, 6, 8]
104:     >>> indices = [0, 1, 4]
105:     >>> np.take(a, indices)
106:     array([4, 3, 6])
107: 
108:     In this example if `a` is an ndarray, "fancy" indexing can be used.
109: 
110:     >>> a = np.array(a)
111:     >>> a[indices]
112:     array([4, 3, 6])
113: 
114:     If `indices` is not one dimensional, the output also has these dimensions.
115: 
116:     >>> np.take(a, [[0, 1], [2, 3]])
117:     array([[4, 3],
118:            [5, 7]])
119:     '''
120:     try:
121:         take = a.take
122:     except AttributeError:
123:         return _wrapit(a, 'take', indices, axis, out, mode)
124:     return take(indices, axis, out, mode)
125: 
126: 
127: # not deprecated --- copy if necessary, view otherwise
128: def reshape(a, newshape, order='C'):
129:     '''
130:     Gives a new shape to an array without changing its data.
131: 
132:     Parameters
133:     ----------
134:     a : array_like
135:         Array to be reshaped.
136:     newshape : int or tuple of ints
137:         The new shape should be compatible with the original shape. If
138:         an integer, then the result will be a 1-D array of that length.
139:         One shape dimension can be -1. In this case, the value is inferred
140:         from the length of the array and remaining dimensions.
141:     order : {'C', 'F', 'A'}, optional
142:         Read the elements of `a` using this index order, and place the elements
143:         into the reshaped array using this index order.  'C' means to
144:         read / write the elements using C-like index order, with the last axis
145:         index changing fastest, back to the first axis index changing slowest.
146:         'F' means to read / write the elements using Fortran-like index order,
147:         with the first index changing fastest, and the last index changing
148:         slowest.
149:         Note that the 'C' and 'F' options take no account of the memory layout
150:         of the underlying array, and only refer to the order of indexing.  'A'
151:         means to read / write the elements in Fortran-like index order if `a`
152:         is Fortran *contiguous* in memory, C-like order otherwise.
153: 
154:     Returns
155:     -------
156:     reshaped_array : ndarray
157:         This will be a new view object if possible; otherwise, it will
158:         be a copy.  Note there is no guarantee of the *memory layout* (C- or
159:         Fortran- contiguous) of the returned array.
160: 
161:     See Also
162:     --------
163:     ndarray.reshape : Equivalent method.
164: 
165:     Notes
166:     -----
167:     It is not always possible to change the shape of an array without
168:     copying the data. If you want an error to be raise if the data is copied,
169:     you should assign the new shape to the shape attribute of the array::
170: 
171:      >>> a = np.zeros((10, 2))
172:      # A transpose make the array non-contiguous
173:      >>> b = a.T
174:      # Taking a view makes it possible to modify the shape without modifying
175:      # the initial object.
176:      >>> c = b.view()
177:      >>> c.shape = (20)
178:      AttributeError: incompatible shape for a non-contiguous array
179: 
180:     The `order` keyword gives the index ordering both for *fetching* the values
181:     from `a`, and then *placing* the values into the output array.
182:     For example, let's say you have an array:
183: 
184:     >>> a = np.arange(6).reshape((3, 2))
185:     >>> a
186:     array([[0, 1],
187:            [2, 3],
188:            [4, 5]])
189: 
190:     You can think of reshaping as first raveling the array (using the given
191:     index order), then inserting the elements from the raveled array into the
192:     new array using the same kind of index ordering as was used for the
193:     raveling.
194: 
195:     >>> np.reshape(a, (2, 3)) # C-like index ordering
196:     array([[0, 1, 2],
197:            [3, 4, 5]])
198:     >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
199:     array([[0, 1, 2],
200:            [3, 4, 5]])
201:     >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
202:     array([[0, 4, 3],
203:            [2, 1, 5]])
204:     >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
205:     array([[0, 4, 3],
206:            [2, 1, 5]])
207: 
208:     Examples
209:     --------
210:     >>> a = np.array([[1,2,3], [4,5,6]])
211:     >>> np.reshape(a, 6)
212:     array([1, 2, 3, 4, 5, 6])
213:     >>> np.reshape(a, 6, order='F')
214:     array([1, 4, 2, 5, 3, 6])
215: 
216:     >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
217:     array([[1, 2],
218:            [3, 4],
219:            [5, 6]])
220:     '''
221:     try:
222:         reshape = a.reshape
223:     except AttributeError:
224:         return _wrapit(a, 'reshape', newshape, order=order)
225:     return reshape(newshape, order=order)
226: 
227: 
228: def choose(a, choices, out=None, mode='raise'):
229:     '''
230:     Construct an array from an index array and a set of arrays to choose from.
231: 
232:     First of all, if confused or uncertain, definitely look at the Examples -
233:     in its full generality, this function is less simple than it might
234:     seem from the following code description (below ndi =
235:     `numpy.lib.index_tricks`):
236: 
237:     ``np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)])``.
238: 
239:     But this omits some subtleties.  Here is a fully general summary:
240: 
241:     Given an "index" array (`a`) of integers and a sequence of `n` arrays
242:     (`choices`), `a` and each choice array are first broadcast, as necessary,
243:     to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
244:     0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
245:     for each `i`.  Then, a new array with shape ``Ba.shape`` is created as
246:     follows:
247: 
248:     * if ``mode=raise`` (the default), then, first of all, each element of
249:       `a` (and thus `Ba`) must be in the range `[0, n-1]`; now, suppose that
250:       `i` (in that range) is the value at the `(j0, j1, ..., jm)` position
251:       in `Ba` - then the value at the same position in the new array is the
252:       value in `Bchoices[i]` at that same position;
253: 
254:     * if ``mode=wrap``, values in `a` (and thus `Ba`) may be any (signed)
255:       integer; modular arithmetic is used to map integers outside the range
256:       `[0, n-1]` back into that range; and then the new array is constructed
257:       as above;
258: 
259:     * if ``mode=clip``, values in `a` (and thus `Ba`) may be any (signed)
260:       integer; negative integers are mapped to 0; values greater than `n-1`
261:       are mapped to `n-1`; and then the new array is constructed as above.
262: 
263:     Parameters
264:     ----------
265:     a : int array
266:         This array must contain integers in `[0, n-1]`, where `n` is the number
267:         of choices, unless ``mode=wrap`` or ``mode=clip``, in which cases any
268:         integers are permissible.
269:     choices : sequence of arrays
270:         Choice arrays. `a` and all of the choices must be broadcastable to the
271:         same shape.  If `choices` is itself an array (not recommended), then
272:         its outermost dimension (i.e., the one corresponding to
273:         ``choices.shape[0]``) is taken as defining the "sequence".
274:     out : array, optional
275:         If provided, the result will be inserted into this array. It should
276:         be of the appropriate shape and dtype.
277:     mode : {'raise' (default), 'wrap', 'clip'}, optional
278:         Specifies how indices outside `[0, n-1]` will be treated:
279: 
280:           * 'raise' : an exception is raised
281:           * 'wrap' : value becomes value mod `n`
282:           * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1
283: 
284:     Returns
285:     -------
286:     merged_array : array
287:         The merged result.
288: 
289:     Raises
290:     ------
291:     ValueError: shape mismatch
292:         If `a` and each choice array are not all broadcastable to the same
293:         shape.
294: 
295:     See Also
296:     --------
297:     ndarray.choose : equivalent method
298: 
299:     Notes
300:     -----
301:     To reduce the chance of misinterpretation, even though the following
302:     "abuse" is nominally supported, `choices` should neither be, nor be
303:     thought of as, a single array, i.e., the outermost sequence-like container
304:     should be either a list or a tuple.
305: 
306:     Examples
307:     --------
308: 
309:     >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
310:     ...   [20, 21, 22, 23], [30, 31, 32, 33]]
311:     >>> np.choose([2, 3, 1, 0], choices
312:     ... # the first element of the result will be the first element of the
313:     ... # third (2+1) "array" in choices, namely, 20; the second element
314:     ... # will be the second element of the fourth (3+1) choice array, i.e.,
315:     ... # 31, etc.
316:     ... )
317:     array([20, 31, 12,  3])
318:     >>> np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
319:     array([20, 31, 12,  3])
320:     >>> # because there are 4 choice arrays
321:     >>> np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
322:     array([20,  1, 12,  3])
323:     >>> # i.e., 0
324: 
325:     A couple examples illustrating how choose broadcasts:
326: 
327:     >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
328:     >>> choices = [-10, 10]
329:     >>> np.choose(a, choices)
330:     array([[ 10, -10,  10],
331:            [-10,  10, -10],
332:            [ 10, -10,  10]])
333: 
334:     >>> # With thanks to Anne Archibald
335:     >>> a = np.array([0, 1]).reshape((2,1,1))
336:     >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))
337:     >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
338:     >>> np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
339:     array([[[ 1,  1,  1,  1,  1],
340:             [ 2,  2,  2,  2,  2],
341:             [ 3,  3,  3,  3,  3]],
342:            [[-1, -2, -3, -4, -5],
343:             [-1, -2, -3, -4, -5],
344:             [-1, -2, -3, -4, -5]]])
345: 
346:     '''
347:     try:
348:         choose = a.choose
349:     except AttributeError:
350:         return _wrapit(a, 'choose', choices, out=out, mode=mode)
351:     return choose(choices, out=out, mode=mode)
352: 
353: 
354: def repeat(a, repeats, axis=None):
355:     '''
356:     Repeat elements of an array.
357: 
358:     Parameters
359:     ----------
360:     a : array_like
361:         Input array.
362:     repeats : int or array of ints
363:         The number of repetitions for each element.  `repeats` is broadcasted
364:         to fit the shape of the given axis.
365:     axis : int, optional
366:         The axis along which to repeat values.  By default, use the
367:         flattened input array, and return a flat output array.
368: 
369:     Returns
370:     -------
371:     repeated_array : ndarray
372:         Output array which has the same shape as `a`, except along
373:         the given axis.
374: 
375:     See Also
376:     --------
377:     tile : Tile an array.
378: 
379:     Examples
380:     --------
381:     >>> x = np.array([[1,2],[3,4]])
382:     >>> np.repeat(x, 2)
383:     array([1, 1, 2, 2, 3, 3, 4, 4])
384:     >>> np.repeat(x, 3, axis=1)
385:     array([[1, 1, 1, 2, 2, 2],
386:            [3, 3, 3, 4, 4, 4]])
387:     >>> np.repeat(x, [1, 2], axis=0)
388:     array([[1, 2],
389:            [3, 4],
390:            [3, 4]])
391: 
392:     '''
393:     try:
394:         repeat = a.repeat
395:     except AttributeError:
396:         return _wrapit(a, 'repeat', repeats, axis)
397:     return repeat(repeats, axis)
398: 
399: 
400: def put(a, ind, v, mode='raise'):
401:     '''
402:     Replaces specified elements of an array with given values.
403: 
404:     The indexing works on the flattened target array. `put` is roughly
405:     equivalent to:
406: 
407:     ::
408: 
409:         a.flat[ind] = v
410: 
411:     Parameters
412:     ----------
413:     a : ndarray
414:         Target array.
415:     ind : array_like
416:         Target indices, interpreted as integers.
417:     v : array_like
418:         Values to place in `a` at target indices. If `v` is shorter than
419:         `ind` it will be repeated as necessary.
420:     mode : {'raise', 'wrap', 'clip'}, optional
421:         Specifies how out-of-bounds indices will behave.
422: 
423:         * 'raise' -- raise an error (default)
424:         * 'wrap' -- wrap around
425:         * 'clip' -- clip to the range
426: 
427:         'clip' mode means that all indices that are too large are replaced
428:         by the index that addresses the last element along that axis. Note
429:         that this disables indexing with negative numbers.
430: 
431:     See Also
432:     --------
433:     putmask, place
434: 
435:     Examples
436:     --------
437:     >>> a = np.arange(5)
438:     >>> np.put(a, [0, 2], [-44, -55])
439:     >>> a
440:     array([-44,   1, -55,   3,   4])
441: 
442:     >>> a = np.arange(5)
443:     >>> np.put(a, 22, -5, mode='clip')
444:     >>> a
445:     array([ 0,  1,  2,  3, -5])
446: 
447:     '''
448:     try:
449:         put = a.put
450:     except AttributeError:
451:         raise TypeError("argument 1 must be numpy.ndarray, "
452:                         "not {name}".format(name=type(a).__name__))
453: 
454:     return put(ind, v, mode)
455: 
456: 
457: def swapaxes(a, axis1, axis2):
458:     '''
459:     Interchange two axes of an array.
460: 
461:     Parameters
462:     ----------
463:     a : array_like
464:         Input array.
465:     axis1 : int
466:         First axis.
467:     axis2 : int
468:         Second axis.
469: 
470:     Returns
471:     -------
472:     a_swapped : ndarray
473:         For Numpy >= 1.10, if `a` is an ndarray, then a view of `a` is
474:         returned; otherwise a new array is created. For earlier Numpy
475:         versions a view of `a` is returned only if the order of the
476:         axes is changed, otherwise the input array is returned.
477: 
478:     Examples
479:     --------
480:     >>> x = np.array([[1,2,3]])
481:     >>> np.swapaxes(x,0,1)
482:     array([[1],
483:            [2],
484:            [3]])
485: 
486:     >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
487:     >>> x
488:     array([[[0, 1],
489:             [2, 3]],
490:            [[4, 5],
491:             [6, 7]]])
492: 
493:     >>> np.swapaxes(x,0,2)
494:     array([[[0, 4],
495:             [2, 6]],
496:            [[1, 5],
497:             [3, 7]]])
498: 
499:     '''
500:     try:
501:         swapaxes = a.swapaxes
502:     except AttributeError:
503:         return _wrapit(a, 'swapaxes', axis1, axis2)
504:     return swapaxes(axis1, axis2)
505: 
506: 
507: def transpose(a, axes=None):
508:     '''
509:     Permute the dimensions of an array.
510: 
511:     Parameters
512:     ----------
513:     a : array_like
514:         Input array.
515:     axes : list of ints, optional
516:         By default, reverse the dimensions, otherwise permute the axes
517:         according to the values given.
518: 
519:     Returns
520:     -------
521:     p : ndarray
522:         `a` with its axes permuted.  A view is returned whenever
523:         possible.
524: 
525:     See Also
526:     --------
527:     moveaxis
528:     argsort
529: 
530:     Notes
531:     -----
532:     Use `transpose(a, argsort(axes))` to invert the transposition of tensors
533:     when using the `axes` keyword argument.
534: 
535:     Transposing a 1-D array returns an unchanged view of the original array.
536: 
537:     Examples
538:     --------
539:     >>> x = np.arange(4).reshape((2,2))
540:     >>> x
541:     array([[0, 1],
542:            [2, 3]])
543: 
544:     >>> np.transpose(x)
545:     array([[0, 2],
546:            [1, 3]])
547: 
548:     >>> x = np.ones((1, 2, 3))
549:     >>> np.transpose(x, (1, 0, 2)).shape
550:     (2, 1, 3)
551: 
552:     '''
553:     try:
554:         transpose = a.transpose
555:     except AttributeError:
556:         return _wrapit(a, 'transpose', axes)
557:     return transpose(axes)
558: 
559: 
560: def partition(a, kth, axis=-1, kind='introselect', order=None):
561:     '''
562:     Return a partitioned copy of an array.
563: 
564:     Creates a copy of the array with its elements rearranged in such a way that
565:     the value of the element in kth position is in the position it would be in
566:     a sorted array. All elements smaller than the kth element are moved before
567:     this element and all equal or greater are moved behind it. The ordering of
568:     the elements in the two partitions is undefined.
569: 
570:     .. versionadded:: 1.8.0
571: 
572:     Parameters
573:     ----------
574:     a : array_like
575:         Array to be sorted.
576:     kth : int or sequence of ints
577:         Element index to partition by. The kth value of the element will be in
578:         its final sorted position and all smaller elements will be moved before
579:         it and all equal or greater elements behind it.
580:         The order all elements in the partitions is undefined.
581:         If provided with a sequence of kth it will partition all elements
582:         indexed by kth  of them into their sorted position at once.
583:     axis : int or None, optional
584:         Axis along which to sort. If None, the array is flattened before
585:         sorting. The default is -1, which sorts along the last axis.
586:     kind : {'introselect'}, optional
587:         Selection algorithm. Default is 'introselect'.
588:     order : str or list of str, optional
589:         When `a` is an array with fields defined, this argument specifies
590:         which fields to compare first, second, etc.  A single field can
591:         be specified as a string.  Not all fields need be specified, but
592:         unspecified fields will still be used, in the order in which they
593:         come up in the dtype, to break ties.
594: 
595:     Returns
596:     -------
597:     partitioned_array : ndarray
598:         Array of the same type and shape as `a`.
599: 
600:     See Also
601:     --------
602:     ndarray.partition : Method to sort an array in-place.
603:     argpartition : Indirect partition.
604:     sort : Full sorting
605: 
606:     Notes
607:     -----
608:     The various selection algorithms are characterized by their average speed,
609:     worst case performance, work space size, and whether they are stable. A
610:     stable sort keeps items with the same key in the same relative order. The
611:     available algorithms have the following properties:
612: 
613:     ================= ======= ============= ============ =======
614:        kind            speed   worst case    work space  stable
615:     ================= ======= ============= ============ =======
616:     'introselect'        1        O(n)           0         no
617:     ================= ======= ============= ============ =======
618: 
619:     All the partition algorithms make temporary copies of the data when
620:     partitioning along any but the last axis.  Consequently, partitioning
621:     along the last axis is faster and uses less space than partitioning
622:     along any other axis.
623: 
624:     The sort order for complex numbers is lexicographic. If both the real
625:     and imaginary parts are non-nan then the order is determined by the
626:     real parts except when they are equal, in which case the order is
627:     determined by the imaginary parts.
628: 
629:     Examples
630:     --------
631:     >>> a = np.array([3, 4, 2, 1])
632:     >>> np.partition(a, 3)
633:     array([2, 1, 3, 4])
634: 
635:     >>> np.partition(a, (1, 3))
636:     array([1, 2, 3, 4])
637: 
638:     '''
639:     if axis is None:
640:         a = asanyarray(a).flatten()
641:         axis = 0
642:     else:
643:         a = asanyarray(a).copy(order="K")
644:     a.partition(kth, axis=axis, kind=kind, order=order)
645:     return a
646: 
647: 
648: def argpartition(a, kth, axis=-1, kind='introselect', order=None):
649:     '''
650:     Perform an indirect partition along the given axis using the algorithm
651:     specified by the `kind` keyword. It returns an array of indices of the
652:     same shape as `a` that index data along the given axis in partitioned
653:     order.
654: 
655:     .. versionadded:: 1.8.0
656: 
657:     Parameters
658:     ----------
659:     a : array_like
660:         Array to sort.
661:     kth : int or sequence of ints
662:         Element index to partition by. The kth element will be in its final
663:         sorted position and all smaller elements will be moved before it and
664:         all larger elements behind it.
665:         The order all elements in the partitions is undefined.
666:         If provided with a sequence of kth it will partition all of them into
667:         their sorted position at once.
668:     axis : int or None, optional
669:         Axis along which to sort.  The default is -1 (the last axis). If None,
670:         the flattened array is used.
671:     kind : {'introselect'}, optional
672:         Selection algorithm. Default is 'introselect'
673:     order : str or list of str, optional
674:         When `a` is an array with fields defined, this argument specifies
675:         which fields to compare first, second, etc.  A single field can
676:         be specified as a string, and not all fields need be specified,
677:         but unspecified fields will still be used, in the order in which
678:         they come up in the dtype, to break ties.
679: 
680:     Returns
681:     -------
682:     index_array : ndarray, int
683:         Array of indices that partition `a` along the specified axis.
684:         In other words, ``a[index_array]`` yields a sorted `a`.
685: 
686:     See Also
687:     --------
688:     partition : Describes partition algorithms used.
689:     ndarray.partition : Inplace partition.
690:     argsort : Full indirect sort
691: 
692:     Notes
693:     -----
694:     See `partition` for notes on the different selection algorithms.
695: 
696:     Examples
697:     --------
698:     One dimensional array:
699: 
700:     >>> x = np.array([3, 4, 2, 1])
701:     >>> x[np.argpartition(x, 3)]
702:     array([2, 1, 3, 4])
703:     >>> x[np.argpartition(x, (1, 3))]
704:     array([1, 2, 3, 4])
705: 
706:     >>> x = [3, 4, 2, 1]
707:     >>> np.array(x)[np.argpartition(x, 3)]
708:     array([2, 1, 3, 4])
709: 
710:     '''
711:     try:
712:         argpartition = a.argpartition
713:     except AttributeError:
714:         return _wrapit(a, 'argpartition',kth, axis, kind, order)
715:     return argpartition(kth, axis, kind=kind, order=order)
716: 
717: 
718: def sort(a, axis=-1, kind='quicksort', order=None):
719:     '''
720:     Return a sorted copy of an array.
721: 
722:     Parameters
723:     ----------
724:     a : array_like
725:         Array to be sorted.
726:     axis : int or None, optional
727:         Axis along which to sort. If None, the array is flattened before
728:         sorting. The default is -1, which sorts along the last axis.
729:     kind : {'quicksort', 'mergesort', 'heapsort'}, optional
730:         Sorting algorithm. Default is 'quicksort'.
731:     order : str or list of str, optional
732:         When `a` is an array with fields defined, this argument specifies
733:         which fields to compare first, second, etc.  A single field can
734:         be specified as a string, and not all fields need be specified,
735:         but unspecified fields will still be used, in the order in which
736:         they come up in the dtype, to break ties.
737: 
738:     Returns
739:     -------
740:     sorted_array : ndarray
741:         Array of the same type and shape as `a`.
742: 
743:     See Also
744:     --------
745:     ndarray.sort : Method to sort an array in-place.
746:     argsort : Indirect sort.
747:     lexsort : Indirect stable sort on multiple keys.
748:     searchsorted : Find elements in a sorted array.
749:     partition : Partial sort.
750: 
751:     Notes
752:     -----
753:     The various sorting algorithms are characterized by their average speed,
754:     worst case performance, work space size, and whether they are stable. A
755:     stable sort keeps items with the same key in the same relative
756:     order. The three available algorithms have the following
757:     properties:
758: 
759:     =========== ======= ============= ============ =======
760:        kind      speed   worst case    work space  stable
761:     =========== ======= ============= ============ =======
762:     'quicksort'    1     O(n^2)            0          no
763:     'mergesort'    2     O(n*log(n))      ~n/2        yes
764:     'heapsort'     3     O(n*log(n))       0          no
765:     =========== ======= ============= ============ =======
766: 
767:     All the sort algorithms make temporary copies of the data when
768:     sorting along any but the last axis.  Consequently, sorting along
769:     the last axis is faster and uses less space than sorting along
770:     any other axis.
771: 
772:     The sort order for complex numbers is lexicographic. If both the real
773:     and imaginary parts are non-nan then the order is determined by the
774:     real parts except when they are equal, in which case the order is
775:     determined by the imaginary parts.
776: 
777:     Previous to numpy 1.4.0 sorting real and complex arrays containing nan
778:     values led to undefined behaviour. In numpy versions >= 1.4.0 nan
779:     values are sorted to the end. The extended sort order is:
780: 
781:       * Real: [R, nan]
782:       * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]
783: 
784:     where R is a non-nan real value. Complex values with the same nan
785:     placements are sorted according to the non-nan part if it exists.
786:     Non-nan values are sorted as before.
787: 
788:     Examples
789:     --------
790:     >>> a = np.array([[1,4],[3,1]])
791:     >>> np.sort(a)                # sort along the last axis
792:     array([[1, 4],
793:            [1, 3]])
794:     >>> np.sort(a, axis=None)     # sort the flattened array
795:     array([1, 1, 3, 4])
796:     >>> np.sort(a, axis=0)        # sort along the first axis
797:     array([[1, 1],
798:            [3, 4]])
799: 
800:     Use the `order` keyword to specify a field to use when sorting a
801:     structured array:
802: 
803:     >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
804:     >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
805:     ...           ('Galahad', 1.7, 38)]
806:     >>> a = np.array(values, dtype=dtype)       # create a structured array
807:     >>> np.sort(a, order='height')                        # doctest: +SKIP
808:     array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
809:            ('Lancelot', 1.8999999999999999, 38)],
810:           dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
811: 
812:     Sort by age, then height if ages are equal:
813: 
814:     >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
815:     array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
816:            ('Arthur', 1.8, 41)],
817:           dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
818: 
819:     '''
820:     if axis is None:
821:         a = asanyarray(a).flatten()
822:         axis = 0
823:     else:
824:         a = asanyarray(a).copy(order="K")
825:     a.sort(axis, kind, order)
826:     return a
827: 
828: 
829: def argsort(a, axis=-1, kind='quicksort', order=None):
830:     '''
831:     Returns the indices that would sort an array.
832: 
833:     Perform an indirect sort along the given axis using the algorithm specified
834:     by the `kind` keyword. It returns an array of indices of the same shape as
835:     `a` that index data along the given axis in sorted order.
836: 
837:     Parameters
838:     ----------
839:     a : array_like
840:         Array to sort.
841:     axis : int or None, optional
842:         Axis along which to sort.  The default is -1 (the last axis). If None,
843:         the flattened array is used.
844:     kind : {'quicksort', 'mergesort', 'heapsort'}, optional
845:         Sorting algorithm.
846:     order : str or list of str, optional
847:         When `a` is an array with fields defined, this argument specifies
848:         which fields to compare first, second, etc.  A single field can
849:         be specified as a string, and not all fields need be specified,
850:         but unspecified fields will still be used, in the order in which
851:         they come up in the dtype, to break ties.
852: 
853:     Returns
854:     -------
855:     index_array : ndarray, int
856:         Array of indices that sort `a` along the specified axis.
857:         If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
858: 
859:     See Also
860:     --------
861:     sort : Describes sorting algorithms used.
862:     lexsort : Indirect stable sort with multiple keys.
863:     ndarray.sort : Inplace sort.
864:     argpartition : Indirect partial sort.
865: 
866:     Notes
867:     -----
868:     See `sort` for notes on the different sorting algorithms.
869: 
870:     As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
871:     nan values. The enhanced sort order is documented in `sort`.
872: 
873:     Examples
874:     --------
875:     One dimensional array:
876: 
877:     >>> x = np.array([3, 1, 2])
878:     >>> np.argsort(x)
879:     array([1, 2, 0])
880: 
881:     Two-dimensional array:
882: 
883:     >>> x = np.array([[0, 3], [2, 2]])
884:     >>> x
885:     array([[0, 3],
886:            [2, 2]])
887: 
888:     >>> np.argsort(x, axis=0)
889:     array([[0, 1],
890:            [1, 0]])
891: 
892:     >>> np.argsort(x, axis=1)
893:     array([[0, 1],
894:            [0, 1]])
895: 
896:     Sorting with keys:
897: 
898:     >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
899:     >>> x
900:     array([(1, 0), (0, 1)],
901:           dtype=[('x', '<i4'), ('y', '<i4')])
902: 
903:     >>> np.argsort(x, order=('x','y'))
904:     array([1, 0])
905: 
906:     >>> np.argsort(x, order=('y','x'))
907:     array([0, 1])
908: 
909:     '''
910:     try:
911:         argsort = a.argsort
912:     except AttributeError:
913:         return _wrapit(a, 'argsort', axis, kind, order)
914:     return argsort(axis, kind, order)
915: 
916: 
917: def argmax(a, axis=None, out=None):
918:     '''
919:     Returns the indices of the maximum values along an axis.
920: 
921:     Parameters
922:     ----------
923:     a : array_like
924:         Input array.
925:     axis : int, optional
926:         By default, the index is into the flattened array, otherwise
927:         along the specified axis.
928:     out : array, optional
929:         If provided, the result will be inserted into this array. It should
930:         be of the appropriate shape and dtype.
931: 
932:     Returns
933:     -------
934:     index_array : ndarray of ints
935:         Array of indices into the array. It has the same shape as `a.shape`
936:         with the dimension along `axis` removed.
937: 
938:     See Also
939:     --------
940:     ndarray.argmax, argmin
941:     amax : The maximum value along a given axis.
942:     unravel_index : Convert a flat index into an index tuple.
943: 
944:     Notes
945:     -----
946:     In case of multiple occurrences of the maximum values, the indices
947:     corresponding to the first occurrence are returned.
948: 
949:     Examples
950:     --------
951:     >>> a = np.arange(6).reshape(2,3)
952:     >>> a
953:     array([[0, 1, 2],
954:            [3, 4, 5]])
955:     >>> np.argmax(a)
956:     5
957:     >>> np.argmax(a, axis=0)
958:     array([1, 1, 1])
959:     >>> np.argmax(a, axis=1)
960:     array([2, 2])
961: 
962:     >>> b = np.arange(6)
963:     >>> b[1] = 5
964:     >>> b
965:     array([0, 5, 2, 3, 4, 5])
966:     >>> np.argmax(b) # Only the first occurrence is returned.
967:     1
968: 
969:     '''
970:     try:
971:         argmax = a.argmax
972:     except AttributeError:
973:         return _wrapit(a, 'argmax', axis, out)
974:     return argmax(axis, out)
975: 
976: 
977: def argmin(a, axis=None, out=None):
978:     '''
979:     Returns the indices of the minimum values along an axis.
980: 
981:     Parameters
982:     ----------
983:     a : array_like
984:         Input array.
985:     axis : int, optional
986:         By default, the index is into the flattened array, otherwise
987:         along the specified axis.
988:     out : array, optional
989:         If provided, the result will be inserted into this array. It should
990:         be of the appropriate shape and dtype.
991: 
992:     Returns
993:     -------
994:     index_array : ndarray of ints
995:         Array of indices into the array. It has the same shape as `a.shape`
996:         with the dimension along `axis` removed.
997: 
998:     See Also
999:     --------
1000:     ndarray.argmin, argmax
1001:     amin : The minimum value along a given axis.
1002:     unravel_index : Convert a flat index into an index tuple.
1003: 
1004:     Notes
1005:     -----
1006:     In case of multiple occurrences of the minimum values, the indices
1007:     corresponding to the first occurrence are returned.
1008: 
1009:     Examples
1010:     --------
1011:     >>> a = np.arange(6).reshape(2,3)
1012:     >>> a
1013:     array([[0, 1, 2],
1014:            [3, 4, 5]])
1015:     >>> np.argmin(a)
1016:     0
1017:     >>> np.argmin(a, axis=0)
1018:     array([0, 0, 0])
1019:     >>> np.argmin(a, axis=1)
1020:     array([0, 0])
1021: 
1022:     >>> b = np.arange(6)
1023:     >>> b[4] = 0
1024:     >>> b
1025:     array([0, 1, 2, 3, 0, 5])
1026:     >>> np.argmin(b) # Only the first occurrence is returned.
1027:     0
1028: 
1029:     '''
1030:     try:
1031:         argmin = a.argmin
1032:     except AttributeError:
1033:         return _wrapit(a, 'argmin', axis, out)
1034:     return argmin(axis, out)
1035: 
1036: 
1037: def searchsorted(a, v, side='left', sorter=None):
1038:     '''
1039:     Find indices where elements should be inserted to maintain order.
1040: 
1041:     Find the indices into a sorted array `a` such that, if the
1042:     corresponding elements in `v` were inserted before the indices, the
1043:     order of `a` would be preserved.
1044: 
1045:     Parameters
1046:     ----------
1047:     a : 1-D array_like
1048:         Input array. If `sorter` is None, then it must be sorted in
1049:         ascending order, otherwise `sorter` must be an array of indices
1050:         that sort it.
1051:     v : array_like
1052:         Values to insert into `a`.
1053:     side : {'left', 'right'}, optional
1054:         If 'left', the index of the first suitable location found is given.
1055:         If 'right', return the last such index.  If there is no suitable
1056:         index, return either 0 or N (where N is the length of `a`).
1057:     sorter : 1-D array_like, optional
1058:         Optional array of integer indices that sort array a into ascending
1059:         order. They are typically the result of argsort.
1060: 
1061:         .. versionadded:: 1.7.0
1062: 
1063:     Returns
1064:     -------
1065:     indices : array of ints
1066:         Array of insertion points with the same shape as `v`.
1067: 
1068:     See Also
1069:     --------
1070:     sort : Return a sorted copy of an array.
1071:     histogram : Produce histogram from 1-D data.
1072: 
1073:     Notes
1074:     -----
1075:     Binary search is used to find the required insertion points.
1076: 
1077:     As of Numpy 1.4.0 `searchsorted` works with real/complex arrays containing
1078:     `nan` values. The enhanced sort order is documented in `sort`.
1079: 
1080:     Examples
1081:     --------
1082:     >>> np.searchsorted([1,2,3,4,5], 3)
1083:     2
1084:     >>> np.searchsorted([1,2,3,4,5], 3, side='right')
1085:     3
1086:     >>> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
1087:     array([0, 5, 1, 2])
1088: 
1089:     '''
1090:     try:
1091:         searchsorted = a.searchsorted
1092:     except AttributeError:
1093:         return _wrapit(a, 'searchsorted', v, side, sorter)
1094:     return searchsorted(v, side, sorter)
1095: 
1096: 
1097: def resize(a, new_shape):
1098:     '''
1099:     Return a new array with the specified shape.
1100: 
1101:     If the new array is larger than the original array, then the new
1102:     array is filled with repeated copies of `a`.  Note that this behavior
1103:     is different from a.resize(new_shape) which fills with zeros instead
1104:     of repeated copies of `a`.
1105: 
1106:     Parameters
1107:     ----------
1108:     a : array_like
1109:         Array to be resized.
1110: 
1111:     new_shape : int or tuple of int
1112:         Shape of resized array.
1113: 
1114:     Returns
1115:     -------
1116:     reshaped_array : ndarray
1117:         The new array is formed from the data in the old array, repeated
1118:         if necessary to fill out the required number of elements.  The
1119:         data are repeated in the order that they are stored in memory.
1120: 
1121:     See Also
1122:     --------
1123:     ndarray.resize : resize an array in-place.
1124: 
1125:     Examples
1126:     --------
1127:     >>> a=np.array([[0,1],[2,3]])
1128:     >>> np.resize(a,(2,3))
1129:     array([[0, 1, 2],
1130:            [3, 0, 1]])
1131:     >>> np.resize(a,(1,4))
1132:     array([[0, 1, 2, 3]])
1133:     >>> np.resize(a,(2,4))
1134:     array([[0, 1, 2, 3],
1135:            [0, 1, 2, 3]])
1136: 
1137:     '''
1138:     if isinstance(new_shape, (int, nt.integer)):
1139:         new_shape = (new_shape,)
1140:     a = ravel(a)
1141:     Na = len(a)
1142:     if not Na:
1143:         return mu.zeros(new_shape, a.dtype)
1144:     total_size = um.multiply.reduce(new_shape)
1145:     n_copies = int(total_size / Na)
1146:     extra = total_size % Na
1147: 
1148:     if total_size == 0:
1149:         return a[:0]
1150: 
1151:     if extra != 0:
1152:         n_copies = n_copies+1
1153:         extra = Na-extra
1154: 
1155:     a = concatenate((a,)*n_copies)
1156:     if extra > 0:
1157:         a = a[:-extra]
1158: 
1159:     return reshape(a, new_shape)
1160: 
1161: 
1162: def squeeze(a, axis=None):
1163:     '''
1164:     Remove single-dimensional entries from the shape of an array.
1165: 
1166:     Parameters
1167:     ----------
1168:     a : array_like
1169:         Input data.
1170:     axis : None or int or tuple of ints, optional
1171:         .. versionadded:: 1.7.0
1172: 
1173:         Selects a subset of the single-dimensional entries in the
1174:         shape. If an axis is selected with shape entry greater than
1175:         one, an error is raised.
1176: 
1177:     Returns
1178:     -------
1179:     squeezed : ndarray
1180:         The input array, but with all or a subset of the
1181:         dimensions of length 1 removed. This is always `a` itself
1182:         or a view into `a`.
1183: 
1184:     Examples
1185:     --------
1186:     >>> x = np.array([[[0], [1], [2]]])
1187:     >>> x.shape
1188:     (1, 3, 1)
1189:     >>> np.squeeze(x).shape
1190:     (3,)
1191:     >>> np.squeeze(x, axis=(2,)).shape
1192:     (1, 3)
1193: 
1194:     '''
1195:     try:
1196:         squeeze = a.squeeze
1197:     except AttributeError:
1198:         return _wrapit(a, 'squeeze')
1199:     try:
1200:         # First try to use the new axis= parameter
1201:         return squeeze(axis=axis)
1202:     except TypeError:
1203:         # For backwards compatibility
1204:         return squeeze()
1205: 
1206: 
1207: def diagonal(a, offset=0, axis1=0, axis2=1):
1208:     '''
1209:     Return specified diagonals.
1210: 
1211:     If `a` is 2-D, returns the diagonal of `a` with the given offset,
1212:     i.e., the collection of elements of the form ``a[i, i+offset]``.  If
1213:     `a` has more than two dimensions, then the axes specified by `axis1`
1214:     and `axis2` are used to determine the 2-D sub-array whose diagonal is
1215:     returned.  The shape of the resulting array can be determined by
1216:     removing `axis1` and `axis2` and appending an index to the right equal
1217:     to the size of the resulting diagonals.
1218: 
1219:     In versions of NumPy prior to 1.7, this function always returned a new,
1220:     independent array containing a copy of the values in the diagonal.
1221: 
1222:     In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,
1223:     but depending on this fact is deprecated. Writing to the resulting
1224:     array continues to work as it used to, but a FutureWarning is issued.
1225: 
1226:     Starting in NumPy 1.9 it returns a read-only view on the original array.
1227:     Attempting to write to the resulting array will produce an error.
1228: 
1229:     In some future release, it will return a read/write view and writing to
1230:     the returned array will alter your original array.  The returned array
1231:     will have the same type as the input array.
1232: 
1233:     If you don't write to the array returned by this function, then you can
1234:     just ignore all of the above.
1235: 
1236:     If you depend on the current behavior, then we suggest copying the
1237:     returned array explicitly, i.e., use ``np.diagonal(a).copy()`` instead
1238:     of just ``np.diagonal(a)``. This will work with both past and future
1239:     versions of NumPy.
1240: 
1241:     Parameters
1242:     ----------
1243:     a : array_like
1244:         Array from which the diagonals are taken.
1245:     offset : int, optional
1246:         Offset of the diagonal from the main diagonal.  Can be positive or
1247:         negative.  Defaults to main diagonal (0).
1248:     axis1 : int, optional
1249:         Axis to be used as the first axis of the 2-D sub-arrays from which
1250:         the diagonals should be taken.  Defaults to first axis (0).
1251:     axis2 : int, optional
1252:         Axis to be used as the second axis of the 2-D sub-arrays from
1253:         which the diagonals should be taken. Defaults to second axis (1).
1254: 
1255:     Returns
1256:     -------
1257:     array_of_diagonals : ndarray
1258:         If `a` is 2-D and not a matrix, a 1-D array of the same type as `a`
1259:         containing the diagonal is returned. If `a` is a matrix, a 1-D
1260:         array containing the diagonal is returned in order to maintain
1261:         backward compatibility.  If the dimension of `a` is greater than
1262:         two, then an array of diagonals is returned, "packed" from
1263:         left-most dimension to right-most (e.g., if `a` is 3-D, then the
1264:         diagonals are "packed" along rows).
1265: 
1266:     Raises
1267:     ------
1268:     ValueError
1269:         If the dimension of `a` is less than 2.
1270: 
1271:     See Also
1272:     --------
1273:     diag : MATLAB work-a-like for 1-D and 2-D arrays.
1274:     diagflat : Create diagonal arrays.
1275:     trace : Sum along diagonals.
1276: 
1277:     Examples
1278:     --------
1279:     >>> a = np.arange(4).reshape(2,2)
1280:     >>> a
1281:     array([[0, 1],
1282:            [2, 3]])
1283:     >>> a.diagonal()
1284:     array([0, 3])
1285:     >>> a.diagonal(1)
1286:     array([1])
1287: 
1288:     A 3-D example:
1289: 
1290:     >>> a = np.arange(8).reshape(2,2,2); a
1291:     array([[[0, 1],
1292:             [2, 3]],
1293:            [[4, 5],
1294:             [6, 7]]])
1295:     >>> a.diagonal(0, # Main diagonals of two arrays created by skipping
1296:     ...            0, # across the outer(left)-most axis last and
1297:     ...            1) # the "middle" (row) axis first.
1298:     array([[0, 6],
1299:            [1, 7]])
1300: 
1301:     The sub-arrays whose main diagonals we just obtained; note that each
1302:     corresponds to fixing the right-most (column) axis, and that the
1303:     diagonals are "packed" in rows.
1304: 
1305:     >>> a[:,:,0] # main diagonal is [0 6]
1306:     array([[0, 2],
1307:            [4, 6]])
1308:     >>> a[:,:,1] # main diagonal is [1 7]
1309:     array([[1, 3],
1310:            [5, 7]])
1311: 
1312:     '''
1313:     if isinstance(a, np.matrix):
1314:         # Make diagonal of matrix 1-D to preserve backward compatibility.
1315:         return asarray(a).diagonal(offset, axis1, axis2)
1316:     else:
1317:         return asanyarray(a).diagonal(offset, axis1, axis2)
1318: 
1319: 
1320: def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
1321:     '''
1322:     Return the sum along diagonals of the array.
1323: 
1324:     If `a` is 2-D, the sum along its diagonal with the given offset
1325:     is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.
1326: 
1327:     If `a` has more than two dimensions, then the axes specified by axis1 and
1328:     axis2 are used to determine the 2-D sub-arrays whose traces are returned.
1329:     The shape of the resulting array is the same as that of `a` with `axis1`
1330:     and `axis2` removed.
1331: 
1332:     Parameters
1333:     ----------
1334:     a : array_like
1335:         Input array, from which the diagonals are taken.
1336:     offset : int, optional
1337:         Offset of the diagonal from the main diagonal. Can be both positive
1338:         and negative. Defaults to 0.
1339:     axis1, axis2 : int, optional
1340:         Axes to be used as the first and second axis of the 2-D sub-arrays
1341:         from which the diagonals should be taken. Defaults are the first two
1342:         axes of `a`.
1343:     dtype : dtype, optional
1344:         Determines the data-type of the returned array and of the accumulator
1345:         where the elements are summed. If dtype has the value None and `a` is
1346:         of integer type of precision less than the default integer
1347:         precision, then the default integer precision is used. Otherwise,
1348:         the precision is the same as that of `a`.
1349:     out : ndarray, optional
1350:         Array into which the output is placed. Its type is preserved and
1351:         it must be of the right shape to hold the output.
1352: 
1353:     Returns
1354:     -------
1355:     sum_along_diagonals : ndarray
1356:         If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
1357:         larger dimensions, then an array of sums along diagonals is returned.
1358: 
1359:     See Also
1360:     --------
1361:     diag, diagonal, diagflat
1362: 
1363:     Examples
1364:     --------
1365:     >>> np.trace(np.eye(3))
1366:     3.0
1367:     >>> a = np.arange(8).reshape((2,2,2))
1368:     >>> np.trace(a)
1369:     array([6, 8])
1370: 
1371:     >>> a = np.arange(24).reshape((2,2,2,3))
1372:     >>> np.trace(a).shape
1373:     (2, 3)
1374: 
1375:     '''
1376:     if isinstance(a, np.matrix):
1377:         # Get trace of matrix via an array to preserve backward compatibility.
1378:         return asarray(a).trace(offset, axis1, axis2, dtype, out)
1379:     else:
1380:         return asanyarray(a).trace(offset, axis1, axis2, dtype, out)
1381: 
1382: 
1383: def ravel(a, order='C'):
1384:     '''Return a contiguous flattened array.
1385: 
1386:     A 1-D array, containing the elements of the input, is returned.  A copy is
1387:     made only if needed.
1388: 
1389:     As of NumPy 1.10, the returned array will have the same type as the input
1390:     array. (for example, a masked array will be returned for a masked array
1391:     input)
1392: 
1393:     Parameters
1394:     ----------
1395:     a : array_like
1396:         Input array.  The elements in `a` are read in the order specified by
1397:         `order`, and packed as a 1-D array.
1398:     order : {'C','F', 'A', 'K'}, optional
1399: 
1400:         The elements of `a` are read using this index order. 'C' means
1401:         to index the elements in row-major, C-style order,
1402:         with the last axis index changing fastest, back to the first
1403:         axis index changing slowest.  'F' means to index the elements
1404:         in column-major, Fortran-style order, with the
1405:         first index changing fastest, and the last index changing
1406:         slowest. Note that the 'C' and 'F' options take no account of
1407:         the memory layout of the underlying array, and only refer to
1408:         the order of axis indexing.  'A' means to read the elements in
1409:         Fortran-like index order if `a` is Fortran *contiguous* in
1410:         memory, C-like order otherwise.  'K' means to read the
1411:         elements in the order they occur in memory, except for
1412:         reversing the data when strides are negative.  By default, 'C'
1413:         index order is used.
1414: 
1415:     Returns
1416:     -------
1417:     y : array_like
1418:         If `a` is a matrix, y is a 1-D ndarray, otherwise y is an array of
1419:         the same subtype as `a`. The shape of the returned array is
1420:         ``(a.size,)``. Matrices are special cased for backward
1421:         compatibility.
1422: 
1423:     See Also
1424:     --------
1425:     ndarray.flat : 1-D iterator over an array.
1426:     ndarray.flatten : 1-D array copy of the elements of an array
1427:                       in row-major order.
1428:     ndarray.reshape : Change the shape of an array without changing its data.
1429: 
1430:     Notes
1431:     -----
1432:     In row-major, C-style order, in two dimensions, the row index
1433:     varies the slowest, and the column index the quickest.  This can
1434:     be generalized to multiple dimensions, where row-major order
1435:     implies that the index along the first axis varies slowest, and
1436:     the index along the last quickest.  The opposite holds for
1437:     column-major, Fortran-style index ordering.
1438: 
1439:     When a view is desired in as many cases as possible, ``arr.reshape(-1)``
1440:     may be preferable.
1441: 
1442:     Examples
1443:     --------
1444:     It is equivalent to ``reshape(-1, order=order)``.
1445: 
1446:     >>> x = np.array([[1, 2, 3], [4, 5, 6]])
1447:     >>> print(np.ravel(x))
1448:     [1 2 3 4 5 6]
1449: 
1450:     >>> print(x.reshape(-1))
1451:     [1 2 3 4 5 6]
1452: 
1453:     >>> print(np.ravel(x, order='F'))
1454:     [1 4 2 5 3 6]
1455: 
1456:     When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:
1457: 
1458:     >>> print(np.ravel(x.T))
1459:     [1 4 2 5 3 6]
1460:     >>> print(np.ravel(x.T, order='A'))
1461:     [1 2 3 4 5 6]
1462: 
1463:     When ``order`` is 'K', it will preserve orderings that are neither 'C'
1464:     nor 'F', but won't reverse axes:
1465: 
1466:     >>> a = np.arange(3)[::-1]; a
1467:     array([2, 1, 0])
1468:     >>> a.ravel(order='C')
1469:     array([2, 1, 0])
1470:     >>> a.ravel(order='K')
1471:     array([2, 1, 0])
1472: 
1473:     >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
1474:     array([[[ 0,  2,  4],
1475:             [ 1,  3,  5]],
1476:            [[ 6,  8, 10],
1477:             [ 7,  9, 11]]])
1478:     >>> a.ravel(order='C')
1479:     array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
1480:     >>> a.ravel(order='K')
1481:     array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
1482: 
1483:     '''
1484:     if isinstance(a, np.matrix):
1485:         return asarray(a).ravel(order)
1486:     else:
1487:         return asanyarray(a).ravel(order)
1488: 
1489: 
1490: def nonzero(a):
1491:     '''
1492:     Return the indices of the elements that are non-zero.
1493: 
1494:     Returns a tuple of arrays, one for each dimension of `a`,
1495:     containing the indices of the non-zero elements in that
1496:     dimension. The values in `a` are always tested and returned in
1497:     row-major, C-style order. The corresponding non-zero
1498:     values can be obtained with::
1499: 
1500:         a[nonzero(a)]
1501: 
1502:     To group the indices by element, rather than dimension, use::
1503: 
1504:         transpose(nonzero(a))
1505: 
1506:     The result of this is always a 2-D array, with a row for
1507:     each non-zero element.
1508: 
1509:     Parameters
1510:     ----------
1511:     a : array_like
1512:         Input array.
1513: 
1514:     Returns
1515:     -------
1516:     tuple_of_arrays : tuple
1517:         Indices of elements that are non-zero.
1518: 
1519:     See Also
1520:     --------
1521:     flatnonzero :
1522:         Return indices that are non-zero in the flattened version of the input
1523:         array.
1524:     ndarray.nonzero :
1525:         Equivalent ndarray method.
1526:     count_nonzero :
1527:         Counts the number of non-zero elements in the input array.
1528: 
1529:     Examples
1530:     --------
1531:     >>> x = np.eye(3)
1532:     >>> x
1533:     array([[ 1.,  0.,  0.],
1534:            [ 0.,  1.,  0.],
1535:            [ 0.,  0.,  1.]])
1536:     >>> np.nonzero(x)
1537:     (array([0, 1, 2]), array([0, 1, 2]))
1538: 
1539:     >>> x[np.nonzero(x)]
1540:     array([ 1.,  1.,  1.])
1541:     >>> np.transpose(np.nonzero(x))
1542:     array([[0, 0],
1543:            [1, 1],
1544:            [2, 2]])
1545: 
1546:     A common use for ``nonzero`` is to find the indices of an array, where
1547:     a condition is True.  Given an array `a`, the condition `a` > 3 is a
1548:     boolean array and since False is interpreted as 0, np.nonzero(a > 3)
1549:     yields the indices of the `a` where the condition is true.
1550: 
1551:     >>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
1552:     >>> a > 3
1553:     array([[False, False, False],
1554:            [ True,  True,  True],
1555:            [ True,  True,  True]], dtype=bool)
1556:     >>> np.nonzero(a > 3)
1557:     (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
1558: 
1559:     The ``nonzero`` method of the boolean array can also be called.
1560: 
1561:     >>> (a > 3).nonzero()
1562:     (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
1563: 
1564:     '''
1565:     try:
1566:         nonzero = a.nonzero
1567:     except AttributeError:
1568:         res = _wrapit(a, 'nonzero')
1569:     else:
1570:         res = nonzero()
1571:     return res
1572: 
1573: 
1574: def shape(a):
1575:     '''
1576:     Return the shape of an array.
1577: 
1578:     Parameters
1579:     ----------
1580:     a : array_like
1581:         Input array.
1582: 
1583:     Returns
1584:     -------
1585:     shape : tuple of ints
1586:         The elements of the shape tuple give the lengths of the
1587:         corresponding array dimensions.
1588: 
1589:     See Also
1590:     --------
1591:     alen
1592:     ndarray.shape : Equivalent array method.
1593: 
1594:     Examples
1595:     --------
1596:     >>> np.shape(np.eye(3))
1597:     (3, 3)
1598:     >>> np.shape([[1, 2]])
1599:     (1, 2)
1600:     >>> np.shape([0])
1601:     (1,)
1602:     >>> np.shape(0)
1603:     ()
1604: 
1605:     >>> a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])
1606:     >>> np.shape(a)
1607:     (2,)
1608:     >>> a.shape
1609:     (2,)
1610: 
1611:     '''
1612:     try:
1613:         result = a.shape
1614:     except AttributeError:
1615:         result = asarray(a).shape
1616:     return result
1617: 
1618: 
1619: def compress(condition, a, axis=None, out=None):
1620:     '''
1621:     Return selected slices of an array along given axis.
1622: 
1623:     When working along a given axis, a slice along that axis is returned in
1624:     `output` for each index where `condition` evaluates to True. When
1625:     working on a 1-D array, `compress` is equivalent to `extract`.
1626: 
1627:     Parameters
1628:     ----------
1629:     condition : 1-D array of bools
1630:         Array that selects which entries to return. If len(condition)
1631:         is less than the size of `a` along the given axis, then output is
1632:         truncated to the length of the condition array.
1633:     a : array_like
1634:         Array from which to extract a part.
1635:     axis : int, optional
1636:         Axis along which to take slices. If None (default), work on the
1637:         flattened array.
1638:     out : ndarray, optional
1639:         Output array.  Its type is preserved and it must be of the right
1640:         shape to hold the output.
1641: 
1642:     Returns
1643:     -------
1644:     compressed_array : ndarray
1645:         A copy of `a` without the slices along axis for which `condition`
1646:         is false.
1647: 
1648:     See Also
1649:     --------
1650:     take, choose, diag, diagonal, select
1651:     ndarray.compress : Equivalent method in ndarray
1652:     np.extract: Equivalent method when working on 1-D arrays
1653:     numpy.doc.ufuncs : Section "Output arguments"
1654: 
1655:     Examples
1656:     --------
1657:     >>> a = np.array([[1, 2], [3, 4], [5, 6]])
1658:     >>> a
1659:     array([[1, 2],
1660:            [3, 4],
1661:            [5, 6]])
1662:     >>> np.compress([0, 1], a, axis=0)
1663:     array([[3, 4]])
1664:     >>> np.compress([False, True, True], a, axis=0)
1665:     array([[3, 4],
1666:            [5, 6]])
1667:     >>> np.compress([False, True], a, axis=1)
1668:     array([[2],
1669:            [4],
1670:            [6]])
1671: 
1672:     Working on the flattened array does not return slices along an axis but
1673:     selects elements.
1674: 
1675:     >>> np.compress([False, True], a)
1676:     array([2])
1677: 
1678:     '''
1679:     try:
1680:         compress = a.compress
1681:     except AttributeError:
1682:         return _wrapit(a, 'compress', condition, axis, out)
1683:     return compress(condition, axis, out)
1684: 
1685: 
1686: def clip(a, a_min, a_max, out=None):
1687:     '''
1688:     Clip (limit) the values in an array.
1689: 
1690:     Given an interval, values outside the interval are clipped to
1691:     the interval edges.  For example, if an interval of ``[0, 1]``
1692:     is specified, values smaller than 0 become 0, and values larger
1693:     than 1 become 1.
1694: 
1695:     Parameters
1696:     ----------
1697:     a : array_like
1698:         Array containing elements to clip.
1699:     a_min : scalar or array_like
1700:         Minimum value.
1701:     a_max : scalar or array_like
1702:         Maximum value.  If `a_min` or `a_max` are array_like, then they will
1703:         be broadcasted to the shape of `a`.
1704:     out : ndarray, optional
1705:         The results will be placed in this array. It may be the input
1706:         array for in-place clipping.  `out` must be of the right shape
1707:         to hold the output.  Its type is preserved.
1708: 
1709:     Returns
1710:     -------
1711:     clipped_array : ndarray
1712:         An array with the elements of `a`, but where values
1713:         < `a_min` are replaced with `a_min`, and those > `a_max`
1714:         with `a_max`.
1715: 
1716:     See Also
1717:     --------
1718:     numpy.doc.ufuncs : Section "Output arguments"
1719: 
1720:     Examples
1721:     --------
1722:     >>> a = np.arange(10)
1723:     >>> np.clip(a, 1, 8)
1724:     array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
1725:     >>> a
1726:     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
1727:     >>> np.clip(a, 3, 6, out=a)
1728:     array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
1729:     >>> a = np.arange(10)
1730:     >>> a
1731:     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
1732:     >>> np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)
1733:     array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
1734: 
1735:     '''
1736:     try:
1737:         clip = a.clip
1738:     except AttributeError:
1739:         return _wrapit(a, 'clip', a_min, a_max, out)
1740:     return clip(a_min, a_max, out)
1741: 
1742: 
1743: def sum(a, axis=None, dtype=None, out=None, keepdims=False):
1744:     '''
1745:     Sum of array elements over a given axis.
1746: 
1747:     Parameters
1748:     ----------
1749:     a : array_like
1750:         Elements to sum.
1751:     axis : None or int or tuple of ints, optional
1752:         Axis or axes along which a sum is performed.  The default,
1753:         axis=None, will sum all of the elements of the input array.  If
1754:         axis is negative it counts from the last to the first axis.
1755: 
1756:         .. versionadded:: 1.7.0
1757: 
1758:         If axis is a tuple of ints, a sum is performed on all of the axes
1759:         specified in the tuple instead of a single axis or all the axes as
1760:         before.
1761:     dtype : dtype, optional
1762:         The type of the returned array and of the accumulator in which the
1763:         elements are summed.  The dtype of `a` is used by default unless `a`
1764:         has an integer dtype of less precision than the default platform
1765:         integer.  In that case, if `a` is signed then the platform integer
1766:         is used while if `a` is unsigned then an unsigned integer of the
1767:         same precision as the platform integer is used.
1768:     out : ndarray, optional
1769:         Alternative output array in which to place the result. It must have
1770:         the same shape as the expected output, but the type of the output
1771:         values will be cast if necessary.
1772:     keepdims : bool, optional
1773:         If this is set to True, the axes which are reduced are left in the
1774:         result as dimensions with size one. With this option, the result
1775:         will broadcast correctly against the input array.
1776: 
1777:     Returns
1778:     -------
1779:     sum_along_axis : ndarray
1780:         An array with the same shape as `a`, with the specified
1781:         axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
1782:         is returned.  If an output array is specified, a reference to
1783:         `out` is returned.
1784: 
1785:     See Also
1786:     --------
1787:     ndarray.sum : Equivalent method.
1788: 
1789:     cumsum : Cumulative sum of array elements.
1790: 
1791:     trapz : Integration of array values using the composite trapezoidal rule.
1792: 
1793:     mean, average
1794: 
1795:     Notes
1796:     -----
1797:     Arithmetic is modular when using integer types, and no error is
1798:     raised on overflow.
1799: 
1800:     The sum of an empty array is the neutral element 0:
1801: 
1802:     >>> np.sum([])
1803:     0.0
1804: 
1805:     Examples
1806:     --------
1807:     >>> np.sum([0.5, 1.5])
1808:     2.0
1809:     >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
1810:     1
1811:     >>> np.sum([[0, 1], [0, 5]])
1812:     6
1813:     >>> np.sum([[0, 1], [0, 5]], axis=0)
1814:     array([0, 6])
1815:     >>> np.sum([[0, 1], [0, 5]], axis=1)
1816:     array([1, 5])
1817: 
1818:     If the accumulator is too small, overflow occurs:
1819: 
1820:     >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
1821:     -128
1822: 
1823:     '''
1824:     if isinstance(a, _gentype):
1825:         res = _sum_(a)
1826:         if out is not None:
1827:             out[...] = res
1828:             return out
1829:         return res
1830:     elif type(a) is not mu.ndarray:
1831:         try:
1832:             sum = a.sum
1833:         except AttributeError:
1834:             return _methods._sum(a, axis=axis, dtype=dtype,
1835:                                  out=out, keepdims=keepdims)
1836:         # NOTE: Dropping the keepdims parameters here...
1837:         return sum(axis=axis, dtype=dtype, out=out)
1838:     else:
1839:         return _methods._sum(a, axis=axis, dtype=dtype,
1840:                              out=out, keepdims=keepdims)
1841: 
1842: 
1843: def product(a, axis=None, dtype=None, out=None, keepdims=False):
1844:     '''
1845:     Return the product of array elements over a given axis.
1846: 
1847:     See Also
1848:     --------
1849:     prod : equivalent function; see for details.
1850: 
1851:     '''
1852:     return um.multiply.reduce(a, axis=axis, dtype=dtype,
1853:                               out=out, keepdims=keepdims)
1854: 
1855: 
1856: def sometrue(a, axis=None, out=None, keepdims=False):
1857:     '''
1858:     Check whether some values are true.
1859: 
1860:     Refer to `any` for full documentation.
1861: 
1862:     See Also
1863:     --------
1864:     any : equivalent function
1865: 
1866:     '''
1867:     arr = asanyarray(a)
1868: 
1869:     try:
1870:         return arr.any(axis=axis, out=out, keepdims=keepdims)
1871:     except TypeError:
1872:         return arr.any(axis=axis, out=out)
1873: 
1874: 
1875: def alltrue(a, axis=None, out=None, keepdims=False):
1876:     '''
1877:     Check if all elements of input array are true.
1878: 
1879:     See Also
1880:     --------
1881:     numpy.all : Equivalent function; see for details.
1882: 
1883:     '''
1884:     arr = asanyarray(a)
1885: 
1886:     try:
1887:         return arr.all(axis=axis, out=out, keepdims=keepdims)
1888:     except TypeError:
1889:         return arr.all(axis=axis, out=out)
1890: 
1891: 
1892: def any(a, axis=None, out=None, keepdims=False):
1893:     '''
1894:     Test whether any array element along a given axis evaluates to True.
1895: 
1896:     Returns single boolean unless `axis` is not ``None``
1897: 
1898:     Parameters
1899:     ----------
1900:     a : array_like
1901:         Input array or object that can be converted to an array.
1902:     axis : None or int or tuple of ints, optional
1903:         Axis or axes along which a logical OR reduction is performed.
1904:         The default (`axis` = `None`) is to perform a logical OR over all
1905:         the dimensions of the input array. `axis` may be negative, in
1906:         which case it counts from the last to the first axis.
1907: 
1908:         .. versionadded:: 1.7.0
1909: 
1910:         If this is a tuple of ints, a reduction is performed on multiple
1911:         axes, instead of a single axis or all the axes as before.
1912:     out : ndarray, optional
1913:         Alternate output array in which to place the result.  It must have
1914:         the same shape as the expected output and its type is preserved
1915:         (e.g., if it is of type float, then it will remain so, returning
1916:         1.0 for True and 0.0 for False, regardless of the type of `a`).
1917:         See `doc.ufuncs` (Section "Output arguments") for details.
1918:     keepdims : bool, optional
1919:         If this is set to True, the axes which are reduced are left
1920:         in the result as dimensions with size one. With this option,
1921:         the result will broadcast correctly against the original `arr`.
1922: 
1923:     Returns
1924:     -------
1925:     any : bool or ndarray
1926:         A new boolean or `ndarray` is returned unless `out` is specified,
1927:         in which case a reference to `out` is returned.
1928: 
1929:     See Also
1930:     --------
1931:     ndarray.any : equivalent method
1932: 
1933:     all : Test whether all elements along a given axis evaluate to True.
1934: 
1935:     Notes
1936:     -----
1937:     Not a Number (NaN), positive infinity and negative infinity evaluate
1938:     to `True` because these are not equal to zero.
1939: 
1940:     Examples
1941:     --------
1942:     >>> np.any([[True, False], [True, True]])
1943:     True
1944: 
1945:     >>> np.any([[True, False], [False, False]], axis=0)
1946:     array([ True, False], dtype=bool)
1947: 
1948:     >>> np.any([-1, 0, 5])
1949:     True
1950: 
1951:     >>> np.any(np.nan)
1952:     True
1953: 
1954:     >>> o=np.array([False])
1955:     >>> z=np.any([-1, 4, 5], out=o)
1956:     >>> z, o
1957:     (array([ True], dtype=bool), array([ True], dtype=bool))
1958:     >>> # Check now that z is a reference to o
1959:     >>> z is o
1960:     True
1961:     >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
1962:     (191614240, 191614240)
1963: 
1964:     '''
1965:     arr = asanyarray(a)
1966: 
1967:     try:
1968:         return arr.any(axis=axis, out=out, keepdims=keepdims)
1969:     except TypeError:
1970:         return arr.any(axis=axis, out=out)
1971: 
1972: 
1973: def all(a, axis=None, out=None, keepdims=False):
1974:     '''
1975:     Test whether all array elements along a given axis evaluate to True.
1976: 
1977:     Parameters
1978:     ----------
1979:     a : array_like
1980:         Input array or object that can be converted to an array.
1981:     axis : None or int or tuple of ints, optional
1982:         Axis or axes along which a logical AND reduction is performed.
1983:         The default (`axis` = `None`) is to perform a logical AND over all
1984:         the dimensions of the input array. `axis` may be negative, in
1985:         which case it counts from the last to the first axis.
1986: 
1987:         .. versionadded:: 1.7.0
1988: 
1989:         If this is a tuple of ints, a reduction is performed on multiple
1990:         axes, instead of a single axis or all the axes as before.
1991:     out : ndarray, optional
1992:         Alternate output array in which to place the result.
1993:         It must have the same shape as the expected output and its
1994:         type is preserved (e.g., if ``dtype(out)`` is float, the result
1995:         will consist of 0.0's and 1.0's).  See `doc.ufuncs` (Section
1996:         "Output arguments") for more details.
1997:     keepdims : bool, optional
1998:         If this is set to True, the axes which are reduced are left
1999:         in the result as dimensions with size one. With this option,
2000:         the result will broadcast correctly against the original `arr`.
2001: 
2002:     Returns
2003:     -------
2004:     all : ndarray, bool
2005:         A new boolean or array is returned unless `out` is specified,
2006:         in which case a reference to `out` is returned.
2007: 
2008:     See Also
2009:     --------
2010:     ndarray.all : equivalent method
2011: 
2012:     any : Test whether any element along a given axis evaluates to True.
2013: 
2014:     Notes
2015:     -----
2016:     Not a Number (NaN), positive infinity and negative infinity
2017:     evaluate to `True` because these are not equal to zero.
2018: 
2019:     Examples
2020:     --------
2021:     >>> np.all([[True,False],[True,True]])
2022:     False
2023: 
2024:     >>> np.all([[True,False],[True,True]], axis=0)
2025:     array([ True, False], dtype=bool)
2026: 
2027:     >>> np.all([-1, 4, 5])
2028:     True
2029: 
2030:     >>> np.all([1.0, np.nan])
2031:     True
2032: 
2033:     >>> o=np.array([False])
2034:     >>> z=np.all([-1, 4, 5], out=o)
2035:     >>> id(z), id(o), z                             # doctest: +SKIP
2036:     (28293632, 28293632, array([ True], dtype=bool))
2037: 
2038:     '''
2039:     arr = asanyarray(a)
2040: 
2041:     try:
2042:         return arr.all(axis=axis, out=out, keepdims=keepdims)
2043:     except TypeError:
2044:         return arr.all(axis=axis, out=out)
2045: 
2046: 
2047: def cumsum(a, axis=None, dtype=None, out=None):
2048:     '''
2049:     Return the cumulative sum of the elements along a given axis.
2050: 
2051:     Parameters
2052:     ----------
2053:     a : array_like
2054:         Input array.
2055:     axis : int, optional
2056:         Axis along which the cumulative sum is computed. The default
2057:         (None) is to compute the cumsum over the flattened array.
2058:     dtype : dtype, optional
2059:         Type of the returned array and of the accumulator in which the
2060:         elements are summed.  If `dtype` is not specified, it defaults
2061:         to the dtype of `a`, unless `a` has an integer dtype with a
2062:         precision less than that of the default platform integer.  In
2063:         that case, the default platform integer is used.
2064:     out : ndarray, optional
2065:         Alternative output array in which to place the result. It must
2066:         have the same shape and buffer length as the expected output
2067:         but the type will be cast if necessary. See `doc.ufuncs`
2068:         (Section "Output arguments") for more details.
2069: 
2070:     Returns
2071:     -------
2072:     cumsum_along_axis : ndarray.
2073:         A new array holding the result is returned unless `out` is
2074:         specified, in which case a reference to `out` is returned. The
2075:         result has the same size as `a`, and the same shape as `a` if
2076:         `axis` is not None or `a` is a 1-d array.
2077: 
2078: 
2079:     See Also
2080:     --------
2081:     sum : Sum array elements.
2082: 
2083:     trapz : Integration of array values using the composite trapezoidal rule.
2084: 
2085:     diff :  Calculate the n-th discrete difference along given axis.
2086: 
2087:     Notes
2088:     -----
2089:     Arithmetic is modular when using integer types, and no error is
2090:     raised on overflow.
2091: 
2092:     Examples
2093:     --------
2094:     >>> a = np.array([[1,2,3], [4,5,6]])
2095:     >>> a
2096:     array([[1, 2, 3],
2097:            [4, 5, 6]])
2098:     >>> np.cumsum(a)
2099:     array([ 1,  3,  6, 10, 15, 21])
2100:     >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
2101:     array([  1.,   3.,   6.,  10.,  15.,  21.])
2102: 
2103:     >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
2104:     array([[1, 2, 3],
2105:            [5, 7, 9]])
2106:     >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
2107:     array([[ 1,  3,  6],
2108:            [ 4,  9, 15]])
2109: 
2110:     '''
2111:     try:
2112:         cumsum = a.cumsum
2113:     except AttributeError:
2114:         return _wrapit(a, 'cumsum', axis, dtype, out)
2115:     return cumsum(axis, dtype, out)
2116: 
2117: 
2118: def cumproduct(a, axis=None, dtype=None, out=None):
2119:     '''
2120:     Return the cumulative product over the given axis.
2121: 
2122: 
2123:     See Also
2124:     --------
2125:     cumprod : equivalent function; see for details.
2126: 
2127:     '''
2128:     try:
2129:         cumprod = a.cumprod
2130:     except AttributeError:
2131:         return _wrapit(a, 'cumprod', axis, dtype, out)
2132:     return cumprod(axis, dtype, out)
2133: 
2134: 
2135: def ptp(a, axis=None, out=None):
2136:     '''
2137:     Range of values (maximum - minimum) along an axis.
2138: 
2139:     The name of the function comes from the acronym for 'peak to peak'.
2140: 
2141:     Parameters
2142:     ----------
2143:     a : array_like
2144:         Input values.
2145:     axis : int, optional
2146:         Axis along which to find the peaks.  By default, flatten the
2147:         array.
2148:     out : array_like
2149:         Alternative output array in which to place the result. It must
2150:         have the same shape and buffer length as the expected output,
2151:         but the type of the output values will be cast if necessary.
2152: 
2153:     Returns
2154:     -------
2155:     ptp : ndarray
2156:         A new array holding the result, unless `out` was
2157:         specified, in which case a reference to `out` is returned.
2158: 
2159:     Examples
2160:     --------
2161:     >>> x = np.arange(4).reshape((2,2))
2162:     >>> x
2163:     array([[0, 1],
2164:            [2, 3]])
2165: 
2166:     >>> np.ptp(x, axis=0)
2167:     array([2, 2])
2168: 
2169:     >>> np.ptp(x, axis=1)
2170:     array([1, 1])
2171: 
2172:     '''
2173:     try:
2174:         ptp = a.ptp
2175:     except AttributeError:
2176:         return _wrapit(a, 'ptp', axis, out)
2177:     return ptp(axis, out)
2178: 
2179: 
2180: def amax(a, axis=None, out=None, keepdims=False):
2181:     '''
2182:     Return the maximum of an array or maximum along an axis.
2183: 
2184:     Parameters
2185:     ----------
2186:     a : array_like
2187:         Input data.
2188:     axis : None or int or tuple of ints, optional
2189:         Axis or axes along which to operate.  By default, flattened input is
2190:         used.
2191: 
2192:         .. versionadded: 1.7.0
2193: 
2194:         If this is a tuple of ints, the maximum is selected over multiple axes,
2195:         instead of a single axis or all the axes as before.
2196:     out : ndarray, optional
2197:         Alternative output array in which to place the result.  Must
2198:         be of the same shape and buffer length as the expected output.
2199:         See `doc.ufuncs` (Section "Output arguments") for more details.
2200:     keepdims : bool, optional
2201:         If this is set to True, the axes which are reduced are left
2202:         in the result as dimensions with size one. With this option,
2203:         the result will broadcast correctly against the original `arr`.
2204: 
2205:     Returns
2206:     -------
2207:     amax : ndarray or scalar
2208:         Maximum of `a`. If `axis` is None, the result is a scalar value.
2209:         If `axis` is given, the result is an array of dimension
2210:         ``a.ndim - 1``.
2211: 
2212:     See Also
2213:     --------
2214:     amin :
2215:         The minimum value of an array along a given axis, propagating any NaNs.
2216:     nanmax :
2217:         The maximum value of an array along a given axis, ignoring any NaNs.
2218:     maximum :
2219:         Element-wise maximum of two arrays, propagating any NaNs.
2220:     fmax :
2221:         Element-wise maximum of two arrays, ignoring any NaNs.
2222:     argmax :
2223:         Return the indices of the maximum values.
2224: 
2225:     nanmin, minimum, fmin
2226: 
2227:     Notes
2228:     -----
2229:     NaN values are propagated, that is if at least one item is NaN, the
2230:     corresponding max value will be NaN as well. To ignore NaN values
2231:     (MATLAB behavior), please use nanmax.
2232: 
2233:     Don't use `amax` for element-wise comparison of 2 arrays; when
2234:     ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
2235:     ``amax(a, axis=0)``.
2236: 
2237:     Examples
2238:     --------
2239:     >>> a = np.arange(4).reshape((2,2))
2240:     >>> a
2241:     array([[0, 1],
2242:            [2, 3]])
2243:     >>> np.amax(a)           # Maximum of the flattened array
2244:     3
2245:     >>> np.amax(a, axis=0)   # Maxima along the first axis
2246:     array([2, 3])
2247:     >>> np.amax(a, axis=1)   # Maxima along the second axis
2248:     array([1, 3])
2249: 
2250:     >>> b = np.arange(5, dtype=np.float)
2251:     >>> b[2] = np.NaN
2252:     >>> np.amax(b)
2253:     nan
2254:     >>> np.nanmax(b)
2255:     4.0
2256: 
2257:     '''
2258:     if type(a) is not mu.ndarray:
2259:         try:
2260:             amax = a.max
2261:         except AttributeError:
2262:             return _methods._amax(a, axis=axis,
2263:                                   out=out, keepdims=keepdims)
2264:         # NOTE: Dropping the keepdims parameter
2265:         return amax(axis=axis, out=out)
2266:     else:
2267:         return _methods._amax(a, axis=axis,
2268:                               out=out, keepdims=keepdims)
2269: 
2270: 
2271: def amin(a, axis=None, out=None, keepdims=False):
2272:     '''
2273:     Return the minimum of an array or minimum along an axis.
2274: 
2275:     Parameters
2276:     ----------
2277:     a : array_like
2278:         Input data.
2279:     axis : None or int or tuple of ints, optional
2280:         Axis or axes along which to operate.  By default, flattened input is
2281:         used.
2282: 
2283:         .. versionadded: 1.7.0
2284: 
2285:         If this is a tuple of ints, the minimum is selected over multiple axes,
2286:         instead of a single axis or all the axes as before.
2287:     out : ndarray, optional
2288:         Alternative output array in which to place the result.  Must
2289:         be of the same shape and buffer length as the expected output.
2290:         See `doc.ufuncs` (Section "Output arguments") for more details.
2291:     keepdims : bool, optional
2292:         If this is set to True, the axes which are reduced are left
2293:         in the result as dimensions with size one. With this option,
2294:         the result will broadcast correctly against the original `arr`.
2295: 
2296:     Returns
2297:     -------
2298:     amin : ndarray or scalar
2299:         Minimum of `a`. If `axis` is None, the result is a scalar value.
2300:         If `axis` is given, the result is an array of dimension
2301:         ``a.ndim - 1``.
2302: 
2303:     See Also
2304:     --------
2305:     amax :
2306:         The maximum value of an array along a given axis, propagating any NaNs.
2307:     nanmin :
2308:         The minimum value of an array along a given axis, ignoring any NaNs.
2309:     minimum :
2310:         Element-wise minimum of two arrays, propagating any NaNs.
2311:     fmin :
2312:         Element-wise minimum of two arrays, ignoring any NaNs.
2313:     argmin :
2314:         Return the indices of the minimum values.
2315: 
2316:     nanmax, maximum, fmax
2317: 
2318:     Notes
2319:     -----
2320:     NaN values are propagated, that is if at least one item is NaN, the
2321:     corresponding min value will be NaN as well. To ignore NaN values
2322:     (MATLAB behavior), please use nanmin.
2323: 
2324:     Don't use `amin` for element-wise comparison of 2 arrays; when
2325:     ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
2326:     ``amin(a, axis=0)``.
2327: 
2328:     Examples
2329:     --------
2330:     >>> a = np.arange(4).reshape((2,2))
2331:     >>> a
2332:     array([[0, 1],
2333:            [2, 3]])
2334:     >>> np.amin(a)           # Minimum of the flattened array
2335:     0
2336:     >>> np.amin(a, axis=0)   # Minima along the first axis
2337:     array([0, 1])
2338:     >>> np.amin(a, axis=1)   # Minima along the second axis
2339:     array([0, 2])
2340: 
2341:     >>> b = np.arange(5, dtype=np.float)
2342:     >>> b[2] = np.NaN
2343:     >>> np.amin(b)
2344:     nan
2345:     >>> np.nanmin(b)
2346:     0.0
2347: 
2348:     '''
2349:     if type(a) is not mu.ndarray:
2350:         try:
2351:             amin = a.min
2352:         except AttributeError:
2353:             return _methods._amin(a, axis=axis,
2354:                                   out=out, keepdims=keepdims)
2355:         # NOTE: Dropping the keepdims parameter
2356:         return amin(axis=axis, out=out)
2357:     else:
2358:         return _methods._amin(a, axis=axis,
2359:                               out=out, keepdims=keepdims)
2360: 
2361: 
2362: def alen(a):
2363:     '''
2364:     Return the length of the first dimension of the input array.
2365: 
2366:     Parameters
2367:     ----------
2368:     a : array_like
2369:        Input array.
2370: 
2371:     Returns
2372:     -------
2373:     alen : int
2374:        Length of the first dimension of `a`.
2375: 
2376:     See Also
2377:     --------
2378:     shape, size
2379: 
2380:     Examples
2381:     --------
2382:     >>> a = np.zeros((7,4,5))
2383:     >>> a.shape[0]
2384:     7
2385:     >>> np.alen(a)
2386:     7
2387: 
2388:     '''
2389:     try:
2390:         return len(a)
2391:     except TypeError:
2392:         return len(array(a, ndmin=1))
2393: 
2394: 
2395: def prod(a, axis=None, dtype=None, out=None, keepdims=False):
2396:     '''
2397:     Return the product of array elements over a given axis.
2398: 
2399:     Parameters
2400:     ----------
2401:     a : array_like
2402:         Input data.
2403:     axis : None or int or tuple of ints, optional
2404:         Axis or axes along which a product is performed.  The default,
2405:         axis=None, will calculate the product of all the elements in the
2406:         input array. If axis is negative it counts from the last to the
2407:         first axis.
2408: 
2409:         .. versionadded:: 1.7.0
2410: 
2411:         If axis is a tuple of ints, a product is performed on all of the
2412:         axes specified in the tuple instead of a single axis or all the
2413:         axes as before.
2414:     dtype : dtype, optional
2415:         The type of the returned array, as well as of the accumulator in
2416:         which the elements are multiplied.  The dtype of `a` is used by
2417:         default unless `a` has an integer dtype of less precision than the
2418:         default platform integer.  In that case, if `a` is signed then the
2419:         platform integer is used while if `a` is unsigned then an unsigned
2420:         integer of the same precision as the platform integer is used.
2421:     out : ndarray, optional
2422:         Alternative output array in which to place the result. It must have
2423:         the same shape as the expected output, but the type of the output
2424:         values will be cast if necessary.
2425:     keepdims : bool, optional
2426:         If this is set to True, the axes which are reduced are left in the
2427:         result as dimensions with size one. With this option, the result
2428:         will broadcast correctly against the input array.
2429: 
2430:     Returns
2431:     -------
2432:     product_along_axis : ndarray, see `dtype` parameter above.
2433:         An array shaped as `a` but with the specified axis removed.
2434:         Returns a reference to `out` if specified.
2435: 
2436:     See Also
2437:     --------
2438:     ndarray.prod : equivalent method
2439:     numpy.doc.ufuncs : Section "Output arguments"
2440: 
2441:     Notes
2442:     -----
2443:     Arithmetic is modular when using integer types, and no error is
2444:     raised on overflow.  That means that, on a 32-bit platform:
2445: 
2446:     >>> x = np.array([536870910, 536870910, 536870910, 536870910])
2447:     >>> np.prod(x) #random
2448:     16
2449: 
2450:     The product of an empty array is the neutral element 1:
2451: 
2452:     >>> np.prod([])
2453:     1.0
2454: 
2455:     Examples
2456:     --------
2457:     By default, calculate the product of all elements:
2458: 
2459:     >>> np.prod([1.,2.])
2460:     2.0
2461: 
2462:     Even when the input array is two-dimensional:
2463: 
2464:     >>> np.prod([[1.,2.],[3.,4.]])
2465:     24.0
2466: 
2467:     But we can also specify the axis over which to multiply:
2468: 
2469:     >>> np.prod([[1.,2.],[3.,4.]], axis=1)
2470:     array([  2.,  12.])
2471: 
2472:     If the type of `x` is unsigned, then the output type is
2473:     the unsigned platform integer:
2474: 
2475:     >>> x = np.array([1, 2, 3], dtype=np.uint8)
2476:     >>> np.prod(x).dtype == np.uint
2477:     True
2478: 
2479:     If `x` is of a signed integer type, then the output type
2480:     is the default platform integer:
2481: 
2482:     >>> x = np.array([1, 2, 3], dtype=np.int8)
2483:     >>> np.prod(x).dtype == np.int
2484:     True
2485: 
2486:     '''
2487:     if type(a) is not mu.ndarray:
2488:         try:
2489:             prod = a.prod
2490:         except AttributeError:
2491:             return _methods._prod(a, axis=axis, dtype=dtype,
2492:                                   out=out, keepdims=keepdims)
2493:         return prod(axis=axis, dtype=dtype, out=out)
2494:     else:
2495:         return _methods._prod(a, axis=axis, dtype=dtype,
2496:                               out=out, keepdims=keepdims)
2497: 
2498: 
2499: def cumprod(a, axis=None, dtype=None, out=None):
2500:     '''
2501:     Return the cumulative product of elements along a given axis.
2502: 
2503:     Parameters
2504:     ----------
2505:     a : array_like
2506:         Input array.
2507:     axis : int, optional
2508:         Axis along which the cumulative product is computed.  By default
2509:         the input is flattened.
2510:     dtype : dtype, optional
2511:         Type of the returned array, as well as of the accumulator in which
2512:         the elements are multiplied.  If *dtype* is not specified, it
2513:         defaults to the dtype of `a`, unless `a` has an integer dtype with
2514:         a precision less than that of the default platform integer.  In
2515:         that case, the default platform integer is used instead.
2516:     out : ndarray, optional
2517:         Alternative output array in which to place the result. It must
2518:         have the same shape and buffer length as the expected output
2519:         but the type of the resulting values will be cast if necessary.
2520: 
2521:     Returns
2522:     -------
2523:     cumprod : ndarray
2524:         A new array holding the result is returned unless `out` is
2525:         specified, in which case a reference to out is returned.
2526: 
2527:     See Also
2528:     --------
2529:     numpy.doc.ufuncs : Section "Output arguments"
2530: 
2531:     Notes
2532:     -----
2533:     Arithmetic is modular when using integer types, and no error is
2534:     raised on overflow.
2535: 
2536:     Examples
2537:     --------
2538:     >>> a = np.array([1,2,3])
2539:     >>> np.cumprod(a) # intermediate results 1, 1*2
2540:     ...               # total product 1*2*3 = 6
2541:     array([1, 2, 6])
2542:     >>> a = np.array([[1, 2, 3], [4, 5, 6]])
2543:     >>> np.cumprod(a, dtype=float) # specify type of output
2544:     array([   1.,    2.,    6.,   24.,  120.,  720.])
2545: 
2546:     The cumulative product for each column (i.e., over the rows) of `a`:
2547: 
2548:     >>> np.cumprod(a, axis=0)
2549:     array([[ 1,  2,  3],
2550:            [ 4, 10, 18]])
2551: 
2552:     The cumulative product for each row (i.e. over the columns) of `a`:
2553: 
2554:     >>> np.cumprod(a,axis=1)
2555:     array([[  1,   2,   6],
2556:            [  4,  20, 120]])
2557: 
2558:     '''
2559:     try:
2560:         cumprod = a.cumprod
2561:     except AttributeError:
2562:         return _wrapit(a, 'cumprod', axis, dtype, out)
2563:     return cumprod(axis, dtype, out)
2564: 
2565: 
2566: def ndim(a):
2567:     '''
2568:     Return the number of dimensions of an array.
2569: 
2570:     Parameters
2571:     ----------
2572:     a : array_like
2573:         Input array.  If it is not already an ndarray, a conversion is
2574:         attempted.
2575: 
2576:     Returns
2577:     -------
2578:     number_of_dimensions : int
2579:         The number of dimensions in `a`.  Scalars are zero-dimensional.
2580: 
2581:     See Also
2582:     --------
2583:     ndarray.ndim : equivalent method
2584:     shape : dimensions of array
2585:     ndarray.shape : dimensions of array
2586: 
2587:     Examples
2588:     --------
2589:     >>> np.ndim([[1,2,3],[4,5,6]])
2590:     2
2591:     >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
2592:     2
2593:     >>> np.ndim(1)
2594:     0
2595: 
2596:     '''
2597:     try:
2598:         return a.ndim
2599:     except AttributeError:
2600:         return asarray(a).ndim
2601: 
2602: 
2603: def rank(a):
2604:     '''
2605:     Return the number of dimensions of an array.
2606: 
2607:     If `a` is not already an array, a conversion is attempted.
2608:     Scalars are zero dimensional.
2609: 
2610:     .. note::
2611:         This function is deprecated in NumPy 1.9 to avoid confusion with
2612:         `numpy.linalg.matrix_rank`. The ``ndim`` attribute or function
2613:         should be used instead.
2614: 
2615:     Parameters
2616:     ----------
2617:     a : array_like
2618:         Array whose number of dimensions is desired. If `a` is not an array,
2619:         a conversion is attempted.
2620: 
2621:     Returns
2622:     -------
2623:     number_of_dimensions : int
2624:         The number of dimensions in the array.
2625: 
2626:     See Also
2627:     --------
2628:     ndim : equivalent function
2629:     ndarray.ndim : equivalent property
2630:     shape : dimensions of array
2631:     ndarray.shape : dimensions of array
2632: 
2633:     Notes
2634:     -----
2635:     In the old Numeric package, `rank` was the term used for the number of
2636:     dimensions, but in Numpy `ndim` is used instead.
2637: 
2638:     Examples
2639:     --------
2640:     >>> np.rank([1,2,3])
2641:     1
2642:     >>> np.rank(np.array([[1,2,3],[4,5,6]]))
2643:     2
2644:     >>> np.rank(1)
2645:     0
2646: 
2647:     '''
2648:     # 2014-04-12, 1.9
2649:     warnings.warn(
2650:         "`rank` is deprecated; use the `ndim` attribute or function instead. "
2651:         "To find the rank of a matrix see `numpy.linalg.matrix_rank`.",
2652:         VisibleDeprecationWarning)
2653:     try:
2654:         return a.ndim
2655:     except AttributeError:
2656:         return asarray(a).ndim
2657: 
2658: 
2659: def size(a, axis=None):
2660:     '''
2661:     Return the number of elements along a given axis.
2662: 
2663:     Parameters
2664:     ----------
2665:     a : array_like
2666:         Input data.
2667:     axis : int, optional
2668:         Axis along which the elements are counted.  By default, give
2669:         the total number of elements.
2670: 
2671:     Returns
2672:     -------
2673:     element_count : int
2674:         Number of elements along the specified axis.
2675: 
2676:     See Also
2677:     --------
2678:     shape : dimensions of array
2679:     ndarray.shape : dimensions of array
2680:     ndarray.size : number of elements in array
2681: 
2682:     Examples
2683:     --------
2684:     >>> a = np.array([[1,2,3],[4,5,6]])
2685:     >>> np.size(a)
2686:     6
2687:     >>> np.size(a,1)
2688:     3
2689:     >>> np.size(a,0)
2690:     2
2691: 
2692:     '''
2693:     if axis is None:
2694:         try:
2695:             return a.size
2696:         except AttributeError:
2697:             return asarray(a).size
2698:     else:
2699:         try:
2700:             return a.shape[axis]
2701:         except AttributeError:
2702:             return asarray(a).shape[axis]
2703: 
2704: 
2705: def around(a, decimals=0, out=None):
2706:     '''
2707:     Evenly round to the given number of decimals.
2708: 
2709:     Parameters
2710:     ----------
2711:     a : array_like
2712:         Input data.
2713:     decimals : int, optional
2714:         Number of decimal places to round to (default: 0).  If
2715:         decimals is negative, it specifies the number of positions to
2716:         the left of the decimal point.
2717:     out : ndarray, optional
2718:         Alternative output array in which to place the result. It must have
2719:         the same shape as the expected output, but the type of the output
2720:         values will be cast if necessary. See `doc.ufuncs` (Section
2721:         "Output arguments") for details.
2722: 
2723:     Returns
2724:     -------
2725:     rounded_array : ndarray
2726:         An array of the same type as `a`, containing the rounded values.
2727:         Unless `out` was specified, a new array is created.  A reference to
2728:         the result is returned.
2729: 
2730:         The real and imaginary parts of complex numbers are rounded
2731:         separately.  The result of rounding a float is a float.
2732: 
2733:     See Also
2734:     --------
2735:     ndarray.round : equivalent method
2736: 
2737:     ceil, fix, floor, rint, trunc
2738: 
2739: 
2740:     Notes
2741:     -----
2742:     For values exactly halfway between rounded decimal values, Numpy
2743:     rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
2744:     -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due
2745:     to the inexact representation of decimal fractions in the IEEE
2746:     floating point standard [1]_ and errors introduced when scaling
2747:     by powers of ten.
2748: 
2749:     References
2750:     ----------
2751:     .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,
2752:            http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF
2753:     .. [2] "How Futile are Mindless Assessments of
2754:            Roundoff in Floating-Point Computation?", William Kahan,
2755:            http://www.cs.berkeley.edu/~wkahan/Mindless.pdf
2756: 
2757:     Examples
2758:     --------
2759:     >>> np.around([0.37, 1.64])
2760:     array([ 0.,  2.])
2761:     >>> np.around([0.37, 1.64], decimals=1)
2762:     array([ 0.4,  1.6])
2763:     >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
2764:     array([ 0.,  2.,  2.,  4.,  4.])
2765:     >>> np.around([1,2,3,11], decimals=1) # ndarray of ints is returned
2766:     array([ 1,  2,  3, 11])
2767:     >>> np.around([1,2,3,11], decimals=-1)
2768:     array([ 0,  0,  0, 10])
2769: 
2770:     '''
2771:     try:
2772:         round = a.round
2773:     except AttributeError:
2774:         return _wrapit(a, 'round', decimals, out)
2775:     return round(decimals, out)
2776: 
2777: 
2778: def round_(a, decimals=0, out=None):
2779:     '''
2780:     Round an array to the given number of decimals.
2781: 
2782:     Refer to `around` for full documentation.
2783: 
2784:     See Also
2785:     --------
2786:     around : equivalent function
2787: 
2788:     '''
2789:     try:
2790:         round = a.round
2791:     except AttributeError:
2792:         return _wrapit(a, 'round', decimals, out)
2793:     return round(decimals, out)
2794: 
2795: 
2796: def mean(a, axis=None, dtype=None, out=None, keepdims=False):
2797:     '''
2798:     Compute the arithmetic mean along the specified axis.
2799: 
2800:     Returns the average of the array elements.  The average is taken over
2801:     the flattened array by default, otherwise over the specified axis.
2802:     `float64` intermediate and return values are used for integer inputs.
2803: 
2804:     Parameters
2805:     ----------
2806:     a : array_like
2807:         Array containing numbers whose mean is desired. If `a` is not an
2808:         array, a conversion is attempted.
2809:     axis : None or int or tuple of ints, optional
2810:         Axis or axes along which the means are computed. The default is to
2811:         compute the mean of the flattened array.
2812: 
2813:         .. versionadded: 1.7.0
2814: 
2815:         If this is a tuple of ints, a mean is performed over multiple axes,
2816:         instead of a single axis or all the axes as before.
2817:     dtype : data-type, optional
2818:         Type to use in computing the mean.  For integer inputs, the default
2819:         is `float64`; for floating point inputs, it is the same as the
2820:         input dtype.
2821:     out : ndarray, optional
2822:         Alternate output array in which to place the result.  The default
2823:         is ``None``; if provided, it must have the same shape as the
2824:         expected output, but the type will be cast if necessary.
2825:         See `doc.ufuncs` for details.
2826:     keepdims : bool, optional
2827:         If this is set to True, the axes which are reduced are left
2828:         in the result as dimensions with size one. With this option,
2829:         the result will broadcast correctly against the original `arr`.
2830: 
2831:     Returns
2832:     -------
2833:     m : ndarray, see dtype parameter above
2834:         If `out=None`, returns a new array containing the mean values,
2835:         otherwise a reference to the output array is returned.
2836: 
2837:     See Also
2838:     --------
2839:     average : Weighted average
2840:     std, var, nanmean, nanstd, nanvar
2841: 
2842:     Notes
2843:     -----
2844:     The arithmetic mean is the sum of the elements along the axis divided
2845:     by the number of elements.
2846: 
2847:     Note that for floating-point input, the mean is computed using the
2848:     same precision the input has.  Depending on the input data, this can
2849:     cause the results to be inaccurate, especially for `float32` (see
2850:     example below).  Specifying a higher-precision accumulator using the
2851:     `dtype` keyword can alleviate this issue.
2852: 
2853:     Examples
2854:     --------
2855:     >>> a = np.array([[1, 2], [3, 4]])
2856:     >>> np.mean(a)
2857:     2.5
2858:     >>> np.mean(a, axis=0)
2859:     array([ 2.,  3.])
2860:     >>> np.mean(a, axis=1)
2861:     array([ 1.5,  3.5])
2862: 
2863:     In single precision, `mean` can be inaccurate:
2864: 
2865:     >>> a = np.zeros((2, 512*512), dtype=np.float32)
2866:     >>> a[0, :] = 1.0
2867:     >>> a[1, :] = 0.1
2868:     >>> np.mean(a)
2869:     0.546875
2870: 
2871:     Computing the mean in float64 is more accurate:
2872: 
2873:     >>> np.mean(a, dtype=np.float64)
2874:     0.55000000074505806
2875: 
2876:     '''
2877:     if type(a) is not mu.ndarray:
2878:         try:
2879:             mean = a.mean
2880:             return mean(axis=axis, dtype=dtype, out=out)
2881:         except AttributeError:
2882:             pass
2883: 
2884:     return _methods._mean(a, axis=axis, dtype=dtype,
2885:                           out=out, keepdims=keepdims)
2886: 
2887: 
2888: def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
2889:     '''
2890:     Compute the standard deviation along the specified axis.
2891: 
2892:     Returns the standard deviation, a measure of the spread of a distribution,
2893:     of the array elements. The standard deviation is computed for the
2894:     flattened array by default, otherwise over the specified axis.
2895: 
2896:     Parameters
2897:     ----------
2898:     a : array_like
2899:         Calculate the standard deviation of these values.
2900:     axis : None or int or tuple of ints, optional
2901:         Axis or axes along which the standard deviation is computed. The
2902:         default is to compute the standard deviation of the flattened array.
2903: 
2904:         .. versionadded: 1.7.0
2905: 
2906:         If this is a tuple of ints, a standard deviation is performed over
2907:         multiple axes, instead of a single axis or all the axes as before.
2908:     dtype : dtype, optional
2909:         Type to use in computing the standard deviation. For arrays of
2910:         integer type the default is float64, for arrays of float types it is
2911:         the same as the array type.
2912:     out : ndarray, optional
2913:         Alternative output array in which to place the result. It must have
2914:         the same shape as the expected output but the type (of the calculated
2915:         values) will be cast if necessary.
2916:     ddof : int, optional
2917:         Means Delta Degrees of Freedom.  The divisor used in calculations
2918:         is ``N - ddof``, where ``N`` represents the number of elements.
2919:         By default `ddof` is zero.
2920:     keepdims : bool, optional
2921:         If this is set to True, the axes which are reduced are left
2922:         in the result as dimensions with size one. With this option,
2923:         the result will broadcast correctly against the original `arr`.
2924: 
2925:     Returns
2926:     -------
2927:     standard_deviation : ndarray, see dtype parameter above.
2928:         If `out` is None, return a new array containing the standard deviation,
2929:         otherwise return a reference to the output array.
2930: 
2931:     See Also
2932:     --------
2933:     var, mean, nanmean, nanstd, nanvar
2934:     numpy.doc.ufuncs : Section "Output arguments"
2935: 
2936:     Notes
2937:     -----
2938:     The standard deviation is the square root of the average of the squared
2939:     deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.
2940: 
2941:     The average squared deviation is normally calculated as
2942:     ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
2943:     the divisor ``N - ddof`` is used instead. In standard statistical
2944:     practice, ``ddof=1`` provides an unbiased estimator of the variance
2945:     of the infinite population. ``ddof=0`` provides a maximum likelihood
2946:     estimate of the variance for normally distributed variables. The
2947:     standard deviation computed in this function is the square root of
2948:     the estimated variance, so even with ``ddof=1``, it will not be an
2949:     unbiased estimate of the standard deviation per se.
2950: 
2951:     Note that, for complex numbers, `std` takes the absolute
2952:     value before squaring, so that the result is always real and nonnegative.
2953: 
2954:     For floating-point input, the *std* is computed using the same
2955:     precision the input has. Depending on the input data, this can cause
2956:     the results to be inaccurate, especially for float32 (see example below).
2957:     Specifying a higher-accuracy accumulator using the `dtype` keyword can
2958:     alleviate this issue.
2959: 
2960:     Examples
2961:     --------
2962:     >>> a = np.array([[1, 2], [3, 4]])
2963:     >>> np.std(a)
2964:     1.1180339887498949
2965:     >>> np.std(a, axis=0)
2966:     array([ 1.,  1.])
2967:     >>> np.std(a, axis=1)
2968:     array([ 0.5,  0.5])
2969: 
2970:     In single precision, std() can be inaccurate:
2971: 
2972:     >>> a = np.zeros((2, 512*512), dtype=np.float32)
2973:     >>> a[0, :] = 1.0
2974:     >>> a[1, :] = 0.1
2975:     >>> np.std(a)
2976:     0.45000005
2977: 
2978:     Computing the standard deviation in float64 is more accurate:
2979: 
2980:     >>> np.std(a, dtype=np.float64)
2981:     0.44999999925494177
2982: 
2983:     '''
2984:     if type(a) is not mu.ndarray:
2985:         try:
2986:             std = a.std
2987:             return std(axis=axis, dtype=dtype, out=out, ddof=ddof)
2988:         except AttributeError:
2989:             pass
2990: 
2991:     return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
2992:                          keepdims=keepdims)
2993: 
2994: 
2995: def var(a, axis=None, dtype=None, out=None, ddof=0,
2996:         keepdims=False):
2997:     '''
2998:     Compute the variance along the specified axis.
2999: 
3000:     Returns the variance of the array elements, a measure of the spread of a
3001:     distribution.  The variance is computed for the flattened array by
3002:     default, otherwise over the specified axis.
3003: 
3004:     Parameters
3005:     ----------
3006:     a : array_like
3007:         Array containing numbers whose variance is desired.  If `a` is not an
3008:         array, a conversion is attempted.
3009:     axis : None or int or tuple of ints, optional
3010:         Axis or axes along which the variance is computed.  The default is to
3011:         compute the variance of the flattened array.
3012: 
3013:         .. versionadded: 1.7.0
3014: 
3015:         If this is a tuple of ints, a variance is performed over multiple axes,
3016:         instead of a single axis or all the axes as before.
3017:     dtype : data-type, optional
3018:         Type to use in computing the variance.  For arrays of integer type
3019:         the default is `float32`; for arrays of float types it is the same as
3020:         the array type.
3021:     out : ndarray, optional
3022:         Alternate output array in which to place the result.  It must have
3023:         the same shape as the expected output, but the type is cast if
3024:         necessary.
3025:     ddof : int, optional
3026:         "Delta Degrees of Freedom": the divisor used in the calculation is
3027:         ``N - ddof``, where ``N`` represents the number of elements. By
3028:         default `ddof` is zero.
3029:     keepdims : bool, optional
3030:         If this is set to True, the axes which are reduced are left
3031:         in the result as dimensions with size one. With this option,
3032:         the result will broadcast correctly against the original `arr`.
3033: 
3034:     Returns
3035:     -------
3036:     variance : ndarray, see dtype parameter above
3037:         If ``out=None``, returns a new array containing the variance;
3038:         otherwise, a reference to the output array is returned.
3039: 
3040:     See Also
3041:     --------
3042:     std , mean, nanmean, nanstd, nanvar
3043:     numpy.doc.ufuncs : Section "Output arguments"
3044: 
3045:     Notes
3046:     -----
3047:     The variance is the average of the squared deviations from the mean,
3048:     i.e.,  ``var = mean(abs(x - x.mean())**2)``.
3049: 
3050:     The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
3051:     If, however, `ddof` is specified, the divisor ``N - ddof`` is used
3052:     instead.  In standard statistical practice, ``ddof=1`` provides an
3053:     unbiased estimator of the variance of a hypothetical infinite population.
3054:     ``ddof=0`` provides a maximum likelihood estimate of the variance for
3055:     normally distributed variables.
3056: 
3057:     Note that for complex numbers, the absolute value is taken before
3058:     squaring, so that the result is always real and nonnegative.
3059: 
3060:     For floating-point input, the variance is computed using the same
3061:     precision the input has.  Depending on the input data, this can cause
3062:     the results to be inaccurate, especially for `float32` (see example
3063:     below).  Specifying a higher-accuracy accumulator using the ``dtype``
3064:     keyword can alleviate this issue.
3065: 
3066:     Examples
3067:     --------
3068:     >>> a = np.array([[1, 2], [3, 4]])
3069:     >>> np.var(a)
3070:     1.25
3071:     >>> np.var(a, axis=0)
3072:     array([ 1.,  1.])
3073:     >>> np.var(a, axis=1)
3074:     array([ 0.25,  0.25])
3075: 
3076:     In single precision, var() can be inaccurate:
3077: 
3078:     >>> a = np.zeros((2, 512*512), dtype=np.float32)
3079:     >>> a[0, :] = 1.0
3080:     >>> a[1, :] = 0.1
3081:     >>> np.var(a)
3082:     0.20250003
3083: 
3084:     Computing the variance in float64 is more accurate:
3085: 
3086:     >>> np.var(a, dtype=np.float64)
3087:     0.20249999932944759
3088:     >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
3089:     0.2025
3090: 
3091:     '''
3092:     if type(a) is not mu.ndarray:
3093:         try:
3094:             var = a.var
3095:             return var(axis=axis, dtype=dtype, out=out, ddof=ddof)
3096:         except AttributeError:
3097:             pass
3098: 
3099:     return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
3100:                          keepdims=keepdims)
3101: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_3929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Module containing non-deprecated functions borrowed from Numeric.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import types' statement (line 6)
import types

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import warnings' statement (line 7)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3930 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_3930) is not StypyTypeError):

    if (import_3930 != 'pyd_module'):
        __import__(import_3930)
        sys_modules_3931 = sys.modules[import_3930]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_3931.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_3930)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import VisibleDeprecationWarning' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3932 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_3932) is not StypyTypeError):

    if (import_3932 != 'pyd_module'):
        __import__(import_3932)
        sys_modules_3933 = sys.modules[import_3932]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_3933.module_type_store, module_type_store, ['VisibleDeprecationWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_3933, sys_modules_3933.module_type_store, module_type_store)
    else:
        from numpy import VisibleDeprecationWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['VisibleDeprecationWarning'], [VisibleDeprecationWarning])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_3932)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.core import mu' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3934 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core')

if (type(import_3934) is not StypyTypeError):

    if (import_3934 != 'pyd_module'):
        __import__(import_3934)
        sys_modules_3935 = sys.modules[import_3934]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', sys_modules_3935.module_type_store, module_type_store, ['multiarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_3935, sys_modules_3935.module_type_store, module_type_store)
    else:
        from numpy.core import multiarray as mu

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', None, module_type_store, ['multiarray'], [mu])

else:
    # Assigning a type to the variable 'numpy.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core', import_3934)

# Adding an alias
module_type_store.add_alias('mu', 'multiarray')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.core import um' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3936 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core')

if (type(import_3936) is not StypyTypeError):

    if (import_3936 != 'pyd_module'):
        __import__(import_3936)
        sys_modules_3937 = sys.modules[import_3936]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core', sys_modules_3937.module_type_store, module_type_store, ['umath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_3937, sys_modules_3937.module_type_store, module_type_store)
    else:
        from numpy.core import umath as um

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core', None, module_type_store, ['umath'], [um])

else:
    # Assigning a type to the variable 'numpy.core' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.core', import_3936)

# Adding an alias
module_type_store.add_alias('um', 'umath')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.core import nt' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3938 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core')

if (type(import_3938) is not StypyTypeError):

    if (import_3938 != 'pyd_module'):
        __import__(import_3938)
        sys_modules_3939 = sys.modules[import_3938]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', sys_modules_3939.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_3939, sys_modules_3939.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as nt

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [nt])

else:
    # Assigning a type to the variable 'numpy.core' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.core', import_3938)

# Adding an alias
module_type_store.add_alias('nt', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.core.numeric import asarray, array, asanyarray, concatenate' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3940 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.numeric')

if (type(import_3940) is not StypyTypeError):

    if (import_3940 != 'pyd_module'):
        __import__(import_3940)
        sys_modules_3941 = sys.modules[import_3940]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.numeric', sys_modules_3941.module_type_store, module_type_store, ['asarray', 'array', 'asanyarray', 'concatenate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_3941, sys_modules_3941.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, array, asanyarray, concatenate

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'array', 'asanyarray', 'concatenate'], [asarray, array, asanyarray, concatenate])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.core.numeric', import_3940)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.core import _methods' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_3942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core')

if (type(import_3942) is not StypyTypeError):

    if (import_3942 != 'pyd_module'):
        __import__(import_3942)
        sys_modules_3943 = sys.modules[import_3942]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', sys_modules_3943.module_type_store, module_type_store, ['_methods'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_3943, sys_modules_3943.module_type_store, module_type_store)
    else:
        from numpy.core import _methods

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', None, module_type_store, ['_methods'], [_methods])

else:
    # Assigning a type to the variable 'numpy.core' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.core', import_3942)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a Attribute to a Name (line 18):
# Getting the type of 'nt' (line 18)
nt_3944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'nt')
# Obtaining the member 'sctype2char' of a type (line 18)
sctype2char_3945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 7), nt_3944, 'sctype2char')
# Assigning a type to the variable '_dt_' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_dt_', sctype2char_3945)

# Assigning a List to a Name (line 22):
__all__ = ['alen', 'all', 'alltrue', 'amax', 'amin', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip', 'compress', 'cumprod', 'cumproduct', 'cumsum', 'diagonal', 'mean', 'ndim', 'nonzero', 'partition', 'prod', 'product', 'ptp', 'put', 'rank', 'ravel', 'repeat', 'reshape', 'resize', 'round_', 'searchsorted', 'shape', 'size', 'sometrue', 'sort', 'squeeze', 'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var']
module_type_store.set_exportable_members(['alen', 'all', 'alltrue', 'amax', 'amin', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip', 'compress', 'cumprod', 'cumproduct', 'cumsum', 'diagonal', 'mean', 'ndim', 'nonzero', 'partition', 'prod', 'product', 'ptp', 'put', 'rank', 'ravel', 'repeat', 'reshape', 'resize', 'round_', 'searchsorted', 'shape', 'size', 'sometrue', 'sort', 'squeeze', 'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var'])

# Obtaining an instance of the builtin type 'list' (line 22)
list_3946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_3947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 'alen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3947)
# Adding element type (line 22)
str_3948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'str', 'all')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3948)
# Adding element type (line 22)
str_3949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'str', 'alltrue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3949)
# Adding element type (line 22)
str_3950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'str', 'amax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3950)
# Adding element type (line 22)
str_3951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 38), 'str', 'amin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3951)
# Adding element type (line 22)
str_3952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'str', 'any')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3952)
# Adding element type (line 22)
str_3953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'str', 'argmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3953)
# Adding element type (line 22)
str_3954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'argmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3954)
# Adding element type (line 22)
str_3955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'str', 'argpartition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3955)
# Adding element type (line 22)
str_3956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'str', 'argsort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3956)
# Adding element type (line 22)
str_3957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 41), 'str', 'around')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3957)
# Adding element type (line 22)
str_3958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'str', 'choose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3958)
# Adding element type (line 22)
str_3959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 61), 'str', 'clip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3959)
# Adding element type (line 22)
str_3960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3960)
# Adding element type (line 22)
str_3961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'str', 'cumprod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3961)
# Adding element type (line 22)
str_3962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'str', 'cumproduct')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3962)
# Adding element type (line 22)
str_3963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 41), 'str', 'cumsum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3963)
# Adding element type (line 22)
str_3964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 51), 'str', 'diagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3964)
# Adding element type (line 22)
str_3965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'str', 'mean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3965)
# Adding element type (line 22)
str_3966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', 'ndim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3966)
# Adding element type (line 22)
str_3967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'str', 'nonzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3967)
# Adding element type (line 22)
str_3968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'str', 'partition')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3968)
# Adding element type (line 22)
str_3969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 36), 'str', 'prod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3969)
# Adding element type (line 22)
str_3970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 44), 'str', 'product')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3970)
# Adding element type (line 22)
str_3971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 55), 'str', 'ptp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3971)
# Adding element type (line 22)
str_3972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 62), 'str', 'put')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3972)
# Adding element type (line 22)
str_3973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'rank')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3973)
# Adding element type (line 22)
str_3974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 12), 'str', 'ravel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3974)
# Adding element type (line 22)
str_3975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'str', 'repeat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3975)
# Adding element type (line 22)
str_3976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'str', 'reshape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3976)
# Adding element type (line 22)
str_3977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'str', 'resize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3977)
# Adding element type (line 22)
str_3978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 52), 'str', 'round_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3978)
# Adding element type (line 22)
str_3979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'searchsorted')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3979)
# Adding element type (line 22)
str_3980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'str', 'shape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3980)
# Adding element type (line 22)
str_3981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', 'size')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3981)
# Adding element type (line 22)
str_3982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'str', 'sometrue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3982)
# Adding element type (line 22)
str_3983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 49), 'str', 'sort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3983)
# Adding element type (line 22)
str_3984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'str', 'squeeze')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3984)
# Adding element type (line 22)
str_3985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'std')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3985)
# Adding element type (line 22)
str_3986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', 'sum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3986)
# Adding element type (line 22)
str_3987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'str', 'swapaxes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3987)
# Adding element type (line 22)
str_3988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'str', 'take')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3988)
# Adding element type (line 22)
str_3989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'str', 'trace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3989)
# Adding element type (line 22)
str_3990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 47), 'str', 'transpose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3990)
# Adding element type (line 22)
str_3991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 60), 'str', 'var')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_3946, str_3991)

# Assigning a type to the variable '__all__' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '__all__', list_3946)


# SSA begins for try-except statement (line 33)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Attribute to a Name (line 34):
# Getting the type of 'types' (line 34)
types_3992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'types')
# Obtaining the member 'GeneratorType' of a type (line 34)
GeneratorType_3993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), types_3992, 'GeneratorType')
# Assigning a type to the variable '_gentype' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), '_gentype', GeneratorType_3993)
# SSA branch for the except part of a try statement (line 33)
# SSA branch for the except 'AttributeError' branch of a try statement (line 33)
module_type_store.open_ssa_branch('except')

# Assigning a Call to a Name (line 36):

# Call to type(...): (line 36)
# Processing the call arguments (line 36)
# Getting the type of 'None' (line 36)
None_3995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'None', False)
# Processing the call keyword arguments (line 36)
kwargs_3996 = {}
# Getting the type of 'type' (line 36)
type_3994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'type', False)
# Calling type(args, kwargs) (line 36)
type_call_result_3997 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), type_3994, *[None_3995], **kwargs_3996)

# Assigning a type to the variable '_gentype' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), '_gentype', type_call_result_3997)
# SSA join for try-except statement (line 33)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 39):
# Getting the type of 'sum' (line 39)
sum_3998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'sum')
# Assigning a type to the variable '_sum_' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_sum_', sum_3998)

@norecursion
def _wrapit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_wrapit'
    module_type_store = module_type_store.open_function_context('_wrapit', 43, 0, False)
    
    # Passed parameters checking function
    _wrapit.stypy_localization = localization
    _wrapit.stypy_type_of_self = None
    _wrapit.stypy_type_store = module_type_store
    _wrapit.stypy_function_name = '_wrapit'
    _wrapit.stypy_param_names_list = ['obj', 'method']
    _wrapit.stypy_varargs_param_name = 'args'
    _wrapit.stypy_kwargs_param_name = 'kwds'
    _wrapit.stypy_call_defaults = defaults
    _wrapit.stypy_call_varargs = varargs
    _wrapit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_wrapit', ['obj', 'method'], 'args', 'kwds', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_wrapit', localization, ['obj', 'method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_wrapit(...)' code ##################

    
    
    # SSA begins for try-except statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 45):
    # Getting the type of 'obj' (line 45)
    obj_3999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'obj')
    # Obtaining the member '__array_wrap__' of a type (line 45)
    array_wrap___4000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), obj_3999, '__array_wrap__')
    # Assigning a type to the variable 'wrap' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'wrap', array_wrap___4000)
    # SSA branch for the except part of a try statement (line 44)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 44)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 47):
    # Getting the type of 'None' (line 47)
    None_4001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'None')
    # Assigning a type to the variable 'wrap' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'wrap', None_4001)
    # SSA join for try-except statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 48):
    
    # Call to (...): (line 48)
    # Getting the type of 'args' (line 48)
    args_4010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'args', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'kwds' (line 48)
    kwds_4011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 52), 'kwds', False)
    kwargs_4012 = {'kwds_4011': kwds_4011}
    
    # Call to getattr(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to asarray(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'obj' (line 48)
    obj_4004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'obj', False)
    # Processing the call keyword arguments (line 48)
    kwargs_4005 = {}
    # Getting the type of 'asarray' (line 48)
    asarray_4003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'asarray', False)
    # Calling asarray(args, kwargs) (line 48)
    asarray_call_result_4006 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), asarray_4003, *[obj_4004], **kwargs_4005)
    
    # Getting the type of 'method' (line 48)
    method_4007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'method', False)
    # Processing the call keyword arguments (line 48)
    kwargs_4008 = {}
    # Getting the type of 'getattr' (line 48)
    getattr_4002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 48)
    getattr_call_result_4009 = invoke(stypy.reporting.localization.Localization(__file__, 48, 13), getattr_4002, *[asarray_call_result_4006, method_4007], **kwargs_4008)
    
    # Calling (args, kwargs) (line 48)
    _call_result_4013 = invoke(stypy.reporting.localization.Localization(__file__, 48, 13), getattr_call_result_4009, *[args_4010], **kwargs_4012)
    
    # Assigning a type to the variable 'result' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'result', _call_result_4013)
    
    # Getting the type of 'wrap' (line 49)
    wrap_4014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'wrap')
    # Testing the type of an if condition (line 49)
    if_condition_4015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), wrap_4014)
    # Assigning a type to the variable 'if_condition_4015' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_4015', if_condition_4015)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to isinstance(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'result' (line 50)
    result_4017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'result', False)
    # Getting the type of 'mu' (line 50)
    mu_4018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'mu', False)
    # Obtaining the member 'ndarray' of a type (line 50)
    ndarray_4019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 34), mu_4018, 'ndarray')
    # Processing the call keyword arguments (line 50)
    kwargs_4020 = {}
    # Getting the type of 'isinstance' (line 50)
    isinstance_4016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 50)
    isinstance_call_result_4021 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), isinstance_4016, *[result_4017, ndarray_4019], **kwargs_4020)
    
    # Applying the 'not' unary operator (line 50)
    result_not__4022 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'not', isinstance_call_result_4021)
    
    # Testing the type of an if condition (line 50)
    if_condition_4023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_not__4022)
    # Assigning a type to the variable 'if_condition_4023' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_4023', if_condition_4023)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 51):
    
    # Call to asarray(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'result' (line 51)
    result_4025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'result', False)
    # Processing the call keyword arguments (line 51)
    kwargs_4026 = {}
    # Getting the type of 'asarray' (line 51)
    asarray_4024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'asarray', False)
    # Calling asarray(args, kwargs) (line 51)
    asarray_call_result_4027 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), asarray_4024, *[result_4025], **kwargs_4026)
    
    # Assigning a type to the variable 'result' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'result', asarray_call_result_4027)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 52):
    
    # Call to wrap(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'result' (line 52)
    result_4029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'result', False)
    # Processing the call keyword arguments (line 52)
    kwargs_4030 = {}
    # Getting the type of 'wrap' (line 52)
    wrap_4028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 'wrap', False)
    # Calling wrap(args, kwargs) (line 52)
    wrap_call_result_4031 = invoke(stypy.reporting.localization.Localization(__file__, 52, 17), wrap_4028, *[result_4029], **kwargs_4030)
    
    # Assigning a type to the variable 'result' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'result', wrap_call_result_4031)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 53)
    result_4032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', result_4032)
    
    # ################# End of '_wrapit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_wrapit' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_4033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4033)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_wrapit'
    return stypy_return_type_4033

# Assigning a type to the variable '_wrapit' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '_wrapit', _wrapit)

@norecursion
def take(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 56)
    None_4034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'None')
    # Getting the type of 'None' (line 56)
    None_4035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'None')
    str_4036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 47), 'str', 'raise')
    defaults = [None_4034, None_4035, str_4036]
    # Create a new context for function 'take'
    module_type_store = module_type_store.open_function_context('take', 56, 0, False)
    
    # Passed parameters checking function
    take.stypy_localization = localization
    take.stypy_type_of_self = None
    take.stypy_type_store = module_type_store
    take.stypy_function_name = 'take'
    take.stypy_param_names_list = ['a', 'indices', 'axis', 'out', 'mode']
    take.stypy_varargs_param_name = None
    take.stypy_kwargs_param_name = None
    take.stypy_call_defaults = defaults
    take.stypy_call_varargs = varargs
    take.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'take', ['a', 'indices', 'axis', 'out', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'take', localization, ['a', 'indices', 'axis', 'out', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'take(...)' code ##################

    str_4037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '\n    Take elements from an array along an axis.\n\n    This function does the same thing as "fancy" indexing (indexing arrays\n    using arrays); however, it can be easier to use if you need elements\n    along a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        The source array.\n    indices : array_like\n        The indices of the values to extract.\n\n        .. versionadded:: 1.8.0\n\n        Also allow scalars for indices.\n    axis : int, optional\n        The axis over which to select values. By default, the flattened\n        input array is used.\n    out : ndarray, optional\n        If provided, the result will be placed in this array. It should\n        be of the appropriate shape and dtype.\n    mode : {\'raise\', \'wrap\', \'clip\'}, optional\n        Specifies how out-of-bounds indices will behave.\n\n        * \'raise\' -- raise an error (default)\n        * \'wrap\' -- wrap around\n        * \'clip\' -- clip to the range\n\n        \'clip\' mode means that all indices that are too large are replaced\n        by the index that addresses the last element along that axis. Note\n        that this disables indexing with negative numbers.\n\n    Returns\n    -------\n    subarray : ndarray\n        The returned array has the same type as `a`.\n\n    See Also\n    --------\n    compress : Take elements using a boolean mask\n    ndarray.take : equivalent method\n\n    Examples\n    --------\n    >>> a = [4, 3, 5, 7, 6, 8]\n    >>> indices = [0, 1, 4]\n    >>> np.take(a, indices)\n    array([4, 3, 6])\n\n    In this example if `a` is an ndarray, "fancy" indexing can be used.\n\n    >>> a = np.array(a)\n    >>> a[indices]\n    array([4, 3, 6])\n\n    If `indices` is not one dimensional, the output also has these dimensions.\n\n    >>> np.take(a, [[0, 1], [2, 3]])\n    array([[4, 3],\n           [5, 7]])\n    ')
    
    
    # SSA begins for try-except statement (line 120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 121):
    # Getting the type of 'a' (line 121)
    a_4038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'a')
    # Obtaining the member 'take' of a type (line 121)
    take_4039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), a_4038, 'take')
    # Assigning a type to the variable 'take' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'take', take_4039)
    # SSA branch for the except part of a try statement (line 120)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 120)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'a' (line 123)
    a_4041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'a', False)
    str_4042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 26), 'str', 'take')
    # Getting the type of 'indices' (line 123)
    indices_4043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'indices', False)
    # Getting the type of 'axis' (line 123)
    axis_4044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'axis', False)
    # Getting the type of 'out' (line 123)
    out_4045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'out', False)
    # Getting the type of 'mode' (line 123)
    mode_4046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 54), 'mode', False)
    # Processing the call keyword arguments (line 123)
    kwargs_4047 = {}
    # Getting the type of '_wrapit' (line 123)
    _wrapit_4040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 123)
    _wrapit_call_result_4048 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), _wrapit_4040, *[a_4041, str_4042, indices_4043, axis_4044, out_4045, mode_4046], **kwargs_4047)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', _wrapit_call_result_4048)
    # SSA join for try-except statement (line 120)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to take(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'indices' (line 124)
    indices_4050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'indices', False)
    # Getting the type of 'axis' (line 124)
    axis_4051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'axis', False)
    # Getting the type of 'out' (line 124)
    out_4052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'out', False)
    # Getting the type of 'mode' (line 124)
    mode_4053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'mode', False)
    # Processing the call keyword arguments (line 124)
    kwargs_4054 = {}
    # Getting the type of 'take' (line 124)
    take_4049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'take', False)
    # Calling take(args, kwargs) (line 124)
    take_call_result_4055 = invoke(stypy.reporting.localization.Localization(__file__, 124, 11), take_4049, *[indices_4050, axis_4051, out_4052, mode_4053], **kwargs_4054)
    
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type', take_call_result_4055)
    
    # ################# End of 'take(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'take' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_4056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'take'
    return stypy_return_type_4056

# Assigning a type to the variable 'take' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'take', take)

@norecursion
def reshape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_4057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'str', 'C')
    defaults = [str_4057]
    # Create a new context for function 'reshape'
    module_type_store = module_type_store.open_function_context('reshape', 128, 0, False)
    
    # Passed parameters checking function
    reshape.stypy_localization = localization
    reshape.stypy_type_of_self = None
    reshape.stypy_type_store = module_type_store
    reshape.stypy_function_name = 'reshape'
    reshape.stypy_param_names_list = ['a', 'newshape', 'order']
    reshape.stypy_varargs_param_name = None
    reshape.stypy_kwargs_param_name = None
    reshape.stypy_call_defaults = defaults
    reshape.stypy_call_varargs = varargs
    reshape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reshape', ['a', 'newshape', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reshape', localization, ['a', 'newshape', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reshape(...)' code ##################

    str_4058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', "\n    Gives a new shape to an array without changing its data.\n\n    Parameters\n    ----------\n    a : array_like\n        Array to be reshaped.\n    newshape : int or tuple of ints\n        The new shape should be compatible with the original shape. If\n        an integer, then the result will be a 1-D array of that length.\n        One shape dimension can be -1. In this case, the value is inferred\n        from the length of the array and remaining dimensions.\n    order : {'C', 'F', 'A'}, optional\n        Read the elements of `a` using this index order, and place the elements\n        into the reshaped array using this index order.  'C' means to\n        read / write the elements using C-like index order, with the last axis\n        index changing fastest, back to the first axis index changing slowest.\n        'F' means to read / write the elements using Fortran-like index order,\n        with the first index changing fastest, and the last index changing\n        slowest.\n        Note that the 'C' and 'F' options take no account of the memory layout\n        of the underlying array, and only refer to the order of indexing.  'A'\n        means to read / write the elements in Fortran-like index order if `a`\n        is Fortran *contiguous* in memory, C-like order otherwise.\n\n    Returns\n    -------\n    reshaped_array : ndarray\n        This will be a new view object if possible; otherwise, it will\n        be a copy.  Note there is no guarantee of the *memory layout* (C- or\n        Fortran- contiguous) of the returned array.\n\n    See Also\n    --------\n    ndarray.reshape : Equivalent method.\n\n    Notes\n    -----\n    It is not always possible to change the shape of an array without\n    copying the data. If you want an error to be raise if the data is copied,\n    you should assign the new shape to the shape attribute of the array::\n\n     >>> a = np.zeros((10, 2))\n     # A transpose make the array non-contiguous\n     >>> b = a.T\n     # Taking a view makes it possible to modify the shape without modifying\n     # the initial object.\n     >>> c = b.view()\n     >>> c.shape = (20)\n     AttributeError: incompatible shape for a non-contiguous array\n\n    The `order` keyword gives the index ordering both for *fetching* the values\n    from `a`, and then *placing* the values into the output array.\n    For example, let's say you have an array:\n\n    >>> a = np.arange(6).reshape((3, 2))\n    >>> a\n    array([[0, 1],\n           [2, 3],\n           [4, 5]])\n\n    You can think of reshaping as first raveling the array (using the given\n    index order), then inserting the elements from the raveled array into the\n    new array using the same kind of index ordering as was used for the\n    raveling.\n\n    >>> np.reshape(a, (2, 3)) # C-like index ordering\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering\n    array([[0, 4, 3],\n           [2, 1, 5]])\n    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')\n    array([[0, 4, 3],\n           [2, 1, 5]])\n\n    Examples\n    --------\n    >>> a = np.array([[1,2,3], [4,5,6]])\n    >>> np.reshape(a, 6)\n    array([1, 2, 3, 4, 5, 6])\n    >>> np.reshape(a, 6, order='F')\n    array([1, 4, 2, 5, 3, 6])\n\n    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2\n    array([[1, 2],\n           [3, 4],\n           [5, 6]])\n    ")
    
    
    # SSA begins for try-except statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 222):
    # Getting the type of 'a' (line 222)
    a_4059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 18), 'a')
    # Obtaining the member 'reshape' of a type (line 222)
    reshape_4060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 18), a_4059, 'reshape')
    # Assigning a type to the variable 'reshape' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'reshape', reshape_4060)
    # SSA branch for the except part of a try statement (line 221)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 221)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'a' (line 224)
    a_4062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'a', False)
    str_4063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 26), 'str', 'reshape')
    # Getting the type of 'newshape' (line 224)
    newshape_4064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'newshape', False)
    # Processing the call keyword arguments (line 224)
    # Getting the type of 'order' (line 224)
    order_4065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 53), 'order', False)
    keyword_4066 = order_4065
    kwargs_4067 = {'order': keyword_4066}
    # Getting the type of '_wrapit' (line 224)
    _wrapit_4061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 224)
    _wrapit_call_result_4068 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), _wrapit_4061, *[a_4062, str_4063, newshape_4064], **kwargs_4067)
    
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', _wrapit_call_result_4068)
    # SSA join for try-except statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'newshape' (line 225)
    newshape_4070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'newshape', False)
    # Processing the call keyword arguments (line 225)
    # Getting the type of 'order' (line 225)
    order_4071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 'order', False)
    keyword_4072 = order_4071
    kwargs_4073 = {'order': keyword_4072}
    # Getting the type of 'reshape' (line 225)
    reshape_4069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'reshape', False)
    # Calling reshape(args, kwargs) (line 225)
    reshape_call_result_4074 = invoke(stypy.reporting.localization.Localization(__file__, 225, 11), reshape_4069, *[newshape_4070], **kwargs_4073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type', reshape_call_result_4074)
    
    # ################# End of 'reshape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reshape' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_4075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reshape'
    return stypy_return_type_4075

# Assigning a type to the variable 'reshape' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'reshape', reshape)

@norecursion
def choose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 228)
    None_4076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 27), 'None')
    str_4077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 38), 'str', 'raise')
    defaults = [None_4076, str_4077]
    # Create a new context for function 'choose'
    module_type_store = module_type_store.open_function_context('choose', 228, 0, False)
    
    # Passed parameters checking function
    choose.stypy_localization = localization
    choose.stypy_type_of_self = None
    choose.stypy_type_store = module_type_store
    choose.stypy_function_name = 'choose'
    choose.stypy_param_names_list = ['a', 'choices', 'out', 'mode']
    choose.stypy_varargs_param_name = None
    choose.stypy_kwargs_param_name = None
    choose.stypy_call_defaults = defaults
    choose.stypy_call_varargs = varargs
    choose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'choose', ['a', 'choices', 'out', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'choose', localization, ['a', 'choices', 'out', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'choose(...)' code ##################

    str_4078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'str', '\n    Construct an array from an index array and a set of arrays to choose from.\n\n    First of all, if confused or uncertain, definitely look at the Examples -\n    in its full generality, this function is less simple than it might\n    seem from the following code description (below ndi =\n    `numpy.lib.index_tricks`):\n\n    ``np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)])``.\n\n    But this omits some subtleties.  Here is a fully general summary:\n\n    Given an "index" array (`a`) of integers and a sequence of `n` arrays\n    (`choices`), `a` and each choice array are first broadcast, as necessary,\n    to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =\n    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``\n    for each `i`.  Then, a new array with shape ``Ba.shape`` is created as\n    follows:\n\n    * if ``mode=raise`` (the default), then, first of all, each element of\n      `a` (and thus `Ba`) must be in the range `[0, n-1]`; now, suppose that\n      `i` (in that range) is the value at the `(j0, j1, ..., jm)` position\n      in `Ba` - then the value at the same position in the new array is the\n      value in `Bchoices[i]` at that same position;\n\n    * if ``mode=wrap``, values in `a` (and thus `Ba`) may be any (signed)\n      integer; modular arithmetic is used to map integers outside the range\n      `[0, n-1]` back into that range; and then the new array is constructed\n      as above;\n\n    * if ``mode=clip``, values in `a` (and thus `Ba`) may be any (signed)\n      integer; negative integers are mapped to 0; values greater than `n-1`\n      are mapped to `n-1`; and then the new array is constructed as above.\n\n    Parameters\n    ----------\n    a : int array\n        This array must contain integers in `[0, n-1]`, where `n` is the number\n        of choices, unless ``mode=wrap`` or ``mode=clip``, in which cases any\n        integers are permissible.\n    choices : sequence of arrays\n        Choice arrays. `a` and all of the choices must be broadcastable to the\n        same shape.  If `choices` is itself an array (not recommended), then\n        its outermost dimension (i.e., the one corresponding to\n        ``choices.shape[0]``) is taken as defining the "sequence".\n    out : array, optional\n        If provided, the result will be inserted into this array. It should\n        be of the appropriate shape and dtype.\n    mode : {\'raise\' (default), \'wrap\', \'clip\'}, optional\n        Specifies how indices outside `[0, n-1]` will be treated:\n\n          * \'raise\' : an exception is raised\n          * \'wrap\' : value becomes value mod `n`\n          * \'clip\' : values < 0 are mapped to 0, values > n-1 are mapped to n-1\n\n    Returns\n    -------\n    merged_array : array\n        The merged result.\n\n    Raises\n    ------\n    ValueError: shape mismatch\n        If `a` and each choice array are not all broadcastable to the same\n        shape.\n\n    See Also\n    --------\n    ndarray.choose : equivalent method\n\n    Notes\n    -----\n    To reduce the chance of misinterpretation, even though the following\n    "abuse" is nominally supported, `choices` should neither be, nor be\n    thought of as, a single array, i.e., the outermost sequence-like container\n    should be either a list or a tuple.\n\n    Examples\n    --------\n\n    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],\n    ...   [20, 21, 22, 23], [30, 31, 32, 33]]\n    >>> np.choose([2, 3, 1, 0], choices\n    ... # the first element of the result will be the first element of the\n    ... # third (2+1) "array" in choices, namely, 20; the second element\n    ... # will be the second element of the fourth (3+1) choice array, i.e.,\n    ... # 31, etc.\n    ... )\n    array([20, 31, 12,  3])\n    >>> np.choose([2, 4, 1, 0], choices, mode=\'clip\') # 4 goes to 3 (4-1)\n    array([20, 31, 12,  3])\n    >>> # because there are 4 choice arrays\n    >>> np.choose([2, 4, 1, 0], choices, mode=\'wrap\') # 4 goes to (4 mod 4)\n    array([20,  1, 12,  3])\n    >>> # i.e., 0\n\n    A couple examples illustrating how choose broadcasts:\n\n    >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]\n    >>> choices = [-10, 10]\n    >>> np.choose(a, choices)\n    array([[ 10, -10,  10],\n           [-10,  10, -10],\n           [ 10, -10,  10]])\n\n    >>> # With thanks to Anne Archibald\n    >>> a = np.array([0, 1]).reshape((2,1,1))\n    >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))\n    >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))\n    >>> np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2\n    array([[[ 1,  1,  1,  1,  1],\n            [ 2,  2,  2,  2,  2],\n            [ 3,  3,  3,  3,  3]],\n           [[-1, -2, -3, -4, -5],\n            [-1, -2, -3, -4, -5],\n            [-1, -2, -3, -4, -5]]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 348):
    # Getting the type of 'a' (line 348)
    a_4079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'a')
    # Obtaining the member 'choose' of a type (line 348)
    choose_4080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), a_4079, 'choose')
    # Assigning a type to the variable 'choose' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'choose', choose_4080)
    # SSA branch for the except part of a try statement (line 347)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 347)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'a' (line 350)
    a_4082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 23), 'a', False)
    str_4083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 26), 'str', 'choose')
    # Getting the type of 'choices' (line 350)
    choices_4084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'choices', False)
    # Processing the call keyword arguments (line 350)
    # Getting the type of 'out' (line 350)
    out_4085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'out', False)
    keyword_4086 = out_4085
    # Getting the type of 'mode' (line 350)
    mode_4087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 59), 'mode', False)
    keyword_4088 = mode_4087
    kwargs_4089 = {'mode': keyword_4088, 'out': keyword_4086}
    # Getting the type of '_wrapit' (line 350)
    _wrapit_4081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 350)
    _wrapit_call_result_4090 = invoke(stypy.reporting.localization.Localization(__file__, 350, 15), _wrapit_4081, *[a_4082, str_4083, choices_4084], **kwargs_4089)
    
    # Assigning a type to the variable 'stypy_return_type' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'stypy_return_type', _wrapit_call_result_4090)
    # SSA join for try-except statement (line 347)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to choose(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'choices' (line 351)
    choices_4092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'choices', False)
    # Processing the call keyword arguments (line 351)
    # Getting the type of 'out' (line 351)
    out_4093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'out', False)
    keyword_4094 = out_4093
    # Getting the type of 'mode' (line 351)
    mode_4095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 41), 'mode', False)
    keyword_4096 = mode_4095
    kwargs_4097 = {'mode': keyword_4096, 'out': keyword_4094}
    # Getting the type of 'choose' (line 351)
    choose_4091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'choose', False)
    # Calling choose(args, kwargs) (line 351)
    choose_call_result_4098 = invoke(stypy.reporting.localization.Localization(__file__, 351, 11), choose_4091, *[choices_4092], **kwargs_4097)
    
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type', choose_call_result_4098)
    
    # ################# End of 'choose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'choose' in the type store
    # Getting the type of 'stypy_return_type' (line 228)
    stypy_return_type_4099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'choose'
    return stypy_return_type_4099

# Assigning a type to the variable 'choose' (line 228)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'choose', choose)

@norecursion
def repeat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 354)
    None_4100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 28), 'None')
    defaults = [None_4100]
    # Create a new context for function 'repeat'
    module_type_store = module_type_store.open_function_context('repeat', 354, 0, False)
    
    # Passed parameters checking function
    repeat.stypy_localization = localization
    repeat.stypy_type_of_self = None
    repeat.stypy_type_store = module_type_store
    repeat.stypy_function_name = 'repeat'
    repeat.stypy_param_names_list = ['a', 'repeats', 'axis']
    repeat.stypy_varargs_param_name = None
    repeat.stypy_kwargs_param_name = None
    repeat.stypy_call_defaults = defaults
    repeat.stypy_call_varargs = varargs
    repeat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'repeat', ['a', 'repeats', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'repeat', localization, ['a', 'repeats', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'repeat(...)' code ##################

    str_4101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, (-1)), 'str', '\n    Repeat elements of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    repeats : int or array of ints\n        The number of repetitions for each element.  `repeats` is broadcasted\n        to fit the shape of the given axis.\n    axis : int, optional\n        The axis along which to repeat values.  By default, use the\n        flattened input array, and return a flat output array.\n\n    Returns\n    -------\n    repeated_array : ndarray\n        Output array which has the same shape as `a`, except along\n        the given axis.\n\n    See Also\n    --------\n    tile : Tile an array.\n\n    Examples\n    --------\n    >>> x = np.array([[1,2],[3,4]])\n    >>> np.repeat(x, 2)\n    array([1, 1, 2, 2, 3, 3, 4, 4])\n    >>> np.repeat(x, 3, axis=1)\n    array([[1, 1, 1, 2, 2, 2],\n           [3, 3, 3, 4, 4, 4]])\n    >>> np.repeat(x, [1, 2], axis=0)\n    array([[1, 2],\n           [3, 4],\n           [3, 4]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 394):
    # Getting the type of 'a' (line 394)
    a_4102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 17), 'a')
    # Obtaining the member 'repeat' of a type (line 394)
    repeat_4103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 17), a_4102, 'repeat')
    # Assigning a type to the variable 'repeat' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'repeat', repeat_4103)
    # SSA branch for the except part of a try statement (line 393)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 393)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'a' (line 396)
    a_4105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 23), 'a', False)
    str_4106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 26), 'str', 'repeat')
    # Getting the type of 'repeats' (line 396)
    repeats_4107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 36), 'repeats', False)
    # Getting the type of 'axis' (line 396)
    axis_4108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'axis', False)
    # Processing the call keyword arguments (line 396)
    kwargs_4109 = {}
    # Getting the type of '_wrapit' (line 396)
    _wrapit_4104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 396)
    _wrapit_call_result_4110 = invoke(stypy.reporting.localization.Localization(__file__, 396, 15), _wrapit_4104, *[a_4105, str_4106, repeats_4107, axis_4108], **kwargs_4109)
    
    # Assigning a type to the variable 'stypy_return_type' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'stypy_return_type', _wrapit_call_result_4110)
    # SSA join for try-except statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to repeat(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'repeats' (line 397)
    repeats_4112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 18), 'repeats', False)
    # Getting the type of 'axis' (line 397)
    axis_4113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'axis', False)
    # Processing the call keyword arguments (line 397)
    kwargs_4114 = {}
    # Getting the type of 'repeat' (line 397)
    repeat_4111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'repeat', False)
    # Calling repeat(args, kwargs) (line 397)
    repeat_call_result_4115 = invoke(stypy.reporting.localization.Localization(__file__, 397, 11), repeat_4111, *[repeats_4112, axis_4113], **kwargs_4114)
    
    # Assigning a type to the variable 'stypy_return_type' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type', repeat_call_result_4115)
    
    # ################# End of 'repeat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'repeat' in the type store
    # Getting the type of 'stypy_return_type' (line 354)
    stypy_return_type_4116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'repeat'
    return stypy_return_type_4116

# Assigning a type to the variable 'repeat' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'repeat', repeat)

@norecursion
def put(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_4117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 24), 'str', 'raise')
    defaults = [str_4117]
    # Create a new context for function 'put'
    module_type_store = module_type_store.open_function_context('put', 400, 0, False)
    
    # Passed parameters checking function
    put.stypy_localization = localization
    put.stypy_type_of_self = None
    put.stypy_type_store = module_type_store
    put.stypy_function_name = 'put'
    put.stypy_param_names_list = ['a', 'ind', 'v', 'mode']
    put.stypy_varargs_param_name = None
    put.stypy_kwargs_param_name = None
    put.stypy_call_defaults = defaults
    put.stypy_call_varargs = varargs
    put.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'put', ['a', 'ind', 'v', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'put', localization, ['a', 'ind', 'v', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'put(...)' code ##################

    str_4118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, (-1)), 'str', "\n    Replaces specified elements of an array with given values.\n\n    The indexing works on the flattened target array. `put` is roughly\n    equivalent to:\n\n    ::\n\n        a.flat[ind] = v\n\n    Parameters\n    ----------\n    a : ndarray\n        Target array.\n    ind : array_like\n        Target indices, interpreted as integers.\n    v : array_like\n        Values to place in `a` at target indices. If `v` is shorter than\n        `ind` it will be repeated as necessary.\n    mode : {'raise', 'wrap', 'clip'}, optional\n        Specifies how out-of-bounds indices will behave.\n\n        * 'raise' -- raise an error (default)\n        * 'wrap' -- wrap around\n        * 'clip' -- clip to the range\n\n        'clip' mode means that all indices that are too large are replaced\n        by the index that addresses the last element along that axis. Note\n        that this disables indexing with negative numbers.\n\n    See Also\n    --------\n    putmask, place\n\n    Examples\n    --------\n    >>> a = np.arange(5)\n    >>> np.put(a, [0, 2], [-44, -55])\n    >>> a\n    array([-44,   1, -55,   3,   4])\n\n    >>> a = np.arange(5)\n    >>> np.put(a, 22, -5, mode='clip')\n    >>> a\n    array([ 0,  1,  2,  3, -5])\n\n    ")
    
    
    # SSA begins for try-except statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 449):
    # Getting the type of 'a' (line 449)
    a_4119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 14), 'a')
    # Obtaining the member 'put' of a type (line 449)
    put_4120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 14), a_4119, 'put')
    # Assigning a type to the variable 'put' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'put', put_4120)
    # SSA branch for the except part of a try statement (line 448)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 448)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 451)
    # Processing the call arguments (line 451)
    
    # Call to format(...): (line 451)
    # Processing the call keyword arguments (line 451)
    
    # Call to type(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'a' (line 452)
    a_4125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 54), 'a', False)
    # Processing the call keyword arguments (line 452)
    kwargs_4126 = {}
    # Getting the type of 'type' (line 452)
    type_4124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 49), 'type', False)
    # Calling type(args, kwargs) (line 452)
    type_call_result_4127 = invoke(stypy.reporting.localization.Localization(__file__, 452, 49), type_4124, *[a_4125], **kwargs_4126)
    
    # Obtaining the member '__name__' of a type (line 452)
    name___4128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 49), type_call_result_4127, '__name__')
    keyword_4129 = name___4128
    kwargs_4130 = {'name': keyword_4129}
    str_4122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 24), 'str', 'argument 1 must be numpy.ndarray, not {name}')
    # Obtaining the member 'format' of a type (line 451)
    format_4123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), str_4122, 'format')
    # Calling format(args, kwargs) (line 451)
    format_call_result_4131 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), format_4123, *[], **kwargs_4130)
    
    # Processing the call keyword arguments (line 451)
    kwargs_4132 = {}
    # Getting the type of 'TypeError' (line 451)
    TypeError_4121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 451)
    TypeError_call_result_4133 = invoke(stypy.reporting.localization.Localization(__file__, 451, 14), TypeError_4121, *[format_call_result_4131], **kwargs_4132)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 451, 8), TypeError_call_result_4133, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to put(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'ind' (line 454)
    ind_4135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'ind', False)
    # Getting the type of 'v' (line 454)
    v_4136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 20), 'v', False)
    # Getting the type of 'mode' (line 454)
    mode_4137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 23), 'mode', False)
    # Processing the call keyword arguments (line 454)
    kwargs_4138 = {}
    # Getting the type of 'put' (line 454)
    put_4134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'put', False)
    # Calling put(args, kwargs) (line 454)
    put_call_result_4139 = invoke(stypy.reporting.localization.Localization(__file__, 454, 11), put_4134, *[ind_4135, v_4136, mode_4137], **kwargs_4138)
    
    # Assigning a type to the variable 'stypy_return_type' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'stypy_return_type', put_call_result_4139)
    
    # ################# End of 'put(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'put' in the type store
    # Getting the type of 'stypy_return_type' (line 400)
    stypy_return_type_4140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'put'
    return stypy_return_type_4140

# Assigning a type to the variable 'put' (line 400)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'put', put)

@norecursion
def swapaxes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'swapaxes'
    module_type_store = module_type_store.open_function_context('swapaxes', 457, 0, False)
    
    # Passed parameters checking function
    swapaxes.stypy_localization = localization
    swapaxes.stypy_type_of_self = None
    swapaxes.stypy_type_store = module_type_store
    swapaxes.stypy_function_name = 'swapaxes'
    swapaxes.stypy_param_names_list = ['a', 'axis1', 'axis2']
    swapaxes.stypy_varargs_param_name = None
    swapaxes.stypy_kwargs_param_name = None
    swapaxes.stypy_call_defaults = defaults
    swapaxes.stypy_call_varargs = varargs
    swapaxes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'swapaxes', ['a', 'axis1', 'axis2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'swapaxes', localization, ['a', 'axis1', 'axis2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'swapaxes(...)' code ##################

    str_4141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, (-1)), 'str', '\n    Interchange two axes of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis1 : int\n        First axis.\n    axis2 : int\n        Second axis.\n\n    Returns\n    -------\n    a_swapped : ndarray\n        For Numpy >= 1.10, if `a` is an ndarray, then a view of `a` is\n        returned; otherwise a new array is created. For earlier Numpy\n        versions a view of `a` is returned only if the order of the\n        axes is changed, otherwise the input array is returned.\n\n    Examples\n    --------\n    >>> x = np.array([[1,2,3]])\n    >>> np.swapaxes(x,0,1)\n    array([[1],\n           [2],\n           [3]])\n\n    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])\n    >>> x\n    array([[[0, 1],\n            [2, 3]],\n           [[4, 5],\n            [6, 7]]])\n\n    >>> np.swapaxes(x,0,2)\n    array([[[0, 4],\n            [2, 6]],\n           [[1, 5],\n            [3, 7]]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 501):
    # Getting the type of 'a' (line 501)
    a_4142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'a')
    # Obtaining the member 'swapaxes' of a type (line 501)
    swapaxes_4143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 19), a_4142, 'swapaxes')
    # Assigning a type to the variable 'swapaxes' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'swapaxes', swapaxes_4143)
    # SSA branch for the except part of a try statement (line 500)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 500)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'a' (line 503)
    a_4145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 23), 'a', False)
    str_4146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 26), 'str', 'swapaxes')
    # Getting the type of 'axis1' (line 503)
    axis1_4147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 38), 'axis1', False)
    # Getting the type of 'axis2' (line 503)
    axis2_4148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 45), 'axis2', False)
    # Processing the call keyword arguments (line 503)
    kwargs_4149 = {}
    # Getting the type of '_wrapit' (line 503)
    _wrapit_4144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 503)
    _wrapit_call_result_4150 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), _wrapit_4144, *[a_4145, str_4146, axis1_4147, axis2_4148], **kwargs_4149)
    
    # Assigning a type to the variable 'stypy_return_type' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'stypy_return_type', _wrapit_call_result_4150)
    # SSA join for try-except statement (line 500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to swapaxes(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'axis1' (line 504)
    axis1_4152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'axis1', False)
    # Getting the type of 'axis2' (line 504)
    axis2_4153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 27), 'axis2', False)
    # Processing the call keyword arguments (line 504)
    kwargs_4154 = {}
    # Getting the type of 'swapaxes' (line 504)
    swapaxes_4151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 504)
    swapaxes_call_result_4155 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), swapaxes_4151, *[axis1_4152, axis2_4153], **kwargs_4154)
    
    # Assigning a type to the variable 'stypy_return_type' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type', swapaxes_call_result_4155)
    
    # ################# End of 'swapaxes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'swapaxes' in the type store
    # Getting the type of 'stypy_return_type' (line 457)
    stypy_return_type_4156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4156)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'swapaxes'
    return stypy_return_type_4156

# Assigning a type to the variable 'swapaxes' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 0), 'swapaxes', swapaxes)

@norecursion
def transpose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 507)
    None_4157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 22), 'None')
    defaults = [None_4157]
    # Create a new context for function 'transpose'
    module_type_store = module_type_store.open_function_context('transpose', 507, 0, False)
    
    # Passed parameters checking function
    transpose.stypy_localization = localization
    transpose.stypy_type_of_self = None
    transpose.stypy_type_store = module_type_store
    transpose.stypy_function_name = 'transpose'
    transpose.stypy_param_names_list = ['a', 'axes']
    transpose.stypy_varargs_param_name = None
    transpose.stypy_kwargs_param_name = None
    transpose.stypy_call_defaults = defaults
    transpose.stypy_call_varargs = varargs
    transpose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'transpose', ['a', 'axes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'transpose', localization, ['a', 'axes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'transpose(...)' code ##################

    str_4158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, (-1)), 'str', '\n    Permute the dimensions of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axes : list of ints, optional\n        By default, reverse the dimensions, otherwise permute the axes\n        according to the values given.\n\n    Returns\n    -------\n    p : ndarray\n        `a` with its axes permuted.  A view is returned whenever\n        possible.\n\n    See Also\n    --------\n    moveaxis\n    argsort\n\n    Notes\n    -----\n    Use `transpose(a, argsort(axes))` to invert the transposition of tensors\n    when using the `axes` keyword argument.\n\n    Transposing a 1-D array returns an unchanged view of the original array.\n\n    Examples\n    --------\n    >>> x = np.arange(4).reshape((2,2))\n    >>> x\n    array([[0, 1],\n           [2, 3]])\n\n    >>> np.transpose(x)\n    array([[0, 2],\n           [1, 3]])\n\n    >>> x = np.ones((1, 2, 3))\n    >>> np.transpose(x, (1, 0, 2)).shape\n    (2, 1, 3)\n\n    ')
    
    
    # SSA begins for try-except statement (line 553)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 554):
    # Getting the type of 'a' (line 554)
    a_4159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'a')
    # Obtaining the member 'transpose' of a type (line 554)
    transpose_4160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 20), a_4159, 'transpose')
    # Assigning a type to the variable 'transpose' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'transpose', transpose_4160)
    # SSA branch for the except part of a try statement (line 553)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 553)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'a' (line 556)
    a_4162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'a', False)
    str_4163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 26), 'str', 'transpose')
    # Getting the type of 'axes' (line 556)
    axes_4164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 39), 'axes', False)
    # Processing the call keyword arguments (line 556)
    kwargs_4165 = {}
    # Getting the type of '_wrapit' (line 556)
    _wrapit_4161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 556)
    _wrapit_call_result_4166 = invoke(stypy.reporting.localization.Localization(__file__, 556, 15), _wrapit_4161, *[a_4162, str_4163, axes_4164], **kwargs_4165)
    
    # Assigning a type to the variable 'stypy_return_type' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'stypy_return_type', _wrapit_call_result_4166)
    # SSA join for try-except statement (line 553)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to transpose(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'axes' (line 557)
    axes_4168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 21), 'axes', False)
    # Processing the call keyword arguments (line 557)
    kwargs_4169 = {}
    # Getting the type of 'transpose' (line 557)
    transpose_4167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 11), 'transpose', False)
    # Calling transpose(args, kwargs) (line 557)
    transpose_call_result_4170 = invoke(stypy.reporting.localization.Localization(__file__, 557, 11), transpose_4167, *[axes_4168], **kwargs_4169)
    
    # Assigning a type to the variable 'stypy_return_type' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'stypy_return_type', transpose_call_result_4170)
    
    # ################# End of 'transpose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'transpose' in the type store
    # Getting the type of 'stypy_return_type' (line 507)
    stypy_return_type_4171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'transpose'
    return stypy_return_type_4171

# Assigning a type to the variable 'transpose' (line 507)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'transpose', transpose)

@norecursion
def partition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 27), 'int')
    str_4173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 36), 'str', 'introselect')
    # Getting the type of 'None' (line 560)
    None_4174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 57), 'None')
    defaults = [int_4172, str_4173, None_4174]
    # Create a new context for function 'partition'
    module_type_store = module_type_store.open_function_context('partition', 560, 0, False)
    
    # Passed parameters checking function
    partition.stypy_localization = localization
    partition.stypy_type_of_self = None
    partition.stypy_type_store = module_type_store
    partition.stypy_function_name = 'partition'
    partition.stypy_param_names_list = ['a', 'kth', 'axis', 'kind', 'order']
    partition.stypy_varargs_param_name = None
    partition.stypy_kwargs_param_name = None
    partition.stypy_call_defaults = defaults
    partition.stypy_call_varargs = varargs
    partition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'partition', ['a', 'kth', 'axis', 'kind', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'partition', localization, ['a', 'kth', 'axis', 'kind', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'partition(...)' code ##################

    str_4175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, (-1)), 'str', "\n    Return a partitioned copy of an array.\n\n    Creates a copy of the array with its elements rearranged in such a way that\n    the value of the element in kth position is in the position it would be in\n    a sorted array. All elements smaller than the kth element are moved before\n    this element and all equal or greater are moved behind it. The ordering of\n    the elements in the two partitions is undefined.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        Array to be sorted.\n    kth : int or sequence of ints\n        Element index to partition by. The kth value of the element will be in\n        its final sorted position and all smaller elements will be moved before\n        it and all equal or greater elements behind it.\n        The order all elements in the partitions is undefined.\n        If provided with a sequence of kth it will partition all elements\n        indexed by kth  of them into their sorted position at once.\n    axis : int or None, optional\n        Axis along which to sort. If None, the array is flattened before\n        sorting. The default is -1, which sorts along the last axis.\n    kind : {'introselect'}, optional\n        Selection algorithm. Default is 'introselect'.\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string.  Not all fields need be specified, but\n        unspecified fields will still be used, in the order in which they\n        come up in the dtype, to break ties.\n\n    Returns\n    -------\n    partitioned_array : ndarray\n        Array of the same type and shape as `a`.\n\n    See Also\n    --------\n    ndarray.partition : Method to sort an array in-place.\n    argpartition : Indirect partition.\n    sort : Full sorting\n\n    Notes\n    -----\n    The various selection algorithms are characterized by their average speed,\n    worst case performance, work space size, and whether they are stable. A\n    stable sort keeps items with the same key in the same relative order. The\n    available algorithms have the following properties:\n\n    ================= ======= ============= ============ =======\n       kind            speed   worst case    work space  stable\n    ================= ======= ============= ============ =======\n    'introselect'        1        O(n)           0         no\n    ================= ======= ============= ============ =======\n\n    All the partition algorithms make temporary copies of the data when\n    partitioning along any but the last axis.  Consequently, partitioning\n    along the last axis is faster and uses less space than partitioning\n    along any other axis.\n\n    The sort order for complex numbers is lexicographic. If both the real\n    and imaginary parts are non-nan then the order is determined by the\n    real parts except when they are equal, in which case the order is\n    determined by the imaginary parts.\n\n    Examples\n    --------\n    >>> a = np.array([3, 4, 2, 1])\n    >>> np.partition(a, 3)\n    array([2, 1, 3, 4])\n\n    >>> np.partition(a, (1, 3))\n    array([1, 2, 3, 4])\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 639)
    # Getting the type of 'axis' (line 639)
    axis_4176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 7), 'axis')
    # Getting the type of 'None' (line 639)
    None_4177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 15), 'None')
    
    (may_be_4178, more_types_in_union_4179) = may_be_none(axis_4176, None_4177)

    if may_be_4178:

        if more_types_in_union_4179:
            # Runtime conditional SSA (line 639)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 640):
        
        # Call to flatten(...): (line 640)
        # Processing the call keyword arguments (line 640)
        kwargs_4185 = {}
        
        # Call to asanyarray(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'a' (line 640)
        a_4181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 23), 'a', False)
        # Processing the call keyword arguments (line 640)
        kwargs_4182 = {}
        # Getting the type of 'asanyarray' (line 640)
        asanyarray_4180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 12), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 640)
        asanyarray_call_result_4183 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), asanyarray_4180, *[a_4181], **kwargs_4182)
        
        # Obtaining the member 'flatten' of a type (line 640)
        flatten_4184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 12), asanyarray_call_result_4183, 'flatten')
        # Calling flatten(args, kwargs) (line 640)
        flatten_call_result_4186 = invoke(stypy.reporting.localization.Localization(__file__, 640, 12), flatten_4184, *[], **kwargs_4185)
        
        # Assigning a type to the variable 'a' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'a', flatten_call_result_4186)
        
        # Assigning a Num to a Name (line 641):
        int_4187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 15), 'int')
        # Assigning a type to the variable 'axis' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'axis', int_4187)

        if more_types_in_union_4179:
            # Runtime conditional SSA for else branch (line 639)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_4178) or more_types_in_union_4179):
        
        # Assigning a Call to a Name (line 643):
        
        # Call to copy(...): (line 643)
        # Processing the call keyword arguments (line 643)
        str_4193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 37), 'str', 'K')
        keyword_4194 = str_4193
        kwargs_4195 = {'order': keyword_4194}
        
        # Call to asanyarray(...): (line 643)
        # Processing the call arguments (line 643)
        # Getting the type of 'a' (line 643)
        a_4189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 23), 'a', False)
        # Processing the call keyword arguments (line 643)
        kwargs_4190 = {}
        # Getting the type of 'asanyarray' (line 643)
        asanyarray_4188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 12), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 643)
        asanyarray_call_result_4191 = invoke(stypy.reporting.localization.Localization(__file__, 643, 12), asanyarray_4188, *[a_4189], **kwargs_4190)
        
        # Obtaining the member 'copy' of a type (line 643)
        copy_4192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 12), asanyarray_call_result_4191, 'copy')
        # Calling copy(args, kwargs) (line 643)
        copy_call_result_4196 = invoke(stypy.reporting.localization.Localization(__file__, 643, 12), copy_4192, *[], **kwargs_4195)
        
        # Assigning a type to the variable 'a' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'a', copy_call_result_4196)

        if (may_be_4178 and more_types_in_union_4179):
            # SSA join for if statement (line 639)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to partition(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'kth' (line 644)
    kth_4199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 16), 'kth', False)
    # Processing the call keyword arguments (line 644)
    # Getting the type of 'axis' (line 644)
    axis_4200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 26), 'axis', False)
    keyword_4201 = axis_4200
    # Getting the type of 'kind' (line 644)
    kind_4202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 37), 'kind', False)
    keyword_4203 = kind_4202
    # Getting the type of 'order' (line 644)
    order_4204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 49), 'order', False)
    keyword_4205 = order_4204
    kwargs_4206 = {'kind': keyword_4203, 'order': keyword_4205, 'axis': keyword_4201}
    # Getting the type of 'a' (line 644)
    a_4197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'a', False)
    # Obtaining the member 'partition' of a type (line 644)
    partition_4198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 4), a_4197, 'partition')
    # Calling partition(args, kwargs) (line 644)
    partition_call_result_4207 = invoke(stypy.reporting.localization.Localization(__file__, 644, 4), partition_4198, *[kth_4199], **kwargs_4206)
    
    # Getting the type of 'a' (line 645)
    a_4208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 4), 'stypy_return_type', a_4208)
    
    # ################# End of 'partition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'partition' in the type store
    # Getting the type of 'stypy_return_type' (line 560)
    stypy_return_type_4209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4209)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'partition'
    return stypy_return_type_4209

# Assigning a type to the variable 'partition' (line 560)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'partition', partition)

@norecursion
def argpartition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 30), 'int')
    str_4211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 39), 'str', 'introselect')
    # Getting the type of 'None' (line 648)
    None_4212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 60), 'None')
    defaults = [int_4210, str_4211, None_4212]
    # Create a new context for function 'argpartition'
    module_type_store = module_type_store.open_function_context('argpartition', 648, 0, False)
    
    # Passed parameters checking function
    argpartition.stypy_localization = localization
    argpartition.stypy_type_of_self = None
    argpartition.stypy_type_store = module_type_store
    argpartition.stypy_function_name = 'argpartition'
    argpartition.stypy_param_names_list = ['a', 'kth', 'axis', 'kind', 'order']
    argpartition.stypy_varargs_param_name = None
    argpartition.stypy_kwargs_param_name = None
    argpartition.stypy_call_defaults = defaults
    argpartition.stypy_call_varargs = varargs
    argpartition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argpartition', ['a', 'kth', 'axis', 'kind', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argpartition', localization, ['a', 'kth', 'axis', 'kind', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argpartition(...)' code ##################

    str_4213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, (-1)), 'str', "\n    Perform an indirect partition along the given axis using the algorithm\n    specified by the `kind` keyword. It returns an array of indices of the\n    same shape as `a` that index data along the given axis in partitioned\n    order.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        Array to sort.\n    kth : int or sequence of ints\n        Element index to partition by. The kth element will be in its final\n        sorted position and all smaller elements will be moved before it and\n        all larger elements behind it.\n        The order all elements in the partitions is undefined.\n        If provided with a sequence of kth it will partition all of them into\n        their sorted position at once.\n    axis : int or None, optional\n        Axis along which to sort.  The default is -1 (the last axis). If None,\n        the flattened array is used.\n    kind : {'introselect'}, optional\n        Selection algorithm. Default is 'introselect'\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string, and not all fields need be specified,\n        but unspecified fields will still be used, in the order in which\n        they come up in the dtype, to break ties.\n\n    Returns\n    -------\n    index_array : ndarray, int\n        Array of indices that partition `a` along the specified axis.\n        In other words, ``a[index_array]`` yields a sorted `a`.\n\n    See Also\n    --------\n    partition : Describes partition algorithms used.\n    ndarray.partition : Inplace partition.\n    argsort : Full indirect sort\n\n    Notes\n    -----\n    See `partition` for notes on the different selection algorithms.\n\n    Examples\n    --------\n    One dimensional array:\n\n    >>> x = np.array([3, 4, 2, 1])\n    >>> x[np.argpartition(x, 3)]\n    array([2, 1, 3, 4])\n    >>> x[np.argpartition(x, (1, 3))]\n    array([1, 2, 3, 4])\n\n    >>> x = [3, 4, 2, 1]\n    >>> np.array(x)[np.argpartition(x, 3)]\n    array([2, 1, 3, 4])\n\n    ")
    
    
    # SSA begins for try-except statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 712):
    # Getting the type of 'a' (line 712)
    a_4214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 23), 'a')
    # Obtaining the member 'argpartition' of a type (line 712)
    argpartition_4215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 23), a_4214, 'argpartition')
    # Assigning a type to the variable 'argpartition' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'argpartition', argpartition_4215)
    # SSA branch for the except part of a try statement (line 711)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 711)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'a' (line 714)
    a_4217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 23), 'a', False)
    str_4218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 26), 'str', 'argpartition')
    # Getting the type of 'kth' (line 714)
    kth_4219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 41), 'kth', False)
    # Getting the type of 'axis' (line 714)
    axis_4220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 46), 'axis', False)
    # Getting the type of 'kind' (line 714)
    kind_4221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 52), 'kind', False)
    # Getting the type of 'order' (line 714)
    order_4222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 58), 'order', False)
    # Processing the call keyword arguments (line 714)
    kwargs_4223 = {}
    # Getting the type of '_wrapit' (line 714)
    _wrapit_4216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 714)
    _wrapit_call_result_4224 = invoke(stypy.reporting.localization.Localization(__file__, 714, 15), _wrapit_4216, *[a_4217, str_4218, kth_4219, axis_4220, kind_4221, order_4222], **kwargs_4223)
    
    # Assigning a type to the variable 'stypy_return_type' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'stypy_return_type', _wrapit_call_result_4224)
    # SSA join for try-except statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to argpartition(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 'kth' (line 715)
    kth_4226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 24), 'kth', False)
    # Getting the type of 'axis' (line 715)
    axis_4227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 29), 'axis', False)
    # Processing the call keyword arguments (line 715)
    # Getting the type of 'kind' (line 715)
    kind_4228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 40), 'kind', False)
    keyword_4229 = kind_4228
    # Getting the type of 'order' (line 715)
    order_4230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 52), 'order', False)
    keyword_4231 = order_4230
    kwargs_4232 = {'kind': keyword_4229, 'order': keyword_4231}
    # Getting the type of 'argpartition' (line 715)
    argpartition_4225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 11), 'argpartition', False)
    # Calling argpartition(args, kwargs) (line 715)
    argpartition_call_result_4233 = invoke(stypy.reporting.localization.Localization(__file__, 715, 11), argpartition_4225, *[kth_4226, axis_4227], **kwargs_4232)
    
    # Assigning a type to the variable 'stypy_return_type' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'stypy_return_type', argpartition_call_result_4233)
    
    # ################# End of 'argpartition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argpartition' in the type store
    # Getting the type of 'stypy_return_type' (line 648)
    stypy_return_type_4234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4234)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argpartition'
    return stypy_return_type_4234

# Assigning a type to the variable 'argpartition' (line 648)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 0), 'argpartition', argpartition)

@norecursion
def sort(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 17), 'int')
    str_4236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 26), 'str', 'quicksort')
    # Getting the type of 'None' (line 718)
    None_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 45), 'None')
    defaults = [int_4235, str_4236, None_4237]
    # Create a new context for function 'sort'
    module_type_store = module_type_store.open_function_context('sort', 718, 0, False)
    
    # Passed parameters checking function
    sort.stypy_localization = localization
    sort.stypy_type_of_self = None
    sort.stypy_type_store = module_type_store
    sort.stypy_function_name = 'sort'
    sort.stypy_param_names_list = ['a', 'axis', 'kind', 'order']
    sort.stypy_varargs_param_name = None
    sort.stypy_kwargs_param_name = None
    sort.stypy_call_defaults = defaults
    sort.stypy_call_varargs = varargs
    sort.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sort', ['a', 'axis', 'kind', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sort', localization, ['a', 'axis', 'kind', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sort(...)' code ##################

    str_4238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, (-1)), 'str', "\n    Return a sorted copy of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Array to be sorted.\n    axis : int or None, optional\n        Axis along which to sort. If None, the array is flattened before\n        sorting. The default is -1, which sorts along the last axis.\n    kind : {'quicksort', 'mergesort', 'heapsort'}, optional\n        Sorting algorithm. Default is 'quicksort'.\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string, and not all fields need be specified,\n        but unspecified fields will still be used, in the order in which\n        they come up in the dtype, to break ties.\n\n    Returns\n    -------\n    sorted_array : ndarray\n        Array of the same type and shape as `a`.\n\n    See Also\n    --------\n    ndarray.sort : Method to sort an array in-place.\n    argsort : Indirect sort.\n    lexsort : Indirect stable sort on multiple keys.\n    searchsorted : Find elements in a sorted array.\n    partition : Partial sort.\n\n    Notes\n    -----\n    The various sorting algorithms are characterized by their average speed,\n    worst case performance, work space size, and whether they are stable. A\n    stable sort keeps items with the same key in the same relative\n    order. The three available algorithms have the following\n    properties:\n\n    =========== ======= ============= ============ =======\n       kind      speed   worst case    work space  stable\n    =========== ======= ============= ============ =======\n    'quicksort'    1     O(n^2)            0          no\n    'mergesort'    2     O(n*log(n))      ~n/2        yes\n    'heapsort'     3     O(n*log(n))       0          no\n    =========== ======= ============= ============ =======\n\n    All the sort algorithms make temporary copies of the data when\n    sorting along any but the last axis.  Consequently, sorting along\n    the last axis is faster and uses less space than sorting along\n    any other axis.\n\n    The sort order for complex numbers is lexicographic. If both the real\n    and imaginary parts are non-nan then the order is determined by the\n    real parts except when they are equal, in which case the order is\n    determined by the imaginary parts.\n\n    Previous to numpy 1.4.0 sorting real and complex arrays containing nan\n    values led to undefined behaviour. In numpy versions >= 1.4.0 nan\n    values are sorted to the end. The extended sort order is:\n\n      * Real: [R, nan]\n      * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]\n\n    where R is a non-nan real value. Complex values with the same nan\n    placements are sorted according to the non-nan part if it exists.\n    Non-nan values are sorted as before.\n\n    Examples\n    --------\n    >>> a = np.array([[1,4],[3,1]])\n    >>> np.sort(a)                # sort along the last axis\n    array([[1, 4],\n           [1, 3]])\n    >>> np.sort(a, axis=None)     # sort the flattened array\n    array([1, 1, 3, 4])\n    >>> np.sort(a, axis=0)        # sort along the first axis\n    array([[1, 1],\n           [3, 4]])\n\n    Use the `order` keyword to specify a field to use when sorting a\n    structured array:\n\n    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]\n    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),\n    ...           ('Galahad', 1.7, 38)]\n    >>> a = np.array(values, dtype=dtype)       # create a structured array\n    >>> np.sort(a, order='height')                        # doctest: +SKIP\n    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),\n           ('Lancelot', 1.8999999999999999, 38)],\n          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])\n\n    Sort by age, then height if ages are equal:\n\n    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP\n    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),\n           ('Arthur', 1.8, 41)],\n          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 820)
    # Getting the type of 'axis' (line 820)
    axis_4239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 7), 'axis')
    # Getting the type of 'None' (line 820)
    None_4240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 15), 'None')
    
    (may_be_4241, more_types_in_union_4242) = may_be_none(axis_4239, None_4240)

    if may_be_4241:

        if more_types_in_union_4242:
            # Runtime conditional SSA (line 820)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 821):
        
        # Call to flatten(...): (line 821)
        # Processing the call keyword arguments (line 821)
        kwargs_4248 = {}
        
        # Call to asanyarray(...): (line 821)
        # Processing the call arguments (line 821)
        # Getting the type of 'a' (line 821)
        a_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 23), 'a', False)
        # Processing the call keyword arguments (line 821)
        kwargs_4245 = {}
        # Getting the type of 'asanyarray' (line 821)
        asanyarray_4243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 821)
        asanyarray_call_result_4246 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), asanyarray_4243, *[a_4244], **kwargs_4245)
        
        # Obtaining the member 'flatten' of a type (line 821)
        flatten_4247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 12), asanyarray_call_result_4246, 'flatten')
        # Calling flatten(args, kwargs) (line 821)
        flatten_call_result_4249 = invoke(stypy.reporting.localization.Localization(__file__, 821, 12), flatten_4247, *[], **kwargs_4248)
        
        # Assigning a type to the variable 'a' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'a', flatten_call_result_4249)
        
        # Assigning a Num to a Name (line 822):
        int_4250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 15), 'int')
        # Assigning a type to the variable 'axis' (line 822)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 8), 'axis', int_4250)

        if more_types_in_union_4242:
            # Runtime conditional SSA for else branch (line 820)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_4241) or more_types_in_union_4242):
        
        # Assigning a Call to a Name (line 824):
        
        # Call to copy(...): (line 824)
        # Processing the call keyword arguments (line 824)
        str_4256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 37), 'str', 'K')
        keyword_4257 = str_4256
        kwargs_4258 = {'order': keyword_4257}
        
        # Call to asanyarray(...): (line 824)
        # Processing the call arguments (line 824)
        # Getting the type of 'a' (line 824)
        a_4252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 23), 'a', False)
        # Processing the call keyword arguments (line 824)
        kwargs_4253 = {}
        # Getting the type of 'asanyarray' (line 824)
        asanyarray_4251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 824)
        asanyarray_call_result_4254 = invoke(stypy.reporting.localization.Localization(__file__, 824, 12), asanyarray_4251, *[a_4252], **kwargs_4253)
        
        # Obtaining the member 'copy' of a type (line 824)
        copy_4255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 12), asanyarray_call_result_4254, 'copy')
        # Calling copy(args, kwargs) (line 824)
        copy_call_result_4259 = invoke(stypy.reporting.localization.Localization(__file__, 824, 12), copy_4255, *[], **kwargs_4258)
        
        # Assigning a type to the variable 'a' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'a', copy_call_result_4259)

        if (may_be_4241 and more_types_in_union_4242):
            # SSA join for if statement (line 820)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to sort(...): (line 825)
    # Processing the call arguments (line 825)
    # Getting the type of 'axis' (line 825)
    axis_4262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 11), 'axis', False)
    # Getting the type of 'kind' (line 825)
    kind_4263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 17), 'kind', False)
    # Getting the type of 'order' (line 825)
    order_4264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 23), 'order', False)
    # Processing the call keyword arguments (line 825)
    kwargs_4265 = {}
    # Getting the type of 'a' (line 825)
    a_4260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'a', False)
    # Obtaining the member 'sort' of a type (line 825)
    sort_4261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 4), a_4260, 'sort')
    # Calling sort(args, kwargs) (line 825)
    sort_call_result_4266 = invoke(stypy.reporting.localization.Localization(__file__, 825, 4), sort_4261, *[axis_4262, kind_4263, order_4264], **kwargs_4265)
    
    # Getting the type of 'a' (line 826)
    a_4267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'stypy_return_type', a_4267)
    
    # ################# End of 'sort(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sort' in the type store
    # Getting the type of 'stypy_return_type' (line 718)
    stypy_return_type_4268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sort'
    return stypy_return_type_4268

# Assigning a type to the variable 'sort' (line 718)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 0), 'sort', sort)

@norecursion
def argsort(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 20), 'int')
    str_4270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 29), 'str', 'quicksort')
    # Getting the type of 'None' (line 829)
    None_4271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 48), 'None')
    defaults = [int_4269, str_4270, None_4271]
    # Create a new context for function 'argsort'
    module_type_store = module_type_store.open_function_context('argsort', 829, 0, False)
    
    # Passed parameters checking function
    argsort.stypy_localization = localization
    argsort.stypy_type_of_self = None
    argsort.stypy_type_store = module_type_store
    argsort.stypy_function_name = 'argsort'
    argsort.stypy_param_names_list = ['a', 'axis', 'kind', 'order']
    argsort.stypy_varargs_param_name = None
    argsort.stypy_kwargs_param_name = None
    argsort.stypy_call_defaults = defaults
    argsort.stypy_call_varargs = varargs
    argsort.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argsort', ['a', 'axis', 'kind', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argsort', localization, ['a', 'axis', 'kind', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argsort(...)' code ##################

    str_4272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, (-1)), 'str', "\n    Returns the indices that would sort an array.\n\n    Perform an indirect sort along the given axis using the algorithm specified\n    by the `kind` keyword. It returns an array of indices of the same shape as\n    `a` that index data along the given axis in sorted order.\n\n    Parameters\n    ----------\n    a : array_like\n        Array to sort.\n    axis : int or None, optional\n        Axis along which to sort.  The default is -1 (the last axis). If None,\n        the flattened array is used.\n    kind : {'quicksort', 'mergesort', 'heapsort'}, optional\n        Sorting algorithm.\n    order : str or list of str, optional\n        When `a` is an array with fields defined, this argument specifies\n        which fields to compare first, second, etc.  A single field can\n        be specified as a string, and not all fields need be specified,\n        but unspecified fields will still be used, in the order in which\n        they come up in the dtype, to break ties.\n\n    Returns\n    -------\n    index_array : ndarray, int\n        Array of indices that sort `a` along the specified axis.\n        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.\n\n    See Also\n    --------\n    sort : Describes sorting algorithms used.\n    lexsort : Indirect stable sort with multiple keys.\n    ndarray.sort : Inplace sort.\n    argpartition : Indirect partial sort.\n\n    Notes\n    -----\n    See `sort` for notes on the different sorting algorithms.\n\n    As of NumPy 1.4.0 `argsort` works with real/complex arrays containing\n    nan values. The enhanced sort order is documented in `sort`.\n\n    Examples\n    --------\n    One dimensional array:\n\n    >>> x = np.array([3, 1, 2])\n    >>> np.argsort(x)\n    array([1, 2, 0])\n\n    Two-dimensional array:\n\n    >>> x = np.array([[0, 3], [2, 2]])\n    >>> x\n    array([[0, 3],\n           [2, 2]])\n\n    >>> np.argsort(x, axis=0)\n    array([[0, 1],\n           [1, 0]])\n\n    >>> np.argsort(x, axis=1)\n    array([[0, 1],\n           [0, 1]])\n\n    Sorting with keys:\n\n    >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])\n    >>> x\n    array([(1, 0), (0, 1)],\n          dtype=[('x', '<i4'), ('y', '<i4')])\n\n    >>> np.argsort(x, order=('x','y'))\n    array([1, 0])\n\n    >>> np.argsort(x, order=('y','x'))\n    array([0, 1])\n\n    ")
    
    
    # SSA begins for try-except statement (line 910)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 911):
    # Getting the type of 'a' (line 911)
    a_4273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 18), 'a')
    # Obtaining the member 'argsort' of a type (line 911)
    argsort_4274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 18), a_4273, 'argsort')
    # Assigning a type to the variable 'argsort' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 8), 'argsort', argsort_4274)
    # SSA branch for the except part of a try statement (line 910)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 910)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 913)
    # Processing the call arguments (line 913)
    # Getting the type of 'a' (line 913)
    a_4276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 23), 'a', False)
    str_4277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 26), 'str', 'argsort')
    # Getting the type of 'axis' (line 913)
    axis_4278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 37), 'axis', False)
    # Getting the type of 'kind' (line 913)
    kind_4279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 43), 'kind', False)
    # Getting the type of 'order' (line 913)
    order_4280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 49), 'order', False)
    # Processing the call keyword arguments (line 913)
    kwargs_4281 = {}
    # Getting the type of '_wrapit' (line 913)
    _wrapit_4275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 913)
    _wrapit_call_result_4282 = invoke(stypy.reporting.localization.Localization(__file__, 913, 15), _wrapit_4275, *[a_4276, str_4277, axis_4278, kind_4279, order_4280], **kwargs_4281)
    
    # Assigning a type to the variable 'stypy_return_type' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 8), 'stypy_return_type', _wrapit_call_result_4282)
    # SSA join for try-except statement (line 910)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to argsort(...): (line 914)
    # Processing the call arguments (line 914)
    # Getting the type of 'axis' (line 914)
    axis_4284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 19), 'axis', False)
    # Getting the type of 'kind' (line 914)
    kind_4285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 25), 'kind', False)
    # Getting the type of 'order' (line 914)
    order_4286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 31), 'order', False)
    # Processing the call keyword arguments (line 914)
    kwargs_4287 = {}
    # Getting the type of 'argsort' (line 914)
    argsort_4283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 11), 'argsort', False)
    # Calling argsort(args, kwargs) (line 914)
    argsort_call_result_4288 = invoke(stypy.reporting.localization.Localization(__file__, 914, 11), argsort_4283, *[axis_4284, kind_4285, order_4286], **kwargs_4287)
    
    # Assigning a type to the variable 'stypy_return_type' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'stypy_return_type', argsort_call_result_4288)
    
    # ################# End of 'argsort(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argsort' in the type store
    # Getting the type of 'stypy_return_type' (line 829)
    stypy_return_type_4289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4289)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argsort'
    return stypy_return_type_4289

# Assigning a type to the variable 'argsort' (line 829)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 0), 'argsort', argsort)

@norecursion
def argmax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 917)
    None_4290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 19), 'None')
    # Getting the type of 'None' (line 917)
    None_4291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 29), 'None')
    defaults = [None_4290, None_4291]
    # Create a new context for function 'argmax'
    module_type_store = module_type_store.open_function_context('argmax', 917, 0, False)
    
    # Passed parameters checking function
    argmax.stypy_localization = localization
    argmax.stypy_type_of_self = None
    argmax.stypy_type_store = module_type_store
    argmax.stypy_function_name = 'argmax'
    argmax.stypy_param_names_list = ['a', 'axis', 'out']
    argmax.stypy_varargs_param_name = None
    argmax.stypy_kwargs_param_name = None
    argmax.stypy_call_defaults = defaults
    argmax.stypy_call_varargs = varargs
    argmax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argmax', ['a', 'axis', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argmax', localization, ['a', 'axis', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argmax(...)' code ##################

    str_4292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, (-1)), 'str', '\n    Returns the indices of the maximum values along an axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int, optional\n        By default, the index is into the flattened array, otherwise\n        along the specified axis.\n    out : array, optional\n        If provided, the result will be inserted into this array. It should\n        be of the appropriate shape and dtype.\n\n    Returns\n    -------\n    index_array : ndarray of ints\n        Array of indices into the array. It has the same shape as `a.shape`\n        with the dimension along `axis` removed.\n\n    See Also\n    --------\n    ndarray.argmax, argmin\n    amax : The maximum value along a given axis.\n    unravel_index : Convert a flat index into an index tuple.\n\n    Notes\n    -----\n    In case of multiple occurrences of the maximum values, the indices\n    corresponding to the first occurrence are returned.\n\n    Examples\n    --------\n    >>> a = np.arange(6).reshape(2,3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> np.argmax(a)\n    5\n    >>> np.argmax(a, axis=0)\n    array([1, 1, 1])\n    >>> np.argmax(a, axis=1)\n    array([2, 2])\n\n    >>> b = np.arange(6)\n    >>> b[1] = 5\n    >>> b\n    array([0, 5, 2, 3, 4, 5])\n    >>> np.argmax(b) # Only the first occurrence is returned.\n    1\n\n    ')
    
    
    # SSA begins for try-except statement (line 970)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 971):
    # Getting the type of 'a' (line 971)
    a_4293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 17), 'a')
    # Obtaining the member 'argmax' of a type (line 971)
    argmax_4294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 17), a_4293, 'argmax')
    # Assigning a type to the variable 'argmax' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 8), 'argmax', argmax_4294)
    # SSA branch for the except part of a try statement (line 970)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 970)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'a' (line 973)
    a_4296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 23), 'a', False)
    str_4297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 26), 'str', 'argmax')
    # Getting the type of 'axis' (line 973)
    axis_4298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 36), 'axis', False)
    # Getting the type of 'out' (line 973)
    out_4299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 42), 'out', False)
    # Processing the call keyword arguments (line 973)
    kwargs_4300 = {}
    # Getting the type of '_wrapit' (line 973)
    _wrapit_4295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 973)
    _wrapit_call_result_4301 = invoke(stypy.reporting.localization.Localization(__file__, 973, 15), _wrapit_4295, *[a_4296, str_4297, axis_4298, out_4299], **kwargs_4300)
    
    # Assigning a type to the variable 'stypy_return_type' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'stypy_return_type', _wrapit_call_result_4301)
    # SSA join for try-except statement (line 970)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to argmax(...): (line 974)
    # Processing the call arguments (line 974)
    # Getting the type of 'axis' (line 974)
    axis_4303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 18), 'axis', False)
    # Getting the type of 'out' (line 974)
    out_4304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 24), 'out', False)
    # Processing the call keyword arguments (line 974)
    kwargs_4305 = {}
    # Getting the type of 'argmax' (line 974)
    argmax_4302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 11), 'argmax', False)
    # Calling argmax(args, kwargs) (line 974)
    argmax_call_result_4306 = invoke(stypy.reporting.localization.Localization(__file__, 974, 11), argmax_4302, *[axis_4303, out_4304], **kwargs_4305)
    
    # Assigning a type to the variable 'stypy_return_type' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'stypy_return_type', argmax_call_result_4306)
    
    # ################# End of 'argmax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argmax' in the type store
    # Getting the type of 'stypy_return_type' (line 917)
    stypy_return_type_4307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argmax'
    return stypy_return_type_4307

# Assigning a type to the variable 'argmax' (line 917)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 0), 'argmax', argmax)

@norecursion
def argmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 977)
    None_4308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 19), 'None')
    # Getting the type of 'None' (line 977)
    None_4309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 29), 'None')
    defaults = [None_4308, None_4309]
    # Create a new context for function 'argmin'
    module_type_store = module_type_store.open_function_context('argmin', 977, 0, False)
    
    # Passed parameters checking function
    argmin.stypy_localization = localization
    argmin.stypy_type_of_self = None
    argmin.stypy_type_store = module_type_store
    argmin.stypy_function_name = 'argmin'
    argmin.stypy_param_names_list = ['a', 'axis', 'out']
    argmin.stypy_varargs_param_name = None
    argmin.stypy_kwargs_param_name = None
    argmin.stypy_call_defaults = defaults
    argmin.stypy_call_varargs = varargs
    argmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'argmin', ['a', 'axis', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'argmin', localization, ['a', 'axis', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'argmin(...)' code ##################

    str_4310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, (-1)), 'str', '\n    Returns the indices of the minimum values along an axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int, optional\n        By default, the index is into the flattened array, otherwise\n        along the specified axis.\n    out : array, optional\n        If provided, the result will be inserted into this array. It should\n        be of the appropriate shape and dtype.\n\n    Returns\n    -------\n    index_array : ndarray of ints\n        Array of indices into the array. It has the same shape as `a.shape`\n        with the dimension along `axis` removed.\n\n    See Also\n    --------\n    ndarray.argmin, argmax\n    amin : The minimum value along a given axis.\n    unravel_index : Convert a flat index into an index tuple.\n\n    Notes\n    -----\n    In case of multiple occurrences of the minimum values, the indices\n    corresponding to the first occurrence are returned.\n\n    Examples\n    --------\n    >>> a = np.arange(6).reshape(2,3)\n    >>> a\n    array([[0, 1, 2],\n           [3, 4, 5]])\n    >>> np.argmin(a)\n    0\n    >>> np.argmin(a, axis=0)\n    array([0, 0, 0])\n    >>> np.argmin(a, axis=1)\n    array([0, 0])\n\n    >>> b = np.arange(6)\n    >>> b[4] = 0\n    >>> b\n    array([0, 1, 2, 3, 0, 5])\n    >>> np.argmin(b) # Only the first occurrence is returned.\n    0\n\n    ')
    
    
    # SSA begins for try-except statement (line 1030)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1031):
    # Getting the type of 'a' (line 1031)
    a_4311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 17), 'a')
    # Obtaining the member 'argmin' of a type (line 1031)
    argmin_4312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 17), a_4311, 'argmin')
    # Assigning a type to the variable 'argmin' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'argmin', argmin_4312)
    # SSA branch for the except part of a try statement (line 1030)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1030)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'a' (line 1033)
    a_4314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 23), 'a', False)
    str_4315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 26), 'str', 'argmin')
    # Getting the type of 'axis' (line 1033)
    axis_4316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 36), 'axis', False)
    # Getting the type of 'out' (line 1033)
    out_4317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 42), 'out', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_4318 = {}
    # Getting the type of '_wrapit' (line 1033)
    _wrapit_4313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1033)
    _wrapit_call_result_4319 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 15), _wrapit_4313, *[a_4314, str_4315, axis_4316, out_4317], **kwargs_4318)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'stypy_return_type', _wrapit_call_result_4319)
    # SSA join for try-except statement (line 1030)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to argmin(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'axis' (line 1034)
    axis_4321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 18), 'axis', False)
    # Getting the type of 'out' (line 1034)
    out_4322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 24), 'out', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_4323 = {}
    # Getting the type of 'argmin' (line 1034)
    argmin_4320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 11), 'argmin', False)
    # Calling argmin(args, kwargs) (line 1034)
    argmin_call_result_4324 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 11), argmin_4320, *[axis_4321, out_4322], **kwargs_4323)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'stypy_return_type', argmin_call_result_4324)
    
    # ################# End of 'argmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'argmin' in the type store
    # Getting the type of 'stypy_return_type' (line 977)
    stypy_return_type_4325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4325)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'argmin'
    return stypy_return_type_4325

# Assigning a type to the variable 'argmin' (line 977)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 0), 'argmin', argmin)

@norecursion
def searchsorted(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_4326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 28), 'str', 'left')
    # Getting the type of 'None' (line 1037)
    None_4327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 43), 'None')
    defaults = [str_4326, None_4327]
    # Create a new context for function 'searchsorted'
    module_type_store = module_type_store.open_function_context('searchsorted', 1037, 0, False)
    
    # Passed parameters checking function
    searchsorted.stypy_localization = localization
    searchsorted.stypy_type_of_self = None
    searchsorted.stypy_type_store = module_type_store
    searchsorted.stypy_function_name = 'searchsorted'
    searchsorted.stypy_param_names_list = ['a', 'v', 'side', 'sorter']
    searchsorted.stypy_varargs_param_name = None
    searchsorted.stypy_kwargs_param_name = None
    searchsorted.stypy_call_defaults = defaults
    searchsorted.stypy_call_varargs = varargs
    searchsorted.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'searchsorted', ['a', 'v', 'side', 'sorter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'searchsorted', localization, ['a', 'v', 'side', 'sorter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'searchsorted(...)' code ##################

    str_4328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, (-1)), 'str', "\n    Find indices where elements should be inserted to maintain order.\n\n    Find the indices into a sorted array `a` such that, if the\n    corresponding elements in `v` were inserted before the indices, the\n    order of `a` would be preserved.\n\n    Parameters\n    ----------\n    a : 1-D array_like\n        Input array. If `sorter` is None, then it must be sorted in\n        ascending order, otherwise `sorter` must be an array of indices\n        that sort it.\n    v : array_like\n        Values to insert into `a`.\n    side : {'left', 'right'}, optional\n        If 'left', the index of the first suitable location found is given.\n        If 'right', return the last such index.  If there is no suitable\n        index, return either 0 or N (where N is the length of `a`).\n    sorter : 1-D array_like, optional\n        Optional array of integer indices that sort array a into ascending\n        order. They are typically the result of argsort.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    indices : array of ints\n        Array of insertion points with the same shape as `v`.\n\n    See Also\n    --------\n    sort : Return a sorted copy of an array.\n    histogram : Produce histogram from 1-D data.\n\n    Notes\n    -----\n    Binary search is used to find the required insertion points.\n\n    As of Numpy 1.4.0 `searchsorted` works with real/complex arrays containing\n    `nan` values. The enhanced sort order is documented in `sort`.\n\n    Examples\n    --------\n    >>> np.searchsorted([1,2,3,4,5], 3)\n    2\n    >>> np.searchsorted([1,2,3,4,5], 3, side='right')\n    3\n    >>> np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])\n    array([0, 5, 1, 2])\n\n    ")
    
    
    # SSA begins for try-except statement (line 1090)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1091):
    # Getting the type of 'a' (line 1091)
    a_4329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 23), 'a')
    # Obtaining the member 'searchsorted' of a type (line 1091)
    searchsorted_4330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 23), a_4329, 'searchsorted')
    # Assigning a type to the variable 'searchsorted' (line 1091)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1091, 8), 'searchsorted', searchsorted_4330)
    # SSA branch for the except part of a try statement (line 1090)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1090)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 1093)
    # Processing the call arguments (line 1093)
    # Getting the type of 'a' (line 1093)
    a_4332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 23), 'a', False)
    str_4333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 26), 'str', 'searchsorted')
    # Getting the type of 'v' (line 1093)
    v_4334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 42), 'v', False)
    # Getting the type of 'side' (line 1093)
    side_4335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 45), 'side', False)
    # Getting the type of 'sorter' (line 1093)
    sorter_4336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 51), 'sorter', False)
    # Processing the call keyword arguments (line 1093)
    kwargs_4337 = {}
    # Getting the type of '_wrapit' (line 1093)
    _wrapit_4331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1093)
    _wrapit_call_result_4338 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 15), _wrapit_4331, *[a_4332, str_4333, v_4334, side_4335, sorter_4336], **kwargs_4337)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'stypy_return_type', _wrapit_call_result_4338)
    # SSA join for try-except statement (line 1090)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to searchsorted(...): (line 1094)
    # Processing the call arguments (line 1094)
    # Getting the type of 'v' (line 1094)
    v_4340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 24), 'v', False)
    # Getting the type of 'side' (line 1094)
    side_4341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 27), 'side', False)
    # Getting the type of 'sorter' (line 1094)
    sorter_4342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 33), 'sorter', False)
    # Processing the call keyword arguments (line 1094)
    kwargs_4343 = {}
    # Getting the type of 'searchsorted' (line 1094)
    searchsorted_4339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 11), 'searchsorted', False)
    # Calling searchsorted(args, kwargs) (line 1094)
    searchsorted_call_result_4344 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 11), searchsorted_4339, *[v_4340, side_4341, sorter_4342], **kwargs_4343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'stypy_return_type', searchsorted_call_result_4344)
    
    # ################# End of 'searchsorted(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'searchsorted' in the type store
    # Getting the type of 'stypy_return_type' (line 1037)
    stypy_return_type_4345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'searchsorted'
    return stypy_return_type_4345

# Assigning a type to the variable 'searchsorted' (line 1037)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 0), 'searchsorted', searchsorted)

@norecursion
def resize(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'resize'
    module_type_store = module_type_store.open_function_context('resize', 1097, 0, False)
    
    # Passed parameters checking function
    resize.stypy_localization = localization
    resize.stypy_type_of_self = None
    resize.stypy_type_store = module_type_store
    resize.stypy_function_name = 'resize'
    resize.stypy_param_names_list = ['a', 'new_shape']
    resize.stypy_varargs_param_name = None
    resize.stypy_kwargs_param_name = None
    resize.stypy_call_defaults = defaults
    resize.stypy_call_varargs = varargs
    resize.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'resize', ['a', 'new_shape'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'resize', localization, ['a', 'new_shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'resize(...)' code ##################

    str_4346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, (-1)), 'str', '\n    Return a new array with the specified shape.\n\n    If the new array is larger than the original array, then the new\n    array is filled with repeated copies of `a`.  Note that this behavior\n    is different from a.resize(new_shape) which fills with zeros instead\n    of repeated copies of `a`.\n\n    Parameters\n    ----------\n    a : array_like\n        Array to be resized.\n\n    new_shape : int or tuple of int\n        Shape of resized array.\n\n    Returns\n    -------\n    reshaped_array : ndarray\n        The new array is formed from the data in the old array, repeated\n        if necessary to fill out the required number of elements.  The\n        data are repeated in the order that they are stored in memory.\n\n    See Also\n    --------\n    ndarray.resize : resize an array in-place.\n\n    Examples\n    --------\n    >>> a=np.array([[0,1],[2,3]])\n    >>> np.resize(a,(2,3))\n    array([[0, 1, 2],\n           [3, 0, 1]])\n    >>> np.resize(a,(1,4))\n    array([[0, 1, 2, 3]])\n    >>> np.resize(a,(2,4))\n    array([[0, 1, 2, 3],\n           [0, 1, 2, 3]])\n\n    ')
    
    
    # Call to isinstance(...): (line 1138)
    # Processing the call arguments (line 1138)
    # Getting the type of 'new_shape' (line 1138)
    new_shape_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 18), 'new_shape', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1138)
    tuple_4349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1138)
    # Adding element type (line 1138)
    # Getting the type of 'int' (line 1138)
    int_4350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 30), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1138, 30), tuple_4349, int_4350)
    # Adding element type (line 1138)
    # Getting the type of 'nt' (line 1138)
    nt_4351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 35), 'nt', False)
    # Obtaining the member 'integer' of a type (line 1138)
    integer_4352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1138, 35), nt_4351, 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1138, 30), tuple_4349, integer_4352)
    
    # Processing the call keyword arguments (line 1138)
    kwargs_4353 = {}
    # Getting the type of 'isinstance' (line 1138)
    isinstance_4347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1138)
    isinstance_call_result_4354 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 7), isinstance_4347, *[new_shape_4348, tuple_4349], **kwargs_4353)
    
    # Testing the type of an if condition (line 1138)
    if_condition_4355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1138, 4), isinstance_call_result_4354)
    # Assigning a type to the variable 'if_condition_4355' (line 1138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 4), 'if_condition_4355', if_condition_4355)
    # SSA begins for if statement (line 1138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 1139):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1139)
    tuple_4356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1139)
    # Adding element type (line 1139)
    # Getting the type of 'new_shape' (line 1139)
    new_shape_4357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 21), 'new_shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 21), tuple_4356, new_shape_4357)
    
    # Assigning a type to the variable 'new_shape' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'new_shape', tuple_4356)
    # SSA join for if statement (line 1138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1140):
    
    # Call to ravel(...): (line 1140)
    # Processing the call arguments (line 1140)
    # Getting the type of 'a' (line 1140)
    a_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 14), 'a', False)
    # Processing the call keyword arguments (line 1140)
    kwargs_4360 = {}
    # Getting the type of 'ravel' (line 1140)
    ravel_4358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'ravel', False)
    # Calling ravel(args, kwargs) (line 1140)
    ravel_call_result_4361 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 8), ravel_4358, *[a_4359], **kwargs_4360)
    
    # Assigning a type to the variable 'a' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 4), 'a', ravel_call_result_4361)
    
    # Assigning a Call to a Name (line 1141):
    
    # Call to len(...): (line 1141)
    # Processing the call arguments (line 1141)
    # Getting the type of 'a' (line 1141)
    a_4363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 13), 'a', False)
    # Processing the call keyword arguments (line 1141)
    kwargs_4364 = {}
    # Getting the type of 'len' (line 1141)
    len_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 9), 'len', False)
    # Calling len(args, kwargs) (line 1141)
    len_call_result_4365 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 9), len_4362, *[a_4363], **kwargs_4364)
    
    # Assigning a type to the variable 'Na' (line 1141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 4), 'Na', len_call_result_4365)
    
    
    # Getting the type of 'Na' (line 1142)
    Na_4366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 11), 'Na')
    # Applying the 'not' unary operator (line 1142)
    result_not__4367 = python_operator(stypy.reporting.localization.Localization(__file__, 1142, 7), 'not', Na_4366)
    
    # Testing the type of an if condition (line 1142)
    if_condition_4368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1142, 4), result_not__4367)
    # Assigning a type to the variable 'if_condition_4368' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'if_condition_4368', if_condition_4368)
    # SSA begins for if statement (line 1142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to zeros(...): (line 1143)
    # Processing the call arguments (line 1143)
    # Getting the type of 'new_shape' (line 1143)
    new_shape_4371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 24), 'new_shape', False)
    # Getting the type of 'a' (line 1143)
    a_4372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 35), 'a', False)
    # Obtaining the member 'dtype' of a type (line 1143)
    dtype_4373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1143, 35), a_4372, 'dtype')
    # Processing the call keyword arguments (line 1143)
    kwargs_4374 = {}
    # Getting the type of 'mu' (line 1143)
    mu_4369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 15), 'mu', False)
    # Obtaining the member 'zeros' of a type (line 1143)
    zeros_4370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1143, 15), mu_4369, 'zeros')
    # Calling zeros(args, kwargs) (line 1143)
    zeros_call_result_4375 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 15), zeros_4370, *[new_shape_4371, dtype_4373], **kwargs_4374)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1143, 8), 'stypy_return_type', zeros_call_result_4375)
    # SSA join for if statement (line 1142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1144):
    
    # Call to reduce(...): (line 1144)
    # Processing the call arguments (line 1144)
    # Getting the type of 'new_shape' (line 1144)
    new_shape_4379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 36), 'new_shape', False)
    # Processing the call keyword arguments (line 1144)
    kwargs_4380 = {}
    # Getting the type of 'um' (line 1144)
    um_4376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 17), 'um', False)
    # Obtaining the member 'multiply' of a type (line 1144)
    multiply_4377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 17), um_4376, 'multiply')
    # Obtaining the member 'reduce' of a type (line 1144)
    reduce_4378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 17), multiply_4377, 'reduce')
    # Calling reduce(args, kwargs) (line 1144)
    reduce_call_result_4381 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 17), reduce_4378, *[new_shape_4379], **kwargs_4380)
    
    # Assigning a type to the variable 'total_size' (line 1144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'total_size', reduce_call_result_4381)
    
    # Assigning a Call to a Name (line 1145):
    
    # Call to int(...): (line 1145)
    # Processing the call arguments (line 1145)
    # Getting the type of 'total_size' (line 1145)
    total_size_4383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 19), 'total_size', False)
    # Getting the type of 'Na' (line 1145)
    Na_4384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 32), 'Na', False)
    # Applying the binary operator 'div' (line 1145)
    result_div_4385 = python_operator(stypy.reporting.localization.Localization(__file__, 1145, 19), 'div', total_size_4383, Na_4384)
    
    # Processing the call keyword arguments (line 1145)
    kwargs_4386 = {}
    # Getting the type of 'int' (line 1145)
    int_4382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 15), 'int', False)
    # Calling int(args, kwargs) (line 1145)
    int_call_result_4387 = invoke(stypy.reporting.localization.Localization(__file__, 1145, 15), int_4382, *[result_div_4385], **kwargs_4386)
    
    # Assigning a type to the variable 'n_copies' (line 1145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 4), 'n_copies', int_call_result_4387)
    
    # Assigning a BinOp to a Name (line 1146):
    # Getting the type of 'total_size' (line 1146)
    total_size_4388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 12), 'total_size')
    # Getting the type of 'Na' (line 1146)
    Na_4389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 25), 'Na')
    # Applying the binary operator '%' (line 1146)
    result_mod_4390 = python_operator(stypy.reporting.localization.Localization(__file__, 1146, 12), '%', total_size_4388, Na_4389)
    
    # Assigning a type to the variable 'extra' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 4), 'extra', result_mod_4390)
    
    
    # Getting the type of 'total_size' (line 1148)
    total_size_4391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 7), 'total_size')
    int_4392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1148, 21), 'int')
    # Applying the binary operator '==' (line 1148)
    result_eq_4393 = python_operator(stypy.reporting.localization.Localization(__file__, 1148, 7), '==', total_size_4391, int_4392)
    
    # Testing the type of an if condition (line 1148)
    if_condition_4394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1148, 4), result_eq_4393)
    # Assigning a type to the variable 'if_condition_4394' (line 1148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 4), 'if_condition_4394', if_condition_4394)
    # SSA begins for if statement (line 1148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_4395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1149, 18), 'int')
    slice_4396 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1149, 15), None, int_4395, None)
    # Getting the type of 'a' (line 1149)
    a_4397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 15), 'a')
    # Obtaining the member '__getitem__' of a type (line 1149)
    getitem___4398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 15), a_4397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1149)
    subscript_call_result_4399 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 15), getitem___4398, slice_4396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 8), 'stypy_return_type', subscript_call_result_4399)
    # SSA join for if statement (line 1148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'extra' (line 1151)
    extra_4400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 7), 'extra')
    int_4401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 16), 'int')
    # Applying the binary operator '!=' (line 1151)
    result_ne_4402 = python_operator(stypy.reporting.localization.Localization(__file__, 1151, 7), '!=', extra_4400, int_4401)
    
    # Testing the type of an if condition (line 1151)
    if_condition_4403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1151, 4), result_ne_4402)
    # Assigning a type to the variable 'if_condition_4403' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 4), 'if_condition_4403', if_condition_4403)
    # SSA begins for if statement (line 1151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1152):
    # Getting the type of 'n_copies' (line 1152)
    n_copies_4404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 19), 'n_copies')
    int_4405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, 28), 'int')
    # Applying the binary operator '+' (line 1152)
    result_add_4406 = python_operator(stypy.reporting.localization.Localization(__file__, 1152, 19), '+', n_copies_4404, int_4405)
    
    # Assigning a type to the variable 'n_copies' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 8), 'n_copies', result_add_4406)
    
    # Assigning a BinOp to a Name (line 1153):
    # Getting the type of 'Na' (line 1153)
    Na_4407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 16), 'Na')
    # Getting the type of 'extra' (line 1153)
    extra_4408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 19), 'extra')
    # Applying the binary operator '-' (line 1153)
    result_sub_4409 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 16), '-', Na_4407, extra_4408)
    
    # Assigning a type to the variable 'extra' (line 1153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 8), 'extra', result_sub_4409)
    # SSA join for if statement (line 1151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1155):
    
    # Call to concatenate(...): (line 1155)
    # Processing the call arguments (line 1155)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1155)
    tuple_4411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1155, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1155)
    # Adding element type (line 1155)
    # Getting the type of 'a' (line 1155)
    a_4412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 21), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1155, 21), tuple_4411, a_4412)
    
    # Getting the type of 'n_copies' (line 1155)
    n_copies_4413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 25), 'n_copies', False)
    # Applying the binary operator '*' (line 1155)
    result_mul_4414 = python_operator(stypy.reporting.localization.Localization(__file__, 1155, 20), '*', tuple_4411, n_copies_4413)
    
    # Processing the call keyword arguments (line 1155)
    kwargs_4415 = {}
    # Getting the type of 'concatenate' (line 1155)
    concatenate_4410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 8), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 1155)
    concatenate_call_result_4416 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 8), concatenate_4410, *[result_mul_4414], **kwargs_4415)
    
    # Assigning a type to the variable 'a' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 4), 'a', concatenate_call_result_4416)
    
    
    # Getting the type of 'extra' (line 1156)
    extra_4417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 7), 'extra')
    int_4418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1156, 15), 'int')
    # Applying the binary operator '>' (line 1156)
    result_gt_4419 = python_operator(stypy.reporting.localization.Localization(__file__, 1156, 7), '>', extra_4417, int_4418)
    
    # Testing the type of an if condition (line 1156)
    if_condition_4420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1156, 4), result_gt_4419)
    # Assigning a type to the variable 'if_condition_4420' (line 1156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'if_condition_4420', if_condition_4420)
    # SSA begins for if statement (line 1156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1157):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'extra' (line 1157)
    extra_4421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 16), 'extra')
    # Applying the 'usub' unary operator (line 1157)
    result___neg___4422 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 15), 'usub', extra_4421)
    
    slice_4423 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1157, 12), None, result___neg___4422, None)
    # Getting the type of 'a' (line 1157)
    a_4424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 12), 'a')
    # Obtaining the member '__getitem__' of a type (line 1157)
    getitem___4425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1157, 12), a_4424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1157)
    subscript_call_result_4426 = invoke(stypy.reporting.localization.Localization(__file__, 1157, 12), getitem___4425, slice_4423)
    
    # Assigning a type to the variable 'a' (line 1157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 8), 'a', subscript_call_result_4426)
    # SSA join for if statement (line 1156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 1159)
    # Processing the call arguments (line 1159)
    # Getting the type of 'a' (line 1159)
    a_4428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 19), 'a', False)
    # Getting the type of 'new_shape' (line 1159)
    new_shape_4429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 22), 'new_shape', False)
    # Processing the call keyword arguments (line 1159)
    kwargs_4430 = {}
    # Getting the type of 'reshape' (line 1159)
    reshape_4427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 11), 'reshape', False)
    # Calling reshape(args, kwargs) (line 1159)
    reshape_call_result_4431 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 11), reshape_4427, *[a_4428, new_shape_4429], **kwargs_4430)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'stypy_return_type', reshape_call_result_4431)
    
    # ################# End of 'resize(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'resize' in the type store
    # Getting the type of 'stypy_return_type' (line 1097)
    stypy_return_type_4432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4432)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'resize'
    return stypy_return_type_4432

# Assigning a type to the variable 'resize' (line 1097)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 0), 'resize', resize)

@norecursion
def squeeze(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1162)
    None_4433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 20), 'None')
    defaults = [None_4433]
    # Create a new context for function 'squeeze'
    module_type_store = module_type_store.open_function_context('squeeze', 1162, 0, False)
    
    # Passed parameters checking function
    squeeze.stypy_localization = localization
    squeeze.stypy_type_of_self = None
    squeeze.stypy_type_store = module_type_store
    squeeze.stypy_function_name = 'squeeze'
    squeeze.stypy_param_names_list = ['a', 'axis']
    squeeze.stypy_varargs_param_name = None
    squeeze.stypy_kwargs_param_name = None
    squeeze.stypy_call_defaults = defaults
    squeeze.stypy_call_varargs = varargs
    squeeze.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'squeeze', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'squeeze', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'squeeze(...)' code ##################

    str_4434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1194, (-1)), 'str', '\n    Remove single-dimensional entries from the shape of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : None or int or tuple of ints, optional\n        .. versionadded:: 1.7.0\n\n        Selects a subset of the single-dimensional entries in the\n        shape. If an axis is selected with shape entry greater than\n        one, an error is raised.\n\n    Returns\n    -------\n    squeezed : ndarray\n        The input array, but with all or a subset of the\n        dimensions of length 1 removed. This is always `a` itself\n        or a view into `a`.\n\n    Examples\n    --------\n    >>> x = np.array([[[0], [1], [2]]])\n    >>> x.shape\n    (1, 3, 1)\n    >>> np.squeeze(x).shape\n    (3,)\n    >>> np.squeeze(x, axis=(2,)).shape\n    (1, 3)\n\n    ')
    
    
    # SSA begins for try-except statement (line 1195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1196):
    # Getting the type of 'a' (line 1196)
    a_4435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 18), 'a')
    # Obtaining the member 'squeeze' of a type (line 1196)
    squeeze_4436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 18), a_4435, 'squeeze')
    # Assigning a type to the variable 'squeeze' (line 1196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 8), 'squeeze', squeeze_4436)
    # SSA branch for the except part of a try statement (line 1195)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1195)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 1198)
    # Processing the call arguments (line 1198)
    # Getting the type of 'a' (line 1198)
    a_4438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 23), 'a', False)
    str_4439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1198, 26), 'str', 'squeeze')
    # Processing the call keyword arguments (line 1198)
    kwargs_4440 = {}
    # Getting the type of '_wrapit' (line 1198)
    _wrapit_4437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1198)
    _wrapit_call_result_4441 = invoke(stypy.reporting.localization.Localization(__file__, 1198, 15), _wrapit_4437, *[a_4438, str_4439], **kwargs_4440)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 8), 'stypy_return_type', _wrapit_call_result_4441)
    # SSA join for try-except statement (line 1195)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 1199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to squeeze(...): (line 1201)
    # Processing the call keyword arguments (line 1201)
    # Getting the type of 'axis' (line 1201)
    axis_4443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 28), 'axis', False)
    keyword_4444 = axis_4443
    kwargs_4445 = {'axis': keyword_4444}
    # Getting the type of 'squeeze' (line 1201)
    squeeze_4442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 15), 'squeeze', False)
    # Calling squeeze(args, kwargs) (line 1201)
    squeeze_call_result_4446 = invoke(stypy.reporting.localization.Localization(__file__, 1201, 15), squeeze_4442, *[], **kwargs_4445)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1201, 8), 'stypy_return_type', squeeze_call_result_4446)
    # SSA branch for the except part of a try statement (line 1199)
    # SSA branch for the except 'TypeError' branch of a try statement (line 1199)
    module_type_store.open_ssa_branch('except')
    
    # Call to squeeze(...): (line 1204)
    # Processing the call keyword arguments (line 1204)
    kwargs_4448 = {}
    # Getting the type of 'squeeze' (line 1204)
    squeeze_4447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 15), 'squeeze', False)
    # Calling squeeze(args, kwargs) (line 1204)
    squeeze_call_result_4449 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 15), squeeze_4447, *[], **kwargs_4448)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 8), 'stypy_return_type', squeeze_call_result_4449)
    # SSA join for try-except statement (line 1199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'squeeze(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'squeeze' in the type store
    # Getting the type of 'stypy_return_type' (line 1162)
    stypy_return_type_4450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4450)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'squeeze'
    return stypy_return_type_4450

# Assigning a type to the variable 'squeeze' (line 1162)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 0), 'squeeze', squeeze)

@norecursion
def diagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 23), 'int')
    int_4452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 32), 'int')
    int_4453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 41), 'int')
    defaults = [int_4451, int_4452, int_4453]
    # Create a new context for function 'diagonal'
    module_type_store = module_type_store.open_function_context('diagonal', 1207, 0, False)
    
    # Passed parameters checking function
    diagonal.stypy_localization = localization
    diagonal.stypy_type_of_self = None
    diagonal.stypy_type_store = module_type_store
    diagonal.stypy_function_name = 'diagonal'
    diagonal.stypy_param_names_list = ['a', 'offset', 'axis1', 'axis2']
    diagonal.stypy_varargs_param_name = None
    diagonal.stypy_kwargs_param_name = None
    diagonal.stypy_call_defaults = defaults
    diagonal.stypy_call_varargs = varargs
    diagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diagonal', ['a', 'offset', 'axis1', 'axis2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diagonal', localization, ['a', 'offset', 'axis1', 'axis2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diagonal(...)' code ##################

    str_4454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1312, (-1)), 'str', '\n    Return specified diagonals.\n\n    If `a` is 2-D, returns the diagonal of `a` with the given offset,\n    i.e., the collection of elements of the form ``a[i, i+offset]``.  If\n    `a` has more than two dimensions, then the axes specified by `axis1`\n    and `axis2` are used to determine the 2-D sub-array whose diagonal is\n    returned.  The shape of the resulting array can be determined by\n    removing `axis1` and `axis2` and appending an index to the right equal\n    to the size of the resulting diagonals.\n\n    In versions of NumPy prior to 1.7, this function always returned a new,\n    independent array containing a copy of the values in the diagonal.\n\n    In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,\n    but depending on this fact is deprecated. Writing to the resulting\n    array continues to work as it used to, but a FutureWarning is issued.\n\n    Starting in NumPy 1.9 it returns a read-only view on the original array.\n    Attempting to write to the resulting array will produce an error.\n\n    In some future release, it will return a read/write view and writing to\n    the returned array will alter your original array.  The returned array\n    will have the same type as the input array.\n\n    If you don\'t write to the array returned by this function, then you can\n    just ignore all of the above.\n\n    If you depend on the current behavior, then we suggest copying the\n    returned array explicitly, i.e., use ``np.diagonal(a).copy()`` instead\n    of just ``np.diagonal(a)``. This will work with both past and future\n    versions of NumPy.\n\n    Parameters\n    ----------\n    a : array_like\n        Array from which the diagonals are taken.\n    offset : int, optional\n        Offset of the diagonal from the main diagonal.  Can be positive or\n        negative.  Defaults to main diagonal (0).\n    axis1 : int, optional\n        Axis to be used as the first axis of the 2-D sub-arrays from which\n        the diagonals should be taken.  Defaults to first axis (0).\n    axis2 : int, optional\n        Axis to be used as the second axis of the 2-D sub-arrays from\n        which the diagonals should be taken. Defaults to second axis (1).\n\n    Returns\n    -------\n    array_of_diagonals : ndarray\n        If `a` is 2-D and not a matrix, a 1-D array of the same type as `a`\n        containing the diagonal is returned. If `a` is a matrix, a 1-D\n        array containing the diagonal is returned in order to maintain\n        backward compatibility.  If the dimension of `a` is greater than\n        two, then an array of diagonals is returned, "packed" from\n        left-most dimension to right-most (e.g., if `a` is 3-D, then the\n        diagonals are "packed" along rows).\n\n    Raises\n    ------\n    ValueError\n        If the dimension of `a` is less than 2.\n\n    See Also\n    --------\n    diag : MATLAB work-a-like for 1-D and 2-D arrays.\n    diagflat : Create diagonal arrays.\n    trace : Sum along diagonals.\n\n    Examples\n    --------\n    >>> a = np.arange(4).reshape(2,2)\n    >>> a\n    array([[0, 1],\n           [2, 3]])\n    >>> a.diagonal()\n    array([0, 3])\n    >>> a.diagonal(1)\n    array([1])\n\n    A 3-D example:\n\n    >>> a = np.arange(8).reshape(2,2,2); a\n    array([[[0, 1],\n            [2, 3]],\n           [[4, 5],\n            [6, 7]]])\n    >>> a.diagonal(0, # Main diagonals of two arrays created by skipping\n    ...            0, # across the outer(left)-most axis last and\n    ...            1) # the "middle" (row) axis first.\n    array([[0, 6],\n           [1, 7]])\n\n    The sub-arrays whose main diagonals we just obtained; note that each\n    corresponds to fixing the right-most (column) axis, and that the\n    diagonals are "packed" in rows.\n\n    >>> a[:,:,0] # main diagonal is [0 6]\n    array([[0, 2],\n           [4, 6]])\n    >>> a[:,:,1] # main diagonal is [1 7]\n    array([[1, 3],\n           [5, 7]])\n\n    ')
    
    
    # Call to isinstance(...): (line 1313)
    # Processing the call arguments (line 1313)
    # Getting the type of 'a' (line 1313)
    a_4456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 18), 'a', False)
    # Getting the type of 'np' (line 1313)
    np_4457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 21), 'np', False)
    # Obtaining the member 'matrix' of a type (line 1313)
    matrix_4458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1313, 21), np_4457, 'matrix')
    # Processing the call keyword arguments (line 1313)
    kwargs_4459 = {}
    # Getting the type of 'isinstance' (line 1313)
    isinstance_4455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1313)
    isinstance_call_result_4460 = invoke(stypy.reporting.localization.Localization(__file__, 1313, 7), isinstance_4455, *[a_4456, matrix_4458], **kwargs_4459)
    
    # Testing the type of an if condition (line 1313)
    if_condition_4461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1313, 4), isinstance_call_result_4460)
    # Assigning a type to the variable 'if_condition_4461' (line 1313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1313, 4), 'if_condition_4461', if_condition_4461)
    # SSA begins for if statement (line 1313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to diagonal(...): (line 1315)
    # Processing the call arguments (line 1315)
    # Getting the type of 'offset' (line 1315)
    offset_4467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 35), 'offset', False)
    # Getting the type of 'axis1' (line 1315)
    axis1_4468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 43), 'axis1', False)
    # Getting the type of 'axis2' (line 1315)
    axis2_4469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 50), 'axis2', False)
    # Processing the call keyword arguments (line 1315)
    kwargs_4470 = {}
    
    # Call to asarray(...): (line 1315)
    # Processing the call arguments (line 1315)
    # Getting the type of 'a' (line 1315)
    a_4463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 23), 'a', False)
    # Processing the call keyword arguments (line 1315)
    kwargs_4464 = {}
    # Getting the type of 'asarray' (line 1315)
    asarray_4462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 1315)
    asarray_call_result_4465 = invoke(stypy.reporting.localization.Localization(__file__, 1315, 15), asarray_4462, *[a_4463], **kwargs_4464)
    
    # Obtaining the member 'diagonal' of a type (line 1315)
    diagonal_4466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1315, 15), asarray_call_result_4465, 'diagonal')
    # Calling diagonal(args, kwargs) (line 1315)
    diagonal_call_result_4471 = invoke(stypy.reporting.localization.Localization(__file__, 1315, 15), diagonal_4466, *[offset_4467, axis1_4468, axis2_4469], **kwargs_4470)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1315, 8), 'stypy_return_type', diagonal_call_result_4471)
    # SSA branch for the else part of an if statement (line 1313)
    module_type_store.open_ssa_branch('else')
    
    # Call to diagonal(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'offset' (line 1317)
    offset_4477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 38), 'offset', False)
    # Getting the type of 'axis1' (line 1317)
    axis1_4478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 46), 'axis1', False)
    # Getting the type of 'axis2' (line 1317)
    axis2_4479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 53), 'axis2', False)
    # Processing the call keyword arguments (line 1317)
    kwargs_4480 = {}
    
    # Call to asanyarray(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'a' (line 1317)
    a_4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 26), 'a', False)
    # Processing the call keyword arguments (line 1317)
    kwargs_4474 = {}
    # Getting the type of 'asanyarray' (line 1317)
    asanyarray_4472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 15), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1317)
    asanyarray_call_result_4475 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 15), asanyarray_4472, *[a_4473], **kwargs_4474)
    
    # Obtaining the member 'diagonal' of a type (line 1317)
    diagonal_4476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1317, 15), asanyarray_call_result_4475, 'diagonal')
    # Calling diagonal(args, kwargs) (line 1317)
    diagonal_call_result_4481 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 15), diagonal_4476, *[offset_4477, axis1_4478, axis2_4479], **kwargs_4480)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 8), 'stypy_return_type', diagonal_call_result_4481)
    # SSA join for if statement (line 1313)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'diagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 1207)
    stypy_return_type_4482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4482)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diagonal'
    return stypy_return_type_4482

# Assigning a type to the variable 'diagonal' (line 1207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 0), 'diagonal', diagonal)

@norecursion
def trace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_4483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 20), 'int')
    int_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 29), 'int')
    int_4485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 38), 'int')
    # Getting the type of 'None' (line 1320)
    None_4486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 47), 'None')
    # Getting the type of 'None' (line 1320)
    None_4487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 57), 'None')
    defaults = [int_4483, int_4484, int_4485, None_4486, None_4487]
    # Create a new context for function 'trace'
    module_type_store = module_type_store.open_function_context('trace', 1320, 0, False)
    
    # Passed parameters checking function
    trace.stypy_localization = localization
    trace.stypy_type_of_self = None
    trace.stypy_type_store = module_type_store
    trace.stypy_function_name = 'trace'
    trace.stypy_param_names_list = ['a', 'offset', 'axis1', 'axis2', 'dtype', 'out']
    trace.stypy_varargs_param_name = None
    trace.stypy_kwargs_param_name = None
    trace.stypy_call_defaults = defaults
    trace.stypy_call_varargs = varargs
    trace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'trace', ['a', 'offset', 'axis1', 'axis2', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'trace', localization, ['a', 'offset', 'axis1', 'axis2', 'dtype', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'trace(...)' code ##################

    str_4488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1375, (-1)), 'str', '\n    Return the sum along diagonals of the array.\n\n    If `a` is 2-D, the sum along its diagonal with the given offset\n    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.\n\n    If `a` has more than two dimensions, then the axes specified by axis1 and\n    axis2 are used to determine the 2-D sub-arrays whose traces are returned.\n    The shape of the resulting array is the same as that of `a` with `axis1`\n    and `axis2` removed.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, from which the diagonals are taken.\n    offset : int, optional\n        Offset of the diagonal from the main diagonal. Can be both positive\n        and negative. Defaults to 0.\n    axis1, axis2 : int, optional\n        Axes to be used as the first and second axis of the 2-D sub-arrays\n        from which the diagonals should be taken. Defaults are the first two\n        axes of `a`.\n    dtype : dtype, optional\n        Determines the data-type of the returned array and of the accumulator\n        where the elements are summed. If dtype has the value None and `a` is\n        of integer type of precision less than the default integer\n        precision, then the default integer precision is used. Otherwise,\n        the precision is the same as that of `a`.\n    out : ndarray, optional\n        Array into which the output is placed. Its type is preserved and\n        it must be of the right shape to hold the output.\n\n    Returns\n    -------\n    sum_along_diagonals : ndarray\n        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has\n        larger dimensions, then an array of sums along diagonals is returned.\n\n    See Also\n    --------\n    diag, diagonal, diagflat\n\n    Examples\n    --------\n    >>> np.trace(np.eye(3))\n    3.0\n    >>> a = np.arange(8).reshape((2,2,2))\n    >>> np.trace(a)\n    array([6, 8])\n\n    >>> a = np.arange(24).reshape((2,2,2,3))\n    >>> np.trace(a).shape\n    (2, 3)\n\n    ')
    
    
    # Call to isinstance(...): (line 1376)
    # Processing the call arguments (line 1376)
    # Getting the type of 'a' (line 1376)
    a_4490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 18), 'a', False)
    # Getting the type of 'np' (line 1376)
    np_4491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 21), 'np', False)
    # Obtaining the member 'matrix' of a type (line 1376)
    matrix_4492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 21), np_4491, 'matrix')
    # Processing the call keyword arguments (line 1376)
    kwargs_4493 = {}
    # Getting the type of 'isinstance' (line 1376)
    isinstance_4489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1376)
    isinstance_call_result_4494 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 7), isinstance_4489, *[a_4490, matrix_4492], **kwargs_4493)
    
    # Testing the type of an if condition (line 1376)
    if_condition_4495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1376, 4), isinstance_call_result_4494)
    # Assigning a type to the variable 'if_condition_4495' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'if_condition_4495', if_condition_4495)
    # SSA begins for if statement (line 1376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to trace(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'offset' (line 1378)
    offset_4501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 32), 'offset', False)
    # Getting the type of 'axis1' (line 1378)
    axis1_4502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 40), 'axis1', False)
    # Getting the type of 'axis2' (line 1378)
    axis2_4503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 47), 'axis2', False)
    # Getting the type of 'dtype' (line 1378)
    dtype_4504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 54), 'dtype', False)
    # Getting the type of 'out' (line 1378)
    out_4505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 61), 'out', False)
    # Processing the call keyword arguments (line 1378)
    kwargs_4506 = {}
    
    # Call to asarray(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'a' (line 1378)
    a_4497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 23), 'a', False)
    # Processing the call keyword arguments (line 1378)
    kwargs_4498 = {}
    # Getting the type of 'asarray' (line 1378)
    asarray_4496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 1378)
    asarray_call_result_4499 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 15), asarray_4496, *[a_4497], **kwargs_4498)
    
    # Obtaining the member 'trace' of a type (line 1378)
    trace_4500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1378, 15), asarray_call_result_4499, 'trace')
    # Calling trace(args, kwargs) (line 1378)
    trace_call_result_4507 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 15), trace_4500, *[offset_4501, axis1_4502, axis2_4503, dtype_4504, out_4505], **kwargs_4506)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 8), 'stypy_return_type', trace_call_result_4507)
    # SSA branch for the else part of an if statement (line 1376)
    module_type_store.open_ssa_branch('else')
    
    # Call to trace(...): (line 1380)
    # Processing the call arguments (line 1380)
    # Getting the type of 'offset' (line 1380)
    offset_4513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 35), 'offset', False)
    # Getting the type of 'axis1' (line 1380)
    axis1_4514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 43), 'axis1', False)
    # Getting the type of 'axis2' (line 1380)
    axis2_4515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 50), 'axis2', False)
    # Getting the type of 'dtype' (line 1380)
    dtype_4516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 57), 'dtype', False)
    # Getting the type of 'out' (line 1380)
    out_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 64), 'out', False)
    # Processing the call keyword arguments (line 1380)
    kwargs_4518 = {}
    
    # Call to asanyarray(...): (line 1380)
    # Processing the call arguments (line 1380)
    # Getting the type of 'a' (line 1380)
    a_4509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 26), 'a', False)
    # Processing the call keyword arguments (line 1380)
    kwargs_4510 = {}
    # Getting the type of 'asanyarray' (line 1380)
    asanyarray_4508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 15), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1380)
    asanyarray_call_result_4511 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 15), asanyarray_4508, *[a_4509], **kwargs_4510)
    
    # Obtaining the member 'trace' of a type (line 1380)
    trace_4512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 15), asanyarray_call_result_4511, 'trace')
    # Calling trace(args, kwargs) (line 1380)
    trace_call_result_4519 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 15), trace_4512, *[offset_4513, axis1_4514, axis2_4515, dtype_4516, out_4517], **kwargs_4518)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 8), 'stypy_return_type', trace_call_result_4519)
    # SSA join for if statement (line 1376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'trace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'trace' in the type store
    # Getting the type of 'stypy_return_type' (line 1320)
    stypy_return_type_4520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4520)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'trace'
    return stypy_return_type_4520

# Assigning a type to the variable 'trace' (line 1320)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 0), 'trace', trace)

@norecursion
def ravel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_4521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 19), 'str', 'C')
    defaults = [str_4521]
    # Create a new context for function 'ravel'
    module_type_store = module_type_store.open_function_context('ravel', 1383, 0, False)
    
    # Passed parameters checking function
    ravel.stypy_localization = localization
    ravel.stypy_type_of_self = None
    ravel.stypy_type_store = module_type_store
    ravel.stypy_function_name = 'ravel'
    ravel.stypy_param_names_list = ['a', 'order']
    ravel.stypy_varargs_param_name = None
    ravel.stypy_kwargs_param_name = None
    ravel.stypy_call_defaults = defaults
    ravel.stypy_call_varargs = varargs
    ravel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ravel', ['a', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ravel', localization, ['a', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ravel(...)' code ##################

    str_4522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1483, (-1)), 'str', "Return a contiguous flattened array.\n\n    A 1-D array, containing the elements of the input, is returned.  A copy is\n    made only if needed.\n\n    As of NumPy 1.10, the returned array will have the same type as the input\n    array. (for example, a masked array will be returned for a masked array\n    input)\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.  The elements in `a` are read in the order specified by\n        `order`, and packed as a 1-D array.\n    order : {'C','F', 'A', 'K'}, optional\n\n        The elements of `a` are read using this index order. 'C' means\n        to index the elements in row-major, C-style order,\n        with the last axis index changing fastest, back to the first\n        axis index changing slowest.  'F' means to index the elements\n        in column-major, Fortran-style order, with the\n        first index changing fastest, and the last index changing\n        slowest. Note that the 'C' and 'F' options take no account of\n        the memory layout of the underlying array, and only refer to\n        the order of axis indexing.  'A' means to read the elements in\n        Fortran-like index order if `a` is Fortran *contiguous* in\n        memory, C-like order otherwise.  'K' means to read the\n        elements in the order they occur in memory, except for\n        reversing the data when strides are negative.  By default, 'C'\n        index order is used.\n\n    Returns\n    -------\n    y : array_like\n        If `a` is a matrix, y is a 1-D ndarray, otherwise y is an array of\n        the same subtype as `a`. The shape of the returned array is\n        ``(a.size,)``. Matrices are special cased for backward\n        compatibility.\n\n    See Also\n    --------\n    ndarray.flat : 1-D iterator over an array.\n    ndarray.flatten : 1-D array copy of the elements of an array\n                      in row-major order.\n    ndarray.reshape : Change the shape of an array without changing its data.\n\n    Notes\n    -----\n    In row-major, C-style order, in two dimensions, the row index\n    varies the slowest, and the column index the quickest.  This can\n    be generalized to multiple dimensions, where row-major order\n    implies that the index along the first axis varies slowest, and\n    the index along the last quickest.  The opposite holds for\n    column-major, Fortran-style index ordering.\n\n    When a view is desired in as many cases as possible, ``arr.reshape(-1)``\n    may be preferable.\n\n    Examples\n    --------\n    It is equivalent to ``reshape(-1, order=order)``.\n\n    >>> x = np.array([[1, 2, 3], [4, 5, 6]])\n    >>> print(np.ravel(x))\n    [1 2 3 4 5 6]\n\n    >>> print(x.reshape(-1))\n    [1 2 3 4 5 6]\n\n    >>> print(np.ravel(x, order='F'))\n    [1 4 2 5 3 6]\n\n    When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:\n\n    >>> print(np.ravel(x.T))\n    [1 4 2 5 3 6]\n    >>> print(np.ravel(x.T, order='A'))\n    [1 2 3 4 5 6]\n\n    When ``order`` is 'K', it will preserve orderings that are neither 'C'\n    nor 'F', but won't reverse axes:\n\n    >>> a = np.arange(3)[::-1]; a\n    array([2, 1, 0])\n    >>> a.ravel(order='C')\n    array([2, 1, 0])\n    >>> a.ravel(order='K')\n    array([2, 1, 0])\n\n    >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a\n    array([[[ 0,  2,  4],\n            [ 1,  3,  5]],\n           [[ 6,  8, 10],\n            [ 7,  9, 11]]])\n    >>> a.ravel(order='C')\n    array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])\n    >>> a.ravel(order='K')\n    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])\n\n    ")
    
    
    # Call to isinstance(...): (line 1484)
    # Processing the call arguments (line 1484)
    # Getting the type of 'a' (line 1484)
    a_4524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 18), 'a', False)
    # Getting the type of 'np' (line 1484)
    np_4525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 21), 'np', False)
    # Obtaining the member 'matrix' of a type (line 1484)
    matrix_4526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1484, 21), np_4525, 'matrix')
    # Processing the call keyword arguments (line 1484)
    kwargs_4527 = {}
    # Getting the type of 'isinstance' (line 1484)
    isinstance_4523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1484)
    isinstance_call_result_4528 = invoke(stypy.reporting.localization.Localization(__file__, 1484, 7), isinstance_4523, *[a_4524, matrix_4526], **kwargs_4527)
    
    # Testing the type of an if condition (line 1484)
    if_condition_4529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1484, 4), isinstance_call_result_4528)
    # Assigning a type to the variable 'if_condition_4529' (line 1484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1484, 4), 'if_condition_4529', if_condition_4529)
    # SSA begins for if statement (line 1484)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ravel(...): (line 1485)
    # Processing the call arguments (line 1485)
    # Getting the type of 'order' (line 1485)
    order_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 32), 'order', False)
    # Processing the call keyword arguments (line 1485)
    kwargs_4536 = {}
    
    # Call to asarray(...): (line 1485)
    # Processing the call arguments (line 1485)
    # Getting the type of 'a' (line 1485)
    a_4531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 23), 'a', False)
    # Processing the call keyword arguments (line 1485)
    kwargs_4532 = {}
    # Getting the type of 'asarray' (line 1485)
    asarray_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 1485)
    asarray_call_result_4533 = invoke(stypy.reporting.localization.Localization(__file__, 1485, 15), asarray_4530, *[a_4531], **kwargs_4532)
    
    # Obtaining the member 'ravel' of a type (line 1485)
    ravel_4534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1485, 15), asarray_call_result_4533, 'ravel')
    # Calling ravel(args, kwargs) (line 1485)
    ravel_call_result_4537 = invoke(stypy.reporting.localization.Localization(__file__, 1485, 15), ravel_4534, *[order_4535], **kwargs_4536)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1485, 8), 'stypy_return_type', ravel_call_result_4537)
    # SSA branch for the else part of an if statement (line 1484)
    module_type_store.open_ssa_branch('else')
    
    # Call to ravel(...): (line 1487)
    # Processing the call arguments (line 1487)
    # Getting the type of 'order' (line 1487)
    order_4543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 35), 'order', False)
    # Processing the call keyword arguments (line 1487)
    kwargs_4544 = {}
    
    # Call to asanyarray(...): (line 1487)
    # Processing the call arguments (line 1487)
    # Getting the type of 'a' (line 1487)
    a_4539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 26), 'a', False)
    # Processing the call keyword arguments (line 1487)
    kwargs_4540 = {}
    # Getting the type of 'asanyarray' (line 1487)
    asanyarray_4538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 15), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1487)
    asanyarray_call_result_4541 = invoke(stypy.reporting.localization.Localization(__file__, 1487, 15), asanyarray_4538, *[a_4539], **kwargs_4540)
    
    # Obtaining the member 'ravel' of a type (line 1487)
    ravel_4542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1487, 15), asanyarray_call_result_4541, 'ravel')
    # Calling ravel(args, kwargs) (line 1487)
    ravel_call_result_4545 = invoke(stypy.reporting.localization.Localization(__file__, 1487, 15), ravel_4542, *[order_4543], **kwargs_4544)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1487, 8), 'stypy_return_type', ravel_call_result_4545)
    # SSA join for if statement (line 1484)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'ravel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ravel' in the type store
    # Getting the type of 'stypy_return_type' (line 1383)
    stypy_return_type_4546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4546)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ravel'
    return stypy_return_type_4546

# Assigning a type to the variable 'ravel' (line 1383)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1383, 0), 'ravel', ravel)

@norecursion
def nonzero(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nonzero'
    module_type_store = module_type_store.open_function_context('nonzero', 1490, 0, False)
    
    # Passed parameters checking function
    nonzero.stypy_localization = localization
    nonzero.stypy_type_of_self = None
    nonzero.stypy_type_store = module_type_store
    nonzero.stypy_function_name = 'nonzero'
    nonzero.stypy_param_names_list = ['a']
    nonzero.stypy_varargs_param_name = None
    nonzero.stypy_kwargs_param_name = None
    nonzero.stypy_call_defaults = defaults
    nonzero.stypy_call_varargs = varargs
    nonzero.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nonzero', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nonzero', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nonzero(...)' code ##################

    str_4547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1564, (-1)), 'str', '\n    Return the indices of the elements that are non-zero.\n\n    Returns a tuple of arrays, one for each dimension of `a`,\n    containing the indices of the non-zero elements in that\n    dimension. The values in `a` are always tested and returned in\n    row-major, C-style order. The corresponding non-zero\n    values can be obtained with::\n\n        a[nonzero(a)]\n\n    To group the indices by element, rather than dimension, use::\n\n        transpose(nonzero(a))\n\n    The result of this is always a 2-D array, with a row for\n    each non-zero element.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n\n    Returns\n    -------\n    tuple_of_arrays : tuple\n        Indices of elements that are non-zero.\n\n    See Also\n    --------\n    flatnonzero :\n        Return indices that are non-zero in the flattened version of the input\n        array.\n    ndarray.nonzero :\n        Equivalent ndarray method.\n    count_nonzero :\n        Counts the number of non-zero elements in the input array.\n\n    Examples\n    --------\n    >>> x = np.eye(3)\n    >>> x\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n    >>> np.nonzero(x)\n    (array([0, 1, 2]), array([0, 1, 2]))\n\n    >>> x[np.nonzero(x)]\n    array([ 1.,  1.,  1.])\n    >>> np.transpose(np.nonzero(x))\n    array([[0, 0],\n           [1, 1],\n           [2, 2]])\n\n    A common use for ``nonzero`` is to find the indices of an array, where\n    a condition is True.  Given an array `a`, the condition `a` > 3 is a\n    boolean array and since False is interpreted as 0, np.nonzero(a > 3)\n    yields the indices of the `a` where the condition is true.\n\n    >>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])\n    >>> a > 3\n    array([[False, False, False],\n           [ True,  True,  True],\n           [ True,  True,  True]], dtype=bool)\n    >>> np.nonzero(a > 3)\n    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))\n\n    The ``nonzero`` method of the boolean array can also be called.\n\n    >>> (a > 3).nonzero()\n    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))\n\n    ')
    
    
    # SSA begins for try-except statement (line 1565)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1566):
    # Getting the type of 'a' (line 1566)
    a_4548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 18), 'a')
    # Obtaining the member 'nonzero' of a type (line 1566)
    nonzero_4549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1566, 18), a_4548, 'nonzero')
    # Assigning a type to the variable 'nonzero' (line 1566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1566, 8), 'nonzero', nonzero_4549)
    # SSA branch for the except part of a try statement (line 1565)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1565)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 1568):
    
    # Call to _wrapit(...): (line 1568)
    # Processing the call arguments (line 1568)
    # Getting the type of 'a' (line 1568)
    a_4551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1568, 22), 'a', False)
    str_4552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1568, 25), 'str', 'nonzero')
    # Processing the call keyword arguments (line 1568)
    kwargs_4553 = {}
    # Getting the type of '_wrapit' (line 1568)
    _wrapit_4550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1568, 14), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1568)
    _wrapit_call_result_4554 = invoke(stypy.reporting.localization.Localization(__file__, 1568, 14), _wrapit_4550, *[a_4551, str_4552], **kwargs_4553)
    
    # Assigning a type to the variable 'res' (line 1568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1568, 8), 'res', _wrapit_call_result_4554)
    # SSA branch for the else branch of a try statement (line 1565)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Call to a Name (line 1570):
    
    # Call to nonzero(...): (line 1570)
    # Processing the call keyword arguments (line 1570)
    kwargs_4556 = {}
    # Getting the type of 'nonzero' (line 1570)
    nonzero_4555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1570, 14), 'nonzero', False)
    # Calling nonzero(args, kwargs) (line 1570)
    nonzero_call_result_4557 = invoke(stypy.reporting.localization.Localization(__file__, 1570, 14), nonzero_4555, *[], **kwargs_4556)
    
    # Assigning a type to the variable 'res' (line 1570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1570, 8), 'res', nonzero_call_result_4557)
    # SSA join for try-except statement (line 1565)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 1571)
    res_4558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1571, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 1571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1571, 4), 'stypy_return_type', res_4558)
    
    # ################# End of 'nonzero(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nonzero' in the type store
    # Getting the type of 'stypy_return_type' (line 1490)
    stypy_return_type_4559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4559)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nonzero'
    return stypy_return_type_4559

# Assigning a type to the variable 'nonzero' (line 1490)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 0), 'nonzero', nonzero)

@norecursion
def shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'shape'
    module_type_store = module_type_store.open_function_context('shape', 1574, 0, False)
    
    # Passed parameters checking function
    shape.stypy_localization = localization
    shape.stypy_type_of_self = None
    shape.stypy_type_store = module_type_store
    shape.stypy_function_name = 'shape'
    shape.stypy_param_names_list = ['a']
    shape.stypy_varargs_param_name = None
    shape.stypy_kwargs_param_name = None
    shape.stypy_call_defaults = defaults
    shape.stypy_call_varargs = varargs
    shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shape', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shape', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shape(...)' code ##################

    str_4560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1611, (-1)), 'str', "\n    Return the shape of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n\n    Returns\n    -------\n    shape : tuple of ints\n        The elements of the shape tuple give the lengths of the\n        corresponding array dimensions.\n\n    See Also\n    --------\n    alen\n    ndarray.shape : Equivalent array method.\n\n    Examples\n    --------\n    >>> np.shape(np.eye(3))\n    (3, 3)\n    >>> np.shape([[1, 2]])\n    (1, 2)\n    >>> np.shape([0])\n    (1,)\n    >>> np.shape(0)\n    ()\n\n    >>> a = np.array([(1, 2), (3, 4)], dtype=[('x', 'i4'), ('y', 'i4')])\n    >>> np.shape(a)\n    (2,)\n    >>> a.shape\n    (2,)\n\n    ")
    
    
    # SSA begins for try-except statement (line 1612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1613):
    # Getting the type of 'a' (line 1613)
    a_4561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1613, 17), 'a')
    # Obtaining the member 'shape' of a type (line 1613)
    shape_4562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1613, 17), a_4561, 'shape')
    # Assigning a type to the variable 'result' (line 1613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1613, 8), 'result', shape_4562)
    # SSA branch for the except part of a try statement (line 1612)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1612)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Attribute to a Name (line 1615):
    
    # Call to asarray(...): (line 1615)
    # Processing the call arguments (line 1615)
    # Getting the type of 'a' (line 1615)
    a_4564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 25), 'a', False)
    # Processing the call keyword arguments (line 1615)
    kwargs_4565 = {}
    # Getting the type of 'asarray' (line 1615)
    asarray_4563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 17), 'asarray', False)
    # Calling asarray(args, kwargs) (line 1615)
    asarray_call_result_4566 = invoke(stypy.reporting.localization.Localization(__file__, 1615, 17), asarray_4563, *[a_4564], **kwargs_4565)
    
    # Obtaining the member 'shape' of a type (line 1615)
    shape_4567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1615, 17), asarray_call_result_4566, 'shape')
    # Assigning a type to the variable 'result' (line 1615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1615, 8), 'result', shape_4567)
    # SSA join for try-except statement (line 1612)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 1616)
    result_4568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1616, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1616, 4), 'stypy_return_type', result_4568)
    
    # ################# End of 'shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shape' in the type store
    # Getting the type of 'stypy_return_type' (line 1574)
    stypy_return_type_4569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4569)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shape'
    return stypy_return_type_4569

# Assigning a type to the variable 'shape' (line 1574)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1574, 0), 'shape', shape)

@norecursion
def compress(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1619)
    None_4570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 32), 'None')
    # Getting the type of 'None' (line 1619)
    None_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 42), 'None')
    defaults = [None_4570, None_4571]
    # Create a new context for function 'compress'
    module_type_store = module_type_store.open_function_context('compress', 1619, 0, False)
    
    # Passed parameters checking function
    compress.stypy_localization = localization
    compress.stypy_type_of_self = None
    compress.stypy_type_store = module_type_store
    compress.stypy_function_name = 'compress'
    compress.stypy_param_names_list = ['condition', 'a', 'axis', 'out']
    compress.stypy_varargs_param_name = None
    compress.stypy_kwargs_param_name = None
    compress.stypy_call_defaults = defaults
    compress.stypy_call_varargs = varargs
    compress.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compress', ['condition', 'a', 'axis', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compress', localization, ['condition', 'a', 'axis', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compress(...)' code ##################

    str_4572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1678, (-1)), 'str', '\n    Return selected slices of an array along given axis.\n\n    When working along a given axis, a slice along that axis is returned in\n    `output` for each index where `condition` evaluates to True. When\n    working on a 1-D array, `compress` is equivalent to `extract`.\n\n    Parameters\n    ----------\n    condition : 1-D array of bools\n        Array that selects which entries to return. If len(condition)\n        is less than the size of `a` along the given axis, then output is\n        truncated to the length of the condition array.\n    a : array_like\n        Array from which to extract a part.\n    axis : int, optional\n        Axis along which to take slices. If None (default), work on the\n        flattened array.\n    out : ndarray, optional\n        Output array.  Its type is preserved and it must be of the right\n        shape to hold the output.\n\n    Returns\n    -------\n    compressed_array : ndarray\n        A copy of `a` without the slices along axis for which `condition`\n        is false.\n\n    See Also\n    --------\n    take, choose, diag, diagonal, select\n    ndarray.compress : Equivalent method in ndarray\n    np.extract: Equivalent method when working on 1-D arrays\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4], [5, 6]])\n    >>> a\n    array([[1, 2],\n           [3, 4],\n           [5, 6]])\n    >>> np.compress([0, 1], a, axis=0)\n    array([[3, 4]])\n    >>> np.compress([False, True, True], a, axis=0)\n    array([[3, 4],\n           [5, 6]])\n    >>> np.compress([False, True], a, axis=1)\n    array([[2],\n           [4],\n           [6]])\n\n    Working on the flattened array does not return slices along an axis but\n    selects elements.\n\n    >>> np.compress([False, True], a)\n    array([2])\n\n    ')
    
    
    # SSA begins for try-except statement (line 1679)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1680):
    # Getting the type of 'a' (line 1680)
    a_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1680, 19), 'a')
    # Obtaining the member 'compress' of a type (line 1680)
    compress_4574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1680, 19), a_4573, 'compress')
    # Assigning a type to the variable 'compress' (line 1680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1680, 8), 'compress', compress_4574)
    # SSA branch for the except part of a try statement (line 1679)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1679)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 1682)
    # Processing the call arguments (line 1682)
    # Getting the type of 'a' (line 1682)
    a_4576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 23), 'a', False)
    str_4577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1682, 26), 'str', 'compress')
    # Getting the type of 'condition' (line 1682)
    condition_4578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 38), 'condition', False)
    # Getting the type of 'axis' (line 1682)
    axis_4579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 49), 'axis', False)
    # Getting the type of 'out' (line 1682)
    out_4580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 55), 'out', False)
    # Processing the call keyword arguments (line 1682)
    kwargs_4581 = {}
    # Getting the type of '_wrapit' (line 1682)
    _wrapit_4575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1682, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1682)
    _wrapit_call_result_4582 = invoke(stypy.reporting.localization.Localization(__file__, 1682, 15), _wrapit_4575, *[a_4576, str_4577, condition_4578, axis_4579, out_4580], **kwargs_4581)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1682, 8), 'stypy_return_type', _wrapit_call_result_4582)
    # SSA join for try-except statement (line 1679)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to compress(...): (line 1683)
    # Processing the call arguments (line 1683)
    # Getting the type of 'condition' (line 1683)
    condition_4584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 20), 'condition', False)
    # Getting the type of 'axis' (line 1683)
    axis_4585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 31), 'axis', False)
    # Getting the type of 'out' (line 1683)
    out_4586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 37), 'out', False)
    # Processing the call keyword arguments (line 1683)
    kwargs_4587 = {}
    # Getting the type of 'compress' (line 1683)
    compress_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1683, 11), 'compress', False)
    # Calling compress(args, kwargs) (line 1683)
    compress_call_result_4588 = invoke(stypy.reporting.localization.Localization(__file__, 1683, 11), compress_4583, *[condition_4584, axis_4585, out_4586], **kwargs_4587)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1683, 4), 'stypy_return_type', compress_call_result_4588)
    
    # ################# End of 'compress(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compress' in the type store
    # Getting the type of 'stypy_return_type' (line 1619)
    stypy_return_type_4589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4589)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compress'
    return stypy_return_type_4589

# Assigning a type to the variable 'compress' (line 1619)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1619, 0), 'compress', compress)

@norecursion
def clip(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1686)
    None_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 30), 'None')
    defaults = [None_4590]
    # Create a new context for function 'clip'
    module_type_store = module_type_store.open_function_context('clip', 1686, 0, False)
    
    # Passed parameters checking function
    clip.stypy_localization = localization
    clip.stypy_type_of_self = None
    clip.stypy_type_store = module_type_store
    clip.stypy_function_name = 'clip'
    clip.stypy_param_names_list = ['a', 'a_min', 'a_max', 'out']
    clip.stypy_varargs_param_name = None
    clip.stypy_kwargs_param_name = None
    clip.stypy_call_defaults = defaults
    clip.stypy_call_varargs = varargs
    clip.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clip', ['a', 'a_min', 'a_max', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clip', localization, ['a', 'a_min', 'a_max', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clip(...)' code ##################

    str_4591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1735, (-1)), 'str', '\n    Clip (limit) the values in an array.\n\n    Given an interval, values outside the interval are clipped to\n    the interval edges.  For example, if an interval of ``[0, 1]``\n    is specified, values smaller than 0 become 0, and values larger\n    than 1 become 1.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing elements to clip.\n    a_min : scalar or array_like\n        Minimum value.\n    a_max : scalar or array_like\n        Maximum value.  If `a_min` or `a_max` are array_like, then they will\n        be broadcasted to the shape of `a`.\n    out : ndarray, optional\n        The results will be placed in this array. It may be the input\n        array for in-place clipping.  `out` must be of the right shape\n        to hold the output.  Its type is preserved.\n\n    Returns\n    -------\n    clipped_array : ndarray\n        An array with the elements of `a`, but where values\n        < `a_min` are replaced with `a_min`, and those > `a_max`\n        with `a_max`.\n\n    See Also\n    --------\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Examples\n    --------\n    >>> a = np.arange(10)\n    >>> np.clip(a, 1, 8)\n    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])\n    >>> a\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n    >>> np.clip(a, 3, 6, out=a)\n    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])\n    >>> a = np.arange(10)\n    >>> a\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n    >>> np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)\n    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])\n\n    ')
    
    
    # SSA begins for try-except statement (line 1736)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1737):
    # Getting the type of 'a' (line 1737)
    a_4592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 15), 'a')
    # Obtaining the member 'clip' of a type (line 1737)
    clip_4593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1737, 15), a_4592, 'clip')
    # Assigning a type to the variable 'clip' (line 1737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1737, 8), 'clip', clip_4593)
    # SSA branch for the except part of a try statement (line 1736)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1736)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 1739)
    # Processing the call arguments (line 1739)
    # Getting the type of 'a' (line 1739)
    a_4595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 23), 'a', False)
    str_4596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1739, 26), 'str', 'clip')
    # Getting the type of 'a_min' (line 1739)
    a_min_4597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 34), 'a_min', False)
    # Getting the type of 'a_max' (line 1739)
    a_max_4598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 41), 'a_max', False)
    # Getting the type of 'out' (line 1739)
    out_4599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 48), 'out', False)
    # Processing the call keyword arguments (line 1739)
    kwargs_4600 = {}
    # Getting the type of '_wrapit' (line 1739)
    _wrapit_4594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1739, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 1739)
    _wrapit_call_result_4601 = invoke(stypy.reporting.localization.Localization(__file__, 1739, 15), _wrapit_4594, *[a_4595, str_4596, a_min_4597, a_max_4598, out_4599], **kwargs_4600)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1739, 8), 'stypy_return_type', _wrapit_call_result_4601)
    # SSA join for try-except statement (line 1736)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to clip(...): (line 1740)
    # Processing the call arguments (line 1740)
    # Getting the type of 'a_min' (line 1740)
    a_min_4603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 16), 'a_min', False)
    # Getting the type of 'a_max' (line 1740)
    a_max_4604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 23), 'a_max', False)
    # Getting the type of 'out' (line 1740)
    out_4605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 30), 'out', False)
    # Processing the call keyword arguments (line 1740)
    kwargs_4606 = {}
    # Getting the type of 'clip' (line 1740)
    clip_4602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 11), 'clip', False)
    # Calling clip(args, kwargs) (line 1740)
    clip_call_result_4607 = invoke(stypy.reporting.localization.Localization(__file__, 1740, 11), clip_4602, *[a_min_4603, a_max_4604, out_4605], **kwargs_4606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1740, 4), 'stypy_return_type', clip_call_result_4607)
    
    # ################# End of 'clip(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clip' in the type store
    # Getting the type of 'stypy_return_type' (line 1686)
    stypy_return_type_4608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4608)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clip'
    return stypy_return_type_4608

# Assigning a type to the variable 'clip' (line 1686)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1686, 0), 'clip', clip)

@norecursion
def sum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1743)
    None_4609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 16), 'None')
    # Getting the type of 'None' (line 1743)
    None_4610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 28), 'None')
    # Getting the type of 'None' (line 1743)
    None_4611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 38), 'None')
    # Getting the type of 'False' (line 1743)
    False_4612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 53), 'False')
    defaults = [None_4609, None_4610, None_4611, False_4612]
    # Create a new context for function 'sum'
    module_type_store = module_type_store.open_function_context('sum', 1743, 0, False)
    
    # Passed parameters checking function
    sum.stypy_localization = localization
    sum.stypy_type_of_self = None
    sum.stypy_type_store = module_type_store
    sum.stypy_function_name = 'sum'
    sum.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    sum.stypy_varargs_param_name = None
    sum.stypy_kwargs_param_name = None
    sum.stypy_call_defaults = defaults
    sum.stypy_call_varargs = varargs
    sum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sum', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sum', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sum(...)' code ##################

    str_4613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1823, (-1)), 'str', '\n    Sum of array elements over a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Elements to sum.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which a sum is performed.  The default,\n        axis=None, will sum all of the elements of the input array.  If\n        axis is negative it counts from the last to the first axis.\n\n        .. versionadded:: 1.7.0\n\n        If axis is a tuple of ints, a sum is performed on all of the axes\n        specified in the tuple instead of a single axis or all the axes as\n        before.\n    dtype : dtype, optional\n        The type of the returned array and of the accumulator in which the\n        elements are summed.  The dtype of `a` is used by default unless `a`\n        has an integer dtype of less precision than the default platform\n        integer.  In that case, if `a` is signed then the platform integer\n        is used while if `a` is unsigned then an unsigned integer of the\n        same precision as the platform integer is used.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape as the expected output, but the type of the output\n        values will be cast if necessary.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the input array.\n\n    Returns\n    -------\n    sum_along_axis : ndarray\n        An array with the same shape as `a`, with the specified\n        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar\n        is returned.  If an output array is specified, a reference to\n        `out` is returned.\n\n    See Also\n    --------\n    ndarray.sum : Equivalent method.\n\n    cumsum : Cumulative sum of array elements.\n\n    trapz : Integration of array values using the composite trapezoidal rule.\n\n    mean, average\n\n    Notes\n    -----\n    Arithmetic is modular when using integer types, and no error is\n    raised on overflow.\n\n    The sum of an empty array is the neutral element 0:\n\n    >>> np.sum([])\n    0.0\n\n    Examples\n    --------\n    >>> np.sum([0.5, 1.5])\n    2.0\n    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)\n    1\n    >>> np.sum([[0, 1], [0, 5]])\n    6\n    >>> np.sum([[0, 1], [0, 5]], axis=0)\n    array([0, 6])\n    >>> np.sum([[0, 1], [0, 5]], axis=1)\n    array([1, 5])\n\n    If the accumulator is too small, overflow occurs:\n\n    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)\n    -128\n\n    ')
    
    
    # Call to isinstance(...): (line 1824)
    # Processing the call arguments (line 1824)
    # Getting the type of 'a' (line 1824)
    a_4615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 18), 'a', False)
    # Getting the type of '_gentype' (line 1824)
    _gentype_4616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 21), '_gentype', False)
    # Processing the call keyword arguments (line 1824)
    kwargs_4617 = {}
    # Getting the type of 'isinstance' (line 1824)
    isinstance_4614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1824, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1824)
    isinstance_call_result_4618 = invoke(stypy.reporting.localization.Localization(__file__, 1824, 7), isinstance_4614, *[a_4615, _gentype_4616], **kwargs_4617)
    
    # Testing the type of an if condition (line 1824)
    if_condition_4619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1824, 4), isinstance_call_result_4618)
    # Assigning a type to the variable 'if_condition_4619' (line 1824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1824, 4), 'if_condition_4619', if_condition_4619)
    # SSA begins for if statement (line 1824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1825):
    
    # Call to _sum_(...): (line 1825)
    # Processing the call arguments (line 1825)
    # Getting the type of 'a' (line 1825)
    a_4621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 20), 'a', False)
    # Processing the call keyword arguments (line 1825)
    kwargs_4622 = {}
    # Getting the type of '_sum_' (line 1825)
    _sum__4620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1825, 14), '_sum_', False)
    # Calling _sum_(args, kwargs) (line 1825)
    _sum__call_result_4623 = invoke(stypy.reporting.localization.Localization(__file__, 1825, 14), _sum__4620, *[a_4621], **kwargs_4622)
    
    # Assigning a type to the variable 'res' (line 1825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1825, 8), 'res', _sum__call_result_4623)
    
    # Type idiom detected: calculating its left and rigth part (line 1826)
    # Getting the type of 'out' (line 1826)
    out_4624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 8), 'out')
    # Getting the type of 'None' (line 1826)
    None_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1826, 22), 'None')
    
    (may_be_4626, more_types_in_union_4627) = may_not_be_none(out_4624, None_4625)

    if may_be_4626:

        if more_types_in_union_4627:
            # Runtime conditional SSA (line 1826)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 1827):
        # Getting the type of 'res' (line 1827)
        res_4628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 23), 'res')
        # Getting the type of 'out' (line 1827)
        out_4629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1827, 12), 'out')
        Ellipsis_4630 = Ellipsis
        # Storing an element on a container (line 1827)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1827, 12), out_4629, (Ellipsis_4630, res_4628))
        # Getting the type of 'out' (line 1828)
        out_4631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1828, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 1828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1828, 12), 'stypy_return_type', out_4631)

        if more_types_in_union_4627:
            # SSA join for if statement (line 1826)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'res' (line 1829)
    res_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1829, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 1829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1829, 8), 'stypy_return_type', res_4632)
    # SSA branch for the else part of an if statement (line 1824)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to type(...): (line 1830)
    # Processing the call arguments (line 1830)
    # Getting the type of 'a' (line 1830)
    a_4634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 14), 'a', False)
    # Processing the call keyword arguments (line 1830)
    kwargs_4635 = {}
    # Getting the type of 'type' (line 1830)
    type_4633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 9), 'type', False)
    # Calling type(args, kwargs) (line 1830)
    type_call_result_4636 = invoke(stypy.reporting.localization.Localization(__file__, 1830, 9), type_4633, *[a_4634], **kwargs_4635)
    
    # Getting the type of 'mu' (line 1830)
    mu_4637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1830, 24), 'mu')
    # Obtaining the member 'ndarray' of a type (line 1830)
    ndarray_4638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1830, 24), mu_4637, 'ndarray')
    # Applying the binary operator 'isnot' (line 1830)
    result_is_not_4639 = python_operator(stypy.reporting.localization.Localization(__file__, 1830, 9), 'isnot', type_call_result_4636, ndarray_4638)
    
    # Testing the type of an if condition (line 1830)
    if_condition_4640 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1830, 9), result_is_not_4639)
    # Assigning a type to the variable 'if_condition_4640' (line 1830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1830, 9), 'if_condition_4640', if_condition_4640)
    # SSA begins for if statement (line 1830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 1831)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1832):
    # Getting the type of 'a' (line 1832)
    a_4641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1832, 18), 'a')
    # Obtaining the member 'sum' of a type (line 1832)
    sum_4642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1832, 18), a_4641, 'sum')
    # Assigning a type to the variable 'sum' (line 1832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1832, 12), 'sum', sum_4642)
    # SSA branch for the except part of a try statement (line 1831)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 1831)
    module_type_store.open_ssa_branch('except')
    
    # Call to _sum(...): (line 1834)
    # Processing the call arguments (line 1834)
    # Getting the type of 'a' (line 1834)
    a_4645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 33), 'a', False)
    # Processing the call keyword arguments (line 1834)
    # Getting the type of 'axis' (line 1834)
    axis_4646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 41), 'axis', False)
    keyword_4647 = axis_4646
    # Getting the type of 'dtype' (line 1834)
    dtype_4648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 53), 'dtype', False)
    keyword_4649 = dtype_4648
    # Getting the type of 'out' (line 1835)
    out_4650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 37), 'out', False)
    keyword_4651 = out_4650
    # Getting the type of 'keepdims' (line 1835)
    keepdims_4652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 51), 'keepdims', False)
    keyword_4653 = keepdims_4652
    kwargs_4654 = {'dtype': keyword_4649, 'out': keyword_4651, 'keepdims': keyword_4653, 'axis': keyword_4647}
    # Getting the type of '_methods' (line 1834)
    _methods_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1834, 19), '_methods', False)
    # Obtaining the member '_sum' of a type (line 1834)
    _sum_4644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1834, 19), _methods_4643, '_sum')
    # Calling _sum(args, kwargs) (line 1834)
    _sum_call_result_4655 = invoke(stypy.reporting.localization.Localization(__file__, 1834, 19), _sum_4644, *[a_4645], **kwargs_4654)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1834, 12), 'stypy_return_type', _sum_call_result_4655)
    # SSA join for try-except statement (line 1831)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sum(...): (line 1837)
    # Processing the call keyword arguments (line 1837)
    # Getting the type of 'axis' (line 1837)
    axis_4657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 24), 'axis', False)
    keyword_4658 = axis_4657
    # Getting the type of 'dtype' (line 1837)
    dtype_4659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 36), 'dtype', False)
    keyword_4660 = dtype_4659
    # Getting the type of 'out' (line 1837)
    out_4661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 47), 'out', False)
    keyword_4662 = out_4661
    kwargs_4663 = {'dtype': keyword_4660, 'out': keyword_4662, 'axis': keyword_4658}
    # Getting the type of 'sum' (line 1837)
    sum_4656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 15), 'sum', False)
    # Calling sum(args, kwargs) (line 1837)
    sum_call_result_4664 = invoke(stypy.reporting.localization.Localization(__file__, 1837, 15), sum_4656, *[], **kwargs_4663)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1837, 8), 'stypy_return_type', sum_call_result_4664)
    # SSA branch for the else part of an if statement (line 1830)
    module_type_store.open_ssa_branch('else')
    
    # Call to _sum(...): (line 1839)
    # Processing the call arguments (line 1839)
    # Getting the type of 'a' (line 1839)
    a_4667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 29), 'a', False)
    # Processing the call keyword arguments (line 1839)
    # Getting the type of 'axis' (line 1839)
    axis_4668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 37), 'axis', False)
    keyword_4669 = axis_4668
    # Getting the type of 'dtype' (line 1839)
    dtype_4670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 49), 'dtype', False)
    keyword_4671 = dtype_4670
    # Getting the type of 'out' (line 1840)
    out_4672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 33), 'out', False)
    keyword_4673 = out_4672
    # Getting the type of 'keepdims' (line 1840)
    keepdims_4674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 47), 'keepdims', False)
    keyword_4675 = keepdims_4674
    kwargs_4676 = {'dtype': keyword_4671, 'out': keyword_4673, 'keepdims': keyword_4675, 'axis': keyword_4669}
    # Getting the type of '_methods' (line 1839)
    _methods_4665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 15), '_methods', False)
    # Obtaining the member '_sum' of a type (line 1839)
    _sum_4666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1839, 15), _methods_4665, '_sum')
    # Calling _sum(args, kwargs) (line 1839)
    _sum_call_result_4677 = invoke(stypy.reporting.localization.Localization(__file__, 1839, 15), _sum_4666, *[a_4667], **kwargs_4676)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1839)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1839, 8), 'stypy_return_type', _sum_call_result_4677)
    # SSA join for if statement (line 1830)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1824)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 1743)
    stypy_return_type_4678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4678)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_4678

# Assigning a type to the variable 'sum' (line 1743)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1743, 0), 'sum', sum)

@norecursion
def product(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1843)
    None_4679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 20), 'None')
    # Getting the type of 'None' (line 1843)
    None_4680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 32), 'None')
    # Getting the type of 'None' (line 1843)
    None_4681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 42), 'None')
    # Getting the type of 'False' (line 1843)
    False_4682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 57), 'False')
    defaults = [None_4679, None_4680, None_4681, False_4682]
    # Create a new context for function 'product'
    module_type_store = module_type_store.open_function_context('product', 1843, 0, False)
    
    # Passed parameters checking function
    product.stypy_localization = localization
    product.stypy_type_of_self = None
    product.stypy_type_store = module_type_store
    product.stypy_function_name = 'product'
    product.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    product.stypy_varargs_param_name = None
    product.stypy_kwargs_param_name = None
    product.stypy_call_defaults = defaults
    product.stypy_call_varargs = varargs
    product.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'product', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'product', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'product(...)' code ##################

    str_4683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1851, (-1)), 'str', '\n    Return the product of array elements over a given axis.\n\n    See Also\n    --------\n    prod : equivalent function; see for details.\n\n    ')
    
    # Call to reduce(...): (line 1852)
    # Processing the call arguments (line 1852)
    # Getting the type of 'a' (line 1852)
    a_4687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 30), 'a', False)
    # Processing the call keyword arguments (line 1852)
    # Getting the type of 'axis' (line 1852)
    axis_4688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 38), 'axis', False)
    keyword_4689 = axis_4688
    # Getting the type of 'dtype' (line 1852)
    dtype_4690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 50), 'dtype', False)
    keyword_4691 = dtype_4690
    # Getting the type of 'out' (line 1853)
    out_4692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 34), 'out', False)
    keyword_4693 = out_4692
    # Getting the type of 'keepdims' (line 1853)
    keepdims_4694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 48), 'keepdims', False)
    keyword_4695 = keepdims_4694
    kwargs_4696 = {'dtype': keyword_4691, 'out': keyword_4693, 'keepdims': keyword_4695, 'axis': keyword_4689}
    # Getting the type of 'um' (line 1852)
    um_4684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1852, 11), 'um', False)
    # Obtaining the member 'multiply' of a type (line 1852)
    multiply_4685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1852, 11), um_4684, 'multiply')
    # Obtaining the member 'reduce' of a type (line 1852)
    reduce_4686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1852, 11), multiply_4685, 'reduce')
    # Calling reduce(args, kwargs) (line 1852)
    reduce_call_result_4697 = invoke(stypy.reporting.localization.Localization(__file__, 1852, 11), reduce_4686, *[a_4687], **kwargs_4696)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1852, 4), 'stypy_return_type', reduce_call_result_4697)
    
    # ################# End of 'product(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'product' in the type store
    # Getting the type of 'stypy_return_type' (line 1843)
    stypy_return_type_4698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4698)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'product'
    return stypy_return_type_4698

# Assigning a type to the variable 'product' (line 1843)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1843, 0), 'product', product)

@norecursion
def sometrue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1856)
    None_4699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 21), 'None')
    # Getting the type of 'None' (line 1856)
    None_4700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 31), 'None')
    # Getting the type of 'False' (line 1856)
    False_4701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 46), 'False')
    defaults = [None_4699, None_4700, False_4701]
    # Create a new context for function 'sometrue'
    module_type_store = module_type_store.open_function_context('sometrue', 1856, 0, False)
    
    # Passed parameters checking function
    sometrue.stypy_localization = localization
    sometrue.stypy_type_of_self = None
    sometrue.stypy_type_store = module_type_store
    sometrue.stypy_function_name = 'sometrue'
    sometrue.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    sometrue.stypy_varargs_param_name = None
    sometrue.stypy_kwargs_param_name = None
    sometrue.stypy_call_defaults = defaults
    sometrue.stypy_call_varargs = varargs
    sometrue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sometrue', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sometrue', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sometrue(...)' code ##################

    str_4702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1866, (-1)), 'str', '\n    Check whether some values are true.\n\n    Refer to `any` for full documentation.\n\n    See Also\n    --------\n    any : equivalent function\n\n    ')
    
    # Assigning a Call to a Name (line 1867):
    
    # Call to asanyarray(...): (line 1867)
    # Processing the call arguments (line 1867)
    # Getting the type of 'a' (line 1867)
    a_4704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1867, 21), 'a', False)
    # Processing the call keyword arguments (line 1867)
    kwargs_4705 = {}
    # Getting the type of 'asanyarray' (line 1867)
    asanyarray_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1867, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1867)
    asanyarray_call_result_4706 = invoke(stypy.reporting.localization.Localization(__file__, 1867, 10), asanyarray_4703, *[a_4704], **kwargs_4705)
    
    # Assigning a type to the variable 'arr' (line 1867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1867, 4), 'arr', asanyarray_call_result_4706)
    
    
    # SSA begins for try-except statement (line 1869)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to any(...): (line 1870)
    # Processing the call keyword arguments (line 1870)
    # Getting the type of 'axis' (line 1870)
    axis_4709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1870, 28), 'axis', False)
    keyword_4710 = axis_4709
    # Getting the type of 'out' (line 1870)
    out_4711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1870, 38), 'out', False)
    keyword_4712 = out_4711
    # Getting the type of 'keepdims' (line 1870)
    keepdims_4713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1870, 52), 'keepdims', False)
    keyword_4714 = keepdims_4713
    kwargs_4715 = {'out': keyword_4712, 'keepdims': keyword_4714, 'axis': keyword_4710}
    # Getting the type of 'arr' (line 1870)
    arr_4707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1870, 15), 'arr', False)
    # Obtaining the member 'any' of a type (line 1870)
    any_4708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1870, 15), arr_4707, 'any')
    # Calling any(args, kwargs) (line 1870)
    any_call_result_4716 = invoke(stypy.reporting.localization.Localization(__file__, 1870, 15), any_4708, *[], **kwargs_4715)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1870, 8), 'stypy_return_type', any_call_result_4716)
    # SSA branch for the except part of a try statement (line 1869)
    # SSA branch for the except 'TypeError' branch of a try statement (line 1869)
    module_type_store.open_ssa_branch('except')
    
    # Call to any(...): (line 1872)
    # Processing the call keyword arguments (line 1872)
    # Getting the type of 'axis' (line 1872)
    axis_4719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 28), 'axis', False)
    keyword_4720 = axis_4719
    # Getting the type of 'out' (line 1872)
    out_4721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 38), 'out', False)
    keyword_4722 = out_4721
    kwargs_4723 = {'out': keyword_4722, 'axis': keyword_4720}
    # Getting the type of 'arr' (line 1872)
    arr_4717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1872, 15), 'arr', False)
    # Obtaining the member 'any' of a type (line 1872)
    any_4718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1872, 15), arr_4717, 'any')
    # Calling any(args, kwargs) (line 1872)
    any_call_result_4724 = invoke(stypy.reporting.localization.Localization(__file__, 1872, 15), any_4718, *[], **kwargs_4723)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1872, 8), 'stypy_return_type', any_call_result_4724)
    # SSA join for try-except statement (line 1869)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'sometrue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sometrue' in the type store
    # Getting the type of 'stypy_return_type' (line 1856)
    stypy_return_type_4725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1856, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4725)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sometrue'
    return stypy_return_type_4725

# Assigning a type to the variable 'sometrue' (line 1856)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1856, 0), 'sometrue', sometrue)

@norecursion
def alltrue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1875)
    None_4726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 20), 'None')
    # Getting the type of 'None' (line 1875)
    None_4727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 30), 'None')
    # Getting the type of 'False' (line 1875)
    False_4728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 45), 'False')
    defaults = [None_4726, None_4727, False_4728]
    # Create a new context for function 'alltrue'
    module_type_store = module_type_store.open_function_context('alltrue', 1875, 0, False)
    
    # Passed parameters checking function
    alltrue.stypy_localization = localization
    alltrue.stypy_type_of_self = None
    alltrue.stypy_type_store = module_type_store
    alltrue.stypy_function_name = 'alltrue'
    alltrue.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    alltrue.stypy_varargs_param_name = None
    alltrue.stypy_kwargs_param_name = None
    alltrue.stypy_call_defaults = defaults
    alltrue.stypy_call_varargs = varargs
    alltrue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alltrue', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alltrue', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alltrue(...)' code ##################

    str_4729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1883, (-1)), 'str', '\n    Check if all elements of input array are true.\n\n    See Also\n    --------\n    numpy.all : Equivalent function; see for details.\n\n    ')
    
    # Assigning a Call to a Name (line 1884):
    
    # Call to asanyarray(...): (line 1884)
    # Processing the call arguments (line 1884)
    # Getting the type of 'a' (line 1884)
    a_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1884, 21), 'a', False)
    # Processing the call keyword arguments (line 1884)
    kwargs_4732 = {}
    # Getting the type of 'asanyarray' (line 1884)
    asanyarray_4730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1884, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1884)
    asanyarray_call_result_4733 = invoke(stypy.reporting.localization.Localization(__file__, 1884, 10), asanyarray_4730, *[a_4731], **kwargs_4732)
    
    # Assigning a type to the variable 'arr' (line 1884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1884, 4), 'arr', asanyarray_call_result_4733)
    
    
    # SSA begins for try-except statement (line 1886)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to all(...): (line 1887)
    # Processing the call keyword arguments (line 1887)
    # Getting the type of 'axis' (line 1887)
    axis_4736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 28), 'axis', False)
    keyword_4737 = axis_4736
    # Getting the type of 'out' (line 1887)
    out_4738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 38), 'out', False)
    keyword_4739 = out_4738
    # Getting the type of 'keepdims' (line 1887)
    keepdims_4740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 52), 'keepdims', False)
    keyword_4741 = keepdims_4740
    kwargs_4742 = {'out': keyword_4739, 'keepdims': keyword_4741, 'axis': keyword_4737}
    # Getting the type of 'arr' (line 1887)
    arr_4734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1887, 15), 'arr', False)
    # Obtaining the member 'all' of a type (line 1887)
    all_4735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1887, 15), arr_4734, 'all')
    # Calling all(args, kwargs) (line 1887)
    all_call_result_4743 = invoke(stypy.reporting.localization.Localization(__file__, 1887, 15), all_4735, *[], **kwargs_4742)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1887)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1887, 8), 'stypy_return_type', all_call_result_4743)
    # SSA branch for the except part of a try statement (line 1886)
    # SSA branch for the except 'TypeError' branch of a try statement (line 1886)
    module_type_store.open_ssa_branch('except')
    
    # Call to all(...): (line 1889)
    # Processing the call keyword arguments (line 1889)
    # Getting the type of 'axis' (line 1889)
    axis_4746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1889, 28), 'axis', False)
    keyword_4747 = axis_4746
    # Getting the type of 'out' (line 1889)
    out_4748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1889, 38), 'out', False)
    keyword_4749 = out_4748
    kwargs_4750 = {'out': keyword_4749, 'axis': keyword_4747}
    # Getting the type of 'arr' (line 1889)
    arr_4744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1889, 15), 'arr', False)
    # Obtaining the member 'all' of a type (line 1889)
    all_4745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1889, 15), arr_4744, 'all')
    # Calling all(args, kwargs) (line 1889)
    all_call_result_4751 = invoke(stypy.reporting.localization.Localization(__file__, 1889, 15), all_4745, *[], **kwargs_4750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1889, 8), 'stypy_return_type', all_call_result_4751)
    # SSA join for try-except statement (line 1886)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'alltrue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alltrue' in the type store
    # Getting the type of 'stypy_return_type' (line 1875)
    stypy_return_type_4752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1875, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alltrue'
    return stypy_return_type_4752

# Assigning a type to the variable 'alltrue' (line 1875)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1875, 0), 'alltrue', alltrue)

@norecursion
def any(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1892)
    None_4753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1892, 16), 'None')
    # Getting the type of 'None' (line 1892)
    None_4754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1892, 26), 'None')
    # Getting the type of 'False' (line 1892)
    False_4755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1892, 41), 'False')
    defaults = [None_4753, None_4754, False_4755]
    # Create a new context for function 'any'
    module_type_store = module_type_store.open_function_context('any', 1892, 0, False)
    
    # Passed parameters checking function
    any.stypy_localization = localization
    any.stypy_type_of_self = None
    any.stypy_type_store = module_type_store
    any.stypy_function_name = 'any'
    any.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    any.stypy_varargs_param_name = None
    any.stypy_kwargs_param_name = None
    any.stypy_call_defaults = defaults
    any.stypy_call_varargs = varargs
    any.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'any', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'any', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'any(...)' code ##################

    str_4756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1964, (-1)), 'str', '\n    Test whether any array element along a given axis evaluates to True.\n\n    Returns single boolean unless `axis` is not ``None``\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which a logical OR reduction is performed.\n        The default (`axis` = `None`) is to perform a logical OR over all\n        the dimensions of the input array. `axis` may be negative, in\n        which case it counts from the last to the first axis.\n\n        .. versionadded:: 1.7.0\n\n        If this is a tuple of ints, a reduction is performed on multiple\n        axes, instead of a single axis or all the axes as before.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  It must have\n        the same shape as the expected output and its type is preserved\n        (e.g., if it is of type float, then it will remain so, returning\n        1.0 for True and 0.0 for False, regardless of the type of `a`).\n        See `doc.ufuncs` (Section "Output arguments") for details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    any : bool or ndarray\n        A new boolean or `ndarray` is returned unless `out` is specified,\n        in which case a reference to `out` is returned.\n\n    See Also\n    --------\n    ndarray.any : equivalent method\n\n    all : Test whether all elements along a given axis evaluate to True.\n\n    Notes\n    -----\n    Not a Number (NaN), positive infinity and negative infinity evaluate\n    to `True` because these are not equal to zero.\n\n    Examples\n    --------\n    >>> np.any([[True, False], [True, True]])\n    True\n\n    >>> np.any([[True, False], [False, False]], axis=0)\n    array([ True, False], dtype=bool)\n\n    >>> np.any([-1, 0, 5])\n    True\n\n    >>> np.any(np.nan)\n    True\n\n    >>> o=np.array([False])\n    >>> z=np.any([-1, 4, 5], out=o)\n    >>> z, o\n    (array([ True], dtype=bool), array([ True], dtype=bool))\n    >>> # Check now that z is a reference to o\n    >>> z is o\n    True\n    >>> id(z), id(o) # identity of z and o              # doctest: +SKIP\n    (191614240, 191614240)\n\n    ')
    
    # Assigning a Call to a Name (line 1965):
    
    # Call to asanyarray(...): (line 1965)
    # Processing the call arguments (line 1965)
    # Getting the type of 'a' (line 1965)
    a_4758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 21), 'a', False)
    # Processing the call keyword arguments (line 1965)
    kwargs_4759 = {}
    # Getting the type of 'asanyarray' (line 1965)
    asanyarray_4757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1965, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 1965)
    asanyarray_call_result_4760 = invoke(stypy.reporting.localization.Localization(__file__, 1965, 10), asanyarray_4757, *[a_4758], **kwargs_4759)
    
    # Assigning a type to the variable 'arr' (line 1965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1965, 4), 'arr', asanyarray_call_result_4760)
    
    
    # SSA begins for try-except statement (line 1967)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to any(...): (line 1968)
    # Processing the call keyword arguments (line 1968)
    # Getting the type of 'axis' (line 1968)
    axis_4763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 28), 'axis', False)
    keyword_4764 = axis_4763
    # Getting the type of 'out' (line 1968)
    out_4765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 38), 'out', False)
    keyword_4766 = out_4765
    # Getting the type of 'keepdims' (line 1968)
    keepdims_4767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 52), 'keepdims', False)
    keyword_4768 = keepdims_4767
    kwargs_4769 = {'out': keyword_4766, 'keepdims': keyword_4768, 'axis': keyword_4764}
    # Getting the type of 'arr' (line 1968)
    arr_4761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1968, 15), 'arr', False)
    # Obtaining the member 'any' of a type (line 1968)
    any_4762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1968, 15), arr_4761, 'any')
    # Calling any(args, kwargs) (line 1968)
    any_call_result_4770 = invoke(stypy.reporting.localization.Localization(__file__, 1968, 15), any_4762, *[], **kwargs_4769)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1968)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1968, 8), 'stypy_return_type', any_call_result_4770)
    # SSA branch for the except part of a try statement (line 1967)
    # SSA branch for the except 'TypeError' branch of a try statement (line 1967)
    module_type_store.open_ssa_branch('except')
    
    # Call to any(...): (line 1970)
    # Processing the call keyword arguments (line 1970)
    # Getting the type of 'axis' (line 1970)
    axis_4773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 28), 'axis', False)
    keyword_4774 = axis_4773
    # Getting the type of 'out' (line 1970)
    out_4775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 38), 'out', False)
    keyword_4776 = out_4775
    kwargs_4777 = {'out': keyword_4776, 'axis': keyword_4774}
    # Getting the type of 'arr' (line 1970)
    arr_4771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1970, 15), 'arr', False)
    # Obtaining the member 'any' of a type (line 1970)
    any_4772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1970, 15), arr_4771, 'any')
    # Calling any(args, kwargs) (line 1970)
    any_call_result_4778 = invoke(stypy.reporting.localization.Localization(__file__, 1970, 15), any_4772, *[], **kwargs_4777)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1970, 8), 'stypy_return_type', any_call_result_4778)
    # SSA join for try-except statement (line 1967)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'any(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'any' in the type store
    # Getting the type of 'stypy_return_type' (line 1892)
    stypy_return_type_4779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1892, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'any'
    return stypy_return_type_4779

# Assigning a type to the variable 'any' (line 1892)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1892, 0), 'any', any)

@norecursion
def all(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1973)
    None_4780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1973, 16), 'None')
    # Getting the type of 'None' (line 1973)
    None_4781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1973, 26), 'None')
    # Getting the type of 'False' (line 1973)
    False_4782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1973, 41), 'False')
    defaults = [None_4780, None_4781, False_4782]
    # Create a new context for function 'all'
    module_type_store = module_type_store.open_function_context('all', 1973, 0, False)
    
    # Passed parameters checking function
    all.stypy_localization = localization
    all.stypy_type_of_self = None
    all.stypy_type_store = module_type_store
    all.stypy_function_name = 'all'
    all.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    all.stypy_varargs_param_name = None
    all.stypy_kwargs_param_name = None
    all.stypy_call_defaults = defaults
    all.stypy_call_varargs = varargs
    all.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'all', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'all', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'all(...)' code ##################

    str_4783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2038, (-1)), 'str', '\n    Test whether all array elements along a given axis evaluate to True.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which a logical AND reduction is performed.\n        The default (`axis` = `None`) is to perform a logical AND over all\n        the dimensions of the input array. `axis` may be negative, in\n        which case it counts from the last to the first axis.\n\n        .. versionadded:: 1.7.0\n\n        If this is a tuple of ints, a reduction is performed on multiple\n        axes, instead of a single axis or all the axes as before.\n    out : ndarray, optional\n        Alternate output array in which to place the result.\n        It must have the same shape as the expected output and its\n        type is preserved (e.g., if ``dtype(out)`` is float, the result\n        will consist of 0.0\'s and 1.0\'s).  See `doc.ufuncs` (Section\n        "Output arguments") for more details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    all : ndarray, bool\n        A new boolean or array is returned unless `out` is specified,\n        in which case a reference to `out` is returned.\n\n    See Also\n    --------\n    ndarray.all : equivalent method\n\n    any : Test whether any element along a given axis evaluates to True.\n\n    Notes\n    -----\n    Not a Number (NaN), positive infinity and negative infinity\n    evaluate to `True` because these are not equal to zero.\n\n    Examples\n    --------\n    >>> np.all([[True,False],[True,True]])\n    False\n\n    >>> np.all([[True,False],[True,True]], axis=0)\n    array([ True, False], dtype=bool)\n\n    >>> np.all([-1, 4, 5])\n    True\n\n    >>> np.all([1.0, np.nan])\n    True\n\n    >>> o=np.array([False])\n    >>> z=np.all([-1, 4, 5], out=o)\n    >>> id(z), id(o), z                             # doctest: +SKIP\n    (28293632, 28293632, array([ True], dtype=bool))\n\n    ')
    
    # Assigning a Call to a Name (line 2039):
    
    # Call to asanyarray(...): (line 2039)
    # Processing the call arguments (line 2039)
    # Getting the type of 'a' (line 2039)
    a_4785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2039, 21), 'a', False)
    # Processing the call keyword arguments (line 2039)
    kwargs_4786 = {}
    # Getting the type of 'asanyarray' (line 2039)
    asanyarray_4784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2039, 10), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 2039)
    asanyarray_call_result_4787 = invoke(stypy.reporting.localization.Localization(__file__, 2039, 10), asanyarray_4784, *[a_4785], **kwargs_4786)
    
    # Assigning a type to the variable 'arr' (line 2039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2039, 4), 'arr', asanyarray_call_result_4787)
    
    
    # SSA begins for try-except statement (line 2041)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to all(...): (line 2042)
    # Processing the call keyword arguments (line 2042)
    # Getting the type of 'axis' (line 2042)
    axis_4790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2042, 28), 'axis', False)
    keyword_4791 = axis_4790
    # Getting the type of 'out' (line 2042)
    out_4792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2042, 38), 'out', False)
    keyword_4793 = out_4792
    # Getting the type of 'keepdims' (line 2042)
    keepdims_4794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2042, 52), 'keepdims', False)
    keyword_4795 = keepdims_4794
    kwargs_4796 = {'out': keyword_4793, 'keepdims': keyword_4795, 'axis': keyword_4791}
    # Getting the type of 'arr' (line 2042)
    arr_4788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2042, 15), 'arr', False)
    # Obtaining the member 'all' of a type (line 2042)
    all_4789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2042, 15), arr_4788, 'all')
    # Calling all(args, kwargs) (line 2042)
    all_call_result_4797 = invoke(stypy.reporting.localization.Localization(__file__, 2042, 15), all_4789, *[], **kwargs_4796)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2042, 8), 'stypy_return_type', all_call_result_4797)
    # SSA branch for the except part of a try statement (line 2041)
    # SSA branch for the except 'TypeError' branch of a try statement (line 2041)
    module_type_store.open_ssa_branch('except')
    
    # Call to all(...): (line 2044)
    # Processing the call keyword arguments (line 2044)
    # Getting the type of 'axis' (line 2044)
    axis_4800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2044, 28), 'axis', False)
    keyword_4801 = axis_4800
    # Getting the type of 'out' (line 2044)
    out_4802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2044, 38), 'out', False)
    keyword_4803 = out_4802
    kwargs_4804 = {'out': keyword_4803, 'axis': keyword_4801}
    # Getting the type of 'arr' (line 2044)
    arr_4798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2044, 15), 'arr', False)
    # Obtaining the member 'all' of a type (line 2044)
    all_4799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2044, 15), arr_4798, 'all')
    # Calling all(args, kwargs) (line 2044)
    all_call_result_4805 = invoke(stypy.reporting.localization.Localization(__file__, 2044, 15), all_4799, *[], **kwargs_4804)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2044)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2044, 8), 'stypy_return_type', all_call_result_4805)
    # SSA join for try-except statement (line 2041)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'all(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'all' in the type store
    # Getting the type of 'stypy_return_type' (line 1973)
    stypy_return_type_4806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1973, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4806)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'all'
    return stypy_return_type_4806

# Assigning a type to the variable 'all' (line 1973)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1973, 0), 'all', all)

@norecursion
def cumsum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2047)
    None_4807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2047, 19), 'None')
    # Getting the type of 'None' (line 2047)
    None_4808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2047, 31), 'None')
    # Getting the type of 'None' (line 2047)
    None_4809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2047, 41), 'None')
    defaults = [None_4807, None_4808, None_4809]
    # Create a new context for function 'cumsum'
    module_type_store = module_type_store.open_function_context('cumsum', 2047, 0, False)
    
    # Passed parameters checking function
    cumsum.stypy_localization = localization
    cumsum.stypy_type_of_self = None
    cumsum.stypy_type_store = module_type_store
    cumsum.stypy_function_name = 'cumsum'
    cumsum.stypy_param_names_list = ['a', 'axis', 'dtype', 'out']
    cumsum.stypy_varargs_param_name = None
    cumsum.stypy_kwargs_param_name = None
    cumsum.stypy_call_defaults = defaults
    cumsum.stypy_call_varargs = varargs
    cumsum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cumsum', ['a', 'axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cumsum', localization, ['a', 'axis', 'dtype', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cumsum(...)' code ##################

    str_4810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2110, (-1)), 'str', '\n    Return the cumulative sum of the elements along a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int, optional\n        Axis along which the cumulative sum is computed. The default\n        (None) is to compute the cumsum over the flattened array.\n    dtype : dtype, optional\n        Type of the returned array and of the accumulator in which the\n        elements are summed.  If `dtype` is not specified, it defaults\n        to the dtype of `a`, unless `a` has an integer dtype with a\n        precision less than that of the default platform integer.  In\n        that case, the default platform integer is used.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output\n        but the type will be cast if necessary. See `doc.ufuncs`\n        (Section "Output arguments") for more details.\n\n    Returns\n    -------\n    cumsum_along_axis : ndarray.\n        A new array holding the result is returned unless `out` is\n        specified, in which case a reference to `out` is returned. The\n        result has the same size as `a`, and the same shape as `a` if\n        `axis` is not None or `a` is a 1-d array.\n\n\n    See Also\n    --------\n    sum : Sum array elements.\n\n    trapz : Integration of array values using the composite trapezoidal rule.\n\n    diff :  Calculate the n-th discrete difference along given axis.\n\n    Notes\n    -----\n    Arithmetic is modular when using integer types, and no error is\n    raised on overflow.\n\n    Examples\n    --------\n    >>> a = np.array([[1,2,3], [4,5,6]])\n    >>> a\n    array([[1, 2, 3],\n           [4, 5, 6]])\n    >>> np.cumsum(a)\n    array([ 1,  3,  6, 10, 15, 21])\n    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)\n    array([  1.,   3.,   6.,  10.,  15.,  21.])\n\n    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns\n    array([[1, 2, 3],\n           [5, 7, 9]])\n    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows\n    array([[ 1,  3,  6],\n           [ 4,  9, 15]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 2111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2112):
    # Getting the type of 'a' (line 2112)
    a_4811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2112, 17), 'a')
    # Obtaining the member 'cumsum' of a type (line 2112)
    cumsum_4812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2112, 17), a_4811, 'cumsum')
    # Assigning a type to the variable 'cumsum' (line 2112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2112, 8), 'cumsum', cumsum_4812)
    # SSA branch for the except part of a try statement (line 2111)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2111)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2114)
    # Processing the call arguments (line 2114)
    # Getting the type of 'a' (line 2114)
    a_4814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 23), 'a', False)
    str_4815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2114, 26), 'str', 'cumsum')
    # Getting the type of 'axis' (line 2114)
    axis_4816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 36), 'axis', False)
    # Getting the type of 'dtype' (line 2114)
    dtype_4817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 42), 'dtype', False)
    # Getting the type of 'out' (line 2114)
    out_4818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 49), 'out', False)
    # Processing the call keyword arguments (line 2114)
    kwargs_4819 = {}
    # Getting the type of '_wrapit' (line 2114)
    _wrapit_4813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2114, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2114)
    _wrapit_call_result_4820 = invoke(stypy.reporting.localization.Localization(__file__, 2114, 15), _wrapit_4813, *[a_4814, str_4815, axis_4816, dtype_4817, out_4818], **kwargs_4819)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2114, 8), 'stypy_return_type', _wrapit_call_result_4820)
    # SSA join for try-except statement (line 2111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cumsum(...): (line 2115)
    # Processing the call arguments (line 2115)
    # Getting the type of 'axis' (line 2115)
    axis_4822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2115, 18), 'axis', False)
    # Getting the type of 'dtype' (line 2115)
    dtype_4823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2115, 24), 'dtype', False)
    # Getting the type of 'out' (line 2115)
    out_4824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2115, 31), 'out', False)
    # Processing the call keyword arguments (line 2115)
    kwargs_4825 = {}
    # Getting the type of 'cumsum' (line 2115)
    cumsum_4821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2115, 11), 'cumsum', False)
    # Calling cumsum(args, kwargs) (line 2115)
    cumsum_call_result_4826 = invoke(stypy.reporting.localization.Localization(__file__, 2115, 11), cumsum_4821, *[axis_4822, dtype_4823, out_4824], **kwargs_4825)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2115, 4), 'stypy_return_type', cumsum_call_result_4826)
    
    # ################# End of 'cumsum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cumsum' in the type store
    # Getting the type of 'stypy_return_type' (line 2047)
    stypy_return_type_4827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2047, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4827)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cumsum'
    return stypy_return_type_4827

# Assigning a type to the variable 'cumsum' (line 2047)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2047, 0), 'cumsum', cumsum)

@norecursion
def cumproduct(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2118)
    None_4828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2118, 23), 'None')
    # Getting the type of 'None' (line 2118)
    None_4829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2118, 35), 'None')
    # Getting the type of 'None' (line 2118)
    None_4830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2118, 45), 'None')
    defaults = [None_4828, None_4829, None_4830]
    # Create a new context for function 'cumproduct'
    module_type_store = module_type_store.open_function_context('cumproduct', 2118, 0, False)
    
    # Passed parameters checking function
    cumproduct.stypy_localization = localization
    cumproduct.stypy_type_of_self = None
    cumproduct.stypy_type_store = module_type_store
    cumproduct.stypy_function_name = 'cumproduct'
    cumproduct.stypy_param_names_list = ['a', 'axis', 'dtype', 'out']
    cumproduct.stypy_varargs_param_name = None
    cumproduct.stypy_kwargs_param_name = None
    cumproduct.stypy_call_defaults = defaults
    cumproduct.stypy_call_varargs = varargs
    cumproduct.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cumproduct', ['a', 'axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cumproduct', localization, ['a', 'axis', 'dtype', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cumproduct(...)' code ##################

    str_4831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2127, (-1)), 'str', '\n    Return the cumulative product over the given axis.\n\n\n    See Also\n    --------\n    cumprod : equivalent function; see for details.\n\n    ')
    
    
    # SSA begins for try-except statement (line 2128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2129):
    # Getting the type of 'a' (line 2129)
    a_4832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2129, 18), 'a')
    # Obtaining the member 'cumprod' of a type (line 2129)
    cumprod_4833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2129, 18), a_4832, 'cumprod')
    # Assigning a type to the variable 'cumprod' (line 2129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2129, 8), 'cumprod', cumprod_4833)
    # SSA branch for the except part of a try statement (line 2128)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2128)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2131)
    # Processing the call arguments (line 2131)
    # Getting the type of 'a' (line 2131)
    a_4835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 23), 'a', False)
    str_4836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2131, 26), 'str', 'cumprod')
    # Getting the type of 'axis' (line 2131)
    axis_4837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 37), 'axis', False)
    # Getting the type of 'dtype' (line 2131)
    dtype_4838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 43), 'dtype', False)
    # Getting the type of 'out' (line 2131)
    out_4839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 50), 'out', False)
    # Processing the call keyword arguments (line 2131)
    kwargs_4840 = {}
    # Getting the type of '_wrapit' (line 2131)
    _wrapit_4834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2131, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2131)
    _wrapit_call_result_4841 = invoke(stypy.reporting.localization.Localization(__file__, 2131, 15), _wrapit_4834, *[a_4835, str_4836, axis_4837, dtype_4838, out_4839], **kwargs_4840)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2131, 8), 'stypy_return_type', _wrapit_call_result_4841)
    # SSA join for try-except statement (line 2128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cumprod(...): (line 2132)
    # Processing the call arguments (line 2132)
    # Getting the type of 'axis' (line 2132)
    axis_4843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2132, 19), 'axis', False)
    # Getting the type of 'dtype' (line 2132)
    dtype_4844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2132, 25), 'dtype', False)
    # Getting the type of 'out' (line 2132)
    out_4845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2132, 32), 'out', False)
    # Processing the call keyword arguments (line 2132)
    kwargs_4846 = {}
    # Getting the type of 'cumprod' (line 2132)
    cumprod_4842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2132, 11), 'cumprod', False)
    # Calling cumprod(args, kwargs) (line 2132)
    cumprod_call_result_4847 = invoke(stypy.reporting.localization.Localization(__file__, 2132, 11), cumprod_4842, *[axis_4843, dtype_4844, out_4845], **kwargs_4846)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2132, 4), 'stypy_return_type', cumprod_call_result_4847)
    
    # ################# End of 'cumproduct(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cumproduct' in the type store
    # Getting the type of 'stypy_return_type' (line 2118)
    stypy_return_type_4848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cumproduct'
    return stypy_return_type_4848

# Assigning a type to the variable 'cumproduct' (line 2118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2118, 0), 'cumproduct', cumproduct)

@norecursion
def ptp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2135)
    None_4849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2135, 16), 'None')
    # Getting the type of 'None' (line 2135)
    None_4850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2135, 26), 'None')
    defaults = [None_4849, None_4850]
    # Create a new context for function 'ptp'
    module_type_store = module_type_store.open_function_context('ptp', 2135, 0, False)
    
    # Passed parameters checking function
    ptp.stypy_localization = localization
    ptp.stypy_type_of_self = None
    ptp.stypy_type_store = module_type_store
    ptp.stypy_function_name = 'ptp'
    ptp.stypy_param_names_list = ['a', 'axis', 'out']
    ptp.stypy_varargs_param_name = None
    ptp.stypy_kwargs_param_name = None
    ptp.stypy_call_defaults = defaults
    ptp.stypy_call_varargs = varargs
    ptp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ptp', ['a', 'axis', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ptp', localization, ['a', 'axis', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ptp(...)' code ##################

    str_4851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2172, (-1)), 'str', "\n    Range of values (maximum - minimum) along an axis.\n\n    The name of the function comes from the acronym for 'peak to peak'.\n\n    Parameters\n    ----------\n    a : array_like\n        Input values.\n    axis : int, optional\n        Axis along which to find the peaks.  By default, flatten the\n        array.\n    out : array_like\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output,\n        but the type of the output values will be cast if necessary.\n\n    Returns\n    -------\n    ptp : ndarray\n        A new array holding the result, unless `out` was\n        specified, in which case a reference to `out` is returned.\n\n    Examples\n    --------\n    >>> x = np.arange(4).reshape((2,2))\n    >>> x\n    array([[0, 1],\n           [2, 3]])\n\n    >>> np.ptp(x, axis=0)\n    array([2, 2])\n\n    >>> np.ptp(x, axis=1)\n    array([1, 1])\n\n    ")
    
    
    # SSA begins for try-except statement (line 2173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2174):
    # Getting the type of 'a' (line 2174)
    a_4852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2174, 14), 'a')
    # Obtaining the member 'ptp' of a type (line 2174)
    ptp_4853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2174, 14), a_4852, 'ptp')
    # Assigning a type to the variable 'ptp' (line 2174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2174, 8), 'ptp', ptp_4853)
    # SSA branch for the except part of a try statement (line 2173)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2173)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2176)
    # Processing the call arguments (line 2176)
    # Getting the type of 'a' (line 2176)
    a_4855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2176, 23), 'a', False)
    str_4856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2176, 26), 'str', 'ptp')
    # Getting the type of 'axis' (line 2176)
    axis_4857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2176, 33), 'axis', False)
    # Getting the type of 'out' (line 2176)
    out_4858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2176, 39), 'out', False)
    # Processing the call keyword arguments (line 2176)
    kwargs_4859 = {}
    # Getting the type of '_wrapit' (line 2176)
    _wrapit_4854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2176, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2176)
    _wrapit_call_result_4860 = invoke(stypy.reporting.localization.Localization(__file__, 2176, 15), _wrapit_4854, *[a_4855, str_4856, axis_4857, out_4858], **kwargs_4859)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2176, 8), 'stypy_return_type', _wrapit_call_result_4860)
    # SSA join for try-except statement (line 2173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ptp(...): (line 2177)
    # Processing the call arguments (line 2177)
    # Getting the type of 'axis' (line 2177)
    axis_4862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2177, 15), 'axis', False)
    # Getting the type of 'out' (line 2177)
    out_4863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2177, 21), 'out', False)
    # Processing the call keyword arguments (line 2177)
    kwargs_4864 = {}
    # Getting the type of 'ptp' (line 2177)
    ptp_4861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2177, 11), 'ptp', False)
    # Calling ptp(args, kwargs) (line 2177)
    ptp_call_result_4865 = invoke(stypy.reporting.localization.Localization(__file__, 2177, 11), ptp_4861, *[axis_4862, out_4863], **kwargs_4864)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2177, 4), 'stypy_return_type', ptp_call_result_4865)
    
    # ################# End of 'ptp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ptp' in the type store
    # Getting the type of 'stypy_return_type' (line 2135)
    stypy_return_type_4866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4866)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ptp'
    return stypy_return_type_4866

# Assigning a type to the variable 'ptp' (line 2135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2135, 0), 'ptp', ptp)

@norecursion
def amax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2180)
    None_4867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2180, 17), 'None')
    # Getting the type of 'None' (line 2180)
    None_4868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2180, 27), 'None')
    # Getting the type of 'False' (line 2180)
    False_4869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2180, 42), 'False')
    defaults = [None_4867, None_4868, False_4869]
    # Create a new context for function 'amax'
    module_type_store = module_type_store.open_function_context('amax', 2180, 0, False)
    
    # Passed parameters checking function
    amax.stypy_localization = localization
    amax.stypy_type_of_self = None
    amax.stypy_type_store = module_type_store
    amax.stypy_function_name = 'amax'
    amax.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    amax.stypy_varargs_param_name = None
    amax.stypy_kwargs_param_name = None
    amax.stypy_call_defaults = defaults
    amax.stypy_call_varargs = varargs
    amax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'amax', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'amax', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'amax(...)' code ##################

    str_4870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2257, (-1)), 'str', '\n    Return the maximum of an array or maximum along an axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which to operate.  By default, flattened input is\n        used.\n\n        .. versionadded: 1.7.0\n\n        If this is a tuple of ints, the maximum is selected over multiple axes,\n        instead of a single axis or all the axes as before.\n    out : ndarray, optional\n        Alternative output array in which to place the result.  Must\n        be of the same shape and buffer length as the expected output.\n        See `doc.ufuncs` (Section "Output arguments") for more details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    amax : ndarray or scalar\n        Maximum of `a`. If `axis` is None, the result is a scalar value.\n        If `axis` is given, the result is an array of dimension\n        ``a.ndim - 1``.\n\n    See Also\n    --------\n    amin :\n        The minimum value of an array along a given axis, propagating any NaNs.\n    nanmax :\n        The maximum value of an array along a given axis, ignoring any NaNs.\n    maximum :\n        Element-wise maximum of two arrays, propagating any NaNs.\n    fmax :\n        Element-wise maximum of two arrays, ignoring any NaNs.\n    argmax :\n        Return the indices of the maximum values.\n\n    nanmin, minimum, fmin\n\n    Notes\n    -----\n    NaN values are propagated, that is if at least one item is NaN, the\n    corresponding max value will be NaN as well. To ignore NaN values\n    (MATLAB behavior), please use nanmax.\n\n    Don\'t use `amax` for element-wise comparison of 2 arrays; when\n    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than\n    ``amax(a, axis=0)``.\n\n    Examples\n    --------\n    >>> a = np.arange(4).reshape((2,2))\n    >>> a\n    array([[0, 1],\n           [2, 3]])\n    >>> np.amax(a)           # Maximum of the flattened array\n    3\n    >>> np.amax(a, axis=0)   # Maxima along the first axis\n    array([2, 3])\n    >>> np.amax(a, axis=1)   # Maxima along the second axis\n    array([1, 3])\n\n    >>> b = np.arange(5, dtype=np.float)\n    >>> b[2] = np.NaN\n    >>> np.amax(b)\n    nan\n    >>> np.nanmax(b)\n    4.0\n\n    ')
    
    
    
    # Call to type(...): (line 2258)
    # Processing the call arguments (line 2258)
    # Getting the type of 'a' (line 2258)
    a_4872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2258, 12), 'a', False)
    # Processing the call keyword arguments (line 2258)
    kwargs_4873 = {}
    # Getting the type of 'type' (line 2258)
    type_4871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2258, 7), 'type', False)
    # Calling type(args, kwargs) (line 2258)
    type_call_result_4874 = invoke(stypy.reporting.localization.Localization(__file__, 2258, 7), type_4871, *[a_4872], **kwargs_4873)
    
    # Getting the type of 'mu' (line 2258)
    mu_4875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2258, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 2258)
    ndarray_4876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2258, 22), mu_4875, 'ndarray')
    # Applying the binary operator 'isnot' (line 2258)
    result_is_not_4877 = python_operator(stypy.reporting.localization.Localization(__file__, 2258, 7), 'isnot', type_call_result_4874, ndarray_4876)
    
    # Testing the type of an if condition (line 2258)
    if_condition_4878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2258, 4), result_is_not_4877)
    # Assigning a type to the variable 'if_condition_4878' (line 2258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2258, 4), 'if_condition_4878', if_condition_4878)
    # SSA begins for if statement (line 2258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 2259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2260):
    # Getting the type of 'a' (line 2260)
    a_4879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2260, 19), 'a')
    # Obtaining the member 'max' of a type (line 2260)
    max_4880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2260, 19), a_4879, 'max')
    # Assigning a type to the variable 'amax' (line 2260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2260, 12), 'amax', max_4880)
    # SSA branch for the except part of a try statement (line 2259)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2259)
    module_type_store.open_ssa_branch('except')
    
    # Call to _amax(...): (line 2262)
    # Processing the call arguments (line 2262)
    # Getting the type of 'a' (line 2262)
    a_4883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2262, 34), 'a', False)
    # Processing the call keyword arguments (line 2262)
    # Getting the type of 'axis' (line 2262)
    axis_4884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2262, 42), 'axis', False)
    keyword_4885 = axis_4884
    # Getting the type of 'out' (line 2263)
    out_4886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2263, 38), 'out', False)
    keyword_4887 = out_4886
    # Getting the type of 'keepdims' (line 2263)
    keepdims_4888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2263, 52), 'keepdims', False)
    keyword_4889 = keepdims_4888
    kwargs_4890 = {'out': keyword_4887, 'keepdims': keyword_4889, 'axis': keyword_4885}
    # Getting the type of '_methods' (line 2262)
    _methods_4881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2262, 19), '_methods', False)
    # Obtaining the member '_amax' of a type (line 2262)
    _amax_4882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2262, 19), _methods_4881, '_amax')
    # Calling _amax(args, kwargs) (line 2262)
    _amax_call_result_4891 = invoke(stypy.reporting.localization.Localization(__file__, 2262, 19), _amax_4882, *[a_4883], **kwargs_4890)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2262, 12), 'stypy_return_type', _amax_call_result_4891)
    # SSA join for try-except statement (line 2259)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to amax(...): (line 2265)
    # Processing the call keyword arguments (line 2265)
    # Getting the type of 'axis' (line 2265)
    axis_4893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2265, 25), 'axis', False)
    keyword_4894 = axis_4893
    # Getting the type of 'out' (line 2265)
    out_4895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2265, 35), 'out', False)
    keyword_4896 = out_4895
    kwargs_4897 = {'out': keyword_4896, 'axis': keyword_4894}
    # Getting the type of 'amax' (line 2265)
    amax_4892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2265, 15), 'amax', False)
    # Calling amax(args, kwargs) (line 2265)
    amax_call_result_4898 = invoke(stypy.reporting.localization.Localization(__file__, 2265, 15), amax_4892, *[], **kwargs_4897)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2265, 8), 'stypy_return_type', amax_call_result_4898)
    # SSA branch for the else part of an if statement (line 2258)
    module_type_store.open_ssa_branch('else')
    
    # Call to _amax(...): (line 2267)
    # Processing the call arguments (line 2267)
    # Getting the type of 'a' (line 2267)
    a_4901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2267, 30), 'a', False)
    # Processing the call keyword arguments (line 2267)
    # Getting the type of 'axis' (line 2267)
    axis_4902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2267, 38), 'axis', False)
    keyword_4903 = axis_4902
    # Getting the type of 'out' (line 2268)
    out_4904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2268, 34), 'out', False)
    keyword_4905 = out_4904
    # Getting the type of 'keepdims' (line 2268)
    keepdims_4906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2268, 48), 'keepdims', False)
    keyword_4907 = keepdims_4906
    kwargs_4908 = {'out': keyword_4905, 'keepdims': keyword_4907, 'axis': keyword_4903}
    # Getting the type of '_methods' (line 2267)
    _methods_4899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2267, 15), '_methods', False)
    # Obtaining the member '_amax' of a type (line 2267)
    _amax_4900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2267, 15), _methods_4899, '_amax')
    # Calling _amax(args, kwargs) (line 2267)
    _amax_call_result_4909 = invoke(stypy.reporting.localization.Localization(__file__, 2267, 15), _amax_4900, *[a_4901], **kwargs_4908)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2267, 8), 'stypy_return_type', _amax_call_result_4909)
    # SSA join for if statement (line 2258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'amax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'amax' in the type store
    # Getting the type of 'stypy_return_type' (line 2180)
    stypy_return_type_4910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4910)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'amax'
    return stypy_return_type_4910

# Assigning a type to the variable 'amax' (line 2180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2180, 0), 'amax', amax)

@norecursion
def amin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2271)
    None_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 17), 'None')
    # Getting the type of 'None' (line 2271)
    None_4912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 27), 'None')
    # Getting the type of 'False' (line 2271)
    False_4913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 42), 'False')
    defaults = [None_4911, None_4912, False_4913]
    # Create a new context for function 'amin'
    module_type_store = module_type_store.open_function_context('amin', 2271, 0, False)
    
    # Passed parameters checking function
    amin.stypy_localization = localization
    amin.stypy_type_of_self = None
    amin.stypy_type_store = module_type_store
    amin.stypy_function_name = 'amin'
    amin.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    amin.stypy_varargs_param_name = None
    amin.stypy_kwargs_param_name = None
    amin.stypy_call_defaults = defaults
    amin.stypy_call_varargs = varargs
    amin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'amin', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'amin', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'amin(...)' code ##################

    str_4914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2348, (-1)), 'str', '\n    Return the minimum of an array or minimum along an axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which to operate.  By default, flattened input is\n        used.\n\n        .. versionadded: 1.7.0\n\n        If this is a tuple of ints, the minimum is selected over multiple axes,\n        instead of a single axis or all the axes as before.\n    out : ndarray, optional\n        Alternative output array in which to place the result.  Must\n        be of the same shape and buffer length as the expected output.\n        See `doc.ufuncs` (Section "Output arguments") for more details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    amin : ndarray or scalar\n        Minimum of `a`. If `axis` is None, the result is a scalar value.\n        If `axis` is given, the result is an array of dimension\n        ``a.ndim - 1``.\n\n    See Also\n    --------\n    amax :\n        The maximum value of an array along a given axis, propagating any NaNs.\n    nanmin :\n        The minimum value of an array along a given axis, ignoring any NaNs.\n    minimum :\n        Element-wise minimum of two arrays, propagating any NaNs.\n    fmin :\n        Element-wise minimum of two arrays, ignoring any NaNs.\n    argmin :\n        Return the indices of the minimum values.\n\n    nanmax, maximum, fmax\n\n    Notes\n    -----\n    NaN values are propagated, that is if at least one item is NaN, the\n    corresponding min value will be NaN as well. To ignore NaN values\n    (MATLAB behavior), please use nanmin.\n\n    Don\'t use `amin` for element-wise comparison of 2 arrays; when\n    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than\n    ``amin(a, axis=0)``.\n\n    Examples\n    --------\n    >>> a = np.arange(4).reshape((2,2))\n    >>> a\n    array([[0, 1],\n           [2, 3]])\n    >>> np.amin(a)           # Minimum of the flattened array\n    0\n    >>> np.amin(a, axis=0)   # Minima along the first axis\n    array([0, 1])\n    >>> np.amin(a, axis=1)   # Minima along the second axis\n    array([0, 2])\n\n    >>> b = np.arange(5, dtype=np.float)\n    >>> b[2] = np.NaN\n    >>> np.amin(b)\n    nan\n    >>> np.nanmin(b)\n    0.0\n\n    ')
    
    
    
    # Call to type(...): (line 2349)
    # Processing the call arguments (line 2349)
    # Getting the type of 'a' (line 2349)
    a_4916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2349, 12), 'a', False)
    # Processing the call keyword arguments (line 2349)
    kwargs_4917 = {}
    # Getting the type of 'type' (line 2349)
    type_4915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2349, 7), 'type', False)
    # Calling type(args, kwargs) (line 2349)
    type_call_result_4918 = invoke(stypy.reporting.localization.Localization(__file__, 2349, 7), type_4915, *[a_4916], **kwargs_4917)
    
    # Getting the type of 'mu' (line 2349)
    mu_4919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2349, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 2349)
    ndarray_4920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2349, 22), mu_4919, 'ndarray')
    # Applying the binary operator 'isnot' (line 2349)
    result_is_not_4921 = python_operator(stypy.reporting.localization.Localization(__file__, 2349, 7), 'isnot', type_call_result_4918, ndarray_4920)
    
    # Testing the type of an if condition (line 2349)
    if_condition_4922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2349, 4), result_is_not_4921)
    # Assigning a type to the variable 'if_condition_4922' (line 2349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2349, 4), 'if_condition_4922', if_condition_4922)
    # SSA begins for if statement (line 2349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 2350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2351):
    # Getting the type of 'a' (line 2351)
    a_4923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2351, 19), 'a')
    # Obtaining the member 'min' of a type (line 2351)
    min_4924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2351, 19), a_4923, 'min')
    # Assigning a type to the variable 'amin' (line 2351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2351, 12), 'amin', min_4924)
    # SSA branch for the except part of a try statement (line 2350)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2350)
    module_type_store.open_ssa_branch('except')
    
    # Call to _amin(...): (line 2353)
    # Processing the call arguments (line 2353)
    # Getting the type of 'a' (line 2353)
    a_4927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2353, 34), 'a', False)
    # Processing the call keyword arguments (line 2353)
    # Getting the type of 'axis' (line 2353)
    axis_4928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2353, 42), 'axis', False)
    keyword_4929 = axis_4928
    # Getting the type of 'out' (line 2354)
    out_4930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2354, 38), 'out', False)
    keyword_4931 = out_4930
    # Getting the type of 'keepdims' (line 2354)
    keepdims_4932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2354, 52), 'keepdims', False)
    keyword_4933 = keepdims_4932
    kwargs_4934 = {'out': keyword_4931, 'keepdims': keyword_4933, 'axis': keyword_4929}
    # Getting the type of '_methods' (line 2353)
    _methods_4925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2353, 19), '_methods', False)
    # Obtaining the member '_amin' of a type (line 2353)
    _amin_4926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2353, 19), _methods_4925, '_amin')
    # Calling _amin(args, kwargs) (line 2353)
    _amin_call_result_4935 = invoke(stypy.reporting.localization.Localization(__file__, 2353, 19), _amin_4926, *[a_4927], **kwargs_4934)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2353, 12), 'stypy_return_type', _amin_call_result_4935)
    # SSA join for try-except statement (line 2350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to amin(...): (line 2356)
    # Processing the call keyword arguments (line 2356)
    # Getting the type of 'axis' (line 2356)
    axis_4937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2356, 25), 'axis', False)
    keyword_4938 = axis_4937
    # Getting the type of 'out' (line 2356)
    out_4939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2356, 35), 'out', False)
    keyword_4940 = out_4939
    kwargs_4941 = {'out': keyword_4940, 'axis': keyword_4938}
    # Getting the type of 'amin' (line 2356)
    amin_4936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2356, 15), 'amin', False)
    # Calling amin(args, kwargs) (line 2356)
    amin_call_result_4942 = invoke(stypy.reporting.localization.Localization(__file__, 2356, 15), amin_4936, *[], **kwargs_4941)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2356, 8), 'stypy_return_type', amin_call_result_4942)
    # SSA branch for the else part of an if statement (line 2349)
    module_type_store.open_ssa_branch('else')
    
    # Call to _amin(...): (line 2358)
    # Processing the call arguments (line 2358)
    # Getting the type of 'a' (line 2358)
    a_4945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2358, 30), 'a', False)
    # Processing the call keyword arguments (line 2358)
    # Getting the type of 'axis' (line 2358)
    axis_4946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2358, 38), 'axis', False)
    keyword_4947 = axis_4946
    # Getting the type of 'out' (line 2359)
    out_4948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2359, 34), 'out', False)
    keyword_4949 = out_4948
    # Getting the type of 'keepdims' (line 2359)
    keepdims_4950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2359, 48), 'keepdims', False)
    keyword_4951 = keepdims_4950
    kwargs_4952 = {'out': keyword_4949, 'keepdims': keyword_4951, 'axis': keyword_4947}
    # Getting the type of '_methods' (line 2358)
    _methods_4943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2358, 15), '_methods', False)
    # Obtaining the member '_amin' of a type (line 2358)
    _amin_4944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2358, 15), _methods_4943, '_amin')
    # Calling _amin(args, kwargs) (line 2358)
    _amin_call_result_4953 = invoke(stypy.reporting.localization.Localization(__file__, 2358, 15), _amin_4944, *[a_4945], **kwargs_4952)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2358, 8), 'stypy_return_type', _amin_call_result_4953)
    # SSA join for if statement (line 2349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'amin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'amin' in the type store
    # Getting the type of 'stypy_return_type' (line 2271)
    stypy_return_type_4954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4954)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'amin'
    return stypy_return_type_4954

# Assigning a type to the variable 'amin' (line 2271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2271, 0), 'amin', amin)

@norecursion
def alen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'alen'
    module_type_store = module_type_store.open_function_context('alen', 2362, 0, False)
    
    # Passed parameters checking function
    alen.stypy_localization = localization
    alen.stypy_type_of_self = None
    alen.stypy_type_store = module_type_store
    alen.stypy_function_name = 'alen'
    alen.stypy_param_names_list = ['a']
    alen.stypy_varargs_param_name = None
    alen.stypy_kwargs_param_name = None
    alen.stypy_call_defaults = defaults
    alen.stypy_call_varargs = varargs
    alen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'alen', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'alen', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'alen(...)' code ##################

    str_4955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2388, (-1)), 'str', '\n    Return the length of the first dimension of the input array.\n\n    Parameters\n    ----------\n    a : array_like\n       Input array.\n\n    Returns\n    -------\n    alen : int\n       Length of the first dimension of `a`.\n\n    See Also\n    --------\n    shape, size\n\n    Examples\n    --------\n    >>> a = np.zeros((7,4,5))\n    >>> a.shape[0]\n    7\n    >>> np.alen(a)\n    7\n\n    ')
    
    
    # SSA begins for try-except statement (line 2389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to len(...): (line 2390)
    # Processing the call arguments (line 2390)
    # Getting the type of 'a' (line 2390)
    a_4957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2390, 19), 'a', False)
    # Processing the call keyword arguments (line 2390)
    kwargs_4958 = {}
    # Getting the type of 'len' (line 2390)
    len_4956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2390, 15), 'len', False)
    # Calling len(args, kwargs) (line 2390)
    len_call_result_4959 = invoke(stypy.reporting.localization.Localization(__file__, 2390, 15), len_4956, *[a_4957], **kwargs_4958)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2390, 8), 'stypy_return_type', len_call_result_4959)
    # SSA branch for the except part of a try statement (line 2389)
    # SSA branch for the except 'TypeError' branch of a try statement (line 2389)
    module_type_store.open_ssa_branch('except')
    
    # Call to len(...): (line 2392)
    # Processing the call arguments (line 2392)
    
    # Call to array(...): (line 2392)
    # Processing the call arguments (line 2392)
    # Getting the type of 'a' (line 2392)
    a_4962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2392, 25), 'a', False)
    # Processing the call keyword arguments (line 2392)
    int_4963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2392, 34), 'int')
    keyword_4964 = int_4963
    kwargs_4965 = {'ndmin': keyword_4964}
    # Getting the type of 'array' (line 2392)
    array_4961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2392, 19), 'array', False)
    # Calling array(args, kwargs) (line 2392)
    array_call_result_4966 = invoke(stypy.reporting.localization.Localization(__file__, 2392, 19), array_4961, *[a_4962], **kwargs_4965)
    
    # Processing the call keyword arguments (line 2392)
    kwargs_4967 = {}
    # Getting the type of 'len' (line 2392)
    len_4960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2392, 15), 'len', False)
    # Calling len(args, kwargs) (line 2392)
    len_call_result_4968 = invoke(stypy.reporting.localization.Localization(__file__, 2392, 15), len_4960, *[array_call_result_4966], **kwargs_4967)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2392, 8), 'stypy_return_type', len_call_result_4968)
    # SSA join for try-except statement (line 2389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'alen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'alen' in the type store
    # Getting the type of 'stypy_return_type' (line 2362)
    stypy_return_type_4969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2362, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4969)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'alen'
    return stypy_return_type_4969

# Assigning a type to the variable 'alen' (line 2362)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2362, 0), 'alen', alen)

@norecursion
def prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2395)
    None_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2395, 17), 'None')
    # Getting the type of 'None' (line 2395)
    None_4971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2395, 29), 'None')
    # Getting the type of 'None' (line 2395)
    None_4972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2395, 39), 'None')
    # Getting the type of 'False' (line 2395)
    False_4973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2395, 54), 'False')
    defaults = [None_4970, None_4971, None_4972, False_4973]
    # Create a new context for function 'prod'
    module_type_store = module_type_store.open_function_context('prod', 2395, 0, False)
    
    # Passed parameters checking function
    prod.stypy_localization = localization
    prod.stypy_type_of_self = None
    prod.stypy_type_store = module_type_store
    prod.stypy_function_name = 'prod'
    prod.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    prod.stypy_varargs_param_name = None
    prod.stypy_kwargs_param_name = None
    prod.stypy_call_defaults = defaults
    prod.stypy_call_varargs = varargs
    prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prod', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prod', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prod(...)' code ##################

    str_4974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2486, (-1)), 'str', '\n    Return the product of array elements over a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which a product is performed.  The default,\n        axis=None, will calculate the product of all the elements in the\n        input array. If axis is negative it counts from the last to the\n        first axis.\n\n        .. versionadded:: 1.7.0\n\n        If axis is a tuple of ints, a product is performed on all of the\n        axes specified in the tuple instead of a single axis or all the\n        axes as before.\n    dtype : dtype, optional\n        The type of the returned array, as well as of the accumulator in\n        which the elements are multiplied.  The dtype of `a` is used by\n        default unless `a` has an integer dtype of less precision than the\n        default platform integer.  In that case, if `a` is signed then the\n        platform integer is used while if `a` is unsigned then an unsigned\n        integer of the same precision as the platform integer is used.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape as the expected output, but the type of the output\n        values will be cast if necessary.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the input array.\n\n    Returns\n    -------\n    product_along_axis : ndarray, see `dtype` parameter above.\n        An array shaped as `a` but with the specified axis removed.\n        Returns a reference to `out` if specified.\n\n    See Also\n    --------\n    ndarray.prod : equivalent method\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    Arithmetic is modular when using integer types, and no error is\n    raised on overflow.  That means that, on a 32-bit platform:\n\n    >>> x = np.array([536870910, 536870910, 536870910, 536870910])\n    >>> np.prod(x) #random\n    16\n\n    The product of an empty array is the neutral element 1:\n\n    >>> np.prod([])\n    1.0\n\n    Examples\n    --------\n    By default, calculate the product of all elements:\n\n    >>> np.prod([1.,2.])\n    2.0\n\n    Even when the input array is two-dimensional:\n\n    >>> np.prod([[1.,2.],[3.,4.]])\n    24.0\n\n    But we can also specify the axis over which to multiply:\n\n    >>> np.prod([[1.,2.],[3.,4.]], axis=1)\n    array([  2.,  12.])\n\n    If the type of `x` is unsigned, then the output type is\n    the unsigned platform integer:\n\n    >>> x = np.array([1, 2, 3], dtype=np.uint8)\n    >>> np.prod(x).dtype == np.uint\n    True\n\n    If `x` is of a signed integer type, then the output type\n    is the default platform integer:\n\n    >>> x = np.array([1, 2, 3], dtype=np.int8)\n    >>> np.prod(x).dtype == np.int\n    True\n\n    ')
    
    
    
    # Call to type(...): (line 2487)
    # Processing the call arguments (line 2487)
    # Getting the type of 'a' (line 2487)
    a_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2487, 12), 'a', False)
    # Processing the call keyword arguments (line 2487)
    kwargs_4977 = {}
    # Getting the type of 'type' (line 2487)
    type_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2487, 7), 'type', False)
    # Calling type(args, kwargs) (line 2487)
    type_call_result_4978 = invoke(stypy.reporting.localization.Localization(__file__, 2487, 7), type_4975, *[a_4976], **kwargs_4977)
    
    # Getting the type of 'mu' (line 2487)
    mu_4979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2487, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 2487)
    ndarray_4980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2487, 22), mu_4979, 'ndarray')
    # Applying the binary operator 'isnot' (line 2487)
    result_is_not_4981 = python_operator(stypy.reporting.localization.Localization(__file__, 2487, 7), 'isnot', type_call_result_4978, ndarray_4980)
    
    # Testing the type of an if condition (line 2487)
    if_condition_4982 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2487, 4), result_is_not_4981)
    # Assigning a type to the variable 'if_condition_4982' (line 2487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2487, 4), 'if_condition_4982', if_condition_4982)
    # SSA begins for if statement (line 2487)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 2488)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2489):
    # Getting the type of 'a' (line 2489)
    a_4983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2489, 19), 'a')
    # Obtaining the member 'prod' of a type (line 2489)
    prod_4984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2489, 19), a_4983, 'prod')
    # Assigning a type to the variable 'prod' (line 2489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2489, 12), 'prod', prod_4984)
    # SSA branch for the except part of a try statement (line 2488)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2488)
    module_type_store.open_ssa_branch('except')
    
    # Call to _prod(...): (line 2491)
    # Processing the call arguments (line 2491)
    # Getting the type of 'a' (line 2491)
    a_4987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2491, 34), 'a', False)
    # Processing the call keyword arguments (line 2491)
    # Getting the type of 'axis' (line 2491)
    axis_4988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2491, 42), 'axis', False)
    keyword_4989 = axis_4988
    # Getting the type of 'dtype' (line 2491)
    dtype_4990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2491, 54), 'dtype', False)
    keyword_4991 = dtype_4990
    # Getting the type of 'out' (line 2492)
    out_4992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2492, 38), 'out', False)
    keyword_4993 = out_4992
    # Getting the type of 'keepdims' (line 2492)
    keepdims_4994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2492, 52), 'keepdims', False)
    keyword_4995 = keepdims_4994
    kwargs_4996 = {'dtype': keyword_4991, 'out': keyword_4993, 'keepdims': keyword_4995, 'axis': keyword_4989}
    # Getting the type of '_methods' (line 2491)
    _methods_4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2491, 19), '_methods', False)
    # Obtaining the member '_prod' of a type (line 2491)
    _prod_4986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2491, 19), _methods_4985, '_prod')
    # Calling _prod(args, kwargs) (line 2491)
    _prod_call_result_4997 = invoke(stypy.reporting.localization.Localization(__file__, 2491, 19), _prod_4986, *[a_4987], **kwargs_4996)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2491, 12), 'stypy_return_type', _prod_call_result_4997)
    # SSA join for try-except statement (line 2488)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to prod(...): (line 2493)
    # Processing the call keyword arguments (line 2493)
    # Getting the type of 'axis' (line 2493)
    axis_4999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2493, 25), 'axis', False)
    keyword_5000 = axis_4999
    # Getting the type of 'dtype' (line 2493)
    dtype_5001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2493, 37), 'dtype', False)
    keyword_5002 = dtype_5001
    # Getting the type of 'out' (line 2493)
    out_5003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2493, 48), 'out', False)
    keyword_5004 = out_5003
    kwargs_5005 = {'dtype': keyword_5002, 'out': keyword_5004, 'axis': keyword_5000}
    # Getting the type of 'prod' (line 2493)
    prod_4998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2493, 15), 'prod', False)
    # Calling prod(args, kwargs) (line 2493)
    prod_call_result_5006 = invoke(stypy.reporting.localization.Localization(__file__, 2493, 15), prod_4998, *[], **kwargs_5005)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2493, 8), 'stypy_return_type', prod_call_result_5006)
    # SSA branch for the else part of an if statement (line 2487)
    module_type_store.open_ssa_branch('else')
    
    # Call to _prod(...): (line 2495)
    # Processing the call arguments (line 2495)
    # Getting the type of 'a' (line 2495)
    a_5009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2495, 30), 'a', False)
    # Processing the call keyword arguments (line 2495)
    # Getting the type of 'axis' (line 2495)
    axis_5010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2495, 38), 'axis', False)
    keyword_5011 = axis_5010
    # Getting the type of 'dtype' (line 2495)
    dtype_5012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2495, 50), 'dtype', False)
    keyword_5013 = dtype_5012
    # Getting the type of 'out' (line 2496)
    out_5014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2496, 34), 'out', False)
    keyword_5015 = out_5014
    # Getting the type of 'keepdims' (line 2496)
    keepdims_5016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2496, 48), 'keepdims', False)
    keyword_5017 = keepdims_5016
    kwargs_5018 = {'dtype': keyword_5013, 'out': keyword_5015, 'keepdims': keyword_5017, 'axis': keyword_5011}
    # Getting the type of '_methods' (line 2495)
    _methods_5007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2495, 15), '_methods', False)
    # Obtaining the member '_prod' of a type (line 2495)
    _prod_5008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2495, 15), _methods_5007, '_prod')
    # Calling _prod(args, kwargs) (line 2495)
    _prod_call_result_5019 = invoke(stypy.reporting.localization.Localization(__file__, 2495, 15), _prod_5008, *[a_5009], **kwargs_5018)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2495, 8), 'stypy_return_type', _prod_call_result_5019)
    # SSA join for if statement (line 2487)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prod' in the type store
    # Getting the type of 'stypy_return_type' (line 2395)
    stypy_return_type_5020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2395, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5020)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prod'
    return stypy_return_type_5020

# Assigning a type to the variable 'prod' (line 2395)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2395, 0), 'prod', prod)

@norecursion
def cumprod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2499)
    None_5021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2499, 20), 'None')
    # Getting the type of 'None' (line 2499)
    None_5022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2499, 32), 'None')
    # Getting the type of 'None' (line 2499)
    None_5023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2499, 42), 'None')
    defaults = [None_5021, None_5022, None_5023]
    # Create a new context for function 'cumprod'
    module_type_store = module_type_store.open_function_context('cumprod', 2499, 0, False)
    
    # Passed parameters checking function
    cumprod.stypy_localization = localization
    cumprod.stypy_type_of_self = None
    cumprod.stypy_type_store = module_type_store
    cumprod.stypy_function_name = 'cumprod'
    cumprod.stypy_param_names_list = ['a', 'axis', 'dtype', 'out']
    cumprod.stypy_varargs_param_name = None
    cumprod.stypy_kwargs_param_name = None
    cumprod.stypy_call_defaults = defaults
    cumprod.stypy_call_varargs = varargs
    cumprod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cumprod', ['a', 'axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cumprod', localization, ['a', 'axis', 'dtype', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cumprod(...)' code ##################

    str_5024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2558, (-1)), 'str', '\n    Return the cumulative product of elements along a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int, optional\n        Axis along which the cumulative product is computed.  By default\n        the input is flattened.\n    dtype : dtype, optional\n        Type of the returned array, as well as of the accumulator in which\n        the elements are multiplied.  If *dtype* is not specified, it\n        defaults to the dtype of `a`, unless `a` has an integer dtype with\n        a precision less than that of the default platform integer.  In\n        that case, the default platform integer is used instead.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output\n        but the type of the resulting values will be cast if necessary.\n\n    Returns\n    -------\n    cumprod : ndarray\n        A new array holding the result is returned unless `out` is\n        specified, in which case a reference to out is returned.\n\n    See Also\n    --------\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    Arithmetic is modular when using integer types, and no error is\n    raised on overflow.\n\n    Examples\n    --------\n    >>> a = np.array([1,2,3])\n    >>> np.cumprod(a) # intermediate results 1, 1*2\n    ...               # total product 1*2*3 = 6\n    array([1, 2, 6])\n    >>> a = np.array([[1, 2, 3], [4, 5, 6]])\n    >>> np.cumprod(a, dtype=float) # specify type of output\n    array([   1.,    2.,    6.,   24.,  120.,  720.])\n\n    The cumulative product for each column (i.e., over the rows) of `a`:\n\n    >>> np.cumprod(a, axis=0)\n    array([[ 1,  2,  3],\n           [ 4, 10, 18]])\n\n    The cumulative product for each row (i.e. over the columns) of `a`:\n\n    >>> np.cumprod(a,axis=1)\n    array([[  1,   2,   6],\n           [  4,  20, 120]])\n\n    ')
    
    
    # SSA begins for try-except statement (line 2559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2560):
    # Getting the type of 'a' (line 2560)
    a_5025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2560, 18), 'a')
    # Obtaining the member 'cumprod' of a type (line 2560)
    cumprod_5026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2560, 18), a_5025, 'cumprod')
    # Assigning a type to the variable 'cumprod' (line 2560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2560, 8), 'cumprod', cumprod_5026)
    # SSA branch for the except part of a try statement (line 2559)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2559)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2562)
    # Processing the call arguments (line 2562)
    # Getting the type of 'a' (line 2562)
    a_5028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 23), 'a', False)
    str_5029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2562, 26), 'str', 'cumprod')
    # Getting the type of 'axis' (line 2562)
    axis_5030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 37), 'axis', False)
    # Getting the type of 'dtype' (line 2562)
    dtype_5031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 43), 'dtype', False)
    # Getting the type of 'out' (line 2562)
    out_5032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 50), 'out', False)
    # Processing the call keyword arguments (line 2562)
    kwargs_5033 = {}
    # Getting the type of '_wrapit' (line 2562)
    _wrapit_5027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2562, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2562)
    _wrapit_call_result_5034 = invoke(stypy.reporting.localization.Localization(__file__, 2562, 15), _wrapit_5027, *[a_5028, str_5029, axis_5030, dtype_5031, out_5032], **kwargs_5033)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2562, 8), 'stypy_return_type', _wrapit_call_result_5034)
    # SSA join for try-except statement (line 2559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to cumprod(...): (line 2563)
    # Processing the call arguments (line 2563)
    # Getting the type of 'axis' (line 2563)
    axis_5036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2563, 19), 'axis', False)
    # Getting the type of 'dtype' (line 2563)
    dtype_5037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2563, 25), 'dtype', False)
    # Getting the type of 'out' (line 2563)
    out_5038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2563, 32), 'out', False)
    # Processing the call keyword arguments (line 2563)
    kwargs_5039 = {}
    # Getting the type of 'cumprod' (line 2563)
    cumprod_5035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2563, 11), 'cumprod', False)
    # Calling cumprod(args, kwargs) (line 2563)
    cumprod_call_result_5040 = invoke(stypy.reporting.localization.Localization(__file__, 2563, 11), cumprod_5035, *[axis_5036, dtype_5037, out_5038], **kwargs_5039)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2563, 4), 'stypy_return_type', cumprod_call_result_5040)
    
    # ################# End of 'cumprod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cumprod' in the type store
    # Getting the type of 'stypy_return_type' (line 2499)
    stypy_return_type_5041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2499, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cumprod'
    return stypy_return_type_5041

# Assigning a type to the variable 'cumprod' (line 2499)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2499, 0), 'cumprod', cumprod)

@norecursion
def ndim(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ndim'
    module_type_store = module_type_store.open_function_context('ndim', 2566, 0, False)
    
    # Passed parameters checking function
    ndim.stypy_localization = localization
    ndim.stypy_type_of_self = None
    ndim.stypy_type_store = module_type_store
    ndim.stypy_function_name = 'ndim'
    ndim.stypy_param_names_list = ['a']
    ndim.stypy_varargs_param_name = None
    ndim.stypy_kwargs_param_name = None
    ndim.stypy_call_defaults = defaults
    ndim.stypy_call_varargs = varargs
    ndim.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ndim', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ndim', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ndim(...)' code ##################

    str_5042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2596, (-1)), 'str', '\n    Return the number of dimensions of an array.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.  If it is not already an ndarray, a conversion is\n        attempted.\n\n    Returns\n    -------\n    number_of_dimensions : int\n        The number of dimensions in `a`.  Scalars are zero-dimensional.\n\n    See Also\n    --------\n    ndarray.ndim : equivalent method\n    shape : dimensions of array\n    ndarray.shape : dimensions of array\n\n    Examples\n    --------\n    >>> np.ndim([[1,2,3],[4,5,6]])\n    2\n    >>> np.ndim(np.array([[1,2,3],[4,5,6]]))\n    2\n    >>> np.ndim(1)\n    0\n\n    ')
    
    
    # SSA begins for try-except statement (line 2597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'a' (line 2598)
    a_5043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2598, 15), 'a')
    # Obtaining the member 'ndim' of a type (line 2598)
    ndim_5044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2598, 15), a_5043, 'ndim')
    # Assigning a type to the variable 'stypy_return_type' (line 2598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2598, 8), 'stypy_return_type', ndim_5044)
    # SSA branch for the except part of a try statement (line 2597)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2597)
    module_type_store.open_ssa_branch('except')
    
    # Call to asarray(...): (line 2600)
    # Processing the call arguments (line 2600)
    # Getting the type of 'a' (line 2600)
    a_5046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2600, 23), 'a', False)
    # Processing the call keyword arguments (line 2600)
    kwargs_5047 = {}
    # Getting the type of 'asarray' (line 2600)
    asarray_5045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2600, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 2600)
    asarray_call_result_5048 = invoke(stypy.reporting.localization.Localization(__file__, 2600, 15), asarray_5045, *[a_5046], **kwargs_5047)
    
    # Obtaining the member 'ndim' of a type (line 2600)
    ndim_5049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2600, 15), asarray_call_result_5048, 'ndim')
    # Assigning a type to the variable 'stypy_return_type' (line 2600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2600, 8), 'stypy_return_type', ndim_5049)
    # SSA join for try-except statement (line 2597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'ndim(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ndim' in the type store
    # Getting the type of 'stypy_return_type' (line 2566)
    stypy_return_type_5050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2566, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ndim'
    return stypy_return_type_5050

# Assigning a type to the variable 'ndim' (line 2566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2566, 0), 'ndim', ndim)

@norecursion
def rank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rank'
    module_type_store = module_type_store.open_function_context('rank', 2603, 0, False)
    
    # Passed parameters checking function
    rank.stypy_localization = localization
    rank.stypy_type_of_self = None
    rank.stypy_type_store = module_type_store
    rank.stypy_function_name = 'rank'
    rank.stypy_param_names_list = ['a']
    rank.stypy_varargs_param_name = None
    rank.stypy_kwargs_param_name = None
    rank.stypy_call_defaults = defaults
    rank.stypy_call_varargs = varargs
    rank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rank', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rank', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rank(...)' code ##################

    str_5051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2647, (-1)), 'str', '\n    Return the number of dimensions of an array.\n\n    If `a` is not already an array, a conversion is attempted.\n    Scalars are zero dimensional.\n\n    .. note::\n        This function is deprecated in NumPy 1.9 to avoid confusion with\n        `numpy.linalg.matrix_rank`. The ``ndim`` attribute or function\n        should be used instead.\n\n    Parameters\n    ----------\n    a : array_like\n        Array whose number of dimensions is desired. If `a` is not an array,\n        a conversion is attempted.\n\n    Returns\n    -------\n    number_of_dimensions : int\n        The number of dimensions in the array.\n\n    See Also\n    --------\n    ndim : equivalent function\n    ndarray.ndim : equivalent property\n    shape : dimensions of array\n    ndarray.shape : dimensions of array\n\n    Notes\n    -----\n    In the old Numeric package, `rank` was the term used for the number of\n    dimensions, but in Numpy `ndim` is used instead.\n\n    Examples\n    --------\n    >>> np.rank([1,2,3])\n    1\n    >>> np.rank(np.array([[1,2,3],[4,5,6]]))\n    2\n    >>> np.rank(1)\n    0\n\n    ')
    
    # Call to warn(...): (line 2649)
    # Processing the call arguments (line 2649)
    str_5054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2650, 8), 'str', '`rank` is deprecated; use the `ndim` attribute or function instead. To find the rank of a matrix see `numpy.linalg.matrix_rank`.')
    # Getting the type of 'VisibleDeprecationWarning' (line 2652)
    VisibleDeprecationWarning_5055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2652, 8), 'VisibleDeprecationWarning', False)
    # Processing the call keyword arguments (line 2649)
    kwargs_5056 = {}
    # Getting the type of 'warnings' (line 2649)
    warnings_5052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2649, 4), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 2649)
    warn_5053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2649, 4), warnings_5052, 'warn')
    # Calling warn(args, kwargs) (line 2649)
    warn_call_result_5057 = invoke(stypy.reporting.localization.Localization(__file__, 2649, 4), warn_5053, *[str_5054, VisibleDeprecationWarning_5055], **kwargs_5056)
    
    
    
    # SSA begins for try-except statement (line 2653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'a' (line 2654)
    a_5058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2654, 15), 'a')
    # Obtaining the member 'ndim' of a type (line 2654)
    ndim_5059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2654, 15), a_5058, 'ndim')
    # Assigning a type to the variable 'stypy_return_type' (line 2654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2654, 8), 'stypy_return_type', ndim_5059)
    # SSA branch for the except part of a try statement (line 2653)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2653)
    module_type_store.open_ssa_branch('except')
    
    # Call to asarray(...): (line 2656)
    # Processing the call arguments (line 2656)
    # Getting the type of 'a' (line 2656)
    a_5061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2656, 23), 'a', False)
    # Processing the call keyword arguments (line 2656)
    kwargs_5062 = {}
    # Getting the type of 'asarray' (line 2656)
    asarray_5060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2656, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 2656)
    asarray_call_result_5063 = invoke(stypy.reporting.localization.Localization(__file__, 2656, 15), asarray_5060, *[a_5061], **kwargs_5062)
    
    # Obtaining the member 'ndim' of a type (line 2656)
    ndim_5064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2656, 15), asarray_call_result_5063, 'ndim')
    # Assigning a type to the variable 'stypy_return_type' (line 2656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2656, 8), 'stypy_return_type', ndim_5064)
    # SSA join for try-except statement (line 2653)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'rank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rank' in the type store
    # Getting the type of 'stypy_return_type' (line 2603)
    stypy_return_type_5065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2603, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rank'
    return stypy_return_type_5065

# Assigning a type to the variable 'rank' (line 2603)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2603, 0), 'rank', rank)

@norecursion
def size(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2659)
    None_5066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2659, 17), 'None')
    defaults = [None_5066]
    # Create a new context for function 'size'
    module_type_store = module_type_store.open_function_context('size', 2659, 0, False)
    
    # Passed parameters checking function
    size.stypy_localization = localization
    size.stypy_type_of_self = None
    size.stypy_type_store = module_type_store
    size.stypy_function_name = 'size'
    size.stypy_param_names_list = ['a', 'axis']
    size.stypy_varargs_param_name = None
    size.stypy_kwargs_param_name = None
    size.stypy_call_defaults = defaults
    size.stypy_call_varargs = varargs
    size.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'size', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'size', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'size(...)' code ##################

    str_5067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2692, (-1)), 'str', '\n    Return the number of elements along a given axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : int, optional\n        Axis along which the elements are counted.  By default, give\n        the total number of elements.\n\n    Returns\n    -------\n    element_count : int\n        Number of elements along the specified axis.\n\n    See Also\n    --------\n    shape : dimensions of array\n    ndarray.shape : dimensions of array\n    ndarray.size : number of elements in array\n\n    Examples\n    --------\n    >>> a = np.array([[1,2,3],[4,5,6]])\n    >>> np.size(a)\n    6\n    >>> np.size(a,1)\n    3\n    >>> np.size(a,0)\n    2\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 2693)
    # Getting the type of 'axis' (line 2693)
    axis_5068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2693, 7), 'axis')
    # Getting the type of 'None' (line 2693)
    None_5069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2693, 15), 'None')
    
    (may_be_5070, more_types_in_union_5071) = may_be_none(axis_5068, None_5069)

    if may_be_5070:

        if more_types_in_union_5071:
            # Runtime conditional SSA (line 2693)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 2694)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'a' (line 2695)
        a_5072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2695, 19), 'a')
        # Obtaining the member 'size' of a type (line 2695)
        size_5073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2695, 19), a_5072, 'size')
        # Assigning a type to the variable 'stypy_return_type' (line 2695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2695, 12), 'stypy_return_type', size_5073)
        # SSA branch for the except part of a try statement (line 2694)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 2694)
        module_type_store.open_ssa_branch('except')
        
        # Call to asarray(...): (line 2697)
        # Processing the call arguments (line 2697)
        # Getting the type of 'a' (line 2697)
        a_5075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2697, 27), 'a', False)
        # Processing the call keyword arguments (line 2697)
        kwargs_5076 = {}
        # Getting the type of 'asarray' (line 2697)
        asarray_5074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2697, 19), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2697)
        asarray_call_result_5077 = invoke(stypy.reporting.localization.Localization(__file__, 2697, 19), asarray_5074, *[a_5075], **kwargs_5076)
        
        # Obtaining the member 'size' of a type (line 2697)
        size_5078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2697, 19), asarray_call_result_5077, 'size')
        # Assigning a type to the variable 'stypy_return_type' (line 2697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2697, 12), 'stypy_return_type', size_5078)
        # SSA join for try-except statement (line 2694)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_5071:
            # Runtime conditional SSA for else branch (line 2693)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_5070) or more_types_in_union_5071):
        
        
        # SSA begins for try-except statement (line 2699)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 2700)
        axis_5079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2700, 27), 'axis')
        # Getting the type of 'a' (line 2700)
        a_5080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2700, 19), 'a')
        # Obtaining the member 'shape' of a type (line 2700)
        shape_5081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2700, 19), a_5080, 'shape')
        # Obtaining the member '__getitem__' of a type (line 2700)
        getitem___5082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2700, 19), shape_5081, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 2700)
        subscript_call_result_5083 = invoke(stypy.reporting.localization.Localization(__file__, 2700, 19), getitem___5082, axis_5079)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2700, 12), 'stypy_return_type', subscript_call_result_5083)
        # SSA branch for the except part of a try statement (line 2699)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 2699)
        module_type_store.open_ssa_branch('except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 2702)
        axis_5084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2702, 36), 'axis')
        
        # Call to asarray(...): (line 2702)
        # Processing the call arguments (line 2702)
        # Getting the type of 'a' (line 2702)
        a_5086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2702, 27), 'a', False)
        # Processing the call keyword arguments (line 2702)
        kwargs_5087 = {}
        # Getting the type of 'asarray' (line 2702)
        asarray_5085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2702, 19), 'asarray', False)
        # Calling asarray(args, kwargs) (line 2702)
        asarray_call_result_5088 = invoke(stypy.reporting.localization.Localization(__file__, 2702, 19), asarray_5085, *[a_5086], **kwargs_5087)
        
        # Obtaining the member 'shape' of a type (line 2702)
        shape_5089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2702, 19), asarray_call_result_5088, 'shape')
        # Obtaining the member '__getitem__' of a type (line 2702)
        getitem___5090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2702, 19), shape_5089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 2702)
        subscript_call_result_5091 = invoke(stypy.reporting.localization.Localization(__file__, 2702, 19), getitem___5090, axis_5084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 2702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2702, 12), 'stypy_return_type', subscript_call_result_5091)
        # SSA join for try-except statement (line 2699)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_5070 and more_types_in_union_5071):
            # SSA join for if statement (line 2693)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'size(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'size' in the type store
    # Getting the type of 'stypy_return_type' (line 2659)
    stypy_return_type_5092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2659, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'size'
    return stypy_return_type_5092

# Assigning a type to the variable 'size' (line 2659)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2659, 0), 'size', size)

@norecursion
def around(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_5093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2705, 23), 'int')
    # Getting the type of 'None' (line 2705)
    None_5094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2705, 30), 'None')
    defaults = [int_5093, None_5094]
    # Create a new context for function 'around'
    module_type_store = module_type_store.open_function_context('around', 2705, 0, False)
    
    # Passed parameters checking function
    around.stypy_localization = localization
    around.stypy_type_of_self = None
    around.stypy_type_store = module_type_store
    around.stypy_function_name = 'around'
    around.stypy_param_names_list = ['a', 'decimals', 'out']
    around.stypy_varargs_param_name = None
    around.stypy_kwargs_param_name = None
    around.stypy_call_defaults = defaults
    around.stypy_call_varargs = varargs
    around.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'around', ['a', 'decimals', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'around', localization, ['a', 'decimals', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'around(...)' code ##################

    str_5095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2770, (-1)), 'str', '\n    Evenly round to the given number of decimals.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    decimals : int, optional\n        Number of decimal places to round to (default: 0).  If\n        decimals is negative, it specifies the number of positions to\n        the left of the decimal point.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape as the expected output, but the type of the output\n        values will be cast if necessary. See `doc.ufuncs` (Section\n        "Output arguments") for details.\n\n    Returns\n    -------\n    rounded_array : ndarray\n        An array of the same type as `a`, containing the rounded values.\n        Unless `out` was specified, a new array is created.  A reference to\n        the result is returned.\n\n        The real and imaginary parts of complex numbers are rounded\n        separately.  The result of rounding a float is a float.\n\n    See Also\n    --------\n    ndarray.round : equivalent method\n\n    ceil, fix, floor, rint, trunc\n\n\n    Notes\n    -----\n    For values exactly halfway between rounded decimal values, Numpy\n    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,\n    -0.5 and 0.5 round to 0.0, etc. Results may also be surprising due\n    to the inexact representation of decimal fractions in the IEEE\n    floating point standard [1]_ and errors introduced when scaling\n    by powers of ten.\n\n    References\n    ----------\n    .. [1] "Lecture Notes on the Status of  IEEE 754", William Kahan,\n           http://www.cs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF\n    .. [2] "How Futile are Mindless Assessments of\n           Roundoff in Floating-Point Computation?", William Kahan,\n           http://www.cs.berkeley.edu/~wkahan/Mindless.pdf\n\n    Examples\n    --------\n    >>> np.around([0.37, 1.64])\n    array([ 0.,  2.])\n    >>> np.around([0.37, 1.64], decimals=1)\n    array([ 0.4,  1.6])\n    >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value\n    array([ 0.,  2.,  2.,  4.,  4.])\n    >>> np.around([1,2,3,11], decimals=1) # ndarray of ints is returned\n    array([ 1,  2,  3, 11])\n    >>> np.around([1,2,3,11], decimals=-1)\n    array([ 0,  0,  0, 10])\n\n    ')
    
    
    # SSA begins for try-except statement (line 2771)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2772):
    # Getting the type of 'a' (line 2772)
    a_5096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2772, 16), 'a')
    # Obtaining the member 'round' of a type (line 2772)
    round_5097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2772, 16), a_5096, 'round')
    # Assigning a type to the variable 'round' (line 2772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2772, 8), 'round', round_5097)
    # SSA branch for the except part of a try statement (line 2771)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2771)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2774)
    # Processing the call arguments (line 2774)
    # Getting the type of 'a' (line 2774)
    a_5099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2774, 23), 'a', False)
    str_5100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2774, 26), 'str', 'round')
    # Getting the type of 'decimals' (line 2774)
    decimals_5101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2774, 35), 'decimals', False)
    # Getting the type of 'out' (line 2774)
    out_5102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2774, 45), 'out', False)
    # Processing the call keyword arguments (line 2774)
    kwargs_5103 = {}
    # Getting the type of '_wrapit' (line 2774)
    _wrapit_5098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2774, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2774)
    _wrapit_call_result_5104 = invoke(stypy.reporting.localization.Localization(__file__, 2774, 15), _wrapit_5098, *[a_5099, str_5100, decimals_5101, out_5102], **kwargs_5103)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2774, 8), 'stypy_return_type', _wrapit_call_result_5104)
    # SSA join for try-except statement (line 2771)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to round(...): (line 2775)
    # Processing the call arguments (line 2775)
    # Getting the type of 'decimals' (line 2775)
    decimals_5106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2775, 17), 'decimals', False)
    # Getting the type of 'out' (line 2775)
    out_5107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2775, 27), 'out', False)
    # Processing the call keyword arguments (line 2775)
    kwargs_5108 = {}
    # Getting the type of 'round' (line 2775)
    round_5105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2775, 11), 'round', False)
    # Calling round(args, kwargs) (line 2775)
    round_call_result_5109 = invoke(stypy.reporting.localization.Localization(__file__, 2775, 11), round_5105, *[decimals_5106, out_5107], **kwargs_5108)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2775, 4), 'stypy_return_type', round_call_result_5109)
    
    # ################# End of 'around(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'around' in the type store
    # Getting the type of 'stypy_return_type' (line 2705)
    stypy_return_type_5110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2705, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'around'
    return stypy_return_type_5110

# Assigning a type to the variable 'around' (line 2705)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2705, 0), 'around', around)

@norecursion
def round_(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_5111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2778, 23), 'int')
    # Getting the type of 'None' (line 2778)
    None_5112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2778, 30), 'None')
    defaults = [int_5111, None_5112]
    # Create a new context for function 'round_'
    module_type_store = module_type_store.open_function_context('round_', 2778, 0, False)
    
    # Passed parameters checking function
    round_.stypy_localization = localization
    round_.stypy_type_of_self = None
    round_.stypy_type_store = module_type_store
    round_.stypy_function_name = 'round_'
    round_.stypy_param_names_list = ['a', 'decimals', 'out']
    round_.stypy_varargs_param_name = None
    round_.stypy_kwargs_param_name = None
    round_.stypy_call_defaults = defaults
    round_.stypy_call_varargs = varargs
    round_.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'round_', ['a', 'decimals', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'round_', localization, ['a', 'decimals', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'round_(...)' code ##################

    str_5113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2788, (-1)), 'str', '\n    Round an array to the given number of decimals.\n\n    Refer to `around` for full documentation.\n\n    See Also\n    --------\n    around : equivalent function\n\n    ')
    
    
    # SSA begins for try-except statement (line 2789)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2790):
    # Getting the type of 'a' (line 2790)
    a_5114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2790, 16), 'a')
    # Obtaining the member 'round' of a type (line 2790)
    round_5115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2790, 16), a_5114, 'round')
    # Assigning a type to the variable 'round' (line 2790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2790, 8), 'round', round_5115)
    # SSA branch for the except part of a try statement (line 2789)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2789)
    module_type_store.open_ssa_branch('except')
    
    # Call to _wrapit(...): (line 2792)
    # Processing the call arguments (line 2792)
    # Getting the type of 'a' (line 2792)
    a_5117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2792, 23), 'a', False)
    str_5118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2792, 26), 'str', 'round')
    # Getting the type of 'decimals' (line 2792)
    decimals_5119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2792, 35), 'decimals', False)
    # Getting the type of 'out' (line 2792)
    out_5120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2792, 45), 'out', False)
    # Processing the call keyword arguments (line 2792)
    kwargs_5121 = {}
    # Getting the type of '_wrapit' (line 2792)
    _wrapit_5116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2792, 15), '_wrapit', False)
    # Calling _wrapit(args, kwargs) (line 2792)
    _wrapit_call_result_5122 = invoke(stypy.reporting.localization.Localization(__file__, 2792, 15), _wrapit_5116, *[a_5117, str_5118, decimals_5119, out_5120], **kwargs_5121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2792, 8), 'stypy_return_type', _wrapit_call_result_5122)
    # SSA join for try-except statement (line 2789)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to round(...): (line 2793)
    # Processing the call arguments (line 2793)
    # Getting the type of 'decimals' (line 2793)
    decimals_5124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2793, 17), 'decimals', False)
    # Getting the type of 'out' (line 2793)
    out_5125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2793, 27), 'out', False)
    # Processing the call keyword arguments (line 2793)
    kwargs_5126 = {}
    # Getting the type of 'round' (line 2793)
    round_5123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2793, 11), 'round', False)
    # Calling round(args, kwargs) (line 2793)
    round_call_result_5127 = invoke(stypy.reporting.localization.Localization(__file__, 2793, 11), round_5123, *[decimals_5124, out_5125], **kwargs_5126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2793)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2793, 4), 'stypy_return_type', round_call_result_5127)
    
    # ################# End of 'round_(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'round_' in the type store
    # Getting the type of 'stypy_return_type' (line 2778)
    stypy_return_type_5128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2778, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'round_'
    return stypy_return_type_5128

# Assigning a type to the variable 'round_' (line 2778)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2778, 0), 'round_', round_)

@norecursion
def mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2796)
    None_5129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2796, 17), 'None')
    # Getting the type of 'None' (line 2796)
    None_5130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2796, 29), 'None')
    # Getting the type of 'None' (line 2796)
    None_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2796, 39), 'None')
    # Getting the type of 'False' (line 2796)
    False_5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2796, 54), 'False')
    defaults = [None_5129, None_5130, None_5131, False_5132]
    # Create a new context for function 'mean'
    module_type_store = module_type_store.open_function_context('mean', 2796, 0, False)
    
    # Passed parameters checking function
    mean.stypy_localization = localization
    mean.stypy_type_of_self = None
    mean.stypy_type_store = module_type_store
    mean.stypy_function_name = 'mean'
    mean.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    mean.stypy_varargs_param_name = None
    mean.stypy_kwargs_param_name = None
    mean.stypy_call_defaults = defaults
    mean.stypy_call_varargs = varargs
    mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mean', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mean', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mean(...)' code ##################

    str_5133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2876, (-1)), 'str', '\n    Compute the arithmetic mean along the specified axis.\n\n    Returns the average of the array elements.  The average is taken over\n    the flattened array by default, otherwise over the specified axis.\n    `float64` intermediate and return values are used for integer inputs.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose mean is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which the means are computed. The default is to\n        compute the mean of the flattened array.\n\n        .. versionadded: 1.7.0\n\n        If this is a tuple of ints, a mean is performed over multiple axes,\n        instead of a single axis or all the axes as before.\n    dtype : data-type, optional\n        Type to use in computing the mean.  For integer inputs, the default\n        is `float64`; for floating point inputs, it is the same as the\n        input dtype.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``; if provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.\n        See `doc.ufuncs` for details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    m : ndarray, see dtype parameter above\n        If `out=None`, returns a new array containing the mean values,\n        otherwise a reference to the output array is returned.\n\n    See Also\n    --------\n    average : Weighted average\n    std, var, nanmean, nanstd, nanvar\n\n    Notes\n    -----\n    The arithmetic mean is the sum of the elements along the axis divided\n    by the number of elements.\n\n    Note that for floating-point input, the mean is computed using the\n    same precision the input has.  Depending on the input data, this can\n    cause the results to be inaccurate, especially for `float32` (see\n    example below).  Specifying a higher-precision accumulator using the\n    `dtype` keyword can alleviate this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> np.mean(a)\n    2.5\n    >>> np.mean(a, axis=0)\n    array([ 2.,  3.])\n    >>> np.mean(a, axis=1)\n    array([ 1.5,  3.5])\n\n    In single precision, `mean` can be inaccurate:\n\n    >>> a = np.zeros((2, 512*512), dtype=np.float32)\n    >>> a[0, :] = 1.0\n    >>> a[1, :] = 0.1\n    >>> np.mean(a)\n    0.546875\n\n    Computing the mean in float64 is more accurate:\n\n    >>> np.mean(a, dtype=np.float64)\n    0.55000000074505806\n\n    ')
    
    
    
    # Call to type(...): (line 2877)
    # Processing the call arguments (line 2877)
    # Getting the type of 'a' (line 2877)
    a_5135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2877, 12), 'a', False)
    # Processing the call keyword arguments (line 2877)
    kwargs_5136 = {}
    # Getting the type of 'type' (line 2877)
    type_5134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2877, 7), 'type', False)
    # Calling type(args, kwargs) (line 2877)
    type_call_result_5137 = invoke(stypy.reporting.localization.Localization(__file__, 2877, 7), type_5134, *[a_5135], **kwargs_5136)
    
    # Getting the type of 'mu' (line 2877)
    mu_5138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2877, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 2877)
    ndarray_5139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2877, 22), mu_5138, 'ndarray')
    # Applying the binary operator 'isnot' (line 2877)
    result_is_not_5140 = python_operator(stypy.reporting.localization.Localization(__file__, 2877, 7), 'isnot', type_call_result_5137, ndarray_5139)
    
    # Testing the type of an if condition (line 2877)
    if_condition_5141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2877, 4), result_is_not_5140)
    # Assigning a type to the variable 'if_condition_5141' (line 2877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2877, 4), 'if_condition_5141', if_condition_5141)
    # SSA begins for if statement (line 2877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 2878)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2879):
    # Getting the type of 'a' (line 2879)
    a_5142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2879, 19), 'a')
    # Obtaining the member 'mean' of a type (line 2879)
    mean_5143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2879, 19), a_5142, 'mean')
    # Assigning a type to the variable 'mean' (line 2879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2879, 12), 'mean', mean_5143)
    
    # Call to mean(...): (line 2880)
    # Processing the call keyword arguments (line 2880)
    # Getting the type of 'axis' (line 2880)
    axis_5145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2880, 29), 'axis', False)
    keyword_5146 = axis_5145
    # Getting the type of 'dtype' (line 2880)
    dtype_5147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2880, 41), 'dtype', False)
    keyword_5148 = dtype_5147
    # Getting the type of 'out' (line 2880)
    out_5149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2880, 52), 'out', False)
    keyword_5150 = out_5149
    kwargs_5151 = {'dtype': keyword_5148, 'out': keyword_5150, 'axis': keyword_5146}
    # Getting the type of 'mean' (line 2880)
    mean_5144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2880, 19), 'mean', False)
    # Calling mean(args, kwargs) (line 2880)
    mean_call_result_5152 = invoke(stypy.reporting.localization.Localization(__file__, 2880, 19), mean_5144, *[], **kwargs_5151)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2880, 12), 'stypy_return_type', mean_call_result_5152)
    # SSA branch for the except part of a try statement (line 2878)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2878)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 2878)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 2877)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _mean(...): (line 2884)
    # Processing the call arguments (line 2884)
    # Getting the type of 'a' (line 2884)
    a_5155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2884, 26), 'a', False)
    # Processing the call keyword arguments (line 2884)
    # Getting the type of 'axis' (line 2884)
    axis_5156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2884, 34), 'axis', False)
    keyword_5157 = axis_5156
    # Getting the type of 'dtype' (line 2884)
    dtype_5158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2884, 46), 'dtype', False)
    keyword_5159 = dtype_5158
    # Getting the type of 'out' (line 2885)
    out_5160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2885, 30), 'out', False)
    keyword_5161 = out_5160
    # Getting the type of 'keepdims' (line 2885)
    keepdims_5162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2885, 44), 'keepdims', False)
    keyword_5163 = keepdims_5162
    kwargs_5164 = {'dtype': keyword_5159, 'out': keyword_5161, 'keepdims': keyword_5163, 'axis': keyword_5157}
    # Getting the type of '_methods' (line 2884)
    _methods_5153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2884, 11), '_methods', False)
    # Obtaining the member '_mean' of a type (line 2884)
    _mean_5154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2884, 11), _methods_5153, '_mean')
    # Calling _mean(args, kwargs) (line 2884)
    _mean_call_result_5165 = invoke(stypy.reporting.localization.Localization(__file__, 2884, 11), _mean_5154, *[a_5155], **kwargs_5164)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2884, 4), 'stypy_return_type', _mean_call_result_5165)
    
    # ################# End of 'mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mean' in the type store
    # Getting the type of 'stypy_return_type' (line 2796)
    stypy_return_type_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2796, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5166)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mean'
    return stypy_return_type_5166

# Assigning a type to the variable 'mean' (line 2796)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2796, 0), 'mean', mean)

@norecursion
def std(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2888)
    None_5167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2888, 16), 'None')
    # Getting the type of 'None' (line 2888)
    None_5168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2888, 28), 'None')
    # Getting the type of 'None' (line 2888)
    None_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2888, 38), 'None')
    int_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2888, 49), 'int')
    # Getting the type of 'False' (line 2888)
    False_5171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2888, 61), 'False')
    defaults = [None_5167, None_5168, None_5169, int_5170, False_5171]
    # Create a new context for function 'std'
    module_type_store = module_type_store.open_function_context('std', 2888, 0, False)
    
    # Passed parameters checking function
    std.stypy_localization = localization
    std.stypy_type_of_self = None
    std.stypy_type_store = module_type_store
    std.stypy_function_name = 'std'
    std.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    std.stypy_varargs_param_name = None
    std.stypy_kwargs_param_name = None
    std.stypy_call_defaults = defaults
    std.stypy_call_varargs = varargs
    std.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'std', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'std', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'std(...)' code ##################

    str_5172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2983, (-1)), 'str', '\n    Compute the standard deviation along the specified axis.\n\n    Returns the standard deviation, a measure of the spread of a distribution,\n    of the array elements. The standard deviation is computed for the\n    flattened array by default, otherwise over the specified axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Calculate the standard deviation of these values.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which the standard deviation is computed. The\n        default is to compute the standard deviation of the flattened array.\n\n        .. versionadded: 1.7.0\n\n        If this is a tuple of ints, a standard deviation is performed over\n        multiple axes, instead of a single axis or all the axes as before.\n    dtype : dtype, optional\n        Type to use in computing the standard deviation. For arrays of\n        integer type the default is float64, for arrays of float types it is\n        the same as the array type.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape as the expected output but the type (of the calculated\n        values) will be cast if necessary.\n    ddof : int, optional\n        Means Delta Degrees of Freedom.  The divisor used in calculations\n        is ``N - ddof``, where ``N`` represents the number of elements.\n        By default `ddof` is zero.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    standard_deviation : ndarray, see dtype parameter above.\n        If `out` is None, return a new array containing the standard deviation,\n        otherwise return a reference to the output array.\n\n    See Also\n    --------\n    var, mean, nanmean, nanstd, nanvar\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    The standard deviation is the square root of the average of the squared\n    deviations from the mean, i.e., ``std = sqrt(mean(abs(x - x.mean())**2))``.\n\n    The average squared deviation is normally calculated as\n    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,\n    the divisor ``N - ddof`` is used instead. In standard statistical\n    practice, ``ddof=1`` provides an unbiased estimator of the variance\n    of the infinite population. ``ddof=0`` provides a maximum likelihood\n    estimate of the variance for normally distributed variables. The\n    standard deviation computed in this function is the square root of\n    the estimated variance, so even with ``ddof=1``, it will not be an\n    unbiased estimate of the standard deviation per se.\n\n    Note that, for complex numbers, `std` takes the absolute\n    value before squaring, so that the result is always real and nonnegative.\n\n    For floating-point input, the *std* is computed using the same\n    precision the input has. Depending on the input data, this can cause\n    the results to be inaccurate, especially for float32 (see example below).\n    Specifying a higher-accuracy accumulator using the `dtype` keyword can\n    alleviate this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> np.std(a)\n    1.1180339887498949\n    >>> np.std(a, axis=0)\n    array([ 1.,  1.])\n    >>> np.std(a, axis=1)\n    array([ 0.5,  0.5])\n\n    In single precision, std() can be inaccurate:\n\n    >>> a = np.zeros((2, 512*512), dtype=np.float32)\n    >>> a[0, :] = 1.0\n    >>> a[1, :] = 0.1\n    >>> np.std(a)\n    0.45000005\n\n    Computing the standard deviation in float64 is more accurate:\n\n    >>> np.std(a, dtype=np.float64)\n    0.44999999925494177\n\n    ')
    
    
    
    # Call to type(...): (line 2984)
    # Processing the call arguments (line 2984)
    # Getting the type of 'a' (line 2984)
    a_5174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2984, 12), 'a', False)
    # Processing the call keyword arguments (line 2984)
    kwargs_5175 = {}
    # Getting the type of 'type' (line 2984)
    type_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2984, 7), 'type', False)
    # Calling type(args, kwargs) (line 2984)
    type_call_result_5176 = invoke(stypy.reporting.localization.Localization(__file__, 2984, 7), type_5173, *[a_5174], **kwargs_5175)
    
    # Getting the type of 'mu' (line 2984)
    mu_5177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2984, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 2984)
    ndarray_5178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2984, 22), mu_5177, 'ndarray')
    # Applying the binary operator 'isnot' (line 2984)
    result_is_not_5179 = python_operator(stypy.reporting.localization.Localization(__file__, 2984, 7), 'isnot', type_call_result_5176, ndarray_5178)
    
    # Testing the type of an if condition (line 2984)
    if_condition_5180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2984, 4), result_is_not_5179)
    # Assigning a type to the variable 'if_condition_5180' (line 2984)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2984, 4), 'if_condition_5180', if_condition_5180)
    # SSA begins for if statement (line 2984)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 2985)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 2986):
    # Getting the type of 'a' (line 2986)
    a_5181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2986, 18), 'a')
    # Obtaining the member 'std' of a type (line 2986)
    std_5182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2986, 18), a_5181, 'std')
    # Assigning a type to the variable 'std' (line 2986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2986, 12), 'std', std_5182)
    
    # Call to std(...): (line 2987)
    # Processing the call keyword arguments (line 2987)
    # Getting the type of 'axis' (line 2987)
    axis_5184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2987, 28), 'axis', False)
    keyword_5185 = axis_5184
    # Getting the type of 'dtype' (line 2987)
    dtype_5186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2987, 40), 'dtype', False)
    keyword_5187 = dtype_5186
    # Getting the type of 'out' (line 2987)
    out_5188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2987, 51), 'out', False)
    keyword_5189 = out_5188
    # Getting the type of 'ddof' (line 2987)
    ddof_5190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2987, 61), 'ddof', False)
    keyword_5191 = ddof_5190
    kwargs_5192 = {'dtype': keyword_5187, 'out': keyword_5189, 'ddof': keyword_5191, 'axis': keyword_5185}
    # Getting the type of 'std' (line 2987)
    std_5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2987, 19), 'std', False)
    # Calling std(args, kwargs) (line 2987)
    std_call_result_5193 = invoke(stypy.reporting.localization.Localization(__file__, 2987, 19), std_5183, *[], **kwargs_5192)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2987)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2987, 12), 'stypy_return_type', std_call_result_5193)
    # SSA branch for the except part of a try statement (line 2985)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 2985)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 2985)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 2984)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _std(...): (line 2991)
    # Processing the call arguments (line 2991)
    # Getting the type of 'a' (line 2991)
    a_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 25), 'a', False)
    # Processing the call keyword arguments (line 2991)
    # Getting the type of 'axis' (line 2991)
    axis_5197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 33), 'axis', False)
    keyword_5198 = axis_5197
    # Getting the type of 'dtype' (line 2991)
    dtype_5199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 45), 'dtype', False)
    keyword_5200 = dtype_5199
    # Getting the type of 'out' (line 2991)
    out_5201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 56), 'out', False)
    keyword_5202 = out_5201
    # Getting the type of 'ddof' (line 2991)
    ddof_5203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 66), 'ddof', False)
    keyword_5204 = ddof_5203
    # Getting the type of 'keepdims' (line 2992)
    keepdims_5205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2992, 34), 'keepdims', False)
    keyword_5206 = keepdims_5205
    kwargs_5207 = {'dtype': keyword_5200, 'out': keyword_5202, 'ddof': keyword_5204, 'keepdims': keyword_5206, 'axis': keyword_5198}
    # Getting the type of '_methods' (line 2991)
    _methods_5194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2991, 11), '_methods', False)
    # Obtaining the member '_std' of a type (line 2991)
    _std_5195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2991, 11), _methods_5194, '_std')
    # Calling _std(args, kwargs) (line 2991)
    _std_call_result_5208 = invoke(stypy.reporting.localization.Localization(__file__, 2991, 11), _std_5195, *[a_5196], **kwargs_5207)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2991)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2991, 4), 'stypy_return_type', _std_call_result_5208)
    
    # ################# End of 'std(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'std' in the type store
    # Getting the type of 'stypy_return_type' (line 2888)
    stypy_return_type_5209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2888, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5209)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'std'
    return stypy_return_type_5209

# Assigning a type to the variable 'std' (line 2888)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2888, 0), 'std', std)

@norecursion
def var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 2995)
    None_5210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2995, 16), 'None')
    # Getting the type of 'None' (line 2995)
    None_5211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2995, 28), 'None')
    # Getting the type of 'None' (line 2995)
    None_5212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2995, 38), 'None')
    int_5213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2995, 49), 'int')
    # Getting the type of 'False' (line 2996)
    False_5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2996, 17), 'False')
    defaults = [None_5210, None_5211, None_5212, int_5213, False_5214]
    # Create a new context for function 'var'
    module_type_store = module_type_store.open_function_context('var', 2995, 0, False)
    
    # Passed parameters checking function
    var.stypy_localization = localization
    var.stypy_type_of_self = None
    var.stypy_type_store = module_type_store
    var.stypy_function_name = 'var'
    var.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    var.stypy_varargs_param_name = None
    var.stypy_kwargs_param_name = None
    var.stypy_call_defaults = defaults
    var.stypy_call_varargs = varargs
    var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'var', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'var', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'var(...)' code ##################

    str_5215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3091, (-1)), 'str', '\n    Compute the variance along the specified axis.\n\n    Returns the variance of the array elements, a measure of the spread of a\n    distribution.  The variance is computed for the flattened array by\n    default, otherwise over the specified axis.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose variance is desired.  If `a` is not an\n        array, a conversion is attempted.\n    axis : None or int or tuple of ints, optional\n        Axis or axes along which the variance is computed.  The default is to\n        compute the variance of the flattened array.\n\n        .. versionadded: 1.7.0\n\n        If this is a tuple of ints, a variance is performed over multiple axes,\n        instead of a single axis or all the axes as before.\n    dtype : data-type, optional\n        Type to use in computing the variance.  For arrays of integer type\n        the default is `float32`; for arrays of float types it is the same as\n        the array type.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  It must have\n        the same shape as the expected output, but the type is cast if\n        necessary.\n    ddof : int, optional\n        "Delta Degrees of Freedom": the divisor used in the calculation is\n        ``N - ddof``, where ``N`` represents the number of elements. By\n        default `ddof` is zero.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    variance : ndarray, see dtype parameter above\n        If ``out=None``, returns a new array containing the variance;\n        otherwise, a reference to the output array is returned.\n\n    See Also\n    --------\n    std , mean, nanmean, nanstd, nanvar\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    The variance is the average of the squared deviations from the mean,\n    i.e.,  ``var = mean(abs(x - x.mean())**2)``.\n\n    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.\n    If, however, `ddof` is specified, the divisor ``N - ddof`` is used\n    instead.  In standard statistical practice, ``ddof=1`` provides an\n    unbiased estimator of the variance of a hypothetical infinite population.\n    ``ddof=0`` provides a maximum likelihood estimate of the variance for\n    normally distributed variables.\n\n    Note that for complex numbers, the absolute value is taken before\n    squaring, so that the result is always real and nonnegative.\n\n    For floating-point input, the variance is computed using the same\n    precision the input has.  Depending on the input data, this can cause\n    the results to be inaccurate, especially for `float32` (see example\n    below).  Specifying a higher-accuracy accumulator using the ``dtype``\n    keyword can alleviate this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, 4]])\n    >>> np.var(a)\n    1.25\n    >>> np.var(a, axis=0)\n    array([ 1.,  1.])\n    >>> np.var(a, axis=1)\n    array([ 0.25,  0.25])\n\n    In single precision, var() can be inaccurate:\n\n    >>> a = np.zeros((2, 512*512), dtype=np.float32)\n    >>> a[0, :] = 1.0\n    >>> a[1, :] = 0.1\n    >>> np.var(a)\n    0.20250003\n\n    Computing the variance in float64 is more accurate:\n\n    >>> np.var(a, dtype=np.float64)\n    0.20249999932944759\n    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2\n    0.2025\n\n    ')
    
    
    
    # Call to type(...): (line 3092)
    # Processing the call arguments (line 3092)
    # Getting the type of 'a' (line 3092)
    a_5217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3092, 12), 'a', False)
    # Processing the call keyword arguments (line 3092)
    kwargs_5218 = {}
    # Getting the type of 'type' (line 3092)
    type_5216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3092, 7), 'type', False)
    # Calling type(args, kwargs) (line 3092)
    type_call_result_5219 = invoke(stypy.reporting.localization.Localization(__file__, 3092, 7), type_5216, *[a_5217], **kwargs_5218)
    
    # Getting the type of 'mu' (line 3092)
    mu_5220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3092, 22), 'mu')
    # Obtaining the member 'ndarray' of a type (line 3092)
    ndarray_5221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3092, 22), mu_5220, 'ndarray')
    # Applying the binary operator 'isnot' (line 3092)
    result_is_not_5222 = python_operator(stypy.reporting.localization.Localization(__file__, 3092, 7), 'isnot', type_call_result_5219, ndarray_5221)
    
    # Testing the type of an if condition (line 3092)
    if_condition_5223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3092, 4), result_is_not_5222)
    # Assigning a type to the variable 'if_condition_5223' (line 3092)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3092, 4), 'if_condition_5223', if_condition_5223)
    # SSA begins for if statement (line 3092)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 3093)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 3094):
    # Getting the type of 'a' (line 3094)
    a_5224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3094, 18), 'a')
    # Obtaining the member 'var' of a type (line 3094)
    var_5225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3094, 18), a_5224, 'var')
    # Assigning a type to the variable 'var' (line 3094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3094, 12), 'var', var_5225)
    
    # Call to var(...): (line 3095)
    # Processing the call keyword arguments (line 3095)
    # Getting the type of 'axis' (line 3095)
    axis_5227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 28), 'axis', False)
    keyword_5228 = axis_5227
    # Getting the type of 'dtype' (line 3095)
    dtype_5229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 40), 'dtype', False)
    keyword_5230 = dtype_5229
    # Getting the type of 'out' (line 3095)
    out_5231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 51), 'out', False)
    keyword_5232 = out_5231
    # Getting the type of 'ddof' (line 3095)
    ddof_5233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 61), 'ddof', False)
    keyword_5234 = ddof_5233
    kwargs_5235 = {'dtype': keyword_5230, 'out': keyword_5232, 'ddof': keyword_5234, 'axis': keyword_5228}
    # Getting the type of 'var' (line 3095)
    var_5226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3095, 19), 'var', False)
    # Calling var(args, kwargs) (line 3095)
    var_call_result_5236 = invoke(stypy.reporting.localization.Localization(__file__, 3095, 19), var_5226, *[], **kwargs_5235)
    
    # Assigning a type to the variable 'stypy_return_type' (line 3095)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3095, 12), 'stypy_return_type', var_call_result_5236)
    # SSA branch for the except part of a try statement (line 3093)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 3093)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 3093)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 3092)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _var(...): (line 3099)
    # Processing the call arguments (line 3099)
    # Getting the type of 'a' (line 3099)
    a_5239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 25), 'a', False)
    # Processing the call keyword arguments (line 3099)
    # Getting the type of 'axis' (line 3099)
    axis_5240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 33), 'axis', False)
    keyword_5241 = axis_5240
    # Getting the type of 'dtype' (line 3099)
    dtype_5242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 45), 'dtype', False)
    keyword_5243 = dtype_5242
    # Getting the type of 'out' (line 3099)
    out_5244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 56), 'out', False)
    keyword_5245 = out_5244
    # Getting the type of 'ddof' (line 3099)
    ddof_5246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 66), 'ddof', False)
    keyword_5247 = ddof_5246
    # Getting the type of 'keepdims' (line 3100)
    keepdims_5248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3100, 34), 'keepdims', False)
    keyword_5249 = keepdims_5248
    kwargs_5250 = {'dtype': keyword_5243, 'out': keyword_5245, 'ddof': keyword_5247, 'keepdims': keyword_5249, 'axis': keyword_5241}
    # Getting the type of '_methods' (line 3099)
    _methods_5237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3099, 11), '_methods', False)
    # Obtaining the member '_var' of a type (line 3099)
    _var_5238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3099, 11), _methods_5237, '_var')
    # Calling _var(args, kwargs) (line 3099)
    _var_call_result_5251 = invoke(stypy.reporting.localization.Localization(__file__, 3099, 11), _var_5238, *[a_5239], **kwargs_5250)
    
    # Assigning a type to the variable 'stypy_return_type' (line 3099)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3099, 4), 'stypy_return_type', _var_call_result_5251)
    
    # ################# End of 'var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'var' in the type store
    # Getting the type of 'stypy_return_type' (line 2995)
    stypy_return_type_5252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2995, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5252)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'var'
    return stypy_return_type_5252

# Assigning a type to the variable 'var' (line 2995)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2995, 0), 'var', var)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
