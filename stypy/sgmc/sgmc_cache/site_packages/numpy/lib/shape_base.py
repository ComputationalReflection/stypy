
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import warnings
4: 
5: import numpy.core.numeric as _nx
6: from numpy.core.numeric import (
7:     asarray, zeros, outer, concatenate, isscalar, array, asanyarray
8:     )
9: from numpy.core.fromnumeric import product, reshape
10: from numpy.core import vstack, atleast_3d
11: 
12: 
13: __all__ = [
14:     'column_stack', 'row_stack', 'dstack', 'array_split', 'split',
15:     'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims',
16:     'apply_along_axis', 'kron', 'tile', 'get_array_wrap'
17:     ]
18: 
19: 
20: def apply_along_axis(func1d, axis, arr, *args, **kwargs):
21:     '''
22:     Apply a function to 1-D slices along the given axis.
23: 
24:     Execute `func1d(a, *args)` where `func1d` operates on 1-D arrays and `a`
25:     is a 1-D slice of `arr` along `axis`.
26: 
27:     Parameters
28:     ----------
29:     func1d : function
30:         This function should accept 1-D arrays. It is applied to 1-D
31:         slices of `arr` along the specified axis.
32:     axis : integer
33:         Axis along which `arr` is sliced.
34:     arr : ndarray
35:         Input array.
36:     args : any
37:         Additional arguments to `func1d`.
38:     kwargs: any
39:         Additional named arguments to `func1d`.
40: 
41:         .. versionadded:: 1.9.0
42: 
43: 
44:     Returns
45:     -------
46:     apply_along_axis : ndarray
47:         The output array. The shape of `outarr` is identical to the shape of
48:         `arr`, except along the `axis` dimension, where the length of `outarr`
49:         is equal to the size of the return value of `func1d`.  If `func1d`
50:         returns a scalar `outarr` will have one fewer dimensions than `arr`.
51: 
52:     See Also
53:     --------
54:     apply_over_axes : Apply a function repeatedly over multiple axes.
55: 
56:     Examples
57:     --------
58:     >>> def my_func(a):
59:     ...     \"\"\"Average first and last element of a 1-D array\"\"\"
60:     ...     return (a[0] + a[-1]) * 0.5
61:     >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
62:     >>> np.apply_along_axis(my_func, 0, b)
63:     array([ 4.,  5.,  6.])
64:     >>> np.apply_along_axis(my_func, 1, b)
65:     array([ 2.,  5.,  8.])
66: 
67:     For a function that doesn't return a scalar, the number of dimensions in
68:     `outarr` is the same as `arr`.
69: 
70:     >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])
71:     >>> np.apply_along_axis(sorted, 1, b)
72:     array([[1, 7, 8],
73:            [3, 4, 9],
74:            [2, 5, 6]])
75: 
76:     '''
77:     arr = asarray(arr)
78:     nd = arr.ndim
79:     if axis < 0:
80:         axis += nd
81:     if (axis >= nd):
82:         raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."
83:             % (axis, nd))
84:     ind = [0]*(nd-1)
85:     i = zeros(nd, 'O')
86:     indlist = list(range(nd))
87:     indlist.remove(axis)
88:     i[axis] = slice(None, None)
89:     outshape = asarray(arr.shape).take(indlist)
90:     i.put(indlist, ind)
91:     res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
92:     #  if res is a number, then we have a smaller output array
93:     if isscalar(res):
94:         outarr = zeros(outshape, asarray(res).dtype)
95:         outarr[tuple(ind)] = res
96:         Ntot = product(outshape)
97:         k = 1
98:         while k < Ntot:
99:             # increment the index
100:             ind[-1] += 1
101:             n = -1
102:             while (ind[n] >= outshape[n]) and (n > (1-nd)):
103:                 ind[n-1] += 1
104:                 ind[n] = 0
105:                 n -= 1
106:             i.put(indlist, ind)
107:             res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
108:             outarr[tuple(ind)] = res
109:             k += 1
110:         return outarr
111:     else:
112:         Ntot = product(outshape)
113:         holdshape = outshape
114:         outshape = list(arr.shape)
115:         outshape[axis] = len(res)
116:         outarr = zeros(outshape, asarray(res).dtype)
117:         outarr[tuple(i.tolist())] = res
118:         k = 1
119:         while k < Ntot:
120:             # increment the index
121:             ind[-1] += 1
122:             n = -1
123:             while (ind[n] >= holdshape[n]) and (n > (1-nd)):
124:                 ind[n-1] += 1
125:                 ind[n] = 0
126:                 n -= 1
127:             i.put(indlist, ind)
128:             res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
129:             outarr[tuple(i.tolist())] = res
130:             k += 1
131:         return outarr
132: 
133: 
134: def apply_over_axes(func, a, axes):
135:     '''
136:     Apply a function repeatedly over multiple axes.
137: 
138:     `func` is called as `res = func(a, axis)`, where `axis` is the first
139:     element of `axes`.  The result `res` of the function call must have
140:     either the same dimensions as `a` or one less dimension.  If `res`
141:     has one less dimension than `a`, a dimension is inserted before
142:     `axis`.  The call to `func` is then repeated for each axis in `axes`,
143:     with `res` as the first argument.
144: 
145:     Parameters
146:     ----------
147:     func : function
148:         This function must take two arguments, `func(a, axis)`.
149:     a : array_like
150:         Input array.
151:     axes : array_like
152:         Axes over which `func` is applied; the elements must be integers.
153: 
154:     Returns
155:     -------
156:     apply_over_axis : ndarray
157:         The output array.  The number of dimensions is the same as `a`,
158:         but the shape can be different.  This depends on whether `func`
159:         changes the shape of its output with respect to its input.
160: 
161:     See Also
162:     --------
163:     apply_along_axis :
164:         Apply a function to 1-D slices of an array along the given axis.
165: 
166:     Notes
167:     ------
168:     This function is equivalent to tuple axis arguments to reorderable ufuncs
169:     with keepdims=True. Tuple axis arguments to ufuncs have been availabe since
170:     version 1.7.0.
171: 
172:     Examples
173:     --------
174:     >>> a = np.arange(24).reshape(2,3,4)
175:     >>> a
176:     array([[[ 0,  1,  2,  3],
177:             [ 4,  5,  6,  7],
178:             [ 8,  9, 10, 11]],
179:            [[12, 13, 14, 15],
180:             [16, 17, 18, 19],
181:             [20, 21, 22, 23]]])
182: 
183:     Sum over axes 0 and 2. The result has same number of dimensions
184:     as the original array:
185: 
186:     >>> np.apply_over_axes(np.sum, a, [0,2])
187:     array([[[ 60],
188:             [ 92],
189:             [124]]])
190: 
191:     Tuple axis arguments to ufuncs are equivalent:
192: 
193:     >>> np.sum(a, axis=(0,2), keepdims=True)
194:     array([[[ 60],
195:             [ 92],
196:             [124]]])
197: 
198:     '''
199:     val = asarray(a)
200:     N = a.ndim
201:     if array(axes).ndim == 0:
202:         axes = (axes,)
203:     for axis in axes:
204:         if axis < 0:
205:             axis = N + axis
206:         args = (val, axis)
207:         res = func(*args)
208:         if res.ndim == val.ndim:
209:             val = res
210:         else:
211:             res = expand_dims(res, axis)
212:             if res.ndim == val.ndim:
213:                 val = res
214:             else:
215:                 raise ValueError("function is not returning "
216:                         "an array of the correct shape")
217:     return val
218: 
219: def expand_dims(a, axis):
220:     '''
221:     Expand the shape of an array.
222: 
223:     Insert a new axis, corresponding to a given position in the array shape.
224: 
225:     Parameters
226:     ----------
227:     a : array_like
228:         Input array.
229:     axis : int
230:         Position (amongst axes) where new axis is to be inserted.
231: 
232:     Returns
233:     -------
234:     res : ndarray
235:         Output array. The number of dimensions is one greater than that of
236:         the input array.
237: 
238:     See Also
239:     --------
240:     doc.indexing, atleast_1d, atleast_2d, atleast_3d
241: 
242:     Examples
243:     --------
244:     >>> x = np.array([1,2])
245:     >>> x.shape
246:     (2,)
247: 
248:     The following is equivalent to ``x[np.newaxis,:]`` or ``x[np.newaxis]``:
249: 
250:     >>> y = np.expand_dims(x, axis=0)
251:     >>> y
252:     array([[1, 2]])
253:     >>> y.shape
254:     (1, 2)
255: 
256:     >>> y = np.expand_dims(x, axis=1)  # Equivalent to x[:,newaxis]
257:     >>> y
258:     array([[1],
259:            [2]])
260:     >>> y.shape
261:     (2, 1)
262: 
263:     Note that some examples may use ``None`` instead of ``np.newaxis``.  These
264:     are the same objects:
265: 
266:     >>> np.newaxis is None
267:     True
268: 
269:     '''
270:     a = asarray(a)
271:     shape = a.shape
272:     if axis < 0:
273:         axis = axis + len(shape) + 1
274:     return a.reshape(shape[:axis] + (1,) + shape[axis:])
275: 
276: row_stack = vstack
277: 
278: def column_stack(tup):
279:     '''
280:     Stack 1-D arrays as columns into a 2-D array.
281: 
282:     Take a sequence of 1-D arrays and stack them as columns
283:     to make a single 2-D array. 2-D arrays are stacked as-is,
284:     just like with `hstack`.  1-D arrays are turned into 2-D columns
285:     first.
286: 
287:     Parameters
288:     ----------
289:     tup : sequence of 1-D or 2-D arrays.
290:         Arrays to stack. All of them must have the same first dimension.
291: 
292:     Returns
293:     -------
294:     stacked : 2-D array
295:         The array formed by stacking the given arrays.
296: 
297:     See Also
298:     --------
299:     hstack, vstack, concatenate
300: 
301:     Examples
302:     --------
303:     >>> a = np.array((1,2,3))
304:     >>> b = np.array((2,3,4))
305:     >>> np.column_stack((a,b))
306:     array([[1, 2],
307:            [2, 3],
308:            [3, 4]])
309: 
310:     '''
311:     arrays = []
312:     for v in tup:
313:         arr = array(v, copy=False, subok=True)
314:         if arr.ndim < 2:
315:             arr = array(arr, copy=False, subok=True, ndmin=2).T
316:         arrays.append(arr)
317:     return _nx.concatenate(arrays, 1)
318: 
319: def dstack(tup):
320:     '''
321:     Stack arrays in sequence depth wise (along third axis).
322: 
323:     Takes a sequence of arrays and stack them along the third axis
324:     to make a single array. Rebuilds arrays divided by `dsplit`.
325:     This is a simple way to stack 2D arrays (images) into a single
326:     3D array for processing.
327: 
328:     Parameters
329:     ----------
330:     tup : sequence of arrays
331:         Arrays to stack. All of them must have the same shape along all
332:         but the third axis.
333: 
334:     Returns
335:     -------
336:     stacked : ndarray
337:         The array formed by stacking the given arrays.
338: 
339:     See Also
340:     --------
341:     stack : Join a sequence of arrays along a new axis.
342:     vstack : Stack along first axis.
343:     hstack : Stack along second axis.
344:     concatenate : Join a sequence of arrays along an existing axis.
345:     dsplit : Split array along third axis.
346: 
347:     Notes
348:     -----
349:     Equivalent to ``np.concatenate(tup, axis=2)``.
350: 
351:     Examples
352:     --------
353:     >>> a = np.array((1,2,3))
354:     >>> b = np.array((2,3,4))
355:     >>> np.dstack((a,b))
356:     array([[[1, 2],
357:             [2, 3],
358:             [3, 4]]])
359: 
360:     >>> a = np.array([[1],[2],[3]])
361:     >>> b = np.array([[2],[3],[4]])
362:     >>> np.dstack((a,b))
363:     array([[[1, 2]],
364:            [[2, 3]],
365:            [[3, 4]]])
366: 
367:     '''
368:     return _nx.concatenate([atleast_3d(_m) for _m in tup], 2)
369: 
370: def _replace_zero_by_x_arrays(sub_arys):
371:     for i in range(len(sub_arys)):
372:         if len(_nx.shape(sub_arys[i])) == 0:
373:             sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
374:         elif _nx.sometrue(_nx.equal(_nx.shape(sub_arys[i]), 0)):
375:             sub_arys[i] = _nx.empty(0, dtype=sub_arys[i].dtype)
376:     return sub_arys
377: 
378: def array_split(ary, indices_or_sections, axis=0):
379:     '''
380:     Split an array into multiple sub-arrays.
381: 
382:     Please refer to the ``split`` documentation.  The only difference
383:     between these functions is that ``array_split`` allows
384:     `indices_or_sections` to be an integer that does *not* equally
385:     divide the axis.
386: 
387:     See Also
388:     --------
389:     split : Split array into multiple sub-arrays of equal size.
390: 
391:     Examples
392:     --------
393:     >>> x = np.arange(8.0)
394:     >>> np.array_split(x, 3)
395:         [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]
396: 
397:     '''
398:     try:
399:         Ntotal = ary.shape[axis]
400:     except AttributeError:
401:         Ntotal = len(ary)
402:     try:
403:         # handle scalar case.
404:         Nsections = len(indices_or_sections) + 1
405:         div_points = [0] + list(indices_or_sections) + [Ntotal]
406:     except TypeError:
407:         # indices_or_sections is a scalar, not an array.
408:         Nsections = int(indices_or_sections)
409:         if Nsections <= 0:
410:             raise ValueError('number sections must be larger than 0.')
411:         Neach_section, extras = divmod(Ntotal, Nsections)
412:         section_sizes = ([0] +
413:                          extras * [Neach_section+1] +
414:                          (Nsections-extras) * [Neach_section])
415:         div_points = _nx.array(section_sizes).cumsum()
416: 
417:     sub_arys = []
418:     sary = _nx.swapaxes(ary, axis, 0)
419:     for i in range(Nsections):
420:         st = div_points[i]
421:         end = div_points[i + 1]
422:         sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
423: 
424:     return sub_arys
425: 
426: 
427: def split(ary,indices_or_sections,axis=0):
428:     '''
429:     Split an array into multiple sub-arrays.
430: 
431:     Parameters
432:     ----------
433:     ary : ndarray
434:         Array to be divided into sub-arrays.
435:     indices_or_sections : int or 1-D array
436:         If `indices_or_sections` is an integer, N, the array will be divided
437:         into N equal arrays along `axis`.  If such a split is not possible,
438:         an error is raised.
439: 
440:         If `indices_or_sections` is a 1-D array of sorted integers, the entries
441:         indicate where along `axis` the array is split.  For example,
442:         ``[2, 3]`` would, for ``axis=0``, result in
443: 
444:           - ary[:2]
445:           - ary[2:3]
446:           - ary[3:]
447: 
448:         If an index exceeds the dimension of the array along `axis`,
449:         an empty sub-array is returned correspondingly.
450:     axis : int, optional
451:         The axis along which to split, default is 0.
452: 
453:     Returns
454:     -------
455:     sub-arrays : list of ndarrays
456:         A list of sub-arrays.
457: 
458:     Raises
459:     ------
460:     ValueError
461:         If `indices_or_sections` is given as an integer, but
462:         a split does not result in equal division.
463: 
464:     See Also
465:     --------
466:     array_split : Split an array into multiple sub-arrays of equal or
467:                   near-equal size.  Does not raise an exception if
468:                   an equal division cannot be made.
469:     hsplit : Split array into multiple sub-arrays horizontally (column-wise).
470:     vsplit : Split array into multiple sub-arrays vertically (row wise).
471:     dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
472:     concatenate : Join a sequence of arrays along an existing axis.
473:     stack : Join a sequence of arrays along a new axis.
474:     hstack : Stack arrays in sequence horizontally (column wise).
475:     vstack : Stack arrays in sequence vertically (row wise).
476:     dstack : Stack arrays in sequence depth wise (along third dimension).
477: 
478:     Examples
479:     --------
480:     >>> x = np.arange(9.0)
481:     >>> np.split(x, 3)
482:     [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]
483: 
484:     >>> x = np.arange(8.0)
485:     >>> np.split(x, [3, 5, 6, 10])
486:     [array([ 0.,  1.,  2.]),
487:      array([ 3.,  4.]),
488:      array([ 5.]),
489:      array([ 6.,  7.]),
490:      array([], dtype=float64)]
491: 
492:     '''
493:     try:
494:         len(indices_or_sections)
495:     except TypeError:
496:         sections = indices_or_sections
497:         N = ary.shape[axis]
498:         if N % sections:
499:             raise ValueError(
500:                 'array split does not result in an equal division')
501:     res = array_split(ary, indices_or_sections, axis)
502:     return res
503: 
504: def hsplit(ary, indices_or_sections):
505:     '''
506:     Split an array into multiple sub-arrays horizontally (column-wise).
507: 
508:     Please refer to the `split` documentation.  `hsplit` is equivalent
509:     to `split` with ``axis=1``, the array is always split along the second
510:     axis regardless of the array dimension.
511: 
512:     See Also
513:     --------
514:     split : Split an array into multiple sub-arrays of equal size.
515: 
516:     Examples
517:     --------
518:     >>> x = np.arange(16.0).reshape(4, 4)
519:     >>> x
520:     array([[  0.,   1.,   2.,   3.],
521:            [  4.,   5.,   6.,   7.],
522:            [  8.,   9.,  10.,  11.],
523:            [ 12.,  13.,  14.,  15.]])
524:     >>> np.hsplit(x, 2)
525:     [array([[  0.,   1.],
526:            [  4.,   5.],
527:            [  8.,   9.],
528:            [ 12.,  13.]]),
529:      array([[  2.,   3.],
530:            [  6.,   7.],
531:            [ 10.,  11.],
532:            [ 14.,  15.]])]
533:     >>> np.hsplit(x, np.array([3, 6]))
534:     [array([[  0.,   1.,   2.],
535:            [  4.,   5.,   6.],
536:            [  8.,   9.,  10.],
537:            [ 12.,  13.,  14.]]),
538:      array([[  3.],
539:            [  7.],
540:            [ 11.],
541:            [ 15.]]),
542:      array([], dtype=float64)]
543: 
544:     With a higher dimensional array the split is still along the second axis.
545: 
546:     >>> x = np.arange(8.0).reshape(2, 2, 2)
547:     >>> x
548:     array([[[ 0.,  1.],
549:             [ 2.,  3.]],
550:            [[ 4.,  5.],
551:             [ 6.,  7.]]])
552:     >>> np.hsplit(x, 2)
553:     [array([[[ 0.,  1.]],
554:            [[ 4.,  5.]]]),
555:      array([[[ 2.,  3.]],
556:            [[ 6.,  7.]]])]
557: 
558:     '''
559:     if len(_nx.shape(ary)) == 0:
560:         raise ValueError('hsplit only works on arrays of 1 or more dimensions')
561:     if len(ary.shape) > 1:
562:         return split(ary, indices_or_sections, 1)
563:     else:
564:         return split(ary, indices_or_sections, 0)
565: 
566: def vsplit(ary, indices_or_sections):
567:     '''
568:     Split an array into multiple sub-arrays vertically (row-wise).
569: 
570:     Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
571:     to ``split`` with `axis=0` (default), the array is always split along the
572:     first axis regardless of the array dimension.
573: 
574:     See Also
575:     --------
576:     split : Split an array into multiple sub-arrays of equal size.
577: 
578:     Examples
579:     --------
580:     >>> x = np.arange(16.0).reshape(4, 4)
581:     >>> x
582:     array([[  0.,   1.,   2.,   3.],
583:            [  4.,   5.,   6.,   7.],
584:            [  8.,   9.,  10.,  11.],
585:            [ 12.,  13.,  14.,  15.]])
586:     >>> np.vsplit(x, 2)
587:     [array([[ 0.,  1.,  2.,  3.],
588:            [ 4.,  5.,  6.,  7.]]),
589:      array([[  8.,   9.,  10.,  11.],
590:            [ 12.,  13.,  14.,  15.]])]
591:     >>> np.vsplit(x, np.array([3, 6]))
592:     [array([[  0.,   1.,   2.,   3.],
593:            [  4.,   5.,   6.,   7.],
594:            [  8.,   9.,  10.,  11.]]),
595:      array([[ 12.,  13.,  14.,  15.]]),
596:      array([], dtype=float64)]
597: 
598:     With a higher dimensional array the split is still along the first axis.
599: 
600:     >>> x = np.arange(8.0).reshape(2, 2, 2)
601:     >>> x
602:     array([[[ 0.,  1.],
603:             [ 2.,  3.]],
604:            [[ 4.,  5.],
605:             [ 6.,  7.]]])
606:     >>> np.vsplit(x, 2)
607:     [array([[[ 0.,  1.],
608:             [ 2.,  3.]]]),
609:      array([[[ 4.,  5.],
610:             [ 6.,  7.]]])]
611: 
612:     '''
613:     if len(_nx.shape(ary)) < 2:
614:         raise ValueError('vsplit only works on arrays of 2 or more dimensions')
615:     return split(ary, indices_or_sections, 0)
616: 
617: def dsplit(ary, indices_or_sections):
618:     '''
619:     Split array into multiple sub-arrays along the 3rd axis (depth).
620: 
621:     Please refer to the `split` documentation.  `dsplit` is equivalent
622:     to `split` with ``axis=2``, the array is always split along the third
623:     axis provided the array dimension is greater than or equal to 3.
624: 
625:     See Also
626:     --------
627:     split : Split an array into multiple sub-arrays of equal size.
628: 
629:     Examples
630:     --------
631:     >>> x = np.arange(16.0).reshape(2, 2, 4)
632:     >>> x
633:     array([[[  0.,   1.,   2.,   3.],
634:             [  4.,   5.,   6.,   7.]],
635:            [[  8.,   9.,  10.,  11.],
636:             [ 12.,  13.,  14.,  15.]]])
637:     >>> np.dsplit(x, 2)
638:     [array([[[  0.,   1.],
639:             [  4.,   5.]],
640:            [[  8.,   9.],
641:             [ 12.,  13.]]]),
642:      array([[[  2.,   3.],
643:             [  6.,   7.]],
644:            [[ 10.,  11.],
645:             [ 14.,  15.]]])]
646:     >>> np.dsplit(x, np.array([3, 6]))
647:     [array([[[  0.,   1.,   2.],
648:             [  4.,   5.,   6.]],
649:            [[  8.,   9.,  10.],
650:             [ 12.,  13.,  14.]]]),
651:      array([[[  3.],
652:             [  7.]],
653:            [[ 11.],
654:             [ 15.]]]),
655:      array([], dtype=float64)]
656: 
657:     '''
658:     if len(_nx.shape(ary)) < 3:
659:         raise ValueError('dsplit only works on arrays of 3 or more dimensions')
660:     return split(ary, indices_or_sections, 2)
661: 
662: def get_array_prepare(*args):
663:     '''Find the wrapper for the array with the highest priority.
664: 
665:     In case of ties, leftmost wins. If no wrapper is found, return None
666:     '''
667:     wrappers = sorted((getattr(x, '__array_priority__', 0), -i,
668:                  x.__array_prepare__) for i, x in enumerate(args)
669:                                    if hasattr(x, '__array_prepare__'))
670:     if wrappers:
671:         return wrappers[-1][-1]
672:     return None
673: 
674: def get_array_wrap(*args):
675:     '''Find the wrapper for the array with the highest priority.
676: 
677:     In case of ties, leftmost wins. If no wrapper is found, return None
678:     '''
679:     wrappers = sorted((getattr(x, '__array_priority__', 0), -i,
680:                  x.__array_wrap__) for i, x in enumerate(args)
681:                                    if hasattr(x, '__array_wrap__'))
682:     if wrappers:
683:         return wrappers[-1][-1]
684:     return None
685: 
686: def kron(a, b):
687:     '''
688:     Kronecker product of two arrays.
689: 
690:     Computes the Kronecker product, a composite array made of blocks of the
691:     second array scaled by the first.
692: 
693:     Parameters
694:     ----------
695:     a, b : array_like
696: 
697:     Returns
698:     -------
699:     out : ndarray
700: 
701:     See Also
702:     --------
703:     outer : The outer product
704: 
705:     Notes
706:     -----
707:     The function assumes that the number of dimensions of `a` and `b`
708:     are the same, if necessary prepending the smallest with ones.
709:     If `a.shape = (r0,r1,..,rN)` and `b.shape = (s0,s1,...,sN)`,
710:     the Kronecker product has shape `(r0*s0, r1*s1, ..., rN*SN)`.
711:     The elements are products of elements from `a` and `b`, organized
712:     explicitly by::
713: 
714:         kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]
715: 
716:     where::
717: 
718:         kt = it * st + jt,  t = 0,...,N
719: 
720:     In the common 2-D case (N=1), the block structure can be visualized::
721: 
722:         [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
723:          [  ...                              ...   ],
724:          [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]
725: 
726: 
727:     Examples
728:     --------
729:     >>> np.kron([1,10,100], [5,6,7])
730:     array([  5,   6,   7,  50,  60,  70, 500, 600, 700])
731:     >>> np.kron([5,6,7], [1,10,100])
732:     array([  5,  50, 500,   6,  60, 600,   7,  70, 700])
733: 
734:     >>> np.kron(np.eye(2), np.ones((2,2)))
735:     array([[ 1.,  1.,  0.,  0.],
736:            [ 1.,  1.,  0.,  0.],
737:            [ 0.,  0.,  1.,  1.],
738:            [ 0.,  0.,  1.,  1.]])
739: 
740:     >>> a = np.arange(100).reshape((2,5,2,5))
741:     >>> b = np.arange(24).reshape((2,3,4))
742:     >>> c = np.kron(a,b)
743:     >>> c.shape
744:     (2, 10, 6, 20)
745:     >>> I = (1,3,0,2)
746:     >>> J = (0,2,1)
747:     >>> J1 = (0,) + J             # extend to ndim=4
748:     >>> S1 = (1,) + b.shape
749:     >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))
750:     >>> c[K] == a[I]*b[J]
751:     True
752: 
753:     '''
754:     b = asanyarray(b)
755:     a = array(a, copy=False, subok=True, ndmin=b.ndim)
756:     ndb, nda = b.ndim, a.ndim
757:     if (nda == 0 or ndb == 0):
758:         return _nx.multiply(a, b)
759:     as_ = a.shape
760:     bs = b.shape
761:     if not a.flags.contiguous:
762:         a = reshape(a, as_)
763:     if not b.flags.contiguous:
764:         b = reshape(b, bs)
765:     nd = ndb
766:     if (ndb != nda):
767:         if (ndb > nda):
768:             as_ = (1,)*(ndb-nda) + as_
769:         else:
770:             bs = (1,)*(nda-ndb) + bs
771:             nd = nda
772:     result = outer(a, b).reshape(as_+bs)
773:     axis = nd-1
774:     for _ in range(nd):
775:         result = concatenate(result, axis=axis)
776:     wrapper = get_array_prepare(a, b)
777:     if wrapper is not None:
778:         result = wrapper(result)
779:     wrapper = get_array_wrap(a, b)
780:     if wrapper is not None:
781:         result = wrapper(result)
782:     return result
783: 
784: 
785: def tile(A, reps):
786:     '''
787:     Construct an array by repeating A the number of times given by reps.
788: 
789:     If `reps` has length ``d``, the result will have dimension of
790:     ``max(d, A.ndim)``.
791: 
792:     If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
793:     axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
794:     or shape (1, 1, 3) for 3-D replication. If this is not the desired
795:     behavior, promote `A` to d-dimensions manually before calling this
796:     function.
797: 
798:     If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
799:     Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
800:     (1, 1, 2, 2).
801: 
802:     Note : Although tile may be used for broadcasting, it is strongly
803:     recommended to use numpy's broadcasting operations and functions.
804: 
805:     Parameters
806:     ----------
807:     A : array_like
808:         The input array.
809:     reps : array_like
810:         The number of repetitions of `A` along each axis.
811: 
812:     Returns
813:     -------
814:     c : ndarray
815:         The tiled output array.
816: 
817:     See Also
818:     --------
819:     repeat : Repeat elements of an array.
820:     broadcast_to : Broadcast an array to a new shape
821: 
822:     Examples
823:     --------
824:     >>> a = np.array([0, 1, 2])
825:     >>> np.tile(a, 2)
826:     array([0, 1, 2, 0, 1, 2])
827:     >>> np.tile(a, (2, 2))
828:     array([[0, 1, 2, 0, 1, 2],
829:            [0, 1, 2, 0, 1, 2]])
830:     >>> np.tile(a, (2, 1, 2))
831:     array([[[0, 1, 2, 0, 1, 2]],
832:            [[0, 1, 2, 0, 1, 2]]])
833: 
834:     >>> b = np.array([[1, 2], [3, 4]])
835:     >>> np.tile(b, 2)
836:     array([[1, 2, 1, 2],
837:            [3, 4, 3, 4]])
838:     >>> np.tile(b, (2, 1))
839:     array([[1, 2],
840:            [3, 4],
841:            [1, 2],
842:            [3, 4]])
843: 
844:     >>> c = np.array([1,2,3,4])
845:     >>> np.tile(c,(4,1))
846:     array([[1, 2, 3, 4],
847:            [1, 2, 3, 4],
848:            [1, 2, 3, 4],
849:            [1, 2, 3, 4]])
850:     '''
851:     try:
852:         tup = tuple(reps)
853:     except TypeError:
854:         tup = (reps,)
855:     d = len(tup)
856:     if all(x == 1 for x in tup) and isinstance(A, _nx.ndarray):
857:         # Fixes the problem that the function does not make a copy if A is a
858:         # numpy array and the repetitions are 1 in all dimensions
859:         return _nx.array(A, copy=True, subok=True, ndmin=d)
860:     else:
861:         # Note that no copy of zero-sized arrays is made. However since they
862:         # have no data there is no risk of an inadvertent overwrite.
863:         c = _nx.array(A, copy=False, subok=True, ndmin=d)
864:     if (d < c.ndim):
865:         tup = (1,)*(c.ndim-d) + tup
866:     shape_out = tuple(s*t for s, t in zip(c.shape, tup))
867:     n = c.size
868:     if n > 0:
869:         for dim_in, nrep in zip(c.shape, tup):
870:             if nrep != 1:
871:                 c = c.reshape(-1, n).repeat(nrep, 0)
872:             n //= dim_in
873:     return c.reshape(shape_out)
874: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import warnings' statement (line 3)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy.core.numeric' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_125188 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core.numeric')

if (type(import_125188) is not StypyTypeError):

    if (import_125188 != 'pyd_module'):
        __import__(import_125188)
        sys_modules_125189 = sys.modules[import_125188]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), '_nx', sys_modules_125189.module_type_store, module_type_store)
    else:
        import numpy.core.numeric as _nx

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), '_nx', numpy.core.numeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.core.numeric', import_125188)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core.numeric import asarray, zeros, outer, concatenate, isscalar, array, asanyarray' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_125190 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric')

if (type(import_125190) is not StypyTypeError):

    if (import_125190 != 'pyd_module'):
        __import__(import_125190)
        sys_modules_125191 = sys.modules[import_125190]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', sys_modules_125191.module_type_store, module_type_store, ['asarray', 'zeros', 'outer', 'concatenate', 'isscalar', 'array', 'asanyarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_125191, sys_modules_125191.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, zeros, outer, concatenate, isscalar, array, asanyarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'zeros', 'outer', 'concatenate', 'isscalar', 'array', 'asanyarray'], [asarray, zeros, outer, concatenate, isscalar, array, asanyarray])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core.numeric', import_125190)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.core.fromnumeric import product, reshape' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_125192 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.fromnumeric')

if (type(import_125192) is not StypyTypeError):

    if (import_125192 != 'pyd_module'):
        __import__(import_125192)
        sys_modules_125193 = sys.modules[import_125192]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.fromnumeric', sys_modules_125193.module_type_store, module_type_store, ['product', 'reshape'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_125193, sys_modules_125193.module_type_store, module_type_store)
    else:
        from numpy.core.fromnumeric import product, reshape

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.fromnumeric', None, module_type_store, ['product', 'reshape'], [product, reshape])

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.fromnumeric', import_125192)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import vstack, atleast_3d' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_125194 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_125194) is not StypyTypeError):

    if (import_125194 != 'pyd_module'):
        __import__(import_125194)
        sys_modules_125195 = sys.modules[import_125194]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_125195.module_type_store, module_type_store, ['vstack', 'atleast_3d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_125195, sys_modules_125195.module_type_store, module_type_store)
    else:
        from numpy.core import vstack, atleast_3d

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['vstack', 'atleast_3d'], [vstack, atleast_3d])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_125194)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['column_stack', 'row_stack', 'dstack', 'array_split', 'split', 'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims', 'apply_along_axis', 'kron', 'tile', 'get_array_wrap']
module_type_store.set_exportable_members(['column_stack', 'row_stack', 'dstack', 'array_split', 'split', 'hsplit', 'vsplit', 'dsplit', 'apply_over_axes', 'expand_dims', 'apply_along_axis', 'kron', 'tile', 'get_array_wrap'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_125196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_125197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'column_stack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125197)
# Adding element type (line 13)
str_125198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'str', 'row_stack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125198)
# Adding element type (line 13)
str_125199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', 'dstack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125199)
# Adding element type (line 13)
str_125200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 43), 'str', 'array_split')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125200)
# Adding element type (line 13)
str_125201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 58), 'str', 'split')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125201)
# Adding element type (line 13)
str_125202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'hsplit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125202)
# Adding element type (line 13)
str_125203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'str', 'vsplit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125203)
# Adding element type (line 13)
str_125204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'str', 'dsplit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125204)
# Adding element type (line 13)
str_125205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', 'apply_over_axes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125205)
# Adding element type (line 13)
str_125206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 53), 'str', 'expand_dims')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125206)
# Adding element type (line 13)
str_125207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'apply_along_axis')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125207)
# Adding element type (line 13)
str_125208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'str', 'kron')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125208)
# Adding element type (line 13)
str_125209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 32), 'str', 'tile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125209)
# Adding element type (line 13)
str_125210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 40), 'str', 'get_array_wrap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_125196, str_125210)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_125196)

@norecursion
def apply_along_axis(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'apply_along_axis'
    module_type_store = module_type_store.open_function_context('apply_along_axis', 20, 0, False)
    
    # Passed parameters checking function
    apply_along_axis.stypy_localization = localization
    apply_along_axis.stypy_type_of_self = None
    apply_along_axis.stypy_type_store = module_type_store
    apply_along_axis.stypy_function_name = 'apply_along_axis'
    apply_along_axis.stypy_param_names_list = ['func1d', 'axis', 'arr']
    apply_along_axis.stypy_varargs_param_name = 'args'
    apply_along_axis.stypy_kwargs_param_name = 'kwargs'
    apply_along_axis.stypy_call_defaults = defaults
    apply_along_axis.stypy_call_varargs = varargs
    apply_along_axis.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'apply_along_axis', ['func1d', 'axis', 'arr'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'apply_along_axis', localization, ['func1d', 'axis', 'arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'apply_along_axis(...)' code ##################

    str_125211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n    Apply a function to 1-D slices along the given axis.\n\n    Execute `func1d(a, *args)` where `func1d` operates on 1-D arrays and `a`\n    is a 1-D slice of `arr` along `axis`.\n\n    Parameters\n    ----------\n    func1d : function\n        This function should accept 1-D arrays. It is applied to 1-D\n        slices of `arr` along the specified axis.\n    axis : integer\n        Axis along which `arr` is sliced.\n    arr : ndarray\n        Input array.\n    args : any\n        Additional arguments to `func1d`.\n    kwargs: any\n        Additional named arguments to `func1d`.\n\n        .. versionadded:: 1.9.0\n\n\n    Returns\n    -------\n    apply_along_axis : ndarray\n        The output array. The shape of `outarr` is identical to the shape of\n        `arr`, except along the `axis` dimension, where the length of `outarr`\n        is equal to the size of the return value of `func1d`.  If `func1d`\n        returns a scalar `outarr` will have one fewer dimensions than `arr`.\n\n    See Also\n    --------\n    apply_over_axes : Apply a function repeatedly over multiple axes.\n\n    Examples\n    --------\n    >>> def my_func(a):\n    ...     """Average first and last element of a 1-D array"""\n    ...     return (a[0] + a[-1]) * 0.5\n    >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])\n    >>> np.apply_along_axis(my_func, 0, b)\n    array([ 4.,  5.,  6.])\n    >>> np.apply_along_axis(my_func, 1, b)\n    array([ 2.,  5.,  8.])\n\n    For a function that doesn\'t return a scalar, the number of dimensions in\n    `outarr` is the same as `arr`.\n\n    >>> b = np.array([[8,1,7], [4,3,9], [5,2,6]])\n    >>> np.apply_along_axis(sorted, 1, b)\n    array([[1, 7, 8],\n           [3, 4, 9],\n           [2, 5, 6]])\n\n    ')
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to asarray(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'arr' (line 77)
    arr_125213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'arr', False)
    # Processing the call keyword arguments (line 77)
    kwargs_125214 = {}
    # Getting the type of 'asarray' (line 77)
    asarray_125212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 77)
    asarray_call_result_125215 = invoke(stypy.reporting.localization.Localization(__file__, 77, 10), asarray_125212, *[arr_125213], **kwargs_125214)
    
    # Assigning a type to the variable 'arr' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'arr', asarray_call_result_125215)
    
    # Assigning a Attribute to a Name (line 78):
    
    # Assigning a Attribute to a Name (line 78):
    # Getting the type of 'arr' (line 78)
    arr_125216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'arr')
    # Obtaining the member 'ndim' of a type (line 78)
    ndim_125217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 9), arr_125216, 'ndim')
    # Assigning a type to the variable 'nd' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'nd', ndim_125217)
    
    
    # Getting the type of 'axis' (line 79)
    axis_125218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'axis')
    int_125219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 14), 'int')
    # Applying the binary operator '<' (line 79)
    result_lt_125220 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), '<', axis_125218, int_125219)
    
    # Testing the type of an if condition (line 79)
    if_condition_125221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_lt_125220)
    # Assigning a type to the variable 'if_condition_125221' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_125221', if_condition_125221)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axis' (line 80)
    axis_125222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'axis')
    # Getting the type of 'nd' (line 80)
    nd_125223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'nd')
    # Applying the binary operator '+=' (line 80)
    result_iadd_125224 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), '+=', axis_125222, nd_125223)
    # Assigning a type to the variable 'axis' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'axis', result_iadd_125224)
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'axis' (line 81)
    axis_125225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'axis')
    # Getting the type of 'nd' (line 81)
    nd_125226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'nd')
    # Applying the binary operator '>=' (line 81)
    result_ge_125227 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), '>=', axis_125225, nd_125226)
    
    # Testing the type of an if condition (line 81)
    if_condition_125228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_ge_125227)
    # Assigning a type to the variable 'if_condition_125228' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_125228', if_condition_125228)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 82)
    # Processing the call arguments (line 82)
    str_125230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', 'axis must be less than arr.ndim; axis=%d, rank=%d.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 83)
    tuple_125231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'axis' (line 83)
    axis_125232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_125231, axis_125232)
    # Adding element type (line 83)
    # Getting the type of 'nd' (line 83)
    nd_125233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'nd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 15), tuple_125231, nd_125233)
    
    # Applying the binary operator '%' (line 82)
    result_mod_125234 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 25), '%', str_125230, tuple_125231)
    
    # Processing the call keyword arguments (line 82)
    kwargs_125235 = {}
    # Getting the type of 'ValueError' (line 82)
    ValueError_125229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 82)
    ValueError_call_result_125236 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), ValueError_125229, *[result_mod_125234], **kwargs_125235)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 82, 8), ValueError_call_result_125236, 'raise parameter', BaseException)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    
    # Obtaining an instance of the builtin type 'list' (line 84)
    list_125237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 84)
    # Adding element type (line 84)
    int_125238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 10), list_125237, int_125238)
    
    # Getting the type of 'nd' (line 84)
    nd_125239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'nd')
    int_125240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'int')
    # Applying the binary operator '-' (line 84)
    result_sub_125241 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), '-', nd_125239, int_125240)
    
    # Applying the binary operator '*' (line 84)
    result_mul_125242 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 10), '*', list_125237, result_sub_125241)
    
    # Assigning a type to the variable 'ind' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'ind', result_mul_125242)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to zeros(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'nd' (line 85)
    nd_125244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'nd', False)
    str_125245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', 'O')
    # Processing the call keyword arguments (line 85)
    kwargs_125246 = {}
    # Getting the type of 'zeros' (line 85)
    zeros_125243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 85)
    zeros_call_result_125247 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), zeros_125243, *[nd_125244, str_125245], **kwargs_125246)
    
    # Assigning a type to the variable 'i' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'i', zeros_call_result_125247)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to list(...): (line 86)
    # Processing the call arguments (line 86)
    
    # Call to range(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'nd' (line 86)
    nd_125250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'nd', False)
    # Processing the call keyword arguments (line 86)
    kwargs_125251 = {}
    # Getting the type of 'range' (line 86)
    range_125249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'range', False)
    # Calling range(args, kwargs) (line 86)
    range_call_result_125252 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), range_125249, *[nd_125250], **kwargs_125251)
    
    # Processing the call keyword arguments (line 86)
    kwargs_125253 = {}
    # Getting the type of 'list' (line 86)
    list_125248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'list', False)
    # Calling list(args, kwargs) (line 86)
    list_call_result_125254 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), list_125248, *[range_call_result_125252], **kwargs_125253)
    
    # Assigning a type to the variable 'indlist' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'indlist', list_call_result_125254)
    
    # Call to remove(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'axis' (line 87)
    axis_125257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'axis', False)
    # Processing the call keyword arguments (line 87)
    kwargs_125258 = {}
    # Getting the type of 'indlist' (line 87)
    indlist_125255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'indlist', False)
    # Obtaining the member 'remove' of a type (line 87)
    remove_125256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 4), indlist_125255, 'remove')
    # Calling remove(args, kwargs) (line 87)
    remove_call_result_125259 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), remove_125256, *[axis_125257], **kwargs_125258)
    
    
    # Assigning a Call to a Subscript (line 88):
    
    # Assigning a Call to a Subscript (line 88):
    
    # Call to slice(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'None' (line 88)
    None_125261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'None', False)
    # Getting the type of 'None' (line 88)
    None_125262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'None', False)
    # Processing the call keyword arguments (line 88)
    kwargs_125263 = {}
    # Getting the type of 'slice' (line 88)
    slice_125260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'slice', False)
    # Calling slice(args, kwargs) (line 88)
    slice_call_result_125264 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), slice_125260, *[None_125261, None_125262], **kwargs_125263)
    
    # Getting the type of 'i' (line 88)
    i_125265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'i')
    # Getting the type of 'axis' (line 88)
    axis_125266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 6), 'axis')
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 4), i_125265, (axis_125266, slice_call_result_125264))
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to take(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'indlist' (line 89)
    indlist_125273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'indlist', False)
    # Processing the call keyword arguments (line 89)
    kwargs_125274 = {}
    
    # Call to asarray(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'arr' (line 89)
    arr_125268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'arr', False)
    # Obtaining the member 'shape' of a type (line 89)
    shape_125269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), arr_125268, 'shape')
    # Processing the call keyword arguments (line 89)
    kwargs_125270 = {}
    # Getting the type of 'asarray' (line 89)
    asarray_125267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'asarray', False)
    # Calling asarray(args, kwargs) (line 89)
    asarray_call_result_125271 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), asarray_125267, *[shape_125269], **kwargs_125270)
    
    # Obtaining the member 'take' of a type (line 89)
    take_125272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), asarray_call_result_125271, 'take')
    # Calling take(args, kwargs) (line 89)
    take_call_result_125275 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), take_125272, *[indlist_125273], **kwargs_125274)
    
    # Assigning a type to the variable 'outshape' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'outshape', take_call_result_125275)
    
    # Call to put(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'indlist' (line 90)
    indlist_125278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 10), 'indlist', False)
    # Getting the type of 'ind' (line 90)
    ind_125279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'ind', False)
    # Processing the call keyword arguments (line 90)
    kwargs_125280 = {}
    # Getting the type of 'i' (line 90)
    i_125276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'i', False)
    # Obtaining the member 'put' of a type (line 90)
    put_125277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), i_125276, 'put')
    # Calling put(args, kwargs) (line 90)
    put_call_result_125281 = invoke(stypy.reporting.localization.Localization(__file__, 90, 4), put_125277, *[indlist_125278, ind_125279], **kwargs_125280)
    
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to func1d(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to tolist(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_125286 = {}
    # Getting the type of 'i' (line 91)
    i_125284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'i', False)
    # Obtaining the member 'tolist' of a type (line 91)
    tolist_125285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), i_125284, 'tolist')
    # Calling tolist(args, kwargs) (line 91)
    tolist_call_result_125287 = invoke(stypy.reporting.localization.Localization(__file__, 91, 27), tolist_125285, *[], **kwargs_125286)
    
    # Processing the call keyword arguments (line 91)
    kwargs_125288 = {}
    # Getting the type of 'tuple' (line 91)
    tuple_125283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'tuple', False)
    # Calling tuple(args, kwargs) (line 91)
    tuple_call_result_125289 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), tuple_125283, *[tolist_call_result_125287], **kwargs_125288)
    
    # Getting the type of 'arr' (line 91)
    arr_125290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 17), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___125291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 17), arr_125290, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_125292 = invoke(stypy.reporting.localization.Localization(__file__, 91, 17), getitem___125291, tuple_call_result_125289)
    
    # Getting the type of 'args' (line 91)
    args_125293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'args', False)
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'kwargs' (line 91)
    kwargs_125294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 50), 'kwargs', False)
    kwargs_125295 = {'kwargs_125294': kwargs_125294}
    # Getting the type of 'func1d' (line 91)
    func1d_125282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'func1d', False)
    # Calling func1d(args, kwargs) (line 91)
    func1d_call_result_125296 = invoke(stypy.reporting.localization.Localization(__file__, 91, 10), func1d_125282, *[subscript_call_result_125292, args_125293], **kwargs_125295)
    
    # Assigning a type to the variable 'res' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'res', func1d_call_result_125296)
    
    
    # Call to isscalar(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'res' (line 93)
    res_125298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'res', False)
    # Processing the call keyword arguments (line 93)
    kwargs_125299 = {}
    # Getting the type of 'isscalar' (line 93)
    isscalar_125297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 93)
    isscalar_call_result_125300 = invoke(stypy.reporting.localization.Localization(__file__, 93, 7), isscalar_125297, *[res_125298], **kwargs_125299)
    
    # Testing the type of an if condition (line 93)
    if_condition_125301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), isscalar_call_result_125300)
    # Assigning a type to the variable 'if_condition_125301' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_125301', if_condition_125301)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to zeros(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'outshape' (line 94)
    outshape_125303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'outshape', False)
    
    # Call to asarray(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'res' (line 94)
    res_125305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 41), 'res', False)
    # Processing the call keyword arguments (line 94)
    kwargs_125306 = {}
    # Getting the type of 'asarray' (line 94)
    asarray_125304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'asarray', False)
    # Calling asarray(args, kwargs) (line 94)
    asarray_call_result_125307 = invoke(stypy.reporting.localization.Localization(__file__, 94, 33), asarray_125304, *[res_125305], **kwargs_125306)
    
    # Obtaining the member 'dtype' of a type (line 94)
    dtype_125308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), asarray_call_result_125307, 'dtype')
    # Processing the call keyword arguments (line 94)
    kwargs_125309 = {}
    # Getting the type of 'zeros' (line 94)
    zeros_125302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'zeros', False)
    # Calling zeros(args, kwargs) (line 94)
    zeros_call_result_125310 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), zeros_125302, *[outshape_125303, dtype_125308], **kwargs_125309)
    
    # Assigning a type to the variable 'outarr' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'outarr', zeros_call_result_125310)
    
    # Assigning a Name to a Subscript (line 95):
    
    # Assigning a Name to a Subscript (line 95):
    # Getting the type of 'res' (line 95)
    res_125311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'res')
    # Getting the type of 'outarr' (line 95)
    outarr_125312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'outarr')
    
    # Call to tuple(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'ind' (line 95)
    ind_125314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'ind', False)
    # Processing the call keyword arguments (line 95)
    kwargs_125315 = {}
    # Getting the type of 'tuple' (line 95)
    tuple_125313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 95)
    tuple_call_result_125316 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), tuple_125313, *[ind_125314], **kwargs_125315)
    
    # Storing an element on a container (line 95)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), outarr_125312, (tuple_call_result_125316, res_125311))
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to product(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'outshape' (line 96)
    outshape_125318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'outshape', False)
    # Processing the call keyword arguments (line 96)
    kwargs_125319 = {}
    # Getting the type of 'product' (line 96)
    product_125317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'product', False)
    # Calling product(args, kwargs) (line 96)
    product_call_result_125320 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), product_125317, *[outshape_125318], **kwargs_125319)
    
    # Assigning a type to the variable 'Ntot' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'Ntot', product_call_result_125320)
    
    # Assigning a Num to a Name (line 97):
    
    # Assigning a Num to a Name (line 97):
    int_125321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'int')
    # Assigning a type to the variable 'k' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'k', int_125321)
    
    
    # Getting the type of 'k' (line 98)
    k_125322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'k')
    # Getting the type of 'Ntot' (line 98)
    Ntot_125323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'Ntot')
    # Applying the binary operator '<' (line 98)
    result_lt_125324 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 14), '<', k_125322, Ntot_125323)
    
    # Testing the type of an if condition (line 98)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_lt_125324)
    # SSA begins for while statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'ind' (line 100)
    ind_125325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ind')
    
    # Obtaining the type of the subscript
    int_125326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'int')
    # Getting the type of 'ind' (line 100)
    ind_125327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ind')
    # Obtaining the member '__getitem__' of a type (line 100)
    getitem___125328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), ind_125327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 100)
    subscript_call_result_125329 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), getitem___125328, int_125326)
    
    int_125330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 23), 'int')
    # Applying the binary operator '+=' (line 100)
    result_iadd_125331 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '+=', subscript_call_result_125329, int_125330)
    # Getting the type of 'ind' (line 100)
    ind_125332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ind')
    int_125333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'int')
    # Storing an element on a container (line 100)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 12), ind_125332, (int_125333, result_iadd_125331))
    
    
    # Assigning a Num to a Name (line 101):
    
    # Assigning a Num to a Name (line 101):
    int_125334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'int')
    # Assigning a type to the variable 'n' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'n', int_125334)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 102)
    n_125335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'n')
    # Getting the type of 'ind' (line 102)
    ind_125336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'ind')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___125337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 19), ind_125336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_125338 = invoke(stypy.reporting.localization.Localization(__file__, 102, 19), getitem___125337, n_125335)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 102)
    n_125339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'n')
    # Getting the type of 'outshape' (line 102)
    outshape_125340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'outshape')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___125341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 29), outshape_125340, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_125342 = invoke(stypy.reporting.localization.Localization(__file__, 102, 29), getitem___125341, n_125339)
    
    # Applying the binary operator '>=' (line 102)
    result_ge_125343 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 19), '>=', subscript_call_result_125338, subscript_call_result_125342)
    
    
    # Getting the type of 'n' (line 102)
    n_125344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 47), 'n')
    int_125345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'int')
    # Getting the type of 'nd' (line 102)
    nd_125346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'nd')
    # Applying the binary operator '-' (line 102)
    result_sub_125347 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 52), '-', int_125345, nd_125346)
    
    # Applying the binary operator '>' (line 102)
    result_gt_125348 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 47), '>', n_125344, result_sub_125347)
    
    # Applying the binary operator 'and' (line 102)
    result_and_keyword_125349 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 18), 'and', result_ge_125343, result_gt_125348)
    
    # Testing the type of an if condition (line 102)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 12), result_and_keyword_125349)
    # SSA begins for while statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'ind' (line 103)
    ind_125350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'ind')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 103)
    n_125351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'n')
    int_125352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'int')
    # Applying the binary operator '-' (line 103)
    result_sub_125353 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 20), '-', n_125351, int_125352)
    
    # Getting the type of 'ind' (line 103)
    ind_125354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'ind')
    # Obtaining the member '__getitem__' of a type (line 103)
    getitem___125355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), ind_125354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 103)
    subscript_call_result_125356 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), getitem___125355, result_sub_125353)
    
    int_125357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 28), 'int')
    # Applying the binary operator '+=' (line 103)
    result_iadd_125358 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 16), '+=', subscript_call_result_125356, int_125357)
    # Getting the type of 'ind' (line 103)
    ind_125359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'ind')
    # Getting the type of 'n' (line 103)
    n_125360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'n')
    int_125361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'int')
    # Applying the binary operator '-' (line 103)
    result_sub_125362 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 20), '-', n_125360, int_125361)
    
    # Storing an element on a container (line 103)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 16), ind_125359, (result_sub_125362, result_iadd_125358))
    
    
    # Assigning a Num to a Subscript (line 104):
    
    # Assigning a Num to a Subscript (line 104):
    int_125363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
    # Getting the type of 'ind' (line 104)
    ind_125364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'ind')
    # Getting the type of 'n' (line 104)
    n_125365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'n')
    # Storing an element on a container (line 104)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), ind_125364, (n_125365, int_125363))
    
    # Getting the type of 'n' (line 105)
    n_125366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'n')
    int_125367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'int')
    # Applying the binary operator '-=' (line 105)
    result_isub_125368 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 16), '-=', n_125366, int_125367)
    # Assigning a type to the variable 'n' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'n', result_isub_125368)
    
    # SSA join for while statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to put(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'indlist' (line 106)
    indlist_125371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'indlist', False)
    # Getting the type of 'ind' (line 106)
    ind_125372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'ind', False)
    # Processing the call keyword arguments (line 106)
    kwargs_125373 = {}
    # Getting the type of 'i' (line 106)
    i_125369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'i', False)
    # Obtaining the member 'put' of a type (line 106)
    put_125370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), i_125369, 'put')
    # Calling put(args, kwargs) (line 106)
    put_call_result_125374 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), put_125370, *[indlist_125371, ind_125372], **kwargs_125373)
    
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to func1d(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Call to tolist(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_125379 = {}
    # Getting the type of 'i' (line 107)
    i_125377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 35), 'i', False)
    # Obtaining the member 'tolist' of a type (line 107)
    tolist_125378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 35), i_125377, 'tolist')
    # Calling tolist(args, kwargs) (line 107)
    tolist_call_result_125380 = invoke(stypy.reporting.localization.Localization(__file__, 107, 35), tolist_125378, *[], **kwargs_125379)
    
    # Processing the call keyword arguments (line 107)
    kwargs_125381 = {}
    # Getting the type of 'tuple' (line 107)
    tuple_125376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'tuple', False)
    # Calling tuple(args, kwargs) (line 107)
    tuple_call_result_125382 = invoke(stypy.reporting.localization.Localization(__file__, 107, 29), tuple_125376, *[tolist_call_result_125380], **kwargs_125381)
    
    # Getting the type of 'arr' (line 107)
    arr_125383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___125384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 25), arr_125383, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_125385 = invoke(stypy.reporting.localization.Localization(__file__, 107, 25), getitem___125384, tuple_call_result_125382)
    
    # Getting the type of 'args' (line 107)
    args_125386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 50), 'args', False)
    # Processing the call keyword arguments (line 107)
    # Getting the type of 'kwargs' (line 107)
    kwargs_125387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 58), 'kwargs', False)
    kwargs_125388 = {'kwargs_125387': kwargs_125387}
    # Getting the type of 'func1d' (line 107)
    func1d_125375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'func1d', False)
    # Calling func1d(args, kwargs) (line 107)
    func1d_call_result_125389 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), func1d_125375, *[subscript_call_result_125385, args_125386], **kwargs_125388)
    
    # Assigning a type to the variable 'res' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'res', func1d_call_result_125389)
    
    # Assigning a Name to a Subscript (line 108):
    
    # Assigning a Name to a Subscript (line 108):
    # Getting the type of 'res' (line 108)
    res_125390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'res')
    # Getting the type of 'outarr' (line 108)
    outarr_125391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'outarr')
    
    # Call to tuple(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'ind' (line 108)
    ind_125393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'ind', False)
    # Processing the call keyword arguments (line 108)
    kwargs_125394 = {}
    # Getting the type of 'tuple' (line 108)
    tuple_125392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'tuple', False)
    # Calling tuple(args, kwargs) (line 108)
    tuple_call_result_125395 = invoke(stypy.reporting.localization.Localization(__file__, 108, 19), tuple_125392, *[ind_125393], **kwargs_125394)
    
    # Storing an element on a container (line 108)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), outarr_125391, (tuple_call_result_125395, res_125390))
    
    # Getting the type of 'k' (line 109)
    k_125396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'k')
    int_125397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 17), 'int')
    # Applying the binary operator '+=' (line 109)
    result_iadd_125398 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '+=', k_125396, int_125397)
    # Assigning a type to the variable 'k' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'k', result_iadd_125398)
    
    # SSA join for while statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'outarr' (line 110)
    outarr_125399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'outarr')
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'stypy_return_type', outarr_125399)
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to product(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'outshape' (line 112)
    outshape_125401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'outshape', False)
    # Processing the call keyword arguments (line 112)
    kwargs_125402 = {}
    # Getting the type of 'product' (line 112)
    product_125400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'product', False)
    # Calling product(args, kwargs) (line 112)
    product_call_result_125403 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), product_125400, *[outshape_125401], **kwargs_125402)
    
    # Assigning a type to the variable 'Ntot' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'Ntot', product_call_result_125403)
    
    # Assigning a Name to a Name (line 113):
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'outshape' (line 113)
    outshape_125404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'outshape')
    # Assigning a type to the variable 'holdshape' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'holdshape', outshape_125404)
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to list(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'arr' (line 114)
    arr_125406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'arr', False)
    # Obtaining the member 'shape' of a type (line 114)
    shape_125407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), arr_125406, 'shape')
    # Processing the call keyword arguments (line 114)
    kwargs_125408 = {}
    # Getting the type of 'list' (line 114)
    list_125405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'list', False)
    # Calling list(args, kwargs) (line 114)
    list_call_result_125409 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), list_125405, *[shape_125407], **kwargs_125408)
    
    # Assigning a type to the variable 'outshape' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'outshape', list_call_result_125409)
    
    # Assigning a Call to a Subscript (line 115):
    
    # Assigning a Call to a Subscript (line 115):
    
    # Call to len(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'res' (line 115)
    res_125411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'res', False)
    # Processing the call keyword arguments (line 115)
    kwargs_125412 = {}
    # Getting the type of 'len' (line 115)
    len_125410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'len', False)
    # Calling len(args, kwargs) (line 115)
    len_call_result_125413 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), len_125410, *[res_125411], **kwargs_125412)
    
    # Getting the type of 'outshape' (line 115)
    outshape_125414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'outshape')
    # Getting the type of 'axis' (line 115)
    axis_125415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'axis')
    # Storing an element on a container (line 115)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), outshape_125414, (axis_125415, len_call_result_125413))
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to zeros(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'outshape' (line 116)
    outshape_125417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'outshape', False)
    
    # Call to asarray(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'res' (line 116)
    res_125419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'res', False)
    # Processing the call keyword arguments (line 116)
    kwargs_125420 = {}
    # Getting the type of 'asarray' (line 116)
    asarray_125418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'asarray', False)
    # Calling asarray(args, kwargs) (line 116)
    asarray_call_result_125421 = invoke(stypy.reporting.localization.Localization(__file__, 116, 33), asarray_125418, *[res_125419], **kwargs_125420)
    
    # Obtaining the member 'dtype' of a type (line 116)
    dtype_125422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 33), asarray_call_result_125421, 'dtype')
    # Processing the call keyword arguments (line 116)
    kwargs_125423 = {}
    # Getting the type of 'zeros' (line 116)
    zeros_125416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'zeros', False)
    # Calling zeros(args, kwargs) (line 116)
    zeros_call_result_125424 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), zeros_125416, *[outshape_125417, dtype_125422], **kwargs_125423)
    
    # Assigning a type to the variable 'outarr' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'outarr', zeros_call_result_125424)
    
    # Assigning a Name to a Subscript (line 117):
    
    # Assigning a Name to a Subscript (line 117):
    # Getting the type of 'res' (line 117)
    res_125425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'res')
    # Getting the type of 'outarr' (line 117)
    outarr_125426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'outarr')
    
    # Call to tuple(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Call to tolist(...): (line 117)
    # Processing the call keyword arguments (line 117)
    kwargs_125430 = {}
    # Getting the type of 'i' (line 117)
    i_125428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'i', False)
    # Obtaining the member 'tolist' of a type (line 117)
    tolist_125429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 21), i_125428, 'tolist')
    # Calling tolist(args, kwargs) (line 117)
    tolist_call_result_125431 = invoke(stypy.reporting.localization.Localization(__file__, 117, 21), tolist_125429, *[], **kwargs_125430)
    
    # Processing the call keyword arguments (line 117)
    kwargs_125432 = {}
    # Getting the type of 'tuple' (line 117)
    tuple_125427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 117)
    tuple_call_result_125433 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), tuple_125427, *[tolist_call_result_125431], **kwargs_125432)
    
    # Storing an element on a container (line 117)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 8), outarr_125426, (tuple_call_result_125433, res_125425))
    
    # Assigning a Num to a Name (line 118):
    
    # Assigning a Num to a Name (line 118):
    int_125434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'int')
    # Assigning a type to the variable 'k' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'k', int_125434)
    
    
    # Getting the type of 'k' (line 119)
    k_125435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'k')
    # Getting the type of 'Ntot' (line 119)
    Ntot_125436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'Ntot')
    # Applying the binary operator '<' (line 119)
    result_lt_125437 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 14), '<', k_125435, Ntot_125436)
    
    # Testing the type of an if condition (line 119)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_lt_125437)
    # SSA begins for while statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'ind' (line 121)
    ind_125438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'ind')
    
    # Obtaining the type of the subscript
    int_125439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'int')
    # Getting the type of 'ind' (line 121)
    ind_125440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'ind')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___125441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), ind_125440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_125442 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), getitem___125441, int_125439)
    
    int_125443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'int')
    # Applying the binary operator '+=' (line 121)
    result_iadd_125444 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 12), '+=', subscript_call_result_125442, int_125443)
    # Getting the type of 'ind' (line 121)
    ind_125445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'ind')
    int_125446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 16), 'int')
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 12), ind_125445, (int_125446, result_iadd_125444))
    
    
    # Assigning a Num to a Name (line 122):
    
    # Assigning a Num to a Name (line 122):
    int_125447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'int')
    # Assigning a type to the variable 'n' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'n', int_125447)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 123)
    n_125448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'n')
    # Getting the type of 'ind' (line 123)
    ind_125449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'ind')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___125450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 19), ind_125449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_125451 = invoke(stypy.reporting.localization.Localization(__file__, 123, 19), getitem___125450, n_125448)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 123)
    n_125452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'n')
    # Getting the type of 'holdshape' (line 123)
    holdshape_125453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'holdshape')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___125454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 29), holdshape_125453, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_125455 = invoke(stypy.reporting.localization.Localization(__file__, 123, 29), getitem___125454, n_125452)
    
    # Applying the binary operator '>=' (line 123)
    result_ge_125456 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 19), '>=', subscript_call_result_125451, subscript_call_result_125455)
    
    
    # Getting the type of 'n' (line 123)
    n_125457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 48), 'n')
    int_125458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 53), 'int')
    # Getting the type of 'nd' (line 123)
    nd_125459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'nd')
    # Applying the binary operator '-' (line 123)
    result_sub_125460 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 53), '-', int_125458, nd_125459)
    
    # Applying the binary operator '>' (line 123)
    result_gt_125461 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 48), '>', n_125457, result_sub_125460)
    
    # Applying the binary operator 'and' (line 123)
    result_and_keyword_125462 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 18), 'and', result_ge_125456, result_gt_125461)
    
    # Testing the type of an if condition (line 123)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), result_and_keyword_125462)
    # SSA begins for while statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'ind' (line 124)
    ind_125463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'ind')
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 124)
    n_125464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'n')
    int_125465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
    # Applying the binary operator '-' (line 124)
    result_sub_125466 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '-', n_125464, int_125465)
    
    # Getting the type of 'ind' (line 124)
    ind_125467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'ind')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___125468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), ind_125467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_125469 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), getitem___125468, result_sub_125466)
    
    int_125470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'int')
    # Applying the binary operator '+=' (line 124)
    result_iadd_125471 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '+=', subscript_call_result_125469, int_125470)
    # Getting the type of 'ind' (line 124)
    ind_125472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'ind')
    # Getting the type of 'n' (line 124)
    n_125473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'n')
    int_125474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
    # Applying the binary operator '-' (line 124)
    result_sub_125475 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 20), '-', n_125473, int_125474)
    
    # Storing an element on a container (line 124)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 16), ind_125472, (result_sub_125475, result_iadd_125471))
    
    
    # Assigning a Num to a Subscript (line 125):
    
    # Assigning a Num to a Subscript (line 125):
    int_125476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'int')
    # Getting the type of 'ind' (line 125)
    ind_125477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'ind')
    # Getting the type of 'n' (line 125)
    n_125478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'n')
    # Storing an element on a container (line 125)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 16), ind_125477, (n_125478, int_125476))
    
    # Getting the type of 'n' (line 126)
    n_125479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'n')
    int_125480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'int')
    # Applying the binary operator '-=' (line 126)
    result_isub_125481 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), '-=', n_125479, int_125480)
    # Assigning a type to the variable 'n' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'n', result_isub_125481)
    
    # SSA join for while statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to put(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'indlist' (line 127)
    indlist_125484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'indlist', False)
    # Getting the type of 'ind' (line 127)
    ind_125485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'ind', False)
    # Processing the call keyword arguments (line 127)
    kwargs_125486 = {}
    # Getting the type of 'i' (line 127)
    i_125482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'i', False)
    # Obtaining the member 'put' of a type (line 127)
    put_125483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), i_125482, 'put')
    # Calling put(args, kwargs) (line 127)
    put_call_result_125487 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), put_125483, *[indlist_125484, ind_125485], **kwargs_125486)
    
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to func1d(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 128)
    # Processing the call arguments (line 128)
    
    # Call to tolist(...): (line 128)
    # Processing the call keyword arguments (line 128)
    kwargs_125492 = {}
    # Getting the type of 'i' (line 128)
    i_125490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'i', False)
    # Obtaining the member 'tolist' of a type (line 128)
    tolist_125491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 35), i_125490, 'tolist')
    # Calling tolist(args, kwargs) (line 128)
    tolist_call_result_125493 = invoke(stypy.reporting.localization.Localization(__file__, 128, 35), tolist_125491, *[], **kwargs_125492)
    
    # Processing the call keyword arguments (line 128)
    kwargs_125494 = {}
    # Getting the type of 'tuple' (line 128)
    tuple_125489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'tuple', False)
    # Calling tuple(args, kwargs) (line 128)
    tuple_call_result_125495 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), tuple_125489, *[tolist_call_result_125493], **kwargs_125494)
    
    # Getting the type of 'arr' (line 128)
    arr_125496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___125497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), arr_125496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_125498 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), getitem___125497, tuple_call_result_125495)
    
    # Getting the type of 'args' (line 128)
    args_125499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'args', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'kwargs' (line 128)
    kwargs_125500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 58), 'kwargs', False)
    kwargs_125501 = {'kwargs_125500': kwargs_125500}
    # Getting the type of 'func1d' (line 128)
    func1d_125488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'func1d', False)
    # Calling func1d(args, kwargs) (line 128)
    func1d_call_result_125502 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), func1d_125488, *[subscript_call_result_125498, args_125499], **kwargs_125501)
    
    # Assigning a type to the variable 'res' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'res', func1d_call_result_125502)
    
    # Assigning a Name to a Subscript (line 129):
    
    # Assigning a Name to a Subscript (line 129):
    # Getting the type of 'res' (line 129)
    res_125503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'res')
    # Getting the type of 'outarr' (line 129)
    outarr_125504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'outarr')
    
    # Call to tuple(...): (line 129)
    # Processing the call arguments (line 129)
    
    # Call to tolist(...): (line 129)
    # Processing the call keyword arguments (line 129)
    kwargs_125508 = {}
    # Getting the type of 'i' (line 129)
    i_125506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'i', False)
    # Obtaining the member 'tolist' of a type (line 129)
    tolist_125507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), i_125506, 'tolist')
    # Calling tolist(args, kwargs) (line 129)
    tolist_call_result_125509 = invoke(stypy.reporting.localization.Localization(__file__, 129, 25), tolist_125507, *[], **kwargs_125508)
    
    # Processing the call keyword arguments (line 129)
    kwargs_125510 = {}
    # Getting the type of 'tuple' (line 129)
    tuple_125505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'tuple', False)
    # Calling tuple(args, kwargs) (line 129)
    tuple_call_result_125511 = invoke(stypy.reporting.localization.Localization(__file__, 129, 19), tuple_125505, *[tolist_call_result_125509], **kwargs_125510)
    
    # Storing an element on a container (line 129)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 12), outarr_125504, (tuple_call_result_125511, res_125503))
    
    # Getting the type of 'k' (line 130)
    k_125512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'k')
    int_125513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'int')
    # Applying the binary operator '+=' (line 130)
    result_iadd_125514 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), '+=', k_125512, int_125513)
    # Assigning a type to the variable 'k' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'k', result_iadd_125514)
    
    # SSA join for while statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'outarr' (line 131)
    outarr_125515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'outarr')
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', outarr_125515)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'apply_along_axis(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'apply_along_axis' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_125516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125516)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'apply_along_axis'
    return stypy_return_type_125516

# Assigning a type to the variable 'apply_along_axis' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'apply_along_axis', apply_along_axis)

@norecursion
def apply_over_axes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'apply_over_axes'
    module_type_store = module_type_store.open_function_context('apply_over_axes', 134, 0, False)
    
    # Passed parameters checking function
    apply_over_axes.stypy_localization = localization
    apply_over_axes.stypy_type_of_self = None
    apply_over_axes.stypy_type_store = module_type_store
    apply_over_axes.stypy_function_name = 'apply_over_axes'
    apply_over_axes.stypy_param_names_list = ['func', 'a', 'axes']
    apply_over_axes.stypy_varargs_param_name = None
    apply_over_axes.stypy_kwargs_param_name = None
    apply_over_axes.stypy_call_defaults = defaults
    apply_over_axes.stypy_call_varargs = varargs
    apply_over_axes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'apply_over_axes', ['func', 'a', 'axes'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'apply_over_axes', localization, ['func', 'a', 'axes'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'apply_over_axes(...)' code ##################

    str_125517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'str', '\n    Apply a function repeatedly over multiple axes.\n\n    `func` is called as `res = func(a, axis)`, where `axis` is the first\n    element of `axes`.  The result `res` of the function call must have\n    either the same dimensions as `a` or one less dimension.  If `res`\n    has one less dimension than `a`, a dimension is inserted before\n    `axis`.  The call to `func` is then repeated for each axis in `axes`,\n    with `res` as the first argument.\n\n    Parameters\n    ----------\n    func : function\n        This function must take two arguments, `func(a, axis)`.\n    a : array_like\n        Input array.\n    axes : array_like\n        Axes over which `func` is applied; the elements must be integers.\n\n    Returns\n    -------\n    apply_over_axis : ndarray\n        The output array.  The number of dimensions is the same as `a`,\n        but the shape can be different.  This depends on whether `func`\n        changes the shape of its output with respect to its input.\n\n    See Also\n    --------\n    apply_along_axis :\n        Apply a function to 1-D slices of an array along the given axis.\n\n    Notes\n    ------\n    This function is equivalent to tuple axis arguments to reorderable ufuncs\n    with keepdims=True. Tuple axis arguments to ufuncs have been availabe since\n    version 1.7.0.\n\n    Examples\n    --------\n    >>> a = np.arange(24).reshape(2,3,4)\n    >>> a\n    array([[[ 0,  1,  2,  3],\n            [ 4,  5,  6,  7],\n            [ 8,  9, 10, 11]],\n           [[12, 13, 14, 15],\n            [16, 17, 18, 19],\n            [20, 21, 22, 23]]])\n\n    Sum over axes 0 and 2. The result has same number of dimensions\n    as the original array:\n\n    >>> np.apply_over_axes(np.sum, a, [0,2])\n    array([[[ 60],\n            [ 92],\n            [124]]])\n\n    Tuple axis arguments to ufuncs are equivalent:\n\n    >>> np.sum(a, axis=(0,2), keepdims=True)\n    array([[[ 60],\n            [ 92],\n            [124]]])\n\n    ')
    
    # Assigning a Call to a Name (line 199):
    
    # Assigning a Call to a Name (line 199):
    
    # Call to asarray(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'a' (line 199)
    a_125519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'a', False)
    # Processing the call keyword arguments (line 199)
    kwargs_125520 = {}
    # Getting the type of 'asarray' (line 199)
    asarray_125518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 10), 'asarray', False)
    # Calling asarray(args, kwargs) (line 199)
    asarray_call_result_125521 = invoke(stypy.reporting.localization.Localization(__file__, 199, 10), asarray_125518, *[a_125519], **kwargs_125520)
    
    # Assigning a type to the variable 'val' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'val', asarray_call_result_125521)
    
    # Assigning a Attribute to a Name (line 200):
    
    # Assigning a Attribute to a Name (line 200):
    # Getting the type of 'a' (line 200)
    a_125522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'a')
    # Obtaining the member 'ndim' of a type (line 200)
    ndim_125523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), a_125522, 'ndim')
    # Assigning a type to the variable 'N' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'N', ndim_125523)
    
    
    
    # Call to array(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'axes' (line 201)
    axes_125525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'axes', False)
    # Processing the call keyword arguments (line 201)
    kwargs_125526 = {}
    # Getting the type of 'array' (line 201)
    array_125524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 7), 'array', False)
    # Calling array(args, kwargs) (line 201)
    array_call_result_125527 = invoke(stypy.reporting.localization.Localization(__file__, 201, 7), array_125524, *[axes_125525], **kwargs_125526)
    
    # Obtaining the member 'ndim' of a type (line 201)
    ndim_125528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 7), array_call_result_125527, 'ndim')
    int_125529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 27), 'int')
    # Applying the binary operator '==' (line 201)
    result_eq_125530 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 7), '==', ndim_125528, int_125529)
    
    # Testing the type of an if condition (line 201)
    if_condition_125531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 4), result_eq_125530)
    # Assigning a type to the variable 'if_condition_125531' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'if_condition_125531', if_condition_125531)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 202):
    
    # Assigning a Tuple to a Name (line 202):
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_125532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    # Getting the type of 'axes' (line 202)
    axes_125533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'axes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 16), tuple_125532, axes_125533)
    
    # Assigning a type to the variable 'axes' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'axes', tuple_125532)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'axes' (line 203)
    axes_125534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'axes')
    # Testing the type of a for loop iterable (line 203)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 4), axes_125534)
    # Getting the type of the for loop variable (line 203)
    for_loop_var_125535 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 4), axes_125534)
    # Assigning a type to the variable 'axis' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'axis', for_loop_var_125535)
    # SSA begins for a for statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'axis' (line 204)
    axis_125536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'axis')
    int_125537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 18), 'int')
    # Applying the binary operator '<' (line 204)
    result_lt_125538 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), '<', axis_125536, int_125537)
    
    # Testing the type of an if condition (line 204)
    if_condition_125539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_lt_125538)
    # Assigning a type to the variable 'if_condition_125539' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_125539', if_condition_125539)
    # SSA begins for if statement (line 204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 205):
    
    # Assigning a BinOp to a Name (line 205):
    # Getting the type of 'N' (line 205)
    N_125540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'N')
    # Getting the type of 'axis' (line 205)
    axis_125541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'axis')
    # Applying the binary operator '+' (line 205)
    result_add_125542 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 19), '+', N_125540, axis_125541)
    
    # Assigning a type to the variable 'axis' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'axis', result_add_125542)
    # SSA join for if statement (line 204)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 206):
    
    # Assigning a Tuple to a Name (line 206):
    
    # Obtaining an instance of the builtin type 'tuple' (line 206)
    tuple_125543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 206)
    # Adding element type (line 206)
    # Getting the type of 'val' (line 206)
    val_125544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'val')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 16), tuple_125543, val_125544)
    # Adding element type (line 206)
    # Getting the type of 'axis' (line 206)
    axis_125545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 16), tuple_125543, axis_125545)
    
    # Assigning a type to the variable 'args' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'args', tuple_125543)
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to func(...): (line 207)
    # Getting the type of 'args' (line 207)
    args_125547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'args', False)
    # Processing the call keyword arguments (line 207)
    kwargs_125548 = {}
    # Getting the type of 'func' (line 207)
    func_125546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 14), 'func', False)
    # Calling func(args, kwargs) (line 207)
    func_call_result_125549 = invoke(stypy.reporting.localization.Localization(__file__, 207, 14), func_125546, *[args_125547], **kwargs_125548)
    
    # Assigning a type to the variable 'res' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'res', func_call_result_125549)
    
    
    # Getting the type of 'res' (line 208)
    res_125550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'res')
    # Obtaining the member 'ndim' of a type (line 208)
    ndim_125551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), res_125550, 'ndim')
    # Getting the type of 'val' (line 208)
    val_125552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'val')
    # Obtaining the member 'ndim' of a type (line 208)
    ndim_125553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 23), val_125552, 'ndim')
    # Applying the binary operator '==' (line 208)
    result_eq_125554 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), '==', ndim_125551, ndim_125553)
    
    # Testing the type of an if condition (line 208)
    if_condition_125555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_eq_125554)
    # Assigning a type to the variable 'if_condition_125555' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_125555', if_condition_125555)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 209):
    
    # Assigning a Name to a Name (line 209):
    # Getting the type of 'res' (line 209)
    res_125556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'res')
    # Assigning a type to the variable 'val' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'val', res_125556)
    # SSA branch for the else part of an if statement (line 208)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to expand_dims(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'res' (line 211)
    res_125558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 'res', False)
    # Getting the type of 'axis' (line 211)
    axis_125559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'axis', False)
    # Processing the call keyword arguments (line 211)
    kwargs_125560 = {}
    # Getting the type of 'expand_dims' (line 211)
    expand_dims_125557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'expand_dims', False)
    # Calling expand_dims(args, kwargs) (line 211)
    expand_dims_call_result_125561 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), expand_dims_125557, *[res_125558, axis_125559], **kwargs_125560)
    
    # Assigning a type to the variable 'res' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'res', expand_dims_call_result_125561)
    
    
    # Getting the type of 'res' (line 212)
    res_125562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'res')
    # Obtaining the member 'ndim' of a type (line 212)
    ndim_125563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), res_125562, 'ndim')
    # Getting the type of 'val' (line 212)
    val_125564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'val')
    # Obtaining the member 'ndim' of a type (line 212)
    ndim_125565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 27), val_125564, 'ndim')
    # Applying the binary operator '==' (line 212)
    result_eq_125566 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 15), '==', ndim_125563, ndim_125565)
    
    # Testing the type of an if condition (line 212)
    if_condition_125567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), result_eq_125566)
    # Assigning a type to the variable 'if_condition_125567' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_125567', if_condition_125567)
    # SSA begins for if statement (line 212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 213):
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'res' (line 213)
    res_125568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'res')
    # Assigning a type to the variable 'val' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'val', res_125568)
    # SSA branch for the else part of an if statement (line 212)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 215)
    # Processing the call arguments (line 215)
    str_125570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 33), 'str', 'function is not returning an array of the correct shape')
    # Processing the call keyword arguments (line 215)
    kwargs_125571 = {}
    # Getting the type of 'ValueError' (line 215)
    ValueError_125569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 215)
    ValueError_call_result_125572 = invoke(stypy.reporting.localization.Localization(__file__, 215, 22), ValueError_125569, *[str_125570], **kwargs_125571)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 16), ValueError_call_result_125572, 'raise parameter', BaseException)
    # SSA join for if statement (line 212)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'val' (line 217)
    val_125573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'val')
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', val_125573)
    
    # ################# End of 'apply_over_axes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'apply_over_axes' in the type store
    # Getting the type of 'stypy_return_type' (line 134)
    stypy_return_type_125574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125574)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'apply_over_axes'
    return stypy_return_type_125574

# Assigning a type to the variable 'apply_over_axes' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'apply_over_axes', apply_over_axes)

@norecursion
def expand_dims(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expand_dims'
    module_type_store = module_type_store.open_function_context('expand_dims', 219, 0, False)
    
    # Passed parameters checking function
    expand_dims.stypy_localization = localization
    expand_dims.stypy_type_of_self = None
    expand_dims.stypy_type_store = module_type_store
    expand_dims.stypy_function_name = 'expand_dims'
    expand_dims.stypy_param_names_list = ['a', 'axis']
    expand_dims.stypy_varargs_param_name = None
    expand_dims.stypy_kwargs_param_name = None
    expand_dims.stypy_call_defaults = defaults
    expand_dims.stypy_call_varargs = varargs
    expand_dims.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expand_dims', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expand_dims', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expand_dims(...)' code ##################

    str_125575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, (-1)), 'str', '\n    Expand the shape of an array.\n\n    Insert a new axis, corresponding to a given position in the array shape.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    axis : int\n        Position (amongst axes) where new axis is to be inserted.\n\n    Returns\n    -------\n    res : ndarray\n        Output array. The number of dimensions is one greater than that of\n        the input array.\n\n    See Also\n    --------\n    doc.indexing, atleast_1d, atleast_2d, atleast_3d\n\n    Examples\n    --------\n    >>> x = np.array([1,2])\n    >>> x.shape\n    (2,)\n\n    The following is equivalent to ``x[np.newaxis,:]`` or ``x[np.newaxis]``:\n\n    >>> y = np.expand_dims(x, axis=0)\n    >>> y\n    array([[1, 2]])\n    >>> y.shape\n    (1, 2)\n\n    >>> y = np.expand_dims(x, axis=1)  # Equivalent to x[:,newaxis]\n    >>> y\n    array([[1],\n           [2]])\n    >>> y.shape\n    (2, 1)\n\n    Note that some examples may use ``None`` instead of ``np.newaxis``.  These\n    are the same objects:\n\n    >>> np.newaxis is None\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to asarray(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'a' (line 270)
    a_125577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'a', False)
    # Processing the call keyword arguments (line 270)
    kwargs_125578 = {}
    # Getting the type of 'asarray' (line 270)
    asarray_125576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 270)
    asarray_call_result_125579 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), asarray_125576, *[a_125577], **kwargs_125578)
    
    # Assigning a type to the variable 'a' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'a', asarray_call_result_125579)
    
    # Assigning a Attribute to a Name (line 271):
    
    # Assigning a Attribute to a Name (line 271):
    # Getting the type of 'a' (line 271)
    a_125580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'a')
    # Obtaining the member 'shape' of a type (line 271)
    shape_125581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), a_125580, 'shape')
    # Assigning a type to the variable 'shape' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'shape', shape_125581)
    
    
    # Getting the type of 'axis' (line 272)
    axis_125582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 7), 'axis')
    int_125583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 14), 'int')
    # Applying the binary operator '<' (line 272)
    result_lt_125584 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 7), '<', axis_125582, int_125583)
    
    # Testing the type of an if condition (line 272)
    if_condition_125585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 4), result_lt_125584)
    # Assigning a type to the variable 'if_condition_125585' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'if_condition_125585', if_condition_125585)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 273):
    
    # Assigning a BinOp to a Name (line 273):
    # Getting the type of 'axis' (line 273)
    axis_125586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'axis')
    
    # Call to len(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'shape' (line 273)
    shape_125588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 26), 'shape', False)
    # Processing the call keyword arguments (line 273)
    kwargs_125589 = {}
    # Getting the type of 'len' (line 273)
    len_125587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'len', False)
    # Calling len(args, kwargs) (line 273)
    len_call_result_125590 = invoke(stypy.reporting.localization.Localization(__file__, 273, 22), len_125587, *[shape_125588], **kwargs_125589)
    
    # Applying the binary operator '+' (line 273)
    result_add_125591 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 15), '+', axis_125586, len_call_result_125590)
    
    int_125592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 35), 'int')
    # Applying the binary operator '+' (line 273)
    result_add_125593 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 33), '+', result_add_125591, int_125592)
    
    # Assigning a type to the variable 'axis' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'axis', result_add_125593)
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 274)
    # Processing the call arguments (line 274)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 274)
    axis_125596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 28), 'axis', False)
    slice_125597 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 21), None, axis_125596, None)
    # Getting the type of 'shape' (line 274)
    shape_125598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___125599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), shape_125598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_125600 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), getitem___125599, slice_125597)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 274)
    tuple_125601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 274)
    # Adding element type (line 274)
    int_125602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 37), tuple_125601, int_125602)
    
    # Applying the binary operator '+' (line 274)
    result_add_125603 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 21), '+', subscript_call_result_125600, tuple_125601)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 274)
    axis_125604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 49), 'axis', False)
    slice_125605 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 43), axis_125604, None, None)
    # Getting the type of 'shape' (line 274)
    shape_125606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 43), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 274)
    getitem___125607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 43), shape_125606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 274)
    subscript_call_result_125608 = invoke(stypy.reporting.localization.Localization(__file__, 274, 43), getitem___125607, slice_125605)
    
    # Applying the binary operator '+' (line 274)
    result_add_125609 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 41), '+', result_add_125603, subscript_call_result_125608)
    
    # Processing the call keyword arguments (line 274)
    kwargs_125610 = {}
    # Getting the type of 'a' (line 274)
    a_125594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 11), 'a', False)
    # Obtaining the member 'reshape' of a type (line 274)
    reshape_125595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 11), a_125594, 'reshape')
    # Calling reshape(args, kwargs) (line 274)
    reshape_call_result_125611 = invoke(stypy.reporting.localization.Localization(__file__, 274, 11), reshape_125595, *[result_add_125609], **kwargs_125610)
    
    # Assigning a type to the variable 'stypy_return_type' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type', reshape_call_result_125611)
    
    # ################# End of 'expand_dims(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expand_dims' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_125612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125612)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expand_dims'
    return stypy_return_type_125612

# Assigning a type to the variable 'expand_dims' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'expand_dims', expand_dims)

# Assigning a Name to a Name (line 276):

# Assigning a Name to a Name (line 276):
# Getting the type of 'vstack' (line 276)
vstack_125613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'vstack')
# Assigning a type to the variable 'row_stack' (line 276)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 0), 'row_stack', vstack_125613)

@norecursion
def column_stack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'column_stack'
    module_type_store = module_type_store.open_function_context('column_stack', 278, 0, False)
    
    # Passed parameters checking function
    column_stack.stypy_localization = localization
    column_stack.stypy_type_of_self = None
    column_stack.stypy_type_store = module_type_store
    column_stack.stypy_function_name = 'column_stack'
    column_stack.stypy_param_names_list = ['tup']
    column_stack.stypy_varargs_param_name = None
    column_stack.stypy_kwargs_param_name = None
    column_stack.stypy_call_defaults = defaults
    column_stack.stypy_call_varargs = varargs
    column_stack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'column_stack', ['tup'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'column_stack', localization, ['tup'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'column_stack(...)' code ##################

    str_125614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, (-1)), 'str', '\n    Stack 1-D arrays as columns into a 2-D array.\n\n    Take a sequence of 1-D arrays and stack them as columns\n    to make a single 2-D array. 2-D arrays are stacked as-is,\n    just like with `hstack`.  1-D arrays are turned into 2-D columns\n    first.\n\n    Parameters\n    ----------\n    tup : sequence of 1-D or 2-D arrays.\n        Arrays to stack. All of them must have the same first dimension.\n\n    Returns\n    -------\n    stacked : 2-D array\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    hstack, vstack, concatenate\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((2,3,4))\n    >>> np.column_stack((a,b))\n    array([[1, 2],\n           [2, 3],\n           [3, 4]])\n\n    ')
    
    # Assigning a List to a Name (line 311):
    
    # Assigning a List to a Name (line 311):
    
    # Obtaining an instance of the builtin type 'list' (line 311)
    list_125615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 311)
    
    # Assigning a type to the variable 'arrays' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'arrays', list_125615)
    
    # Getting the type of 'tup' (line 312)
    tup_125616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'tup')
    # Testing the type of a for loop iterable (line 312)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 4), tup_125616)
    # Getting the type of the for loop variable (line 312)
    for_loop_var_125617 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 4), tup_125616)
    # Assigning a type to the variable 'v' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'v', for_loop_var_125617)
    # SSA begins for a for statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to array(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'v' (line 313)
    v_125619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'v', False)
    # Processing the call keyword arguments (line 313)
    # Getting the type of 'False' (line 313)
    False_125620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'False', False)
    keyword_125621 = False_125620
    # Getting the type of 'True' (line 313)
    True_125622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 41), 'True', False)
    keyword_125623 = True_125622
    kwargs_125624 = {'subok': keyword_125623, 'copy': keyword_125621}
    # Getting the type of 'array' (line 313)
    array_125618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 14), 'array', False)
    # Calling array(args, kwargs) (line 313)
    array_call_result_125625 = invoke(stypy.reporting.localization.Localization(__file__, 313, 14), array_125618, *[v_125619], **kwargs_125624)
    
    # Assigning a type to the variable 'arr' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'arr', array_call_result_125625)
    
    
    # Getting the type of 'arr' (line 314)
    arr_125626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 11), 'arr')
    # Obtaining the member 'ndim' of a type (line 314)
    ndim_125627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 11), arr_125626, 'ndim')
    int_125628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 22), 'int')
    # Applying the binary operator '<' (line 314)
    result_lt_125629 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 11), '<', ndim_125627, int_125628)
    
    # Testing the type of an if condition (line 314)
    if_condition_125630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 314, 8), result_lt_125629)
    # Assigning a type to the variable 'if_condition_125630' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'if_condition_125630', if_condition_125630)
    # SSA begins for if statement (line 314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 315):
    
    # Assigning a Attribute to a Name (line 315):
    
    # Call to array(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'arr' (line 315)
    arr_125632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'arr', False)
    # Processing the call keyword arguments (line 315)
    # Getting the type of 'False' (line 315)
    False_125633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 34), 'False', False)
    keyword_125634 = False_125633
    # Getting the type of 'True' (line 315)
    True_125635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 47), 'True', False)
    keyword_125636 = True_125635
    int_125637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 59), 'int')
    keyword_125638 = int_125637
    kwargs_125639 = {'subok': keyword_125636, 'copy': keyword_125634, 'ndmin': keyword_125638}
    # Getting the type of 'array' (line 315)
    array_125631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 18), 'array', False)
    # Calling array(args, kwargs) (line 315)
    array_call_result_125640 = invoke(stypy.reporting.localization.Localization(__file__, 315, 18), array_125631, *[arr_125632], **kwargs_125639)
    
    # Obtaining the member 'T' of a type (line 315)
    T_125641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 18), array_call_result_125640, 'T')
    # Assigning a type to the variable 'arr' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'arr', T_125641)
    # SSA join for if statement (line 314)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'arr' (line 316)
    arr_125644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'arr', False)
    # Processing the call keyword arguments (line 316)
    kwargs_125645 = {}
    # Getting the type of 'arrays' (line 316)
    arrays_125642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'arrays', False)
    # Obtaining the member 'append' of a type (line 316)
    append_125643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), arrays_125642, 'append')
    # Calling append(args, kwargs) (line 316)
    append_call_result_125646 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), append_125643, *[arr_125644], **kwargs_125645)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to concatenate(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'arrays' (line 317)
    arrays_125649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 27), 'arrays', False)
    int_125650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 35), 'int')
    # Processing the call keyword arguments (line 317)
    kwargs_125651 = {}
    # Getting the type of '_nx' (line 317)
    _nx_125647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 317)
    concatenate_125648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), _nx_125647, 'concatenate')
    # Calling concatenate(args, kwargs) (line 317)
    concatenate_call_result_125652 = invoke(stypy.reporting.localization.Localization(__file__, 317, 11), concatenate_125648, *[arrays_125649, int_125650], **kwargs_125651)
    
    # Assigning a type to the variable 'stypy_return_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type', concatenate_call_result_125652)
    
    # ################# End of 'column_stack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'column_stack' in the type store
    # Getting the type of 'stypy_return_type' (line 278)
    stypy_return_type_125653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125653)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'column_stack'
    return stypy_return_type_125653

# Assigning a type to the variable 'column_stack' (line 278)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'column_stack', column_stack)

@norecursion
def dstack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dstack'
    module_type_store = module_type_store.open_function_context('dstack', 319, 0, False)
    
    # Passed parameters checking function
    dstack.stypy_localization = localization
    dstack.stypy_type_of_self = None
    dstack.stypy_type_store = module_type_store
    dstack.stypy_function_name = 'dstack'
    dstack.stypy_param_names_list = ['tup']
    dstack.stypy_varargs_param_name = None
    dstack.stypy_kwargs_param_name = None
    dstack.stypy_call_defaults = defaults
    dstack.stypy_call_varargs = varargs
    dstack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dstack', ['tup'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dstack', localization, ['tup'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dstack(...)' code ##################

    str_125654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, (-1)), 'str', '\n    Stack arrays in sequence depth wise (along third axis).\n\n    Takes a sequence of arrays and stack them along the third axis\n    to make a single array. Rebuilds arrays divided by `dsplit`.\n    This is a simple way to stack 2D arrays (images) into a single\n    3D array for processing.\n\n    Parameters\n    ----------\n    tup : sequence of arrays\n        Arrays to stack. All of them must have the same shape along all\n        but the third axis.\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    stack : Join a sequence of arrays along a new axis.\n    vstack : Stack along first axis.\n    hstack : Stack along second axis.\n    concatenate : Join a sequence of arrays along an existing axis.\n    dsplit : Split array along third axis.\n\n    Notes\n    -----\n    Equivalent to ``np.concatenate(tup, axis=2)``.\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((2,3,4))\n    >>> np.dstack((a,b))\n    array([[[1, 2],\n            [2, 3],\n            [3, 4]]])\n\n    >>> a = np.array([[1],[2],[3]])\n    >>> b = np.array([[2],[3],[4]])\n    >>> np.dstack((a,b))\n    array([[[1, 2]],\n           [[2, 3]],\n           [[3, 4]]])\n\n    ')
    
    # Call to concatenate(...): (line 368)
    # Processing the call arguments (line 368)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'tup' (line 368)
    tup_125661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 53), 'tup', False)
    comprehension_125662 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 28), tup_125661)
    # Assigning a type to the variable '_m' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 28), '_m', comprehension_125662)
    
    # Call to atleast_3d(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of '_m' (line 368)
    _m_125658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 39), '_m', False)
    # Processing the call keyword arguments (line 368)
    kwargs_125659 = {}
    # Getting the type of 'atleast_3d' (line 368)
    atleast_3d_125657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 28), 'atleast_3d', False)
    # Calling atleast_3d(args, kwargs) (line 368)
    atleast_3d_call_result_125660 = invoke(stypy.reporting.localization.Localization(__file__, 368, 28), atleast_3d_125657, *[_m_125658], **kwargs_125659)
    
    list_125663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 28), list_125663, atleast_3d_call_result_125660)
    int_125664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 59), 'int')
    # Processing the call keyword arguments (line 368)
    kwargs_125665 = {}
    # Getting the type of '_nx' (line 368)
    _nx_125655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 368)
    concatenate_125656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 11), _nx_125655, 'concatenate')
    # Calling concatenate(args, kwargs) (line 368)
    concatenate_call_result_125666 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), concatenate_125656, *[list_125663, int_125664], **kwargs_125665)
    
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type', concatenate_call_result_125666)
    
    # ################# End of 'dstack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dstack' in the type store
    # Getting the type of 'stypy_return_type' (line 319)
    stypy_return_type_125667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125667)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dstack'
    return stypy_return_type_125667

# Assigning a type to the variable 'dstack' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'dstack', dstack)

@norecursion
def _replace_zero_by_x_arrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_replace_zero_by_x_arrays'
    module_type_store = module_type_store.open_function_context('_replace_zero_by_x_arrays', 370, 0, False)
    
    # Passed parameters checking function
    _replace_zero_by_x_arrays.stypy_localization = localization
    _replace_zero_by_x_arrays.stypy_type_of_self = None
    _replace_zero_by_x_arrays.stypy_type_store = module_type_store
    _replace_zero_by_x_arrays.stypy_function_name = '_replace_zero_by_x_arrays'
    _replace_zero_by_x_arrays.stypy_param_names_list = ['sub_arys']
    _replace_zero_by_x_arrays.stypy_varargs_param_name = None
    _replace_zero_by_x_arrays.stypy_kwargs_param_name = None
    _replace_zero_by_x_arrays.stypy_call_defaults = defaults
    _replace_zero_by_x_arrays.stypy_call_varargs = varargs
    _replace_zero_by_x_arrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_replace_zero_by_x_arrays', ['sub_arys'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_replace_zero_by_x_arrays', localization, ['sub_arys'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_replace_zero_by_x_arrays(...)' code ##################

    
    
    # Call to range(...): (line 371)
    # Processing the call arguments (line 371)
    
    # Call to len(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'sub_arys' (line 371)
    sub_arys_125670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 23), 'sub_arys', False)
    # Processing the call keyword arguments (line 371)
    kwargs_125671 = {}
    # Getting the type of 'len' (line 371)
    len_125669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 19), 'len', False)
    # Calling len(args, kwargs) (line 371)
    len_call_result_125672 = invoke(stypy.reporting.localization.Localization(__file__, 371, 19), len_125669, *[sub_arys_125670], **kwargs_125671)
    
    # Processing the call keyword arguments (line 371)
    kwargs_125673 = {}
    # Getting the type of 'range' (line 371)
    range_125668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'range', False)
    # Calling range(args, kwargs) (line 371)
    range_call_result_125674 = invoke(stypy.reporting.localization.Localization(__file__, 371, 13), range_125668, *[len_call_result_125672], **kwargs_125673)
    
    # Testing the type of a for loop iterable (line 371)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 371, 4), range_call_result_125674)
    # Getting the type of the for loop variable (line 371)
    for_loop_var_125675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 371, 4), range_call_result_125674)
    # Assigning a type to the variable 'i' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'i', for_loop_var_125675)
    # SSA begins for a for statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to len(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Call to shape(...): (line 372)
    # Processing the call arguments (line 372)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 372)
    i_125679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'i', False)
    # Getting the type of 'sub_arys' (line 372)
    sub_arys_125680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 25), 'sub_arys', False)
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___125681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 25), sub_arys_125680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_125682 = invoke(stypy.reporting.localization.Localization(__file__, 372, 25), getitem___125681, i_125679)
    
    # Processing the call keyword arguments (line 372)
    kwargs_125683 = {}
    # Getting the type of '_nx' (line 372)
    _nx_125677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), '_nx', False)
    # Obtaining the member 'shape' of a type (line 372)
    shape_125678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 15), _nx_125677, 'shape')
    # Calling shape(args, kwargs) (line 372)
    shape_call_result_125684 = invoke(stypy.reporting.localization.Localization(__file__, 372, 15), shape_125678, *[subscript_call_result_125682], **kwargs_125683)
    
    # Processing the call keyword arguments (line 372)
    kwargs_125685 = {}
    # Getting the type of 'len' (line 372)
    len_125676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'len', False)
    # Calling len(args, kwargs) (line 372)
    len_call_result_125686 = invoke(stypy.reporting.localization.Localization(__file__, 372, 11), len_125676, *[shape_call_result_125684], **kwargs_125685)
    
    int_125687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 42), 'int')
    # Applying the binary operator '==' (line 372)
    result_eq_125688 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), '==', len_call_result_125686, int_125687)
    
    # Testing the type of an if condition (line 372)
    if_condition_125689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), result_eq_125688)
    # Assigning a type to the variable 'if_condition_125689' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_125689', if_condition_125689)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 373):
    
    # Assigning a Call to a Subscript (line 373):
    
    # Call to empty(...): (line 373)
    # Processing the call arguments (line 373)
    int_125692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 36), 'int')
    # Processing the call keyword arguments (line 373)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 373)
    i_125693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 54), 'i', False)
    # Getting the type of 'sub_arys' (line 373)
    sub_arys_125694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 45), 'sub_arys', False)
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___125695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 45), sub_arys_125694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_125696 = invoke(stypy.reporting.localization.Localization(__file__, 373, 45), getitem___125695, i_125693)
    
    # Obtaining the member 'dtype' of a type (line 373)
    dtype_125697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 45), subscript_call_result_125696, 'dtype')
    keyword_125698 = dtype_125697
    kwargs_125699 = {'dtype': keyword_125698}
    # Getting the type of '_nx' (line 373)
    _nx_125690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), '_nx', False)
    # Obtaining the member 'empty' of a type (line 373)
    empty_125691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 26), _nx_125690, 'empty')
    # Calling empty(args, kwargs) (line 373)
    empty_call_result_125700 = invoke(stypy.reporting.localization.Localization(__file__, 373, 26), empty_125691, *[int_125692], **kwargs_125699)
    
    # Getting the type of 'sub_arys' (line 373)
    sub_arys_125701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'sub_arys')
    # Getting the type of 'i' (line 373)
    i_125702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'i')
    # Storing an element on a container (line 373)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), sub_arys_125701, (i_125702, empty_call_result_125700))
    # SSA branch for the else part of an if statement (line 372)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to sometrue(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Call to equal(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Call to shape(...): (line 374)
    # Processing the call arguments (line 374)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 374)
    i_125709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 55), 'i', False)
    # Getting the type of 'sub_arys' (line 374)
    sub_arys_125710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 46), 'sub_arys', False)
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___125711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 46), sub_arys_125710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_125712 = invoke(stypy.reporting.localization.Localization(__file__, 374, 46), getitem___125711, i_125709)
    
    # Processing the call keyword arguments (line 374)
    kwargs_125713 = {}
    # Getting the type of '_nx' (line 374)
    _nx_125707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 36), '_nx', False)
    # Obtaining the member 'shape' of a type (line 374)
    shape_125708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 36), _nx_125707, 'shape')
    # Calling shape(args, kwargs) (line 374)
    shape_call_result_125714 = invoke(stypy.reporting.localization.Localization(__file__, 374, 36), shape_125708, *[subscript_call_result_125712], **kwargs_125713)
    
    int_125715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 60), 'int')
    # Processing the call keyword arguments (line 374)
    kwargs_125716 = {}
    # Getting the type of '_nx' (line 374)
    _nx_125705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 26), '_nx', False)
    # Obtaining the member 'equal' of a type (line 374)
    equal_125706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 26), _nx_125705, 'equal')
    # Calling equal(args, kwargs) (line 374)
    equal_call_result_125717 = invoke(stypy.reporting.localization.Localization(__file__, 374, 26), equal_125706, *[shape_call_result_125714, int_125715], **kwargs_125716)
    
    # Processing the call keyword arguments (line 374)
    kwargs_125718 = {}
    # Getting the type of '_nx' (line 374)
    _nx_125703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 13), '_nx', False)
    # Obtaining the member 'sometrue' of a type (line 374)
    sometrue_125704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 13), _nx_125703, 'sometrue')
    # Calling sometrue(args, kwargs) (line 374)
    sometrue_call_result_125719 = invoke(stypy.reporting.localization.Localization(__file__, 374, 13), sometrue_125704, *[equal_call_result_125717], **kwargs_125718)
    
    # Testing the type of an if condition (line 374)
    if_condition_125720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 13), sometrue_call_result_125719)
    # Assigning a type to the variable 'if_condition_125720' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 13), 'if_condition_125720', if_condition_125720)
    # SSA begins for if statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 375):
    
    # Assigning a Call to a Subscript (line 375):
    
    # Call to empty(...): (line 375)
    # Processing the call arguments (line 375)
    int_125723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 36), 'int')
    # Processing the call keyword arguments (line 375)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 375)
    i_125724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 54), 'i', False)
    # Getting the type of 'sub_arys' (line 375)
    sub_arys_125725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 45), 'sub_arys', False)
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___125726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 45), sub_arys_125725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 375)
    subscript_call_result_125727 = invoke(stypy.reporting.localization.Localization(__file__, 375, 45), getitem___125726, i_125724)
    
    # Obtaining the member 'dtype' of a type (line 375)
    dtype_125728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 45), subscript_call_result_125727, 'dtype')
    keyword_125729 = dtype_125728
    kwargs_125730 = {'dtype': keyword_125729}
    # Getting the type of '_nx' (line 375)
    _nx_125721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 26), '_nx', False)
    # Obtaining the member 'empty' of a type (line 375)
    empty_125722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 26), _nx_125721, 'empty')
    # Calling empty(args, kwargs) (line 375)
    empty_call_result_125731 = invoke(stypy.reporting.localization.Localization(__file__, 375, 26), empty_125722, *[int_125723], **kwargs_125730)
    
    # Getting the type of 'sub_arys' (line 375)
    sub_arys_125732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'sub_arys')
    # Getting the type of 'i' (line 375)
    i_125733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 'i')
    # Storing an element on a container (line 375)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), sub_arys_125732, (i_125733, empty_call_result_125731))
    # SSA join for if statement (line 374)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sub_arys' (line 376)
    sub_arys_125734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'sub_arys')
    # Assigning a type to the variable 'stypy_return_type' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type', sub_arys_125734)
    
    # ################# End of '_replace_zero_by_x_arrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_replace_zero_by_x_arrays' in the type store
    # Getting the type of 'stypy_return_type' (line 370)
    stypy_return_type_125735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125735)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_replace_zero_by_x_arrays'
    return stypy_return_type_125735

# Assigning a type to the variable '_replace_zero_by_x_arrays' (line 370)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 0), '_replace_zero_by_x_arrays', _replace_zero_by_x_arrays)

@norecursion
def array_split(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_125736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 47), 'int')
    defaults = [int_125736]
    # Create a new context for function 'array_split'
    module_type_store = module_type_store.open_function_context('array_split', 378, 0, False)
    
    # Passed parameters checking function
    array_split.stypy_localization = localization
    array_split.stypy_type_of_self = None
    array_split.stypy_type_store = module_type_store
    array_split.stypy_function_name = 'array_split'
    array_split.stypy_param_names_list = ['ary', 'indices_or_sections', 'axis']
    array_split.stypy_varargs_param_name = None
    array_split.stypy_kwargs_param_name = None
    array_split.stypy_call_defaults = defaults
    array_split.stypy_call_varargs = varargs
    array_split.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'array_split', ['ary', 'indices_or_sections', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'array_split', localization, ['ary', 'indices_or_sections', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'array_split(...)' code ##################

    str_125737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, (-1)), 'str', '\n    Split an array into multiple sub-arrays.\n\n    Please refer to the ``split`` documentation.  The only difference\n    between these functions is that ``array_split`` allows\n    `indices_or_sections` to be an integer that does *not* equally\n    divide the axis.\n\n    See Also\n    --------\n    split : Split array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(8.0)\n    >>> np.array_split(x, 3)\n        [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.])]\n\n    ')
    
    
    # SSA begins for try-except statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 399):
    
    # Assigning a Subscript to a Name (line 399):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 399)
    axis_125738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'axis')
    # Getting the type of 'ary' (line 399)
    ary_125739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'ary')
    # Obtaining the member 'shape' of a type (line 399)
    shape_125740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 17), ary_125739, 'shape')
    # Obtaining the member '__getitem__' of a type (line 399)
    getitem___125741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 17), shape_125740, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 399)
    subscript_call_result_125742 = invoke(stypy.reporting.localization.Localization(__file__, 399, 17), getitem___125741, axis_125738)
    
    # Assigning a type to the variable 'Ntotal' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'Ntotal', subscript_call_result_125742)
    # SSA branch for the except part of a try statement (line 398)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 398)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 401):
    
    # Assigning a Call to a Name (line 401):
    
    # Call to len(...): (line 401)
    # Processing the call arguments (line 401)
    # Getting the type of 'ary' (line 401)
    ary_125744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'ary', False)
    # Processing the call keyword arguments (line 401)
    kwargs_125745 = {}
    # Getting the type of 'len' (line 401)
    len_125743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 17), 'len', False)
    # Calling len(args, kwargs) (line 401)
    len_call_result_125746 = invoke(stypy.reporting.localization.Localization(__file__, 401, 17), len_125743, *[ary_125744], **kwargs_125745)
    
    # Assigning a type to the variable 'Ntotal' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'Ntotal', len_call_result_125746)
    # SSA join for try-except statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a BinOp to a Name (line 404):
    
    # Assigning a BinOp to a Name (line 404):
    
    # Call to len(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'indices_or_sections' (line 404)
    indices_or_sections_125748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 24), 'indices_or_sections', False)
    # Processing the call keyword arguments (line 404)
    kwargs_125749 = {}
    # Getting the type of 'len' (line 404)
    len_125747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'len', False)
    # Calling len(args, kwargs) (line 404)
    len_call_result_125750 = invoke(stypy.reporting.localization.Localization(__file__, 404, 20), len_125747, *[indices_or_sections_125748], **kwargs_125749)
    
    int_125751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 47), 'int')
    # Applying the binary operator '+' (line 404)
    result_add_125752 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 20), '+', len_call_result_125750, int_125751)
    
    # Assigning a type to the variable 'Nsections' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'Nsections', result_add_125752)
    
    # Assigning a BinOp to a Name (line 405):
    
    # Assigning a BinOp to a Name (line 405):
    
    # Obtaining an instance of the builtin type 'list' (line 405)
    list_125753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 405)
    # Adding element type (line 405)
    int_125754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 21), list_125753, int_125754)
    
    
    # Call to list(...): (line 405)
    # Processing the call arguments (line 405)
    # Getting the type of 'indices_or_sections' (line 405)
    indices_or_sections_125756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 32), 'indices_or_sections', False)
    # Processing the call keyword arguments (line 405)
    kwargs_125757 = {}
    # Getting the type of 'list' (line 405)
    list_125755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'list', False)
    # Calling list(args, kwargs) (line 405)
    list_call_result_125758 = invoke(stypy.reporting.localization.Localization(__file__, 405, 27), list_125755, *[indices_or_sections_125756], **kwargs_125757)
    
    # Applying the binary operator '+' (line 405)
    result_add_125759 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 21), '+', list_125753, list_call_result_125758)
    
    
    # Obtaining an instance of the builtin type 'list' (line 405)
    list_125760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 405)
    # Adding element type (line 405)
    # Getting the type of 'Ntotal' (line 405)
    Ntotal_125761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 56), 'Ntotal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 55), list_125760, Ntotal_125761)
    
    # Applying the binary operator '+' (line 405)
    result_add_125762 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 53), '+', result_add_125759, list_125760)
    
    # Assigning a type to the variable 'div_points' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'div_points', result_add_125762)
    # SSA branch for the except part of a try statement (line 402)
    # SSA branch for the except 'TypeError' branch of a try statement (line 402)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to int(...): (line 408)
    # Processing the call arguments (line 408)
    # Getting the type of 'indices_or_sections' (line 408)
    indices_or_sections_125764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'indices_or_sections', False)
    # Processing the call keyword arguments (line 408)
    kwargs_125765 = {}
    # Getting the type of 'int' (line 408)
    int_125763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'int', False)
    # Calling int(args, kwargs) (line 408)
    int_call_result_125766 = invoke(stypy.reporting.localization.Localization(__file__, 408, 20), int_125763, *[indices_or_sections_125764], **kwargs_125765)
    
    # Assigning a type to the variable 'Nsections' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'Nsections', int_call_result_125766)
    
    
    # Getting the type of 'Nsections' (line 409)
    Nsections_125767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'Nsections')
    int_125768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 24), 'int')
    # Applying the binary operator '<=' (line 409)
    result_le_125769 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 11), '<=', Nsections_125767, int_125768)
    
    # Testing the type of an if condition (line 409)
    if_condition_125770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), result_le_125769)
    # Assigning a type to the variable 'if_condition_125770' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_125770', if_condition_125770)
    # SSA begins for if statement (line 409)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 410)
    # Processing the call arguments (line 410)
    str_125772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 29), 'str', 'number sections must be larger than 0.')
    # Processing the call keyword arguments (line 410)
    kwargs_125773 = {}
    # Getting the type of 'ValueError' (line 410)
    ValueError_125771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 410)
    ValueError_call_result_125774 = invoke(stypy.reporting.localization.Localization(__file__, 410, 18), ValueError_125771, *[str_125772], **kwargs_125773)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 410, 12), ValueError_call_result_125774, 'raise parameter', BaseException)
    # SSA join for if statement (line 409)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 411):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'Ntotal' (line 411)
    Ntotal_125776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 39), 'Ntotal', False)
    # Getting the type of 'Nsections' (line 411)
    Nsections_125777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 47), 'Nsections', False)
    # Processing the call keyword arguments (line 411)
    kwargs_125778 = {}
    # Getting the type of 'divmod' (line 411)
    divmod_125775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 32), 'divmod', False)
    # Calling divmod(args, kwargs) (line 411)
    divmod_call_result_125779 = invoke(stypy.reporting.localization.Localization(__file__, 411, 32), divmod_125775, *[Ntotal_125776, Nsections_125777], **kwargs_125778)
    
    # Assigning a type to the variable 'call_assignment_125183' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125183', divmod_call_result_125779)
    
    # Assigning a Call to a Name (line 411):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_125782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
    # Processing the call keyword arguments
    kwargs_125783 = {}
    # Getting the type of 'call_assignment_125183' (line 411)
    call_assignment_125183_125780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125183', False)
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___125781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), call_assignment_125183_125780, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_125784 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___125781, *[int_125782], **kwargs_125783)
    
    # Assigning a type to the variable 'call_assignment_125184' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125184', getitem___call_result_125784)
    
    # Assigning a Name to a Name (line 411):
    # Getting the type of 'call_assignment_125184' (line 411)
    call_assignment_125184_125785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125184')
    # Assigning a type to the variable 'Neach_section' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'Neach_section', call_assignment_125184_125785)
    
    # Assigning a Call to a Name (line 411):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_125788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
    # Processing the call keyword arguments
    kwargs_125789 = {}
    # Getting the type of 'call_assignment_125183' (line 411)
    call_assignment_125183_125786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125183', False)
    # Obtaining the member '__getitem__' of a type (line 411)
    getitem___125787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), call_assignment_125183_125786, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_125790 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___125787, *[int_125788], **kwargs_125789)
    
    # Assigning a type to the variable 'call_assignment_125185' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125185', getitem___call_result_125790)
    
    # Assigning a Name to a Name (line 411):
    # Getting the type of 'call_assignment_125185' (line 411)
    call_assignment_125185_125791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'call_assignment_125185')
    # Assigning a type to the variable 'extras' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 23), 'extras', call_assignment_125185_125791)
    
    # Assigning a BinOp to a Name (line 412):
    
    # Assigning a BinOp to a Name (line 412):
    
    # Obtaining an instance of the builtin type 'list' (line 412)
    list_125792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 412)
    # Adding element type (line 412)
    int_125793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 25), list_125792, int_125793)
    
    # Getting the type of 'extras' (line 413)
    extras_125794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 25), 'extras')
    
    # Obtaining an instance of the builtin type 'list' (line 413)
    list_125795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 413)
    # Adding element type (line 413)
    # Getting the type of 'Neach_section' (line 413)
    Neach_section_125796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'Neach_section')
    int_125797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 49), 'int')
    # Applying the binary operator '+' (line 413)
    result_add_125798 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 35), '+', Neach_section_125796, int_125797)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 34), list_125795, result_add_125798)
    
    # Applying the binary operator '*' (line 413)
    result_mul_125799 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 25), '*', extras_125794, list_125795)
    
    # Applying the binary operator '+' (line 412)
    result_add_125800 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 25), '+', list_125792, result_mul_125799)
    
    # Getting the type of 'Nsections' (line 414)
    Nsections_125801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'Nsections')
    # Getting the type of 'extras' (line 414)
    extras_125802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 36), 'extras')
    # Applying the binary operator '-' (line 414)
    result_sub_125803 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 26), '-', Nsections_125801, extras_125802)
    
    
    # Obtaining an instance of the builtin type 'list' (line 414)
    list_125804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 414)
    # Adding element type (line 414)
    # Getting the type of 'Neach_section' (line 414)
    Neach_section_125805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 47), 'Neach_section')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 46), list_125804, Neach_section_125805)
    
    # Applying the binary operator '*' (line 414)
    result_mul_125806 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 25), '*', result_sub_125803, list_125804)
    
    # Applying the binary operator '+' (line 413)
    result_add_125807 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 52), '+', result_add_125800, result_mul_125806)
    
    # Assigning a type to the variable 'section_sizes' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'section_sizes', result_add_125807)
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to cumsum(...): (line 415)
    # Processing the call keyword arguments (line 415)
    kwargs_125814 = {}
    
    # Call to array(...): (line 415)
    # Processing the call arguments (line 415)
    # Getting the type of 'section_sizes' (line 415)
    section_sizes_125810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 31), 'section_sizes', False)
    # Processing the call keyword arguments (line 415)
    kwargs_125811 = {}
    # Getting the type of '_nx' (line 415)
    _nx_125808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 21), '_nx', False)
    # Obtaining the member 'array' of a type (line 415)
    array_125809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 21), _nx_125808, 'array')
    # Calling array(args, kwargs) (line 415)
    array_call_result_125812 = invoke(stypy.reporting.localization.Localization(__file__, 415, 21), array_125809, *[section_sizes_125810], **kwargs_125811)
    
    # Obtaining the member 'cumsum' of a type (line 415)
    cumsum_125813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 21), array_call_result_125812, 'cumsum')
    # Calling cumsum(args, kwargs) (line 415)
    cumsum_call_result_125815 = invoke(stypy.reporting.localization.Localization(__file__, 415, 21), cumsum_125813, *[], **kwargs_125814)
    
    # Assigning a type to the variable 'div_points' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'div_points', cumsum_call_result_125815)
    # SSA join for try-except statement (line 402)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 417):
    
    # Assigning a List to a Name (line 417):
    
    # Obtaining an instance of the builtin type 'list' (line 417)
    list_125816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 417)
    
    # Assigning a type to the variable 'sub_arys' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'sub_arys', list_125816)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to swapaxes(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'ary' (line 418)
    ary_125819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 24), 'ary', False)
    # Getting the type of 'axis' (line 418)
    axis_125820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 29), 'axis', False)
    int_125821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 35), 'int')
    # Processing the call keyword arguments (line 418)
    kwargs_125822 = {}
    # Getting the type of '_nx' (line 418)
    _nx_125817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 11), '_nx', False)
    # Obtaining the member 'swapaxes' of a type (line 418)
    swapaxes_125818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 11), _nx_125817, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 418)
    swapaxes_call_result_125823 = invoke(stypy.reporting.localization.Localization(__file__, 418, 11), swapaxes_125818, *[ary_125819, axis_125820, int_125821], **kwargs_125822)
    
    # Assigning a type to the variable 'sary' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'sary', swapaxes_call_result_125823)
    
    
    # Call to range(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'Nsections' (line 419)
    Nsections_125825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 19), 'Nsections', False)
    # Processing the call keyword arguments (line 419)
    kwargs_125826 = {}
    # Getting the type of 'range' (line 419)
    range_125824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 13), 'range', False)
    # Calling range(args, kwargs) (line 419)
    range_call_result_125827 = invoke(stypy.reporting.localization.Localization(__file__, 419, 13), range_125824, *[Nsections_125825], **kwargs_125826)
    
    # Testing the type of a for loop iterable (line 419)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 419, 4), range_call_result_125827)
    # Getting the type of the for loop variable (line 419)
    for_loop_var_125828 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 419, 4), range_call_result_125827)
    # Assigning a type to the variable 'i' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'i', for_loop_var_125828)
    # SSA begins for a for statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 420):
    
    # Assigning a Subscript to a Name (line 420):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 420)
    i_125829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'i')
    # Getting the type of 'div_points' (line 420)
    div_points_125830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'div_points')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___125831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 13), div_points_125830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_125832 = invoke(stypy.reporting.localization.Localization(__file__, 420, 13), getitem___125831, i_125829)
    
    # Assigning a type to the variable 'st' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'st', subscript_call_result_125832)
    
    # Assigning a Subscript to a Name (line 421):
    
    # Assigning a Subscript to a Name (line 421):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 421)
    i_125833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'i')
    int_125834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 29), 'int')
    # Applying the binary operator '+' (line 421)
    result_add_125835 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 25), '+', i_125833, int_125834)
    
    # Getting the type of 'div_points' (line 421)
    div_points_125836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 14), 'div_points')
    # Obtaining the member '__getitem__' of a type (line 421)
    getitem___125837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 14), div_points_125836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 421)
    subscript_call_result_125838 = invoke(stypy.reporting.localization.Localization(__file__, 421, 14), getitem___125837, result_add_125835)
    
    # Assigning a type to the variable 'end' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'end', subscript_call_result_125838)
    
    # Call to append(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Call to swapaxes(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Obtaining the type of the subscript
    # Getting the type of 'st' (line 422)
    st_125843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 42), 'st', False)
    # Getting the type of 'end' (line 422)
    end_125844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 45), 'end', False)
    slice_125845 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 422, 37), st_125843, end_125844, None)
    # Getting the type of 'sary' (line 422)
    sary_125846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 37), 'sary', False)
    # Obtaining the member '__getitem__' of a type (line 422)
    getitem___125847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 37), sary_125846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 422)
    subscript_call_result_125848 = invoke(stypy.reporting.localization.Localization(__file__, 422, 37), getitem___125847, slice_125845)
    
    # Getting the type of 'axis' (line 422)
    axis_125849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 51), 'axis', False)
    int_125850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 57), 'int')
    # Processing the call keyword arguments (line 422)
    kwargs_125851 = {}
    # Getting the type of '_nx' (line 422)
    _nx_125841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), '_nx', False)
    # Obtaining the member 'swapaxes' of a type (line 422)
    swapaxes_125842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 24), _nx_125841, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 422)
    swapaxes_call_result_125852 = invoke(stypy.reporting.localization.Localization(__file__, 422, 24), swapaxes_125842, *[subscript_call_result_125848, axis_125849, int_125850], **kwargs_125851)
    
    # Processing the call keyword arguments (line 422)
    kwargs_125853 = {}
    # Getting the type of 'sub_arys' (line 422)
    sub_arys_125839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'sub_arys', False)
    # Obtaining the member 'append' of a type (line 422)
    append_125840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), sub_arys_125839, 'append')
    # Calling append(args, kwargs) (line 422)
    append_call_result_125854 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), append_125840, *[swapaxes_call_result_125852], **kwargs_125853)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sub_arys' (line 424)
    sub_arys_125855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'sub_arys')
    # Assigning a type to the variable 'stypy_return_type' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'stypy_return_type', sub_arys_125855)
    
    # ################# End of 'array_split(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'array_split' in the type store
    # Getting the type of 'stypy_return_type' (line 378)
    stypy_return_type_125856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'array_split'
    return stypy_return_type_125856

# Assigning a type to the variable 'array_split' (line 378)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'array_split', array_split)

@norecursion
def split(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_125857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 39), 'int')
    defaults = [int_125857]
    # Create a new context for function 'split'
    module_type_store = module_type_store.open_function_context('split', 427, 0, False)
    
    # Passed parameters checking function
    split.stypy_localization = localization
    split.stypy_type_of_self = None
    split.stypy_type_store = module_type_store
    split.stypy_function_name = 'split'
    split.stypy_param_names_list = ['ary', 'indices_or_sections', 'axis']
    split.stypy_varargs_param_name = None
    split.stypy_kwargs_param_name = None
    split.stypy_call_defaults = defaults
    split.stypy_call_varargs = varargs
    split.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split', ['ary', 'indices_or_sections', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split', localization, ['ary', 'indices_or_sections', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split(...)' code ##################

    str_125858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, (-1)), 'str', '\n    Split an array into multiple sub-arrays.\n\n    Parameters\n    ----------\n    ary : ndarray\n        Array to be divided into sub-arrays.\n    indices_or_sections : int or 1-D array\n        If `indices_or_sections` is an integer, N, the array will be divided\n        into N equal arrays along `axis`.  If such a split is not possible,\n        an error is raised.\n\n        If `indices_or_sections` is a 1-D array of sorted integers, the entries\n        indicate where along `axis` the array is split.  For example,\n        ``[2, 3]`` would, for ``axis=0``, result in\n\n          - ary[:2]\n          - ary[2:3]\n          - ary[3:]\n\n        If an index exceeds the dimension of the array along `axis`,\n        an empty sub-array is returned correspondingly.\n    axis : int, optional\n        The axis along which to split, default is 0.\n\n    Returns\n    -------\n    sub-arrays : list of ndarrays\n        A list of sub-arrays.\n\n    Raises\n    ------\n    ValueError\n        If `indices_or_sections` is given as an integer, but\n        a split does not result in equal division.\n\n    See Also\n    --------\n    array_split : Split an array into multiple sub-arrays of equal or\n                  near-equal size.  Does not raise an exception if\n                  an equal division cannot be made.\n    hsplit : Split array into multiple sub-arrays horizontally (column-wise).\n    vsplit : Split array into multiple sub-arrays vertically (row wise).\n    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).\n    concatenate : Join a sequence of arrays along an existing axis.\n    stack : Join a sequence of arrays along a new axis.\n    hstack : Stack arrays in sequence horizontally (column wise).\n    vstack : Stack arrays in sequence vertically (row wise).\n    dstack : Stack arrays in sequence depth wise (along third dimension).\n\n    Examples\n    --------\n    >>> x = np.arange(9.0)\n    >>> np.split(x, 3)\n    [array([ 0.,  1.,  2.]), array([ 3.,  4.,  5.]), array([ 6.,  7.,  8.])]\n\n    >>> x = np.arange(8.0)\n    >>> np.split(x, [3, 5, 6, 10])\n    [array([ 0.,  1.,  2.]),\n     array([ 3.,  4.]),\n     array([ 5.]),\n     array([ 6.,  7.]),\n     array([], dtype=float64)]\n\n    ')
    
    
    # SSA begins for try-except statement (line 493)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to len(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'indices_or_sections' (line 494)
    indices_or_sections_125860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'indices_or_sections', False)
    # Processing the call keyword arguments (line 494)
    kwargs_125861 = {}
    # Getting the type of 'len' (line 494)
    len_125859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'len', False)
    # Calling len(args, kwargs) (line 494)
    len_call_result_125862 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), len_125859, *[indices_or_sections_125860], **kwargs_125861)
    
    # SSA branch for the except part of a try statement (line 493)
    # SSA branch for the except 'TypeError' branch of a try statement (line 493)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 496):
    
    # Assigning a Name to a Name (line 496):
    # Getting the type of 'indices_or_sections' (line 496)
    indices_or_sections_125863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 19), 'indices_or_sections')
    # Assigning a type to the variable 'sections' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'sections', indices_or_sections_125863)
    
    # Assigning a Subscript to a Name (line 497):
    
    # Assigning a Subscript to a Name (line 497):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 497)
    axis_125864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 22), 'axis')
    # Getting the type of 'ary' (line 497)
    ary_125865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 12), 'ary')
    # Obtaining the member 'shape' of a type (line 497)
    shape_125866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), ary_125865, 'shape')
    # Obtaining the member '__getitem__' of a type (line 497)
    getitem___125867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 12), shape_125866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 497)
    subscript_call_result_125868 = invoke(stypy.reporting.localization.Localization(__file__, 497, 12), getitem___125867, axis_125864)
    
    # Assigning a type to the variable 'N' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'N', subscript_call_result_125868)
    
    # Getting the type of 'N' (line 498)
    N_125869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 11), 'N')
    # Getting the type of 'sections' (line 498)
    sections_125870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 15), 'sections')
    # Applying the binary operator '%' (line 498)
    result_mod_125871 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 11), '%', N_125869, sections_125870)
    
    # Testing the type of an if condition (line 498)
    if_condition_125872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 8), result_mod_125871)
    # Assigning a type to the variable 'if_condition_125872' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'if_condition_125872', if_condition_125872)
    # SSA begins for if statement (line 498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 499)
    # Processing the call arguments (line 499)
    str_125874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 16), 'str', 'array split does not result in an equal division')
    # Processing the call keyword arguments (line 499)
    kwargs_125875 = {}
    # Getting the type of 'ValueError' (line 499)
    ValueError_125873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 499)
    ValueError_call_result_125876 = invoke(stypy.reporting.localization.Localization(__file__, 499, 18), ValueError_125873, *[str_125874], **kwargs_125875)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 499, 12), ValueError_call_result_125876, 'raise parameter', BaseException)
    # SSA join for if statement (line 498)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 493)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 501):
    
    # Assigning a Call to a Name (line 501):
    
    # Call to array_split(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'ary' (line 501)
    ary_125878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 22), 'ary', False)
    # Getting the type of 'indices_or_sections' (line 501)
    indices_or_sections_125879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 27), 'indices_or_sections', False)
    # Getting the type of 'axis' (line 501)
    axis_125880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 48), 'axis', False)
    # Processing the call keyword arguments (line 501)
    kwargs_125881 = {}
    # Getting the type of 'array_split' (line 501)
    array_split_125877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 10), 'array_split', False)
    # Calling array_split(args, kwargs) (line 501)
    array_split_call_result_125882 = invoke(stypy.reporting.localization.Localization(__file__, 501, 10), array_split_125877, *[ary_125878, indices_or_sections_125879, axis_125880], **kwargs_125881)
    
    # Assigning a type to the variable 'res' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'res', array_split_call_result_125882)
    # Getting the type of 'res' (line 502)
    res_125883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'stypy_return_type', res_125883)
    
    # ################# End of 'split(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split' in the type store
    # Getting the type of 'stypy_return_type' (line 427)
    stypy_return_type_125884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125884)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split'
    return stypy_return_type_125884

# Assigning a type to the variable 'split' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'split', split)

@norecursion
def hsplit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hsplit'
    module_type_store = module_type_store.open_function_context('hsplit', 504, 0, False)
    
    # Passed parameters checking function
    hsplit.stypy_localization = localization
    hsplit.stypy_type_of_self = None
    hsplit.stypy_type_store = module_type_store
    hsplit.stypy_function_name = 'hsplit'
    hsplit.stypy_param_names_list = ['ary', 'indices_or_sections']
    hsplit.stypy_varargs_param_name = None
    hsplit.stypy_kwargs_param_name = None
    hsplit.stypy_call_defaults = defaults
    hsplit.stypy_call_varargs = varargs
    hsplit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hsplit', ['ary', 'indices_or_sections'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hsplit', localization, ['ary', 'indices_or_sections'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hsplit(...)' code ##################

    str_125885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, (-1)), 'str', '\n    Split an array into multiple sub-arrays horizontally (column-wise).\n\n    Please refer to the `split` documentation.  `hsplit` is equivalent\n    to `split` with ``axis=1``, the array is always split along the second\n    axis regardless of the array dimension.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(4, 4)\n    >>> x\n    array([[  0.,   1.,   2.,   3.],\n           [  4.,   5.,   6.,   7.],\n           [  8.,   9.,  10.,  11.],\n           [ 12.,  13.,  14.,  15.]])\n    >>> np.hsplit(x, 2)\n    [array([[  0.,   1.],\n           [  4.,   5.],\n           [  8.,   9.],\n           [ 12.,  13.]]),\n     array([[  2.,   3.],\n           [  6.,   7.],\n           [ 10.,  11.],\n           [ 14.,  15.]])]\n    >>> np.hsplit(x, np.array([3, 6]))\n    [array([[  0.,   1.,   2.],\n           [  4.,   5.,   6.],\n           [  8.,   9.,  10.],\n           [ 12.,  13.,  14.]]),\n     array([[  3.],\n           [  7.],\n           [ 11.],\n           [ 15.]]),\n     array([], dtype=float64)]\n\n    With a higher dimensional array the split is still along the second axis.\n\n    >>> x = np.arange(8.0).reshape(2, 2, 2)\n    >>> x\n    array([[[ 0.,  1.],\n            [ 2.,  3.]],\n           [[ 4.,  5.],\n            [ 6.,  7.]]])\n    >>> np.hsplit(x, 2)\n    [array([[[ 0.,  1.]],\n           [[ 4.,  5.]]]),\n     array([[[ 2.,  3.]],\n           [[ 6.,  7.]]])]\n\n    ')
    
    
    
    # Call to len(...): (line 559)
    # Processing the call arguments (line 559)
    
    # Call to shape(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'ary' (line 559)
    ary_125889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 21), 'ary', False)
    # Processing the call keyword arguments (line 559)
    kwargs_125890 = {}
    # Getting the type of '_nx' (line 559)
    _nx_125887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), '_nx', False)
    # Obtaining the member 'shape' of a type (line 559)
    shape_125888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 11), _nx_125887, 'shape')
    # Calling shape(args, kwargs) (line 559)
    shape_call_result_125891 = invoke(stypy.reporting.localization.Localization(__file__, 559, 11), shape_125888, *[ary_125889], **kwargs_125890)
    
    # Processing the call keyword arguments (line 559)
    kwargs_125892 = {}
    # Getting the type of 'len' (line 559)
    len_125886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'len', False)
    # Calling len(args, kwargs) (line 559)
    len_call_result_125893 = invoke(stypy.reporting.localization.Localization(__file__, 559, 7), len_125886, *[shape_call_result_125891], **kwargs_125892)
    
    int_125894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 30), 'int')
    # Applying the binary operator '==' (line 559)
    result_eq_125895 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 7), '==', len_call_result_125893, int_125894)
    
    # Testing the type of an if condition (line 559)
    if_condition_125896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 4), result_eq_125895)
    # Assigning a type to the variable 'if_condition_125896' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'if_condition_125896', if_condition_125896)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 560)
    # Processing the call arguments (line 560)
    str_125898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 25), 'str', 'hsplit only works on arrays of 1 or more dimensions')
    # Processing the call keyword arguments (line 560)
    kwargs_125899 = {}
    # Getting the type of 'ValueError' (line 560)
    ValueError_125897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 560)
    ValueError_call_result_125900 = invoke(stypy.reporting.localization.Localization(__file__, 560, 14), ValueError_125897, *[str_125898], **kwargs_125899)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 560, 8), ValueError_call_result_125900, 'raise parameter', BaseException)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'ary' (line 561)
    ary_125902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 11), 'ary', False)
    # Obtaining the member 'shape' of a type (line 561)
    shape_125903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 11), ary_125902, 'shape')
    # Processing the call keyword arguments (line 561)
    kwargs_125904 = {}
    # Getting the type of 'len' (line 561)
    len_125901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 7), 'len', False)
    # Calling len(args, kwargs) (line 561)
    len_call_result_125905 = invoke(stypy.reporting.localization.Localization(__file__, 561, 7), len_125901, *[shape_125903], **kwargs_125904)
    
    int_125906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 24), 'int')
    # Applying the binary operator '>' (line 561)
    result_gt_125907 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 7), '>', len_call_result_125905, int_125906)
    
    # Testing the type of an if condition (line 561)
    if_condition_125908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 4), result_gt_125907)
    # Assigning a type to the variable 'if_condition_125908' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'if_condition_125908', if_condition_125908)
    # SSA begins for if statement (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to split(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'ary' (line 562)
    ary_125910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 21), 'ary', False)
    # Getting the type of 'indices_or_sections' (line 562)
    indices_or_sections_125911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 26), 'indices_or_sections', False)
    int_125912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 47), 'int')
    # Processing the call keyword arguments (line 562)
    kwargs_125913 = {}
    # Getting the type of 'split' (line 562)
    split_125909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'split', False)
    # Calling split(args, kwargs) (line 562)
    split_call_result_125914 = invoke(stypy.reporting.localization.Localization(__file__, 562, 15), split_125909, *[ary_125910, indices_or_sections_125911, int_125912], **kwargs_125913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'stypy_return_type', split_call_result_125914)
    # SSA branch for the else part of an if statement (line 561)
    module_type_store.open_ssa_branch('else')
    
    # Call to split(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'ary' (line 564)
    ary_125916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 21), 'ary', False)
    # Getting the type of 'indices_or_sections' (line 564)
    indices_or_sections_125917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 26), 'indices_or_sections', False)
    int_125918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 47), 'int')
    # Processing the call keyword arguments (line 564)
    kwargs_125919 = {}
    # Getting the type of 'split' (line 564)
    split_125915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 15), 'split', False)
    # Calling split(args, kwargs) (line 564)
    split_call_result_125920 = invoke(stypy.reporting.localization.Localization(__file__, 564, 15), split_125915, *[ary_125916, indices_or_sections_125917, int_125918], **kwargs_125919)
    
    # Assigning a type to the variable 'stypy_return_type' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'stypy_return_type', split_call_result_125920)
    # SSA join for if statement (line 561)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hsplit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hsplit' in the type store
    # Getting the type of 'stypy_return_type' (line 504)
    stypy_return_type_125921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125921)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hsplit'
    return stypy_return_type_125921

# Assigning a type to the variable 'hsplit' (line 504)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 0), 'hsplit', hsplit)

@norecursion
def vsplit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vsplit'
    module_type_store = module_type_store.open_function_context('vsplit', 566, 0, False)
    
    # Passed parameters checking function
    vsplit.stypy_localization = localization
    vsplit.stypy_type_of_self = None
    vsplit.stypy_type_store = module_type_store
    vsplit.stypy_function_name = 'vsplit'
    vsplit.stypy_param_names_list = ['ary', 'indices_or_sections']
    vsplit.stypy_varargs_param_name = None
    vsplit.stypy_kwargs_param_name = None
    vsplit.stypy_call_defaults = defaults
    vsplit.stypy_call_varargs = varargs
    vsplit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vsplit', ['ary', 'indices_or_sections'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vsplit', localization, ['ary', 'indices_or_sections'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vsplit(...)' code ##################

    str_125922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, (-1)), 'str', '\n    Split an array into multiple sub-arrays vertically (row-wise).\n\n    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent\n    to ``split`` with `axis=0` (default), the array is always split along the\n    first axis regardless of the array dimension.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(4, 4)\n    >>> x\n    array([[  0.,   1.,   2.,   3.],\n           [  4.,   5.,   6.,   7.],\n           [  8.,   9.,  10.,  11.],\n           [ 12.,  13.,  14.,  15.]])\n    >>> np.vsplit(x, 2)\n    [array([[ 0.,  1.,  2.,  3.],\n           [ 4.,  5.,  6.,  7.]]),\n     array([[  8.,   9.,  10.,  11.],\n           [ 12.,  13.,  14.,  15.]])]\n    >>> np.vsplit(x, np.array([3, 6]))\n    [array([[  0.,   1.,   2.,   3.],\n           [  4.,   5.,   6.,   7.],\n           [  8.,   9.,  10.,  11.]]),\n     array([[ 12.,  13.,  14.,  15.]]),\n     array([], dtype=float64)]\n\n    With a higher dimensional array the split is still along the first axis.\n\n    >>> x = np.arange(8.0).reshape(2, 2, 2)\n    >>> x\n    array([[[ 0.,  1.],\n            [ 2.,  3.]],\n           [[ 4.,  5.],\n            [ 6.,  7.]]])\n    >>> np.vsplit(x, 2)\n    [array([[[ 0.,  1.],\n            [ 2.,  3.]]]),\n     array([[[ 4.,  5.],\n            [ 6.,  7.]]])]\n\n    ')
    
    
    
    # Call to len(...): (line 613)
    # Processing the call arguments (line 613)
    
    # Call to shape(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'ary' (line 613)
    ary_125926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'ary', False)
    # Processing the call keyword arguments (line 613)
    kwargs_125927 = {}
    # Getting the type of '_nx' (line 613)
    _nx_125924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), '_nx', False)
    # Obtaining the member 'shape' of a type (line 613)
    shape_125925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), _nx_125924, 'shape')
    # Calling shape(args, kwargs) (line 613)
    shape_call_result_125928 = invoke(stypy.reporting.localization.Localization(__file__, 613, 11), shape_125925, *[ary_125926], **kwargs_125927)
    
    # Processing the call keyword arguments (line 613)
    kwargs_125929 = {}
    # Getting the type of 'len' (line 613)
    len_125923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 7), 'len', False)
    # Calling len(args, kwargs) (line 613)
    len_call_result_125930 = invoke(stypy.reporting.localization.Localization(__file__, 613, 7), len_125923, *[shape_call_result_125928], **kwargs_125929)
    
    int_125931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 29), 'int')
    # Applying the binary operator '<' (line 613)
    result_lt_125932 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 7), '<', len_call_result_125930, int_125931)
    
    # Testing the type of an if condition (line 613)
    if_condition_125933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 4), result_lt_125932)
    # Assigning a type to the variable 'if_condition_125933' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'if_condition_125933', if_condition_125933)
    # SSA begins for if statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 614)
    # Processing the call arguments (line 614)
    str_125935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 25), 'str', 'vsplit only works on arrays of 2 or more dimensions')
    # Processing the call keyword arguments (line 614)
    kwargs_125936 = {}
    # Getting the type of 'ValueError' (line 614)
    ValueError_125934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 614)
    ValueError_call_result_125937 = invoke(stypy.reporting.localization.Localization(__file__, 614, 14), ValueError_125934, *[str_125935], **kwargs_125936)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 614, 8), ValueError_call_result_125937, 'raise parameter', BaseException)
    # SSA join for if statement (line 613)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to split(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'ary' (line 615)
    ary_125939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 17), 'ary', False)
    # Getting the type of 'indices_or_sections' (line 615)
    indices_or_sections_125940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 22), 'indices_or_sections', False)
    int_125941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 43), 'int')
    # Processing the call keyword arguments (line 615)
    kwargs_125942 = {}
    # Getting the type of 'split' (line 615)
    split_125938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'split', False)
    # Calling split(args, kwargs) (line 615)
    split_call_result_125943 = invoke(stypy.reporting.localization.Localization(__file__, 615, 11), split_125938, *[ary_125939, indices_or_sections_125940, int_125941], **kwargs_125942)
    
    # Assigning a type to the variable 'stypy_return_type' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'stypy_return_type', split_call_result_125943)
    
    # ################# End of 'vsplit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vsplit' in the type store
    # Getting the type of 'stypy_return_type' (line 566)
    stypy_return_type_125944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125944)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vsplit'
    return stypy_return_type_125944

# Assigning a type to the variable 'vsplit' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'vsplit', vsplit)

@norecursion
def dsplit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dsplit'
    module_type_store = module_type_store.open_function_context('dsplit', 617, 0, False)
    
    # Passed parameters checking function
    dsplit.stypy_localization = localization
    dsplit.stypy_type_of_self = None
    dsplit.stypy_type_store = module_type_store
    dsplit.stypy_function_name = 'dsplit'
    dsplit.stypy_param_names_list = ['ary', 'indices_or_sections']
    dsplit.stypy_varargs_param_name = None
    dsplit.stypy_kwargs_param_name = None
    dsplit.stypy_call_defaults = defaults
    dsplit.stypy_call_varargs = varargs
    dsplit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dsplit', ['ary', 'indices_or_sections'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dsplit', localization, ['ary', 'indices_or_sections'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dsplit(...)' code ##################

    str_125945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, (-1)), 'str', '\n    Split array into multiple sub-arrays along the 3rd axis (depth).\n\n    Please refer to the `split` documentation.  `dsplit` is equivalent\n    to `split` with ``axis=2``, the array is always split along the third\n    axis provided the array dimension is greater than or equal to 3.\n\n    See Also\n    --------\n    split : Split an array into multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> x = np.arange(16.0).reshape(2, 2, 4)\n    >>> x\n    array([[[  0.,   1.,   2.,   3.],\n            [  4.,   5.,   6.,   7.]],\n           [[  8.,   9.,  10.,  11.],\n            [ 12.,  13.,  14.,  15.]]])\n    >>> np.dsplit(x, 2)\n    [array([[[  0.,   1.],\n            [  4.,   5.]],\n           [[  8.,   9.],\n            [ 12.,  13.]]]),\n     array([[[  2.,   3.],\n            [  6.,   7.]],\n           [[ 10.,  11.],\n            [ 14.,  15.]]])]\n    >>> np.dsplit(x, np.array([3, 6]))\n    [array([[[  0.,   1.,   2.],\n            [  4.,   5.,   6.]],\n           [[  8.,   9.,  10.],\n            [ 12.,  13.,  14.]]]),\n     array([[[  3.],\n            [  7.]],\n           [[ 11.],\n            [ 15.]]]),\n     array([], dtype=float64)]\n\n    ')
    
    
    
    # Call to len(...): (line 658)
    # Processing the call arguments (line 658)
    
    # Call to shape(...): (line 658)
    # Processing the call arguments (line 658)
    # Getting the type of 'ary' (line 658)
    ary_125949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 21), 'ary', False)
    # Processing the call keyword arguments (line 658)
    kwargs_125950 = {}
    # Getting the type of '_nx' (line 658)
    _nx_125947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 11), '_nx', False)
    # Obtaining the member 'shape' of a type (line 658)
    shape_125948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 11), _nx_125947, 'shape')
    # Calling shape(args, kwargs) (line 658)
    shape_call_result_125951 = invoke(stypy.reporting.localization.Localization(__file__, 658, 11), shape_125948, *[ary_125949], **kwargs_125950)
    
    # Processing the call keyword arguments (line 658)
    kwargs_125952 = {}
    # Getting the type of 'len' (line 658)
    len_125946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 7), 'len', False)
    # Calling len(args, kwargs) (line 658)
    len_call_result_125953 = invoke(stypy.reporting.localization.Localization(__file__, 658, 7), len_125946, *[shape_call_result_125951], **kwargs_125952)
    
    int_125954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 29), 'int')
    # Applying the binary operator '<' (line 658)
    result_lt_125955 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 7), '<', len_call_result_125953, int_125954)
    
    # Testing the type of an if condition (line 658)
    if_condition_125956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 658, 4), result_lt_125955)
    # Assigning a type to the variable 'if_condition_125956' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'if_condition_125956', if_condition_125956)
    # SSA begins for if statement (line 658)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 659)
    # Processing the call arguments (line 659)
    str_125958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 25), 'str', 'dsplit only works on arrays of 3 or more dimensions')
    # Processing the call keyword arguments (line 659)
    kwargs_125959 = {}
    # Getting the type of 'ValueError' (line 659)
    ValueError_125957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 659)
    ValueError_call_result_125960 = invoke(stypy.reporting.localization.Localization(__file__, 659, 14), ValueError_125957, *[str_125958], **kwargs_125959)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 659, 8), ValueError_call_result_125960, 'raise parameter', BaseException)
    # SSA join for if statement (line 658)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to split(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'ary' (line 660)
    ary_125962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 17), 'ary', False)
    # Getting the type of 'indices_or_sections' (line 660)
    indices_or_sections_125963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 22), 'indices_or_sections', False)
    int_125964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 43), 'int')
    # Processing the call keyword arguments (line 660)
    kwargs_125965 = {}
    # Getting the type of 'split' (line 660)
    split_125961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 11), 'split', False)
    # Calling split(args, kwargs) (line 660)
    split_call_result_125966 = invoke(stypy.reporting.localization.Localization(__file__, 660, 11), split_125961, *[ary_125962, indices_or_sections_125963, int_125964], **kwargs_125965)
    
    # Assigning a type to the variable 'stypy_return_type' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 4), 'stypy_return_type', split_call_result_125966)
    
    # ################# End of 'dsplit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dsplit' in the type store
    # Getting the type of 'stypy_return_type' (line 617)
    stypy_return_type_125967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_125967)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dsplit'
    return stypy_return_type_125967

# Assigning a type to the variable 'dsplit' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'dsplit', dsplit)

@norecursion
def get_array_prepare(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_array_prepare'
    module_type_store = module_type_store.open_function_context('get_array_prepare', 662, 0, False)
    
    # Passed parameters checking function
    get_array_prepare.stypy_localization = localization
    get_array_prepare.stypy_type_of_self = None
    get_array_prepare.stypy_type_store = module_type_store
    get_array_prepare.stypy_function_name = 'get_array_prepare'
    get_array_prepare.stypy_param_names_list = []
    get_array_prepare.stypy_varargs_param_name = 'args'
    get_array_prepare.stypy_kwargs_param_name = None
    get_array_prepare.stypy_call_defaults = defaults
    get_array_prepare.stypy_call_varargs = varargs
    get_array_prepare.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_array_prepare', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_array_prepare', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_array_prepare(...)' code ##################

    str_125968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, (-1)), 'str', 'Find the wrapper for the array with the highest priority.\n\n    In case of ties, leftmost wins. If no wrapper is found, return None\n    ')
    
    # Assigning a Call to a Name (line 667):
    
    # Assigning a Call to a Name (line 667):
    
    # Call to sorted(...): (line 667)
    # Processing the call arguments (line 667)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 667, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'args' (line 668)
    args_125987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 60), 'args', False)
    # Processing the call keyword arguments (line 668)
    kwargs_125988 = {}
    # Getting the type of 'enumerate' (line 668)
    enumerate_125986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 50), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 668)
    enumerate_call_result_125989 = invoke(stypy.reporting.localization.Localization(__file__, 668, 50), enumerate_125986, *[args_125987], **kwargs_125988)
    
    comprehension_125990 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 22), enumerate_call_result_125989)
    # Assigning a type to the variable 'i' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 22), comprehension_125990))
    # Assigning a type to the variable 'x' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 22), comprehension_125990))
    
    # Call to hasattr(...): (line 669)
    # Processing the call arguments (line 669)
    # Getting the type of 'x' (line 669)
    x_125982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 46), 'x', False)
    str_125983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 49), 'str', '__array_prepare__')
    # Processing the call keyword arguments (line 669)
    kwargs_125984 = {}
    # Getting the type of 'hasattr' (line 669)
    hasattr_125981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 38), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 669)
    hasattr_call_result_125985 = invoke(stypy.reporting.localization.Localization(__file__, 669, 38), hasattr_125981, *[x_125982, str_125983], **kwargs_125984)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 667)
    tuple_125970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 667)
    # Adding element type (line 667)
    
    # Call to getattr(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 'x' (line 667)
    x_125972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 31), 'x', False)
    str_125973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 34), 'str', '__array_priority__')
    int_125974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 56), 'int')
    # Processing the call keyword arguments (line 667)
    kwargs_125975 = {}
    # Getting the type of 'getattr' (line 667)
    getattr_125971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 23), 'getattr', False)
    # Calling getattr(args, kwargs) (line 667)
    getattr_call_result_125976 = invoke(stypy.reporting.localization.Localization(__file__, 667, 23), getattr_125971, *[x_125972, str_125973, int_125974], **kwargs_125975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), tuple_125970, getattr_call_result_125976)
    # Adding element type (line 667)
    
    # Getting the type of 'i' (line 667)
    i_125977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 61), 'i', False)
    # Applying the 'usub' unary operator (line 667)
    result___neg___125978 = python_operator(stypy.reporting.localization.Localization(__file__, 667, 60), 'usub', i_125977)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), tuple_125970, result___neg___125978)
    # Adding element type (line 667)
    # Getting the type of 'x' (line 668)
    x_125979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 17), 'x', False)
    # Obtaining the member '__array_prepare__' of a type (line 668)
    array_prepare___125980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 17), x_125979, '__array_prepare__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 23), tuple_125970, array_prepare___125980)
    
    list_125991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 22), list_125991, tuple_125970)
    # Processing the call keyword arguments (line 667)
    kwargs_125992 = {}
    # Getting the type of 'sorted' (line 667)
    sorted_125969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 15), 'sorted', False)
    # Calling sorted(args, kwargs) (line 667)
    sorted_call_result_125993 = invoke(stypy.reporting.localization.Localization(__file__, 667, 15), sorted_125969, *[list_125991], **kwargs_125992)
    
    # Assigning a type to the variable 'wrappers' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'wrappers', sorted_call_result_125993)
    
    # Getting the type of 'wrappers' (line 670)
    wrappers_125994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 7), 'wrappers')
    # Testing the type of an if condition (line 670)
    if_condition_125995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 4), wrappers_125994)
    # Assigning a type to the variable 'if_condition_125995' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'if_condition_125995', if_condition_125995)
    # SSA begins for if statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_125996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 28), 'int')
    
    # Obtaining the type of the subscript
    int_125997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 24), 'int')
    # Getting the type of 'wrappers' (line 671)
    wrappers_125998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 15), 'wrappers')
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___125999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 15), wrappers_125998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_126000 = invoke(stypy.reporting.localization.Localization(__file__, 671, 15), getitem___125999, int_125997)
    
    # Obtaining the member '__getitem__' of a type (line 671)
    getitem___126001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 15), subscript_call_result_126000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 671)
    subscript_call_result_126002 = invoke(stypy.reporting.localization.Localization(__file__, 671, 15), getitem___126001, int_125996)
    
    # Assigning a type to the variable 'stypy_return_type' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'stypy_return_type', subscript_call_result_126002)
    # SSA join for if statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 672)
    None_126003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'stypy_return_type', None_126003)
    
    # ################# End of 'get_array_prepare(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_array_prepare' in the type store
    # Getting the type of 'stypy_return_type' (line 662)
    stypy_return_type_126004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_array_prepare'
    return stypy_return_type_126004

# Assigning a type to the variable 'get_array_prepare' (line 662)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 0), 'get_array_prepare', get_array_prepare)

@norecursion
def get_array_wrap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_array_wrap'
    module_type_store = module_type_store.open_function_context('get_array_wrap', 674, 0, False)
    
    # Passed parameters checking function
    get_array_wrap.stypy_localization = localization
    get_array_wrap.stypy_type_of_self = None
    get_array_wrap.stypy_type_store = module_type_store
    get_array_wrap.stypy_function_name = 'get_array_wrap'
    get_array_wrap.stypy_param_names_list = []
    get_array_wrap.stypy_varargs_param_name = 'args'
    get_array_wrap.stypy_kwargs_param_name = None
    get_array_wrap.stypy_call_defaults = defaults
    get_array_wrap.stypy_call_varargs = varargs
    get_array_wrap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_array_wrap', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_array_wrap', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_array_wrap(...)' code ##################

    str_126005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, (-1)), 'str', 'Find the wrapper for the array with the highest priority.\n\n    In case of ties, leftmost wins. If no wrapper is found, return None\n    ')
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to sorted(...): (line 679)
    # Processing the call arguments (line 679)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 679, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 680)
    # Processing the call arguments (line 680)
    # Getting the type of 'args' (line 680)
    args_126024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 57), 'args', False)
    # Processing the call keyword arguments (line 680)
    kwargs_126025 = {}
    # Getting the type of 'enumerate' (line 680)
    enumerate_126023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 47), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 680)
    enumerate_call_result_126026 = invoke(stypy.reporting.localization.Localization(__file__, 680, 47), enumerate_126023, *[args_126024], **kwargs_126025)
    
    comprehension_126027 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 22), enumerate_call_result_126026)
    # Assigning a type to the variable 'i' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 22), comprehension_126027))
    # Assigning a type to the variable 'x' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 22), comprehension_126027))
    
    # Call to hasattr(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'x' (line 681)
    x_126019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 46), 'x', False)
    str_126020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 49), 'str', '__array_wrap__')
    # Processing the call keyword arguments (line 681)
    kwargs_126021 = {}
    # Getting the type of 'hasattr' (line 681)
    hasattr_126018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 38), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 681)
    hasattr_call_result_126022 = invoke(stypy.reporting.localization.Localization(__file__, 681, 38), hasattr_126018, *[x_126019, str_126020], **kwargs_126021)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 679)
    tuple_126007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 679)
    # Adding element type (line 679)
    
    # Call to getattr(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'x' (line 679)
    x_126009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 31), 'x', False)
    str_126010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 34), 'str', '__array_priority__')
    int_126011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 56), 'int')
    # Processing the call keyword arguments (line 679)
    kwargs_126012 = {}
    # Getting the type of 'getattr' (line 679)
    getattr_126008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 23), 'getattr', False)
    # Calling getattr(args, kwargs) (line 679)
    getattr_call_result_126013 = invoke(stypy.reporting.localization.Localization(__file__, 679, 23), getattr_126008, *[x_126009, str_126010, int_126011], **kwargs_126012)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 23), tuple_126007, getattr_call_result_126013)
    # Adding element type (line 679)
    
    # Getting the type of 'i' (line 679)
    i_126014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 61), 'i', False)
    # Applying the 'usub' unary operator (line 679)
    result___neg___126015 = python_operator(stypy.reporting.localization.Localization(__file__, 679, 60), 'usub', i_126014)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 23), tuple_126007, result___neg___126015)
    # Adding element type (line 679)
    # Getting the type of 'x' (line 680)
    x_126016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 17), 'x', False)
    # Obtaining the member '__array_wrap__' of a type (line 680)
    array_wrap___126017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 17), x_126016, '__array_wrap__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 23), tuple_126007, array_wrap___126017)
    
    list_126028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 22), list_126028, tuple_126007)
    # Processing the call keyword arguments (line 679)
    kwargs_126029 = {}
    # Getting the type of 'sorted' (line 679)
    sorted_126006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 15), 'sorted', False)
    # Calling sorted(args, kwargs) (line 679)
    sorted_call_result_126030 = invoke(stypy.reporting.localization.Localization(__file__, 679, 15), sorted_126006, *[list_126028], **kwargs_126029)
    
    # Assigning a type to the variable 'wrappers' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'wrappers', sorted_call_result_126030)
    
    # Getting the type of 'wrappers' (line 682)
    wrappers_126031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 7), 'wrappers')
    # Testing the type of an if condition (line 682)
    if_condition_126032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 4), wrappers_126031)
    # Assigning a type to the variable 'if_condition_126032' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'if_condition_126032', if_condition_126032)
    # SSA begins for if statement (line 682)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_126033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 28), 'int')
    
    # Obtaining the type of the subscript
    int_126034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 24), 'int')
    # Getting the type of 'wrappers' (line 683)
    wrappers_126035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'wrappers')
    # Obtaining the member '__getitem__' of a type (line 683)
    getitem___126036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 15), wrappers_126035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 683)
    subscript_call_result_126037 = invoke(stypy.reporting.localization.Localization(__file__, 683, 15), getitem___126036, int_126034)
    
    # Obtaining the member '__getitem__' of a type (line 683)
    getitem___126038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 15), subscript_call_result_126037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 683)
    subscript_call_result_126039 = invoke(stypy.reporting.localization.Localization(__file__, 683, 15), getitem___126038, int_126033)
    
    # Assigning a type to the variable 'stypy_return_type' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'stypy_return_type', subscript_call_result_126039)
    # SSA join for if statement (line 682)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 684)
    None_126040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'stypy_return_type', None_126040)
    
    # ################# End of 'get_array_wrap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_array_wrap' in the type store
    # Getting the type of 'stypy_return_type' (line 674)
    stypy_return_type_126041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_array_wrap'
    return stypy_return_type_126041

# Assigning a type to the variable 'get_array_wrap' (line 674)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 0), 'get_array_wrap', get_array_wrap)

@norecursion
def kron(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kron'
    module_type_store = module_type_store.open_function_context('kron', 686, 0, False)
    
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

    str_126042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', '\n    Kronecker product of two arrays.\n\n    Computes the Kronecker product, a composite array made of blocks of the\n    second array scaled by the first.\n\n    Parameters\n    ----------\n    a, b : array_like\n\n    Returns\n    -------\n    out : ndarray\n\n    See Also\n    --------\n    outer : The outer product\n\n    Notes\n    -----\n    The function assumes that the number of dimensions of `a` and `b`\n    are the same, if necessary prepending the smallest with ones.\n    If `a.shape = (r0,r1,..,rN)` and `b.shape = (s0,s1,...,sN)`,\n    the Kronecker product has shape `(r0*s0, r1*s1, ..., rN*SN)`.\n    The elements are products of elements from `a` and `b`, organized\n    explicitly by::\n\n        kron(a,b)[k0,k1,...,kN] = a[i0,i1,...,iN] * b[j0,j1,...,jN]\n\n    where::\n\n        kt = it * st + jt,  t = 0,...,N\n\n    In the common 2-D case (N=1), the block structure can be visualized::\n\n        [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],\n         [  ...                              ...   ],\n         [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]\n\n\n    Examples\n    --------\n    >>> np.kron([1,10,100], [5,6,7])\n    array([  5,   6,   7,  50,  60,  70, 500, 600, 700])\n    >>> np.kron([5,6,7], [1,10,100])\n    array([  5,  50, 500,   6,  60, 600,   7,  70, 700])\n\n    >>> np.kron(np.eye(2), np.ones((2,2)))\n    array([[ 1.,  1.,  0.,  0.],\n           [ 1.,  1.,  0.,  0.],\n           [ 0.,  0.,  1.,  1.],\n           [ 0.,  0.,  1.,  1.]])\n\n    >>> a = np.arange(100).reshape((2,5,2,5))\n    >>> b = np.arange(24).reshape((2,3,4))\n    >>> c = np.kron(a,b)\n    >>> c.shape\n    (2, 10, 6, 20)\n    >>> I = (1,3,0,2)\n    >>> J = (0,2,1)\n    >>> J1 = (0,) + J             # extend to ndim=4\n    >>> S1 = (1,) + b.shape\n    >>> K = tuple(np.array(I) * np.array(S1) + np.array(J1))\n    >>> c[K] == a[I]*b[J]\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 754):
    
    # Assigning a Call to a Name (line 754):
    
    # Call to asanyarray(...): (line 754)
    # Processing the call arguments (line 754)
    # Getting the type of 'b' (line 754)
    b_126044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 19), 'b', False)
    # Processing the call keyword arguments (line 754)
    kwargs_126045 = {}
    # Getting the type of 'asanyarray' (line 754)
    asanyarray_126043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 754)
    asanyarray_call_result_126046 = invoke(stypy.reporting.localization.Localization(__file__, 754, 8), asanyarray_126043, *[b_126044], **kwargs_126045)
    
    # Assigning a type to the variable 'b' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'b', asanyarray_call_result_126046)
    
    # Assigning a Call to a Name (line 755):
    
    # Assigning a Call to a Name (line 755):
    
    # Call to array(...): (line 755)
    # Processing the call arguments (line 755)
    # Getting the type of 'a' (line 755)
    a_126048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 14), 'a', False)
    # Processing the call keyword arguments (line 755)
    # Getting the type of 'False' (line 755)
    False_126049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 22), 'False', False)
    keyword_126050 = False_126049
    # Getting the type of 'True' (line 755)
    True_126051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 35), 'True', False)
    keyword_126052 = True_126051
    # Getting the type of 'b' (line 755)
    b_126053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 47), 'b', False)
    # Obtaining the member 'ndim' of a type (line 755)
    ndim_126054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 47), b_126053, 'ndim')
    keyword_126055 = ndim_126054
    kwargs_126056 = {'subok': keyword_126052, 'copy': keyword_126050, 'ndmin': keyword_126055}
    # Getting the type of 'array' (line 755)
    array_126047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'array', False)
    # Calling array(args, kwargs) (line 755)
    array_call_result_126057 = invoke(stypy.reporting.localization.Localization(__file__, 755, 8), array_126047, *[a_126048], **kwargs_126056)
    
    # Assigning a type to the variable 'a' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'a', array_call_result_126057)
    
    # Assigning a Tuple to a Tuple (line 756):
    
    # Assigning a Attribute to a Name (line 756):
    # Getting the type of 'b' (line 756)
    b_126058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 15), 'b')
    # Obtaining the member 'ndim' of a type (line 756)
    ndim_126059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 15), b_126058, 'ndim')
    # Assigning a type to the variable 'tuple_assignment_125186' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'tuple_assignment_125186', ndim_126059)
    
    # Assigning a Attribute to a Name (line 756):
    # Getting the type of 'a' (line 756)
    a_126060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 23), 'a')
    # Obtaining the member 'ndim' of a type (line 756)
    ndim_126061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 23), a_126060, 'ndim')
    # Assigning a type to the variable 'tuple_assignment_125187' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'tuple_assignment_125187', ndim_126061)
    
    # Assigning a Name to a Name (line 756):
    # Getting the type of 'tuple_assignment_125186' (line 756)
    tuple_assignment_125186_126062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'tuple_assignment_125186')
    # Assigning a type to the variable 'ndb' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'ndb', tuple_assignment_125186_126062)
    
    # Assigning a Name to a Name (line 756):
    # Getting the type of 'tuple_assignment_125187' (line 756)
    tuple_assignment_125187_126063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 4), 'tuple_assignment_125187')
    # Assigning a type to the variable 'nda' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 9), 'nda', tuple_assignment_125187_126063)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'nda' (line 757)
    nda_126064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'nda')
    int_126065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 15), 'int')
    # Applying the binary operator '==' (line 757)
    result_eq_126066 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 8), '==', nda_126064, int_126065)
    
    
    # Getting the type of 'ndb' (line 757)
    ndb_126067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 20), 'ndb')
    int_126068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 27), 'int')
    # Applying the binary operator '==' (line 757)
    result_eq_126069 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 20), '==', ndb_126067, int_126068)
    
    # Applying the binary operator 'or' (line 757)
    result_or_keyword_126070 = python_operator(stypy.reporting.localization.Localization(__file__, 757, 8), 'or', result_eq_126066, result_eq_126069)
    
    # Testing the type of an if condition (line 757)
    if_condition_126071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 757, 4), result_or_keyword_126070)
    # Assigning a type to the variable 'if_condition_126071' (line 757)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'if_condition_126071', if_condition_126071)
    # SSA begins for if statement (line 757)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to multiply(...): (line 758)
    # Processing the call arguments (line 758)
    # Getting the type of 'a' (line 758)
    a_126074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 28), 'a', False)
    # Getting the type of 'b' (line 758)
    b_126075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 31), 'b', False)
    # Processing the call keyword arguments (line 758)
    kwargs_126076 = {}
    # Getting the type of '_nx' (line 758)
    _nx_126072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 15), '_nx', False)
    # Obtaining the member 'multiply' of a type (line 758)
    multiply_126073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 15), _nx_126072, 'multiply')
    # Calling multiply(args, kwargs) (line 758)
    multiply_call_result_126077 = invoke(stypy.reporting.localization.Localization(__file__, 758, 15), multiply_126073, *[a_126074, b_126075], **kwargs_126076)
    
    # Assigning a type to the variable 'stypy_return_type' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'stypy_return_type', multiply_call_result_126077)
    # SSA join for if statement (line 757)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 759):
    
    # Assigning a Attribute to a Name (line 759):
    # Getting the type of 'a' (line 759)
    a_126078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 10), 'a')
    # Obtaining the member 'shape' of a type (line 759)
    shape_126079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 10), a_126078, 'shape')
    # Assigning a type to the variable 'as_' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'as_', shape_126079)
    
    # Assigning a Attribute to a Name (line 760):
    
    # Assigning a Attribute to a Name (line 760):
    # Getting the type of 'b' (line 760)
    b_126080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 9), 'b')
    # Obtaining the member 'shape' of a type (line 760)
    shape_126081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 9), b_126080, 'shape')
    # Assigning a type to the variable 'bs' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'bs', shape_126081)
    
    
    # Getting the type of 'a' (line 761)
    a_126082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 11), 'a')
    # Obtaining the member 'flags' of a type (line 761)
    flags_126083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 11), a_126082, 'flags')
    # Obtaining the member 'contiguous' of a type (line 761)
    contiguous_126084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 11), flags_126083, 'contiguous')
    # Applying the 'not' unary operator (line 761)
    result_not__126085 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 7), 'not', contiguous_126084)
    
    # Testing the type of an if condition (line 761)
    if_condition_126086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 761, 4), result_not__126085)
    # Assigning a type to the variable 'if_condition_126086' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'if_condition_126086', if_condition_126086)
    # SSA begins for if statement (line 761)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 762):
    
    # Assigning a Call to a Name (line 762):
    
    # Call to reshape(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'a' (line 762)
    a_126088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 20), 'a', False)
    # Getting the type of 'as_' (line 762)
    as__126089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 23), 'as_', False)
    # Processing the call keyword arguments (line 762)
    kwargs_126090 = {}
    # Getting the type of 'reshape' (line 762)
    reshape_126087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'reshape', False)
    # Calling reshape(args, kwargs) (line 762)
    reshape_call_result_126091 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), reshape_126087, *[a_126088, as__126089], **kwargs_126090)
    
    # Assigning a type to the variable 'a' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'a', reshape_call_result_126091)
    # SSA join for if statement (line 761)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b' (line 763)
    b_126092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 11), 'b')
    # Obtaining the member 'flags' of a type (line 763)
    flags_126093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 11), b_126092, 'flags')
    # Obtaining the member 'contiguous' of a type (line 763)
    contiguous_126094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 11), flags_126093, 'contiguous')
    # Applying the 'not' unary operator (line 763)
    result_not__126095 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 7), 'not', contiguous_126094)
    
    # Testing the type of an if condition (line 763)
    if_condition_126096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 763, 4), result_not__126095)
    # Assigning a type to the variable 'if_condition_126096' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'if_condition_126096', if_condition_126096)
    # SSA begins for if statement (line 763)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 764):
    
    # Assigning a Call to a Name (line 764):
    
    # Call to reshape(...): (line 764)
    # Processing the call arguments (line 764)
    # Getting the type of 'b' (line 764)
    b_126098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 20), 'b', False)
    # Getting the type of 'bs' (line 764)
    bs_126099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 23), 'bs', False)
    # Processing the call keyword arguments (line 764)
    kwargs_126100 = {}
    # Getting the type of 'reshape' (line 764)
    reshape_126097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'reshape', False)
    # Calling reshape(args, kwargs) (line 764)
    reshape_call_result_126101 = invoke(stypy.reporting.localization.Localization(__file__, 764, 12), reshape_126097, *[b_126098, bs_126099], **kwargs_126100)
    
    # Assigning a type to the variable 'b' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'b', reshape_call_result_126101)
    # SSA join for if statement (line 763)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 765):
    
    # Assigning a Name to a Name (line 765):
    # Getting the type of 'ndb' (line 765)
    ndb_126102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 9), 'ndb')
    # Assigning a type to the variable 'nd' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 4), 'nd', ndb_126102)
    
    
    # Getting the type of 'ndb' (line 766)
    ndb_126103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'ndb')
    # Getting the type of 'nda' (line 766)
    nda_126104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 15), 'nda')
    # Applying the binary operator '!=' (line 766)
    result_ne_126105 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 8), '!=', ndb_126103, nda_126104)
    
    # Testing the type of an if condition (line 766)
    if_condition_126106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 4), result_ne_126105)
    # Assigning a type to the variable 'if_condition_126106' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 4), 'if_condition_126106', if_condition_126106)
    # SSA begins for if statement (line 766)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'ndb' (line 767)
    ndb_126107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'ndb')
    # Getting the type of 'nda' (line 767)
    nda_126108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 18), 'nda')
    # Applying the binary operator '>' (line 767)
    result_gt_126109 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 12), '>', ndb_126107, nda_126108)
    
    # Testing the type of an if condition (line 767)
    if_condition_126110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 767, 8), result_gt_126109)
    # Assigning a type to the variable 'if_condition_126110' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'if_condition_126110', if_condition_126110)
    # SSA begins for if statement (line 767)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 768):
    
    # Assigning a BinOp to a Name (line 768):
    
    # Obtaining an instance of the builtin type 'tuple' (line 768)
    tuple_126111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 768)
    # Adding element type (line 768)
    int_126112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 768, 19), tuple_126111, int_126112)
    
    # Getting the type of 'ndb' (line 768)
    ndb_126113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 24), 'ndb')
    # Getting the type of 'nda' (line 768)
    nda_126114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 28), 'nda')
    # Applying the binary operator '-' (line 768)
    result_sub_126115 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 24), '-', ndb_126113, nda_126114)
    
    # Applying the binary operator '*' (line 768)
    result_mul_126116 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 18), '*', tuple_126111, result_sub_126115)
    
    # Getting the type of 'as_' (line 768)
    as__126117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 35), 'as_')
    # Applying the binary operator '+' (line 768)
    result_add_126118 = python_operator(stypy.reporting.localization.Localization(__file__, 768, 18), '+', result_mul_126116, as__126117)
    
    # Assigning a type to the variable 'as_' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 12), 'as_', result_add_126118)
    # SSA branch for the else part of an if statement (line 767)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 770):
    
    # Assigning a BinOp to a Name (line 770):
    
    # Obtaining an instance of the builtin type 'tuple' (line 770)
    tuple_126119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 770)
    # Adding element type (line 770)
    int_126120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 770, 18), tuple_126119, int_126120)
    
    # Getting the type of 'nda' (line 770)
    nda_126121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 23), 'nda')
    # Getting the type of 'ndb' (line 770)
    ndb_126122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 27), 'ndb')
    # Applying the binary operator '-' (line 770)
    result_sub_126123 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 23), '-', nda_126121, ndb_126122)
    
    # Applying the binary operator '*' (line 770)
    result_mul_126124 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 17), '*', tuple_126119, result_sub_126123)
    
    # Getting the type of 'bs' (line 770)
    bs_126125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 34), 'bs')
    # Applying the binary operator '+' (line 770)
    result_add_126126 = python_operator(stypy.reporting.localization.Localization(__file__, 770, 17), '+', result_mul_126124, bs_126125)
    
    # Assigning a type to the variable 'bs' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 12), 'bs', result_add_126126)
    
    # Assigning a Name to a Name (line 771):
    
    # Assigning a Name to a Name (line 771):
    # Getting the type of 'nda' (line 771)
    nda_126127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 17), 'nda')
    # Assigning a type to the variable 'nd' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 12), 'nd', nda_126127)
    # SSA join for if statement (line 767)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 766)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 772):
    
    # Assigning a Call to a Name (line 772):
    
    # Call to reshape(...): (line 772)
    # Processing the call arguments (line 772)
    # Getting the type of 'as_' (line 772)
    as__126134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 33), 'as_', False)
    # Getting the type of 'bs' (line 772)
    bs_126135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 37), 'bs', False)
    # Applying the binary operator '+' (line 772)
    result_add_126136 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 33), '+', as__126134, bs_126135)
    
    # Processing the call keyword arguments (line 772)
    kwargs_126137 = {}
    
    # Call to outer(...): (line 772)
    # Processing the call arguments (line 772)
    # Getting the type of 'a' (line 772)
    a_126129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 19), 'a', False)
    # Getting the type of 'b' (line 772)
    b_126130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 22), 'b', False)
    # Processing the call keyword arguments (line 772)
    kwargs_126131 = {}
    # Getting the type of 'outer' (line 772)
    outer_126128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 13), 'outer', False)
    # Calling outer(args, kwargs) (line 772)
    outer_call_result_126132 = invoke(stypy.reporting.localization.Localization(__file__, 772, 13), outer_126128, *[a_126129, b_126130], **kwargs_126131)
    
    # Obtaining the member 'reshape' of a type (line 772)
    reshape_126133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 13), outer_call_result_126132, 'reshape')
    # Calling reshape(args, kwargs) (line 772)
    reshape_call_result_126138 = invoke(stypy.reporting.localization.Localization(__file__, 772, 13), reshape_126133, *[result_add_126136], **kwargs_126137)
    
    # Assigning a type to the variable 'result' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'result', reshape_call_result_126138)
    
    # Assigning a BinOp to a Name (line 773):
    
    # Assigning a BinOp to a Name (line 773):
    # Getting the type of 'nd' (line 773)
    nd_126139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 11), 'nd')
    int_126140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 773, 14), 'int')
    # Applying the binary operator '-' (line 773)
    result_sub_126141 = python_operator(stypy.reporting.localization.Localization(__file__, 773, 11), '-', nd_126139, int_126140)
    
    # Assigning a type to the variable 'axis' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'axis', result_sub_126141)
    
    
    # Call to range(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'nd' (line 774)
    nd_126143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 19), 'nd', False)
    # Processing the call keyword arguments (line 774)
    kwargs_126144 = {}
    # Getting the type of 'range' (line 774)
    range_126142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 13), 'range', False)
    # Calling range(args, kwargs) (line 774)
    range_call_result_126145 = invoke(stypy.reporting.localization.Localization(__file__, 774, 13), range_126142, *[nd_126143], **kwargs_126144)
    
    # Testing the type of a for loop iterable (line 774)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 774, 4), range_call_result_126145)
    # Getting the type of the for loop variable (line 774)
    for_loop_var_126146 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 774, 4), range_call_result_126145)
    # Assigning a type to the variable '_' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), '_', for_loop_var_126146)
    # SSA begins for a for statement (line 774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 775):
    
    # Assigning a Call to a Name (line 775):
    
    # Call to concatenate(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'result' (line 775)
    result_126148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 29), 'result', False)
    # Processing the call keyword arguments (line 775)
    # Getting the type of 'axis' (line 775)
    axis_126149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 42), 'axis', False)
    keyword_126150 = axis_126149
    kwargs_126151 = {'axis': keyword_126150}
    # Getting the type of 'concatenate' (line 775)
    concatenate_126147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 17), 'concatenate', False)
    # Calling concatenate(args, kwargs) (line 775)
    concatenate_call_result_126152 = invoke(stypy.reporting.localization.Localization(__file__, 775, 17), concatenate_126147, *[result_126148], **kwargs_126151)
    
    # Assigning a type to the variable 'result' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'result', concatenate_call_result_126152)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 776):
    
    # Assigning a Call to a Name (line 776):
    
    # Call to get_array_prepare(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'a' (line 776)
    a_126154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 32), 'a', False)
    # Getting the type of 'b' (line 776)
    b_126155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 35), 'b', False)
    # Processing the call keyword arguments (line 776)
    kwargs_126156 = {}
    # Getting the type of 'get_array_prepare' (line 776)
    get_array_prepare_126153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 14), 'get_array_prepare', False)
    # Calling get_array_prepare(args, kwargs) (line 776)
    get_array_prepare_call_result_126157 = invoke(stypy.reporting.localization.Localization(__file__, 776, 14), get_array_prepare_126153, *[a_126154, b_126155], **kwargs_126156)
    
    # Assigning a type to the variable 'wrapper' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'wrapper', get_array_prepare_call_result_126157)
    
    # Type idiom detected: calculating its left and rigth part (line 777)
    # Getting the type of 'wrapper' (line 777)
    wrapper_126158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 4), 'wrapper')
    # Getting the type of 'None' (line 777)
    None_126159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 22), 'None')
    
    (may_be_126160, more_types_in_union_126161) = may_not_be_none(wrapper_126158, None_126159)

    if may_be_126160:

        if more_types_in_union_126161:
            # Runtime conditional SSA (line 777)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 778):
        
        # Assigning a Call to a Name (line 778):
        
        # Call to wrapper(...): (line 778)
        # Processing the call arguments (line 778)
        # Getting the type of 'result' (line 778)
        result_126163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 25), 'result', False)
        # Processing the call keyword arguments (line 778)
        kwargs_126164 = {}
        # Getting the type of 'wrapper' (line 778)
        wrapper_126162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 17), 'wrapper', False)
        # Calling wrapper(args, kwargs) (line 778)
        wrapper_call_result_126165 = invoke(stypy.reporting.localization.Localization(__file__, 778, 17), wrapper_126162, *[result_126163], **kwargs_126164)
        
        # Assigning a type to the variable 'result' (line 778)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 778, 8), 'result', wrapper_call_result_126165)

        if more_types_in_union_126161:
            # SSA join for if statement (line 777)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 779):
    
    # Assigning a Call to a Name (line 779):
    
    # Call to get_array_wrap(...): (line 779)
    # Processing the call arguments (line 779)
    # Getting the type of 'a' (line 779)
    a_126167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 29), 'a', False)
    # Getting the type of 'b' (line 779)
    b_126168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 32), 'b', False)
    # Processing the call keyword arguments (line 779)
    kwargs_126169 = {}
    # Getting the type of 'get_array_wrap' (line 779)
    get_array_wrap_126166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 14), 'get_array_wrap', False)
    # Calling get_array_wrap(args, kwargs) (line 779)
    get_array_wrap_call_result_126170 = invoke(stypy.reporting.localization.Localization(__file__, 779, 14), get_array_wrap_126166, *[a_126167, b_126168], **kwargs_126169)
    
    # Assigning a type to the variable 'wrapper' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'wrapper', get_array_wrap_call_result_126170)
    
    # Type idiom detected: calculating its left and rigth part (line 780)
    # Getting the type of 'wrapper' (line 780)
    wrapper_126171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 4), 'wrapper')
    # Getting the type of 'None' (line 780)
    None_126172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 22), 'None')
    
    (may_be_126173, more_types_in_union_126174) = may_not_be_none(wrapper_126171, None_126172)

    if may_be_126173:

        if more_types_in_union_126174:
            # Runtime conditional SSA (line 780)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 781):
        
        # Assigning a Call to a Name (line 781):
        
        # Call to wrapper(...): (line 781)
        # Processing the call arguments (line 781)
        # Getting the type of 'result' (line 781)
        result_126176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 25), 'result', False)
        # Processing the call keyword arguments (line 781)
        kwargs_126177 = {}
        # Getting the type of 'wrapper' (line 781)
        wrapper_126175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 17), 'wrapper', False)
        # Calling wrapper(args, kwargs) (line 781)
        wrapper_call_result_126178 = invoke(stypy.reporting.localization.Localization(__file__, 781, 17), wrapper_126175, *[result_126176], **kwargs_126177)
        
        # Assigning a type to the variable 'result' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 'result', wrapper_call_result_126178)

        if more_types_in_union_126174:
            # SSA join for if statement (line 780)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 782)
    result_126179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'stypy_return_type', result_126179)
    
    # ################# End of 'kron(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kron' in the type store
    # Getting the type of 'stypy_return_type' (line 686)
    stypy_return_type_126180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126180)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kron'
    return stypy_return_type_126180

# Assigning a type to the variable 'kron' (line 686)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 0), 'kron', kron)

@norecursion
def tile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tile'
    module_type_store = module_type_store.open_function_context('tile', 785, 0, False)
    
    # Passed parameters checking function
    tile.stypy_localization = localization
    tile.stypy_type_of_self = None
    tile.stypy_type_store = module_type_store
    tile.stypy_function_name = 'tile'
    tile.stypy_param_names_list = ['A', 'reps']
    tile.stypy_varargs_param_name = None
    tile.stypy_kwargs_param_name = None
    tile.stypy_call_defaults = defaults
    tile.stypy_call_varargs = varargs
    tile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tile', ['A', 'reps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tile', localization, ['A', 'reps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tile(...)' code ##################

    str_126181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, (-1)), 'str', "\n    Construct an array by repeating A the number of times given by reps.\n\n    If `reps` has length ``d``, the result will have dimension of\n    ``max(d, A.ndim)``.\n\n    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new\n    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,\n    or shape (1, 1, 3) for 3-D replication. If this is not the desired\n    behavior, promote `A` to d-dimensions manually before calling this\n    function.\n\n    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.\n    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as\n    (1, 1, 2, 2).\n\n    Note : Although tile may be used for broadcasting, it is strongly\n    recommended to use numpy's broadcasting operations and functions.\n\n    Parameters\n    ----------\n    A : array_like\n        The input array.\n    reps : array_like\n        The number of repetitions of `A` along each axis.\n\n    Returns\n    -------\n    c : ndarray\n        The tiled output array.\n\n    See Also\n    --------\n    repeat : Repeat elements of an array.\n    broadcast_to : Broadcast an array to a new shape\n\n    Examples\n    --------\n    >>> a = np.array([0, 1, 2])\n    >>> np.tile(a, 2)\n    array([0, 1, 2, 0, 1, 2])\n    >>> np.tile(a, (2, 2))\n    array([[0, 1, 2, 0, 1, 2],\n           [0, 1, 2, 0, 1, 2]])\n    >>> np.tile(a, (2, 1, 2))\n    array([[[0, 1, 2, 0, 1, 2]],\n           [[0, 1, 2, 0, 1, 2]]])\n\n    >>> b = np.array([[1, 2], [3, 4]])\n    >>> np.tile(b, 2)\n    array([[1, 2, 1, 2],\n           [3, 4, 3, 4]])\n    >>> np.tile(b, (2, 1))\n    array([[1, 2],\n           [3, 4],\n           [1, 2],\n           [3, 4]])\n\n    >>> c = np.array([1,2,3,4])\n    >>> np.tile(c,(4,1))\n    array([[1, 2, 3, 4],\n           [1, 2, 3, 4],\n           [1, 2, 3, 4],\n           [1, 2, 3, 4]])\n    ")
    
    
    # SSA begins for try-except statement (line 851)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 852):
    
    # Assigning a Call to a Name (line 852):
    
    # Call to tuple(...): (line 852)
    # Processing the call arguments (line 852)
    # Getting the type of 'reps' (line 852)
    reps_126183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 20), 'reps', False)
    # Processing the call keyword arguments (line 852)
    kwargs_126184 = {}
    # Getting the type of 'tuple' (line 852)
    tuple_126182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 14), 'tuple', False)
    # Calling tuple(args, kwargs) (line 852)
    tuple_call_result_126185 = invoke(stypy.reporting.localization.Localization(__file__, 852, 14), tuple_126182, *[reps_126183], **kwargs_126184)
    
    # Assigning a type to the variable 'tup' (line 852)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'tup', tuple_call_result_126185)
    # SSA branch for the except part of a try statement (line 851)
    # SSA branch for the except 'TypeError' branch of a try statement (line 851)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Tuple to a Name (line 854):
    
    # Assigning a Tuple to a Name (line 854):
    
    # Obtaining an instance of the builtin type 'tuple' (line 854)
    tuple_126186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 854)
    # Adding element type (line 854)
    # Getting the type of 'reps' (line 854)
    reps_126187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 15), 'reps')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 854, 15), tuple_126186, reps_126187)
    
    # Assigning a type to the variable 'tup' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'tup', tuple_126186)
    # SSA join for try-except statement (line 851)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 855):
    
    # Assigning a Call to a Name (line 855):
    
    # Call to len(...): (line 855)
    # Processing the call arguments (line 855)
    # Getting the type of 'tup' (line 855)
    tup_126189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 12), 'tup', False)
    # Processing the call keyword arguments (line 855)
    kwargs_126190 = {}
    # Getting the type of 'len' (line 855)
    len_126188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'len', False)
    # Calling len(args, kwargs) (line 855)
    len_call_result_126191 = invoke(stypy.reporting.localization.Localization(__file__, 855, 8), len_126188, *[tup_126189], **kwargs_126190)
    
    # Assigning a type to the variable 'd' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 4), 'd', len_call_result_126191)
    
    
    # Evaluating a boolean operation
    
    # Call to all(...): (line 856)
    # Processing the call arguments (line 856)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 856, 11, True)
    # Calculating comprehension expression
    # Getting the type of 'tup' (line 856)
    tup_126196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 27), 'tup', False)
    comprehension_126197 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 856, 11), tup_126196)
    # Assigning a type to the variable 'x' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 11), 'x', comprehension_126197)
    
    # Getting the type of 'x' (line 856)
    x_126193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 11), 'x', False)
    int_126194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 16), 'int')
    # Applying the binary operator '==' (line 856)
    result_eq_126195 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 11), '==', x_126193, int_126194)
    
    list_126198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 856, 11), list_126198, result_eq_126195)
    # Processing the call keyword arguments (line 856)
    kwargs_126199 = {}
    # Getting the type of 'all' (line 856)
    all_126192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 7), 'all', False)
    # Calling all(args, kwargs) (line 856)
    all_call_result_126200 = invoke(stypy.reporting.localization.Localization(__file__, 856, 7), all_126192, *[list_126198], **kwargs_126199)
    
    
    # Call to isinstance(...): (line 856)
    # Processing the call arguments (line 856)
    # Getting the type of 'A' (line 856)
    A_126202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 47), 'A', False)
    # Getting the type of '_nx' (line 856)
    _nx_126203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 50), '_nx', False)
    # Obtaining the member 'ndarray' of a type (line 856)
    ndarray_126204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 50), _nx_126203, 'ndarray')
    # Processing the call keyword arguments (line 856)
    kwargs_126205 = {}
    # Getting the type of 'isinstance' (line 856)
    isinstance_126201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 36), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 856)
    isinstance_call_result_126206 = invoke(stypy.reporting.localization.Localization(__file__, 856, 36), isinstance_126201, *[A_126202, ndarray_126204], **kwargs_126205)
    
    # Applying the binary operator 'and' (line 856)
    result_and_keyword_126207 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 7), 'and', all_call_result_126200, isinstance_call_result_126206)
    
    # Testing the type of an if condition (line 856)
    if_condition_126208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 856, 4), result_and_keyword_126207)
    # Assigning a type to the variable 'if_condition_126208' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'if_condition_126208', if_condition_126208)
    # SSA begins for if statement (line 856)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 859)
    # Processing the call arguments (line 859)
    # Getting the type of 'A' (line 859)
    A_126211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 25), 'A', False)
    # Processing the call keyword arguments (line 859)
    # Getting the type of 'True' (line 859)
    True_126212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 33), 'True', False)
    keyword_126213 = True_126212
    # Getting the type of 'True' (line 859)
    True_126214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 45), 'True', False)
    keyword_126215 = True_126214
    # Getting the type of 'd' (line 859)
    d_126216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 57), 'd', False)
    keyword_126217 = d_126216
    kwargs_126218 = {'subok': keyword_126215, 'copy': keyword_126213, 'ndmin': keyword_126217}
    # Getting the type of '_nx' (line 859)
    _nx_126209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 15), '_nx', False)
    # Obtaining the member 'array' of a type (line 859)
    array_126210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 15), _nx_126209, 'array')
    # Calling array(args, kwargs) (line 859)
    array_call_result_126219 = invoke(stypy.reporting.localization.Localization(__file__, 859, 15), array_126210, *[A_126211], **kwargs_126218)
    
    # Assigning a type to the variable 'stypy_return_type' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'stypy_return_type', array_call_result_126219)
    # SSA branch for the else part of an if statement (line 856)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 863):
    
    # Assigning a Call to a Name (line 863):
    
    # Call to array(...): (line 863)
    # Processing the call arguments (line 863)
    # Getting the type of 'A' (line 863)
    A_126222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 22), 'A', False)
    # Processing the call keyword arguments (line 863)
    # Getting the type of 'False' (line 863)
    False_126223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 30), 'False', False)
    keyword_126224 = False_126223
    # Getting the type of 'True' (line 863)
    True_126225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 43), 'True', False)
    keyword_126226 = True_126225
    # Getting the type of 'd' (line 863)
    d_126227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 55), 'd', False)
    keyword_126228 = d_126227
    kwargs_126229 = {'subok': keyword_126226, 'copy': keyword_126224, 'ndmin': keyword_126228}
    # Getting the type of '_nx' (line 863)
    _nx_126220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 12), '_nx', False)
    # Obtaining the member 'array' of a type (line 863)
    array_126221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 863, 12), _nx_126220, 'array')
    # Calling array(args, kwargs) (line 863)
    array_call_result_126230 = invoke(stypy.reporting.localization.Localization(__file__, 863, 12), array_126221, *[A_126222], **kwargs_126229)
    
    # Assigning a type to the variable 'c' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'c', array_call_result_126230)
    # SSA join for if statement (line 856)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 864)
    d_126231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 8), 'd')
    # Getting the type of 'c' (line 864)
    c_126232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 864)
    ndim_126233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 864, 12), c_126232, 'ndim')
    # Applying the binary operator '<' (line 864)
    result_lt_126234 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 8), '<', d_126231, ndim_126233)
    
    # Testing the type of an if condition (line 864)
    if_condition_126235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 864, 4), result_lt_126234)
    # Assigning a type to the variable 'if_condition_126235' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'if_condition_126235', if_condition_126235)
    # SSA begins for if statement (line 864)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 865):
    
    # Assigning a BinOp to a Name (line 865):
    
    # Obtaining an instance of the builtin type 'tuple' (line 865)
    tuple_126236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 865)
    # Adding element type (line 865)
    int_126237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 865, 15), tuple_126236, int_126237)
    
    # Getting the type of 'c' (line 865)
    c_126238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 20), 'c')
    # Obtaining the member 'ndim' of a type (line 865)
    ndim_126239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 20), c_126238, 'ndim')
    # Getting the type of 'd' (line 865)
    d_126240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 27), 'd')
    # Applying the binary operator '-' (line 865)
    result_sub_126241 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 20), '-', ndim_126239, d_126240)
    
    # Applying the binary operator '*' (line 865)
    result_mul_126242 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 14), '*', tuple_126236, result_sub_126241)
    
    # Getting the type of 'tup' (line 865)
    tup_126243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 32), 'tup')
    # Applying the binary operator '+' (line 865)
    result_add_126244 = python_operator(stypy.reporting.localization.Localization(__file__, 865, 14), '+', result_mul_126242, tup_126243)
    
    # Assigning a type to the variable 'tup' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'tup', result_add_126244)
    # SSA join for if statement (line 864)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 866):
    
    # Assigning a Call to a Name (line 866):
    
    # Call to tuple(...): (line 866)
    # Processing the call arguments (line 866)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 866, 22, True)
    # Calculating comprehension expression
    
    # Call to zip(...): (line 866)
    # Processing the call arguments (line 866)
    # Getting the type of 'c' (line 866)
    c_126250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 42), 'c', False)
    # Obtaining the member 'shape' of a type (line 866)
    shape_126251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 42), c_126250, 'shape')
    # Getting the type of 'tup' (line 866)
    tup_126252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 51), 'tup', False)
    # Processing the call keyword arguments (line 866)
    kwargs_126253 = {}
    # Getting the type of 'zip' (line 866)
    zip_126249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 38), 'zip', False)
    # Calling zip(args, kwargs) (line 866)
    zip_call_result_126254 = invoke(stypy.reporting.localization.Localization(__file__, 866, 38), zip_126249, *[shape_126251, tup_126252], **kwargs_126253)
    
    comprehension_126255 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 866, 22), zip_call_result_126254)
    # Assigning a type to the variable 's' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 22), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 866, 22), comprehension_126255))
    # Assigning a type to the variable 't' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 22), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 866, 22), comprehension_126255))
    # Getting the type of 's' (line 866)
    s_126246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 22), 's', False)
    # Getting the type of 't' (line 866)
    t_126247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 24), 't', False)
    # Applying the binary operator '*' (line 866)
    result_mul_126248 = python_operator(stypy.reporting.localization.Localization(__file__, 866, 22), '*', s_126246, t_126247)
    
    list_126256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 866, 22), list_126256, result_mul_126248)
    # Processing the call keyword arguments (line 866)
    kwargs_126257 = {}
    # Getting the type of 'tuple' (line 866)
    tuple_126245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 866)
    tuple_call_result_126258 = invoke(stypy.reporting.localization.Localization(__file__, 866, 16), tuple_126245, *[list_126256], **kwargs_126257)
    
    # Assigning a type to the variable 'shape_out' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), 'shape_out', tuple_call_result_126258)
    
    # Assigning a Attribute to a Name (line 867):
    
    # Assigning a Attribute to a Name (line 867):
    # Getting the type of 'c' (line 867)
    c_126259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'c')
    # Obtaining the member 'size' of a type (line 867)
    size_126260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 8), c_126259, 'size')
    # Assigning a type to the variable 'n' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'n', size_126260)
    
    
    # Getting the type of 'n' (line 868)
    n_126261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 7), 'n')
    int_126262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 11), 'int')
    # Applying the binary operator '>' (line 868)
    result_gt_126263 = python_operator(stypy.reporting.localization.Localization(__file__, 868, 7), '>', n_126261, int_126262)
    
    # Testing the type of an if condition (line 868)
    if_condition_126264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 868, 4), result_gt_126263)
    # Assigning a type to the variable 'if_condition_126264' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'if_condition_126264', if_condition_126264)
    # SSA begins for if statement (line 868)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to zip(...): (line 869)
    # Processing the call arguments (line 869)
    # Getting the type of 'c' (line 869)
    c_126266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 32), 'c', False)
    # Obtaining the member 'shape' of a type (line 869)
    shape_126267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 32), c_126266, 'shape')
    # Getting the type of 'tup' (line 869)
    tup_126268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 41), 'tup', False)
    # Processing the call keyword arguments (line 869)
    kwargs_126269 = {}
    # Getting the type of 'zip' (line 869)
    zip_126265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 28), 'zip', False)
    # Calling zip(args, kwargs) (line 869)
    zip_call_result_126270 = invoke(stypy.reporting.localization.Localization(__file__, 869, 28), zip_126265, *[shape_126267, tup_126268], **kwargs_126269)
    
    # Testing the type of a for loop iterable (line 869)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 869, 8), zip_call_result_126270)
    # Getting the type of the for loop variable (line 869)
    for_loop_var_126271 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 869, 8), zip_call_result_126270)
    # Assigning a type to the variable 'dim_in' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'dim_in', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 869, 8), for_loop_var_126271))
    # Assigning a type to the variable 'nrep' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'nrep', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 869, 8), for_loop_var_126271))
    # SSA begins for a for statement (line 869)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'nrep' (line 870)
    nrep_126272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 15), 'nrep')
    int_126273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 23), 'int')
    # Applying the binary operator '!=' (line 870)
    result_ne_126274 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 15), '!=', nrep_126272, int_126273)
    
    # Testing the type of an if condition (line 870)
    if_condition_126275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 870, 12), result_ne_126274)
    # Assigning a type to the variable 'if_condition_126275' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 12), 'if_condition_126275', if_condition_126275)
    # SSA begins for if statement (line 870)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 871):
    
    # Assigning a Call to a Name (line 871):
    
    # Call to repeat(...): (line 871)
    # Processing the call arguments (line 871)
    # Getting the type of 'nrep' (line 871)
    nrep_126283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 44), 'nrep', False)
    int_126284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 50), 'int')
    # Processing the call keyword arguments (line 871)
    kwargs_126285 = {}
    
    # Call to reshape(...): (line 871)
    # Processing the call arguments (line 871)
    int_126278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 30), 'int')
    # Getting the type of 'n' (line 871)
    n_126279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 34), 'n', False)
    # Processing the call keyword arguments (line 871)
    kwargs_126280 = {}
    # Getting the type of 'c' (line 871)
    c_126276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 20), 'c', False)
    # Obtaining the member 'reshape' of a type (line 871)
    reshape_126277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 20), c_126276, 'reshape')
    # Calling reshape(args, kwargs) (line 871)
    reshape_call_result_126281 = invoke(stypy.reporting.localization.Localization(__file__, 871, 20), reshape_126277, *[int_126278, n_126279], **kwargs_126280)
    
    # Obtaining the member 'repeat' of a type (line 871)
    repeat_126282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 20), reshape_call_result_126281, 'repeat')
    # Calling repeat(args, kwargs) (line 871)
    repeat_call_result_126286 = invoke(stypy.reporting.localization.Localization(__file__, 871, 20), repeat_126282, *[nrep_126283, int_126284], **kwargs_126285)
    
    # Assigning a type to the variable 'c' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 16), 'c', repeat_call_result_126286)
    # SSA join for if statement (line 870)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'n' (line 872)
    n_126287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'n')
    # Getting the type of 'dim_in' (line 872)
    dim_in_126288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 18), 'dim_in')
    # Applying the binary operator '//=' (line 872)
    result_ifloordiv_126289 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 12), '//=', n_126287, dim_in_126288)
    # Assigning a type to the variable 'n' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'n', result_ifloordiv_126289)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 868)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reshape(...): (line 873)
    # Processing the call arguments (line 873)
    # Getting the type of 'shape_out' (line 873)
    shape_out_126292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 21), 'shape_out', False)
    # Processing the call keyword arguments (line 873)
    kwargs_126293 = {}
    # Getting the type of 'c' (line 873)
    c_126290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 11), 'c', False)
    # Obtaining the member 'reshape' of a type (line 873)
    reshape_126291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 11), c_126290, 'reshape')
    # Calling reshape(args, kwargs) (line 873)
    reshape_call_result_126294 = invoke(stypy.reporting.localization.Localization(__file__, 873, 11), reshape_126291, *[shape_out_126292], **kwargs_126293)
    
    # Assigning a type to the variable 'stypy_return_type' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 4), 'stypy_return_type', reshape_call_result_126294)
    
    # ################# End of 'tile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tile' in the type store
    # Getting the type of 'stypy_return_type' (line 785)
    stypy_return_type_126295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126295)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tile'
    return stypy_return_type_126295

# Assigning a type to the variable 'tile' (line 785)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 0), 'tile', tile)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
