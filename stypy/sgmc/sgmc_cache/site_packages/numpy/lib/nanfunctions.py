
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Functions that ignore NaN.
3: 
4: Functions
5: ---------
6: 
7: - `nanmin` -- minimum non-NaN value
8: - `nanmax` -- maximum non-NaN value
9: - `nanargmin` -- index of minimum non-NaN value
10: - `nanargmax` -- index of maximum non-NaN value
11: - `nansum` -- sum of non-NaN values
12: - `nanprod` -- product of non-NaN values
13: - `nanmean` -- mean of non-NaN values
14: - `nanvar` -- variance of non-NaN values
15: - `nanstd` -- standard deviation of non-NaN values
16: - `nanmedian` -- median of non-NaN values
17: - `nanpercentile` -- qth percentile of non-NaN values
18: 
19: '''
20: from __future__ import division, absolute_import, print_function
21: 
22: import warnings
23: import numpy as np
24: from numpy.lib.function_base import _ureduce as _ureduce
25: 
26: __all__ = [
27:     'nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean',
28:     'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod',
29:     ]
30: 
31: 
32: def _replace_nan(a, val):
33:     '''
34:     If `a` is of inexact type, make a copy of `a`, replace NaNs with
35:     the `val` value, and return the copy together with a boolean mask
36:     marking the locations where NaNs were present. If `a` is not of
37:     inexact type, do nothing and return `a` together with a mask of None.
38: 
39:     Note that scalars will end up as array scalars, which is important
40:     for using the result as the value of the out argument in some
41:     operations.
42: 
43:     Parameters
44:     ----------
45:     a : array-like
46:         Input array.
47:     val : float
48:         NaN values are set to val before doing the operation.
49: 
50:     Returns
51:     -------
52:     y : ndarray
53:         If `a` is of inexact type, return a copy of `a` with the NaNs
54:         replaced by the fill value, otherwise return `a`.
55:     mask: {bool, None}
56:         If `a` is of inexact type, return a boolean mask marking locations of
57:         NaNs, otherwise return None.
58: 
59:     '''
60:     is_new = not isinstance(a, np.ndarray)
61:     if is_new:
62:         a = np.array(a)
63:     if not issubclass(a.dtype.type, np.inexact):
64:         return a, None
65:     if not is_new:
66:         # need copy
67:         a = np.array(a, subok=True)
68: 
69:     mask = np.isnan(a)
70:     np.copyto(a, val, where=mask)
71:     return a, mask
72: 
73: 
74: def _copyto(a, val, mask):
75:     '''
76:     Replace values in `a` with NaN where `mask` is True.  This differs from
77:     copyto in that it will deal with the case where `a` is a numpy scalar.
78: 
79:     Parameters
80:     ----------
81:     a : ndarray or numpy scalar
82:         Array or numpy scalar some of whose values are to be replaced
83:         by val.
84:     val : numpy scalar
85:         Value used a replacement.
86:     mask : ndarray, scalar
87:         Boolean array. Where True the corresponding element of `a` is
88:         replaced by `val`. Broadcasts.
89: 
90:     Returns
91:     -------
92:     res : ndarray, scalar
93:         Array with elements replaced or scalar `val`.
94: 
95:     '''
96:     if isinstance(a, np.ndarray):
97:         np.copyto(a, val, where=mask, casting='unsafe')
98:     else:
99:         a = a.dtype.type(val)
100:     return a
101: 
102: 
103: def _divide_by_count(a, b, out=None):
104:     '''
105:     Compute a/b ignoring invalid results. If `a` is an array the division
106:     is done in place. If `a` is a scalar, then its type is preserved in the
107:     output. If out is None, then then a is used instead so that the
108:     division is in place. Note that this is only called with `a` an inexact
109:     type.
110: 
111:     Parameters
112:     ----------
113:     a : {ndarray, numpy scalar}
114:         Numerator. Expected to be of inexact type but not checked.
115:     b : {ndarray, numpy scalar}
116:         Denominator.
117:     out : ndarray, optional
118:         Alternate output array in which to place the result.  The default
119:         is ``None``; if provided, it must have the same shape as the
120:         expected output, but the type will be cast if necessary.
121: 
122:     Returns
123:     -------
124:     ret : {ndarray, numpy scalar}
125:         The return value is a/b. If `a` was an ndarray the division is done
126:         in place. If `a` is a numpy scalar, the division preserves its type.
127: 
128:     '''
129:     with np.errstate(invalid='ignore'):
130:         if isinstance(a, np.ndarray):
131:             if out is None:
132:                 return np.divide(a, b, out=a, casting='unsafe')
133:             else:
134:                 return np.divide(a, b, out=out, casting='unsafe')
135:         else:
136:             if out is None:
137:                 return a.dtype.type(a / b)
138:             else:
139:                 # This is questionable, but currently a numpy scalar can
140:                 # be output to a zero dimensional array.
141:                 return np.divide(a, b, out=out, casting='unsafe')
142: 
143: 
144: def nanmin(a, axis=None, out=None, keepdims=False):
145:     '''
146:     Return minimum of an array or minimum along an axis, ignoring any NaNs.
147:     When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
148:     Nan is returned for that slice.
149: 
150:     Parameters
151:     ----------
152:     a : array_like
153:         Array containing numbers whose minimum is desired. If `a` is not an
154:         array, a conversion is attempted.
155:     axis : int, optional
156:         Axis along which the minimum is computed. The default is to compute
157:         the minimum of the flattened array.
158:     out : ndarray, optional
159:         Alternate output array in which to place the result.  The default
160:         is ``None``; if provided, it must have the same shape as the
161:         expected output, but the type will be cast if necessary.  See
162:         `doc.ufuncs` for details.
163: 
164:         .. versionadded:: 1.8.0
165:     keepdims : bool, optional
166:         If this is set to True, the axes which are reduced are left in the
167:         result as dimensions with size one. With this option, the result
168:         will broadcast correctly against the original `a`.
169: 
170:         .. versionadded:: 1.8.0
171: 
172:     Returns
173:     -------
174:     nanmin : ndarray
175:         An array with the same shape as `a`, with the specified axis
176:         removed.  If `a` is a 0-d array, or if axis is None, an ndarray
177:         scalar is returned.  The same dtype as `a` is returned.
178: 
179:     See Also
180:     --------
181:     nanmax :
182:         The maximum value of an array along a given axis, ignoring any NaNs.
183:     amin :
184:         The minimum value of an array along a given axis, propagating any NaNs.
185:     fmin :
186:         Element-wise minimum of two arrays, ignoring any NaNs.
187:     minimum :
188:         Element-wise minimum of two arrays, propagating any NaNs.
189:     isnan :
190:         Shows which elements are Not a Number (NaN).
191:     isfinite:
192:         Shows which elements are neither NaN nor infinity.
193: 
194:     amax, fmax, maximum
195: 
196:     Notes
197:     -----
198:     Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
199:     (IEEE 754). This means that Not a Number is not equivalent to infinity.
200:     Positive infinity is treated as a very large number and negative
201:     infinity is treated as a very small (i.e. negative) number.
202: 
203:     If the input has a integer type the function is equivalent to np.min.
204: 
205:     Examples
206:     --------
207:     >>> a = np.array([[1, 2], [3, np.nan]])
208:     >>> np.nanmin(a)
209:     1.0
210:     >>> np.nanmin(a, axis=0)
211:     array([ 1.,  2.])
212:     >>> np.nanmin(a, axis=1)
213:     array([ 1.,  3.])
214: 
215:     When positive infinity and negative infinity are present:
216: 
217:     >>> np.nanmin([1, 2, np.nan, np.inf])
218:     1.0
219:     >>> np.nanmin([1, 2, np.nan, np.NINF])
220:     -inf
221: 
222:     '''
223:     if not isinstance(a, np.ndarray) or type(a) is np.ndarray:
224:         # Fast, but not safe for subclasses of ndarray
225:         res = np.fmin.reduce(a, axis=axis, out=out, keepdims=keepdims)
226:         if np.isnan(res).any():
227:             warnings.warn("All-NaN axis encountered", RuntimeWarning)
228:     else:
229:         # Slow, but safe for subclasses of ndarray
230:         a, mask = _replace_nan(a, +np.inf)
231:         res = np.amin(a, axis=axis, out=out, keepdims=keepdims)
232:         if mask is None:
233:             return res
234: 
235:         # Check for all-NaN axis
236:         mask = np.all(mask, axis=axis, keepdims=keepdims)
237:         if np.any(mask):
238:             res = _copyto(res, np.nan, mask)
239:             warnings.warn("All-NaN axis encountered", RuntimeWarning)
240:     return res
241: 
242: 
243: def nanmax(a, axis=None, out=None, keepdims=False):
244:     '''
245:     Return the maximum of an array or maximum along an axis, ignoring any
246:     NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is
247:     raised and NaN is returned for that slice.
248: 
249:     Parameters
250:     ----------
251:     a : array_like
252:         Array containing numbers whose maximum is desired. If `a` is not an
253:         array, a conversion is attempted.
254:     axis : int, optional
255:         Axis along which the maximum is computed. The default is to compute
256:         the maximum of the flattened array.
257:     out : ndarray, optional
258:         Alternate output array in which to place the result.  The default
259:         is ``None``; if provided, it must have the same shape as the
260:         expected output, but the type will be cast if necessary.  See
261:         `doc.ufuncs` for details.
262: 
263:         .. versionadded:: 1.8.0
264:     keepdims : bool, optional
265:         If this is set to True, the axes which are reduced are left in the
266:         result as dimensions with size one. With this option, the result
267:         will broadcast correctly against the original `a`.
268: 
269:         .. versionadded:: 1.8.0
270: 
271:     Returns
272:     -------
273:     nanmax : ndarray
274:         An array with the same shape as `a`, with the specified axis removed.
275:         If `a` is a 0-d array, or if axis is None, an ndarray scalar is
276:         returned.  The same dtype as `a` is returned.
277: 
278:     See Also
279:     --------
280:     nanmin :
281:         The minimum value of an array along a given axis, ignoring any NaNs.
282:     amax :
283:         The maximum value of an array along a given axis, propagating any NaNs.
284:     fmax :
285:         Element-wise maximum of two arrays, ignoring any NaNs.
286:     maximum :
287:         Element-wise maximum of two arrays, propagating any NaNs.
288:     isnan :
289:         Shows which elements are Not a Number (NaN).
290:     isfinite:
291:         Shows which elements are neither NaN nor infinity.
292: 
293:     amin, fmin, minimum
294: 
295:     Notes
296:     -----
297:     Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
298:     (IEEE 754). This means that Not a Number is not equivalent to infinity.
299:     Positive infinity is treated as a very large number and negative
300:     infinity is treated as a very small (i.e. negative) number.
301: 
302:     If the input has a integer type the function is equivalent to np.max.
303: 
304:     Examples
305:     --------
306:     >>> a = np.array([[1, 2], [3, np.nan]])
307:     >>> np.nanmax(a)
308:     3.0
309:     >>> np.nanmax(a, axis=0)
310:     array([ 3.,  2.])
311:     >>> np.nanmax(a, axis=1)
312:     array([ 2.,  3.])
313: 
314:     When positive infinity and negative infinity are present:
315: 
316:     >>> np.nanmax([1, 2, np.nan, np.NINF])
317:     2.0
318:     >>> np.nanmax([1, 2, np.nan, np.inf])
319:     inf
320: 
321:     '''
322:     if not isinstance(a, np.ndarray) or type(a) is np.ndarray:
323:         # Fast, but not safe for subclasses of ndarray
324:         res = np.fmax.reduce(a, axis=axis, out=out, keepdims=keepdims)
325:         if np.isnan(res).any():
326:             warnings.warn("All-NaN slice encountered", RuntimeWarning)
327:     else:
328:         # Slow, but safe for subclasses of ndarray
329:         a, mask = _replace_nan(a, -np.inf)
330:         res = np.amax(a, axis=axis, out=out, keepdims=keepdims)
331:         if mask is None:
332:             return res
333: 
334:         # Check for all-NaN axis
335:         mask = np.all(mask, axis=axis, keepdims=keepdims)
336:         if np.any(mask):
337:             res = _copyto(res, np.nan, mask)
338:             warnings.warn("All-NaN axis encountered", RuntimeWarning)
339:     return res
340: 
341: 
342: def nanargmin(a, axis=None):
343:     '''
344:     Return the indices of the minimum values in the specified axis ignoring
345:     NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results
346:     cannot be trusted if a slice contains only NaNs and Infs.
347: 
348:     Parameters
349:     ----------
350:     a : array_like
351:         Input data.
352:     axis : int, optional
353:         Axis along which to operate.  By default flattened input is used.
354: 
355:     Returns
356:     -------
357:     index_array : ndarray
358:         An array of indices or a single index value.
359: 
360:     See Also
361:     --------
362:     argmin, nanargmax
363: 
364:     Examples
365:     --------
366:     >>> a = np.array([[np.nan, 4], [2, 3]])
367:     >>> np.argmin(a)
368:     0
369:     >>> np.nanargmin(a)
370:     2
371:     >>> np.nanargmin(a, axis=0)
372:     array([1, 1])
373:     >>> np.nanargmin(a, axis=1)
374:     array([1, 0])
375: 
376:     '''
377:     a, mask = _replace_nan(a, np.inf)
378:     res = np.argmin(a, axis=axis)
379:     if mask is not None:
380:         mask = np.all(mask, axis=axis)
381:         if np.any(mask):
382:             raise ValueError("All-NaN slice encountered")
383:     return res
384: 
385: 
386: def nanargmax(a, axis=None):
387:     '''
388:     Return the indices of the maximum values in the specified axis ignoring
389:     NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the
390:     results cannot be trusted if a slice contains only NaNs and -Infs.
391: 
392: 
393:     Parameters
394:     ----------
395:     a : array_like
396:         Input data.
397:     axis : int, optional
398:         Axis along which to operate.  By default flattened input is used.
399: 
400:     Returns
401:     -------
402:     index_array : ndarray
403:         An array of indices or a single index value.
404: 
405:     See Also
406:     --------
407:     argmax, nanargmin
408: 
409:     Examples
410:     --------
411:     >>> a = np.array([[np.nan, 4], [2, 3]])
412:     >>> np.argmax(a)
413:     0
414:     >>> np.nanargmax(a)
415:     1
416:     >>> np.nanargmax(a, axis=0)
417:     array([1, 0])
418:     >>> np.nanargmax(a, axis=1)
419:     array([1, 1])
420: 
421:     '''
422:     a, mask = _replace_nan(a, -np.inf)
423:     res = np.argmax(a, axis=axis)
424:     if mask is not None:
425:         mask = np.all(mask, axis=axis)
426:         if np.any(mask):
427:             raise ValueError("All-NaN slice encountered")
428:     return res
429: 
430: 
431: def nansum(a, axis=None, dtype=None, out=None, keepdims=0):
432:     '''
433:     Return the sum of array elements over a given axis treating Not a
434:     Numbers (NaNs) as zero.
435: 
436:     In Numpy versions <= 1.8 Nan is returned for slices that are all-NaN or
437:     empty. In later versions zero is returned.
438: 
439:     Parameters
440:     ----------
441:     a : array_like
442:         Array containing numbers whose sum is desired. If `a` is not an
443:         array, a conversion is attempted.
444:     axis : int, optional
445:         Axis along which the sum is computed. The default is to compute the
446:         sum of the flattened array.
447:     dtype : data-type, optional
448:         The type of the returned array and of the accumulator in which the
449:         elements are summed.  By default, the dtype of `a` is used.  An
450:         exception is when `a` has an integer type with less precision than
451:         the platform (u)intp. In that case, the default will be either
452:         (u)int32 or (u)int64 depending on whether the platform is 32 or 64
453:         bits. For inexact inputs, dtype must be inexact.
454: 
455:         .. versionadded:: 1.8.0
456:     out : ndarray, optional
457:         Alternate output array in which to place the result.  The default
458:         is ``None``. If provided, it must have the same shape as the
459:         expected output, but the type will be cast if necessary.  See
460:         `doc.ufuncs` for details. The casting of NaN to integer can yield
461:         unexpected results.
462: 
463:         .. versionadded:: 1.8.0
464:     keepdims : bool, optional
465:         If True, the axes which are reduced are left in the result as
466:         dimensions with size one. With this option, the result will
467:         broadcast correctly against the original `arr`.
468: 
469:         .. versionadded:: 1.8.0
470: 
471:     Returns
472:     -------
473:     y : ndarray or numpy scalar
474: 
475:     See Also
476:     --------
477:     numpy.sum : Sum across array propagating NaNs.
478:     isnan : Show which elements are NaN.
479:     isfinite: Show which elements are not NaN or +/-inf.
480: 
481:     Notes
482:     -----
483:     If both positive and negative infinity are present, the sum will be Not
484:     A Number (NaN).
485: 
486:     Numpy integer arithmetic is modular. If the size of a sum exceeds the
487:     size of an integer accumulator, its value will wrap around and the
488:     result will be incorrect. Specifying ``dtype=double`` can alleviate
489:     that problem.
490: 
491:     Examples
492:     --------
493:     >>> np.nansum(1)
494:     1
495:     >>> np.nansum([1])
496:     1
497:     >>> np.nansum([1, np.nan])
498:     1.0
499:     >>> a = np.array([[1, 1], [1, np.nan]])
500:     >>> np.nansum(a)
501:     3.0
502:     >>> np.nansum(a, axis=0)
503:     array([ 2.,  1.])
504:     >>> np.nansum([1, np.nan, np.inf])
505:     inf
506:     >>> np.nansum([1, np.nan, np.NINF])
507:     -inf
508:     >>> np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present
509:     nan
510: 
511:     '''
512:     a, mask = _replace_nan(a, 0)
513:     return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
514: 
515: 
516: def nanprod(a, axis=None, dtype=None, out=None, keepdims=0):
517:     '''
518:     Return the product of array elements over a given axis treating Not a
519:     Numbers (NaNs) as zero.
520: 
521:     One is returned for slices that are all-NaN or empty.
522: 
523:     .. versionadded:: 1.10.0
524: 
525:     Parameters
526:     ----------
527:     a : array_like
528:         Array containing numbers whose sum is desired. If `a` is not an
529:         array, a conversion is attempted.
530:     axis : int, optional
531:         Axis along which the product is computed. The default is to compute
532:         the product of the flattened array.
533:     dtype : data-type, optional
534:         The type of the returned array and of the accumulator in which the
535:         elements are summed.  By default, the dtype of `a` is used.  An
536:         exception is when `a` has an integer type with less precision than
537:         the platform (u)intp. In that case, the default will be either
538:         (u)int32 or (u)int64 depending on whether the platform is 32 or 64
539:         bits. For inexact inputs, dtype must be inexact.
540:     out : ndarray, optional
541:         Alternate output array in which to place the result.  The default
542:         is ``None``. If provided, it must have the same shape as the
543:         expected output, but the type will be cast if necessary.  See
544:         `doc.ufuncs` for details. The casting of NaN to integer can yield
545:         unexpected results.
546:     keepdims : bool, optional
547:         If True, the axes which are reduced are left in the result as
548:         dimensions with size one. With this option, the result will
549:         broadcast correctly against the original `arr`.
550: 
551:     Returns
552:     -------
553:     y : ndarray or numpy scalar
554: 
555:     See Also
556:     --------
557:     numpy.prod : Product across array propagating NaNs.
558:     isnan : Show which elements are NaN.
559: 
560:     Notes
561:     -----
562:     Numpy integer arithmetic is modular. If the size of a product exceeds
563:     the size of an integer accumulator, its value will wrap around and the
564:     result will be incorrect. Specifying ``dtype=double`` can alleviate
565:     that problem.
566: 
567:     Examples
568:     --------
569:     >>> np.nanprod(1)
570:     1
571:     >>> np.nanprod([1])
572:     1
573:     >>> np.nanprod([1, np.nan])
574:     1.0
575:     >>> a = np.array([[1, 2], [3, np.nan]])
576:     >>> np.nanprod(a)
577:     6.0
578:     >>> np.nanprod(a, axis=0)
579:     array([ 3.,  2.])
580: 
581:     '''
582:     a, mask = _replace_nan(a, 1)
583:     return np.prod(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
584: 
585: 
586: def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
587:     '''
588:     Compute the arithmetic mean along the specified axis, ignoring NaNs.
589: 
590:     Returns the average of the array elements.  The average is taken over
591:     the flattened array by default, otherwise over the specified axis.
592:     `float64` intermediate and return values are used for integer inputs.
593: 
594:     For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.
595: 
596:     .. versionadded:: 1.8.0
597: 
598:     Parameters
599:     ----------
600:     a : array_like
601:         Array containing numbers whose mean is desired. If `a` is not an
602:         array, a conversion is attempted.
603:     axis : int, optional
604:         Axis along which the means are computed. The default is to compute
605:         the mean of the flattened array.
606:     dtype : data-type, optional
607:         Type to use in computing the mean.  For integer inputs, the default
608:         is `float64`; for inexact inputs, it is the same as the input
609:         dtype.
610:     out : ndarray, optional
611:         Alternate output array in which to place the result.  The default
612:         is ``None``; if provided, it must have the same shape as the
613:         expected output, but the type will be cast if necessary.  See
614:         `doc.ufuncs` for details.
615:     keepdims : bool, optional
616:         If this is set to True, the axes which are reduced are left in the
617:         result as dimensions with size one. With this option, the result
618:         will broadcast correctly against the original `arr`.
619: 
620:     Returns
621:     -------
622:     m : ndarray, see dtype parameter above
623:         If `out=None`, returns a new array containing the mean values,
624:         otherwise a reference to the output array is returned. Nan is
625:         returned for slices that contain only NaNs.
626: 
627:     See Also
628:     --------
629:     average : Weighted average
630:     mean : Arithmetic mean taken while not ignoring NaNs
631:     var, nanvar
632: 
633:     Notes
634:     -----
635:     The arithmetic mean is the sum of the non-NaN elements along the axis
636:     divided by the number of non-NaN elements.
637: 
638:     Note that for floating-point input, the mean is computed using the same
639:     precision the input has.  Depending on the input data, this can cause
640:     the results to be inaccurate, especially for `float32`.  Specifying a
641:     higher-precision accumulator using the `dtype` keyword can alleviate
642:     this issue.
643: 
644:     Examples
645:     --------
646:     >>> a = np.array([[1, np.nan], [3, 4]])
647:     >>> np.nanmean(a)
648:     2.6666666666666665
649:     >>> np.nanmean(a, axis=0)
650:     array([ 2.,  4.])
651:     >>> np.nanmean(a, axis=1)
652:     array([ 1.,  3.5])
653: 
654:     '''
655:     arr, mask = _replace_nan(a, 0)
656:     if mask is None:
657:         return np.mean(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
658: 
659:     if dtype is not None:
660:         dtype = np.dtype(dtype)
661:     if dtype is not None and not issubclass(dtype.type, np.inexact):
662:         raise TypeError("If a is inexact, then dtype must be inexact")
663:     if out is not None and not issubclass(out.dtype.type, np.inexact):
664:         raise TypeError("If a is inexact, then out must be inexact")
665: 
666:     # The warning context speeds things up.
667:     with warnings.catch_warnings():
668:         warnings.simplefilter('ignore')
669:         cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=keepdims)
670:         tot = np.sum(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
671:         avg = _divide_by_count(tot, cnt, out=out)
672: 
673:     isbad = (cnt == 0)
674:     if isbad.any():
675:         warnings.warn("Mean of empty slice", RuntimeWarning)
676:         # NaN is the only possible bad value, so no further
677:         # action is needed to handle bad results.
678:     return avg
679: 
680: 
681: def _nanmedian1d(arr1d, overwrite_input=False):
682:     '''
683:     Private function for rank 1 arrays. Compute the median ignoring NaNs.
684:     See nanmedian for parameter usage
685:     '''
686:     c = np.isnan(arr1d)
687:     s = np.where(c)[0]
688:     if s.size == arr1d.size:
689:         warnings.warn("All-NaN slice encountered", RuntimeWarning)
690:         return np.nan
691:     elif s.size == 0:
692:         return np.median(arr1d, overwrite_input=overwrite_input)
693:     else:
694:         if overwrite_input:
695:             x = arr1d
696:         else:
697:             x = arr1d.copy()
698:         # select non-nans at end of array
699:         enonan = arr1d[-s.size:][~c[-s.size:]]
700:         # fill nans in beginning of array with non-nans of end
701:         x[s[:enonan.size]] = enonan
702:         # slice nans away
703:         return np.median(x[:-s.size], overwrite_input=True)
704: 
705: 
706: def _nanmedian(a, axis=None, out=None, overwrite_input=False):
707:     '''
708:     Private function that doesn't support extended axis or keepdims.
709:     These methods are extended to this function using _ureduce
710:     See nanmedian for parameter usage
711: 
712:     '''
713:     if axis is None or a.ndim == 1:
714:         part = a.ravel()
715:         if out is None:
716:             return _nanmedian1d(part, overwrite_input)
717:         else:
718:             out[...] = _nanmedian1d(part, overwrite_input)
719:             return out
720:     else:
721:         # for small medians use sort + indexing which is still faster than
722:         # apply_along_axis
723:         if a.shape[axis] < 400:
724:             return _nanmedian_small(a, axis, out, overwrite_input)
725:         result = np.apply_along_axis(_nanmedian1d, axis, a, overwrite_input)
726:         if out is not None:
727:             out[...] = result
728:         return result
729: 
730: def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
731:     '''
732:     sort + indexing median, faster for small medians along multiple
733:     dimensions due to the high overhead of apply_along_axis
734: 
735:     see nanmedian for parameter usage
736:     '''
737:     a = np.ma.masked_array(a, np.isnan(a))
738:     m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
739:     for i in range(np.count_nonzero(m.mask.ravel())):
740:         warnings.warn("All-NaN slice encountered", RuntimeWarning)
741:     if out is not None:
742:         out[...] = m.filled(np.nan)
743:         return out
744:     return m.filled(np.nan)
745: 
746: def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
747:     '''
748:     Compute the median along the specified axis, while ignoring NaNs.
749: 
750:     Returns the median of the array elements.
751: 
752:     .. versionadded:: 1.9.0
753: 
754:     Parameters
755:     ----------
756:     a : array_like
757:         Input array or object that can be converted to an array.
758:     axis : {int, sequence of int, None}, optional
759:         Axis or axes along which the medians are computed. The default
760:         is to compute the median along a flattened version of the array.
761:         A sequence of axes is supported since version 1.9.0.
762:     out : ndarray, optional
763:         Alternative output array in which to place the result. It must
764:         have the same shape and buffer length as the expected output,
765:         but the type (of the output) will be cast if necessary.
766:     overwrite_input : bool, optional
767:        If True, then allow use of memory of input array `a` for
768:        calculations. The input array will be modified by the call to
769:        `median`. This will save memory when you do not need to preserve
770:        the contents of the input array. Treat the input as undefined,
771:        but it will probably be fully or partially sorted. Default is
772:        False. If `overwrite_input` is ``True`` and `a` is not already an
773:        `ndarray`, an error will be raised.
774:     keepdims : bool, optional
775:         If this is set to True, the axes which are reduced are left in
776:         the result as dimensions with size one. With this option, the
777:         result will broadcast correctly against the original `arr`.
778: 
779:     Returns
780:     -------
781:     median : ndarray
782:         A new array holding the result. If the input contains integers
783:         or floats smaller than ``float64``, then the output data-type is
784:         ``np.float64``.  Otherwise, the data-type of the output is the
785:         same as that of the input. If `out` is specified, that array is
786:         returned instead.
787: 
788:     See Also
789:     --------
790:     mean, median, percentile
791: 
792:     Notes
793:     -----
794:     Given a vector ``V`` of length ``N``, the median of ``V`` is the
795:     middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
796:     ``V_sorted[(N-1)/2]``, when ``N`` is odd and the average of the two
797:     middle values of ``V_sorted`` when ``N`` is even.
798: 
799:     Examples
800:     --------
801:     >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])
802:     >>> a[0, 1] = np.nan
803:     >>> a
804:     array([[ 10.,  nan,   4.],
805:        [  3.,   2.,   1.]])
806:     >>> np.median(a)
807:     nan
808:     >>> np.nanmedian(a)
809:     3.0
810:     >>> np.nanmedian(a, axis=0)
811:     array([ 6.5,  2.,  2.5])
812:     >>> np.median(a, axis=1)
813:     array([ 7.,  2.])
814:     >>> b = a.copy()
815:     >>> np.nanmedian(b, axis=1, overwrite_input=True)
816:     array([ 7.,  2.])
817:     >>> assert not np.all(a==b)
818:     >>> b = a.copy()
819:     >>> np.nanmedian(b, axis=None, overwrite_input=True)
820:     3.0
821:     >>> assert not np.all(a==b)
822: 
823:     '''
824:     a = np.asanyarray(a)
825:     # apply_along_axis in _nanmedian doesn't handle empty arrays well,
826:     # so deal them upfront
827:     if a.size == 0:
828:         return np.nanmean(a, axis, out=out, keepdims=keepdims)
829: 
830:     r, k = _ureduce(a, func=_nanmedian, axis=axis, out=out,
831:                     overwrite_input=overwrite_input)
832:     if keepdims:
833:         return r.reshape(k)
834:     else:
835:         return r
836: 
837: 
838: def nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
839:                   interpolation='linear', keepdims=False):
840:     '''
841:     Compute the qth percentile of the data along the specified axis,
842:     while ignoring nan values.
843: 
844:     Returns the qth percentile(s) of the array elements.
845: 
846:     .. versionadded:: 1.9.0
847: 
848:     Parameters
849:     ----------
850:     a : array_like
851:         Input array or object that can be converted to an array.
852:     q : float in range of [0,100] (or sequence of floats)
853:         Percentile to compute, which must be between 0 and 100
854:         inclusive.
855:     axis : {int, sequence of int, None}, optional
856:         Axis or axes along which the percentiles are computed. The
857:         default is to compute the percentile(s) along a flattened
858:         version of the array. A sequence of axes is supported since
859:         version 1.9.0.
860:     out : ndarray, optional
861:         Alternative output array in which to place the result. It must
862:         have the same shape and buffer length as the expected output,
863:         but the type (of the output) will be cast if necessary.
864:     overwrite_input : bool, optional
865:         If True, then allow use of memory of input array `a` for
866:         calculations. The input array will be modified by the call to
867:         `percentile`. This will save memory when you do not need to
868:         preserve the contents of the input array. In this case you
869:         should not make any assumptions about the contents of the input
870:         `a` after this function completes -- treat it as undefined.
871:         Default is False. If `a` is not already an array, this parameter
872:         will have no effect as `a` will be converted to an array
873:         internally regardless of the value of this parameter.
874:     interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
875:         This optional parameter specifies the interpolation method to
876:         use when the desired quantile lies between two data points
877:         ``i < j``:
878:             * linear: ``i + (j - i) * fraction``, where ``fraction`` is
879:               the fractional part of the index surrounded by ``i`` and
880:               ``j``.
881:             * lower: ``i``.
882:             * higher: ``j``.
883:             * nearest: ``i`` or ``j``, whichever is nearest.
884:             * midpoint: ``(i + j) / 2``.
885:     keepdims : bool, optional
886:         If this is set to True, the axes which are reduced are left in
887:         the result as dimensions with size one. With this option, the
888:         result will broadcast correctly against the original array `a`.
889: 
890:     Returns
891:     -------
892:     percentile : scalar or ndarray
893:         If `q` is a single percentile and `axis=None`, then the result
894:         is a scalar. If multiple percentiles are given, first axis of
895:         the result corresponds to the percentiles. The other axes are
896:         the axes that remain after the reduction of `a`. If the input 
897:         contains integers or floats smaller than ``float64``, the output
898:         data-type is ``float64``. Otherwise, the output data-type is the
899:         same as that of the input. If `out` is specified, that array is
900:         returned instead.
901: 
902:     See Also
903:     --------
904:     nanmean, nanmedian, percentile, median, mean
905: 
906:     Notes
907:     -----
908:     Given a vector ``V`` of length ``N``, the ``q``-th percentile of
909:     ``V`` is the value ``q/100`` of the way from the mimumum to the
910:     maximum in in a sorted copy of ``V``. The values and distances of
911:     the two nearest neighbors as well as the `interpolation` parameter
912:     will determine the percentile if the normalized ranking does not
913:     match the location of ``q`` exactly. This function is the same as
914:     the median if ``q=50``, the same as the minimum if ``q=0`` and the
915:     same as the maximum if ``q=100``.
916: 
917:     Examples
918:     --------
919:     >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])
920:     >>> a[0][1] = np.nan
921:     >>> a
922:     array([[ 10.,  nan,   4.],
923:        [  3.,   2.,   1.]])
924:     >>> np.percentile(a, 50)
925:     nan
926:     >>> np.nanpercentile(a, 50)
927:     3.5
928:     >>> np.nanpercentile(a, 50, axis=0)
929:     array([ 6.5,  2.,   2.5])
930:     >>> np.nanpercentile(a, 50, axis=1, keepdims=True)
931:     array([[ 7.],
932:            [ 2.]])
933:     >>> m = np.nanpercentile(a, 50, axis=0)
934:     >>> out = np.zeros_like(m)
935:     >>> np.nanpercentile(a, 50, axis=0, out=out)
936:     array([ 6.5,  2.,   2.5])
937:     >>> m
938:     array([ 6.5,  2. ,  2.5])
939: 
940:     >>> b = a.copy()
941:     >>> np.nanpercentile(b, 50, axis=1, overwrite_input=True)
942:     array([  7.,  2.])
943:     >>> assert not np.all(a==b)
944: 
945:     '''
946: 
947:     a = np.asanyarray(a)
948:     q = np.asanyarray(q)
949:     # apply_along_axis in _nanpercentile doesn't handle empty arrays well,
950:     # so deal them upfront
951:     if a.size == 0:
952:         return np.nanmean(a, axis, out=out, keepdims=keepdims)
953: 
954:     r, k = _ureduce(a, func=_nanpercentile, q=q, axis=axis, out=out,
955:                     overwrite_input=overwrite_input,
956:                     interpolation=interpolation)
957:     if keepdims:
958:         if q.ndim == 0:
959:             return r.reshape(k)
960:         else:
961:             return r.reshape([len(q)] + k)
962:     else:
963:         return r
964: 
965: 
966: def _nanpercentile(a, q, axis=None, out=None, overwrite_input=False,
967:                    interpolation='linear', keepdims=False):
968:     '''
969:     Private function that doesn't support extended axis or keepdims.
970:     These methods are extended to this function using _ureduce
971:     See nanpercentile for parameter usage
972: 
973:     '''
974:     if axis is None:
975:         part = a.ravel()
976:         result = _nanpercentile1d(part, q, overwrite_input, interpolation)
977:     else:
978:         result = np.apply_along_axis(_nanpercentile1d, axis, a, q,
979:                                      overwrite_input, interpolation)
980:         # apply_along_axis fills in collapsed axis with results.
981:         # Move that axis to the beginning to match percentile's
982:         # convention.
983:         if q.ndim != 0:
984:             result = np.rollaxis(result, axis)   
985: 
986:     if out is not None:
987:         out[...] = result
988:     return result
989: 
990: 
991: def _nanpercentile1d(arr1d, q, overwrite_input=False, interpolation='linear'):
992:     '''
993:     Private function for rank 1 arrays. Compute percentile ignoring
994:     NaNs.
995: 
996:     See nanpercentile for parameter usage
997:     '''
998:     c = np.isnan(arr1d)
999:     s = np.where(c)[0]
1000:     if s.size == arr1d.size:
1001:         warnings.warn("All-NaN slice encountered", RuntimeWarning)
1002:         if q.ndim == 0:
1003:             return np.nan
1004:         else:
1005:             return np.nan * np.ones((len(q),))
1006:     elif s.size == 0:
1007:         return np.percentile(arr1d, q, overwrite_input=overwrite_input,
1008:                              interpolation=interpolation)
1009:     else:
1010:         if overwrite_input:
1011:             x = arr1d
1012:         else:
1013:             x = arr1d.copy()
1014:         # select non-nans at end of array
1015:         enonan = arr1d[-s.size:][~c[-s.size:]]
1016:         # fill nans in beginning of array with non-nans of end
1017:         x[s[:enonan.size]] = enonan
1018:         # slice nans away
1019:         return np.percentile(x[:-s.size], q, overwrite_input=True,
1020:                              interpolation=interpolation)
1021: 
1022: 
1023: def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
1024:     '''
1025:     Compute the variance along the specified axis, while ignoring NaNs.
1026: 
1027:     Returns the variance of the array elements, a measure of the spread of
1028:     a distribution.  The variance is computed for the flattened array by
1029:     default, otherwise over the specified axis.
1030: 
1031:     For all-NaN slices or slices with zero degrees of freedom, NaN is
1032:     returned and a `RuntimeWarning` is raised.
1033: 
1034:     .. versionadded:: 1.8.0
1035: 
1036:     Parameters
1037:     ----------
1038:     a : array_like
1039:         Array containing numbers whose variance is desired.  If `a` is not an
1040:         array, a conversion is attempted.
1041:     axis : int, optional
1042:         Axis along which the variance is computed.  The default is to compute
1043:         the variance of the flattened array.
1044:     dtype : data-type, optional
1045:         Type to use in computing the variance.  For arrays of integer type
1046:         the default is `float32`; for arrays of float types it is the same as
1047:         the array type.
1048:     out : ndarray, optional
1049:         Alternate output array in which to place the result.  It must have
1050:         the same shape as the expected output, but the type is cast if
1051:         necessary.
1052:     ddof : int, optional
1053:         "Delta Degrees of Freedom": the divisor used in the calculation is
1054:         ``N - ddof``, where ``N`` represents the number of non-NaN
1055:         elements. By default `ddof` is zero.
1056:     keepdims : bool, optional
1057:         If this is set to True, the axes which are reduced are left
1058:         in the result as dimensions with size one. With this option,
1059:         the result will broadcast correctly against the original `arr`.
1060: 
1061:     Returns
1062:     -------
1063:     variance : ndarray, see dtype parameter above
1064:         If `out` is None, return a new array containing the variance,
1065:         otherwise return a reference to the output array. If ddof is >= the
1066:         number of non-NaN elements in a slice or the slice contains only
1067:         NaNs, then the result for that slice is NaN.
1068: 
1069:     See Also
1070:     --------
1071:     std : Standard deviation
1072:     mean : Average
1073:     var : Variance while not ignoring NaNs
1074:     nanstd, nanmean
1075:     numpy.doc.ufuncs : Section "Output arguments"
1076: 
1077:     Notes
1078:     -----
1079:     The variance is the average of the squared deviations from the mean,
1080:     i.e.,  ``var = mean(abs(x - x.mean())**2)``.
1081: 
1082:     The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
1083:     If, however, `ddof` is specified, the divisor ``N - ddof`` is used
1084:     instead.  In standard statistical practice, ``ddof=1`` provides an
1085:     unbiased estimator of the variance of a hypothetical infinite
1086:     population.  ``ddof=0`` provides a maximum likelihood estimate of the
1087:     variance for normally distributed variables.
1088: 
1089:     Note that for complex numbers, the absolute value is taken before
1090:     squaring, so that the result is always real and nonnegative.
1091: 
1092:     For floating-point input, the variance is computed using the same
1093:     precision the input has.  Depending on the input data, this can cause
1094:     the results to be inaccurate, especially for `float32` (see example
1095:     below).  Specifying a higher-accuracy accumulator using the ``dtype``
1096:     keyword can alleviate this issue.
1097: 
1098:     Examples
1099:     --------
1100:     >>> a = np.array([[1, np.nan], [3, 4]])
1101:     >>> np.var(a)
1102:     1.5555555555555554
1103:     >>> np.nanvar(a, axis=0)
1104:     array([ 1.,  0.])
1105:     >>> np.nanvar(a, axis=1)
1106:     array([ 0.,  0.25])
1107: 
1108:     '''
1109:     arr, mask = _replace_nan(a, 0)
1110:     if mask is None:
1111:         return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
1112:                       keepdims=keepdims)
1113: 
1114:     if dtype is not None:
1115:         dtype = np.dtype(dtype)
1116:     if dtype is not None and not issubclass(dtype.type, np.inexact):
1117:         raise TypeError("If a is inexact, then dtype must be inexact")
1118:     if out is not None and not issubclass(out.dtype.type, np.inexact):
1119:         raise TypeError("If a is inexact, then out must be inexact")
1120: 
1121:     with warnings.catch_warnings():
1122:         warnings.simplefilter('ignore')
1123: 
1124:         # Compute mean
1125:         cnt = np.sum(~mask, axis=axis, dtype=np.intp, keepdims=True)
1126:         avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=True)
1127:         avg = _divide_by_count(avg, cnt)
1128: 
1129:         # Compute squared deviation from mean.
1130:         np.subtract(arr, avg, out=arr, casting='unsafe')
1131:         arr = _copyto(arr, 0, mask)
1132:         if issubclass(arr.dtype.type, np.complexfloating):
1133:             sqr = np.multiply(arr, arr.conj(), out=arr).real
1134:         else:
1135:             sqr = np.multiply(arr, arr, out=arr)
1136: 
1137:         # Compute variance.
1138:         var = np.sum(sqr, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
1139:         if var.ndim < cnt.ndim:
1140:             # Subclasses of ndarray may ignore keepdims, so check here.
1141:             cnt = cnt.squeeze(axis)
1142:         dof = cnt - ddof
1143:         var = _divide_by_count(var, dof)
1144: 
1145:     isbad = (dof <= 0)
1146:     if np.any(isbad):
1147:         warnings.warn("Degrees of freedom <= 0 for slice.", RuntimeWarning)
1148:         # NaN, inf, or negative numbers are all possible bad
1149:         # values, so explicitly replace them with NaN.
1150:         var = _copyto(var, np.nan, isbad)
1151:     return var
1152: 
1153: 
1154: def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
1155:     '''
1156:     Compute the standard deviation along the specified axis, while
1157:     ignoring NaNs.
1158: 
1159:     Returns the standard deviation, a measure of the spread of a
1160:     distribution, of the non-NaN array elements. The standard deviation is
1161:     computed for the flattened array by default, otherwise over the
1162:     specified axis.
1163: 
1164:     For all-NaN slices or slices with zero degrees of freedom, NaN is
1165:     returned and a `RuntimeWarning` is raised.
1166: 
1167:     .. versionadded:: 1.8.0
1168: 
1169:     Parameters
1170:     ----------
1171:     a : array_like
1172:         Calculate the standard deviation of the non-NaN values.
1173:     axis : int, optional
1174:         Axis along which the standard deviation is computed. The default is
1175:         to compute the standard deviation of the flattened array.
1176:     dtype : dtype, optional
1177:         Type to use in computing the standard deviation. For arrays of
1178:         integer type the default is float64, for arrays of float types it
1179:         is the same as the array type.
1180:     out : ndarray, optional
1181:         Alternative output array in which to place the result. It must have
1182:         the same shape as the expected output but the type (of the
1183:         calculated values) will be cast if necessary.
1184:     ddof : int, optional
1185:         Means Delta Degrees of Freedom.  The divisor used in calculations
1186:         is ``N - ddof``, where ``N`` represents the number of non-NaN
1187:         elements.  By default `ddof` is zero.
1188:     keepdims : bool, optional
1189:         If this is set to True, the axes which are reduced are left
1190:         in the result as dimensions with size one. With this option,
1191:         the result will broadcast correctly against the original `arr`.
1192: 
1193:     Returns
1194:     -------
1195:     standard_deviation : ndarray, see dtype parameter above.
1196:         If `out` is None, return a new array containing the standard
1197:         deviation, otherwise return a reference to the output array. If
1198:         ddof is >= the number of non-NaN elements in a slice or the slice
1199:         contains only NaNs, then the result for that slice is NaN.
1200: 
1201:     See Also
1202:     --------
1203:     var, mean, std
1204:     nanvar, nanmean
1205:     numpy.doc.ufuncs : Section "Output arguments"
1206: 
1207:     Notes
1208:     -----
1209:     The standard deviation is the square root of the average of the squared
1210:     deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.
1211: 
1212:     The average squared deviation is normally calculated as
1213:     ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is
1214:     specified, the divisor ``N - ddof`` is used instead. In standard
1215:     statistical practice, ``ddof=1`` provides an unbiased estimator of the
1216:     variance of the infinite population. ``ddof=0`` provides a maximum
1217:     likelihood estimate of the variance for normally distributed variables.
1218:     The standard deviation computed in this function is the square root of
1219:     the estimated variance, so even with ``ddof=1``, it will not be an
1220:     unbiased estimate of the standard deviation per se.
1221: 
1222:     Note that, for complex numbers, `std` takes the absolute value before
1223:     squaring, so that the result is always real and nonnegative.
1224: 
1225:     For floating-point input, the *std* is computed using the same
1226:     precision the input has. Depending on the input data, this can cause
1227:     the results to be inaccurate, especially for float32 (see example
1228:     below).  Specifying a higher-accuracy accumulator using the `dtype`
1229:     keyword can alleviate this issue.
1230: 
1231:     Examples
1232:     --------
1233:     >>> a = np.array([[1, np.nan], [3, 4]])
1234:     >>> np.nanstd(a)
1235:     1.247219128924647
1236:     >>> np.nanstd(a, axis=0)
1237:     array([ 1.,  0.])
1238:     >>> np.nanstd(a, axis=1)
1239:     array([ 0.,  0.5])
1240: 
1241:     '''
1242:     var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
1243:                  keepdims=keepdims)
1244:     if isinstance(var, np.ndarray):
1245:         std = np.sqrt(var, out=var)
1246:     else:
1247:         std = var.dtype.type(np.sqrt(var))
1248:     return std
1249: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_115627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\nFunctions that ignore NaN.\n\nFunctions\n---------\n\n- `nanmin` -- minimum non-NaN value\n- `nanmax` -- maximum non-NaN value\n- `nanargmin` -- index of minimum non-NaN value\n- `nanargmax` -- index of maximum non-NaN value\n- `nansum` -- sum of non-NaN values\n- `nanprod` -- product of non-NaN values\n- `nanmean` -- mean of non-NaN values\n- `nanvar` -- variance of non-NaN values\n- `nanstd` -- standard deviation of non-NaN values\n- `nanmedian` -- median of non-NaN values\n- `nanpercentile` -- qth percentile of non-NaN values\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import warnings' statement (line 22)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import numpy' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_115628 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy')

if (type(import_115628) is not StypyTypeError):

    if (import_115628 != 'pyd_module'):
        __import__(import_115628)
        sys_modules_115629 = sys.modules[import_115628]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', sys_modules_115629.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'numpy', import_115628)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.lib.function_base import _ureduce' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_115630 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.function_base')

if (type(import_115630) is not StypyTypeError):

    if (import_115630 != 'pyd_module'):
        __import__(import_115630)
        sys_modules_115631 = sys.modules[import_115630]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.function_base', sys_modules_115631.module_type_store, module_type_store, ['_ureduce'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_115631, sys_modules_115631.module_type_store, module_type_store)
    else:
        from numpy.lib.function_base import _ureduce as _ureduce

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.function_base', None, module_type_store, ['_ureduce'], [_ureduce])

else:
    # Assigning a type to the variable 'numpy.lib.function_base' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.lib.function_base', import_115630)

# Adding an alias
module_type_store.add_alias('_ureduce', '_ureduce')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 26):

# Assigning a List to a Name (line 26):
__all__ = ['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean', 'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod']
module_type_store.set_exportable_members(['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean', 'nanmedian', 'nanpercentile', 'nanvar', 'nanstd', 'nanprod'])

# Obtaining an instance of the builtin type 'list' (line 26)
list_115632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_115633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'nansum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115633)
# Adding element type (line 26)
str_115634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'str', 'nanmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115634)
# Adding element type (line 26)
str_115635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'str', 'nanmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115635)
# Adding element type (line 26)
str_115636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 34), 'str', 'nanargmax')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115636)
# Adding element type (line 26)
str_115637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 47), 'str', 'nanargmin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115637)
# Adding element type (line 26)
str_115638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 60), 'str', 'nanmean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115638)
# Adding element type (line 26)
str_115639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'nanmedian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115639)
# Adding element type (line 26)
str_115640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'str', 'nanpercentile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115640)
# Adding element type (line 26)
str_115641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'str', 'nanvar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115641)
# Adding element type (line 26)
str_115642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'str', 'nanstd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115642)
# Adding element type (line 26)
str_115643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 54), 'str', 'nanprod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 10), list_115632, str_115643)

# Assigning a type to the variable '__all__' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__all__', list_115632)

@norecursion
def _replace_nan(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_replace_nan'
    module_type_store = module_type_store.open_function_context('_replace_nan', 32, 0, False)
    
    # Passed parameters checking function
    _replace_nan.stypy_localization = localization
    _replace_nan.stypy_type_of_self = None
    _replace_nan.stypy_type_store = module_type_store
    _replace_nan.stypy_function_name = '_replace_nan'
    _replace_nan.stypy_param_names_list = ['a', 'val']
    _replace_nan.stypy_varargs_param_name = None
    _replace_nan.stypy_kwargs_param_name = None
    _replace_nan.stypy_call_defaults = defaults
    _replace_nan.stypy_call_varargs = varargs
    _replace_nan.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_replace_nan', ['a', 'val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_replace_nan', localization, ['a', 'val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_replace_nan(...)' code ##################

    str_115644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    If `a` is of inexact type, make a copy of `a`, replace NaNs with\n    the `val` value, and return the copy together with a boolean mask\n    marking the locations where NaNs were present. If `a` is not of\n    inexact type, do nothing and return `a` together with a mask of None.\n\n    Note that scalars will end up as array scalars, which is important\n    for using the result as the value of the out argument in some\n    operations.\n\n    Parameters\n    ----------\n    a : array-like\n        Input array.\n    val : float\n        NaN values are set to val before doing the operation.\n\n    Returns\n    -------\n    y : ndarray\n        If `a` is of inexact type, return a copy of `a` with the NaNs\n        replaced by the fill value, otherwise return `a`.\n    mask: {bool, None}\n        If `a` is of inexact type, return a boolean mask marking locations of\n        NaNs, otherwise return None.\n\n    ')
    
    # Assigning a UnaryOp to a Name (line 60):
    
    # Assigning a UnaryOp to a Name (line 60):
    
    
    # Call to isinstance(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'a' (line 60)
    a_115646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 28), 'a', False)
    # Getting the type of 'np' (line 60)
    np_115647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 60)
    ndarray_115648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 31), np_115647, 'ndarray')
    # Processing the call keyword arguments (line 60)
    kwargs_115649 = {}
    # Getting the type of 'isinstance' (line 60)
    isinstance_115645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 60)
    isinstance_call_result_115650 = invoke(stypy.reporting.localization.Localization(__file__, 60, 17), isinstance_115645, *[a_115646, ndarray_115648], **kwargs_115649)
    
    # Applying the 'not' unary operator (line 60)
    result_not__115651 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 13), 'not', isinstance_call_result_115650)
    
    # Assigning a type to the variable 'is_new' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'is_new', result_not__115651)
    
    # Getting the type of 'is_new' (line 61)
    is_new_115652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 'is_new')
    # Testing the type of an if condition (line 61)
    if_condition_115653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), is_new_115652)
    # Assigning a type to the variable 'if_condition_115653' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_115653', if_condition_115653)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to array(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'a' (line 62)
    a_115656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 21), 'a', False)
    # Processing the call keyword arguments (line 62)
    kwargs_115657 = {}
    # Getting the type of 'np' (line 62)
    np_115654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 62)
    array_115655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), np_115654, 'array')
    # Calling array(args, kwargs) (line 62)
    array_call_result_115658 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), array_115655, *[a_115656], **kwargs_115657)
    
    # Assigning a type to the variable 'a' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'a', array_call_result_115658)
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to issubclass(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'a' (line 63)
    a_115660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'a', False)
    # Obtaining the member 'dtype' of a type (line 63)
    dtype_115661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), a_115660, 'dtype')
    # Obtaining the member 'type' of a type (line 63)
    type_115662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), dtype_115661, 'type')
    # Getting the type of 'np' (line 63)
    np_115663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'np', False)
    # Obtaining the member 'inexact' of a type (line 63)
    inexact_115664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 36), np_115663, 'inexact')
    # Processing the call keyword arguments (line 63)
    kwargs_115665 = {}
    # Getting the type of 'issubclass' (line 63)
    issubclass_115659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 63)
    issubclass_call_result_115666 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), issubclass_115659, *[type_115662, inexact_115664], **kwargs_115665)
    
    # Applying the 'not' unary operator (line 63)
    result_not__115667 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 7), 'not', issubclass_call_result_115666)
    
    # Testing the type of an if condition (line 63)
    if_condition_115668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), result_not__115667)
    # Assigning a type to the variable 'if_condition_115668' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_115668', if_condition_115668)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 64)
    tuple_115669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 64)
    # Adding element type (line 64)
    # Getting the type of 'a' (line 64)
    a_115670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 15), tuple_115669, a_115670)
    # Adding element type (line 64)
    # Getting the type of 'None' (line 64)
    None_115671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 15), tuple_115669, None_115671)
    
    # Assigning a type to the variable 'stypy_return_type' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', tuple_115669)
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'is_new' (line 65)
    is_new_115672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'is_new')
    # Applying the 'not' unary operator (line 65)
    result_not__115673 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 7), 'not', is_new_115672)
    
    # Testing the type of an if condition (line 65)
    if_condition_115674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 4), result_not__115673)
    # Assigning a type to the variable 'if_condition_115674' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'if_condition_115674', if_condition_115674)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to array(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'a' (line 67)
    a_115677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'a', False)
    # Processing the call keyword arguments (line 67)
    # Getting the type of 'True' (line 67)
    True_115678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'True', False)
    keyword_115679 = True_115678
    kwargs_115680 = {'subok': keyword_115679}
    # Getting the type of 'np' (line 67)
    np_115675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 67)
    array_115676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), np_115675, 'array')
    # Calling array(args, kwargs) (line 67)
    array_call_result_115681 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), array_115676, *[a_115677], **kwargs_115680)
    
    # Assigning a type to the variable 'a' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'a', array_call_result_115681)
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to isnan(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'a' (line 69)
    a_115684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'a', False)
    # Processing the call keyword arguments (line 69)
    kwargs_115685 = {}
    # Getting the type of 'np' (line 69)
    np_115682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'np', False)
    # Obtaining the member 'isnan' of a type (line 69)
    isnan_115683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), np_115682, 'isnan')
    # Calling isnan(args, kwargs) (line 69)
    isnan_call_result_115686 = invoke(stypy.reporting.localization.Localization(__file__, 69, 11), isnan_115683, *[a_115684], **kwargs_115685)
    
    # Assigning a type to the variable 'mask' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'mask', isnan_call_result_115686)
    
    # Call to copyto(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'a' (line 70)
    a_115689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'a', False)
    # Getting the type of 'val' (line 70)
    val_115690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'val', False)
    # Processing the call keyword arguments (line 70)
    # Getting the type of 'mask' (line 70)
    mask_115691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'mask', False)
    keyword_115692 = mask_115691
    kwargs_115693 = {'where': keyword_115692}
    # Getting the type of 'np' (line 70)
    np_115687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'np', False)
    # Obtaining the member 'copyto' of a type (line 70)
    copyto_115688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), np_115687, 'copyto')
    # Calling copyto(args, kwargs) (line 70)
    copyto_call_result_115694 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), copyto_115688, *[a_115689, val_115690], **kwargs_115693)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 71)
    tuple_115695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'a' (line 71)
    a_115696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 11), tuple_115695, a_115696)
    # Adding element type (line 71)
    # Getting the type of 'mask' (line 71)
    mask_115697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 14), 'mask')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 11), tuple_115695, mask_115697)
    
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type', tuple_115695)
    
    # ################# End of '_replace_nan(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_replace_nan' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_115698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115698)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_replace_nan'
    return stypy_return_type_115698

# Assigning a type to the variable '_replace_nan' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_replace_nan', _replace_nan)

@norecursion
def _copyto(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_copyto'
    module_type_store = module_type_store.open_function_context('_copyto', 74, 0, False)
    
    # Passed parameters checking function
    _copyto.stypy_localization = localization
    _copyto.stypy_type_of_self = None
    _copyto.stypy_type_store = module_type_store
    _copyto.stypy_function_name = '_copyto'
    _copyto.stypy_param_names_list = ['a', 'val', 'mask']
    _copyto.stypy_varargs_param_name = None
    _copyto.stypy_kwargs_param_name = None
    _copyto.stypy_call_defaults = defaults
    _copyto.stypy_call_varargs = varargs
    _copyto.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_copyto', ['a', 'val', 'mask'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_copyto', localization, ['a', 'val', 'mask'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_copyto(...)' code ##################

    str_115699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    Replace values in `a` with NaN where `mask` is True.  This differs from\n    copyto in that it will deal with the case where `a` is a numpy scalar.\n\n    Parameters\n    ----------\n    a : ndarray or numpy scalar\n        Array or numpy scalar some of whose values are to be replaced\n        by val.\n    val : numpy scalar\n        Value used a replacement.\n    mask : ndarray, scalar\n        Boolean array. Where True the corresponding element of `a` is\n        replaced by `val`. Broadcasts.\n\n    Returns\n    -------\n    res : ndarray, scalar\n        Array with elements replaced or scalar `val`.\n\n    ')
    
    
    # Call to isinstance(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'a' (line 96)
    a_115701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'a', False)
    # Getting the type of 'np' (line 96)
    np_115702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 96)
    ndarray_115703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), np_115702, 'ndarray')
    # Processing the call keyword arguments (line 96)
    kwargs_115704 = {}
    # Getting the type of 'isinstance' (line 96)
    isinstance_115700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 96)
    isinstance_call_result_115705 = invoke(stypy.reporting.localization.Localization(__file__, 96, 7), isinstance_115700, *[a_115701, ndarray_115703], **kwargs_115704)
    
    # Testing the type of an if condition (line 96)
    if_condition_115706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), isinstance_call_result_115705)
    # Assigning a type to the variable 'if_condition_115706' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_115706', if_condition_115706)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to copyto(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'a' (line 97)
    a_115709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'a', False)
    # Getting the type of 'val' (line 97)
    val_115710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'val', False)
    # Processing the call keyword arguments (line 97)
    # Getting the type of 'mask' (line 97)
    mask_115711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'mask', False)
    keyword_115712 = mask_115711
    str_115713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 46), 'str', 'unsafe')
    keyword_115714 = str_115713
    kwargs_115715 = {'casting': keyword_115714, 'where': keyword_115712}
    # Getting the type of 'np' (line 97)
    np_115707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'np', False)
    # Obtaining the member 'copyto' of a type (line 97)
    copyto_115708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), np_115707, 'copyto')
    # Calling copyto(args, kwargs) (line 97)
    copyto_call_result_115716 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), copyto_115708, *[a_115709, val_115710], **kwargs_115715)
    
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to type(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'val' (line 99)
    val_115720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'val', False)
    # Processing the call keyword arguments (line 99)
    kwargs_115721 = {}
    # Getting the type of 'a' (line 99)
    a_115717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'a', False)
    # Obtaining the member 'dtype' of a type (line 99)
    dtype_115718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), a_115717, 'dtype')
    # Obtaining the member 'type' of a type (line 99)
    type_115719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), dtype_115718, 'type')
    # Calling type(args, kwargs) (line 99)
    type_call_result_115722 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), type_115719, *[val_115720], **kwargs_115721)
    
    # Assigning a type to the variable 'a' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'a', type_call_result_115722)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 100)
    a_115723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', a_115723)
    
    # ################# End of '_copyto(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_copyto' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_115724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115724)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_copyto'
    return stypy_return_type_115724

# Assigning a type to the variable '_copyto' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_copyto', _copyto)

@norecursion
def _divide_by_count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 103)
    None_115725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'None')
    defaults = [None_115725]
    # Create a new context for function '_divide_by_count'
    module_type_store = module_type_store.open_function_context('_divide_by_count', 103, 0, False)
    
    # Passed parameters checking function
    _divide_by_count.stypy_localization = localization
    _divide_by_count.stypy_type_of_self = None
    _divide_by_count.stypy_type_store = module_type_store
    _divide_by_count.stypy_function_name = '_divide_by_count'
    _divide_by_count.stypy_param_names_list = ['a', 'b', 'out']
    _divide_by_count.stypy_varargs_param_name = None
    _divide_by_count.stypy_kwargs_param_name = None
    _divide_by_count.stypy_call_defaults = defaults
    _divide_by_count.stypy_call_varargs = varargs
    _divide_by_count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_divide_by_count', ['a', 'b', 'out'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_divide_by_count', localization, ['a', 'b', 'out'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_divide_by_count(...)' code ##################

    str_115726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n    Compute a/b ignoring invalid results. If `a` is an array the division\n    is done in place. If `a` is a scalar, then its type is preserved in the\n    output. If out is None, then then a is used instead so that the\n    division is in place. Note that this is only called with `a` an inexact\n    type.\n\n    Parameters\n    ----------\n    a : {ndarray, numpy scalar}\n        Numerator. Expected to be of inexact type but not checked.\n    b : {ndarray, numpy scalar}\n        Denominator.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``; if provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.\n\n    Returns\n    -------\n    ret : {ndarray, numpy scalar}\n        The return value is a/b. If `a` was an ndarray the division is done\n        in place. If `a` is a numpy scalar, the division preserves its type.\n\n    ')
    
    # Call to errstate(...): (line 129)
    # Processing the call keyword arguments (line 129)
    str_115729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'str', 'ignore')
    keyword_115730 = str_115729
    kwargs_115731 = {'invalid': keyword_115730}
    # Getting the type of 'np' (line 129)
    np_115727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'np', False)
    # Obtaining the member 'errstate' of a type (line 129)
    errstate_115728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), np_115727, 'errstate')
    # Calling errstate(args, kwargs) (line 129)
    errstate_call_result_115732 = invoke(stypy.reporting.localization.Localization(__file__, 129, 9), errstate_115728, *[], **kwargs_115731)
    
    with_115733 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 129, 9), errstate_call_result_115732, 'with parameter', '__enter__', '__exit__')

    if with_115733:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 129)
        enter___115734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), errstate_call_result_115732, '__enter__')
        with_enter_115735 = invoke(stypy.reporting.localization.Localization(__file__, 129, 9), enter___115734)
        
        
        # Call to isinstance(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'a' (line 130)
        a_115737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'a', False)
        # Getting the type of 'np' (line 130)
        np_115738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 130)
        ndarray_115739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), np_115738, 'ndarray')
        # Processing the call keyword arguments (line 130)
        kwargs_115740 = {}
        # Getting the type of 'isinstance' (line 130)
        isinstance_115736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 130)
        isinstance_call_result_115741 = invoke(stypy.reporting.localization.Localization(__file__, 130, 11), isinstance_115736, *[a_115737, ndarray_115739], **kwargs_115740)
        
        # Testing the type of an if condition (line 130)
        if_condition_115742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), isinstance_call_result_115741)
        # Assigning a type to the variable 'if_condition_115742' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_115742', if_condition_115742)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 131)
        # Getting the type of 'out' (line 131)
        out_115743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'out')
        # Getting the type of 'None' (line 131)
        None_115744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'None')
        
        (may_be_115745, more_types_in_union_115746) = may_be_none(out_115743, None_115744)

        if may_be_115745:

            if more_types_in_union_115746:
                # Runtime conditional SSA (line 131)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to divide(...): (line 132)
            # Processing the call arguments (line 132)
            # Getting the type of 'a' (line 132)
            a_115749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'a', False)
            # Getting the type of 'b' (line 132)
            b_115750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'b', False)
            # Processing the call keyword arguments (line 132)
            # Getting the type of 'a' (line 132)
            a_115751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'a', False)
            keyword_115752 = a_115751
            str_115753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'str', 'unsafe')
            keyword_115754 = str_115753
            kwargs_115755 = {'casting': keyword_115754, 'out': keyword_115752}
            # Getting the type of 'np' (line 132)
            np_115747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'np', False)
            # Obtaining the member 'divide' of a type (line 132)
            divide_115748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), np_115747, 'divide')
            # Calling divide(args, kwargs) (line 132)
            divide_call_result_115756 = invoke(stypy.reporting.localization.Localization(__file__, 132, 23), divide_115748, *[a_115749, b_115750], **kwargs_115755)
            
            # Assigning a type to the variable 'stypy_return_type' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'stypy_return_type', divide_call_result_115756)

            if more_types_in_union_115746:
                # Runtime conditional SSA for else branch (line 131)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_115745) or more_types_in_union_115746):
            
            # Call to divide(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 'a' (line 134)
            a_115759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 33), 'a', False)
            # Getting the type of 'b' (line 134)
            b_115760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'b', False)
            # Processing the call keyword arguments (line 134)
            # Getting the type of 'out' (line 134)
            out_115761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'out', False)
            keyword_115762 = out_115761
            str_115763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 56), 'str', 'unsafe')
            keyword_115764 = str_115763
            kwargs_115765 = {'casting': keyword_115764, 'out': keyword_115762}
            # Getting the type of 'np' (line 134)
            np_115757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'np', False)
            # Obtaining the member 'divide' of a type (line 134)
            divide_115758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), np_115757, 'divide')
            # Calling divide(args, kwargs) (line 134)
            divide_call_result_115766 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), divide_115758, *[a_115759, b_115760], **kwargs_115765)
            
            # Assigning a type to the variable 'stypy_return_type' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'stypy_return_type', divide_call_result_115766)

            if (may_be_115745 and more_types_in_union_115746):
                # SSA join for if statement (line 131)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        # Getting the type of 'out' (line 136)
        out_115767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'out')
        # Getting the type of 'None' (line 136)
        None_115768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 22), 'None')
        
        (may_be_115769, more_types_in_union_115770) = may_be_none(out_115767, None_115768)

        if may_be_115769:

            if more_types_in_union_115770:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to type(...): (line 137)
            # Processing the call arguments (line 137)
            # Getting the type of 'a' (line 137)
            a_115774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'a', False)
            # Getting the type of 'b' (line 137)
            b_115775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'b', False)
            # Applying the binary operator 'div' (line 137)
            result_div_115776 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 36), 'div', a_115774, b_115775)
            
            # Processing the call keyword arguments (line 137)
            kwargs_115777 = {}
            # Getting the type of 'a' (line 137)
            a_115771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'a', False)
            # Obtaining the member 'dtype' of a type (line 137)
            dtype_115772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 23), a_115771, 'dtype')
            # Obtaining the member 'type' of a type (line 137)
            type_115773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 23), dtype_115772, 'type')
            # Calling type(args, kwargs) (line 137)
            type_call_result_115778 = invoke(stypy.reporting.localization.Localization(__file__, 137, 23), type_115773, *[result_div_115776], **kwargs_115777)
            
            # Assigning a type to the variable 'stypy_return_type' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'stypy_return_type', type_call_result_115778)

            if more_types_in_union_115770:
                # Runtime conditional SSA for else branch (line 136)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_115769) or more_types_in_union_115770):
            
            # Call to divide(...): (line 141)
            # Processing the call arguments (line 141)
            # Getting the type of 'a' (line 141)
            a_115781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'a', False)
            # Getting the type of 'b' (line 141)
            b_115782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'b', False)
            # Processing the call keyword arguments (line 141)
            # Getting the type of 'out' (line 141)
            out_115783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'out', False)
            keyword_115784 = out_115783
            str_115785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 56), 'str', 'unsafe')
            keyword_115786 = str_115785
            kwargs_115787 = {'casting': keyword_115786, 'out': keyword_115784}
            # Getting the type of 'np' (line 141)
            np_115779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'np', False)
            # Obtaining the member 'divide' of a type (line 141)
            divide_115780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 23), np_115779, 'divide')
            # Calling divide(args, kwargs) (line 141)
            divide_call_result_115788 = invoke(stypy.reporting.localization.Localization(__file__, 141, 23), divide_115780, *[a_115781, b_115782], **kwargs_115787)
            
            # Assigning a type to the variable 'stypy_return_type' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'stypy_return_type', divide_call_result_115788)

            if (may_be_115769 and more_types_in_union_115770):
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 129)
        exit___115789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), errstate_call_result_115732, '__exit__')
        with_exit_115790 = invoke(stypy.reporting.localization.Localization(__file__, 129, 9), exit___115789, None, None, None)

    
    # ################# End of '_divide_by_count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_divide_by_count' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_115791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115791)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_divide_by_count'
    return stypy_return_type_115791

# Assigning a type to the variable '_divide_by_count' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), '_divide_by_count', _divide_by_count)

@norecursion
def nanmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 144)
    None_115792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'None')
    # Getting the type of 'None' (line 144)
    None_115793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'None')
    # Getting the type of 'False' (line 144)
    False_115794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'False')
    defaults = [None_115792, None_115793, False_115794]
    # Create a new context for function 'nanmin'
    module_type_store = module_type_store.open_function_context('nanmin', 144, 0, False)
    
    # Passed parameters checking function
    nanmin.stypy_localization = localization
    nanmin.stypy_type_of_self = None
    nanmin.stypy_type_store = module_type_store
    nanmin.stypy_function_name = 'nanmin'
    nanmin.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    nanmin.stypy_varargs_param_name = None
    nanmin.stypy_kwargs_param_name = None
    nanmin.stypy_call_defaults = defaults
    nanmin.stypy_call_varargs = varargs
    nanmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanmin', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanmin', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanmin(...)' code ##################

    str_115795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, (-1)), 'str', '\n    Return minimum of an array or minimum along an axis, ignoring any NaNs.\n    When all-NaN slices are encountered a ``RuntimeWarning`` is raised and\n    Nan is returned for that slice.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose minimum is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the minimum is computed. The default is to compute\n        the minimum of the flattened array.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``; if provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.  See\n        `doc.ufuncs` for details.\n\n        .. versionadded:: 1.8.0\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the original `a`.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    nanmin : ndarray\n        An array with the same shape as `a`, with the specified axis\n        removed.  If `a` is a 0-d array, or if axis is None, an ndarray\n        scalar is returned.  The same dtype as `a` is returned.\n\n    See Also\n    --------\n    nanmax :\n        The maximum value of an array along a given axis, ignoring any NaNs.\n    amin :\n        The minimum value of an array along a given axis, propagating any NaNs.\n    fmin :\n        Element-wise minimum of two arrays, ignoring any NaNs.\n    minimum :\n        Element-wise minimum of two arrays, propagating any NaNs.\n    isnan :\n        Shows which elements are Not a Number (NaN).\n    isfinite:\n        Shows which elements are neither NaN nor infinity.\n\n    amax, fmax, maximum\n\n    Notes\n    -----\n    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n    Positive infinity is treated as a very large number and negative\n    infinity is treated as a very small (i.e. negative) number.\n\n    If the input has a integer type the function is equivalent to np.min.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, np.nan]])\n    >>> np.nanmin(a)\n    1.0\n    >>> np.nanmin(a, axis=0)\n    array([ 1.,  2.])\n    >>> np.nanmin(a, axis=1)\n    array([ 1.,  3.])\n\n    When positive infinity and negative infinity are present:\n\n    >>> np.nanmin([1, 2, np.nan, np.inf])\n    1.0\n    >>> np.nanmin([1, 2, np.nan, np.NINF])\n    -inf\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'a' (line 223)
    a_115797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 22), 'a', False)
    # Getting the type of 'np' (line 223)
    np_115798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 223)
    ndarray_115799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 25), np_115798, 'ndarray')
    # Processing the call keyword arguments (line 223)
    kwargs_115800 = {}
    # Getting the type of 'isinstance' (line 223)
    isinstance_115796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 223)
    isinstance_call_result_115801 = invoke(stypy.reporting.localization.Localization(__file__, 223, 11), isinstance_115796, *[a_115797, ndarray_115799], **kwargs_115800)
    
    # Applying the 'not' unary operator (line 223)
    result_not__115802 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), 'not', isinstance_call_result_115801)
    
    
    
    # Call to type(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'a' (line 223)
    a_115804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 45), 'a', False)
    # Processing the call keyword arguments (line 223)
    kwargs_115805 = {}
    # Getting the type of 'type' (line 223)
    type_115803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'type', False)
    # Calling type(args, kwargs) (line 223)
    type_call_result_115806 = invoke(stypy.reporting.localization.Localization(__file__, 223, 40), type_115803, *[a_115804], **kwargs_115805)
    
    # Getting the type of 'np' (line 223)
    np_115807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 51), 'np')
    # Obtaining the member 'ndarray' of a type (line 223)
    ndarray_115808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 51), np_115807, 'ndarray')
    # Applying the binary operator 'is' (line 223)
    result_is__115809 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 40), 'is', type_call_result_115806, ndarray_115808)
    
    # Applying the binary operator 'or' (line 223)
    result_or_keyword_115810 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 7), 'or', result_not__115802, result_is__115809)
    
    # Testing the type of an if condition (line 223)
    if_condition_115811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 4), result_or_keyword_115810)
    # Assigning a type to the variable 'if_condition_115811' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'if_condition_115811', if_condition_115811)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to reduce(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'a' (line 225)
    a_115815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'a', False)
    # Processing the call keyword arguments (line 225)
    # Getting the type of 'axis' (line 225)
    axis_115816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 37), 'axis', False)
    keyword_115817 = axis_115816
    # Getting the type of 'out' (line 225)
    out_115818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 47), 'out', False)
    keyword_115819 = out_115818
    # Getting the type of 'keepdims' (line 225)
    keepdims_115820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 61), 'keepdims', False)
    keyword_115821 = keepdims_115820
    kwargs_115822 = {'out': keyword_115819, 'keepdims': keyword_115821, 'axis': keyword_115817}
    # Getting the type of 'np' (line 225)
    np_115812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 14), 'np', False)
    # Obtaining the member 'fmin' of a type (line 225)
    fmin_115813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 14), np_115812, 'fmin')
    # Obtaining the member 'reduce' of a type (line 225)
    reduce_115814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 14), fmin_115813, 'reduce')
    # Calling reduce(args, kwargs) (line 225)
    reduce_call_result_115823 = invoke(stypy.reporting.localization.Localization(__file__, 225, 14), reduce_115814, *[a_115815], **kwargs_115822)
    
    # Assigning a type to the variable 'res' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'res', reduce_call_result_115823)
    
    
    # Call to any(...): (line 226)
    # Processing the call keyword arguments (line 226)
    kwargs_115830 = {}
    
    # Call to isnan(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'res' (line 226)
    res_115826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'res', False)
    # Processing the call keyword arguments (line 226)
    kwargs_115827 = {}
    # Getting the type of 'np' (line 226)
    np_115824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'np', False)
    # Obtaining the member 'isnan' of a type (line 226)
    isnan_115825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), np_115824, 'isnan')
    # Calling isnan(args, kwargs) (line 226)
    isnan_call_result_115828 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), isnan_115825, *[res_115826], **kwargs_115827)
    
    # Obtaining the member 'any' of a type (line 226)
    any_115829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), isnan_call_result_115828, 'any')
    # Calling any(args, kwargs) (line 226)
    any_call_result_115831 = invoke(stypy.reporting.localization.Localization(__file__, 226, 11), any_115829, *[], **kwargs_115830)
    
    # Testing the type of an if condition (line 226)
    if_condition_115832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), any_call_result_115831)
    # Assigning a type to the variable 'if_condition_115832' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_115832', if_condition_115832)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 227)
    # Processing the call arguments (line 227)
    str_115835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 26), 'str', 'All-NaN axis encountered')
    # Getting the type of 'RuntimeWarning' (line 227)
    RuntimeWarning_115836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 54), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 227)
    kwargs_115837 = {}
    # Getting the type of 'warnings' (line 227)
    warnings_115833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 227)
    warn_115834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), warnings_115833, 'warn')
    # Calling warn(args, kwargs) (line 227)
    warn_call_result_115838 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), warn_115834, *[str_115835, RuntimeWarning_115836], **kwargs_115837)
    
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 230):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'a' (line 230)
    a_115840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 31), 'a', False)
    
    # Getting the type of 'np' (line 230)
    np_115841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'np', False)
    # Obtaining the member 'inf' of a type (line 230)
    inf_115842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 35), np_115841, 'inf')
    # Applying the 'uadd' unary operator (line 230)
    result___pos___115843 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 34), 'uadd', inf_115842)
    
    # Processing the call keyword arguments (line 230)
    kwargs_115844 = {}
    # Getting the type of '_replace_nan' (line 230)
    _replace_nan_115839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 18), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 230)
    _replace_nan_call_result_115845 = invoke(stypy.reporting.localization.Localization(__file__, 230, 18), _replace_nan_115839, *[a_115840, result___pos___115843], **kwargs_115844)
    
    # Assigning a type to the variable 'call_assignment_115597' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115597', _replace_nan_call_result_115845)
    
    # Assigning a Call to a Name (line 230):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_115848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
    # Processing the call keyword arguments
    kwargs_115849 = {}
    # Getting the type of 'call_assignment_115597' (line 230)
    call_assignment_115597_115846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115597', False)
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___115847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), call_assignment_115597_115846, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_115850 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___115847, *[int_115848], **kwargs_115849)
    
    # Assigning a type to the variable 'call_assignment_115598' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115598', getitem___call_result_115850)
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'call_assignment_115598' (line 230)
    call_assignment_115598_115851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115598')
    # Assigning a type to the variable 'a' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'a', call_assignment_115598_115851)
    
    # Assigning a Call to a Name (line 230):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_115854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
    # Processing the call keyword arguments
    kwargs_115855 = {}
    # Getting the type of 'call_assignment_115597' (line 230)
    call_assignment_115597_115852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115597', False)
    # Obtaining the member '__getitem__' of a type (line 230)
    getitem___115853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), call_assignment_115597_115852, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_115856 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___115853, *[int_115854], **kwargs_115855)
    
    # Assigning a type to the variable 'call_assignment_115599' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115599', getitem___call_result_115856)
    
    # Assigning a Name to a Name (line 230):
    # Getting the type of 'call_assignment_115599' (line 230)
    call_assignment_115599_115857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'call_assignment_115599')
    # Assigning a type to the variable 'mask' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'mask', call_assignment_115599_115857)
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to amin(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'a' (line 231)
    a_115860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'a', False)
    # Processing the call keyword arguments (line 231)
    # Getting the type of 'axis' (line 231)
    axis_115861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'axis', False)
    keyword_115862 = axis_115861
    # Getting the type of 'out' (line 231)
    out_115863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'out', False)
    keyword_115864 = out_115863
    # Getting the type of 'keepdims' (line 231)
    keepdims_115865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 54), 'keepdims', False)
    keyword_115866 = keepdims_115865
    kwargs_115867 = {'out': keyword_115864, 'keepdims': keyword_115866, 'axis': keyword_115862}
    # Getting the type of 'np' (line 231)
    np_115858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 14), 'np', False)
    # Obtaining the member 'amin' of a type (line 231)
    amin_115859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 14), np_115858, 'amin')
    # Calling amin(args, kwargs) (line 231)
    amin_call_result_115868 = invoke(stypy.reporting.localization.Localization(__file__, 231, 14), amin_115859, *[a_115860], **kwargs_115867)
    
    # Assigning a type to the variable 'res' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'res', amin_call_result_115868)
    
    # Type idiom detected: calculating its left and rigth part (line 232)
    # Getting the type of 'mask' (line 232)
    mask_115869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 11), 'mask')
    # Getting the type of 'None' (line 232)
    None_115870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'None')
    
    (may_be_115871, more_types_in_union_115872) = may_be_none(mask_115869, None_115870)

    if may_be_115871:

        if more_types_in_union_115872:
            # Runtime conditional SSA (line 232)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'res' (line 233)
        res_115873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'stypy_return_type', res_115873)

        if more_types_in_union_115872:
            # SSA join for if statement (line 232)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to all(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'mask' (line 236)
    mask_115876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 22), 'mask', False)
    # Processing the call keyword arguments (line 236)
    # Getting the type of 'axis' (line 236)
    axis_115877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'axis', False)
    keyword_115878 = axis_115877
    # Getting the type of 'keepdims' (line 236)
    keepdims_115879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 48), 'keepdims', False)
    keyword_115880 = keepdims_115879
    kwargs_115881 = {'keepdims': keyword_115880, 'axis': keyword_115878}
    # Getting the type of 'np' (line 236)
    np_115874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 236)
    all_115875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), np_115874, 'all')
    # Calling all(args, kwargs) (line 236)
    all_call_result_115882 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), all_115875, *[mask_115876], **kwargs_115881)
    
    # Assigning a type to the variable 'mask' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'mask', all_call_result_115882)
    
    
    # Call to any(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'mask' (line 237)
    mask_115885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'mask', False)
    # Processing the call keyword arguments (line 237)
    kwargs_115886 = {}
    # Getting the type of 'np' (line 237)
    np_115883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 237)
    any_115884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 11), np_115883, 'any')
    # Calling any(args, kwargs) (line 237)
    any_call_result_115887 = invoke(stypy.reporting.localization.Localization(__file__, 237, 11), any_115884, *[mask_115885], **kwargs_115886)
    
    # Testing the type of an if condition (line 237)
    if_condition_115888 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), any_call_result_115887)
    # Assigning a type to the variable 'if_condition_115888' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_115888', if_condition_115888)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to _copyto(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'res' (line 238)
    res_115890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'res', False)
    # Getting the type of 'np' (line 238)
    np_115891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 31), 'np', False)
    # Obtaining the member 'nan' of a type (line 238)
    nan_115892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 31), np_115891, 'nan')
    # Getting the type of 'mask' (line 238)
    mask_115893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 39), 'mask', False)
    # Processing the call keyword arguments (line 238)
    kwargs_115894 = {}
    # Getting the type of '_copyto' (line 238)
    _copyto_115889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 18), '_copyto', False)
    # Calling _copyto(args, kwargs) (line 238)
    _copyto_call_result_115895 = invoke(stypy.reporting.localization.Localization(__file__, 238, 18), _copyto_115889, *[res_115890, nan_115892, mask_115893], **kwargs_115894)
    
    # Assigning a type to the variable 'res' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'res', _copyto_call_result_115895)
    
    # Call to warn(...): (line 239)
    # Processing the call arguments (line 239)
    str_115898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'str', 'All-NaN axis encountered')
    # Getting the type of 'RuntimeWarning' (line 239)
    RuntimeWarning_115899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 54), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 239)
    kwargs_115900 = {}
    # Getting the type of 'warnings' (line 239)
    warnings_115896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 239)
    warn_115897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), warnings_115896, 'warn')
    # Calling warn(args, kwargs) (line 239)
    warn_call_result_115901 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), warn_115897, *[str_115898, RuntimeWarning_115899], **kwargs_115900)
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 240)
    res_115902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type', res_115902)
    
    # ################# End of 'nanmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanmin' in the type store
    # Getting the type of 'stypy_return_type' (line 144)
    stypy_return_type_115903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115903)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanmin'
    return stypy_return_type_115903

# Assigning a type to the variable 'nanmin' (line 144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 0), 'nanmin', nanmin)

@norecursion
def nanmax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 243)
    None_115904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'None')
    # Getting the type of 'None' (line 243)
    None_115905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'None')
    # Getting the type of 'False' (line 243)
    False_115906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 44), 'False')
    defaults = [None_115904, None_115905, False_115906]
    # Create a new context for function 'nanmax'
    module_type_store = module_type_store.open_function_context('nanmax', 243, 0, False)
    
    # Passed parameters checking function
    nanmax.stypy_localization = localization
    nanmax.stypy_type_of_self = None
    nanmax.stypy_type_store = module_type_store
    nanmax.stypy_function_name = 'nanmax'
    nanmax.stypy_param_names_list = ['a', 'axis', 'out', 'keepdims']
    nanmax.stypy_varargs_param_name = None
    nanmax.stypy_kwargs_param_name = None
    nanmax.stypy_call_defaults = defaults
    nanmax.stypy_call_varargs = varargs
    nanmax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanmax', ['a', 'axis', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanmax', localization, ['a', 'axis', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanmax(...)' code ##################

    str_115907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, (-1)), 'str', '\n    Return the maximum of an array or maximum along an axis, ignoring any\n    NaNs.  When all-NaN slices are encountered a ``RuntimeWarning`` is\n    raised and NaN is returned for that slice.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose maximum is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the maximum is computed. The default is to compute\n        the maximum of the flattened array.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``; if provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.  See\n        `doc.ufuncs` for details.\n\n        .. versionadded:: 1.8.0\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the original `a`.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    nanmax : ndarray\n        An array with the same shape as `a`, with the specified axis removed.\n        If `a` is a 0-d array, or if axis is None, an ndarray scalar is\n        returned.  The same dtype as `a` is returned.\n\n    See Also\n    --------\n    nanmin :\n        The minimum value of an array along a given axis, ignoring any NaNs.\n    amax :\n        The maximum value of an array along a given axis, propagating any NaNs.\n    fmax :\n        Element-wise maximum of two arrays, ignoring any NaNs.\n    maximum :\n        Element-wise maximum of two arrays, propagating any NaNs.\n    isnan :\n        Shows which elements are Not a Number (NaN).\n    isfinite:\n        Shows which elements are neither NaN nor infinity.\n\n    amin, fmin, minimum\n\n    Notes\n    -----\n    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n    Positive infinity is treated as a very large number and negative\n    infinity is treated as a very small (i.e. negative) number.\n\n    If the input has a integer type the function is equivalent to np.max.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2], [3, np.nan]])\n    >>> np.nanmax(a)\n    3.0\n    >>> np.nanmax(a, axis=0)\n    array([ 3.,  2.])\n    >>> np.nanmax(a, axis=1)\n    array([ 2.,  3.])\n\n    When positive infinity and negative infinity are present:\n\n    >>> np.nanmax([1, 2, np.nan, np.NINF])\n    2.0\n    >>> np.nanmax([1, 2, np.nan, np.inf])\n    inf\n\n    ')
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'a' (line 322)
    a_115909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 22), 'a', False)
    # Getting the type of 'np' (line 322)
    np_115910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 322)
    ndarray_115911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 25), np_115910, 'ndarray')
    # Processing the call keyword arguments (line 322)
    kwargs_115912 = {}
    # Getting the type of 'isinstance' (line 322)
    isinstance_115908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 322)
    isinstance_call_result_115913 = invoke(stypy.reporting.localization.Localization(__file__, 322, 11), isinstance_115908, *[a_115909, ndarray_115911], **kwargs_115912)
    
    # Applying the 'not' unary operator (line 322)
    result_not__115914 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), 'not', isinstance_call_result_115913)
    
    
    
    # Call to type(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'a' (line 322)
    a_115916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 45), 'a', False)
    # Processing the call keyword arguments (line 322)
    kwargs_115917 = {}
    # Getting the type of 'type' (line 322)
    type_115915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 40), 'type', False)
    # Calling type(args, kwargs) (line 322)
    type_call_result_115918 = invoke(stypy.reporting.localization.Localization(__file__, 322, 40), type_115915, *[a_115916], **kwargs_115917)
    
    # Getting the type of 'np' (line 322)
    np_115919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 51), 'np')
    # Obtaining the member 'ndarray' of a type (line 322)
    ndarray_115920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 51), np_115919, 'ndarray')
    # Applying the binary operator 'is' (line 322)
    result_is__115921 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 40), 'is', type_call_result_115918, ndarray_115920)
    
    # Applying the binary operator 'or' (line 322)
    result_or_keyword_115922 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), 'or', result_not__115914, result_is__115921)
    
    # Testing the type of an if condition (line 322)
    if_condition_115923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_or_keyword_115922)
    # Assigning a type to the variable 'if_condition_115923' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_115923', if_condition_115923)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to reduce(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'a' (line 324)
    a_115927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 29), 'a', False)
    # Processing the call keyword arguments (line 324)
    # Getting the type of 'axis' (line 324)
    axis_115928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 37), 'axis', False)
    keyword_115929 = axis_115928
    # Getting the type of 'out' (line 324)
    out_115930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 47), 'out', False)
    keyword_115931 = out_115930
    # Getting the type of 'keepdims' (line 324)
    keepdims_115932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 61), 'keepdims', False)
    keyword_115933 = keepdims_115932
    kwargs_115934 = {'out': keyword_115931, 'keepdims': keyword_115933, 'axis': keyword_115929}
    # Getting the type of 'np' (line 324)
    np_115924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 14), 'np', False)
    # Obtaining the member 'fmax' of a type (line 324)
    fmax_115925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 14), np_115924, 'fmax')
    # Obtaining the member 'reduce' of a type (line 324)
    reduce_115926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 14), fmax_115925, 'reduce')
    # Calling reduce(args, kwargs) (line 324)
    reduce_call_result_115935 = invoke(stypy.reporting.localization.Localization(__file__, 324, 14), reduce_115926, *[a_115927], **kwargs_115934)
    
    # Assigning a type to the variable 'res' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'res', reduce_call_result_115935)
    
    
    # Call to any(...): (line 325)
    # Processing the call keyword arguments (line 325)
    kwargs_115942 = {}
    
    # Call to isnan(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'res' (line 325)
    res_115938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'res', False)
    # Processing the call keyword arguments (line 325)
    kwargs_115939 = {}
    # Getting the type of 'np' (line 325)
    np_115936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 11), 'np', False)
    # Obtaining the member 'isnan' of a type (line 325)
    isnan_115937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), np_115936, 'isnan')
    # Calling isnan(args, kwargs) (line 325)
    isnan_call_result_115940 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), isnan_115937, *[res_115938], **kwargs_115939)
    
    # Obtaining the member 'any' of a type (line 325)
    any_115941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 11), isnan_call_result_115940, 'any')
    # Calling any(args, kwargs) (line 325)
    any_call_result_115943 = invoke(stypy.reporting.localization.Localization(__file__, 325, 11), any_115941, *[], **kwargs_115942)
    
    # Testing the type of an if condition (line 325)
    if_condition_115944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 8), any_call_result_115943)
    # Assigning a type to the variable 'if_condition_115944' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'if_condition_115944', if_condition_115944)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 326)
    # Processing the call arguments (line 326)
    str_115947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 26), 'str', 'All-NaN slice encountered')
    # Getting the type of 'RuntimeWarning' (line 326)
    RuntimeWarning_115948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 55), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 326)
    kwargs_115949 = {}
    # Getting the type of 'warnings' (line 326)
    warnings_115945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 326)
    warn_115946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 12), warnings_115945, 'warn')
    # Calling warn(args, kwargs) (line 326)
    warn_call_result_115950 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), warn_115946, *[str_115947, RuntimeWarning_115948], **kwargs_115949)
    
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 322)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 329):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'a' (line 329)
    a_115952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'a', False)
    
    # Getting the type of 'np' (line 329)
    np_115953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 35), 'np', False)
    # Obtaining the member 'inf' of a type (line 329)
    inf_115954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 35), np_115953, 'inf')
    # Applying the 'usub' unary operator (line 329)
    result___neg___115955 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 34), 'usub', inf_115954)
    
    # Processing the call keyword arguments (line 329)
    kwargs_115956 = {}
    # Getting the type of '_replace_nan' (line 329)
    _replace_nan_115951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 329)
    _replace_nan_call_result_115957 = invoke(stypy.reporting.localization.Localization(__file__, 329, 18), _replace_nan_115951, *[a_115952, result___neg___115955], **kwargs_115956)
    
    # Assigning a type to the variable 'call_assignment_115600' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115600', _replace_nan_call_result_115957)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_115960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_115961 = {}
    # Getting the type of 'call_assignment_115600' (line 329)
    call_assignment_115600_115958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115600', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___115959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_115600_115958, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_115962 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___115959, *[int_115960], **kwargs_115961)
    
    # Assigning a type to the variable 'call_assignment_115601' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115601', getitem___call_result_115962)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_115601' (line 329)
    call_assignment_115601_115963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115601')
    # Assigning a type to the variable 'a' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'a', call_assignment_115601_115963)
    
    # Assigning a Call to a Name (line 329):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_115966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 8), 'int')
    # Processing the call keyword arguments
    kwargs_115967 = {}
    # Getting the type of 'call_assignment_115600' (line 329)
    call_assignment_115600_115964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115600', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___115965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), call_assignment_115600_115964, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_115968 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___115965, *[int_115966], **kwargs_115967)
    
    # Assigning a type to the variable 'call_assignment_115602' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115602', getitem___call_result_115968)
    
    # Assigning a Name to a Name (line 329):
    # Getting the type of 'call_assignment_115602' (line 329)
    call_assignment_115602_115969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'call_assignment_115602')
    # Assigning a type to the variable 'mask' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'mask', call_assignment_115602_115969)
    
    # Assigning a Call to a Name (line 330):
    
    # Assigning a Call to a Name (line 330):
    
    # Call to amax(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'a' (line 330)
    a_115972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'a', False)
    # Processing the call keyword arguments (line 330)
    # Getting the type of 'axis' (line 330)
    axis_115973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 30), 'axis', False)
    keyword_115974 = axis_115973
    # Getting the type of 'out' (line 330)
    out_115975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 40), 'out', False)
    keyword_115976 = out_115975
    # Getting the type of 'keepdims' (line 330)
    keepdims_115977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 54), 'keepdims', False)
    keyword_115978 = keepdims_115977
    kwargs_115979 = {'out': keyword_115976, 'keepdims': keyword_115978, 'axis': keyword_115974}
    # Getting the type of 'np' (line 330)
    np_115970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 14), 'np', False)
    # Obtaining the member 'amax' of a type (line 330)
    amax_115971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 14), np_115970, 'amax')
    # Calling amax(args, kwargs) (line 330)
    amax_call_result_115980 = invoke(stypy.reporting.localization.Localization(__file__, 330, 14), amax_115971, *[a_115972], **kwargs_115979)
    
    # Assigning a type to the variable 'res' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'res', amax_call_result_115980)
    
    # Type idiom detected: calculating its left and rigth part (line 331)
    # Getting the type of 'mask' (line 331)
    mask_115981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'mask')
    # Getting the type of 'None' (line 331)
    None_115982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'None')
    
    (may_be_115983, more_types_in_union_115984) = may_be_none(mask_115981, None_115982)

    if may_be_115983:

        if more_types_in_union_115984:
            # Runtime conditional SSA (line 331)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'res' (line 332)
        res_115985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', res_115985)

        if more_types_in_union_115984:
            # SSA join for if statement (line 331)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to all(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'mask' (line 335)
    mask_115988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'mask', False)
    # Processing the call keyword arguments (line 335)
    # Getting the type of 'axis' (line 335)
    axis_115989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'axis', False)
    keyword_115990 = axis_115989
    # Getting the type of 'keepdims' (line 335)
    keepdims_115991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 48), 'keepdims', False)
    keyword_115992 = keepdims_115991
    kwargs_115993 = {'keepdims': keyword_115992, 'axis': keyword_115990}
    # Getting the type of 'np' (line 335)
    np_115986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 335)
    all_115987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 15), np_115986, 'all')
    # Calling all(args, kwargs) (line 335)
    all_call_result_115994 = invoke(stypy.reporting.localization.Localization(__file__, 335, 15), all_115987, *[mask_115988], **kwargs_115993)
    
    # Assigning a type to the variable 'mask' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'mask', all_call_result_115994)
    
    
    # Call to any(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'mask' (line 336)
    mask_115997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'mask', False)
    # Processing the call keyword arguments (line 336)
    kwargs_115998 = {}
    # Getting the type of 'np' (line 336)
    np_115995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 336)
    any_115996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 11), np_115995, 'any')
    # Calling any(args, kwargs) (line 336)
    any_call_result_115999 = invoke(stypy.reporting.localization.Localization(__file__, 336, 11), any_115996, *[mask_115997], **kwargs_115998)
    
    # Testing the type of an if condition (line 336)
    if_condition_116000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), any_call_result_115999)
    # Assigning a type to the variable 'if_condition_116000' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_116000', if_condition_116000)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to _copyto(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'res' (line 337)
    res_116002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 26), 'res', False)
    # Getting the type of 'np' (line 337)
    np_116003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'np', False)
    # Obtaining the member 'nan' of a type (line 337)
    nan_116004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 31), np_116003, 'nan')
    # Getting the type of 'mask' (line 337)
    mask_116005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 39), 'mask', False)
    # Processing the call keyword arguments (line 337)
    kwargs_116006 = {}
    # Getting the type of '_copyto' (line 337)
    _copyto_116001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), '_copyto', False)
    # Calling _copyto(args, kwargs) (line 337)
    _copyto_call_result_116007 = invoke(stypy.reporting.localization.Localization(__file__, 337, 18), _copyto_116001, *[res_116002, nan_116004, mask_116005], **kwargs_116006)
    
    # Assigning a type to the variable 'res' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'res', _copyto_call_result_116007)
    
    # Call to warn(...): (line 338)
    # Processing the call arguments (line 338)
    str_116010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 26), 'str', 'All-NaN axis encountered')
    # Getting the type of 'RuntimeWarning' (line 338)
    RuntimeWarning_116011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 54), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 338)
    kwargs_116012 = {}
    # Getting the type of 'warnings' (line 338)
    warnings_116008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 338)
    warn_116009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), warnings_116008, 'warn')
    # Calling warn(args, kwargs) (line 338)
    warn_call_result_116013 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), warn_116009, *[str_116010, RuntimeWarning_116011], **kwargs_116012)
    
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 339)
    res_116014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type', res_116014)
    
    # ################# End of 'nanmax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanmax' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_116015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116015)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanmax'
    return stypy_return_type_116015

# Assigning a type to the variable 'nanmax' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'nanmax', nanmax)

@norecursion
def nanargmin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 342)
    None_116016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'None')
    defaults = [None_116016]
    # Create a new context for function 'nanargmin'
    module_type_store = module_type_store.open_function_context('nanargmin', 342, 0, False)
    
    # Passed parameters checking function
    nanargmin.stypy_localization = localization
    nanargmin.stypy_type_of_self = None
    nanargmin.stypy_type_store = module_type_store
    nanargmin.stypy_function_name = 'nanargmin'
    nanargmin.stypy_param_names_list = ['a', 'axis']
    nanargmin.stypy_varargs_param_name = None
    nanargmin.stypy_kwargs_param_name = None
    nanargmin.stypy_call_defaults = defaults
    nanargmin.stypy_call_varargs = varargs
    nanargmin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanargmin', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanargmin', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanargmin(...)' code ##################

    str_116017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, (-1)), 'str', '\n    Return the indices of the minimum values in the specified axis ignoring\n    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the results\n    cannot be trusted if a slice contains only NaNs and Infs.\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : int, optional\n        Axis along which to operate.  By default flattened input is used.\n\n    Returns\n    -------\n    index_array : ndarray\n        An array of indices or a single index value.\n\n    See Also\n    --------\n    argmin, nanargmax\n\n    Examples\n    --------\n    >>> a = np.array([[np.nan, 4], [2, 3]])\n    >>> np.argmin(a)\n    0\n    >>> np.nanargmin(a)\n    2\n    >>> np.nanargmin(a, axis=0)\n    array([1, 1])\n    >>> np.nanargmin(a, axis=1)\n    array([1, 0])\n\n    ')
    
    # Assigning a Call to a Tuple (line 377):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'a' (line 377)
    a_116019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 27), 'a', False)
    # Getting the type of 'np' (line 377)
    np_116020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'np', False)
    # Obtaining the member 'inf' of a type (line 377)
    inf_116021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 30), np_116020, 'inf')
    # Processing the call keyword arguments (line 377)
    kwargs_116022 = {}
    # Getting the type of '_replace_nan' (line 377)
    _replace_nan_116018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 377)
    _replace_nan_call_result_116023 = invoke(stypy.reporting.localization.Localization(__file__, 377, 14), _replace_nan_116018, *[a_116019, inf_116021], **kwargs_116022)
    
    # Assigning a type to the variable 'call_assignment_115603' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115603', _replace_nan_call_result_116023)
    
    # Assigning a Call to a Name (line 377):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116027 = {}
    # Getting the type of 'call_assignment_115603' (line 377)
    call_assignment_115603_116024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115603', False)
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___116025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 4), call_assignment_115603_116024, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116028 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116025, *[int_116026], **kwargs_116027)
    
    # Assigning a type to the variable 'call_assignment_115604' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115604', getitem___call_result_116028)
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'call_assignment_115604' (line 377)
    call_assignment_115604_116029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115604')
    # Assigning a type to the variable 'a' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'a', call_assignment_115604_116029)
    
    # Assigning a Call to a Name (line 377):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116033 = {}
    # Getting the type of 'call_assignment_115603' (line 377)
    call_assignment_115603_116030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115603', False)
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___116031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 4), call_assignment_115603_116030, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116034 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116031, *[int_116032], **kwargs_116033)
    
    # Assigning a type to the variable 'call_assignment_115605' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115605', getitem___call_result_116034)
    
    # Assigning a Name to a Name (line 377):
    # Getting the type of 'call_assignment_115605' (line 377)
    call_assignment_115605_116035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'call_assignment_115605')
    # Assigning a type to the variable 'mask' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 7), 'mask', call_assignment_115605_116035)
    
    # Assigning a Call to a Name (line 378):
    
    # Assigning a Call to a Name (line 378):
    
    # Call to argmin(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'a' (line 378)
    a_116038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 20), 'a', False)
    # Processing the call keyword arguments (line 378)
    # Getting the type of 'axis' (line 378)
    axis_116039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 28), 'axis', False)
    keyword_116040 = axis_116039
    kwargs_116041 = {'axis': keyword_116040}
    # Getting the type of 'np' (line 378)
    np_116036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 10), 'np', False)
    # Obtaining the member 'argmin' of a type (line 378)
    argmin_116037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 10), np_116036, 'argmin')
    # Calling argmin(args, kwargs) (line 378)
    argmin_call_result_116042 = invoke(stypy.reporting.localization.Localization(__file__, 378, 10), argmin_116037, *[a_116038], **kwargs_116041)
    
    # Assigning a type to the variable 'res' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'res', argmin_call_result_116042)
    
    # Type idiom detected: calculating its left and rigth part (line 379)
    # Getting the type of 'mask' (line 379)
    mask_116043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'mask')
    # Getting the type of 'None' (line 379)
    None_116044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'None')
    
    (may_be_116045, more_types_in_union_116046) = may_not_be_none(mask_116043, None_116044)

    if may_be_116045:

        if more_types_in_union_116046:
            # Runtime conditional SSA (line 379)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to all(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'mask' (line 380)
        mask_116049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'mask', False)
        # Processing the call keyword arguments (line 380)
        # Getting the type of 'axis' (line 380)
        axis_116050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 33), 'axis', False)
        keyword_116051 = axis_116050
        kwargs_116052 = {'axis': keyword_116051}
        # Getting the type of 'np' (line 380)
        np_116047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 380)
        all_116048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 15), np_116047, 'all')
        # Calling all(args, kwargs) (line 380)
        all_call_result_116053 = invoke(stypy.reporting.localization.Localization(__file__, 380, 15), all_116048, *[mask_116049], **kwargs_116052)
        
        # Assigning a type to the variable 'mask' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'mask', all_call_result_116053)
        
        
        # Call to any(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'mask' (line 381)
        mask_116056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 18), 'mask', False)
        # Processing the call keyword arguments (line 381)
        kwargs_116057 = {}
        # Getting the type of 'np' (line 381)
        np_116054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 381)
        any_116055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 11), np_116054, 'any')
        # Calling any(args, kwargs) (line 381)
        any_call_result_116058 = invoke(stypy.reporting.localization.Localization(__file__, 381, 11), any_116055, *[mask_116056], **kwargs_116057)
        
        # Testing the type of an if condition (line 381)
        if_condition_116059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), any_call_result_116058)
        # Assigning a type to the variable 'if_condition_116059' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_116059', if_condition_116059)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 382)
        # Processing the call arguments (line 382)
        str_116061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 29), 'str', 'All-NaN slice encountered')
        # Processing the call keyword arguments (line 382)
        kwargs_116062 = {}
        # Getting the type of 'ValueError' (line 382)
        ValueError_116060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 382)
        ValueError_call_result_116063 = invoke(stypy.reporting.localization.Localization(__file__, 382, 18), ValueError_116060, *[str_116061], **kwargs_116062)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 382, 12), ValueError_call_result_116063, 'raise parameter', BaseException)
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_116046:
            # SSA join for if statement (line 379)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'res' (line 383)
    res_116064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'stypy_return_type', res_116064)
    
    # ################# End of 'nanargmin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanargmin' in the type store
    # Getting the type of 'stypy_return_type' (line 342)
    stypy_return_type_116065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanargmin'
    return stypy_return_type_116065

# Assigning a type to the variable 'nanargmin' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'nanargmin', nanargmin)

@norecursion
def nanargmax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 386)
    None_116066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 22), 'None')
    defaults = [None_116066]
    # Create a new context for function 'nanargmax'
    module_type_store = module_type_store.open_function_context('nanargmax', 386, 0, False)
    
    # Passed parameters checking function
    nanargmax.stypy_localization = localization
    nanargmax.stypy_type_of_self = None
    nanargmax.stypy_type_store = module_type_store
    nanargmax.stypy_function_name = 'nanargmax'
    nanargmax.stypy_param_names_list = ['a', 'axis']
    nanargmax.stypy_varargs_param_name = None
    nanargmax.stypy_kwargs_param_name = None
    nanargmax.stypy_call_defaults = defaults
    nanargmax.stypy_call_varargs = varargs
    nanargmax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanargmax', ['a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanargmax', localization, ['a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanargmax(...)' code ##################

    str_116067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', '\n    Return the indices of the maximum values in the specified axis ignoring\n    NaNs. For all-NaN slices ``ValueError`` is raised. Warning: the\n    results cannot be trusted if a slice contains only NaNs and -Infs.\n\n\n    Parameters\n    ----------\n    a : array_like\n        Input data.\n    axis : int, optional\n        Axis along which to operate.  By default flattened input is used.\n\n    Returns\n    -------\n    index_array : ndarray\n        An array of indices or a single index value.\n\n    See Also\n    --------\n    argmax, nanargmin\n\n    Examples\n    --------\n    >>> a = np.array([[np.nan, 4], [2, 3]])\n    >>> np.argmax(a)\n    0\n    >>> np.nanargmax(a)\n    1\n    >>> np.nanargmax(a, axis=0)\n    array([1, 0])\n    >>> np.nanargmax(a, axis=1)\n    array([1, 1])\n\n    ')
    
    # Assigning a Call to a Tuple (line 422):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'a' (line 422)
    a_116069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 27), 'a', False)
    
    # Getting the type of 'np' (line 422)
    np_116070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 31), 'np', False)
    # Obtaining the member 'inf' of a type (line 422)
    inf_116071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 31), np_116070, 'inf')
    # Applying the 'usub' unary operator (line 422)
    result___neg___116072 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 30), 'usub', inf_116071)
    
    # Processing the call keyword arguments (line 422)
    kwargs_116073 = {}
    # Getting the type of '_replace_nan' (line 422)
    _replace_nan_116068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 14), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 422)
    _replace_nan_call_result_116074 = invoke(stypy.reporting.localization.Localization(__file__, 422, 14), _replace_nan_116068, *[a_116069, result___neg___116072], **kwargs_116073)
    
    # Assigning a type to the variable 'call_assignment_115606' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115606', _replace_nan_call_result_116074)
    
    # Assigning a Call to a Name (line 422):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116078 = {}
    # Getting the type of 'call_assignment_115606' (line 422)
    call_assignment_115606_116075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115606', False)
    # Obtaining the member '__getitem__' of a type (line 422)
    getitem___116076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 4), call_assignment_115606_116075, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116079 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116076, *[int_116077], **kwargs_116078)
    
    # Assigning a type to the variable 'call_assignment_115607' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115607', getitem___call_result_116079)
    
    # Assigning a Name to a Name (line 422):
    # Getting the type of 'call_assignment_115607' (line 422)
    call_assignment_115607_116080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115607')
    # Assigning a type to the variable 'a' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'a', call_assignment_115607_116080)
    
    # Assigning a Call to a Name (line 422):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116084 = {}
    # Getting the type of 'call_assignment_115606' (line 422)
    call_assignment_115606_116081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115606', False)
    # Obtaining the member '__getitem__' of a type (line 422)
    getitem___116082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 4), call_assignment_115606_116081, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116085 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116082, *[int_116083], **kwargs_116084)
    
    # Assigning a type to the variable 'call_assignment_115608' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115608', getitem___call_result_116085)
    
    # Assigning a Name to a Name (line 422):
    # Getting the type of 'call_assignment_115608' (line 422)
    call_assignment_115608_116086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'call_assignment_115608')
    # Assigning a type to the variable 'mask' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'mask', call_assignment_115608_116086)
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to argmax(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'a' (line 423)
    a_116089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'a', False)
    # Processing the call keyword arguments (line 423)
    # Getting the type of 'axis' (line 423)
    axis_116090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 28), 'axis', False)
    keyword_116091 = axis_116090
    kwargs_116092 = {'axis': keyword_116091}
    # Getting the type of 'np' (line 423)
    np_116087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 10), 'np', False)
    # Obtaining the member 'argmax' of a type (line 423)
    argmax_116088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 10), np_116087, 'argmax')
    # Calling argmax(args, kwargs) (line 423)
    argmax_call_result_116093 = invoke(stypy.reporting.localization.Localization(__file__, 423, 10), argmax_116088, *[a_116089], **kwargs_116092)
    
    # Assigning a type to the variable 'res' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'res', argmax_call_result_116093)
    
    # Type idiom detected: calculating its left and rigth part (line 424)
    # Getting the type of 'mask' (line 424)
    mask_116094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'mask')
    # Getting the type of 'None' (line 424)
    None_116095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'None')
    
    (may_be_116096, more_types_in_union_116097) = may_not_be_none(mask_116094, None_116095)

    if may_be_116096:

        if more_types_in_union_116097:
            # Runtime conditional SSA (line 424)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to all(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'mask' (line 425)
        mask_116100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'mask', False)
        # Processing the call keyword arguments (line 425)
        # Getting the type of 'axis' (line 425)
        axis_116101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 33), 'axis', False)
        keyword_116102 = axis_116101
        kwargs_116103 = {'axis': keyword_116102}
        # Getting the type of 'np' (line 425)
        np_116098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'np', False)
        # Obtaining the member 'all' of a type (line 425)
        all_116099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), np_116098, 'all')
        # Calling all(args, kwargs) (line 425)
        all_call_result_116104 = invoke(stypy.reporting.localization.Localization(__file__, 425, 15), all_116099, *[mask_116100], **kwargs_116103)
        
        # Assigning a type to the variable 'mask' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'mask', all_call_result_116104)
        
        
        # Call to any(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'mask' (line 426)
        mask_116107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 18), 'mask', False)
        # Processing the call keyword arguments (line 426)
        kwargs_116108 = {}
        # Getting the type of 'np' (line 426)
        np_116105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 426)
        any_116106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 11), np_116105, 'any')
        # Calling any(args, kwargs) (line 426)
        any_call_result_116109 = invoke(stypy.reporting.localization.Localization(__file__, 426, 11), any_116106, *[mask_116107], **kwargs_116108)
        
        # Testing the type of an if condition (line 426)
        if_condition_116110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 426, 8), any_call_result_116109)
        # Assigning a type to the variable 'if_condition_116110' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'if_condition_116110', if_condition_116110)
        # SSA begins for if statement (line 426)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 427)
        # Processing the call arguments (line 427)
        str_116112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 29), 'str', 'All-NaN slice encountered')
        # Processing the call keyword arguments (line 427)
        kwargs_116113 = {}
        # Getting the type of 'ValueError' (line 427)
        ValueError_116111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 427)
        ValueError_call_result_116114 = invoke(stypy.reporting.localization.Localization(__file__, 427, 18), ValueError_116111, *[str_116112], **kwargs_116113)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 427, 12), ValueError_call_result_116114, 'raise parameter', BaseException)
        # SSA join for if statement (line 426)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_116097:
            # SSA join for if statement (line 424)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'res' (line 428)
    res_116115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type', res_116115)
    
    # ################# End of 'nanargmax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanargmax' in the type store
    # Getting the type of 'stypy_return_type' (line 386)
    stypy_return_type_116116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanargmax'
    return stypy_return_type_116116

# Assigning a type to the variable 'nanargmax' (line 386)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'nanargmax', nanargmax)

@norecursion
def nansum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 431)
    None_116117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'None')
    # Getting the type of 'None' (line 431)
    None_116118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'None')
    # Getting the type of 'None' (line 431)
    None_116119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 41), 'None')
    int_116120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 56), 'int')
    defaults = [None_116117, None_116118, None_116119, int_116120]
    # Create a new context for function 'nansum'
    module_type_store = module_type_store.open_function_context('nansum', 431, 0, False)
    
    # Passed parameters checking function
    nansum.stypy_localization = localization
    nansum.stypy_type_of_self = None
    nansum.stypy_type_store = module_type_store
    nansum.stypy_function_name = 'nansum'
    nansum.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    nansum.stypy_varargs_param_name = None
    nansum.stypy_kwargs_param_name = None
    nansum.stypy_call_defaults = defaults
    nansum.stypy_call_varargs = varargs
    nansum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nansum', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nansum', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nansum(...)' code ##################

    str_116121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, (-1)), 'str', '\n    Return the sum of array elements over a given axis treating Not a\n    Numbers (NaNs) as zero.\n\n    In Numpy versions <= 1.8 Nan is returned for slices that are all-NaN or\n    empty. In later versions zero is returned.\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose sum is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the sum is computed. The default is to compute the\n        sum of the flattened array.\n    dtype : data-type, optional\n        The type of the returned array and of the accumulator in which the\n        elements are summed.  By default, the dtype of `a` is used.  An\n        exception is when `a` has an integer type with less precision than\n        the platform (u)intp. In that case, the default will be either\n        (u)int32 or (u)int64 depending on whether the platform is 32 or 64\n        bits. For inexact inputs, dtype must be inexact.\n\n        .. versionadded:: 1.8.0\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``. If provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.  See\n        `doc.ufuncs` for details. The casting of NaN to integer can yield\n        unexpected results.\n\n        .. versionadded:: 1.8.0\n    keepdims : bool, optional\n        If True, the axes which are reduced are left in the result as\n        dimensions with size one. With this option, the result will\n        broadcast correctly against the original `arr`.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    y : ndarray or numpy scalar\n\n    See Also\n    --------\n    numpy.sum : Sum across array propagating NaNs.\n    isnan : Show which elements are NaN.\n    isfinite: Show which elements are not NaN or +/-inf.\n\n    Notes\n    -----\n    If both positive and negative infinity are present, the sum will be Not\n    A Number (NaN).\n\n    Numpy integer arithmetic is modular. If the size of a sum exceeds the\n    size of an integer accumulator, its value will wrap around and the\n    result will be incorrect. Specifying ``dtype=double`` can alleviate\n    that problem.\n\n    Examples\n    --------\n    >>> np.nansum(1)\n    1\n    >>> np.nansum([1])\n    1\n    >>> np.nansum([1, np.nan])\n    1.0\n    >>> a = np.array([[1, 1], [1, np.nan]])\n    >>> np.nansum(a)\n    3.0\n    >>> np.nansum(a, axis=0)\n    array([ 2.,  1.])\n    >>> np.nansum([1, np.nan, np.inf])\n    inf\n    >>> np.nansum([1, np.nan, np.NINF])\n    -inf\n    >>> np.nansum([1, np.nan, np.inf, -np.inf]) # both +/- infinity present\n    nan\n\n    ')
    
    # Assigning a Call to a Tuple (line 512):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'a' (line 512)
    a_116123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 27), 'a', False)
    int_116124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 30), 'int')
    # Processing the call keyword arguments (line 512)
    kwargs_116125 = {}
    # Getting the type of '_replace_nan' (line 512)
    _replace_nan_116122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 14), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 512)
    _replace_nan_call_result_116126 = invoke(stypy.reporting.localization.Localization(__file__, 512, 14), _replace_nan_116122, *[a_116123, int_116124], **kwargs_116125)
    
    # Assigning a type to the variable 'call_assignment_115609' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115609', _replace_nan_call_result_116126)
    
    # Assigning a Call to a Name (line 512):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116130 = {}
    # Getting the type of 'call_assignment_115609' (line 512)
    call_assignment_115609_116127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115609', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___116128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 4), call_assignment_115609_116127, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116131 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116128, *[int_116129], **kwargs_116130)
    
    # Assigning a type to the variable 'call_assignment_115610' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115610', getitem___call_result_116131)
    
    # Assigning a Name to a Name (line 512):
    # Getting the type of 'call_assignment_115610' (line 512)
    call_assignment_115610_116132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115610')
    # Assigning a type to the variable 'a' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'a', call_assignment_115610_116132)
    
    # Assigning a Call to a Name (line 512):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116136 = {}
    # Getting the type of 'call_assignment_115609' (line 512)
    call_assignment_115609_116133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115609', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___116134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 4), call_assignment_115609_116133, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116137 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116134, *[int_116135], **kwargs_116136)
    
    # Assigning a type to the variable 'call_assignment_115611' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115611', getitem___call_result_116137)
    
    # Assigning a Name to a Name (line 512):
    # Getting the type of 'call_assignment_115611' (line 512)
    call_assignment_115611_116138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'call_assignment_115611')
    # Assigning a type to the variable 'mask' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 7), 'mask', call_assignment_115611_116138)
    
    # Call to sum(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'a' (line 513)
    a_116141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'a', False)
    # Processing the call keyword arguments (line 513)
    # Getting the type of 'axis' (line 513)
    axis_116142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 26), 'axis', False)
    keyword_116143 = axis_116142
    # Getting the type of 'dtype' (line 513)
    dtype_116144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 38), 'dtype', False)
    keyword_116145 = dtype_116144
    # Getting the type of 'out' (line 513)
    out_116146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 49), 'out', False)
    keyword_116147 = out_116146
    # Getting the type of 'keepdims' (line 513)
    keepdims_116148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 63), 'keepdims', False)
    keyword_116149 = keepdims_116148
    kwargs_116150 = {'dtype': keyword_116145, 'out': keyword_116147, 'keepdims': keyword_116149, 'axis': keyword_116143}
    # Getting the type of 'np' (line 513)
    np_116139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 11), 'np', False)
    # Obtaining the member 'sum' of a type (line 513)
    sum_116140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 11), np_116139, 'sum')
    # Calling sum(args, kwargs) (line 513)
    sum_call_result_116151 = invoke(stypy.reporting.localization.Localization(__file__, 513, 11), sum_116140, *[a_116141], **kwargs_116150)
    
    # Assigning a type to the variable 'stypy_return_type' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type', sum_call_result_116151)
    
    # ################# End of 'nansum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nansum' in the type store
    # Getting the type of 'stypy_return_type' (line 431)
    stypy_return_type_116152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nansum'
    return stypy_return_type_116152

# Assigning a type to the variable 'nansum' (line 431)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 0), 'nansum', nansum)

@norecursion
def nanprod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 516)
    None_116153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'None')
    # Getting the type of 'None' (line 516)
    None_116154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 32), 'None')
    # Getting the type of 'None' (line 516)
    None_116155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 42), 'None')
    int_116156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 57), 'int')
    defaults = [None_116153, None_116154, None_116155, int_116156]
    # Create a new context for function 'nanprod'
    module_type_store = module_type_store.open_function_context('nanprod', 516, 0, False)
    
    # Passed parameters checking function
    nanprod.stypy_localization = localization
    nanprod.stypy_type_of_self = None
    nanprod.stypy_type_store = module_type_store
    nanprod.stypy_function_name = 'nanprod'
    nanprod.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    nanprod.stypy_varargs_param_name = None
    nanprod.stypy_kwargs_param_name = None
    nanprod.stypy_call_defaults = defaults
    nanprod.stypy_call_varargs = varargs
    nanprod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanprod', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanprod', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanprod(...)' code ##################

    str_116157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, (-1)), 'str', '\n    Return the product of array elements over a given axis treating Not a\n    Numbers (NaNs) as zero.\n\n    One is returned for slices that are all-NaN or empty.\n\n    .. versionadded:: 1.10.0\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose sum is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the product is computed. The default is to compute\n        the product of the flattened array.\n    dtype : data-type, optional\n        The type of the returned array and of the accumulator in which the\n        elements are summed.  By default, the dtype of `a` is used.  An\n        exception is when `a` has an integer type with less precision than\n        the platform (u)intp. In that case, the default will be either\n        (u)int32 or (u)int64 depending on whether the platform is 32 or 64\n        bits. For inexact inputs, dtype must be inexact.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``. If provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.  See\n        `doc.ufuncs` for details. The casting of NaN to integer can yield\n        unexpected results.\n    keepdims : bool, optional\n        If True, the axes which are reduced are left in the result as\n        dimensions with size one. With this option, the result will\n        broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    y : ndarray or numpy scalar\n\n    See Also\n    --------\n    numpy.prod : Product across array propagating NaNs.\n    isnan : Show which elements are NaN.\n\n    Notes\n    -----\n    Numpy integer arithmetic is modular. If the size of a product exceeds\n    the size of an integer accumulator, its value will wrap around and the\n    result will be incorrect. Specifying ``dtype=double`` can alleviate\n    that problem.\n\n    Examples\n    --------\n    >>> np.nanprod(1)\n    1\n    >>> np.nanprod([1])\n    1\n    >>> np.nanprod([1, np.nan])\n    1.0\n    >>> a = np.array([[1, 2], [3, np.nan]])\n    >>> np.nanprod(a)\n    6.0\n    >>> np.nanprod(a, axis=0)\n    array([ 3.,  2.])\n\n    ')
    
    # Assigning a Call to a Tuple (line 582):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'a' (line 582)
    a_116159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 27), 'a', False)
    int_116160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 30), 'int')
    # Processing the call keyword arguments (line 582)
    kwargs_116161 = {}
    # Getting the type of '_replace_nan' (line 582)
    _replace_nan_116158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 14), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 582)
    _replace_nan_call_result_116162 = invoke(stypy.reporting.localization.Localization(__file__, 582, 14), _replace_nan_116158, *[a_116159, int_116160], **kwargs_116161)
    
    # Assigning a type to the variable 'call_assignment_115612' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115612', _replace_nan_call_result_116162)
    
    # Assigning a Call to a Name (line 582):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116166 = {}
    # Getting the type of 'call_assignment_115612' (line 582)
    call_assignment_115612_116163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115612', False)
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___116164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 4), call_assignment_115612_116163, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116167 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116164, *[int_116165], **kwargs_116166)
    
    # Assigning a type to the variable 'call_assignment_115613' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115613', getitem___call_result_116167)
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'call_assignment_115613' (line 582)
    call_assignment_115613_116168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115613')
    # Assigning a type to the variable 'a' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'a', call_assignment_115613_116168)
    
    # Assigning a Call to a Name (line 582):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116172 = {}
    # Getting the type of 'call_assignment_115612' (line 582)
    call_assignment_115612_116169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115612', False)
    # Obtaining the member '__getitem__' of a type (line 582)
    getitem___116170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 4), call_assignment_115612_116169, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116173 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116170, *[int_116171], **kwargs_116172)
    
    # Assigning a type to the variable 'call_assignment_115614' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115614', getitem___call_result_116173)
    
    # Assigning a Name to a Name (line 582):
    # Getting the type of 'call_assignment_115614' (line 582)
    call_assignment_115614_116174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'call_assignment_115614')
    # Assigning a type to the variable 'mask' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 7), 'mask', call_assignment_115614_116174)
    
    # Call to prod(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'a' (line 583)
    a_116177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 19), 'a', False)
    # Processing the call keyword arguments (line 583)
    # Getting the type of 'axis' (line 583)
    axis_116178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 27), 'axis', False)
    keyword_116179 = axis_116178
    # Getting the type of 'dtype' (line 583)
    dtype_116180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 39), 'dtype', False)
    keyword_116181 = dtype_116180
    # Getting the type of 'out' (line 583)
    out_116182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 50), 'out', False)
    keyword_116183 = out_116182
    # Getting the type of 'keepdims' (line 583)
    keepdims_116184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 64), 'keepdims', False)
    keyword_116185 = keepdims_116184
    kwargs_116186 = {'dtype': keyword_116181, 'out': keyword_116183, 'keepdims': keyword_116185, 'axis': keyword_116179}
    # Getting the type of 'np' (line 583)
    np_116175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 11), 'np', False)
    # Obtaining the member 'prod' of a type (line 583)
    prod_116176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 11), np_116175, 'prod')
    # Calling prod(args, kwargs) (line 583)
    prod_call_result_116187 = invoke(stypy.reporting.localization.Localization(__file__, 583, 11), prod_116176, *[a_116177], **kwargs_116186)
    
    # Assigning a type to the variable 'stypy_return_type' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'stypy_return_type', prod_call_result_116187)
    
    # ################# End of 'nanprod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanprod' in the type store
    # Getting the type of 'stypy_return_type' (line 516)
    stypy_return_type_116188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanprod'
    return stypy_return_type_116188

# Assigning a type to the variable 'nanprod' (line 516)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'nanprod', nanprod)

@norecursion
def nanmean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 586)
    None_116189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'None')
    # Getting the type of 'None' (line 586)
    None_116190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 32), 'None')
    # Getting the type of 'None' (line 586)
    None_116191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 42), 'None')
    # Getting the type of 'False' (line 586)
    False_116192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 57), 'False')
    defaults = [None_116189, None_116190, None_116191, False_116192]
    # Create a new context for function 'nanmean'
    module_type_store = module_type_store.open_function_context('nanmean', 586, 0, False)
    
    # Passed parameters checking function
    nanmean.stypy_localization = localization
    nanmean.stypy_type_of_self = None
    nanmean.stypy_type_store = module_type_store
    nanmean.stypy_function_name = 'nanmean'
    nanmean.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'keepdims']
    nanmean.stypy_varargs_param_name = None
    nanmean.stypy_kwargs_param_name = None
    nanmean.stypy_call_defaults = defaults
    nanmean.stypy_call_varargs = varargs
    nanmean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanmean', ['a', 'axis', 'dtype', 'out', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanmean', localization, ['a', 'axis', 'dtype', 'out', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanmean(...)' code ##################

    str_116193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, (-1)), 'str', '\n    Compute the arithmetic mean along the specified axis, ignoring NaNs.\n\n    Returns the average of the array elements.  The average is taken over\n    the flattened array by default, otherwise over the specified axis.\n    `float64` intermediate and return values are used for integer inputs.\n\n    For all-NaN slices, NaN is returned and a `RuntimeWarning` is raised.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose mean is desired. If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the means are computed. The default is to compute\n        the mean of the flattened array.\n    dtype : data-type, optional\n        Type to use in computing the mean.  For integer inputs, the default\n        is `float64`; for inexact inputs, it is the same as the input\n        dtype.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  The default\n        is ``None``; if provided, it must have the same shape as the\n        expected output, but the type will be cast if necessary.  See\n        `doc.ufuncs` for details.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in the\n        result as dimensions with size one. With this option, the result\n        will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    m : ndarray, see dtype parameter above\n        If `out=None`, returns a new array containing the mean values,\n        otherwise a reference to the output array is returned. Nan is\n        returned for slices that contain only NaNs.\n\n    See Also\n    --------\n    average : Weighted average\n    mean : Arithmetic mean taken while not ignoring NaNs\n    var, nanvar\n\n    Notes\n    -----\n    The arithmetic mean is the sum of the non-NaN elements along the axis\n    divided by the number of non-NaN elements.\n\n    Note that for floating-point input, the mean is computed using the same\n    precision the input has.  Depending on the input data, this can cause\n    the results to be inaccurate, especially for `float32`.  Specifying a\n    higher-precision accumulator using the `dtype` keyword can alleviate\n    this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, np.nan], [3, 4]])\n    >>> np.nanmean(a)\n    2.6666666666666665\n    >>> np.nanmean(a, axis=0)\n    array([ 2.,  4.])\n    >>> np.nanmean(a, axis=1)\n    array([ 1.,  3.5])\n\n    ')
    
    # Assigning a Call to a Tuple (line 655):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 655)
    # Processing the call arguments (line 655)
    # Getting the type of 'a' (line 655)
    a_116195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 29), 'a', False)
    int_116196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 32), 'int')
    # Processing the call keyword arguments (line 655)
    kwargs_116197 = {}
    # Getting the type of '_replace_nan' (line 655)
    _replace_nan_116194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 655)
    _replace_nan_call_result_116198 = invoke(stypy.reporting.localization.Localization(__file__, 655, 16), _replace_nan_116194, *[a_116195, int_116196], **kwargs_116197)
    
    # Assigning a type to the variable 'call_assignment_115615' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115615', _replace_nan_call_result_116198)
    
    # Assigning a Call to a Name (line 655):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116202 = {}
    # Getting the type of 'call_assignment_115615' (line 655)
    call_assignment_115615_116199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115615', False)
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___116200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), call_assignment_115615_116199, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116203 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116200, *[int_116201], **kwargs_116202)
    
    # Assigning a type to the variable 'call_assignment_115616' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115616', getitem___call_result_116203)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'call_assignment_115616' (line 655)
    call_assignment_115616_116204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115616')
    # Assigning a type to the variable 'arr' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'arr', call_assignment_115616_116204)
    
    # Assigning a Call to a Name (line 655):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116208 = {}
    # Getting the type of 'call_assignment_115615' (line 655)
    call_assignment_115615_116205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115615', False)
    # Obtaining the member '__getitem__' of a type (line 655)
    getitem___116206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 4), call_assignment_115615_116205, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116209 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116206, *[int_116207], **kwargs_116208)
    
    # Assigning a type to the variable 'call_assignment_115617' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115617', getitem___call_result_116209)
    
    # Assigning a Name to a Name (line 655):
    # Getting the type of 'call_assignment_115617' (line 655)
    call_assignment_115617_116210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 4), 'call_assignment_115617')
    # Assigning a type to the variable 'mask' (line 655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 9), 'mask', call_assignment_115617_116210)
    
    # Type idiom detected: calculating its left and rigth part (line 656)
    # Getting the type of 'mask' (line 656)
    mask_116211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 7), 'mask')
    # Getting the type of 'None' (line 656)
    None_116212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 15), 'None')
    
    (may_be_116213, more_types_in_union_116214) = may_be_none(mask_116211, None_116212)

    if may_be_116213:

        if more_types_in_union_116214:
            # Runtime conditional SSA (line 656)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to mean(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'arr' (line 657)
        arr_116217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 23), 'arr', False)
        # Processing the call keyword arguments (line 657)
        # Getting the type of 'axis' (line 657)
        axis_116218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 33), 'axis', False)
        keyword_116219 = axis_116218
        # Getting the type of 'dtype' (line 657)
        dtype_116220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 45), 'dtype', False)
        keyword_116221 = dtype_116220
        # Getting the type of 'out' (line 657)
        out_116222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 56), 'out', False)
        keyword_116223 = out_116222
        # Getting the type of 'keepdims' (line 657)
        keepdims_116224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 70), 'keepdims', False)
        keyword_116225 = keepdims_116224
        kwargs_116226 = {'dtype': keyword_116221, 'out': keyword_116223, 'keepdims': keyword_116225, 'axis': keyword_116219}
        # Getting the type of 'np' (line 657)
        np_116215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'np', False)
        # Obtaining the member 'mean' of a type (line 657)
        mean_116216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 15), np_116215, 'mean')
        # Calling mean(args, kwargs) (line 657)
        mean_call_result_116227 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), mean_116216, *[arr_116217], **kwargs_116226)
        
        # Assigning a type to the variable 'stypy_return_type' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'stypy_return_type', mean_call_result_116227)

        if more_types_in_union_116214:
            # SSA join for if statement (line 656)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 659)
    # Getting the type of 'dtype' (line 659)
    dtype_116228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'dtype')
    # Getting the type of 'None' (line 659)
    None_116229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'None')
    
    (may_be_116230, more_types_in_union_116231) = may_not_be_none(dtype_116228, None_116229)

    if may_be_116230:

        if more_types_in_union_116231:
            # Runtime conditional SSA (line 659)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 660):
        
        # Assigning a Call to a Name (line 660):
        
        # Call to dtype(...): (line 660)
        # Processing the call arguments (line 660)
        # Getting the type of 'dtype' (line 660)
        dtype_116234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 25), 'dtype', False)
        # Processing the call keyword arguments (line 660)
        kwargs_116235 = {}
        # Getting the type of 'np' (line 660)
        np_116232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'np', False)
        # Obtaining the member 'dtype' of a type (line 660)
        dtype_116233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 16), np_116232, 'dtype')
        # Calling dtype(args, kwargs) (line 660)
        dtype_call_result_116236 = invoke(stypy.reporting.localization.Localization(__file__, 660, 16), dtype_116233, *[dtype_116234], **kwargs_116235)
        
        # Assigning a type to the variable 'dtype' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 8), 'dtype', dtype_call_result_116236)

        if more_types_in_union_116231:
            # SSA join for if statement (line 659)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 661)
    dtype_116237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 7), 'dtype')
    # Getting the type of 'None' (line 661)
    None_116238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 20), 'None')
    # Applying the binary operator 'isnot' (line 661)
    result_is_not_116239 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 7), 'isnot', dtype_116237, None_116238)
    
    
    
    # Call to issubclass(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'dtype' (line 661)
    dtype_116241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 44), 'dtype', False)
    # Obtaining the member 'type' of a type (line 661)
    type_116242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 44), dtype_116241, 'type')
    # Getting the type of 'np' (line 661)
    np_116243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 56), 'np', False)
    # Obtaining the member 'inexact' of a type (line 661)
    inexact_116244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 56), np_116243, 'inexact')
    # Processing the call keyword arguments (line 661)
    kwargs_116245 = {}
    # Getting the type of 'issubclass' (line 661)
    issubclass_116240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 33), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 661)
    issubclass_call_result_116246 = invoke(stypy.reporting.localization.Localization(__file__, 661, 33), issubclass_116240, *[type_116242, inexact_116244], **kwargs_116245)
    
    # Applying the 'not' unary operator (line 661)
    result_not__116247 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 29), 'not', issubclass_call_result_116246)
    
    # Applying the binary operator 'and' (line 661)
    result_and_keyword_116248 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 7), 'and', result_is_not_116239, result_not__116247)
    
    # Testing the type of an if condition (line 661)
    if_condition_116249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 4), result_and_keyword_116248)
    # Assigning a type to the variable 'if_condition_116249' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'if_condition_116249', if_condition_116249)
    # SSA begins for if statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 662)
    # Processing the call arguments (line 662)
    str_116251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 24), 'str', 'If a is inexact, then dtype must be inexact')
    # Processing the call keyword arguments (line 662)
    kwargs_116252 = {}
    # Getting the type of 'TypeError' (line 662)
    TypeError_116250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 662)
    TypeError_call_result_116253 = invoke(stypy.reporting.localization.Localization(__file__, 662, 14), TypeError_116250, *[str_116251], **kwargs_116252)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 662, 8), TypeError_call_result_116253, 'raise parameter', BaseException)
    # SSA join for if statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'out' (line 663)
    out_116254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 7), 'out')
    # Getting the type of 'None' (line 663)
    None_116255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 18), 'None')
    # Applying the binary operator 'isnot' (line 663)
    result_is_not_116256 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 7), 'isnot', out_116254, None_116255)
    
    
    
    # Call to issubclass(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'out' (line 663)
    out_116258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 42), 'out', False)
    # Obtaining the member 'dtype' of a type (line 663)
    dtype_116259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 42), out_116258, 'dtype')
    # Obtaining the member 'type' of a type (line 663)
    type_116260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 42), dtype_116259, 'type')
    # Getting the type of 'np' (line 663)
    np_116261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 58), 'np', False)
    # Obtaining the member 'inexact' of a type (line 663)
    inexact_116262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 58), np_116261, 'inexact')
    # Processing the call keyword arguments (line 663)
    kwargs_116263 = {}
    # Getting the type of 'issubclass' (line 663)
    issubclass_116257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 31), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 663)
    issubclass_call_result_116264 = invoke(stypy.reporting.localization.Localization(__file__, 663, 31), issubclass_116257, *[type_116260, inexact_116262], **kwargs_116263)
    
    # Applying the 'not' unary operator (line 663)
    result_not__116265 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 27), 'not', issubclass_call_result_116264)
    
    # Applying the binary operator 'and' (line 663)
    result_and_keyword_116266 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 7), 'and', result_is_not_116256, result_not__116265)
    
    # Testing the type of an if condition (line 663)
    if_condition_116267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 663, 4), result_and_keyword_116266)
    # Assigning a type to the variable 'if_condition_116267' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 4), 'if_condition_116267', if_condition_116267)
    # SSA begins for if statement (line 663)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 664)
    # Processing the call arguments (line 664)
    str_116269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 24), 'str', 'If a is inexact, then out must be inexact')
    # Processing the call keyword arguments (line 664)
    kwargs_116270 = {}
    # Getting the type of 'TypeError' (line 664)
    TypeError_116268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 664)
    TypeError_call_result_116271 = invoke(stypy.reporting.localization.Localization(__file__, 664, 14), TypeError_116268, *[str_116269], **kwargs_116270)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 664, 8), TypeError_call_result_116271, 'raise parameter', BaseException)
    # SSA join for if statement (line 663)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to catch_warnings(...): (line 667)
    # Processing the call keyword arguments (line 667)
    kwargs_116274 = {}
    # Getting the type of 'warnings' (line 667)
    warnings_116272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 9), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 667)
    catch_warnings_116273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 9), warnings_116272, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 667)
    catch_warnings_call_result_116275 = invoke(stypy.reporting.localization.Localization(__file__, 667, 9), catch_warnings_116273, *[], **kwargs_116274)
    
    with_116276 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 667, 9), catch_warnings_call_result_116275, 'with parameter', '__enter__', '__exit__')

    if with_116276:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 667)
        enter___116277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 9), catch_warnings_call_result_116275, '__enter__')
        with_enter_116278 = invoke(stypy.reporting.localization.Localization(__file__, 667, 9), enter___116277)
        
        # Call to simplefilter(...): (line 668)
        # Processing the call arguments (line 668)
        str_116281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 30), 'str', 'ignore')
        # Processing the call keyword arguments (line 668)
        kwargs_116282 = {}
        # Getting the type of 'warnings' (line 668)
        warnings_116279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 668)
        simplefilter_116280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 8), warnings_116279, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 668)
        simplefilter_call_result_116283 = invoke(stypy.reporting.localization.Localization(__file__, 668, 8), simplefilter_116280, *[str_116281], **kwargs_116282)
        
        
        # Assigning a Call to a Name (line 669):
        
        # Assigning a Call to a Name (line 669):
        
        # Call to sum(...): (line 669)
        # Processing the call arguments (line 669)
        
        # Getting the type of 'mask' (line 669)
        mask_116286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 22), 'mask', False)
        # Applying the '~' unary operator (line 669)
        result_inv_116287 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 21), '~', mask_116286)
        
        # Processing the call keyword arguments (line 669)
        # Getting the type of 'axis' (line 669)
        axis_116288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 33), 'axis', False)
        keyword_116289 = axis_116288
        # Getting the type of 'np' (line 669)
        np_116290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 45), 'np', False)
        # Obtaining the member 'intp' of a type (line 669)
        intp_116291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 45), np_116290, 'intp')
        keyword_116292 = intp_116291
        # Getting the type of 'keepdims' (line 669)
        keepdims_116293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 63), 'keepdims', False)
        keyword_116294 = keepdims_116293
        kwargs_116295 = {'dtype': keyword_116292, 'keepdims': keyword_116294, 'axis': keyword_116289}
        # Getting the type of 'np' (line 669)
        np_116284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 14), 'np', False)
        # Obtaining the member 'sum' of a type (line 669)
        sum_116285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 14), np_116284, 'sum')
        # Calling sum(args, kwargs) (line 669)
        sum_call_result_116296 = invoke(stypy.reporting.localization.Localization(__file__, 669, 14), sum_116285, *[result_inv_116287], **kwargs_116295)
        
        # Assigning a type to the variable 'cnt' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 8), 'cnt', sum_call_result_116296)
        
        # Assigning a Call to a Name (line 670):
        
        # Assigning a Call to a Name (line 670):
        
        # Call to sum(...): (line 670)
        # Processing the call arguments (line 670)
        # Getting the type of 'arr' (line 670)
        arr_116299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 21), 'arr', False)
        # Processing the call keyword arguments (line 670)
        # Getting the type of 'axis' (line 670)
        axis_116300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 31), 'axis', False)
        keyword_116301 = axis_116300
        # Getting the type of 'dtype' (line 670)
        dtype_116302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 43), 'dtype', False)
        keyword_116303 = dtype_116302
        # Getting the type of 'out' (line 670)
        out_116304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 54), 'out', False)
        keyword_116305 = out_116304
        # Getting the type of 'keepdims' (line 670)
        keepdims_116306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 68), 'keepdims', False)
        keyword_116307 = keepdims_116306
        kwargs_116308 = {'dtype': keyword_116303, 'out': keyword_116305, 'keepdims': keyword_116307, 'axis': keyword_116301}
        # Getting the type of 'np' (line 670)
        np_116297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 14), 'np', False)
        # Obtaining the member 'sum' of a type (line 670)
        sum_116298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 14), np_116297, 'sum')
        # Calling sum(args, kwargs) (line 670)
        sum_call_result_116309 = invoke(stypy.reporting.localization.Localization(__file__, 670, 14), sum_116298, *[arr_116299], **kwargs_116308)
        
        # Assigning a type to the variable 'tot' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'tot', sum_call_result_116309)
        
        # Assigning a Call to a Name (line 671):
        
        # Assigning a Call to a Name (line 671):
        
        # Call to _divide_by_count(...): (line 671)
        # Processing the call arguments (line 671)
        # Getting the type of 'tot' (line 671)
        tot_116311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 31), 'tot', False)
        # Getting the type of 'cnt' (line 671)
        cnt_116312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 36), 'cnt', False)
        # Processing the call keyword arguments (line 671)
        # Getting the type of 'out' (line 671)
        out_116313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 45), 'out', False)
        keyword_116314 = out_116313
        kwargs_116315 = {'out': keyword_116314}
        # Getting the type of '_divide_by_count' (line 671)
        _divide_by_count_116310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 14), '_divide_by_count', False)
        # Calling _divide_by_count(args, kwargs) (line 671)
        _divide_by_count_call_result_116316 = invoke(stypy.reporting.localization.Localization(__file__, 671, 14), _divide_by_count_116310, *[tot_116311, cnt_116312], **kwargs_116315)
        
        # Assigning a type to the variable 'avg' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'avg', _divide_by_count_call_result_116316)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 667)
        exit___116317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 9), catch_warnings_call_result_116275, '__exit__')
        with_exit_116318 = invoke(stypy.reporting.localization.Localization(__file__, 667, 9), exit___116317, None, None, None)

    
    # Assigning a Compare to a Name (line 673):
    
    # Assigning a Compare to a Name (line 673):
    
    # Getting the type of 'cnt' (line 673)
    cnt_116319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 13), 'cnt')
    int_116320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 20), 'int')
    # Applying the binary operator '==' (line 673)
    result_eq_116321 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 13), '==', cnt_116319, int_116320)
    
    # Assigning a type to the variable 'isbad' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'isbad', result_eq_116321)
    
    
    # Call to any(...): (line 674)
    # Processing the call keyword arguments (line 674)
    kwargs_116324 = {}
    # Getting the type of 'isbad' (line 674)
    isbad_116322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 7), 'isbad', False)
    # Obtaining the member 'any' of a type (line 674)
    any_116323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 7), isbad_116322, 'any')
    # Calling any(args, kwargs) (line 674)
    any_call_result_116325 = invoke(stypy.reporting.localization.Localization(__file__, 674, 7), any_116323, *[], **kwargs_116324)
    
    # Testing the type of an if condition (line 674)
    if_condition_116326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 4), any_call_result_116325)
    # Assigning a type to the variable 'if_condition_116326' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'if_condition_116326', if_condition_116326)
    # SSA begins for if statement (line 674)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 675)
    # Processing the call arguments (line 675)
    str_116329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 22), 'str', 'Mean of empty slice')
    # Getting the type of 'RuntimeWarning' (line 675)
    RuntimeWarning_116330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 45), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 675)
    kwargs_116331 = {}
    # Getting the type of 'warnings' (line 675)
    warnings_116327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 675)
    warn_116328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 8), warnings_116327, 'warn')
    # Calling warn(args, kwargs) (line 675)
    warn_call_result_116332 = invoke(stypy.reporting.localization.Localization(__file__, 675, 8), warn_116328, *[str_116329, RuntimeWarning_116330], **kwargs_116331)
    
    # SSA join for if statement (line 674)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'avg' (line 678)
    avg_116333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 11), 'avg')
    # Assigning a type to the variable 'stypy_return_type' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'stypy_return_type', avg_116333)
    
    # ################# End of 'nanmean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanmean' in the type store
    # Getting the type of 'stypy_return_type' (line 586)
    stypy_return_type_116334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116334)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanmean'
    return stypy_return_type_116334

# Assigning a type to the variable 'nanmean' (line 586)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 0), 'nanmean', nanmean)

@norecursion
def _nanmedian1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 681)
    False_116335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 40), 'False')
    defaults = [False_116335]
    # Create a new context for function '_nanmedian1d'
    module_type_store = module_type_store.open_function_context('_nanmedian1d', 681, 0, False)
    
    # Passed parameters checking function
    _nanmedian1d.stypy_localization = localization
    _nanmedian1d.stypy_type_of_self = None
    _nanmedian1d.stypy_type_store = module_type_store
    _nanmedian1d.stypy_function_name = '_nanmedian1d'
    _nanmedian1d.stypy_param_names_list = ['arr1d', 'overwrite_input']
    _nanmedian1d.stypy_varargs_param_name = None
    _nanmedian1d.stypy_kwargs_param_name = None
    _nanmedian1d.stypy_call_defaults = defaults
    _nanmedian1d.stypy_call_varargs = varargs
    _nanmedian1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nanmedian1d', ['arr1d', 'overwrite_input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nanmedian1d', localization, ['arr1d', 'overwrite_input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nanmedian1d(...)' code ##################

    str_116336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, (-1)), 'str', '\n    Private function for rank 1 arrays. Compute the median ignoring NaNs.\n    See nanmedian for parameter usage\n    ')
    
    # Assigning a Call to a Name (line 686):
    
    # Assigning a Call to a Name (line 686):
    
    # Call to isnan(...): (line 686)
    # Processing the call arguments (line 686)
    # Getting the type of 'arr1d' (line 686)
    arr1d_116339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 17), 'arr1d', False)
    # Processing the call keyword arguments (line 686)
    kwargs_116340 = {}
    # Getting the type of 'np' (line 686)
    np_116337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'np', False)
    # Obtaining the member 'isnan' of a type (line 686)
    isnan_116338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 8), np_116337, 'isnan')
    # Calling isnan(args, kwargs) (line 686)
    isnan_call_result_116341 = invoke(stypy.reporting.localization.Localization(__file__, 686, 8), isnan_116338, *[arr1d_116339], **kwargs_116340)
    
    # Assigning a type to the variable 'c' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'c', isnan_call_result_116341)
    
    # Assigning a Subscript to a Name (line 687):
    
    # Assigning a Subscript to a Name (line 687):
    
    # Obtaining the type of the subscript
    int_116342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 20), 'int')
    
    # Call to where(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'c' (line 687)
    c_116345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 17), 'c', False)
    # Processing the call keyword arguments (line 687)
    kwargs_116346 = {}
    # Getting the type of 'np' (line 687)
    np_116343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'np', False)
    # Obtaining the member 'where' of a type (line 687)
    where_116344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), np_116343, 'where')
    # Calling where(args, kwargs) (line 687)
    where_call_result_116347 = invoke(stypy.reporting.localization.Localization(__file__, 687, 8), where_116344, *[c_116345], **kwargs_116346)
    
    # Obtaining the member '__getitem__' of a type (line 687)
    getitem___116348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), where_call_result_116347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 687)
    subscript_call_result_116349 = invoke(stypy.reporting.localization.Localization(__file__, 687, 8), getitem___116348, int_116342)
    
    # Assigning a type to the variable 's' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 's', subscript_call_result_116349)
    
    
    # Getting the type of 's' (line 688)
    s_116350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 7), 's')
    # Obtaining the member 'size' of a type (line 688)
    size_116351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 7), s_116350, 'size')
    # Getting the type of 'arr1d' (line 688)
    arr1d_116352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 17), 'arr1d')
    # Obtaining the member 'size' of a type (line 688)
    size_116353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 17), arr1d_116352, 'size')
    # Applying the binary operator '==' (line 688)
    result_eq_116354 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 7), '==', size_116351, size_116353)
    
    # Testing the type of an if condition (line 688)
    if_condition_116355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 688, 4), result_eq_116354)
    # Assigning a type to the variable 'if_condition_116355' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'if_condition_116355', if_condition_116355)
    # SSA begins for if statement (line 688)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 689)
    # Processing the call arguments (line 689)
    str_116358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 22), 'str', 'All-NaN slice encountered')
    # Getting the type of 'RuntimeWarning' (line 689)
    RuntimeWarning_116359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 51), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 689)
    kwargs_116360 = {}
    # Getting the type of 'warnings' (line 689)
    warnings_116356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 689)
    warn_116357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), warnings_116356, 'warn')
    # Calling warn(args, kwargs) (line 689)
    warn_call_result_116361 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), warn_116357, *[str_116358, RuntimeWarning_116359], **kwargs_116360)
    
    # Getting the type of 'np' (line 690)
    np_116362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'np')
    # Obtaining the member 'nan' of a type (line 690)
    nan_116363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 15), np_116362, 'nan')
    # Assigning a type to the variable 'stypy_return_type' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'stypy_return_type', nan_116363)
    # SSA branch for the else part of an if statement (line 688)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 's' (line 691)
    s_116364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 9), 's')
    # Obtaining the member 'size' of a type (line 691)
    size_116365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 9), s_116364, 'size')
    int_116366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 19), 'int')
    # Applying the binary operator '==' (line 691)
    result_eq_116367 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 9), '==', size_116365, int_116366)
    
    # Testing the type of an if condition (line 691)
    if_condition_116368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 691, 9), result_eq_116367)
    # Assigning a type to the variable 'if_condition_116368' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 9), 'if_condition_116368', if_condition_116368)
    # SSA begins for if statement (line 691)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to median(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'arr1d' (line 692)
    arr1d_116371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 25), 'arr1d', False)
    # Processing the call keyword arguments (line 692)
    # Getting the type of 'overwrite_input' (line 692)
    overwrite_input_116372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 48), 'overwrite_input', False)
    keyword_116373 = overwrite_input_116372
    kwargs_116374 = {'overwrite_input': keyword_116373}
    # Getting the type of 'np' (line 692)
    np_116369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 15), 'np', False)
    # Obtaining the member 'median' of a type (line 692)
    median_116370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 15), np_116369, 'median')
    # Calling median(args, kwargs) (line 692)
    median_call_result_116375 = invoke(stypy.reporting.localization.Localization(__file__, 692, 15), median_116370, *[arr1d_116371], **kwargs_116374)
    
    # Assigning a type to the variable 'stypy_return_type' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'stypy_return_type', median_call_result_116375)
    # SSA branch for the else part of an if statement (line 691)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'overwrite_input' (line 694)
    overwrite_input_116376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 11), 'overwrite_input')
    # Testing the type of an if condition (line 694)
    if_condition_116377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 8), overwrite_input_116376)
    # Assigning a type to the variable 'if_condition_116377' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'if_condition_116377', if_condition_116377)
    # SSA begins for if statement (line 694)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 695):
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'arr1d' (line 695)
    arr1d_116378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 16), 'arr1d')
    # Assigning a type to the variable 'x' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'x', arr1d_116378)
    # SSA branch for the else part of an if statement (line 694)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to copy(...): (line 697)
    # Processing the call keyword arguments (line 697)
    kwargs_116381 = {}
    # Getting the type of 'arr1d' (line 697)
    arr1d_116379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'arr1d', False)
    # Obtaining the member 'copy' of a type (line 697)
    copy_116380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 16), arr1d_116379, 'copy')
    # Calling copy(args, kwargs) (line 697)
    copy_call_result_116382 = invoke(stypy.reporting.localization.Localization(__file__, 697, 16), copy_116380, *[], **kwargs_116381)
    
    # Assigning a type to the variable 'x' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'x', copy_call_result_116382)
    # SSA join for if statement (line 694)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 699):
    
    # Assigning a Subscript to a Name (line 699):
    
    # Obtaining the type of the subscript
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 699)
    s_116383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 37), 's')
    # Obtaining the member 'size' of a type (line 699)
    size_116384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 37), s_116383, 'size')
    # Applying the 'usub' unary operator (line 699)
    result___neg___116385 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 36), 'usub', size_116384)
    
    slice_116386 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 34), result___neg___116385, None, None)
    # Getting the type of 'c' (line 699)
    c_116387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 34), 'c')
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___116388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 34), c_116387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_116389 = invoke(stypy.reporting.localization.Localization(__file__, 699, 34), getitem___116388, slice_116386)
    
    # Applying the '~' unary operator (line 699)
    result_inv_116390 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 33), '~', subscript_call_result_116389)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 699)
    s_116391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 24), 's')
    # Obtaining the member 'size' of a type (line 699)
    size_116392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 24), s_116391, 'size')
    # Applying the 'usub' unary operator (line 699)
    result___neg___116393 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 23), 'usub', size_116392)
    
    slice_116394 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 17), result___neg___116393, None, None)
    # Getting the type of 'arr1d' (line 699)
    arr1d_116395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 17), 'arr1d')
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___116396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 17), arr1d_116395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_116397 = invoke(stypy.reporting.localization.Localization(__file__, 699, 17), getitem___116396, slice_116394)
    
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___116398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 17), subscript_call_result_116397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_116399 = invoke(stypy.reporting.localization.Localization(__file__, 699, 17), getitem___116398, result_inv_116390)
    
    # Assigning a type to the variable 'enonan' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'enonan', subscript_call_result_116399)
    
    # Assigning a Name to a Subscript (line 701):
    
    # Assigning a Name to a Subscript (line 701):
    # Getting the type of 'enonan' (line 701)
    enonan_116400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 29), 'enonan')
    # Getting the type of 'x' (line 701)
    x_116401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'enonan' (line 701)
    enonan_116402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 13), 'enonan')
    # Obtaining the member 'size' of a type (line 701)
    size_116403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 13), enonan_116402, 'size')
    slice_116404 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 701, 10), None, size_116403, None)
    # Getting the type of 's' (line 701)
    s_116405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 10), 's')
    # Obtaining the member '__getitem__' of a type (line 701)
    getitem___116406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 10), s_116405, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 701)
    subscript_call_result_116407 = invoke(stypy.reporting.localization.Localization(__file__, 701, 10), getitem___116406, slice_116404)
    
    # Storing an element on a container (line 701)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 8), x_116401, (subscript_call_result_116407, enonan_116400))
    
    # Call to median(...): (line 703)
    # Processing the call arguments (line 703)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 703)
    s_116410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 29), 's', False)
    # Obtaining the member 'size' of a type (line 703)
    size_116411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 29), s_116410, 'size')
    # Applying the 'usub' unary operator (line 703)
    result___neg___116412 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 28), 'usub', size_116411)
    
    slice_116413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 703, 25), None, result___neg___116412, None)
    # Getting the type of 'x' (line 703)
    x_116414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 25), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 703)
    getitem___116415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 25), x_116414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 703)
    subscript_call_result_116416 = invoke(stypy.reporting.localization.Localization(__file__, 703, 25), getitem___116415, slice_116413)
    
    # Processing the call keyword arguments (line 703)
    # Getting the type of 'True' (line 703)
    True_116417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 54), 'True', False)
    keyword_116418 = True_116417
    kwargs_116419 = {'overwrite_input': keyword_116418}
    # Getting the type of 'np' (line 703)
    np_116408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 15), 'np', False)
    # Obtaining the member 'median' of a type (line 703)
    median_116409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 15), np_116408, 'median')
    # Calling median(args, kwargs) (line 703)
    median_call_result_116420 = invoke(stypy.reporting.localization.Localization(__file__, 703, 15), median_116409, *[subscript_call_result_116416], **kwargs_116419)
    
    # Assigning a type to the variable 'stypy_return_type' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'stypy_return_type', median_call_result_116420)
    # SSA join for if statement (line 691)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 688)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_nanmedian1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nanmedian1d' in the type store
    # Getting the type of 'stypy_return_type' (line 681)
    stypy_return_type_116421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nanmedian1d'
    return stypy_return_type_116421

# Assigning a type to the variable '_nanmedian1d' (line 681)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 0), '_nanmedian1d', _nanmedian1d)

@norecursion
def _nanmedian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 706)
    None_116422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 23), 'None')
    # Getting the type of 'None' (line 706)
    None_116423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 33), 'None')
    # Getting the type of 'False' (line 706)
    False_116424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 55), 'False')
    defaults = [None_116422, None_116423, False_116424]
    # Create a new context for function '_nanmedian'
    module_type_store = module_type_store.open_function_context('_nanmedian', 706, 0, False)
    
    # Passed parameters checking function
    _nanmedian.stypy_localization = localization
    _nanmedian.stypy_type_of_self = None
    _nanmedian.stypy_type_store = module_type_store
    _nanmedian.stypy_function_name = '_nanmedian'
    _nanmedian.stypy_param_names_list = ['a', 'axis', 'out', 'overwrite_input']
    _nanmedian.stypy_varargs_param_name = None
    _nanmedian.stypy_kwargs_param_name = None
    _nanmedian.stypy_call_defaults = defaults
    _nanmedian.stypy_call_varargs = varargs
    _nanmedian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nanmedian', ['a', 'axis', 'out', 'overwrite_input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nanmedian', localization, ['a', 'axis', 'out', 'overwrite_input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nanmedian(...)' code ##################

    str_116425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, (-1)), 'str', "\n    Private function that doesn't support extended axis or keepdims.\n    These methods are extended to this function using _ureduce\n    See nanmedian for parameter usage\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 713)
    axis_116426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 7), 'axis')
    # Getting the type of 'None' (line 713)
    None_116427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'None')
    # Applying the binary operator 'is' (line 713)
    result_is__116428 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 7), 'is', axis_116426, None_116427)
    
    
    # Getting the type of 'a' (line 713)
    a_116429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 23), 'a')
    # Obtaining the member 'ndim' of a type (line 713)
    ndim_116430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 23), a_116429, 'ndim')
    int_116431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 33), 'int')
    # Applying the binary operator '==' (line 713)
    result_eq_116432 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 23), '==', ndim_116430, int_116431)
    
    # Applying the binary operator 'or' (line 713)
    result_or_keyword_116433 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 7), 'or', result_is__116428, result_eq_116432)
    
    # Testing the type of an if condition (line 713)
    if_condition_116434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 713, 4), result_or_keyword_116433)
    # Assigning a type to the variable 'if_condition_116434' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'if_condition_116434', if_condition_116434)
    # SSA begins for if statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 714):
    
    # Assigning a Call to a Name (line 714):
    
    # Call to ravel(...): (line 714)
    # Processing the call keyword arguments (line 714)
    kwargs_116437 = {}
    # Getting the type of 'a' (line 714)
    a_116435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 15), 'a', False)
    # Obtaining the member 'ravel' of a type (line 714)
    ravel_116436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 15), a_116435, 'ravel')
    # Calling ravel(args, kwargs) (line 714)
    ravel_call_result_116438 = invoke(stypy.reporting.localization.Localization(__file__, 714, 15), ravel_116436, *[], **kwargs_116437)
    
    # Assigning a type to the variable 'part' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'part', ravel_call_result_116438)
    
    # Type idiom detected: calculating its left and rigth part (line 715)
    # Getting the type of 'out' (line 715)
    out_116439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 11), 'out')
    # Getting the type of 'None' (line 715)
    None_116440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 18), 'None')
    
    (may_be_116441, more_types_in_union_116442) = may_be_none(out_116439, None_116440)

    if may_be_116441:

        if more_types_in_union_116442:
            # Runtime conditional SSA (line 715)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to _nanmedian1d(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'part' (line 716)
        part_116444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 32), 'part', False)
        # Getting the type of 'overwrite_input' (line 716)
        overwrite_input_116445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 38), 'overwrite_input', False)
        # Processing the call keyword arguments (line 716)
        kwargs_116446 = {}
        # Getting the type of '_nanmedian1d' (line 716)
        _nanmedian1d_116443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 19), '_nanmedian1d', False)
        # Calling _nanmedian1d(args, kwargs) (line 716)
        _nanmedian1d_call_result_116447 = invoke(stypy.reporting.localization.Localization(__file__, 716, 19), _nanmedian1d_116443, *[part_116444, overwrite_input_116445], **kwargs_116446)
        
        # Assigning a type to the variable 'stypy_return_type' (line 716)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'stypy_return_type', _nanmedian1d_call_result_116447)

        if more_types_in_union_116442:
            # Runtime conditional SSA for else branch (line 715)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_116441) or more_types_in_union_116442):
        
        # Assigning a Call to a Subscript (line 718):
        
        # Assigning a Call to a Subscript (line 718):
        
        # Call to _nanmedian1d(...): (line 718)
        # Processing the call arguments (line 718)
        # Getting the type of 'part' (line 718)
        part_116449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 36), 'part', False)
        # Getting the type of 'overwrite_input' (line 718)
        overwrite_input_116450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 42), 'overwrite_input', False)
        # Processing the call keyword arguments (line 718)
        kwargs_116451 = {}
        # Getting the type of '_nanmedian1d' (line 718)
        _nanmedian1d_116448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 23), '_nanmedian1d', False)
        # Calling _nanmedian1d(args, kwargs) (line 718)
        _nanmedian1d_call_result_116452 = invoke(stypy.reporting.localization.Localization(__file__, 718, 23), _nanmedian1d_116448, *[part_116449, overwrite_input_116450], **kwargs_116451)
        
        # Getting the type of 'out' (line 718)
        out_116453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'out')
        Ellipsis_116454 = Ellipsis
        # Storing an element on a container (line 718)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 12), out_116453, (Ellipsis_116454, _nanmedian1d_call_result_116452))
        # Getting the type of 'out' (line 719)
        out_116455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 719)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'stypy_return_type', out_116455)

        if (may_be_116441 and more_types_in_union_116442):
            # SSA join for if statement (line 715)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 713)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 723)
    axis_116456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 19), 'axis')
    # Getting the type of 'a' (line 723)
    a_116457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'a')
    # Obtaining the member 'shape' of a type (line 723)
    shape_116458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 11), a_116457, 'shape')
    # Obtaining the member '__getitem__' of a type (line 723)
    getitem___116459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 11), shape_116458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 723)
    subscript_call_result_116460 = invoke(stypy.reporting.localization.Localization(__file__, 723, 11), getitem___116459, axis_116456)
    
    int_116461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 27), 'int')
    # Applying the binary operator '<' (line 723)
    result_lt_116462 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 11), '<', subscript_call_result_116460, int_116461)
    
    # Testing the type of an if condition (line 723)
    if_condition_116463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 723, 8), result_lt_116462)
    # Assigning a type to the variable 'if_condition_116463' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'if_condition_116463', if_condition_116463)
    # SSA begins for if statement (line 723)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _nanmedian_small(...): (line 724)
    # Processing the call arguments (line 724)
    # Getting the type of 'a' (line 724)
    a_116465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 36), 'a', False)
    # Getting the type of 'axis' (line 724)
    axis_116466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 39), 'axis', False)
    # Getting the type of 'out' (line 724)
    out_116467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 45), 'out', False)
    # Getting the type of 'overwrite_input' (line 724)
    overwrite_input_116468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 50), 'overwrite_input', False)
    # Processing the call keyword arguments (line 724)
    kwargs_116469 = {}
    # Getting the type of '_nanmedian_small' (line 724)
    _nanmedian_small_116464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 19), '_nanmedian_small', False)
    # Calling _nanmedian_small(args, kwargs) (line 724)
    _nanmedian_small_call_result_116470 = invoke(stypy.reporting.localization.Localization(__file__, 724, 19), _nanmedian_small_116464, *[a_116465, axis_116466, out_116467, overwrite_input_116468], **kwargs_116469)
    
    # Assigning a type to the variable 'stypy_return_type' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 12), 'stypy_return_type', _nanmedian_small_call_result_116470)
    # SSA join for if statement (line 723)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 725):
    
    # Assigning a Call to a Name (line 725):
    
    # Call to apply_along_axis(...): (line 725)
    # Processing the call arguments (line 725)
    # Getting the type of '_nanmedian1d' (line 725)
    _nanmedian1d_116473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 37), '_nanmedian1d', False)
    # Getting the type of 'axis' (line 725)
    axis_116474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 51), 'axis', False)
    # Getting the type of 'a' (line 725)
    a_116475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 57), 'a', False)
    # Getting the type of 'overwrite_input' (line 725)
    overwrite_input_116476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 60), 'overwrite_input', False)
    # Processing the call keyword arguments (line 725)
    kwargs_116477 = {}
    # Getting the type of 'np' (line 725)
    np_116471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 17), 'np', False)
    # Obtaining the member 'apply_along_axis' of a type (line 725)
    apply_along_axis_116472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 17), np_116471, 'apply_along_axis')
    # Calling apply_along_axis(args, kwargs) (line 725)
    apply_along_axis_call_result_116478 = invoke(stypy.reporting.localization.Localization(__file__, 725, 17), apply_along_axis_116472, *[_nanmedian1d_116473, axis_116474, a_116475, overwrite_input_116476], **kwargs_116477)
    
    # Assigning a type to the variable 'result' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'result', apply_along_axis_call_result_116478)
    
    # Type idiom detected: calculating its left and rigth part (line 726)
    # Getting the type of 'out' (line 726)
    out_116479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'out')
    # Getting the type of 'None' (line 726)
    None_116480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 22), 'None')
    
    (may_be_116481, more_types_in_union_116482) = may_not_be_none(out_116479, None_116480)

    if may_be_116481:

        if more_types_in_union_116482:
            # Runtime conditional SSA (line 726)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 727):
        
        # Assigning a Name to a Subscript (line 727):
        # Getting the type of 'result' (line 727)
        result_116483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 23), 'result')
        # Getting the type of 'out' (line 727)
        out_116484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 12), 'out')
        Ellipsis_116485 = Ellipsis
        # Storing an element on a container (line 727)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 727, 12), out_116484, (Ellipsis_116485, result_116483))

        if more_types_in_union_116482:
            # SSA join for if statement (line 726)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 728)
    result_116486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'stypy_return_type', result_116486)
    # SSA join for if statement (line 713)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_nanmedian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nanmedian' in the type store
    # Getting the type of 'stypy_return_type' (line 706)
    stypy_return_type_116487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116487)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nanmedian'
    return stypy_return_type_116487

# Assigning a type to the variable '_nanmedian' (line 706)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 0), '_nanmedian', _nanmedian)

@norecursion
def _nanmedian_small(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 730)
    None_116488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 29), 'None')
    # Getting the type of 'None' (line 730)
    None_116489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 39), 'None')
    # Getting the type of 'False' (line 730)
    False_116490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 61), 'False')
    defaults = [None_116488, None_116489, False_116490]
    # Create a new context for function '_nanmedian_small'
    module_type_store = module_type_store.open_function_context('_nanmedian_small', 730, 0, False)
    
    # Passed parameters checking function
    _nanmedian_small.stypy_localization = localization
    _nanmedian_small.stypy_type_of_self = None
    _nanmedian_small.stypy_type_store = module_type_store
    _nanmedian_small.stypy_function_name = '_nanmedian_small'
    _nanmedian_small.stypy_param_names_list = ['a', 'axis', 'out', 'overwrite_input']
    _nanmedian_small.stypy_varargs_param_name = None
    _nanmedian_small.stypy_kwargs_param_name = None
    _nanmedian_small.stypy_call_defaults = defaults
    _nanmedian_small.stypy_call_varargs = varargs
    _nanmedian_small.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nanmedian_small', ['a', 'axis', 'out', 'overwrite_input'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nanmedian_small', localization, ['a', 'axis', 'out', 'overwrite_input'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nanmedian_small(...)' code ##################

    str_116491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, (-1)), 'str', '\n    sort + indexing median, faster for small medians along multiple\n    dimensions due to the high overhead of apply_along_axis\n\n    see nanmedian for parameter usage\n    ')
    
    # Assigning a Call to a Name (line 737):
    
    # Assigning a Call to a Name (line 737):
    
    # Call to masked_array(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'a' (line 737)
    a_116495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 27), 'a', False)
    
    # Call to isnan(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'a' (line 737)
    a_116498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 39), 'a', False)
    # Processing the call keyword arguments (line 737)
    kwargs_116499 = {}
    # Getting the type of 'np' (line 737)
    np_116496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 30), 'np', False)
    # Obtaining the member 'isnan' of a type (line 737)
    isnan_116497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 30), np_116496, 'isnan')
    # Calling isnan(args, kwargs) (line 737)
    isnan_call_result_116500 = invoke(stypy.reporting.localization.Localization(__file__, 737, 30), isnan_116497, *[a_116498], **kwargs_116499)
    
    # Processing the call keyword arguments (line 737)
    kwargs_116501 = {}
    # Getting the type of 'np' (line 737)
    np_116492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'np', False)
    # Obtaining the member 'ma' of a type (line 737)
    ma_116493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 8), np_116492, 'ma')
    # Obtaining the member 'masked_array' of a type (line 737)
    masked_array_116494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 8), ma_116493, 'masked_array')
    # Calling masked_array(args, kwargs) (line 737)
    masked_array_call_result_116502 = invoke(stypy.reporting.localization.Localization(__file__, 737, 8), masked_array_116494, *[a_116495, isnan_call_result_116500], **kwargs_116501)
    
    # Assigning a type to the variable 'a' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'a', masked_array_call_result_116502)
    
    # Assigning a Call to a Name (line 738):
    
    # Assigning a Call to a Name (line 738):
    
    # Call to median(...): (line 738)
    # Processing the call arguments (line 738)
    # Getting the type of 'a' (line 738)
    a_116506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 21), 'a', False)
    # Processing the call keyword arguments (line 738)
    # Getting the type of 'axis' (line 738)
    axis_116507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 29), 'axis', False)
    keyword_116508 = axis_116507
    # Getting the type of 'overwrite_input' (line 738)
    overwrite_input_116509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 51), 'overwrite_input', False)
    keyword_116510 = overwrite_input_116509
    kwargs_116511 = {'overwrite_input': keyword_116510, 'axis': keyword_116508}
    # Getting the type of 'np' (line 738)
    np_116503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'np', False)
    # Obtaining the member 'ma' of a type (line 738)
    ma_116504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 8), np_116503, 'ma')
    # Obtaining the member 'median' of a type (line 738)
    median_116505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 8), ma_116504, 'median')
    # Calling median(args, kwargs) (line 738)
    median_call_result_116512 = invoke(stypy.reporting.localization.Localization(__file__, 738, 8), median_116505, *[a_116506], **kwargs_116511)
    
    # Assigning a type to the variable 'm' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'm', median_call_result_116512)
    
    
    # Call to range(...): (line 739)
    # Processing the call arguments (line 739)
    
    # Call to count_nonzero(...): (line 739)
    # Processing the call arguments (line 739)
    
    # Call to ravel(...): (line 739)
    # Processing the call keyword arguments (line 739)
    kwargs_116519 = {}
    # Getting the type of 'm' (line 739)
    m_116516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 36), 'm', False)
    # Obtaining the member 'mask' of a type (line 739)
    mask_116517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 36), m_116516, 'mask')
    # Obtaining the member 'ravel' of a type (line 739)
    ravel_116518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 36), mask_116517, 'ravel')
    # Calling ravel(args, kwargs) (line 739)
    ravel_call_result_116520 = invoke(stypy.reporting.localization.Localization(__file__, 739, 36), ravel_116518, *[], **kwargs_116519)
    
    # Processing the call keyword arguments (line 739)
    kwargs_116521 = {}
    # Getting the type of 'np' (line 739)
    np_116514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 19), 'np', False)
    # Obtaining the member 'count_nonzero' of a type (line 739)
    count_nonzero_116515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 19), np_116514, 'count_nonzero')
    # Calling count_nonzero(args, kwargs) (line 739)
    count_nonzero_call_result_116522 = invoke(stypy.reporting.localization.Localization(__file__, 739, 19), count_nonzero_116515, *[ravel_call_result_116520], **kwargs_116521)
    
    # Processing the call keyword arguments (line 739)
    kwargs_116523 = {}
    # Getting the type of 'range' (line 739)
    range_116513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 13), 'range', False)
    # Calling range(args, kwargs) (line 739)
    range_call_result_116524 = invoke(stypy.reporting.localization.Localization(__file__, 739, 13), range_116513, *[count_nonzero_call_result_116522], **kwargs_116523)
    
    # Testing the type of a for loop iterable (line 739)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 739, 4), range_call_result_116524)
    # Getting the type of the for loop variable (line 739)
    for_loop_var_116525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 739, 4), range_call_result_116524)
    # Assigning a type to the variable 'i' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'i', for_loop_var_116525)
    # SSA begins for a for statement (line 739)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to warn(...): (line 740)
    # Processing the call arguments (line 740)
    str_116528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 22), 'str', 'All-NaN slice encountered')
    # Getting the type of 'RuntimeWarning' (line 740)
    RuntimeWarning_116529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 51), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 740)
    kwargs_116530 = {}
    # Getting the type of 'warnings' (line 740)
    warnings_116526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 740)
    warn_116527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 8), warnings_116526, 'warn')
    # Calling warn(args, kwargs) (line 740)
    warn_call_result_116531 = invoke(stypy.reporting.localization.Localization(__file__, 740, 8), warn_116527, *[str_116528, RuntimeWarning_116529], **kwargs_116530)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 741)
    # Getting the type of 'out' (line 741)
    out_116532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'out')
    # Getting the type of 'None' (line 741)
    None_116533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 18), 'None')
    
    (may_be_116534, more_types_in_union_116535) = may_not_be_none(out_116532, None_116533)

    if may_be_116534:

        if more_types_in_union_116535:
            # Runtime conditional SSA (line 741)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 742):
        
        # Assigning a Call to a Subscript (line 742):
        
        # Call to filled(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'np' (line 742)
        np_116538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 28), 'np', False)
        # Obtaining the member 'nan' of a type (line 742)
        nan_116539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 28), np_116538, 'nan')
        # Processing the call keyword arguments (line 742)
        kwargs_116540 = {}
        # Getting the type of 'm' (line 742)
        m_116536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 19), 'm', False)
        # Obtaining the member 'filled' of a type (line 742)
        filled_116537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 19), m_116536, 'filled')
        # Calling filled(args, kwargs) (line 742)
        filled_call_result_116541 = invoke(stypy.reporting.localization.Localization(__file__, 742, 19), filled_116537, *[nan_116539], **kwargs_116540)
        
        # Getting the type of 'out' (line 742)
        out_116542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'out')
        Ellipsis_116543 = Ellipsis
        # Storing an element on a container (line 742)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 8), out_116542, (Ellipsis_116543, filled_call_result_116541))
        # Getting the type of 'out' (line 743)
        out_116544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'stypy_return_type', out_116544)

        if more_types_in_union_116535:
            # SSA join for if statement (line 741)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to filled(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'np' (line 744)
    np_116547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 20), 'np', False)
    # Obtaining the member 'nan' of a type (line 744)
    nan_116548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 20), np_116547, 'nan')
    # Processing the call keyword arguments (line 744)
    kwargs_116549 = {}
    # Getting the type of 'm' (line 744)
    m_116545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 11), 'm', False)
    # Obtaining the member 'filled' of a type (line 744)
    filled_116546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 11), m_116545, 'filled')
    # Calling filled(args, kwargs) (line 744)
    filled_call_result_116550 = invoke(stypy.reporting.localization.Localization(__file__, 744, 11), filled_116546, *[nan_116548], **kwargs_116549)
    
    # Assigning a type to the variable 'stypy_return_type' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type', filled_call_result_116550)
    
    # ################# End of '_nanmedian_small(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nanmedian_small' in the type store
    # Getting the type of 'stypy_return_type' (line 730)
    stypy_return_type_116551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nanmedian_small'
    return stypy_return_type_116551

# Assigning a type to the variable '_nanmedian_small' (line 730)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 0), '_nanmedian_small', _nanmedian_small)

@norecursion
def nanmedian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 746)
    None_116552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 22), 'None')
    # Getting the type of 'None' (line 746)
    None_116553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 32), 'None')
    # Getting the type of 'False' (line 746)
    False_116554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 54), 'False')
    # Getting the type of 'False' (line 746)
    False_116555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 70), 'False')
    defaults = [None_116552, None_116553, False_116554, False_116555]
    # Create a new context for function 'nanmedian'
    module_type_store = module_type_store.open_function_context('nanmedian', 746, 0, False)
    
    # Passed parameters checking function
    nanmedian.stypy_localization = localization
    nanmedian.stypy_type_of_self = None
    nanmedian.stypy_type_store = module_type_store
    nanmedian.stypy_function_name = 'nanmedian'
    nanmedian.stypy_param_names_list = ['a', 'axis', 'out', 'overwrite_input', 'keepdims']
    nanmedian.stypy_varargs_param_name = None
    nanmedian.stypy_kwargs_param_name = None
    nanmedian.stypy_call_defaults = defaults
    nanmedian.stypy_call_varargs = varargs
    nanmedian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanmedian', ['a', 'axis', 'out', 'overwrite_input', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanmedian', localization, ['a', 'axis', 'out', 'overwrite_input', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanmedian(...)' code ##################

    str_116556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, (-1)), 'str', '\n    Compute the median along the specified axis, while ignoring NaNs.\n\n    Returns the median of the array elements.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    axis : {int, sequence of int, None}, optional\n        Axis or axes along which the medians are computed. The default\n        is to compute the median along a flattened version of the array.\n        A sequence of axes is supported since version 1.9.0.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output,\n        but the type (of the output) will be cast if necessary.\n    overwrite_input : bool, optional\n       If True, then allow use of memory of input array `a` for\n       calculations. The input array will be modified by the call to\n       `median`. This will save memory when you do not need to preserve\n       the contents of the input array. Treat the input as undefined,\n       but it will probably be fully or partially sorted. Default is\n       False. If `overwrite_input` is ``True`` and `a` is not already an\n       `ndarray`, an error will be raised.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in\n        the result as dimensions with size one. With this option, the\n        result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    median : ndarray\n        A new array holding the result. If the input contains integers\n        or floats smaller than ``float64``, then the output data-type is\n        ``np.float64``.  Otherwise, the data-type of the output is the\n        same as that of the input. If `out` is specified, that array is\n        returned instead.\n\n    See Also\n    --------\n    mean, median, percentile\n\n    Notes\n    -----\n    Given a vector ``V`` of length ``N``, the median of ``V`` is the\n    middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,\n    ``V_sorted[(N-1)/2]``, when ``N`` is odd and the average of the two\n    middle values of ``V_sorted`` when ``N`` is even.\n\n    Examples\n    --------\n    >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])\n    >>> a[0, 1] = np.nan\n    >>> a\n    array([[ 10.,  nan,   4.],\n       [  3.,   2.,   1.]])\n    >>> np.median(a)\n    nan\n    >>> np.nanmedian(a)\n    3.0\n    >>> np.nanmedian(a, axis=0)\n    array([ 6.5,  2.,  2.5])\n    >>> np.median(a, axis=1)\n    array([ 7.,  2.])\n    >>> b = a.copy()\n    >>> np.nanmedian(b, axis=1, overwrite_input=True)\n    array([ 7.,  2.])\n    >>> assert not np.all(a==b)\n    >>> b = a.copy()\n    >>> np.nanmedian(b, axis=None, overwrite_input=True)\n    3.0\n    >>> assert not np.all(a==b)\n\n    ')
    
    # Assigning a Call to a Name (line 824):
    
    # Assigning a Call to a Name (line 824):
    
    # Call to asanyarray(...): (line 824)
    # Processing the call arguments (line 824)
    # Getting the type of 'a' (line 824)
    a_116559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 22), 'a', False)
    # Processing the call keyword arguments (line 824)
    kwargs_116560 = {}
    # Getting the type of 'np' (line 824)
    np_116557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 824)
    asanyarray_116558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 8), np_116557, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 824)
    asanyarray_call_result_116561 = invoke(stypy.reporting.localization.Localization(__file__, 824, 8), asanyarray_116558, *[a_116559], **kwargs_116560)
    
    # Assigning a type to the variable 'a' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'a', asanyarray_call_result_116561)
    
    
    # Getting the type of 'a' (line 827)
    a_116562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 7), 'a')
    # Obtaining the member 'size' of a type (line 827)
    size_116563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 7), a_116562, 'size')
    int_116564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 17), 'int')
    # Applying the binary operator '==' (line 827)
    result_eq_116565 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 7), '==', size_116563, int_116564)
    
    # Testing the type of an if condition (line 827)
    if_condition_116566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 4), result_eq_116565)
    # Assigning a type to the variable 'if_condition_116566' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'if_condition_116566', if_condition_116566)
    # SSA begins for if statement (line 827)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to nanmean(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'a' (line 828)
    a_116569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 26), 'a', False)
    # Getting the type of 'axis' (line 828)
    axis_116570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 29), 'axis', False)
    # Processing the call keyword arguments (line 828)
    # Getting the type of 'out' (line 828)
    out_116571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 39), 'out', False)
    keyword_116572 = out_116571
    # Getting the type of 'keepdims' (line 828)
    keepdims_116573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 53), 'keepdims', False)
    keyword_116574 = keepdims_116573
    kwargs_116575 = {'keepdims': keyword_116574, 'out': keyword_116572}
    # Getting the type of 'np' (line 828)
    np_116567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 15), 'np', False)
    # Obtaining the member 'nanmean' of a type (line 828)
    nanmean_116568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 15), np_116567, 'nanmean')
    # Calling nanmean(args, kwargs) (line 828)
    nanmean_call_result_116576 = invoke(stypy.reporting.localization.Localization(__file__, 828, 15), nanmean_116568, *[a_116569, axis_116570], **kwargs_116575)
    
    # Assigning a type to the variable 'stypy_return_type' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'stypy_return_type', nanmean_call_result_116576)
    # SSA join for if statement (line 827)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 830):
    
    # Assigning a Call to a Name:
    
    # Call to _ureduce(...): (line 830)
    # Processing the call arguments (line 830)
    # Getting the type of 'a' (line 830)
    a_116578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 20), 'a', False)
    # Processing the call keyword arguments (line 830)
    # Getting the type of '_nanmedian' (line 830)
    _nanmedian_116579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 28), '_nanmedian', False)
    keyword_116580 = _nanmedian_116579
    # Getting the type of 'axis' (line 830)
    axis_116581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 45), 'axis', False)
    keyword_116582 = axis_116581
    # Getting the type of 'out' (line 830)
    out_116583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 55), 'out', False)
    keyword_116584 = out_116583
    # Getting the type of 'overwrite_input' (line 831)
    overwrite_input_116585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 36), 'overwrite_input', False)
    keyword_116586 = overwrite_input_116585
    kwargs_116587 = {'overwrite_input': keyword_116586, 'out': keyword_116584, 'func': keyword_116580, 'axis': keyword_116582}
    # Getting the type of '_ureduce' (line 830)
    _ureduce_116577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 11), '_ureduce', False)
    # Calling _ureduce(args, kwargs) (line 830)
    _ureduce_call_result_116588 = invoke(stypy.reporting.localization.Localization(__file__, 830, 11), _ureduce_116577, *[a_116578], **kwargs_116587)
    
    # Assigning a type to the variable 'call_assignment_115618' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115618', _ureduce_call_result_116588)
    
    # Assigning a Call to a Name (line 830):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116592 = {}
    # Getting the type of 'call_assignment_115618' (line 830)
    call_assignment_115618_116589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115618', False)
    # Obtaining the member '__getitem__' of a type (line 830)
    getitem___116590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 4), call_assignment_115618_116589, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116593 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116590, *[int_116591], **kwargs_116592)
    
    # Assigning a type to the variable 'call_assignment_115619' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115619', getitem___call_result_116593)
    
    # Assigning a Name to a Name (line 830):
    # Getting the type of 'call_assignment_115619' (line 830)
    call_assignment_115619_116594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115619')
    # Assigning a type to the variable 'r' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'r', call_assignment_115619_116594)
    
    # Assigning a Call to a Name (line 830):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116598 = {}
    # Getting the type of 'call_assignment_115618' (line 830)
    call_assignment_115618_116595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115618', False)
    # Obtaining the member '__getitem__' of a type (line 830)
    getitem___116596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 4), call_assignment_115618_116595, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116599 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116596, *[int_116597], **kwargs_116598)
    
    # Assigning a type to the variable 'call_assignment_115620' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115620', getitem___call_result_116599)
    
    # Assigning a Name to a Name (line 830):
    # Getting the type of 'call_assignment_115620' (line 830)
    call_assignment_115620_116600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'call_assignment_115620')
    # Assigning a type to the variable 'k' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 7), 'k', call_assignment_115620_116600)
    
    # Getting the type of 'keepdims' (line 832)
    keepdims_116601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 7), 'keepdims')
    # Testing the type of an if condition (line 832)
    if_condition_116602 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 4), keepdims_116601)
    # Assigning a type to the variable 'if_condition_116602' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'if_condition_116602', if_condition_116602)
    # SSA begins for if statement (line 832)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to reshape(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'k' (line 833)
    k_116605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 25), 'k', False)
    # Processing the call keyword arguments (line 833)
    kwargs_116606 = {}
    # Getting the type of 'r' (line 833)
    r_116603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 15), 'r', False)
    # Obtaining the member 'reshape' of a type (line 833)
    reshape_116604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 15), r_116603, 'reshape')
    # Calling reshape(args, kwargs) (line 833)
    reshape_call_result_116607 = invoke(stypy.reporting.localization.Localization(__file__, 833, 15), reshape_116604, *[k_116605], **kwargs_116606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'stypy_return_type', reshape_call_result_116607)
    # SSA branch for the else part of an if statement (line 832)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'r' (line 835)
    r_116608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'stypy_return_type', r_116608)
    # SSA join for if statement (line 832)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'nanmedian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanmedian' in the type store
    # Getting the type of 'stypy_return_type' (line 746)
    stypy_return_type_116609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116609)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanmedian'
    return stypy_return_type_116609

# Assigning a type to the variable 'nanmedian' (line 746)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 0), 'nanmedian', nanmedian)

@norecursion
def nanpercentile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 838)
    None_116610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 29), 'None')
    # Getting the type of 'None' (line 838)
    None_116611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 39), 'None')
    # Getting the type of 'False' (line 838)
    False_116612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 61), 'False')
    str_116613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 32), 'str', 'linear')
    # Getting the type of 'False' (line 839)
    False_116614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 51), 'False')
    defaults = [None_116610, None_116611, False_116612, str_116613, False_116614]
    # Create a new context for function 'nanpercentile'
    module_type_store = module_type_store.open_function_context('nanpercentile', 838, 0, False)
    
    # Passed parameters checking function
    nanpercentile.stypy_localization = localization
    nanpercentile.stypy_type_of_self = None
    nanpercentile.stypy_type_store = module_type_store
    nanpercentile.stypy_function_name = 'nanpercentile'
    nanpercentile.stypy_param_names_list = ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims']
    nanpercentile.stypy_varargs_param_name = None
    nanpercentile.stypy_kwargs_param_name = None
    nanpercentile.stypy_call_defaults = defaults
    nanpercentile.stypy_call_varargs = varargs
    nanpercentile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanpercentile', ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanpercentile', localization, ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanpercentile(...)' code ##################

    str_116615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, (-1)), 'str', "\n    Compute the qth percentile of the data along the specified axis,\n    while ignoring nan values.\n\n    Returns the qth percentile(s) of the array elements.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    a : array_like\n        Input array or object that can be converted to an array.\n    q : float in range of [0,100] (or sequence of floats)\n        Percentile to compute, which must be between 0 and 100\n        inclusive.\n    axis : {int, sequence of int, None}, optional\n        Axis or axes along which the percentiles are computed. The\n        default is to compute the percentile(s) along a flattened\n        version of the array. A sequence of axes is supported since\n        version 1.9.0.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must\n        have the same shape and buffer length as the expected output,\n        but the type (of the output) will be cast if necessary.\n    overwrite_input : bool, optional\n        If True, then allow use of memory of input array `a` for\n        calculations. The input array will be modified by the call to\n        `percentile`. This will save memory when you do not need to\n        preserve the contents of the input array. In this case you\n        should not make any assumptions about the contents of the input\n        `a` after this function completes -- treat it as undefined.\n        Default is False. If `a` is not already an array, this parameter\n        will have no effect as `a` will be converted to an array\n        internally regardless of the value of this parameter.\n    interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}\n        This optional parameter specifies the interpolation method to\n        use when the desired quantile lies between two data points\n        ``i < j``:\n            * linear: ``i + (j - i) * fraction``, where ``fraction`` is\n              the fractional part of the index surrounded by ``i`` and\n              ``j``.\n            * lower: ``i``.\n            * higher: ``j``.\n            * nearest: ``i`` or ``j``, whichever is nearest.\n            * midpoint: ``(i + j) / 2``.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left in\n        the result as dimensions with size one. With this option, the\n        result will broadcast correctly against the original array `a`.\n\n    Returns\n    -------\n    percentile : scalar or ndarray\n        If `q` is a single percentile and `axis=None`, then the result\n        is a scalar. If multiple percentiles are given, first axis of\n        the result corresponds to the percentiles. The other axes are\n        the axes that remain after the reduction of `a`. If the input \n        contains integers or floats smaller than ``float64``, the output\n        data-type is ``float64``. Otherwise, the output data-type is the\n        same as that of the input. If `out` is specified, that array is\n        returned instead.\n\n    See Also\n    --------\n    nanmean, nanmedian, percentile, median, mean\n\n    Notes\n    -----\n    Given a vector ``V`` of length ``N``, the ``q``-th percentile of\n    ``V`` is the value ``q/100`` of the way from the mimumum to the\n    maximum in in a sorted copy of ``V``. The values and distances of\n    the two nearest neighbors as well as the `interpolation` parameter\n    will determine the percentile if the normalized ranking does not\n    match the location of ``q`` exactly. This function is the same as\n    the median if ``q=50``, the same as the minimum if ``q=0`` and the\n    same as the maximum if ``q=100``.\n\n    Examples\n    --------\n    >>> a = np.array([[10., 7., 4.], [3., 2., 1.]])\n    >>> a[0][1] = np.nan\n    >>> a\n    array([[ 10.,  nan,   4.],\n       [  3.,   2.,   1.]])\n    >>> np.percentile(a, 50)\n    nan\n    >>> np.nanpercentile(a, 50)\n    3.5\n    >>> np.nanpercentile(a, 50, axis=0)\n    array([ 6.5,  2.,   2.5])\n    >>> np.nanpercentile(a, 50, axis=1, keepdims=True)\n    array([[ 7.],\n           [ 2.]])\n    >>> m = np.nanpercentile(a, 50, axis=0)\n    >>> out = np.zeros_like(m)\n    >>> np.nanpercentile(a, 50, axis=0, out=out)\n    array([ 6.5,  2.,   2.5])\n    >>> m\n    array([ 6.5,  2. ,  2.5])\n\n    >>> b = a.copy()\n    >>> np.nanpercentile(b, 50, axis=1, overwrite_input=True)\n    array([  7.,  2.])\n    >>> assert not np.all(a==b)\n\n    ")
    
    # Assigning a Call to a Name (line 947):
    
    # Assigning a Call to a Name (line 947):
    
    # Call to asanyarray(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'a' (line 947)
    a_116618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 22), 'a', False)
    # Processing the call keyword arguments (line 947)
    kwargs_116619 = {}
    # Getting the type of 'np' (line 947)
    np_116616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 947)
    asanyarray_116617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 8), np_116616, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 947)
    asanyarray_call_result_116620 = invoke(stypy.reporting.localization.Localization(__file__, 947, 8), asanyarray_116617, *[a_116618], **kwargs_116619)
    
    # Assigning a type to the variable 'a' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'a', asanyarray_call_result_116620)
    
    # Assigning a Call to a Name (line 948):
    
    # Assigning a Call to a Name (line 948):
    
    # Call to asanyarray(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'q' (line 948)
    q_116623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 22), 'q', False)
    # Processing the call keyword arguments (line 948)
    kwargs_116624 = {}
    # Getting the type of 'np' (line 948)
    np_116621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 948)
    asanyarray_116622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 8), np_116621, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 948)
    asanyarray_call_result_116625 = invoke(stypy.reporting.localization.Localization(__file__, 948, 8), asanyarray_116622, *[q_116623], **kwargs_116624)
    
    # Assigning a type to the variable 'q' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'q', asanyarray_call_result_116625)
    
    
    # Getting the type of 'a' (line 951)
    a_116626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 7), 'a')
    # Obtaining the member 'size' of a type (line 951)
    size_116627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 7), a_116626, 'size')
    int_116628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 17), 'int')
    # Applying the binary operator '==' (line 951)
    result_eq_116629 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 7), '==', size_116627, int_116628)
    
    # Testing the type of an if condition (line 951)
    if_condition_116630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 951, 4), result_eq_116629)
    # Assigning a type to the variable 'if_condition_116630' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'if_condition_116630', if_condition_116630)
    # SSA begins for if statement (line 951)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to nanmean(...): (line 952)
    # Processing the call arguments (line 952)
    # Getting the type of 'a' (line 952)
    a_116633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 26), 'a', False)
    # Getting the type of 'axis' (line 952)
    axis_116634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 29), 'axis', False)
    # Processing the call keyword arguments (line 952)
    # Getting the type of 'out' (line 952)
    out_116635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 39), 'out', False)
    keyword_116636 = out_116635
    # Getting the type of 'keepdims' (line 952)
    keepdims_116637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 53), 'keepdims', False)
    keyword_116638 = keepdims_116637
    kwargs_116639 = {'keepdims': keyword_116638, 'out': keyword_116636}
    # Getting the type of 'np' (line 952)
    np_116631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 15), 'np', False)
    # Obtaining the member 'nanmean' of a type (line 952)
    nanmean_116632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 15), np_116631, 'nanmean')
    # Calling nanmean(args, kwargs) (line 952)
    nanmean_call_result_116640 = invoke(stypy.reporting.localization.Localization(__file__, 952, 15), nanmean_116632, *[a_116633, axis_116634], **kwargs_116639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 8), 'stypy_return_type', nanmean_call_result_116640)
    # SSA join for if statement (line 951)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 954):
    
    # Assigning a Call to a Name:
    
    # Call to _ureduce(...): (line 954)
    # Processing the call arguments (line 954)
    # Getting the type of 'a' (line 954)
    a_116642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 20), 'a', False)
    # Processing the call keyword arguments (line 954)
    # Getting the type of '_nanpercentile' (line 954)
    _nanpercentile_116643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 28), '_nanpercentile', False)
    keyword_116644 = _nanpercentile_116643
    # Getting the type of 'q' (line 954)
    q_116645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 46), 'q', False)
    keyword_116646 = q_116645
    # Getting the type of 'axis' (line 954)
    axis_116647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 54), 'axis', False)
    keyword_116648 = axis_116647
    # Getting the type of 'out' (line 954)
    out_116649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 64), 'out', False)
    keyword_116650 = out_116649
    # Getting the type of 'overwrite_input' (line 955)
    overwrite_input_116651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 36), 'overwrite_input', False)
    keyword_116652 = overwrite_input_116651
    # Getting the type of 'interpolation' (line 956)
    interpolation_116653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 34), 'interpolation', False)
    keyword_116654 = interpolation_116653
    kwargs_116655 = {'q': keyword_116646, 'axis': keyword_116648, 'func': keyword_116644, 'overwrite_input': keyword_116652, 'interpolation': keyword_116654, 'out': keyword_116650}
    # Getting the type of '_ureduce' (line 954)
    _ureduce_116641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 11), '_ureduce', False)
    # Calling _ureduce(args, kwargs) (line 954)
    _ureduce_call_result_116656 = invoke(stypy.reporting.localization.Localization(__file__, 954, 11), _ureduce_116641, *[a_116642], **kwargs_116655)
    
    # Assigning a type to the variable 'call_assignment_115621' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115621', _ureduce_call_result_116656)
    
    # Assigning a Call to a Name (line 954):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116660 = {}
    # Getting the type of 'call_assignment_115621' (line 954)
    call_assignment_115621_116657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115621', False)
    # Obtaining the member '__getitem__' of a type (line 954)
    getitem___116658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 4), call_assignment_115621_116657, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116661 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116658, *[int_116659], **kwargs_116660)
    
    # Assigning a type to the variable 'call_assignment_115622' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115622', getitem___call_result_116661)
    
    # Assigning a Name to a Name (line 954):
    # Getting the type of 'call_assignment_115622' (line 954)
    call_assignment_115622_116662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115622')
    # Assigning a type to the variable 'r' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'r', call_assignment_115622_116662)
    
    # Assigning a Call to a Name (line 954):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116666 = {}
    # Getting the type of 'call_assignment_115621' (line 954)
    call_assignment_115621_116663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115621', False)
    # Obtaining the member '__getitem__' of a type (line 954)
    getitem___116664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 4), call_assignment_115621_116663, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116667 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116664, *[int_116665], **kwargs_116666)
    
    # Assigning a type to the variable 'call_assignment_115623' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115623', getitem___call_result_116667)
    
    # Assigning a Name to a Name (line 954):
    # Getting the type of 'call_assignment_115623' (line 954)
    call_assignment_115623_116668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'call_assignment_115623')
    # Assigning a type to the variable 'k' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 7), 'k', call_assignment_115623_116668)
    
    # Getting the type of 'keepdims' (line 957)
    keepdims_116669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 7), 'keepdims')
    # Testing the type of an if condition (line 957)
    if_condition_116670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 957, 4), keepdims_116669)
    # Assigning a type to the variable 'if_condition_116670' (line 957)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 4), 'if_condition_116670', if_condition_116670)
    # SSA begins for if statement (line 957)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'q' (line 958)
    q_116671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 11), 'q')
    # Obtaining the member 'ndim' of a type (line 958)
    ndim_116672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 11), q_116671, 'ndim')
    int_116673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, 21), 'int')
    # Applying the binary operator '==' (line 958)
    result_eq_116674 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 11), '==', ndim_116672, int_116673)
    
    # Testing the type of an if condition (line 958)
    if_condition_116675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 958, 8), result_eq_116674)
    # Assigning a type to the variable 'if_condition_116675' (line 958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 8), 'if_condition_116675', if_condition_116675)
    # SSA begins for if statement (line 958)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to reshape(...): (line 959)
    # Processing the call arguments (line 959)
    # Getting the type of 'k' (line 959)
    k_116678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 29), 'k', False)
    # Processing the call keyword arguments (line 959)
    kwargs_116679 = {}
    # Getting the type of 'r' (line 959)
    r_116676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 19), 'r', False)
    # Obtaining the member 'reshape' of a type (line 959)
    reshape_116677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 959, 19), r_116676, 'reshape')
    # Calling reshape(args, kwargs) (line 959)
    reshape_call_result_116680 = invoke(stypy.reporting.localization.Localization(__file__, 959, 19), reshape_116677, *[k_116678], **kwargs_116679)
    
    # Assigning a type to the variable 'stypy_return_type' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 12), 'stypy_return_type', reshape_call_result_116680)
    # SSA branch for the else part of an if statement (line 958)
    module_type_store.open_ssa_branch('else')
    
    # Call to reshape(...): (line 961)
    # Processing the call arguments (line 961)
    
    # Obtaining an instance of the builtin type 'list' (line 961)
    list_116683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 961)
    # Adding element type (line 961)
    
    # Call to len(...): (line 961)
    # Processing the call arguments (line 961)
    # Getting the type of 'q' (line 961)
    q_116685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 34), 'q', False)
    # Processing the call keyword arguments (line 961)
    kwargs_116686 = {}
    # Getting the type of 'len' (line 961)
    len_116684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 30), 'len', False)
    # Calling len(args, kwargs) (line 961)
    len_call_result_116687 = invoke(stypy.reporting.localization.Localization(__file__, 961, 30), len_116684, *[q_116685], **kwargs_116686)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 961, 29), list_116683, len_call_result_116687)
    
    # Getting the type of 'k' (line 961)
    k_116688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 40), 'k', False)
    # Applying the binary operator '+' (line 961)
    result_add_116689 = python_operator(stypy.reporting.localization.Localization(__file__, 961, 29), '+', list_116683, k_116688)
    
    # Processing the call keyword arguments (line 961)
    kwargs_116690 = {}
    # Getting the type of 'r' (line 961)
    r_116681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 19), 'r', False)
    # Obtaining the member 'reshape' of a type (line 961)
    reshape_116682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 19), r_116681, 'reshape')
    # Calling reshape(args, kwargs) (line 961)
    reshape_call_result_116691 = invoke(stypy.reporting.localization.Localization(__file__, 961, 19), reshape_116682, *[result_add_116689], **kwargs_116690)
    
    # Assigning a type to the variable 'stypy_return_type' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'stypy_return_type', reshape_call_result_116691)
    # SSA join for if statement (line 958)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 957)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'r' (line 963)
    r_116692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'stypy_return_type', r_116692)
    # SSA join for if statement (line 957)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'nanpercentile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanpercentile' in the type store
    # Getting the type of 'stypy_return_type' (line 838)
    stypy_return_type_116693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanpercentile'
    return stypy_return_type_116693

# Assigning a type to the variable 'nanpercentile' (line 838)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 0), 'nanpercentile', nanpercentile)

@norecursion
def _nanpercentile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 966)
    None_116694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 30), 'None')
    # Getting the type of 'None' (line 966)
    None_116695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 40), 'None')
    # Getting the type of 'False' (line 966)
    False_116696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 62), 'False')
    str_116697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 967, 33), 'str', 'linear')
    # Getting the type of 'False' (line 967)
    False_116698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 52), 'False')
    defaults = [None_116694, None_116695, False_116696, str_116697, False_116698]
    # Create a new context for function '_nanpercentile'
    module_type_store = module_type_store.open_function_context('_nanpercentile', 966, 0, False)
    
    # Passed parameters checking function
    _nanpercentile.stypy_localization = localization
    _nanpercentile.stypy_type_of_self = None
    _nanpercentile.stypy_type_store = module_type_store
    _nanpercentile.stypy_function_name = '_nanpercentile'
    _nanpercentile.stypy_param_names_list = ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims']
    _nanpercentile.stypy_varargs_param_name = None
    _nanpercentile.stypy_kwargs_param_name = None
    _nanpercentile.stypy_call_defaults = defaults
    _nanpercentile.stypy_call_varargs = varargs
    _nanpercentile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nanpercentile', ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nanpercentile', localization, ['a', 'q', 'axis', 'out', 'overwrite_input', 'interpolation', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nanpercentile(...)' code ##################

    str_116699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, (-1)), 'str', "\n    Private function that doesn't support extended axis or keepdims.\n    These methods are extended to this function using _ureduce\n    See nanpercentile for parameter usage\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 974)
    # Getting the type of 'axis' (line 974)
    axis_116700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 7), 'axis')
    # Getting the type of 'None' (line 974)
    None_116701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 15), 'None')
    
    (may_be_116702, more_types_in_union_116703) = may_be_none(axis_116700, None_116701)

    if may_be_116702:

        if more_types_in_union_116703:
            # Runtime conditional SSA (line 974)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 975):
        
        # Assigning a Call to a Name (line 975):
        
        # Call to ravel(...): (line 975)
        # Processing the call keyword arguments (line 975)
        kwargs_116706 = {}
        # Getting the type of 'a' (line 975)
        a_116704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 15), 'a', False)
        # Obtaining the member 'ravel' of a type (line 975)
        ravel_116705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 15), a_116704, 'ravel')
        # Calling ravel(args, kwargs) (line 975)
        ravel_call_result_116707 = invoke(stypy.reporting.localization.Localization(__file__, 975, 15), ravel_116705, *[], **kwargs_116706)
        
        # Assigning a type to the variable 'part' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'part', ravel_call_result_116707)
        
        # Assigning a Call to a Name (line 976):
        
        # Assigning a Call to a Name (line 976):
        
        # Call to _nanpercentile1d(...): (line 976)
        # Processing the call arguments (line 976)
        # Getting the type of 'part' (line 976)
        part_116709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 34), 'part', False)
        # Getting the type of 'q' (line 976)
        q_116710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 40), 'q', False)
        # Getting the type of 'overwrite_input' (line 976)
        overwrite_input_116711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 43), 'overwrite_input', False)
        # Getting the type of 'interpolation' (line 976)
        interpolation_116712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 60), 'interpolation', False)
        # Processing the call keyword arguments (line 976)
        kwargs_116713 = {}
        # Getting the type of '_nanpercentile1d' (line 976)
        _nanpercentile1d_116708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 17), '_nanpercentile1d', False)
        # Calling _nanpercentile1d(args, kwargs) (line 976)
        _nanpercentile1d_call_result_116714 = invoke(stypy.reporting.localization.Localization(__file__, 976, 17), _nanpercentile1d_116708, *[part_116709, q_116710, overwrite_input_116711, interpolation_116712], **kwargs_116713)
        
        # Assigning a type to the variable 'result' (line 976)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'result', _nanpercentile1d_call_result_116714)

        if more_types_in_union_116703:
            # Runtime conditional SSA for else branch (line 974)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_116702) or more_types_in_union_116703):
        
        # Assigning a Call to a Name (line 978):
        
        # Assigning a Call to a Name (line 978):
        
        # Call to apply_along_axis(...): (line 978)
        # Processing the call arguments (line 978)
        # Getting the type of '_nanpercentile1d' (line 978)
        _nanpercentile1d_116717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 37), '_nanpercentile1d', False)
        # Getting the type of 'axis' (line 978)
        axis_116718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 55), 'axis', False)
        # Getting the type of 'a' (line 978)
        a_116719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 61), 'a', False)
        # Getting the type of 'q' (line 978)
        q_116720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 64), 'q', False)
        # Getting the type of 'overwrite_input' (line 979)
        overwrite_input_116721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 37), 'overwrite_input', False)
        # Getting the type of 'interpolation' (line 979)
        interpolation_116722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 54), 'interpolation', False)
        # Processing the call keyword arguments (line 978)
        kwargs_116723 = {}
        # Getting the type of 'np' (line 978)
        np_116715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 17), 'np', False)
        # Obtaining the member 'apply_along_axis' of a type (line 978)
        apply_along_axis_116716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 17), np_116715, 'apply_along_axis')
        # Calling apply_along_axis(args, kwargs) (line 978)
        apply_along_axis_call_result_116724 = invoke(stypy.reporting.localization.Localization(__file__, 978, 17), apply_along_axis_116716, *[_nanpercentile1d_116717, axis_116718, a_116719, q_116720, overwrite_input_116721, interpolation_116722], **kwargs_116723)
        
        # Assigning a type to the variable 'result' (line 978)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'result', apply_along_axis_call_result_116724)
        
        
        # Getting the type of 'q' (line 983)
        q_116725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 11), 'q')
        # Obtaining the member 'ndim' of a type (line 983)
        ndim_116726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 11), q_116725, 'ndim')
        int_116727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 21), 'int')
        # Applying the binary operator '!=' (line 983)
        result_ne_116728 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 11), '!=', ndim_116726, int_116727)
        
        # Testing the type of an if condition (line 983)
        if_condition_116729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 983, 8), result_ne_116728)
        # Assigning a type to the variable 'if_condition_116729' (line 983)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 8), 'if_condition_116729', if_condition_116729)
        # SSA begins for if statement (line 983)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 984):
        
        # Assigning a Call to a Name (line 984):
        
        # Call to rollaxis(...): (line 984)
        # Processing the call arguments (line 984)
        # Getting the type of 'result' (line 984)
        result_116732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 33), 'result', False)
        # Getting the type of 'axis' (line 984)
        axis_116733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 41), 'axis', False)
        # Processing the call keyword arguments (line 984)
        kwargs_116734 = {}
        # Getting the type of 'np' (line 984)
        np_116730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 21), 'np', False)
        # Obtaining the member 'rollaxis' of a type (line 984)
        rollaxis_116731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 984, 21), np_116730, 'rollaxis')
        # Calling rollaxis(args, kwargs) (line 984)
        rollaxis_call_result_116735 = invoke(stypy.reporting.localization.Localization(__file__, 984, 21), rollaxis_116731, *[result_116732, axis_116733], **kwargs_116734)
        
        # Assigning a type to the variable 'result' (line 984)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 12), 'result', rollaxis_call_result_116735)
        # SSA join for if statement (line 983)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_116702 and more_types_in_union_116703):
            # SSA join for if statement (line 974)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 986)
    # Getting the type of 'out' (line 986)
    out_116736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'out')
    # Getting the type of 'None' (line 986)
    None_116737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 18), 'None')
    
    (may_be_116738, more_types_in_union_116739) = may_not_be_none(out_116736, None_116737)

    if may_be_116738:

        if more_types_in_union_116739:
            # Runtime conditional SSA (line 986)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 987):
        
        # Assigning a Name to a Subscript (line 987):
        # Getting the type of 'result' (line 987)
        result_116740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 19), 'result')
        # Getting the type of 'out' (line 987)
        out_116741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'out')
        Ellipsis_116742 = Ellipsis
        # Storing an element on a container (line 987)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 987, 8), out_116741, (Ellipsis_116742, result_116740))

        if more_types_in_union_116739:
            # SSA join for if statement (line 986)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'result' (line 988)
    result_116743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 988)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 988, 4), 'stypy_return_type', result_116743)
    
    # ################# End of '_nanpercentile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nanpercentile' in the type store
    # Getting the type of 'stypy_return_type' (line 966)
    stypy_return_type_116744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116744)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nanpercentile'
    return stypy_return_type_116744

# Assigning a type to the variable '_nanpercentile' (line 966)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 0), '_nanpercentile', _nanpercentile)

@norecursion
def _nanpercentile1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 991)
    False_116745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 47), 'False')
    str_116746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, 68), 'str', 'linear')
    defaults = [False_116745, str_116746]
    # Create a new context for function '_nanpercentile1d'
    module_type_store = module_type_store.open_function_context('_nanpercentile1d', 991, 0, False)
    
    # Passed parameters checking function
    _nanpercentile1d.stypy_localization = localization
    _nanpercentile1d.stypy_type_of_self = None
    _nanpercentile1d.stypy_type_store = module_type_store
    _nanpercentile1d.stypy_function_name = '_nanpercentile1d'
    _nanpercentile1d.stypy_param_names_list = ['arr1d', 'q', 'overwrite_input', 'interpolation']
    _nanpercentile1d.stypy_varargs_param_name = None
    _nanpercentile1d.stypy_kwargs_param_name = None
    _nanpercentile1d.stypy_call_defaults = defaults
    _nanpercentile1d.stypy_call_varargs = varargs
    _nanpercentile1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nanpercentile1d', ['arr1d', 'q', 'overwrite_input', 'interpolation'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nanpercentile1d', localization, ['arr1d', 'q', 'overwrite_input', 'interpolation'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nanpercentile1d(...)' code ##################

    str_116747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, (-1)), 'str', '\n    Private function for rank 1 arrays. Compute percentile ignoring\n    NaNs.\n\n    See nanpercentile for parameter usage\n    ')
    
    # Assigning a Call to a Name (line 998):
    
    # Assigning a Call to a Name (line 998):
    
    # Call to isnan(...): (line 998)
    # Processing the call arguments (line 998)
    # Getting the type of 'arr1d' (line 998)
    arr1d_116750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 17), 'arr1d', False)
    # Processing the call keyword arguments (line 998)
    kwargs_116751 = {}
    # Getting the type of 'np' (line 998)
    np_116748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 8), 'np', False)
    # Obtaining the member 'isnan' of a type (line 998)
    isnan_116749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 998, 8), np_116748, 'isnan')
    # Calling isnan(args, kwargs) (line 998)
    isnan_call_result_116752 = invoke(stypy.reporting.localization.Localization(__file__, 998, 8), isnan_116749, *[arr1d_116750], **kwargs_116751)
    
    # Assigning a type to the variable 'c' (line 998)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'c', isnan_call_result_116752)
    
    # Assigning a Subscript to a Name (line 999):
    
    # Assigning a Subscript to a Name (line 999):
    
    # Obtaining the type of the subscript
    int_116753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 20), 'int')
    
    # Call to where(...): (line 999)
    # Processing the call arguments (line 999)
    # Getting the type of 'c' (line 999)
    c_116756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 17), 'c', False)
    # Processing the call keyword arguments (line 999)
    kwargs_116757 = {}
    # Getting the type of 'np' (line 999)
    np_116754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 8), 'np', False)
    # Obtaining the member 'where' of a type (line 999)
    where_116755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 999, 8), np_116754, 'where')
    # Calling where(args, kwargs) (line 999)
    where_call_result_116758 = invoke(stypy.reporting.localization.Localization(__file__, 999, 8), where_116755, *[c_116756], **kwargs_116757)
    
    # Obtaining the member '__getitem__' of a type (line 999)
    getitem___116759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 999, 8), where_call_result_116758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 999)
    subscript_call_result_116760 = invoke(stypy.reporting.localization.Localization(__file__, 999, 8), getitem___116759, int_116753)
    
    # Assigning a type to the variable 's' (line 999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 's', subscript_call_result_116760)
    
    
    # Getting the type of 's' (line 1000)
    s_116761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 7), 's')
    # Obtaining the member 'size' of a type (line 1000)
    size_116762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 7), s_116761, 'size')
    # Getting the type of 'arr1d' (line 1000)
    arr1d_116763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 17), 'arr1d')
    # Obtaining the member 'size' of a type (line 1000)
    size_116764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 17), arr1d_116763, 'size')
    # Applying the binary operator '==' (line 1000)
    result_eq_116765 = python_operator(stypy.reporting.localization.Localization(__file__, 1000, 7), '==', size_116762, size_116764)
    
    # Testing the type of an if condition (line 1000)
    if_condition_116766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1000, 4), result_eq_116765)
    # Assigning a type to the variable 'if_condition_116766' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'if_condition_116766', if_condition_116766)
    # SSA begins for if statement (line 1000)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1001)
    # Processing the call arguments (line 1001)
    str_116769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 22), 'str', 'All-NaN slice encountered')
    # Getting the type of 'RuntimeWarning' (line 1001)
    RuntimeWarning_116770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 51), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 1001)
    kwargs_116771 = {}
    # Getting the type of 'warnings' (line 1001)
    warnings_116767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1001)
    warn_116768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 8), warnings_116767, 'warn')
    # Calling warn(args, kwargs) (line 1001)
    warn_call_result_116772 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 8), warn_116768, *[str_116769, RuntimeWarning_116770], **kwargs_116771)
    
    
    
    # Getting the type of 'q' (line 1002)
    q_116773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 11), 'q')
    # Obtaining the member 'ndim' of a type (line 1002)
    ndim_116774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 11), q_116773, 'ndim')
    int_116775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 21), 'int')
    # Applying the binary operator '==' (line 1002)
    result_eq_116776 = python_operator(stypy.reporting.localization.Localization(__file__, 1002, 11), '==', ndim_116774, int_116775)
    
    # Testing the type of an if condition (line 1002)
    if_condition_116777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1002, 8), result_eq_116776)
    # Assigning a type to the variable 'if_condition_116777' (line 1002)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'if_condition_116777', if_condition_116777)
    # SSA begins for if statement (line 1002)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 1003)
    np_116778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 19), 'np')
    # Obtaining the member 'nan' of a type (line 1003)
    nan_116779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 19), np_116778, 'nan')
    # Assigning a type to the variable 'stypy_return_type' (line 1003)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1003, 12), 'stypy_return_type', nan_116779)
    # SSA branch for the else part of an if statement (line 1002)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'np' (line 1005)
    np_116780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 19), 'np')
    # Obtaining the member 'nan' of a type (line 1005)
    nan_116781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 19), np_116780, 'nan')
    
    # Call to ones(...): (line 1005)
    # Processing the call arguments (line 1005)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1005)
    tuple_116784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1005)
    # Adding element type (line 1005)
    
    # Call to len(...): (line 1005)
    # Processing the call arguments (line 1005)
    # Getting the type of 'q' (line 1005)
    q_116786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 41), 'q', False)
    # Processing the call keyword arguments (line 1005)
    kwargs_116787 = {}
    # Getting the type of 'len' (line 1005)
    len_116785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 37), 'len', False)
    # Calling len(args, kwargs) (line 1005)
    len_call_result_116788 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 37), len_116785, *[q_116786], **kwargs_116787)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1005, 37), tuple_116784, len_call_result_116788)
    
    # Processing the call keyword arguments (line 1005)
    kwargs_116789 = {}
    # Getting the type of 'np' (line 1005)
    np_116782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 28), 'np', False)
    # Obtaining the member 'ones' of a type (line 1005)
    ones_116783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1005, 28), np_116782, 'ones')
    # Calling ones(args, kwargs) (line 1005)
    ones_call_result_116790 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 28), ones_116783, *[tuple_116784], **kwargs_116789)
    
    # Applying the binary operator '*' (line 1005)
    result_mul_116791 = python_operator(stypy.reporting.localization.Localization(__file__, 1005, 19), '*', nan_116781, ones_call_result_116790)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1005)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 12), 'stypy_return_type', result_mul_116791)
    # SSA join for if statement (line 1002)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1000)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 's' (line 1006)
    s_116792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 9), 's')
    # Obtaining the member 'size' of a type (line 1006)
    size_116793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 9), s_116792, 'size')
    int_116794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 19), 'int')
    # Applying the binary operator '==' (line 1006)
    result_eq_116795 = python_operator(stypy.reporting.localization.Localization(__file__, 1006, 9), '==', size_116793, int_116794)
    
    # Testing the type of an if condition (line 1006)
    if_condition_116796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1006, 9), result_eq_116795)
    # Assigning a type to the variable 'if_condition_116796' (line 1006)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1006, 9), 'if_condition_116796', if_condition_116796)
    # SSA begins for if statement (line 1006)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to percentile(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'arr1d' (line 1007)
    arr1d_116799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 29), 'arr1d', False)
    # Getting the type of 'q' (line 1007)
    q_116800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 36), 'q', False)
    # Processing the call keyword arguments (line 1007)
    # Getting the type of 'overwrite_input' (line 1007)
    overwrite_input_116801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 55), 'overwrite_input', False)
    keyword_116802 = overwrite_input_116801
    # Getting the type of 'interpolation' (line 1008)
    interpolation_116803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 43), 'interpolation', False)
    keyword_116804 = interpolation_116803
    kwargs_116805 = {'overwrite_input': keyword_116802, 'interpolation': keyword_116804}
    # Getting the type of 'np' (line 1007)
    np_116797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 15), 'np', False)
    # Obtaining the member 'percentile' of a type (line 1007)
    percentile_116798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 15), np_116797, 'percentile')
    # Calling percentile(args, kwargs) (line 1007)
    percentile_call_result_116806 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 15), percentile_116798, *[arr1d_116799, q_116800], **kwargs_116805)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 8), 'stypy_return_type', percentile_call_result_116806)
    # SSA branch for the else part of an if statement (line 1006)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'overwrite_input' (line 1010)
    overwrite_input_116807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 11), 'overwrite_input')
    # Testing the type of an if condition (line 1010)
    if_condition_116808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1010, 8), overwrite_input_116807)
    # Assigning a type to the variable 'if_condition_116808' (line 1010)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1010, 8), 'if_condition_116808', if_condition_116808)
    # SSA begins for if statement (line 1010)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1011):
    
    # Assigning a Name to a Name (line 1011):
    # Getting the type of 'arr1d' (line 1011)
    arr1d_116809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1011, 16), 'arr1d')
    # Assigning a type to the variable 'x' (line 1011)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 12), 'x', arr1d_116809)
    # SSA branch for the else part of an if statement (line 1010)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1013):
    
    # Assigning a Call to a Name (line 1013):
    
    # Call to copy(...): (line 1013)
    # Processing the call keyword arguments (line 1013)
    kwargs_116812 = {}
    # Getting the type of 'arr1d' (line 1013)
    arr1d_116810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 16), 'arr1d', False)
    # Obtaining the member 'copy' of a type (line 1013)
    copy_116811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1013, 16), arr1d_116810, 'copy')
    # Calling copy(args, kwargs) (line 1013)
    copy_call_result_116813 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 16), copy_116811, *[], **kwargs_116812)
    
    # Assigning a type to the variable 'x' (line 1013)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1013, 12), 'x', copy_call_result_116813)
    # SSA join for if statement (line 1010)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1015):
    
    # Assigning a Subscript to a Name (line 1015):
    
    # Obtaining the type of the subscript
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 1015)
    s_116814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 37), 's')
    # Obtaining the member 'size' of a type (line 1015)
    size_116815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 37), s_116814, 'size')
    # Applying the 'usub' unary operator (line 1015)
    result___neg___116816 = python_operator(stypy.reporting.localization.Localization(__file__, 1015, 36), 'usub', size_116815)
    
    slice_116817 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1015, 34), result___neg___116816, None, None)
    # Getting the type of 'c' (line 1015)
    c_116818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 34), 'c')
    # Obtaining the member '__getitem__' of a type (line 1015)
    getitem___116819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 34), c_116818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1015)
    subscript_call_result_116820 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 34), getitem___116819, slice_116817)
    
    # Applying the '~' unary operator (line 1015)
    result_inv_116821 = python_operator(stypy.reporting.localization.Localization(__file__, 1015, 33), '~', subscript_call_result_116820)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 1015)
    s_116822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 24), 's')
    # Obtaining the member 'size' of a type (line 1015)
    size_116823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 24), s_116822, 'size')
    # Applying the 'usub' unary operator (line 1015)
    result___neg___116824 = python_operator(stypy.reporting.localization.Localization(__file__, 1015, 23), 'usub', size_116823)
    
    slice_116825 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1015, 17), result___neg___116824, None, None)
    # Getting the type of 'arr1d' (line 1015)
    arr1d_116826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 17), 'arr1d')
    # Obtaining the member '__getitem__' of a type (line 1015)
    getitem___116827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 17), arr1d_116826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1015)
    subscript_call_result_116828 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 17), getitem___116827, slice_116825)
    
    # Obtaining the member '__getitem__' of a type (line 1015)
    getitem___116829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1015, 17), subscript_call_result_116828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1015)
    subscript_call_result_116830 = invoke(stypy.reporting.localization.Localization(__file__, 1015, 17), getitem___116829, result_inv_116821)
    
    # Assigning a type to the variable 'enonan' (line 1015)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 8), 'enonan', subscript_call_result_116830)
    
    # Assigning a Name to a Subscript (line 1017):
    
    # Assigning a Name to a Subscript (line 1017):
    # Getting the type of 'enonan' (line 1017)
    enonan_116831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 29), 'enonan')
    # Getting the type of 'x' (line 1017)
    x_116832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 8), 'x')
    
    # Obtaining the type of the subscript
    # Getting the type of 'enonan' (line 1017)
    enonan_116833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 13), 'enonan')
    # Obtaining the member 'size' of a type (line 1017)
    size_116834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 13), enonan_116833, 'size')
    slice_116835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1017, 10), None, size_116834, None)
    # Getting the type of 's' (line 1017)
    s_116836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 10), 's')
    # Obtaining the member '__getitem__' of a type (line 1017)
    getitem___116837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 10), s_116836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1017)
    subscript_call_result_116838 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 10), getitem___116837, slice_116835)
    
    # Storing an element on a container (line 1017)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 8), x_116832, (subscript_call_result_116838, enonan_116831))
    
    # Call to percentile(...): (line 1019)
    # Processing the call arguments (line 1019)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 's' (line 1019)
    s_116841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 33), 's', False)
    # Obtaining the member 'size' of a type (line 1019)
    size_116842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 33), s_116841, 'size')
    # Applying the 'usub' unary operator (line 1019)
    result___neg___116843 = python_operator(stypy.reporting.localization.Localization(__file__, 1019, 32), 'usub', size_116842)
    
    slice_116844 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1019, 29), None, result___neg___116843, None)
    # Getting the type of 'x' (line 1019)
    x_116845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 29), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 1019)
    getitem___116846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 29), x_116845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1019)
    subscript_call_result_116847 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 29), getitem___116846, slice_116844)
    
    # Getting the type of 'q' (line 1019)
    q_116848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 42), 'q', False)
    # Processing the call keyword arguments (line 1019)
    # Getting the type of 'True' (line 1019)
    True_116849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 61), 'True', False)
    keyword_116850 = True_116849
    # Getting the type of 'interpolation' (line 1020)
    interpolation_116851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1020, 43), 'interpolation', False)
    keyword_116852 = interpolation_116851
    kwargs_116853 = {'overwrite_input': keyword_116850, 'interpolation': keyword_116852}
    # Getting the type of 'np' (line 1019)
    np_116839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 15), 'np', False)
    # Obtaining the member 'percentile' of a type (line 1019)
    percentile_116840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1019, 15), np_116839, 'percentile')
    # Calling percentile(args, kwargs) (line 1019)
    percentile_call_result_116854 = invoke(stypy.reporting.localization.Localization(__file__, 1019, 15), percentile_116840, *[subscript_call_result_116847, q_116848], **kwargs_116853)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1019)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1019, 8), 'stypy_return_type', percentile_call_result_116854)
    # SSA join for if statement (line 1006)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1000)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_nanpercentile1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nanpercentile1d' in the type store
    # Getting the type of 'stypy_return_type' (line 991)
    stypy_return_type_116855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116855)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nanpercentile1d'
    return stypy_return_type_116855

# Assigning a type to the variable '_nanpercentile1d' (line 991)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 0), '_nanpercentile1d', _nanpercentile1d)

@norecursion
def nanvar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1023)
    None_116856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 19), 'None')
    # Getting the type of 'None' (line 1023)
    None_116857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 31), 'None')
    # Getting the type of 'None' (line 1023)
    None_116858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 41), 'None')
    int_116859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 52), 'int')
    # Getting the type of 'False' (line 1023)
    False_116860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 64), 'False')
    defaults = [None_116856, None_116857, None_116858, int_116859, False_116860]
    # Create a new context for function 'nanvar'
    module_type_store = module_type_store.open_function_context('nanvar', 1023, 0, False)
    
    # Passed parameters checking function
    nanvar.stypy_localization = localization
    nanvar.stypy_type_of_self = None
    nanvar.stypy_type_store = module_type_store
    nanvar.stypy_function_name = 'nanvar'
    nanvar.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    nanvar.stypy_varargs_param_name = None
    nanvar.stypy_kwargs_param_name = None
    nanvar.stypy_call_defaults = defaults
    nanvar.stypy_call_varargs = varargs
    nanvar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanvar', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanvar', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanvar(...)' code ##################

    str_116861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, (-1)), 'str', '\n    Compute the variance along the specified axis, while ignoring NaNs.\n\n    Returns the variance of the array elements, a measure of the spread of\n    a distribution.  The variance is computed for the flattened array by\n    default, otherwise over the specified axis.\n\n    For all-NaN slices or slices with zero degrees of freedom, NaN is\n    returned and a `RuntimeWarning` is raised.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        Array containing numbers whose variance is desired.  If `a` is not an\n        array, a conversion is attempted.\n    axis : int, optional\n        Axis along which the variance is computed.  The default is to compute\n        the variance of the flattened array.\n    dtype : data-type, optional\n        Type to use in computing the variance.  For arrays of integer type\n        the default is `float32`; for arrays of float types it is the same as\n        the array type.\n    out : ndarray, optional\n        Alternate output array in which to place the result.  It must have\n        the same shape as the expected output, but the type is cast if\n        necessary.\n    ddof : int, optional\n        "Delta Degrees of Freedom": the divisor used in the calculation is\n        ``N - ddof``, where ``N`` represents the number of non-NaN\n        elements. By default `ddof` is zero.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    variance : ndarray, see dtype parameter above\n        If `out` is None, return a new array containing the variance,\n        otherwise return a reference to the output array. If ddof is >= the\n        number of non-NaN elements in a slice or the slice contains only\n        NaNs, then the result for that slice is NaN.\n\n    See Also\n    --------\n    std : Standard deviation\n    mean : Average\n    var : Variance while not ignoring NaNs\n    nanstd, nanmean\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    The variance is the average of the squared deviations from the mean,\n    i.e.,  ``var = mean(abs(x - x.mean())**2)``.\n\n    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.\n    If, however, `ddof` is specified, the divisor ``N - ddof`` is used\n    instead.  In standard statistical practice, ``ddof=1`` provides an\n    unbiased estimator of the variance of a hypothetical infinite\n    population.  ``ddof=0`` provides a maximum likelihood estimate of the\n    variance for normally distributed variables.\n\n    Note that for complex numbers, the absolute value is taken before\n    squaring, so that the result is always real and nonnegative.\n\n    For floating-point input, the variance is computed using the same\n    precision the input has.  Depending on the input data, this can cause\n    the results to be inaccurate, especially for `float32` (see example\n    below).  Specifying a higher-accuracy accumulator using the ``dtype``\n    keyword can alleviate this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, np.nan], [3, 4]])\n    >>> np.var(a)\n    1.5555555555555554\n    >>> np.nanvar(a, axis=0)\n    array([ 1.,  0.])\n    >>> np.nanvar(a, axis=1)\n    array([ 0.,  0.25])\n\n    ')
    
    # Assigning a Call to a Tuple (line 1109):
    
    # Assigning a Call to a Name:
    
    # Call to _replace_nan(...): (line 1109)
    # Processing the call arguments (line 1109)
    # Getting the type of 'a' (line 1109)
    a_116863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 29), 'a', False)
    int_116864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 32), 'int')
    # Processing the call keyword arguments (line 1109)
    kwargs_116865 = {}
    # Getting the type of '_replace_nan' (line 1109)
    _replace_nan_116862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 16), '_replace_nan', False)
    # Calling _replace_nan(args, kwargs) (line 1109)
    _replace_nan_call_result_116866 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 16), _replace_nan_116862, *[a_116863, int_116864], **kwargs_116865)
    
    # Assigning a type to the variable 'call_assignment_115624' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115624', _replace_nan_call_result_116866)
    
    # Assigning a Call to a Name (line 1109):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116870 = {}
    # Getting the type of 'call_assignment_115624' (line 1109)
    call_assignment_115624_116867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115624', False)
    # Obtaining the member '__getitem__' of a type (line 1109)
    getitem___116868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 4), call_assignment_115624_116867, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116871 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116868, *[int_116869], **kwargs_116870)
    
    # Assigning a type to the variable 'call_assignment_115625' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115625', getitem___call_result_116871)
    
    # Assigning a Name to a Name (line 1109):
    # Getting the type of 'call_assignment_115625' (line 1109)
    call_assignment_115625_116872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115625')
    # Assigning a type to the variable 'arr' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'arr', call_assignment_115625_116872)
    
    # Assigning a Call to a Name (line 1109):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_116875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 4), 'int')
    # Processing the call keyword arguments
    kwargs_116876 = {}
    # Getting the type of 'call_assignment_115624' (line 1109)
    call_assignment_115624_116873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115624', False)
    # Obtaining the member '__getitem__' of a type (line 1109)
    getitem___116874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1109, 4), call_assignment_115624_116873, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_116877 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___116874, *[int_116875], **kwargs_116876)
    
    # Assigning a type to the variable 'call_assignment_115626' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115626', getitem___call_result_116877)
    
    # Assigning a Name to a Name (line 1109):
    # Getting the type of 'call_assignment_115626' (line 1109)
    call_assignment_115626_116878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'call_assignment_115626')
    # Assigning a type to the variable 'mask' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 9), 'mask', call_assignment_115626_116878)
    
    # Type idiom detected: calculating its left and rigth part (line 1110)
    # Getting the type of 'mask' (line 1110)
    mask_116879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 7), 'mask')
    # Getting the type of 'None' (line 1110)
    None_116880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 15), 'None')
    
    (may_be_116881, more_types_in_union_116882) = may_be_none(mask_116879, None_116880)

    if may_be_116881:

        if more_types_in_union_116882:
            # Runtime conditional SSA (line 1110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to var(...): (line 1111)
        # Processing the call arguments (line 1111)
        # Getting the type of 'arr' (line 1111)
        arr_116885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 22), 'arr', False)
        # Processing the call keyword arguments (line 1111)
        # Getting the type of 'axis' (line 1111)
        axis_116886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 32), 'axis', False)
        keyword_116887 = axis_116886
        # Getting the type of 'dtype' (line 1111)
        dtype_116888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 44), 'dtype', False)
        keyword_116889 = dtype_116888
        # Getting the type of 'out' (line 1111)
        out_116890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 55), 'out', False)
        keyword_116891 = out_116890
        # Getting the type of 'ddof' (line 1111)
        ddof_116892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 65), 'ddof', False)
        keyword_116893 = ddof_116892
        # Getting the type of 'keepdims' (line 1112)
        keepdims_116894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 31), 'keepdims', False)
        keyword_116895 = keepdims_116894
        kwargs_116896 = {'dtype': keyword_116889, 'out': keyword_116891, 'ddof': keyword_116893, 'keepdims': keyword_116895, 'axis': keyword_116887}
        # Getting the type of 'np' (line 1111)
        np_116883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 15), 'np', False)
        # Obtaining the member 'var' of a type (line 1111)
        var_116884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1111, 15), np_116883, 'var')
        # Calling var(args, kwargs) (line 1111)
        var_call_result_116897 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 15), var_116884, *[arr_116885], **kwargs_116896)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1111, 8), 'stypy_return_type', var_call_result_116897)

        if more_types_in_union_116882:
            # SSA join for if statement (line 1110)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1114)
    # Getting the type of 'dtype' (line 1114)
    dtype_116898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'dtype')
    # Getting the type of 'None' (line 1114)
    None_116899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 20), 'None')
    
    (may_be_116900, more_types_in_union_116901) = may_not_be_none(dtype_116898, None_116899)

    if may_be_116900:

        if more_types_in_union_116901:
            # Runtime conditional SSA (line 1114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1115):
        
        # Assigning a Call to a Name (line 1115):
        
        # Call to dtype(...): (line 1115)
        # Processing the call arguments (line 1115)
        # Getting the type of 'dtype' (line 1115)
        dtype_116904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 25), 'dtype', False)
        # Processing the call keyword arguments (line 1115)
        kwargs_116905 = {}
        # Getting the type of 'np' (line 1115)
        np_116902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 16), 'np', False)
        # Obtaining the member 'dtype' of a type (line 1115)
        dtype_116903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1115, 16), np_116902, 'dtype')
        # Calling dtype(args, kwargs) (line 1115)
        dtype_call_result_116906 = invoke(stypy.reporting.localization.Localization(__file__, 1115, 16), dtype_116903, *[dtype_116904], **kwargs_116905)
        
        # Assigning a type to the variable 'dtype' (line 1115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 8), 'dtype', dtype_call_result_116906)

        if more_types_in_union_116901:
            # SSA join for if statement (line 1114)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 1116)
    dtype_116907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 7), 'dtype')
    # Getting the type of 'None' (line 1116)
    None_116908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 20), 'None')
    # Applying the binary operator 'isnot' (line 1116)
    result_is_not_116909 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 7), 'isnot', dtype_116907, None_116908)
    
    
    
    # Call to issubclass(...): (line 1116)
    # Processing the call arguments (line 1116)
    # Getting the type of 'dtype' (line 1116)
    dtype_116911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 44), 'dtype', False)
    # Obtaining the member 'type' of a type (line 1116)
    type_116912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1116, 44), dtype_116911, 'type')
    # Getting the type of 'np' (line 1116)
    np_116913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 56), 'np', False)
    # Obtaining the member 'inexact' of a type (line 1116)
    inexact_116914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1116, 56), np_116913, 'inexact')
    # Processing the call keyword arguments (line 1116)
    kwargs_116915 = {}
    # Getting the type of 'issubclass' (line 1116)
    issubclass_116910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 33), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1116)
    issubclass_call_result_116916 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 33), issubclass_116910, *[type_116912, inexact_116914], **kwargs_116915)
    
    # Applying the 'not' unary operator (line 1116)
    result_not__116917 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 29), 'not', issubclass_call_result_116916)
    
    # Applying the binary operator 'and' (line 1116)
    result_and_keyword_116918 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 7), 'and', result_is_not_116909, result_not__116917)
    
    # Testing the type of an if condition (line 1116)
    if_condition_116919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1116, 4), result_and_keyword_116918)
    # Assigning a type to the variable 'if_condition_116919' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 4), 'if_condition_116919', if_condition_116919)
    # SSA begins for if statement (line 1116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1117)
    # Processing the call arguments (line 1117)
    str_116921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 24), 'str', 'If a is inexact, then dtype must be inexact')
    # Processing the call keyword arguments (line 1117)
    kwargs_116922 = {}
    # Getting the type of 'TypeError' (line 1117)
    TypeError_116920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1117)
    TypeError_call_result_116923 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 14), TypeError_116920, *[str_116921], **kwargs_116922)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1117, 8), TypeError_call_result_116923, 'raise parameter', BaseException)
    # SSA join for if statement (line 1116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'out' (line 1118)
    out_116924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 7), 'out')
    # Getting the type of 'None' (line 1118)
    None_116925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 18), 'None')
    # Applying the binary operator 'isnot' (line 1118)
    result_is_not_116926 = python_operator(stypy.reporting.localization.Localization(__file__, 1118, 7), 'isnot', out_116924, None_116925)
    
    
    
    # Call to issubclass(...): (line 1118)
    # Processing the call arguments (line 1118)
    # Getting the type of 'out' (line 1118)
    out_116928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 42), 'out', False)
    # Obtaining the member 'dtype' of a type (line 1118)
    dtype_116929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 42), out_116928, 'dtype')
    # Obtaining the member 'type' of a type (line 1118)
    type_116930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 42), dtype_116929, 'type')
    # Getting the type of 'np' (line 1118)
    np_116931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 58), 'np', False)
    # Obtaining the member 'inexact' of a type (line 1118)
    inexact_116932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1118, 58), np_116931, 'inexact')
    # Processing the call keyword arguments (line 1118)
    kwargs_116933 = {}
    # Getting the type of 'issubclass' (line 1118)
    issubclass_116927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 31), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1118)
    issubclass_call_result_116934 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 31), issubclass_116927, *[type_116930, inexact_116932], **kwargs_116933)
    
    # Applying the 'not' unary operator (line 1118)
    result_not__116935 = python_operator(stypy.reporting.localization.Localization(__file__, 1118, 27), 'not', issubclass_call_result_116934)
    
    # Applying the binary operator 'and' (line 1118)
    result_and_keyword_116936 = python_operator(stypy.reporting.localization.Localization(__file__, 1118, 7), 'and', result_is_not_116926, result_not__116935)
    
    # Testing the type of an if condition (line 1118)
    if_condition_116937 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1118, 4), result_and_keyword_116936)
    # Assigning a type to the variable 'if_condition_116937' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'if_condition_116937', if_condition_116937)
    # SSA begins for if statement (line 1118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1119)
    # Processing the call arguments (line 1119)
    str_116939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 24), 'str', 'If a is inexact, then out must be inexact')
    # Processing the call keyword arguments (line 1119)
    kwargs_116940 = {}
    # Getting the type of 'TypeError' (line 1119)
    TypeError_116938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1119)
    TypeError_call_result_116941 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 14), TypeError_116938, *[str_116939], **kwargs_116940)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1119, 8), TypeError_call_result_116941, 'raise parameter', BaseException)
    # SSA join for if statement (line 1118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to catch_warnings(...): (line 1121)
    # Processing the call keyword arguments (line 1121)
    kwargs_116944 = {}
    # Getting the type of 'warnings' (line 1121)
    warnings_116942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 9), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 1121)
    catch_warnings_116943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 9), warnings_116942, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 1121)
    catch_warnings_call_result_116945 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 9), catch_warnings_116943, *[], **kwargs_116944)
    
    with_116946 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 1121, 9), catch_warnings_call_result_116945, 'with parameter', '__enter__', '__exit__')

    if with_116946:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 1121)
        enter___116947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 9), catch_warnings_call_result_116945, '__enter__')
        with_enter_116948 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 9), enter___116947)
        
        # Call to simplefilter(...): (line 1122)
        # Processing the call arguments (line 1122)
        str_116951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1122, 30), 'str', 'ignore')
        # Processing the call keyword arguments (line 1122)
        kwargs_116952 = {}
        # Getting the type of 'warnings' (line 1122)
        warnings_116949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 8), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 1122)
        simplefilter_116950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1122, 8), warnings_116949, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 1122)
        simplefilter_call_result_116953 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 8), simplefilter_116950, *[str_116951], **kwargs_116952)
        
        
        # Assigning a Call to a Name (line 1125):
        
        # Assigning a Call to a Name (line 1125):
        
        # Call to sum(...): (line 1125)
        # Processing the call arguments (line 1125)
        
        # Getting the type of 'mask' (line 1125)
        mask_116956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 22), 'mask', False)
        # Applying the '~' unary operator (line 1125)
        result_inv_116957 = python_operator(stypy.reporting.localization.Localization(__file__, 1125, 21), '~', mask_116956)
        
        # Processing the call keyword arguments (line 1125)
        # Getting the type of 'axis' (line 1125)
        axis_116958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 33), 'axis', False)
        keyword_116959 = axis_116958
        # Getting the type of 'np' (line 1125)
        np_116960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 45), 'np', False)
        # Obtaining the member 'intp' of a type (line 1125)
        intp_116961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 45), np_116960, 'intp')
        keyword_116962 = intp_116961
        # Getting the type of 'True' (line 1125)
        True_116963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 63), 'True', False)
        keyword_116964 = True_116963
        kwargs_116965 = {'dtype': keyword_116962, 'keepdims': keyword_116964, 'axis': keyword_116959}
        # Getting the type of 'np' (line 1125)
        np_116954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 14), 'np', False)
        # Obtaining the member 'sum' of a type (line 1125)
        sum_116955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 14), np_116954, 'sum')
        # Calling sum(args, kwargs) (line 1125)
        sum_call_result_116966 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 14), sum_116955, *[result_inv_116957], **kwargs_116965)
        
        # Assigning a type to the variable 'cnt' (line 1125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1125, 8), 'cnt', sum_call_result_116966)
        
        # Assigning a Call to a Name (line 1126):
        
        # Assigning a Call to a Name (line 1126):
        
        # Call to sum(...): (line 1126)
        # Processing the call arguments (line 1126)
        # Getting the type of 'arr' (line 1126)
        arr_116969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 21), 'arr', False)
        # Processing the call keyword arguments (line 1126)
        # Getting the type of 'axis' (line 1126)
        axis_116970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 31), 'axis', False)
        keyword_116971 = axis_116970
        # Getting the type of 'dtype' (line 1126)
        dtype_116972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 43), 'dtype', False)
        keyword_116973 = dtype_116972
        # Getting the type of 'True' (line 1126)
        True_116974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 59), 'True', False)
        keyword_116975 = True_116974
        kwargs_116976 = {'dtype': keyword_116973, 'keepdims': keyword_116975, 'axis': keyword_116971}
        # Getting the type of 'np' (line 1126)
        np_116967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 14), 'np', False)
        # Obtaining the member 'sum' of a type (line 1126)
        sum_116968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 14), np_116967, 'sum')
        # Calling sum(args, kwargs) (line 1126)
        sum_call_result_116977 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 14), sum_116968, *[arr_116969], **kwargs_116976)
        
        # Assigning a type to the variable 'avg' (line 1126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 8), 'avg', sum_call_result_116977)
        
        # Assigning a Call to a Name (line 1127):
        
        # Assigning a Call to a Name (line 1127):
        
        # Call to _divide_by_count(...): (line 1127)
        # Processing the call arguments (line 1127)
        # Getting the type of 'avg' (line 1127)
        avg_116979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 31), 'avg', False)
        # Getting the type of 'cnt' (line 1127)
        cnt_116980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 36), 'cnt', False)
        # Processing the call keyword arguments (line 1127)
        kwargs_116981 = {}
        # Getting the type of '_divide_by_count' (line 1127)
        _divide_by_count_116978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 14), '_divide_by_count', False)
        # Calling _divide_by_count(args, kwargs) (line 1127)
        _divide_by_count_call_result_116982 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 14), _divide_by_count_116978, *[avg_116979, cnt_116980], **kwargs_116981)
        
        # Assigning a type to the variable 'avg' (line 1127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 8), 'avg', _divide_by_count_call_result_116982)
        
        # Call to subtract(...): (line 1130)
        # Processing the call arguments (line 1130)
        # Getting the type of 'arr' (line 1130)
        arr_116985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 20), 'arr', False)
        # Getting the type of 'avg' (line 1130)
        avg_116986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 25), 'avg', False)
        # Processing the call keyword arguments (line 1130)
        # Getting the type of 'arr' (line 1130)
        arr_116987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 34), 'arr', False)
        keyword_116988 = arr_116987
        str_116989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 47), 'str', 'unsafe')
        keyword_116990 = str_116989
        kwargs_116991 = {'casting': keyword_116990, 'out': keyword_116988}
        # Getting the type of 'np' (line 1130)
        np_116983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 8), 'np', False)
        # Obtaining the member 'subtract' of a type (line 1130)
        subtract_116984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 8), np_116983, 'subtract')
        # Calling subtract(args, kwargs) (line 1130)
        subtract_call_result_116992 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 8), subtract_116984, *[arr_116985, avg_116986], **kwargs_116991)
        
        
        # Assigning a Call to a Name (line 1131):
        
        # Assigning a Call to a Name (line 1131):
        
        # Call to _copyto(...): (line 1131)
        # Processing the call arguments (line 1131)
        # Getting the type of 'arr' (line 1131)
        arr_116994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 22), 'arr', False)
        int_116995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1131, 27), 'int')
        # Getting the type of 'mask' (line 1131)
        mask_116996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 30), 'mask', False)
        # Processing the call keyword arguments (line 1131)
        kwargs_116997 = {}
        # Getting the type of '_copyto' (line 1131)
        _copyto_116993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 14), '_copyto', False)
        # Calling _copyto(args, kwargs) (line 1131)
        _copyto_call_result_116998 = invoke(stypy.reporting.localization.Localization(__file__, 1131, 14), _copyto_116993, *[arr_116994, int_116995, mask_116996], **kwargs_116997)
        
        # Assigning a type to the variable 'arr' (line 1131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'arr', _copyto_call_result_116998)
        
        
        # Call to issubclass(...): (line 1132)
        # Processing the call arguments (line 1132)
        # Getting the type of 'arr' (line 1132)
        arr_117000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 22), 'arr', False)
        # Obtaining the member 'dtype' of a type (line 1132)
        dtype_117001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 22), arr_117000, 'dtype')
        # Obtaining the member 'type' of a type (line 1132)
        type_117002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 22), dtype_117001, 'type')
        # Getting the type of 'np' (line 1132)
        np_117003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 38), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 1132)
        complexfloating_117004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 38), np_117003, 'complexfloating')
        # Processing the call keyword arguments (line 1132)
        kwargs_117005 = {}
        # Getting the type of 'issubclass' (line 1132)
        issubclass_116999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 11), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 1132)
        issubclass_call_result_117006 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 11), issubclass_116999, *[type_117002, complexfloating_117004], **kwargs_117005)
        
        # Testing the type of an if condition (line 1132)
        if_condition_117007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1132, 8), issubclass_call_result_117006)
        # Assigning a type to the variable 'if_condition_117007' (line 1132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 8), 'if_condition_117007', if_condition_117007)
        # SSA begins for if statement (line 1132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 1133):
        
        # Assigning a Attribute to a Name (line 1133):
        
        # Call to multiply(...): (line 1133)
        # Processing the call arguments (line 1133)
        # Getting the type of 'arr' (line 1133)
        arr_117010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 30), 'arr', False)
        
        # Call to conj(...): (line 1133)
        # Processing the call keyword arguments (line 1133)
        kwargs_117013 = {}
        # Getting the type of 'arr' (line 1133)
        arr_117011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 35), 'arr', False)
        # Obtaining the member 'conj' of a type (line 1133)
        conj_117012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1133, 35), arr_117011, 'conj')
        # Calling conj(args, kwargs) (line 1133)
        conj_call_result_117014 = invoke(stypy.reporting.localization.Localization(__file__, 1133, 35), conj_117012, *[], **kwargs_117013)
        
        # Processing the call keyword arguments (line 1133)
        # Getting the type of 'arr' (line 1133)
        arr_117015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 51), 'arr', False)
        keyword_117016 = arr_117015
        kwargs_117017 = {'out': keyword_117016}
        # Getting the type of 'np' (line 1133)
        np_117008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 18), 'np', False)
        # Obtaining the member 'multiply' of a type (line 1133)
        multiply_117009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1133, 18), np_117008, 'multiply')
        # Calling multiply(args, kwargs) (line 1133)
        multiply_call_result_117018 = invoke(stypy.reporting.localization.Localization(__file__, 1133, 18), multiply_117009, *[arr_117010, conj_call_result_117014], **kwargs_117017)
        
        # Obtaining the member 'real' of a type (line 1133)
        real_117019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1133, 18), multiply_call_result_117018, 'real')
        # Assigning a type to the variable 'sqr' (line 1133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1133, 12), 'sqr', real_117019)
        # SSA branch for the else part of an if statement (line 1132)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1135):
        
        # Assigning a Call to a Name (line 1135):
        
        # Call to multiply(...): (line 1135)
        # Processing the call arguments (line 1135)
        # Getting the type of 'arr' (line 1135)
        arr_117022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 30), 'arr', False)
        # Getting the type of 'arr' (line 1135)
        arr_117023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 35), 'arr', False)
        # Processing the call keyword arguments (line 1135)
        # Getting the type of 'arr' (line 1135)
        arr_117024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 44), 'arr', False)
        keyword_117025 = arr_117024
        kwargs_117026 = {'out': keyword_117025}
        # Getting the type of 'np' (line 1135)
        np_117020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 18), 'np', False)
        # Obtaining the member 'multiply' of a type (line 1135)
        multiply_117021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1135, 18), np_117020, 'multiply')
        # Calling multiply(args, kwargs) (line 1135)
        multiply_call_result_117027 = invoke(stypy.reporting.localization.Localization(__file__, 1135, 18), multiply_117021, *[arr_117022, arr_117023], **kwargs_117026)
        
        # Assigning a type to the variable 'sqr' (line 1135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 12), 'sqr', multiply_call_result_117027)
        # SSA join for if statement (line 1132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 1138):
        
        # Assigning a Call to a Name (line 1138):
        
        # Call to sum(...): (line 1138)
        # Processing the call arguments (line 1138)
        # Getting the type of 'sqr' (line 1138)
        sqr_117030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 21), 'sqr', False)
        # Processing the call keyword arguments (line 1138)
        # Getting the type of 'axis' (line 1138)
        axis_117031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 31), 'axis', False)
        keyword_117032 = axis_117031
        # Getting the type of 'dtype' (line 1138)
        dtype_117033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 43), 'dtype', False)
        keyword_117034 = dtype_117033
        # Getting the type of 'out' (line 1138)
        out_117035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 54), 'out', False)
        keyword_117036 = out_117035
        # Getting the type of 'keepdims' (line 1138)
        keepdims_117037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 68), 'keepdims', False)
        keyword_117038 = keepdims_117037
        kwargs_117039 = {'dtype': keyword_117034, 'out': keyword_117036, 'keepdims': keyword_117038, 'axis': keyword_117032}
        # Getting the type of 'np' (line 1138)
        np_117028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 14), 'np', False)
        # Obtaining the member 'sum' of a type (line 1138)
        sum_117029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1138, 14), np_117028, 'sum')
        # Calling sum(args, kwargs) (line 1138)
        sum_call_result_117040 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 14), sum_117029, *[sqr_117030], **kwargs_117039)
        
        # Assigning a type to the variable 'var' (line 1138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 8), 'var', sum_call_result_117040)
        
        
        # Getting the type of 'var' (line 1139)
        var_117041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 11), 'var')
        # Obtaining the member 'ndim' of a type (line 1139)
        ndim_117042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 11), var_117041, 'ndim')
        # Getting the type of 'cnt' (line 1139)
        cnt_117043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 22), 'cnt')
        # Obtaining the member 'ndim' of a type (line 1139)
        ndim_117044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 22), cnt_117043, 'ndim')
        # Applying the binary operator '<' (line 1139)
        result_lt_117045 = python_operator(stypy.reporting.localization.Localization(__file__, 1139, 11), '<', ndim_117042, ndim_117044)
        
        # Testing the type of an if condition (line 1139)
        if_condition_117046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1139, 8), result_lt_117045)
        # Assigning a type to the variable 'if_condition_117046' (line 1139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 8), 'if_condition_117046', if_condition_117046)
        # SSA begins for if statement (line 1139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1141):
        
        # Assigning a Call to a Name (line 1141):
        
        # Call to squeeze(...): (line 1141)
        # Processing the call arguments (line 1141)
        # Getting the type of 'axis' (line 1141)
        axis_117049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 30), 'axis', False)
        # Processing the call keyword arguments (line 1141)
        kwargs_117050 = {}
        # Getting the type of 'cnt' (line 1141)
        cnt_117047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 18), 'cnt', False)
        # Obtaining the member 'squeeze' of a type (line 1141)
        squeeze_117048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 18), cnt_117047, 'squeeze')
        # Calling squeeze(args, kwargs) (line 1141)
        squeeze_call_result_117051 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 18), squeeze_117048, *[axis_117049], **kwargs_117050)
        
        # Assigning a type to the variable 'cnt' (line 1141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 12), 'cnt', squeeze_call_result_117051)
        # SSA join for if statement (line 1139)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1142):
        
        # Assigning a BinOp to a Name (line 1142):
        # Getting the type of 'cnt' (line 1142)
        cnt_117052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 14), 'cnt')
        # Getting the type of 'ddof' (line 1142)
        ddof_117053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 20), 'ddof')
        # Applying the binary operator '-' (line 1142)
        result_sub_117054 = python_operator(stypy.reporting.localization.Localization(__file__, 1142, 14), '-', cnt_117052, ddof_117053)
        
        # Assigning a type to the variable 'dof' (line 1142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'dof', result_sub_117054)
        
        # Assigning a Call to a Name (line 1143):
        
        # Assigning a Call to a Name (line 1143):
        
        # Call to _divide_by_count(...): (line 1143)
        # Processing the call arguments (line 1143)
        # Getting the type of 'var' (line 1143)
        var_117056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 31), 'var', False)
        # Getting the type of 'dof' (line 1143)
        dof_117057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 36), 'dof', False)
        # Processing the call keyword arguments (line 1143)
        kwargs_117058 = {}
        # Getting the type of '_divide_by_count' (line 1143)
        _divide_by_count_117055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 14), '_divide_by_count', False)
        # Calling _divide_by_count(args, kwargs) (line 1143)
        _divide_by_count_call_result_117059 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 14), _divide_by_count_117055, *[var_117056, dof_117057], **kwargs_117058)
        
        # Assigning a type to the variable 'var' (line 1143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1143, 8), 'var', _divide_by_count_call_result_117059)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 1121)
        exit___117060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 9), catch_warnings_call_result_116945, '__exit__')
        with_exit_117061 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 9), exit___117060, None, None, None)

    
    # Assigning a Compare to a Name (line 1145):
    
    # Assigning a Compare to a Name (line 1145):
    
    # Getting the type of 'dof' (line 1145)
    dof_117062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 13), 'dof')
    int_117063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, 20), 'int')
    # Applying the binary operator '<=' (line 1145)
    result_le_117064 = python_operator(stypy.reporting.localization.Localization(__file__, 1145, 13), '<=', dof_117062, int_117063)
    
    # Assigning a type to the variable 'isbad' (line 1145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1145, 4), 'isbad', result_le_117064)
    
    
    # Call to any(...): (line 1146)
    # Processing the call arguments (line 1146)
    # Getting the type of 'isbad' (line 1146)
    isbad_117067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 14), 'isbad', False)
    # Processing the call keyword arguments (line 1146)
    kwargs_117068 = {}
    # Getting the type of 'np' (line 1146)
    np_117065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1146, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 1146)
    any_117066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1146, 7), np_117065, 'any')
    # Calling any(args, kwargs) (line 1146)
    any_call_result_117069 = invoke(stypy.reporting.localization.Localization(__file__, 1146, 7), any_117066, *[isbad_117067], **kwargs_117068)
    
    # Testing the type of an if condition (line 1146)
    if_condition_117070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1146, 4), any_call_result_117069)
    # Assigning a type to the variable 'if_condition_117070' (line 1146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1146, 4), 'if_condition_117070', if_condition_117070)
    # SSA begins for if statement (line 1146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1147)
    # Processing the call arguments (line 1147)
    str_117073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1147, 22), 'str', 'Degrees of freedom <= 0 for slice.')
    # Getting the type of 'RuntimeWarning' (line 1147)
    RuntimeWarning_117074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 60), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 1147)
    kwargs_117075 = {}
    # Getting the type of 'warnings' (line 1147)
    warnings_117071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1147)
    warn_117072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1147, 8), warnings_117071, 'warn')
    # Calling warn(args, kwargs) (line 1147)
    warn_call_result_117076 = invoke(stypy.reporting.localization.Localization(__file__, 1147, 8), warn_117072, *[str_117073, RuntimeWarning_117074], **kwargs_117075)
    
    
    # Assigning a Call to a Name (line 1150):
    
    # Assigning a Call to a Name (line 1150):
    
    # Call to _copyto(...): (line 1150)
    # Processing the call arguments (line 1150)
    # Getting the type of 'var' (line 1150)
    var_117078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 22), 'var', False)
    # Getting the type of 'np' (line 1150)
    np_117079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 27), 'np', False)
    # Obtaining the member 'nan' of a type (line 1150)
    nan_117080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1150, 27), np_117079, 'nan')
    # Getting the type of 'isbad' (line 1150)
    isbad_117081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 35), 'isbad', False)
    # Processing the call keyword arguments (line 1150)
    kwargs_117082 = {}
    # Getting the type of '_copyto' (line 1150)
    _copyto_117077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1150, 14), '_copyto', False)
    # Calling _copyto(args, kwargs) (line 1150)
    _copyto_call_result_117083 = invoke(stypy.reporting.localization.Localization(__file__, 1150, 14), _copyto_117077, *[var_117078, nan_117080, isbad_117081], **kwargs_117082)
    
    # Assigning a type to the variable 'var' (line 1150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1150, 8), 'var', _copyto_call_result_117083)
    # SSA join for if statement (line 1146)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'var' (line 1151)
    var_117084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 11), 'var')
    # Assigning a type to the variable 'stypy_return_type' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 4), 'stypy_return_type', var_117084)
    
    # ################# End of 'nanvar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanvar' in the type store
    # Getting the type of 'stypy_return_type' (line 1023)
    stypy_return_type_117085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117085)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanvar'
    return stypy_return_type_117085

# Assigning a type to the variable 'nanvar' (line 1023)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1023, 0), 'nanvar', nanvar)

@norecursion
def nanstd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1154)
    None_117086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 19), 'None')
    # Getting the type of 'None' (line 1154)
    None_117087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 31), 'None')
    # Getting the type of 'None' (line 1154)
    None_117088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 41), 'None')
    int_117089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 52), 'int')
    # Getting the type of 'False' (line 1154)
    False_117090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 64), 'False')
    defaults = [None_117086, None_117087, None_117088, int_117089, False_117090]
    # Create a new context for function 'nanstd'
    module_type_store = module_type_store.open_function_context('nanstd', 1154, 0, False)
    
    # Passed parameters checking function
    nanstd.stypy_localization = localization
    nanstd.stypy_type_of_self = None
    nanstd.stypy_type_store = module_type_store
    nanstd.stypy_function_name = 'nanstd'
    nanstd.stypy_param_names_list = ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims']
    nanstd.stypy_varargs_param_name = None
    nanstd.stypy_kwargs_param_name = None
    nanstd.stypy_call_defaults = defaults
    nanstd.stypy_call_varargs = varargs
    nanstd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nanstd', ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nanstd', localization, ['a', 'axis', 'dtype', 'out', 'ddof', 'keepdims'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nanstd(...)' code ##################

    str_117091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1241, (-1)), 'str', '\n    Compute the standard deviation along the specified axis, while\n    ignoring NaNs.\n\n    Returns the standard deviation, a measure of the spread of a\n    distribution, of the non-NaN array elements. The standard deviation is\n    computed for the flattened array by default, otherwise over the\n    specified axis.\n\n    For all-NaN slices or slices with zero degrees of freedom, NaN is\n    returned and a `RuntimeWarning` is raised.\n\n    .. versionadded:: 1.8.0\n\n    Parameters\n    ----------\n    a : array_like\n        Calculate the standard deviation of the non-NaN values.\n    axis : int, optional\n        Axis along which the standard deviation is computed. The default is\n        to compute the standard deviation of the flattened array.\n    dtype : dtype, optional\n        Type to use in computing the standard deviation. For arrays of\n        integer type the default is float64, for arrays of float types it\n        is the same as the array type.\n    out : ndarray, optional\n        Alternative output array in which to place the result. It must have\n        the same shape as the expected output but the type (of the\n        calculated values) will be cast if necessary.\n    ddof : int, optional\n        Means Delta Degrees of Freedom.  The divisor used in calculations\n        is ``N - ddof``, where ``N`` represents the number of non-NaN\n        elements.  By default `ddof` is zero.\n    keepdims : bool, optional\n        If this is set to True, the axes which are reduced are left\n        in the result as dimensions with size one. With this option,\n        the result will broadcast correctly against the original `arr`.\n\n    Returns\n    -------\n    standard_deviation : ndarray, see dtype parameter above.\n        If `out` is None, return a new array containing the standard\n        deviation, otherwise return a reference to the output array. If\n        ddof is >= the number of non-NaN elements in a slice or the slice\n        contains only NaNs, then the result for that slice is NaN.\n\n    See Also\n    --------\n    var, mean, std\n    nanvar, nanmean\n    numpy.doc.ufuncs : Section "Output arguments"\n\n    Notes\n    -----\n    The standard deviation is the square root of the average of the squared\n    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.\n\n    The average squared deviation is normally calculated as\n    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is\n    specified, the divisor ``N - ddof`` is used instead. In standard\n    statistical practice, ``ddof=1`` provides an unbiased estimator of the\n    variance of the infinite population. ``ddof=0`` provides a maximum\n    likelihood estimate of the variance for normally distributed variables.\n    The standard deviation computed in this function is the square root of\n    the estimated variance, so even with ``ddof=1``, it will not be an\n    unbiased estimate of the standard deviation per se.\n\n    Note that, for complex numbers, `std` takes the absolute value before\n    squaring, so that the result is always real and nonnegative.\n\n    For floating-point input, the *std* is computed using the same\n    precision the input has. Depending on the input data, this can cause\n    the results to be inaccurate, especially for float32 (see example\n    below).  Specifying a higher-accuracy accumulator using the `dtype`\n    keyword can alleviate this issue.\n\n    Examples\n    --------\n    >>> a = np.array([[1, np.nan], [3, 4]])\n    >>> np.nanstd(a)\n    1.247219128924647\n    >>> np.nanstd(a, axis=0)\n    array([ 1.,  0.])\n    >>> np.nanstd(a, axis=1)\n    array([ 0.,  0.5])\n\n    ')
    
    # Assigning a Call to a Name (line 1242):
    
    # Assigning a Call to a Name (line 1242):
    
    # Call to nanvar(...): (line 1242)
    # Processing the call arguments (line 1242)
    # Getting the type of 'a' (line 1242)
    a_117093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 17), 'a', False)
    # Processing the call keyword arguments (line 1242)
    # Getting the type of 'axis' (line 1242)
    axis_117094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 25), 'axis', False)
    keyword_117095 = axis_117094
    # Getting the type of 'dtype' (line 1242)
    dtype_117096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 37), 'dtype', False)
    keyword_117097 = dtype_117096
    # Getting the type of 'out' (line 1242)
    out_117098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 48), 'out', False)
    keyword_117099 = out_117098
    # Getting the type of 'ddof' (line 1242)
    ddof_117100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 58), 'ddof', False)
    keyword_117101 = ddof_117100
    # Getting the type of 'keepdims' (line 1243)
    keepdims_117102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 26), 'keepdims', False)
    keyword_117103 = keepdims_117102
    kwargs_117104 = {'dtype': keyword_117097, 'out': keyword_117099, 'ddof': keyword_117101, 'keepdims': keyword_117103, 'axis': keyword_117095}
    # Getting the type of 'nanvar' (line 1242)
    nanvar_117092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 10), 'nanvar', False)
    # Calling nanvar(args, kwargs) (line 1242)
    nanvar_call_result_117105 = invoke(stypy.reporting.localization.Localization(__file__, 1242, 10), nanvar_117092, *[a_117093], **kwargs_117104)
    
    # Assigning a type to the variable 'var' (line 1242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 4), 'var', nanvar_call_result_117105)
    
    
    # Call to isinstance(...): (line 1244)
    # Processing the call arguments (line 1244)
    # Getting the type of 'var' (line 1244)
    var_117107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 18), 'var', False)
    # Getting the type of 'np' (line 1244)
    np_117108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 23), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 1244)
    ndarray_117109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1244, 23), np_117108, 'ndarray')
    # Processing the call keyword arguments (line 1244)
    kwargs_117110 = {}
    # Getting the type of 'isinstance' (line 1244)
    isinstance_117106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1244)
    isinstance_call_result_117111 = invoke(stypy.reporting.localization.Localization(__file__, 1244, 7), isinstance_117106, *[var_117107, ndarray_117109], **kwargs_117110)
    
    # Testing the type of an if condition (line 1244)
    if_condition_117112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1244, 4), isinstance_call_result_117111)
    # Assigning a type to the variable 'if_condition_117112' (line 1244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1244, 4), 'if_condition_117112', if_condition_117112)
    # SSA begins for if statement (line 1244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1245):
    
    # Assigning a Call to a Name (line 1245):
    
    # Call to sqrt(...): (line 1245)
    # Processing the call arguments (line 1245)
    # Getting the type of 'var' (line 1245)
    var_117115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 22), 'var', False)
    # Processing the call keyword arguments (line 1245)
    # Getting the type of 'var' (line 1245)
    var_117116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 31), 'var', False)
    keyword_117117 = var_117116
    kwargs_117118 = {'out': keyword_117117}
    # Getting the type of 'np' (line 1245)
    np_117113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1245)
    sqrt_117114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1245, 14), np_117113, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1245)
    sqrt_call_result_117119 = invoke(stypy.reporting.localization.Localization(__file__, 1245, 14), sqrt_117114, *[var_117115], **kwargs_117118)
    
    # Assigning a type to the variable 'std' (line 1245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1245, 8), 'std', sqrt_call_result_117119)
    # SSA branch for the else part of an if statement (line 1244)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1247):
    
    # Assigning a Call to a Name (line 1247):
    
    # Call to type(...): (line 1247)
    # Processing the call arguments (line 1247)
    
    # Call to sqrt(...): (line 1247)
    # Processing the call arguments (line 1247)
    # Getting the type of 'var' (line 1247)
    var_117125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 37), 'var', False)
    # Processing the call keyword arguments (line 1247)
    kwargs_117126 = {}
    # Getting the type of 'np' (line 1247)
    np_117123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 29), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1247)
    sqrt_117124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 29), np_117123, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1247)
    sqrt_call_result_117127 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 29), sqrt_117124, *[var_117125], **kwargs_117126)
    
    # Processing the call keyword arguments (line 1247)
    kwargs_117128 = {}
    # Getting the type of 'var' (line 1247)
    var_117120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 14), 'var', False)
    # Obtaining the member 'dtype' of a type (line 1247)
    dtype_117121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 14), var_117120, 'dtype')
    # Obtaining the member 'type' of a type (line 1247)
    type_117122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 14), dtype_117121, 'type')
    # Calling type(args, kwargs) (line 1247)
    type_call_result_117129 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 14), type_117122, *[sqrt_call_result_117127], **kwargs_117128)
    
    # Assigning a type to the variable 'std' (line 1247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'std', type_call_result_117129)
    # SSA join for if statement (line 1244)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'std' (line 1248)
    std_117130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 11), 'std')
    # Assigning a type to the variable 'stypy_return_type' (line 1248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1248, 4), 'stypy_return_type', std_117130)
    
    # ################# End of 'nanstd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nanstd' in the type store
    # Getting the type of 'stypy_return_type' (line 1154)
    stypy_return_type_117131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nanstd'
    return stypy_return_type_117131

# Assigning a type to the variable 'nanstd' (line 1154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 0), 'nanstd', nanstd)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
