
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: The arraypad module contains a group of functions to pad values onto the edges
3: of an n-dimensional array.
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: import numpy as np
9: 
10: 
11: __all__ = ['pad']
12: 
13: 
14: ###############################################################################
15: # Private utility functions.
16: 
17: 
18: def _arange_ndarray(arr, shape, axis, reverse=False):
19:     '''
20:     Create an ndarray of `shape` with increments along specified `axis`
21: 
22:     Parameters
23:     ----------
24:     arr : ndarray
25:         Input array of arbitrary shape.
26:     shape : tuple of ints
27:         Shape of desired array. Should be equivalent to `arr.shape` except
28:         `shape[axis]` which may have any positive value.
29:     axis : int
30:         Axis to increment along.
31:     reverse : bool
32:         If False, increment in a positive fashion from 1 to `shape[axis]`,
33:         inclusive. If True, the bounds are the same but the order reversed.
34: 
35:     Returns
36:     -------
37:     padarr : ndarray
38:         Output array sized to pad `arr` along `axis`, with linear range from
39:         1 to `shape[axis]` along specified `axis`.
40: 
41:     Notes
42:     -----
43:     The range is deliberately 1-indexed for this specific use case. Think of
44:     this algorithm as broadcasting `np.arange` to a single `axis` of an
45:     arbitrarily shaped ndarray.
46: 
47:     '''
48:     initshape = tuple(1 if i != axis else shape[axis]
49:                       for (i, x) in enumerate(arr.shape))
50:     if not reverse:
51:         padarr = np.arange(1, shape[axis] + 1)
52:     else:
53:         padarr = np.arange(shape[axis], 0, -1)
54:     padarr = padarr.reshape(initshape)
55:     for i, dim in enumerate(shape):
56:         if padarr.shape[i] != dim:
57:             padarr = padarr.repeat(dim, axis=i)
58:     return padarr
59: 
60: 
61: def _round_ifneeded(arr, dtype):
62:     '''
63:     Rounds arr inplace if destination dtype is integer.
64: 
65:     Parameters
66:     ----------
67:     arr : ndarray
68:         Input array.
69:     dtype : dtype
70:         The dtype of the destination array.
71: 
72:     '''
73:     if np.issubdtype(dtype, np.integer):
74:         arr.round(out=arr)
75: 
76: 
77: def _prepend_const(arr, pad_amt, val, axis=-1):
78:     '''
79:     Prepend constant `val` along `axis` of `arr`.
80: 
81:     Parameters
82:     ----------
83:     arr : ndarray
84:         Input array of arbitrary shape.
85:     pad_amt : int
86:         Amount of padding to prepend.
87:     val : scalar
88:         Constant value to use. For best results should be of type `arr.dtype`;
89:         if not `arr.dtype` will be cast to `arr.dtype`.
90:     axis : int
91:         Axis along which to pad `arr`.
92: 
93:     Returns
94:     -------
95:     padarr : ndarray
96:         Output array, with `pad_amt` constant `val` prepended along `axis`.
97: 
98:     '''
99:     if pad_amt == 0:
100:         return arr
101:     padshape = tuple(x if i != axis else pad_amt
102:                      for (i, x) in enumerate(arr.shape))
103:     if val == 0:
104:         return np.concatenate((np.zeros(padshape, dtype=arr.dtype), arr),
105:                               axis=axis)
106:     else:
107:         return np.concatenate(((np.zeros(padshape) + val).astype(arr.dtype),
108:                                arr), axis=axis)
109: 
110: 
111: def _append_const(arr, pad_amt, val, axis=-1):
112:     '''
113:     Append constant `val` along `axis` of `arr`.
114: 
115:     Parameters
116:     ----------
117:     arr : ndarray
118:         Input array of arbitrary shape.
119:     pad_amt : int
120:         Amount of padding to append.
121:     val : scalar
122:         Constant value to use. For best results should be of type `arr.dtype`;
123:         if not `arr.dtype` will be cast to `arr.dtype`.
124:     axis : int
125:         Axis along which to pad `arr`.
126: 
127:     Returns
128:     -------
129:     padarr : ndarray
130:         Output array, with `pad_amt` constant `val` appended along `axis`.
131: 
132:     '''
133:     if pad_amt == 0:
134:         return arr
135:     padshape = tuple(x if i != axis else pad_amt
136:                      for (i, x) in enumerate(arr.shape))
137:     if val == 0:
138:         return np.concatenate((arr, np.zeros(padshape, dtype=arr.dtype)),
139:                               axis=axis)
140:     else:
141:         return np.concatenate(
142:             (arr, (np.zeros(padshape) + val).astype(arr.dtype)), axis=axis)
143: 
144: 
145: def _prepend_edge(arr, pad_amt, axis=-1):
146:     '''
147:     Prepend `pad_amt` to `arr` along `axis` by extending edge values.
148: 
149:     Parameters
150:     ----------
151:     arr : ndarray
152:         Input array of arbitrary shape.
153:     pad_amt : int
154:         Amount of padding to prepend.
155:     axis : int
156:         Axis along which to pad `arr`.
157: 
158:     Returns
159:     -------
160:     padarr : ndarray
161:         Output array, extended by `pad_amt` edge values appended along `axis`.
162: 
163:     '''
164:     if pad_amt == 0:
165:         return arr
166: 
167:     edge_slice = tuple(slice(None) if i != axis else 0
168:                        for (i, x) in enumerate(arr.shape))
169: 
170:     # Shape to restore singleton dimension after slicing
171:     pad_singleton = tuple(x if i != axis else 1
172:                           for (i, x) in enumerate(arr.shape))
173:     edge_arr = arr[edge_slice].reshape(pad_singleton)
174:     return np.concatenate((edge_arr.repeat(pad_amt, axis=axis), arr),
175:                           axis=axis)
176: 
177: 
178: def _append_edge(arr, pad_amt, axis=-1):
179:     '''
180:     Append `pad_amt` to `arr` along `axis` by extending edge values.
181: 
182:     Parameters
183:     ----------
184:     arr : ndarray
185:         Input array of arbitrary shape.
186:     pad_amt : int
187:         Amount of padding to append.
188:     axis : int
189:         Axis along which to pad `arr`.
190: 
191:     Returns
192:     -------
193:     padarr : ndarray
194:         Output array, extended by `pad_amt` edge values prepended along
195:         `axis`.
196: 
197:     '''
198:     if pad_amt == 0:
199:         return arr
200: 
201:     edge_slice = tuple(slice(None) if i != axis else arr.shape[axis] - 1
202:                        for (i, x) in enumerate(arr.shape))
203: 
204:     # Shape to restore singleton dimension after slicing
205:     pad_singleton = tuple(x if i != axis else 1
206:                           for (i, x) in enumerate(arr.shape))
207:     edge_arr = arr[edge_slice].reshape(pad_singleton)
208:     return np.concatenate((arr, edge_arr.repeat(pad_amt, axis=axis)),
209:                           axis=axis)
210: 
211: 
212: def _prepend_ramp(arr, pad_amt, end, axis=-1):
213:     '''
214:     Prepend linear ramp along `axis`.
215: 
216:     Parameters
217:     ----------
218:     arr : ndarray
219:         Input array of arbitrary shape.
220:     pad_amt : int
221:         Amount of padding to prepend.
222:     end : scalar
223:         Constal value to use. For best results should be of type `arr.dtype`;
224:         if not `arr.dtype` will be cast to `arr.dtype`.
225:     axis : int
226:         Axis along which to pad `arr`.
227: 
228:     Returns
229:     -------
230:     padarr : ndarray
231:         Output array, with `pad_amt` values prepended along `axis`. The
232:         prepended region ramps linearly from the edge value to `end`.
233: 
234:     '''
235:     if pad_amt == 0:
236:         return arr
237: 
238:     # Generate shape for final concatenated array
239:     padshape = tuple(x if i != axis else pad_amt
240:                      for (i, x) in enumerate(arr.shape))
241: 
242:     # Generate an n-dimensional array incrementing along `axis`
243:     ramp_arr = _arange_ndarray(arr, padshape, axis,
244:                                reverse=True).astype(np.float64)
245: 
246:     # Appropriate slicing to extract n-dimensional edge along `axis`
247:     edge_slice = tuple(slice(None) if i != axis else 0
248:                        for (i, x) in enumerate(arr.shape))
249: 
250:     # Shape to restore singleton dimension after slicing
251:     pad_singleton = tuple(x if i != axis else 1
252:                           for (i, x) in enumerate(arr.shape))
253: 
254:     # Extract edge, reshape to original rank, and extend along `axis`
255:     edge_pad = arr[edge_slice].reshape(pad_singleton).repeat(pad_amt, axis)
256: 
257:     # Linear ramp
258:     slope = (end - edge_pad) / float(pad_amt)
259:     ramp_arr = ramp_arr * slope
260:     ramp_arr += edge_pad
261:     _round_ifneeded(ramp_arr, arr.dtype)
262: 
263:     # Ramp values will most likely be float, cast them to the same type as arr
264:     return np.concatenate((ramp_arr.astype(arr.dtype), arr), axis=axis)
265: 
266: 
267: def _append_ramp(arr, pad_amt, end, axis=-1):
268:     '''
269:     Append linear ramp along `axis`.
270: 
271:     Parameters
272:     ----------
273:     arr : ndarray
274:         Input array of arbitrary shape.
275:     pad_amt : int
276:         Amount of padding to append.
277:     end : scalar
278:         Constal value to use. For best results should be of type `arr.dtype`;
279:         if not `arr.dtype` will be cast to `arr.dtype`.
280:     axis : int
281:         Axis along which to pad `arr`.
282: 
283:     Returns
284:     -------
285:     padarr : ndarray
286:         Output array, with `pad_amt` values appended along `axis`. The
287:         appended region ramps linearly from the edge value to `end`.
288: 
289:     '''
290:     if pad_amt == 0:
291:         return arr
292: 
293:     # Generate shape for final concatenated array
294:     padshape = tuple(x if i != axis else pad_amt
295:                      for (i, x) in enumerate(arr.shape))
296: 
297:     # Generate an n-dimensional array incrementing along `axis`
298:     ramp_arr = _arange_ndarray(arr, padshape, axis,
299:                                reverse=False).astype(np.float64)
300: 
301:     # Slice a chunk from the edge to calculate stats on
302:     edge_slice = tuple(slice(None) if i != axis else -1
303:                        for (i, x) in enumerate(arr.shape))
304: 
305:     # Shape to restore singleton dimension after slicing
306:     pad_singleton = tuple(x if i != axis else 1
307:                           for (i, x) in enumerate(arr.shape))
308: 
309:     # Extract edge, reshape to original rank, and extend along `axis`
310:     edge_pad = arr[edge_slice].reshape(pad_singleton).repeat(pad_amt, axis)
311: 
312:     # Linear ramp
313:     slope = (end - edge_pad) / float(pad_amt)
314:     ramp_arr = ramp_arr * slope
315:     ramp_arr += edge_pad
316:     _round_ifneeded(ramp_arr, arr.dtype)
317: 
318:     # Ramp values will most likely be float, cast them to the same type as arr
319:     return np.concatenate((arr, ramp_arr.astype(arr.dtype)), axis=axis)
320: 
321: 
322: def _prepend_max(arr, pad_amt, num, axis=-1):
323:     '''
324:     Prepend `pad_amt` maximum values along `axis`.
325: 
326:     Parameters
327:     ----------
328:     arr : ndarray
329:         Input array of arbitrary shape.
330:     pad_amt : int
331:         Amount of padding to prepend.
332:     num : int
333:         Depth into `arr` along `axis` to calculate maximum.
334:         Range: [1, `arr.shape[axis]`] or None (entire axis)
335:     axis : int
336:         Axis along which to pad `arr`.
337: 
338:     Returns
339:     -------
340:     padarr : ndarray
341:         Output array, with `pad_amt` values appended along `axis`. The
342:         prepended region is the maximum of the first `num` values along
343:         `axis`.
344: 
345:     '''
346:     if pad_amt == 0:
347:         return arr
348: 
349:     # Equivalent to edge padding for single value, so do that instead
350:     if num == 1:
351:         return _prepend_edge(arr, pad_amt, axis)
352: 
353:     # Use entire array if `num` is too large
354:     if num is not None:
355:         if num >= arr.shape[axis]:
356:             num = None
357: 
358:     # Slice a chunk from the edge to calculate stats on
359:     max_slice = tuple(slice(None) if i != axis else slice(num)
360:                       for (i, x) in enumerate(arr.shape))
361: 
362:     # Shape to restore singleton dimension after slicing
363:     pad_singleton = tuple(x if i != axis else 1
364:                           for (i, x) in enumerate(arr.shape))
365: 
366:     # Extract slice, calculate max, reshape to add singleton dimension back
367:     max_chunk = arr[max_slice].max(axis=axis).reshape(pad_singleton)
368: 
369:     # Concatenate `arr` with `max_chunk`, extended along `axis` by `pad_amt`
370:     return np.concatenate((max_chunk.repeat(pad_amt, axis=axis), arr),
371:                           axis=axis)
372: 
373: 
374: def _append_max(arr, pad_amt, num, axis=-1):
375:     '''
376:     Pad one `axis` of `arr` with the maximum of the last `num` elements.
377: 
378:     Parameters
379:     ----------
380:     arr : ndarray
381:         Input array of arbitrary shape.
382:     pad_amt : int
383:         Amount of padding to append.
384:     num : int
385:         Depth into `arr` along `axis` to calculate maximum.
386:         Range: [1, `arr.shape[axis]`] or None (entire axis)
387:     axis : int
388:         Axis along which to pad `arr`.
389: 
390:     Returns
391:     -------
392:     padarr : ndarray
393:         Output array, with `pad_amt` values appended along `axis`. The
394:         appended region is the maximum of the final `num` values along `axis`.
395: 
396:     '''
397:     if pad_amt == 0:
398:         return arr
399: 
400:     # Equivalent to edge padding for single value, so do that instead
401:     if num == 1:
402:         return _append_edge(arr, pad_amt, axis)
403: 
404:     # Use entire array if `num` is too large
405:     if num is not None:
406:         if num >= arr.shape[axis]:
407:             num = None
408: 
409:     # Slice a chunk from the edge to calculate stats on
410:     end = arr.shape[axis] - 1
411:     if num is not None:
412:         max_slice = tuple(
413:             slice(None) if i != axis else slice(end, end - num, -1)
414:             for (i, x) in enumerate(arr.shape))
415:     else:
416:         max_slice = tuple(slice(None) for x in arr.shape)
417: 
418:     # Shape to restore singleton dimension after slicing
419:     pad_singleton = tuple(x if i != axis else 1
420:                           for (i, x) in enumerate(arr.shape))
421: 
422:     # Extract slice, calculate max, reshape to add singleton dimension back
423:     max_chunk = arr[max_slice].max(axis=axis).reshape(pad_singleton)
424: 
425:     # Concatenate `arr` with `max_chunk`, extended along `axis` by `pad_amt`
426:     return np.concatenate((arr, max_chunk.repeat(pad_amt, axis=axis)),
427:                           axis=axis)
428: 
429: 
430: def _prepend_mean(arr, pad_amt, num, axis=-1):
431:     '''
432:     Prepend `pad_amt` mean values along `axis`.
433: 
434:     Parameters
435:     ----------
436:     arr : ndarray
437:         Input array of arbitrary shape.
438:     pad_amt : int
439:         Amount of padding to prepend.
440:     num : int
441:         Depth into `arr` along `axis` to calculate mean.
442:         Range: [1, `arr.shape[axis]`] or None (entire axis)
443:     axis : int
444:         Axis along which to pad `arr`.
445: 
446:     Returns
447:     -------
448:     padarr : ndarray
449:         Output array, with `pad_amt` values prepended along `axis`. The
450:         prepended region is the mean of the first `num` values along `axis`.
451: 
452:     '''
453:     if pad_amt == 0:
454:         return arr
455: 
456:     # Equivalent to edge padding for single value, so do that instead
457:     if num == 1:
458:         return _prepend_edge(arr, pad_amt, axis)
459: 
460:     # Use entire array if `num` is too large
461:     if num is not None:
462:         if num >= arr.shape[axis]:
463:             num = None
464: 
465:     # Slice a chunk from the edge to calculate stats on
466:     mean_slice = tuple(slice(None) if i != axis else slice(num)
467:                        for (i, x) in enumerate(arr.shape))
468: 
469:     # Shape to restore singleton dimension after slicing
470:     pad_singleton = tuple(x if i != axis else 1
471:                           for (i, x) in enumerate(arr.shape))
472: 
473:     # Extract slice, calculate mean, reshape to add singleton dimension back
474:     mean_chunk = arr[mean_slice].mean(axis).reshape(pad_singleton)
475:     _round_ifneeded(mean_chunk, arr.dtype)
476: 
477:     # Concatenate `arr` with `mean_chunk`, extended along `axis` by `pad_amt`
478:     return np.concatenate((mean_chunk.repeat(pad_amt, axis).astype(arr.dtype),
479:                            arr), axis=axis)
480: 
481: 
482: def _append_mean(arr, pad_amt, num, axis=-1):
483:     '''
484:     Append `pad_amt` mean values along `axis`.
485: 
486:     Parameters
487:     ----------
488:     arr : ndarray
489:         Input array of arbitrary shape.
490:     pad_amt : int
491:         Amount of padding to append.
492:     num : int
493:         Depth into `arr` along `axis` to calculate mean.
494:         Range: [1, `arr.shape[axis]`] or None (entire axis)
495:     axis : int
496:         Axis along which to pad `arr`.
497: 
498:     Returns
499:     -------
500:     padarr : ndarray
501:         Output array, with `pad_amt` values appended along `axis`. The
502:         appended region is the maximum of the final `num` values along `axis`.
503: 
504:     '''
505:     if pad_amt == 0:
506:         return arr
507: 
508:     # Equivalent to edge padding for single value, so do that instead
509:     if num == 1:
510:         return _append_edge(arr, pad_amt, axis)
511: 
512:     # Use entire array if `num` is too large
513:     if num is not None:
514:         if num >= arr.shape[axis]:
515:             num = None
516: 
517:     # Slice a chunk from the edge to calculate stats on
518:     end = arr.shape[axis] - 1
519:     if num is not None:
520:         mean_slice = tuple(
521:             slice(None) if i != axis else slice(end, end - num, -1)
522:             for (i, x) in enumerate(arr.shape))
523:     else:
524:         mean_slice = tuple(slice(None) for x in arr.shape)
525: 
526:     # Shape to restore singleton dimension after slicing
527:     pad_singleton = tuple(x if i != axis else 1
528:                           for (i, x) in enumerate(arr.shape))
529: 
530:     # Extract slice, calculate mean, reshape to add singleton dimension back
531:     mean_chunk = arr[mean_slice].mean(axis=axis).reshape(pad_singleton)
532:     _round_ifneeded(mean_chunk, arr.dtype)
533: 
534:     # Concatenate `arr` with `mean_chunk`, extended along `axis` by `pad_amt`
535:     return np.concatenate(
536:         (arr, mean_chunk.repeat(pad_amt, axis).astype(arr.dtype)), axis=axis)
537: 
538: 
539: def _prepend_med(arr, pad_amt, num, axis=-1):
540:     '''
541:     Prepend `pad_amt` median values along `axis`.
542: 
543:     Parameters
544:     ----------
545:     arr : ndarray
546:         Input array of arbitrary shape.
547:     pad_amt : int
548:         Amount of padding to prepend.
549:     num : int
550:         Depth into `arr` along `axis` to calculate median.
551:         Range: [1, `arr.shape[axis]`] or None (entire axis)
552:     axis : int
553:         Axis along which to pad `arr`.
554: 
555:     Returns
556:     -------
557:     padarr : ndarray
558:         Output array, with `pad_amt` values prepended along `axis`. The
559:         prepended region is the median of the first `num` values along `axis`.
560: 
561:     '''
562:     if pad_amt == 0:
563:         return arr
564: 
565:     # Equivalent to edge padding for single value, so do that instead
566:     if num == 1:
567:         return _prepend_edge(arr, pad_amt, axis)
568: 
569:     # Use entire array if `num` is too large
570:     if num is not None:
571:         if num >= arr.shape[axis]:
572:             num = None
573: 
574:     # Slice a chunk from the edge to calculate stats on
575:     med_slice = tuple(slice(None) if i != axis else slice(num)
576:                       for (i, x) in enumerate(arr.shape))
577: 
578:     # Shape to restore singleton dimension after slicing
579:     pad_singleton = tuple(x if i != axis else 1
580:                           for (i, x) in enumerate(arr.shape))
581: 
582:     # Extract slice, calculate median, reshape to add singleton dimension back
583:     med_chunk = np.median(arr[med_slice], axis=axis).reshape(pad_singleton)
584:     _round_ifneeded(med_chunk, arr.dtype)
585: 
586:     # Concatenate `arr` with `med_chunk`, extended along `axis` by `pad_amt`
587:     return np.concatenate(
588:         (med_chunk.repeat(pad_amt, axis).astype(arr.dtype), arr), axis=axis)
589: 
590: 
591: def _append_med(arr, pad_amt, num, axis=-1):
592:     '''
593:     Append `pad_amt` median values along `axis`.
594: 
595:     Parameters
596:     ----------
597:     arr : ndarray
598:         Input array of arbitrary shape.
599:     pad_amt : int
600:         Amount of padding to append.
601:     num : int
602:         Depth into `arr` along `axis` to calculate median.
603:         Range: [1, `arr.shape[axis]`] or None (entire axis)
604:     axis : int
605:         Axis along which to pad `arr`.
606: 
607:     Returns
608:     -------
609:     padarr : ndarray
610:         Output array, with `pad_amt` values appended along `axis`. The
611:         appended region is the median of the final `num` values along `axis`.
612: 
613:     '''
614:     if pad_amt == 0:
615:         return arr
616: 
617:     # Equivalent to edge padding for single value, so do that instead
618:     if num == 1:
619:         return _append_edge(arr, pad_amt, axis)
620: 
621:     # Use entire array if `num` is too large
622:     if num is not None:
623:         if num >= arr.shape[axis]:
624:             num = None
625: 
626:     # Slice a chunk from the edge to calculate stats on
627:     end = arr.shape[axis] - 1
628:     if num is not None:
629:         med_slice = tuple(
630:             slice(None) if i != axis else slice(end, end - num, -1)
631:             for (i, x) in enumerate(arr.shape))
632:     else:
633:         med_slice = tuple(slice(None) for x in arr.shape)
634: 
635:     # Shape to restore singleton dimension after slicing
636:     pad_singleton = tuple(x if i != axis else 1
637:                           for (i, x) in enumerate(arr.shape))
638: 
639:     # Extract slice, calculate median, reshape to add singleton dimension back
640:     med_chunk = np.median(arr[med_slice], axis=axis).reshape(pad_singleton)
641:     _round_ifneeded(med_chunk, arr.dtype)
642: 
643:     # Concatenate `arr` with `med_chunk`, extended along `axis` by `pad_amt`
644:     return np.concatenate(
645:         (arr, med_chunk.repeat(pad_amt, axis).astype(arr.dtype)), axis=axis)
646: 
647: 
648: def _prepend_min(arr, pad_amt, num, axis=-1):
649:     '''
650:     Prepend `pad_amt` minimum values along `axis`.
651: 
652:     Parameters
653:     ----------
654:     arr : ndarray
655:         Input array of arbitrary shape.
656:     pad_amt : int
657:         Amount of padding to prepend.
658:     num : int
659:         Depth into `arr` along `axis` to calculate minimum.
660:         Range: [1, `arr.shape[axis]`] or None (entire axis)
661:     axis : int
662:         Axis along which to pad `arr`.
663: 
664:     Returns
665:     -------
666:     padarr : ndarray
667:         Output array, with `pad_amt` values prepended along `axis`. The
668:         prepended region is the minimum of the first `num` values along
669:         `axis`.
670: 
671:     '''
672:     if pad_amt == 0:
673:         return arr
674: 
675:     # Equivalent to edge padding for single value, so do that instead
676:     if num == 1:
677:         return _prepend_edge(arr, pad_amt, axis)
678: 
679:     # Use entire array if `num` is too large
680:     if num is not None:
681:         if num >= arr.shape[axis]:
682:             num = None
683: 
684:     # Slice a chunk from the edge to calculate stats on
685:     min_slice = tuple(slice(None) if i != axis else slice(num)
686:                       for (i, x) in enumerate(arr.shape))
687: 
688:     # Shape to restore singleton dimension after slicing
689:     pad_singleton = tuple(x if i != axis else 1
690:                           for (i, x) in enumerate(arr.shape))
691: 
692:     # Extract slice, calculate min, reshape to add singleton dimension back
693:     min_chunk = arr[min_slice].min(axis=axis).reshape(pad_singleton)
694: 
695:     # Concatenate `arr` with `min_chunk`, extended along `axis` by `pad_amt`
696:     return np.concatenate((min_chunk.repeat(pad_amt, axis=axis), arr),
697:                           axis=axis)
698: 
699: 
700: def _append_min(arr, pad_amt, num, axis=-1):
701:     '''
702:     Append `pad_amt` median values along `axis`.
703: 
704:     Parameters
705:     ----------
706:     arr : ndarray
707:         Input array of arbitrary shape.
708:     pad_amt : int
709:         Amount of padding to append.
710:     num : int
711:         Depth into `arr` along `axis` to calculate minimum.
712:         Range: [1, `arr.shape[axis]`] or None (entire axis)
713:     axis : int
714:         Axis along which to pad `arr`.
715: 
716:     Returns
717:     -------
718:     padarr : ndarray
719:         Output array, with `pad_amt` values appended along `axis`. The
720:         appended region is the minimum of the final `num` values along `axis`.
721: 
722:     '''
723:     if pad_amt == 0:
724:         return arr
725: 
726:     # Equivalent to edge padding for single value, so do that instead
727:     if num == 1:
728:         return _append_edge(arr, pad_amt, axis)
729: 
730:     # Use entire array if `num` is too large
731:     if num is not None:
732:         if num >= arr.shape[axis]:
733:             num = None
734: 
735:     # Slice a chunk from the edge to calculate stats on
736:     end = arr.shape[axis] - 1
737:     if num is not None:
738:         min_slice = tuple(
739:             slice(None) if i != axis else slice(end, end - num, -1)
740:             for (i, x) in enumerate(arr.shape))
741:     else:
742:         min_slice = tuple(slice(None) for x in arr.shape)
743: 
744:     # Shape to restore singleton dimension after slicing
745:     pad_singleton = tuple(x if i != axis else 1
746:                           for (i, x) in enumerate(arr.shape))
747: 
748:     # Extract slice, calculate min, reshape to add singleton dimension back
749:     min_chunk = arr[min_slice].min(axis=axis).reshape(pad_singleton)
750: 
751:     # Concatenate `arr` with `min_chunk`, extended along `axis` by `pad_amt`
752:     return np.concatenate((arr, min_chunk.repeat(pad_amt, axis=axis)),
753:                           axis=axis)
754: 
755: 
756: def _pad_ref(arr, pad_amt, method, axis=-1):
757:     '''
758:     Pad `axis` of `arr` by reflection.
759: 
760:     Parameters
761:     ----------
762:     arr : ndarray
763:         Input array of arbitrary shape.
764:     pad_amt : tuple of ints, length 2
765:         Padding to (prepend, append) along `axis`.
766:     method : str
767:         Controls method of reflection; options are 'even' or 'odd'.
768:     axis : int
769:         Axis along which to pad `arr`.
770: 
771:     Returns
772:     -------
773:     padarr : ndarray
774:         Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`
775:         values appended along `axis`. Both regions are padded with reflected
776:         values from the original array.
777: 
778:     Notes
779:     -----
780:     This algorithm does not pad with repetition, i.e. the edges are not
781:     repeated in the reflection. For that behavior, use `mode='symmetric'`.
782: 
783:     The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
784:     single function, lest the indexing tricks in non-integer multiples of the
785:     original shape would violate repetition in the final iteration.
786: 
787:     '''
788:     # Implicit booleanness to test for zero (or None) in any scalar type
789:     if pad_amt[0] == 0 and pad_amt[1] == 0:
790:         return arr
791: 
792:     ##########################################################################
793:     # Prepended region
794: 
795:     # Slice off a reverse indexed chunk from near edge to pad `arr` before
796:     ref_slice = tuple(slice(None) if i != axis else slice(pad_amt[0], 0, -1)
797:                       for (i, x) in enumerate(arr.shape))
798: 
799:     ref_chunk1 = arr[ref_slice]
800: 
801:     # Shape to restore singleton dimension after slicing
802:     pad_singleton = tuple(x if i != axis else 1
803:                           for (i, x) in enumerate(arr.shape))
804:     if pad_amt[0] == 1:
805:         ref_chunk1 = ref_chunk1.reshape(pad_singleton)
806: 
807:     # Memory/computationally more expensive, only do this if `method='odd'`
808:     if 'odd' in method and pad_amt[0] > 0:
809:         edge_slice1 = tuple(slice(None) if i != axis else 0
810:                             for (i, x) in enumerate(arr.shape))
811:         edge_chunk = arr[edge_slice1].reshape(pad_singleton)
812:         ref_chunk1 = 2 * edge_chunk - ref_chunk1
813:         del edge_chunk
814: 
815:     ##########################################################################
816:     # Appended region
817: 
818:     # Slice off a reverse indexed chunk from far edge to pad `arr` after
819:     start = arr.shape[axis] - pad_amt[1] - 1
820:     end = arr.shape[axis] - 1
821:     ref_slice = tuple(slice(None) if i != axis else slice(start, end)
822:                       for (i, x) in enumerate(arr.shape))
823:     rev_idx = tuple(slice(None) if i != axis else slice(None, None, -1)
824:                     for (i, x) in enumerate(arr.shape))
825:     ref_chunk2 = arr[ref_slice][rev_idx]
826: 
827:     if pad_amt[1] == 1:
828:         ref_chunk2 = ref_chunk2.reshape(pad_singleton)
829: 
830:     if 'odd' in method:
831:         edge_slice2 = tuple(slice(None) if i != axis else -1
832:                             for (i, x) in enumerate(arr.shape))
833:         edge_chunk = arr[edge_slice2].reshape(pad_singleton)
834:         ref_chunk2 = 2 * edge_chunk - ref_chunk2
835:         del edge_chunk
836: 
837:     # Concatenate `arr` with both chunks, extending along `axis`
838:     return np.concatenate((ref_chunk1, arr, ref_chunk2), axis=axis)
839: 
840: 
841: def _pad_sym(arr, pad_amt, method, axis=-1):
842:     '''
843:     Pad `axis` of `arr` by symmetry.
844: 
845:     Parameters
846:     ----------
847:     arr : ndarray
848:         Input array of arbitrary shape.
849:     pad_amt : tuple of ints, length 2
850:         Padding to (prepend, append) along `axis`.
851:     method : str
852:         Controls method of symmetry; options are 'even' or 'odd'.
853:     axis : int
854:         Axis along which to pad `arr`.
855: 
856:     Returns
857:     -------
858:     padarr : ndarray
859:         Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`
860:         values appended along `axis`. Both regions are padded with symmetric
861:         values from the original array.
862: 
863:     Notes
864:     -----
865:     This algorithm DOES pad with repetition, i.e. the edges are repeated.
866:     For padding without repeated edges, use `mode='reflect'`.
867: 
868:     The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
869:     single function, lest the indexing tricks in non-integer multiples of the
870:     original shape would violate repetition in the final iteration.
871: 
872:     '''
873:     # Implicit booleanness to test for zero (or None) in any scalar type
874:     if pad_amt[0] == 0 and pad_amt[1] == 0:
875:         return arr
876: 
877:     ##########################################################################
878:     # Prepended region
879: 
880:     # Slice off a reverse indexed chunk from near edge to pad `arr` before
881:     sym_slice = tuple(slice(None) if i != axis else slice(0, pad_amt[0])
882:                       for (i, x) in enumerate(arr.shape))
883:     rev_idx = tuple(slice(None) if i != axis else slice(None, None, -1)
884:                     for (i, x) in enumerate(arr.shape))
885:     sym_chunk1 = arr[sym_slice][rev_idx]
886: 
887:     # Shape to restore singleton dimension after slicing
888:     pad_singleton = tuple(x if i != axis else 1
889:                           for (i, x) in enumerate(arr.shape))
890:     if pad_amt[0] == 1:
891:         sym_chunk1 = sym_chunk1.reshape(pad_singleton)
892: 
893:     # Memory/computationally more expensive, only do this if `method='odd'`
894:     if 'odd' in method and pad_amt[0] > 0:
895:         edge_slice1 = tuple(slice(None) if i != axis else 0
896:                             for (i, x) in enumerate(arr.shape))
897:         edge_chunk = arr[edge_slice1].reshape(pad_singleton)
898:         sym_chunk1 = 2 * edge_chunk - sym_chunk1
899:         del edge_chunk
900: 
901:     ##########################################################################
902:     # Appended region
903: 
904:     # Slice off a reverse indexed chunk from far edge to pad `arr` after
905:     start = arr.shape[axis] - pad_amt[1]
906:     end = arr.shape[axis]
907:     sym_slice = tuple(slice(None) if i != axis else slice(start, end)
908:                       for (i, x) in enumerate(arr.shape))
909:     sym_chunk2 = arr[sym_slice][rev_idx]
910: 
911:     if pad_amt[1] == 1:
912:         sym_chunk2 = sym_chunk2.reshape(pad_singleton)
913: 
914:     if 'odd' in method:
915:         edge_slice2 = tuple(slice(None) if i != axis else -1
916:                             for (i, x) in enumerate(arr.shape))
917:         edge_chunk = arr[edge_slice2].reshape(pad_singleton)
918:         sym_chunk2 = 2 * edge_chunk - sym_chunk2
919:         del edge_chunk
920: 
921:     # Concatenate `arr` with both chunks, extending along `axis`
922:     return np.concatenate((sym_chunk1, arr, sym_chunk2), axis=axis)
923: 
924: 
925: def _pad_wrap(arr, pad_amt, axis=-1):
926:     '''
927:     Pad `axis` of `arr` via wrapping.
928: 
929:     Parameters
930:     ----------
931:     arr : ndarray
932:         Input array of arbitrary shape.
933:     pad_amt : tuple of ints, length 2
934:         Padding to (prepend, append) along `axis`.
935:     axis : int
936:         Axis along which to pad `arr`.
937: 
938:     Returns
939:     -------
940:     padarr : ndarray
941:         Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`
942:         values appended along `axis`. Both regions are padded wrapped values
943:         from the opposite end of `axis`.
944: 
945:     Notes
946:     -----
947:     This method of padding is also known as 'tile' or 'tiling'.
948: 
949:     The modes 'reflect', 'symmetric', and 'wrap' must be padded with a
950:     single function, lest the indexing tricks in non-integer multiples of the
951:     original shape would violate repetition in the final iteration.
952: 
953:     '''
954:     # Implicit booleanness to test for zero (or None) in any scalar type
955:     if pad_amt[0] == 0 and pad_amt[1] == 0:
956:         return arr
957: 
958:     ##########################################################################
959:     # Prepended region
960: 
961:     # Slice off a reverse indexed chunk from near edge to pad `arr` before
962:     start = arr.shape[axis] - pad_amt[0]
963:     end = arr.shape[axis]
964:     wrap_slice = tuple(slice(None) if i != axis else slice(start, end)
965:                        for (i, x) in enumerate(arr.shape))
966:     wrap_chunk1 = arr[wrap_slice]
967: 
968:     # Shape to restore singleton dimension after slicing
969:     pad_singleton = tuple(x if i != axis else 1
970:                           for (i, x) in enumerate(arr.shape))
971:     if pad_amt[0] == 1:
972:         wrap_chunk1 = wrap_chunk1.reshape(pad_singleton)
973: 
974:     ##########################################################################
975:     # Appended region
976: 
977:     # Slice off a reverse indexed chunk from far edge to pad `arr` after
978:     wrap_slice = tuple(slice(None) if i != axis else slice(0, pad_amt[1])
979:                        for (i, x) in enumerate(arr.shape))
980:     wrap_chunk2 = arr[wrap_slice]
981: 
982:     if pad_amt[1] == 1:
983:         wrap_chunk2 = wrap_chunk2.reshape(pad_singleton)
984: 
985:     # Concatenate `arr` with both chunks, extending along `axis`
986:     return np.concatenate((wrap_chunk1, arr, wrap_chunk2), axis=axis)
987: 
988: 
989: def _normalize_shape(ndarray, shape, cast_to_int=True):
990:     '''
991:     Private function which does some checks and normalizes the possibly
992:     much simpler representations of 'pad_width', 'stat_length',
993:     'constant_values', 'end_values'.
994: 
995:     Parameters
996:     ----------
997:     narray : ndarray
998:         Input ndarray
999:     shape : {sequence, array_like, float, int}, optional
1000:         The width of padding (pad_width), the number of elements on the
1001:         edge of the narray used for statistics (stat_length), the constant
1002:         value(s) to use when filling padded regions (constant_values), or the
1003:         endpoint target(s) for linear ramps (end_values).
1004:         ((before_1, after_1), ... (before_N, after_N)) unique number of
1005:         elements for each axis where `N` is rank of `narray`.
1006:         ((before, after),) yields same before and after constants for each
1007:         axis.
1008:         (constant,) or val is a shortcut for before = after = constant for
1009:         all axes.
1010:     cast_to_int : bool, optional
1011:         Controls if values in ``shape`` will be rounded and cast to int
1012:         before being returned.
1013: 
1014:     Returns
1015:     -------
1016:     normalized_shape : tuple of tuples
1017:         val                               => ((val, val), (val, val), ...)
1018:         [[val1, val2], [val3, val4], ...] => ((val1, val2), (val3, val4), ...)
1019:         ((val1, val2), (val3, val4), ...) => no change
1020:         [[val1, val2], ]                  => ((val1, val2), (val1, val2), ...)
1021:         ((val1, val2), )                  => ((val1, val2), (val1, val2), ...)
1022:         [[val ,     ], ]                  => ((val, val), (val, val), ...)
1023:         ((val ,     ), )                  => ((val, val), (val, val), ...)
1024: 
1025:     '''
1026:     ndims = ndarray.ndim
1027: 
1028:     # Shortcut shape=None
1029:     if shape is None:
1030:         return ((None, None), ) * ndims
1031: 
1032:     # Convert any input `info` to a NumPy array
1033:     arr = np.asarray(shape)
1034: 
1035:     # Switch based on what input looks like
1036:     if arr.ndim <= 1:
1037:         if arr.shape == () or arr.shape == (1,):
1038:             # Single scalar input
1039:             #   Create new array of ones, multiply by the scalar
1040:             arr = np.ones((ndims, 2), dtype=ndarray.dtype) * arr
1041:         elif arr.shape == (2,):
1042:             # Apply padding (before, after) each axis
1043:             #   Create new axis 0, repeat along it for every axis
1044:             arr = arr[np.newaxis, :].repeat(ndims, axis=0)
1045:         else:
1046:             fmt = "Unable to create correctly shaped tuple from %s"
1047:             raise ValueError(fmt % (shape,))
1048: 
1049:     elif arr.ndim == 2:
1050:         if arr.shape[1] == 1 and arr.shape[0] == ndims:
1051:             # Padded before and after by the same amount
1052:             arr = arr.repeat(2, axis=1)
1053:         elif arr.shape[0] == ndims:
1054:             # Input correctly formatted, pass it on as `arr`
1055:             arr = shape
1056:         else:
1057:             fmt = "Unable to create correctly shaped tuple from %s"
1058:             raise ValueError(fmt % (shape,))
1059: 
1060:     else:
1061:         fmt = "Unable to create correctly shaped tuple from %s"
1062:         raise ValueError(fmt % (shape,))
1063: 
1064:     # Cast if necessary
1065:     if cast_to_int is True:
1066:         arr = np.round(arr).astype(int)
1067: 
1068:     # Convert list of lists to tuple of tuples
1069:     return tuple(tuple(axis) for axis in arr.tolist())
1070: 
1071: 
1072: def _validate_lengths(narray, number_elements):
1073:     '''
1074:     Private function which does some checks and reformats pad_width and
1075:     stat_length using _normalize_shape.
1076: 
1077:     Parameters
1078:     ----------
1079:     narray : ndarray
1080:         Input ndarray
1081:     number_elements : {sequence, int}, optional
1082:         The width of padding (pad_width) or the number of elements on the edge
1083:         of the narray used for statistics (stat_length).
1084:         ((before_1, after_1), ... (before_N, after_N)) unique number of
1085:         elements for each axis.
1086:         ((before, after),) yields same before and after constants for each
1087:         axis.
1088:         (constant,) or int is a shortcut for before = after = constant for all
1089:         axes.
1090: 
1091:     Returns
1092:     -------
1093:     _validate_lengths : tuple of tuples
1094:         int                               => ((int, int), (int, int), ...)
1095:         [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)
1096:         ((int1, int2), (int3, int4), ...) => no change
1097:         [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...)
1098:         ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)
1099:         [[int ,     ], ]                  => ((int, int), (int, int), ...)
1100:         ((int ,     ), )                  => ((int, int), (int, int), ...)
1101: 
1102:     '''
1103:     normshp = _normalize_shape(narray, number_elements)
1104:     for i in normshp:
1105:         chk = [1 if x is None else x for x in i]
1106:         chk = [1 if x >= 0 else -1 for x in chk]
1107:         if (chk[0] < 0) or (chk[1] < 0):
1108:             fmt = "%s cannot contain negative values."
1109:             raise ValueError(fmt % (number_elements,))
1110:     return normshp
1111: 
1112: 
1113: ###############################################################################
1114: # Public functions
1115: 
1116: 
1117: def pad(array, pad_width, mode, **kwargs):
1118:     '''
1119:     Pads an array.
1120: 
1121:     Parameters
1122:     ----------
1123:     array : array_like of rank N
1124:         Input array
1125:     pad_width : {sequence, array_like, int}
1126:         Number of values padded to the edges of each axis.
1127:         ((before_1, after_1), ... (before_N, after_N)) unique pad widths
1128:         for each axis.
1129:         ((before, after),) yields same before and after pad for each axis.
1130:         (pad,) or int is a shortcut for before = after = pad width for all
1131:         axes.
1132:     mode : str or function
1133:         One of the following string values or a user supplied function.
1134: 
1135:         'constant'
1136:             Pads with a constant value.
1137:         'edge'
1138:             Pads with the edge values of array.
1139:         'linear_ramp'
1140:             Pads with the linear ramp between end_value and the
1141:             array edge value.
1142:         'maximum'
1143:             Pads with the maximum value of all or part of the
1144:             vector along each axis.
1145:         'mean'
1146:             Pads with the mean value of all or part of the
1147:             vector along each axis.
1148:         'median'
1149:             Pads with the median value of all or part of the
1150:             vector along each axis.
1151:         'minimum'
1152:             Pads with the minimum value of all or part of the
1153:             vector along each axis.
1154:         'reflect'
1155:             Pads with the reflection of the vector mirrored on
1156:             the first and last values of the vector along each
1157:             axis.
1158:         'symmetric'
1159:             Pads with the reflection of the vector mirrored
1160:             along the edge of the array.
1161:         'wrap'
1162:             Pads with the wrap of the vector along the axis.
1163:             The first values are used to pad the end and the
1164:             end values are used to pad the beginning.
1165:         <function>
1166:             Padding function, see Notes.
1167:     stat_length : sequence or int, optional
1168:         Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
1169:         values at edge of each axis used to calculate the statistic value.
1170: 
1171:         ((before_1, after_1), ... (before_N, after_N)) unique statistic
1172:         lengths for each axis.
1173: 
1174:         ((before, after),) yields same before and after statistic lengths
1175:         for each axis.
1176: 
1177:         (stat_length,) or int is a shortcut for before = after = statistic
1178:         length for all axes.
1179: 
1180:         Default is ``None``, to use the entire axis.
1181:     constant_values : sequence or int, optional
1182:         Used in 'constant'.  The values to set the padded values for each
1183:         axis.
1184: 
1185:         ((before_1, after_1), ... (before_N, after_N)) unique pad constants
1186:         for each axis.
1187: 
1188:         ((before, after),) yields same before and after constants for each
1189:         axis.
1190: 
1191:         (constant,) or int is a shortcut for before = after = constant for
1192:         all axes.
1193: 
1194:         Default is 0.
1195:     end_values : sequence or int, optional
1196:         Used in 'linear_ramp'.  The values used for the ending value of the
1197:         linear_ramp and that will form the edge of the padded array.
1198: 
1199:         ((before_1, after_1), ... (before_N, after_N)) unique end values
1200:         for each axis.
1201: 
1202:         ((before, after),) yields same before and after end values for each
1203:         axis.
1204: 
1205:         (constant,) or int is a shortcut for before = after = end value for
1206:         all axes.
1207: 
1208:         Default is 0.
1209:     reflect_type : {'even', 'odd'}, optional
1210:         Used in 'reflect', and 'symmetric'.  The 'even' style is the
1211:         default with an unaltered reflection around the edge value.  For
1212:         the 'odd' style, the extented part of the array is created by
1213:         subtracting the reflected values from two times the edge value.
1214: 
1215:     Returns
1216:     -------
1217:     pad : ndarray
1218:         Padded array of rank equal to `array` with shape increased
1219:         according to `pad_width`.
1220: 
1221:     Notes
1222:     -----
1223:     .. versionadded:: 1.7.0
1224: 
1225:     For an array with rank greater than 1, some of the padding of later
1226:     axes is calculated from padding of previous axes.  This is easiest to
1227:     think about with a rank 2 array where the corners of the padded array
1228:     are calculated by using padded values from the first axis.
1229: 
1230:     The padding function, if used, should return a rank 1 array equal in
1231:     length to the vector argument with padded values replaced. It has the
1232:     following signature::
1233: 
1234:         padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
1235: 
1236:     where
1237: 
1238:         vector : ndarray
1239:             A rank 1 array already padded with zeros.  Padded values are
1240:             vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].
1241:         iaxis_pad_width : tuple
1242:             A 2-tuple of ints, iaxis_pad_width[0] represents the number of
1243:             values padded at the beginning of vector where
1244:             iaxis_pad_width[1] represents the number of values padded at
1245:             the end of vector.
1246:         iaxis : int
1247:             The axis currently being calculated.
1248:         kwargs : misc
1249:             Any keyword arguments the function requires.
1250: 
1251:     Examples
1252:     --------
1253:     >>> a = [1, 2, 3, 4, 5]
1254:     >>> np.lib.pad(a, (2,3), 'constant', constant_values=(4, 6))
1255:     array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])
1256: 
1257:     >>> np.lib.pad(a, (2, 3), 'edge')
1258:     array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])
1259: 
1260:     >>> np.lib.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
1261:     array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])
1262: 
1263:     >>> np.lib.pad(a, (2,), 'maximum')
1264:     array([5, 5, 1, 2, 3, 4, 5, 5, 5])
1265: 
1266:     >>> np.lib.pad(a, (2,), 'mean')
1267:     array([3, 3, 1, 2, 3, 4, 5, 3, 3])
1268: 
1269:     >>> np.lib.pad(a, (2,), 'median')
1270:     array([3, 3, 1, 2, 3, 4, 5, 3, 3])
1271: 
1272:     >>> a = [[1, 2], [3, 4]]
1273:     >>> np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')
1274:     array([[1, 1, 1, 2, 1, 1, 1],
1275:            [1, 1, 1, 2, 1, 1, 1],
1276:            [1, 1, 1, 2, 1, 1, 1],
1277:            [1, 1, 1, 2, 1, 1, 1],
1278:            [3, 3, 3, 4, 3, 3, 3],
1279:            [1, 1, 1, 2, 1, 1, 1],
1280:            [1, 1, 1, 2, 1, 1, 1]])
1281: 
1282:     >>> a = [1, 2, 3, 4, 5]
1283:     >>> np.lib.pad(a, (2, 3), 'reflect')
1284:     array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])
1285: 
1286:     >>> np.lib.pad(a, (2, 3), 'reflect', reflect_type='odd')
1287:     array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])
1288: 
1289:     >>> np.lib.pad(a, (2, 3), 'symmetric')
1290:     array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])
1291: 
1292:     >>> np.lib.pad(a, (2, 3), 'symmetric', reflect_type='odd')
1293:     array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])
1294: 
1295:     >>> np.lib.pad(a, (2, 3), 'wrap')
1296:     array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])
1297: 
1298:     >>> def padwithtens(vector, pad_width, iaxis, kwargs):
1299:     ...     vector[:pad_width[0]] = 10
1300:     ...     vector[-pad_width[1]:] = 10
1301:     ...     return vector
1302: 
1303:     >>> a = np.arange(6)
1304:     >>> a = a.reshape((2, 3))
1305: 
1306:     >>> np.lib.pad(a, 2, padwithtens)
1307:     array([[10, 10, 10, 10, 10, 10, 10],
1308:            [10, 10, 10, 10, 10, 10, 10],
1309:            [10, 10,  0,  1,  2, 10, 10],
1310:            [10, 10,  3,  4,  5, 10, 10],
1311:            [10, 10, 10, 10, 10, 10, 10],
1312:            [10, 10, 10, 10, 10, 10, 10]])
1313:     '''
1314:     if not np.asarray(pad_width).dtype.kind == 'i':
1315:         raise TypeError('`pad_width` must be of integral type.')
1316: 
1317:     narray = np.array(array)
1318:     pad_width = _validate_lengths(narray, pad_width)
1319: 
1320:     allowedkwargs = {
1321:         'constant': ['constant_values'],
1322:         'edge': [],
1323:         'linear_ramp': ['end_values'],
1324:         'maximum': ['stat_length'],
1325:         'mean': ['stat_length'],
1326:         'median': ['stat_length'],
1327:         'minimum': ['stat_length'],
1328:         'reflect': ['reflect_type'],
1329:         'symmetric': ['reflect_type'],
1330:         'wrap': [],
1331:         }
1332: 
1333:     kwdefaults = {
1334:         'stat_length': None,
1335:         'constant_values': 0,
1336:         'end_values': 0,
1337:         'reflect_type': 'even',
1338:         }
1339: 
1340:     if isinstance(mode, np.compat.basestring):
1341:         # Make sure have allowed kwargs appropriate for mode
1342:         for key in kwargs:
1343:             if key not in allowedkwargs[mode]:
1344:                 raise ValueError('%s keyword not in allowed keywords %s' %
1345:                                  (key, allowedkwargs[mode]))
1346: 
1347:         # Set kwarg defaults
1348:         for kw in allowedkwargs[mode]:
1349:             kwargs.setdefault(kw, kwdefaults[kw])
1350: 
1351:         # Need to only normalize particular keywords.
1352:         for i in kwargs:
1353:             if i == 'stat_length':
1354:                 kwargs[i] = _validate_lengths(narray, kwargs[i])
1355:             if i in ['end_values', 'constant_values']:
1356:                 kwargs[i] = _normalize_shape(narray, kwargs[i],
1357:                                              cast_to_int=False)
1358:     else:
1359:         # Drop back to old, slower np.apply_along_axis mode for user-supplied
1360:         # vector function
1361:         function = mode
1362: 
1363:         # Create a new padded array
1364:         rank = list(range(len(narray.shape)))
1365:         total_dim_increase = [np.sum(pad_width[i]) for i in rank]
1366:         offset_slices = [slice(pad_width[i][0],
1367:                                pad_width[i][0] + narray.shape[i])
1368:                          for i in rank]
1369:         new_shape = np.array(narray.shape) + total_dim_increase
1370:         newmat = np.zeros(new_shape, narray.dtype)
1371: 
1372:         # Insert the original array into the padded array
1373:         newmat[offset_slices] = narray
1374: 
1375:         # This is the core of pad ...
1376:         for iaxis in rank:
1377:             np.apply_along_axis(function,
1378:                                 iaxis,
1379:                                 newmat,
1380:                                 pad_width[iaxis],
1381:                                 iaxis,
1382:                                 kwargs)
1383:         return newmat
1384: 
1385:     # If we get here, use new padding method
1386:     newmat = narray.copy()
1387: 
1388:     # API preserved, but completely new algorithm which pads by building the
1389:     # entire block to pad before/after `arr` with in one step, for each axis.
1390:     if mode == 'constant':
1391:         for axis, ((pad_before, pad_after), (before_val, after_val)) \
1392:                 in enumerate(zip(pad_width, kwargs['constant_values'])):
1393:             newmat = _prepend_const(newmat, pad_before, before_val, axis)
1394:             newmat = _append_const(newmat, pad_after, after_val, axis)
1395: 
1396:     elif mode == 'edge':
1397:         for axis, (pad_before, pad_after) in enumerate(pad_width):
1398:             newmat = _prepend_edge(newmat, pad_before, axis)
1399:             newmat = _append_edge(newmat, pad_after, axis)
1400: 
1401:     elif mode == 'linear_ramp':
1402:         for axis, ((pad_before, pad_after), (before_val, after_val)) \
1403:                 in enumerate(zip(pad_width, kwargs['end_values'])):
1404:             newmat = _prepend_ramp(newmat, pad_before, before_val, axis)
1405:             newmat = _append_ramp(newmat, pad_after, after_val, axis)
1406: 
1407:     elif mode == 'maximum':
1408:         for axis, ((pad_before, pad_after), (chunk_before, chunk_after)) \
1409:                 in enumerate(zip(pad_width, kwargs['stat_length'])):
1410:             newmat = _prepend_max(newmat, pad_before, chunk_before, axis)
1411:             newmat = _append_max(newmat, pad_after, chunk_after, axis)
1412: 
1413:     elif mode == 'mean':
1414:         for axis, ((pad_before, pad_after), (chunk_before, chunk_after)) \
1415:                 in enumerate(zip(pad_width, kwargs['stat_length'])):
1416:             newmat = _prepend_mean(newmat, pad_before, chunk_before, axis)
1417:             newmat = _append_mean(newmat, pad_after, chunk_after, axis)
1418: 
1419:     elif mode == 'median':
1420:         for axis, ((pad_before, pad_after), (chunk_before, chunk_after)) \
1421:                 in enumerate(zip(pad_width, kwargs['stat_length'])):
1422:             newmat = _prepend_med(newmat, pad_before, chunk_before, axis)
1423:             newmat = _append_med(newmat, pad_after, chunk_after, axis)
1424: 
1425:     elif mode == 'minimum':
1426:         for axis, ((pad_before, pad_after), (chunk_before, chunk_after)) \
1427:                 in enumerate(zip(pad_width, kwargs['stat_length'])):
1428:             newmat = _prepend_min(newmat, pad_before, chunk_before, axis)
1429:             newmat = _append_min(newmat, pad_after, chunk_after, axis)
1430: 
1431:     elif mode == 'reflect':
1432:         for axis, (pad_before, pad_after) in enumerate(pad_width):
1433:             # Recursive padding along any axis where `pad_amt` is too large
1434:             # for indexing tricks. We can only safely pad the original axis
1435:             # length, to keep the period of the reflections consistent.
1436:             if ((pad_before > 0) or
1437:                     (pad_after > 0)) and newmat.shape[axis] == 1:
1438:                 # Extending singleton dimension for 'reflect' is legacy
1439:                 # behavior; it really should raise an error.
1440:                 newmat = _prepend_edge(newmat, pad_before, axis)
1441:                 newmat = _append_edge(newmat, pad_after, axis)
1442:                 continue
1443: 
1444:             method = kwargs['reflect_type']
1445:             safe_pad = newmat.shape[axis] - 1
1446:             while ((pad_before > safe_pad) or (pad_after > safe_pad)):
1447:                 pad_iter_b = min(safe_pad,
1448:                                  safe_pad * (pad_before // safe_pad))
1449:                 pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
1450:                 newmat = _pad_ref(newmat, (pad_iter_b,
1451:                                            pad_iter_a), method, axis)
1452:                 pad_before -= pad_iter_b
1453:                 pad_after -= pad_iter_a
1454:                 safe_pad += pad_iter_b + pad_iter_a
1455:             newmat = _pad_ref(newmat, (pad_before, pad_after), method, axis)
1456: 
1457:     elif mode == 'symmetric':
1458:         for axis, (pad_before, pad_after) in enumerate(pad_width):
1459:             # Recursive padding along any axis where `pad_amt` is too large
1460:             # for indexing tricks. We can only safely pad the original axis
1461:             # length, to keep the period of the reflections consistent.
1462:             method = kwargs['reflect_type']
1463:             safe_pad = newmat.shape[axis]
1464:             while ((pad_before > safe_pad) or
1465:                    (pad_after > safe_pad)):
1466:                 pad_iter_b = min(safe_pad,
1467:                                  safe_pad * (pad_before // safe_pad))
1468:                 pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
1469:                 newmat = _pad_sym(newmat, (pad_iter_b,
1470:                                            pad_iter_a), method, axis)
1471:                 pad_before -= pad_iter_b
1472:                 pad_after -= pad_iter_a
1473:                 safe_pad += pad_iter_b + pad_iter_a
1474:             newmat = _pad_sym(newmat, (pad_before, pad_after), method, axis)
1475: 
1476:     elif mode == 'wrap':
1477:         for axis, (pad_before, pad_after) in enumerate(pad_width):
1478:             # Recursive padding along any axis where `pad_amt` is too large
1479:             # for indexing tricks. We can only safely pad the original axis
1480:             # length, to keep the period of the reflections consistent.
1481:             safe_pad = newmat.shape[axis]
1482:             while ((pad_before > safe_pad) or
1483:                    (pad_after > safe_pad)):
1484:                 pad_iter_b = min(safe_pad,
1485:                                  safe_pad * (pad_before // safe_pad))
1486:                 pad_iter_a = min(safe_pad, safe_pad * (pad_after // safe_pad))
1487:                 newmat = _pad_wrap(newmat, (pad_iter_b, pad_iter_a), axis)
1488: 
1489:                 pad_before -= pad_iter_b
1490:                 pad_after -= pad_iter_a
1491:                 safe_pad += pad_iter_b + pad_iter_a
1492:             newmat = _pad_wrap(newmat, (pad_before, pad_after), axis)
1493: 
1494:     return newmat
1495: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_101133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nThe arraypad module contains a group of functions to pad values onto the edges\nof an n-dimensional array.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_101134 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_101134) is not StypyTypeError):

    if (import_101134 != 'pyd_module'):
        __import__(import_101134)
        sys_modules_101135 = sys.modules[import_101134]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_101135.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_101134)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 11):
__all__ = ['pad']
module_type_store.set_exportable_members(['pad'])

# Obtaining an instance of the builtin type 'list' (line 11)
list_101136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_101137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'pad')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 10), list_101136, str_101137)

# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', list_101136)

@norecursion
def _arange_ndarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 18)
    False_101138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 46), 'False')
    defaults = [False_101138]
    # Create a new context for function '_arange_ndarray'
    module_type_store = module_type_store.open_function_context('_arange_ndarray', 18, 0, False)
    
    # Passed parameters checking function
    _arange_ndarray.stypy_localization = localization
    _arange_ndarray.stypy_type_of_self = None
    _arange_ndarray.stypy_type_store = module_type_store
    _arange_ndarray.stypy_function_name = '_arange_ndarray'
    _arange_ndarray.stypy_param_names_list = ['arr', 'shape', 'axis', 'reverse']
    _arange_ndarray.stypy_varargs_param_name = None
    _arange_ndarray.stypy_kwargs_param_name = None
    _arange_ndarray.stypy_call_defaults = defaults
    _arange_ndarray.stypy_call_varargs = varargs
    _arange_ndarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_arange_ndarray', ['arr', 'shape', 'axis', 'reverse'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_arange_ndarray', localization, ['arr', 'shape', 'axis', 'reverse'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_arange_ndarray(...)' code ##################

    str_101139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', '\n    Create an ndarray of `shape` with increments along specified `axis`\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    shape : tuple of ints\n        Shape of desired array. Should be equivalent to `arr.shape` except\n        `shape[axis]` which may have any positive value.\n    axis : int\n        Axis to increment along.\n    reverse : bool\n        If False, increment in a positive fashion from 1 to `shape[axis]`,\n        inclusive. If True, the bounds are the same but the order reversed.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array sized to pad `arr` along `axis`, with linear range from\n        1 to `shape[axis]` along specified `axis`.\n\n    Notes\n    -----\n    The range is deliberately 1-indexed for this specific use case. Think of\n    this algorithm as broadcasting `np.arange` to a single `axis` of an\n    arbitrarily shaped ndarray.\n\n    ')
    
    # Assigning a Call to a Name (line 48):
    
    # Call to tuple(...): (line 48)
    # Processing the call arguments (line 48)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 48, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'arr' (line 49)
    arr_101151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 49)
    shape_101152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 46), arr_101151, 'shape')
    # Processing the call keyword arguments (line 49)
    kwargs_101153 = {}
    # Getting the type of 'enumerate' (line 49)
    enumerate_101150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 49)
    enumerate_call_result_101154 = invoke(stypy.reporting.localization.Localization(__file__, 49, 36), enumerate_101150, *[shape_101152], **kwargs_101153)
    
    comprehension_101155 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), enumerate_call_result_101154)
    # Assigning a type to the variable 'i' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), comprehension_101155))
    # Assigning a type to the variable 'x' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), comprehension_101155))
    
    
    # Getting the type of 'i' (line 48)
    i_101141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'i', False)
    # Getting the type of 'axis' (line 48)
    axis_101142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'axis', False)
    # Applying the binary operator '!=' (line 48)
    result_ne_101143 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 27), '!=', i_101141, axis_101142)
    
    # Testing the type of an if expression (line 48)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 22), result_ne_101143)
    # SSA begins for if expression (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_101144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'int')
    # SSA branch for the else part of an if expression (line 48)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 48)
    axis_101145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 48), 'axis', False)
    # Getting the type of 'shape' (line 48)
    shape_101146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___101147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 42), shape_101146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_101148 = invoke(stypy.reporting.localization.Localization(__file__, 48, 42), getitem___101147, axis_101145)
    
    # SSA join for if expression (line 48)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101149 = union_type.UnionType.add(int_101144, subscript_call_result_101148)
    
    list_101156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_101156, if_exp_101149)
    # Processing the call keyword arguments (line 48)
    kwargs_101157 = {}
    # Getting the type of 'tuple' (line 48)
    tuple_101140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 48)
    tuple_call_result_101158 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), tuple_101140, *[list_101156], **kwargs_101157)
    
    # Assigning a type to the variable 'initshape' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'initshape', tuple_call_result_101158)
    
    
    # Getting the type of 'reverse' (line 50)
    reverse_101159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'reverse')
    # Applying the 'not' unary operator (line 50)
    result_not__101160 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), 'not', reverse_101159)
    
    # Testing the type of an if condition (line 50)
    if_condition_101161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_not__101160)
    # Assigning a type to the variable 'if_condition_101161' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_101161', if_condition_101161)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 51):
    
    # Call to arange(...): (line 51)
    # Processing the call arguments (line 51)
    int_101164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 51)
    axis_101165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'axis', False)
    # Getting the type of 'shape' (line 51)
    shape_101166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___101167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), shape_101166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_101168 = invoke(stypy.reporting.localization.Localization(__file__, 51, 30), getitem___101167, axis_101165)
    
    int_101169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 44), 'int')
    # Applying the binary operator '+' (line 51)
    result_add_101170 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '+', subscript_call_result_101168, int_101169)
    
    # Processing the call keyword arguments (line 51)
    kwargs_101171 = {}
    # Getting the type of 'np' (line 51)
    np_101162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'np', False)
    # Obtaining the member 'arange' of a type (line 51)
    arange_101163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), np_101162, 'arange')
    # Calling arange(args, kwargs) (line 51)
    arange_call_result_101172 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), arange_101163, *[int_101164, result_add_101170], **kwargs_101171)
    
    # Assigning a type to the variable 'padarr' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'padarr', arange_call_result_101172)
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 53):
    
    # Call to arange(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 53)
    axis_101175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'axis', False)
    # Getting the type of 'shape' (line 53)
    shape_101176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'shape', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___101177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 27), shape_101176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_101178 = invoke(stypy.reporting.localization.Localization(__file__, 53, 27), getitem___101177, axis_101175)
    
    int_101179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 40), 'int')
    int_101180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'int')
    # Processing the call keyword arguments (line 53)
    kwargs_101181 = {}
    # Getting the type of 'np' (line 53)
    np_101173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'np', False)
    # Obtaining the member 'arange' of a type (line 53)
    arange_101174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), np_101173, 'arange')
    # Calling arange(args, kwargs) (line 53)
    arange_call_result_101182 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), arange_101174, *[subscript_call_result_101178, int_101179, int_101180], **kwargs_101181)
    
    # Assigning a type to the variable 'padarr' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'padarr', arange_call_result_101182)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 54):
    
    # Call to reshape(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'initshape' (line 54)
    initshape_101185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'initshape', False)
    # Processing the call keyword arguments (line 54)
    kwargs_101186 = {}
    # Getting the type of 'padarr' (line 54)
    padarr_101183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'padarr', False)
    # Obtaining the member 'reshape' of a type (line 54)
    reshape_101184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), padarr_101183, 'reshape')
    # Calling reshape(args, kwargs) (line 54)
    reshape_call_result_101187 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), reshape_101184, *[initshape_101185], **kwargs_101186)
    
    # Assigning a type to the variable 'padarr' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'padarr', reshape_call_result_101187)
    
    
    # Call to enumerate(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'shape' (line 55)
    shape_101189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 28), 'shape', False)
    # Processing the call keyword arguments (line 55)
    kwargs_101190 = {}
    # Getting the type of 'enumerate' (line 55)
    enumerate_101188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 55)
    enumerate_call_result_101191 = invoke(stypy.reporting.localization.Localization(__file__, 55, 18), enumerate_101188, *[shape_101189], **kwargs_101190)
    
    # Testing the type of a for loop iterable (line 55)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 4), enumerate_call_result_101191)
    # Getting the type of the for loop variable (line 55)
    for_loop_var_101192 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 4), enumerate_call_result_101191)
    # Assigning a type to the variable 'i' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 4), for_loop_var_101192))
    # Assigning a type to the variable 'dim' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'dim', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 4), for_loop_var_101192))
    # SSA begins for a for statement (line 55)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 56)
    i_101193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'i')
    # Getting the type of 'padarr' (line 56)
    padarr_101194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'padarr')
    # Obtaining the member 'shape' of a type (line 56)
    shape_101195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), padarr_101194, 'shape')
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___101196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 11), shape_101195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_101197 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), getitem___101196, i_101193)
    
    # Getting the type of 'dim' (line 56)
    dim_101198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'dim')
    # Applying the binary operator '!=' (line 56)
    result_ne_101199 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), '!=', subscript_call_result_101197, dim_101198)
    
    # Testing the type of an if condition (line 56)
    if_condition_101200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), result_ne_101199)
    # Assigning a type to the variable 'if_condition_101200' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_101200', if_condition_101200)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 57):
    
    # Call to repeat(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'dim' (line 57)
    dim_101203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 35), 'dim', False)
    # Processing the call keyword arguments (line 57)
    # Getting the type of 'i' (line 57)
    i_101204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'i', False)
    keyword_101205 = i_101204
    kwargs_101206 = {'axis': keyword_101205}
    # Getting the type of 'padarr' (line 57)
    padarr_101201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'padarr', False)
    # Obtaining the member 'repeat' of a type (line 57)
    repeat_101202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 21), padarr_101201, 'repeat')
    # Calling repeat(args, kwargs) (line 57)
    repeat_call_result_101207 = invoke(stypy.reporting.localization.Localization(__file__, 57, 21), repeat_101202, *[dim_101203], **kwargs_101206)
    
    # Assigning a type to the variable 'padarr' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'padarr', repeat_call_result_101207)
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'padarr' (line 58)
    padarr_101208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'padarr')
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', padarr_101208)
    
    # ################# End of '_arange_ndarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_arange_ndarray' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_101209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101209)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_arange_ndarray'
    return stypy_return_type_101209

# Assigning a type to the variable '_arange_ndarray' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_arange_ndarray', _arange_ndarray)

@norecursion
def _round_ifneeded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_round_ifneeded'
    module_type_store = module_type_store.open_function_context('_round_ifneeded', 61, 0, False)
    
    # Passed parameters checking function
    _round_ifneeded.stypy_localization = localization
    _round_ifneeded.stypy_type_of_self = None
    _round_ifneeded.stypy_type_store = module_type_store
    _round_ifneeded.stypy_function_name = '_round_ifneeded'
    _round_ifneeded.stypy_param_names_list = ['arr', 'dtype']
    _round_ifneeded.stypy_varargs_param_name = None
    _round_ifneeded.stypy_kwargs_param_name = None
    _round_ifneeded.stypy_call_defaults = defaults
    _round_ifneeded.stypy_call_varargs = varargs
    _round_ifneeded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_round_ifneeded', ['arr', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_round_ifneeded', localization, ['arr', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_round_ifneeded(...)' code ##################

    str_101210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', '\n    Rounds arr inplace if destination dtype is integer.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array.\n    dtype : dtype\n        The dtype of the destination array.\n\n    ')
    
    
    # Call to issubdtype(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'dtype' (line 73)
    dtype_101213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'dtype', False)
    # Getting the type of 'np' (line 73)
    np_101214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 28), 'np', False)
    # Obtaining the member 'integer' of a type (line 73)
    integer_101215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 28), np_101214, 'integer')
    # Processing the call keyword arguments (line 73)
    kwargs_101216 = {}
    # Getting the type of 'np' (line 73)
    np_101211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 73)
    issubdtype_101212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 7), np_101211, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 73)
    issubdtype_call_result_101217 = invoke(stypy.reporting.localization.Localization(__file__, 73, 7), issubdtype_101212, *[dtype_101213, integer_101215], **kwargs_101216)
    
    # Testing the type of an if condition (line 73)
    if_condition_101218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), issubdtype_call_result_101217)
    # Assigning a type to the variable 'if_condition_101218' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_101218', if_condition_101218)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to round(...): (line 74)
    # Processing the call keyword arguments (line 74)
    # Getting the type of 'arr' (line 74)
    arr_101221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'arr', False)
    keyword_101222 = arr_101221
    kwargs_101223 = {'out': keyword_101222}
    # Getting the type of 'arr' (line 74)
    arr_101219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'arr', False)
    # Obtaining the member 'round' of a type (line 74)
    round_101220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), arr_101219, 'round')
    # Calling round(args, kwargs) (line 74)
    round_call_result_101224 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), round_101220, *[], **kwargs_101223)
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_round_ifneeded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_round_ifneeded' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_101225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_round_ifneeded'
    return stypy_return_type_101225

# Assigning a type to the variable '_round_ifneeded' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_round_ifneeded', _round_ifneeded)

@norecursion
def _prepend_const(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 43), 'int')
    defaults = [int_101226]
    # Create a new context for function '_prepend_const'
    module_type_store = module_type_store.open_function_context('_prepend_const', 77, 0, False)
    
    # Passed parameters checking function
    _prepend_const.stypy_localization = localization
    _prepend_const.stypy_type_of_self = None
    _prepend_const.stypy_type_store = module_type_store
    _prepend_const.stypy_function_name = '_prepend_const'
    _prepend_const.stypy_param_names_list = ['arr', 'pad_amt', 'val', 'axis']
    _prepend_const.stypy_varargs_param_name = None
    _prepend_const.stypy_kwargs_param_name = None
    _prepend_const.stypy_call_defaults = defaults
    _prepend_const.stypy_call_varargs = varargs
    _prepend_const.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_const', ['arr', 'pad_amt', 'val', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_const', localization, ['arr', 'pad_amt', 'val', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_const(...)' code ##################

    str_101227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n    Prepend constant `val` along `axis` of `arr`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    val : scalar\n        Constant value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` constant `val` prepended along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 99)
    pad_amt_101228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'pad_amt')
    int_101229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'int')
    # Applying the binary operator '==' (line 99)
    result_eq_101230 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '==', pad_amt_101228, int_101229)
    
    # Testing the type of an if condition (line 99)
    if_condition_101231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_eq_101230)
    # Assigning a type to the variable 'if_condition_101231' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_101231', if_condition_101231)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 100)
    arr_101232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', arr_101232)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 101):
    
    # Call to tuple(...): (line 101)
    # Processing the call arguments (line 101)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 101, 21, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'arr' (line 102)
    arr_101241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'arr', False)
    # Obtaining the member 'shape' of a type (line 102)
    shape_101242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 45), arr_101241, 'shape')
    # Processing the call keyword arguments (line 102)
    kwargs_101243 = {}
    # Getting the type of 'enumerate' (line 102)
    enumerate_101240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 102)
    enumerate_call_result_101244 = invoke(stypy.reporting.localization.Localization(__file__, 102, 35), enumerate_101240, *[shape_101242], **kwargs_101243)
    
    comprehension_101245 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), enumerate_call_result_101244)
    # Assigning a type to the variable 'i' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), comprehension_101245))
    # Assigning a type to the variable 'x' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), comprehension_101245))
    
    
    # Getting the type of 'i' (line 101)
    i_101234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'i', False)
    # Getting the type of 'axis' (line 101)
    axis_101235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'axis', False)
    # Applying the binary operator '!=' (line 101)
    result_ne_101236 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 26), '!=', i_101234, axis_101235)
    
    # Testing the type of an if expression (line 101)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 21), result_ne_101236)
    # SSA begins for if expression (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 101)
    x_101237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'x', False)
    # SSA branch for the else part of an if expression (line 101)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'pad_amt' (line 101)
    pad_amt_101238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'pad_amt', False)
    # SSA join for if expression (line 101)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101239 = union_type.UnionType.add(x_101237, pad_amt_101238)
    
    list_101246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), list_101246, if_exp_101239)
    # Processing the call keyword arguments (line 101)
    kwargs_101247 = {}
    # Getting the type of 'tuple' (line 101)
    tuple_101233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 101)
    tuple_call_result_101248 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), tuple_101233, *[list_101246], **kwargs_101247)
    
    # Assigning a type to the variable 'padshape' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'padshape', tuple_call_result_101248)
    
    
    # Getting the type of 'val' (line 103)
    val_101249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'val')
    int_101250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 14), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_101251 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), '==', val_101249, int_101250)
    
    # Testing the type of an if condition (line 103)
    if_condition_101252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_eq_101251)
    # Assigning a type to the variable 'if_condition_101252' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_101252', if_condition_101252)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to concatenate(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining an instance of the builtin type 'tuple' (line 104)
    tuple_101255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 104)
    # Adding element type (line 104)
    
    # Call to zeros(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'padshape' (line 104)
    padshape_101258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 40), 'padshape', False)
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'arr' (line 104)
    arr_101259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 56), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 104)
    dtype_101260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 56), arr_101259, 'dtype')
    keyword_101261 = dtype_101260
    kwargs_101262 = {'dtype': keyword_101261}
    # Getting the type of 'np' (line 104)
    np_101256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'np', False)
    # Obtaining the member 'zeros' of a type (line 104)
    zeros_101257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), np_101256, 'zeros')
    # Calling zeros(args, kwargs) (line 104)
    zeros_call_result_101263 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), zeros_101257, *[padshape_101258], **kwargs_101262)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), tuple_101255, zeros_call_result_101263)
    # Adding element type (line 104)
    # Getting the type of 'arr' (line 104)
    arr_101264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 68), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), tuple_101255, arr_101264)
    
    # Processing the call keyword arguments (line 104)
    # Getting the type of 'axis' (line 105)
    axis_101265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'axis', False)
    keyword_101266 = axis_101265
    kwargs_101267 = {'axis': keyword_101266}
    # Getting the type of 'np' (line 104)
    np_101253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 104)
    concatenate_101254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), np_101253, 'concatenate')
    # Calling concatenate(args, kwargs) (line 104)
    concatenate_call_result_101268 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), concatenate_101254, *[tuple_101255], **kwargs_101267)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', concatenate_call_result_101268)
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    # Call to concatenate(...): (line 107)
    # Processing the call arguments (line 107)
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_101271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    
    # Call to astype(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'arr' (line 107)
    arr_101280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 65), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 107)
    dtype_101281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 65), arr_101280, 'dtype')
    # Processing the call keyword arguments (line 107)
    kwargs_101282 = {}
    
    # Call to zeros(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'padshape' (line 107)
    padshape_101274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 41), 'padshape', False)
    # Processing the call keyword arguments (line 107)
    kwargs_101275 = {}
    # Getting the type of 'np' (line 107)
    np_101272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'np', False)
    # Obtaining the member 'zeros' of a type (line 107)
    zeros_101273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 32), np_101272, 'zeros')
    # Calling zeros(args, kwargs) (line 107)
    zeros_call_result_101276 = invoke(stypy.reporting.localization.Localization(__file__, 107, 32), zeros_101273, *[padshape_101274], **kwargs_101275)
    
    # Getting the type of 'val' (line 107)
    val_101277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 53), 'val', False)
    # Applying the binary operator '+' (line 107)
    result_add_101278 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 32), '+', zeros_call_result_101276, val_101277)
    
    # Obtaining the member 'astype' of a type (line 107)
    astype_101279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 32), result_add_101278, 'astype')
    # Calling astype(args, kwargs) (line 107)
    astype_call_result_101283 = invoke(stypy.reporting.localization.Localization(__file__, 107, 32), astype_101279, *[dtype_101281], **kwargs_101282)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 31), tuple_101271, astype_call_result_101283)
    # Adding element type (line 107)
    # Getting the type of 'arr' (line 108)
    arr_101284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 31), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 31), tuple_101271, arr_101284)
    
    # Processing the call keyword arguments (line 107)
    # Getting the type of 'axis' (line 108)
    axis_101285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'axis', False)
    keyword_101286 = axis_101285
    kwargs_101287 = {'axis': keyword_101286}
    # Getting the type of 'np' (line 107)
    np_101269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 107)
    concatenate_101270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), np_101269, 'concatenate')
    # Calling concatenate(args, kwargs) (line 107)
    concatenate_call_result_101288 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), concatenate_101270, *[tuple_101271], **kwargs_101287)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', concatenate_call_result_101288)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_prepend_const(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_const' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_101289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101289)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_const'
    return stypy_return_type_101289

# Assigning a type to the variable '_prepend_const' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), '_prepend_const', _prepend_const)

@norecursion
def _append_const(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'int')
    defaults = [int_101290]
    # Create a new context for function '_append_const'
    module_type_store = module_type_store.open_function_context('_append_const', 111, 0, False)
    
    # Passed parameters checking function
    _append_const.stypy_localization = localization
    _append_const.stypy_type_of_self = None
    _append_const.stypy_type_store = module_type_store
    _append_const.stypy_function_name = '_append_const'
    _append_const.stypy_param_names_list = ['arr', 'pad_amt', 'val', 'axis']
    _append_const.stypy_varargs_param_name = None
    _append_const.stypy_kwargs_param_name = None
    _append_const.stypy_call_defaults = defaults
    _append_const.stypy_call_varargs = varargs
    _append_const.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_const', ['arr', 'pad_amt', 'val', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_const', localization, ['arr', 'pad_amt', 'val', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_const(...)' code ##################

    str_101291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Append constant `val` along `axis` of `arr`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    val : scalar\n        Constant value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` constant `val` appended along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 133)
    pad_amt_101292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'pad_amt')
    int_101293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 18), 'int')
    # Applying the binary operator '==' (line 133)
    result_eq_101294 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), '==', pad_amt_101292, int_101293)
    
    # Testing the type of an if condition (line 133)
    if_condition_101295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_eq_101294)
    # Assigning a type to the variable 'if_condition_101295' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_101295', if_condition_101295)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 134)
    arr_101296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', arr_101296)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 135):
    
    # Call to tuple(...): (line 135)
    # Processing the call arguments (line 135)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 135, 21, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'arr' (line 136)
    arr_101305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 45), 'arr', False)
    # Obtaining the member 'shape' of a type (line 136)
    shape_101306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 45), arr_101305, 'shape')
    # Processing the call keyword arguments (line 136)
    kwargs_101307 = {}
    # Getting the type of 'enumerate' (line 136)
    enumerate_101304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 136)
    enumerate_call_result_101308 = invoke(stypy.reporting.localization.Localization(__file__, 136, 35), enumerate_101304, *[shape_101306], **kwargs_101307)
    
    comprehension_101309 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), enumerate_call_result_101308)
    # Assigning a type to the variable 'i' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), comprehension_101309))
    # Assigning a type to the variable 'x' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), comprehension_101309))
    
    
    # Getting the type of 'i' (line 135)
    i_101298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'i', False)
    # Getting the type of 'axis' (line 135)
    axis_101299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'axis', False)
    # Applying the binary operator '!=' (line 135)
    result_ne_101300 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 26), '!=', i_101298, axis_101299)
    
    # Testing the type of an if expression (line 135)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 21), result_ne_101300)
    # SSA begins for if expression (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 135)
    x_101301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'x', False)
    # SSA branch for the else part of an if expression (line 135)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'pad_amt' (line 135)
    pad_amt_101302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 41), 'pad_amt', False)
    # SSA join for if expression (line 135)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101303 = union_type.UnionType.add(x_101301, pad_amt_101302)
    
    list_101310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_101310, if_exp_101303)
    # Processing the call keyword arguments (line 135)
    kwargs_101311 = {}
    # Getting the type of 'tuple' (line 135)
    tuple_101297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 135)
    tuple_call_result_101312 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), tuple_101297, *[list_101310], **kwargs_101311)
    
    # Assigning a type to the variable 'padshape' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'padshape', tuple_call_result_101312)
    
    
    # Getting the type of 'val' (line 137)
    val_101313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'val')
    int_101314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 14), 'int')
    # Applying the binary operator '==' (line 137)
    result_eq_101315 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 7), '==', val_101313, int_101314)
    
    # Testing the type of an if condition (line 137)
    if_condition_101316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), result_eq_101315)
    # Assigning a type to the variable 'if_condition_101316' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_101316', if_condition_101316)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to concatenate(...): (line 138)
    # Processing the call arguments (line 138)
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_101319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'arr' (line 138)
    arr_101320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 31), tuple_101319, arr_101320)
    # Adding element type (line 138)
    
    # Call to zeros(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'padshape' (line 138)
    padshape_101323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 45), 'padshape', False)
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'arr' (line 138)
    arr_101324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 61), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 138)
    dtype_101325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 61), arr_101324, 'dtype')
    keyword_101326 = dtype_101325
    kwargs_101327 = {'dtype': keyword_101326}
    # Getting the type of 'np' (line 138)
    np_101321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 36), 'np', False)
    # Obtaining the member 'zeros' of a type (line 138)
    zeros_101322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 36), np_101321, 'zeros')
    # Calling zeros(args, kwargs) (line 138)
    zeros_call_result_101328 = invoke(stypy.reporting.localization.Localization(__file__, 138, 36), zeros_101322, *[padshape_101323], **kwargs_101327)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 31), tuple_101319, zeros_call_result_101328)
    
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'axis' (line 139)
    axis_101329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'axis', False)
    keyword_101330 = axis_101329
    kwargs_101331 = {'axis': keyword_101330}
    # Getting the type of 'np' (line 138)
    np_101317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 138)
    concatenate_101318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), np_101317, 'concatenate')
    # Calling concatenate(args, kwargs) (line 138)
    concatenate_call_result_101332 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), concatenate_101318, *[tuple_101319], **kwargs_101331)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', concatenate_call_result_101332)
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    # Call to concatenate(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_101335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    # Getting the type of 'arr' (line 142)
    arr_101336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 13), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 13), tuple_101335, arr_101336)
    # Adding element type (line 142)
    
    # Call to astype(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'arr' (line 142)
    arr_101345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 142)
    dtype_101346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 52), arr_101345, 'dtype')
    # Processing the call keyword arguments (line 142)
    kwargs_101347 = {}
    
    # Call to zeros(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'padshape' (line 142)
    padshape_101339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'padshape', False)
    # Processing the call keyword arguments (line 142)
    kwargs_101340 = {}
    # Getting the type of 'np' (line 142)
    np_101337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 142)
    zeros_101338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), np_101337, 'zeros')
    # Calling zeros(args, kwargs) (line 142)
    zeros_call_result_101341 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), zeros_101338, *[padshape_101339], **kwargs_101340)
    
    # Getting the type of 'val' (line 142)
    val_101342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), 'val', False)
    # Applying the binary operator '+' (line 142)
    result_add_101343 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 19), '+', zeros_call_result_101341, val_101342)
    
    # Obtaining the member 'astype' of a type (line 142)
    astype_101344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), result_add_101343, 'astype')
    # Calling astype(args, kwargs) (line 142)
    astype_call_result_101348 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), astype_101344, *[dtype_101346], **kwargs_101347)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 13), tuple_101335, astype_call_result_101348)
    
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'axis' (line 142)
    axis_101349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 70), 'axis', False)
    keyword_101350 = axis_101349
    kwargs_101351 = {'axis': keyword_101350}
    # Getting the type of 'np' (line 141)
    np_101333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 141)
    concatenate_101334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), np_101333, 'concatenate')
    # Calling concatenate(args, kwargs) (line 141)
    concatenate_call_result_101352 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), concatenate_101334, *[tuple_101335], **kwargs_101351)
    
    # Assigning a type to the variable 'stypy_return_type' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', concatenate_call_result_101352)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_append_const(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_const' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_101353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101353)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_const'
    return stypy_return_type_101353

# Assigning a type to the variable '_append_const' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), '_append_const', _append_const)

@norecursion
def _prepend_edge(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 37), 'int')
    defaults = [int_101354]
    # Create a new context for function '_prepend_edge'
    module_type_store = module_type_store.open_function_context('_prepend_edge', 145, 0, False)
    
    # Passed parameters checking function
    _prepend_edge.stypy_localization = localization
    _prepend_edge.stypy_type_of_self = None
    _prepend_edge.stypy_type_store = module_type_store
    _prepend_edge.stypy_function_name = '_prepend_edge'
    _prepend_edge.stypy_param_names_list = ['arr', 'pad_amt', 'axis']
    _prepend_edge.stypy_varargs_param_name = None
    _prepend_edge.stypy_kwargs_param_name = None
    _prepend_edge.stypy_call_defaults = defaults
    _prepend_edge.stypy_call_varargs = varargs
    _prepend_edge.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_edge', ['arr', 'pad_amt', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_edge', localization, ['arr', 'pad_amt', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_edge(...)' code ##################

    str_101355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Prepend `pad_amt` to `arr` along `axis` by extending edge values.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, extended by `pad_amt` edge values appended along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 164)
    pad_amt_101356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 7), 'pad_amt')
    int_101357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'int')
    # Applying the binary operator '==' (line 164)
    result_eq_101358 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), '==', pad_amt_101356, int_101357)
    
    # Testing the type of an if condition (line 164)
    if_condition_101359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), result_eq_101358)
    # Assigning a type to the variable 'if_condition_101359' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'if_condition_101359', if_condition_101359)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 165)
    arr_101360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', arr_101360)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 167):
    
    # Call to tuple(...): (line 167)
    # Processing the call arguments (line 167)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 167, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'arr' (line 168)
    arr_101372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 168)
    shape_101373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 47), arr_101372, 'shape')
    # Processing the call keyword arguments (line 168)
    kwargs_101374 = {}
    # Getting the type of 'enumerate' (line 168)
    enumerate_101371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 168)
    enumerate_call_result_101375 = invoke(stypy.reporting.localization.Localization(__file__, 168, 37), enumerate_101371, *[shape_101373], **kwargs_101374)
    
    comprehension_101376 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), enumerate_call_result_101375)
    # Assigning a type to the variable 'i' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), comprehension_101376))
    # Assigning a type to the variable 'x' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), comprehension_101376))
    
    
    # Getting the type of 'i' (line 167)
    i_101362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 38), 'i', False)
    # Getting the type of 'axis' (line 167)
    axis_101363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'axis', False)
    # Applying the binary operator '!=' (line 167)
    result_ne_101364 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 38), '!=', i_101362, axis_101363)
    
    # Testing the type of an if expression (line 167)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 23), result_ne_101364)
    # SSA begins for if expression (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'None' (line 167)
    None_101366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'None', False)
    # Processing the call keyword arguments (line 167)
    kwargs_101367 = {}
    # Getting the type of 'slice' (line 167)
    slice_101365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 167)
    slice_call_result_101368 = invoke(stypy.reporting.localization.Localization(__file__, 167, 23), slice_101365, *[None_101366], **kwargs_101367)
    
    # SSA branch for the else part of an if expression (line 167)
    module_type_store.open_ssa_branch('if expression else')
    int_101369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 53), 'int')
    # SSA join for if expression (line 167)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101370 = union_type.UnionType.add(slice_call_result_101368, int_101369)
    
    list_101377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), list_101377, if_exp_101370)
    # Processing the call keyword arguments (line 167)
    kwargs_101378 = {}
    # Getting the type of 'tuple' (line 167)
    tuple_101361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 167)
    tuple_call_result_101379 = invoke(stypy.reporting.localization.Localization(__file__, 167, 17), tuple_101361, *[list_101377], **kwargs_101378)
    
    # Assigning a type to the variable 'edge_slice' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'edge_slice', tuple_call_result_101379)
    
    # Assigning a Call to a Name (line 171):
    
    # Call to tuple(...): (line 171)
    # Processing the call arguments (line 171)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 171, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'arr' (line 172)
    arr_101388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 172)
    shape_101389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 50), arr_101388, 'shape')
    # Processing the call keyword arguments (line 172)
    kwargs_101390 = {}
    # Getting the type of 'enumerate' (line 172)
    enumerate_101387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 172)
    enumerate_call_result_101391 = invoke(stypy.reporting.localization.Localization(__file__, 172, 40), enumerate_101387, *[shape_101389], **kwargs_101390)
    
    comprehension_101392 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 26), enumerate_call_result_101391)
    # Assigning a type to the variable 'i' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 26), comprehension_101392))
    # Assigning a type to the variable 'x' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 26), comprehension_101392))
    
    
    # Getting the type of 'i' (line 171)
    i_101381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'i', False)
    # Getting the type of 'axis' (line 171)
    axis_101382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'axis', False)
    # Applying the binary operator '!=' (line 171)
    result_ne_101383 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 31), '!=', i_101381, axis_101382)
    
    # Testing the type of an if expression (line 171)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 26), result_ne_101383)
    # SSA begins for if expression (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 171)
    x_101384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 171)
    module_type_store.open_ssa_branch('if expression else')
    int_101385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 46), 'int')
    # SSA join for if expression (line 171)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101386 = union_type.UnionType.add(x_101384, int_101385)
    
    list_101393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 26), list_101393, if_exp_101386)
    # Processing the call keyword arguments (line 171)
    kwargs_101394 = {}
    # Getting the type of 'tuple' (line 171)
    tuple_101380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 171)
    tuple_call_result_101395 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), tuple_101380, *[list_101393], **kwargs_101394)
    
    # Assigning a type to the variable 'pad_singleton' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'pad_singleton', tuple_call_result_101395)
    
    # Assigning a Call to a Name (line 173):
    
    # Call to reshape(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'pad_singleton' (line 173)
    pad_singleton_101401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 39), 'pad_singleton', False)
    # Processing the call keyword arguments (line 173)
    kwargs_101402 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice' (line 173)
    edge_slice_101396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'edge_slice', False)
    # Getting the type of 'arr' (line 173)
    arr_101397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 173)
    getitem___101398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 15), arr_101397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 173)
    subscript_call_result_101399 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), getitem___101398, edge_slice_101396)
    
    # Obtaining the member 'reshape' of a type (line 173)
    reshape_101400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 15), subscript_call_result_101399, 'reshape')
    # Calling reshape(args, kwargs) (line 173)
    reshape_call_result_101403 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), reshape_101400, *[pad_singleton_101401], **kwargs_101402)
    
    # Assigning a type to the variable 'edge_arr' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'edge_arr', reshape_call_result_101403)
    
    # Call to concatenate(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_101406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    
    # Call to repeat(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'pad_amt' (line 174)
    pad_amt_101409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'pad_amt', False)
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'axis' (line 174)
    axis_101410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 57), 'axis', False)
    keyword_101411 = axis_101410
    kwargs_101412 = {'axis': keyword_101411}
    # Getting the type of 'edge_arr' (line 174)
    edge_arr_101407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'edge_arr', False)
    # Obtaining the member 'repeat' of a type (line 174)
    repeat_101408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 27), edge_arr_101407, 'repeat')
    # Calling repeat(args, kwargs) (line 174)
    repeat_call_result_101413 = invoke(stypy.reporting.localization.Localization(__file__, 174, 27), repeat_101408, *[pad_amt_101409], **kwargs_101412)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), tuple_101406, repeat_call_result_101413)
    # Adding element type (line 174)
    # Getting the type of 'arr' (line 174)
    arr_101414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 64), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 27), tuple_101406, arr_101414)
    
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'axis' (line 175)
    axis_101415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), 'axis', False)
    keyword_101416 = axis_101415
    kwargs_101417 = {'axis': keyword_101416}
    # Getting the type of 'np' (line 174)
    np_101404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 174)
    concatenate_101405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), np_101404, 'concatenate')
    # Calling concatenate(args, kwargs) (line 174)
    concatenate_call_result_101418 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), concatenate_101405, *[tuple_101406], **kwargs_101417)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', concatenate_call_result_101418)
    
    # ################# End of '_prepend_edge(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_edge' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_101419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101419)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_edge'
    return stypy_return_type_101419

# Assigning a type to the variable '_prepend_edge' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), '_prepend_edge', _prepend_edge)

@norecursion
def _append_edge(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'int')
    defaults = [int_101420]
    # Create a new context for function '_append_edge'
    module_type_store = module_type_store.open_function_context('_append_edge', 178, 0, False)
    
    # Passed parameters checking function
    _append_edge.stypy_localization = localization
    _append_edge.stypy_type_of_self = None
    _append_edge.stypy_type_store = module_type_store
    _append_edge.stypy_function_name = '_append_edge'
    _append_edge.stypy_param_names_list = ['arr', 'pad_amt', 'axis']
    _append_edge.stypy_varargs_param_name = None
    _append_edge.stypy_kwargs_param_name = None
    _append_edge.stypy_call_defaults = defaults
    _append_edge.stypy_call_varargs = varargs
    _append_edge.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_edge', ['arr', 'pad_amt', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_edge', localization, ['arr', 'pad_amt', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_edge(...)' code ##################

    str_101421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', '\n    Append `pad_amt` to `arr` along `axis` by extending edge values.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, extended by `pad_amt` edge values prepended along\n        `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 198)
    pad_amt_101422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 7), 'pad_amt')
    int_101423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'int')
    # Applying the binary operator '==' (line 198)
    result_eq_101424 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 7), '==', pad_amt_101422, int_101423)
    
    # Testing the type of an if condition (line 198)
    if_condition_101425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 4), result_eq_101424)
    # Assigning a type to the variable 'if_condition_101425' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'if_condition_101425', if_condition_101425)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 199)
    arr_101426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', arr_101426)
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 201):
    
    # Call to tuple(...): (line 201)
    # Processing the call arguments (line 201)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 201, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'arr' (line 202)
    arr_101444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 202)
    shape_101445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 47), arr_101444, 'shape')
    # Processing the call keyword arguments (line 202)
    kwargs_101446 = {}
    # Getting the type of 'enumerate' (line 202)
    enumerate_101443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 202)
    enumerate_call_result_101447 = invoke(stypy.reporting.localization.Localization(__file__, 202, 37), enumerate_101443, *[shape_101445], **kwargs_101446)
    
    comprehension_101448 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), enumerate_call_result_101447)
    # Assigning a type to the variable 'i' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), comprehension_101448))
    # Assigning a type to the variable 'x' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), comprehension_101448))
    
    
    # Getting the type of 'i' (line 201)
    i_101428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 38), 'i', False)
    # Getting the type of 'axis' (line 201)
    axis_101429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'axis', False)
    # Applying the binary operator '!=' (line 201)
    result_ne_101430 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 38), '!=', i_101428, axis_101429)
    
    # Testing the type of an if expression (line 201)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 23), result_ne_101430)
    # SSA begins for if expression (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'None' (line 201)
    None_101432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 29), 'None', False)
    # Processing the call keyword arguments (line 201)
    kwargs_101433 = {}
    # Getting the type of 'slice' (line 201)
    slice_101431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 201)
    slice_call_result_101434 = invoke(stypy.reporting.localization.Localization(__file__, 201, 23), slice_101431, *[None_101432], **kwargs_101433)
    
    # SSA branch for the else part of an if expression (line 201)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 201)
    axis_101435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 63), 'axis', False)
    # Getting the type of 'arr' (line 201)
    arr_101436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 53), 'arr', False)
    # Obtaining the member 'shape' of a type (line 201)
    shape_101437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 53), arr_101436, 'shape')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___101438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 53), shape_101437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_101439 = invoke(stypy.reporting.localization.Localization(__file__, 201, 53), getitem___101438, axis_101435)
    
    int_101440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 71), 'int')
    # Applying the binary operator '-' (line 201)
    result_sub_101441 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 53), '-', subscript_call_result_101439, int_101440)
    
    # SSA join for if expression (line 201)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101442 = union_type.UnionType.add(slice_call_result_101434, result_sub_101441)
    
    list_101449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 23), list_101449, if_exp_101442)
    # Processing the call keyword arguments (line 201)
    kwargs_101450 = {}
    # Getting the type of 'tuple' (line 201)
    tuple_101427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 201)
    tuple_call_result_101451 = invoke(stypy.reporting.localization.Localization(__file__, 201, 17), tuple_101427, *[list_101449], **kwargs_101450)
    
    # Assigning a type to the variable 'edge_slice' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'edge_slice', tuple_call_result_101451)
    
    # Assigning a Call to a Name (line 205):
    
    # Call to tuple(...): (line 205)
    # Processing the call arguments (line 205)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 205, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'arr' (line 206)
    arr_101460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 206)
    shape_101461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 50), arr_101460, 'shape')
    # Processing the call keyword arguments (line 206)
    kwargs_101462 = {}
    # Getting the type of 'enumerate' (line 206)
    enumerate_101459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 206)
    enumerate_call_result_101463 = invoke(stypy.reporting.localization.Localization(__file__, 206, 40), enumerate_101459, *[shape_101461], **kwargs_101462)
    
    comprehension_101464 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), enumerate_call_result_101463)
    # Assigning a type to the variable 'i' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), comprehension_101464))
    # Assigning a type to the variable 'x' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), comprehension_101464))
    
    
    # Getting the type of 'i' (line 205)
    i_101453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 31), 'i', False)
    # Getting the type of 'axis' (line 205)
    axis_101454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'axis', False)
    # Applying the binary operator '!=' (line 205)
    result_ne_101455 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 31), '!=', i_101453, axis_101454)
    
    # Testing the type of an if expression (line 205)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 26), result_ne_101455)
    # SSA begins for if expression (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 205)
    x_101456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 205)
    module_type_store.open_ssa_branch('if expression else')
    int_101457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 46), 'int')
    # SSA join for if expression (line 205)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101458 = union_type.UnionType.add(x_101456, int_101457)
    
    list_101465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 26), list_101465, if_exp_101458)
    # Processing the call keyword arguments (line 205)
    kwargs_101466 = {}
    # Getting the type of 'tuple' (line 205)
    tuple_101452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 205)
    tuple_call_result_101467 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), tuple_101452, *[list_101465], **kwargs_101466)
    
    # Assigning a type to the variable 'pad_singleton' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'pad_singleton', tuple_call_result_101467)
    
    # Assigning a Call to a Name (line 207):
    
    # Call to reshape(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'pad_singleton' (line 207)
    pad_singleton_101473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 'pad_singleton', False)
    # Processing the call keyword arguments (line 207)
    kwargs_101474 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice' (line 207)
    edge_slice_101468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'edge_slice', False)
    # Getting the type of 'arr' (line 207)
    arr_101469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___101470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), arr_101469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_101471 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), getitem___101470, edge_slice_101468)
    
    # Obtaining the member 'reshape' of a type (line 207)
    reshape_101472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), subscript_call_result_101471, 'reshape')
    # Calling reshape(args, kwargs) (line 207)
    reshape_call_result_101475 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), reshape_101472, *[pad_singleton_101473], **kwargs_101474)
    
    # Assigning a type to the variable 'edge_arr' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'edge_arr', reshape_call_result_101475)
    
    # Call to concatenate(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_101478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'arr' (line 208)
    arr_101479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 27), tuple_101478, arr_101479)
    # Adding element type (line 208)
    
    # Call to repeat(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'pad_amt' (line 208)
    pad_amt_101482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'pad_amt', False)
    # Processing the call keyword arguments (line 208)
    # Getting the type of 'axis' (line 208)
    axis_101483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'axis', False)
    keyword_101484 = axis_101483
    kwargs_101485 = {'axis': keyword_101484}
    # Getting the type of 'edge_arr' (line 208)
    edge_arr_101480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'edge_arr', False)
    # Obtaining the member 'repeat' of a type (line 208)
    repeat_101481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 32), edge_arr_101480, 'repeat')
    # Calling repeat(args, kwargs) (line 208)
    repeat_call_result_101486 = invoke(stypy.reporting.localization.Localization(__file__, 208, 32), repeat_101481, *[pad_amt_101482], **kwargs_101485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 27), tuple_101478, repeat_call_result_101486)
    
    # Processing the call keyword arguments (line 208)
    # Getting the type of 'axis' (line 209)
    axis_101487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'axis', False)
    keyword_101488 = axis_101487
    kwargs_101489 = {'axis': keyword_101488}
    # Getting the type of 'np' (line 208)
    np_101476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 208)
    concatenate_101477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), np_101476, 'concatenate')
    # Calling concatenate(args, kwargs) (line 208)
    concatenate_call_result_101490 = invoke(stypy.reporting.localization.Localization(__file__, 208, 11), concatenate_101477, *[tuple_101478], **kwargs_101489)
    
    # Assigning a type to the variable 'stypy_return_type' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type', concatenate_call_result_101490)
    
    # ################# End of '_append_edge(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_edge' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_101491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101491)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_edge'
    return stypy_return_type_101491

# Assigning a type to the variable '_append_edge' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), '_append_edge', _append_edge)

@norecursion
def _prepend_ramp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 42), 'int')
    defaults = [int_101492]
    # Create a new context for function '_prepend_ramp'
    module_type_store = module_type_store.open_function_context('_prepend_ramp', 212, 0, False)
    
    # Passed parameters checking function
    _prepend_ramp.stypy_localization = localization
    _prepend_ramp.stypy_type_of_self = None
    _prepend_ramp.stypy_type_store = module_type_store
    _prepend_ramp.stypy_function_name = '_prepend_ramp'
    _prepend_ramp.stypy_param_names_list = ['arr', 'pad_amt', 'end', 'axis']
    _prepend_ramp.stypy_varargs_param_name = None
    _prepend_ramp.stypy_kwargs_param_name = None
    _prepend_ramp.stypy_call_defaults = defaults
    _prepend_ramp.stypy_call_varargs = varargs
    _prepend_ramp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_ramp', ['arr', 'pad_amt', 'end', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_ramp', localization, ['arr', 'pad_amt', 'end', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_ramp(...)' code ##################

    str_101493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', '\n    Prepend linear ramp along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    end : scalar\n        Constal value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region ramps linearly from the edge value to `end`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 235)
    pad_amt_101494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 7), 'pad_amt')
    int_101495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 18), 'int')
    # Applying the binary operator '==' (line 235)
    result_eq_101496 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 7), '==', pad_amt_101494, int_101495)
    
    # Testing the type of an if condition (line 235)
    if_condition_101497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 4), result_eq_101496)
    # Assigning a type to the variable 'if_condition_101497' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'if_condition_101497', if_condition_101497)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 236)
    arr_101498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', arr_101498)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 239):
    
    # Call to tuple(...): (line 239)
    # Processing the call arguments (line 239)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 239, 21, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'arr' (line 240)
    arr_101507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'arr', False)
    # Obtaining the member 'shape' of a type (line 240)
    shape_101508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 45), arr_101507, 'shape')
    # Processing the call keyword arguments (line 240)
    kwargs_101509 = {}
    # Getting the type of 'enumerate' (line 240)
    enumerate_101506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 35), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 240)
    enumerate_call_result_101510 = invoke(stypy.reporting.localization.Localization(__file__, 240, 35), enumerate_101506, *[shape_101508], **kwargs_101509)
    
    comprehension_101511 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), enumerate_call_result_101510)
    # Assigning a type to the variable 'i' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), comprehension_101511))
    # Assigning a type to the variable 'x' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), comprehension_101511))
    
    
    # Getting the type of 'i' (line 239)
    i_101500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'i', False)
    # Getting the type of 'axis' (line 239)
    axis_101501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'axis', False)
    # Applying the binary operator '!=' (line 239)
    result_ne_101502 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 26), '!=', i_101500, axis_101501)
    
    # Testing the type of an if expression (line 239)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 21), result_ne_101502)
    # SSA begins for if expression (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 239)
    x_101503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'x', False)
    # SSA branch for the else part of an if expression (line 239)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'pad_amt' (line 239)
    pad_amt_101504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'pad_amt', False)
    # SSA join for if expression (line 239)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101505 = union_type.UnionType.add(x_101503, pad_amt_101504)
    
    list_101512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 21), list_101512, if_exp_101505)
    # Processing the call keyword arguments (line 239)
    kwargs_101513 = {}
    # Getting the type of 'tuple' (line 239)
    tuple_101499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 239)
    tuple_call_result_101514 = invoke(stypy.reporting.localization.Localization(__file__, 239, 15), tuple_101499, *[list_101512], **kwargs_101513)
    
    # Assigning a type to the variable 'padshape' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'padshape', tuple_call_result_101514)
    
    # Assigning a Call to a Name (line 243):
    
    # Call to astype(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'np' (line 244)
    np_101524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 52), 'np', False)
    # Obtaining the member 'float64' of a type (line 244)
    float64_101525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 52), np_101524, 'float64')
    # Processing the call keyword arguments (line 243)
    kwargs_101526 = {}
    
    # Call to _arange_ndarray(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'arr' (line 243)
    arr_101516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 31), 'arr', False)
    # Getting the type of 'padshape' (line 243)
    padshape_101517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 36), 'padshape', False)
    # Getting the type of 'axis' (line 243)
    axis_101518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 46), 'axis', False)
    # Processing the call keyword arguments (line 243)
    # Getting the type of 'True' (line 244)
    True_101519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 39), 'True', False)
    keyword_101520 = True_101519
    kwargs_101521 = {'reverse': keyword_101520}
    # Getting the type of '_arange_ndarray' (line 243)
    _arange_ndarray_101515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), '_arange_ndarray', False)
    # Calling _arange_ndarray(args, kwargs) (line 243)
    _arange_ndarray_call_result_101522 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), _arange_ndarray_101515, *[arr_101516, padshape_101517, axis_101518], **kwargs_101521)
    
    # Obtaining the member 'astype' of a type (line 243)
    astype_101523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), _arange_ndarray_call_result_101522, 'astype')
    # Calling astype(args, kwargs) (line 243)
    astype_call_result_101527 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), astype_101523, *[float64_101525], **kwargs_101526)
    
    # Assigning a type to the variable 'ramp_arr' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'ramp_arr', astype_call_result_101527)
    
    # Assigning a Call to a Name (line 247):
    
    # Call to tuple(...): (line 247)
    # Processing the call arguments (line 247)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 247, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'arr' (line 248)
    arr_101539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 248)
    shape_101540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 47), arr_101539, 'shape')
    # Processing the call keyword arguments (line 248)
    kwargs_101541 = {}
    # Getting the type of 'enumerate' (line 248)
    enumerate_101538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 248)
    enumerate_call_result_101542 = invoke(stypy.reporting.localization.Localization(__file__, 248, 37), enumerate_101538, *[shape_101540], **kwargs_101541)
    
    comprehension_101543 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 23), enumerate_call_result_101542)
    # Assigning a type to the variable 'i' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 23), comprehension_101543))
    # Assigning a type to the variable 'x' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 23), comprehension_101543))
    
    
    # Getting the type of 'i' (line 247)
    i_101529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 'i', False)
    # Getting the type of 'axis' (line 247)
    axis_101530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 43), 'axis', False)
    # Applying the binary operator '!=' (line 247)
    result_ne_101531 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 38), '!=', i_101529, axis_101530)
    
    # Testing the type of an if expression (line 247)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 23), result_ne_101531)
    # SSA begins for if expression (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'None' (line 247)
    None_101533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'None', False)
    # Processing the call keyword arguments (line 247)
    kwargs_101534 = {}
    # Getting the type of 'slice' (line 247)
    slice_101532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 247)
    slice_call_result_101535 = invoke(stypy.reporting.localization.Localization(__file__, 247, 23), slice_101532, *[None_101533], **kwargs_101534)
    
    # SSA branch for the else part of an if expression (line 247)
    module_type_store.open_ssa_branch('if expression else')
    int_101536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 53), 'int')
    # SSA join for if expression (line 247)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101537 = union_type.UnionType.add(slice_call_result_101535, int_101536)
    
    list_101544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 23), list_101544, if_exp_101537)
    # Processing the call keyword arguments (line 247)
    kwargs_101545 = {}
    # Getting the type of 'tuple' (line 247)
    tuple_101528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 247)
    tuple_call_result_101546 = invoke(stypy.reporting.localization.Localization(__file__, 247, 17), tuple_101528, *[list_101544], **kwargs_101545)
    
    # Assigning a type to the variable 'edge_slice' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'edge_slice', tuple_call_result_101546)
    
    # Assigning a Call to a Name (line 251):
    
    # Call to tuple(...): (line 251)
    # Processing the call arguments (line 251)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 251, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'arr' (line 252)
    arr_101555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 252)
    shape_101556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 50), arr_101555, 'shape')
    # Processing the call keyword arguments (line 252)
    kwargs_101557 = {}
    # Getting the type of 'enumerate' (line 252)
    enumerate_101554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 252)
    enumerate_call_result_101558 = invoke(stypy.reporting.localization.Localization(__file__, 252, 40), enumerate_101554, *[shape_101556], **kwargs_101557)
    
    comprehension_101559 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), enumerate_call_result_101558)
    # Assigning a type to the variable 'i' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), comprehension_101559))
    # Assigning a type to the variable 'x' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), comprehension_101559))
    
    
    # Getting the type of 'i' (line 251)
    i_101548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'i', False)
    # Getting the type of 'axis' (line 251)
    axis_101549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'axis', False)
    # Applying the binary operator '!=' (line 251)
    result_ne_101550 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 31), '!=', i_101548, axis_101549)
    
    # Testing the type of an if expression (line 251)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 26), result_ne_101550)
    # SSA begins for if expression (line 251)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 251)
    x_101551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 251)
    module_type_store.open_ssa_branch('if expression else')
    int_101552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 46), 'int')
    # SSA join for if expression (line 251)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101553 = union_type.UnionType.add(x_101551, int_101552)
    
    list_101560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 26), list_101560, if_exp_101553)
    # Processing the call keyword arguments (line 251)
    kwargs_101561 = {}
    # Getting the type of 'tuple' (line 251)
    tuple_101547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 251)
    tuple_call_result_101562 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), tuple_101547, *[list_101560], **kwargs_101561)
    
    # Assigning a type to the variable 'pad_singleton' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'pad_singleton', tuple_call_result_101562)
    
    # Assigning a Call to a Name (line 255):
    
    # Call to repeat(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'pad_amt' (line 255)
    pad_amt_101572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 61), 'pad_amt', False)
    # Getting the type of 'axis' (line 255)
    axis_101573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 70), 'axis', False)
    # Processing the call keyword arguments (line 255)
    kwargs_101574 = {}
    
    # Call to reshape(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'pad_singleton' (line 255)
    pad_singleton_101568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 39), 'pad_singleton', False)
    # Processing the call keyword arguments (line 255)
    kwargs_101569 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice' (line 255)
    edge_slice_101563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'edge_slice', False)
    # Getting the type of 'arr' (line 255)
    arr_101564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 255)
    getitem___101565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 15), arr_101564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 255)
    subscript_call_result_101566 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), getitem___101565, edge_slice_101563)
    
    # Obtaining the member 'reshape' of a type (line 255)
    reshape_101567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 15), subscript_call_result_101566, 'reshape')
    # Calling reshape(args, kwargs) (line 255)
    reshape_call_result_101570 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), reshape_101567, *[pad_singleton_101568], **kwargs_101569)
    
    # Obtaining the member 'repeat' of a type (line 255)
    repeat_101571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 15), reshape_call_result_101570, 'repeat')
    # Calling repeat(args, kwargs) (line 255)
    repeat_call_result_101575 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), repeat_101571, *[pad_amt_101572, axis_101573], **kwargs_101574)
    
    # Assigning a type to the variable 'edge_pad' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'edge_pad', repeat_call_result_101575)
    
    # Assigning a BinOp to a Name (line 258):
    # Getting the type of 'end' (line 258)
    end_101576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'end')
    # Getting the type of 'edge_pad' (line 258)
    edge_pad_101577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'edge_pad')
    # Applying the binary operator '-' (line 258)
    result_sub_101578 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '-', end_101576, edge_pad_101577)
    
    
    # Call to float(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'pad_amt' (line 258)
    pad_amt_101580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'pad_amt', False)
    # Processing the call keyword arguments (line 258)
    kwargs_101581 = {}
    # Getting the type of 'float' (line 258)
    float_101579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'float', False)
    # Calling float(args, kwargs) (line 258)
    float_call_result_101582 = invoke(stypy.reporting.localization.Localization(__file__, 258, 31), float_101579, *[pad_amt_101580], **kwargs_101581)
    
    # Applying the binary operator 'div' (line 258)
    result_div_101583 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 12), 'div', result_sub_101578, float_call_result_101582)
    
    # Assigning a type to the variable 'slope' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'slope', result_div_101583)
    
    # Assigning a BinOp to a Name (line 259):
    # Getting the type of 'ramp_arr' (line 259)
    ramp_arr_101584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'ramp_arr')
    # Getting the type of 'slope' (line 259)
    slope_101585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 26), 'slope')
    # Applying the binary operator '*' (line 259)
    result_mul_101586 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), '*', ramp_arr_101584, slope_101585)
    
    # Assigning a type to the variable 'ramp_arr' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'ramp_arr', result_mul_101586)
    
    # Getting the type of 'ramp_arr' (line 260)
    ramp_arr_101587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'ramp_arr')
    # Getting the type of 'edge_pad' (line 260)
    edge_pad_101588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'edge_pad')
    # Applying the binary operator '+=' (line 260)
    result_iadd_101589 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 4), '+=', ramp_arr_101587, edge_pad_101588)
    # Assigning a type to the variable 'ramp_arr' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'ramp_arr', result_iadd_101589)
    
    
    # Call to _round_ifneeded(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'ramp_arr' (line 261)
    ramp_arr_101591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'ramp_arr', False)
    # Getting the type of 'arr' (line 261)
    arr_101592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 30), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 261)
    dtype_101593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 30), arr_101592, 'dtype')
    # Processing the call keyword arguments (line 261)
    kwargs_101594 = {}
    # Getting the type of '_round_ifneeded' (line 261)
    _round_ifneeded_101590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 261)
    _round_ifneeded_call_result_101595 = invoke(stypy.reporting.localization.Localization(__file__, 261, 4), _round_ifneeded_101590, *[ramp_arr_101591, dtype_101593], **kwargs_101594)
    
    
    # Call to concatenate(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_101598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    
    # Call to astype(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'arr' (line 264)
    arr_101601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 264)
    dtype_101602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 43), arr_101601, 'dtype')
    # Processing the call keyword arguments (line 264)
    kwargs_101603 = {}
    # Getting the type of 'ramp_arr' (line 264)
    ramp_arr_101599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 27), 'ramp_arr', False)
    # Obtaining the member 'astype' of a type (line 264)
    astype_101600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 27), ramp_arr_101599, 'astype')
    # Calling astype(args, kwargs) (line 264)
    astype_call_result_101604 = invoke(stypy.reporting.localization.Localization(__file__, 264, 27), astype_101600, *[dtype_101602], **kwargs_101603)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 27), tuple_101598, astype_call_result_101604)
    # Adding element type (line 264)
    # Getting the type of 'arr' (line 264)
    arr_101605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 55), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 27), tuple_101598, arr_101605)
    
    # Processing the call keyword arguments (line 264)
    # Getting the type of 'axis' (line 264)
    axis_101606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'axis', False)
    keyword_101607 = axis_101606
    kwargs_101608 = {'axis': keyword_101607}
    # Getting the type of 'np' (line 264)
    np_101596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 264)
    concatenate_101597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), np_101596, 'concatenate')
    # Calling concatenate(args, kwargs) (line 264)
    concatenate_call_result_101609 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), concatenate_101597, *[tuple_101598], **kwargs_101608)
    
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type', concatenate_call_result_101609)
    
    # ################# End of '_prepend_ramp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_ramp' in the type store
    # Getting the type of 'stypy_return_type' (line 212)
    stypy_return_type_101610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101610)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_ramp'
    return stypy_return_type_101610

# Assigning a type to the variable '_prepend_ramp' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), '_prepend_ramp', _prepend_ramp)

@norecursion
def _append_ramp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 41), 'int')
    defaults = [int_101611]
    # Create a new context for function '_append_ramp'
    module_type_store = module_type_store.open_function_context('_append_ramp', 267, 0, False)
    
    # Passed parameters checking function
    _append_ramp.stypy_localization = localization
    _append_ramp.stypy_type_of_self = None
    _append_ramp.stypy_type_store = module_type_store
    _append_ramp.stypy_function_name = '_append_ramp'
    _append_ramp.stypy_param_names_list = ['arr', 'pad_amt', 'end', 'axis']
    _append_ramp.stypy_varargs_param_name = None
    _append_ramp.stypy_kwargs_param_name = None
    _append_ramp.stypy_call_defaults = defaults
    _append_ramp.stypy_call_varargs = varargs
    _append_ramp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_ramp', ['arr', 'pad_amt', 'end', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_ramp', localization, ['arr', 'pad_amt', 'end', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_ramp(...)' code ##################

    str_101612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', '\n    Append linear ramp along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    end : scalar\n        Constal value to use. For best results should be of type `arr.dtype`;\n        if not `arr.dtype` will be cast to `arr.dtype`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region ramps linearly from the edge value to `end`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 290)
    pad_amt_101613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 7), 'pad_amt')
    int_101614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'int')
    # Applying the binary operator '==' (line 290)
    result_eq_101615 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 7), '==', pad_amt_101613, int_101614)
    
    # Testing the type of an if condition (line 290)
    if_condition_101616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 4), result_eq_101615)
    # Assigning a type to the variable 'if_condition_101616' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'if_condition_101616', if_condition_101616)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 291)
    arr_101617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', arr_101617)
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 294):
    
    # Call to tuple(...): (line 294)
    # Processing the call arguments (line 294)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 294, 21, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'arr' (line 295)
    arr_101626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 45), 'arr', False)
    # Obtaining the member 'shape' of a type (line 295)
    shape_101627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 45), arr_101626, 'shape')
    # Processing the call keyword arguments (line 295)
    kwargs_101628 = {}
    # Getting the type of 'enumerate' (line 295)
    enumerate_101625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 295)
    enumerate_call_result_101629 = invoke(stypy.reporting.localization.Localization(__file__, 295, 35), enumerate_101625, *[shape_101627], **kwargs_101628)
    
    comprehension_101630 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), enumerate_call_result_101629)
    # Assigning a type to the variable 'i' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), comprehension_101630))
    # Assigning a type to the variable 'x' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), comprehension_101630))
    
    
    # Getting the type of 'i' (line 294)
    i_101619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'i', False)
    # Getting the type of 'axis' (line 294)
    axis_101620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'axis', False)
    # Applying the binary operator '!=' (line 294)
    result_ne_101621 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 26), '!=', i_101619, axis_101620)
    
    # Testing the type of an if expression (line 294)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 21), result_ne_101621)
    # SSA begins for if expression (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 294)
    x_101622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'x', False)
    # SSA branch for the else part of an if expression (line 294)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'pad_amt' (line 294)
    pad_amt_101623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 41), 'pad_amt', False)
    # SSA join for if expression (line 294)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101624 = union_type.UnionType.add(x_101622, pad_amt_101623)
    
    list_101631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 21), list_101631, if_exp_101624)
    # Processing the call keyword arguments (line 294)
    kwargs_101632 = {}
    # Getting the type of 'tuple' (line 294)
    tuple_101618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 294)
    tuple_call_result_101633 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), tuple_101618, *[list_101631], **kwargs_101632)
    
    # Assigning a type to the variable 'padshape' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'padshape', tuple_call_result_101633)
    
    # Assigning a Call to a Name (line 298):
    
    # Call to astype(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'np' (line 299)
    np_101643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 53), 'np', False)
    # Obtaining the member 'float64' of a type (line 299)
    float64_101644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 53), np_101643, 'float64')
    # Processing the call keyword arguments (line 298)
    kwargs_101645 = {}
    
    # Call to _arange_ndarray(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'arr' (line 298)
    arr_101635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 31), 'arr', False)
    # Getting the type of 'padshape' (line 298)
    padshape_101636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'padshape', False)
    # Getting the type of 'axis' (line 298)
    axis_101637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 46), 'axis', False)
    # Processing the call keyword arguments (line 298)
    # Getting the type of 'False' (line 299)
    False_101638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 39), 'False', False)
    keyword_101639 = False_101638
    kwargs_101640 = {'reverse': keyword_101639}
    # Getting the type of '_arange_ndarray' (line 298)
    _arange_ndarray_101634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), '_arange_ndarray', False)
    # Calling _arange_ndarray(args, kwargs) (line 298)
    _arange_ndarray_call_result_101641 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), _arange_ndarray_101634, *[arr_101635, padshape_101636, axis_101637], **kwargs_101640)
    
    # Obtaining the member 'astype' of a type (line 298)
    astype_101642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), _arange_ndarray_call_result_101641, 'astype')
    # Calling astype(args, kwargs) (line 298)
    astype_call_result_101646 = invoke(stypy.reporting.localization.Localization(__file__, 298, 15), astype_101642, *[float64_101644], **kwargs_101645)
    
    # Assigning a type to the variable 'ramp_arr' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'ramp_arr', astype_call_result_101646)
    
    # Assigning a Call to a Name (line 302):
    
    # Call to tuple(...): (line 302)
    # Processing the call arguments (line 302)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 302, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'arr' (line 303)
    arr_101658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 303)
    shape_101659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 47), arr_101658, 'shape')
    # Processing the call keyword arguments (line 303)
    kwargs_101660 = {}
    # Getting the type of 'enumerate' (line 303)
    enumerate_101657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 303)
    enumerate_call_result_101661 = invoke(stypy.reporting.localization.Localization(__file__, 303, 37), enumerate_101657, *[shape_101659], **kwargs_101660)
    
    comprehension_101662 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 23), enumerate_call_result_101661)
    # Assigning a type to the variable 'i' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 23), comprehension_101662))
    # Assigning a type to the variable 'x' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 23), comprehension_101662))
    
    
    # Getting the type of 'i' (line 302)
    i_101648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'i', False)
    # Getting the type of 'axis' (line 302)
    axis_101649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 43), 'axis', False)
    # Applying the binary operator '!=' (line 302)
    result_ne_101650 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 38), '!=', i_101648, axis_101649)
    
    # Testing the type of an if expression (line 302)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 23), result_ne_101650)
    # SSA begins for if expression (line 302)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'None' (line 302)
    None_101652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 29), 'None', False)
    # Processing the call keyword arguments (line 302)
    kwargs_101653 = {}
    # Getting the type of 'slice' (line 302)
    slice_101651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 302)
    slice_call_result_101654 = invoke(stypy.reporting.localization.Localization(__file__, 302, 23), slice_101651, *[None_101652], **kwargs_101653)
    
    # SSA branch for the else part of an if expression (line 302)
    module_type_store.open_ssa_branch('if expression else')
    int_101655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 53), 'int')
    # SSA join for if expression (line 302)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101656 = union_type.UnionType.add(slice_call_result_101654, int_101655)
    
    list_101663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 23), list_101663, if_exp_101656)
    # Processing the call keyword arguments (line 302)
    kwargs_101664 = {}
    # Getting the type of 'tuple' (line 302)
    tuple_101647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 302)
    tuple_call_result_101665 = invoke(stypy.reporting.localization.Localization(__file__, 302, 17), tuple_101647, *[list_101663], **kwargs_101664)
    
    # Assigning a type to the variable 'edge_slice' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'edge_slice', tuple_call_result_101665)
    
    # Assigning a Call to a Name (line 306):
    
    # Call to tuple(...): (line 306)
    # Processing the call arguments (line 306)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 306, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 307)
    # Processing the call arguments (line 307)
    # Getting the type of 'arr' (line 307)
    arr_101674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 307)
    shape_101675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 50), arr_101674, 'shape')
    # Processing the call keyword arguments (line 307)
    kwargs_101676 = {}
    # Getting the type of 'enumerate' (line 307)
    enumerate_101673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 307)
    enumerate_call_result_101677 = invoke(stypy.reporting.localization.Localization(__file__, 307, 40), enumerate_101673, *[shape_101675], **kwargs_101676)
    
    comprehension_101678 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 26), enumerate_call_result_101677)
    # Assigning a type to the variable 'i' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 26), comprehension_101678))
    # Assigning a type to the variable 'x' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 26), comprehension_101678))
    
    
    # Getting the type of 'i' (line 306)
    i_101667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'i', False)
    # Getting the type of 'axis' (line 306)
    axis_101668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 36), 'axis', False)
    # Applying the binary operator '!=' (line 306)
    result_ne_101669 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 31), '!=', i_101667, axis_101668)
    
    # Testing the type of an if expression (line 306)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 26), result_ne_101669)
    # SSA begins for if expression (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 306)
    x_101670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 306)
    module_type_store.open_ssa_branch('if expression else')
    int_101671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 46), 'int')
    # SSA join for if expression (line 306)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101672 = union_type.UnionType.add(x_101670, int_101671)
    
    list_101679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 26), list_101679, if_exp_101672)
    # Processing the call keyword arguments (line 306)
    kwargs_101680 = {}
    # Getting the type of 'tuple' (line 306)
    tuple_101666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 306)
    tuple_call_result_101681 = invoke(stypy.reporting.localization.Localization(__file__, 306, 20), tuple_101666, *[list_101679], **kwargs_101680)
    
    # Assigning a type to the variable 'pad_singleton' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'pad_singleton', tuple_call_result_101681)
    
    # Assigning a Call to a Name (line 310):
    
    # Call to repeat(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'pad_amt' (line 310)
    pad_amt_101691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 61), 'pad_amt', False)
    # Getting the type of 'axis' (line 310)
    axis_101692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 70), 'axis', False)
    # Processing the call keyword arguments (line 310)
    kwargs_101693 = {}
    
    # Call to reshape(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'pad_singleton' (line 310)
    pad_singleton_101687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'pad_singleton', False)
    # Processing the call keyword arguments (line 310)
    kwargs_101688 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice' (line 310)
    edge_slice_101682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'edge_slice', False)
    # Getting the type of 'arr' (line 310)
    arr_101683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 310)
    getitem___101684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), arr_101683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 310)
    subscript_call_result_101685 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), getitem___101684, edge_slice_101682)
    
    # Obtaining the member 'reshape' of a type (line 310)
    reshape_101686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), subscript_call_result_101685, 'reshape')
    # Calling reshape(args, kwargs) (line 310)
    reshape_call_result_101689 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), reshape_101686, *[pad_singleton_101687], **kwargs_101688)
    
    # Obtaining the member 'repeat' of a type (line 310)
    repeat_101690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 15), reshape_call_result_101689, 'repeat')
    # Calling repeat(args, kwargs) (line 310)
    repeat_call_result_101694 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), repeat_101690, *[pad_amt_101691, axis_101692], **kwargs_101693)
    
    # Assigning a type to the variable 'edge_pad' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'edge_pad', repeat_call_result_101694)
    
    # Assigning a BinOp to a Name (line 313):
    # Getting the type of 'end' (line 313)
    end_101695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'end')
    # Getting the type of 'edge_pad' (line 313)
    edge_pad_101696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'edge_pad')
    # Applying the binary operator '-' (line 313)
    result_sub_101697 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 13), '-', end_101695, edge_pad_101696)
    
    
    # Call to float(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'pad_amt' (line 313)
    pad_amt_101699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'pad_amt', False)
    # Processing the call keyword arguments (line 313)
    kwargs_101700 = {}
    # Getting the type of 'float' (line 313)
    float_101698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 31), 'float', False)
    # Calling float(args, kwargs) (line 313)
    float_call_result_101701 = invoke(stypy.reporting.localization.Localization(__file__, 313, 31), float_101698, *[pad_amt_101699], **kwargs_101700)
    
    # Applying the binary operator 'div' (line 313)
    result_div_101702 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 12), 'div', result_sub_101697, float_call_result_101701)
    
    # Assigning a type to the variable 'slope' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'slope', result_div_101702)
    
    # Assigning a BinOp to a Name (line 314):
    # Getting the type of 'ramp_arr' (line 314)
    ramp_arr_101703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'ramp_arr')
    # Getting the type of 'slope' (line 314)
    slope_101704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'slope')
    # Applying the binary operator '*' (line 314)
    result_mul_101705 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 15), '*', ramp_arr_101703, slope_101704)
    
    # Assigning a type to the variable 'ramp_arr' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'ramp_arr', result_mul_101705)
    
    # Getting the type of 'ramp_arr' (line 315)
    ramp_arr_101706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'ramp_arr')
    # Getting the type of 'edge_pad' (line 315)
    edge_pad_101707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'edge_pad')
    # Applying the binary operator '+=' (line 315)
    result_iadd_101708 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 4), '+=', ramp_arr_101706, edge_pad_101707)
    # Assigning a type to the variable 'ramp_arr' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'ramp_arr', result_iadd_101708)
    
    
    # Call to _round_ifneeded(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'ramp_arr' (line 316)
    ramp_arr_101710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'ramp_arr', False)
    # Getting the type of 'arr' (line 316)
    arr_101711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 30), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 316)
    dtype_101712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 30), arr_101711, 'dtype')
    # Processing the call keyword arguments (line 316)
    kwargs_101713 = {}
    # Getting the type of '_round_ifneeded' (line 316)
    _round_ifneeded_101709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 316)
    _round_ifneeded_call_result_101714 = invoke(stypy.reporting.localization.Localization(__file__, 316, 4), _round_ifneeded_101709, *[ramp_arr_101710, dtype_101712], **kwargs_101713)
    
    
    # Call to concatenate(...): (line 319)
    # Processing the call arguments (line 319)
    
    # Obtaining an instance of the builtin type 'tuple' (line 319)
    tuple_101717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 319)
    # Adding element type (line 319)
    # Getting the type of 'arr' (line 319)
    arr_101718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 27), tuple_101717, arr_101718)
    # Adding element type (line 319)
    
    # Call to astype(...): (line 319)
    # Processing the call arguments (line 319)
    # Getting the type of 'arr' (line 319)
    arr_101721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 48), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 319)
    dtype_101722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 48), arr_101721, 'dtype')
    # Processing the call keyword arguments (line 319)
    kwargs_101723 = {}
    # Getting the type of 'ramp_arr' (line 319)
    ramp_arr_101719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 32), 'ramp_arr', False)
    # Obtaining the member 'astype' of a type (line 319)
    astype_101720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 32), ramp_arr_101719, 'astype')
    # Calling astype(args, kwargs) (line 319)
    astype_call_result_101724 = invoke(stypy.reporting.localization.Localization(__file__, 319, 32), astype_101720, *[dtype_101722], **kwargs_101723)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 27), tuple_101717, astype_call_result_101724)
    
    # Processing the call keyword arguments (line 319)
    # Getting the type of 'axis' (line 319)
    axis_101725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 66), 'axis', False)
    keyword_101726 = axis_101725
    kwargs_101727 = {'axis': keyword_101726}
    # Getting the type of 'np' (line 319)
    np_101715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 319)
    concatenate_101716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 11), np_101715, 'concatenate')
    # Calling concatenate(args, kwargs) (line 319)
    concatenate_call_result_101728 = invoke(stypy.reporting.localization.Localization(__file__, 319, 11), concatenate_101716, *[tuple_101717], **kwargs_101727)
    
    # Assigning a type to the variable 'stypy_return_type' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type', concatenate_call_result_101728)
    
    # ################# End of '_append_ramp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_ramp' in the type store
    # Getting the type of 'stypy_return_type' (line 267)
    stypy_return_type_101729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101729)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_ramp'
    return stypy_return_type_101729

# Assigning a type to the variable '_append_ramp' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), '_append_ramp', _append_ramp)

@norecursion
def _prepend_max(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 41), 'int')
    defaults = [int_101730]
    # Create a new context for function '_prepend_max'
    module_type_store = module_type_store.open_function_context('_prepend_max', 322, 0, False)
    
    # Passed parameters checking function
    _prepend_max.stypy_localization = localization
    _prepend_max.stypy_type_of_self = None
    _prepend_max.stypy_type_store = module_type_store
    _prepend_max.stypy_function_name = '_prepend_max'
    _prepend_max.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _prepend_max.stypy_varargs_param_name = None
    _prepend_max.stypy_kwargs_param_name = None
    _prepend_max.stypy_call_defaults = defaults
    _prepend_max.stypy_call_varargs = varargs
    _prepend_max.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_max', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_max', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_max(...)' code ##################

    str_101731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', '\n    Prepend `pad_amt` maximum values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate maximum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        prepended region is the maximum of the first `num` values along\n        `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 346)
    pad_amt_101732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 7), 'pad_amt')
    int_101733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 18), 'int')
    # Applying the binary operator '==' (line 346)
    result_eq_101734 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 7), '==', pad_amt_101732, int_101733)
    
    # Testing the type of an if condition (line 346)
    if_condition_101735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 4), result_eq_101734)
    # Assigning a type to the variable 'if_condition_101735' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'if_condition_101735', if_condition_101735)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 347)
    arr_101736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'stypy_return_type', arr_101736)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 350)
    num_101737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 7), 'num')
    int_101738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 14), 'int')
    # Applying the binary operator '==' (line 350)
    result_eq_101739 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 7), '==', num_101737, int_101738)
    
    # Testing the type of an if condition (line 350)
    if_condition_101740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 4), result_eq_101739)
    # Assigning a type to the variable 'if_condition_101740' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'if_condition_101740', if_condition_101740)
    # SSA begins for if statement (line 350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _prepend_edge(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'arr' (line 351)
    arr_101742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 29), 'arr', False)
    # Getting the type of 'pad_amt' (line 351)
    pad_amt_101743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'pad_amt', False)
    # Getting the type of 'axis' (line 351)
    axis_101744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 43), 'axis', False)
    # Processing the call keyword arguments (line 351)
    kwargs_101745 = {}
    # Getting the type of '_prepend_edge' (line 351)
    _prepend_edge_101741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 351)
    _prepend_edge_call_result_101746 = invoke(stypy.reporting.localization.Localization(__file__, 351, 15), _prepend_edge_101741, *[arr_101742, pad_amt_101743, axis_101744], **kwargs_101745)
    
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', _prepend_edge_call_result_101746)
    # SSA join for if statement (line 350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 354)
    # Getting the type of 'num' (line 354)
    num_101747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'num')
    # Getting the type of 'None' (line 354)
    None_101748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'None')
    
    (may_be_101749, more_types_in_union_101750) = may_not_be_none(num_101747, None_101748)

    if may_be_101749:

        if more_types_in_union_101750:
            # Runtime conditional SSA (line 354)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 355)
        num_101751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 355)
        axis_101752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 28), 'axis')
        # Getting the type of 'arr' (line 355)
        arr_101753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 355)
        shape_101754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 18), arr_101753, 'shape')
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___101755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 18), shape_101754, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_101756 = invoke(stypy.reporting.localization.Localization(__file__, 355, 18), getitem___101755, axis_101752)
        
        # Applying the binary operator '>=' (line 355)
        result_ge_101757 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), '>=', num_101751, subscript_call_result_101756)
        
        # Testing the type of an if condition (line 355)
        if_condition_101758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_ge_101757)
        # Assigning a type to the variable 'if_condition_101758' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_101758', if_condition_101758)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 356):
        # Getting the type of 'None' (line 356)
        None_101759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 18), 'None')
        # Assigning a type to the variable 'num' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'num', None_101759)
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_101750:
            # SSA join for if statement (line 354)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 359):
    
    # Call to tuple(...): (line 359)
    # Processing the call arguments (line 359)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 359, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'arr' (line 360)
    arr_101774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 360)
    shape_101775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 46), arr_101774, 'shape')
    # Processing the call keyword arguments (line 360)
    kwargs_101776 = {}
    # Getting the type of 'enumerate' (line 360)
    enumerate_101773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 360)
    enumerate_call_result_101777 = invoke(stypy.reporting.localization.Localization(__file__, 360, 36), enumerate_101773, *[shape_101775], **kwargs_101776)
    
    comprehension_101778 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 22), enumerate_call_result_101777)
    # Assigning a type to the variable 'i' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 22), comprehension_101778))
    # Assigning a type to the variable 'x' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 22), comprehension_101778))
    
    
    # Getting the type of 'i' (line 359)
    i_101761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 37), 'i', False)
    # Getting the type of 'axis' (line 359)
    axis_101762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 42), 'axis', False)
    # Applying the binary operator '!=' (line 359)
    result_ne_101763 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 37), '!=', i_101761, axis_101762)
    
    # Testing the type of an if expression (line 359)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 22), result_ne_101763)
    # SSA begins for if expression (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'None' (line 359)
    None_101765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 28), 'None', False)
    # Processing the call keyword arguments (line 359)
    kwargs_101766 = {}
    # Getting the type of 'slice' (line 359)
    slice_101764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 359)
    slice_call_result_101767 = invoke(stypy.reporting.localization.Localization(__file__, 359, 22), slice_101764, *[None_101765], **kwargs_101766)
    
    # SSA branch for the else part of an if expression (line 359)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'num' (line 359)
    num_101769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 58), 'num', False)
    # Processing the call keyword arguments (line 359)
    kwargs_101770 = {}
    # Getting the type of 'slice' (line 359)
    slice_101768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 359)
    slice_call_result_101771 = invoke(stypy.reporting.localization.Localization(__file__, 359, 52), slice_101768, *[num_101769], **kwargs_101770)
    
    # SSA join for if expression (line 359)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101772 = union_type.UnionType.add(slice_call_result_101767, slice_call_result_101771)
    
    list_101779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 22), list_101779, if_exp_101772)
    # Processing the call keyword arguments (line 359)
    kwargs_101780 = {}
    # Getting the type of 'tuple' (line 359)
    tuple_101760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 359)
    tuple_call_result_101781 = invoke(stypy.reporting.localization.Localization(__file__, 359, 16), tuple_101760, *[list_101779], **kwargs_101780)
    
    # Assigning a type to the variable 'max_slice' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'max_slice', tuple_call_result_101781)
    
    # Assigning a Call to a Name (line 363):
    
    # Call to tuple(...): (line 363)
    # Processing the call arguments (line 363)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 363, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'arr' (line 364)
    arr_101790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 364)
    shape_101791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 50), arr_101790, 'shape')
    # Processing the call keyword arguments (line 364)
    kwargs_101792 = {}
    # Getting the type of 'enumerate' (line 364)
    enumerate_101789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 364)
    enumerate_call_result_101793 = invoke(stypy.reporting.localization.Localization(__file__, 364, 40), enumerate_101789, *[shape_101791], **kwargs_101792)
    
    comprehension_101794 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), enumerate_call_result_101793)
    # Assigning a type to the variable 'i' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), comprehension_101794))
    # Assigning a type to the variable 'x' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), comprehension_101794))
    
    
    # Getting the type of 'i' (line 363)
    i_101783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 31), 'i', False)
    # Getting the type of 'axis' (line 363)
    axis_101784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 36), 'axis', False)
    # Applying the binary operator '!=' (line 363)
    result_ne_101785 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 31), '!=', i_101783, axis_101784)
    
    # Testing the type of an if expression (line 363)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 363, 26), result_ne_101785)
    # SSA begins for if expression (line 363)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 363)
    x_101786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 363)
    module_type_store.open_ssa_branch('if expression else')
    int_101787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 46), 'int')
    # SSA join for if expression (line 363)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101788 = union_type.UnionType.add(x_101786, int_101787)
    
    list_101795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), list_101795, if_exp_101788)
    # Processing the call keyword arguments (line 363)
    kwargs_101796 = {}
    # Getting the type of 'tuple' (line 363)
    tuple_101782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 363)
    tuple_call_result_101797 = invoke(stypy.reporting.localization.Localization(__file__, 363, 20), tuple_101782, *[list_101795], **kwargs_101796)
    
    # Assigning a type to the variable 'pad_singleton' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'pad_singleton', tuple_call_result_101797)
    
    # Assigning a Call to a Name (line 367):
    
    # Call to reshape(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'pad_singleton' (line 367)
    pad_singleton_101808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 54), 'pad_singleton', False)
    # Processing the call keyword arguments (line 367)
    kwargs_101809 = {}
    
    # Call to max(...): (line 367)
    # Processing the call keyword arguments (line 367)
    # Getting the type of 'axis' (line 367)
    axis_101803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 40), 'axis', False)
    keyword_101804 = axis_101803
    kwargs_101805 = {'axis': keyword_101804}
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_slice' (line 367)
    max_slice_101798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'max_slice', False)
    # Getting the type of 'arr' (line 367)
    arr_101799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 367)
    getitem___101800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), arr_101799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 367)
    subscript_call_result_101801 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), getitem___101800, max_slice_101798)
    
    # Obtaining the member 'max' of a type (line 367)
    max_101802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), subscript_call_result_101801, 'max')
    # Calling max(args, kwargs) (line 367)
    max_call_result_101806 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), max_101802, *[], **kwargs_101805)
    
    # Obtaining the member 'reshape' of a type (line 367)
    reshape_101807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 16), max_call_result_101806, 'reshape')
    # Calling reshape(args, kwargs) (line 367)
    reshape_call_result_101810 = invoke(stypy.reporting.localization.Localization(__file__, 367, 16), reshape_101807, *[pad_singleton_101808], **kwargs_101809)
    
    # Assigning a type to the variable 'max_chunk' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'max_chunk', reshape_call_result_101810)
    
    # Call to concatenate(...): (line 370)
    # Processing the call arguments (line 370)
    
    # Obtaining an instance of the builtin type 'tuple' (line 370)
    tuple_101813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 370)
    # Adding element type (line 370)
    
    # Call to repeat(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'pad_amt' (line 370)
    pad_amt_101816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 44), 'pad_amt', False)
    # Processing the call keyword arguments (line 370)
    # Getting the type of 'axis' (line 370)
    axis_101817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 58), 'axis', False)
    keyword_101818 = axis_101817
    kwargs_101819 = {'axis': keyword_101818}
    # Getting the type of 'max_chunk' (line 370)
    max_chunk_101814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'max_chunk', False)
    # Obtaining the member 'repeat' of a type (line 370)
    repeat_101815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 27), max_chunk_101814, 'repeat')
    # Calling repeat(args, kwargs) (line 370)
    repeat_call_result_101820 = invoke(stypy.reporting.localization.Localization(__file__, 370, 27), repeat_101815, *[pad_amt_101816], **kwargs_101819)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 27), tuple_101813, repeat_call_result_101820)
    # Adding element type (line 370)
    # Getting the type of 'arr' (line 370)
    arr_101821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 65), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 27), tuple_101813, arr_101821)
    
    # Processing the call keyword arguments (line 370)
    # Getting the type of 'axis' (line 371)
    axis_101822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 31), 'axis', False)
    keyword_101823 = axis_101822
    kwargs_101824 = {'axis': keyword_101823}
    # Getting the type of 'np' (line 370)
    np_101811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 370)
    concatenate_101812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 11), np_101811, 'concatenate')
    # Calling concatenate(args, kwargs) (line 370)
    concatenate_call_result_101825 = invoke(stypy.reporting.localization.Localization(__file__, 370, 11), concatenate_101812, *[tuple_101813], **kwargs_101824)
    
    # Assigning a type to the variable 'stypy_return_type' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type', concatenate_call_result_101825)
    
    # ################# End of '_prepend_max(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_max' in the type store
    # Getting the type of 'stypy_return_type' (line 322)
    stypy_return_type_101826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101826)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_max'
    return stypy_return_type_101826

# Assigning a type to the variable '_prepend_max' (line 322)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 0), '_prepend_max', _prepend_max)

@norecursion
def _append_max(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 40), 'int')
    defaults = [int_101827]
    # Create a new context for function '_append_max'
    module_type_store = module_type_store.open_function_context('_append_max', 374, 0, False)
    
    # Passed parameters checking function
    _append_max.stypy_localization = localization
    _append_max.stypy_type_of_self = None
    _append_max.stypy_type_store = module_type_store
    _append_max.stypy_function_name = '_append_max'
    _append_max.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _append_max.stypy_varargs_param_name = None
    _append_max.stypy_kwargs_param_name = None
    _append_max.stypy_call_defaults = defaults
    _append_max.stypy_call_varargs = varargs
    _append_max.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_max', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_max', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_max(...)' code ##################

    str_101828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, (-1)), 'str', '\n    Pad one `axis` of `arr` with the maximum of the last `num` elements.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate maximum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the maximum of the final `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 397)
    pad_amt_101829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 7), 'pad_amt')
    int_101830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 18), 'int')
    # Applying the binary operator '==' (line 397)
    result_eq_101831 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 7), '==', pad_amt_101829, int_101830)
    
    # Testing the type of an if condition (line 397)
    if_condition_101832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 4), result_eq_101831)
    # Assigning a type to the variable 'if_condition_101832' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'if_condition_101832', if_condition_101832)
    # SSA begins for if statement (line 397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 398)
    arr_101833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type', arr_101833)
    # SSA join for if statement (line 397)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 401)
    num_101834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 7), 'num')
    int_101835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 14), 'int')
    # Applying the binary operator '==' (line 401)
    result_eq_101836 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 7), '==', num_101834, int_101835)
    
    # Testing the type of an if condition (line 401)
    if_condition_101837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 4), result_eq_101836)
    # Assigning a type to the variable 'if_condition_101837' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'if_condition_101837', if_condition_101837)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _append_edge(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'arr' (line 402)
    arr_101839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 28), 'arr', False)
    # Getting the type of 'pad_amt' (line 402)
    pad_amt_101840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 33), 'pad_amt', False)
    # Getting the type of 'axis' (line 402)
    axis_101841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 42), 'axis', False)
    # Processing the call keyword arguments (line 402)
    kwargs_101842 = {}
    # Getting the type of '_append_edge' (line 402)
    _append_edge_101838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 402)
    _append_edge_call_result_101843 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), _append_edge_101838, *[arr_101839, pad_amt_101840, axis_101841], **kwargs_101842)
    
    # Assigning a type to the variable 'stypy_return_type' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'stypy_return_type', _append_edge_call_result_101843)
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 405)
    # Getting the type of 'num' (line 405)
    num_101844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'num')
    # Getting the type of 'None' (line 405)
    None_101845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 18), 'None')
    
    (may_be_101846, more_types_in_union_101847) = may_not_be_none(num_101844, None_101845)

    if may_be_101846:

        if more_types_in_union_101847:
            # Runtime conditional SSA (line 405)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 406)
        num_101848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 406)
        axis_101849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 28), 'axis')
        # Getting the type of 'arr' (line 406)
        arr_101850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 406)
        shape_101851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 18), arr_101850, 'shape')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___101852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 18), shape_101851, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_101853 = invoke(stypy.reporting.localization.Localization(__file__, 406, 18), getitem___101852, axis_101849)
        
        # Applying the binary operator '>=' (line 406)
        result_ge_101854 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 11), '>=', num_101848, subscript_call_result_101853)
        
        # Testing the type of an if condition (line 406)
        if_condition_101855 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 8), result_ge_101854)
        # Assigning a type to the variable 'if_condition_101855' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'if_condition_101855', if_condition_101855)
        # SSA begins for if statement (line 406)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 407):
        # Getting the type of 'None' (line 407)
        None_101856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 18), 'None')
        # Assigning a type to the variable 'num' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'num', None_101856)
        # SSA join for if statement (line 406)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_101847:
            # SSA join for if statement (line 405)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 410):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 410)
    axis_101857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'axis')
    # Getting the type of 'arr' (line 410)
    arr_101858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 410)
    shape_101859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 10), arr_101858, 'shape')
    # Obtaining the member '__getitem__' of a type (line 410)
    getitem___101860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 10), shape_101859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 410)
    subscript_call_result_101861 = invoke(stypy.reporting.localization.Localization(__file__, 410, 10), getitem___101860, axis_101857)
    
    int_101862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 28), 'int')
    # Applying the binary operator '-' (line 410)
    result_sub_101863 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 10), '-', subscript_call_result_101861, int_101862)
    
    # Assigning a type to the variable 'end' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'end', result_sub_101863)
    
    # Type idiom detected: calculating its left and rigth part (line 411)
    # Getting the type of 'num' (line 411)
    num_101864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'num')
    # Getting the type of 'None' (line 411)
    None_101865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 18), 'None')
    
    (may_be_101866, more_types_in_union_101867) = may_not_be_none(num_101864, None_101865)

    if may_be_101866:

        if more_types_in_union_101867:
            # Runtime conditional SSA (line 411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 412):
        
        # Call to tuple(...): (line 412)
        # Processing the call arguments (line 412)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 413, 12, True)
        # Calculating comprehension expression
        
        # Call to enumerate(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'arr' (line 414)
        arr_101886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 36), 'arr', False)
        # Obtaining the member 'shape' of a type (line 414)
        shape_101887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 36), arr_101886, 'shape')
        # Processing the call keyword arguments (line 414)
        kwargs_101888 = {}
        # Getting the type of 'enumerate' (line 414)
        enumerate_101885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 414)
        enumerate_call_result_101889 = invoke(stypy.reporting.localization.Localization(__file__, 414, 26), enumerate_101885, *[shape_101887], **kwargs_101888)
        
        comprehension_101890 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), enumerate_call_result_101889)
        # Assigning a type to the variable 'i' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), comprehension_101890))
        # Assigning a type to the variable 'x' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), comprehension_101890))
        
        
        # Getting the type of 'i' (line 413)
        i_101869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 27), 'i', False)
        # Getting the type of 'axis' (line 413)
        axis_101870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'axis', False)
        # Applying the binary operator '!=' (line 413)
        result_ne_101871 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 27), '!=', i_101869, axis_101870)
        
        # Testing the type of an if expression (line 413)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 12), result_ne_101871)
        # SSA begins for if expression (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'None' (line 413)
        None_101873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 18), 'None', False)
        # Processing the call keyword arguments (line 413)
        kwargs_101874 = {}
        # Getting the type of 'slice' (line 413)
        slice_101872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'slice', False)
        # Calling slice(args, kwargs) (line 413)
        slice_call_result_101875 = invoke(stypy.reporting.localization.Localization(__file__, 413, 12), slice_101872, *[None_101873], **kwargs_101874)
        
        # SSA branch for the else part of an if expression (line 413)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to slice(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'end' (line 413)
        end_101877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 48), 'end', False)
        # Getting the type of 'end' (line 413)
        end_101878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 53), 'end', False)
        # Getting the type of 'num' (line 413)
        num_101879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 59), 'num', False)
        # Applying the binary operator '-' (line 413)
        result_sub_101880 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 53), '-', end_101878, num_101879)
        
        int_101881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 64), 'int')
        # Processing the call keyword arguments (line 413)
        kwargs_101882 = {}
        # Getting the type of 'slice' (line 413)
        slice_101876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 42), 'slice', False)
        # Calling slice(args, kwargs) (line 413)
        slice_call_result_101883 = invoke(stypy.reporting.localization.Localization(__file__, 413, 42), slice_101876, *[end_101877, result_sub_101880, int_101881], **kwargs_101882)
        
        # SSA join for if expression (line 413)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_101884 = union_type.UnionType.add(slice_call_result_101875, slice_call_result_101883)
        
        list_101891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 12), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_101891, if_exp_101884)
        # Processing the call keyword arguments (line 412)
        kwargs_101892 = {}
        # Getting the type of 'tuple' (line 412)
        tuple_101868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 412)
        tuple_call_result_101893 = invoke(stypy.reporting.localization.Localization(__file__, 412, 20), tuple_101868, *[list_101891], **kwargs_101892)
        
        # Assigning a type to the variable 'max_slice' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'max_slice', tuple_call_result_101893)

        if more_types_in_union_101867:
            # Runtime conditional SSA for else branch (line 411)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_101866) or more_types_in_union_101867):
        
        # Assigning a Call to a Name (line 416):
        
        # Call to tuple(...): (line 416)
        # Processing the call arguments (line 416)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 416, 26, True)
        # Calculating comprehension expression
        # Getting the type of 'arr' (line 416)
        arr_101899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 47), 'arr', False)
        # Obtaining the member 'shape' of a type (line 416)
        shape_101900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 47), arr_101899, 'shape')
        comprehension_101901 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 26), shape_101900)
        # Assigning a type to the variable 'x' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'x', comprehension_101901)
        
        # Call to slice(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'None' (line 416)
        None_101896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'None', False)
        # Processing the call keyword arguments (line 416)
        kwargs_101897 = {}
        # Getting the type of 'slice' (line 416)
        slice_101895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 26), 'slice', False)
        # Calling slice(args, kwargs) (line 416)
        slice_call_result_101898 = invoke(stypy.reporting.localization.Localization(__file__, 416, 26), slice_101895, *[None_101896], **kwargs_101897)
        
        list_101902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 26), list_101902, slice_call_result_101898)
        # Processing the call keyword arguments (line 416)
        kwargs_101903 = {}
        # Getting the type of 'tuple' (line 416)
        tuple_101894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 416)
        tuple_call_result_101904 = invoke(stypy.reporting.localization.Localization(__file__, 416, 20), tuple_101894, *[list_101902], **kwargs_101903)
        
        # Assigning a type to the variable 'max_slice' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'max_slice', tuple_call_result_101904)

        if (may_be_101866 and more_types_in_union_101867):
            # SSA join for if statement (line 411)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 419):
    
    # Call to tuple(...): (line 419)
    # Processing the call arguments (line 419)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 419, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'arr' (line 420)
    arr_101913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 420)
    shape_101914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 50), arr_101913, 'shape')
    # Processing the call keyword arguments (line 420)
    kwargs_101915 = {}
    # Getting the type of 'enumerate' (line 420)
    enumerate_101912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 420)
    enumerate_call_result_101916 = invoke(stypy.reporting.localization.Localization(__file__, 420, 40), enumerate_101912, *[shape_101914], **kwargs_101915)
    
    comprehension_101917 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 26), enumerate_call_result_101916)
    # Assigning a type to the variable 'i' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 26), comprehension_101917))
    # Assigning a type to the variable 'x' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 26), comprehension_101917))
    
    
    # Getting the type of 'i' (line 419)
    i_101906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 31), 'i', False)
    # Getting the type of 'axis' (line 419)
    axis_101907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 36), 'axis', False)
    # Applying the binary operator '!=' (line 419)
    result_ne_101908 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 31), '!=', i_101906, axis_101907)
    
    # Testing the type of an if expression (line 419)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 26), result_ne_101908)
    # SSA begins for if expression (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 419)
    x_101909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 419)
    module_type_store.open_ssa_branch('if expression else')
    int_101910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 46), 'int')
    # SSA join for if expression (line 419)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101911 = union_type.UnionType.add(x_101909, int_101910)
    
    list_101918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 26), list_101918, if_exp_101911)
    # Processing the call keyword arguments (line 419)
    kwargs_101919 = {}
    # Getting the type of 'tuple' (line 419)
    tuple_101905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 419)
    tuple_call_result_101920 = invoke(stypy.reporting.localization.Localization(__file__, 419, 20), tuple_101905, *[list_101918], **kwargs_101919)
    
    # Assigning a type to the variable 'pad_singleton' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'pad_singleton', tuple_call_result_101920)
    
    # Assigning a Call to a Name (line 423):
    
    # Call to reshape(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'pad_singleton' (line 423)
    pad_singleton_101931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 54), 'pad_singleton', False)
    # Processing the call keyword arguments (line 423)
    kwargs_101932 = {}
    
    # Call to max(...): (line 423)
    # Processing the call keyword arguments (line 423)
    # Getting the type of 'axis' (line 423)
    axis_101926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 40), 'axis', False)
    keyword_101927 = axis_101926
    kwargs_101928 = {'axis': keyword_101927}
    
    # Obtaining the type of the subscript
    # Getting the type of 'max_slice' (line 423)
    max_slice_101921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 20), 'max_slice', False)
    # Getting the type of 'arr' (line 423)
    arr_101922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___101923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), arr_101922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_101924 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), getitem___101923, max_slice_101921)
    
    # Obtaining the member 'max' of a type (line 423)
    max_101925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), subscript_call_result_101924, 'max')
    # Calling max(args, kwargs) (line 423)
    max_call_result_101929 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), max_101925, *[], **kwargs_101928)
    
    # Obtaining the member 'reshape' of a type (line 423)
    reshape_101930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 16), max_call_result_101929, 'reshape')
    # Calling reshape(args, kwargs) (line 423)
    reshape_call_result_101933 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), reshape_101930, *[pad_singleton_101931], **kwargs_101932)
    
    # Assigning a type to the variable 'max_chunk' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'max_chunk', reshape_call_result_101933)
    
    # Call to concatenate(...): (line 426)
    # Processing the call arguments (line 426)
    
    # Obtaining an instance of the builtin type 'tuple' (line 426)
    tuple_101936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 426)
    # Adding element type (line 426)
    # Getting the type of 'arr' (line 426)
    arr_101937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 27), tuple_101936, arr_101937)
    # Adding element type (line 426)
    
    # Call to repeat(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'pad_amt' (line 426)
    pad_amt_101940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 49), 'pad_amt', False)
    # Processing the call keyword arguments (line 426)
    # Getting the type of 'axis' (line 426)
    axis_101941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 63), 'axis', False)
    keyword_101942 = axis_101941
    kwargs_101943 = {'axis': keyword_101942}
    # Getting the type of 'max_chunk' (line 426)
    max_chunk_101938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 32), 'max_chunk', False)
    # Obtaining the member 'repeat' of a type (line 426)
    repeat_101939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 32), max_chunk_101938, 'repeat')
    # Calling repeat(args, kwargs) (line 426)
    repeat_call_result_101944 = invoke(stypy.reporting.localization.Localization(__file__, 426, 32), repeat_101939, *[pad_amt_101940], **kwargs_101943)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 27), tuple_101936, repeat_call_result_101944)
    
    # Processing the call keyword arguments (line 426)
    # Getting the type of 'axis' (line 427)
    axis_101945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 31), 'axis', False)
    keyword_101946 = axis_101945
    kwargs_101947 = {'axis': keyword_101946}
    # Getting the type of 'np' (line 426)
    np_101934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 426)
    concatenate_101935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 11), np_101934, 'concatenate')
    # Calling concatenate(args, kwargs) (line 426)
    concatenate_call_result_101948 = invoke(stypy.reporting.localization.Localization(__file__, 426, 11), concatenate_101935, *[tuple_101936], **kwargs_101947)
    
    # Assigning a type to the variable 'stypy_return_type' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type', concatenate_call_result_101948)
    
    # ################# End of '_append_max(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_max' in the type store
    # Getting the type of 'stypy_return_type' (line 374)
    stypy_return_type_101949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101949)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_max'
    return stypy_return_type_101949

# Assigning a type to the variable '_append_max' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), '_append_max', _append_max)

@norecursion
def _prepend_mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_101950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 42), 'int')
    defaults = [int_101950]
    # Create a new context for function '_prepend_mean'
    module_type_store = module_type_store.open_function_context('_prepend_mean', 430, 0, False)
    
    # Passed parameters checking function
    _prepend_mean.stypy_localization = localization
    _prepend_mean.stypy_type_of_self = None
    _prepend_mean.stypy_type_store = module_type_store
    _prepend_mean.stypy_function_name = '_prepend_mean'
    _prepend_mean.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _prepend_mean.stypy_varargs_param_name = None
    _prepend_mean.stypy_kwargs_param_name = None
    _prepend_mean.stypy_call_defaults = defaults
    _prepend_mean.stypy_call_varargs = varargs
    _prepend_mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_mean', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_mean', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_mean(...)' code ##################

    str_101951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, (-1)), 'str', '\n    Prepend `pad_amt` mean values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate mean.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the mean of the first `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 453)
    pad_amt_101952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 7), 'pad_amt')
    int_101953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 18), 'int')
    # Applying the binary operator '==' (line 453)
    result_eq_101954 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 7), '==', pad_amt_101952, int_101953)
    
    # Testing the type of an if condition (line 453)
    if_condition_101955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 4), result_eq_101954)
    # Assigning a type to the variable 'if_condition_101955' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'if_condition_101955', if_condition_101955)
    # SSA begins for if statement (line 453)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 454)
    arr_101956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'stypy_return_type', arr_101956)
    # SSA join for if statement (line 453)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 457)
    num_101957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 7), 'num')
    int_101958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 14), 'int')
    # Applying the binary operator '==' (line 457)
    result_eq_101959 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 7), '==', num_101957, int_101958)
    
    # Testing the type of an if condition (line 457)
    if_condition_101960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 4), result_eq_101959)
    # Assigning a type to the variable 'if_condition_101960' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'if_condition_101960', if_condition_101960)
    # SSA begins for if statement (line 457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _prepend_edge(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'arr' (line 458)
    arr_101962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 29), 'arr', False)
    # Getting the type of 'pad_amt' (line 458)
    pad_amt_101963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 34), 'pad_amt', False)
    # Getting the type of 'axis' (line 458)
    axis_101964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 43), 'axis', False)
    # Processing the call keyword arguments (line 458)
    kwargs_101965 = {}
    # Getting the type of '_prepend_edge' (line 458)
    _prepend_edge_101961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 15), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 458)
    _prepend_edge_call_result_101966 = invoke(stypy.reporting.localization.Localization(__file__, 458, 15), _prepend_edge_101961, *[arr_101962, pad_amt_101963, axis_101964], **kwargs_101965)
    
    # Assigning a type to the variable 'stypy_return_type' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type', _prepend_edge_call_result_101966)
    # SSA join for if statement (line 457)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 461)
    # Getting the type of 'num' (line 461)
    num_101967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'num')
    # Getting the type of 'None' (line 461)
    None_101968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 18), 'None')
    
    (may_be_101969, more_types_in_union_101970) = may_not_be_none(num_101967, None_101968)

    if may_be_101969:

        if more_types_in_union_101970:
            # Runtime conditional SSA (line 461)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 462)
        num_101971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 462)
        axis_101972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 28), 'axis')
        # Getting the type of 'arr' (line 462)
        arr_101973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 462)
        shape_101974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), arr_101973, 'shape')
        # Obtaining the member '__getitem__' of a type (line 462)
        getitem___101975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 18), shape_101974, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 462)
        subscript_call_result_101976 = invoke(stypy.reporting.localization.Localization(__file__, 462, 18), getitem___101975, axis_101972)
        
        # Applying the binary operator '>=' (line 462)
        result_ge_101977 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 11), '>=', num_101971, subscript_call_result_101976)
        
        # Testing the type of an if condition (line 462)
        if_condition_101978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 462, 8), result_ge_101977)
        # Assigning a type to the variable 'if_condition_101978' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'if_condition_101978', if_condition_101978)
        # SSA begins for if statement (line 462)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 463):
        # Getting the type of 'None' (line 463)
        None_101979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 18), 'None')
        # Assigning a type to the variable 'num' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'num', None_101979)
        # SSA join for if statement (line 462)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_101970:
            # SSA join for if statement (line 461)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 466):
    
    # Call to tuple(...): (line 466)
    # Processing the call arguments (line 466)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 466, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'arr' (line 467)
    arr_101994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 467)
    shape_101995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 47), arr_101994, 'shape')
    # Processing the call keyword arguments (line 467)
    kwargs_101996 = {}
    # Getting the type of 'enumerate' (line 467)
    enumerate_101993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 467)
    enumerate_call_result_101997 = invoke(stypy.reporting.localization.Localization(__file__, 467, 37), enumerate_101993, *[shape_101995], **kwargs_101996)
    
    comprehension_101998 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 23), enumerate_call_result_101997)
    # Assigning a type to the variable 'i' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 23), comprehension_101998))
    # Assigning a type to the variable 'x' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 23), comprehension_101998))
    
    
    # Getting the type of 'i' (line 466)
    i_101981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 38), 'i', False)
    # Getting the type of 'axis' (line 466)
    axis_101982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 43), 'axis', False)
    # Applying the binary operator '!=' (line 466)
    result_ne_101983 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 38), '!=', i_101981, axis_101982)
    
    # Testing the type of an if expression (line 466)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 23), result_ne_101983)
    # SSA begins for if expression (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'None' (line 466)
    None_101985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 29), 'None', False)
    # Processing the call keyword arguments (line 466)
    kwargs_101986 = {}
    # Getting the type of 'slice' (line 466)
    slice_101984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 466)
    slice_call_result_101987 = invoke(stypy.reporting.localization.Localization(__file__, 466, 23), slice_101984, *[None_101985], **kwargs_101986)
    
    # SSA branch for the else part of an if expression (line 466)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'num' (line 466)
    num_101989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 59), 'num', False)
    # Processing the call keyword arguments (line 466)
    kwargs_101990 = {}
    # Getting the type of 'slice' (line 466)
    slice_101988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 53), 'slice', False)
    # Calling slice(args, kwargs) (line 466)
    slice_call_result_101991 = invoke(stypy.reporting.localization.Localization(__file__, 466, 53), slice_101988, *[num_101989], **kwargs_101990)
    
    # SSA join for if expression (line 466)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_101992 = union_type.UnionType.add(slice_call_result_101987, slice_call_result_101991)
    
    list_101999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 23), list_101999, if_exp_101992)
    # Processing the call keyword arguments (line 466)
    kwargs_102000 = {}
    # Getting the type of 'tuple' (line 466)
    tuple_101980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 466)
    tuple_call_result_102001 = invoke(stypy.reporting.localization.Localization(__file__, 466, 17), tuple_101980, *[list_101999], **kwargs_102000)
    
    # Assigning a type to the variable 'mean_slice' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'mean_slice', tuple_call_result_102001)
    
    # Assigning a Call to a Name (line 470):
    
    # Call to tuple(...): (line 470)
    # Processing the call arguments (line 470)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 470, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'arr' (line 471)
    arr_102010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 471)
    shape_102011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 50), arr_102010, 'shape')
    # Processing the call keyword arguments (line 471)
    kwargs_102012 = {}
    # Getting the type of 'enumerate' (line 471)
    enumerate_102009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 471)
    enumerate_call_result_102013 = invoke(stypy.reporting.localization.Localization(__file__, 471, 40), enumerate_102009, *[shape_102011], **kwargs_102012)
    
    comprehension_102014 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 26), enumerate_call_result_102013)
    # Assigning a type to the variable 'i' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 26), comprehension_102014))
    # Assigning a type to the variable 'x' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 26), comprehension_102014))
    
    
    # Getting the type of 'i' (line 470)
    i_102003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'i', False)
    # Getting the type of 'axis' (line 470)
    axis_102004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 36), 'axis', False)
    # Applying the binary operator '!=' (line 470)
    result_ne_102005 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 31), '!=', i_102003, axis_102004)
    
    # Testing the type of an if expression (line 470)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 470, 26), result_ne_102005)
    # SSA begins for if expression (line 470)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 470)
    x_102006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 470)
    module_type_store.open_ssa_branch('if expression else')
    int_102007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 46), 'int')
    # SSA join for if expression (line 470)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102008 = union_type.UnionType.add(x_102006, int_102007)
    
    list_102015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 26), list_102015, if_exp_102008)
    # Processing the call keyword arguments (line 470)
    kwargs_102016 = {}
    # Getting the type of 'tuple' (line 470)
    tuple_102002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 470)
    tuple_call_result_102017 = invoke(stypy.reporting.localization.Localization(__file__, 470, 20), tuple_102002, *[list_102015], **kwargs_102016)
    
    # Assigning a type to the variable 'pad_singleton' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'pad_singleton', tuple_call_result_102017)
    
    # Assigning a Call to a Name (line 474):
    
    # Call to reshape(...): (line 474)
    # Processing the call arguments (line 474)
    # Getting the type of 'pad_singleton' (line 474)
    pad_singleton_102027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 52), 'pad_singleton', False)
    # Processing the call keyword arguments (line 474)
    kwargs_102028 = {}
    
    # Call to mean(...): (line 474)
    # Processing the call arguments (line 474)
    # Getting the type of 'axis' (line 474)
    axis_102023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 38), 'axis', False)
    # Processing the call keyword arguments (line 474)
    kwargs_102024 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'mean_slice' (line 474)
    mean_slice_102018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 21), 'mean_slice', False)
    # Getting the type of 'arr' (line 474)
    arr_102019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 17), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 474)
    getitem___102020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 17), arr_102019, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 474)
    subscript_call_result_102021 = invoke(stypy.reporting.localization.Localization(__file__, 474, 17), getitem___102020, mean_slice_102018)
    
    # Obtaining the member 'mean' of a type (line 474)
    mean_102022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 17), subscript_call_result_102021, 'mean')
    # Calling mean(args, kwargs) (line 474)
    mean_call_result_102025 = invoke(stypy.reporting.localization.Localization(__file__, 474, 17), mean_102022, *[axis_102023], **kwargs_102024)
    
    # Obtaining the member 'reshape' of a type (line 474)
    reshape_102026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 17), mean_call_result_102025, 'reshape')
    # Calling reshape(args, kwargs) (line 474)
    reshape_call_result_102029 = invoke(stypy.reporting.localization.Localization(__file__, 474, 17), reshape_102026, *[pad_singleton_102027], **kwargs_102028)
    
    # Assigning a type to the variable 'mean_chunk' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'mean_chunk', reshape_call_result_102029)
    
    # Call to _round_ifneeded(...): (line 475)
    # Processing the call arguments (line 475)
    # Getting the type of 'mean_chunk' (line 475)
    mean_chunk_102031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'mean_chunk', False)
    # Getting the type of 'arr' (line 475)
    arr_102032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 32), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 475)
    dtype_102033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 32), arr_102032, 'dtype')
    # Processing the call keyword arguments (line 475)
    kwargs_102034 = {}
    # Getting the type of '_round_ifneeded' (line 475)
    _round_ifneeded_102030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 475)
    _round_ifneeded_call_result_102035 = invoke(stypy.reporting.localization.Localization(__file__, 475, 4), _round_ifneeded_102030, *[mean_chunk_102031, dtype_102033], **kwargs_102034)
    
    
    # Call to concatenate(...): (line 478)
    # Processing the call arguments (line 478)
    
    # Obtaining an instance of the builtin type 'tuple' (line 478)
    tuple_102038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 478)
    # Adding element type (line 478)
    
    # Call to astype(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'arr' (line 478)
    arr_102046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 67), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 478)
    dtype_102047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 67), arr_102046, 'dtype')
    # Processing the call keyword arguments (line 478)
    kwargs_102048 = {}
    
    # Call to repeat(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'pad_amt' (line 478)
    pad_amt_102041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 45), 'pad_amt', False)
    # Getting the type of 'axis' (line 478)
    axis_102042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 54), 'axis', False)
    # Processing the call keyword arguments (line 478)
    kwargs_102043 = {}
    # Getting the type of 'mean_chunk' (line 478)
    mean_chunk_102039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 27), 'mean_chunk', False)
    # Obtaining the member 'repeat' of a type (line 478)
    repeat_102040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 27), mean_chunk_102039, 'repeat')
    # Calling repeat(args, kwargs) (line 478)
    repeat_call_result_102044 = invoke(stypy.reporting.localization.Localization(__file__, 478, 27), repeat_102040, *[pad_amt_102041, axis_102042], **kwargs_102043)
    
    # Obtaining the member 'astype' of a type (line 478)
    astype_102045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 27), repeat_call_result_102044, 'astype')
    # Calling astype(args, kwargs) (line 478)
    astype_call_result_102049 = invoke(stypy.reporting.localization.Localization(__file__, 478, 27), astype_102045, *[dtype_102047], **kwargs_102048)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 27), tuple_102038, astype_call_result_102049)
    # Adding element type (line 478)
    # Getting the type of 'arr' (line 479)
    arr_102050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 27), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 27), tuple_102038, arr_102050)
    
    # Processing the call keyword arguments (line 478)
    # Getting the type of 'axis' (line 479)
    axis_102051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 38), 'axis', False)
    keyword_102052 = axis_102051
    kwargs_102053 = {'axis': keyword_102052}
    # Getting the type of 'np' (line 478)
    np_102036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 478)
    concatenate_102037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 11), np_102036, 'concatenate')
    # Calling concatenate(args, kwargs) (line 478)
    concatenate_call_result_102054 = invoke(stypy.reporting.localization.Localization(__file__, 478, 11), concatenate_102037, *[tuple_102038], **kwargs_102053)
    
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type', concatenate_call_result_102054)
    
    # ################# End of '_prepend_mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_mean' in the type store
    # Getting the type of 'stypy_return_type' (line 430)
    stypy_return_type_102055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102055)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_mean'
    return stypy_return_type_102055

# Assigning a type to the variable '_prepend_mean' (line 430)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), '_prepend_mean', _prepend_mean)

@norecursion
def _append_mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 41), 'int')
    defaults = [int_102056]
    # Create a new context for function '_append_mean'
    module_type_store = module_type_store.open_function_context('_append_mean', 482, 0, False)
    
    # Passed parameters checking function
    _append_mean.stypy_localization = localization
    _append_mean.stypy_type_of_self = None
    _append_mean.stypy_type_store = module_type_store
    _append_mean.stypy_function_name = '_append_mean'
    _append_mean.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _append_mean.stypy_varargs_param_name = None
    _append_mean.stypy_kwargs_param_name = None
    _append_mean.stypy_call_defaults = defaults
    _append_mean.stypy_call_varargs = varargs
    _append_mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_mean', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_mean', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_mean(...)' code ##################

    str_102057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, (-1)), 'str', '\n    Append `pad_amt` mean values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate mean.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the maximum of the final `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 505)
    pad_amt_102058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'pad_amt')
    int_102059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 18), 'int')
    # Applying the binary operator '==' (line 505)
    result_eq_102060 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), '==', pad_amt_102058, int_102059)
    
    # Testing the type of an if condition (line 505)
    if_condition_102061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_eq_102060)
    # Assigning a type to the variable 'if_condition_102061' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_102061', if_condition_102061)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 506)
    arr_102062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'stypy_return_type', arr_102062)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 509)
    num_102063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 7), 'num')
    int_102064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 14), 'int')
    # Applying the binary operator '==' (line 509)
    result_eq_102065 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 7), '==', num_102063, int_102064)
    
    # Testing the type of an if condition (line 509)
    if_condition_102066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 4), result_eq_102065)
    # Assigning a type to the variable 'if_condition_102066' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'if_condition_102066', if_condition_102066)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _append_edge(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'arr' (line 510)
    arr_102068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 28), 'arr', False)
    # Getting the type of 'pad_amt' (line 510)
    pad_amt_102069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 33), 'pad_amt', False)
    # Getting the type of 'axis' (line 510)
    axis_102070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 42), 'axis', False)
    # Processing the call keyword arguments (line 510)
    kwargs_102071 = {}
    # Getting the type of '_append_edge' (line 510)
    _append_edge_102067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 15), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 510)
    _append_edge_call_result_102072 = invoke(stypy.reporting.localization.Localization(__file__, 510, 15), _append_edge_102067, *[arr_102068, pad_amt_102069, axis_102070], **kwargs_102071)
    
    # Assigning a type to the variable 'stypy_return_type' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'stypy_return_type', _append_edge_call_result_102072)
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 513)
    # Getting the type of 'num' (line 513)
    num_102073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'num')
    # Getting the type of 'None' (line 513)
    None_102074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'None')
    
    (may_be_102075, more_types_in_union_102076) = may_not_be_none(num_102073, None_102074)

    if may_be_102075:

        if more_types_in_union_102076:
            # Runtime conditional SSA (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 514)
        num_102077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 514)
        axis_102078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 28), 'axis')
        # Getting the type of 'arr' (line 514)
        arr_102079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 514)
        shape_102080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 18), arr_102079, 'shape')
        # Obtaining the member '__getitem__' of a type (line 514)
        getitem___102081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 18), shape_102080, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 514)
        subscript_call_result_102082 = invoke(stypy.reporting.localization.Localization(__file__, 514, 18), getitem___102081, axis_102078)
        
        # Applying the binary operator '>=' (line 514)
        result_ge_102083 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 11), '>=', num_102077, subscript_call_result_102082)
        
        # Testing the type of an if condition (line 514)
        if_condition_102084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 8), result_ge_102083)
        # Assigning a type to the variable 'if_condition_102084' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'if_condition_102084', if_condition_102084)
        # SSA begins for if statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 515):
        # Getting the type of 'None' (line 515)
        None_102085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 18), 'None')
        # Assigning a type to the variable 'num' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'num', None_102085)
        # SSA join for if statement (line 514)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_102076:
            # SSA join for if statement (line 513)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 518):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 518)
    axis_102086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'axis')
    # Getting the type of 'arr' (line 518)
    arr_102087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 518)
    shape_102088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 10), arr_102087, 'shape')
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___102089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 10), shape_102088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_102090 = invoke(stypy.reporting.localization.Localization(__file__, 518, 10), getitem___102089, axis_102086)
    
    int_102091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 28), 'int')
    # Applying the binary operator '-' (line 518)
    result_sub_102092 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 10), '-', subscript_call_result_102090, int_102091)
    
    # Assigning a type to the variable 'end' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'end', result_sub_102092)
    
    # Type idiom detected: calculating its left and rigth part (line 519)
    # Getting the type of 'num' (line 519)
    num_102093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'num')
    # Getting the type of 'None' (line 519)
    None_102094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 18), 'None')
    
    (may_be_102095, more_types_in_union_102096) = may_not_be_none(num_102093, None_102094)

    if may_be_102095:

        if more_types_in_union_102096:
            # Runtime conditional SSA (line 519)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 520):
        
        # Call to tuple(...): (line 520)
        # Processing the call arguments (line 520)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 521, 12, True)
        # Calculating comprehension expression
        
        # Call to enumerate(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'arr' (line 522)
        arr_102115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 36), 'arr', False)
        # Obtaining the member 'shape' of a type (line 522)
        shape_102116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 36), arr_102115, 'shape')
        # Processing the call keyword arguments (line 522)
        kwargs_102117 = {}
        # Getting the type of 'enumerate' (line 522)
        enumerate_102114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 522)
        enumerate_call_result_102118 = invoke(stypy.reporting.localization.Localization(__file__, 522, 26), enumerate_102114, *[shape_102116], **kwargs_102117)
        
        comprehension_102119 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 12), enumerate_call_result_102118)
        # Assigning a type to the variable 'i' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 12), comprehension_102119))
        # Assigning a type to the variable 'x' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 12), comprehension_102119))
        
        
        # Getting the type of 'i' (line 521)
        i_102098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'i', False)
        # Getting the type of 'axis' (line 521)
        axis_102099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'axis', False)
        # Applying the binary operator '!=' (line 521)
        result_ne_102100 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 27), '!=', i_102098, axis_102099)
        
        # Testing the type of an if expression (line 521)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 12), result_ne_102100)
        # SSA begins for if expression (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'None' (line 521)
        None_102102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 18), 'None', False)
        # Processing the call keyword arguments (line 521)
        kwargs_102103 = {}
        # Getting the type of 'slice' (line 521)
        slice_102101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'slice', False)
        # Calling slice(args, kwargs) (line 521)
        slice_call_result_102104 = invoke(stypy.reporting.localization.Localization(__file__, 521, 12), slice_102101, *[None_102102], **kwargs_102103)
        
        # SSA branch for the else part of an if expression (line 521)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to slice(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'end' (line 521)
        end_102106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 48), 'end', False)
        # Getting the type of 'end' (line 521)
        end_102107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 53), 'end', False)
        # Getting the type of 'num' (line 521)
        num_102108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 59), 'num', False)
        # Applying the binary operator '-' (line 521)
        result_sub_102109 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 53), '-', end_102107, num_102108)
        
        int_102110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 64), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_102111 = {}
        # Getting the type of 'slice' (line 521)
        slice_102105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 42), 'slice', False)
        # Calling slice(args, kwargs) (line 521)
        slice_call_result_102112 = invoke(stypy.reporting.localization.Localization(__file__, 521, 42), slice_102105, *[end_102106, result_sub_102109, int_102110], **kwargs_102111)
        
        # SSA join for if expression (line 521)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_102113 = union_type.UnionType.add(slice_call_result_102104, slice_call_result_102112)
        
        list_102120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 12), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 12), list_102120, if_exp_102113)
        # Processing the call keyword arguments (line 520)
        kwargs_102121 = {}
        # Getting the type of 'tuple' (line 520)
        tuple_102097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 520)
        tuple_call_result_102122 = invoke(stypy.reporting.localization.Localization(__file__, 520, 21), tuple_102097, *[list_102120], **kwargs_102121)
        
        # Assigning a type to the variable 'mean_slice' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'mean_slice', tuple_call_result_102122)

        if more_types_in_union_102096:
            # Runtime conditional SSA for else branch (line 519)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_102095) or more_types_in_union_102096):
        
        # Assigning a Call to a Name (line 524):
        
        # Call to tuple(...): (line 524)
        # Processing the call arguments (line 524)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 524, 27, True)
        # Calculating comprehension expression
        # Getting the type of 'arr' (line 524)
        arr_102128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 48), 'arr', False)
        # Obtaining the member 'shape' of a type (line 524)
        shape_102129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 48), arr_102128, 'shape')
        comprehension_102130 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 27), shape_102129)
        # Assigning a type to the variable 'x' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'x', comprehension_102130)
        
        # Call to slice(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'None' (line 524)
        None_102125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 33), 'None', False)
        # Processing the call keyword arguments (line 524)
        kwargs_102126 = {}
        # Getting the type of 'slice' (line 524)
        slice_102124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 27), 'slice', False)
        # Calling slice(args, kwargs) (line 524)
        slice_call_result_102127 = invoke(stypy.reporting.localization.Localization(__file__, 524, 27), slice_102124, *[None_102125], **kwargs_102126)
        
        list_102131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 27), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 524, 27), list_102131, slice_call_result_102127)
        # Processing the call keyword arguments (line 524)
        kwargs_102132 = {}
        # Getting the type of 'tuple' (line 524)
        tuple_102123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 524)
        tuple_call_result_102133 = invoke(stypy.reporting.localization.Localization(__file__, 524, 21), tuple_102123, *[list_102131], **kwargs_102132)
        
        # Assigning a type to the variable 'mean_slice' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'mean_slice', tuple_call_result_102133)

        if (may_be_102095 and more_types_in_union_102096):
            # SSA join for if statement (line 519)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 527):
    
    # Call to tuple(...): (line 527)
    # Processing the call arguments (line 527)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 527, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 528)
    # Processing the call arguments (line 528)
    # Getting the type of 'arr' (line 528)
    arr_102142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 528)
    shape_102143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 50), arr_102142, 'shape')
    # Processing the call keyword arguments (line 528)
    kwargs_102144 = {}
    # Getting the type of 'enumerate' (line 528)
    enumerate_102141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 528)
    enumerate_call_result_102145 = invoke(stypy.reporting.localization.Localization(__file__, 528, 40), enumerate_102141, *[shape_102143], **kwargs_102144)
    
    comprehension_102146 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 26), enumerate_call_result_102145)
    # Assigning a type to the variable 'i' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 26), comprehension_102146))
    # Assigning a type to the variable 'x' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 26), comprehension_102146))
    
    
    # Getting the type of 'i' (line 527)
    i_102135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 31), 'i', False)
    # Getting the type of 'axis' (line 527)
    axis_102136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 36), 'axis', False)
    # Applying the binary operator '!=' (line 527)
    result_ne_102137 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 31), '!=', i_102135, axis_102136)
    
    # Testing the type of an if expression (line 527)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 26), result_ne_102137)
    # SSA begins for if expression (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 527)
    x_102138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 527)
    module_type_store.open_ssa_branch('if expression else')
    int_102139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 46), 'int')
    # SSA join for if expression (line 527)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102140 = union_type.UnionType.add(x_102138, int_102139)
    
    list_102147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 26), list_102147, if_exp_102140)
    # Processing the call keyword arguments (line 527)
    kwargs_102148 = {}
    # Getting the type of 'tuple' (line 527)
    tuple_102134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 527)
    tuple_call_result_102149 = invoke(stypy.reporting.localization.Localization(__file__, 527, 20), tuple_102134, *[list_102147], **kwargs_102148)
    
    # Assigning a type to the variable 'pad_singleton' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'pad_singleton', tuple_call_result_102149)
    
    # Assigning a Call to a Name (line 531):
    
    # Call to reshape(...): (line 531)
    # Processing the call arguments (line 531)
    # Getting the type of 'pad_singleton' (line 531)
    pad_singleton_102160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 57), 'pad_singleton', False)
    # Processing the call keyword arguments (line 531)
    kwargs_102161 = {}
    
    # Call to mean(...): (line 531)
    # Processing the call keyword arguments (line 531)
    # Getting the type of 'axis' (line 531)
    axis_102155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 43), 'axis', False)
    keyword_102156 = axis_102155
    kwargs_102157 = {'axis': keyword_102156}
    
    # Obtaining the type of the subscript
    # Getting the type of 'mean_slice' (line 531)
    mean_slice_102150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 21), 'mean_slice', False)
    # Getting the type of 'arr' (line 531)
    arr_102151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 17), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 531)
    getitem___102152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 17), arr_102151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 531)
    subscript_call_result_102153 = invoke(stypy.reporting.localization.Localization(__file__, 531, 17), getitem___102152, mean_slice_102150)
    
    # Obtaining the member 'mean' of a type (line 531)
    mean_102154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 17), subscript_call_result_102153, 'mean')
    # Calling mean(args, kwargs) (line 531)
    mean_call_result_102158 = invoke(stypy.reporting.localization.Localization(__file__, 531, 17), mean_102154, *[], **kwargs_102157)
    
    # Obtaining the member 'reshape' of a type (line 531)
    reshape_102159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 17), mean_call_result_102158, 'reshape')
    # Calling reshape(args, kwargs) (line 531)
    reshape_call_result_102162 = invoke(stypy.reporting.localization.Localization(__file__, 531, 17), reshape_102159, *[pad_singleton_102160], **kwargs_102161)
    
    # Assigning a type to the variable 'mean_chunk' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'mean_chunk', reshape_call_result_102162)
    
    # Call to _round_ifneeded(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'mean_chunk' (line 532)
    mean_chunk_102164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'mean_chunk', False)
    # Getting the type of 'arr' (line 532)
    arr_102165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 32), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 532)
    dtype_102166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 32), arr_102165, 'dtype')
    # Processing the call keyword arguments (line 532)
    kwargs_102167 = {}
    # Getting the type of '_round_ifneeded' (line 532)
    _round_ifneeded_102163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 532)
    _round_ifneeded_call_result_102168 = invoke(stypy.reporting.localization.Localization(__file__, 532, 4), _round_ifneeded_102163, *[mean_chunk_102164, dtype_102166], **kwargs_102167)
    
    
    # Call to concatenate(...): (line 535)
    # Processing the call arguments (line 535)
    
    # Obtaining an instance of the builtin type 'tuple' (line 536)
    tuple_102171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 536)
    # Adding element type (line 536)
    # Getting the type of 'arr' (line 536)
    arr_102172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 9), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 9), tuple_102171, arr_102172)
    # Adding element type (line 536)
    
    # Call to astype(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'arr' (line 536)
    arr_102180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 54), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 536)
    dtype_102181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 54), arr_102180, 'dtype')
    # Processing the call keyword arguments (line 536)
    kwargs_102182 = {}
    
    # Call to repeat(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'pad_amt' (line 536)
    pad_amt_102175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 32), 'pad_amt', False)
    # Getting the type of 'axis' (line 536)
    axis_102176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 41), 'axis', False)
    # Processing the call keyword arguments (line 536)
    kwargs_102177 = {}
    # Getting the type of 'mean_chunk' (line 536)
    mean_chunk_102173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 14), 'mean_chunk', False)
    # Obtaining the member 'repeat' of a type (line 536)
    repeat_102174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 14), mean_chunk_102173, 'repeat')
    # Calling repeat(args, kwargs) (line 536)
    repeat_call_result_102178 = invoke(stypy.reporting.localization.Localization(__file__, 536, 14), repeat_102174, *[pad_amt_102175, axis_102176], **kwargs_102177)
    
    # Obtaining the member 'astype' of a type (line 536)
    astype_102179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 14), repeat_call_result_102178, 'astype')
    # Calling astype(args, kwargs) (line 536)
    astype_call_result_102183 = invoke(stypy.reporting.localization.Localization(__file__, 536, 14), astype_102179, *[dtype_102181], **kwargs_102182)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 9), tuple_102171, astype_call_result_102183)
    
    # Processing the call keyword arguments (line 535)
    # Getting the type of 'axis' (line 536)
    axis_102184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 72), 'axis', False)
    keyword_102185 = axis_102184
    kwargs_102186 = {'axis': keyword_102185}
    # Getting the type of 'np' (line 535)
    np_102169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 535)
    concatenate_102170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 11), np_102169, 'concatenate')
    # Calling concatenate(args, kwargs) (line 535)
    concatenate_call_result_102187 = invoke(stypy.reporting.localization.Localization(__file__, 535, 11), concatenate_102170, *[tuple_102171], **kwargs_102186)
    
    # Assigning a type to the variable 'stypy_return_type' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'stypy_return_type', concatenate_call_result_102187)
    
    # ################# End of '_append_mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_mean' in the type store
    # Getting the type of 'stypy_return_type' (line 482)
    stypy_return_type_102188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102188)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_mean'
    return stypy_return_type_102188

# Assigning a type to the variable '_append_mean' (line 482)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), '_append_mean', _append_mean)

@norecursion
def _prepend_med(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 41), 'int')
    defaults = [int_102189]
    # Create a new context for function '_prepend_med'
    module_type_store = module_type_store.open_function_context('_prepend_med', 539, 0, False)
    
    # Passed parameters checking function
    _prepend_med.stypy_localization = localization
    _prepend_med.stypy_type_of_self = None
    _prepend_med.stypy_type_store = module_type_store
    _prepend_med.stypy_function_name = '_prepend_med'
    _prepend_med.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _prepend_med.stypy_varargs_param_name = None
    _prepend_med.stypy_kwargs_param_name = None
    _prepend_med.stypy_call_defaults = defaults
    _prepend_med.stypy_call_varargs = varargs
    _prepend_med.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_med', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_med', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_med(...)' code ##################

    str_102190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, (-1)), 'str', '\n    Prepend `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate median.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the median of the first `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 562)
    pad_amt_102191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'pad_amt')
    int_102192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 18), 'int')
    # Applying the binary operator '==' (line 562)
    result_eq_102193 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 7), '==', pad_amt_102191, int_102192)
    
    # Testing the type of an if condition (line 562)
    if_condition_102194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 4), result_eq_102193)
    # Assigning a type to the variable 'if_condition_102194' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'if_condition_102194', if_condition_102194)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 563)
    arr_102195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'stypy_return_type', arr_102195)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 566)
    num_102196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 7), 'num')
    int_102197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 14), 'int')
    # Applying the binary operator '==' (line 566)
    result_eq_102198 = python_operator(stypy.reporting.localization.Localization(__file__, 566, 7), '==', num_102196, int_102197)
    
    # Testing the type of an if condition (line 566)
    if_condition_102199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 566, 4), result_eq_102198)
    # Assigning a type to the variable 'if_condition_102199' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'if_condition_102199', if_condition_102199)
    # SSA begins for if statement (line 566)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _prepend_edge(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'arr' (line 567)
    arr_102201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 29), 'arr', False)
    # Getting the type of 'pad_amt' (line 567)
    pad_amt_102202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 34), 'pad_amt', False)
    # Getting the type of 'axis' (line 567)
    axis_102203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 43), 'axis', False)
    # Processing the call keyword arguments (line 567)
    kwargs_102204 = {}
    # Getting the type of '_prepend_edge' (line 567)
    _prepend_edge_102200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 15), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 567)
    _prepend_edge_call_result_102205 = invoke(stypy.reporting.localization.Localization(__file__, 567, 15), _prepend_edge_102200, *[arr_102201, pad_amt_102202, axis_102203], **kwargs_102204)
    
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'stypy_return_type', _prepend_edge_call_result_102205)
    # SSA join for if statement (line 566)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 570)
    # Getting the type of 'num' (line 570)
    num_102206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'num')
    # Getting the type of 'None' (line 570)
    None_102207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 18), 'None')
    
    (may_be_102208, more_types_in_union_102209) = may_not_be_none(num_102206, None_102207)

    if may_be_102208:

        if more_types_in_union_102209:
            # Runtime conditional SSA (line 570)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 571)
        num_102210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 571)
        axis_102211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 28), 'axis')
        # Getting the type of 'arr' (line 571)
        arr_102212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 571)
        shape_102213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), arr_102212, 'shape')
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___102214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), shape_102213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_102215 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), getitem___102214, axis_102211)
        
        # Applying the binary operator '>=' (line 571)
        result_ge_102216 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 11), '>=', num_102210, subscript_call_result_102215)
        
        # Testing the type of an if condition (line 571)
        if_condition_102217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 8), result_ge_102216)
        # Assigning a type to the variable 'if_condition_102217' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'if_condition_102217', if_condition_102217)
        # SSA begins for if statement (line 571)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 572):
        # Getting the type of 'None' (line 572)
        None_102218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 18), 'None')
        # Assigning a type to the variable 'num' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'num', None_102218)
        # SSA join for if statement (line 571)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_102209:
            # SSA join for if statement (line 570)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 575):
    
    # Call to tuple(...): (line 575)
    # Processing the call arguments (line 575)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 575, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'arr' (line 576)
    arr_102233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 576)
    shape_102234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 46), arr_102233, 'shape')
    # Processing the call keyword arguments (line 576)
    kwargs_102235 = {}
    # Getting the type of 'enumerate' (line 576)
    enumerate_102232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 576)
    enumerate_call_result_102236 = invoke(stypy.reporting.localization.Localization(__file__, 576, 36), enumerate_102232, *[shape_102234], **kwargs_102235)
    
    comprehension_102237 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 22), enumerate_call_result_102236)
    # Assigning a type to the variable 'i' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 22), comprehension_102237))
    # Assigning a type to the variable 'x' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 22), comprehension_102237))
    
    
    # Getting the type of 'i' (line 575)
    i_102220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 37), 'i', False)
    # Getting the type of 'axis' (line 575)
    axis_102221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 42), 'axis', False)
    # Applying the binary operator '!=' (line 575)
    result_ne_102222 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 37), '!=', i_102220, axis_102221)
    
    # Testing the type of an if expression (line 575)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 22), result_ne_102222)
    # SSA begins for if expression (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'None' (line 575)
    None_102224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 28), 'None', False)
    # Processing the call keyword arguments (line 575)
    kwargs_102225 = {}
    # Getting the type of 'slice' (line 575)
    slice_102223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 575)
    slice_call_result_102226 = invoke(stypy.reporting.localization.Localization(__file__, 575, 22), slice_102223, *[None_102224], **kwargs_102225)
    
    # SSA branch for the else part of an if expression (line 575)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'num' (line 575)
    num_102228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 58), 'num', False)
    # Processing the call keyword arguments (line 575)
    kwargs_102229 = {}
    # Getting the type of 'slice' (line 575)
    slice_102227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 575)
    slice_call_result_102230 = invoke(stypy.reporting.localization.Localization(__file__, 575, 52), slice_102227, *[num_102228], **kwargs_102229)
    
    # SSA join for if expression (line 575)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102231 = union_type.UnionType.add(slice_call_result_102226, slice_call_result_102230)
    
    list_102238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 22), list_102238, if_exp_102231)
    # Processing the call keyword arguments (line 575)
    kwargs_102239 = {}
    # Getting the type of 'tuple' (line 575)
    tuple_102219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 575)
    tuple_call_result_102240 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), tuple_102219, *[list_102238], **kwargs_102239)
    
    # Assigning a type to the variable 'med_slice' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'med_slice', tuple_call_result_102240)
    
    # Assigning a Call to a Name (line 579):
    
    # Call to tuple(...): (line 579)
    # Processing the call arguments (line 579)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 579, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'arr' (line 580)
    arr_102249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 580)
    shape_102250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 50), arr_102249, 'shape')
    # Processing the call keyword arguments (line 580)
    kwargs_102251 = {}
    # Getting the type of 'enumerate' (line 580)
    enumerate_102248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 580)
    enumerate_call_result_102252 = invoke(stypy.reporting.localization.Localization(__file__, 580, 40), enumerate_102248, *[shape_102250], **kwargs_102251)
    
    comprehension_102253 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 26), enumerate_call_result_102252)
    # Assigning a type to the variable 'i' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 26), comprehension_102253))
    # Assigning a type to the variable 'x' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 26), comprehension_102253))
    
    
    # Getting the type of 'i' (line 579)
    i_102242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'i', False)
    # Getting the type of 'axis' (line 579)
    axis_102243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 36), 'axis', False)
    # Applying the binary operator '!=' (line 579)
    result_ne_102244 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 31), '!=', i_102242, axis_102243)
    
    # Testing the type of an if expression (line 579)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 26), result_ne_102244)
    # SSA begins for if expression (line 579)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 579)
    x_102245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 579)
    module_type_store.open_ssa_branch('if expression else')
    int_102246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 46), 'int')
    # SSA join for if expression (line 579)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102247 = union_type.UnionType.add(x_102245, int_102246)
    
    list_102254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 26), list_102254, if_exp_102247)
    # Processing the call keyword arguments (line 579)
    kwargs_102255 = {}
    # Getting the type of 'tuple' (line 579)
    tuple_102241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 579)
    tuple_call_result_102256 = invoke(stypy.reporting.localization.Localization(__file__, 579, 20), tuple_102241, *[list_102254], **kwargs_102255)
    
    # Assigning a type to the variable 'pad_singleton' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'pad_singleton', tuple_call_result_102256)
    
    # Assigning a Call to a Name (line 583):
    
    # Call to reshape(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'pad_singleton' (line 583)
    pad_singleton_102268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 61), 'pad_singleton', False)
    # Processing the call keyword arguments (line 583)
    kwargs_102269 = {}
    
    # Call to median(...): (line 583)
    # Processing the call arguments (line 583)
    
    # Obtaining the type of the subscript
    # Getting the type of 'med_slice' (line 583)
    med_slice_102259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 30), 'med_slice', False)
    # Getting the type of 'arr' (line 583)
    arr_102260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 26), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___102261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 26), arr_102260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 583)
    subscript_call_result_102262 = invoke(stypy.reporting.localization.Localization(__file__, 583, 26), getitem___102261, med_slice_102259)
    
    # Processing the call keyword arguments (line 583)
    # Getting the type of 'axis' (line 583)
    axis_102263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 47), 'axis', False)
    keyword_102264 = axis_102263
    kwargs_102265 = {'axis': keyword_102264}
    # Getting the type of 'np' (line 583)
    np_102257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 16), 'np', False)
    # Obtaining the member 'median' of a type (line 583)
    median_102258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), np_102257, 'median')
    # Calling median(args, kwargs) (line 583)
    median_call_result_102266 = invoke(stypy.reporting.localization.Localization(__file__, 583, 16), median_102258, *[subscript_call_result_102262], **kwargs_102265)
    
    # Obtaining the member 'reshape' of a type (line 583)
    reshape_102267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 16), median_call_result_102266, 'reshape')
    # Calling reshape(args, kwargs) (line 583)
    reshape_call_result_102270 = invoke(stypy.reporting.localization.Localization(__file__, 583, 16), reshape_102267, *[pad_singleton_102268], **kwargs_102269)
    
    # Assigning a type to the variable 'med_chunk' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'med_chunk', reshape_call_result_102270)
    
    # Call to _round_ifneeded(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'med_chunk' (line 584)
    med_chunk_102272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'med_chunk', False)
    # Getting the type of 'arr' (line 584)
    arr_102273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 584)
    dtype_102274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 31), arr_102273, 'dtype')
    # Processing the call keyword arguments (line 584)
    kwargs_102275 = {}
    # Getting the type of '_round_ifneeded' (line 584)
    _round_ifneeded_102271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 584)
    _round_ifneeded_call_result_102276 = invoke(stypy.reporting.localization.Localization(__file__, 584, 4), _round_ifneeded_102271, *[med_chunk_102272, dtype_102274], **kwargs_102275)
    
    
    # Call to concatenate(...): (line 587)
    # Processing the call arguments (line 587)
    
    # Obtaining an instance of the builtin type 'tuple' (line 588)
    tuple_102279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 588)
    # Adding element type (line 588)
    
    # Call to astype(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'arr' (line 588)
    arr_102287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 48), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 588)
    dtype_102288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 48), arr_102287, 'dtype')
    # Processing the call keyword arguments (line 588)
    kwargs_102289 = {}
    
    # Call to repeat(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'pad_amt' (line 588)
    pad_amt_102282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 26), 'pad_amt', False)
    # Getting the type of 'axis' (line 588)
    axis_102283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 35), 'axis', False)
    # Processing the call keyword arguments (line 588)
    kwargs_102284 = {}
    # Getting the type of 'med_chunk' (line 588)
    med_chunk_102280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 9), 'med_chunk', False)
    # Obtaining the member 'repeat' of a type (line 588)
    repeat_102281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 9), med_chunk_102280, 'repeat')
    # Calling repeat(args, kwargs) (line 588)
    repeat_call_result_102285 = invoke(stypy.reporting.localization.Localization(__file__, 588, 9), repeat_102281, *[pad_amt_102282, axis_102283], **kwargs_102284)
    
    # Obtaining the member 'astype' of a type (line 588)
    astype_102286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 9), repeat_call_result_102285, 'astype')
    # Calling astype(args, kwargs) (line 588)
    astype_call_result_102290 = invoke(stypy.reporting.localization.Localization(__file__, 588, 9), astype_102286, *[dtype_102288], **kwargs_102289)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 9), tuple_102279, astype_call_result_102290)
    # Adding element type (line 588)
    # Getting the type of 'arr' (line 588)
    arr_102291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 60), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 588, 9), tuple_102279, arr_102291)
    
    # Processing the call keyword arguments (line 587)
    # Getting the type of 'axis' (line 588)
    axis_102292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 71), 'axis', False)
    keyword_102293 = axis_102292
    kwargs_102294 = {'axis': keyword_102293}
    # Getting the type of 'np' (line 587)
    np_102277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 587)
    concatenate_102278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 11), np_102277, 'concatenate')
    # Calling concatenate(args, kwargs) (line 587)
    concatenate_call_result_102295 = invoke(stypy.reporting.localization.Localization(__file__, 587, 11), concatenate_102278, *[tuple_102279], **kwargs_102294)
    
    # Assigning a type to the variable 'stypy_return_type' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'stypy_return_type', concatenate_call_result_102295)
    
    # ################# End of '_prepend_med(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_med' in the type store
    # Getting the type of 'stypy_return_type' (line 539)
    stypy_return_type_102296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102296)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_med'
    return stypy_return_type_102296

# Assigning a type to the variable '_prepend_med' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), '_prepend_med', _prepend_med)

@norecursion
def _append_med(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 40), 'int')
    defaults = [int_102297]
    # Create a new context for function '_append_med'
    module_type_store = module_type_store.open_function_context('_append_med', 591, 0, False)
    
    # Passed parameters checking function
    _append_med.stypy_localization = localization
    _append_med.stypy_type_of_self = None
    _append_med.stypy_type_store = module_type_store
    _append_med.stypy_function_name = '_append_med'
    _append_med.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _append_med.stypy_varargs_param_name = None
    _append_med.stypy_kwargs_param_name = None
    _append_med.stypy_call_defaults = defaults
    _append_med.stypy_call_varargs = varargs
    _append_med.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_med', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_med', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_med(...)' code ##################

    str_102298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, (-1)), 'str', '\n    Append `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate median.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the median of the final `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 614)
    pad_amt_102299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 7), 'pad_amt')
    int_102300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 18), 'int')
    # Applying the binary operator '==' (line 614)
    result_eq_102301 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 7), '==', pad_amt_102299, int_102300)
    
    # Testing the type of an if condition (line 614)
    if_condition_102302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 4), result_eq_102301)
    # Assigning a type to the variable 'if_condition_102302' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'if_condition_102302', if_condition_102302)
    # SSA begins for if statement (line 614)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 615)
    arr_102303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'stypy_return_type', arr_102303)
    # SSA join for if statement (line 614)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 618)
    num_102304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 7), 'num')
    int_102305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 14), 'int')
    # Applying the binary operator '==' (line 618)
    result_eq_102306 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 7), '==', num_102304, int_102305)
    
    # Testing the type of an if condition (line 618)
    if_condition_102307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 618, 4), result_eq_102306)
    # Assigning a type to the variable 'if_condition_102307' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'if_condition_102307', if_condition_102307)
    # SSA begins for if statement (line 618)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _append_edge(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'arr' (line 619)
    arr_102309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'arr', False)
    # Getting the type of 'pad_amt' (line 619)
    pad_amt_102310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 33), 'pad_amt', False)
    # Getting the type of 'axis' (line 619)
    axis_102311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 42), 'axis', False)
    # Processing the call keyword arguments (line 619)
    kwargs_102312 = {}
    # Getting the type of '_append_edge' (line 619)
    _append_edge_102308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 15), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 619)
    _append_edge_call_result_102313 = invoke(stypy.reporting.localization.Localization(__file__, 619, 15), _append_edge_102308, *[arr_102309, pad_amt_102310, axis_102311], **kwargs_102312)
    
    # Assigning a type to the variable 'stypy_return_type' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'stypy_return_type', _append_edge_call_result_102313)
    # SSA join for if statement (line 618)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 622)
    # Getting the type of 'num' (line 622)
    num_102314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'num')
    # Getting the type of 'None' (line 622)
    None_102315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 18), 'None')
    
    (may_be_102316, more_types_in_union_102317) = may_not_be_none(num_102314, None_102315)

    if may_be_102316:

        if more_types_in_union_102317:
            # Runtime conditional SSA (line 622)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 623)
        num_102318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 623)
        axis_102319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'axis')
        # Getting the type of 'arr' (line 623)
        arr_102320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 623)
        shape_102321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 18), arr_102320, 'shape')
        # Obtaining the member '__getitem__' of a type (line 623)
        getitem___102322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 18), shape_102321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 623)
        subscript_call_result_102323 = invoke(stypy.reporting.localization.Localization(__file__, 623, 18), getitem___102322, axis_102319)
        
        # Applying the binary operator '>=' (line 623)
        result_ge_102324 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 11), '>=', num_102318, subscript_call_result_102323)
        
        # Testing the type of an if condition (line 623)
        if_condition_102325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 8), result_ge_102324)
        # Assigning a type to the variable 'if_condition_102325' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'if_condition_102325', if_condition_102325)
        # SSA begins for if statement (line 623)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 624):
        # Getting the type of 'None' (line 624)
        None_102326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 18), 'None')
        # Assigning a type to the variable 'num' (line 624)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'num', None_102326)
        # SSA join for if statement (line 623)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_102317:
            # SSA join for if statement (line 622)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 627):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 627)
    axis_102327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'axis')
    # Getting the type of 'arr' (line 627)
    arr_102328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 627)
    shape_102329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 10), arr_102328, 'shape')
    # Obtaining the member '__getitem__' of a type (line 627)
    getitem___102330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 10), shape_102329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 627)
    subscript_call_result_102331 = invoke(stypy.reporting.localization.Localization(__file__, 627, 10), getitem___102330, axis_102327)
    
    int_102332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 28), 'int')
    # Applying the binary operator '-' (line 627)
    result_sub_102333 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 10), '-', subscript_call_result_102331, int_102332)
    
    # Assigning a type to the variable 'end' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'end', result_sub_102333)
    
    # Type idiom detected: calculating its left and rigth part (line 628)
    # Getting the type of 'num' (line 628)
    num_102334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'num')
    # Getting the type of 'None' (line 628)
    None_102335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 18), 'None')
    
    (may_be_102336, more_types_in_union_102337) = may_not_be_none(num_102334, None_102335)

    if may_be_102336:

        if more_types_in_union_102337:
            # Runtime conditional SSA (line 628)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 629):
        
        # Call to tuple(...): (line 629)
        # Processing the call arguments (line 629)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 630, 12, True)
        # Calculating comprehension expression
        
        # Call to enumerate(...): (line 631)
        # Processing the call arguments (line 631)
        # Getting the type of 'arr' (line 631)
        arr_102356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 36), 'arr', False)
        # Obtaining the member 'shape' of a type (line 631)
        shape_102357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 36), arr_102356, 'shape')
        # Processing the call keyword arguments (line 631)
        kwargs_102358 = {}
        # Getting the type of 'enumerate' (line 631)
        enumerate_102355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 631)
        enumerate_call_result_102359 = invoke(stypy.reporting.localization.Localization(__file__, 631, 26), enumerate_102355, *[shape_102357], **kwargs_102358)
        
        comprehension_102360 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 12), enumerate_call_result_102359)
        # Assigning a type to the variable 'i' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 12), comprehension_102360))
        # Assigning a type to the variable 'x' (line 630)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 12), comprehension_102360))
        
        
        # Getting the type of 'i' (line 630)
        i_102339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 27), 'i', False)
        # Getting the type of 'axis' (line 630)
        axis_102340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 32), 'axis', False)
        # Applying the binary operator '!=' (line 630)
        result_ne_102341 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 27), '!=', i_102339, axis_102340)
        
        # Testing the type of an if expression (line 630)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 12), result_ne_102341)
        # SSA begins for if expression (line 630)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'None' (line 630)
        None_102343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 18), 'None', False)
        # Processing the call keyword arguments (line 630)
        kwargs_102344 = {}
        # Getting the type of 'slice' (line 630)
        slice_102342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'slice', False)
        # Calling slice(args, kwargs) (line 630)
        slice_call_result_102345 = invoke(stypy.reporting.localization.Localization(__file__, 630, 12), slice_102342, *[None_102343], **kwargs_102344)
        
        # SSA branch for the else part of an if expression (line 630)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to slice(...): (line 630)
        # Processing the call arguments (line 630)
        # Getting the type of 'end' (line 630)
        end_102347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 48), 'end', False)
        # Getting the type of 'end' (line 630)
        end_102348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 53), 'end', False)
        # Getting the type of 'num' (line 630)
        num_102349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 59), 'num', False)
        # Applying the binary operator '-' (line 630)
        result_sub_102350 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 53), '-', end_102348, num_102349)
        
        int_102351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 64), 'int')
        # Processing the call keyword arguments (line 630)
        kwargs_102352 = {}
        # Getting the type of 'slice' (line 630)
        slice_102346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 42), 'slice', False)
        # Calling slice(args, kwargs) (line 630)
        slice_call_result_102353 = invoke(stypy.reporting.localization.Localization(__file__, 630, 42), slice_102346, *[end_102347, result_sub_102350, int_102351], **kwargs_102352)
        
        # SSA join for if expression (line 630)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_102354 = union_type.UnionType.add(slice_call_result_102345, slice_call_result_102353)
        
        list_102361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 12), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 12), list_102361, if_exp_102354)
        # Processing the call keyword arguments (line 629)
        kwargs_102362 = {}
        # Getting the type of 'tuple' (line 629)
        tuple_102338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 629)
        tuple_call_result_102363 = invoke(stypy.reporting.localization.Localization(__file__, 629, 20), tuple_102338, *[list_102361], **kwargs_102362)
        
        # Assigning a type to the variable 'med_slice' (line 629)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'med_slice', tuple_call_result_102363)

        if more_types_in_union_102337:
            # Runtime conditional SSA for else branch (line 628)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_102336) or more_types_in_union_102337):
        
        # Assigning a Call to a Name (line 633):
        
        # Call to tuple(...): (line 633)
        # Processing the call arguments (line 633)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 633, 26, True)
        # Calculating comprehension expression
        # Getting the type of 'arr' (line 633)
        arr_102369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 47), 'arr', False)
        # Obtaining the member 'shape' of a type (line 633)
        shape_102370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 47), arr_102369, 'shape')
        comprehension_102371 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 26), shape_102370)
        # Assigning a type to the variable 'x' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 26), 'x', comprehension_102371)
        
        # Call to slice(...): (line 633)
        # Processing the call arguments (line 633)
        # Getting the type of 'None' (line 633)
        None_102366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 32), 'None', False)
        # Processing the call keyword arguments (line 633)
        kwargs_102367 = {}
        # Getting the type of 'slice' (line 633)
        slice_102365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 26), 'slice', False)
        # Calling slice(args, kwargs) (line 633)
        slice_call_result_102368 = invoke(stypy.reporting.localization.Localization(__file__, 633, 26), slice_102365, *[None_102366], **kwargs_102367)
        
        list_102372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 633, 26), list_102372, slice_call_result_102368)
        # Processing the call keyword arguments (line 633)
        kwargs_102373 = {}
        # Getting the type of 'tuple' (line 633)
        tuple_102364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 633)
        tuple_call_result_102374 = invoke(stypy.reporting.localization.Localization(__file__, 633, 20), tuple_102364, *[list_102372], **kwargs_102373)
        
        # Assigning a type to the variable 'med_slice' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'med_slice', tuple_call_result_102374)

        if (may_be_102336 and more_types_in_union_102337):
            # SSA join for if statement (line 628)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 636):
    
    # Call to tuple(...): (line 636)
    # Processing the call arguments (line 636)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 636, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'arr' (line 637)
    arr_102383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 637)
    shape_102384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 50), arr_102383, 'shape')
    # Processing the call keyword arguments (line 637)
    kwargs_102385 = {}
    # Getting the type of 'enumerate' (line 637)
    enumerate_102382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 637)
    enumerate_call_result_102386 = invoke(stypy.reporting.localization.Localization(__file__, 637, 40), enumerate_102382, *[shape_102384], **kwargs_102385)
    
    comprehension_102387 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 26), enumerate_call_result_102386)
    # Assigning a type to the variable 'i' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 26), comprehension_102387))
    # Assigning a type to the variable 'x' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 26), comprehension_102387))
    
    
    # Getting the type of 'i' (line 636)
    i_102376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 31), 'i', False)
    # Getting the type of 'axis' (line 636)
    axis_102377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 36), 'axis', False)
    # Applying the binary operator '!=' (line 636)
    result_ne_102378 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 31), '!=', i_102376, axis_102377)
    
    # Testing the type of an if expression (line 636)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 636, 26), result_ne_102378)
    # SSA begins for if expression (line 636)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 636)
    x_102379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 636)
    module_type_store.open_ssa_branch('if expression else')
    int_102380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 46), 'int')
    # SSA join for if expression (line 636)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102381 = union_type.UnionType.add(x_102379, int_102380)
    
    list_102388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 26), list_102388, if_exp_102381)
    # Processing the call keyword arguments (line 636)
    kwargs_102389 = {}
    # Getting the type of 'tuple' (line 636)
    tuple_102375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 636)
    tuple_call_result_102390 = invoke(stypy.reporting.localization.Localization(__file__, 636, 20), tuple_102375, *[list_102388], **kwargs_102389)
    
    # Assigning a type to the variable 'pad_singleton' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'pad_singleton', tuple_call_result_102390)
    
    # Assigning a Call to a Name (line 640):
    
    # Call to reshape(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'pad_singleton' (line 640)
    pad_singleton_102402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 61), 'pad_singleton', False)
    # Processing the call keyword arguments (line 640)
    kwargs_102403 = {}
    
    # Call to median(...): (line 640)
    # Processing the call arguments (line 640)
    
    # Obtaining the type of the subscript
    # Getting the type of 'med_slice' (line 640)
    med_slice_102393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'med_slice', False)
    # Getting the type of 'arr' (line 640)
    arr_102394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 26), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 640)
    getitem___102395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 26), arr_102394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 640)
    subscript_call_result_102396 = invoke(stypy.reporting.localization.Localization(__file__, 640, 26), getitem___102395, med_slice_102393)
    
    # Processing the call keyword arguments (line 640)
    # Getting the type of 'axis' (line 640)
    axis_102397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 47), 'axis', False)
    keyword_102398 = axis_102397
    kwargs_102399 = {'axis': keyword_102398}
    # Getting the type of 'np' (line 640)
    np_102391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'np', False)
    # Obtaining the member 'median' of a type (line 640)
    median_102392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 16), np_102391, 'median')
    # Calling median(args, kwargs) (line 640)
    median_call_result_102400 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), median_102392, *[subscript_call_result_102396], **kwargs_102399)
    
    # Obtaining the member 'reshape' of a type (line 640)
    reshape_102401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 16), median_call_result_102400, 'reshape')
    # Calling reshape(args, kwargs) (line 640)
    reshape_call_result_102404 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), reshape_102401, *[pad_singleton_102402], **kwargs_102403)
    
    # Assigning a type to the variable 'med_chunk' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'med_chunk', reshape_call_result_102404)
    
    # Call to _round_ifneeded(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'med_chunk' (line 641)
    med_chunk_102406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 20), 'med_chunk', False)
    # Getting the type of 'arr' (line 641)
    arr_102407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 31), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 641)
    dtype_102408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 31), arr_102407, 'dtype')
    # Processing the call keyword arguments (line 641)
    kwargs_102409 = {}
    # Getting the type of '_round_ifneeded' (line 641)
    _round_ifneeded_102405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), '_round_ifneeded', False)
    # Calling _round_ifneeded(args, kwargs) (line 641)
    _round_ifneeded_call_result_102410 = invoke(stypy.reporting.localization.Localization(__file__, 641, 4), _round_ifneeded_102405, *[med_chunk_102406, dtype_102408], **kwargs_102409)
    
    
    # Call to concatenate(...): (line 644)
    # Processing the call arguments (line 644)
    
    # Obtaining an instance of the builtin type 'tuple' (line 645)
    tuple_102413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 645)
    # Adding element type (line 645)
    # Getting the type of 'arr' (line 645)
    arr_102414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 9), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 9), tuple_102413, arr_102414)
    # Adding element type (line 645)
    
    # Call to astype(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'arr' (line 645)
    arr_102422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 53), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 645)
    dtype_102423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 53), arr_102422, 'dtype')
    # Processing the call keyword arguments (line 645)
    kwargs_102424 = {}
    
    # Call to repeat(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'pad_amt' (line 645)
    pad_amt_102417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 31), 'pad_amt', False)
    # Getting the type of 'axis' (line 645)
    axis_102418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 40), 'axis', False)
    # Processing the call keyword arguments (line 645)
    kwargs_102419 = {}
    # Getting the type of 'med_chunk' (line 645)
    med_chunk_102415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 14), 'med_chunk', False)
    # Obtaining the member 'repeat' of a type (line 645)
    repeat_102416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 14), med_chunk_102415, 'repeat')
    # Calling repeat(args, kwargs) (line 645)
    repeat_call_result_102420 = invoke(stypy.reporting.localization.Localization(__file__, 645, 14), repeat_102416, *[pad_amt_102417, axis_102418], **kwargs_102419)
    
    # Obtaining the member 'astype' of a type (line 645)
    astype_102421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 645, 14), repeat_call_result_102420, 'astype')
    # Calling astype(args, kwargs) (line 645)
    astype_call_result_102425 = invoke(stypy.reporting.localization.Localization(__file__, 645, 14), astype_102421, *[dtype_102423], **kwargs_102424)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 645, 9), tuple_102413, astype_call_result_102425)
    
    # Processing the call keyword arguments (line 644)
    # Getting the type of 'axis' (line 645)
    axis_102426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 71), 'axis', False)
    keyword_102427 = axis_102426
    kwargs_102428 = {'axis': keyword_102427}
    # Getting the type of 'np' (line 644)
    np_102411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 644)
    concatenate_102412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 11), np_102411, 'concatenate')
    # Calling concatenate(args, kwargs) (line 644)
    concatenate_call_result_102429 = invoke(stypy.reporting.localization.Localization(__file__, 644, 11), concatenate_102412, *[tuple_102413], **kwargs_102428)
    
    # Assigning a type to the variable 'stypy_return_type' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'stypy_return_type', concatenate_call_result_102429)
    
    # ################# End of '_append_med(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_med' in the type store
    # Getting the type of 'stypy_return_type' (line 591)
    stypy_return_type_102430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102430)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_med'
    return stypy_return_type_102430

# Assigning a type to the variable '_append_med' (line 591)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 0), '_append_med', _append_med)

@norecursion
def _prepend_min(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 41), 'int')
    defaults = [int_102431]
    # Create a new context for function '_prepend_min'
    module_type_store = module_type_store.open_function_context('_prepend_min', 648, 0, False)
    
    # Passed parameters checking function
    _prepend_min.stypy_localization = localization
    _prepend_min.stypy_type_of_self = None
    _prepend_min.stypy_type_store = module_type_store
    _prepend_min.stypy_function_name = '_prepend_min'
    _prepend_min.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _prepend_min.stypy_varargs_param_name = None
    _prepend_min.stypy_kwargs_param_name = None
    _prepend_min.stypy_call_defaults = defaults
    _prepend_min.stypy_call_varargs = varargs
    _prepend_min.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepend_min', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepend_min', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepend_min(...)' code ##################

    str_102432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, (-1)), 'str', '\n    Prepend `pad_amt` minimum values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to prepend.\n    num : int\n        Depth into `arr` along `axis` to calculate minimum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values prepended along `axis`. The\n        prepended region is the minimum of the first `num` values along\n        `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 672)
    pad_amt_102433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 7), 'pad_amt')
    int_102434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 18), 'int')
    # Applying the binary operator '==' (line 672)
    result_eq_102435 = python_operator(stypy.reporting.localization.Localization(__file__, 672, 7), '==', pad_amt_102433, int_102434)
    
    # Testing the type of an if condition (line 672)
    if_condition_102436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 672, 4), result_eq_102435)
    # Assigning a type to the variable 'if_condition_102436' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'if_condition_102436', if_condition_102436)
    # SSA begins for if statement (line 672)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 673)
    arr_102437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'stypy_return_type', arr_102437)
    # SSA join for if statement (line 672)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 676)
    num_102438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 7), 'num')
    int_102439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 14), 'int')
    # Applying the binary operator '==' (line 676)
    result_eq_102440 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 7), '==', num_102438, int_102439)
    
    # Testing the type of an if condition (line 676)
    if_condition_102441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 676, 4), result_eq_102440)
    # Assigning a type to the variable 'if_condition_102441' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'if_condition_102441', if_condition_102441)
    # SSA begins for if statement (line 676)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _prepend_edge(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'arr' (line 677)
    arr_102443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'arr', False)
    # Getting the type of 'pad_amt' (line 677)
    pad_amt_102444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 34), 'pad_amt', False)
    # Getting the type of 'axis' (line 677)
    axis_102445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 43), 'axis', False)
    # Processing the call keyword arguments (line 677)
    kwargs_102446 = {}
    # Getting the type of '_prepend_edge' (line 677)
    _prepend_edge_102442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 677)
    _prepend_edge_call_result_102447 = invoke(stypy.reporting.localization.Localization(__file__, 677, 15), _prepend_edge_102442, *[arr_102443, pad_amt_102444, axis_102445], **kwargs_102446)
    
    # Assigning a type to the variable 'stypy_return_type' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'stypy_return_type', _prepend_edge_call_result_102447)
    # SSA join for if statement (line 676)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 680)
    # Getting the type of 'num' (line 680)
    num_102448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'num')
    # Getting the type of 'None' (line 680)
    None_102449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 18), 'None')
    
    (may_be_102450, more_types_in_union_102451) = may_not_be_none(num_102448, None_102449)

    if may_be_102450:

        if more_types_in_union_102451:
            # Runtime conditional SSA (line 680)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 681)
        num_102452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 681)
        axis_102453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 28), 'axis')
        # Getting the type of 'arr' (line 681)
        arr_102454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 681)
        shape_102455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 18), arr_102454, 'shape')
        # Obtaining the member '__getitem__' of a type (line 681)
        getitem___102456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 681, 18), shape_102455, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 681)
        subscript_call_result_102457 = invoke(stypy.reporting.localization.Localization(__file__, 681, 18), getitem___102456, axis_102453)
        
        # Applying the binary operator '>=' (line 681)
        result_ge_102458 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 11), '>=', num_102452, subscript_call_result_102457)
        
        # Testing the type of an if condition (line 681)
        if_condition_102459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 681, 8), result_ge_102458)
        # Assigning a type to the variable 'if_condition_102459' (line 681)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 8), 'if_condition_102459', if_condition_102459)
        # SSA begins for if statement (line 681)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 682):
        # Getting the type of 'None' (line 682)
        None_102460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 18), 'None')
        # Assigning a type to the variable 'num' (line 682)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 12), 'num', None_102460)
        # SSA join for if statement (line 681)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_102451:
            # SSA join for if statement (line 680)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 685):
    
    # Call to tuple(...): (line 685)
    # Processing the call arguments (line 685)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 685, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 686)
    # Processing the call arguments (line 686)
    # Getting the type of 'arr' (line 686)
    arr_102475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 686)
    shape_102476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 46), arr_102475, 'shape')
    # Processing the call keyword arguments (line 686)
    kwargs_102477 = {}
    # Getting the type of 'enumerate' (line 686)
    enumerate_102474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 686)
    enumerate_call_result_102478 = invoke(stypy.reporting.localization.Localization(__file__, 686, 36), enumerate_102474, *[shape_102476], **kwargs_102477)
    
    comprehension_102479 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 22), enumerate_call_result_102478)
    # Assigning a type to the variable 'i' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 22), comprehension_102479))
    # Assigning a type to the variable 'x' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 22), comprehension_102479))
    
    
    # Getting the type of 'i' (line 685)
    i_102462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 37), 'i', False)
    # Getting the type of 'axis' (line 685)
    axis_102463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 42), 'axis', False)
    # Applying the binary operator '!=' (line 685)
    result_ne_102464 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 37), '!=', i_102462, axis_102463)
    
    # Testing the type of an if expression (line 685)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 22), result_ne_102464)
    # SSA begins for if expression (line 685)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 685)
    # Processing the call arguments (line 685)
    # Getting the type of 'None' (line 685)
    None_102466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 28), 'None', False)
    # Processing the call keyword arguments (line 685)
    kwargs_102467 = {}
    # Getting the type of 'slice' (line 685)
    slice_102465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 685)
    slice_call_result_102468 = invoke(stypy.reporting.localization.Localization(__file__, 685, 22), slice_102465, *[None_102466], **kwargs_102467)
    
    # SSA branch for the else part of an if expression (line 685)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 685)
    # Processing the call arguments (line 685)
    # Getting the type of 'num' (line 685)
    num_102470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 58), 'num', False)
    # Processing the call keyword arguments (line 685)
    kwargs_102471 = {}
    # Getting the type of 'slice' (line 685)
    slice_102469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 685)
    slice_call_result_102472 = invoke(stypy.reporting.localization.Localization(__file__, 685, 52), slice_102469, *[num_102470], **kwargs_102471)
    
    # SSA join for if expression (line 685)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102473 = union_type.UnionType.add(slice_call_result_102468, slice_call_result_102472)
    
    list_102480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 685, 22), list_102480, if_exp_102473)
    # Processing the call keyword arguments (line 685)
    kwargs_102481 = {}
    # Getting the type of 'tuple' (line 685)
    tuple_102461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 685)
    tuple_call_result_102482 = invoke(stypy.reporting.localization.Localization(__file__, 685, 16), tuple_102461, *[list_102480], **kwargs_102481)
    
    # Assigning a type to the variable 'min_slice' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'min_slice', tuple_call_result_102482)
    
    # Assigning a Call to a Name (line 689):
    
    # Call to tuple(...): (line 689)
    # Processing the call arguments (line 689)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 689, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'arr' (line 690)
    arr_102491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 690)
    shape_102492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 50), arr_102491, 'shape')
    # Processing the call keyword arguments (line 690)
    kwargs_102493 = {}
    # Getting the type of 'enumerate' (line 690)
    enumerate_102490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 690)
    enumerate_call_result_102494 = invoke(stypy.reporting.localization.Localization(__file__, 690, 40), enumerate_102490, *[shape_102492], **kwargs_102493)
    
    comprehension_102495 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 26), enumerate_call_result_102494)
    # Assigning a type to the variable 'i' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 26), comprehension_102495))
    # Assigning a type to the variable 'x' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 26), comprehension_102495))
    
    
    # Getting the type of 'i' (line 689)
    i_102484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 31), 'i', False)
    # Getting the type of 'axis' (line 689)
    axis_102485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 36), 'axis', False)
    # Applying the binary operator '!=' (line 689)
    result_ne_102486 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 31), '!=', i_102484, axis_102485)
    
    # Testing the type of an if expression (line 689)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 689, 26), result_ne_102486)
    # SSA begins for if expression (line 689)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 689)
    x_102487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 689)
    module_type_store.open_ssa_branch('if expression else')
    int_102488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 46), 'int')
    # SSA join for if expression (line 689)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102489 = union_type.UnionType.add(x_102487, int_102488)
    
    list_102496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 689, 26), list_102496, if_exp_102489)
    # Processing the call keyword arguments (line 689)
    kwargs_102497 = {}
    # Getting the type of 'tuple' (line 689)
    tuple_102483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 689)
    tuple_call_result_102498 = invoke(stypy.reporting.localization.Localization(__file__, 689, 20), tuple_102483, *[list_102496], **kwargs_102497)
    
    # Assigning a type to the variable 'pad_singleton' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'pad_singleton', tuple_call_result_102498)
    
    # Assigning a Call to a Name (line 693):
    
    # Call to reshape(...): (line 693)
    # Processing the call arguments (line 693)
    # Getting the type of 'pad_singleton' (line 693)
    pad_singleton_102509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 54), 'pad_singleton', False)
    # Processing the call keyword arguments (line 693)
    kwargs_102510 = {}
    
    # Call to min(...): (line 693)
    # Processing the call keyword arguments (line 693)
    # Getting the type of 'axis' (line 693)
    axis_102504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 40), 'axis', False)
    keyword_102505 = axis_102504
    kwargs_102506 = {'axis': keyword_102505}
    
    # Obtaining the type of the subscript
    # Getting the type of 'min_slice' (line 693)
    min_slice_102499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 20), 'min_slice', False)
    # Getting the type of 'arr' (line 693)
    arr_102500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___102501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), arr_102500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_102502 = invoke(stypy.reporting.localization.Localization(__file__, 693, 16), getitem___102501, min_slice_102499)
    
    # Obtaining the member 'min' of a type (line 693)
    min_102503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), subscript_call_result_102502, 'min')
    # Calling min(args, kwargs) (line 693)
    min_call_result_102507 = invoke(stypy.reporting.localization.Localization(__file__, 693, 16), min_102503, *[], **kwargs_102506)
    
    # Obtaining the member 'reshape' of a type (line 693)
    reshape_102508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 16), min_call_result_102507, 'reshape')
    # Calling reshape(args, kwargs) (line 693)
    reshape_call_result_102511 = invoke(stypy.reporting.localization.Localization(__file__, 693, 16), reshape_102508, *[pad_singleton_102509], **kwargs_102510)
    
    # Assigning a type to the variable 'min_chunk' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'min_chunk', reshape_call_result_102511)
    
    # Call to concatenate(...): (line 696)
    # Processing the call arguments (line 696)
    
    # Obtaining an instance of the builtin type 'tuple' (line 696)
    tuple_102514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 696)
    # Adding element type (line 696)
    
    # Call to repeat(...): (line 696)
    # Processing the call arguments (line 696)
    # Getting the type of 'pad_amt' (line 696)
    pad_amt_102517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 44), 'pad_amt', False)
    # Processing the call keyword arguments (line 696)
    # Getting the type of 'axis' (line 696)
    axis_102518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 58), 'axis', False)
    keyword_102519 = axis_102518
    kwargs_102520 = {'axis': keyword_102519}
    # Getting the type of 'min_chunk' (line 696)
    min_chunk_102515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 27), 'min_chunk', False)
    # Obtaining the member 'repeat' of a type (line 696)
    repeat_102516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 27), min_chunk_102515, 'repeat')
    # Calling repeat(args, kwargs) (line 696)
    repeat_call_result_102521 = invoke(stypy.reporting.localization.Localization(__file__, 696, 27), repeat_102516, *[pad_amt_102517], **kwargs_102520)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 27), tuple_102514, repeat_call_result_102521)
    # Adding element type (line 696)
    # Getting the type of 'arr' (line 696)
    arr_102522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 65), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 696, 27), tuple_102514, arr_102522)
    
    # Processing the call keyword arguments (line 696)
    # Getting the type of 'axis' (line 697)
    axis_102523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 31), 'axis', False)
    keyword_102524 = axis_102523
    kwargs_102525 = {'axis': keyword_102524}
    # Getting the type of 'np' (line 696)
    np_102512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 696)
    concatenate_102513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 11), np_102512, 'concatenate')
    # Calling concatenate(args, kwargs) (line 696)
    concatenate_call_result_102526 = invoke(stypy.reporting.localization.Localization(__file__, 696, 11), concatenate_102513, *[tuple_102514], **kwargs_102525)
    
    # Assigning a type to the variable 'stypy_return_type' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'stypy_return_type', concatenate_call_result_102526)
    
    # ################# End of '_prepend_min(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepend_min' in the type store
    # Getting the type of 'stypy_return_type' (line 648)
    stypy_return_type_102527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102527)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepend_min'
    return stypy_return_type_102527

# Assigning a type to the variable '_prepend_min' (line 648)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 0), '_prepend_min', _prepend_min)

@norecursion
def _append_min(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 40), 'int')
    defaults = [int_102528]
    # Create a new context for function '_append_min'
    module_type_store = module_type_store.open_function_context('_append_min', 700, 0, False)
    
    # Passed parameters checking function
    _append_min.stypy_localization = localization
    _append_min.stypy_type_of_self = None
    _append_min.stypy_type_store = module_type_store
    _append_min.stypy_function_name = '_append_min'
    _append_min.stypy_param_names_list = ['arr', 'pad_amt', 'num', 'axis']
    _append_min.stypy_varargs_param_name = None
    _append_min.stypy_kwargs_param_name = None
    _append_min.stypy_call_defaults = defaults
    _append_min.stypy_call_varargs = varargs
    _append_min.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_append_min', ['arr', 'pad_amt', 'num', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_append_min', localization, ['arr', 'pad_amt', 'num', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_append_min(...)' code ##################

    str_102529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, (-1)), 'str', '\n    Append `pad_amt` median values along `axis`.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : int\n        Amount of padding to append.\n    num : int\n        Depth into `arr` along `axis` to calculate minimum.\n        Range: [1, `arr.shape[axis]`] or None (entire axis)\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt` values appended along `axis`. The\n        appended region is the minimum of the final `num` values along `axis`.\n\n    ')
    
    
    # Getting the type of 'pad_amt' (line 723)
    pad_amt_102530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 7), 'pad_amt')
    int_102531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 18), 'int')
    # Applying the binary operator '==' (line 723)
    result_eq_102532 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 7), '==', pad_amt_102530, int_102531)
    
    # Testing the type of an if condition (line 723)
    if_condition_102533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 723, 4), result_eq_102532)
    # Assigning a type to the variable 'if_condition_102533' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'if_condition_102533', if_condition_102533)
    # SSA begins for if statement (line 723)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 724)
    arr_102534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'stypy_return_type', arr_102534)
    # SSA join for if statement (line 723)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'num' (line 727)
    num_102535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 7), 'num')
    int_102536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 14), 'int')
    # Applying the binary operator '==' (line 727)
    result_eq_102537 = python_operator(stypy.reporting.localization.Localization(__file__, 727, 7), '==', num_102535, int_102536)
    
    # Testing the type of an if condition (line 727)
    if_condition_102538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 727, 4), result_eq_102537)
    # Assigning a type to the variable 'if_condition_102538' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'if_condition_102538', if_condition_102538)
    # SSA begins for if statement (line 727)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _append_edge(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'arr' (line 728)
    arr_102540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 28), 'arr', False)
    # Getting the type of 'pad_amt' (line 728)
    pad_amt_102541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 33), 'pad_amt', False)
    # Getting the type of 'axis' (line 728)
    axis_102542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 42), 'axis', False)
    # Processing the call keyword arguments (line 728)
    kwargs_102543 = {}
    # Getting the type of '_append_edge' (line 728)
    _append_edge_102539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 15), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 728)
    _append_edge_call_result_102544 = invoke(stypy.reporting.localization.Localization(__file__, 728, 15), _append_edge_102539, *[arr_102540, pad_amt_102541, axis_102542], **kwargs_102543)
    
    # Assigning a type to the variable 'stypy_return_type' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 8), 'stypy_return_type', _append_edge_call_result_102544)
    # SSA join for if statement (line 727)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 731)
    # Getting the type of 'num' (line 731)
    num_102545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'num')
    # Getting the type of 'None' (line 731)
    None_102546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 'None')
    
    (may_be_102547, more_types_in_union_102548) = may_not_be_none(num_102545, None_102546)

    if may_be_102547:

        if more_types_in_union_102548:
            # Runtime conditional SSA (line 731)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'num' (line 732)
        num_102549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 11), 'num')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 732)
        axis_102550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 28), 'axis')
        # Getting the type of 'arr' (line 732)
        arr_102551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 18), 'arr')
        # Obtaining the member 'shape' of a type (line 732)
        shape_102552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 18), arr_102551, 'shape')
        # Obtaining the member '__getitem__' of a type (line 732)
        getitem___102553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 18), shape_102552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 732)
        subscript_call_result_102554 = invoke(stypy.reporting.localization.Localization(__file__, 732, 18), getitem___102553, axis_102550)
        
        # Applying the binary operator '>=' (line 732)
        result_ge_102555 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 11), '>=', num_102549, subscript_call_result_102554)
        
        # Testing the type of an if condition (line 732)
        if_condition_102556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 732, 8), result_ge_102555)
        # Assigning a type to the variable 'if_condition_102556' (line 732)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'if_condition_102556', if_condition_102556)
        # SSA begins for if statement (line 732)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 733):
        # Getting the type of 'None' (line 733)
        None_102557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 18), 'None')
        # Assigning a type to the variable 'num' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 12), 'num', None_102557)
        # SSA join for if statement (line 732)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_102548:
            # SSA join for if statement (line 731)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 736):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 736)
    axis_102558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 20), 'axis')
    # Getting the type of 'arr' (line 736)
    arr_102559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 736)
    shape_102560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 10), arr_102559, 'shape')
    # Obtaining the member '__getitem__' of a type (line 736)
    getitem___102561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 10), shape_102560, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 736)
    subscript_call_result_102562 = invoke(stypy.reporting.localization.Localization(__file__, 736, 10), getitem___102561, axis_102558)
    
    int_102563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 28), 'int')
    # Applying the binary operator '-' (line 736)
    result_sub_102564 = python_operator(stypy.reporting.localization.Localization(__file__, 736, 10), '-', subscript_call_result_102562, int_102563)
    
    # Assigning a type to the variable 'end' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'end', result_sub_102564)
    
    # Type idiom detected: calculating its left and rigth part (line 737)
    # Getting the type of 'num' (line 737)
    num_102565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'num')
    # Getting the type of 'None' (line 737)
    None_102566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 18), 'None')
    
    (may_be_102567, more_types_in_union_102568) = may_not_be_none(num_102565, None_102566)

    if may_be_102567:

        if more_types_in_union_102568:
            # Runtime conditional SSA (line 737)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 738):
        
        # Call to tuple(...): (line 738)
        # Processing the call arguments (line 738)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 739, 12, True)
        # Calculating comprehension expression
        
        # Call to enumerate(...): (line 740)
        # Processing the call arguments (line 740)
        # Getting the type of 'arr' (line 740)
        arr_102587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 36), 'arr', False)
        # Obtaining the member 'shape' of a type (line 740)
        shape_102588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 740, 36), arr_102587, 'shape')
        # Processing the call keyword arguments (line 740)
        kwargs_102589 = {}
        # Getting the type of 'enumerate' (line 740)
        enumerate_102586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 740)
        enumerate_call_result_102590 = invoke(stypy.reporting.localization.Localization(__file__, 740, 26), enumerate_102586, *[shape_102588], **kwargs_102589)
        
        comprehension_102591 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 12), enumerate_call_result_102590)
        # Assigning a type to the variable 'i' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 12), comprehension_102591))
        # Assigning a type to the variable 'x' (line 739)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 12), comprehension_102591))
        
        
        # Getting the type of 'i' (line 739)
        i_102570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 27), 'i', False)
        # Getting the type of 'axis' (line 739)
        axis_102571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 32), 'axis', False)
        # Applying the binary operator '!=' (line 739)
        result_ne_102572 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 27), '!=', i_102570, axis_102571)
        
        # Testing the type of an if expression (line 739)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 739, 12), result_ne_102572)
        # SSA begins for if expression (line 739)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'None' (line 739)
        None_102574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 18), 'None', False)
        # Processing the call keyword arguments (line 739)
        kwargs_102575 = {}
        # Getting the type of 'slice' (line 739)
        slice_102573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'slice', False)
        # Calling slice(args, kwargs) (line 739)
        slice_call_result_102576 = invoke(stypy.reporting.localization.Localization(__file__, 739, 12), slice_102573, *[None_102574], **kwargs_102575)
        
        # SSA branch for the else part of an if expression (line 739)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to slice(...): (line 739)
        # Processing the call arguments (line 739)
        # Getting the type of 'end' (line 739)
        end_102578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 48), 'end', False)
        # Getting the type of 'end' (line 739)
        end_102579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 53), 'end', False)
        # Getting the type of 'num' (line 739)
        num_102580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 59), 'num', False)
        # Applying the binary operator '-' (line 739)
        result_sub_102581 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 53), '-', end_102579, num_102580)
        
        int_102582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 64), 'int')
        # Processing the call keyword arguments (line 739)
        kwargs_102583 = {}
        # Getting the type of 'slice' (line 739)
        slice_102577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 42), 'slice', False)
        # Calling slice(args, kwargs) (line 739)
        slice_call_result_102584 = invoke(stypy.reporting.localization.Localization(__file__, 739, 42), slice_102577, *[end_102578, result_sub_102581, int_102582], **kwargs_102583)
        
        # SSA join for if expression (line 739)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_102585 = union_type.UnionType.add(slice_call_result_102576, slice_call_result_102584)
        
        list_102592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 12), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 739, 12), list_102592, if_exp_102585)
        # Processing the call keyword arguments (line 738)
        kwargs_102593 = {}
        # Getting the type of 'tuple' (line 738)
        tuple_102569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 738)
        tuple_call_result_102594 = invoke(stypy.reporting.localization.Localization(__file__, 738, 20), tuple_102569, *[list_102592], **kwargs_102593)
        
        # Assigning a type to the variable 'min_slice' (line 738)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'min_slice', tuple_call_result_102594)

        if more_types_in_union_102568:
            # Runtime conditional SSA for else branch (line 737)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_102567) or more_types_in_union_102568):
        
        # Assigning a Call to a Name (line 742):
        
        # Call to tuple(...): (line 742)
        # Processing the call arguments (line 742)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 742, 26, True)
        # Calculating comprehension expression
        # Getting the type of 'arr' (line 742)
        arr_102600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 47), 'arr', False)
        # Obtaining the member 'shape' of a type (line 742)
        shape_102601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 47), arr_102600, 'shape')
        comprehension_102602 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 26), shape_102601)
        # Assigning a type to the variable 'x' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 26), 'x', comprehension_102602)
        
        # Call to slice(...): (line 742)
        # Processing the call arguments (line 742)
        # Getting the type of 'None' (line 742)
        None_102597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'None', False)
        # Processing the call keyword arguments (line 742)
        kwargs_102598 = {}
        # Getting the type of 'slice' (line 742)
        slice_102596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 26), 'slice', False)
        # Calling slice(args, kwargs) (line 742)
        slice_call_result_102599 = invoke(stypy.reporting.localization.Localization(__file__, 742, 26), slice_102596, *[None_102597], **kwargs_102598)
        
        list_102603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 742, 26), list_102603, slice_call_result_102599)
        # Processing the call keyword arguments (line 742)
        kwargs_102604 = {}
        # Getting the type of 'tuple' (line 742)
        tuple_102595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 742)
        tuple_call_result_102605 = invoke(stypy.reporting.localization.Localization(__file__, 742, 20), tuple_102595, *[list_102603], **kwargs_102604)
        
        # Assigning a type to the variable 'min_slice' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'min_slice', tuple_call_result_102605)

        if (may_be_102567 and more_types_in_union_102568):
            # SSA join for if statement (line 737)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 745):
    
    # Call to tuple(...): (line 745)
    # Processing the call arguments (line 745)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 745, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 746)
    # Processing the call arguments (line 746)
    # Getting the type of 'arr' (line 746)
    arr_102614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 746)
    shape_102615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 50), arr_102614, 'shape')
    # Processing the call keyword arguments (line 746)
    kwargs_102616 = {}
    # Getting the type of 'enumerate' (line 746)
    enumerate_102613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 746)
    enumerate_call_result_102617 = invoke(stypy.reporting.localization.Localization(__file__, 746, 40), enumerate_102613, *[shape_102615], **kwargs_102616)
    
    comprehension_102618 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 26), enumerate_call_result_102617)
    # Assigning a type to the variable 'i' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 26), comprehension_102618))
    # Assigning a type to the variable 'x' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 26), comprehension_102618))
    
    
    # Getting the type of 'i' (line 745)
    i_102607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 31), 'i', False)
    # Getting the type of 'axis' (line 745)
    axis_102608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 36), 'axis', False)
    # Applying the binary operator '!=' (line 745)
    result_ne_102609 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 31), '!=', i_102607, axis_102608)
    
    # Testing the type of an if expression (line 745)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 745, 26), result_ne_102609)
    # SSA begins for if expression (line 745)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 745)
    x_102610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 745)
    module_type_store.open_ssa_branch('if expression else')
    int_102611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 46), 'int')
    # SSA join for if expression (line 745)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102612 = union_type.UnionType.add(x_102610, int_102611)
    
    list_102619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 26), list_102619, if_exp_102612)
    # Processing the call keyword arguments (line 745)
    kwargs_102620 = {}
    # Getting the type of 'tuple' (line 745)
    tuple_102606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 745)
    tuple_call_result_102621 = invoke(stypy.reporting.localization.Localization(__file__, 745, 20), tuple_102606, *[list_102619], **kwargs_102620)
    
    # Assigning a type to the variable 'pad_singleton' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'pad_singleton', tuple_call_result_102621)
    
    # Assigning a Call to a Name (line 749):
    
    # Call to reshape(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'pad_singleton' (line 749)
    pad_singleton_102632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 54), 'pad_singleton', False)
    # Processing the call keyword arguments (line 749)
    kwargs_102633 = {}
    
    # Call to min(...): (line 749)
    # Processing the call keyword arguments (line 749)
    # Getting the type of 'axis' (line 749)
    axis_102627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 40), 'axis', False)
    keyword_102628 = axis_102627
    kwargs_102629 = {'axis': keyword_102628}
    
    # Obtaining the type of the subscript
    # Getting the type of 'min_slice' (line 749)
    min_slice_102622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 20), 'min_slice', False)
    # Getting the type of 'arr' (line 749)
    arr_102623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 749)
    getitem___102624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 16), arr_102623, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 749)
    subscript_call_result_102625 = invoke(stypy.reporting.localization.Localization(__file__, 749, 16), getitem___102624, min_slice_102622)
    
    # Obtaining the member 'min' of a type (line 749)
    min_102626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 16), subscript_call_result_102625, 'min')
    # Calling min(args, kwargs) (line 749)
    min_call_result_102630 = invoke(stypy.reporting.localization.Localization(__file__, 749, 16), min_102626, *[], **kwargs_102629)
    
    # Obtaining the member 'reshape' of a type (line 749)
    reshape_102631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 16), min_call_result_102630, 'reshape')
    # Calling reshape(args, kwargs) (line 749)
    reshape_call_result_102634 = invoke(stypy.reporting.localization.Localization(__file__, 749, 16), reshape_102631, *[pad_singleton_102632], **kwargs_102633)
    
    # Assigning a type to the variable 'min_chunk' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 4), 'min_chunk', reshape_call_result_102634)
    
    # Call to concatenate(...): (line 752)
    # Processing the call arguments (line 752)
    
    # Obtaining an instance of the builtin type 'tuple' (line 752)
    tuple_102637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 752)
    # Adding element type (line 752)
    # Getting the type of 'arr' (line 752)
    arr_102638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 27), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 27), tuple_102637, arr_102638)
    # Adding element type (line 752)
    
    # Call to repeat(...): (line 752)
    # Processing the call arguments (line 752)
    # Getting the type of 'pad_amt' (line 752)
    pad_amt_102641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 49), 'pad_amt', False)
    # Processing the call keyword arguments (line 752)
    # Getting the type of 'axis' (line 752)
    axis_102642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 63), 'axis', False)
    keyword_102643 = axis_102642
    kwargs_102644 = {'axis': keyword_102643}
    # Getting the type of 'min_chunk' (line 752)
    min_chunk_102639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 32), 'min_chunk', False)
    # Obtaining the member 'repeat' of a type (line 752)
    repeat_102640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 32), min_chunk_102639, 'repeat')
    # Calling repeat(args, kwargs) (line 752)
    repeat_call_result_102645 = invoke(stypy.reporting.localization.Localization(__file__, 752, 32), repeat_102640, *[pad_amt_102641], **kwargs_102644)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 27), tuple_102637, repeat_call_result_102645)
    
    # Processing the call keyword arguments (line 752)
    # Getting the type of 'axis' (line 753)
    axis_102646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 31), 'axis', False)
    keyword_102647 = axis_102646
    kwargs_102648 = {'axis': keyword_102647}
    # Getting the type of 'np' (line 752)
    np_102635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 752)
    concatenate_102636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 11), np_102635, 'concatenate')
    # Calling concatenate(args, kwargs) (line 752)
    concatenate_call_result_102649 = invoke(stypy.reporting.localization.Localization(__file__, 752, 11), concatenate_102636, *[tuple_102637], **kwargs_102648)
    
    # Assigning a type to the variable 'stypy_return_type' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'stypy_return_type', concatenate_call_result_102649)
    
    # ################# End of '_append_min(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_append_min' in the type store
    # Getting the type of 'stypy_return_type' (line 700)
    stypy_return_type_102650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102650)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_append_min'
    return stypy_return_type_102650

# Assigning a type to the variable '_append_min' (line 700)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 0), '_append_min', _append_min)

@norecursion
def _pad_ref(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 40), 'int')
    defaults = [int_102651]
    # Create a new context for function '_pad_ref'
    module_type_store = module_type_store.open_function_context('_pad_ref', 756, 0, False)
    
    # Passed parameters checking function
    _pad_ref.stypy_localization = localization
    _pad_ref.stypy_type_of_self = None
    _pad_ref.stypy_type_store = module_type_store
    _pad_ref.stypy_function_name = '_pad_ref'
    _pad_ref.stypy_param_names_list = ['arr', 'pad_amt', 'method', 'axis']
    _pad_ref.stypy_varargs_param_name = None
    _pad_ref.stypy_kwargs_param_name = None
    _pad_ref.stypy_call_defaults = defaults
    _pad_ref.stypy_call_varargs = varargs
    _pad_ref.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pad_ref', ['arr', 'pad_amt', 'method', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pad_ref', localization, ['arr', 'pad_amt', 'method', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pad_ref(...)' code ##################

    str_102652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, (-1)), 'str', "\n    Pad `axis` of `arr` by reflection.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    method : str\n        Controls method of reflection; options are 'even' or 'odd'.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded with reflected\n        values from the original array.\n\n    Notes\n    -----\n    This algorithm does not pad with repetition, i.e. the edges are not\n    repeated in the reflection. For that behavior, use `mode='symmetric'`.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_102653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 15), 'int')
    # Getting the type of 'pad_amt' (line 789)
    pad_amt_102654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 789)
    getitem___102655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 7), pad_amt_102654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 789)
    subscript_call_result_102656 = invoke(stypy.reporting.localization.Localization(__file__, 789, 7), getitem___102655, int_102653)
    
    int_102657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 21), 'int')
    # Applying the binary operator '==' (line 789)
    result_eq_102658 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 7), '==', subscript_call_result_102656, int_102657)
    
    
    
    # Obtaining the type of the subscript
    int_102659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 35), 'int')
    # Getting the type of 'pad_amt' (line 789)
    pad_amt_102660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 27), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 789)
    getitem___102661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 27), pad_amt_102660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 789)
    subscript_call_result_102662 = invoke(stypy.reporting.localization.Localization(__file__, 789, 27), getitem___102661, int_102659)
    
    int_102663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 789, 41), 'int')
    # Applying the binary operator '==' (line 789)
    result_eq_102664 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 27), '==', subscript_call_result_102662, int_102663)
    
    # Applying the binary operator 'and' (line 789)
    result_and_keyword_102665 = python_operator(stypy.reporting.localization.Localization(__file__, 789, 7), 'and', result_eq_102658, result_eq_102664)
    
    # Testing the type of an if condition (line 789)
    if_condition_102666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 789, 4), result_and_keyword_102665)
    # Assigning a type to the variable 'if_condition_102666' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 4), 'if_condition_102666', if_condition_102666)
    # SSA begins for if statement (line 789)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 790)
    arr_102667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'stypy_return_type', arr_102667)
    # SSA join for if statement (line 789)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 796):
    
    # Call to tuple(...): (line 796)
    # Processing the call arguments (line 796)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 796, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'arr' (line 797)
    arr_102687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 797)
    shape_102688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 46), arr_102687, 'shape')
    # Processing the call keyword arguments (line 797)
    kwargs_102689 = {}
    # Getting the type of 'enumerate' (line 797)
    enumerate_102686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 797)
    enumerate_call_result_102690 = invoke(stypy.reporting.localization.Localization(__file__, 797, 36), enumerate_102686, *[shape_102688], **kwargs_102689)
    
    comprehension_102691 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 22), enumerate_call_result_102690)
    # Assigning a type to the variable 'i' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 22), comprehension_102691))
    # Assigning a type to the variable 'x' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 22), comprehension_102691))
    
    
    # Getting the type of 'i' (line 796)
    i_102669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 37), 'i', False)
    # Getting the type of 'axis' (line 796)
    axis_102670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 42), 'axis', False)
    # Applying the binary operator '!=' (line 796)
    result_ne_102671 = python_operator(stypy.reporting.localization.Localization(__file__, 796, 37), '!=', i_102669, axis_102670)
    
    # Testing the type of an if expression (line 796)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 796, 22), result_ne_102671)
    # SSA begins for if expression (line 796)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 796)
    # Processing the call arguments (line 796)
    # Getting the type of 'None' (line 796)
    None_102673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 28), 'None', False)
    # Processing the call keyword arguments (line 796)
    kwargs_102674 = {}
    # Getting the type of 'slice' (line 796)
    slice_102672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 796)
    slice_call_result_102675 = invoke(stypy.reporting.localization.Localization(__file__, 796, 22), slice_102672, *[None_102673], **kwargs_102674)
    
    # SSA branch for the else part of an if expression (line 796)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 796)
    # Processing the call arguments (line 796)
    
    # Obtaining the type of the subscript
    int_102677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 66), 'int')
    # Getting the type of 'pad_amt' (line 796)
    pad_amt_102678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 58), 'pad_amt', False)
    # Obtaining the member '__getitem__' of a type (line 796)
    getitem___102679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 796, 58), pad_amt_102678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 796)
    subscript_call_result_102680 = invoke(stypy.reporting.localization.Localization(__file__, 796, 58), getitem___102679, int_102677)
    
    int_102681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 70), 'int')
    int_102682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 73), 'int')
    # Processing the call keyword arguments (line 796)
    kwargs_102683 = {}
    # Getting the type of 'slice' (line 796)
    slice_102676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 796)
    slice_call_result_102684 = invoke(stypy.reporting.localization.Localization(__file__, 796, 52), slice_102676, *[subscript_call_result_102680, int_102681, int_102682], **kwargs_102683)
    
    # SSA join for if expression (line 796)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102685 = union_type.UnionType.add(slice_call_result_102675, slice_call_result_102684)
    
    list_102692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 796, 22), list_102692, if_exp_102685)
    # Processing the call keyword arguments (line 796)
    kwargs_102693 = {}
    # Getting the type of 'tuple' (line 796)
    tuple_102668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 796)
    tuple_call_result_102694 = invoke(stypy.reporting.localization.Localization(__file__, 796, 16), tuple_102668, *[list_102692], **kwargs_102693)
    
    # Assigning a type to the variable 'ref_slice' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'ref_slice', tuple_call_result_102694)
    
    # Assigning a Subscript to a Name (line 799):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ref_slice' (line 799)
    ref_slice_102695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 21), 'ref_slice')
    # Getting the type of 'arr' (line 799)
    arr_102696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 17), 'arr')
    # Obtaining the member '__getitem__' of a type (line 799)
    getitem___102697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 17), arr_102696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 799)
    subscript_call_result_102698 = invoke(stypy.reporting.localization.Localization(__file__, 799, 17), getitem___102697, ref_slice_102695)
    
    # Assigning a type to the variable 'ref_chunk1' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 4), 'ref_chunk1', subscript_call_result_102698)
    
    # Assigning a Call to a Name (line 802):
    
    # Call to tuple(...): (line 802)
    # Processing the call arguments (line 802)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 802, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 803)
    # Processing the call arguments (line 803)
    # Getting the type of 'arr' (line 803)
    arr_102707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 803)
    shape_102708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 50), arr_102707, 'shape')
    # Processing the call keyword arguments (line 803)
    kwargs_102709 = {}
    # Getting the type of 'enumerate' (line 803)
    enumerate_102706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 803)
    enumerate_call_result_102710 = invoke(stypy.reporting.localization.Localization(__file__, 803, 40), enumerate_102706, *[shape_102708], **kwargs_102709)
    
    comprehension_102711 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 26), enumerate_call_result_102710)
    # Assigning a type to the variable 'i' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 26), comprehension_102711))
    # Assigning a type to the variable 'x' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 26), comprehension_102711))
    
    
    # Getting the type of 'i' (line 802)
    i_102700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 31), 'i', False)
    # Getting the type of 'axis' (line 802)
    axis_102701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 36), 'axis', False)
    # Applying the binary operator '!=' (line 802)
    result_ne_102702 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 31), '!=', i_102700, axis_102701)
    
    # Testing the type of an if expression (line 802)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 26), result_ne_102702)
    # SSA begins for if expression (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 802)
    x_102703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 802)
    module_type_store.open_ssa_branch('if expression else')
    int_102704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 46), 'int')
    # SSA join for if expression (line 802)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102705 = union_type.UnionType.add(x_102703, int_102704)
    
    list_102712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 802, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 802, 26), list_102712, if_exp_102705)
    # Processing the call keyword arguments (line 802)
    kwargs_102713 = {}
    # Getting the type of 'tuple' (line 802)
    tuple_102699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 802)
    tuple_call_result_102714 = invoke(stypy.reporting.localization.Localization(__file__, 802, 20), tuple_102699, *[list_102712], **kwargs_102713)
    
    # Assigning a type to the variable 'pad_singleton' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'pad_singleton', tuple_call_result_102714)
    
    
    
    # Obtaining the type of the subscript
    int_102715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 15), 'int')
    # Getting the type of 'pad_amt' (line 804)
    pad_amt_102716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 804)
    getitem___102717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 7), pad_amt_102716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 804)
    subscript_call_result_102718 = invoke(stypy.reporting.localization.Localization(__file__, 804, 7), getitem___102717, int_102715)
    
    int_102719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 21), 'int')
    # Applying the binary operator '==' (line 804)
    result_eq_102720 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 7), '==', subscript_call_result_102718, int_102719)
    
    # Testing the type of an if condition (line 804)
    if_condition_102721 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 804, 4), result_eq_102720)
    # Assigning a type to the variable 'if_condition_102721' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'if_condition_102721', if_condition_102721)
    # SSA begins for if statement (line 804)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 805):
    
    # Call to reshape(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'pad_singleton' (line 805)
    pad_singleton_102724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 40), 'pad_singleton', False)
    # Processing the call keyword arguments (line 805)
    kwargs_102725 = {}
    # Getting the type of 'ref_chunk1' (line 805)
    ref_chunk1_102722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 21), 'ref_chunk1', False)
    # Obtaining the member 'reshape' of a type (line 805)
    reshape_102723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 21), ref_chunk1_102722, 'reshape')
    # Calling reshape(args, kwargs) (line 805)
    reshape_call_result_102726 = invoke(stypy.reporting.localization.Localization(__file__, 805, 21), reshape_102723, *[pad_singleton_102724], **kwargs_102725)
    
    # Assigning a type to the variable 'ref_chunk1' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'ref_chunk1', reshape_call_result_102726)
    # SSA join for if statement (line 804)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_102727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 7), 'str', 'odd')
    # Getting the type of 'method' (line 808)
    method_102728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 16), 'method')
    # Applying the binary operator 'in' (line 808)
    result_contains_102729 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 7), 'in', str_102727, method_102728)
    
    
    
    # Obtaining the type of the subscript
    int_102730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 35), 'int')
    # Getting the type of 'pad_amt' (line 808)
    pad_amt_102731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 27), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___102732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 27), pad_amt_102731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_102733 = invoke(stypy.reporting.localization.Localization(__file__, 808, 27), getitem___102732, int_102730)
    
    int_102734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 40), 'int')
    # Applying the binary operator '>' (line 808)
    result_gt_102735 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 27), '>', subscript_call_result_102733, int_102734)
    
    # Applying the binary operator 'and' (line 808)
    result_and_keyword_102736 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 7), 'and', result_contains_102729, result_gt_102735)
    
    # Testing the type of an if condition (line 808)
    if_condition_102737 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 808, 4), result_and_keyword_102736)
    # Assigning a type to the variable 'if_condition_102737' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'if_condition_102737', if_condition_102737)
    # SSA begins for if statement (line 808)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 809):
    
    # Call to tuple(...): (line 809)
    # Processing the call arguments (line 809)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 809, 28, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'arr' (line 810)
    arr_102749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 52), 'arr', False)
    # Obtaining the member 'shape' of a type (line 810)
    shape_102750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 52), arr_102749, 'shape')
    # Processing the call keyword arguments (line 810)
    kwargs_102751 = {}
    # Getting the type of 'enumerate' (line 810)
    enumerate_102748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 42), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 810)
    enumerate_call_result_102752 = invoke(stypy.reporting.localization.Localization(__file__, 810, 42), enumerate_102748, *[shape_102750], **kwargs_102751)
    
    comprehension_102753 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 809, 28), enumerate_call_result_102752)
    # Assigning a type to the variable 'i' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 809, 28), comprehension_102753))
    # Assigning a type to the variable 'x' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 809, 28), comprehension_102753))
    
    
    # Getting the type of 'i' (line 809)
    i_102739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 43), 'i', False)
    # Getting the type of 'axis' (line 809)
    axis_102740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 48), 'axis', False)
    # Applying the binary operator '!=' (line 809)
    result_ne_102741 = python_operator(stypy.reporting.localization.Localization(__file__, 809, 43), '!=', i_102739, axis_102740)
    
    # Testing the type of an if expression (line 809)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 809, 28), result_ne_102741)
    # SSA begins for if expression (line 809)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'None' (line 809)
    None_102743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 34), 'None', False)
    # Processing the call keyword arguments (line 809)
    kwargs_102744 = {}
    # Getting the type of 'slice' (line 809)
    slice_102742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 28), 'slice', False)
    # Calling slice(args, kwargs) (line 809)
    slice_call_result_102745 = invoke(stypy.reporting.localization.Localization(__file__, 809, 28), slice_102742, *[None_102743], **kwargs_102744)
    
    # SSA branch for the else part of an if expression (line 809)
    module_type_store.open_ssa_branch('if expression else')
    int_102746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 58), 'int')
    # SSA join for if expression (line 809)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102747 = union_type.UnionType.add(slice_call_result_102745, int_102746)
    
    list_102754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 809, 28), list_102754, if_exp_102747)
    # Processing the call keyword arguments (line 809)
    kwargs_102755 = {}
    # Getting the type of 'tuple' (line 809)
    tuple_102738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 22), 'tuple', False)
    # Calling tuple(args, kwargs) (line 809)
    tuple_call_result_102756 = invoke(stypy.reporting.localization.Localization(__file__, 809, 22), tuple_102738, *[list_102754], **kwargs_102755)
    
    # Assigning a type to the variable 'edge_slice1' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'edge_slice1', tuple_call_result_102756)
    
    # Assigning a Call to a Name (line 811):
    
    # Call to reshape(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'pad_singleton' (line 811)
    pad_singleton_102762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 46), 'pad_singleton', False)
    # Processing the call keyword arguments (line 811)
    kwargs_102763 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice1' (line 811)
    edge_slice1_102757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 25), 'edge_slice1', False)
    # Getting the type of 'arr' (line 811)
    arr_102758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 21), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 811)
    getitem___102759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 21), arr_102758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 811)
    subscript_call_result_102760 = invoke(stypy.reporting.localization.Localization(__file__, 811, 21), getitem___102759, edge_slice1_102757)
    
    # Obtaining the member 'reshape' of a type (line 811)
    reshape_102761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 21), subscript_call_result_102760, 'reshape')
    # Calling reshape(args, kwargs) (line 811)
    reshape_call_result_102764 = invoke(stypy.reporting.localization.Localization(__file__, 811, 21), reshape_102761, *[pad_singleton_102762], **kwargs_102763)
    
    # Assigning a type to the variable 'edge_chunk' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'edge_chunk', reshape_call_result_102764)
    
    # Assigning a BinOp to a Name (line 812):
    int_102765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 21), 'int')
    # Getting the type of 'edge_chunk' (line 812)
    edge_chunk_102766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'edge_chunk')
    # Applying the binary operator '*' (line 812)
    result_mul_102767 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 21), '*', int_102765, edge_chunk_102766)
    
    # Getting the type of 'ref_chunk1' (line 812)
    ref_chunk1_102768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 38), 'ref_chunk1')
    # Applying the binary operator '-' (line 812)
    result_sub_102769 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 21), '-', result_mul_102767, ref_chunk1_102768)
    
    # Assigning a type to the variable 'ref_chunk1' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'ref_chunk1', result_sub_102769)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 813, 8), module_type_store, 'edge_chunk')
    # SSA join for if statement (line 808)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 819):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 819)
    axis_102770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 22), 'axis')
    # Getting the type of 'arr' (line 819)
    arr_102771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'arr')
    # Obtaining the member 'shape' of a type (line 819)
    shape_102772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 12), arr_102771, 'shape')
    # Obtaining the member '__getitem__' of a type (line 819)
    getitem___102773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 12), shape_102772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 819)
    subscript_call_result_102774 = invoke(stypy.reporting.localization.Localization(__file__, 819, 12), getitem___102773, axis_102770)
    
    
    # Obtaining the type of the subscript
    int_102775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 38), 'int')
    # Getting the type of 'pad_amt' (line 819)
    pad_amt_102776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 30), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 819)
    getitem___102777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 30), pad_amt_102776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 819)
    subscript_call_result_102778 = invoke(stypy.reporting.localization.Localization(__file__, 819, 30), getitem___102777, int_102775)
    
    # Applying the binary operator '-' (line 819)
    result_sub_102779 = python_operator(stypy.reporting.localization.Localization(__file__, 819, 12), '-', subscript_call_result_102774, subscript_call_result_102778)
    
    int_102780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 43), 'int')
    # Applying the binary operator '-' (line 819)
    result_sub_102781 = python_operator(stypy.reporting.localization.Localization(__file__, 819, 41), '-', result_sub_102779, int_102780)
    
    # Assigning a type to the variable 'start' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'start', result_sub_102781)
    
    # Assigning a BinOp to a Name (line 820):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 820)
    axis_102782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 20), 'axis')
    # Getting the type of 'arr' (line 820)
    arr_102783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 820)
    shape_102784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 10), arr_102783, 'shape')
    # Obtaining the member '__getitem__' of a type (line 820)
    getitem___102785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 10), shape_102784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 820)
    subscript_call_result_102786 = invoke(stypy.reporting.localization.Localization(__file__, 820, 10), getitem___102785, axis_102782)
    
    int_102787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 28), 'int')
    # Applying the binary operator '-' (line 820)
    result_sub_102788 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 10), '-', subscript_call_result_102786, int_102787)
    
    # Assigning a type to the variable 'end' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 4), 'end', result_sub_102788)
    
    # Assigning a Call to a Name (line 821):
    
    # Call to tuple(...): (line 821)
    # Processing the call arguments (line 821)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 821, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 822)
    # Processing the call arguments (line 822)
    # Getting the type of 'arr' (line 822)
    arr_102804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 822)
    shape_102805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 46), arr_102804, 'shape')
    # Processing the call keyword arguments (line 822)
    kwargs_102806 = {}
    # Getting the type of 'enumerate' (line 822)
    enumerate_102803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 822)
    enumerate_call_result_102807 = invoke(stypy.reporting.localization.Localization(__file__, 822, 36), enumerate_102803, *[shape_102805], **kwargs_102806)
    
    comprehension_102808 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 22), enumerate_call_result_102807)
    # Assigning a type to the variable 'i' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 22), comprehension_102808))
    # Assigning a type to the variable 'x' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 22), comprehension_102808))
    
    
    # Getting the type of 'i' (line 821)
    i_102790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 37), 'i', False)
    # Getting the type of 'axis' (line 821)
    axis_102791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 42), 'axis', False)
    # Applying the binary operator '!=' (line 821)
    result_ne_102792 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 37), '!=', i_102790, axis_102791)
    
    # Testing the type of an if expression (line 821)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 22), result_ne_102792)
    # SSA begins for if expression (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'None' (line 821)
    None_102794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 28), 'None', False)
    # Processing the call keyword arguments (line 821)
    kwargs_102795 = {}
    # Getting the type of 'slice' (line 821)
    slice_102793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 821)
    slice_call_result_102796 = invoke(stypy.reporting.localization.Localization(__file__, 821, 22), slice_102793, *[None_102794], **kwargs_102795)
    
    # SSA branch for the else part of an if expression (line 821)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'start' (line 821)
    start_102798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 58), 'start', False)
    # Getting the type of 'end' (line 821)
    end_102799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 65), 'end', False)
    # Processing the call keyword arguments (line 821)
    kwargs_102800 = {}
    # Getting the type of 'slice' (line 821)
    slice_102797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 821)
    slice_call_result_102801 = invoke(stypy.reporting.localization.Localization(__file__, 821, 52), slice_102797, *[start_102798, end_102799], **kwargs_102800)
    
    # SSA join for if expression (line 821)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102802 = union_type.UnionType.add(slice_call_result_102796, slice_call_result_102801)
    
    list_102809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 821, 22), list_102809, if_exp_102802)
    # Processing the call keyword arguments (line 821)
    kwargs_102810 = {}
    # Getting the type of 'tuple' (line 821)
    tuple_102789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 821)
    tuple_call_result_102811 = invoke(stypy.reporting.localization.Localization(__file__, 821, 16), tuple_102789, *[list_102809], **kwargs_102810)
    
    # Assigning a type to the variable 'ref_slice' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'ref_slice', tuple_call_result_102811)
    
    # Assigning a Call to a Name (line 823):
    
    # Call to tuple(...): (line 823)
    # Processing the call arguments (line 823)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 823, 20, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 824)
    # Processing the call arguments (line 824)
    # Getting the type of 'arr' (line 824)
    arr_102828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 44), 'arr', False)
    # Obtaining the member 'shape' of a type (line 824)
    shape_102829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 44), arr_102828, 'shape')
    # Processing the call keyword arguments (line 824)
    kwargs_102830 = {}
    # Getting the type of 'enumerate' (line 824)
    enumerate_102827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 34), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 824)
    enumerate_call_result_102831 = invoke(stypy.reporting.localization.Localization(__file__, 824, 34), enumerate_102827, *[shape_102829], **kwargs_102830)
    
    comprehension_102832 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 20), enumerate_call_result_102831)
    # Assigning a type to the variable 'i' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 20), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 20), comprehension_102832))
    # Assigning a type to the variable 'x' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 20), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 20), comprehension_102832))
    
    
    # Getting the type of 'i' (line 823)
    i_102813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 35), 'i', False)
    # Getting the type of 'axis' (line 823)
    axis_102814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 40), 'axis', False)
    # Applying the binary operator '!=' (line 823)
    result_ne_102815 = python_operator(stypy.reporting.localization.Localization(__file__, 823, 35), '!=', i_102813, axis_102814)
    
    # Testing the type of an if expression (line 823)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 823, 20), result_ne_102815)
    # SSA begins for if expression (line 823)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 823)
    # Processing the call arguments (line 823)
    # Getting the type of 'None' (line 823)
    None_102817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 26), 'None', False)
    # Processing the call keyword arguments (line 823)
    kwargs_102818 = {}
    # Getting the type of 'slice' (line 823)
    slice_102816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 20), 'slice', False)
    # Calling slice(args, kwargs) (line 823)
    slice_call_result_102819 = invoke(stypy.reporting.localization.Localization(__file__, 823, 20), slice_102816, *[None_102817], **kwargs_102818)
    
    # SSA branch for the else part of an if expression (line 823)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 823)
    # Processing the call arguments (line 823)
    # Getting the type of 'None' (line 823)
    None_102821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 56), 'None', False)
    # Getting the type of 'None' (line 823)
    None_102822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 62), 'None', False)
    int_102823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 68), 'int')
    # Processing the call keyword arguments (line 823)
    kwargs_102824 = {}
    # Getting the type of 'slice' (line 823)
    slice_102820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 50), 'slice', False)
    # Calling slice(args, kwargs) (line 823)
    slice_call_result_102825 = invoke(stypy.reporting.localization.Localization(__file__, 823, 50), slice_102820, *[None_102821, None_102822, int_102823], **kwargs_102824)
    
    # SSA join for if expression (line 823)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102826 = union_type.UnionType.add(slice_call_result_102819, slice_call_result_102825)
    
    list_102833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 20), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 20), list_102833, if_exp_102826)
    # Processing the call keyword arguments (line 823)
    kwargs_102834 = {}
    # Getting the type of 'tuple' (line 823)
    tuple_102812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 14), 'tuple', False)
    # Calling tuple(args, kwargs) (line 823)
    tuple_call_result_102835 = invoke(stypy.reporting.localization.Localization(__file__, 823, 14), tuple_102812, *[list_102833], **kwargs_102834)
    
    # Assigning a type to the variable 'rev_idx' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'rev_idx', tuple_call_result_102835)
    
    # Assigning a Subscript to a Name (line 825):
    
    # Obtaining the type of the subscript
    # Getting the type of 'rev_idx' (line 825)
    rev_idx_102836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 32), 'rev_idx')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ref_slice' (line 825)
    ref_slice_102837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 21), 'ref_slice')
    # Getting the type of 'arr' (line 825)
    arr_102838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 17), 'arr')
    # Obtaining the member '__getitem__' of a type (line 825)
    getitem___102839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 17), arr_102838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 825)
    subscript_call_result_102840 = invoke(stypy.reporting.localization.Localization(__file__, 825, 17), getitem___102839, ref_slice_102837)
    
    # Obtaining the member '__getitem__' of a type (line 825)
    getitem___102841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 17), subscript_call_result_102840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 825)
    subscript_call_result_102842 = invoke(stypy.reporting.localization.Localization(__file__, 825, 17), getitem___102841, rev_idx_102836)
    
    # Assigning a type to the variable 'ref_chunk2' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'ref_chunk2', subscript_call_result_102842)
    
    
    
    # Obtaining the type of the subscript
    int_102843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 15), 'int')
    # Getting the type of 'pad_amt' (line 827)
    pad_amt_102844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 827)
    getitem___102845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 7), pad_amt_102844, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 827)
    subscript_call_result_102846 = invoke(stypy.reporting.localization.Localization(__file__, 827, 7), getitem___102845, int_102843)
    
    int_102847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 21), 'int')
    # Applying the binary operator '==' (line 827)
    result_eq_102848 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 7), '==', subscript_call_result_102846, int_102847)
    
    # Testing the type of an if condition (line 827)
    if_condition_102849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 4), result_eq_102848)
    # Assigning a type to the variable 'if_condition_102849' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'if_condition_102849', if_condition_102849)
    # SSA begins for if statement (line 827)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 828):
    
    # Call to reshape(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'pad_singleton' (line 828)
    pad_singleton_102852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 40), 'pad_singleton', False)
    # Processing the call keyword arguments (line 828)
    kwargs_102853 = {}
    # Getting the type of 'ref_chunk2' (line 828)
    ref_chunk2_102850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 21), 'ref_chunk2', False)
    # Obtaining the member 'reshape' of a type (line 828)
    reshape_102851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 21), ref_chunk2_102850, 'reshape')
    # Calling reshape(args, kwargs) (line 828)
    reshape_call_result_102854 = invoke(stypy.reporting.localization.Localization(__file__, 828, 21), reshape_102851, *[pad_singleton_102852], **kwargs_102853)
    
    # Assigning a type to the variable 'ref_chunk2' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'ref_chunk2', reshape_call_result_102854)
    # SSA join for if statement (line 827)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_102855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 7), 'str', 'odd')
    # Getting the type of 'method' (line 830)
    method_102856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 16), 'method')
    # Applying the binary operator 'in' (line 830)
    result_contains_102857 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 7), 'in', str_102855, method_102856)
    
    # Testing the type of an if condition (line 830)
    if_condition_102858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 830, 4), result_contains_102857)
    # Assigning a type to the variable 'if_condition_102858' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'if_condition_102858', if_condition_102858)
    # SSA begins for if statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 831):
    
    # Call to tuple(...): (line 831)
    # Processing the call arguments (line 831)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 831, 28, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'arr' (line 832)
    arr_102870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 52), 'arr', False)
    # Obtaining the member 'shape' of a type (line 832)
    shape_102871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 52), arr_102870, 'shape')
    # Processing the call keyword arguments (line 832)
    kwargs_102872 = {}
    # Getting the type of 'enumerate' (line 832)
    enumerate_102869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 42), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 832)
    enumerate_call_result_102873 = invoke(stypy.reporting.localization.Localization(__file__, 832, 42), enumerate_102869, *[shape_102871], **kwargs_102872)
    
    comprehension_102874 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 28), enumerate_call_result_102873)
    # Assigning a type to the variable 'i' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 28), comprehension_102874))
    # Assigning a type to the variable 'x' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 28), comprehension_102874))
    
    
    # Getting the type of 'i' (line 831)
    i_102860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 43), 'i', False)
    # Getting the type of 'axis' (line 831)
    axis_102861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 48), 'axis', False)
    # Applying the binary operator '!=' (line 831)
    result_ne_102862 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 43), '!=', i_102860, axis_102861)
    
    # Testing the type of an if expression (line 831)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 831, 28), result_ne_102862)
    # SSA begins for if expression (line 831)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 831)
    # Processing the call arguments (line 831)
    # Getting the type of 'None' (line 831)
    None_102864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 34), 'None', False)
    # Processing the call keyword arguments (line 831)
    kwargs_102865 = {}
    # Getting the type of 'slice' (line 831)
    slice_102863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'slice', False)
    # Calling slice(args, kwargs) (line 831)
    slice_call_result_102866 = invoke(stypy.reporting.localization.Localization(__file__, 831, 28), slice_102863, *[None_102864], **kwargs_102865)
    
    # SSA branch for the else part of an if expression (line 831)
    module_type_store.open_ssa_branch('if expression else')
    int_102867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 58), 'int')
    # SSA join for if expression (line 831)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102868 = union_type.UnionType.add(slice_call_result_102866, int_102867)
    
    list_102875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 28), list_102875, if_exp_102868)
    # Processing the call keyword arguments (line 831)
    kwargs_102876 = {}
    # Getting the type of 'tuple' (line 831)
    tuple_102859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 22), 'tuple', False)
    # Calling tuple(args, kwargs) (line 831)
    tuple_call_result_102877 = invoke(stypy.reporting.localization.Localization(__file__, 831, 22), tuple_102859, *[list_102875], **kwargs_102876)
    
    # Assigning a type to the variable 'edge_slice2' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'edge_slice2', tuple_call_result_102877)
    
    # Assigning a Call to a Name (line 833):
    
    # Call to reshape(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'pad_singleton' (line 833)
    pad_singleton_102883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 46), 'pad_singleton', False)
    # Processing the call keyword arguments (line 833)
    kwargs_102884 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice2' (line 833)
    edge_slice2_102878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 25), 'edge_slice2', False)
    # Getting the type of 'arr' (line 833)
    arr_102879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 21), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 833)
    getitem___102880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 21), arr_102879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 833)
    subscript_call_result_102881 = invoke(stypy.reporting.localization.Localization(__file__, 833, 21), getitem___102880, edge_slice2_102878)
    
    # Obtaining the member 'reshape' of a type (line 833)
    reshape_102882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 21), subscript_call_result_102881, 'reshape')
    # Calling reshape(args, kwargs) (line 833)
    reshape_call_result_102885 = invoke(stypy.reporting.localization.Localization(__file__, 833, 21), reshape_102882, *[pad_singleton_102883], **kwargs_102884)
    
    # Assigning a type to the variable 'edge_chunk' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'edge_chunk', reshape_call_result_102885)
    
    # Assigning a BinOp to a Name (line 834):
    int_102886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 21), 'int')
    # Getting the type of 'edge_chunk' (line 834)
    edge_chunk_102887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 25), 'edge_chunk')
    # Applying the binary operator '*' (line 834)
    result_mul_102888 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 21), '*', int_102886, edge_chunk_102887)
    
    # Getting the type of 'ref_chunk2' (line 834)
    ref_chunk2_102889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 38), 'ref_chunk2')
    # Applying the binary operator '-' (line 834)
    result_sub_102890 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 21), '-', result_mul_102888, ref_chunk2_102889)
    
    # Assigning a type to the variable 'ref_chunk2' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'ref_chunk2', result_sub_102890)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 835, 8), module_type_store, 'edge_chunk')
    # SSA join for if statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to concatenate(...): (line 838)
    # Processing the call arguments (line 838)
    
    # Obtaining an instance of the builtin type 'tuple' (line 838)
    tuple_102893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 838)
    # Adding element type (line 838)
    # Getting the type of 'ref_chunk1' (line 838)
    ref_chunk1_102894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 27), 'ref_chunk1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 27), tuple_102893, ref_chunk1_102894)
    # Adding element type (line 838)
    # Getting the type of 'arr' (line 838)
    arr_102895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 39), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 27), tuple_102893, arr_102895)
    # Adding element type (line 838)
    # Getting the type of 'ref_chunk2' (line 838)
    ref_chunk2_102896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 44), 'ref_chunk2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 27), tuple_102893, ref_chunk2_102896)
    
    # Processing the call keyword arguments (line 838)
    # Getting the type of 'axis' (line 838)
    axis_102897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 62), 'axis', False)
    keyword_102898 = axis_102897
    kwargs_102899 = {'axis': keyword_102898}
    # Getting the type of 'np' (line 838)
    np_102891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 838)
    concatenate_102892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 11), np_102891, 'concatenate')
    # Calling concatenate(args, kwargs) (line 838)
    concatenate_call_result_102900 = invoke(stypy.reporting.localization.Localization(__file__, 838, 11), concatenate_102892, *[tuple_102893], **kwargs_102899)
    
    # Assigning a type to the variable 'stypy_return_type' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'stypy_return_type', concatenate_call_result_102900)
    
    # ################# End of '_pad_ref(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pad_ref' in the type store
    # Getting the type of 'stypy_return_type' (line 756)
    stypy_return_type_102901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_102901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pad_ref'
    return stypy_return_type_102901

# Assigning a type to the variable '_pad_ref' (line 756)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 0), '_pad_ref', _pad_ref)

@norecursion
def _pad_sym(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_102902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 40), 'int')
    defaults = [int_102902]
    # Create a new context for function '_pad_sym'
    module_type_store = module_type_store.open_function_context('_pad_sym', 841, 0, False)
    
    # Passed parameters checking function
    _pad_sym.stypy_localization = localization
    _pad_sym.stypy_type_of_self = None
    _pad_sym.stypy_type_store = module_type_store
    _pad_sym.stypy_function_name = '_pad_sym'
    _pad_sym.stypy_param_names_list = ['arr', 'pad_amt', 'method', 'axis']
    _pad_sym.stypy_varargs_param_name = None
    _pad_sym.stypy_kwargs_param_name = None
    _pad_sym.stypy_call_defaults = defaults
    _pad_sym.stypy_call_varargs = varargs
    _pad_sym.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pad_sym', ['arr', 'pad_amt', 'method', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pad_sym', localization, ['arr', 'pad_amt', 'method', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pad_sym(...)' code ##################

    str_102903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, (-1)), 'str', "\n    Pad `axis` of `arr` by symmetry.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    method : str\n        Controls method of symmetry; options are 'even' or 'odd'.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded with symmetric\n        values from the original array.\n\n    Notes\n    -----\n    This algorithm DOES pad with repetition, i.e. the edges are repeated.\n    For padding without repeated edges, use `mode='reflect'`.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_102904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 15), 'int')
    # Getting the type of 'pad_amt' (line 874)
    pad_amt_102905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 874)
    getitem___102906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 7), pad_amt_102905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 874)
    subscript_call_result_102907 = invoke(stypy.reporting.localization.Localization(__file__, 874, 7), getitem___102906, int_102904)
    
    int_102908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 21), 'int')
    # Applying the binary operator '==' (line 874)
    result_eq_102909 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 7), '==', subscript_call_result_102907, int_102908)
    
    
    
    # Obtaining the type of the subscript
    int_102910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 35), 'int')
    # Getting the type of 'pad_amt' (line 874)
    pad_amt_102911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 27), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 874)
    getitem___102912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 27), pad_amt_102911, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 874)
    subscript_call_result_102913 = invoke(stypy.reporting.localization.Localization(__file__, 874, 27), getitem___102912, int_102910)
    
    int_102914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 41), 'int')
    # Applying the binary operator '==' (line 874)
    result_eq_102915 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 27), '==', subscript_call_result_102913, int_102914)
    
    # Applying the binary operator 'and' (line 874)
    result_and_keyword_102916 = python_operator(stypy.reporting.localization.Localization(__file__, 874, 7), 'and', result_eq_102909, result_eq_102915)
    
    # Testing the type of an if condition (line 874)
    if_condition_102917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 874, 4), result_and_keyword_102916)
    # Assigning a type to the variable 'if_condition_102917' (line 874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 4), 'if_condition_102917', if_condition_102917)
    # SSA begins for if statement (line 874)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 875)
    arr_102918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 875)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'stypy_return_type', arr_102918)
    # SSA join for if statement (line 874)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 881):
    
    # Call to tuple(...): (line 881)
    # Processing the call arguments (line 881)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 881, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 882)
    # Processing the call arguments (line 882)
    # Getting the type of 'arr' (line 882)
    arr_102937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 882)
    shape_102938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 46), arr_102937, 'shape')
    # Processing the call keyword arguments (line 882)
    kwargs_102939 = {}
    # Getting the type of 'enumerate' (line 882)
    enumerate_102936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 882)
    enumerate_call_result_102940 = invoke(stypy.reporting.localization.Localization(__file__, 882, 36), enumerate_102936, *[shape_102938], **kwargs_102939)
    
    comprehension_102941 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), enumerate_call_result_102940)
    # Assigning a type to the variable 'i' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), comprehension_102941))
    # Assigning a type to the variable 'x' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), comprehension_102941))
    
    
    # Getting the type of 'i' (line 881)
    i_102920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 37), 'i', False)
    # Getting the type of 'axis' (line 881)
    axis_102921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 42), 'axis', False)
    # Applying the binary operator '!=' (line 881)
    result_ne_102922 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 37), '!=', i_102920, axis_102921)
    
    # Testing the type of an if expression (line 881)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 22), result_ne_102922)
    # SSA begins for if expression (line 881)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 881)
    # Processing the call arguments (line 881)
    # Getting the type of 'None' (line 881)
    None_102924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 28), 'None', False)
    # Processing the call keyword arguments (line 881)
    kwargs_102925 = {}
    # Getting the type of 'slice' (line 881)
    slice_102923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 881)
    slice_call_result_102926 = invoke(stypy.reporting.localization.Localization(__file__, 881, 22), slice_102923, *[None_102924], **kwargs_102925)
    
    # SSA branch for the else part of an if expression (line 881)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 881)
    # Processing the call arguments (line 881)
    int_102928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 58), 'int')
    
    # Obtaining the type of the subscript
    int_102929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 69), 'int')
    # Getting the type of 'pad_amt' (line 881)
    pad_amt_102930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 61), 'pad_amt', False)
    # Obtaining the member '__getitem__' of a type (line 881)
    getitem___102931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 61), pad_amt_102930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 881)
    subscript_call_result_102932 = invoke(stypy.reporting.localization.Localization(__file__, 881, 61), getitem___102931, int_102929)
    
    # Processing the call keyword arguments (line 881)
    kwargs_102933 = {}
    # Getting the type of 'slice' (line 881)
    slice_102927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 881)
    slice_call_result_102934 = invoke(stypy.reporting.localization.Localization(__file__, 881, 52), slice_102927, *[int_102928, subscript_call_result_102932], **kwargs_102933)
    
    # SSA join for if expression (line 881)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102935 = union_type.UnionType.add(slice_call_result_102926, slice_call_result_102934)
    
    list_102942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 881, 22), list_102942, if_exp_102935)
    # Processing the call keyword arguments (line 881)
    kwargs_102943 = {}
    # Getting the type of 'tuple' (line 881)
    tuple_102919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 881)
    tuple_call_result_102944 = invoke(stypy.reporting.localization.Localization(__file__, 881, 16), tuple_102919, *[list_102942], **kwargs_102943)
    
    # Assigning a type to the variable 'sym_slice' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 4), 'sym_slice', tuple_call_result_102944)
    
    # Assigning a Call to a Name (line 883):
    
    # Call to tuple(...): (line 883)
    # Processing the call arguments (line 883)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 883, 20, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'arr' (line 884)
    arr_102961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 44), 'arr', False)
    # Obtaining the member 'shape' of a type (line 884)
    shape_102962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 44), arr_102961, 'shape')
    # Processing the call keyword arguments (line 884)
    kwargs_102963 = {}
    # Getting the type of 'enumerate' (line 884)
    enumerate_102960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 34), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 884)
    enumerate_call_result_102964 = invoke(stypy.reporting.localization.Localization(__file__, 884, 34), enumerate_102960, *[shape_102962], **kwargs_102963)
    
    comprehension_102965 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 20), enumerate_call_result_102964)
    # Assigning a type to the variable 'i' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 20), comprehension_102965))
    # Assigning a type to the variable 'x' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 20), comprehension_102965))
    
    
    # Getting the type of 'i' (line 883)
    i_102946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 35), 'i', False)
    # Getting the type of 'axis' (line 883)
    axis_102947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 40), 'axis', False)
    # Applying the binary operator '!=' (line 883)
    result_ne_102948 = python_operator(stypy.reporting.localization.Localization(__file__, 883, 35), '!=', i_102946, axis_102947)
    
    # Testing the type of an if expression (line 883)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 883, 20), result_ne_102948)
    # SSA begins for if expression (line 883)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 883)
    # Processing the call arguments (line 883)
    # Getting the type of 'None' (line 883)
    None_102950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 26), 'None', False)
    # Processing the call keyword arguments (line 883)
    kwargs_102951 = {}
    # Getting the type of 'slice' (line 883)
    slice_102949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'slice', False)
    # Calling slice(args, kwargs) (line 883)
    slice_call_result_102952 = invoke(stypy.reporting.localization.Localization(__file__, 883, 20), slice_102949, *[None_102950], **kwargs_102951)
    
    # SSA branch for the else part of an if expression (line 883)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 883)
    # Processing the call arguments (line 883)
    # Getting the type of 'None' (line 883)
    None_102954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 56), 'None', False)
    # Getting the type of 'None' (line 883)
    None_102955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 62), 'None', False)
    int_102956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 68), 'int')
    # Processing the call keyword arguments (line 883)
    kwargs_102957 = {}
    # Getting the type of 'slice' (line 883)
    slice_102953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 50), 'slice', False)
    # Calling slice(args, kwargs) (line 883)
    slice_call_result_102958 = invoke(stypy.reporting.localization.Localization(__file__, 883, 50), slice_102953, *[None_102954, None_102955, int_102956], **kwargs_102957)
    
    # SSA join for if expression (line 883)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102959 = union_type.UnionType.add(slice_call_result_102952, slice_call_result_102958)
    
    list_102966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 20), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 20), list_102966, if_exp_102959)
    # Processing the call keyword arguments (line 883)
    kwargs_102967 = {}
    # Getting the type of 'tuple' (line 883)
    tuple_102945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 14), 'tuple', False)
    # Calling tuple(args, kwargs) (line 883)
    tuple_call_result_102968 = invoke(stypy.reporting.localization.Localization(__file__, 883, 14), tuple_102945, *[list_102966], **kwargs_102967)
    
    # Assigning a type to the variable 'rev_idx' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'rev_idx', tuple_call_result_102968)
    
    # Assigning a Subscript to a Name (line 885):
    
    # Obtaining the type of the subscript
    # Getting the type of 'rev_idx' (line 885)
    rev_idx_102969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 32), 'rev_idx')
    
    # Obtaining the type of the subscript
    # Getting the type of 'sym_slice' (line 885)
    sym_slice_102970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 21), 'sym_slice')
    # Getting the type of 'arr' (line 885)
    arr_102971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 17), 'arr')
    # Obtaining the member '__getitem__' of a type (line 885)
    getitem___102972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 17), arr_102971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 885)
    subscript_call_result_102973 = invoke(stypy.reporting.localization.Localization(__file__, 885, 17), getitem___102972, sym_slice_102970)
    
    # Obtaining the member '__getitem__' of a type (line 885)
    getitem___102974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 17), subscript_call_result_102973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 885)
    subscript_call_result_102975 = invoke(stypy.reporting.localization.Localization(__file__, 885, 17), getitem___102974, rev_idx_102969)
    
    # Assigning a type to the variable 'sym_chunk1' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'sym_chunk1', subscript_call_result_102975)
    
    # Assigning a Call to a Name (line 888):
    
    # Call to tuple(...): (line 888)
    # Processing the call arguments (line 888)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 888, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 889)
    # Processing the call arguments (line 889)
    # Getting the type of 'arr' (line 889)
    arr_102984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 889)
    shape_102985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 50), arr_102984, 'shape')
    # Processing the call keyword arguments (line 889)
    kwargs_102986 = {}
    # Getting the type of 'enumerate' (line 889)
    enumerate_102983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 889)
    enumerate_call_result_102987 = invoke(stypy.reporting.localization.Localization(__file__, 889, 40), enumerate_102983, *[shape_102985], **kwargs_102986)
    
    comprehension_102988 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 26), enumerate_call_result_102987)
    # Assigning a type to the variable 'i' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 26), comprehension_102988))
    # Assigning a type to the variable 'x' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 26), comprehension_102988))
    
    
    # Getting the type of 'i' (line 888)
    i_102977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 31), 'i', False)
    # Getting the type of 'axis' (line 888)
    axis_102978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 36), 'axis', False)
    # Applying the binary operator '!=' (line 888)
    result_ne_102979 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 31), '!=', i_102977, axis_102978)
    
    # Testing the type of an if expression (line 888)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 888, 26), result_ne_102979)
    # SSA begins for if expression (line 888)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 888)
    x_102980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 888)
    module_type_store.open_ssa_branch('if expression else')
    int_102981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 46), 'int')
    # SSA join for if expression (line 888)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_102982 = union_type.UnionType.add(x_102980, int_102981)
    
    list_102989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 888, 26), list_102989, if_exp_102982)
    # Processing the call keyword arguments (line 888)
    kwargs_102990 = {}
    # Getting the type of 'tuple' (line 888)
    tuple_102976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 888)
    tuple_call_result_102991 = invoke(stypy.reporting.localization.Localization(__file__, 888, 20), tuple_102976, *[list_102989], **kwargs_102990)
    
    # Assigning a type to the variable 'pad_singleton' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'pad_singleton', tuple_call_result_102991)
    
    
    
    # Obtaining the type of the subscript
    int_102992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 15), 'int')
    # Getting the type of 'pad_amt' (line 890)
    pad_amt_102993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 890)
    getitem___102994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 7), pad_amt_102993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 890)
    subscript_call_result_102995 = invoke(stypy.reporting.localization.Localization(__file__, 890, 7), getitem___102994, int_102992)
    
    int_102996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 21), 'int')
    # Applying the binary operator '==' (line 890)
    result_eq_102997 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 7), '==', subscript_call_result_102995, int_102996)
    
    # Testing the type of an if condition (line 890)
    if_condition_102998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 890, 4), result_eq_102997)
    # Assigning a type to the variable 'if_condition_102998' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'if_condition_102998', if_condition_102998)
    # SSA begins for if statement (line 890)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 891):
    
    # Call to reshape(...): (line 891)
    # Processing the call arguments (line 891)
    # Getting the type of 'pad_singleton' (line 891)
    pad_singleton_103001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 40), 'pad_singleton', False)
    # Processing the call keyword arguments (line 891)
    kwargs_103002 = {}
    # Getting the type of 'sym_chunk1' (line 891)
    sym_chunk1_102999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 21), 'sym_chunk1', False)
    # Obtaining the member 'reshape' of a type (line 891)
    reshape_103000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 21), sym_chunk1_102999, 'reshape')
    # Calling reshape(args, kwargs) (line 891)
    reshape_call_result_103003 = invoke(stypy.reporting.localization.Localization(__file__, 891, 21), reshape_103000, *[pad_singleton_103001], **kwargs_103002)
    
    # Assigning a type to the variable 'sym_chunk1' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'sym_chunk1', reshape_call_result_103003)
    # SSA join for if statement (line 890)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_103004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 7), 'str', 'odd')
    # Getting the type of 'method' (line 894)
    method_103005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 16), 'method')
    # Applying the binary operator 'in' (line 894)
    result_contains_103006 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 7), 'in', str_103004, method_103005)
    
    
    
    # Obtaining the type of the subscript
    int_103007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 35), 'int')
    # Getting the type of 'pad_amt' (line 894)
    pad_amt_103008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 27), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 894)
    getitem___103009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 27), pad_amt_103008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 894)
    subscript_call_result_103010 = invoke(stypy.reporting.localization.Localization(__file__, 894, 27), getitem___103009, int_103007)
    
    int_103011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 40), 'int')
    # Applying the binary operator '>' (line 894)
    result_gt_103012 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 27), '>', subscript_call_result_103010, int_103011)
    
    # Applying the binary operator 'and' (line 894)
    result_and_keyword_103013 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 7), 'and', result_contains_103006, result_gt_103012)
    
    # Testing the type of an if condition (line 894)
    if_condition_103014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 894, 4), result_and_keyword_103013)
    # Assigning a type to the variable 'if_condition_103014' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'if_condition_103014', if_condition_103014)
    # SSA begins for if statement (line 894)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 895):
    
    # Call to tuple(...): (line 895)
    # Processing the call arguments (line 895)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 895, 28, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 896)
    # Processing the call arguments (line 896)
    # Getting the type of 'arr' (line 896)
    arr_103026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 52), 'arr', False)
    # Obtaining the member 'shape' of a type (line 896)
    shape_103027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 52), arr_103026, 'shape')
    # Processing the call keyword arguments (line 896)
    kwargs_103028 = {}
    # Getting the type of 'enumerate' (line 896)
    enumerate_103025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 42), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 896)
    enumerate_call_result_103029 = invoke(stypy.reporting.localization.Localization(__file__, 896, 42), enumerate_103025, *[shape_103027], **kwargs_103028)
    
    comprehension_103030 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 895, 28), enumerate_call_result_103029)
    # Assigning a type to the variable 'i' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 28), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 895, 28), comprehension_103030))
    # Assigning a type to the variable 'x' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 28), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 895, 28), comprehension_103030))
    
    
    # Getting the type of 'i' (line 895)
    i_103016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 43), 'i', False)
    # Getting the type of 'axis' (line 895)
    axis_103017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 48), 'axis', False)
    # Applying the binary operator '!=' (line 895)
    result_ne_103018 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 43), '!=', i_103016, axis_103017)
    
    # Testing the type of an if expression (line 895)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 895, 28), result_ne_103018)
    # SSA begins for if expression (line 895)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 895)
    # Processing the call arguments (line 895)
    # Getting the type of 'None' (line 895)
    None_103020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 34), 'None', False)
    # Processing the call keyword arguments (line 895)
    kwargs_103021 = {}
    # Getting the type of 'slice' (line 895)
    slice_103019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 28), 'slice', False)
    # Calling slice(args, kwargs) (line 895)
    slice_call_result_103022 = invoke(stypy.reporting.localization.Localization(__file__, 895, 28), slice_103019, *[None_103020], **kwargs_103021)
    
    # SSA branch for the else part of an if expression (line 895)
    module_type_store.open_ssa_branch('if expression else')
    int_103023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 58), 'int')
    # SSA join for if expression (line 895)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103024 = union_type.UnionType.add(slice_call_result_103022, int_103023)
    
    list_103031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 895, 28), list_103031, if_exp_103024)
    # Processing the call keyword arguments (line 895)
    kwargs_103032 = {}
    # Getting the type of 'tuple' (line 895)
    tuple_103015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 22), 'tuple', False)
    # Calling tuple(args, kwargs) (line 895)
    tuple_call_result_103033 = invoke(stypy.reporting.localization.Localization(__file__, 895, 22), tuple_103015, *[list_103031], **kwargs_103032)
    
    # Assigning a type to the variable 'edge_slice1' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 8), 'edge_slice1', tuple_call_result_103033)
    
    # Assigning a Call to a Name (line 897):
    
    # Call to reshape(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'pad_singleton' (line 897)
    pad_singleton_103039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 46), 'pad_singleton', False)
    # Processing the call keyword arguments (line 897)
    kwargs_103040 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice1' (line 897)
    edge_slice1_103034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 25), 'edge_slice1', False)
    # Getting the type of 'arr' (line 897)
    arr_103035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 21), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 897)
    getitem___103036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 21), arr_103035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 897)
    subscript_call_result_103037 = invoke(stypy.reporting.localization.Localization(__file__, 897, 21), getitem___103036, edge_slice1_103034)
    
    # Obtaining the member 'reshape' of a type (line 897)
    reshape_103038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 21), subscript_call_result_103037, 'reshape')
    # Calling reshape(args, kwargs) (line 897)
    reshape_call_result_103041 = invoke(stypy.reporting.localization.Localization(__file__, 897, 21), reshape_103038, *[pad_singleton_103039], **kwargs_103040)
    
    # Assigning a type to the variable 'edge_chunk' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'edge_chunk', reshape_call_result_103041)
    
    # Assigning a BinOp to a Name (line 898):
    int_103042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 21), 'int')
    # Getting the type of 'edge_chunk' (line 898)
    edge_chunk_103043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 25), 'edge_chunk')
    # Applying the binary operator '*' (line 898)
    result_mul_103044 = python_operator(stypy.reporting.localization.Localization(__file__, 898, 21), '*', int_103042, edge_chunk_103043)
    
    # Getting the type of 'sym_chunk1' (line 898)
    sym_chunk1_103045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 38), 'sym_chunk1')
    # Applying the binary operator '-' (line 898)
    result_sub_103046 = python_operator(stypy.reporting.localization.Localization(__file__, 898, 21), '-', result_mul_103044, sym_chunk1_103045)
    
    # Assigning a type to the variable 'sym_chunk1' (line 898)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 898, 8), 'sym_chunk1', result_sub_103046)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 899, 8), module_type_store, 'edge_chunk')
    # SSA join for if statement (line 894)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 905):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 905)
    axis_103047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 22), 'axis')
    # Getting the type of 'arr' (line 905)
    arr_103048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'arr')
    # Obtaining the member 'shape' of a type (line 905)
    shape_103049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 12), arr_103048, 'shape')
    # Obtaining the member '__getitem__' of a type (line 905)
    getitem___103050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 12), shape_103049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 905)
    subscript_call_result_103051 = invoke(stypy.reporting.localization.Localization(__file__, 905, 12), getitem___103050, axis_103047)
    
    
    # Obtaining the type of the subscript
    int_103052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 38), 'int')
    # Getting the type of 'pad_amt' (line 905)
    pad_amt_103053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 30), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 905)
    getitem___103054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 30), pad_amt_103053, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 905)
    subscript_call_result_103055 = invoke(stypy.reporting.localization.Localization(__file__, 905, 30), getitem___103054, int_103052)
    
    # Applying the binary operator '-' (line 905)
    result_sub_103056 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 12), '-', subscript_call_result_103051, subscript_call_result_103055)
    
    # Assigning a type to the variable 'start' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'start', result_sub_103056)
    
    # Assigning a Subscript to a Name (line 906):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 906)
    axis_103057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 20), 'axis')
    # Getting the type of 'arr' (line 906)
    arr_103058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 906)
    shape_103059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 10), arr_103058, 'shape')
    # Obtaining the member '__getitem__' of a type (line 906)
    getitem___103060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 10), shape_103059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 906)
    subscript_call_result_103061 = invoke(stypy.reporting.localization.Localization(__file__, 906, 10), getitem___103060, axis_103057)
    
    # Assigning a type to the variable 'end' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 4), 'end', subscript_call_result_103061)
    
    # Assigning a Call to a Name (line 907):
    
    # Call to tuple(...): (line 907)
    # Processing the call arguments (line 907)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 907, 22, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 908)
    # Processing the call arguments (line 908)
    # Getting the type of 'arr' (line 908)
    arr_103077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 46), 'arr', False)
    # Obtaining the member 'shape' of a type (line 908)
    shape_103078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 46), arr_103077, 'shape')
    # Processing the call keyword arguments (line 908)
    kwargs_103079 = {}
    # Getting the type of 'enumerate' (line 908)
    enumerate_103076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 36), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 908)
    enumerate_call_result_103080 = invoke(stypy.reporting.localization.Localization(__file__, 908, 36), enumerate_103076, *[shape_103078], **kwargs_103079)
    
    comprehension_103081 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 22), enumerate_call_result_103080)
    # Assigning a type to the variable 'i' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 22), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 22), comprehension_103081))
    # Assigning a type to the variable 'x' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 22), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 22), comprehension_103081))
    
    
    # Getting the type of 'i' (line 907)
    i_103063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 37), 'i', False)
    # Getting the type of 'axis' (line 907)
    axis_103064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 42), 'axis', False)
    # Applying the binary operator '!=' (line 907)
    result_ne_103065 = python_operator(stypy.reporting.localization.Localization(__file__, 907, 37), '!=', i_103063, axis_103064)
    
    # Testing the type of an if expression (line 907)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 907, 22), result_ne_103065)
    # SSA begins for if expression (line 907)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'None' (line 907)
    None_103067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 28), 'None', False)
    # Processing the call keyword arguments (line 907)
    kwargs_103068 = {}
    # Getting the type of 'slice' (line 907)
    slice_103066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 907)
    slice_call_result_103069 = invoke(stypy.reporting.localization.Localization(__file__, 907, 22), slice_103066, *[None_103067], **kwargs_103068)
    
    # SSA branch for the else part of an if expression (line 907)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'start' (line 907)
    start_103071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 58), 'start', False)
    # Getting the type of 'end' (line 907)
    end_103072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 65), 'end', False)
    # Processing the call keyword arguments (line 907)
    kwargs_103073 = {}
    # Getting the type of 'slice' (line 907)
    slice_103070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 52), 'slice', False)
    # Calling slice(args, kwargs) (line 907)
    slice_call_result_103074 = invoke(stypy.reporting.localization.Localization(__file__, 907, 52), slice_103070, *[start_103071, end_103072], **kwargs_103073)
    
    # SSA join for if expression (line 907)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103075 = union_type.UnionType.add(slice_call_result_103069, slice_call_result_103074)
    
    list_103082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 907, 22), list_103082, if_exp_103075)
    # Processing the call keyword arguments (line 907)
    kwargs_103083 = {}
    # Getting the type of 'tuple' (line 907)
    tuple_103062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 907)
    tuple_call_result_103084 = invoke(stypy.reporting.localization.Localization(__file__, 907, 16), tuple_103062, *[list_103082], **kwargs_103083)
    
    # Assigning a type to the variable 'sym_slice' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 4), 'sym_slice', tuple_call_result_103084)
    
    # Assigning a Subscript to a Name (line 909):
    
    # Obtaining the type of the subscript
    # Getting the type of 'rev_idx' (line 909)
    rev_idx_103085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 32), 'rev_idx')
    
    # Obtaining the type of the subscript
    # Getting the type of 'sym_slice' (line 909)
    sym_slice_103086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 21), 'sym_slice')
    # Getting the type of 'arr' (line 909)
    arr_103087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 17), 'arr')
    # Obtaining the member '__getitem__' of a type (line 909)
    getitem___103088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 17), arr_103087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 909)
    subscript_call_result_103089 = invoke(stypy.reporting.localization.Localization(__file__, 909, 17), getitem___103088, sym_slice_103086)
    
    # Obtaining the member '__getitem__' of a type (line 909)
    getitem___103090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 17), subscript_call_result_103089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 909)
    subscript_call_result_103091 = invoke(stypy.reporting.localization.Localization(__file__, 909, 17), getitem___103090, rev_idx_103085)
    
    # Assigning a type to the variable 'sym_chunk2' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 4), 'sym_chunk2', subscript_call_result_103091)
    
    
    
    # Obtaining the type of the subscript
    int_103092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 15), 'int')
    # Getting the type of 'pad_amt' (line 911)
    pad_amt_103093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 911)
    getitem___103094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 7), pad_amt_103093, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 911)
    subscript_call_result_103095 = invoke(stypy.reporting.localization.Localization(__file__, 911, 7), getitem___103094, int_103092)
    
    int_103096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 21), 'int')
    # Applying the binary operator '==' (line 911)
    result_eq_103097 = python_operator(stypy.reporting.localization.Localization(__file__, 911, 7), '==', subscript_call_result_103095, int_103096)
    
    # Testing the type of an if condition (line 911)
    if_condition_103098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 911, 4), result_eq_103097)
    # Assigning a type to the variable 'if_condition_103098' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 4), 'if_condition_103098', if_condition_103098)
    # SSA begins for if statement (line 911)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 912):
    
    # Call to reshape(...): (line 912)
    # Processing the call arguments (line 912)
    # Getting the type of 'pad_singleton' (line 912)
    pad_singleton_103101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 40), 'pad_singleton', False)
    # Processing the call keyword arguments (line 912)
    kwargs_103102 = {}
    # Getting the type of 'sym_chunk2' (line 912)
    sym_chunk2_103099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 21), 'sym_chunk2', False)
    # Obtaining the member 'reshape' of a type (line 912)
    reshape_103100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 21), sym_chunk2_103099, 'reshape')
    # Calling reshape(args, kwargs) (line 912)
    reshape_call_result_103103 = invoke(stypy.reporting.localization.Localization(__file__, 912, 21), reshape_103100, *[pad_singleton_103101], **kwargs_103102)
    
    # Assigning a type to the variable 'sym_chunk2' (line 912)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 8), 'sym_chunk2', reshape_call_result_103103)
    # SSA join for if statement (line 911)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_103104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 7), 'str', 'odd')
    # Getting the type of 'method' (line 914)
    method_103105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 16), 'method')
    # Applying the binary operator 'in' (line 914)
    result_contains_103106 = python_operator(stypy.reporting.localization.Localization(__file__, 914, 7), 'in', str_103104, method_103105)
    
    # Testing the type of an if condition (line 914)
    if_condition_103107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 914, 4), result_contains_103106)
    # Assigning a type to the variable 'if_condition_103107' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'if_condition_103107', if_condition_103107)
    # SSA begins for if statement (line 914)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 915):
    
    # Call to tuple(...): (line 915)
    # Processing the call arguments (line 915)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 915, 28, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 916)
    # Processing the call arguments (line 916)
    # Getting the type of 'arr' (line 916)
    arr_103119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 52), 'arr', False)
    # Obtaining the member 'shape' of a type (line 916)
    shape_103120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 52), arr_103119, 'shape')
    # Processing the call keyword arguments (line 916)
    kwargs_103121 = {}
    # Getting the type of 'enumerate' (line 916)
    enumerate_103118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 42), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 916)
    enumerate_call_result_103122 = invoke(stypy.reporting.localization.Localization(__file__, 916, 42), enumerate_103118, *[shape_103120], **kwargs_103121)
    
    comprehension_103123 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 28), enumerate_call_result_103122)
    # Assigning a type to the variable 'i' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 28), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 28), comprehension_103123))
    # Assigning a type to the variable 'x' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 28), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 28), comprehension_103123))
    
    
    # Getting the type of 'i' (line 915)
    i_103109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 43), 'i', False)
    # Getting the type of 'axis' (line 915)
    axis_103110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 48), 'axis', False)
    # Applying the binary operator '!=' (line 915)
    result_ne_103111 = python_operator(stypy.reporting.localization.Localization(__file__, 915, 43), '!=', i_103109, axis_103110)
    
    # Testing the type of an if expression (line 915)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 915, 28), result_ne_103111)
    # SSA begins for if expression (line 915)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'None' (line 915)
    None_103113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 34), 'None', False)
    # Processing the call keyword arguments (line 915)
    kwargs_103114 = {}
    # Getting the type of 'slice' (line 915)
    slice_103112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 28), 'slice', False)
    # Calling slice(args, kwargs) (line 915)
    slice_call_result_103115 = invoke(stypy.reporting.localization.Localization(__file__, 915, 28), slice_103112, *[None_103113], **kwargs_103114)
    
    # SSA branch for the else part of an if expression (line 915)
    module_type_store.open_ssa_branch('if expression else')
    int_103116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 58), 'int')
    # SSA join for if expression (line 915)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103117 = union_type.UnionType.add(slice_call_result_103115, int_103116)
    
    list_103124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 915, 28), list_103124, if_exp_103117)
    # Processing the call keyword arguments (line 915)
    kwargs_103125 = {}
    # Getting the type of 'tuple' (line 915)
    tuple_103108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 22), 'tuple', False)
    # Calling tuple(args, kwargs) (line 915)
    tuple_call_result_103126 = invoke(stypy.reporting.localization.Localization(__file__, 915, 22), tuple_103108, *[list_103124], **kwargs_103125)
    
    # Assigning a type to the variable 'edge_slice2' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 8), 'edge_slice2', tuple_call_result_103126)
    
    # Assigning a Call to a Name (line 917):
    
    # Call to reshape(...): (line 917)
    # Processing the call arguments (line 917)
    # Getting the type of 'pad_singleton' (line 917)
    pad_singleton_103132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 46), 'pad_singleton', False)
    # Processing the call keyword arguments (line 917)
    kwargs_103133 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'edge_slice2' (line 917)
    edge_slice2_103127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 25), 'edge_slice2', False)
    # Getting the type of 'arr' (line 917)
    arr_103128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 21), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 917)
    getitem___103129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 21), arr_103128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 917)
    subscript_call_result_103130 = invoke(stypy.reporting.localization.Localization(__file__, 917, 21), getitem___103129, edge_slice2_103127)
    
    # Obtaining the member 'reshape' of a type (line 917)
    reshape_103131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 21), subscript_call_result_103130, 'reshape')
    # Calling reshape(args, kwargs) (line 917)
    reshape_call_result_103134 = invoke(stypy.reporting.localization.Localization(__file__, 917, 21), reshape_103131, *[pad_singleton_103132], **kwargs_103133)
    
    # Assigning a type to the variable 'edge_chunk' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'edge_chunk', reshape_call_result_103134)
    
    # Assigning a BinOp to a Name (line 918):
    int_103135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 21), 'int')
    # Getting the type of 'edge_chunk' (line 918)
    edge_chunk_103136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 25), 'edge_chunk')
    # Applying the binary operator '*' (line 918)
    result_mul_103137 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 21), '*', int_103135, edge_chunk_103136)
    
    # Getting the type of 'sym_chunk2' (line 918)
    sym_chunk2_103138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 38), 'sym_chunk2')
    # Applying the binary operator '-' (line 918)
    result_sub_103139 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 21), '-', result_mul_103137, sym_chunk2_103138)
    
    # Assigning a type to the variable 'sym_chunk2' (line 918)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'sym_chunk2', result_sub_103139)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 919, 8), module_type_store, 'edge_chunk')
    # SSA join for if statement (line 914)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to concatenate(...): (line 922)
    # Processing the call arguments (line 922)
    
    # Obtaining an instance of the builtin type 'tuple' (line 922)
    tuple_103142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 922)
    # Adding element type (line 922)
    # Getting the type of 'sym_chunk1' (line 922)
    sym_chunk1_103143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 27), 'sym_chunk1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 27), tuple_103142, sym_chunk1_103143)
    # Adding element type (line 922)
    # Getting the type of 'arr' (line 922)
    arr_103144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 39), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 27), tuple_103142, arr_103144)
    # Adding element type (line 922)
    # Getting the type of 'sym_chunk2' (line 922)
    sym_chunk2_103145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 44), 'sym_chunk2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 27), tuple_103142, sym_chunk2_103145)
    
    # Processing the call keyword arguments (line 922)
    # Getting the type of 'axis' (line 922)
    axis_103146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 62), 'axis', False)
    keyword_103147 = axis_103146
    kwargs_103148 = {'axis': keyword_103147}
    # Getting the type of 'np' (line 922)
    np_103140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 922)
    concatenate_103141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 11), np_103140, 'concatenate')
    # Calling concatenate(args, kwargs) (line 922)
    concatenate_call_result_103149 = invoke(stypy.reporting.localization.Localization(__file__, 922, 11), concatenate_103141, *[tuple_103142], **kwargs_103148)
    
    # Assigning a type to the variable 'stypy_return_type' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'stypy_return_type', concatenate_call_result_103149)
    
    # ################# End of '_pad_sym(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pad_sym' in the type store
    # Getting the type of 'stypy_return_type' (line 841)
    stypy_return_type_103150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_103150)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pad_sym'
    return stypy_return_type_103150

# Assigning a type to the variable '_pad_sym' (line 841)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 0), '_pad_sym', _pad_sym)

@norecursion
def _pad_wrap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_103151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 925, 33), 'int')
    defaults = [int_103151]
    # Create a new context for function '_pad_wrap'
    module_type_store = module_type_store.open_function_context('_pad_wrap', 925, 0, False)
    
    # Passed parameters checking function
    _pad_wrap.stypy_localization = localization
    _pad_wrap.stypy_type_of_self = None
    _pad_wrap.stypy_type_store = module_type_store
    _pad_wrap.stypy_function_name = '_pad_wrap'
    _pad_wrap.stypy_param_names_list = ['arr', 'pad_amt', 'axis']
    _pad_wrap.stypy_varargs_param_name = None
    _pad_wrap.stypy_kwargs_param_name = None
    _pad_wrap.stypy_call_defaults = defaults
    _pad_wrap.stypy_call_varargs = varargs
    _pad_wrap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_pad_wrap', ['arr', 'pad_amt', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_pad_wrap', localization, ['arr', 'pad_amt', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_pad_wrap(...)' code ##################

    str_103152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, (-1)), 'str', "\n    Pad `axis` of `arr` via wrapping.\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array of arbitrary shape.\n    pad_amt : tuple of ints, length 2\n        Padding to (prepend, append) along `axis`.\n    axis : int\n        Axis along which to pad `arr`.\n\n    Returns\n    -------\n    padarr : ndarray\n        Output array, with `pad_amt[0]` values prepended and `pad_amt[1]`\n        values appended along `axis`. Both regions are padded wrapped values\n        from the opposite end of `axis`.\n\n    Notes\n    -----\n    This method of padding is also known as 'tile' or 'tiling'.\n\n    The modes 'reflect', 'symmetric', and 'wrap' must be padded with a\n    single function, lest the indexing tricks in non-integer multiples of the\n    original shape would violate repetition in the final iteration.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_103153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 15), 'int')
    # Getting the type of 'pad_amt' (line 955)
    pad_amt_103154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___103155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 7), pad_amt_103154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_103156 = invoke(stypy.reporting.localization.Localization(__file__, 955, 7), getitem___103155, int_103153)
    
    int_103157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 21), 'int')
    # Applying the binary operator '==' (line 955)
    result_eq_103158 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 7), '==', subscript_call_result_103156, int_103157)
    
    
    
    # Obtaining the type of the subscript
    int_103159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 35), 'int')
    # Getting the type of 'pad_amt' (line 955)
    pad_amt_103160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 27), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 955)
    getitem___103161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 27), pad_amt_103160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 955)
    subscript_call_result_103162 = invoke(stypy.reporting.localization.Localization(__file__, 955, 27), getitem___103161, int_103159)
    
    int_103163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 41), 'int')
    # Applying the binary operator '==' (line 955)
    result_eq_103164 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 27), '==', subscript_call_result_103162, int_103163)
    
    # Applying the binary operator 'and' (line 955)
    result_and_keyword_103165 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 7), 'and', result_eq_103158, result_eq_103164)
    
    # Testing the type of an if condition (line 955)
    if_condition_103166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 955, 4), result_and_keyword_103165)
    # Assigning a type to the variable 'if_condition_103166' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'if_condition_103166', if_condition_103166)
    # SSA begins for if statement (line 955)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'arr' (line 956)
    arr_103167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 15), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 8), 'stypy_return_type', arr_103167)
    # SSA join for if statement (line 955)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 962):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 962)
    axis_103168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 22), 'axis')
    # Getting the type of 'arr' (line 962)
    arr_103169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'arr')
    # Obtaining the member 'shape' of a type (line 962)
    shape_103170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 12), arr_103169, 'shape')
    # Obtaining the member '__getitem__' of a type (line 962)
    getitem___103171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 12), shape_103170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 962)
    subscript_call_result_103172 = invoke(stypy.reporting.localization.Localization(__file__, 962, 12), getitem___103171, axis_103168)
    
    
    # Obtaining the type of the subscript
    int_103173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 38), 'int')
    # Getting the type of 'pad_amt' (line 962)
    pad_amt_103174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 30), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 962)
    getitem___103175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 30), pad_amt_103174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 962)
    subscript_call_result_103176 = invoke(stypy.reporting.localization.Localization(__file__, 962, 30), getitem___103175, int_103173)
    
    # Applying the binary operator '-' (line 962)
    result_sub_103177 = python_operator(stypy.reporting.localization.Localization(__file__, 962, 12), '-', subscript_call_result_103172, subscript_call_result_103176)
    
    # Assigning a type to the variable 'start' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'start', result_sub_103177)
    
    # Assigning a Subscript to a Name (line 963):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 963)
    axis_103178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'axis')
    # Getting the type of 'arr' (line 963)
    arr_103179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 10), 'arr')
    # Obtaining the member 'shape' of a type (line 963)
    shape_103180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 10), arr_103179, 'shape')
    # Obtaining the member '__getitem__' of a type (line 963)
    getitem___103181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 10), shape_103180, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 963)
    subscript_call_result_103182 = invoke(stypy.reporting.localization.Localization(__file__, 963, 10), getitem___103181, axis_103178)
    
    # Assigning a type to the variable 'end' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'end', subscript_call_result_103182)
    
    # Assigning a Call to a Name (line 964):
    
    # Call to tuple(...): (line 964)
    # Processing the call arguments (line 964)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 964, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'arr' (line 965)
    arr_103198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 965)
    shape_103199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 47), arr_103198, 'shape')
    # Processing the call keyword arguments (line 965)
    kwargs_103200 = {}
    # Getting the type of 'enumerate' (line 965)
    enumerate_103197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 965)
    enumerate_call_result_103201 = invoke(stypy.reporting.localization.Localization(__file__, 965, 37), enumerate_103197, *[shape_103199], **kwargs_103200)
    
    comprehension_103202 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 23), enumerate_call_result_103201)
    # Assigning a type to the variable 'i' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 23), comprehension_103202))
    # Assigning a type to the variable 'x' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 23), comprehension_103202))
    
    
    # Getting the type of 'i' (line 964)
    i_103184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 38), 'i', False)
    # Getting the type of 'axis' (line 964)
    axis_103185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 43), 'axis', False)
    # Applying the binary operator '!=' (line 964)
    result_ne_103186 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 38), '!=', i_103184, axis_103185)
    
    # Testing the type of an if expression (line 964)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 964, 23), result_ne_103186)
    # SSA begins for if expression (line 964)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 964)
    # Processing the call arguments (line 964)
    # Getting the type of 'None' (line 964)
    None_103188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 29), 'None', False)
    # Processing the call keyword arguments (line 964)
    kwargs_103189 = {}
    # Getting the type of 'slice' (line 964)
    slice_103187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 964)
    slice_call_result_103190 = invoke(stypy.reporting.localization.Localization(__file__, 964, 23), slice_103187, *[None_103188], **kwargs_103189)
    
    # SSA branch for the else part of an if expression (line 964)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 964)
    # Processing the call arguments (line 964)
    # Getting the type of 'start' (line 964)
    start_103192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 59), 'start', False)
    # Getting the type of 'end' (line 964)
    end_103193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 66), 'end', False)
    # Processing the call keyword arguments (line 964)
    kwargs_103194 = {}
    # Getting the type of 'slice' (line 964)
    slice_103191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 53), 'slice', False)
    # Calling slice(args, kwargs) (line 964)
    slice_call_result_103195 = invoke(stypy.reporting.localization.Localization(__file__, 964, 53), slice_103191, *[start_103192, end_103193], **kwargs_103194)
    
    # SSA join for if expression (line 964)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103196 = union_type.UnionType.add(slice_call_result_103190, slice_call_result_103195)
    
    list_103203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 23), list_103203, if_exp_103196)
    # Processing the call keyword arguments (line 964)
    kwargs_103204 = {}
    # Getting the type of 'tuple' (line 964)
    tuple_103183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 964)
    tuple_call_result_103205 = invoke(stypy.reporting.localization.Localization(__file__, 964, 17), tuple_103183, *[list_103203], **kwargs_103204)
    
    # Assigning a type to the variable 'wrap_slice' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'wrap_slice', tuple_call_result_103205)
    
    # Assigning a Subscript to a Name (line 966):
    
    # Obtaining the type of the subscript
    # Getting the type of 'wrap_slice' (line 966)
    wrap_slice_103206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 22), 'wrap_slice')
    # Getting the type of 'arr' (line 966)
    arr_103207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 18), 'arr')
    # Obtaining the member '__getitem__' of a type (line 966)
    getitem___103208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 966, 18), arr_103207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 966)
    subscript_call_result_103209 = invoke(stypy.reporting.localization.Localization(__file__, 966, 18), getitem___103208, wrap_slice_103206)
    
    # Assigning a type to the variable 'wrap_chunk1' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 4), 'wrap_chunk1', subscript_call_result_103209)
    
    # Assigning a Call to a Name (line 969):
    
    # Call to tuple(...): (line 969)
    # Processing the call arguments (line 969)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 969, 26, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 970)
    # Processing the call arguments (line 970)
    # Getting the type of 'arr' (line 970)
    arr_103218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 50), 'arr', False)
    # Obtaining the member 'shape' of a type (line 970)
    shape_103219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 50), arr_103218, 'shape')
    # Processing the call keyword arguments (line 970)
    kwargs_103220 = {}
    # Getting the type of 'enumerate' (line 970)
    enumerate_103217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 40), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 970)
    enumerate_call_result_103221 = invoke(stypy.reporting.localization.Localization(__file__, 970, 40), enumerate_103217, *[shape_103219], **kwargs_103220)
    
    comprehension_103222 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 969, 26), enumerate_call_result_103221)
    # Assigning a type to the variable 'i' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 26), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 969, 26), comprehension_103222))
    # Assigning a type to the variable 'x' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 26), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 969, 26), comprehension_103222))
    
    
    # Getting the type of 'i' (line 969)
    i_103211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 31), 'i', False)
    # Getting the type of 'axis' (line 969)
    axis_103212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 36), 'axis', False)
    # Applying the binary operator '!=' (line 969)
    result_ne_103213 = python_operator(stypy.reporting.localization.Localization(__file__, 969, 31), '!=', i_103211, axis_103212)
    
    # Testing the type of an if expression (line 969)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 969, 26), result_ne_103213)
    # SSA begins for if expression (line 969)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'x' (line 969)
    x_103214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 26), 'x', False)
    # SSA branch for the else part of an if expression (line 969)
    module_type_store.open_ssa_branch('if expression else')
    int_103215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 46), 'int')
    # SSA join for if expression (line 969)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103216 = union_type.UnionType.add(x_103214, int_103215)
    
    list_103223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 969, 26), list_103223, if_exp_103216)
    # Processing the call keyword arguments (line 969)
    kwargs_103224 = {}
    # Getting the type of 'tuple' (line 969)
    tuple_103210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 20), 'tuple', False)
    # Calling tuple(args, kwargs) (line 969)
    tuple_call_result_103225 = invoke(stypy.reporting.localization.Localization(__file__, 969, 20), tuple_103210, *[list_103223], **kwargs_103224)
    
    # Assigning a type to the variable 'pad_singleton' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'pad_singleton', tuple_call_result_103225)
    
    
    
    # Obtaining the type of the subscript
    int_103226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 15), 'int')
    # Getting the type of 'pad_amt' (line 971)
    pad_amt_103227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 971)
    getitem___103228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 7), pad_amt_103227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 971)
    subscript_call_result_103229 = invoke(stypy.reporting.localization.Localization(__file__, 971, 7), getitem___103228, int_103226)
    
    int_103230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, 21), 'int')
    # Applying the binary operator '==' (line 971)
    result_eq_103231 = python_operator(stypy.reporting.localization.Localization(__file__, 971, 7), '==', subscript_call_result_103229, int_103230)
    
    # Testing the type of an if condition (line 971)
    if_condition_103232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 4), result_eq_103231)
    # Assigning a type to the variable 'if_condition_103232' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 4), 'if_condition_103232', if_condition_103232)
    # SSA begins for if statement (line 971)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 972):
    
    # Call to reshape(...): (line 972)
    # Processing the call arguments (line 972)
    # Getting the type of 'pad_singleton' (line 972)
    pad_singleton_103235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 42), 'pad_singleton', False)
    # Processing the call keyword arguments (line 972)
    kwargs_103236 = {}
    # Getting the type of 'wrap_chunk1' (line 972)
    wrap_chunk1_103233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 22), 'wrap_chunk1', False)
    # Obtaining the member 'reshape' of a type (line 972)
    reshape_103234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 22), wrap_chunk1_103233, 'reshape')
    # Calling reshape(args, kwargs) (line 972)
    reshape_call_result_103237 = invoke(stypy.reporting.localization.Localization(__file__, 972, 22), reshape_103234, *[pad_singleton_103235], **kwargs_103236)
    
    # Assigning a type to the variable 'wrap_chunk1' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 8), 'wrap_chunk1', reshape_call_result_103237)
    # SSA join for if statement (line 971)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 978):
    
    # Call to tuple(...): (line 978)
    # Processing the call arguments (line 978)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 978, 23, True)
    # Calculating comprehension expression
    
    # Call to enumerate(...): (line 979)
    # Processing the call arguments (line 979)
    # Getting the type of 'arr' (line 979)
    arr_103256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 47), 'arr', False)
    # Obtaining the member 'shape' of a type (line 979)
    shape_103257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 47), arr_103256, 'shape')
    # Processing the call keyword arguments (line 979)
    kwargs_103258 = {}
    # Getting the type of 'enumerate' (line 979)
    enumerate_103255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 37), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 979)
    enumerate_call_result_103259 = invoke(stypy.reporting.localization.Localization(__file__, 979, 37), enumerate_103255, *[shape_103257], **kwargs_103258)
    
    comprehension_103260 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 23), enumerate_call_result_103259)
    # Assigning a type to the variable 'i' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 23), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 23), comprehension_103260))
    # Assigning a type to the variable 'x' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 23), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 23), comprehension_103260))
    
    
    # Getting the type of 'i' (line 978)
    i_103239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 38), 'i', False)
    # Getting the type of 'axis' (line 978)
    axis_103240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 43), 'axis', False)
    # Applying the binary operator '!=' (line 978)
    result_ne_103241 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 38), '!=', i_103239, axis_103240)
    
    # Testing the type of an if expression (line 978)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 978, 23), result_ne_103241)
    # SSA begins for if expression (line 978)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to slice(...): (line 978)
    # Processing the call arguments (line 978)
    # Getting the type of 'None' (line 978)
    None_103243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 29), 'None', False)
    # Processing the call keyword arguments (line 978)
    kwargs_103244 = {}
    # Getting the type of 'slice' (line 978)
    slice_103242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 23), 'slice', False)
    # Calling slice(args, kwargs) (line 978)
    slice_call_result_103245 = invoke(stypy.reporting.localization.Localization(__file__, 978, 23), slice_103242, *[None_103243], **kwargs_103244)
    
    # SSA branch for the else part of an if expression (line 978)
    module_type_store.open_ssa_branch('if expression else')
    
    # Call to slice(...): (line 978)
    # Processing the call arguments (line 978)
    int_103247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 59), 'int')
    
    # Obtaining the type of the subscript
    int_103248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 70), 'int')
    # Getting the type of 'pad_amt' (line 978)
    pad_amt_103249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 62), 'pad_amt', False)
    # Obtaining the member '__getitem__' of a type (line 978)
    getitem___103250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 62), pad_amt_103249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 978)
    subscript_call_result_103251 = invoke(stypy.reporting.localization.Localization(__file__, 978, 62), getitem___103250, int_103248)
    
    # Processing the call keyword arguments (line 978)
    kwargs_103252 = {}
    # Getting the type of 'slice' (line 978)
    slice_103246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 53), 'slice', False)
    # Calling slice(args, kwargs) (line 978)
    slice_call_result_103253 = invoke(stypy.reporting.localization.Localization(__file__, 978, 53), slice_103246, *[int_103247, subscript_call_result_103251], **kwargs_103252)
    
    # SSA join for if expression (line 978)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103254 = union_type.UnionType.add(slice_call_result_103245, slice_call_result_103253)
    
    list_103261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 23), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 978, 23), list_103261, if_exp_103254)
    # Processing the call keyword arguments (line 978)
    kwargs_103262 = {}
    # Getting the type of 'tuple' (line 978)
    tuple_103238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 978)
    tuple_call_result_103263 = invoke(stypy.reporting.localization.Localization(__file__, 978, 17), tuple_103238, *[list_103261], **kwargs_103262)
    
    # Assigning a type to the variable 'wrap_slice' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 4), 'wrap_slice', tuple_call_result_103263)
    
    # Assigning a Subscript to a Name (line 980):
    
    # Obtaining the type of the subscript
    # Getting the type of 'wrap_slice' (line 980)
    wrap_slice_103264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 22), 'wrap_slice')
    # Getting the type of 'arr' (line 980)
    arr_103265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 18), 'arr')
    # Obtaining the member '__getitem__' of a type (line 980)
    getitem___103266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 18), arr_103265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 980)
    subscript_call_result_103267 = invoke(stypy.reporting.localization.Localization(__file__, 980, 18), getitem___103266, wrap_slice_103264)
    
    # Assigning a type to the variable 'wrap_chunk2' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), 'wrap_chunk2', subscript_call_result_103267)
    
    
    
    # Obtaining the type of the subscript
    int_103268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 15), 'int')
    # Getting the type of 'pad_amt' (line 982)
    pad_amt_103269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 7), 'pad_amt')
    # Obtaining the member '__getitem__' of a type (line 982)
    getitem___103270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 982, 7), pad_amt_103269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 982)
    subscript_call_result_103271 = invoke(stypy.reporting.localization.Localization(__file__, 982, 7), getitem___103270, int_103268)
    
    int_103272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 982, 21), 'int')
    # Applying the binary operator '==' (line 982)
    result_eq_103273 = python_operator(stypy.reporting.localization.Localization(__file__, 982, 7), '==', subscript_call_result_103271, int_103272)
    
    # Testing the type of an if condition (line 982)
    if_condition_103274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 982, 4), result_eq_103273)
    # Assigning a type to the variable 'if_condition_103274' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'if_condition_103274', if_condition_103274)
    # SSA begins for if statement (line 982)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 983):
    
    # Call to reshape(...): (line 983)
    # Processing the call arguments (line 983)
    # Getting the type of 'pad_singleton' (line 983)
    pad_singleton_103277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 42), 'pad_singleton', False)
    # Processing the call keyword arguments (line 983)
    kwargs_103278 = {}
    # Getting the type of 'wrap_chunk2' (line 983)
    wrap_chunk2_103275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 22), 'wrap_chunk2', False)
    # Obtaining the member 'reshape' of a type (line 983)
    reshape_103276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 22), wrap_chunk2_103275, 'reshape')
    # Calling reshape(args, kwargs) (line 983)
    reshape_call_result_103279 = invoke(stypy.reporting.localization.Localization(__file__, 983, 22), reshape_103276, *[pad_singleton_103277], **kwargs_103278)
    
    # Assigning a type to the variable 'wrap_chunk2' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 8), 'wrap_chunk2', reshape_call_result_103279)
    # SSA join for if statement (line 982)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to concatenate(...): (line 986)
    # Processing the call arguments (line 986)
    
    # Obtaining an instance of the builtin type 'tuple' (line 986)
    tuple_103282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 986)
    # Adding element type (line 986)
    # Getting the type of 'wrap_chunk1' (line 986)
    wrap_chunk1_103283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 27), 'wrap_chunk1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 27), tuple_103282, wrap_chunk1_103283)
    # Adding element type (line 986)
    # Getting the type of 'arr' (line 986)
    arr_103284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 40), 'arr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 27), tuple_103282, arr_103284)
    # Adding element type (line 986)
    # Getting the type of 'wrap_chunk2' (line 986)
    wrap_chunk2_103285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 45), 'wrap_chunk2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 986, 27), tuple_103282, wrap_chunk2_103285)
    
    # Processing the call keyword arguments (line 986)
    # Getting the type of 'axis' (line 986)
    axis_103286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 64), 'axis', False)
    keyword_103287 = axis_103286
    kwargs_103288 = {'axis': keyword_103287}
    # Getting the type of 'np' (line 986)
    np_103280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 986)
    concatenate_103281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 11), np_103280, 'concatenate')
    # Calling concatenate(args, kwargs) (line 986)
    concatenate_call_result_103289 = invoke(stypy.reporting.localization.Localization(__file__, 986, 11), concatenate_103281, *[tuple_103282], **kwargs_103288)
    
    # Assigning a type to the variable 'stypy_return_type' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'stypy_return_type', concatenate_call_result_103289)
    
    # ################# End of '_pad_wrap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_pad_wrap' in the type store
    # Getting the type of 'stypy_return_type' (line 925)
    stypy_return_type_103290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_103290)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_pad_wrap'
    return stypy_return_type_103290

# Assigning a type to the variable '_pad_wrap' (line 925)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 0), '_pad_wrap', _pad_wrap)

@norecursion
def _normalize_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 989)
    True_103291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 49), 'True')
    defaults = [True_103291]
    # Create a new context for function '_normalize_shape'
    module_type_store = module_type_store.open_function_context('_normalize_shape', 989, 0, False)
    
    # Passed parameters checking function
    _normalize_shape.stypy_localization = localization
    _normalize_shape.stypy_type_of_self = None
    _normalize_shape.stypy_type_store = module_type_store
    _normalize_shape.stypy_function_name = '_normalize_shape'
    _normalize_shape.stypy_param_names_list = ['ndarray', 'shape', 'cast_to_int']
    _normalize_shape.stypy_varargs_param_name = None
    _normalize_shape.stypy_kwargs_param_name = None
    _normalize_shape.stypy_call_defaults = defaults
    _normalize_shape.stypy_call_varargs = varargs
    _normalize_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_normalize_shape', ['ndarray', 'shape', 'cast_to_int'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_normalize_shape', localization, ['ndarray', 'shape', 'cast_to_int'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_normalize_shape(...)' code ##################

    str_103292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, (-1)), 'str', "\n    Private function which does some checks and normalizes the possibly\n    much simpler representations of 'pad_width', 'stat_length',\n    'constant_values', 'end_values'.\n\n    Parameters\n    ----------\n    narray : ndarray\n        Input ndarray\n    shape : {sequence, array_like, float, int}, optional\n        The width of padding (pad_width), the number of elements on the\n        edge of the narray used for statistics (stat_length), the constant\n        value(s) to use when filling padded regions (constant_values), or the\n        endpoint target(s) for linear ramps (end_values).\n        ((before_1, after_1), ... (before_N, after_N)) unique number of\n        elements for each axis where `N` is rank of `narray`.\n        ((before, after),) yields same before and after constants for each\n        axis.\n        (constant,) or val is a shortcut for before = after = constant for\n        all axes.\n    cast_to_int : bool, optional\n        Controls if values in ``shape`` will be rounded and cast to int\n        before being returned.\n\n    Returns\n    -------\n    normalized_shape : tuple of tuples\n        val                               => ((val, val), (val, val), ...)\n        [[val1, val2], [val3, val4], ...] => ((val1, val2), (val3, val4), ...)\n        ((val1, val2), (val3, val4), ...) => no change\n        [[val1, val2], ]                  => ((val1, val2), (val1, val2), ...)\n        ((val1, val2), )                  => ((val1, val2), (val1, val2), ...)\n        [[val ,     ], ]                  => ((val, val), (val, val), ...)\n        ((val ,     ), )                  => ((val, val), (val, val), ...)\n\n    ")
    
    # Assigning a Attribute to a Name (line 1026):
    # Getting the type of 'ndarray' (line 1026)
    ndarray_103293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 12), 'ndarray')
    # Obtaining the member 'ndim' of a type (line 1026)
    ndim_103294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 12), ndarray_103293, 'ndim')
    # Assigning a type to the variable 'ndims' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'ndims', ndim_103294)
    
    # Type idiom detected: calculating its left and rigth part (line 1029)
    # Getting the type of 'shape' (line 1029)
    shape_103295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 7), 'shape')
    # Getting the type of 'None' (line 1029)
    None_103296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 16), 'None')
    
    (may_be_103297, more_types_in_union_103298) = may_be_none(shape_103295, None_103296)

    if may_be_103297:

        if more_types_in_union_103298:
            # Runtime conditional SSA (line 1029)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining an instance of the builtin type 'tuple' (line 1030)
        tuple_103299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1030)
        # Adding element type (line 1030)
        
        # Obtaining an instance of the builtin type 'tuple' (line 1030)
        tuple_103300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1030)
        # Adding element type (line 1030)
        # Getting the type of 'None' (line 1030)
        None_103301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 17), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 17), tuple_103300, None_103301)
        # Adding element type (line 1030)
        # Getting the type of 'None' (line 1030)
        None_103302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 23), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 17), tuple_103300, None_103302)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1030, 16), tuple_103299, tuple_103300)
        
        # Getting the type of 'ndims' (line 1030)
        ndims_103303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 34), 'ndims')
        # Applying the binary operator '*' (line 1030)
        result_mul_103304 = python_operator(stypy.reporting.localization.Localization(__file__, 1030, 15), '*', tuple_103299, ndims_103303)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1030)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'stypy_return_type', result_mul_103304)

        if more_types_in_union_103298:
            # SSA join for if statement (line 1029)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1033):
    
    # Call to asarray(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'shape' (line 1033)
    shape_103307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 21), 'shape', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_103308 = {}
    # Getting the type of 'np' (line 1033)
    np_103305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1033)
    asarray_103306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 10), np_103305, 'asarray')
    # Calling asarray(args, kwargs) (line 1033)
    asarray_call_result_103309 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 10), asarray_103306, *[shape_103307], **kwargs_103308)
    
    # Assigning a type to the variable 'arr' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'arr', asarray_call_result_103309)
    
    
    # Getting the type of 'arr' (line 1036)
    arr_103310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 7), 'arr')
    # Obtaining the member 'ndim' of a type (line 1036)
    ndim_103311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 7), arr_103310, 'ndim')
    int_103312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 19), 'int')
    # Applying the binary operator '<=' (line 1036)
    result_le_103313 = python_operator(stypy.reporting.localization.Localization(__file__, 1036, 7), '<=', ndim_103311, int_103312)
    
    # Testing the type of an if condition (line 1036)
    if_condition_103314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1036, 4), result_le_103313)
    # Assigning a type to the variable 'if_condition_103314' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'if_condition_103314', if_condition_103314)
    # SSA begins for if statement (line 1036)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'arr' (line 1037)
    arr_103315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 11), 'arr')
    # Obtaining the member 'shape' of a type (line 1037)
    shape_103316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 11), arr_103315, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1037)
    tuple_103317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1037)
    
    # Applying the binary operator '==' (line 1037)
    result_eq_103318 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 11), '==', shape_103316, tuple_103317)
    
    
    # Getting the type of 'arr' (line 1037)
    arr_103319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 30), 'arr')
    # Obtaining the member 'shape' of a type (line 1037)
    shape_103320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 30), arr_103319, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1037)
    tuple_103321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1037)
    # Adding element type (line 1037)
    int_103322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1037, 44), tuple_103321, int_103322)
    
    # Applying the binary operator '==' (line 1037)
    result_eq_103323 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 30), '==', shape_103320, tuple_103321)
    
    # Applying the binary operator 'or' (line 1037)
    result_or_keyword_103324 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 11), 'or', result_eq_103318, result_eq_103323)
    
    # Testing the type of an if condition (line 1037)
    if_condition_103325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1037, 8), result_or_keyword_103324)
    # Assigning a type to the variable 'if_condition_103325' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'if_condition_103325', if_condition_103325)
    # SSA begins for if statement (line 1037)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1040):
    
    # Call to ones(...): (line 1040)
    # Processing the call arguments (line 1040)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1040)
    tuple_103328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1040)
    # Adding element type (line 1040)
    # Getting the type of 'ndims' (line 1040)
    ndims_103329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 27), 'ndims', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1040, 27), tuple_103328, ndims_103329)
    # Adding element type (line 1040)
    int_103330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1040, 27), tuple_103328, int_103330)
    
    # Processing the call keyword arguments (line 1040)
    # Getting the type of 'ndarray' (line 1040)
    ndarray_103331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 44), 'ndarray', False)
    # Obtaining the member 'dtype' of a type (line 1040)
    dtype_103332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 44), ndarray_103331, 'dtype')
    keyword_103333 = dtype_103332
    kwargs_103334 = {'dtype': keyword_103333}
    # Getting the type of 'np' (line 1040)
    np_103326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 18), 'np', False)
    # Obtaining the member 'ones' of a type (line 1040)
    ones_103327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 18), np_103326, 'ones')
    # Calling ones(args, kwargs) (line 1040)
    ones_call_result_103335 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 18), ones_103327, *[tuple_103328], **kwargs_103334)
    
    # Getting the type of 'arr' (line 1040)
    arr_103336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 61), 'arr')
    # Applying the binary operator '*' (line 1040)
    result_mul_103337 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 18), '*', ones_call_result_103335, arr_103336)
    
    # Assigning a type to the variable 'arr' (line 1040)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 12), 'arr', result_mul_103337)
    # SSA branch for the else part of an if statement (line 1037)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'arr' (line 1041)
    arr_103338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 13), 'arr')
    # Obtaining the member 'shape' of a type (line 1041)
    shape_103339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1041, 13), arr_103338, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1041)
    tuple_103340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1041)
    # Adding element type (line 1041)
    int_103341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1041, 27), tuple_103340, int_103341)
    
    # Applying the binary operator '==' (line 1041)
    result_eq_103342 = python_operator(stypy.reporting.localization.Localization(__file__, 1041, 13), '==', shape_103339, tuple_103340)
    
    # Testing the type of an if condition (line 1041)
    if_condition_103343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1041, 13), result_eq_103342)
    # Assigning a type to the variable 'if_condition_103343' (line 1041)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1041, 13), 'if_condition_103343', if_condition_103343)
    # SSA begins for if statement (line 1041)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1044):
    
    # Call to repeat(...): (line 1044)
    # Processing the call arguments (line 1044)
    # Getting the type of 'ndims' (line 1044)
    ndims_103351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 44), 'ndims', False)
    # Processing the call keyword arguments (line 1044)
    int_103352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 56), 'int')
    keyword_103353 = int_103352
    kwargs_103354 = {'axis': keyword_103353}
    
    # Obtaining the type of the subscript
    # Getting the type of 'np' (line 1044)
    np_103344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 22), 'np', False)
    # Obtaining the member 'newaxis' of a type (line 1044)
    newaxis_103345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 22), np_103344, 'newaxis')
    slice_103346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1044, 18), None, None, None)
    # Getting the type of 'arr' (line 1044)
    arr_103347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 18), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 1044)
    getitem___103348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 18), arr_103347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1044)
    subscript_call_result_103349 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 18), getitem___103348, (newaxis_103345, slice_103346))
    
    # Obtaining the member 'repeat' of a type (line 1044)
    repeat_103350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 18), subscript_call_result_103349, 'repeat')
    # Calling repeat(args, kwargs) (line 1044)
    repeat_call_result_103355 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 18), repeat_103350, *[ndims_103351], **kwargs_103354)
    
    # Assigning a type to the variable 'arr' (line 1044)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1044, 12), 'arr', repeat_call_result_103355)
    # SSA branch for the else part of an if statement (line 1041)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 1046):
    str_103356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 18), 'str', 'Unable to create correctly shaped tuple from %s')
    # Assigning a type to the variable 'fmt' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 12), 'fmt', str_103356)
    
    # Call to ValueError(...): (line 1047)
    # Processing the call arguments (line 1047)
    # Getting the type of 'fmt' (line 1047)
    fmt_103358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 29), 'fmt', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1047)
    tuple_103359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1047)
    # Adding element type (line 1047)
    # Getting the type of 'shape' (line 1047)
    shape_103360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 36), 'shape', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1047, 36), tuple_103359, shape_103360)
    
    # Applying the binary operator '%' (line 1047)
    result_mod_103361 = python_operator(stypy.reporting.localization.Localization(__file__, 1047, 29), '%', fmt_103358, tuple_103359)
    
    # Processing the call keyword arguments (line 1047)
    kwargs_103362 = {}
    # Getting the type of 'ValueError' (line 1047)
    ValueError_103357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1047)
    ValueError_call_result_103363 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 18), ValueError_103357, *[result_mod_103361], **kwargs_103362)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1047, 12), ValueError_call_result_103363, 'raise parameter', BaseException)
    # SSA join for if statement (line 1041)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1037)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1036)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'arr' (line 1049)
    arr_103364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 9), 'arr')
    # Obtaining the member 'ndim' of a type (line 1049)
    ndim_103365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 9), arr_103364, 'ndim')
    int_103366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 21), 'int')
    # Applying the binary operator '==' (line 1049)
    result_eq_103367 = python_operator(stypy.reporting.localization.Localization(__file__, 1049, 9), '==', ndim_103365, int_103366)
    
    # Testing the type of an if condition (line 1049)
    if_condition_103368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1049, 9), result_eq_103367)
    # Assigning a type to the variable 'if_condition_103368' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 9), 'if_condition_103368', if_condition_103368)
    # SSA begins for if statement (line 1049)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_103369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 21), 'int')
    # Getting the type of 'arr' (line 1050)
    arr_103370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 11), 'arr')
    # Obtaining the member 'shape' of a type (line 1050)
    shape_103371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 11), arr_103370, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1050)
    getitem___103372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 11), shape_103371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1050)
    subscript_call_result_103373 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 11), getitem___103372, int_103369)
    
    int_103374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 27), 'int')
    # Applying the binary operator '==' (line 1050)
    result_eq_103375 = python_operator(stypy.reporting.localization.Localization(__file__, 1050, 11), '==', subscript_call_result_103373, int_103374)
    
    
    
    # Obtaining the type of the subscript
    int_103376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 43), 'int')
    # Getting the type of 'arr' (line 1050)
    arr_103377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 33), 'arr')
    # Obtaining the member 'shape' of a type (line 1050)
    shape_103378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 33), arr_103377, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1050)
    getitem___103379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 33), shape_103378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1050)
    subscript_call_result_103380 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 33), getitem___103379, int_103376)
    
    # Getting the type of 'ndims' (line 1050)
    ndims_103381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 49), 'ndims')
    # Applying the binary operator '==' (line 1050)
    result_eq_103382 = python_operator(stypy.reporting.localization.Localization(__file__, 1050, 33), '==', subscript_call_result_103380, ndims_103381)
    
    # Applying the binary operator 'and' (line 1050)
    result_and_keyword_103383 = python_operator(stypy.reporting.localization.Localization(__file__, 1050, 11), 'and', result_eq_103375, result_eq_103382)
    
    # Testing the type of an if condition (line 1050)
    if_condition_103384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1050, 8), result_and_keyword_103383)
    # Assigning a type to the variable 'if_condition_103384' (line 1050)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'if_condition_103384', if_condition_103384)
    # SSA begins for if statement (line 1050)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1052):
    
    # Call to repeat(...): (line 1052)
    # Processing the call arguments (line 1052)
    int_103387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 29), 'int')
    # Processing the call keyword arguments (line 1052)
    int_103388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 37), 'int')
    keyword_103389 = int_103388
    kwargs_103390 = {'axis': keyword_103389}
    # Getting the type of 'arr' (line 1052)
    arr_103385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 18), 'arr', False)
    # Obtaining the member 'repeat' of a type (line 1052)
    repeat_103386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 18), arr_103385, 'repeat')
    # Calling repeat(args, kwargs) (line 1052)
    repeat_call_result_103391 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 18), repeat_103386, *[int_103387], **kwargs_103390)
    
    # Assigning a type to the variable 'arr' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 12), 'arr', repeat_call_result_103391)
    # SSA branch for the else part of an if statement (line 1050)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_103392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, 23), 'int')
    # Getting the type of 'arr' (line 1053)
    arr_103393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'arr')
    # Obtaining the member 'shape' of a type (line 1053)
    shape_103394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 13), arr_103393, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1053)
    getitem___103395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 13), shape_103394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1053)
    subscript_call_result_103396 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 13), getitem___103395, int_103392)
    
    # Getting the type of 'ndims' (line 1053)
    ndims_103397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 29), 'ndims')
    # Applying the binary operator '==' (line 1053)
    result_eq_103398 = python_operator(stypy.reporting.localization.Localization(__file__, 1053, 13), '==', subscript_call_result_103396, ndims_103397)
    
    # Testing the type of an if condition (line 1053)
    if_condition_103399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1053, 13), result_eq_103398)
    # Assigning a type to the variable 'if_condition_103399' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 13), 'if_condition_103399', if_condition_103399)
    # SSA begins for if statement (line 1053)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1055):
    # Getting the type of 'shape' (line 1055)
    shape_103400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 18), 'shape')
    # Assigning a type to the variable 'arr' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 12), 'arr', shape_103400)
    # SSA branch for the else part of an if statement (line 1053)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 1057):
    str_103401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 18), 'str', 'Unable to create correctly shaped tuple from %s')
    # Assigning a type to the variable 'fmt' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 12), 'fmt', str_103401)
    
    # Call to ValueError(...): (line 1058)
    # Processing the call arguments (line 1058)
    # Getting the type of 'fmt' (line 1058)
    fmt_103403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 29), 'fmt', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1058)
    tuple_103404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1058)
    # Adding element type (line 1058)
    # Getting the type of 'shape' (line 1058)
    shape_103405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 36), 'shape', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1058, 36), tuple_103404, shape_103405)
    
    # Applying the binary operator '%' (line 1058)
    result_mod_103406 = python_operator(stypy.reporting.localization.Localization(__file__, 1058, 29), '%', fmt_103403, tuple_103404)
    
    # Processing the call keyword arguments (line 1058)
    kwargs_103407 = {}
    # Getting the type of 'ValueError' (line 1058)
    ValueError_103402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1058)
    ValueError_call_result_103408 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 18), ValueError_103402, *[result_mod_103406], **kwargs_103407)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1058, 12), ValueError_call_result_103408, 'raise parameter', BaseException)
    # SSA join for if statement (line 1053)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1050)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1049)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 1061):
    str_103409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 14), 'str', 'Unable to create correctly shaped tuple from %s')
    # Assigning a type to the variable 'fmt' (line 1061)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'fmt', str_103409)
    
    # Call to ValueError(...): (line 1062)
    # Processing the call arguments (line 1062)
    # Getting the type of 'fmt' (line 1062)
    fmt_103411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 25), 'fmt', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1062)
    tuple_103412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1062)
    # Adding element type (line 1062)
    # Getting the type of 'shape' (line 1062)
    shape_103413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 32), 'shape', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1062, 32), tuple_103412, shape_103413)
    
    # Applying the binary operator '%' (line 1062)
    result_mod_103414 = python_operator(stypy.reporting.localization.Localization(__file__, 1062, 25), '%', fmt_103411, tuple_103412)
    
    # Processing the call keyword arguments (line 1062)
    kwargs_103415 = {}
    # Getting the type of 'ValueError' (line 1062)
    ValueError_103410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1062)
    ValueError_call_result_103416 = invoke(stypy.reporting.localization.Localization(__file__, 1062, 14), ValueError_103410, *[result_mod_103414], **kwargs_103415)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1062, 8), ValueError_call_result_103416, 'raise parameter', BaseException)
    # SSA join for if statement (line 1049)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1036)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cast_to_int' (line 1065)
    cast_to_int_103417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 7), 'cast_to_int')
    # Getting the type of 'True' (line 1065)
    True_103418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 22), 'True')
    # Applying the binary operator 'is' (line 1065)
    result_is__103419 = python_operator(stypy.reporting.localization.Localization(__file__, 1065, 7), 'is', cast_to_int_103417, True_103418)
    
    # Testing the type of an if condition (line 1065)
    if_condition_103420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1065, 4), result_is__103419)
    # Assigning a type to the variable 'if_condition_103420' (line 1065)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 4), 'if_condition_103420', if_condition_103420)
    # SSA begins for if statement (line 1065)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1066):
    
    # Call to astype(...): (line 1066)
    # Processing the call arguments (line 1066)
    # Getting the type of 'int' (line 1066)
    int_103427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 35), 'int', False)
    # Processing the call keyword arguments (line 1066)
    kwargs_103428 = {}
    
    # Call to round(...): (line 1066)
    # Processing the call arguments (line 1066)
    # Getting the type of 'arr' (line 1066)
    arr_103423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 23), 'arr', False)
    # Processing the call keyword arguments (line 1066)
    kwargs_103424 = {}
    # Getting the type of 'np' (line 1066)
    np_103421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 14), 'np', False)
    # Obtaining the member 'round' of a type (line 1066)
    round_103422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 14), np_103421, 'round')
    # Calling round(args, kwargs) (line 1066)
    round_call_result_103425 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 14), round_103422, *[arr_103423], **kwargs_103424)
    
    # Obtaining the member 'astype' of a type (line 1066)
    astype_103426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 14), round_call_result_103425, 'astype')
    # Calling astype(args, kwargs) (line 1066)
    astype_call_result_103429 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 14), astype_103426, *[int_103427], **kwargs_103428)
    
    # Assigning a type to the variable 'arr' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 8), 'arr', astype_call_result_103429)
    # SSA join for if statement (line 1065)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to tuple(...): (line 1069)
    # Processing the call arguments (line 1069)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 1069, 17, True)
    # Calculating comprehension expression
    
    # Call to tolist(...): (line 1069)
    # Processing the call keyword arguments (line 1069)
    kwargs_103437 = {}
    # Getting the type of 'arr' (line 1069)
    arr_103435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 41), 'arr', False)
    # Obtaining the member 'tolist' of a type (line 1069)
    tolist_103436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 41), arr_103435, 'tolist')
    # Calling tolist(args, kwargs) (line 1069)
    tolist_call_result_103438 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 41), tolist_103436, *[], **kwargs_103437)
    
    comprehension_103439 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 17), tolist_call_result_103438)
    # Assigning a type to the variable 'axis' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 17), 'axis', comprehension_103439)
    
    # Call to tuple(...): (line 1069)
    # Processing the call arguments (line 1069)
    # Getting the type of 'axis' (line 1069)
    axis_103432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 23), 'axis', False)
    # Processing the call keyword arguments (line 1069)
    kwargs_103433 = {}
    # Getting the type of 'tuple' (line 1069)
    tuple_103431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 17), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1069)
    tuple_call_result_103434 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 17), tuple_103431, *[axis_103432], **kwargs_103433)
    
    list_103440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 17), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1069, 17), list_103440, tuple_call_result_103434)
    # Processing the call keyword arguments (line 1069)
    kwargs_103441 = {}
    # Getting the type of 'tuple' (line 1069)
    tuple_103430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1069)
    tuple_call_result_103442 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 11), tuple_103430, *[list_103440], **kwargs_103441)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'stypy_return_type', tuple_call_result_103442)
    
    # ################# End of '_normalize_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_normalize_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 989)
    stypy_return_type_103443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_103443)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_normalize_shape'
    return stypy_return_type_103443

# Assigning a type to the variable '_normalize_shape' (line 989)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 0), '_normalize_shape', _normalize_shape)

@norecursion
def _validate_lengths(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_validate_lengths'
    module_type_store = module_type_store.open_function_context('_validate_lengths', 1072, 0, False)
    
    # Passed parameters checking function
    _validate_lengths.stypy_localization = localization
    _validate_lengths.stypy_type_of_self = None
    _validate_lengths.stypy_type_store = module_type_store
    _validate_lengths.stypy_function_name = '_validate_lengths'
    _validate_lengths.stypy_param_names_list = ['narray', 'number_elements']
    _validate_lengths.stypy_varargs_param_name = None
    _validate_lengths.stypy_kwargs_param_name = None
    _validate_lengths.stypy_call_defaults = defaults
    _validate_lengths.stypy_call_varargs = varargs
    _validate_lengths.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_validate_lengths', ['narray', 'number_elements'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_validate_lengths', localization, ['narray', 'number_elements'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_validate_lengths(...)' code ##################

    str_103444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1102, (-1)), 'str', '\n    Private function which does some checks and reformats pad_width and\n    stat_length using _normalize_shape.\n\n    Parameters\n    ----------\n    narray : ndarray\n        Input ndarray\n    number_elements : {sequence, int}, optional\n        The width of padding (pad_width) or the number of elements on the edge\n        of the narray used for statistics (stat_length).\n        ((before_1, after_1), ... (before_N, after_N)) unique number of\n        elements for each axis.\n        ((before, after),) yields same before and after constants for each\n        axis.\n        (constant,) or int is a shortcut for before = after = constant for all\n        axes.\n\n    Returns\n    -------\n    _validate_lengths : tuple of tuples\n        int                               => ((int, int), (int, int), ...)\n        [[int1, int2], [int3, int4], ...] => ((int1, int2), (int3, int4), ...)\n        ((int1, int2), (int3, int4), ...) => no change\n        [[int1, int2], ]                  => ((int1, int2), (int1, int2), ...)\n        ((int1, int2), )                  => ((int1, int2), (int1, int2), ...)\n        [[int ,     ], ]                  => ((int, int), (int, int), ...)\n        ((int ,     ), )                  => ((int, int), (int, int), ...)\n\n    ')
    
    # Assigning a Call to a Name (line 1103):
    
    # Call to _normalize_shape(...): (line 1103)
    # Processing the call arguments (line 1103)
    # Getting the type of 'narray' (line 1103)
    narray_103446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 31), 'narray', False)
    # Getting the type of 'number_elements' (line 1103)
    number_elements_103447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 39), 'number_elements', False)
    # Processing the call keyword arguments (line 1103)
    kwargs_103448 = {}
    # Getting the type of '_normalize_shape' (line 1103)
    _normalize_shape_103445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 14), '_normalize_shape', False)
    # Calling _normalize_shape(args, kwargs) (line 1103)
    _normalize_shape_call_result_103449 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 14), _normalize_shape_103445, *[narray_103446, number_elements_103447], **kwargs_103448)
    
    # Assigning a type to the variable 'normshp' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'normshp', _normalize_shape_call_result_103449)
    
    # Getting the type of 'normshp' (line 1104)
    normshp_103450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 13), 'normshp')
    # Testing the type of a for loop iterable (line 1104)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1104, 4), normshp_103450)
    # Getting the type of the for loop variable (line 1104)
    for_loop_var_103451 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1104, 4), normshp_103450)
    # Assigning a type to the variable 'i' (line 1104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 4), 'i', for_loop_var_103451)
    # SSA begins for a for statement (line 1104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a ListComp to a Name (line 1105):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'i' (line 1105)
    i_103458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 46), 'i')
    comprehension_103459 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 15), i_103458)
    # Assigning a type to the variable 'x' (line 1105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1105, 15), 'x', comprehension_103459)
    
    
    # Getting the type of 'x' (line 1105)
    x_103452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 20), 'x')
    # Getting the type of 'None' (line 1105)
    None_103453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 25), 'None')
    # Applying the binary operator 'is' (line 1105)
    result_is__103454 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 20), 'is', x_103452, None_103453)
    
    # Testing the type of an if expression (line 1105)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1105, 15), result_is__103454)
    # SSA begins for if expression (line 1105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_103455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 15), 'int')
    # SSA branch for the else part of an if expression (line 1105)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'x' (line 1105)
    x_103456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 35), 'x')
    # SSA join for if expression (line 1105)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103457 = union_type.UnionType.add(int_103455, x_103456)
    
    list_103460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1105, 15), list_103460, if_exp_103457)
    # Assigning a type to the variable 'chk' (line 1105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1105, 8), 'chk', list_103460)
    
    # Assigning a ListComp to a Name (line 1106):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'chk' (line 1106)
    chk_103467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 44), 'chk')
    comprehension_103468 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 15), chk_103467)
    # Assigning a type to the variable 'x' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 15), 'x', comprehension_103468)
    
    
    # Getting the type of 'x' (line 1106)
    x_103461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 20), 'x')
    int_103462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 25), 'int')
    # Applying the binary operator '>=' (line 1106)
    result_ge_103463 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 20), '>=', x_103461, int_103462)
    
    # Testing the type of an if expression (line 1106)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1106, 15), result_ge_103463)
    # SSA begins for if expression (line 1106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_103464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 15), 'int')
    # SSA branch for the else part of an if expression (line 1106)
    module_type_store.open_ssa_branch('if expression else')
    int_103465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 32), 'int')
    # SSA join for if expression (line 1106)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_103466 = union_type.UnionType.add(int_103464, int_103465)
    
    list_103469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 15), list_103469, if_exp_103466)
    # Assigning a type to the variable 'chk' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'chk', list_103469)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_103470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 16), 'int')
    # Getting the type of 'chk' (line 1107)
    chk_103471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 12), 'chk')
    # Obtaining the member '__getitem__' of a type (line 1107)
    getitem___103472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1107, 12), chk_103471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1107)
    subscript_call_result_103473 = invoke(stypy.reporting.localization.Localization(__file__, 1107, 12), getitem___103472, int_103470)
    
    int_103474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 21), 'int')
    # Applying the binary operator '<' (line 1107)
    result_lt_103475 = python_operator(stypy.reporting.localization.Localization(__file__, 1107, 12), '<', subscript_call_result_103473, int_103474)
    
    
    
    # Obtaining the type of the subscript
    int_103476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 32), 'int')
    # Getting the type of 'chk' (line 1107)
    chk_103477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 28), 'chk')
    # Obtaining the member '__getitem__' of a type (line 1107)
    getitem___103478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1107, 28), chk_103477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1107)
    subscript_call_result_103479 = invoke(stypy.reporting.localization.Localization(__file__, 1107, 28), getitem___103478, int_103476)
    
    int_103480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1107, 37), 'int')
    # Applying the binary operator '<' (line 1107)
    result_lt_103481 = python_operator(stypy.reporting.localization.Localization(__file__, 1107, 28), '<', subscript_call_result_103479, int_103480)
    
    # Applying the binary operator 'or' (line 1107)
    result_or_keyword_103482 = python_operator(stypy.reporting.localization.Localization(__file__, 1107, 11), 'or', result_lt_103475, result_lt_103481)
    
    # Testing the type of an if condition (line 1107)
    if_condition_103483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1107, 8), result_or_keyword_103482)
    # Assigning a type to the variable 'if_condition_103483' (line 1107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1107, 8), 'if_condition_103483', if_condition_103483)
    # SSA begins for if statement (line 1107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1108):
    str_103484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 18), 'str', '%s cannot contain negative values.')
    # Assigning a type to the variable 'fmt' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 12), 'fmt', str_103484)
    
    # Call to ValueError(...): (line 1109)
    # Processing the call arguments (line 1109)
    # Getting the type of 'fmt' (line 1109)
    fmt_103486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 29), 'fmt', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1109)
    tuple_103487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1109)
    # Adding element type (line 1109)
    # Getting the type of 'number_elements' (line 1109)
    number_elements_103488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 36), 'number_elements', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1109, 36), tuple_103487, number_elements_103488)
    
    # Applying the binary operator '%' (line 1109)
    result_mod_103489 = python_operator(stypy.reporting.localization.Localization(__file__, 1109, 29), '%', fmt_103486, tuple_103487)
    
    # Processing the call keyword arguments (line 1109)
    kwargs_103490 = {}
    # Getting the type of 'ValueError' (line 1109)
    ValueError_103485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1109)
    ValueError_call_result_103491 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 18), ValueError_103485, *[result_mod_103489], **kwargs_103490)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1109, 12), ValueError_call_result_103491, 'raise parameter', BaseException)
    # SSA join for if statement (line 1107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'normshp' (line 1110)
    normshp_103492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 11), 'normshp')
    # Assigning a type to the variable 'stypy_return_type' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 4), 'stypy_return_type', normshp_103492)
    
    # ################# End of '_validate_lengths(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_validate_lengths' in the type store
    # Getting the type of 'stypy_return_type' (line 1072)
    stypy_return_type_103493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_103493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_validate_lengths'
    return stypy_return_type_103493

# Assigning a type to the variable '_validate_lengths' (line 1072)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 0), '_validate_lengths', _validate_lengths)

@norecursion
def pad(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pad'
    module_type_store = module_type_store.open_function_context('pad', 1117, 0, False)
    
    # Passed parameters checking function
    pad.stypy_localization = localization
    pad.stypy_type_of_self = None
    pad.stypy_type_store = module_type_store
    pad.stypy_function_name = 'pad'
    pad.stypy_param_names_list = ['array', 'pad_width', 'mode']
    pad.stypy_varargs_param_name = None
    pad.stypy_kwargs_param_name = 'kwargs'
    pad.stypy_call_defaults = defaults
    pad.stypy_call_varargs = varargs
    pad.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pad', ['array', 'pad_width', 'mode'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pad', localization, ['array', 'pad_width', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pad(...)' code ##################

    str_103494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1313, (-1)), 'str', "\n    Pads an array.\n\n    Parameters\n    ----------\n    array : array_like of rank N\n        Input array\n    pad_width : {sequence, array_like, int}\n        Number of values padded to the edges of each axis.\n        ((before_1, after_1), ... (before_N, after_N)) unique pad widths\n        for each axis.\n        ((before, after),) yields same before and after pad for each axis.\n        (pad,) or int is a shortcut for before = after = pad width for all\n        axes.\n    mode : str or function\n        One of the following string values or a user supplied function.\n\n        'constant'\n            Pads with a constant value.\n        'edge'\n            Pads with the edge values of array.\n        'linear_ramp'\n            Pads with the linear ramp between end_value and the\n            array edge value.\n        'maximum'\n            Pads with the maximum value of all or part of the\n            vector along each axis.\n        'mean'\n            Pads with the mean value of all or part of the\n            vector along each axis.\n        'median'\n            Pads with the median value of all or part of the\n            vector along each axis.\n        'minimum'\n            Pads with the minimum value of all or part of the\n            vector along each axis.\n        'reflect'\n            Pads with the reflection of the vector mirrored on\n            the first and last values of the vector along each\n            axis.\n        'symmetric'\n            Pads with the reflection of the vector mirrored\n            along the edge of the array.\n        'wrap'\n            Pads with the wrap of the vector along the axis.\n            The first values are used to pad the end and the\n            end values are used to pad the beginning.\n        <function>\n            Padding function, see Notes.\n    stat_length : sequence or int, optional\n        Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of\n        values at edge of each axis used to calculate the statistic value.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique statistic\n        lengths for each axis.\n\n        ((before, after),) yields same before and after statistic lengths\n        for each axis.\n\n        (stat_length,) or int is a shortcut for before = after = statistic\n        length for all axes.\n\n        Default is ``None``, to use the entire axis.\n    constant_values : sequence or int, optional\n        Used in 'constant'.  The values to set the padded values for each\n        axis.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique pad constants\n        for each axis.\n\n        ((before, after),) yields same before and after constants for each\n        axis.\n\n        (constant,) or int is a shortcut for before = after = constant for\n        all axes.\n\n        Default is 0.\n    end_values : sequence or int, optional\n        Used in 'linear_ramp'.  The values used for the ending value of the\n        linear_ramp and that will form the edge of the padded array.\n\n        ((before_1, after_1), ... (before_N, after_N)) unique end values\n        for each axis.\n\n        ((before, after),) yields same before and after end values for each\n        axis.\n\n        (constant,) or int is a shortcut for before = after = end value for\n        all axes.\n\n        Default is 0.\n    reflect_type : {'even', 'odd'}, optional\n        Used in 'reflect', and 'symmetric'.  The 'even' style is the\n        default with an unaltered reflection around the edge value.  For\n        the 'odd' style, the extented part of the array is created by\n        subtracting the reflected values from two times the edge value.\n\n    Returns\n    -------\n    pad : ndarray\n        Padded array of rank equal to `array` with shape increased\n        according to `pad_width`.\n\n    Notes\n    -----\n    .. versionadded:: 1.7.0\n\n    For an array with rank greater than 1, some of the padding of later\n    axes is calculated from padding of previous axes.  This is easiest to\n    think about with a rank 2 array where the corners of the padded array\n    are calculated by using padded values from the first axis.\n\n    The padding function, if used, should return a rank 1 array equal in\n    length to the vector argument with padded values replaced. It has the\n    following signature::\n\n        padding_func(vector, iaxis_pad_width, iaxis, **kwargs)\n\n    where\n\n        vector : ndarray\n            A rank 1 array already padded with zeros.  Padded values are\n            vector[:pad_tuple[0]] and vector[-pad_tuple[1]:].\n        iaxis_pad_width : tuple\n            A 2-tuple of ints, iaxis_pad_width[0] represents the number of\n            values padded at the beginning of vector where\n            iaxis_pad_width[1] represents the number of values padded at\n            the end of vector.\n        iaxis : int\n            The axis currently being calculated.\n        kwargs : misc\n            Any keyword arguments the function requires.\n\n    Examples\n    --------\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.lib.pad(a, (2,3), 'constant', constant_values=(4, 6))\n    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])\n\n    >>> np.lib.pad(a, (2, 3), 'edge')\n    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])\n\n    >>> np.lib.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))\n    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])\n\n    >>> np.lib.pad(a, (2,), 'maximum')\n    array([5, 5, 1, 2, 3, 4, 5, 5, 5])\n\n    >>> np.lib.pad(a, (2,), 'mean')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> np.lib.pad(a, (2,), 'median')\n    array([3, 3, 1, 2, 3, 4, 5, 3, 3])\n\n    >>> a = [[1, 2], [3, 4]]\n    >>> np.lib.pad(a, ((3, 2), (2, 3)), 'minimum')\n    array([[1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1],\n           [3, 3, 3, 4, 3, 3, 3],\n           [1, 1, 1, 2, 1, 1, 1],\n           [1, 1, 1, 2, 1, 1, 1]])\n\n    >>> a = [1, 2, 3, 4, 5]\n    >>> np.lib.pad(a, (2, 3), 'reflect')\n    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])\n\n    >>> np.lib.pad(a, (2, 3), 'reflect', reflect_type='odd')\n    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])\n\n    >>> np.lib.pad(a, (2, 3), 'symmetric')\n    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])\n\n    >>> np.lib.pad(a, (2, 3), 'symmetric', reflect_type='odd')\n    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])\n\n    >>> np.lib.pad(a, (2, 3), 'wrap')\n    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])\n\n    >>> def padwithtens(vector, pad_width, iaxis, kwargs):\n    ...     vector[:pad_width[0]] = 10\n    ...     vector[-pad_width[1]:] = 10\n    ...     return vector\n\n    >>> a = np.arange(6)\n    >>> a = a.reshape((2, 3))\n\n    >>> np.lib.pad(a, 2, padwithtens)\n    array([[10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10,  0,  1,  2, 10, 10],\n           [10, 10,  3,  4,  5, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10],\n           [10, 10, 10, 10, 10, 10, 10]])\n    ")
    
    
    
    
    # Call to asarray(...): (line 1314)
    # Processing the call arguments (line 1314)
    # Getting the type of 'pad_width' (line 1314)
    pad_width_103497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 22), 'pad_width', False)
    # Processing the call keyword arguments (line 1314)
    kwargs_103498 = {}
    # Getting the type of 'np' (line 1314)
    np_103495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1314)
    asarray_103496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1314, 11), np_103495, 'asarray')
    # Calling asarray(args, kwargs) (line 1314)
    asarray_call_result_103499 = invoke(stypy.reporting.localization.Localization(__file__, 1314, 11), asarray_103496, *[pad_width_103497], **kwargs_103498)
    
    # Obtaining the member 'dtype' of a type (line 1314)
    dtype_103500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1314, 11), asarray_call_result_103499, 'dtype')
    # Obtaining the member 'kind' of a type (line 1314)
    kind_103501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1314, 11), dtype_103500, 'kind')
    str_103502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, 47), 'str', 'i')
    # Applying the binary operator '==' (line 1314)
    result_eq_103503 = python_operator(stypy.reporting.localization.Localization(__file__, 1314, 11), '==', kind_103501, str_103502)
    
    # Applying the 'not' unary operator (line 1314)
    result_not__103504 = python_operator(stypy.reporting.localization.Localization(__file__, 1314, 7), 'not', result_eq_103503)
    
    # Testing the type of an if condition (line 1314)
    if_condition_103505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1314, 4), result_not__103504)
    # Assigning a type to the variable 'if_condition_103505' (line 1314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1314, 4), 'if_condition_103505', if_condition_103505)
    # SSA begins for if statement (line 1314)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1315)
    # Processing the call arguments (line 1315)
    str_103507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1315, 24), 'str', '`pad_width` must be of integral type.')
    # Processing the call keyword arguments (line 1315)
    kwargs_103508 = {}
    # Getting the type of 'TypeError' (line 1315)
    TypeError_103506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1315)
    TypeError_call_result_103509 = invoke(stypy.reporting.localization.Localization(__file__, 1315, 14), TypeError_103506, *[str_103507], **kwargs_103508)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1315, 8), TypeError_call_result_103509, 'raise parameter', BaseException)
    # SSA join for if statement (line 1314)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1317):
    
    # Call to array(...): (line 1317)
    # Processing the call arguments (line 1317)
    # Getting the type of 'array' (line 1317)
    array_103512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 22), 'array', False)
    # Processing the call keyword arguments (line 1317)
    kwargs_103513 = {}
    # Getting the type of 'np' (line 1317)
    np_103510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 1317)
    array_103511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1317, 13), np_103510, 'array')
    # Calling array(args, kwargs) (line 1317)
    array_call_result_103514 = invoke(stypy.reporting.localization.Localization(__file__, 1317, 13), array_103511, *[array_103512], **kwargs_103513)
    
    # Assigning a type to the variable 'narray' (line 1317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'narray', array_call_result_103514)
    
    # Assigning a Call to a Name (line 1318):
    
    # Call to _validate_lengths(...): (line 1318)
    # Processing the call arguments (line 1318)
    # Getting the type of 'narray' (line 1318)
    narray_103516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 34), 'narray', False)
    # Getting the type of 'pad_width' (line 1318)
    pad_width_103517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 42), 'pad_width', False)
    # Processing the call keyword arguments (line 1318)
    kwargs_103518 = {}
    # Getting the type of '_validate_lengths' (line 1318)
    _validate_lengths_103515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 16), '_validate_lengths', False)
    # Calling _validate_lengths(args, kwargs) (line 1318)
    _validate_lengths_call_result_103519 = invoke(stypy.reporting.localization.Localization(__file__, 1318, 16), _validate_lengths_103515, *[narray_103516, pad_width_103517], **kwargs_103518)
    
    # Assigning a type to the variable 'pad_width' (line 1318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1318, 4), 'pad_width', _validate_lengths_call_result_103519)
    
    # Assigning a Dict to a Name (line 1320):
    
    # Obtaining an instance of the builtin type 'dict' (line 1320)
    dict_103520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 20), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1320)
    # Adding element type (key, value) (line 1320)
    str_103521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1321, 8), 'str', 'constant')
    
    # Obtaining an instance of the builtin type 'list' (line 1321)
    list_103522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1321, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1321)
    # Adding element type (line 1321)
    str_103523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1321, 21), 'str', 'constant_values')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1321, 20), list_103522, str_103523)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103521, list_103522))
    # Adding element type (key, value) (line 1320)
    str_103524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, 8), 'str', 'edge')
    
    # Obtaining an instance of the builtin type 'list' (line 1322)
    list_103525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1322, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1322)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103524, list_103525))
    # Adding element type (key, value) (line 1320)
    str_103526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1323, 8), 'str', 'linear_ramp')
    
    # Obtaining an instance of the builtin type 'list' (line 1323)
    list_103527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1323, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1323)
    # Adding element type (line 1323)
    str_103528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1323, 24), 'str', 'end_values')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1323, 23), list_103527, str_103528)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103526, list_103527))
    # Adding element type (key, value) (line 1320)
    str_103529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 8), 'str', 'maximum')
    
    # Obtaining an instance of the builtin type 'list' (line 1324)
    list_103530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1324)
    # Adding element type (line 1324)
    str_103531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1324, 20), 'str', 'stat_length')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1324, 19), list_103530, str_103531)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103529, list_103530))
    # Adding element type (key, value) (line 1320)
    str_103532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 8), 'str', 'mean')
    
    # Obtaining an instance of the builtin type 'list' (line 1325)
    list_103533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1325)
    # Adding element type (line 1325)
    str_103534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1325, 17), 'str', 'stat_length')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1325, 16), list_103533, str_103534)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103532, list_103533))
    # Adding element type (key, value) (line 1320)
    str_103535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1326, 8), 'str', 'median')
    
    # Obtaining an instance of the builtin type 'list' (line 1326)
    list_103536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1326, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1326)
    # Adding element type (line 1326)
    str_103537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1326, 19), 'str', 'stat_length')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1326, 18), list_103536, str_103537)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103535, list_103536))
    # Adding element type (key, value) (line 1320)
    str_103538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1327, 8), 'str', 'minimum')
    
    # Obtaining an instance of the builtin type 'list' (line 1327)
    list_103539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1327, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1327)
    # Adding element type (line 1327)
    str_103540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1327, 20), 'str', 'stat_length')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1327, 19), list_103539, str_103540)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103538, list_103539))
    # Adding element type (key, value) (line 1320)
    str_103541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1328, 8), 'str', 'reflect')
    
    # Obtaining an instance of the builtin type 'list' (line 1328)
    list_103542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1328, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1328)
    # Adding element type (line 1328)
    str_103543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1328, 20), 'str', 'reflect_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1328, 19), list_103542, str_103543)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103541, list_103542))
    # Adding element type (key, value) (line 1320)
    str_103544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 8), 'str', 'symmetric')
    
    # Obtaining an instance of the builtin type 'list' (line 1329)
    list_103545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1329)
    # Adding element type (line 1329)
    str_103546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 22), 'str', 'reflect_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1329, 21), list_103545, str_103546)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103544, list_103545))
    # Adding element type (key, value) (line 1320)
    str_103547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 8), 'str', 'wrap')
    
    # Obtaining an instance of the builtin type 'list' (line 1330)
    list_103548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1330)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1320, 20), dict_103520, (str_103547, list_103548))
    
    # Assigning a type to the variable 'allowedkwargs' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'allowedkwargs', dict_103520)
    
    # Assigning a Dict to a Name (line 1333):
    
    # Obtaining an instance of the builtin type 'dict' (line 1333)
    dict_103549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1333, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1333)
    # Adding element type (key, value) (line 1333)
    str_103550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1334, 8), 'str', 'stat_length')
    # Getting the type of 'None' (line 1334)
    None_103551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1334, 23), 'None')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1333, 17), dict_103549, (str_103550, None_103551))
    # Adding element type (key, value) (line 1333)
    str_103552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1335, 8), 'str', 'constant_values')
    int_103553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1335, 27), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1333, 17), dict_103549, (str_103552, int_103553))
    # Adding element type (key, value) (line 1333)
    str_103554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1336, 8), 'str', 'end_values')
    int_103555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1336, 22), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1333, 17), dict_103549, (str_103554, int_103555))
    # Adding element type (key, value) (line 1333)
    str_103556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1337, 8), 'str', 'reflect_type')
    str_103557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1337, 24), 'str', 'even')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1333, 17), dict_103549, (str_103556, str_103557))
    
    # Assigning a type to the variable 'kwdefaults' (line 1333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1333, 4), 'kwdefaults', dict_103549)
    
    
    # Call to isinstance(...): (line 1340)
    # Processing the call arguments (line 1340)
    # Getting the type of 'mode' (line 1340)
    mode_103559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 18), 'mode', False)
    # Getting the type of 'np' (line 1340)
    np_103560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 24), 'np', False)
    # Obtaining the member 'compat' of a type (line 1340)
    compat_103561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1340, 24), np_103560, 'compat')
    # Obtaining the member 'basestring' of a type (line 1340)
    basestring_103562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1340, 24), compat_103561, 'basestring')
    # Processing the call keyword arguments (line 1340)
    kwargs_103563 = {}
    # Getting the type of 'isinstance' (line 1340)
    isinstance_103558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1340, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1340)
    isinstance_call_result_103564 = invoke(stypy.reporting.localization.Localization(__file__, 1340, 7), isinstance_103558, *[mode_103559, basestring_103562], **kwargs_103563)
    
    # Testing the type of an if condition (line 1340)
    if_condition_103565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1340, 4), isinstance_call_result_103564)
    # Assigning a type to the variable 'if_condition_103565' (line 1340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1340, 4), 'if_condition_103565', if_condition_103565)
    # SSA begins for if statement (line 1340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'kwargs' (line 1342)
    kwargs_103566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1342, 19), 'kwargs')
    # Testing the type of a for loop iterable (line 1342)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1342, 8), kwargs_103566)
    # Getting the type of the for loop variable (line 1342)
    for_loop_var_103567 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1342, 8), kwargs_103566)
    # Assigning a type to the variable 'key' (line 1342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1342, 8), 'key', for_loop_var_103567)
    # SSA begins for a for statement (line 1342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'key' (line 1343)
    key_103568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 15), 'key')
    
    # Obtaining the type of the subscript
    # Getting the type of 'mode' (line 1343)
    mode_103569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 40), 'mode')
    # Getting the type of 'allowedkwargs' (line 1343)
    allowedkwargs_103570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1343, 26), 'allowedkwargs')
    # Obtaining the member '__getitem__' of a type (line 1343)
    getitem___103571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1343, 26), allowedkwargs_103570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1343)
    subscript_call_result_103572 = invoke(stypy.reporting.localization.Localization(__file__, 1343, 26), getitem___103571, mode_103569)
    
    # Applying the binary operator 'notin' (line 1343)
    result_contains_103573 = python_operator(stypy.reporting.localization.Localization(__file__, 1343, 15), 'notin', key_103568, subscript_call_result_103572)
    
    # Testing the type of an if condition (line 1343)
    if_condition_103574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1343, 12), result_contains_103573)
    # Assigning a type to the variable 'if_condition_103574' (line 1343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1343, 12), 'if_condition_103574', if_condition_103574)
    # SSA begins for if statement (line 1343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1344)
    # Processing the call arguments (line 1344)
    str_103576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1344, 33), 'str', '%s keyword not in allowed keywords %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1345)
    tuple_103577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1345, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1345)
    # Adding element type (line 1345)
    # Getting the type of 'key' (line 1345)
    key_103578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 34), 'key', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 34), tuple_103577, key_103578)
    # Adding element type (line 1345)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mode' (line 1345)
    mode_103579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 53), 'mode', False)
    # Getting the type of 'allowedkwargs' (line 1345)
    allowedkwargs_103580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1345, 39), 'allowedkwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1345)
    getitem___103581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1345, 39), allowedkwargs_103580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1345)
    subscript_call_result_103582 = invoke(stypy.reporting.localization.Localization(__file__, 1345, 39), getitem___103581, mode_103579)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1345, 34), tuple_103577, subscript_call_result_103582)
    
    # Applying the binary operator '%' (line 1344)
    result_mod_103583 = python_operator(stypy.reporting.localization.Localization(__file__, 1344, 33), '%', str_103576, tuple_103577)
    
    # Processing the call keyword arguments (line 1344)
    kwargs_103584 = {}
    # Getting the type of 'ValueError' (line 1344)
    ValueError_103575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1344, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1344)
    ValueError_call_result_103585 = invoke(stypy.reporting.localization.Localization(__file__, 1344, 22), ValueError_103575, *[result_mod_103583], **kwargs_103584)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1344, 16), ValueError_call_result_103585, 'raise parameter', BaseException)
    # SSA join for if statement (line 1343)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'mode' (line 1348)
    mode_103586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 32), 'mode')
    # Getting the type of 'allowedkwargs' (line 1348)
    allowedkwargs_103587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 18), 'allowedkwargs')
    # Obtaining the member '__getitem__' of a type (line 1348)
    getitem___103588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1348, 18), allowedkwargs_103587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1348)
    subscript_call_result_103589 = invoke(stypy.reporting.localization.Localization(__file__, 1348, 18), getitem___103588, mode_103586)
    
    # Testing the type of a for loop iterable (line 1348)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1348, 8), subscript_call_result_103589)
    # Getting the type of the for loop variable (line 1348)
    for_loop_var_103590 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1348, 8), subscript_call_result_103589)
    # Assigning a type to the variable 'kw' (line 1348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1348, 8), 'kw', for_loop_var_103590)
    # SSA begins for a for statement (line 1348)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to setdefault(...): (line 1349)
    # Processing the call arguments (line 1349)
    # Getting the type of 'kw' (line 1349)
    kw_103593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 30), 'kw', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'kw' (line 1349)
    kw_103594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 45), 'kw', False)
    # Getting the type of 'kwdefaults' (line 1349)
    kwdefaults_103595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 34), 'kwdefaults', False)
    # Obtaining the member '__getitem__' of a type (line 1349)
    getitem___103596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1349, 34), kwdefaults_103595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1349)
    subscript_call_result_103597 = invoke(stypy.reporting.localization.Localization(__file__, 1349, 34), getitem___103596, kw_103594)
    
    # Processing the call keyword arguments (line 1349)
    kwargs_103598 = {}
    # Getting the type of 'kwargs' (line 1349)
    kwargs_103591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 12), 'kwargs', False)
    # Obtaining the member 'setdefault' of a type (line 1349)
    setdefault_103592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1349, 12), kwargs_103591, 'setdefault')
    # Calling setdefault(args, kwargs) (line 1349)
    setdefault_call_result_103599 = invoke(stypy.reporting.localization.Localization(__file__, 1349, 12), setdefault_103592, *[kw_103593, subscript_call_result_103597], **kwargs_103598)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'kwargs' (line 1352)
    kwargs_103600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 17), 'kwargs')
    # Testing the type of a for loop iterable (line 1352)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1352, 8), kwargs_103600)
    # Getting the type of the for loop variable (line 1352)
    for_loop_var_103601 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1352, 8), kwargs_103600)
    # Assigning a type to the variable 'i' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 8), 'i', for_loop_var_103601)
    # SSA begins for a for statement (line 1352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'i' (line 1353)
    i_103602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 15), 'i')
    str_103603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 20), 'str', 'stat_length')
    # Applying the binary operator '==' (line 1353)
    result_eq_103604 = python_operator(stypy.reporting.localization.Localization(__file__, 1353, 15), '==', i_103602, str_103603)
    
    # Testing the type of an if condition (line 1353)
    if_condition_103605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1353, 12), result_eq_103604)
    # Assigning a type to the variable 'if_condition_103605' (line 1353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1353, 12), 'if_condition_103605', if_condition_103605)
    # SSA begins for if statement (line 1353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 1354):
    
    # Call to _validate_lengths(...): (line 1354)
    # Processing the call arguments (line 1354)
    # Getting the type of 'narray' (line 1354)
    narray_103607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 46), 'narray', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1354)
    i_103608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 61), 'i', False)
    # Getting the type of 'kwargs' (line 1354)
    kwargs_103609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 54), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1354)
    getitem___103610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1354, 54), kwargs_103609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1354)
    subscript_call_result_103611 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 54), getitem___103610, i_103608)
    
    # Processing the call keyword arguments (line 1354)
    kwargs_103612 = {}
    # Getting the type of '_validate_lengths' (line 1354)
    _validate_lengths_103606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 28), '_validate_lengths', False)
    # Calling _validate_lengths(args, kwargs) (line 1354)
    _validate_lengths_call_result_103613 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 28), _validate_lengths_103606, *[narray_103607, subscript_call_result_103611], **kwargs_103612)
    
    # Getting the type of 'kwargs' (line 1354)
    kwargs_103614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 16), 'kwargs')
    # Getting the type of 'i' (line 1354)
    i_103615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 23), 'i')
    # Storing an element on a container (line 1354)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 16), kwargs_103614, (i_103615, _validate_lengths_call_result_103613))
    # SSA join for if statement (line 1353)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'i' (line 1355)
    i_103616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 15), 'i')
    
    # Obtaining an instance of the builtin type 'list' (line 1355)
    list_103617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1355)
    # Adding element type (line 1355)
    str_103618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 21), 'str', 'end_values')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 20), list_103617, str_103618)
    # Adding element type (line 1355)
    str_103619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 35), 'str', 'constant_values')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 20), list_103617, str_103619)
    
    # Applying the binary operator 'in' (line 1355)
    result_contains_103620 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 15), 'in', i_103616, list_103617)
    
    # Testing the type of an if condition (line 1355)
    if_condition_103621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1355, 12), result_contains_103620)
    # Assigning a type to the variable 'if_condition_103621' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 12), 'if_condition_103621', if_condition_103621)
    # SSA begins for if statement (line 1355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 1356):
    
    # Call to _normalize_shape(...): (line 1356)
    # Processing the call arguments (line 1356)
    # Getting the type of 'narray' (line 1356)
    narray_103623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 45), 'narray', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1356)
    i_103624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 60), 'i', False)
    # Getting the type of 'kwargs' (line 1356)
    kwargs_103625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 53), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1356)
    getitem___103626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 53), kwargs_103625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1356)
    subscript_call_result_103627 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 53), getitem___103626, i_103624)
    
    # Processing the call keyword arguments (line 1356)
    # Getting the type of 'False' (line 1357)
    False_103628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 57), 'False', False)
    keyword_103629 = False_103628
    kwargs_103630 = {'cast_to_int': keyword_103629}
    # Getting the type of '_normalize_shape' (line 1356)
    _normalize_shape_103622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 28), '_normalize_shape', False)
    # Calling _normalize_shape(args, kwargs) (line 1356)
    _normalize_shape_call_result_103631 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 28), _normalize_shape_103622, *[narray_103623, subscript_call_result_103627], **kwargs_103630)
    
    # Getting the type of 'kwargs' (line 1356)
    kwargs_103632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 16), 'kwargs')
    # Getting the type of 'i' (line 1356)
    i_103633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 23), 'i')
    # Storing an element on a container (line 1356)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 16), kwargs_103632, (i_103633, _normalize_shape_call_result_103631))
    # SSA join for if statement (line 1355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1340)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1361):
    # Getting the type of 'mode' (line 1361)
    mode_103634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 19), 'mode')
    # Assigning a type to the variable 'function' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 8), 'function', mode_103634)
    
    # Assigning a Call to a Name (line 1364):
    
    # Call to list(...): (line 1364)
    # Processing the call arguments (line 1364)
    
    # Call to range(...): (line 1364)
    # Processing the call arguments (line 1364)
    
    # Call to len(...): (line 1364)
    # Processing the call arguments (line 1364)
    # Getting the type of 'narray' (line 1364)
    narray_103638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 30), 'narray', False)
    # Obtaining the member 'shape' of a type (line 1364)
    shape_103639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 30), narray_103638, 'shape')
    # Processing the call keyword arguments (line 1364)
    kwargs_103640 = {}
    # Getting the type of 'len' (line 1364)
    len_103637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 26), 'len', False)
    # Calling len(args, kwargs) (line 1364)
    len_call_result_103641 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 26), len_103637, *[shape_103639], **kwargs_103640)
    
    # Processing the call keyword arguments (line 1364)
    kwargs_103642 = {}
    # Getting the type of 'range' (line 1364)
    range_103636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 20), 'range', False)
    # Calling range(args, kwargs) (line 1364)
    range_call_result_103643 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 20), range_103636, *[len_call_result_103641], **kwargs_103642)
    
    # Processing the call keyword arguments (line 1364)
    kwargs_103644 = {}
    # Getting the type of 'list' (line 1364)
    list_103635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 15), 'list', False)
    # Calling list(args, kwargs) (line 1364)
    list_call_result_103645 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 15), list_103635, *[range_call_result_103643], **kwargs_103644)
    
    # Assigning a type to the variable 'rank' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 8), 'rank', list_call_result_103645)
    
    # Assigning a ListComp to a Name (line 1365):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'rank' (line 1365)
    rank_103654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 60), 'rank')
    comprehension_103655 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 30), rank_103654)
    # Assigning a type to the variable 'i' (line 1365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1365, 30), 'i', comprehension_103655)
    
    # Call to sum(...): (line 1365)
    # Processing the call arguments (line 1365)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1365)
    i_103648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 47), 'i', False)
    # Getting the type of 'pad_width' (line 1365)
    pad_width_103649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 37), 'pad_width', False)
    # Obtaining the member '__getitem__' of a type (line 1365)
    getitem___103650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1365, 37), pad_width_103649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1365)
    subscript_call_result_103651 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 37), getitem___103650, i_103648)
    
    # Processing the call keyword arguments (line 1365)
    kwargs_103652 = {}
    # Getting the type of 'np' (line 1365)
    np_103646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 30), 'np', False)
    # Obtaining the member 'sum' of a type (line 1365)
    sum_103647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1365, 30), np_103646, 'sum')
    # Calling sum(args, kwargs) (line 1365)
    sum_call_result_103653 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 30), sum_103647, *[subscript_call_result_103651], **kwargs_103652)
    
    list_103656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 30), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 30), list_103656, sum_call_result_103653)
    # Assigning a type to the variable 'total_dim_increase' (line 1365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1365, 8), 'total_dim_increase', list_103656)
    
    # Assigning a ListComp to a Name (line 1366):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'rank' (line 1368)
    rank_103680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 34), 'rank')
    comprehension_103681 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1366, 25), rank_103680)
    # Assigning a type to the variable 'i' (line 1366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1366, 25), 'i', comprehension_103681)
    
    # Call to slice(...): (line 1366)
    # Processing the call arguments (line 1366)
    
    # Obtaining the type of the subscript
    int_103658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 44), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1366)
    i_103659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 41), 'i', False)
    # Getting the type of 'pad_width' (line 1366)
    pad_width_103660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 31), 'pad_width', False)
    # Obtaining the member '__getitem__' of a type (line 1366)
    getitem___103661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1366, 31), pad_width_103660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1366)
    subscript_call_result_103662 = invoke(stypy.reporting.localization.Localization(__file__, 1366, 31), getitem___103661, i_103659)
    
    # Obtaining the member '__getitem__' of a type (line 1366)
    getitem___103663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1366, 31), subscript_call_result_103662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1366)
    subscript_call_result_103664 = invoke(stypy.reporting.localization.Localization(__file__, 1366, 31), getitem___103663, int_103658)
    
    
    # Obtaining the type of the subscript
    int_103665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 44), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1367)
    i_103666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 41), 'i', False)
    # Getting the type of 'pad_width' (line 1367)
    pad_width_103667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 31), 'pad_width', False)
    # Obtaining the member '__getitem__' of a type (line 1367)
    getitem___103668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 31), pad_width_103667, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1367)
    subscript_call_result_103669 = invoke(stypy.reporting.localization.Localization(__file__, 1367, 31), getitem___103668, i_103666)
    
    # Obtaining the member '__getitem__' of a type (line 1367)
    getitem___103670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 31), subscript_call_result_103669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1367)
    subscript_call_result_103671 = invoke(stypy.reporting.localization.Localization(__file__, 1367, 31), getitem___103670, int_103665)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1367)
    i_103672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 62), 'i', False)
    # Getting the type of 'narray' (line 1367)
    narray_103673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 49), 'narray', False)
    # Obtaining the member 'shape' of a type (line 1367)
    shape_103674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 49), narray_103673, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1367)
    getitem___103675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 49), shape_103674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1367)
    subscript_call_result_103676 = invoke(stypy.reporting.localization.Localization(__file__, 1367, 49), getitem___103675, i_103672)
    
    # Applying the binary operator '+' (line 1367)
    result_add_103677 = python_operator(stypy.reporting.localization.Localization(__file__, 1367, 31), '+', subscript_call_result_103671, subscript_call_result_103676)
    
    # Processing the call keyword arguments (line 1366)
    kwargs_103678 = {}
    # Getting the type of 'slice' (line 1366)
    slice_103657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 25), 'slice', False)
    # Calling slice(args, kwargs) (line 1366)
    slice_call_result_103679 = invoke(stypy.reporting.localization.Localization(__file__, 1366, 25), slice_103657, *[subscript_call_result_103664, result_add_103677], **kwargs_103678)
    
    list_103682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 25), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1366, 25), list_103682, slice_call_result_103679)
    # Assigning a type to the variable 'offset_slices' (line 1366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1366, 8), 'offset_slices', list_103682)
    
    # Assigning a BinOp to a Name (line 1369):
    
    # Call to array(...): (line 1369)
    # Processing the call arguments (line 1369)
    # Getting the type of 'narray' (line 1369)
    narray_103685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 29), 'narray', False)
    # Obtaining the member 'shape' of a type (line 1369)
    shape_103686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1369, 29), narray_103685, 'shape')
    # Processing the call keyword arguments (line 1369)
    kwargs_103687 = {}
    # Getting the type of 'np' (line 1369)
    np_103683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 1369)
    array_103684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1369, 20), np_103683, 'array')
    # Calling array(args, kwargs) (line 1369)
    array_call_result_103688 = invoke(stypy.reporting.localization.Localization(__file__, 1369, 20), array_103684, *[shape_103686], **kwargs_103687)
    
    # Getting the type of 'total_dim_increase' (line 1369)
    total_dim_increase_103689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 45), 'total_dim_increase')
    # Applying the binary operator '+' (line 1369)
    result_add_103690 = python_operator(stypy.reporting.localization.Localization(__file__, 1369, 20), '+', array_call_result_103688, total_dim_increase_103689)
    
    # Assigning a type to the variable 'new_shape' (line 1369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1369, 8), 'new_shape', result_add_103690)
    
    # Assigning a Call to a Name (line 1370):
    
    # Call to zeros(...): (line 1370)
    # Processing the call arguments (line 1370)
    # Getting the type of 'new_shape' (line 1370)
    new_shape_103693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 26), 'new_shape', False)
    # Getting the type of 'narray' (line 1370)
    narray_103694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 37), 'narray', False)
    # Obtaining the member 'dtype' of a type (line 1370)
    dtype_103695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1370, 37), narray_103694, 'dtype')
    # Processing the call keyword arguments (line 1370)
    kwargs_103696 = {}
    # Getting the type of 'np' (line 1370)
    np_103691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1370)
    zeros_103692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1370, 17), np_103691, 'zeros')
    # Calling zeros(args, kwargs) (line 1370)
    zeros_call_result_103697 = invoke(stypy.reporting.localization.Localization(__file__, 1370, 17), zeros_103692, *[new_shape_103693, dtype_103695], **kwargs_103696)
    
    # Assigning a type to the variable 'newmat' (line 1370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1370, 8), 'newmat', zeros_call_result_103697)
    
    # Assigning a Name to a Subscript (line 1373):
    # Getting the type of 'narray' (line 1373)
    narray_103698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 32), 'narray')
    # Getting the type of 'newmat' (line 1373)
    newmat_103699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 8), 'newmat')
    # Getting the type of 'offset_slices' (line 1373)
    offset_slices_103700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 15), 'offset_slices')
    # Storing an element on a container (line 1373)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1373, 8), newmat_103699, (offset_slices_103700, narray_103698))
    
    # Getting the type of 'rank' (line 1376)
    rank_103701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 21), 'rank')
    # Testing the type of a for loop iterable (line 1376)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1376, 8), rank_103701)
    # Getting the type of the for loop variable (line 1376)
    for_loop_var_103702 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1376, 8), rank_103701)
    # Assigning a type to the variable 'iaxis' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 8), 'iaxis', for_loop_var_103702)
    # SSA begins for a for statement (line 1376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to apply_along_axis(...): (line 1377)
    # Processing the call arguments (line 1377)
    # Getting the type of 'function' (line 1377)
    function_103705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 32), 'function', False)
    # Getting the type of 'iaxis' (line 1378)
    iaxis_103706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 32), 'iaxis', False)
    # Getting the type of 'newmat' (line 1379)
    newmat_103707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 32), 'newmat', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'iaxis' (line 1380)
    iaxis_103708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 42), 'iaxis', False)
    # Getting the type of 'pad_width' (line 1380)
    pad_width_103709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 32), 'pad_width', False)
    # Obtaining the member '__getitem__' of a type (line 1380)
    getitem___103710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 32), pad_width_103709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1380)
    subscript_call_result_103711 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 32), getitem___103710, iaxis_103708)
    
    # Getting the type of 'iaxis' (line 1381)
    iaxis_103712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 32), 'iaxis', False)
    # Getting the type of 'kwargs' (line 1382)
    kwargs_103713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1382, 32), 'kwargs', False)
    # Processing the call keyword arguments (line 1377)
    kwargs_103714 = {}
    # Getting the type of 'np' (line 1377)
    np_103703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1377, 12), 'np', False)
    # Obtaining the member 'apply_along_axis' of a type (line 1377)
    apply_along_axis_103704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1377, 12), np_103703, 'apply_along_axis')
    # Calling apply_along_axis(args, kwargs) (line 1377)
    apply_along_axis_call_result_103715 = invoke(stypy.reporting.localization.Localization(__file__, 1377, 12), apply_along_axis_103704, *[function_103705, iaxis_103706, newmat_103707, subscript_call_result_103711, iaxis_103712, kwargs_103713], **kwargs_103714)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newmat' (line 1383)
    newmat_103716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 15), 'newmat')
    # Assigning a type to the variable 'stypy_return_type' (line 1383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1383, 8), 'stypy_return_type', newmat_103716)
    # SSA join for if statement (line 1340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1386):
    
    # Call to copy(...): (line 1386)
    # Processing the call keyword arguments (line 1386)
    kwargs_103719 = {}
    # Getting the type of 'narray' (line 1386)
    narray_103717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 13), 'narray', False)
    # Obtaining the member 'copy' of a type (line 1386)
    copy_103718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 13), narray_103717, 'copy')
    # Calling copy(args, kwargs) (line 1386)
    copy_call_result_103720 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 13), copy_103718, *[], **kwargs_103719)
    
    # Assigning a type to the variable 'newmat' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 4), 'newmat', copy_call_result_103720)
    
    
    # Getting the type of 'mode' (line 1390)
    mode_103721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1390, 7), 'mode')
    str_103722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1390, 15), 'str', 'constant')
    # Applying the binary operator '==' (line 1390)
    result_eq_103723 = python_operator(stypy.reporting.localization.Localization(__file__, 1390, 7), '==', mode_103721, str_103722)
    
    # Testing the type of an if condition (line 1390)
    if_condition_103724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1390, 4), result_eq_103723)
    # Assigning a type to the variable 'if_condition_103724' (line 1390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1390, 4), 'if_condition_103724', if_condition_103724)
    # SSA begins for if statement (line 1390)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1392)
    # Processing the call arguments (line 1392)
    
    # Call to zip(...): (line 1392)
    # Processing the call arguments (line 1392)
    # Getting the type of 'pad_width' (line 1392)
    pad_width_103727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 51), 'str', 'constant_values')
    # Getting the type of 'kwargs' (line 1392)
    kwargs_103729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___103730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 44), kwargs_103729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1392)
    subscript_call_result_103731 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 44), getitem___103730, str_103728)
    
    # Processing the call keyword arguments (line 1392)
    kwargs_103732 = {}
    # Getting the type of 'zip' (line 1392)
    zip_103726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1392)
    zip_call_result_103733 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 29), zip_103726, *[pad_width_103727, subscript_call_result_103731], **kwargs_103732)
    
    # Processing the call keyword arguments (line 1392)
    kwargs_103734 = {}
    # Getting the type of 'enumerate' (line 1392)
    enumerate_103725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1392)
    enumerate_call_result_103735 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 19), enumerate_103725, *[zip_call_result_103733], **kwargs_103734)
    
    # Testing the type of a for loop iterable (line 1391)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1391, 8), enumerate_call_result_103735)
    # Getting the type of the for loop variable (line 1391)
    for_loop_var_103736 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1391, 8), enumerate_call_result_103735)
    # Assigning a type to the variable 'axis' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1391, 8), for_loop_var_103736))
    # Assigning a type to the variable 'pad_before' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1391, 8), for_loop_var_103736))
    # Assigning a type to the variable 'pad_after' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1391, 8), for_loop_var_103736))
    # Assigning a type to the variable 'before_val' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'before_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1391, 8), for_loop_var_103736))
    # Assigning a type to the variable 'after_val' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 8), 'after_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1391, 8), for_loop_var_103736))
    # SSA begins for a for statement (line 1391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1393):
    
    # Call to _prepend_const(...): (line 1393)
    # Processing the call arguments (line 1393)
    # Getting the type of 'newmat' (line 1393)
    newmat_103738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 36), 'newmat', False)
    # Getting the type of 'pad_before' (line 1393)
    pad_before_103739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 44), 'pad_before', False)
    # Getting the type of 'before_val' (line 1393)
    before_val_103740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 56), 'before_val', False)
    # Getting the type of 'axis' (line 1393)
    axis_103741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 68), 'axis', False)
    # Processing the call keyword arguments (line 1393)
    kwargs_103742 = {}
    # Getting the type of '_prepend_const' (line 1393)
    _prepend_const_103737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 21), '_prepend_const', False)
    # Calling _prepend_const(args, kwargs) (line 1393)
    _prepend_const_call_result_103743 = invoke(stypy.reporting.localization.Localization(__file__, 1393, 21), _prepend_const_103737, *[newmat_103738, pad_before_103739, before_val_103740, axis_103741], **kwargs_103742)
    
    # Assigning a type to the variable 'newmat' (line 1393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1393, 12), 'newmat', _prepend_const_call_result_103743)
    
    # Assigning a Call to a Name (line 1394):
    
    # Call to _append_const(...): (line 1394)
    # Processing the call arguments (line 1394)
    # Getting the type of 'newmat' (line 1394)
    newmat_103745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 35), 'newmat', False)
    # Getting the type of 'pad_after' (line 1394)
    pad_after_103746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 43), 'pad_after', False)
    # Getting the type of 'after_val' (line 1394)
    after_val_103747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 54), 'after_val', False)
    # Getting the type of 'axis' (line 1394)
    axis_103748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 65), 'axis', False)
    # Processing the call keyword arguments (line 1394)
    kwargs_103749 = {}
    # Getting the type of '_append_const' (line 1394)
    _append_const_103744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 21), '_append_const', False)
    # Calling _append_const(args, kwargs) (line 1394)
    _append_const_call_result_103750 = invoke(stypy.reporting.localization.Localization(__file__, 1394, 21), _append_const_103744, *[newmat_103745, pad_after_103746, after_val_103747, axis_103748], **kwargs_103749)
    
    # Assigning a type to the variable 'newmat' (line 1394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1394, 12), 'newmat', _append_const_call_result_103750)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1390)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1396)
    mode_103751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1396, 9), 'mode')
    str_103752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1396, 17), 'str', 'edge')
    # Applying the binary operator '==' (line 1396)
    result_eq_103753 = python_operator(stypy.reporting.localization.Localization(__file__, 1396, 9), '==', mode_103751, str_103752)
    
    # Testing the type of an if condition (line 1396)
    if_condition_103754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1396, 9), result_eq_103753)
    # Assigning a type to the variable 'if_condition_103754' (line 1396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1396, 9), 'if_condition_103754', if_condition_103754)
    # SSA begins for if statement (line 1396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1397)
    # Processing the call arguments (line 1397)
    # Getting the type of 'pad_width' (line 1397)
    pad_width_103756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 55), 'pad_width', False)
    # Processing the call keyword arguments (line 1397)
    kwargs_103757 = {}
    # Getting the type of 'enumerate' (line 1397)
    enumerate_103755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 45), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1397)
    enumerate_call_result_103758 = invoke(stypy.reporting.localization.Localization(__file__, 1397, 45), enumerate_103755, *[pad_width_103756], **kwargs_103757)
    
    # Testing the type of a for loop iterable (line 1397)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1397, 8), enumerate_call_result_103758)
    # Getting the type of the for loop variable (line 1397)
    for_loop_var_103759 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1397, 8), enumerate_call_result_103758)
    # Assigning a type to the variable 'axis' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1397, 8), for_loop_var_103759))
    # Assigning a type to the variable 'pad_before' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1397, 8), for_loop_var_103759))
    # Assigning a type to the variable 'pad_after' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1397, 8), for_loop_var_103759))
    # SSA begins for a for statement (line 1397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1398):
    
    # Call to _prepend_edge(...): (line 1398)
    # Processing the call arguments (line 1398)
    # Getting the type of 'newmat' (line 1398)
    newmat_103761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 35), 'newmat', False)
    # Getting the type of 'pad_before' (line 1398)
    pad_before_103762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 43), 'pad_before', False)
    # Getting the type of 'axis' (line 1398)
    axis_103763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 55), 'axis', False)
    # Processing the call keyword arguments (line 1398)
    kwargs_103764 = {}
    # Getting the type of '_prepend_edge' (line 1398)
    _prepend_edge_103760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 21), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 1398)
    _prepend_edge_call_result_103765 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 21), _prepend_edge_103760, *[newmat_103761, pad_before_103762, axis_103763], **kwargs_103764)
    
    # Assigning a type to the variable 'newmat' (line 1398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1398, 12), 'newmat', _prepend_edge_call_result_103765)
    
    # Assigning a Call to a Name (line 1399):
    
    # Call to _append_edge(...): (line 1399)
    # Processing the call arguments (line 1399)
    # Getting the type of 'newmat' (line 1399)
    newmat_103767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 34), 'newmat', False)
    # Getting the type of 'pad_after' (line 1399)
    pad_after_103768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 42), 'pad_after', False)
    # Getting the type of 'axis' (line 1399)
    axis_103769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 53), 'axis', False)
    # Processing the call keyword arguments (line 1399)
    kwargs_103770 = {}
    # Getting the type of '_append_edge' (line 1399)
    _append_edge_103766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 21), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 1399)
    _append_edge_call_result_103771 = invoke(stypy.reporting.localization.Localization(__file__, 1399, 21), _append_edge_103766, *[newmat_103767, pad_after_103768, axis_103769], **kwargs_103770)
    
    # Assigning a type to the variable 'newmat' (line 1399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1399, 12), 'newmat', _append_edge_call_result_103771)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1396)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1401)
    mode_103772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 9), 'mode')
    str_103773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1401, 17), 'str', 'linear_ramp')
    # Applying the binary operator '==' (line 1401)
    result_eq_103774 = python_operator(stypy.reporting.localization.Localization(__file__, 1401, 9), '==', mode_103772, str_103773)
    
    # Testing the type of an if condition (line 1401)
    if_condition_103775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1401, 9), result_eq_103774)
    # Assigning a type to the variable 'if_condition_103775' (line 1401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1401, 9), 'if_condition_103775', if_condition_103775)
    # SSA begins for if statement (line 1401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1403)
    # Processing the call arguments (line 1403)
    
    # Call to zip(...): (line 1403)
    # Processing the call arguments (line 1403)
    # Getting the type of 'pad_width' (line 1403)
    pad_width_103778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1403, 51), 'str', 'end_values')
    # Getting the type of 'kwargs' (line 1403)
    kwargs_103780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1403)
    getitem___103781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1403, 44), kwargs_103780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1403)
    subscript_call_result_103782 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 44), getitem___103781, str_103779)
    
    # Processing the call keyword arguments (line 1403)
    kwargs_103783 = {}
    # Getting the type of 'zip' (line 1403)
    zip_103777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1403)
    zip_call_result_103784 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 29), zip_103777, *[pad_width_103778, subscript_call_result_103782], **kwargs_103783)
    
    # Processing the call keyword arguments (line 1403)
    kwargs_103785 = {}
    # Getting the type of 'enumerate' (line 1403)
    enumerate_103776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1403)
    enumerate_call_result_103786 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 19), enumerate_103776, *[zip_call_result_103784], **kwargs_103785)
    
    # Testing the type of a for loop iterable (line 1402)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1402, 8), enumerate_call_result_103786)
    # Getting the type of the for loop variable (line 1402)
    for_loop_var_103787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1402, 8), enumerate_call_result_103786)
    # Assigning a type to the variable 'axis' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 8), for_loop_var_103787))
    # Assigning a type to the variable 'pad_before' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 8), for_loop_var_103787))
    # Assigning a type to the variable 'pad_after' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 8), for_loop_var_103787))
    # Assigning a type to the variable 'before_val' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'before_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 8), for_loop_var_103787))
    # Assigning a type to the variable 'after_val' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 8), 'after_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 8), for_loop_var_103787))
    # SSA begins for a for statement (line 1402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1404):
    
    # Call to _prepend_ramp(...): (line 1404)
    # Processing the call arguments (line 1404)
    # Getting the type of 'newmat' (line 1404)
    newmat_103789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 35), 'newmat', False)
    # Getting the type of 'pad_before' (line 1404)
    pad_before_103790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 43), 'pad_before', False)
    # Getting the type of 'before_val' (line 1404)
    before_val_103791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 55), 'before_val', False)
    # Getting the type of 'axis' (line 1404)
    axis_103792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 67), 'axis', False)
    # Processing the call keyword arguments (line 1404)
    kwargs_103793 = {}
    # Getting the type of '_prepend_ramp' (line 1404)
    _prepend_ramp_103788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 21), '_prepend_ramp', False)
    # Calling _prepend_ramp(args, kwargs) (line 1404)
    _prepend_ramp_call_result_103794 = invoke(stypy.reporting.localization.Localization(__file__, 1404, 21), _prepend_ramp_103788, *[newmat_103789, pad_before_103790, before_val_103791, axis_103792], **kwargs_103793)
    
    # Assigning a type to the variable 'newmat' (line 1404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1404, 12), 'newmat', _prepend_ramp_call_result_103794)
    
    # Assigning a Call to a Name (line 1405):
    
    # Call to _append_ramp(...): (line 1405)
    # Processing the call arguments (line 1405)
    # Getting the type of 'newmat' (line 1405)
    newmat_103796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 34), 'newmat', False)
    # Getting the type of 'pad_after' (line 1405)
    pad_after_103797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 42), 'pad_after', False)
    # Getting the type of 'after_val' (line 1405)
    after_val_103798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 53), 'after_val', False)
    # Getting the type of 'axis' (line 1405)
    axis_103799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 64), 'axis', False)
    # Processing the call keyword arguments (line 1405)
    kwargs_103800 = {}
    # Getting the type of '_append_ramp' (line 1405)
    _append_ramp_103795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 21), '_append_ramp', False)
    # Calling _append_ramp(args, kwargs) (line 1405)
    _append_ramp_call_result_103801 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 21), _append_ramp_103795, *[newmat_103796, pad_after_103797, after_val_103798, axis_103799], **kwargs_103800)
    
    # Assigning a type to the variable 'newmat' (line 1405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1405, 12), 'newmat', _append_ramp_call_result_103801)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1401)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1407)
    mode_103802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 9), 'mode')
    str_103803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1407, 17), 'str', 'maximum')
    # Applying the binary operator '==' (line 1407)
    result_eq_103804 = python_operator(stypy.reporting.localization.Localization(__file__, 1407, 9), '==', mode_103802, str_103803)
    
    # Testing the type of an if condition (line 1407)
    if_condition_103805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1407, 9), result_eq_103804)
    # Assigning a type to the variable 'if_condition_103805' (line 1407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1407, 9), 'if_condition_103805', if_condition_103805)
    # SSA begins for if statement (line 1407)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1409)
    # Processing the call arguments (line 1409)
    
    # Call to zip(...): (line 1409)
    # Processing the call arguments (line 1409)
    # Getting the type of 'pad_width' (line 1409)
    pad_width_103808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1409, 51), 'str', 'stat_length')
    # Getting the type of 'kwargs' (line 1409)
    kwargs_103810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1409)
    getitem___103811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1409, 44), kwargs_103810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1409)
    subscript_call_result_103812 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 44), getitem___103811, str_103809)
    
    # Processing the call keyword arguments (line 1409)
    kwargs_103813 = {}
    # Getting the type of 'zip' (line 1409)
    zip_103807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1409)
    zip_call_result_103814 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 29), zip_103807, *[pad_width_103808, subscript_call_result_103812], **kwargs_103813)
    
    # Processing the call keyword arguments (line 1409)
    kwargs_103815 = {}
    # Getting the type of 'enumerate' (line 1409)
    enumerate_103806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1409)
    enumerate_call_result_103816 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 19), enumerate_103806, *[zip_call_result_103814], **kwargs_103815)
    
    # Testing the type of a for loop iterable (line 1408)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1408, 8), enumerate_call_result_103816)
    # Getting the type of the for loop variable (line 1408)
    for_loop_var_103817 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1408, 8), enumerate_call_result_103816)
    # Assigning a type to the variable 'axis' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 8), for_loop_var_103817))
    # Assigning a type to the variable 'pad_before' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 8), for_loop_var_103817))
    # Assigning a type to the variable 'pad_after' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 8), for_loop_var_103817))
    # Assigning a type to the variable 'chunk_before' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'chunk_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 8), for_loop_var_103817))
    # Assigning a type to the variable 'chunk_after' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'chunk_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 8), for_loop_var_103817))
    # SSA begins for a for statement (line 1408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1410):
    
    # Call to _prepend_max(...): (line 1410)
    # Processing the call arguments (line 1410)
    # Getting the type of 'newmat' (line 1410)
    newmat_103819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 34), 'newmat', False)
    # Getting the type of 'pad_before' (line 1410)
    pad_before_103820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 42), 'pad_before', False)
    # Getting the type of 'chunk_before' (line 1410)
    chunk_before_103821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 54), 'chunk_before', False)
    # Getting the type of 'axis' (line 1410)
    axis_103822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 68), 'axis', False)
    # Processing the call keyword arguments (line 1410)
    kwargs_103823 = {}
    # Getting the type of '_prepend_max' (line 1410)
    _prepend_max_103818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 21), '_prepend_max', False)
    # Calling _prepend_max(args, kwargs) (line 1410)
    _prepend_max_call_result_103824 = invoke(stypy.reporting.localization.Localization(__file__, 1410, 21), _prepend_max_103818, *[newmat_103819, pad_before_103820, chunk_before_103821, axis_103822], **kwargs_103823)
    
    # Assigning a type to the variable 'newmat' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 12), 'newmat', _prepend_max_call_result_103824)
    
    # Assigning a Call to a Name (line 1411):
    
    # Call to _append_max(...): (line 1411)
    # Processing the call arguments (line 1411)
    # Getting the type of 'newmat' (line 1411)
    newmat_103826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 33), 'newmat', False)
    # Getting the type of 'pad_after' (line 1411)
    pad_after_103827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 41), 'pad_after', False)
    # Getting the type of 'chunk_after' (line 1411)
    chunk_after_103828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 52), 'chunk_after', False)
    # Getting the type of 'axis' (line 1411)
    axis_103829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 65), 'axis', False)
    # Processing the call keyword arguments (line 1411)
    kwargs_103830 = {}
    # Getting the type of '_append_max' (line 1411)
    _append_max_103825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 21), '_append_max', False)
    # Calling _append_max(args, kwargs) (line 1411)
    _append_max_call_result_103831 = invoke(stypy.reporting.localization.Localization(__file__, 1411, 21), _append_max_103825, *[newmat_103826, pad_after_103827, chunk_after_103828, axis_103829], **kwargs_103830)
    
    # Assigning a type to the variable 'newmat' (line 1411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1411, 12), 'newmat', _append_max_call_result_103831)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1407)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1413)
    mode_103832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 9), 'mode')
    str_103833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 17), 'str', 'mean')
    # Applying the binary operator '==' (line 1413)
    result_eq_103834 = python_operator(stypy.reporting.localization.Localization(__file__, 1413, 9), '==', mode_103832, str_103833)
    
    # Testing the type of an if condition (line 1413)
    if_condition_103835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1413, 9), result_eq_103834)
    # Assigning a type to the variable 'if_condition_103835' (line 1413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1413, 9), 'if_condition_103835', if_condition_103835)
    # SSA begins for if statement (line 1413)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1415)
    # Processing the call arguments (line 1415)
    
    # Call to zip(...): (line 1415)
    # Processing the call arguments (line 1415)
    # Getting the type of 'pad_width' (line 1415)
    pad_width_103838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1415, 51), 'str', 'stat_length')
    # Getting the type of 'kwargs' (line 1415)
    kwargs_103840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1415)
    getitem___103841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1415, 44), kwargs_103840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1415)
    subscript_call_result_103842 = invoke(stypy.reporting.localization.Localization(__file__, 1415, 44), getitem___103841, str_103839)
    
    # Processing the call keyword arguments (line 1415)
    kwargs_103843 = {}
    # Getting the type of 'zip' (line 1415)
    zip_103837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1415)
    zip_call_result_103844 = invoke(stypy.reporting.localization.Localization(__file__, 1415, 29), zip_103837, *[pad_width_103838, subscript_call_result_103842], **kwargs_103843)
    
    # Processing the call keyword arguments (line 1415)
    kwargs_103845 = {}
    # Getting the type of 'enumerate' (line 1415)
    enumerate_103836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1415)
    enumerate_call_result_103846 = invoke(stypy.reporting.localization.Localization(__file__, 1415, 19), enumerate_103836, *[zip_call_result_103844], **kwargs_103845)
    
    # Testing the type of a for loop iterable (line 1414)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1414, 8), enumerate_call_result_103846)
    # Getting the type of the for loop variable (line 1414)
    for_loop_var_103847 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1414, 8), enumerate_call_result_103846)
    # Assigning a type to the variable 'axis' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), for_loop_var_103847))
    # Assigning a type to the variable 'pad_before' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), for_loop_var_103847))
    # Assigning a type to the variable 'pad_after' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), for_loop_var_103847))
    # Assigning a type to the variable 'chunk_before' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'chunk_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), for_loop_var_103847))
    # Assigning a type to the variable 'chunk_after' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'chunk_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1414, 8), for_loop_var_103847))
    # SSA begins for a for statement (line 1414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1416):
    
    # Call to _prepend_mean(...): (line 1416)
    # Processing the call arguments (line 1416)
    # Getting the type of 'newmat' (line 1416)
    newmat_103849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 35), 'newmat', False)
    # Getting the type of 'pad_before' (line 1416)
    pad_before_103850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 43), 'pad_before', False)
    # Getting the type of 'chunk_before' (line 1416)
    chunk_before_103851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 55), 'chunk_before', False)
    # Getting the type of 'axis' (line 1416)
    axis_103852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 69), 'axis', False)
    # Processing the call keyword arguments (line 1416)
    kwargs_103853 = {}
    # Getting the type of '_prepend_mean' (line 1416)
    _prepend_mean_103848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 21), '_prepend_mean', False)
    # Calling _prepend_mean(args, kwargs) (line 1416)
    _prepend_mean_call_result_103854 = invoke(stypy.reporting.localization.Localization(__file__, 1416, 21), _prepend_mean_103848, *[newmat_103849, pad_before_103850, chunk_before_103851, axis_103852], **kwargs_103853)
    
    # Assigning a type to the variable 'newmat' (line 1416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1416, 12), 'newmat', _prepend_mean_call_result_103854)
    
    # Assigning a Call to a Name (line 1417):
    
    # Call to _append_mean(...): (line 1417)
    # Processing the call arguments (line 1417)
    # Getting the type of 'newmat' (line 1417)
    newmat_103856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 34), 'newmat', False)
    # Getting the type of 'pad_after' (line 1417)
    pad_after_103857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 42), 'pad_after', False)
    # Getting the type of 'chunk_after' (line 1417)
    chunk_after_103858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 53), 'chunk_after', False)
    # Getting the type of 'axis' (line 1417)
    axis_103859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 66), 'axis', False)
    # Processing the call keyword arguments (line 1417)
    kwargs_103860 = {}
    # Getting the type of '_append_mean' (line 1417)
    _append_mean_103855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 21), '_append_mean', False)
    # Calling _append_mean(args, kwargs) (line 1417)
    _append_mean_call_result_103861 = invoke(stypy.reporting.localization.Localization(__file__, 1417, 21), _append_mean_103855, *[newmat_103856, pad_after_103857, chunk_after_103858, axis_103859], **kwargs_103860)
    
    # Assigning a type to the variable 'newmat' (line 1417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1417, 12), 'newmat', _append_mean_call_result_103861)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1413)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1419)
    mode_103862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1419, 9), 'mode')
    str_103863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1419, 17), 'str', 'median')
    # Applying the binary operator '==' (line 1419)
    result_eq_103864 = python_operator(stypy.reporting.localization.Localization(__file__, 1419, 9), '==', mode_103862, str_103863)
    
    # Testing the type of an if condition (line 1419)
    if_condition_103865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1419, 9), result_eq_103864)
    # Assigning a type to the variable 'if_condition_103865' (line 1419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1419, 9), 'if_condition_103865', if_condition_103865)
    # SSA begins for if statement (line 1419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1421)
    # Processing the call arguments (line 1421)
    
    # Call to zip(...): (line 1421)
    # Processing the call arguments (line 1421)
    # Getting the type of 'pad_width' (line 1421)
    pad_width_103868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1421, 51), 'str', 'stat_length')
    # Getting the type of 'kwargs' (line 1421)
    kwargs_103870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1421)
    getitem___103871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 44), kwargs_103870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1421)
    subscript_call_result_103872 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 44), getitem___103871, str_103869)
    
    # Processing the call keyword arguments (line 1421)
    kwargs_103873 = {}
    # Getting the type of 'zip' (line 1421)
    zip_103867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1421)
    zip_call_result_103874 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 29), zip_103867, *[pad_width_103868, subscript_call_result_103872], **kwargs_103873)
    
    # Processing the call keyword arguments (line 1421)
    kwargs_103875 = {}
    # Getting the type of 'enumerate' (line 1421)
    enumerate_103866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1421)
    enumerate_call_result_103876 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 19), enumerate_103866, *[zip_call_result_103874], **kwargs_103875)
    
    # Testing the type of a for loop iterable (line 1420)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1420, 8), enumerate_call_result_103876)
    # Getting the type of the for loop variable (line 1420)
    for_loop_var_103877 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1420, 8), enumerate_call_result_103876)
    # Assigning a type to the variable 'axis' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 8), for_loop_var_103877))
    # Assigning a type to the variable 'pad_before' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 8), for_loop_var_103877))
    # Assigning a type to the variable 'pad_after' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 8), for_loop_var_103877))
    # Assigning a type to the variable 'chunk_before' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 8), 'chunk_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 8), for_loop_var_103877))
    # Assigning a type to the variable 'chunk_after' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 8), 'chunk_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 8), for_loop_var_103877))
    # SSA begins for a for statement (line 1420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1422):
    
    # Call to _prepend_med(...): (line 1422)
    # Processing the call arguments (line 1422)
    # Getting the type of 'newmat' (line 1422)
    newmat_103879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 34), 'newmat', False)
    # Getting the type of 'pad_before' (line 1422)
    pad_before_103880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 42), 'pad_before', False)
    # Getting the type of 'chunk_before' (line 1422)
    chunk_before_103881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 54), 'chunk_before', False)
    # Getting the type of 'axis' (line 1422)
    axis_103882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 68), 'axis', False)
    # Processing the call keyword arguments (line 1422)
    kwargs_103883 = {}
    # Getting the type of '_prepend_med' (line 1422)
    _prepend_med_103878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 21), '_prepend_med', False)
    # Calling _prepend_med(args, kwargs) (line 1422)
    _prepend_med_call_result_103884 = invoke(stypy.reporting.localization.Localization(__file__, 1422, 21), _prepend_med_103878, *[newmat_103879, pad_before_103880, chunk_before_103881, axis_103882], **kwargs_103883)
    
    # Assigning a type to the variable 'newmat' (line 1422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1422, 12), 'newmat', _prepend_med_call_result_103884)
    
    # Assigning a Call to a Name (line 1423):
    
    # Call to _append_med(...): (line 1423)
    # Processing the call arguments (line 1423)
    # Getting the type of 'newmat' (line 1423)
    newmat_103886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 33), 'newmat', False)
    # Getting the type of 'pad_after' (line 1423)
    pad_after_103887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 41), 'pad_after', False)
    # Getting the type of 'chunk_after' (line 1423)
    chunk_after_103888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 52), 'chunk_after', False)
    # Getting the type of 'axis' (line 1423)
    axis_103889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 65), 'axis', False)
    # Processing the call keyword arguments (line 1423)
    kwargs_103890 = {}
    # Getting the type of '_append_med' (line 1423)
    _append_med_103885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 21), '_append_med', False)
    # Calling _append_med(args, kwargs) (line 1423)
    _append_med_call_result_103891 = invoke(stypy.reporting.localization.Localization(__file__, 1423, 21), _append_med_103885, *[newmat_103886, pad_after_103887, chunk_after_103888, axis_103889], **kwargs_103890)
    
    # Assigning a type to the variable 'newmat' (line 1423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1423, 12), 'newmat', _append_med_call_result_103891)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1419)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1425)
    mode_103892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 9), 'mode')
    str_103893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, 17), 'str', 'minimum')
    # Applying the binary operator '==' (line 1425)
    result_eq_103894 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 9), '==', mode_103892, str_103893)
    
    # Testing the type of an if condition (line 1425)
    if_condition_103895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1425, 9), result_eq_103894)
    # Assigning a type to the variable 'if_condition_103895' (line 1425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1425, 9), 'if_condition_103895', if_condition_103895)
    # SSA begins for if statement (line 1425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1427)
    # Processing the call arguments (line 1427)
    
    # Call to zip(...): (line 1427)
    # Processing the call arguments (line 1427)
    # Getting the type of 'pad_width' (line 1427)
    pad_width_103898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 33), 'pad_width', False)
    
    # Obtaining the type of the subscript
    str_103899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 51), 'str', 'stat_length')
    # Getting the type of 'kwargs' (line 1427)
    kwargs_103900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 44), 'kwargs', False)
    # Obtaining the member '__getitem__' of a type (line 1427)
    getitem___103901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1427, 44), kwargs_103900, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1427)
    subscript_call_result_103902 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 44), getitem___103901, str_103899)
    
    # Processing the call keyword arguments (line 1427)
    kwargs_103903 = {}
    # Getting the type of 'zip' (line 1427)
    zip_103897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 29), 'zip', False)
    # Calling zip(args, kwargs) (line 1427)
    zip_call_result_103904 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 29), zip_103897, *[pad_width_103898, subscript_call_result_103902], **kwargs_103903)
    
    # Processing the call keyword arguments (line 1427)
    kwargs_103905 = {}
    # Getting the type of 'enumerate' (line 1427)
    enumerate_103896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1427)
    enumerate_call_result_103906 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 19), enumerate_103896, *[zip_call_result_103904], **kwargs_103905)
    
    # Testing the type of a for loop iterable (line 1426)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1426, 8), enumerate_call_result_103906)
    # Getting the type of the for loop variable (line 1426)
    for_loop_var_103907 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1426, 8), enumerate_call_result_103906)
    # Assigning a type to the variable 'axis' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 8), for_loop_var_103907))
    # Assigning a type to the variable 'pad_before' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 8), for_loop_var_103907))
    # Assigning a type to the variable 'pad_after' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 8), for_loop_var_103907))
    # Assigning a type to the variable 'chunk_before' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 8), 'chunk_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 8), for_loop_var_103907))
    # Assigning a type to the variable 'chunk_after' (line 1426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1426, 8), 'chunk_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1426, 8), for_loop_var_103907))
    # SSA begins for a for statement (line 1426)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1428):
    
    # Call to _prepend_min(...): (line 1428)
    # Processing the call arguments (line 1428)
    # Getting the type of 'newmat' (line 1428)
    newmat_103909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 34), 'newmat', False)
    # Getting the type of 'pad_before' (line 1428)
    pad_before_103910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 42), 'pad_before', False)
    # Getting the type of 'chunk_before' (line 1428)
    chunk_before_103911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 54), 'chunk_before', False)
    # Getting the type of 'axis' (line 1428)
    axis_103912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 68), 'axis', False)
    # Processing the call keyword arguments (line 1428)
    kwargs_103913 = {}
    # Getting the type of '_prepend_min' (line 1428)
    _prepend_min_103908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 21), '_prepend_min', False)
    # Calling _prepend_min(args, kwargs) (line 1428)
    _prepend_min_call_result_103914 = invoke(stypy.reporting.localization.Localization(__file__, 1428, 21), _prepend_min_103908, *[newmat_103909, pad_before_103910, chunk_before_103911, axis_103912], **kwargs_103913)
    
    # Assigning a type to the variable 'newmat' (line 1428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1428, 12), 'newmat', _prepend_min_call_result_103914)
    
    # Assigning a Call to a Name (line 1429):
    
    # Call to _append_min(...): (line 1429)
    # Processing the call arguments (line 1429)
    # Getting the type of 'newmat' (line 1429)
    newmat_103916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 33), 'newmat', False)
    # Getting the type of 'pad_after' (line 1429)
    pad_after_103917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 41), 'pad_after', False)
    # Getting the type of 'chunk_after' (line 1429)
    chunk_after_103918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 52), 'chunk_after', False)
    # Getting the type of 'axis' (line 1429)
    axis_103919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 65), 'axis', False)
    # Processing the call keyword arguments (line 1429)
    kwargs_103920 = {}
    # Getting the type of '_append_min' (line 1429)
    _append_min_103915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 21), '_append_min', False)
    # Calling _append_min(args, kwargs) (line 1429)
    _append_min_call_result_103921 = invoke(stypy.reporting.localization.Localization(__file__, 1429, 21), _append_min_103915, *[newmat_103916, pad_after_103917, chunk_after_103918, axis_103919], **kwargs_103920)
    
    # Assigning a type to the variable 'newmat' (line 1429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1429, 12), 'newmat', _append_min_call_result_103921)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1425)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1431)
    mode_103922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1431, 9), 'mode')
    str_103923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1431, 17), 'str', 'reflect')
    # Applying the binary operator '==' (line 1431)
    result_eq_103924 = python_operator(stypy.reporting.localization.Localization(__file__, 1431, 9), '==', mode_103922, str_103923)
    
    # Testing the type of an if condition (line 1431)
    if_condition_103925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1431, 9), result_eq_103924)
    # Assigning a type to the variable 'if_condition_103925' (line 1431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1431, 9), 'if_condition_103925', if_condition_103925)
    # SSA begins for if statement (line 1431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1432)
    # Processing the call arguments (line 1432)
    # Getting the type of 'pad_width' (line 1432)
    pad_width_103927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 55), 'pad_width', False)
    # Processing the call keyword arguments (line 1432)
    kwargs_103928 = {}
    # Getting the type of 'enumerate' (line 1432)
    enumerate_103926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 45), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1432)
    enumerate_call_result_103929 = invoke(stypy.reporting.localization.Localization(__file__, 1432, 45), enumerate_103926, *[pad_width_103927], **kwargs_103928)
    
    # Testing the type of a for loop iterable (line 1432)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1432, 8), enumerate_call_result_103929)
    # Getting the type of the for loop variable (line 1432)
    for_loop_var_103930 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1432, 8), enumerate_call_result_103929)
    # Assigning a type to the variable 'axis' (line 1432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1432, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1432, 8), for_loop_var_103930))
    # Assigning a type to the variable 'pad_before' (line 1432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1432, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1432, 8), for_loop_var_103930))
    # Assigning a type to the variable 'pad_after' (line 1432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1432, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1432, 8), for_loop_var_103930))
    # SSA begins for a for statement (line 1432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'pad_before' (line 1436)
    pad_before_103931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1436, 17), 'pad_before')
    int_103932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1436, 30), 'int')
    # Applying the binary operator '>' (line 1436)
    result_gt_103933 = python_operator(stypy.reporting.localization.Localization(__file__, 1436, 17), '>', pad_before_103931, int_103932)
    
    
    # Getting the type of 'pad_after' (line 1437)
    pad_after_103934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 21), 'pad_after')
    int_103935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1437, 33), 'int')
    # Applying the binary operator '>' (line 1437)
    result_gt_103936 = python_operator(stypy.reporting.localization.Localization(__file__, 1437, 21), '>', pad_after_103934, int_103935)
    
    # Applying the binary operator 'or' (line 1436)
    result_or_keyword_103937 = python_operator(stypy.reporting.localization.Localization(__file__, 1436, 16), 'or', result_gt_103933, result_gt_103936)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1437)
    axis_103938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 54), 'axis')
    # Getting the type of 'newmat' (line 1437)
    newmat_103939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 41), 'newmat')
    # Obtaining the member 'shape' of a type (line 1437)
    shape_103940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1437, 41), newmat_103939, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1437)
    getitem___103941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1437, 41), shape_103940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1437)
    subscript_call_result_103942 = invoke(stypy.reporting.localization.Localization(__file__, 1437, 41), getitem___103941, axis_103938)
    
    int_103943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1437, 63), 'int')
    # Applying the binary operator '==' (line 1437)
    result_eq_103944 = python_operator(stypy.reporting.localization.Localization(__file__, 1437, 41), '==', subscript_call_result_103942, int_103943)
    
    # Applying the binary operator 'and' (line 1436)
    result_and_keyword_103945 = python_operator(stypy.reporting.localization.Localization(__file__, 1436, 15), 'and', result_or_keyword_103937, result_eq_103944)
    
    # Testing the type of an if condition (line 1436)
    if_condition_103946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1436, 12), result_and_keyword_103945)
    # Assigning a type to the variable 'if_condition_103946' (line 1436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1436, 12), 'if_condition_103946', if_condition_103946)
    # SSA begins for if statement (line 1436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1440):
    
    # Call to _prepend_edge(...): (line 1440)
    # Processing the call arguments (line 1440)
    # Getting the type of 'newmat' (line 1440)
    newmat_103948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 39), 'newmat', False)
    # Getting the type of 'pad_before' (line 1440)
    pad_before_103949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 47), 'pad_before', False)
    # Getting the type of 'axis' (line 1440)
    axis_103950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 59), 'axis', False)
    # Processing the call keyword arguments (line 1440)
    kwargs_103951 = {}
    # Getting the type of '_prepend_edge' (line 1440)
    _prepend_edge_103947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 25), '_prepend_edge', False)
    # Calling _prepend_edge(args, kwargs) (line 1440)
    _prepend_edge_call_result_103952 = invoke(stypy.reporting.localization.Localization(__file__, 1440, 25), _prepend_edge_103947, *[newmat_103948, pad_before_103949, axis_103950], **kwargs_103951)
    
    # Assigning a type to the variable 'newmat' (line 1440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1440, 16), 'newmat', _prepend_edge_call_result_103952)
    
    # Assigning a Call to a Name (line 1441):
    
    # Call to _append_edge(...): (line 1441)
    # Processing the call arguments (line 1441)
    # Getting the type of 'newmat' (line 1441)
    newmat_103954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 38), 'newmat', False)
    # Getting the type of 'pad_after' (line 1441)
    pad_after_103955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 46), 'pad_after', False)
    # Getting the type of 'axis' (line 1441)
    axis_103956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 57), 'axis', False)
    # Processing the call keyword arguments (line 1441)
    kwargs_103957 = {}
    # Getting the type of '_append_edge' (line 1441)
    _append_edge_103953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1441, 25), '_append_edge', False)
    # Calling _append_edge(args, kwargs) (line 1441)
    _append_edge_call_result_103958 = invoke(stypy.reporting.localization.Localization(__file__, 1441, 25), _append_edge_103953, *[newmat_103954, pad_after_103955, axis_103956], **kwargs_103957)
    
    # Assigning a type to the variable 'newmat' (line 1441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1441, 16), 'newmat', _append_edge_call_result_103958)
    # SSA join for if statement (line 1436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1444):
    
    # Obtaining the type of the subscript
    str_103959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1444, 28), 'str', 'reflect_type')
    # Getting the type of 'kwargs' (line 1444)
    kwargs_103960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1444, 21), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 1444)
    getitem___103961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1444, 21), kwargs_103960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1444)
    subscript_call_result_103962 = invoke(stypy.reporting.localization.Localization(__file__, 1444, 21), getitem___103961, str_103959)
    
    # Assigning a type to the variable 'method' (line 1444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1444, 12), 'method', subscript_call_result_103962)
    
    # Assigning a BinOp to a Name (line 1445):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1445)
    axis_103963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 36), 'axis')
    # Getting the type of 'newmat' (line 1445)
    newmat_103964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 23), 'newmat')
    # Obtaining the member 'shape' of a type (line 1445)
    shape_103965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 23), newmat_103964, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___103966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 23), shape_103965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_103967 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 23), getitem___103966, axis_103963)
    
    int_103968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 44), 'int')
    # Applying the binary operator '-' (line 1445)
    result_sub_103969 = python_operator(stypy.reporting.localization.Localization(__file__, 1445, 23), '-', subscript_call_result_103967, int_103968)
    
    # Assigning a type to the variable 'safe_pad' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 12), 'safe_pad', result_sub_103969)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'pad_before' (line 1446)
    pad_before_103970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 20), 'pad_before')
    # Getting the type of 'safe_pad' (line 1446)
    safe_pad_103971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 33), 'safe_pad')
    # Applying the binary operator '>' (line 1446)
    result_gt_103972 = python_operator(stypy.reporting.localization.Localization(__file__, 1446, 20), '>', pad_before_103970, safe_pad_103971)
    
    
    # Getting the type of 'pad_after' (line 1446)
    pad_after_103973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 47), 'pad_after')
    # Getting the type of 'safe_pad' (line 1446)
    safe_pad_103974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 59), 'safe_pad')
    # Applying the binary operator '>' (line 1446)
    result_gt_103975 = python_operator(stypy.reporting.localization.Localization(__file__, 1446, 47), '>', pad_after_103973, safe_pad_103974)
    
    # Applying the binary operator 'or' (line 1446)
    result_or_keyword_103976 = python_operator(stypy.reporting.localization.Localization(__file__, 1446, 19), 'or', result_gt_103972, result_gt_103975)
    
    # Testing the type of an if condition (line 1446)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1446, 12), result_or_keyword_103976)
    # SSA begins for while statement (line 1446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 1447):
    
    # Call to min(...): (line 1447)
    # Processing the call arguments (line 1447)
    # Getting the type of 'safe_pad' (line 1447)
    safe_pad_103978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1448)
    safe_pad_103979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 33), 'safe_pad', False)
    # Getting the type of 'pad_before' (line 1448)
    pad_before_103980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 45), 'pad_before', False)
    # Getting the type of 'safe_pad' (line 1448)
    safe_pad_103981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 59), 'safe_pad', False)
    # Applying the binary operator '//' (line 1448)
    result_floordiv_103982 = python_operator(stypy.reporting.localization.Localization(__file__, 1448, 45), '//', pad_before_103980, safe_pad_103981)
    
    # Applying the binary operator '*' (line 1448)
    result_mul_103983 = python_operator(stypy.reporting.localization.Localization(__file__, 1448, 33), '*', safe_pad_103979, result_floordiv_103982)
    
    # Processing the call keyword arguments (line 1447)
    kwargs_103984 = {}
    # Getting the type of 'min' (line 1447)
    min_103977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 29), 'min', False)
    # Calling min(args, kwargs) (line 1447)
    min_call_result_103985 = invoke(stypy.reporting.localization.Localization(__file__, 1447, 29), min_103977, *[safe_pad_103978, result_mul_103983], **kwargs_103984)
    
    # Assigning a type to the variable 'pad_iter_b' (line 1447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1447, 16), 'pad_iter_b', min_call_result_103985)
    
    # Assigning a Call to a Name (line 1449):
    
    # Call to min(...): (line 1449)
    # Processing the call arguments (line 1449)
    # Getting the type of 'safe_pad' (line 1449)
    safe_pad_103987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1449)
    safe_pad_103988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 43), 'safe_pad', False)
    # Getting the type of 'pad_after' (line 1449)
    pad_after_103989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 55), 'pad_after', False)
    # Getting the type of 'safe_pad' (line 1449)
    safe_pad_103990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 68), 'safe_pad', False)
    # Applying the binary operator '//' (line 1449)
    result_floordiv_103991 = python_operator(stypy.reporting.localization.Localization(__file__, 1449, 55), '//', pad_after_103989, safe_pad_103990)
    
    # Applying the binary operator '*' (line 1449)
    result_mul_103992 = python_operator(stypy.reporting.localization.Localization(__file__, 1449, 43), '*', safe_pad_103988, result_floordiv_103991)
    
    # Processing the call keyword arguments (line 1449)
    kwargs_103993 = {}
    # Getting the type of 'min' (line 1449)
    min_103986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1449, 29), 'min', False)
    # Calling min(args, kwargs) (line 1449)
    min_call_result_103994 = invoke(stypy.reporting.localization.Localization(__file__, 1449, 29), min_103986, *[safe_pad_103987, result_mul_103992], **kwargs_103993)
    
    # Assigning a type to the variable 'pad_iter_a' (line 1449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1449, 16), 'pad_iter_a', min_call_result_103994)
    
    # Assigning a Call to a Name (line 1450):
    
    # Call to _pad_ref(...): (line 1450)
    # Processing the call arguments (line 1450)
    # Getting the type of 'newmat' (line 1450)
    newmat_103996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 34), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1450)
    tuple_103997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1450, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1450)
    # Adding element type (line 1450)
    # Getting the type of 'pad_iter_b' (line 1450)
    pad_iter_b_103998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 43), 'pad_iter_b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1450, 43), tuple_103997, pad_iter_b_103998)
    # Adding element type (line 1450)
    # Getting the type of 'pad_iter_a' (line 1451)
    pad_iter_a_103999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 43), 'pad_iter_a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1450, 43), tuple_103997, pad_iter_a_103999)
    
    # Getting the type of 'method' (line 1451)
    method_104000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 56), 'method', False)
    # Getting the type of 'axis' (line 1451)
    axis_104001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 64), 'axis', False)
    # Processing the call keyword arguments (line 1450)
    kwargs_104002 = {}
    # Getting the type of '_pad_ref' (line 1450)
    _pad_ref_103995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 25), '_pad_ref', False)
    # Calling _pad_ref(args, kwargs) (line 1450)
    _pad_ref_call_result_104003 = invoke(stypy.reporting.localization.Localization(__file__, 1450, 25), _pad_ref_103995, *[newmat_103996, tuple_103997, method_104000, axis_104001], **kwargs_104002)
    
    # Assigning a type to the variable 'newmat' (line 1450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1450, 16), 'newmat', _pad_ref_call_result_104003)
    
    # Getting the type of 'pad_before' (line 1452)
    pad_before_104004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1452, 16), 'pad_before')
    # Getting the type of 'pad_iter_b' (line 1452)
    pad_iter_b_104005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1452, 30), 'pad_iter_b')
    # Applying the binary operator '-=' (line 1452)
    result_isub_104006 = python_operator(stypy.reporting.localization.Localization(__file__, 1452, 16), '-=', pad_before_104004, pad_iter_b_104005)
    # Assigning a type to the variable 'pad_before' (line 1452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1452, 16), 'pad_before', result_isub_104006)
    
    
    # Getting the type of 'pad_after' (line 1453)
    pad_after_104007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1453, 16), 'pad_after')
    # Getting the type of 'pad_iter_a' (line 1453)
    pad_iter_a_104008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1453, 29), 'pad_iter_a')
    # Applying the binary operator '-=' (line 1453)
    result_isub_104009 = python_operator(stypy.reporting.localization.Localization(__file__, 1453, 16), '-=', pad_after_104007, pad_iter_a_104008)
    # Assigning a type to the variable 'pad_after' (line 1453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1453, 16), 'pad_after', result_isub_104009)
    
    
    # Getting the type of 'safe_pad' (line 1454)
    safe_pad_104010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1454, 16), 'safe_pad')
    # Getting the type of 'pad_iter_b' (line 1454)
    pad_iter_b_104011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1454, 28), 'pad_iter_b')
    # Getting the type of 'pad_iter_a' (line 1454)
    pad_iter_a_104012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1454, 41), 'pad_iter_a')
    # Applying the binary operator '+' (line 1454)
    result_add_104013 = python_operator(stypy.reporting.localization.Localization(__file__, 1454, 28), '+', pad_iter_b_104011, pad_iter_a_104012)
    
    # Applying the binary operator '+=' (line 1454)
    result_iadd_104014 = python_operator(stypy.reporting.localization.Localization(__file__, 1454, 16), '+=', safe_pad_104010, result_add_104013)
    # Assigning a type to the variable 'safe_pad' (line 1454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1454, 16), 'safe_pad', result_iadd_104014)
    
    # SSA join for while statement (line 1446)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1455):
    
    # Call to _pad_ref(...): (line 1455)
    # Processing the call arguments (line 1455)
    # Getting the type of 'newmat' (line 1455)
    newmat_104016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 30), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1455)
    tuple_104017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1455)
    # Adding element type (line 1455)
    # Getting the type of 'pad_before' (line 1455)
    pad_before_104018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 39), 'pad_before', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1455, 39), tuple_104017, pad_before_104018)
    # Adding element type (line 1455)
    # Getting the type of 'pad_after' (line 1455)
    pad_after_104019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 51), 'pad_after', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1455, 39), tuple_104017, pad_after_104019)
    
    # Getting the type of 'method' (line 1455)
    method_104020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 63), 'method', False)
    # Getting the type of 'axis' (line 1455)
    axis_104021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 71), 'axis', False)
    # Processing the call keyword arguments (line 1455)
    kwargs_104022 = {}
    # Getting the type of '_pad_ref' (line 1455)
    _pad_ref_104015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 21), '_pad_ref', False)
    # Calling _pad_ref(args, kwargs) (line 1455)
    _pad_ref_call_result_104023 = invoke(stypy.reporting.localization.Localization(__file__, 1455, 21), _pad_ref_104015, *[newmat_104016, tuple_104017, method_104020, axis_104021], **kwargs_104022)
    
    # Assigning a type to the variable 'newmat' (line 1455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1455, 12), 'newmat', _pad_ref_call_result_104023)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1431)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1457)
    mode_104024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1457, 9), 'mode')
    str_104025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 17), 'str', 'symmetric')
    # Applying the binary operator '==' (line 1457)
    result_eq_104026 = python_operator(stypy.reporting.localization.Localization(__file__, 1457, 9), '==', mode_104024, str_104025)
    
    # Testing the type of an if condition (line 1457)
    if_condition_104027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1457, 9), result_eq_104026)
    # Assigning a type to the variable 'if_condition_104027' (line 1457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1457, 9), 'if_condition_104027', if_condition_104027)
    # SSA begins for if statement (line 1457)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1458)
    # Processing the call arguments (line 1458)
    # Getting the type of 'pad_width' (line 1458)
    pad_width_104029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 55), 'pad_width', False)
    # Processing the call keyword arguments (line 1458)
    kwargs_104030 = {}
    # Getting the type of 'enumerate' (line 1458)
    enumerate_104028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 45), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1458)
    enumerate_call_result_104031 = invoke(stypy.reporting.localization.Localization(__file__, 1458, 45), enumerate_104028, *[pad_width_104029], **kwargs_104030)
    
    # Testing the type of a for loop iterable (line 1458)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1458, 8), enumerate_call_result_104031)
    # Getting the type of the for loop variable (line 1458)
    for_loop_var_104032 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1458, 8), enumerate_call_result_104031)
    # Assigning a type to the variable 'axis' (line 1458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1458, 8), for_loop_var_104032))
    # Assigning a type to the variable 'pad_before' (line 1458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1458, 8), for_loop_var_104032))
    # Assigning a type to the variable 'pad_after' (line 1458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1458, 8), for_loop_var_104032))
    # SSA begins for a for statement (line 1458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 1462):
    
    # Obtaining the type of the subscript
    str_104033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1462, 28), 'str', 'reflect_type')
    # Getting the type of 'kwargs' (line 1462)
    kwargs_104034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 21), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 1462)
    getitem___104035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1462, 21), kwargs_104034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1462)
    subscript_call_result_104036 = invoke(stypy.reporting.localization.Localization(__file__, 1462, 21), getitem___104035, str_104033)
    
    # Assigning a type to the variable 'method' (line 1462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1462, 12), 'method', subscript_call_result_104036)
    
    # Assigning a Subscript to a Name (line 1463):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1463)
    axis_104037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 36), 'axis')
    # Getting the type of 'newmat' (line 1463)
    newmat_104038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 23), 'newmat')
    # Obtaining the member 'shape' of a type (line 1463)
    shape_104039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1463, 23), newmat_104038, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1463)
    getitem___104040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1463, 23), shape_104039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1463)
    subscript_call_result_104041 = invoke(stypy.reporting.localization.Localization(__file__, 1463, 23), getitem___104040, axis_104037)
    
    # Assigning a type to the variable 'safe_pad' (line 1463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1463, 12), 'safe_pad', subscript_call_result_104041)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'pad_before' (line 1464)
    pad_before_104042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1464, 20), 'pad_before')
    # Getting the type of 'safe_pad' (line 1464)
    safe_pad_104043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1464, 33), 'safe_pad')
    # Applying the binary operator '>' (line 1464)
    result_gt_104044 = python_operator(stypy.reporting.localization.Localization(__file__, 1464, 20), '>', pad_before_104042, safe_pad_104043)
    
    
    # Getting the type of 'pad_after' (line 1465)
    pad_after_104045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 20), 'pad_after')
    # Getting the type of 'safe_pad' (line 1465)
    safe_pad_104046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 32), 'safe_pad')
    # Applying the binary operator '>' (line 1465)
    result_gt_104047 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 20), '>', pad_after_104045, safe_pad_104046)
    
    # Applying the binary operator 'or' (line 1464)
    result_or_keyword_104048 = python_operator(stypy.reporting.localization.Localization(__file__, 1464, 19), 'or', result_gt_104044, result_gt_104047)
    
    # Testing the type of an if condition (line 1464)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1464, 12), result_or_keyword_104048)
    # SSA begins for while statement (line 1464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 1466):
    
    # Call to min(...): (line 1466)
    # Processing the call arguments (line 1466)
    # Getting the type of 'safe_pad' (line 1466)
    safe_pad_104050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1467)
    safe_pad_104051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1467, 33), 'safe_pad', False)
    # Getting the type of 'pad_before' (line 1467)
    pad_before_104052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1467, 45), 'pad_before', False)
    # Getting the type of 'safe_pad' (line 1467)
    safe_pad_104053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1467, 59), 'safe_pad', False)
    # Applying the binary operator '//' (line 1467)
    result_floordiv_104054 = python_operator(stypy.reporting.localization.Localization(__file__, 1467, 45), '//', pad_before_104052, safe_pad_104053)
    
    # Applying the binary operator '*' (line 1467)
    result_mul_104055 = python_operator(stypy.reporting.localization.Localization(__file__, 1467, 33), '*', safe_pad_104051, result_floordiv_104054)
    
    # Processing the call keyword arguments (line 1466)
    kwargs_104056 = {}
    # Getting the type of 'min' (line 1466)
    min_104049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1466, 29), 'min', False)
    # Calling min(args, kwargs) (line 1466)
    min_call_result_104057 = invoke(stypy.reporting.localization.Localization(__file__, 1466, 29), min_104049, *[safe_pad_104050, result_mul_104055], **kwargs_104056)
    
    # Assigning a type to the variable 'pad_iter_b' (line 1466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1466, 16), 'pad_iter_b', min_call_result_104057)
    
    # Assigning a Call to a Name (line 1468):
    
    # Call to min(...): (line 1468)
    # Processing the call arguments (line 1468)
    # Getting the type of 'safe_pad' (line 1468)
    safe_pad_104059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1468)
    safe_pad_104060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 43), 'safe_pad', False)
    # Getting the type of 'pad_after' (line 1468)
    pad_after_104061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 55), 'pad_after', False)
    # Getting the type of 'safe_pad' (line 1468)
    safe_pad_104062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 68), 'safe_pad', False)
    # Applying the binary operator '//' (line 1468)
    result_floordiv_104063 = python_operator(stypy.reporting.localization.Localization(__file__, 1468, 55), '//', pad_after_104061, safe_pad_104062)
    
    # Applying the binary operator '*' (line 1468)
    result_mul_104064 = python_operator(stypy.reporting.localization.Localization(__file__, 1468, 43), '*', safe_pad_104060, result_floordiv_104063)
    
    # Processing the call keyword arguments (line 1468)
    kwargs_104065 = {}
    # Getting the type of 'min' (line 1468)
    min_104058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 29), 'min', False)
    # Calling min(args, kwargs) (line 1468)
    min_call_result_104066 = invoke(stypy.reporting.localization.Localization(__file__, 1468, 29), min_104058, *[safe_pad_104059, result_mul_104064], **kwargs_104065)
    
    # Assigning a type to the variable 'pad_iter_a' (line 1468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1468, 16), 'pad_iter_a', min_call_result_104066)
    
    # Assigning a Call to a Name (line 1469):
    
    # Call to _pad_sym(...): (line 1469)
    # Processing the call arguments (line 1469)
    # Getting the type of 'newmat' (line 1469)
    newmat_104068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1469, 34), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1469)
    tuple_104069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1469, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1469)
    # Adding element type (line 1469)
    # Getting the type of 'pad_iter_b' (line 1469)
    pad_iter_b_104070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1469, 43), 'pad_iter_b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1469, 43), tuple_104069, pad_iter_b_104070)
    # Adding element type (line 1469)
    # Getting the type of 'pad_iter_a' (line 1470)
    pad_iter_a_104071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 43), 'pad_iter_a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1469, 43), tuple_104069, pad_iter_a_104071)
    
    # Getting the type of 'method' (line 1470)
    method_104072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 56), 'method', False)
    # Getting the type of 'axis' (line 1470)
    axis_104073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 64), 'axis', False)
    # Processing the call keyword arguments (line 1469)
    kwargs_104074 = {}
    # Getting the type of '_pad_sym' (line 1469)
    _pad_sym_104067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1469, 25), '_pad_sym', False)
    # Calling _pad_sym(args, kwargs) (line 1469)
    _pad_sym_call_result_104075 = invoke(stypy.reporting.localization.Localization(__file__, 1469, 25), _pad_sym_104067, *[newmat_104068, tuple_104069, method_104072, axis_104073], **kwargs_104074)
    
    # Assigning a type to the variable 'newmat' (line 1469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1469, 16), 'newmat', _pad_sym_call_result_104075)
    
    # Getting the type of 'pad_before' (line 1471)
    pad_before_104076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1471, 16), 'pad_before')
    # Getting the type of 'pad_iter_b' (line 1471)
    pad_iter_b_104077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1471, 30), 'pad_iter_b')
    # Applying the binary operator '-=' (line 1471)
    result_isub_104078 = python_operator(stypy.reporting.localization.Localization(__file__, 1471, 16), '-=', pad_before_104076, pad_iter_b_104077)
    # Assigning a type to the variable 'pad_before' (line 1471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1471, 16), 'pad_before', result_isub_104078)
    
    
    # Getting the type of 'pad_after' (line 1472)
    pad_after_104079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 16), 'pad_after')
    # Getting the type of 'pad_iter_a' (line 1472)
    pad_iter_a_104080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1472, 29), 'pad_iter_a')
    # Applying the binary operator '-=' (line 1472)
    result_isub_104081 = python_operator(stypy.reporting.localization.Localization(__file__, 1472, 16), '-=', pad_after_104079, pad_iter_a_104080)
    # Assigning a type to the variable 'pad_after' (line 1472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1472, 16), 'pad_after', result_isub_104081)
    
    
    # Getting the type of 'safe_pad' (line 1473)
    safe_pad_104082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 16), 'safe_pad')
    # Getting the type of 'pad_iter_b' (line 1473)
    pad_iter_b_104083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 28), 'pad_iter_b')
    # Getting the type of 'pad_iter_a' (line 1473)
    pad_iter_a_104084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 41), 'pad_iter_a')
    # Applying the binary operator '+' (line 1473)
    result_add_104085 = python_operator(stypy.reporting.localization.Localization(__file__, 1473, 28), '+', pad_iter_b_104083, pad_iter_a_104084)
    
    # Applying the binary operator '+=' (line 1473)
    result_iadd_104086 = python_operator(stypy.reporting.localization.Localization(__file__, 1473, 16), '+=', safe_pad_104082, result_add_104085)
    # Assigning a type to the variable 'safe_pad' (line 1473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1473, 16), 'safe_pad', result_iadd_104086)
    
    # SSA join for while statement (line 1464)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1474):
    
    # Call to _pad_sym(...): (line 1474)
    # Processing the call arguments (line 1474)
    # Getting the type of 'newmat' (line 1474)
    newmat_104088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 30), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1474)
    tuple_104089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1474, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1474)
    # Adding element type (line 1474)
    # Getting the type of 'pad_before' (line 1474)
    pad_before_104090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 39), 'pad_before', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1474, 39), tuple_104089, pad_before_104090)
    # Adding element type (line 1474)
    # Getting the type of 'pad_after' (line 1474)
    pad_after_104091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 51), 'pad_after', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1474, 39), tuple_104089, pad_after_104091)
    
    # Getting the type of 'method' (line 1474)
    method_104092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 63), 'method', False)
    # Getting the type of 'axis' (line 1474)
    axis_104093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 71), 'axis', False)
    # Processing the call keyword arguments (line 1474)
    kwargs_104094 = {}
    # Getting the type of '_pad_sym' (line 1474)
    _pad_sym_104087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 21), '_pad_sym', False)
    # Calling _pad_sym(args, kwargs) (line 1474)
    _pad_sym_call_result_104095 = invoke(stypy.reporting.localization.Localization(__file__, 1474, 21), _pad_sym_104087, *[newmat_104088, tuple_104089, method_104092, axis_104093], **kwargs_104094)
    
    # Assigning a type to the variable 'newmat' (line 1474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1474, 12), 'newmat', _pad_sym_call_result_104095)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1457)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1476)
    mode_104096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 9), 'mode')
    str_104097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1476, 17), 'str', 'wrap')
    # Applying the binary operator '==' (line 1476)
    result_eq_104098 = python_operator(stypy.reporting.localization.Localization(__file__, 1476, 9), '==', mode_104096, str_104097)
    
    # Testing the type of an if condition (line 1476)
    if_condition_104099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1476, 9), result_eq_104098)
    # Assigning a type to the variable 'if_condition_104099' (line 1476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1476, 9), 'if_condition_104099', if_condition_104099)
    # SSA begins for if statement (line 1476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1477)
    # Processing the call arguments (line 1477)
    # Getting the type of 'pad_width' (line 1477)
    pad_width_104101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 55), 'pad_width', False)
    # Processing the call keyword arguments (line 1477)
    kwargs_104102 = {}
    # Getting the type of 'enumerate' (line 1477)
    enumerate_104100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 45), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1477)
    enumerate_call_result_104103 = invoke(stypy.reporting.localization.Localization(__file__, 1477, 45), enumerate_104100, *[pad_width_104101], **kwargs_104102)
    
    # Testing the type of a for loop iterable (line 1477)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1477, 8), enumerate_call_result_104103)
    # Getting the type of the for loop variable (line 1477)
    for_loop_var_104104 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1477, 8), enumerate_call_result_104103)
    # Assigning a type to the variable 'axis' (line 1477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1477, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1477, 8), for_loop_var_104104))
    # Assigning a type to the variable 'pad_before' (line 1477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1477, 8), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1477, 8), for_loop_var_104104))
    # Assigning a type to the variable 'pad_after' (line 1477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1477, 8), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1477, 8), for_loop_var_104104))
    # SSA begins for a for statement (line 1477)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 1481):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1481)
    axis_104105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1481, 36), 'axis')
    # Getting the type of 'newmat' (line 1481)
    newmat_104106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1481, 23), 'newmat')
    # Obtaining the member 'shape' of a type (line 1481)
    shape_104107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1481, 23), newmat_104106, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1481)
    getitem___104108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1481, 23), shape_104107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1481)
    subscript_call_result_104109 = invoke(stypy.reporting.localization.Localization(__file__, 1481, 23), getitem___104108, axis_104105)
    
    # Assigning a type to the variable 'safe_pad' (line 1481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1481, 12), 'safe_pad', subscript_call_result_104109)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'pad_before' (line 1482)
    pad_before_104110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1482, 20), 'pad_before')
    # Getting the type of 'safe_pad' (line 1482)
    safe_pad_104111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1482, 33), 'safe_pad')
    # Applying the binary operator '>' (line 1482)
    result_gt_104112 = python_operator(stypy.reporting.localization.Localization(__file__, 1482, 20), '>', pad_before_104110, safe_pad_104111)
    
    
    # Getting the type of 'pad_after' (line 1483)
    pad_after_104113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1483, 20), 'pad_after')
    # Getting the type of 'safe_pad' (line 1483)
    safe_pad_104114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1483, 32), 'safe_pad')
    # Applying the binary operator '>' (line 1483)
    result_gt_104115 = python_operator(stypy.reporting.localization.Localization(__file__, 1483, 20), '>', pad_after_104113, safe_pad_104114)
    
    # Applying the binary operator 'or' (line 1482)
    result_or_keyword_104116 = python_operator(stypy.reporting.localization.Localization(__file__, 1482, 19), 'or', result_gt_104112, result_gt_104115)
    
    # Testing the type of an if condition (line 1482)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1482, 12), result_or_keyword_104116)
    # SSA begins for while statement (line 1482)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 1484):
    
    # Call to min(...): (line 1484)
    # Processing the call arguments (line 1484)
    # Getting the type of 'safe_pad' (line 1484)
    safe_pad_104118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1485)
    safe_pad_104119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 33), 'safe_pad', False)
    # Getting the type of 'pad_before' (line 1485)
    pad_before_104120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 45), 'pad_before', False)
    # Getting the type of 'safe_pad' (line 1485)
    safe_pad_104121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 59), 'safe_pad', False)
    # Applying the binary operator '//' (line 1485)
    result_floordiv_104122 = python_operator(stypy.reporting.localization.Localization(__file__, 1485, 45), '//', pad_before_104120, safe_pad_104121)
    
    # Applying the binary operator '*' (line 1485)
    result_mul_104123 = python_operator(stypy.reporting.localization.Localization(__file__, 1485, 33), '*', safe_pad_104119, result_floordiv_104122)
    
    # Processing the call keyword arguments (line 1484)
    kwargs_104124 = {}
    # Getting the type of 'min' (line 1484)
    min_104117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 29), 'min', False)
    # Calling min(args, kwargs) (line 1484)
    min_call_result_104125 = invoke(stypy.reporting.localization.Localization(__file__, 1484, 29), min_104117, *[safe_pad_104118, result_mul_104123], **kwargs_104124)
    
    # Assigning a type to the variable 'pad_iter_b' (line 1484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1484, 16), 'pad_iter_b', min_call_result_104125)
    
    # Assigning a Call to a Name (line 1486):
    
    # Call to min(...): (line 1486)
    # Processing the call arguments (line 1486)
    # Getting the type of 'safe_pad' (line 1486)
    safe_pad_104127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 33), 'safe_pad', False)
    # Getting the type of 'safe_pad' (line 1486)
    safe_pad_104128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 43), 'safe_pad', False)
    # Getting the type of 'pad_after' (line 1486)
    pad_after_104129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 55), 'pad_after', False)
    # Getting the type of 'safe_pad' (line 1486)
    safe_pad_104130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 68), 'safe_pad', False)
    # Applying the binary operator '//' (line 1486)
    result_floordiv_104131 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 55), '//', pad_after_104129, safe_pad_104130)
    
    # Applying the binary operator '*' (line 1486)
    result_mul_104132 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 43), '*', safe_pad_104128, result_floordiv_104131)
    
    # Processing the call keyword arguments (line 1486)
    kwargs_104133 = {}
    # Getting the type of 'min' (line 1486)
    min_104126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 29), 'min', False)
    # Calling min(args, kwargs) (line 1486)
    min_call_result_104134 = invoke(stypy.reporting.localization.Localization(__file__, 1486, 29), min_104126, *[safe_pad_104127, result_mul_104132], **kwargs_104133)
    
    # Assigning a type to the variable 'pad_iter_a' (line 1486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1486, 16), 'pad_iter_a', min_call_result_104134)
    
    # Assigning a Call to a Name (line 1487):
    
    # Call to _pad_wrap(...): (line 1487)
    # Processing the call arguments (line 1487)
    # Getting the type of 'newmat' (line 1487)
    newmat_104136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 35), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1487)
    tuple_104137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1487)
    # Adding element type (line 1487)
    # Getting the type of 'pad_iter_b' (line 1487)
    pad_iter_b_104138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 44), 'pad_iter_b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1487, 44), tuple_104137, pad_iter_b_104138)
    # Adding element type (line 1487)
    # Getting the type of 'pad_iter_a' (line 1487)
    pad_iter_a_104139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 56), 'pad_iter_a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1487, 44), tuple_104137, pad_iter_a_104139)
    
    # Getting the type of 'axis' (line 1487)
    axis_104140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 69), 'axis', False)
    # Processing the call keyword arguments (line 1487)
    kwargs_104141 = {}
    # Getting the type of '_pad_wrap' (line 1487)
    _pad_wrap_104135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 25), '_pad_wrap', False)
    # Calling _pad_wrap(args, kwargs) (line 1487)
    _pad_wrap_call_result_104142 = invoke(stypy.reporting.localization.Localization(__file__, 1487, 25), _pad_wrap_104135, *[newmat_104136, tuple_104137, axis_104140], **kwargs_104141)
    
    # Assigning a type to the variable 'newmat' (line 1487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1487, 16), 'newmat', _pad_wrap_call_result_104142)
    
    # Getting the type of 'pad_before' (line 1489)
    pad_before_104143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 16), 'pad_before')
    # Getting the type of 'pad_iter_b' (line 1489)
    pad_iter_b_104144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 30), 'pad_iter_b')
    # Applying the binary operator '-=' (line 1489)
    result_isub_104145 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 16), '-=', pad_before_104143, pad_iter_b_104144)
    # Assigning a type to the variable 'pad_before' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 16), 'pad_before', result_isub_104145)
    
    
    # Getting the type of 'pad_after' (line 1490)
    pad_after_104146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 16), 'pad_after')
    # Getting the type of 'pad_iter_a' (line 1490)
    pad_iter_a_104147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 29), 'pad_iter_a')
    # Applying the binary operator '-=' (line 1490)
    result_isub_104148 = python_operator(stypy.reporting.localization.Localization(__file__, 1490, 16), '-=', pad_after_104146, pad_iter_a_104147)
    # Assigning a type to the variable 'pad_after' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 16), 'pad_after', result_isub_104148)
    
    
    # Getting the type of 'safe_pad' (line 1491)
    safe_pad_104149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 16), 'safe_pad')
    # Getting the type of 'pad_iter_b' (line 1491)
    pad_iter_b_104150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 28), 'pad_iter_b')
    # Getting the type of 'pad_iter_a' (line 1491)
    pad_iter_a_104151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 41), 'pad_iter_a')
    # Applying the binary operator '+' (line 1491)
    result_add_104152 = python_operator(stypy.reporting.localization.Localization(__file__, 1491, 28), '+', pad_iter_b_104150, pad_iter_a_104151)
    
    # Applying the binary operator '+=' (line 1491)
    result_iadd_104153 = python_operator(stypy.reporting.localization.Localization(__file__, 1491, 16), '+=', safe_pad_104149, result_add_104152)
    # Assigning a type to the variable 'safe_pad' (line 1491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1491, 16), 'safe_pad', result_iadd_104153)
    
    # SSA join for while statement (line 1482)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1492):
    
    # Call to _pad_wrap(...): (line 1492)
    # Processing the call arguments (line 1492)
    # Getting the type of 'newmat' (line 1492)
    newmat_104155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 31), 'newmat', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1492)
    tuple_104156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1492, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1492)
    # Adding element type (line 1492)
    # Getting the type of 'pad_before' (line 1492)
    pad_before_104157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 40), 'pad_before', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1492, 40), tuple_104156, pad_before_104157)
    # Adding element type (line 1492)
    # Getting the type of 'pad_after' (line 1492)
    pad_after_104158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 52), 'pad_after', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1492, 40), tuple_104156, pad_after_104158)
    
    # Getting the type of 'axis' (line 1492)
    axis_104159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 64), 'axis', False)
    # Processing the call keyword arguments (line 1492)
    kwargs_104160 = {}
    # Getting the type of '_pad_wrap' (line 1492)
    _pad_wrap_104154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 21), '_pad_wrap', False)
    # Calling _pad_wrap(args, kwargs) (line 1492)
    _pad_wrap_call_result_104161 = invoke(stypy.reporting.localization.Localization(__file__, 1492, 21), _pad_wrap_104154, *[newmat_104155, tuple_104156, axis_104159], **kwargs_104160)
    
    # Assigning a type to the variable 'newmat' (line 1492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1492, 12), 'newmat', _pad_wrap_call_result_104161)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1476)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1457)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1431)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1425)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1419)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1413)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1407)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1401)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1396)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1390)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newmat' (line 1494)
    newmat_104162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 11), 'newmat')
    # Assigning a type to the variable 'stypy_return_type' (line 1494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1494, 4), 'stypy_return_type', newmat_104162)
    
    # ################# End of 'pad(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pad' in the type store
    # Getting the type of 'stypy_return_type' (line 1117)
    stypy_return_type_104163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104163)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pad'
    return stypy_return_type_104163

# Assigning a type to the variable 'pad' (line 1117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 0), 'pad', pad)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
