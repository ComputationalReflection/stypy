
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Copyright (C) 2003-2005 Peter J. Verveer
2: #
3: # Redistribution and use in source and binary forms, with or without
4: # modification, are permitted provided that the following conditions
5: # are met:
6: #
7: # 1. Redistributions of source code must retain the above copyright
8: #    notice, this list of conditions and the following disclaimer.
9: #
10: # 2. Redistributions in binary form must reproduce the above
11: #    copyright notice, this list of conditions and the following
12: #    disclaimer in the documentation and/or other materials provided
13: #    with the distribution.
14: #
15: # 3. The name of the author may not be used to endorse or promote
16: #    products derived from this software without specific prior
17: #    written permission.
18: #
19: # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
20: # OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
21: # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
22: # ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
23: # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
24: # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
25: # GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
26: # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
27: # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
28: # NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
29: # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
30: 
31: from __future__ import division, print_function, absolute_import
32: 
33: import numpy
34: import numpy as np
35: from . import _ni_support
36: from . import _ni_label
37: from . import _nd_image
38: from . import morphology
39: 
40: __all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean',
41:            'variance', 'standard_deviation', 'minimum', 'maximum', 'median',
42:            'minimum_position', 'maximum_position', 'extrema', 'center_of_mass',
43:            'histogram', 'watershed_ift']
44: 
45: 
46: def label(input, structure=None, output=None):
47:     '''
48:     Label features in an array.
49: 
50:     Parameters
51:     ----------
52:     input : array_like
53:         An array-like object to be labeled.  Any non-zero values in `input` are
54:         counted as features and zero values are considered the background.
55:     structure : array_like, optional
56:         A structuring element that defines feature connections.
57:         `structure` must be symmetric.  If no structuring element is provided,
58:         one is automatically generated with a squared connectivity equal to
59:         one.  That is, for a 2-D `input` array, the default structuring element
60:         is::
61: 
62:             [[0,1,0],
63:              [1,1,1],
64:              [0,1,0]]
65: 
66:     output : (None, data-type, array_like), optional
67:         If `output` is a data type, it specifies the type of the resulting
68:         labeled feature array
69:         If `output` is an array-like object, then `output` will be updated
70:         with the labeled features from this function.  This function can
71:         operate in-place, by passing output=input.
72:         Note that the output must be able to store the largest label, or this
73:         function will raise an Exception.
74: 
75:     Returns
76:     -------
77:     label : ndarray or int
78:         An integer ndarray where each unique feature in `input` has a unique
79:         label in the returned array.
80:     num_features : int
81:         How many objects were found.
82: 
83:         If `output` is None, this function returns a tuple of
84:         (`labeled_array`, `num_features`).
85: 
86:         If `output` is a ndarray, then it will be updated with values in
87:         `labeled_array` and only `num_features` will be returned by this
88:         function.
89: 
90:     See Also
91:     --------
92:     find_objects : generate a list of slices for the labeled features (or
93:                    objects); useful for finding features' position or
94:                    dimensions
95: 
96:     Examples
97:     --------
98:     Create an image with some features, then label it using the default
99:     (cross-shaped) structuring element:
100: 
101:     >>> from scipy.ndimage import label, generate_binary_structure
102:     >>> a = np.array([[0,0,1,1,0,0],
103:     ...               [0,0,0,1,0,0],
104:     ...               [1,1,0,0,1,0],
105:     ...               [0,0,0,1,0,0]])
106:     >>> labeled_array, num_features = label(a)
107: 
108:     Each of the 4 features are labeled with a different integer:
109: 
110:     >>> num_features
111:     4
112:     >>> labeled_array
113:     array([[0, 0, 1, 1, 0, 0],
114:            [0, 0, 0, 1, 0, 0],
115:            [2, 2, 0, 0, 3, 0],
116:            [0, 0, 0, 4, 0, 0]])
117: 
118:     Generate a structuring element that will consider features connected even
119:     if they touch diagonally:
120: 
121:     >>> s = generate_binary_structure(2,2)
122: 
123:     or,
124: 
125:     >>> s = [[1,1,1],
126:     ...      [1,1,1],
127:     ...      [1,1,1]]
128: 
129:     Label the image using the new structuring element:
130: 
131:     >>> labeled_array, num_features = label(a, structure=s)
132: 
133:     Show the 2 labeled features (note that features 1, 3, and 4 from above are
134:     now considered a single feature):
135: 
136:     >>> num_features
137:     2
138:     >>> labeled_array
139:     array([[0, 0, 1, 1, 0, 0],
140:            [0, 0, 0, 1, 0, 0],
141:            [2, 2, 0, 0, 1, 0],
142:            [0, 0, 0, 1, 0, 0]])
143: 
144:     '''
145:     input = numpy.asarray(input)
146:     if numpy.iscomplexobj(input):
147:         raise TypeError('Complex type not supported')
148:     if structure is None:
149:         structure = morphology.generate_binary_structure(input.ndim, 1)
150:     structure = numpy.asarray(structure, dtype=bool)
151:     if structure.ndim != input.ndim:
152:         raise RuntimeError('structure and input must have equal rank')
153:     for ii in structure.shape:
154:         if ii != 3:
155:             raise ValueError('structure dimensions must be equal to 3')
156: 
157:     # Use 32 bits if it's large enough for this image.
158:     # _ni_label.label()  needs two entries for background and
159:     # foreground tracking
160:     need_64bits = input.size >= (2**31 - 2)
161: 
162:     if isinstance(output, numpy.ndarray):
163:         if output.shape != input.shape:
164:             raise ValueError("output shape not correct")
165:         caller_provided_output = True
166:     else:
167:         caller_provided_output = False
168:         if output is None:
169:             output = np.empty(input.shape, np.intp if need_64bits else np.int32)
170:         else:
171:             output = np.empty(input.shape, output)
172: 
173:     # handle scalars, 0-dim arrays
174:     if input.ndim == 0 or input.size == 0:
175:         if input.ndim == 0:
176:             # scalar
177:             maxlabel = 1 if (input != 0) else 0
178:             output[...] = maxlabel
179:         else:
180:             # 0-dim
181:             maxlabel = 0
182:         if caller_provided_output:
183:             return maxlabel
184:         else:
185:             return output, maxlabel
186: 
187:     try:
188:         max_label = _ni_label._label(input, structure, output)
189:     except _ni_label.NeedMoreBits:
190:         # Make another attempt with enough bits, then try to cast to the
191:         # new type.
192:         tmp_output = np.empty(input.shape, np.intp if need_64bits else np.int32)
193:         max_label = _ni_label._label(input, structure, tmp_output)
194:         output[...] = tmp_output[...]
195:         if not np.all(output == tmp_output):
196:             # refuse to return bad results
197:             raise RuntimeError("insufficient bit-depth in requested output type")
198: 
199:     if caller_provided_output:
200:         # result was written in-place
201:         return max_label
202:     else:
203:         return output, max_label
204: 
205: 
206: def find_objects(input, max_label=0):
207:     '''
208:     Find objects in a labeled array.
209: 
210:     Parameters
211:     ----------
212:     input : ndarray of ints
213:         Array containing objects defined by different labels. Labels with
214:         value 0 are ignored.
215:     max_label : int, optional
216:         Maximum label to be searched for in `input`. If max_label is not
217:         given, the positions of all objects are returned.
218: 
219:     Returns
220:     -------
221:     object_slices : list of tuples
222:         A list of tuples, with each tuple containing N slices (with N the
223:         dimension of the input array).  Slices correspond to the minimal
224:         parallelepiped that contains the object. If a number is missing,
225:         None is returned instead of a slice.
226: 
227:     See Also
228:     --------
229:     label, center_of_mass
230: 
231:     Notes
232:     -----
233:     This function is very useful for isolating a volume of interest inside
234:     a 3-D array, that cannot be "seen through".
235: 
236:     Examples
237:     --------
238:     >>> from scipy import ndimage
239:     >>> a = np.zeros((6,6), dtype=int)
240:     >>> a[2:4, 2:4] = 1
241:     >>> a[4, 4] = 1
242:     >>> a[:2, :3] = 2
243:     >>> a[0, 5] = 3
244:     >>> a
245:     array([[2, 2, 2, 0, 0, 3],
246:            [2, 2, 2, 0, 0, 0],
247:            [0, 0, 1, 1, 0, 0],
248:            [0, 0, 1, 1, 0, 0],
249:            [0, 0, 0, 0, 1, 0],
250:            [0, 0, 0, 0, 0, 0]])
251:     >>> ndimage.find_objects(a)
252:     [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None)), (slice(0, 1, None), slice(5, 6, None))]
253:     >>> ndimage.find_objects(a, max_label=2)
254:     [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]
255:     >>> ndimage.find_objects(a == 1, max_label=2)
256:     [(slice(2, 5, None), slice(2, 5, None)), None]
257: 
258:     >>> loc = ndimage.find_objects(a)[0]
259:     >>> a[loc]
260:     array([[1, 1, 0],
261:            [1, 1, 0],
262:            [0, 0, 1]])
263: 
264:     '''
265:     input = numpy.asarray(input)
266:     if numpy.iscomplexobj(input):
267:         raise TypeError('Complex type not supported')
268: 
269:     if max_label < 1:
270:         max_label = input.max()
271: 
272:     return _nd_image.find_objects(input, max_label)
273: 
274: 
275: def labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False):
276:     '''
277:     Roughly equivalent to [func(input[labels == i]) for i in index].
278: 
279:     Sequentially applies an arbitrary function (that works on array_like input)
280:     to subsets of an n-D image array specified by `labels` and `index`.
281:     The option exists to provide the function with positional parameters as the
282:     second argument.
283: 
284:     Parameters
285:     ----------
286:     input : array_like
287:         Data from which to select `labels` to process.
288:     labels : array_like or None
289:         Labels to objects in `input`.
290:         If not None, array must be same shape as `input`.
291:         If None, `func` is applied to raveled `input`.
292:     index : int, sequence of ints or None
293:         Subset of `labels` to which to apply `func`.
294:         If a scalar, a single value is returned.
295:         If None, `func` is applied to all non-zero values of `labels`.
296:     func : callable
297:         Python function to apply to `labels` from `input`.
298:     out_dtype : dtype
299:         Dtype to use for `result`.
300:     default : int, float or None
301:         Default return value when a element of `index` does not exist
302:         in `labels`.
303:     pass_positions : bool, optional
304:         If True, pass linear indices to `func` as a second argument.
305:         Default is False.
306: 
307:     Returns
308:     -------
309:     result : ndarray
310:         Result of applying `func` to each of `labels` to `input` in `index`.
311: 
312:     Examples
313:     --------
314:     >>> a = np.array([[1, 2, 0, 0],
315:     ...               [5, 3, 0, 4],
316:     ...               [0, 0, 0, 7],
317:     ...               [9, 3, 0, 0]])
318:     >>> from scipy import ndimage
319:     >>> lbl, nlbl = ndimage.label(a)
320:     >>> lbls = np.arange(1, nlbl+1)
321:     >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)
322:     array([ 2.75,  5.5 ,  6.  ])
323: 
324:     Falling back to `default`:
325: 
326:     >>> lbls = np.arange(1, nlbl+2)
327:     >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)
328:     array([ 2.75,  5.5 ,  6.  , -1.  ])
329: 
330:     Passing positions:
331: 
332:     >>> def fn(val, pos):
333:     ...     print("fn says: %s : %s" % (val, pos))
334:     ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
335:     ...
336:     >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
337:     fn says: [1 2 5 3] : [0 1 4 5]
338:     fn says: [4 7] : [ 7 11]
339:     fn says: [9 3] : [12 13]
340:     array([ 11.,  11., -12.,   0.])
341: 
342:     '''
343: 
344:     as_scalar = numpy.isscalar(index)
345:     input = numpy.asarray(input)
346: 
347:     if pass_positions:
348:         positions = numpy.arange(input.size).reshape(input.shape)
349: 
350:     if labels is None:
351:         if index is not None:
352:             raise ValueError("index without defined labels")
353:         if not pass_positions:
354:             return func(input.ravel())
355:         else:
356:             return func(input.ravel(), positions.ravel())
357: 
358:     try:
359:         input, labels = numpy.broadcast_arrays(input, labels)
360:     except ValueError:
361:         raise ValueError("input and labels must have the same shape "
362:                             "(excepting dimensions with width 1)")
363: 
364:     if index is None:
365:         if not pass_positions:
366:             return func(input[labels > 0])
367:         else:
368:             return func(input[labels > 0], positions[labels > 0])
369: 
370:     index = numpy.atleast_1d(index)
371:     if np.any(index.astype(labels.dtype).astype(index.dtype) != index):
372:         raise ValueError("Cannot convert index values from <%s> to <%s> "
373:                             "(labels' type) without loss of precision" %
374:                             (index.dtype, labels.dtype))
375: 
376:     index = index.astype(labels.dtype)
377: 
378:     # optimization: find min/max in index, and select those parts of labels, input, and positions
379:     lo = index.min()
380:     hi = index.max()
381:     mask = (labels >= lo) & (labels <= hi)
382: 
383:     # this also ravels the arrays
384:     labels = labels[mask]
385:     input = input[mask]
386:     if pass_positions:
387:         positions = positions[mask]
388: 
389:     # sort everything by labels
390:     label_order = labels.argsort()
391:     labels = labels[label_order]
392:     input = input[label_order]
393:     if pass_positions:
394:         positions = positions[label_order]
395: 
396:     index_order = index.argsort()
397:     sorted_index = index[index_order]
398: 
399:     def do_map(inputs, output):
400:         '''labels must be sorted'''
401:         nidx = sorted_index.size
402: 
403:         # Find boundaries for each stretch of constant labels
404:         # This could be faster, but we already paid N log N to sort labels.
405:         lo = numpy.searchsorted(labels, sorted_index, side='left')
406:         hi = numpy.searchsorted(labels, sorted_index, side='right')
407: 
408:         for i, l, h in zip(range(nidx), lo, hi):
409:             if l == h:
410:                 continue
411:             output[i] = func(*[inp[l:h] for inp in inputs])
412: 
413:     temp = numpy.empty(index.shape, out_dtype)
414:     temp[:] = default
415:     if not pass_positions:
416:         do_map([input], temp)
417:     else:
418:         do_map([input, positions], temp)
419: 
420:     output = numpy.zeros(index.shape, out_dtype)
421:     output[index_order] = temp
422:     if as_scalar:
423:         output = output[0]
424: 
425:     return output
426: 
427: 
428: def _safely_castable_to_int(dt):
429:     '''Test whether the numpy data type `dt` can be safely cast to an int.'''
430:     int_size = np.dtype(int).itemsize
431:     safe = ((np.issubdtype(dt, np.signedinteger) and dt.itemsize <= int_size) or
432:             (np.issubdtype(dt, np.unsignedinteger) and dt.itemsize < int_size))
433:     return safe
434: 
435: 
436: def _stats(input, labels=None, index=None, centered=False):
437:     '''Count, sum, and optionally compute (sum - centre)^2 of input by label
438: 
439:     Parameters
440:     ----------
441:     input : array_like, n-dimensional
442:         The input data to be analyzed.
443:     labels : array_like (n-dimensional), optional
444:         The labels of the data in `input`.  This array must be broadcast
445:         compatible with `input`; typically it is the same shape as `input`.
446:         If `labels` is None, all nonzero values in `input` are treated as
447:         the single labeled group.
448:     index : label or sequence of labels, optional
449:         These are the labels of the groups for which the stats are computed.
450:         If `index` is None, the stats are computed for the single group where
451:         `labels` is greater than 0.
452:     centered : bool, optional
453:         If True, the centered sum of squares for each labeled group is
454:         also returned.  Default is False.
455: 
456:     Returns
457:     -------
458:     counts : int or ndarray of ints
459:         The number of elements in each labeled group.
460:     sums : scalar or ndarray of scalars
461:         The sums of the values in each labeled group.
462:     sums_c : scalar or ndarray of scalars, optional
463:         The sums of mean-centered squares of the values in each labeled group.
464:         This is only returned if `centered` is True.
465: 
466:     '''
467:     def single_group(vals):
468:         if centered:
469:             vals_c = vals - vals.mean()
470:             return vals.size, vals.sum(), (vals_c * vals_c.conjugate()).sum()
471:         else:
472:             return vals.size, vals.sum()
473: 
474:     if labels is None:
475:         return single_group(input)
476: 
477:     # ensure input and labels match sizes
478:     input, labels = numpy.broadcast_arrays(input, labels)
479: 
480:     if index is None:
481:         return single_group(input[labels > 0])
482: 
483:     if numpy.isscalar(index):
484:         return single_group(input[labels == index])
485: 
486:     def _sum_centered(labels):
487:         # `labels` is expected to be an ndarray with the same shape as `input`.
488:         # It must contain the label indices (which are not necessarily the labels
489:         # themselves).
490:         means = sums / counts
491:         centered_input = input - means[labels]
492:         # bincount expects 1d inputs, so we ravel the arguments.
493:         bc = numpy.bincount(labels.ravel(),
494:                               weights=(centered_input *
495:                                        centered_input.conjugate()).ravel())
496:         return bc
497: 
498:     # Remap labels to unique integers if necessary, or if the largest
499:     # label is larger than the number of values.
500: 
501:     if (not _safely_castable_to_int(labels.dtype) or
502:             labels.min() < 0 or labels.max() > labels.size):
503:         # Use numpy.unique to generate the label indices.  `new_labels` will
504:         # be 1-d, but it should be interpreted as the flattened n-d array of
505:         # label indices.
506:         unique_labels, new_labels = numpy.unique(labels, return_inverse=True)
507:         counts = numpy.bincount(new_labels)
508:         sums = numpy.bincount(new_labels, weights=input.ravel())
509:         if centered:
510:             # Compute the sum of the mean-centered squares.
511:             # We must reshape new_labels to the n-d shape of `input` before
512:             # passing it _sum_centered.
513:             sums_c = _sum_centered(new_labels.reshape(labels.shape))
514:         idxs = numpy.searchsorted(unique_labels, index)
515:         # make all of idxs valid
516:         idxs[idxs >= unique_labels.size] = 0
517:         found = (unique_labels[idxs] == index)
518:     else:
519:         # labels are an integer type allowed by bincount, and there aren't too
520:         # many, so call bincount directly.
521:         counts = numpy.bincount(labels.ravel())
522:         sums = numpy.bincount(labels.ravel(), weights=input.ravel())
523:         if centered:
524:             sums_c = _sum_centered(labels)
525:         # make sure all index values are valid
526:         idxs = numpy.asanyarray(index, numpy.int).copy()
527:         found = (idxs >= 0) & (idxs < counts.size)
528:         idxs[~found] = 0
529: 
530:     counts = counts[idxs]
531:     counts[~found] = 0
532:     sums = sums[idxs]
533:     sums[~found] = 0
534: 
535:     if not centered:
536:         return (counts, sums)
537:     else:
538:         sums_c = sums_c[idxs]
539:         sums_c[~found] = 0
540:         return (counts, sums, sums_c)
541: 
542: 
543: def sum(input, labels=None, index=None):
544:     '''
545:     Calculate the sum of the values of the array.
546: 
547:     Parameters
548:     ----------
549:     input : array_like
550:         Values of `input` inside the regions defined by `labels`
551:         are summed together.
552:     labels : array_like of ints, optional
553:         Assign labels to the values of the array. Has to have the same shape as
554:         `input`.
555:     index : array_like, optional
556:         A single label number or a sequence of label numbers of
557:         the objects to be measured.
558: 
559:     Returns
560:     -------
561:     sum : ndarray or scalar
562:         An array of the sums of values of `input` inside the regions defined
563:         by `labels` with the same shape as `index`. If 'index' is None or scalar,
564:         a scalar is returned.
565: 
566:     See also
567:     --------
568:     mean, median
569: 
570:     Examples
571:     --------
572:     >>> from scipy import ndimage
573:     >>> input =  [0,1,2,3]
574:     >>> labels = [1,1,2,2]
575:     >>> ndimage.sum(input, labels, index=[1,2])
576:     [1.0, 5.0]
577:     >>> ndimage.sum(input, labels, index=1)
578:     1
579:     >>> ndimage.sum(input, labels)
580:     6
581: 
582: 
583:     '''
584:     count, sum = _stats(input, labels, index)
585:     return sum
586: 
587: 
588: def mean(input, labels=None, index=None):
589:     '''
590:     Calculate the mean of the values of an array at labels.
591: 
592:     Parameters
593:     ----------
594:     input : array_like
595:         Array on which to compute the mean of elements over distinct
596:         regions.
597:     labels : array_like, optional
598:         Array of labels of same shape, or broadcastable to the same shape as
599:         `input`. All elements sharing the same label form one region over
600:         which the mean of the elements is computed.
601:     index : int or sequence of ints, optional
602:         Labels of the objects over which the mean is to be computed.
603:         Default is None, in which case the mean for all values where label is
604:         greater than 0 is calculated.
605: 
606:     Returns
607:     -------
608:     out : list
609:         Sequence of same length as `index`, with the mean of the different
610:         regions labeled by the labels in `index`.
611: 
612:     See also
613:     --------
614:     ndimage.variance, ndimage.standard_deviation, ndimage.minimum,
615:     ndimage.maximum, ndimage.sum
616:     ndimage.label
617: 
618:     Examples
619:     --------
620:     >>> from scipy import ndimage
621:     >>> a = np.arange(25).reshape((5,5))
622:     >>> labels = np.zeros_like(a)
623:     >>> labels[3:5,3:5] = 1
624:     >>> index = np.unique(labels)
625:     >>> labels
626:     array([[0, 0, 0, 0, 0],
627:            [0, 0, 0, 0, 0],
628:            [0, 0, 0, 0, 0],
629:            [0, 0, 0, 1, 1],
630:            [0, 0, 0, 1, 1]])
631:     >>> index
632:     array([0, 1])
633:     >>> ndimage.mean(a, labels=labels, index=index)
634:     [10.285714285714286, 21.0]
635: 
636:     '''
637: 
638:     count, sum = _stats(input, labels, index)
639:     return sum / numpy.asanyarray(count).astype(numpy.float)
640: 
641: 
642: def variance(input, labels=None, index=None):
643:     '''
644:     Calculate the variance of the values of an n-D image array, optionally at
645:     specified sub-regions.
646: 
647:     Parameters
648:     ----------
649:     input : array_like
650:         Nd-image data to process.
651:     labels : array_like, optional
652:         Labels defining sub-regions in `input`.
653:         If not None, must be same shape as `input`.
654:     index : int or sequence of ints, optional
655:         `labels` to include in output.  If None (default), all values where
656:         `labels` is non-zero are used.
657: 
658:     Returns
659:     -------
660:     variance : float or ndarray
661:         Values of variance, for each sub-region if `labels` and `index` are
662:         specified.
663: 
664:     See Also
665:     --------
666:     label, standard_deviation, maximum, minimum, extrema
667: 
668:     Examples
669:     --------
670:     >>> a = np.array([[1, 2, 0, 0],
671:     ...               [5, 3, 0, 4],
672:     ...               [0, 0, 0, 7],
673:     ...               [9, 3, 0, 0]])
674:     >>> from scipy import ndimage
675:     >>> ndimage.variance(a)
676:     7.609375
677: 
678:     Features to process can be specified using `labels` and `index`:
679: 
680:     >>> lbl, nlbl = ndimage.label(a)
681:     >>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))
682:     array([ 2.1875,  2.25  ,  9.    ])
683: 
684:     If no index is given, all non-zero `labels` are processed:
685: 
686:     >>> ndimage.variance(a, lbl)
687:     6.1875
688: 
689:     '''
690:     count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
691:     return sum_c_sq / np.asanyarray(count).astype(float)
692: 
693: 
694: def standard_deviation(input, labels=None, index=None):
695:     '''
696:     Calculate the standard deviation of the values of an n-D image array,
697:     optionally at specified sub-regions.
698: 
699:     Parameters
700:     ----------
701:     input : array_like
702:         Nd-image data to process.
703:     labels : array_like, optional
704:         Labels to identify sub-regions in `input`.
705:         If not None, must be same shape as `input`.
706:     index : int or sequence of ints, optional
707:         `labels` to include in output.  If None (default), all values where
708:         `labels` is non-zero are used.
709: 
710:     Returns
711:     -------
712:     standard_deviation : float or ndarray
713:         Values of standard deviation, for each sub-region if `labels` and
714:         `index` are specified.
715: 
716:     See Also
717:     --------
718:     label, variance, maximum, minimum, extrema
719: 
720:     Examples
721:     --------
722:     >>> a = np.array([[1, 2, 0, 0],
723:     ...               [5, 3, 0, 4],
724:     ...               [0, 0, 0, 7],
725:     ...               [9, 3, 0, 0]])
726:     >>> from scipy import ndimage
727:     >>> ndimage.standard_deviation(a)
728:     2.7585095613392387
729: 
730:     Features to process can be specified using `labels` and `index`:
731: 
732:     >>> lbl, nlbl = ndimage.label(a)
733:     >>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))
734:     array([ 1.479,  1.5  ,  3.   ])
735: 
736:     If no index is given, non-zero `labels` are processed:
737: 
738:     >>> ndimage.standard_deviation(a, lbl)
739:     2.4874685927665499
740: 
741:     '''
742:     return numpy.sqrt(variance(input, labels, index))
743: 
744: 
745: def _select(input, labels=None, index=None, find_min=False, find_max=False,
746:             find_min_positions=False, find_max_positions=False,
747:             find_median=False):
748:     '''Returns min, max, or both, plus their positions (if requested), and
749:     median.'''
750: 
751:     input = numpy.asanyarray(input)
752: 
753:     find_positions = find_min_positions or find_max_positions
754:     positions = None
755:     if find_positions:
756:         positions = numpy.arange(input.size).reshape(input.shape)
757: 
758:     def single_group(vals, positions):
759:         result = []
760:         if find_min:
761:             result += [vals.min()]
762:         if find_min_positions:
763:             result += [positions[vals == vals.min()][0]]
764:         if find_max:
765:             result += [vals.max()]
766:         if find_max_positions:
767:             result += [positions[vals == vals.max()][0]]
768:         if find_median:
769:             result += [numpy.median(vals)]
770:         return result
771: 
772:     if labels is None:
773:         return single_group(input, positions)
774: 
775:     # ensure input and labels match sizes
776:     input, labels = numpy.broadcast_arrays(input, labels)
777: 
778:     if index is None:
779:         mask = (labels > 0)
780:         masked_positions = None
781:         if find_positions:
782:             masked_positions = positions[mask]
783:         return single_group(input[mask], masked_positions)
784: 
785:     if numpy.isscalar(index):
786:         mask = (labels == index)
787:         masked_positions = None
788:         if find_positions:
789:             masked_positions = positions[mask]
790:         return single_group(input[mask], masked_positions)
791: 
792:     # remap labels to unique integers if necessary, or if the largest
793:     # label is larger than the number of values.
794:     if (not _safely_castable_to_int(labels.dtype) or
795:             labels.min() < 0 or labels.max() > labels.size):
796:         # remap labels, and indexes
797:         unique_labels, labels = numpy.unique(labels, return_inverse=True)
798:         idxs = numpy.searchsorted(unique_labels, index)
799: 
800:         # make all of idxs valid
801:         idxs[idxs >= unique_labels.size] = 0
802:         found = (unique_labels[idxs] == index)
803:     else:
804:         # labels are an integer type, and there aren't too many.
805:         idxs = numpy.asanyarray(index, numpy.int).copy()
806:         found = (idxs >= 0) & (idxs <= labels.max())
807: 
808:     idxs[~ found] = labels.max() + 1
809: 
810:     if find_median:
811:         order = numpy.lexsort((input.ravel(), labels.ravel()))
812:     else:
813:         order = input.ravel().argsort()
814:     input = input.ravel()[order]
815:     labels = labels.ravel()[order]
816:     if find_positions:
817:         positions = positions.ravel()[order]
818: 
819:     result = []
820:     if find_min:
821:         mins = numpy.zeros(labels.max() + 2, input.dtype)
822:         mins[labels[::-1]] = input[::-1]
823:         result += [mins[idxs]]
824:     if find_min_positions:
825:         minpos = numpy.zeros(labels.max() + 2, int)
826:         minpos[labels[::-1]] = positions[::-1]
827:         result += [minpos[idxs]]
828:     if find_max:
829:         maxs = numpy.zeros(labels.max() + 2, input.dtype)
830:         maxs[labels] = input
831:         result += [maxs[idxs]]
832:     if find_max_positions:
833:         maxpos = numpy.zeros(labels.max() + 2, int)
834:         maxpos[labels] = positions
835:         result += [maxpos[idxs]]
836:     if find_median:
837:         locs = numpy.arange(len(labels))
838:         lo = numpy.zeros(labels.max() + 2, numpy.int)
839:         lo[labels[::-1]] = locs[::-1]
840:         hi = numpy.zeros(labels.max() + 2, numpy.int)
841:         hi[labels] = locs
842:         lo = lo[idxs]
843:         hi = hi[idxs]
844:         # lo is an index to the lowest value in input for each label,
845:         # hi is an index to the largest value.
846:         # move them to be either the same ((hi - lo) % 2 == 0) or next
847:         # to each other ((hi - lo) % 2 == 1), then average.
848:         step = (hi - lo) // 2
849:         lo += step
850:         hi -= step
851:         result += [(input[lo] + input[hi]) / 2.0]
852: 
853:     return result
854: 
855: 
856: def minimum(input, labels=None, index=None):
857:     '''
858:     Calculate the minimum of the values of an array over labeled regions.
859: 
860:     Parameters
861:     ----------
862:     input : array_like
863:         Array_like of values. For each region specified by `labels`, the
864:         minimal values of `input` over the region is computed.
865:     labels : array_like, optional
866:         An array_like of integers marking different regions over which the
867:         minimum value of `input` is to be computed. `labels` must have the
868:         same shape as `input`. If `labels` is not specified, the minimum
869:         over the whole array is returned.
870:     index : array_like, optional
871:         A list of region labels that are taken into account for computing the
872:         minima. If index is None, the minimum over all elements where `labels`
873:         is non-zero is returned.
874: 
875:     Returns
876:     -------
877:     minimum : float or list of floats
878:         List of minima of `input` over the regions determined by `labels` and
879:         whose index is in `index`. If `index` or `labels` are not specified, a
880:         float is returned: the minimal value of `input` if `labels` is None,
881:         and the minimal value of elements where `labels` is greater than zero
882:         if `index` is None.
883: 
884:     See also
885:     --------
886:     label, maximum, median, minimum_position, extrema, sum, mean, variance,
887:     standard_deviation
888: 
889:     Notes
890:     -----
891:     The function returns a Python list and not a Numpy array, use
892:     `np.array` to convert the list to an array.
893: 
894:     Examples
895:     --------
896:     >>> from scipy import ndimage
897:     >>> a = np.array([[1, 2, 0, 0],
898:     ...               [5, 3, 0, 4],
899:     ...               [0, 0, 0, 7],
900:     ...               [9, 3, 0, 0]])
901:     >>> labels, labels_nb = ndimage.label(a)
902:     >>> labels
903:     array([[1, 1, 0, 0],
904:            [1, 1, 0, 2],
905:            [0, 0, 0, 2],
906:            [3, 3, 0, 0]])
907:     >>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))
908:     [1.0, 4.0, 3.0]
909:     >>> ndimage.minimum(a)
910:     0.0
911:     >>> ndimage.minimum(a, labels=labels)
912:     1.0
913: 
914:     '''
915:     return _select(input, labels, index, find_min=True)[0]
916: 
917: 
918: def maximum(input, labels=None, index=None):
919:     '''
920:     Calculate the maximum of the values of an array over labeled regions.
921: 
922:     Parameters
923:     ----------
924:     input : array_like
925:         Array_like of values. For each region specified by `labels`, the
926:         maximal values of `input` over the region is computed.
927:     labels : array_like, optional
928:         An array of integers marking different regions over which the
929:         maximum value of `input` is to be computed. `labels` must have the
930:         same shape as `input`. If `labels` is not specified, the maximum
931:         over the whole array is returned.
932:     index : array_like, optional
933:         A list of region labels that are taken into account for computing the
934:         maxima. If index is None, the maximum over all elements where `labels`
935:         is non-zero is returned.
936: 
937:     Returns
938:     -------
939:     output : float or list of floats
940:         List of maxima of `input` over the regions determined by `labels` and
941:         whose index is in `index`. If `index` or `labels` are not specified, a
942:         float is returned: the maximal value of `input` if `labels` is None,
943:         and the maximal value of elements where `labels` is greater than zero
944:         if `index` is None.
945: 
946:     See also
947:     --------
948:     label, minimum, median, maximum_position, extrema, sum, mean, variance,
949:     standard_deviation
950: 
951:     Notes
952:     -----
953:     The function returns a Python list and not a Numpy array, use
954:     `np.array` to convert the list to an array.
955: 
956:     Examples
957:     --------
958:     >>> a = np.arange(16).reshape((4,4))
959:     >>> a
960:     array([[ 0,  1,  2,  3],
961:            [ 4,  5,  6,  7],
962:            [ 8,  9, 10, 11],
963:            [12, 13, 14, 15]])
964:     >>> labels = np.zeros_like(a)
965:     >>> labels[:2,:2] = 1
966:     >>> labels[2:, 1:3] = 2
967:     >>> labels
968:     array([[1, 1, 0, 0],
969:            [1, 1, 0, 0],
970:            [0, 2, 2, 0],
971:            [0, 2, 2, 0]])
972:     >>> from scipy import ndimage
973:     >>> ndimage.maximum(a)
974:     15.0
975:     >>> ndimage.maximum(a, labels=labels, index=[1,2])
976:     [5.0, 14.0]
977:     >>> ndimage.maximum(a, labels=labels)
978:     14.0
979: 
980:     >>> b = np.array([[1, 2, 0, 0],
981:     ...               [5, 3, 0, 4],
982:     ...               [0, 0, 0, 7],
983:     ...               [9, 3, 0, 0]])
984:     >>> labels, labels_nb = ndimage.label(b)
985:     >>> labels
986:     array([[1, 1, 0, 0],
987:            [1, 1, 0, 2],
988:            [0, 0, 0, 2],
989:            [3, 3, 0, 0]])
990:     >>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))
991:     [5.0, 7.0, 9.0]
992: 
993:     '''
994:     return _select(input, labels, index, find_max=True)[0]
995: 
996: 
997: def median(input, labels=None, index=None):
998:     '''
999:     Calculate the median of the values of an array over labeled regions.
1000: 
1001:     Parameters
1002:     ----------
1003:     input : array_like
1004:         Array_like of values. For each region specified by `labels`, the
1005:         median value of `input` over the region is computed.
1006:     labels : array_like, optional
1007:         An array_like of integers marking different regions over which the
1008:         median value of `input` is to be computed. `labels` must have the
1009:         same shape as `input`. If `labels` is not specified, the median
1010:         over the whole array is returned.
1011:     index : array_like, optional
1012:         A list of region labels that are taken into account for computing the
1013:         medians. If index is None, the median over all elements where `labels`
1014:         is non-zero is returned.
1015: 
1016:     Returns
1017:     -------
1018:     median : float or list of floats
1019:         List of medians of `input` over the regions determined by `labels` and
1020:         whose index is in `index`. If `index` or `labels` are not specified, a
1021:         float is returned: the median value of `input` if `labels` is None,
1022:         and the median value of elements where `labels` is greater than zero
1023:         if `index` is None.
1024: 
1025:     See also
1026:     --------
1027:     label, minimum, maximum, extrema, sum, mean, variance, standard_deviation
1028: 
1029:     Notes
1030:     -----
1031:     The function returns a Python list and not a Numpy array, use
1032:     `np.array` to convert the list to an array.
1033: 
1034:     Examples
1035:     --------
1036:     >>> from scipy import ndimage
1037:     >>> a = np.array([[1, 2, 0, 1],
1038:     ...               [5, 3, 0, 4],
1039:     ...               [0, 0, 0, 7],
1040:     ...               [9, 3, 0, 0]])
1041:     >>> labels, labels_nb = ndimage.label(a)
1042:     >>> labels
1043:     array([[1, 1, 0, 2],
1044:            [1, 1, 0, 2],
1045:            [0, 0, 0, 2],
1046:            [3, 3, 0, 0]])
1047:     >>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))
1048:     [2.5, 4.0, 6.0]
1049:     >>> ndimage.median(a)
1050:     1.0
1051:     >>> ndimage.median(a, labels=labels)
1052:     3.0
1053: 
1054:     '''
1055:     return _select(input, labels, index, find_median=True)[0]
1056: 
1057: 
1058: def minimum_position(input, labels=None, index=None):
1059:     '''
1060:     Find the positions of the minimums of the values of an array at labels.
1061: 
1062:     Parameters
1063:     ----------
1064:     input : array_like
1065:         Array_like of values.
1066:     labels : array_like, optional
1067:         An array of integers marking different regions over which the
1068:         position of the minimum value of `input` is to be computed.
1069:         `labels` must have the same shape as `input`. If `labels` is not
1070:         specified, the location of the first minimum over the whole
1071:         array is returned.
1072: 
1073:         The `labels` argument only works when `index` is specified.
1074:     index : array_like, optional
1075:         A list of region labels that are taken into account for finding the
1076:         location of the minima. If `index` is None, the ``first`` minimum
1077:         over all elements where `labels` is non-zero is returned.
1078: 
1079:         The `index` argument only works when `labels` is specified.
1080: 
1081:     Returns
1082:     -------
1083:     output : list of tuples of ints
1084:         Tuple of ints or list of tuples of ints that specify the location
1085:         of minima of `input` over the regions determined by `labels` and
1086:         whose index is in `index`.
1087: 
1088:         If `index` or `labels` are not specified, a tuple of ints is
1089:         returned specifying the location of the first minimal value of `input`.
1090: 
1091:     See also
1092:     --------
1093:     label, minimum, median, maximum_position, extrema, sum, mean, variance,
1094:     standard_deviation
1095: 
1096:     '''
1097:     dims = numpy.array(numpy.asarray(input).shape)
1098:     # see numpy.unravel_index to understand this line.
1099:     dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
1100: 
1101:     result = _select(input, labels, index, find_min_positions=True)[0]
1102: 
1103:     if numpy.isscalar(result):
1104:         return tuple((result // dim_prod) % dims)
1105: 
1106:     return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]
1107: 
1108: 
1109: def maximum_position(input, labels=None, index=None):
1110:     '''
1111:     Find the positions of the maximums of the values of an array at labels.
1112: 
1113:     For each region specified by `labels`, the position of the maximum
1114:     value of `input` within the region is returned.
1115: 
1116:     Parameters
1117:     ----------
1118:     input : array_like
1119:         Array_like of values.
1120:     labels : array_like, optional
1121:         An array of integers marking different regions over which the
1122:         position of the maximum value of `input` is to be computed.
1123:         `labels` must have the same shape as `input`. If `labels` is not
1124:         specified, the location of the first maximum over the whole
1125:         array is returned.
1126: 
1127:         The `labels` argument only works when `index` is specified.
1128:     index : array_like, optional
1129:         A list of region labels that are taken into account for finding the
1130:         location of the maxima.  If `index` is None, the first maximum
1131:         over all elements where `labels` is non-zero is returned.
1132: 
1133:         The `index` argument only works when `labels` is specified.
1134: 
1135:     Returns
1136:     -------
1137:     output : list of tuples of ints
1138:         List of tuples of ints that specify the location of maxima of
1139:         `input` over the regions determined by `labels` and whose index
1140:         is in `index`.
1141: 
1142:         If `index` or `labels` are not specified, a tuple of ints is
1143:         returned specifying the location of the ``first`` maximal value
1144:         of `input`.
1145: 
1146:     See also
1147:     --------
1148:     label, minimum, median, maximum_position, extrema, sum, mean, variance,
1149:     standard_deviation
1150: 
1151:     '''
1152:     dims = numpy.array(numpy.asarray(input).shape)
1153:     # see numpy.unravel_index to understand this line.
1154:     dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
1155: 
1156:     result = _select(input, labels, index, find_max_positions=True)[0]
1157: 
1158:     if numpy.isscalar(result):
1159:         return tuple((result // dim_prod) % dims)
1160: 
1161:     return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]
1162: 
1163: 
1164: def extrema(input, labels=None, index=None):
1165:     '''
1166:     Calculate the minimums and maximums of the values of an array
1167:     at labels, along with their positions.
1168: 
1169:     Parameters
1170:     ----------
1171:     input : ndarray
1172:         Nd-image data to process.
1173:     labels : ndarray, optional
1174:         Labels of features in input.
1175:         If not None, must be same shape as `input`.
1176:     index : int or sequence of ints, optional
1177:         Labels to include in output.  If None (default), all values where
1178:         non-zero `labels` are used.
1179: 
1180:     Returns
1181:     -------
1182:     minimums, maximums : int or ndarray
1183:         Values of minimums and maximums in each feature.
1184:     min_positions, max_positions : tuple or list of tuples
1185:         Each tuple gives the n-D coordinates of the corresponding minimum
1186:         or maximum.
1187: 
1188:     See Also
1189:     --------
1190:     maximum, minimum, maximum_position, minimum_position, center_of_mass
1191: 
1192:     Examples
1193:     --------
1194:     >>> a = np.array([[1, 2, 0, 0],
1195:     ...               [5, 3, 0, 4],
1196:     ...               [0, 0, 0, 7],
1197:     ...               [9, 3, 0, 0]])
1198:     >>> from scipy import ndimage
1199:     >>> ndimage.extrema(a)
1200:     (0, 9, (0, 2), (3, 0))
1201: 
1202:     Features to process can be specified using `labels` and `index`:
1203: 
1204:     >>> lbl, nlbl = ndimage.label(a)
1205:     >>> ndimage.extrema(a, lbl, index=np.arange(1, nlbl+1))
1206:     (array([1, 4, 3]),
1207:      array([5, 7, 9]),
1208:      [(0, 0), (1, 3), (3, 1)],
1209:      [(1, 0), (2, 3), (3, 0)])
1210: 
1211:     If no index is given, non-zero `labels` are processed:
1212: 
1213:     >>> ndimage.extrema(a, lbl)
1214:     (1, 9, (0, 0), (3, 0))
1215: 
1216:     '''
1217:     dims = numpy.array(numpy.asarray(input).shape)
1218:     # see numpy.unravel_index to understand this line.
1219:     dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
1220: 
1221:     minimums, min_positions, maximums, max_positions = _select(input, labels,
1222:                                                                index,
1223:                                                                find_min=True,
1224:                                                                find_max=True,
1225:                                                                find_min_positions=True,
1226:                                                                find_max_positions=True)
1227: 
1228:     if numpy.isscalar(minimums):
1229:         return (minimums, maximums, tuple((min_positions // dim_prod) % dims),
1230:                 tuple((max_positions // dim_prod) % dims))
1231: 
1232:     min_positions = [tuple(v) for v in (min_positions.reshape(-1, 1) // dim_prod) % dims]
1233:     max_positions = [tuple(v) for v in (max_positions.reshape(-1, 1) // dim_prod) % dims]
1234: 
1235:     return minimums, maximums, min_positions, max_positions
1236: 
1237: 
1238: def center_of_mass(input, labels=None, index=None):
1239:     '''
1240:     Calculate the center of mass of the values of an array at labels.
1241: 
1242:     Parameters
1243:     ----------
1244:     input : ndarray
1245:         Data from which to calculate center-of-mass. The masses can either
1246:         be positive or negative.
1247:     labels : ndarray, optional
1248:         Labels for objects in `input`, as generated by `ndimage.label`.
1249:         Only used with `index`.  Dimensions must be the same as `input`.
1250:     index : int or sequence of ints, optional
1251:         Labels for which to calculate centers-of-mass. If not specified,
1252:         all labels greater than zero are used.  Only used with `labels`.
1253: 
1254:     Returns
1255:     -------
1256:     center_of_mass : tuple, or list of tuples
1257:         Coordinates of centers-of-mass.
1258: 
1259:     Examples
1260:     --------
1261:     >>> a = np.array(([0,0,0,0],
1262:     ...               [0,1,1,0],
1263:     ...               [0,1,1,0],
1264:     ...               [0,1,1,0]))
1265:     >>> from scipy import ndimage
1266:     >>> ndimage.measurements.center_of_mass(a)
1267:     (2.0, 1.5)
1268: 
1269:     Calculation of multiple objects in an image
1270: 
1271:     >>> b = np.array(([0,1,1,0],
1272:     ...               [0,1,0,0],
1273:     ...               [0,0,0,0],
1274:     ...               [0,0,1,1],
1275:     ...               [0,0,1,1]))
1276:     >>> lbl = ndimage.label(b)[0]
1277:     >>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
1278:     [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]
1279: 
1280:     Negative masses are also accepted, which can occur for example when
1281:     bias is removed from measured data due to random noise.
1282: 
1283:     >>> c = np.array(([-1,0,0,0],
1284:     ...               [0,-1,-1,0],
1285:     ...               [0,1,-1,0],
1286:     ...               [0,1,1,0]))
1287:     >>> ndimage.measurements.center_of_mass(c)
1288:     (-4.0, 1.0)
1289: 
1290:     If there are division by zero issues, the function does not raise an
1291:     error but rather issues a RuntimeWarning before returning inf and/or NaN.
1292: 
1293:     >>> d = np.array([-1, 1])
1294:     >>> ndimage.measurements.center_of_mass(d)
1295:     (inf,)
1296:     '''
1297:     normalizer = sum(input, labels, index)
1298:     grids = numpy.ogrid[[slice(0, i) for i in input.shape]]
1299: 
1300:     results = [sum(input * grids[dir].astype(float), labels, index) / normalizer
1301:                for dir in range(input.ndim)]
1302: 
1303:     if numpy.isscalar(results[0]):
1304:         return tuple(results)
1305: 
1306:     return [tuple(v) for v in numpy.array(results).T]
1307: 
1308: 
1309: def histogram(input, min, max, bins, labels=None, index=None):
1310:     '''
1311:     Calculate the histogram of the values of an array, optionally at labels.
1312: 
1313:     Histogram calculates the frequency of values in an array within bins
1314:     determined by `min`, `max`, and `bins`. The `labels` and `index`
1315:     keywords can limit the scope of the histogram to specified sub-regions
1316:     within the array.
1317: 
1318:     Parameters
1319:     ----------
1320:     input : array_like
1321:         Data for which to calculate histogram.
1322:     min, max : int
1323:         Minimum and maximum values of range of histogram bins.
1324:     bins : int
1325:         Number of bins.
1326:     labels : array_like, optional
1327:         Labels for objects in `input`.
1328:         If not None, must be same shape as `input`.
1329:     index : int or sequence of ints, optional
1330:         Label or labels for which to calculate histogram. If None, all values
1331:         where label is greater than zero are used
1332: 
1333:     Returns
1334:     -------
1335:     hist : ndarray
1336:         Histogram counts.
1337: 
1338:     Examples
1339:     --------
1340:     >>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],
1341:     ...               [ 0.    ,  0.7778,  0.    ,  0.    ],
1342:     ...               [ 0.    ,  0.    ,  0.    ,  0.    ],
1343:     ...               [ 0.    ,  0.    ,  0.7181,  0.2787],
1344:     ...               [ 0.    ,  0.    ,  0.6573,  0.3094]])
1345:     >>> from scipy import ndimage
1346:     >>> ndimage.measurements.histogram(a, 0, 1, 10)
1347:     array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])
1348: 
1349:     With labels and no indices, non-zero elements are counted:
1350: 
1351:     >>> lbl, nlbl = ndimage.label(a)
1352:     >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)
1353:     array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])
1354: 
1355:     Indices can be used to count only certain objects:
1356: 
1357:     >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)
1358:     array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
1359: 
1360:     '''
1361:     _bins = numpy.linspace(min, max, bins + 1)
1362: 
1363:     def _hist(vals):
1364:         return numpy.histogram(vals, _bins)[0]
1365: 
1366:     return labeled_comprehension(input, labels, index, _hist, object, None,
1367:                                  pass_positions=False)
1368: 
1369: 
1370: def watershed_ift(input, markers, structure=None, output=None):
1371:     '''
1372:     Apply watershed from markers using image foresting transform algorithm.
1373: 
1374:     Parameters
1375:     ----------
1376:     input : array_like
1377:         Input.
1378:     markers : array_like
1379:         Markers are points within each watershed that form the beginning
1380:         of the process.  Negative markers are considered background markers
1381:         which are processed after the other markers.
1382:     structure : structure element, optional
1383:         A structuring element defining the connectivity of the object can be
1384:         provided. If None, an element is generated with a squared
1385:         connectivity equal to one.
1386:     output : ndarray, optional
1387:         An output array can optionally be provided.  The same shape as input.
1388: 
1389:     Returns
1390:     -------
1391:     watershed_ift : ndarray
1392:         Output.  Same shape as `input`.
1393: 
1394:     References
1395:     ----------
1396:     .. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, "The image
1397:            foresting transform: theory, algorithms, and applications",
1398:            Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.
1399: 
1400:     '''
1401:     input = numpy.asarray(input)
1402:     if input.dtype.type not in [numpy.uint8, numpy.uint16]:
1403:         raise TypeError('only 8 and 16 unsigned inputs are supported')
1404: 
1405:     if structure is None:
1406:         structure = morphology.generate_binary_structure(input.ndim, 1)
1407:     structure = numpy.asarray(structure, dtype=bool)
1408:     if structure.ndim != input.ndim:
1409:         raise RuntimeError('structure and input must have equal rank')
1410:     for ii in structure.shape:
1411:         if ii != 3:
1412:             raise RuntimeError('structure dimensions must be equal to 3')
1413: 
1414:     if not structure.flags.contiguous:
1415:         structure = structure.copy()
1416:     markers = numpy.asarray(markers)
1417:     if input.shape != markers.shape:
1418:         raise RuntimeError('input and markers must have equal shape')
1419: 
1420:     integral_types = [numpy.int0,
1421:                       numpy.int8,
1422:                       numpy.int16,
1423:                       numpy.int32,
1424:                       numpy.int_,
1425:                       numpy.int64,
1426:                       numpy.intc,
1427:                       numpy.intp]
1428: 
1429:     if markers.dtype.type not in integral_types:
1430:         raise RuntimeError('marker should be of integer type')
1431: 
1432:     if isinstance(output, numpy.ndarray):
1433:         if output.dtype.type not in integral_types:
1434:             raise RuntimeError('output should be of integer type')
1435:     else:
1436:         output = markers.dtype
1437: 
1438:     output, return_value = _ni_support._get_output(output, input)
1439:     _nd_image.watershed_ift(input, markers, structure, output)
1440:     return return_value
1441: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import numpy' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121925 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy')

if (type(import_121925) is not StypyTypeError):

    if (import_121925 != 'pyd_module'):
        __import__(import_121925)
        sys_modules_121926 = sys.modules[import_121925]
        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', sys_modules_121926.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'numpy', import_121925)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121927 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_121927) is not StypyTypeError):

    if (import_121927 != 'pyd_module'):
        __import__(import_121927)
        sys_modules_121928 = sys.modules[import_121927]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', sys_modules_121928.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_121927)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.ndimage import _ni_support' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121929 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage')

if (type(import_121929) is not StypyTypeError):

    if (import_121929 != 'pyd_module'):
        __import__(import_121929)
        sys_modules_121930 = sys.modules[import_121929]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', sys_modules_121930.module_type_store, module_type_store, ['_ni_support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_121930, sys_modules_121930.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_support

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', None, module_type_store, ['_ni_support'], [_ni_support])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', import_121929)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.ndimage import _ni_label' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121931 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage')

if (type(import_121931) is not StypyTypeError):

    if (import_121931 != 'pyd_module'):
        __import__(import_121931)
        sys_modules_121932 = sys.modules[import_121931]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', sys_modules_121932.module_type_store, module_type_store, ['_ni_label'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_121932, sys_modules_121932.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_label

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', None, module_type_store, ['_ni_label'], [_ni_label])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', import_121931)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from scipy.ndimage import _nd_image' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121933 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.ndimage')

if (type(import_121933) is not StypyTypeError):

    if (import_121933 != 'pyd_module'):
        __import__(import_121933)
        sys_modules_121934 = sys.modules[import_121933]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.ndimage', sys_modules_121934.module_type_store, module_type_store, ['_nd_image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_121934, sys_modules_121934.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _nd_image

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.ndimage', None, module_type_store, ['_nd_image'], [_nd_image])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.ndimage', import_121933)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from scipy.ndimage import morphology' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_121935 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.ndimage')

if (type(import_121935) is not StypyTypeError):

    if (import_121935 != 'pyd_module'):
        __import__(import_121935)
        sys_modules_121936 = sys.modules[import_121935]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.ndimage', sys_modules_121936.module_type_store, module_type_store, ['morphology'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_121936, sys_modules_121936.module_type_store, module_type_store)
    else:
        from scipy.ndimage import morphology

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.ndimage', None, module_type_store, ['morphology'], [morphology])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy.ndimage', import_121935)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a List to a Name (line 40):

# Assigning a List to a Name (line 40):
__all__ = ['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean', 'variance', 'standard_deviation', 'minimum', 'maximum', 'median', 'minimum_position', 'maximum_position', 'extrema', 'center_of_mass', 'histogram', 'watershed_ift']
module_type_store.set_exportable_members(['label', 'find_objects', 'labeled_comprehension', 'sum', 'mean', 'variance', 'standard_deviation', 'minimum', 'maximum', 'median', 'minimum_position', 'maximum_position', 'extrema', 'center_of_mass', 'histogram', 'watershed_ift'])

# Obtaining an instance of the builtin type 'list' (line 40)
list_121937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_121938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'str', 'label')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121938)
# Adding element type (line 40)
str_121939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'str', 'find_objects')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121939)
# Adding element type (line 40)
str_121940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'str', 'labeled_comprehension')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121940)
# Adding element type (line 40)
str_121941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'str', 'sum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121941)
# Adding element type (line 40)
str_121942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 68), 'str', 'mean')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121942)
# Adding element type (line 40)
str_121943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'variance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121943)
# Adding element type (line 40)
str_121944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'str', 'standard_deviation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121944)
# Adding element type (line 40)
str_121945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 45), 'str', 'minimum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121945)
# Adding element type (line 40)
str_121946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 56), 'str', 'maximum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121946)
# Adding element type (line 40)
str_121947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 67), 'str', 'median')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121947)
# Adding element type (line 40)
str_121948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'minimum_position')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121948)
# Adding element type (line 40)
str_121949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'str', 'maximum_position')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121949)
# Adding element type (line 40)
str_121950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 51), 'str', 'extrema')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121950)
# Adding element type (line 40)
str_121951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 62), 'str', 'center_of_mass')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121951)
# Adding element type (line 40)
str_121952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'histogram')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121952)
# Adding element type (line 40)
str_121953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'str', 'watershed_ift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_121937, str_121953)

# Assigning a type to the variable '__all__' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '__all__', list_121937)

@norecursion
def label(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 46)
    None_121954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'None')
    # Getting the type of 'None' (line 46)
    None_121955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'None')
    defaults = [None_121954, None_121955]
    # Create a new context for function 'label'
    module_type_store = module_type_store.open_function_context('label', 46, 0, False)
    
    # Passed parameters checking function
    label.stypy_localization = localization
    label.stypy_type_of_self = None
    label.stypy_type_store = module_type_store
    label.stypy_function_name = 'label'
    label.stypy_param_names_list = ['input', 'structure', 'output']
    label.stypy_varargs_param_name = None
    label.stypy_kwargs_param_name = None
    label.stypy_call_defaults = defaults
    label.stypy_call_varargs = varargs
    label.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'label', ['input', 'structure', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'label', localization, ['input', 'structure', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'label(...)' code ##################

    str_121956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', "\n    Label features in an array.\n\n    Parameters\n    ----------\n    input : array_like\n        An array-like object to be labeled.  Any non-zero values in `input` are\n        counted as features and zero values are considered the background.\n    structure : array_like, optional\n        A structuring element that defines feature connections.\n        `structure` must be symmetric.  If no structuring element is provided,\n        one is automatically generated with a squared connectivity equal to\n        one.  That is, for a 2-D `input` array, the default structuring element\n        is::\n\n            [[0,1,0],\n             [1,1,1],\n             [0,1,0]]\n\n    output : (None, data-type, array_like), optional\n        If `output` is a data type, it specifies the type of the resulting\n        labeled feature array\n        If `output` is an array-like object, then `output` will be updated\n        with the labeled features from this function.  This function can\n        operate in-place, by passing output=input.\n        Note that the output must be able to store the largest label, or this\n        function will raise an Exception.\n\n    Returns\n    -------\n    label : ndarray or int\n        An integer ndarray where each unique feature in `input` has a unique\n        label in the returned array.\n    num_features : int\n        How many objects were found.\n\n        If `output` is None, this function returns a tuple of\n        (`labeled_array`, `num_features`).\n\n        If `output` is a ndarray, then it will be updated with values in\n        `labeled_array` and only `num_features` will be returned by this\n        function.\n\n    See Also\n    --------\n    find_objects : generate a list of slices for the labeled features (or\n                   objects); useful for finding features' position or\n                   dimensions\n\n    Examples\n    --------\n    Create an image with some features, then label it using the default\n    (cross-shaped) structuring element:\n\n    >>> from scipy.ndimage import label, generate_binary_structure\n    >>> a = np.array([[0,0,1,1,0,0],\n    ...               [0,0,0,1,0,0],\n    ...               [1,1,0,0,1,0],\n    ...               [0,0,0,1,0,0]])\n    >>> labeled_array, num_features = label(a)\n\n    Each of the 4 features are labeled with a different integer:\n\n    >>> num_features\n    4\n    >>> labeled_array\n    array([[0, 0, 1, 1, 0, 0],\n           [0, 0, 0, 1, 0, 0],\n           [2, 2, 0, 0, 3, 0],\n           [0, 0, 0, 4, 0, 0]])\n\n    Generate a structuring element that will consider features connected even\n    if they touch diagonally:\n\n    >>> s = generate_binary_structure(2,2)\n\n    or,\n\n    >>> s = [[1,1,1],\n    ...      [1,1,1],\n    ...      [1,1,1]]\n\n    Label the image using the new structuring element:\n\n    >>> labeled_array, num_features = label(a, structure=s)\n\n    Show the 2 labeled features (note that features 1, 3, and 4 from above are\n    now considered a single feature):\n\n    >>> num_features\n    2\n    >>> labeled_array\n    array([[0, 0, 1, 1, 0, 0],\n           [0, 0, 0, 1, 0, 0],\n           [2, 2, 0, 0, 1, 0],\n           [0, 0, 0, 1, 0, 0]])\n\n    ")
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to asarray(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'input' (line 145)
    input_121959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'input', False)
    # Processing the call keyword arguments (line 145)
    kwargs_121960 = {}
    # Getting the type of 'numpy' (line 145)
    numpy_121957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 145)
    asarray_121958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), numpy_121957, 'asarray')
    # Calling asarray(args, kwargs) (line 145)
    asarray_call_result_121961 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), asarray_121958, *[input_121959], **kwargs_121960)
    
    # Assigning a type to the variable 'input' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'input', asarray_call_result_121961)
    
    
    # Call to iscomplexobj(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'input' (line 146)
    input_121964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'input', False)
    # Processing the call keyword arguments (line 146)
    kwargs_121965 = {}
    # Getting the type of 'numpy' (line 146)
    numpy_121962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 146)
    iscomplexobj_121963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 7), numpy_121962, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 146)
    iscomplexobj_call_result_121966 = invoke(stypy.reporting.localization.Localization(__file__, 146, 7), iscomplexobj_121963, *[input_121964], **kwargs_121965)
    
    # Testing the type of an if condition (line 146)
    if_condition_121967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), iscomplexobj_call_result_121966)
    # Assigning a type to the variable 'if_condition_121967' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_121967', if_condition_121967)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 147)
    # Processing the call arguments (line 147)
    str_121969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 147)
    kwargs_121970 = {}
    # Getting the type of 'TypeError' (line 147)
    TypeError_121968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 147)
    TypeError_call_result_121971 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), TypeError_121968, *[str_121969], **kwargs_121970)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 8), TypeError_call_result_121971, 'raise parameter', BaseException)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 148)
    # Getting the type of 'structure' (line 148)
    structure_121972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'structure')
    # Getting the type of 'None' (line 148)
    None_121973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'None')
    
    (may_be_121974, more_types_in_union_121975) = may_be_none(structure_121972, None_121973)

    if may_be_121974:

        if more_types_in_union_121975:
            # Runtime conditional SSA (line 148)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to generate_binary_structure(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'input' (line 149)
        input_121978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 57), 'input', False)
        # Obtaining the member 'ndim' of a type (line 149)
        ndim_121979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 57), input_121978, 'ndim')
        int_121980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 69), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_121981 = {}
        # Getting the type of 'morphology' (line 149)
        morphology_121976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'morphology', False)
        # Obtaining the member 'generate_binary_structure' of a type (line 149)
        generate_binary_structure_121977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), morphology_121976, 'generate_binary_structure')
        # Calling generate_binary_structure(args, kwargs) (line 149)
        generate_binary_structure_call_result_121982 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), generate_binary_structure_121977, *[ndim_121979, int_121980], **kwargs_121981)
        
        # Assigning a type to the variable 'structure' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'structure', generate_binary_structure_call_result_121982)

        if more_types_in_union_121975:
            # SSA join for if statement (line 148)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 150):
    
    # Call to asarray(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'structure' (line 150)
    structure_121985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'structure', False)
    # Processing the call keyword arguments (line 150)
    # Getting the type of 'bool' (line 150)
    bool_121986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'bool', False)
    keyword_121987 = bool_121986
    kwargs_121988 = {'dtype': keyword_121987}
    # Getting the type of 'numpy' (line 150)
    numpy_121983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 150)
    asarray_121984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), numpy_121983, 'asarray')
    # Calling asarray(args, kwargs) (line 150)
    asarray_call_result_121989 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), asarray_121984, *[structure_121985], **kwargs_121988)
    
    # Assigning a type to the variable 'structure' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'structure', asarray_call_result_121989)
    
    
    # Getting the type of 'structure' (line 151)
    structure_121990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'structure')
    # Obtaining the member 'ndim' of a type (line 151)
    ndim_121991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 7), structure_121990, 'ndim')
    # Getting the type of 'input' (line 151)
    input_121992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'input')
    # Obtaining the member 'ndim' of a type (line 151)
    ndim_121993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), input_121992, 'ndim')
    # Applying the binary operator '!=' (line 151)
    result_ne_121994 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '!=', ndim_121991, ndim_121993)
    
    # Testing the type of an if condition (line 151)
    if_condition_121995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_ne_121994)
    # Assigning a type to the variable 'if_condition_121995' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_121995', if_condition_121995)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 152)
    # Processing the call arguments (line 152)
    str_121997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'str', 'structure and input must have equal rank')
    # Processing the call keyword arguments (line 152)
    kwargs_121998 = {}
    # Getting the type of 'RuntimeError' (line 152)
    RuntimeError_121996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 152)
    RuntimeError_call_result_121999 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), RuntimeError_121996, *[str_121997], **kwargs_121998)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 8), RuntimeError_call_result_121999, 'raise parameter', BaseException)
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'structure' (line 153)
    structure_122000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'structure')
    # Obtaining the member 'shape' of a type (line 153)
    shape_122001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 14), structure_122000, 'shape')
    # Testing the type of a for loop iterable (line 153)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 4), shape_122001)
    # Getting the type of the for loop variable (line 153)
    for_loop_var_122002 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 4), shape_122001)
    # Assigning a type to the variable 'ii' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'ii', for_loop_var_122002)
    # SSA begins for a for statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ii' (line 154)
    ii_122003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'ii')
    int_122004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'int')
    # Applying the binary operator '!=' (line 154)
    result_ne_122005 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 11), '!=', ii_122003, int_122004)
    
    # Testing the type of an if condition (line 154)
    if_condition_122006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), result_ne_122005)
    # Assigning a type to the variable 'if_condition_122006' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_122006', if_condition_122006)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 155)
    # Processing the call arguments (line 155)
    str_122008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'str', 'structure dimensions must be equal to 3')
    # Processing the call keyword arguments (line 155)
    kwargs_122009 = {}
    # Getting the type of 'ValueError' (line 155)
    ValueError_122007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 155)
    ValueError_call_result_122010 = invoke(stypy.reporting.localization.Localization(__file__, 155, 18), ValueError_122007, *[str_122008], **kwargs_122009)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 12), ValueError_call_result_122010, 'raise parameter', BaseException)
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 160):
    
    # Assigning a Compare to a Name (line 160):
    
    # Getting the type of 'input' (line 160)
    input_122011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'input')
    # Obtaining the member 'size' of a type (line 160)
    size_122012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 18), input_122011, 'size')
    int_122013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'int')
    int_122014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
    # Applying the binary operator '**' (line 160)
    result_pow_122015 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 33), '**', int_122013, int_122014)
    
    int_122016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'int')
    # Applying the binary operator '-' (line 160)
    result_sub_122017 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 33), '-', result_pow_122015, int_122016)
    
    # Applying the binary operator '>=' (line 160)
    result_ge_122018 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 18), '>=', size_122012, result_sub_122017)
    
    # Assigning a type to the variable 'need_64bits' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'need_64bits', result_ge_122018)
    
    
    # Call to isinstance(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'output' (line 162)
    output_122020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'output', False)
    # Getting the type of 'numpy' (line 162)
    numpy_122021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 162)
    ndarray_122022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 26), numpy_122021, 'ndarray')
    # Processing the call keyword arguments (line 162)
    kwargs_122023 = {}
    # Getting the type of 'isinstance' (line 162)
    isinstance_122019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 162)
    isinstance_call_result_122024 = invoke(stypy.reporting.localization.Localization(__file__, 162, 7), isinstance_122019, *[output_122020, ndarray_122022], **kwargs_122023)
    
    # Testing the type of an if condition (line 162)
    if_condition_122025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), isinstance_call_result_122024)
    # Assigning a type to the variable 'if_condition_122025' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_122025', if_condition_122025)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'output' (line 163)
    output_122026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'output')
    # Obtaining the member 'shape' of a type (line 163)
    shape_122027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 11), output_122026, 'shape')
    # Getting the type of 'input' (line 163)
    input_122028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'input')
    # Obtaining the member 'shape' of a type (line 163)
    shape_122029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), input_122028, 'shape')
    # Applying the binary operator '!=' (line 163)
    result_ne_122030 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), '!=', shape_122027, shape_122029)
    
    # Testing the type of an if condition (line 163)
    if_condition_122031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_ne_122030)
    # Assigning a type to the variable 'if_condition_122031' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_122031', if_condition_122031)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 164)
    # Processing the call arguments (line 164)
    str_122033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'str', 'output shape not correct')
    # Processing the call keyword arguments (line 164)
    kwargs_122034 = {}
    # Getting the type of 'ValueError' (line 164)
    ValueError_122032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 164)
    ValueError_call_result_122035 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), ValueError_122032, *[str_122033], **kwargs_122034)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 12), ValueError_call_result_122035, 'raise parameter', BaseException)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 165):
    
    # Assigning a Name to a Name (line 165):
    # Getting the type of 'True' (line 165)
    True_122036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'True')
    # Assigning a type to the variable 'caller_provided_output' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'caller_provided_output', True_122036)
    # SSA branch for the else part of an if statement (line 162)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 167):
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'False' (line 167)
    False_122037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'False')
    # Assigning a type to the variable 'caller_provided_output' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'caller_provided_output', False_122037)
    
    # Type idiom detected: calculating its left and rigth part (line 168)
    # Getting the type of 'output' (line 168)
    output_122038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'output')
    # Getting the type of 'None' (line 168)
    None_122039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'None')
    
    (may_be_122040, more_types_in_union_122041) = may_be_none(output_122038, None_122039)

    if may_be_122040:

        if more_types_in_union_122041:
            # Runtime conditional SSA (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to empty(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'input' (line 169)
        input_122044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'input', False)
        # Obtaining the member 'shape' of a type (line 169)
        shape_122045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 30), input_122044, 'shape')
        
        # Getting the type of 'need_64bits' (line 169)
        need_64bits_122046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 54), 'need_64bits', False)
        # Testing the type of an if expression (line 169)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 43), need_64bits_122046)
        # SSA begins for if expression (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'np' (line 169)
        np_122047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'np', False)
        # Obtaining the member 'intp' of a type (line 169)
        intp_122048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 43), np_122047, 'intp')
        # SSA branch for the else part of an if expression (line 169)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'np' (line 169)
        np_122049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 71), 'np', False)
        # Obtaining the member 'int32' of a type (line 169)
        int32_122050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 71), np_122049, 'int32')
        # SSA join for if expression (line 169)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_122051 = union_type.UnionType.add(intp_122048, int32_122050)
        
        # Processing the call keyword arguments (line 169)
        kwargs_122052 = {}
        # Getting the type of 'np' (line 169)
        np_122042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 169)
        empty_122043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), np_122042, 'empty')
        # Calling empty(args, kwargs) (line 169)
        empty_call_result_122053 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), empty_122043, *[shape_122045, if_exp_122051], **kwargs_122052)
        
        # Assigning a type to the variable 'output' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'output', empty_call_result_122053)

        if more_types_in_union_122041:
            # Runtime conditional SSA for else branch (line 168)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_122040) or more_types_in_union_122041):
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to empty(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'input' (line 171)
        input_122056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'input', False)
        # Obtaining the member 'shape' of a type (line 171)
        shape_122057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 30), input_122056, 'shape')
        # Getting the type of 'output' (line 171)
        output_122058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), 'output', False)
        # Processing the call keyword arguments (line 171)
        kwargs_122059 = {}
        # Getting the type of 'np' (line 171)
        np_122054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 171)
        empty_122055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 21), np_122054, 'empty')
        # Calling empty(args, kwargs) (line 171)
        empty_call_result_122060 = invoke(stypy.reporting.localization.Localization(__file__, 171, 21), empty_122055, *[shape_122057, output_122058], **kwargs_122059)
        
        # Assigning a type to the variable 'output' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'output', empty_call_result_122060)

        if (may_be_122040 and more_types_in_union_122041):
            # SSA join for if statement (line 168)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'input' (line 174)
    input_122061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 174)
    ndim_122062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 7), input_122061, 'ndim')
    int_122063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'int')
    # Applying the binary operator '==' (line 174)
    result_eq_122064 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), '==', ndim_122062, int_122063)
    
    
    # Getting the type of 'input' (line 174)
    input_122065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'input')
    # Obtaining the member 'size' of a type (line 174)
    size_122066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), input_122065, 'size')
    int_122067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
    # Applying the binary operator '==' (line 174)
    result_eq_122068 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 26), '==', size_122066, int_122067)
    
    # Applying the binary operator 'or' (line 174)
    result_or_keyword_122069 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), 'or', result_eq_122064, result_eq_122068)
    
    # Testing the type of an if condition (line 174)
    if_condition_122070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_or_keyword_122069)
    # Assigning a type to the variable 'if_condition_122070' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_122070', if_condition_122070)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'input' (line 175)
    input_122071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 11), 'input')
    # Obtaining the member 'ndim' of a type (line 175)
    ndim_122072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 11), input_122071, 'ndim')
    int_122073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 25), 'int')
    # Applying the binary operator '==' (line 175)
    result_eq_122074 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 11), '==', ndim_122072, int_122073)
    
    # Testing the type of an if condition (line 175)
    if_condition_122075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 8), result_eq_122074)
    # Assigning a type to the variable 'if_condition_122075' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'if_condition_122075', if_condition_122075)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a IfExp to a Name (line 177):
    
    # Assigning a IfExp to a Name (line 177):
    
    
    # Getting the type of 'input' (line 177)
    input_122076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 29), 'input')
    int_122077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 38), 'int')
    # Applying the binary operator '!=' (line 177)
    result_ne_122078 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 29), '!=', input_122076, int_122077)
    
    # Testing the type of an if expression (line 177)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 23), result_ne_122078)
    # SSA begins for if expression (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    int_122079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'int')
    # SSA branch for the else part of an if expression (line 177)
    module_type_store.open_ssa_branch('if expression else')
    int_122080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 46), 'int')
    # SSA join for if expression (line 177)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_122081 = union_type.UnionType.add(int_122079, int_122080)
    
    # Assigning a type to the variable 'maxlabel' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'maxlabel', if_exp_122081)
    
    # Assigning a Name to a Subscript (line 178):
    
    # Assigning a Name to a Subscript (line 178):
    # Getting the type of 'maxlabel' (line 178)
    maxlabel_122082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'maxlabel')
    # Getting the type of 'output' (line 178)
    output_122083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'output')
    Ellipsis_122084 = Ellipsis
    # Storing an element on a container (line 178)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 12), output_122083, (Ellipsis_122084, maxlabel_122082))
    # SSA branch for the else part of an if statement (line 175)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 181):
    
    # Assigning a Num to a Name (line 181):
    int_122085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
    # Assigning a type to the variable 'maxlabel' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'maxlabel', int_122085)
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'caller_provided_output' (line 182)
    caller_provided_output_122086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'caller_provided_output')
    # Testing the type of an if condition (line 182)
    if_condition_122087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), caller_provided_output_122086)
    # Assigning a type to the variable 'if_condition_122087' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_122087', if_condition_122087)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'maxlabel' (line 183)
    maxlabel_122088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'maxlabel')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type', maxlabel_122088)
    # SSA branch for the else part of an if statement (line 182)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 185)
    tuple_122089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'output' (line 185)
    output_122090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_122089, output_122090)
    # Adding element type (line 185)
    # Getting the type of 'maxlabel' (line 185)
    maxlabel_122091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 27), 'maxlabel')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_122089, maxlabel_122091)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'stypy_return_type', tuple_122089)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to _label(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'input' (line 188)
    input_122094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'input', False)
    # Getting the type of 'structure' (line 188)
    structure_122095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 44), 'structure', False)
    # Getting the type of 'output' (line 188)
    output_122096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 55), 'output', False)
    # Processing the call keyword arguments (line 188)
    kwargs_122097 = {}
    # Getting the type of '_ni_label' (line 188)
    _ni_label_122092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), '_ni_label', False)
    # Obtaining the member '_label' of a type (line 188)
    _label_122093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 20), _ni_label_122092, '_label')
    # Calling _label(args, kwargs) (line 188)
    _label_call_result_122098 = invoke(stypy.reporting.localization.Localization(__file__, 188, 20), _label_122093, *[input_122094, structure_122095, output_122096], **kwargs_122097)
    
    # Assigning a type to the variable 'max_label' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'max_label', _label_call_result_122098)
    # SSA branch for the except part of a try statement (line 187)
    # SSA branch for the except 'Attribute' branch of a try statement (line 187)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to empty(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'input' (line 192)
    input_122101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 30), 'input', False)
    # Obtaining the member 'shape' of a type (line 192)
    shape_122102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 30), input_122101, 'shape')
    
    # Getting the type of 'need_64bits' (line 192)
    need_64bits_122103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 54), 'need_64bits', False)
    # Testing the type of an if expression (line 192)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 43), need_64bits_122103)
    # SSA begins for if expression (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'np' (line 192)
    np_122104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 43), 'np', False)
    # Obtaining the member 'intp' of a type (line 192)
    intp_122105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 43), np_122104, 'intp')
    # SSA branch for the else part of an if expression (line 192)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'np' (line 192)
    np_122106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 71), 'np', False)
    # Obtaining the member 'int32' of a type (line 192)
    int32_122107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 71), np_122106, 'int32')
    # SSA join for if expression (line 192)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_122108 = union_type.UnionType.add(intp_122105, int32_122107)
    
    # Processing the call keyword arguments (line 192)
    kwargs_122109 = {}
    # Getting the type of 'np' (line 192)
    np_122099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 21), 'np', False)
    # Obtaining the member 'empty' of a type (line 192)
    empty_122100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 21), np_122099, 'empty')
    # Calling empty(args, kwargs) (line 192)
    empty_call_result_122110 = invoke(stypy.reporting.localization.Localization(__file__, 192, 21), empty_122100, *[shape_122102, if_exp_122108], **kwargs_122109)
    
    # Assigning a type to the variable 'tmp_output' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'tmp_output', empty_call_result_122110)
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to _label(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'input' (line 193)
    input_122113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'input', False)
    # Getting the type of 'structure' (line 193)
    structure_122114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 44), 'structure', False)
    # Getting the type of 'tmp_output' (line 193)
    tmp_output_122115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'tmp_output', False)
    # Processing the call keyword arguments (line 193)
    kwargs_122116 = {}
    # Getting the type of '_ni_label' (line 193)
    _ni_label_122111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), '_ni_label', False)
    # Obtaining the member '_label' of a type (line 193)
    _label_122112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 20), _ni_label_122111, '_label')
    # Calling _label(args, kwargs) (line 193)
    _label_call_result_122117 = invoke(stypy.reporting.localization.Localization(__file__, 193, 20), _label_122112, *[input_122113, structure_122114, tmp_output_122115], **kwargs_122116)
    
    # Assigning a type to the variable 'max_label' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'max_label', _label_call_result_122117)
    
    # Assigning a Subscript to a Subscript (line 194):
    
    # Assigning a Subscript to a Subscript (line 194):
    
    # Obtaining the type of the subscript
    Ellipsis_122118 = Ellipsis
    # Getting the type of 'tmp_output' (line 194)
    tmp_output_122119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'tmp_output')
    # Obtaining the member '__getitem__' of a type (line 194)
    getitem___122120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 22), tmp_output_122119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 194)
    subscript_call_result_122121 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), getitem___122120, Ellipsis_122118)
    
    # Getting the type of 'output' (line 194)
    output_122122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'output')
    Ellipsis_122123 = Ellipsis
    # Storing an element on a container (line 194)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 8), output_122122, (Ellipsis_122123, subscript_call_result_122121))
    
    
    
    # Call to all(...): (line 195)
    # Processing the call arguments (line 195)
    
    # Getting the type of 'output' (line 195)
    output_122126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'output', False)
    # Getting the type of 'tmp_output' (line 195)
    tmp_output_122127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'tmp_output', False)
    # Applying the binary operator '==' (line 195)
    result_eq_122128 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 22), '==', output_122126, tmp_output_122127)
    
    # Processing the call keyword arguments (line 195)
    kwargs_122129 = {}
    # Getting the type of 'np' (line 195)
    np_122124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 195)
    all_122125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), np_122124, 'all')
    # Calling all(args, kwargs) (line 195)
    all_call_result_122130 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), all_122125, *[result_eq_122128], **kwargs_122129)
    
    # Applying the 'not' unary operator (line 195)
    result_not__122131 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'not', all_call_result_122130)
    
    # Testing the type of an if condition (line 195)
    if_condition_122132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 8), result_not__122131)
    # Assigning a type to the variable 'if_condition_122132' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'if_condition_122132', if_condition_122132)
    # SSA begins for if statement (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 197)
    # Processing the call arguments (line 197)
    str_122134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 31), 'str', 'insufficient bit-depth in requested output type')
    # Processing the call keyword arguments (line 197)
    kwargs_122135 = {}
    # Getting the type of 'RuntimeError' (line 197)
    RuntimeError_122133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 197)
    RuntimeError_call_result_122136 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), RuntimeError_122133, *[str_122134], **kwargs_122135)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 197, 12), RuntimeError_call_result_122136, 'raise parameter', BaseException)
    # SSA join for if statement (line 195)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'caller_provided_output' (line 199)
    caller_provided_output_122137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 7), 'caller_provided_output')
    # Testing the type of an if condition (line 199)
    if_condition_122138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), caller_provided_output_122137)
    # Assigning a type to the variable 'if_condition_122138' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_122138', if_condition_122138)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'max_label' (line 201)
    max_label_122139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'max_label')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', max_label_122139)
    # SSA branch for the else part of an if statement (line 199)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_122140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    # Getting the type of 'output' (line 203)
    output_122141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'output')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), tuple_122140, output_122141)
    # Adding element type (line 203)
    # Getting the type of 'max_label' (line 203)
    max_label_122142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'max_label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 15), tuple_122140, max_label_122142)
    
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'stypy_return_type', tuple_122140)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'label(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'label' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_122143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122143)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'label'
    return stypy_return_type_122143

# Assigning a type to the variable 'label' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'label', label)

@norecursion
def find_objects(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_122144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 34), 'int')
    defaults = [int_122144]
    # Create a new context for function 'find_objects'
    module_type_store = module_type_store.open_function_context('find_objects', 206, 0, False)
    
    # Passed parameters checking function
    find_objects.stypy_localization = localization
    find_objects.stypy_type_of_self = None
    find_objects.stypy_type_store = module_type_store
    find_objects.stypy_function_name = 'find_objects'
    find_objects.stypy_param_names_list = ['input', 'max_label']
    find_objects.stypy_varargs_param_name = None
    find_objects.stypy_kwargs_param_name = None
    find_objects.stypy_call_defaults = defaults
    find_objects.stypy_call_varargs = varargs
    find_objects.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_objects', ['input', 'max_label'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_objects', localization, ['input', 'max_label'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_objects(...)' code ##################

    str_122145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, (-1)), 'str', '\n    Find objects in a labeled array.\n\n    Parameters\n    ----------\n    input : ndarray of ints\n        Array containing objects defined by different labels. Labels with\n        value 0 are ignored.\n    max_label : int, optional\n        Maximum label to be searched for in `input`. If max_label is not\n        given, the positions of all objects are returned.\n\n    Returns\n    -------\n    object_slices : list of tuples\n        A list of tuples, with each tuple containing N slices (with N the\n        dimension of the input array).  Slices correspond to the minimal\n        parallelepiped that contains the object. If a number is missing,\n        None is returned instead of a slice.\n\n    See Also\n    --------\n    label, center_of_mass\n\n    Notes\n    -----\n    This function is very useful for isolating a volume of interest inside\n    a 3-D array, that cannot be "seen through".\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.zeros((6,6), dtype=int)\n    >>> a[2:4, 2:4] = 1\n    >>> a[4, 4] = 1\n    >>> a[:2, :3] = 2\n    >>> a[0, 5] = 3\n    >>> a\n    array([[2, 2, 2, 0, 0, 3],\n           [2, 2, 2, 0, 0, 0],\n           [0, 0, 1, 1, 0, 0],\n           [0, 0, 1, 1, 0, 0],\n           [0, 0, 0, 0, 1, 0],\n           [0, 0, 0, 0, 0, 0]])\n    >>> ndimage.find_objects(a)\n    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None)), (slice(0, 1, None), slice(5, 6, None))]\n    >>> ndimage.find_objects(a, max_label=2)\n    [(slice(2, 5, None), slice(2, 5, None)), (slice(0, 2, None), slice(0, 3, None))]\n    >>> ndimage.find_objects(a == 1, max_label=2)\n    [(slice(2, 5, None), slice(2, 5, None)), None]\n\n    >>> loc = ndimage.find_objects(a)[0]\n    >>> a[loc]\n    array([[1, 1, 0],\n           [1, 1, 0],\n           [0, 0, 1]])\n\n    ')
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to asarray(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'input' (line 265)
    input_122148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 26), 'input', False)
    # Processing the call keyword arguments (line 265)
    kwargs_122149 = {}
    # Getting the type of 'numpy' (line 265)
    numpy_122146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 265)
    asarray_122147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), numpy_122146, 'asarray')
    # Calling asarray(args, kwargs) (line 265)
    asarray_call_result_122150 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), asarray_122147, *[input_122148], **kwargs_122149)
    
    # Assigning a type to the variable 'input' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'input', asarray_call_result_122150)
    
    
    # Call to iscomplexobj(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'input' (line 266)
    input_122153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'input', False)
    # Processing the call keyword arguments (line 266)
    kwargs_122154 = {}
    # Getting the type of 'numpy' (line 266)
    numpy_122151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 266)
    iscomplexobj_122152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 7), numpy_122151, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 266)
    iscomplexobj_call_result_122155 = invoke(stypy.reporting.localization.Localization(__file__, 266, 7), iscomplexobj_122152, *[input_122153], **kwargs_122154)
    
    # Testing the type of an if condition (line 266)
    if_condition_122156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 4), iscomplexobj_call_result_122155)
    # Assigning a type to the variable 'if_condition_122156' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'if_condition_122156', if_condition_122156)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 267)
    # Processing the call arguments (line 267)
    str_122158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 267)
    kwargs_122159 = {}
    # Getting the type of 'TypeError' (line 267)
    TypeError_122157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 267)
    TypeError_call_result_122160 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), TypeError_122157, *[str_122158], **kwargs_122159)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 267, 8), TypeError_call_result_122160, 'raise parameter', BaseException)
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'max_label' (line 269)
    max_label_122161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 7), 'max_label')
    int_122162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 19), 'int')
    # Applying the binary operator '<' (line 269)
    result_lt_122163 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 7), '<', max_label_122161, int_122162)
    
    # Testing the type of an if condition (line 269)
    if_condition_122164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 4), result_lt_122163)
    # Assigning a type to the variable 'if_condition_122164' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'if_condition_122164', if_condition_122164)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 270):
    
    # Assigning a Call to a Name (line 270):
    
    # Call to max(...): (line 270)
    # Processing the call keyword arguments (line 270)
    kwargs_122167 = {}
    # Getting the type of 'input' (line 270)
    input_122165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'input', False)
    # Obtaining the member 'max' of a type (line 270)
    max_122166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), input_122165, 'max')
    # Calling max(args, kwargs) (line 270)
    max_call_result_122168 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), max_122166, *[], **kwargs_122167)
    
    # Assigning a type to the variable 'max_label' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'max_label', max_call_result_122168)
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to find_objects(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'input' (line 272)
    input_122171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'input', False)
    # Getting the type of 'max_label' (line 272)
    max_label_122172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 41), 'max_label', False)
    # Processing the call keyword arguments (line 272)
    kwargs_122173 = {}
    # Getting the type of '_nd_image' (line 272)
    _nd_image_122169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), '_nd_image', False)
    # Obtaining the member 'find_objects' of a type (line 272)
    find_objects_122170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 11), _nd_image_122169, 'find_objects')
    # Calling find_objects(args, kwargs) (line 272)
    find_objects_call_result_122174 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), find_objects_122170, *[input_122171, max_label_122172], **kwargs_122173)
    
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', find_objects_call_result_122174)
    
    # ################# End of 'find_objects(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_objects' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_122175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122175)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_objects'
    return stypy_return_type_122175

# Assigning a type to the variable 'find_objects' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'find_objects', find_objects)

@norecursion
def labeled_comprehension(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 275)
    False_122176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'False')
    defaults = [False_122176]
    # Create a new context for function 'labeled_comprehension'
    module_type_store = module_type_store.open_function_context('labeled_comprehension', 275, 0, False)
    
    # Passed parameters checking function
    labeled_comprehension.stypy_localization = localization
    labeled_comprehension.stypy_type_of_self = None
    labeled_comprehension.stypy_type_store = module_type_store
    labeled_comprehension.stypy_function_name = 'labeled_comprehension'
    labeled_comprehension.stypy_param_names_list = ['input', 'labels', 'index', 'func', 'out_dtype', 'default', 'pass_positions']
    labeled_comprehension.stypy_varargs_param_name = None
    labeled_comprehension.stypy_kwargs_param_name = None
    labeled_comprehension.stypy_call_defaults = defaults
    labeled_comprehension.stypy_call_varargs = varargs
    labeled_comprehension.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'labeled_comprehension', ['input', 'labels', 'index', 'func', 'out_dtype', 'default', 'pass_positions'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'labeled_comprehension', localization, ['input', 'labels', 'index', 'func', 'out_dtype', 'default', 'pass_positions'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'labeled_comprehension(...)' code ##################

    str_122177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, (-1)), 'str', '\n    Roughly equivalent to [func(input[labels == i]) for i in index].\n\n    Sequentially applies an arbitrary function (that works on array_like input)\n    to subsets of an n-D image array specified by `labels` and `index`.\n    The option exists to provide the function with positional parameters as the\n    second argument.\n\n    Parameters\n    ----------\n    input : array_like\n        Data from which to select `labels` to process.\n    labels : array_like or None\n        Labels to objects in `input`.\n        If not None, array must be same shape as `input`.\n        If None, `func` is applied to raveled `input`.\n    index : int, sequence of ints or None\n        Subset of `labels` to which to apply `func`.\n        If a scalar, a single value is returned.\n        If None, `func` is applied to all non-zero values of `labels`.\n    func : callable\n        Python function to apply to `labels` from `input`.\n    out_dtype : dtype\n        Dtype to use for `result`.\n    default : int, float or None\n        Default return value when a element of `index` does not exist\n        in `labels`.\n    pass_positions : bool, optional\n        If True, pass linear indices to `func` as a second argument.\n        Default is False.\n\n    Returns\n    -------\n    result : ndarray\n        Result of applying `func` to each of `labels` to `input` in `index`.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> from scipy import ndimage\n    >>> lbl, nlbl = ndimage.label(a)\n    >>> lbls = np.arange(1, nlbl+1)\n    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, 0)\n    array([ 2.75,  5.5 ,  6.  ])\n\n    Falling back to `default`:\n\n    >>> lbls = np.arange(1, nlbl+2)\n    >>> ndimage.labeled_comprehension(a, lbl, lbls, np.mean, float, -1)\n    array([ 2.75,  5.5 ,  6.  , -1.  ])\n\n    Passing positions:\n\n    >>> def fn(val, pos):\n    ...     print("fn says: %s : %s" % (val, pos))\n    ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())\n    ...\n    >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)\n    fn says: [1 2 5 3] : [0 1 4 5]\n    fn says: [4 7] : [ 7 11]\n    fn says: [9 3] : [12 13]\n    array([ 11.,  11., -12.,   0.])\n\n    ')
    
    # Assigning a Call to a Name (line 344):
    
    # Assigning a Call to a Name (line 344):
    
    # Call to isscalar(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'index' (line 344)
    index_122180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 'index', False)
    # Processing the call keyword arguments (line 344)
    kwargs_122181 = {}
    # Getting the type of 'numpy' (line 344)
    numpy_122178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 344)
    isscalar_122179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 16), numpy_122178, 'isscalar')
    # Calling isscalar(args, kwargs) (line 344)
    isscalar_call_result_122182 = invoke(stypy.reporting.localization.Localization(__file__, 344, 16), isscalar_122179, *[index_122180], **kwargs_122181)
    
    # Assigning a type to the variable 'as_scalar' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'as_scalar', isscalar_call_result_122182)
    
    # Assigning a Call to a Name (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to asarray(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'input' (line 345)
    input_122185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'input', False)
    # Processing the call keyword arguments (line 345)
    kwargs_122186 = {}
    # Getting the type of 'numpy' (line 345)
    numpy_122183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 345)
    asarray_122184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), numpy_122183, 'asarray')
    # Calling asarray(args, kwargs) (line 345)
    asarray_call_result_122187 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), asarray_122184, *[input_122185], **kwargs_122186)
    
    # Assigning a type to the variable 'input' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'input', asarray_call_result_122187)
    
    # Getting the type of 'pass_positions' (line 347)
    pass_positions_122188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 7), 'pass_positions')
    # Testing the type of an if condition (line 347)
    if_condition_122189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 4), pass_positions_122188)
    # Assigning a type to the variable 'if_condition_122189' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'if_condition_122189', if_condition_122189)
    # SSA begins for if statement (line 347)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to reshape(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'input' (line 348)
    input_122197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 53), 'input', False)
    # Obtaining the member 'shape' of a type (line 348)
    shape_122198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 53), input_122197, 'shape')
    # Processing the call keyword arguments (line 348)
    kwargs_122199 = {}
    
    # Call to arange(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'input' (line 348)
    input_122192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 33), 'input', False)
    # Obtaining the member 'size' of a type (line 348)
    size_122193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 33), input_122192, 'size')
    # Processing the call keyword arguments (line 348)
    kwargs_122194 = {}
    # Getting the type of 'numpy' (line 348)
    numpy_122190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 20), 'numpy', False)
    # Obtaining the member 'arange' of a type (line 348)
    arange_122191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), numpy_122190, 'arange')
    # Calling arange(args, kwargs) (line 348)
    arange_call_result_122195 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), arange_122191, *[size_122193], **kwargs_122194)
    
    # Obtaining the member 'reshape' of a type (line 348)
    reshape_122196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 20), arange_call_result_122195, 'reshape')
    # Calling reshape(args, kwargs) (line 348)
    reshape_call_result_122200 = invoke(stypy.reporting.localization.Localization(__file__, 348, 20), reshape_122196, *[shape_122198], **kwargs_122199)
    
    # Assigning a type to the variable 'positions' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'positions', reshape_call_result_122200)
    # SSA join for if statement (line 347)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 350)
    # Getting the type of 'labels' (line 350)
    labels_122201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 7), 'labels')
    # Getting the type of 'None' (line 350)
    None_122202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 17), 'None')
    
    (may_be_122203, more_types_in_union_122204) = may_be_none(labels_122201, None_122202)

    if may_be_122203:

        if more_types_in_union_122204:
            # Runtime conditional SSA (line 350)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 351)
        # Getting the type of 'index' (line 351)
        index_122205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'index')
        # Getting the type of 'None' (line 351)
        None_122206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 24), 'None')
        
        (may_be_122207, more_types_in_union_122208) = may_not_be_none(index_122205, None_122206)

        if may_be_122207:

            if more_types_in_union_122208:
                # Runtime conditional SSA (line 351)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 352)
            # Processing the call arguments (line 352)
            str_122210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 29), 'str', 'index without defined labels')
            # Processing the call keyword arguments (line 352)
            kwargs_122211 = {}
            # Getting the type of 'ValueError' (line 352)
            ValueError_122209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 352)
            ValueError_call_result_122212 = invoke(stypy.reporting.localization.Localization(__file__, 352, 18), ValueError_122209, *[str_122210], **kwargs_122211)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 352, 12), ValueError_call_result_122212, 'raise parameter', BaseException)

            if more_types_in_union_122208:
                # SSA join for if statement (line 351)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'pass_positions' (line 353)
        pass_positions_122213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'pass_positions')
        # Applying the 'not' unary operator (line 353)
        result_not__122214 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 11), 'not', pass_positions_122213)
        
        # Testing the type of an if condition (line 353)
        if_condition_122215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), result_not__122214)
        # Assigning a type to the variable 'if_condition_122215' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_122215', if_condition_122215)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to ravel(...): (line 354)
        # Processing the call keyword arguments (line 354)
        kwargs_122219 = {}
        # Getting the type of 'input' (line 354)
        input_122217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'input', False)
        # Obtaining the member 'ravel' of a type (line 354)
        ravel_122218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), input_122217, 'ravel')
        # Calling ravel(args, kwargs) (line 354)
        ravel_call_result_122220 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), ravel_122218, *[], **kwargs_122219)
        
        # Processing the call keyword arguments (line 354)
        kwargs_122221 = {}
        # Getting the type of 'func' (line 354)
        func_122216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'func', False)
        # Calling func(args, kwargs) (line 354)
        func_call_result_122222 = invoke(stypy.reporting.localization.Localization(__file__, 354, 19), func_122216, *[ravel_call_result_122220], **kwargs_122221)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'stypy_return_type', func_call_result_122222)
        # SSA branch for the else part of an if statement (line 353)
        module_type_store.open_ssa_branch('else')
        
        # Call to func(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to ravel(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_122226 = {}
        # Getting the type of 'input' (line 356)
        input_122224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'input', False)
        # Obtaining the member 'ravel' of a type (line 356)
        ravel_122225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 24), input_122224, 'ravel')
        # Calling ravel(args, kwargs) (line 356)
        ravel_call_result_122227 = invoke(stypy.reporting.localization.Localization(__file__, 356, 24), ravel_122225, *[], **kwargs_122226)
        
        
        # Call to ravel(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_122230 = {}
        # Getting the type of 'positions' (line 356)
        positions_122228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'positions', False)
        # Obtaining the member 'ravel' of a type (line 356)
        ravel_122229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 39), positions_122228, 'ravel')
        # Calling ravel(args, kwargs) (line 356)
        ravel_call_result_122231 = invoke(stypy.reporting.localization.Localization(__file__, 356, 39), ravel_122229, *[], **kwargs_122230)
        
        # Processing the call keyword arguments (line 356)
        kwargs_122232 = {}
        # Getting the type of 'func' (line 356)
        func_122223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'func', False)
        # Calling func(args, kwargs) (line 356)
        func_call_result_122233 = invoke(stypy.reporting.localization.Localization(__file__, 356, 19), func_122223, *[ravel_call_result_122227, ravel_call_result_122231], **kwargs_122232)
        
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type', func_call_result_122233)
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_122204:
            # SSA join for if statement (line 350)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 359):
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_122234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 8), 'int')
    
    # Call to broadcast_arrays(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'input' (line 359)
    input_122237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), 'input', False)
    # Getting the type of 'labels' (line 359)
    labels_122238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 54), 'labels', False)
    # Processing the call keyword arguments (line 359)
    kwargs_122239 = {}
    # Getting the type of 'numpy' (line 359)
    numpy_122235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 359)
    broadcast_arrays_122236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), numpy_122235, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 359)
    broadcast_arrays_call_result_122240 = invoke(stypy.reporting.localization.Localization(__file__, 359, 24), broadcast_arrays_122236, *[input_122237, labels_122238], **kwargs_122239)
    
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___122241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), broadcast_arrays_call_result_122240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_122242 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___122241, int_122234)
    
    # Assigning a type to the variable 'tuple_var_assignment_121902' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_121902', subscript_call_result_122242)
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_122243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 8), 'int')
    
    # Call to broadcast_arrays(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'input' (line 359)
    input_122246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), 'input', False)
    # Getting the type of 'labels' (line 359)
    labels_122247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 54), 'labels', False)
    # Processing the call keyword arguments (line 359)
    kwargs_122248 = {}
    # Getting the type of 'numpy' (line 359)
    numpy_122244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 359)
    broadcast_arrays_122245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), numpy_122244, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 359)
    broadcast_arrays_call_result_122249 = invoke(stypy.reporting.localization.Localization(__file__, 359, 24), broadcast_arrays_122245, *[input_122246, labels_122247], **kwargs_122248)
    
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___122250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), broadcast_arrays_call_result_122249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_122251 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___122250, int_122243)
    
    # Assigning a type to the variable 'tuple_var_assignment_121903' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_121903', subscript_call_result_122251)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_121902' (line 359)
    tuple_var_assignment_121902_122252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_121902')
    # Assigning a type to the variable 'input' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'input', tuple_var_assignment_121902_122252)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_121903' (line 359)
    tuple_var_assignment_121903_122253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'tuple_var_assignment_121903')
    # Assigning a type to the variable 'labels' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'labels', tuple_var_assignment_121903_122253)
    # SSA branch for the except part of a try statement (line 358)
    # SSA branch for the except 'ValueError' branch of a try statement (line 358)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 361)
    # Processing the call arguments (line 361)
    str_122255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 25), 'str', 'input and labels must have the same shape (excepting dimensions with width 1)')
    # Processing the call keyword arguments (line 361)
    kwargs_122256 = {}
    # Getting the type of 'ValueError' (line 361)
    ValueError_122254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 361)
    ValueError_call_result_122257 = invoke(stypy.reporting.localization.Localization(__file__, 361, 14), ValueError_122254, *[str_122255], **kwargs_122256)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 361, 8), ValueError_call_result_122257, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 358)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 364)
    # Getting the type of 'index' (line 364)
    index_122258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 7), 'index')
    # Getting the type of 'None' (line 364)
    None_122259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'None')
    
    (may_be_122260, more_types_in_union_122261) = may_be_none(index_122258, None_122259)

    if may_be_122260:

        if more_types_in_union_122261:
            # Runtime conditional SSA (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'pass_positions' (line 365)
        pass_positions_122262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'pass_positions')
        # Applying the 'not' unary operator (line 365)
        result_not__122263 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 11), 'not', pass_positions_122262)
        
        # Testing the type of an if condition (line 365)
        if_condition_122264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), result_not__122263)
        # Assigning a type to the variable 'if_condition_122264' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_122264', if_condition_122264)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'labels' (line 366)
        labels_122266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 30), 'labels', False)
        int_122267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'int')
        # Applying the binary operator '>' (line 366)
        result_gt_122268 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 30), '>', labels_122266, int_122267)
        
        # Getting the type of 'input' (line 366)
        input_122269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'input', False)
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___122270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 24), input_122269, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_122271 = invoke(stypy.reporting.localization.Localization(__file__, 366, 24), getitem___122270, result_gt_122268)
        
        # Processing the call keyword arguments (line 366)
        kwargs_122272 = {}
        # Getting the type of 'func' (line 366)
        func_122265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'func', False)
        # Calling func(args, kwargs) (line 366)
        func_call_result_122273 = invoke(stypy.reporting.localization.Localization(__file__, 366, 19), func_122265, *[subscript_call_result_122271], **kwargs_122272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'stypy_return_type', func_call_result_122273)
        # SSA branch for the else part of an if statement (line 365)
        module_type_store.open_ssa_branch('else')
        
        # Call to func(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'labels' (line 368)
        labels_122275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'labels', False)
        int_122276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 39), 'int')
        # Applying the binary operator '>' (line 368)
        result_gt_122277 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 30), '>', labels_122275, int_122276)
        
        # Getting the type of 'input' (line 368)
        input_122278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 24), 'input', False)
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___122279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 24), input_122278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_122280 = invoke(stypy.reporting.localization.Localization(__file__, 368, 24), getitem___122279, result_gt_122277)
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'labels' (line 368)
        labels_122281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 53), 'labels', False)
        int_122282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 62), 'int')
        # Applying the binary operator '>' (line 368)
        result_gt_122283 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 53), '>', labels_122281, int_122282)
        
        # Getting the type of 'positions' (line 368)
        positions_122284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 43), 'positions', False)
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___122285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 43), positions_122284, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_122286 = invoke(stypy.reporting.localization.Localization(__file__, 368, 43), getitem___122285, result_gt_122283)
        
        # Processing the call keyword arguments (line 368)
        kwargs_122287 = {}
        # Getting the type of 'func' (line 368)
        func_122274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 19), 'func', False)
        # Calling func(args, kwargs) (line 368)
        func_call_result_122288 = invoke(stypy.reporting.localization.Localization(__file__, 368, 19), func_122274, *[subscript_call_result_122280, subscript_call_result_122286], **kwargs_122287)
        
        # Assigning a type to the variable 'stypy_return_type' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'stypy_return_type', func_call_result_122288)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_122261:
            # SSA join for if statement (line 364)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 370):
    
    # Assigning a Call to a Name (line 370):
    
    # Call to atleast_1d(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'index' (line 370)
    index_122291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 29), 'index', False)
    # Processing the call keyword arguments (line 370)
    kwargs_122292 = {}
    # Getting the type of 'numpy' (line 370)
    numpy_122289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'numpy', False)
    # Obtaining the member 'atleast_1d' of a type (line 370)
    atleast_1d_122290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), numpy_122289, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 370)
    atleast_1d_call_result_122293 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), atleast_1d_122290, *[index_122291], **kwargs_122292)
    
    # Assigning a type to the variable 'index' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'index', atleast_1d_call_result_122293)
    
    
    # Call to any(...): (line 371)
    # Processing the call arguments (line 371)
    
    
    # Call to astype(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'index' (line 371)
    index_122303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 48), 'index', False)
    # Obtaining the member 'dtype' of a type (line 371)
    dtype_122304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 48), index_122303, 'dtype')
    # Processing the call keyword arguments (line 371)
    kwargs_122305 = {}
    
    # Call to astype(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'labels' (line 371)
    labels_122298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 27), 'labels', False)
    # Obtaining the member 'dtype' of a type (line 371)
    dtype_122299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 27), labels_122298, 'dtype')
    # Processing the call keyword arguments (line 371)
    kwargs_122300 = {}
    # Getting the type of 'index' (line 371)
    index_122296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'index', False)
    # Obtaining the member 'astype' of a type (line 371)
    astype_122297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 14), index_122296, 'astype')
    # Calling astype(args, kwargs) (line 371)
    astype_call_result_122301 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), astype_122297, *[dtype_122299], **kwargs_122300)
    
    # Obtaining the member 'astype' of a type (line 371)
    astype_122302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 14), astype_call_result_122301, 'astype')
    # Calling astype(args, kwargs) (line 371)
    astype_call_result_122306 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), astype_122302, *[dtype_122304], **kwargs_122305)
    
    # Getting the type of 'index' (line 371)
    index_122307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 64), 'index', False)
    # Applying the binary operator '!=' (line 371)
    result_ne_122308 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 14), '!=', astype_call_result_122306, index_122307)
    
    # Processing the call keyword arguments (line 371)
    kwargs_122309 = {}
    # Getting the type of 'np' (line 371)
    np_122294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 371)
    any_122295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 7), np_122294, 'any')
    # Calling any(args, kwargs) (line 371)
    any_call_result_122310 = invoke(stypy.reporting.localization.Localization(__file__, 371, 7), any_122295, *[result_ne_122308], **kwargs_122309)
    
    # Testing the type of an if condition (line 371)
    if_condition_122311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 4), any_call_result_122310)
    # Assigning a type to the variable 'if_condition_122311' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'if_condition_122311', if_condition_122311)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 372)
    # Processing the call arguments (line 372)
    str_122313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 25), 'str', "Cannot convert index values from <%s> to <%s> (labels' type) without loss of precision")
    
    # Obtaining an instance of the builtin type 'tuple' (line 374)
    tuple_122314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 374)
    # Adding element type (line 374)
    # Getting the type of 'index' (line 374)
    index_122315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 29), 'index', False)
    # Obtaining the member 'dtype' of a type (line 374)
    dtype_122316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 29), index_122315, 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), tuple_122314, dtype_122316)
    # Adding element type (line 374)
    # Getting the type of 'labels' (line 374)
    labels_122317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 42), 'labels', False)
    # Obtaining the member 'dtype' of a type (line 374)
    dtype_122318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 42), labels_122317, 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), tuple_122314, dtype_122318)
    
    # Applying the binary operator '%' (line 372)
    result_mod_122319 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 25), '%', str_122313, tuple_122314)
    
    # Processing the call keyword arguments (line 372)
    kwargs_122320 = {}
    # Getting the type of 'ValueError' (line 372)
    ValueError_122312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 372)
    ValueError_call_result_122321 = invoke(stypy.reporting.localization.Localization(__file__, 372, 14), ValueError_122312, *[result_mod_122319], **kwargs_122320)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 372, 8), ValueError_call_result_122321, 'raise parameter', BaseException)
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 376):
    
    # Assigning a Call to a Name (line 376):
    
    # Call to astype(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'labels' (line 376)
    labels_122324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 25), 'labels', False)
    # Obtaining the member 'dtype' of a type (line 376)
    dtype_122325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 25), labels_122324, 'dtype')
    # Processing the call keyword arguments (line 376)
    kwargs_122326 = {}
    # Getting the type of 'index' (line 376)
    index_122322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'index', False)
    # Obtaining the member 'astype' of a type (line 376)
    astype_122323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), index_122322, 'astype')
    # Calling astype(args, kwargs) (line 376)
    astype_call_result_122327 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), astype_122323, *[dtype_122325], **kwargs_122326)
    
    # Assigning a type to the variable 'index' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'index', astype_call_result_122327)
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to min(...): (line 379)
    # Processing the call keyword arguments (line 379)
    kwargs_122330 = {}
    # Getting the type of 'index' (line 379)
    index_122328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 9), 'index', False)
    # Obtaining the member 'min' of a type (line 379)
    min_122329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 9), index_122328, 'min')
    # Calling min(args, kwargs) (line 379)
    min_call_result_122331 = invoke(stypy.reporting.localization.Localization(__file__, 379, 9), min_122329, *[], **kwargs_122330)
    
    # Assigning a type to the variable 'lo' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'lo', min_call_result_122331)
    
    # Assigning a Call to a Name (line 380):
    
    # Assigning a Call to a Name (line 380):
    
    # Call to max(...): (line 380)
    # Processing the call keyword arguments (line 380)
    kwargs_122334 = {}
    # Getting the type of 'index' (line 380)
    index_122332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 9), 'index', False)
    # Obtaining the member 'max' of a type (line 380)
    max_122333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 9), index_122332, 'max')
    # Calling max(args, kwargs) (line 380)
    max_call_result_122335 = invoke(stypy.reporting.localization.Localization(__file__, 380, 9), max_122333, *[], **kwargs_122334)
    
    # Assigning a type to the variable 'hi' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'hi', max_call_result_122335)
    
    # Assigning a BinOp to a Name (line 381):
    
    # Assigning a BinOp to a Name (line 381):
    
    # Getting the type of 'labels' (line 381)
    labels_122336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'labels')
    # Getting the type of 'lo' (line 381)
    lo_122337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 22), 'lo')
    # Applying the binary operator '>=' (line 381)
    result_ge_122338 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), '>=', labels_122336, lo_122337)
    
    
    # Getting the type of 'labels' (line 381)
    labels_122339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 29), 'labels')
    # Getting the type of 'hi' (line 381)
    hi_122340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 39), 'hi')
    # Applying the binary operator '<=' (line 381)
    result_le_122341 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 29), '<=', labels_122339, hi_122340)
    
    # Applying the binary operator '&' (line 381)
    result_and__122342 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), '&', result_ge_122338, result_le_122341)
    
    # Assigning a type to the variable 'mask' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'mask', result_and__122342)
    
    # Assigning a Subscript to a Name (line 384):
    
    # Assigning a Subscript to a Name (line 384):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 384)
    mask_122343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'mask')
    # Getting the type of 'labels' (line 384)
    labels_122344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'labels')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___122345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 13), labels_122344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_122346 = invoke(stypy.reporting.localization.Localization(__file__, 384, 13), getitem___122345, mask_122343)
    
    # Assigning a type to the variable 'labels' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'labels', subscript_call_result_122346)
    
    # Assigning a Subscript to a Name (line 385):
    
    # Assigning a Subscript to a Name (line 385):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 385)
    mask_122347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'mask')
    # Getting the type of 'input' (line 385)
    input_122348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'input')
    # Obtaining the member '__getitem__' of a type (line 385)
    getitem___122349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 12), input_122348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 385)
    subscript_call_result_122350 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), getitem___122349, mask_122347)
    
    # Assigning a type to the variable 'input' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'input', subscript_call_result_122350)
    
    # Getting the type of 'pass_positions' (line 386)
    pass_positions_122351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 7), 'pass_positions')
    # Testing the type of an if condition (line 386)
    if_condition_122352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 4), pass_positions_122351)
    # Assigning a type to the variable 'if_condition_122352' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'if_condition_122352', if_condition_122352)
    # SSA begins for if statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 387):
    
    # Assigning a Subscript to a Name (line 387):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 387)
    mask_122353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 30), 'mask')
    # Getting the type of 'positions' (line 387)
    positions_122354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'positions')
    # Obtaining the member '__getitem__' of a type (line 387)
    getitem___122355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), positions_122354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 387)
    subscript_call_result_122356 = invoke(stypy.reporting.localization.Localization(__file__, 387, 20), getitem___122355, mask_122353)
    
    # Assigning a type to the variable 'positions' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'positions', subscript_call_result_122356)
    # SSA join for if statement (line 386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to argsort(...): (line 390)
    # Processing the call keyword arguments (line 390)
    kwargs_122359 = {}
    # Getting the type of 'labels' (line 390)
    labels_122357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 18), 'labels', False)
    # Obtaining the member 'argsort' of a type (line 390)
    argsort_122358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 18), labels_122357, 'argsort')
    # Calling argsort(args, kwargs) (line 390)
    argsort_call_result_122360 = invoke(stypy.reporting.localization.Localization(__file__, 390, 18), argsort_122358, *[], **kwargs_122359)
    
    # Assigning a type to the variable 'label_order' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'label_order', argsort_call_result_122360)
    
    # Assigning a Subscript to a Name (line 391):
    
    # Assigning a Subscript to a Name (line 391):
    
    # Obtaining the type of the subscript
    # Getting the type of 'label_order' (line 391)
    label_order_122361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 20), 'label_order')
    # Getting the type of 'labels' (line 391)
    labels_122362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 13), 'labels')
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___122363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 13), labels_122362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_122364 = invoke(stypy.reporting.localization.Localization(__file__, 391, 13), getitem___122363, label_order_122361)
    
    # Assigning a type to the variable 'labels' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'labels', subscript_call_result_122364)
    
    # Assigning a Subscript to a Name (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    # Getting the type of 'label_order' (line 392)
    label_order_122365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 18), 'label_order')
    # Getting the type of 'input' (line 392)
    input_122366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'input')
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___122367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), input_122366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_122368 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), getitem___122367, label_order_122365)
    
    # Assigning a type to the variable 'input' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'input', subscript_call_result_122368)
    
    # Getting the type of 'pass_positions' (line 393)
    pass_positions_122369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 7), 'pass_positions')
    # Testing the type of an if condition (line 393)
    if_condition_122370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), pass_positions_122369)
    # Assigning a type to the variable 'if_condition_122370' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_122370', if_condition_122370)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 394):
    
    # Assigning a Subscript to a Name (line 394):
    
    # Obtaining the type of the subscript
    # Getting the type of 'label_order' (line 394)
    label_order_122371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 30), 'label_order')
    # Getting the type of 'positions' (line 394)
    positions_122372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'positions')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___122373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 20), positions_122372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_122374 = invoke(stypy.reporting.localization.Localization(__file__, 394, 20), getitem___122373, label_order_122371)
    
    # Assigning a type to the variable 'positions' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'positions', subscript_call_result_122374)
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 396):
    
    # Assigning a Call to a Name (line 396):
    
    # Call to argsort(...): (line 396)
    # Processing the call keyword arguments (line 396)
    kwargs_122377 = {}
    # Getting the type of 'index' (line 396)
    index_122375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 18), 'index', False)
    # Obtaining the member 'argsort' of a type (line 396)
    argsort_122376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 18), index_122375, 'argsort')
    # Calling argsort(args, kwargs) (line 396)
    argsort_call_result_122378 = invoke(stypy.reporting.localization.Localization(__file__, 396, 18), argsort_122376, *[], **kwargs_122377)
    
    # Assigning a type to the variable 'index_order' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'index_order', argsort_call_result_122378)
    
    # Assigning a Subscript to a Name (line 397):
    
    # Assigning a Subscript to a Name (line 397):
    
    # Obtaining the type of the subscript
    # Getting the type of 'index_order' (line 397)
    index_order_122379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 25), 'index_order')
    # Getting the type of 'index' (line 397)
    index_122380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'index')
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___122381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), index_122380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_122382 = invoke(stypy.reporting.localization.Localization(__file__, 397, 19), getitem___122381, index_order_122379)
    
    # Assigning a type to the variable 'sorted_index' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'sorted_index', subscript_call_result_122382)

    @norecursion
    def do_map(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'do_map'
        module_type_store = module_type_store.open_function_context('do_map', 399, 4, False)
        
        # Passed parameters checking function
        do_map.stypy_localization = localization
        do_map.stypy_type_of_self = None
        do_map.stypy_type_store = module_type_store
        do_map.stypy_function_name = 'do_map'
        do_map.stypy_param_names_list = ['inputs', 'output']
        do_map.stypy_varargs_param_name = None
        do_map.stypy_kwargs_param_name = None
        do_map.stypy_call_defaults = defaults
        do_map.stypy_call_varargs = varargs
        do_map.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'do_map', ['inputs', 'output'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'do_map', localization, ['inputs', 'output'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'do_map(...)' code ##################

        str_122383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'str', 'labels must be sorted')
        
        # Assigning a Attribute to a Name (line 401):
        
        # Assigning a Attribute to a Name (line 401):
        # Getting the type of 'sorted_index' (line 401)
        sorted_index_122384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'sorted_index')
        # Obtaining the member 'size' of a type (line 401)
        size_122385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 15), sorted_index_122384, 'size')
        # Assigning a type to the variable 'nidx' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'nidx', size_122385)
        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to searchsorted(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'labels' (line 405)
        labels_122388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 32), 'labels', False)
        # Getting the type of 'sorted_index' (line 405)
        sorted_index_122389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 40), 'sorted_index', False)
        # Processing the call keyword arguments (line 405)
        str_122390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 59), 'str', 'left')
        keyword_122391 = str_122390
        kwargs_122392 = {'side': keyword_122391}
        # Getting the type of 'numpy' (line 405)
        numpy_122386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'numpy', False)
        # Obtaining the member 'searchsorted' of a type (line 405)
        searchsorted_122387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), numpy_122386, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 405)
        searchsorted_call_result_122393 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), searchsorted_122387, *[labels_122388, sorted_index_122389], **kwargs_122392)
        
        # Assigning a type to the variable 'lo' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'lo', searchsorted_call_result_122393)
        
        # Assigning a Call to a Name (line 406):
        
        # Assigning a Call to a Name (line 406):
        
        # Call to searchsorted(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'labels' (line 406)
        labels_122396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 32), 'labels', False)
        # Getting the type of 'sorted_index' (line 406)
        sorted_index_122397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 40), 'sorted_index', False)
        # Processing the call keyword arguments (line 406)
        str_122398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 59), 'str', 'right')
        keyword_122399 = str_122398
        kwargs_122400 = {'side': keyword_122399}
        # Getting the type of 'numpy' (line 406)
        numpy_122394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 13), 'numpy', False)
        # Obtaining the member 'searchsorted' of a type (line 406)
        searchsorted_122395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 13), numpy_122394, 'searchsorted')
        # Calling searchsorted(args, kwargs) (line 406)
        searchsorted_call_result_122401 = invoke(stypy.reporting.localization.Localization(__file__, 406, 13), searchsorted_122395, *[labels_122396, sorted_index_122397], **kwargs_122400)
        
        # Assigning a type to the variable 'hi' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'hi', searchsorted_call_result_122401)
        
        
        # Call to zip(...): (line 408)
        # Processing the call arguments (line 408)
        
        # Call to range(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'nidx' (line 408)
        nidx_122404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 33), 'nidx', False)
        # Processing the call keyword arguments (line 408)
        kwargs_122405 = {}
        # Getting the type of 'range' (line 408)
        range_122403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'range', False)
        # Calling range(args, kwargs) (line 408)
        range_call_result_122406 = invoke(stypy.reporting.localization.Localization(__file__, 408, 27), range_122403, *[nidx_122404], **kwargs_122405)
        
        # Getting the type of 'lo' (line 408)
        lo_122407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 40), 'lo', False)
        # Getting the type of 'hi' (line 408)
        hi_122408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 44), 'hi', False)
        # Processing the call keyword arguments (line 408)
        kwargs_122409 = {}
        # Getting the type of 'zip' (line 408)
        zip_122402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 23), 'zip', False)
        # Calling zip(args, kwargs) (line 408)
        zip_call_result_122410 = invoke(stypy.reporting.localization.Localization(__file__, 408, 23), zip_122402, *[range_call_result_122406, lo_122407, hi_122408], **kwargs_122409)
        
        # Testing the type of a for loop iterable (line 408)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 408, 8), zip_call_result_122410)
        # Getting the type of the for loop variable (line 408)
        for_loop_var_122411 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 408, 8), zip_call_result_122410)
        # Assigning a type to the variable 'i' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 8), for_loop_var_122411))
        # Assigning a type to the variable 'l' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'l', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 8), for_loop_var_122411))
        # Assigning a type to the variable 'h' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'h', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 8), for_loop_var_122411))
        # SSA begins for a for statement (line 408)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'l' (line 409)
        l_122412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 15), 'l')
        # Getting the type of 'h' (line 409)
        h_122413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 20), 'h')
        # Applying the binary operator '==' (line 409)
        result_eq_122414 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 15), '==', l_122412, h_122413)
        
        # Testing the type of an if condition (line 409)
        if_condition_122415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 12), result_eq_122414)
        # Assigning a type to the variable 'if_condition_122415' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'if_condition_122415', if_condition_122415)
        # SSA begins for if statement (line 409)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 409)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 411):
        
        # Assigning a Call to a Subscript (line 411):
        
        # Call to func(...): (line 411)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'inputs' (line 411)
        inputs_122423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 51), 'inputs', False)
        comprehension_122424 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 31), inputs_122423)
        # Assigning a type to the variable 'inp' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'inp', comprehension_122424)
        
        # Obtaining the type of the subscript
        # Getting the type of 'l' (line 411)
        l_122417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 35), 'l', False)
        # Getting the type of 'h' (line 411)
        h_122418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 37), 'h', False)
        slice_122419 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 411, 31), l_122417, h_122418, None)
        # Getting the type of 'inp' (line 411)
        inp_122420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'inp', False)
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___122421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 31), inp_122420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_122422 = invoke(stypy.reporting.localization.Localization(__file__, 411, 31), getitem___122421, slice_122419)
        
        list_122425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 31), list_122425, subscript_call_result_122422)
        # Processing the call keyword arguments (line 411)
        kwargs_122426 = {}
        # Getting the type of 'func' (line 411)
        func_122416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 24), 'func', False)
        # Calling func(args, kwargs) (line 411)
        func_call_result_122427 = invoke(stypy.reporting.localization.Localization(__file__, 411, 24), func_122416, *[list_122425], **kwargs_122426)
        
        # Getting the type of 'output' (line 411)
        output_122428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'output')
        # Getting the type of 'i' (line 411)
        i_122429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'i')
        # Storing an element on a container (line 411)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), output_122428, (i_122429, func_call_result_122427))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'do_map(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'do_map' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_122430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'do_map'
        return stypy_return_type_122430

    # Assigning a type to the variable 'do_map' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'do_map', do_map)
    
    # Assigning a Call to a Name (line 413):
    
    # Assigning a Call to a Name (line 413):
    
    # Call to empty(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'index' (line 413)
    index_122433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), 'index', False)
    # Obtaining the member 'shape' of a type (line 413)
    shape_122434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 23), index_122433, 'shape')
    # Getting the type of 'out_dtype' (line 413)
    out_dtype_122435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 36), 'out_dtype', False)
    # Processing the call keyword arguments (line 413)
    kwargs_122436 = {}
    # Getting the type of 'numpy' (line 413)
    numpy_122431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), 'numpy', False)
    # Obtaining the member 'empty' of a type (line 413)
    empty_122432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 11), numpy_122431, 'empty')
    # Calling empty(args, kwargs) (line 413)
    empty_call_result_122437 = invoke(stypy.reporting.localization.Localization(__file__, 413, 11), empty_122432, *[shape_122434, out_dtype_122435], **kwargs_122436)
    
    # Assigning a type to the variable 'temp' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'temp', empty_call_result_122437)
    
    # Assigning a Name to a Subscript (line 414):
    
    # Assigning a Name to a Subscript (line 414):
    # Getting the type of 'default' (line 414)
    default_122438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 14), 'default')
    # Getting the type of 'temp' (line 414)
    temp_122439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'temp')
    slice_122440 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 414, 4), None, None, None)
    # Storing an element on a container (line 414)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 4), temp_122439, (slice_122440, default_122438))
    
    
    # Getting the type of 'pass_positions' (line 415)
    pass_positions_122441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'pass_positions')
    # Applying the 'not' unary operator (line 415)
    result_not__122442 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 7), 'not', pass_positions_122441)
    
    # Testing the type of an if condition (line 415)
    if_condition_122443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 4), result_not__122442)
    # Assigning a type to the variable 'if_condition_122443' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'if_condition_122443', if_condition_122443)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to do_map(...): (line 416)
    # Processing the call arguments (line 416)
    
    # Obtaining an instance of the builtin type 'list' (line 416)
    list_122445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 416)
    # Adding element type (line 416)
    # Getting the type of 'input' (line 416)
    input_122446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'input', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 15), list_122445, input_122446)
    
    # Getting the type of 'temp' (line 416)
    temp_122447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'temp', False)
    # Processing the call keyword arguments (line 416)
    kwargs_122448 = {}
    # Getting the type of 'do_map' (line 416)
    do_map_122444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'do_map', False)
    # Calling do_map(args, kwargs) (line 416)
    do_map_call_result_122449 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), do_map_122444, *[list_122445, temp_122447], **kwargs_122448)
    
    # SSA branch for the else part of an if statement (line 415)
    module_type_store.open_ssa_branch('else')
    
    # Call to do_map(...): (line 418)
    # Processing the call arguments (line 418)
    
    # Obtaining an instance of the builtin type 'list' (line 418)
    list_122451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 418)
    # Adding element type (line 418)
    # Getting the type of 'input' (line 418)
    input_122452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 16), 'input', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 15), list_122451, input_122452)
    # Adding element type (line 418)
    # Getting the type of 'positions' (line 418)
    positions_122453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'positions', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 15), list_122451, positions_122453)
    
    # Getting the type of 'temp' (line 418)
    temp_122454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 35), 'temp', False)
    # Processing the call keyword arguments (line 418)
    kwargs_122455 = {}
    # Getting the type of 'do_map' (line 418)
    do_map_122450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'do_map', False)
    # Calling do_map(args, kwargs) (line 418)
    do_map_call_result_122456 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), do_map_122450, *[list_122451, temp_122454], **kwargs_122455)
    
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to zeros(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'index' (line 420)
    index_122459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 25), 'index', False)
    # Obtaining the member 'shape' of a type (line 420)
    shape_122460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 25), index_122459, 'shape')
    # Getting the type of 'out_dtype' (line 420)
    out_dtype_122461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 38), 'out_dtype', False)
    # Processing the call keyword arguments (line 420)
    kwargs_122462 = {}
    # Getting the type of 'numpy' (line 420)
    numpy_122457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 420)
    zeros_122458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 13), numpy_122457, 'zeros')
    # Calling zeros(args, kwargs) (line 420)
    zeros_call_result_122463 = invoke(stypy.reporting.localization.Localization(__file__, 420, 13), zeros_122458, *[shape_122460, out_dtype_122461], **kwargs_122462)
    
    # Assigning a type to the variable 'output' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'output', zeros_call_result_122463)
    
    # Assigning a Name to a Subscript (line 421):
    
    # Assigning a Name to a Subscript (line 421):
    # Getting the type of 'temp' (line 421)
    temp_122464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 26), 'temp')
    # Getting the type of 'output' (line 421)
    output_122465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'output')
    # Getting the type of 'index_order' (line 421)
    index_order_122466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'index_order')
    # Storing an element on a container (line 421)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 4), output_122465, (index_order_122466, temp_122464))
    
    # Getting the type of 'as_scalar' (line 422)
    as_scalar_122467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'as_scalar')
    # Testing the type of an if condition (line 422)
    if_condition_122468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), as_scalar_122467)
    # Assigning a type to the variable 'if_condition_122468' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_122468', if_condition_122468)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 423):
    
    # Assigning a Subscript to a Name (line 423):
    
    # Obtaining the type of the subscript
    int_122469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 24), 'int')
    # Getting the type of 'output' (line 423)
    output_122470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 17), 'output')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___122471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 17), output_122470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_122472 = invoke(stypy.reporting.localization.Localization(__file__, 423, 17), getitem___122471, int_122469)
    
    # Assigning a type to the variable 'output' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'output', subscript_call_result_122472)
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 425)
    output_122473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type', output_122473)
    
    # ################# End of 'labeled_comprehension(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'labeled_comprehension' in the type store
    # Getting the type of 'stypy_return_type' (line 275)
    stypy_return_type_122474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122474)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'labeled_comprehension'
    return stypy_return_type_122474

# Assigning a type to the variable 'labeled_comprehension' (line 275)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'labeled_comprehension', labeled_comprehension)

@norecursion
def _safely_castable_to_int(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_safely_castable_to_int'
    module_type_store = module_type_store.open_function_context('_safely_castable_to_int', 428, 0, False)
    
    # Passed parameters checking function
    _safely_castable_to_int.stypy_localization = localization
    _safely_castable_to_int.stypy_type_of_self = None
    _safely_castable_to_int.stypy_type_store = module_type_store
    _safely_castable_to_int.stypy_function_name = '_safely_castable_to_int'
    _safely_castable_to_int.stypy_param_names_list = ['dt']
    _safely_castable_to_int.stypy_varargs_param_name = None
    _safely_castable_to_int.stypy_kwargs_param_name = None
    _safely_castable_to_int.stypy_call_defaults = defaults
    _safely_castable_to_int.stypy_call_varargs = varargs
    _safely_castable_to_int.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_safely_castable_to_int', ['dt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_safely_castable_to_int', localization, ['dt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_safely_castable_to_int(...)' code ##################

    str_122475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 4), 'str', 'Test whether the numpy data type `dt` can be safely cast to an int.')
    
    # Assigning a Attribute to a Name (line 430):
    
    # Assigning a Attribute to a Name (line 430):
    
    # Call to dtype(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'int' (line 430)
    int_122478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 24), 'int', False)
    # Processing the call keyword arguments (line 430)
    kwargs_122479 = {}
    # Getting the type of 'np' (line 430)
    np_122476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'np', False)
    # Obtaining the member 'dtype' of a type (line 430)
    dtype_122477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 15), np_122476, 'dtype')
    # Calling dtype(args, kwargs) (line 430)
    dtype_call_result_122480 = invoke(stypy.reporting.localization.Localization(__file__, 430, 15), dtype_122477, *[int_122478], **kwargs_122479)
    
    # Obtaining the member 'itemsize' of a type (line 430)
    itemsize_122481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 15), dtype_call_result_122480, 'itemsize')
    # Assigning a type to the variable 'int_size' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'int_size', itemsize_122481)
    
    # Assigning a BoolOp to a Name (line 431):
    
    # Assigning a BoolOp to a Name (line 431):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'dt' (line 431)
    dt_122484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 27), 'dt', False)
    # Getting the type of 'np' (line 431)
    np_122485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'np', False)
    # Obtaining the member 'signedinteger' of a type (line 431)
    signedinteger_122486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 31), np_122485, 'signedinteger')
    # Processing the call keyword arguments (line 431)
    kwargs_122487 = {}
    # Getting the type of 'np' (line 431)
    np_122482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 13), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 431)
    issubdtype_122483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 13), np_122482, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 431)
    issubdtype_call_result_122488 = invoke(stypy.reporting.localization.Localization(__file__, 431, 13), issubdtype_122483, *[dt_122484, signedinteger_122486], **kwargs_122487)
    
    
    # Getting the type of 'dt' (line 431)
    dt_122489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 53), 'dt')
    # Obtaining the member 'itemsize' of a type (line 431)
    itemsize_122490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 53), dt_122489, 'itemsize')
    # Getting the type of 'int_size' (line 431)
    int_size_122491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 68), 'int_size')
    # Applying the binary operator '<=' (line 431)
    result_le_122492 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 53), '<=', itemsize_122490, int_size_122491)
    
    # Applying the binary operator 'and' (line 431)
    result_and_keyword_122493 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 13), 'and', issubdtype_call_result_122488, result_le_122492)
    
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'dt' (line 432)
    dt_122496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 27), 'dt', False)
    # Getting the type of 'np' (line 432)
    np_122497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'np', False)
    # Obtaining the member 'unsignedinteger' of a type (line 432)
    unsignedinteger_122498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 31), np_122497, 'unsignedinteger')
    # Processing the call keyword arguments (line 432)
    kwargs_122499 = {}
    # Getting the type of 'np' (line 432)
    np_122494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 13), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 432)
    issubdtype_122495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 13), np_122494, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 432)
    issubdtype_call_result_122500 = invoke(stypy.reporting.localization.Localization(__file__, 432, 13), issubdtype_122495, *[dt_122496, unsignedinteger_122498], **kwargs_122499)
    
    
    # Getting the type of 'dt' (line 432)
    dt_122501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 55), 'dt')
    # Obtaining the member 'itemsize' of a type (line 432)
    itemsize_122502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 55), dt_122501, 'itemsize')
    # Getting the type of 'int_size' (line 432)
    int_size_122503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 69), 'int_size')
    # Applying the binary operator '<' (line 432)
    result_lt_122504 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 55), '<', itemsize_122502, int_size_122503)
    
    # Applying the binary operator 'and' (line 432)
    result_and_keyword_122505 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 13), 'and', issubdtype_call_result_122500, result_lt_122504)
    
    # Applying the binary operator 'or' (line 431)
    result_or_keyword_122506 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 12), 'or', result_and_keyword_122493, result_and_keyword_122505)
    
    # Assigning a type to the variable 'safe' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'safe', result_or_keyword_122506)
    # Getting the type of 'safe' (line 433)
    safe_122507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'safe')
    # Assigning a type to the variable 'stypy_return_type' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type', safe_122507)
    
    # ################# End of '_safely_castable_to_int(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_safely_castable_to_int' in the type store
    # Getting the type of 'stypy_return_type' (line 428)
    stypy_return_type_122508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_safely_castable_to_int'
    return stypy_return_type_122508

# Assigning a type to the variable '_safely_castable_to_int' (line 428)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 0), '_safely_castable_to_int', _safely_castable_to_int)

@norecursion
def _stats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 436)
    None_122509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 'None')
    # Getting the type of 'None' (line 436)
    None_122510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 37), 'None')
    # Getting the type of 'False' (line 436)
    False_122511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'False')
    defaults = [None_122509, None_122510, False_122511]
    # Create a new context for function '_stats'
    module_type_store = module_type_store.open_function_context('_stats', 436, 0, False)
    
    # Passed parameters checking function
    _stats.stypy_localization = localization
    _stats.stypy_type_of_self = None
    _stats.stypy_type_store = module_type_store
    _stats.stypy_function_name = '_stats'
    _stats.stypy_param_names_list = ['input', 'labels', 'index', 'centered']
    _stats.stypy_varargs_param_name = None
    _stats.stypy_kwargs_param_name = None
    _stats.stypy_call_defaults = defaults
    _stats.stypy_call_varargs = varargs
    _stats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stats', ['input', 'labels', 'index', 'centered'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_stats', localization, ['input', 'labels', 'index', 'centered'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_stats(...)' code ##################

    str_122512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, (-1)), 'str', 'Count, sum, and optionally compute (sum - centre)^2 of input by label\n\n    Parameters\n    ----------\n    input : array_like, n-dimensional\n        The input data to be analyzed.\n    labels : array_like (n-dimensional), optional\n        The labels of the data in `input`.  This array must be broadcast\n        compatible with `input`; typically it is the same shape as `input`.\n        If `labels` is None, all nonzero values in `input` are treated as\n        the single labeled group.\n    index : label or sequence of labels, optional\n        These are the labels of the groups for which the stats are computed.\n        If `index` is None, the stats are computed for the single group where\n        `labels` is greater than 0.\n    centered : bool, optional\n        If True, the centered sum of squares for each labeled group is\n        also returned.  Default is False.\n\n    Returns\n    -------\n    counts : int or ndarray of ints\n        The number of elements in each labeled group.\n    sums : scalar or ndarray of scalars\n        The sums of the values in each labeled group.\n    sums_c : scalar or ndarray of scalars, optional\n        The sums of mean-centered squares of the values in each labeled group.\n        This is only returned if `centered` is True.\n\n    ')

    @norecursion
    def single_group(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'single_group'
        module_type_store = module_type_store.open_function_context('single_group', 467, 4, False)
        
        # Passed parameters checking function
        single_group.stypy_localization = localization
        single_group.stypy_type_of_self = None
        single_group.stypy_type_store = module_type_store
        single_group.stypy_function_name = 'single_group'
        single_group.stypy_param_names_list = ['vals']
        single_group.stypy_varargs_param_name = None
        single_group.stypy_kwargs_param_name = None
        single_group.stypy_call_defaults = defaults
        single_group.stypy_call_varargs = varargs
        single_group.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'single_group', ['vals'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'single_group', localization, ['vals'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'single_group(...)' code ##################

        
        # Getting the type of 'centered' (line 468)
        centered_122513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'centered')
        # Testing the type of an if condition (line 468)
        if_condition_122514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), centered_122513)
        # Assigning a type to the variable 'if_condition_122514' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_122514', if_condition_122514)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 469):
        
        # Assigning a BinOp to a Name (line 469):
        # Getting the type of 'vals' (line 469)
        vals_122515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 21), 'vals')
        
        # Call to mean(...): (line 469)
        # Processing the call keyword arguments (line 469)
        kwargs_122518 = {}
        # Getting the type of 'vals' (line 469)
        vals_122516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 28), 'vals', False)
        # Obtaining the member 'mean' of a type (line 469)
        mean_122517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 28), vals_122516, 'mean')
        # Calling mean(args, kwargs) (line 469)
        mean_call_result_122519 = invoke(stypy.reporting.localization.Localization(__file__, 469, 28), mean_122517, *[], **kwargs_122518)
        
        # Applying the binary operator '-' (line 469)
        result_sub_122520 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 21), '-', vals_122515, mean_call_result_122519)
        
        # Assigning a type to the variable 'vals_c' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'vals_c', result_sub_122520)
        
        # Obtaining an instance of the builtin type 'tuple' (line 470)
        tuple_122521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 470)
        # Adding element type (line 470)
        # Getting the type of 'vals' (line 470)
        vals_122522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 19), 'vals')
        # Obtaining the member 'size' of a type (line 470)
        size_122523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 19), vals_122522, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 19), tuple_122521, size_122523)
        # Adding element type (line 470)
        
        # Call to sum(...): (line 470)
        # Processing the call keyword arguments (line 470)
        kwargs_122526 = {}
        # Getting the type of 'vals' (line 470)
        vals_122524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 30), 'vals', False)
        # Obtaining the member 'sum' of a type (line 470)
        sum_122525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 30), vals_122524, 'sum')
        # Calling sum(args, kwargs) (line 470)
        sum_call_result_122527 = invoke(stypy.reporting.localization.Localization(__file__, 470, 30), sum_122525, *[], **kwargs_122526)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 19), tuple_122521, sum_call_result_122527)
        # Adding element type (line 470)
        
        # Call to sum(...): (line 470)
        # Processing the call keyword arguments (line 470)
        kwargs_122535 = {}
        # Getting the type of 'vals_c' (line 470)
        vals_c_122528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 43), 'vals_c', False)
        
        # Call to conjugate(...): (line 470)
        # Processing the call keyword arguments (line 470)
        kwargs_122531 = {}
        # Getting the type of 'vals_c' (line 470)
        vals_c_122529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 52), 'vals_c', False)
        # Obtaining the member 'conjugate' of a type (line 470)
        conjugate_122530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 52), vals_c_122529, 'conjugate')
        # Calling conjugate(args, kwargs) (line 470)
        conjugate_call_result_122532 = invoke(stypy.reporting.localization.Localization(__file__, 470, 52), conjugate_122530, *[], **kwargs_122531)
        
        # Applying the binary operator '*' (line 470)
        result_mul_122533 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 43), '*', vals_c_122528, conjugate_call_result_122532)
        
        # Obtaining the member 'sum' of a type (line 470)
        sum_122534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 43), result_mul_122533, 'sum')
        # Calling sum(args, kwargs) (line 470)
        sum_call_result_122536 = invoke(stypy.reporting.localization.Localization(__file__, 470, 43), sum_122534, *[], **kwargs_122535)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 470, 19), tuple_122521, sum_call_result_122536)
        
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'stypy_return_type', tuple_122521)
        # SSA branch for the else part of an if statement (line 468)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 472)
        tuple_122537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 472)
        # Adding element type (line 472)
        # Getting the type of 'vals' (line 472)
        vals_122538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'vals')
        # Obtaining the member 'size' of a type (line 472)
        size_122539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 19), vals_122538, 'size')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 19), tuple_122537, size_122539)
        # Adding element type (line 472)
        
        # Call to sum(...): (line 472)
        # Processing the call keyword arguments (line 472)
        kwargs_122542 = {}
        # Getting the type of 'vals' (line 472)
        vals_122540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 30), 'vals', False)
        # Obtaining the member 'sum' of a type (line 472)
        sum_122541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 30), vals_122540, 'sum')
        # Calling sum(args, kwargs) (line 472)
        sum_call_result_122543 = invoke(stypy.reporting.localization.Localization(__file__, 472, 30), sum_122541, *[], **kwargs_122542)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 472, 19), tuple_122537, sum_call_result_122543)
        
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'stypy_return_type', tuple_122537)
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'single_group(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'single_group' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_122544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'single_group'
        return stypy_return_type_122544

    # Assigning a type to the variable 'single_group' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'single_group', single_group)
    
    # Type idiom detected: calculating its left and rigth part (line 474)
    # Getting the type of 'labels' (line 474)
    labels_122545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 7), 'labels')
    # Getting the type of 'None' (line 474)
    None_122546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 17), 'None')
    
    (may_be_122547, more_types_in_union_122548) = may_be_none(labels_122545, None_122546)

    if may_be_122547:

        if more_types_in_union_122548:
            # Runtime conditional SSA (line 474)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to single_group(...): (line 475)
        # Processing the call arguments (line 475)
        # Getting the type of 'input' (line 475)
        input_122550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 28), 'input', False)
        # Processing the call keyword arguments (line 475)
        kwargs_122551 = {}
        # Getting the type of 'single_group' (line 475)
        single_group_122549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'single_group', False)
        # Calling single_group(args, kwargs) (line 475)
        single_group_call_result_122552 = invoke(stypy.reporting.localization.Localization(__file__, 475, 15), single_group_122549, *[input_122550], **kwargs_122551)
        
        # Assigning a type to the variable 'stypy_return_type' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'stypy_return_type', single_group_call_result_122552)

        if more_types_in_union_122548:
            # SSA join for if statement (line 474)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 478):
    
    # Assigning a Subscript to a Name (line 478):
    
    # Obtaining the type of the subscript
    int_122553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 4), 'int')
    
    # Call to broadcast_arrays(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'input' (line 478)
    input_122556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 43), 'input', False)
    # Getting the type of 'labels' (line 478)
    labels_122557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 50), 'labels', False)
    # Processing the call keyword arguments (line 478)
    kwargs_122558 = {}
    # Getting the type of 'numpy' (line 478)
    numpy_122554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 478)
    broadcast_arrays_122555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), numpy_122554, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 478)
    broadcast_arrays_call_result_122559 = invoke(stypy.reporting.localization.Localization(__file__, 478, 20), broadcast_arrays_122555, *[input_122556, labels_122557], **kwargs_122558)
    
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___122560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 4), broadcast_arrays_call_result_122559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_122561 = invoke(stypy.reporting.localization.Localization(__file__, 478, 4), getitem___122560, int_122553)
    
    # Assigning a type to the variable 'tuple_var_assignment_121904' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'tuple_var_assignment_121904', subscript_call_result_122561)
    
    # Assigning a Subscript to a Name (line 478):
    
    # Obtaining the type of the subscript
    int_122562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 4), 'int')
    
    # Call to broadcast_arrays(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'input' (line 478)
    input_122565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 43), 'input', False)
    # Getting the type of 'labels' (line 478)
    labels_122566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 50), 'labels', False)
    # Processing the call keyword arguments (line 478)
    kwargs_122567 = {}
    # Getting the type of 'numpy' (line 478)
    numpy_122563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 478)
    broadcast_arrays_122564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), numpy_122563, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 478)
    broadcast_arrays_call_result_122568 = invoke(stypy.reporting.localization.Localization(__file__, 478, 20), broadcast_arrays_122564, *[input_122565, labels_122566], **kwargs_122567)
    
    # Obtaining the member '__getitem__' of a type (line 478)
    getitem___122569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 4), broadcast_arrays_call_result_122568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 478)
    subscript_call_result_122570 = invoke(stypy.reporting.localization.Localization(__file__, 478, 4), getitem___122569, int_122562)
    
    # Assigning a type to the variable 'tuple_var_assignment_121905' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'tuple_var_assignment_121905', subscript_call_result_122570)
    
    # Assigning a Name to a Name (line 478):
    # Getting the type of 'tuple_var_assignment_121904' (line 478)
    tuple_var_assignment_121904_122571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'tuple_var_assignment_121904')
    # Assigning a type to the variable 'input' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'input', tuple_var_assignment_121904_122571)
    
    # Assigning a Name to a Name (line 478):
    # Getting the type of 'tuple_var_assignment_121905' (line 478)
    tuple_var_assignment_121905_122572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'tuple_var_assignment_121905')
    # Assigning a type to the variable 'labels' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'labels', tuple_var_assignment_121905_122572)
    
    # Type idiom detected: calculating its left and rigth part (line 480)
    # Getting the type of 'index' (line 480)
    index_122573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 7), 'index')
    # Getting the type of 'None' (line 480)
    None_122574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'None')
    
    (may_be_122575, more_types_in_union_122576) = may_be_none(index_122573, None_122574)

    if may_be_122575:

        if more_types_in_union_122576:
            # Runtime conditional SSA (line 480)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to single_group(...): (line 481)
        # Processing the call arguments (line 481)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'labels' (line 481)
        labels_122578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'labels', False)
        int_122579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 43), 'int')
        # Applying the binary operator '>' (line 481)
        result_gt_122580 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 34), '>', labels_122578, int_122579)
        
        # Getting the type of 'input' (line 481)
        input_122581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 28), 'input', False)
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___122582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 28), input_122581, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_122583 = invoke(stypy.reporting.localization.Localization(__file__, 481, 28), getitem___122582, result_gt_122580)
        
        # Processing the call keyword arguments (line 481)
        kwargs_122584 = {}
        # Getting the type of 'single_group' (line 481)
        single_group_122577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'single_group', False)
        # Calling single_group(args, kwargs) (line 481)
        single_group_call_result_122585 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), single_group_122577, *[subscript_call_result_122583], **kwargs_122584)
        
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', single_group_call_result_122585)

        if more_types_in_union_122576:
            # SSA join for if statement (line 480)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isscalar(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'index' (line 483)
    index_122588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 22), 'index', False)
    # Processing the call keyword arguments (line 483)
    kwargs_122589 = {}
    # Getting the type of 'numpy' (line 483)
    numpy_122586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 483)
    isscalar_122587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 7), numpy_122586, 'isscalar')
    # Calling isscalar(args, kwargs) (line 483)
    isscalar_call_result_122590 = invoke(stypy.reporting.localization.Localization(__file__, 483, 7), isscalar_122587, *[index_122588], **kwargs_122589)
    
    # Testing the type of an if condition (line 483)
    if_condition_122591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 483, 4), isscalar_call_result_122590)
    # Assigning a type to the variable 'if_condition_122591' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'if_condition_122591', if_condition_122591)
    # SSA begins for if statement (line 483)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to single_group(...): (line 484)
    # Processing the call arguments (line 484)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'labels' (line 484)
    labels_122593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 34), 'labels', False)
    # Getting the type of 'index' (line 484)
    index_122594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 44), 'index', False)
    # Applying the binary operator '==' (line 484)
    result_eq_122595 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 34), '==', labels_122593, index_122594)
    
    # Getting the type of 'input' (line 484)
    input_122596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 28), 'input', False)
    # Obtaining the member '__getitem__' of a type (line 484)
    getitem___122597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 28), input_122596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 484)
    subscript_call_result_122598 = invoke(stypy.reporting.localization.Localization(__file__, 484, 28), getitem___122597, result_eq_122595)
    
    # Processing the call keyword arguments (line 484)
    kwargs_122599 = {}
    # Getting the type of 'single_group' (line 484)
    single_group_122592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 15), 'single_group', False)
    # Calling single_group(args, kwargs) (line 484)
    single_group_call_result_122600 = invoke(stypy.reporting.localization.Localization(__file__, 484, 15), single_group_122592, *[subscript_call_result_122598], **kwargs_122599)
    
    # Assigning a type to the variable 'stypy_return_type' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'stypy_return_type', single_group_call_result_122600)
    # SSA join for if statement (line 483)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def _sum_centered(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sum_centered'
        module_type_store = module_type_store.open_function_context('_sum_centered', 486, 4, False)
        
        # Passed parameters checking function
        _sum_centered.stypy_localization = localization
        _sum_centered.stypy_type_of_self = None
        _sum_centered.stypy_type_store = module_type_store
        _sum_centered.stypy_function_name = '_sum_centered'
        _sum_centered.stypy_param_names_list = ['labels']
        _sum_centered.stypy_varargs_param_name = None
        _sum_centered.stypy_kwargs_param_name = None
        _sum_centered.stypy_call_defaults = defaults
        _sum_centered.stypy_call_varargs = varargs
        _sum_centered.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_sum_centered', ['labels'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sum_centered', localization, ['labels'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sum_centered(...)' code ##################

        
        # Assigning a BinOp to a Name (line 490):
        
        # Assigning a BinOp to a Name (line 490):
        # Getting the type of 'sums' (line 490)
        sums_122601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 16), 'sums')
        # Getting the type of 'counts' (line 490)
        counts_122602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'counts')
        # Applying the binary operator 'div' (line 490)
        result_div_122603 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 16), 'div', sums_122601, counts_122602)
        
        # Assigning a type to the variable 'means' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'means', result_div_122603)
        
        # Assigning a BinOp to a Name (line 491):
        
        # Assigning a BinOp to a Name (line 491):
        # Getting the type of 'input' (line 491)
        input_122604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 25), 'input')
        
        # Obtaining the type of the subscript
        # Getting the type of 'labels' (line 491)
        labels_122605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 39), 'labels')
        # Getting the type of 'means' (line 491)
        means_122606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 33), 'means')
        # Obtaining the member '__getitem__' of a type (line 491)
        getitem___122607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 33), means_122606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 491)
        subscript_call_result_122608 = invoke(stypy.reporting.localization.Localization(__file__, 491, 33), getitem___122607, labels_122605)
        
        # Applying the binary operator '-' (line 491)
        result_sub_122609 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 25), '-', input_122604, subscript_call_result_122608)
        
        # Assigning a type to the variable 'centered_input' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'centered_input', result_sub_122609)
        
        # Assigning a Call to a Name (line 493):
        
        # Assigning a Call to a Name (line 493):
        
        # Call to bincount(...): (line 493)
        # Processing the call arguments (line 493)
        
        # Call to ravel(...): (line 493)
        # Processing the call keyword arguments (line 493)
        kwargs_122614 = {}
        # Getting the type of 'labels' (line 493)
        labels_122612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 28), 'labels', False)
        # Obtaining the member 'ravel' of a type (line 493)
        ravel_122613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 28), labels_122612, 'ravel')
        # Calling ravel(args, kwargs) (line 493)
        ravel_call_result_122615 = invoke(stypy.reporting.localization.Localization(__file__, 493, 28), ravel_122613, *[], **kwargs_122614)
        
        # Processing the call keyword arguments (line 493)
        
        # Call to ravel(...): (line 494)
        # Processing the call keyword arguments (line 494)
        kwargs_122623 = {}
        # Getting the type of 'centered_input' (line 494)
        centered_input_122616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 39), 'centered_input', False)
        
        # Call to conjugate(...): (line 495)
        # Processing the call keyword arguments (line 495)
        kwargs_122619 = {}
        # Getting the type of 'centered_input' (line 495)
        centered_input_122617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 39), 'centered_input', False)
        # Obtaining the member 'conjugate' of a type (line 495)
        conjugate_122618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 39), centered_input_122617, 'conjugate')
        # Calling conjugate(args, kwargs) (line 495)
        conjugate_call_result_122620 = invoke(stypy.reporting.localization.Localization(__file__, 495, 39), conjugate_122618, *[], **kwargs_122619)
        
        # Applying the binary operator '*' (line 494)
        result_mul_122621 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 39), '*', centered_input_122616, conjugate_call_result_122620)
        
        # Obtaining the member 'ravel' of a type (line 494)
        ravel_122622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 39), result_mul_122621, 'ravel')
        # Calling ravel(args, kwargs) (line 494)
        ravel_call_result_122624 = invoke(stypy.reporting.localization.Localization(__file__, 494, 39), ravel_122622, *[], **kwargs_122623)
        
        keyword_122625 = ravel_call_result_122624
        kwargs_122626 = {'weights': keyword_122625}
        # Getting the type of 'numpy' (line 493)
        numpy_122610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 13), 'numpy', False)
        # Obtaining the member 'bincount' of a type (line 493)
        bincount_122611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 13), numpy_122610, 'bincount')
        # Calling bincount(args, kwargs) (line 493)
        bincount_call_result_122627 = invoke(stypy.reporting.localization.Localization(__file__, 493, 13), bincount_122611, *[ravel_call_result_122615], **kwargs_122626)
        
        # Assigning a type to the variable 'bc' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'bc', bincount_call_result_122627)
        # Getting the type of 'bc' (line 496)
        bc_122628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 15), 'bc')
        # Assigning a type to the variable 'stypy_return_type' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'stypy_return_type', bc_122628)
        
        # ################# End of '_sum_centered(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sum_centered' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_122629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_122629)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sum_centered'
        return stypy_return_type_122629

    # Assigning a type to the variable '_sum_centered' (line 486)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), '_sum_centered', _sum_centered)
    
    
    # Evaluating a boolean operation
    
    
    # Call to _safely_castable_to_int(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'labels' (line 501)
    labels_122631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 36), 'labels', False)
    # Obtaining the member 'dtype' of a type (line 501)
    dtype_122632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 36), labels_122631, 'dtype')
    # Processing the call keyword arguments (line 501)
    kwargs_122633 = {}
    # Getting the type of '_safely_castable_to_int' (line 501)
    _safely_castable_to_int_122630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), '_safely_castable_to_int', False)
    # Calling _safely_castable_to_int(args, kwargs) (line 501)
    _safely_castable_to_int_call_result_122634 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), _safely_castable_to_int_122630, *[dtype_122632], **kwargs_122633)
    
    # Applying the 'not' unary operator (line 501)
    result_not__122635 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 8), 'not', _safely_castable_to_int_call_result_122634)
    
    
    
    # Call to min(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_122638 = {}
    # Getting the type of 'labels' (line 502)
    labels_122636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'labels', False)
    # Obtaining the member 'min' of a type (line 502)
    min_122637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 12), labels_122636, 'min')
    # Calling min(args, kwargs) (line 502)
    min_call_result_122639 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), min_122637, *[], **kwargs_122638)
    
    int_122640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 27), 'int')
    # Applying the binary operator '<' (line 502)
    result_lt_122641 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 12), '<', min_call_result_122639, int_122640)
    
    # Applying the binary operator 'or' (line 501)
    result_or_keyword_122642 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 8), 'or', result_not__122635, result_lt_122641)
    
    
    # Call to max(...): (line 502)
    # Processing the call keyword arguments (line 502)
    kwargs_122645 = {}
    # Getting the type of 'labels' (line 502)
    labels_122643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 32), 'labels', False)
    # Obtaining the member 'max' of a type (line 502)
    max_122644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 32), labels_122643, 'max')
    # Calling max(args, kwargs) (line 502)
    max_call_result_122646 = invoke(stypy.reporting.localization.Localization(__file__, 502, 32), max_122644, *[], **kwargs_122645)
    
    # Getting the type of 'labels' (line 502)
    labels_122647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 47), 'labels')
    # Obtaining the member 'size' of a type (line 502)
    size_122648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 47), labels_122647, 'size')
    # Applying the binary operator '>' (line 502)
    result_gt_122649 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 32), '>', max_call_result_122646, size_122648)
    
    # Applying the binary operator 'or' (line 501)
    result_or_keyword_122650 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 8), 'or', result_or_keyword_122642, result_gt_122649)
    
    # Testing the type of an if condition (line 501)
    if_condition_122651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), result_or_keyword_122650)
    # Assigning a type to the variable 'if_condition_122651' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_122651', if_condition_122651)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 506):
    
    # Assigning a Subscript to a Name (line 506):
    
    # Obtaining the type of the subscript
    int_122652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 8), 'int')
    
    # Call to unique(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'labels' (line 506)
    labels_122655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 49), 'labels', False)
    # Processing the call keyword arguments (line 506)
    # Getting the type of 'True' (line 506)
    True_122656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 72), 'True', False)
    keyword_122657 = True_122656
    kwargs_122658 = {'return_inverse': keyword_122657}
    # Getting the type of 'numpy' (line 506)
    numpy_122653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'numpy', False)
    # Obtaining the member 'unique' of a type (line 506)
    unique_122654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 36), numpy_122653, 'unique')
    # Calling unique(args, kwargs) (line 506)
    unique_call_result_122659 = invoke(stypy.reporting.localization.Localization(__file__, 506, 36), unique_122654, *[labels_122655], **kwargs_122658)
    
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___122660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), unique_call_result_122659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_122661 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), getitem___122660, int_122652)
    
    # Assigning a type to the variable 'tuple_var_assignment_121906' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'tuple_var_assignment_121906', subscript_call_result_122661)
    
    # Assigning a Subscript to a Name (line 506):
    
    # Obtaining the type of the subscript
    int_122662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 8), 'int')
    
    # Call to unique(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'labels' (line 506)
    labels_122665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 49), 'labels', False)
    # Processing the call keyword arguments (line 506)
    # Getting the type of 'True' (line 506)
    True_122666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 72), 'True', False)
    keyword_122667 = True_122666
    kwargs_122668 = {'return_inverse': keyword_122667}
    # Getting the type of 'numpy' (line 506)
    numpy_122663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'numpy', False)
    # Obtaining the member 'unique' of a type (line 506)
    unique_122664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 36), numpy_122663, 'unique')
    # Calling unique(args, kwargs) (line 506)
    unique_call_result_122669 = invoke(stypy.reporting.localization.Localization(__file__, 506, 36), unique_122664, *[labels_122665], **kwargs_122668)
    
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___122670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), unique_call_result_122669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_122671 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), getitem___122670, int_122662)
    
    # Assigning a type to the variable 'tuple_var_assignment_121907' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'tuple_var_assignment_121907', subscript_call_result_122671)
    
    # Assigning a Name to a Name (line 506):
    # Getting the type of 'tuple_var_assignment_121906' (line 506)
    tuple_var_assignment_121906_122672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'tuple_var_assignment_121906')
    # Assigning a type to the variable 'unique_labels' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'unique_labels', tuple_var_assignment_121906_122672)
    
    # Assigning a Name to a Name (line 506):
    # Getting the type of 'tuple_var_assignment_121907' (line 506)
    tuple_var_assignment_121907_122673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'tuple_var_assignment_121907')
    # Assigning a type to the variable 'new_labels' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 23), 'new_labels', tuple_var_assignment_121907_122673)
    
    # Assigning a Call to a Name (line 507):
    
    # Assigning a Call to a Name (line 507):
    
    # Call to bincount(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'new_labels' (line 507)
    new_labels_122676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 32), 'new_labels', False)
    # Processing the call keyword arguments (line 507)
    kwargs_122677 = {}
    # Getting the type of 'numpy' (line 507)
    numpy_122674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 17), 'numpy', False)
    # Obtaining the member 'bincount' of a type (line 507)
    bincount_122675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 17), numpy_122674, 'bincount')
    # Calling bincount(args, kwargs) (line 507)
    bincount_call_result_122678 = invoke(stypy.reporting.localization.Localization(__file__, 507, 17), bincount_122675, *[new_labels_122676], **kwargs_122677)
    
    # Assigning a type to the variable 'counts' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'counts', bincount_call_result_122678)
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to bincount(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'new_labels' (line 508)
    new_labels_122681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 30), 'new_labels', False)
    # Processing the call keyword arguments (line 508)
    
    # Call to ravel(...): (line 508)
    # Processing the call keyword arguments (line 508)
    kwargs_122684 = {}
    # Getting the type of 'input' (line 508)
    input_122682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 50), 'input', False)
    # Obtaining the member 'ravel' of a type (line 508)
    ravel_122683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 50), input_122682, 'ravel')
    # Calling ravel(args, kwargs) (line 508)
    ravel_call_result_122685 = invoke(stypy.reporting.localization.Localization(__file__, 508, 50), ravel_122683, *[], **kwargs_122684)
    
    keyword_122686 = ravel_call_result_122685
    kwargs_122687 = {'weights': keyword_122686}
    # Getting the type of 'numpy' (line 508)
    numpy_122679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'numpy', False)
    # Obtaining the member 'bincount' of a type (line 508)
    bincount_122680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), numpy_122679, 'bincount')
    # Calling bincount(args, kwargs) (line 508)
    bincount_call_result_122688 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), bincount_122680, *[new_labels_122681], **kwargs_122687)
    
    # Assigning a type to the variable 'sums' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'sums', bincount_call_result_122688)
    
    # Getting the type of 'centered' (line 509)
    centered_122689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'centered')
    # Testing the type of an if condition (line 509)
    if_condition_122690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 8), centered_122689)
    # Assigning a type to the variable 'if_condition_122690' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'if_condition_122690', if_condition_122690)
    # SSA begins for if statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to _sum_centered(...): (line 513)
    # Processing the call arguments (line 513)
    
    # Call to reshape(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'labels' (line 513)
    labels_122694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 54), 'labels', False)
    # Obtaining the member 'shape' of a type (line 513)
    shape_122695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 54), labels_122694, 'shape')
    # Processing the call keyword arguments (line 513)
    kwargs_122696 = {}
    # Getting the type of 'new_labels' (line 513)
    new_labels_122692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 35), 'new_labels', False)
    # Obtaining the member 'reshape' of a type (line 513)
    reshape_122693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 35), new_labels_122692, 'reshape')
    # Calling reshape(args, kwargs) (line 513)
    reshape_call_result_122697 = invoke(stypy.reporting.localization.Localization(__file__, 513, 35), reshape_122693, *[shape_122695], **kwargs_122696)
    
    # Processing the call keyword arguments (line 513)
    kwargs_122698 = {}
    # Getting the type of '_sum_centered' (line 513)
    _sum_centered_122691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), '_sum_centered', False)
    # Calling _sum_centered(args, kwargs) (line 513)
    _sum_centered_call_result_122699 = invoke(stypy.reporting.localization.Localization(__file__, 513, 21), _sum_centered_122691, *[reshape_call_result_122697], **kwargs_122698)
    
    # Assigning a type to the variable 'sums_c' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'sums_c', _sum_centered_call_result_122699)
    # SSA join for if statement (line 509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 514):
    
    # Assigning a Call to a Name (line 514):
    
    # Call to searchsorted(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'unique_labels' (line 514)
    unique_labels_122702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'unique_labels', False)
    # Getting the type of 'index' (line 514)
    index_122703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 49), 'index', False)
    # Processing the call keyword arguments (line 514)
    kwargs_122704 = {}
    # Getting the type of 'numpy' (line 514)
    numpy_122700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 15), 'numpy', False)
    # Obtaining the member 'searchsorted' of a type (line 514)
    searchsorted_122701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 15), numpy_122700, 'searchsorted')
    # Calling searchsorted(args, kwargs) (line 514)
    searchsorted_call_result_122705 = invoke(stypy.reporting.localization.Localization(__file__, 514, 15), searchsorted_122701, *[unique_labels_122702, index_122703], **kwargs_122704)
    
    # Assigning a type to the variable 'idxs' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'idxs', searchsorted_call_result_122705)
    
    # Assigning a Num to a Subscript (line 516):
    
    # Assigning a Num to a Subscript (line 516):
    int_122706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 43), 'int')
    # Getting the type of 'idxs' (line 516)
    idxs_122707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'idxs')
    
    # Getting the type of 'idxs' (line 516)
    idxs_122708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 13), 'idxs')
    # Getting the type of 'unique_labels' (line 516)
    unique_labels_122709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 21), 'unique_labels')
    # Obtaining the member 'size' of a type (line 516)
    size_122710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 21), unique_labels_122709, 'size')
    # Applying the binary operator '>=' (line 516)
    result_ge_122711 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 13), '>=', idxs_122708, size_122710)
    
    # Storing an element on a container (line 516)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 516, 8), idxs_122707, (result_ge_122711, int_122706))
    
    # Assigning a Compare to a Name (line 517):
    
    # Assigning a Compare to a Name (line 517):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 517)
    idxs_122712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 31), 'idxs')
    # Getting the type of 'unique_labels' (line 517)
    unique_labels_122713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), 'unique_labels')
    # Obtaining the member '__getitem__' of a type (line 517)
    getitem___122714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 17), unique_labels_122713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 517)
    subscript_call_result_122715 = invoke(stypy.reporting.localization.Localization(__file__, 517, 17), getitem___122714, idxs_122712)
    
    # Getting the type of 'index' (line 517)
    index_122716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 40), 'index')
    # Applying the binary operator '==' (line 517)
    result_eq_122717 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 17), '==', subscript_call_result_122715, index_122716)
    
    # Assigning a type to the variable 'found' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'found', result_eq_122717)
    # SSA branch for the else part of an if statement (line 501)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 521):
    
    # Assigning a Call to a Name (line 521):
    
    # Call to bincount(...): (line 521)
    # Processing the call arguments (line 521)
    
    # Call to ravel(...): (line 521)
    # Processing the call keyword arguments (line 521)
    kwargs_122722 = {}
    # Getting the type of 'labels' (line 521)
    labels_122720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 32), 'labels', False)
    # Obtaining the member 'ravel' of a type (line 521)
    ravel_122721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 32), labels_122720, 'ravel')
    # Calling ravel(args, kwargs) (line 521)
    ravel_call_result_122723 = invoke(stypy.reporting.localization.Localization(__file__, 521, 32), ravel_122721, *[], **kwargs_122722)
    
    # Processing the call keyword arguments (line 521)
    kwargs_122724 = {}
    # Getting the type of 'numpy' (line 521)
    numpy_122718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 17), 'numpy', False)
    # Obtaining the member 'bincount' of a type (line 521)
    bincount_122719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 17), numpy_122718, 'bincount')
    # Calling bincount(args, kwargs) (line 521)
    bincount_call_result_122725 = invoke(stypy.reporting.localization.Localization(__file__, 521, 17), bincount_122719, *[ravel_call_result_122723], **kwargs_122724)
    
    # Assigning a type to the variable 'counts' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'counts', bincount_call_result_122725)
    
    # Assigning a Call to a Name (line 522):
    
    # Assigning a Call to a Name (line 522):
    
    # Call to bincount(...): (line 522)
    # Processing the call arguments (line 522)
    
    # Call to ravel(...): (line 522)
    # Processing the call keyword arguments (line 522)
    kwargs_122730 = {}
    # Getting the type of 'labels' (line 522)
    labels_122728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 30), 'labels', False)
    # Obtaining the member 'ravel' of a type (line 522)
    ravel_122729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 30), labels_122728, 'ravel')
    # Calling ravel(args, kwargs) (line 522)
    ravel_call_result_122731 = invoke(stypy.reporting.localization.Localization(__file__, 522, 30), ravel_122729, *[], **kwargs_122730)
    
    # Processing the call keyword arguments (line 522)
    
    # Call to ravel(...): (line 522)
    # Processing the call keyword arguments (line 522)
    kwargs_122734 = {}
    # Getting the type of 'input' (line 522)
    input_122732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 54), 'input', False)
    # Obtaining the member 'ravel' of a type (line 522)
    ravel_122733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 54), input_122732, 'ravel')
    # Calling ravel(args, kwargs) (line 522)
    ravel_call_result_122735 = invoke(stypy.reporting.localization.Localization(__file__, 522, 54), ravel_122733, *[], **kwargs_122734)
    
    keyword_122736 = ravel_call_result_122735
    kwargs_122737 = {'weights': keyword_122736}
    # Getting the type of 'numpy' (line 522)
    numpy_122726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'numpy', False)
    # Obtaining the member 'bincount' of a type (line 522)
    bincount_122727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), numpy_122726, 'bincount')
    # Calling bincount(args, kwargs) (line 522)
    bincount_call_result_122738 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), bincount_122727, *[ravel_call_result_122731], **kwargs_122737)
    
    # Assigning a type to the variable 'sums' (line 522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'sums', bincount_call_result_122738)
    
    # Getting the type of 'centered' (line 523)
    centered_122739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 11), 'centered')
    # Testing the type of an if condition (line 523)
    if_condition_122740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 523, 8), centered_122739)
    # Assigning a type to the variable 'if_condition_122740' (line 523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'if_condition_122740', if_condition_122740)
    # SSA begins for if statement (line 523)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 524):
    
    # Assigning a Call to a Name (line 524):
    
    # Call to _sum_centered(...): (line 524)
    # Processing the call arguments (line 524)
    # Getting the type of 'labels' (line 524)
    labels_122742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 35), 'labels', False)
    # Processing the call keyword arguments (line 524)
    kwargs_122743 = {}
    # Getting the type of '_sum_centered' (line 524)
    _sum_centered_122741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 21), '_sum_centered', False)
    # Calling _sum_centered(args, kwargs) (line 524)
    _sum_centered_call_result_122744 = invoke(stypy.reporting.localization.Localization(__file__, 524, 21), _sum_centered_122741, *[labels_122742], **kwargs_122743)
    
    # Assigning a type to the variable 'sums_c' (line 524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'sums_c', _sum_centered_call_result_122744)
    # SSA join for if statement (line 523)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 526):
    
    # Assigning a Call to a Name (line 526):
    
    # Call to copy(...): (line 526)
    # Processing the call keyword arguments (line 526)
    kwargs_122753 = {}
    
    # Call to asanyarray(...): (line 526)
    # Processing the call arguments (line 526)
    # Getting the type of 'index' (line 526)
    index_122747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 32), 'index', False)
    # Getting the type of 'numpy' (line 526)
    numpy_122748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 39), 'numpy', False)
    # Obtaining the member 'int' of a type (line 526)
    int_122749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 39), numpy_122748, 'int')
    # Processing the call keyword arguments (line 526)
    kwargs_122750 = {}
    # Getting the type of 'numpy' (line 526)
    numpy_122745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 15), 'numpy', False)
    # Obtaining the member 'asanyarray' of a type (line 526)
    asanyarray_122746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), numpy_122745, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 526)
    asanyarray_call_result_122751 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), asanyarray_122746, *[index_122747, int_122749], **kwargs_122750)
    
    # Obtaining the member 'copy' of a type (line 526)
    copy_122752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 15), asanyarray_call_result_122751, 'copy')
    # Calling copy(args, kwargs) (line 526)
    copy_call_result_122754 = invoke(stypy.reporting.localization.Localization(__file__, 526, 15), copy_122752, *[], **kwargs_122753)
    
    # Assigning a type to the variable 'idxs' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'idxs', copy_call_result_122754)
    
    # Assigning a BinOp to a Name (line 527):
    
    # Assigning a BinOp to a Name (line 527):
    
    # Getting the type of 'idxs' (line 527)
    idxs_122755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 17), 'idxs')
    int_122756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 25), 'int')
    # Applying the binary operator '>=' (line 527)
    result_ge_122757 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 17), '>=', idxs_122755, int_122756)
    
    
    # Getting the type of 'idxs' (line 527)
    idxs_122758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 31), 'idxs')
    # Getting the type of 'counts' (line 527)
    counts_122759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 38), 'counts')
    # Obtaining the member 'size' of a type (line 527)
    size_122760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 38), counts_122759, 'size')
    # Applying the binary operator '<' (line 527)
    result_lt_122761 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 31), '<', idxs_122758, size_122760)
    
    # Applying the binary operator '&' (line 527)
    result_and__122762 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 16), '&', result_ge_122757, result_lt_122761)
    
    # Assigning a type to the variable 'found' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'found', result_and__122762)
    
    # Assigning a Num to a Subscript (line 528):
    
    # Assigning a Num to a Subscript (line 528):
    int_122763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 23), 'int')
    # Getting the type of 'idxs' (line 528)
    idxs_122764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'idxs')
    
    # Getting the type of 'found' (line 528)
    found_122765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 14), 'found')
    # Applying the '~' unary operator (line 528)
    result_inv_122766 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 13), '~', found_122765)
    
    # Storing an element on a container (line 528)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 8), idxs_122764, (result_inv_122766, int_122763))
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 530):
    
    # Assigning a Subscript to a Name (line 530):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 530)
    idxs_122767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 20), 'idxs')
    # Getting the type of 'counts' (line 530)
    counts_122768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 13), 'counts')
    # Obtaining the member '__getitem__' of a type (line 530)
    getitem___122769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 13), counts_122768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 530)
    subscript_call_result_122770 = invoke(stypy.reporting.localization.Localization(__file__, 530, 13), getitem___122769, idxs_122767)
    
    # Assigning a type to the variable 'counts' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'counts', subscript_call_result_122770)
    
    # Assigning a Num to a Subscript (line 531):
    
    # Assigning a Num to a Subscript (line 531):
    int_122771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 21), 'int')
    # Getting the type of 'counts' (line 531)
    counts_122772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'counts')
    
    # Getting the type of 'found' (line 531)
    found_122773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'found')
    # Applying the '~' unary operator (line 531)
    result_inv_122774 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 11), '~', found_122773)
    
    # Storing an element on a container (line 531)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 4), counts_122772, (result_inv_122774, int_122771))
    
    # Assigning a Subscript to a Name (line 532):
    
    # Assigning a Subscript to a Name (line 532):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 532)
    idxs_122775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 16), 'idxs')
    # Getting the type of 'sums' (line 532)
    sums_122776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'sums')
    # Obtaining the member '__getitem__' of a type (line 532)
    getitem___122777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), sums_122776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 532)
    subscript_call_result_122778 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), getitem___122777, idxs_122775)
    
    # Assigning a type to the variable 'sums' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'sums', subscript_call_result_122778)
    
    # Assigning a Num to a Subscript (line 533):
    
    # Assigning a Num to a Subscript (line 533):
    int_122779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 19), 'int')
    # Getting the type of 'sums' (line 533)
    sums_122780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'sums')
    
    # Getting the type of 'found' (line 533)
    found_122781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 10), 'found')
    # Applying the '~' unary operator (line 533)
    result_inv_122782 = python_operator(stypy.reporting.localization.Localization(__file__, 533, 9), '~', found_122781)
    
    # Storing an element on a container (line 533)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 4), sums_122780, (result_inv_122782, int_122779))
    
    
    # Getting the type of 'centered' (line 535)
    centered_122783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'centered')
    # Applying the 'not' unary operator (line 535)
    result_not__122784 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 7), 'not', centered_122783)
    
    # Testing the type of an if condition (line 535)
    if_condition_122785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 4), result_not__122784)
    # Assigning a type to the variable 'if_condition_122785' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'if_condition_122785', if_condition_122785)
    # SSA begins for if statement (line 535)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 536)
    tuple_122786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 536)
    # Adding element type (line 536)
    # Getting the type of 'counts' (line 536)
    counts_122787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'counts')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 16), tuple_122786, counts_122787)
    # Adding element type (line 536)
    # Getting the type of 'sums' (line 536)
    sums_122788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'sums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 16), tuple_122786, sums_122788)
    
    # Assigning a type to the variable 'stypy_return_type' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'stypy_return_type', tuple_122786)
    # SSA branch for the else part of an if statement (line 535)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 538):
    
    # Assigning a Subscript to a Name (line 538):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 538)
    idxs_122789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 24), 'idxs')
    # Getting the type of 'sums_c' (line 538)
    sums_c_122790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 17), 'sums_c')
    # Obtaining the member '__getitem__' of a type (line 538)
    getitem___122791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 17), sums_c_122790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 538)
    subscript_call_result_122792 = invoke(stypy.reporting.localization.Localization(__file__, 538, 17), getitem___122791, idxs_122789)
    
    # Assigning a type to the variable 'sums_c' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'sums_c', subscript_call_result_122792)
    
    # Assigning a Num to a Subscript (line 539):
    
    # Assigning a Num to a Subscript (line 539):
    int_122793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 25), 'int')
    # Getting the type of 'sums_c' (line 539)
    sums_c_122794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'sums_c')
    
    # Getting the type of 'found' (line 539)
    found_122795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'found')
    # Applying the '~' unary operator (line 539)
    result_inv_122796 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 15), '~', found_122795)
    
    # Storing an element on a container (line 539)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 539, 8), sums_c_122794, (result_inv_122796, int_122793))
    
    # Obtaining an instance of the builtin type 'tuple' (line 540)
    tuple_122797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 540)
    # Adding element type (line 540)
    # Getting the type of 'counts' (line 540)
    counts_122798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'counts')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 16), tuple_122797, counts_122798)
    # Adding element type (line 540)
    # Getting the type of 'sums' (line 540)
    sums_122799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'sums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 16), tuple_122797, sums_122799)
    # Adding element type (line 540)
    # Getting the type of 'sums_c' (line 540)
    sums_c_122800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 30), 'sums_c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 540, 16), tuple_122797, sums_c_122800)
    
    # Assigning a type to the variable 'stypy_return_type' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'stypy_return_type', tuple_122797)
    # SSA join for if statement (line 535)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_stats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_stats' in the type store
    # Getting the type of 'stypy_return_type' (line 436)
    stypy_return_type_122801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122801)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stats'
    return stypy_return_type_122801

# Assigning a type to the variable '_stats' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), '_stats', _stats)

@norecursion
def sum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 543)
    None_122802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 22), 'None')
    # Getting the type of 'None' (line 543)
    None_122803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 34), 'None')
    defaults = [None_122802, None_122803]
    # Create a new context for function 'sum'
    module_type_store = module_type_store.open_function_context('sum', 543, 0, False)
    
    # Passed parameters checking function
    sum.stypy_localization = localization
    sum.stypy_type_of_self = None
    sum.stypy_type_store = module_type_store
    sum.stypy_function_name = 'sum'
    sum.stypy_param_names_list = ['input', 'labels', 'index']
    sum.stypy_varargs_param_name = None
    sum.stypy_kwargs_param_name = None
    sum.stypy_call_defaults = defaults
    sum.stypy_call_varargs = varargs
    sum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sum', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sum', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sum(...)' code ##################

    str_122804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, (-1)), 'str', "\n    Calculate the sum of the values of the array.\n\n    Parameters\n    ----------\n    input : array_like\n        Values of `input` inside the regions defined by `labels`\n        are summed together.\n    labels : array_like of ints, optional\n        Assign labels to the values of the array. Has to have the same shape as\n        `input`.\n    index : array_like, optional\n        A single label number or a sequence of label numbers of\n        the objects to be measured.\n\n    Returns\n    -------\n    sum : ndarray or scalar\n        An array of the sums of values of `input` inside the regions defined\n        by `labels` with the same shape as `index`. If 'index' is None or scalar,\n        a scalar is returned.\n\n    See also\n    --------\n    mean, median\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> input =  [0,1,2,3]\n    >>> labels = [1,1,2,2]\n    >>> ndimage.sum(input, labels, index=[1,2])\n    [1.0, 5.0]\n    >>> ndimage.sum(input, labels, index=1)\n    1\n    >>> ndimage.sum(input, labels)\n    6\n\n\n    ")
    
    # Assigning a Call to a Tuple (line 584):
    
    # Assigning a Subscript to a Name (line 584):
    
    # Obtaining the type of the subscript
    int_122805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 4), 'int')
    
    # Call to _stats(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'input' (line 584)
    input_122807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 24), 'input', False)
    # Getting the type of 'labels' (line 584)
    labels_122808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'labels', False)
    # Getting the type of 'index' (line 584)
    index_122809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 39), 'index', False)
    # Processing the call keyword arguments (line 584)
    kwargs_122810 = {}
    # Getting the type of '_stats' (line 584)
    _stats_122806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 17), '_stats', False)
    # Calling _stats(args, kwargs) (line 584)
    _stats_call_result_122811 = invoke(stypy.reporting.localization.Localization(__file__, 584, 17), _stats_122806, *[input_122807, labels_122808, index_122809], **kwargs_122810)
    
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___122812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 4), _stats_call_result_122811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_122813 = invoke(stypy.reporting.localization.Localization(__file__, 584, 4), getitem___122812, int_122805)
    
    # Assigning a type to the variable 'tuple_var_assignment_121908' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'tuple_var_assignment_121908', subscript_call_result_122813)
    
    # Assigning a Subscript to a Name (line 584):
    
    # Obtaining the type of the subscript
    int_122814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 4), 'int')
    
    # Call to _stats(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'input' (line 584)
    input_122816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 24), 'input', False)
    # Getting the type of 'labels' (line 584)
    labels_122817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 31), 'labels', False)
    # Getting the type of 'index' (line 584)
    index_122818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 39), 'index', False)
    # Processing the call keyword arguments (line 584)
    kwargs_122819 = {}
    # Getting the type of '_stats' (line 584)
    _stats_122815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 17), '_stats', False)
    # Calling _stats(args, kwargs) (line 584)
    _stats_call_result_122820 = invoke(stypy.reporting.localization.Localization(__file__, 584, 17), _stats_122815, *[input_122816, labels_122817, index_122818], **kwargs_122819)
    
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___122821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 4), _stats_call_result_122820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_122822 = invoke(stypy.reporting.localization.Localization(__file__, 584, 4), getitem___122821, int_122814)
    
    # Assigning a type to the variable 'tuple_var_assignment_121909' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'tuple_var_assignment_121909', subscript_call_result_122822)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'tuple_var_assignment_121908' (line 584)
    tuple_var_assignment_121908_122823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'tuple_var_assignment_121908')
    # Assigning a type to the variable 'count' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'count', tuple_var_assignment_121908_122823)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'tuple_var_assignment_121909' (line 584)
    tuple_var_assignment_121909_122824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'tuple_var_assignment_121909')
    # Assigning a type to the variable 'sum' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'sum', tuple_var_assignment_121909_122824)
    # Getting the type of 'sum' (line 585)
    sum_122825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'sum')
    # Assigning a type to the variable 'stypy_return_type' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'stypy_return_type', sum_122825)
    
    # ################# End of 'sum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sum' in the type store
    # Getting the type of 'stypy_return_type' (line 543)
    stypy_return_type_122826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122826)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sum'
    return stypy_return_type_122826

# Assigning a type to the variable 'sum' (line 543)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'sum', sum)

@norecursion
def mean(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 588)
    None_122827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 23), 'None')
    # Getting the type of 'None' (line 588)
    None_122828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 35), 'None')
    defaults = [None_122827, None_122828]
    # Create a new context for function 'mean'
    module_type_store = module_type_store.open_function_context('mean', 588, 0, False)
    
    # Passed parameters checking function
    mean.stypy_localization = localization
    mean.stypy_type_of_self = None
    mean.stypy_type_store = module_type_store
    mean.stypy_function_name = 'mean'
    mean.stypy_param_names_list = ['input', 'labels', 'index']
    mean.stypy_varargs_param_name = None
    mean.stypy_kwargs_param_name = None
    mean.stypy_call_defaults = defaults
    mean.stypy_call_varargs = varargs
    mean.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mean', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mean', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mean(...)' code ##################

    str_122829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, (-1)), 'str', '\n    Calculate the mean of the values of an array at labels.\n\n    Parameters\n    ----------\n    input : array_like\n        Array on which to compute the mean of elements over distinct\n        regions.\n    labels : array_like, optional\n        Array of labels of same shape, or broadcastable to the same shape as\n        `input`. All elements sharing the same label form one region over\n        which the mean of the elements is computed.\n    index : int or sequence of ints, optional\n        Labels of the objects over which the mean is to be computed.\n        Default is None, in which case the mean for all values where label is\n        greater than 0 is calculated.\n\n    Returns\n    -------\n    out : list\n        Sequence of same length as `index`, with the mean of the different\n        regions labeled by the labels in `index`.\n\n    See also\n    --------\n    ndimage.variance, ndimage.standard_deviation, ndimage.minimum,\n    ndimage.maximum, ndimage.sum\n    ndimage.label\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.arange(25).reshape((5,5))\n    >>> labels = np.zeros_like(a)\n    >>> labels[3:5,3:5] = 1\n    >>> index = np.unique(labels)\n    >>> labels\n    array([[0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 0, 0, 0],\n           [0, 0, 0, 1, 1],\n           [0, 0, 0, 1, 1]])\n    >>> index\n    array([0, 1])\n    >>> ndimage.mean(a, labels=labels, index=index)\n    [10.285714285714286, 21.0]\n\n    ')
    
    # Assigning a Call to a Tuple (line 638):
    
    # Assigning a Subscript to a Name (line 638):
    
    # Obtaining the type of the subscript
    int_122830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 4), 'int')
    
    # Call to _stats(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'input' (line 638)
    input_122832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'input', False)
    # Getting the type of 'labels' (line 638)
    labels_122833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 31), 'labels', False)
    # Getting the type of 'index' (line 638)
    index_122834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 39), 'index', False)
    # Processing the call keyword arguments (line 638)
    kwargs_122835 = {}
    # Getting the type of '_stats' (line 638)
    _stats_122831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 17), '_stats', False)
    # Calling _stats(args, kwargs) (line 638)
    _stats_call_result_122836 = invoke(stypy.reporting.localization.Localization(__file__, 638, 17), _stats_122831, *[input_122832, labels_122833, index_122834], **kwargs_122835)
    
    # Obtaining the member '__getitem__' of a type (line 638)
    getitem___122837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 4), _stats_call_result_122836, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 638)
    subscript_call_result_122838 = invoke(stypy.reporting.localization.Localization(__file__, 638, 4), getitem___122837, int_122830)
    
    # Assigning a type to the variable 'tuple_var_assignment_121910' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_var_assignment_121910', subscript_call_result_122838)
    
    # Assigning a Subscript to a Name (line 638):
    
    # Obtaining the type of the subscript
    int_122839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 4), 'int')
    
    # Call to _stats(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'input' (line 638)
    input_122841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'input', False)
    # Getting the type of 'labels' (line 638)
    labels_122842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 31), 'labels', False)
    # Getting the type of 'index' (line 638)
    index_122843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 39), 'index', False)
    # Processing the call keyword arguments (line 638)
    kwargs_122844 = {}
    # Getting the type of '_stats' (line 638)
    _stats_122840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 17), '_stats', False)
    # Calling _stats(args, kwargs) (line 638)
    _stats_call_result_122845 = invoke(stypy.reporting.localization.Localization(__file__, 638, 17), _stats_122840, *[input_122841, labels_122842, index_122843], **kwargs_122844)
    
    # Obtaining the member '__getitem__' of a type (line 638)
    getitem___122846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 4), _stats_call_result_122845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 638)
    subscript_call_result_122847 = invoke(stypy.reporting.localization.Localization(__file__, 638, 4), getitem___122846, int_122839)
    
    # Assigning a type to the variable 'tuple_var_assignment_121911' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_var_assignment_121911', subscript_call_result_122847)
    
    # Assigning a Name to a Name (line 638):
    # Getting the type of 'tuple_var_assignment_121910' (line 638)
    tuple_var_assignment_121910_122848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_var_assignment_121910')
    # Assigning a type to the variable 'count' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'count', tuple_var_assignment_121910_122848)
    
    # Assigning a Name to a Name (line 638):
    # Getting the type of 'tuple_var_assignment_121911' (line 638)
    tuple_var_assignment_121911_122849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'tuple_var_assignment_121911')
    # Assigning a type to the variable 'sum' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'sum', tuple_var_assignment_121911_122849)
    # Getting the type of 'sum' (line 639)
    sum_122850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 11), 'sum')
    
    # Call to astype(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'numpy' (line 639)
    numpy_122857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 48), 'numpy', False)
    # Obtaining the member 'float' of a type (line 639)
    float_122858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 48), numpy_122857, 'float')
    # Processing the call keyword arguments (line 639)
    kwargs_122859 = {}
    
    # Call to asanyarray(...): (line 639)
    # Processing the call arguments (line 639)
    # Getting the type of 'count' (line 639)
    count_122853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 34), 'count', False)
    # Processing the call keyword arguments (line 639)
    kwargs_122854 = {}
    # Getting the type of 'numpy' (line 639)
    numpy_122851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 17), 'numpy', False)
    # Obtaining the member 'asanyarray' of a type (line 639)
    asanyarray_122852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 17), numpy_122851, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 639)
    asanyarray_call_result_122855 = invoke(stypy.reporting.localization.Localization(__file__, 639, 17), asanyarray_122852, *[count_122853], **kwargs_122854)
    
    # Obtaining the member 'astype' of a type (line 639)
    astype_122856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 17), asanyarray_call_result_122855, 'astype')
    # Calling astype(args, kwargs) (line 639)
    astype_call_result_122860 = invoke(stypy.reporting.localization.Localization(__file__, 639, 17), astype_122856, *[float_122858], **kwargs_122859)
    
    # Applying the binary operator 'div' (line 639)
    result_div_122861 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 11), 'div', sum_122850, astype_call_result_122860)
    
    # Assigning a type to the variable 'stypy_return_type' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'stypy_return_type', result_div_122861)
    
    # ################# End of 'mean(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mean' in the type store
    # Getting the type of 'stypy_return_type' (line 588)
    stypy_return_type_122862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122862)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mean'
    return stypy_return_type_122862

# Assigning a type to the variable 'mean' (line 588)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 0), 'mean', mean)

@norecursion
def variance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 642)
    None_122863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 27), 'None')
    # Getting the type of 'None' (line 642)
    None_122864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 39), 'None')
    defaults = [None_122863, None_122864]
    # Create a new context for function 'variance'
    module_type_store = module_type_store.open_function_context('variance', 642, 0, False)
    
    # Passed parameters checking function
    variance.stypy_localization = localization
    variance.stypy_type_of_self = None
    variance.stypy_type_store = module_type_store
    variance.stypy_function_name = 'variance'
    variance.stypy_param_names_list = ['input', 'labels', 'index']
    variance.stypy_varargs_param_name = None
    variance.stypy_kwargs_param_name = None
    variance.stypy_call_defaults = defaults
    variance.stypy_call_varargs = varargs
    variance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'variance', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'variance', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'variance(...)' code ##################

    str_122865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, (-1)), 'str', '\n    Calculate the variance of the values of an n-D image array, optionally at\n    specified sub-regions.\n\n    Parameters\n    ----------\n    input : array_like\n        Nd-image data to process.\n    labels : array_like, optional\n        Labels defining sub-regions in `input`.\n        If not None, must be same shape as `input`.\n    index : int or sequence of ints, optional\n        `labels` to include in output.  If None (default), all values where\n        `labels` is non-zero are used.\n\n    Returns\n    -------\n    variance : float or ndarray\n        Values of variance, for each sub-region if `labels` and `index` are\n        specified.\n\n    See Also\n    --------\n    label, standard_deviation, maximum, minimum, extrema\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> from scipy import ndimage\n    >>> ndimage.variance(a)\n    7.609375\n\n    Features to process can be specified using `labels` and `index`:\n\n    >>> lbl, nlbl = ndimage.label(a)\n    >>> ndimage.variance(a, lbl, index=np.arange(1, nlbl+1))\n    array([ 2.1875,  2.25  ,  9.    ])\n\n    If no index is given, all non-zero `labels` are processed:\n\n    >>> ndimage.variance(a, lbl)\n    6.1875\n\n    ')
    
    # Assigning a Call to a Tuple (line 690):
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_122866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to _stats(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'input' (line 690)
    input_122868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 34), 'input', False)
    # Getting the type of 'labels' (line 690)
    labels_122869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 41), 'labels', False)
    # Getting the type of 'index' (line 690)
    index_122870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 49), 'index', False)
    # Processing the call keyword arguments (line 690)
    # Getting the type of 'True' (line 690)
    True_122871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 65), 'True', False)
    keyword_122872 = True_122871
    kwargs_122873 = {'centered': keyword_122872}
    # Getting the type of '_stats' (line 690)
    _stats_122867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 27), '_stats', False)
    # Calling _stats(args, kwargs) (line 690)
    _stats_call_result_122874 = invoke(stypy.reporting.localization.Localization(__file__, 690, 27), _stats_122867, *[input_122868, labels_122869, index_122870], **kwargs_122873)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___122875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), _stats_call_result_122874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_122876 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___122875, int_122866)
    
    # Assigning a type to the variable 'tuple_var_assignment_121912' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121912', subscript_call_result_122876)
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_122877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to _stats(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'input' (line 690)
    input_122879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 34), 'input', False)
    # Getting the type of 'labels' (line 690)
    labels_122880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 41), 'labels', False)
    # Getting the type of 'index' (line 690)
    index_122881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 49), 'index', False)
    # Processing the call keyword arguments (line 690)
    # Getting the type of 'True' (line 690)
    True_122882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 65), 'True', False)
    keyword_122883 = True_122882
    kwargs_122884 = {'centered': keyword_122883}
    # Getting the type of '_stats' (line 690)
    _stats_122878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 27), '_stats', False)
    # Calling _stats(args, kwargs) (line 690)
    _stats_call_result_122885 = invoke(stypy.reporting.localization.Localization(__file__, 690, 27), _stats_122878, *[input_122879, labels_122880, index_122881], **kwargs_122884)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___122886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), _stats_call_result_122885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_122887 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___122886, int_122877)
    
    # Assigning a type to the variable 'tuple_var_assignment_121913' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121913', subscript_call_result_122887)
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_122888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to _stats(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'input' (line 690)
    input_122890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 34), 'input', False)
    # Getting the type of 'labels' (line 690)
    labels_122891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 41), 'labels', False)
    # Getting the type of 'index' (line 690)
    index_122892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 49), 'index', False)
    # Processing the call keyword arguments (line 690)
    # Getting the type of 'True' (line 690)
    True_122893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 65), 'True', False)
    keyword_122894 = True_122893
    kwargs_122895 = {'centered': keyword_122894}
    # Getting the type of '_stats' (line 690)
    _stats_122889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 27), '_stats', False)
    # Calling _stats(args, kwargs) (line 690)
    _stats_call_result_122896 = invoke(stypy.reporting.localization.Localization(__file__, 690, 27), _stats_122889, *[input_122890, labels_122891, index_122892], **kwargs_122895)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___122897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), _stats_call_result_122896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_122898 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___122897, int_122888)
    
    # Assigning a type to the variable 'tuple_var_assignment_121914' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121914', subscript_call_result_122898)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_121912' (line 690)
    tuple_var_assignment_121912_122899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121912')
    # Assigning a type to the variable 'count' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'count', tuple_var_assignment_121912_122899)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_121913' (line 690)
    tuple_var_assignment_121913_122900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121913')
    # Assigning a type to the variable 'sum' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 11), 'sum', tuple_var_assignment_121913_122900)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_121914' (line 690)
    tuple_var_assignment_121914_122901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_121914')
    # Assigning a type to the variable 'sum_c_sq' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 16), 'sum_c_sq', tuple_var_assignment_121914_122901)
    # Getting the type of 'sum_c_sq' (line 691)
    sum_c_sq_122902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 11), 'sum_c_sq')
    
    # Call to astype(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'float' (line 691)
    float_122909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 50), 'float', False)
    # Processing the call keyword arguments (line 691)
    kwargs_122910 = {}
    
    # Call to asanyarray(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'count' (line 691)
    count_122905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 36), 'count', False)
    # Processing the call keyword arguments (line 691)
    kwargs_122906 = {}
    # Getting the type of 'np' (line 691)
    np_122903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 22), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 691)
    asanyarray_122904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 22), np_122903, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 691)
    asanyarray_call_result_122907 = invoke(stypy.reporting.localization.Localization(__file__, 691, 22), asanyarray_122904, *[count_122905], **kwargs_122906)
    
    # Obtaining the member 'astype' of a type (line 691)
    astype_122908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 22), asanyarray_call_result_122907, 'astype')
    # Calling astype(args, kwargs) (line 691)
    astype_call_result_122911 = invoke(stypy.reporting.localization.Localization(__file__, 691, 22), astype_122908, *[float_122909], **kwargs_122910)
    
    # Applying the binary operator 'div' (line 691)
    result_div_122912 = python_operator(stypy.reporting.localization.Localization(__file__, 691, 11), 'div', sum_c_sq_122902, astype_call_result_122911)
    
    # Assigning a type to the variable 'stypy_return_type' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'stypy_return_type', result_div_122912)
    
    # ################# End of 'variance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'variance' in the type store
    # Getting the type of 'stypy_return_type' (line 642)
    stypy_return_type_122913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122913)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'variance'
    return stypy_return_type_122913

# Assigning a type to the variable 'variance' (line 642)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 0), 'variance', variance)

@norecursion
def standard_deviation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 694)
    None_122914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 37), 'None')
    # Getting the type of 'None' (line 694)
    None_122915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 49), 'None')
    defaults = [None_122914, None_122915]
    # Create a new context for function 'standard_deviation'
    module_type_store = module_type_store.open_function_context('standard_deviation', 694, 0, False)
    
    # Passed parameters checking function
    standard_deviation.stypy_localization = localization
    standard_deviation.stypy_type_of_self = None
    standard_deviation.stypy_type_store = module_type_store
    standard_deviation.stypy_function_name = 'standard_deviation'
    standard_deviation.stypy_param_names_list = ['input', 'labels', 'index']
    standard_deviation.stypy_varargs_param_name = None
    standard_deviation.stypy_kwargs_param_name = None
    standard_deviation.stypy_call_defaults = defaults
    standard_deviation.stypy_call_varargs = varargs
    standard_deviation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'standard_deviation', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'standard_deviation', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'standard_deviation(...)' code ##################

    str_122916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, (-1)), 'str', '\n    Calculate the standard deviation of the values of an n-D image array,\n    optionally at specified sub-regions.\n\n    Parameters\n    ----------\n    input : array_like\n        Nd-image data to process.\n    labels : array_like, optional\n        Labels to identify sub-regions in `input`.\n        If not None, must be same shape as `input`.\n    index : int or sequence of ints, optional\n        `labels` to include in output.  If None (default), all values where\n        `labels` is non-zero are used.\n\n    Returns\n    -------\n    standard_deviation : float or ndarray\n        Values of standard deviation, for each sub-region if `labels` and\n        `index` are specified.\n\n    See Also\n    --------\n    label, variance, maximum, minimum, extrema\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> from scipy import ndimage\n    >>> ndimage.standard_deviation(a)\n    2.7585095613392387\n\n    Features to process can be specified using `labels` and `index`:\n\n    >>> lbl, nlbl = ndimage.label(a)\n    >>> ndimage.standard_deviation(a, lbl, index=np.arange(1, nlbl+1))\n    array([ 1.479,  1.5  ,  3.   ])\n\n    If no index is given, non-zero `labels` are processed:\n\n    >>> ndimage.standard_deviation(a, lbl)\n    2.4874685927665499\n\n    ')
    
    # Call to sqrt(...): (line 742)
    # Processing the call arguments (line 742)
    
    # Call to variance(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'input' (line 742)
    input_122920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 31), 'input', False)
    # Getting the type of 'labels' (line 742)
    labels_122921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 38), 'labels', False)
    # Getting the type of 'index' (line 742)
    index_122922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 46), 'index', False)
    # Processing the call keyword arguments (line 742)
    kwargs_122923 = {}
    # Getting the type of 'variance' (line 742)
    variance_122919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 22), 'variance', False)
    # Calling variance(args, kwargs) (line 742)
    variance_call_result_122924 = invoke(stypy.reporting.localization.Localization(__file__, 742, 22), variance_122919, *[input_122920, labels_122921, index_122922], **kwargs_122923)
    
    # Processing the call keyword arguments (line 742)
    kwargs_122925 = {}
    # Getting the type of 'numpy' (line 742)
    numpy_122917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'numpy', False)
    # Obtaining the member 'sqrt' of a type (line 742)
    sqrt_122918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 11), numpy_122917, 'sqrt')
    # Calling sqrt(args, kwargs) (line 742)
    sqrt_call_result_122926 = invoke(stypy.reporting.localization.Localization(__file__, 742, 11), sqrt_122918, *[variance_call_result_122924], **kwargs_122925)
    
    # Assigning a type to the variable 'stypy_return_type' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'stypy_return_type', sqrt_call_result_122926)
    
    # ################# End of 'standard_deviation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'standard_deviation' in the type store
    # Getting the type of 'stypy_return_type' (line 694)
    stypy_return_type_122927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_122927)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'standard_deviation'
    return stypy_return_type_122927

# Assigning a type to the variable 'standard_deviation' (line 694)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 0), 'standard_deviation', standard_deviation)

@norecursion
def _select(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 745)
    None_122928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 26), 'None')
    # Getting the type of 'None' (line 745)
    None_122929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 38), 'None')
    # Getting the type of 'False' (line 745)
    False_122930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 53), 'False')
    # Getting the type of 'False' (line 745)
    False_122931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 69), 'False')
    # Getting the type of 'False' (line 746)
    False_122932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 31), 'False')
    # Getting the type of 'False' (line 746)
    False_122933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 57), 'False')
    # Getting the type of 'False' (line 747)
    False_122934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 24), 'False')
    defaults = [None_122928, None_122929, False_122930, False_122931, False_122932, False_122933, False_122934]
    # Create a new context for function '_select'
    module_type_store = module_type_store.open_function_context('_select', 745, 0, False)
    
    # Passed parameters checking function
    _select.stypy_localization = localization
    _select.stypy_type_of_self = None
    _select.stypy_type_store = module_type_store
    _select.stypy_function_name = '_select'
    _select.stypy_param_names_list = ['input', 'labels', 'index', 'find_min', 'find_max', 'find_min_positions', 'find_max_positions', 'find_median']
    _select.stypy_varargs_param_name = None
    _select.stypy_kwargs_param_name = None
    _select.stypy_call_defaults = defaults
    _select.stypy_call_varargs = varargs
    _select.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_select', ['input', 'labels', 'index', 'find_min', 'find_max', 'find_min_positions', 'find_max_positions', 'find_median'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_select', localization, ['input', 'labels', 'index', 'find_min', 'find_max', 'find_min_positions', 'find_max_positions', 'find_median'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_select(...)' code ##################

    str_122935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, (-1)), 'str', 'Returns min, max, or both, plus their positions (if requested), and\n    median.')
    
    # Assigning a Call to a Name (line 751):
    
    # Assigning a Call to a Name (line 751):
    
    # Call to asanyarray(...): (line 751)
    # Processing the call arguments (line 751)
    # Getting the type of 'input' (line 751)
    input_122938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 29), 'input', False)
    # Processing the call keyword arguments (line 751)
    kwargs_122939 = {}
    # Getting the type of 'numpy' (line 751)
    numpy_122936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 12), 'numpy', False)
    # Obtaining the member 'asanyarray' of a type (line 751)
    asanyarray_122937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 12), numpy_122936, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 751)
    asanyarray_call_result_122940 = invoke(stypy.reporting.localization.Localization(__file__, 751, 12), asanyarray_122937, *[input_122938], **kwargs_122939)
    
    # Assigning a type to the variable 'input' (line 751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 4), 'input', asanyarray_call_result_122940)
    
    # Assigning a BoolOp to a Name (line 753):
    
    # Assigning a BoolOp to a Name (line 753):
    
    # Evaluating a boolean operation
    # Getting the type of 'find_min_positions' (line 753)
    find_min_positions_122941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 21), 'find_min_positions')
    # Getting the type of 'find_max_positions' (line 753)
    find_max_positions_122942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 43), 'find_max_positions')
    # Applying the binary operator 'or' (line 753)
    result_or_keyword_122943 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 21), 'or', find_min_positions_122941, find_max_positions_122942)
    
    # Assigning a type to the variable 'find_positions' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'find_positions', result_or_keyword_122943)
    
    # Assigning a Name to a Name (line 754):
    
    # Assigning a Name to a Name (line 754):
    # Getting the type of 'None' (line 754)
    None_122944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 16), 'None')
    # Assigning a type to the variable 'positions' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'positions', None_122944)
    
    # Getting the type of 'find_positions' (line 755)
    find_positions_122945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 7), 'find_positions')
    # Testing the type of an if condition (line 755)
    if_condition_122946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 755, 4), find_positions_122945)
    # Assigning a type to the variable 'if_condition_122946' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 4), 'if_condition_122946', if_condition_122946)
    # SSA begins for if statement (line 755)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 756):
    
    # Assigning a Call to a Name (line 756):
    
    # Call to reshape(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'input' (line 756)
    input_122954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 53), 'input', False)
    # Obtaining the member 'shape' of a type (line 756)
    shape_122955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 53), input_122954, 'shape')
    # Processing the call keyword arguments (line 756)
    kwargs_122956 = {}
    
    # Call to arange(...): (line 756)
    # Processing the call arguments (line 756)
    # Getting the type of 'input' (line 756)
    input_122949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 33), 'input', False)
    # Obtaining the member 'size' of a type (line 756)
    size_122950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 33), input_122949, 'size')
    # Processing the call keyword arguments (line 756)
    kwargs_122951 = {}
    # Getting the type of 'numpy' (line 756)
    numpy_122947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 20), 'numpy', False)
    # Obtaining the member 'arange' of a type (line 756)
    arange_122948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 20), numpy_122947, 'arange')
    # Calling arange(args, kwargs) (line 756)
    arange_call_result_122952 = invoke(stypy.reporting.localization.Localization(__file__, 756, 20), arange_122948, *[size_122950], **kwargs_122951)
    
    # Obtaining the member 'reshape' of a type (line 756)
    reshape_122953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 20), arange_call_result_122952, 'reshape')
    # Calling reshape(args, kwargs) (line 756)
    reshape_call_result_122957 = invoke(stypy.reporting.localization.Localization(__file__, 756, 20), reshape_122953, *[shape_122955], **kwargs_122956)
    
    # Assigning a type to the variable 'positions' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'positions', reshape_call_result_122957)
    # SSA join for if statement (line 755)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def single_group(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'single_group'
        module_type_store = module_type_store.open_function_context('single_group', 758, 4, False)
        
        # Passed parameters checking function
        single_group.stypy_localization = localization
        single_group.stypy_type_of_self = None
        single_group.stypy_type_store = module_type_store
        single_group.stypy_function_name = 'single_group'
        single_group.stypy_param_names_list = ['vals', 'positions']
        single_group.stypy_varargs_param_name = None
        single_group.stypy_kwargs_param_name = None
        single_group.stypy_call_defaults = defaults
        single_group.stypy_call_varargs = varargs
        single_group.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'single_group', ['vals', 'positions'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'single_group', localization, ['vals', 'positions'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'single_group(...)' code ##################

        
        # Assigning a List to a Name (line 759):
        
        # Assigning a List to a Name (line 759):
        
        # Obtaining an instance of the builtin type 'list' (line 759)
        list_122958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 759)
        
        # Assigning a type to the variable 'result' (line 759)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'result', list_122958)
        
        # Getting the type of 'find_min' (line 760)
        find_min_122959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 11), 'find_min')
        # Testing the type of an if condition (line 760)
        if_condition_122960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 760, 8), find_min_122959)
        # Assigning a type to the variable 'if_condition_122960' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'if_condition_122960', if_condition_122960)
        # SSA begins for if statement (line 760)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 761)
        result_122961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'result')
        
        # Obtaining an instance of the builtin type 'list' (line 761)
        list_122962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 761, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 761)
        # Adding element type (line 761)
        
        # Call to min(...): (line 761)
        # Processing the call keyword arguments (line 761)
        kwargs_122965 = {}
        # Getting the type of 'vals' (line 761)
        vals_122963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 23), 'vals', False)
        # Obtaining the member 'min' of a type (line 761)
        min_122964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 23), vals_122963, 'min')
        # Calling min(args, kwargs) (line 761)
        min_call_result_122966 = invoke(stypy.reporting.localization.Localization(__file__, 761, 23), min_122964, *[], **kwargs_122965)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 761, 22), list_122962, min_call_result_122966)
        
        # Applying the binary operator '+=' (line 761)
        result_iadd_122967 = python_operator(stypy.reporting.localization.Localization(__file__, 761, 12), '+=', result_122961, list_122962)
        # Assigning a type to the variable 'result' (line 761)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'result', result_iadd_122967)
        
        # SSA join for if statement (line 760)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'find_min_positions' (line 762)
        find_min_positions_122968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 11), 'find_min_positions')
        # Testing the type of an if condition (line 762)
        if_condition_122969 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 762, 8), find_min_positions_122968)
        # Assigning a type to the variable 'if_condition_122969' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 8), 'if_condition_122969', if_condition_122969)
        # SSA begins for if statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 763)
        result_122970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'result')
        
        # Obtaining an instance of the builtin type 'list' (line 763)
        list_122971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 763)
        # Adding element type (line 763)
        
        # Obtaining the type of the subscript
        int_122972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 53), 'int')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'vals' (line 763)
        vals_122973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 33), 'vals')
        
        # Call to min(...): (line 763)
        # Processing the call keyword arguments (line 763)
        kwargs_122976 = {}
        # Getting the type of 'vals' (line 763)
        vals_122974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 41), 'vals', False)
        # Obtaining the member 'min' of a type (line 763)
        min_122975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 41), vals_122974, 'min')
        # Calling min(args, kwargs) (line 763)
        min_call_result_122977 = invoke(stypy.reporting.localization.Localization(__file__, 763, 41), min_122975, *[], **kwargs_122976)
        
        # Applying the binary operator '==' (line 763)
        result_eq_122978 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 33), '==', vals_122973, min_call_result_122977)
        
        # Getting the type of 'positions' (line 763)
        positions_122979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 23), 'positions')
        # Obtaining the member '__getitem__' of a type (line 763)
        getitem___122980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 23), positions_122979, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 763)
        subscript_call_result_122981 = invoke(stypy.reporting.localization.Localization(__file__, 763, 23), getitem___122980, result_eq_122978)
        
        # Obtaining the member '__getitem__' of a type (line 763)
        getitem___122982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 23), subscript_call_result_122981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 763)
        subscript_call_result_122983 = invoke(stypy.reporting.localization.Localization(__file__, 763, 23), getitem___122982, int_122972)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 22), list_122971, subscript_call_result_122983)
        
        # Applying the binary operator '+=' (line 763)
        result_iadd_122984 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 12), '+=', result_122970, list_122971)
        # Assigning a type to the variable 'result' (line 763)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'result', result_iadd_122984)
        
        # SSA join for if statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'find_max' (line 764)
        find_max_122985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 11), 'find_max')
        # Testing the type of an if condition (line 764)
        if_condition_122986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 764, 8), find_max_122985)
        # Assigning a type to the variable 'if_condition_122986' (line 764)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 8), 'if_condition_122986', if_condition_122986)
        # SSA begins for if statement (line 764)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 765)
        result_122987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'result')
        
        # Obtaining an instance of the builtin type 'list' (line 765)
        list_122988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 765)
        # Adding element type (line 765)
        
        # Call to max(...): (line 765)
        # Processing the call keyword arguments (line 765)
        kwargs_122991 = {}
        # Getting the type of 'vals' (line 765)
        vals_122989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 23), 'vals', False)
        # Obtaining the member 'max' of a type (line 765)
        max_122990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 23), vals_122989, 'max')
        # Calling max(args, kwargs) (line 765)
        max_call_result_122992 = invoke(stypy.reporting.localization.Localization(__file__, 765, 23), max_122990, *[], **kwargs_122991)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 765, 22), list_122988, max_call_result_122992)
        
        # Applying the binary operator '+=' (line 765)
        result_iadd_122993 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 12), '+=', result_122987, list_122988)
        # Assigning a type to the variable 'result' (line 765)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'result', result_iadd_122993)
        
        # SSA join for if statement (line 764)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'find_max_positions' (line 766)
        find_max_positions_122994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 11), 'find_max_positions')
        # Testing the type of an if condition (line 766)
        if_condition_122995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 766, 8), find_max_positions_122994)
        # Assigning a type to the variable 'if_condition_122995' (line 766)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'if_condition_122995', if_condition_122995)
        # SSA begins for if statement (line 766)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 767)
        result_122996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'result')
        
        # Obtaining an instance of the builtin type 'list' (line 767)
        list_122997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 767)
        # Adding element type (line 767)
        
        # Obtaining the type of the subscript
        int_122998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 767, 53), 'int')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'vals' (line 767)
        vals_122999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 33), 'vals')
        
        # Call to max(...): (line 767)
        # Processing the call keyword arguments (line 767)
        kwargs_123002 = {}
        # Getting the type of 'vals' (line 767)
        vals_123000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 41), 'vals', False)
        # Obtaining the member 'max' of a type (line 767)
        max_123001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 41), vals_123000, 'max')
        # Calling max(args, kwargs) (line 767)
        max_call_result_123003 = invoke(stypy.reporting.localization.Localization(__file__, 767, 41), max_123001, *[], **kwargs_123002)
        
        # Applying the binary operator '==' (line 767)
        result_eq_123004 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 33), '==', vals_122999, max_call_result_123003)
        
        # Getting the type of 'positions' (line 767)
        positions_123005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 23), 'positions')
        # Obtaining the member '__getitem__' of a type (line 767)
        getitem___123006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 23), positions_123005, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 767)
        subscript_call_result_123007 = invoke(stypy.reporting.localization.Localization(__file__, 767, 23), getitem___123006, result_eq_123004)
        
        # Obtaining the member '__getitem__' of a type (line 767)
        getitem___123008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 23), subscript_call_result_123007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 767)
        subscript_call_result_123009 = invoke(stypy.reporting.localization.Localization(__file__, 767, 23), getitem___123008, int_122998)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 767, 22), list_122997, subscript_call_result_123009)
        
        # Applying the binary operator '+=' (line 767)
        result_iadd_123010 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 12), '+=', result_122996, list_122997)
        # Assigning a type to the variable 'result' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'result', result_iadd_123010)
        
        # SSA join for if statement (line 766)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'find_median' (line 768)
        find_median_123011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 11), 'find_median')
        # Testing the type of an if condition (line 768)
        if_condition_123012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 768, 8), find_median_123011)
        # Assigning a type to the variable 'if_condition_123012' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'if_condition_123012', if_condition_123012)
        # SSA begins for if statement (line 768)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'result' (line 769)
        result_123013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'result')
        
        # Obtaining an instance of the builtin type 'list' (line 769)
        list_123014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 769)
        # Adding element type (line 769)
        
        # Call to median(...): (line 769)
        # Processing the call arguments (line 769)
        # Getting the type of 'vals' (line 769)
        vals_123017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 36), 'vals', False)
        # Processing the call keyword arguments (line 769)
        kwargs_123018 = {}
        # Getting the type of 'numpy' (line 769)
        numpy_123015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 23), 'numpy', False)
        # Obtaining the member 'median' of a type (line 769)
        median_123016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 23), numpy_123015, 'median')
        # Calling median(args, kwargs) (line 769)
        median_call_result_123019 = invoke(stypy.reporting.localization.Localization(__file__, 769, 23), median_123016, *[vals_123017], **kwargs_123018)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 22), list_123014, median_call_result_123019)
        
        # Applying the binary operator '+=' (line 769)
        result_iadd_123020 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 12), '+=', result_123013, list_123014)
        # Assigning a type to the variable 'result' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'result', result_iadd_123020)
        
        # SSA join for if statement (line 768)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 770)
        result_123021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'stypy_return_type', result_123021)
        
        # ################# End of 'single_group(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'single_group' in the type store
        # Getting the type of 'stypy_return_type' (line 758)
        stypy_return_type_123022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123022)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'single_group'
        return stypy_return_type_123022

    # Assigning a type to the variable 'single_group' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'single_group', single_group)
    
    # Type idiom detected: calculating its left and rigth part (line 772)
    # Getting the type of 'labels' (line 772)
    labels_123023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 7), 'labels')
    # Getting the type of 'None' (line 772)
    None_123024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 17), 'None')
    
    (may_be_123025, more_types_in_union_123026) = may_be_none(labels_123023, None_123024)

    if may_be_123025:

        if more_types_in_union_123026:
            # Runtime conditional SSA (line 772)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to single_group(...): (line 773)
        # Processing the call arguments (line 773)
        # Getting the type of 'input' (line 773)
        input_123028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 28), 'input', False)
        # Getting the type of 'positions' (line 773)
        positions_123029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 35), 'positions', False)
        # Processing the call keyword arguments (line 773)
        kwargs_123030 = {}
        # Getting the type of 'single_group' (line 773)
        single_group_123027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 15), 'single_group', False)
        # Calling single_group(args, kwargs) (line 773)
        single_group_call_result_123031 = invoke(stypy.reporting.localization.Localization(__file__, 773, 15), single_group_123027, *[input_123028, positions_123029], **kwargs_123030)
        
        # Assigning a type to the variable 'stypy_return_type' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 8), 'stypy_return_type', single_group_call_result_123031)

        if more_types_in_union_123026:
            # SSA join for if statement (line 772)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 776):
    
    # Assigning a Subscript to a Name (line 776):
    
    # Obtaining the type of the subscript
    int_123032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 4), 'int')
    
    # Call to broadcast_arrays(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'input' (line 776)
    input_123035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 43), 'input', False)
    # Getting the type of 'labels' (line 776)
    labels_123036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 50), 'labels', False)
    # Processing the call keyword arguments (line 776)
    kwargs_123037 = {}
    # Getting the type of 'numpy' (line 776)
    numpy_123033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 20), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 776)
    broadcast_arrays_123034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 20), numpy_123033, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 776)
    broadcast_arrays_call_result_123038 = invoke(stypy.reporting.localization.Localization(__file__, 776, 20), broadcast_arrays_123034, *[input_123035, labels_123036], **kwargs_123037)
    
    # Obtaining the member '__getitem__' of a type (line 776)
    getitem___123039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 4), broadcast_arrays_call_result_123038, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 776)
    subscript_call_result_123040 = invoke(stypy.reporting.localization.Localization(__file__, 776, 4), getitem___123039, int_123032)
    
    # Assigning a type to the variable 'tuple_var_assignment_121915' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_121915', subscript_call_result_123040)
    
    # Assigning a Subscript to a Name (line 776):
    
    # Obtaining the type of the subscript
    int_123041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 4), 'int')
    
    # Call to broadcast_arrays(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'input' (line 776)
    input_123044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 43), 'input', False)
    # Getting the type of 'labels' (line 776)
    labels_123045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 50), 'labels', False)
    # Processing the call keyword arguments (line 776)
    kwargs_123046 = {}
    # Getting the type of 'numpy' (line 776)
    numpy_123042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 20), 'numpy', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 776)
    broadcast_arrays_123043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 20), numpy_123042, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 776)
    broadcast_arrays_call_result_123047 = invoke(stypy.reporting.localization.Localization(__file__, 776, 20), broadcast_arrays_123043, *[input_123044, labels_123045], **kwargs_123046)
    
    # Obtaining the member '__getitem__' of a type (line 776)
    getitem___123048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 4), broadcast_arrays_call_result_123047, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 776)
    subscript_call_result_123049 = invoke(stypy.reporting.localization.Localization(__file__, 776, 4), getitem___123048, int_123041)
    
    # Assigning a type to the variable 'tuple_var_assignment_121916' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_121916', subscript_call_result_123049)
    
    # Assigning a Name to a Name (line 776):
    # Getting the type of 'tuple_var_assignment_121915' (line 776)
    tuple_var_assignment_121915_123050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_121915')
    # Assigning a type to the variable 'input' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'input', tuple_var_assignment_121915_123050)
    
    # Assigning a Name to a Name (line 776):
    # Getting the type of 'tuple_var_assignment_121916' (line 776)
    tuple_var_assignment_121916_123051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 4), 'tuple_var_assignment_121916')
    # Assigning a type to the variable 'labels' (line 776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 776, 11), 'labels', tuple_var_assignment_121916_123051)
    
    # Type idiom detected: calculating its left and rigth part (line 778)
    # Getting the type of 'index' (line 778)
    index_123052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 7), 'index')
    # Getting the type of 'None' (line 778)
    None_123053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 16), 'None')
    
    (may_be_123054, more_types_in_union_123055) = may_be_none(index_123052, None_123053)

    if may_be_123054:

        if more_types_in_union_123055:
            # Runtime conditional SSA (line 778)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Compare to a Name (line 779):
        
        # Assigning a Compare to a Name (line 779):
        
        # Getting the type of 'labels' (line 779)
        labels_123056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 16), 'labels')
        int_123057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 25), 'int')
        # Applying the binary operator '>' (line 779)
        result_gt_123058 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 16), '>', labels_123056, int_123057)
        
        # Assigning a type to the variable 'mask' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 8), 'mask', result_gt_123058)
        
        # Assigning a Name to a Name (line 780):
        
        # Assigning a Name to a Name (line 780):
        # Getting the type of 'None' (line 780)
        None_123059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 27), 'None')
        # Assigning a type to the variable 'masked_positions' (line 780)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 8), 'masked_positions', None_123059)
        
        # Getting the type of 'find_positions' (line 781)
        find_positions_123060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 11), 'find_positions')
        # Testing the type of an if condition (line 781)
        if_condition_123061 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 781, 8), find_positions_123060)
        # Assigning a type to the variable 'if_condition_123061' (line 781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 781, 8), 'if_condition_123061', if_condition_123061)
        # SSA begins for if statement (line 781)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 782):
        
        # Assigning a Subscript to a Name (line 782):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 782)
        mask_123062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 41), 'mask')
        # Getting the type of 'positions' (line 782)
        positions_123063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 31), 'positions')
        # Obtaining the member '__getitem__' of a type (line 782)
        getitem___123064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 31), positions_123063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 782)
        subscript_call_result_123065 = invoke(stypy.reporting.localization.Localization(__file__, 782, 31), getitem___123064, mask_123062)
        
        # Assigning a type to the variable 'masked_positions' (line 782)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 12), 'masked_positions', subscript_call_result_123065)
        # SSA join for if statement (line 781)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to single_group(...): (line 783)
        # Processing the call arguments (line 783)
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 783)
        mask_123067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 34), 'mask', False)
        # Getting the type of 'input' (line 783)
        input_123068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 28), 'input', False)
        # Obtaining the member '__getitem__' of a type (line 783)
        getitem___123069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 28), input_123068, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 783)
        subscript_call_result_123070 = invoke(stypy.reporting.localization.Localization(__file__, 783, 28), getitem___123069, mask_123067)
        
        # Getting the type of 'masked_positions' (line 783)
        masked_positions_123071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 41), 'masked_positions', False)
        # Processing the call keyword arguments (line 783)
        kwargs_123072 = {}
        # Getting the type of 'single_group' (line 783)
        single_group_123066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 15), 'single_group', False)
        # Calling single_group(args, kwargs) (line 783)
        single_group_call_result_123073 = invoke(stypy.reporting.localization.Localization(__file__, 783, 15), single_group_123066, *[subscript_call_result_123070, masked_positions_123071], **kwargs_123072)
        
        # Assigning a type to the variable 'stypy_return_type' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'stypy_return_type', single_group_call_result_123073)

        if more_types_in_union_123055:
            # SSA join for if statement (line 778)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isscalar(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'index' (line 785)
    index_123076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 22), 'index', False)
    # Processing the call keyword arguments (line 785)
    kwargs_123077 = {}
    # Getting the type of 'numpy' (line 785)
    numpy_123074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 785)
    isscalar_123075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 7), numpy_123074, 'isscalar')
    # Calling isscalar(args, kwargs) (line 785)
    isscalar_call_result_123078 = invoke(stypy.reporting.localization.Localization(__file__, 785, 7), isscalar_123075, *[index_123076], **kwargs_123077)
    
    # Testing the type of an if condition (line 785)
    if_condition_123079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 785, 4), isscalar_call_result_123078)
    # Assigning a type to the variable 'if_condition_123079' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'if_condition_123079', if_condition_123079)
    # SSA begins for if statement (line 785)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 786):
    
    # Assigning a Compare to a Name (line 786):
    
    # Getting the type of 'labels' (line 786)
    labels_123080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 16), 'labels')
    # Getting the type of 'index' (line 786)
    index_123081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 26), 'index')
    # Applying the binary operator '==' (line 786)
    result_eq_123082 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 16), '==', labels_123080, index_123081)
    
    # Assigning a type to the variable 'mask' (line 786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'mask', result_eq_123082)
    
    # Assigning a Name to a Name (line 787):
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'None' (line 787)
    None_123083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 27), 'None')
    # Assigning a type to the variable 'masked_positions' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'masked_positions', None_123083)
    
    # Getting the type of 'find_positions' (line 788)
    find_positions_123084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 11), 'find_positions')
    # Testing the type of an if condition (line 788)
    if_condition_123085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 8), find_positions_123084)
    # Assigning a type to the variable 'if_condition_123085' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'if_condition_123085', if_condition_123085)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 789):
    
    # Assigning a Subscript to a Name (line 789):
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 789)
    mask_123086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 41), 'mask')
    # Getting the type of 'positions' (line 789)
    positions_123087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 31), 'positions')
    # Obtaining the member '__getitem__' of a type (line 789)
    getitem___123088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 31), positions_123087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 789)
    subscript_call_result_123089 = invoke(stypy.reporting.localization.Localization(__file__, 789, 31), getitem___123088, mask_123086)
    
    # Assigning a type to the variable 'masked_positions' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 12), 'masked_positions', subscript_call_result_123089)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to single_group(...): (line 790)
    # Processing the call arguments (line 790)
    
    # Obtaining the type of the subscript
    # Getting the type of 'mask' (line 790)
    mask_123091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 34), 'mask', False)
    # Getting the type of 'input' (line 790)
    input_123092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 28), 'input', False)
    # Obtaining the member '__getitem__' of a type (line 790)
    getitem___123093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 790, 28), input_123092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 790)
    subscript_call_result_123094 = invoke(stypy.reporting.localization.Localization(__file__, 790, 28), getitem___123093, mask_123091)
    
    # Getting the type of 'masked_positions' (line 790)
    masked_positions_123095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 41), 'masked_positions', False)
    # Processing the call keyword arguments (line 790)
    kwargs_123096 = {}
    # Getting the type of 'single_group' (line 790)
    single_group_123090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 15), 'single_group', False)
    # Calling single_group(args, kwargs) (line 790)
    single_group_call_result_123097 = invoke(stypy.reporting.localization.Localization(__file__, 790, 15), single_group_123090, *[subscript_call_result_123094, masked_positions_123095], **kwargs_123096)
    
    # Assigning a type to the variable 'stypy_return_type' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'stypy_return_type', single_group_call_result_123097)
    # SSA join for if statement (line 785)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to _safely_castable_to_int(...): (line 794)
    # Processing the call arguments (line 794)
    # Getting the type of 'labels' (line 794)
    labels_123099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 36), 'labels', False)
    # Obtaining the member 'dtype' of a type (line 794)
    dtype_123100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 36), labels_123099, 'dtype')
    # Processing the call keyword arguments (line 794)
    kwargs_123101 = {}
    # Getting the type of '_safely_castable_to_int' (line 794)
    _safely_castable_to_int_123098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 12), '_safely_castable_to_int', False)
    # Calling _safely_castable_to_int(args, kwargs) (line 794)
    _safely_castable_to_int_call_result_123102 = invoke(stypy.reporting.localization.Localization(__file__, 794, 12), _safely_castable_to_int_123098, *[dtype_123100], **kwargs_123101)
    
    # Applying the 'not' unary operator (line 794)
    result_not__123103 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), 'not', _safely_castable_to_int_call_result_123102)
    
    
    
    # Call to min(...): (line 795)
    # Processing the call keyword arguments (line 795)
    kwargs_123106 = {}
    # Getting the type of 'labels' (line 795)
    labels_123104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 12), 'labels', False)
    # Obtaining the member 'min' of a type (line 795)
    min_123105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 12), labels_123104, 'min')
    # Calling min(args, kwargs) (line 795)
    min_call_result_123107 = invoke(stypy.reporting.localization.Localization(__file__, 795, 12), min_123105, *[], **kwargs_123106)
    
    int_123108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 27), 'int')
    # Applying the binary operator '<' (line 795)
    result_lt_123109 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 12), '<', min_call_result_123107, int_123108)
    
    # Applying the binary operator 'or' (line 794)
    result_or_keyword_123110 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), 'or', result_not__123103, result_lt_123109)
    
    
    # Call to max(...): (line 795)
    # Processing the call keyword arguments (line 795)
    kwargs_123113 = {}
    # Getting the type of 'labels' (line 795)
    labels_123111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 32), 'labels', False)
    # Obtaining the member 'max' of a type (line 795)
    max_123112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 32), labels_123111, 'max')
    # Calling max(args, kwargs) (line 795)
    max_call_result_123114 = invoke(stypy.reporting.localization.Localization(__file__, 795, 32), max_123112, *[], **kwargs_123113)
    
    # Getting the type of 'labels' (line 795)
    labels_123115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 47), 'labels')
    # Obtaining the member 'size' of a type (line 795)
    size_123116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 47), labels_123115, 'size')
    # Applying the binary operator '>' (line 795)
    result_gt_123117 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 32), '>', max_call_result_123114, size_123116)
    
    # Applying the binary operator 'or' (line 794)
    result_or_keyword_123118 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), 'or', result_or_keyword_123110, result_gt_123117)
    
    # Testing the type of an if condition (line 794)
    if_condition_123119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 794, 4), result_or_keyword_123118)
    # Assigning a type to the variable 'if_condition_123119' (line 794)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 4), 'if_condition_123119', if_condition_123119)
    # SSA begins for if statement (line 794)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 797):
    
    # Assigning a Subscript to a Name (line 797):
    
    # Obtaining the type of the subscript
    int_123120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 8), 'int')
    
    # Call to unique(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'labels' (line 797)
    labels_123123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 45), 'labels', False)
    # Processing the call keyword arguments (line 797)
    # Getting the type of 'True' (line 797)
    True_123124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 68), 'True', False)
    keyword_123125 = True_123124
    kwargs_123126 = {'return_inverse': keyword_123125}
    # Getting the type of 'numpy' (line 797)
    numpy_123121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 32), 'numpy', False)
    # Obtaining the member 'unique' of a type (line 797)
    unique_123122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 32), numpy_123121, 'unique')
    # Calling unique(args, kwargs) (line 797)
    unique_call_result_123127 = invoke(stypy.reporting.localization.Localization(__file__, 797, 32), unique_123122, *[labels_123123], **kwargs_123126)
    
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___123128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), unique_call_result_123127, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 797)
    subscript_call_result_123129 = invoke(stypy.reporting.localization.Localization(__file__, 797, 8), getitem___123128, int_123120)
    
    # Assigning a type to the variable 'tuple_var_assignment_121917' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'tuple_var_assignment_121917', subscript_call_result_123129)
    
    # Assigning a Subscript to a Name (line 797):
    
    # Obtaining the type of the subscript
    int_123130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 797, 8), 'int')
    
    # Call to unique(...): (line 797)
    # Processing the call arguments (line 797)
    # Getting the type of 'labels' (line 797)
    labels_123133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 45), 'labels', False)
    # Processing the call keyword arguments (line 797)
    # Getting the type of 'True' (line 797)
    True_123134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 68), 'True', False)
    keyword_123135 = True_123134
    kwargs_123136 = {'return_inverse': keyword_123135}
    # Getting the type of 'numpy' (line 797)
    numpy_123131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 32), 'numpy', False)
    # Obtaining the member 'unique' of a type (line 797)
    unique_123132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 32), numpy_123131, 'unique')
    # Calling unique(args, kwargs) (line 797)
    unique_call_result_123137 = invoke(stypy.reporting.localization.Localization(__file__, 797, 32), unique_123132, *[labels_123133], **kwargs_123136)
    
    # Obtaining the member '__getitem__' of a type (line 797)
    getitem___123138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 797, 8), unique_call_result_123137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 797)
    subscript_call_result_123139 = invoke(stypy.reporting.localization.Localization(__file__, 797, 8), getitem___123138, int_123130)
    
    # Assigning a type to the variable 'tuple_var_assignment_121918' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'tuple_var_assignment_121918', subscript_call_result_123139)
    
    # Assigning a Name to a Name (line 797):
    # Getting the type of 'tuple_var_assignment_121917' (line 797)
    tuple_var_assignment_121917_123140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'tuple_var_assignment_121917')
    # Assigning a type to the variable 'unique_labels' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'unique_labels', tuple_var_assignment_121917_123140)
    
    # Assigning a Name to a Name (line 797):
    # Getting the type of 'tuple_var_assignment_121918' (line 797)
    tuple_var_assignment_121918_123141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 8), 'tuple_var_assignment_121918')
    # Assigning a type to the variable 'labels' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 23), 'labels', tuple_var_assignment_121918_123141)
    
    # Assigning a Call to a Name (line 798):
    
    # Assigning a Call to a Name (line 798):
    
    # Call to searchsorted(...): (line 798)
    # Processing the call arguments (line 798)
    # Getting the type of 'unique_labels' (line 798)
    unique_labels_123144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 34), 'unique_labels', False)
    # Getting the type of 'index' (line 798)
    index_123145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 49), 'index', False)
    # Processing the call keyword arguments (line 798)
    kwargs_123146 = {}
    # Getting the type of 'numpy' (line 798)
    numpy_123142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 15), 'numpy', False)
    # Obtaining the member 'searchsorted' of a type (line 798)
    searchsorted_123143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 15), numpy_123142, 'searchsorted')
    # Calling searchsorted(args, kwargs) (line 798)
    searchsorted_call_result_123147 = invoke(stypy.reporting.localization.Localization(__file__, 798, 15), searchsorted_123143, *[unique_labels_123144, index_123145], **kwargs_123146)
    
    # Assigning a type to the variable 'idxs' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'idxs', searchsorted_call_result_123147)
    
    # Assigning a Num to a Subscript (line 801):
    
    # Assigning a Num to a Subscript (line 801):
    int_123148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 43), 'int')
    # Getting the type of 'idxs' (line 801)
    idxs_123149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 8), 'idxs')
    
    # Getting the type of 'idxs' (line 801)
    idxs_123150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 13), 'idxs')
    # Getting the type of 'unique_labels' (line 801)
    unique_labels_123151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 21), 'unique_labels')
    # Obtaining the member 'size' of a type (line 801)
    size_123152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 21), unique_labels_123151, 'size')
    # Applying the binary operator '>=' (line 801)
    result_ge_123153 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 13), '>=', idxs_123150, size_123152)
    
    # Storing an element on a container (line 801)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 801, 8), idxs_123149, (result_ge_123153, int_123148))
    
    # Assigning a Compare to a Name (line 802):
    
    # Assigning a Compare to a Name (line 802):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 802)
    idxs_123154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 31), 'idxs')
    # Getting the type of 'unique_labels' (line 802)
    unique_labels_123155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 17), 'unique_labels')
    # Obtaining the member '__getitem__' of a type (line 802)
    getitem___123156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 802, 17), unique_labels_123155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 802)
    subscript_call_result_123157 = invoke(stypy.reporting.localization.Localization(__file__, 802, 17), getitem___123156, idxs_123154)
    
    # Getting the type of 'index' (line 802)
    index_123158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 40), 'index')
    # Applying the binary operator '==' (line 802)
    result_eq_123159 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 17), '==', subscript_call_result_123157, index_123158)
    
    # Assigning a type to the variable 'found' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 8), 'found', result_eq_123159)
    # SSA branch for the else part of an if statement (line 794)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 805):
    
    # Assigning a Call to a Name (line 805):
    
    # Call to copy(...): (line 805)
    # Processing the call keyword arguments (line 805)
    kwargs_123168 = {}
    
    # Call to asanyarray(...): (line 805)
    # Processing the call arguments (line 805)
    # Getting the type of 'index' (line 805)
    index_123162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 32), 'index', False)
    # Getting the type of 'numpy' (line 805)
    numpy_123163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 39), 'numpy', False)
    # Obtaining the member 'int' of a type (line 805)
    int_123164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 39), numpy_123163, 'int')
    # Processing the call keyword arguments (line 805)
    kwargs_123165 = {}
    # Getting the type of 'numpy' (line 805)
    numpy_123160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 15), 'numpy', False)
    # Obtaining the member 'asanyarray' of a type (line 805)
    asanyarray_123161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 15), numpy_123160, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 805)
    asanyarray_call_result_123166 = invoke(stypy.reporting.localization.Localization(__file__, 805, 15), asanyarray_123161, *[index_123162, int_123164], **kwargs_123165)
    
    # Obtaining the member 'copy' of a type (line 805)
    copy_123167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 15), asanyarray_call_result_123166, 'copy')
    # Calling copy(args, kwargs) (line 805)
    copy_call_result_123169 = invoke(stypy.reporting.localization.Localization(__file__, 805, 15), copy_123167, *[], **kwargs_123168)
    
    # Assigning a type to the variable 'idxs' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'idxs', copy_call_result_123169)
    
    # Assigning a BinOp to a Name (line 806):
    
    # Assigning a BinOp to a Name (line 806):
    
    # Getting the type of 'idxs' (line 806)
    idxs_123170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 17), 'idxs')
    int_123171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 25), 'int')
    # Applying the binary operator '>=' (line 806)
    result_ge_123172 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 17), '>=', idxs_123170, int_123171)
    
    
    # Getting the type of 'idxs' (line 806)
    idxs_123173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 31), 'idxs')
    
    # Call to max(...): (line 806)
    # Processing the call keyword arguments (line 806)
    kwargs_123176 = {}
    # Getting the type of 'labels' (line 806)
    labels_123174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 39), 'labels', False)
    # Obtaining the member 'max' of a type (line 806)
    max_123175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 39), labels_123174, 'max')
    # Calling max(args, kwargs) (line 806)
    max_call_result_123177 = invoke(stypy.reporting.localization.Localization(__file__, 806, 39), max_123175, *[], **kwargs_123176)
    
    # Applying the binary operator '<=' (line 806)
    result_le_123178 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 31), '<=', idxs_123173, max_call_result_123177)
    
    # Applying the binary operator '&' (line 806)
    result_and__123179 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 16), '&', result_ge_123172, result_le_123178)
    
    # Assigning a type to the variable 'found' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'found', result_and__123179)
    # SSA join for if statement (line 794)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 808):
    
    # Assigning a BinOp to a Subscript (line 808):
    
    # Call to max(...): (line 808)
    # Processing the call keyword arguments (line 808)
    kwargs_123182 = {}
    # Getting the type of 'labels' (line 808)
    labels_123180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 20), 'labels', False)
    # Obtaining the member 'max' of a type (line 808)
    max_123181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 20), labels_123180, 'max')
    # Calling max(args, kwargs) (line 808)
    max_call_result_123183 = invoke(stypy.reporting.localization.Localization(__file__, 808, 20), max_123181, *[], **kwargs_123182)
    
    int_123184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 35), 'int')
    # Applying the binary operator '+' (line 808)
    result_add_123185 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 20), '+', max_call_result_123183, int_123184)
    
    # Getting the type of 'idxs' (line 808)
    idxs_123186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'idxs')
    
    # Getting the type of 'found' (line 808)
    found_123187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 11), 'found')
    # Applying the '~' unary operator (line 808)
    result_inv_123188 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 9), '~', found_123187)
    
    # Storing an element on a container (line 808)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 4), idxs_123186, (result_inv_123188, result_add_123185))
    
    # Getting the type of 'find_median' (line 810)
    find_median_123189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 7), 'find_median')
    # Testing the type of an if condition (line 810)
    if_condition_123190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 4), find_median_123189)
    # Assigning a type to the variable 'if_condition_123190' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'if_condition_123190', if_condition_123190)
    # SSA begins for if statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 811):
    
    # Assigning a Call to a Name (line 811):
    
    # Call to lexsort(...): (line 811)
    # Processing the call arguments (line 811)
    
    # Obtaining an instance of the builtin type 'tuple' (line 811)
    tuple_123193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 811)
    # Adding element type (line 811)
    
    # Call to ravel(...): (line 811)
    # Processing the call keyword arguments (line 811)
    kwargs_123196 = {}
    # Getting the type of 'input' (line 811)
    input_123194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 31), 'input', False)
    # Obtaining the member 'ravel' of a type (line 811)
    ravel_123195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 31), input_123194, 'ravel')
    # Calling ravel(args, kwargs) (line 811)
    ravel_call_result_123197 = invoke(stypy.reporting.localization.Localization(__file__, 811, 31), ravel_123195, *[], **kwargs_123196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 31), tuple_123193, ravel_call_result_123197)
    # Adding element type (line 811)
    
    # Call to ravel(...): (line 811)
    # Processing the call keyword arguments (line 811)
    kwargs_123200 = {}
    # Getting the type of 'labels' (line 811)
    labels_123198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 46), 'labels', False)
    # Obtaining the member 'ravel' of a type (line 811)
    ravel_123199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 46), labels_123198, 'ravel')
    # Calling ravel(args, kwargs) (line 811)
    ravel_call_result_123201 = invoke(stypy.reporting.localization.Localization(__file__, 811, 46), ravel_123199, *[], **kwargs_123200)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 31), tuple_123193, ravel_call_result_123201)
    
    # Processing the call keyword arguments (line 811)
    kwargs_123202 = {}
    # Getting the type of 'numpy' (line 811)
    numpy_123191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 16), 'numpy', False)
    # Obtaining the member 'lexsort' of a type (line 811)
    lexsort_123192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 16), numpy_123191, 'lexsort')
    # Calling lexsort(args, kwargs) (line 811)
    lexsort_call_result_123203 = invoke(stypy.reporting.localization.Localization(__file__, 811, 16), lexsort_123192, *[tuple_123193], **kwargs_123202)
    
    # Assigning a type to the variable 'order' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'order', lexsort_call_result_123203)
    # SSA branch for the else part of an if statement (line 810)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 813):
    
    # Assigning a Call to a Name (line 813):
    
    # Call to argsort(...): (line 813)
    # Processing the call keyword arguments (line 813)
    kwargs_123209 = {}
    
    # Call to ravel(...): (line 813)
    # Processing the call keyword arguments (line 813)
    kwargs_123206 = {}
    # Getting the type of 'input' (line 813)
    input_123204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 16), 'input', False)
    # Obtaining the member 'ravel' of a type (line 813)
    ravel_123205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 16), input_123204, 'ravel')
    # Calling ravel(args, kwargs) (line 813)
    ravel_call_result_123207 = invoke(stypy.reporting.localization.Localization(__file__, 813, 16), ravel_123205, *[], **kwargs_123206)
    
    # Obtaining the member 'argsort' of a type (line 813)
    argsort_123208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 16), ravel_call_result_123207, 'argsort')
    # Calling argsort(args, kwargs) (line 813)
    argsort_call_result_123210 = invoke(stypy.reporting.localization.Localization(__file__, 813, 16), argsort_123208, *[], **kwargs_123209)
    
    # Assigning a type to the variable 'order' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'order', argsort_call_result_123210)
    # SSA join for if statement (line 810)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 814):
    
    # Assigning a Subscript to a Name (line 814):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 814)
    order_123211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 26), 'order')
    
    # Call to ravel(...): (line 814)
    # Processing the call keyword arguments (line 814)
    kwargs_123214 = {}
    # Getting the type of 'input' (line 814)
    input_123212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 12), 'input', False)
    # Obtaining the member 'ravel' of a type (line 814)
    ravel_123213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 12), input_123212, 'ravel')
    # Calling ravel(args, kwargs) (line 814)
    ravel_call_result_123215 = invoke(stypy.reporting.localization.Localization(__file__, 814, 12), ravel_123213, *[], **kwargs_123214)
    
    # Obtaining the member '__getitem__' of a type (line 814)
    getitem___123216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 12), ravel_call_result_123215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 814)
    subscript_call_result_123217 = invoke(stypy.reporting.localization.Localization(__file__, 814, 12), getitem___123216, order_123211)
    
    # Assigning a type to the variable 'input' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'input', subscript_call_result_123217)
    
    # Assigning a Subscript to a Name (line 815):
    
    # Assigning a Subscript to a Name (line 815):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 815)
    order_123218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'order')
    
    # Call to ravel(...): (line 815)
    # Processing the call keyword arguments (line 815)
    kwargs_123221 = {}
    # Getting the type of 'labels' (line 815)
    labels_123219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 13), 'labels', False)
    # Obtaining the member 'ravel' of a type (line 815)
    ravel_123220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 13), labels_123219, 'ravel')
    # Calling ravel(args, kwargs) (line 815)
    ravel_call_result_123222 = invoke(stypy.reporting.localization.Localization(__file__, 815, 13), ravel_123220, *[], **kwargs_123221)
    
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___123223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 13), ravel_call_result_123222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_123224 = invoke(stypy.reporting.localization.Localization(__file__, 815, 13), getitem___123223, order_123218)
    
    # Assigning a type to the variable 'labels' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'labels', subscript_call_result_123224)
    
    # Getting the type of 'find_positions' (line 816)
    find_positions_123225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 7), 'find_positions')
    # Testing the type of an if condition (line 816)
    if_condition_123226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 816, 4), find_positions_123225)
    # Assigning a type to the variable 'if_condition_123226' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'if_condition_123226', if_condition_123226)
    # SSA begins for if statement (line 816)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 817):
    
    # Assigning a Subscript to a Name (line 817):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 817)
    order_123227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 38), 'order')
    
    # Call to ravel(...): (line 817)
    # Processing the call keyword arguments (line 817)
    kwargs_123230 = {}
    # Getting the type of 'positions' (line 817)
    positions_123228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 20), 'positions', False)
    # Obtaining the member 'ravel' of a type (line 817)
    ravel_123229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 20), positions_123228, 'ravel')
    # Calling ravel(args, kwargs) (line 817)
    ravel_call_result_123231 = invoke(stypy.reporting.localization.Localization(__file__, 817, 20), ravel_123229, *[], **kwargs_123230)
    
    # Obtaining the member '__getitem__' of a type (line 817)
    getitem___123232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 20), ravel_call_result_123231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 817)
    subscript_call_result_123233 = invoke(stypy.reporting.localization.Localization(__file__, 817, 20), getitem___123232, order_123227)
    
    # Assigning a type to the variable 'positions' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'positions', subscript_call_result_123233)
    # SSA join for if statement (line 816)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 819):
    
    # Assigning a List to a Name (line 819):
    
    # Obtaining an instance of the builtin type 'list' (line 819)
    list_123234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 819)
    
    # Assigning a type to the variable 'result' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'result', list_123234)
    
    # Getting the type of 'find_min' (line 820)
    find_min_123235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 7), 'find_min')
    # Testing the type of an if condition (line 820)
    if_condition_123236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 820, 4), find_min_123235)
    # Assigning a type to the variable 'if_condition_123236' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 4), 'if_condition_123236', if_condition_123236)
    # SSA begins for if statement (line 820)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 821):
    
    # Assigning a Call to a Name (line 821):
    
    # Call to zeros(...): (line 821)
    # Processing the call arguments (line 821)
    
    # Call to max(...): (line 821)
    # Processing the call keyword arguments (line 821)
    kwargs_123241 = {}
    # Getting the type of 'labels' (line 821)
    labels_123239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 27), 'labels', False)
    # Obtaining the member 'max' of a type (line 821)
    max_123240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 27), labels_123239, 'max')
    # Calling max(args, kwargs) (line 821)
    max_call_result_123242 = invoke(stypy.reporting.localization.Localization(__file__, 821, 27), max_123240, *[], **kwargs_123241)
    
    int_123243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 42), 'int')
    # Applying the binary operator '+' (line 821)
    result_add_123244 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 27), '+', max_call_result_123242, int_123243)
    
    # Getting the type of 'input' (line 821)
    input_123245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 45), 'input', False)
    # Obtaining the member 'dtype' of a type (line 821)
    dtype_123246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 45), input_123245, 'dtype')
    # Processing the call keyword arguments (line 821)
    kwargs_123247 = {}
    # Getting the type of 'numpy' (line 821)
    numpy_123237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 15), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 821)
    zeros_123238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 15), numpy_123237, 'zeros')
    # Calling zeros(args, kwargs) (line 821)
    zeros_call_result_123248 = invoke(stypy.reporting.localization.Localization(__file__, 821, 15), zeros_123238, *[result_add_123244, dtype_123246], **kwargs_123247)
    
    # Assigning a type to the variable 'mins' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'mins', zeros_call_result_123248)
    
    # Assigning a Subscript to a Subscript (line 822):
    
    # Assigning a Subscript to a Subscript (line 822):
    
    # Obtaining the type of the subscript
    int_123249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 37), 'int')
    slice_123250 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 822, 29), None, None, int_123249)
    # Getting the type of 'input' (line 822)
    input_123251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 29), 'input')
    # Obtaining the member '__getitem__' of a type (line 822)
    getitem___123252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 29), input_123251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 822)
    subscript_call_result_123253 = invoke(stypy.reporting.localization.Localization(__file__, 822, 29), getitem___123252, slice_123250)
    
    # Getting the type of 'mins' (line 822)
    mins_123254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 8), 'mins')
    
    # Obtaining the type of the subscript
    int_123255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 22), 'int')
    slice_123256 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 822, 13), None, None, int_123255)
    # Getting the type of 'labels' (line 822)
    labels_123257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 13), 'labels')
    # Obtaining the member '__getitem__' of a type (line 822)
    getitem___123258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 13), labels_123257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 822)
    subscript_call_result_123259 = invoke(stypy.reporting.localization.Localization(__file__, 822, 13), getitem___123258, slice_123256)
    
    # Storing an element on a container (line 822)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 822, 8), mins_123254, (subscript_call_result_123259, subscript_call_result_123253))
    
    # Getting the type of 'result' (line 823)
    result_123260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 8), 'result')
    
    # Obtaining an instance of the builtin type 'list' (line 823)
    list_123261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 823)
    # Adding element type (line 823)
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 823)
    idxs_123262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 24), 'idxs')
    # Getting the type of 'mins' (line 823)
    mins_123263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 19), 'mins')
    # Obtaining the member '__getitem__' of a type (line 823)
    getitem___123264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 19), mins_123263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 823)
    subscript_call_result_123265 = invoke(stypy.reporting.localization.Localization(__file__, 823, 19), getitem___123264, idxs_123262)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 823, 18), list_123261, subscript_call_result_123265)
    
    # Applying the binary operator '+=' (line 823)
    result_iadd_123266 = python_operator(stypy.reporting.localization.Localization(__file__, 823, 8), '+=', result_123260, list_123261)
    # Assigning a type to the variable 'result' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 8), 'result', result_iadd_123266)
    
    # SSA join for if statement (line 820)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'find_min_positions' (line 824)
    find_min_positions_123267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 7), 'find_min_positions')
    # Testing the type of an if condition (line 824)
    if_condition_123268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 4), find_min_positions_123267)
    # Assigning a type to the variable 'if_condition_123268' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'if_condition_123268', if_condition_123268)
    # SSA begins for if statement (line 824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 825):
    
    # Assigning a Call to a Name (line 825):
    
    # Call to zeros(...): (line 825)
    # Processing the call arguments (line 825)
    
    # Call to max(...): (line 825)
    # Processing the call keyword arguments (line 825)
    kwargs_123273 = {}
    # Getting the type of 'labels' (line 825)
    labels_123271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 29), 'labels', False)
    # Obtaining the member 'max' of a type (line 825)
    max_123272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 29), labels_123271, 'max')
    # Calling max(args, kwargs) (line 825)
    max_call_result_123274 = invoke(stypy.reporting.localization.Localization(__file__, 825, 29), max_123272, *[], **kwargs_123273)
    
    int_123275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 44), 'int')
    # Applying the binary operator '+' (line 825)
    result_add_123276 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 29), '+', max_call_result_123274, int_123275)
    
    # Getting the type of 'int' (line 825)
    int_123277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 47), 'int', False)
    # Processing the call keyword arguments (line 825)
    kwargs_123278 = {}
    # Getting the type of 'numpy' (line 825)
    numpy_123269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 17), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 825)
    zeros_123270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 17), numpy_123269, 'zeros')
    # Calling zeros(args, kwargs) (line 825)
    zeros_call_result_123279 = invoke(stypy.reporting.localization.Localization(__file__, 825, 17), zeros_123270, *[result_add_123276, int_123277], **kwargs_123278)
    
    # Assigning a type to the variable 'minpos' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'minpos', zeros_call_result_123279)
    
    # Assigning a Subscript to a Subscript (line 826):
    
    # Assigning a Subscript to a Subscript (line 826):
    
    # Obtaining the type of the subscript
    int_123280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 43), 'int')
    slice_123281 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 826, 31), None, None, int_123280)
    # Getting the type of 'positions' (line 826)
    positions_123282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 31), 'positions')
    # Obtaining the member '__getitem__' of a type (line 826)
    getitem___123283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 31), positions_123282, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 826)
    subscript_call_result_123284 = invoke(stypy.reporting.localization.Localization(__file__, 826, 31), getitem___123283, slice_123281)
    
    # Getting the type of 'minpos' (line 826)
    minpos_123285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 8), 'minpos')
    
    # Obtaining the type of the subscript
    int_123286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 24), 'int')
    slice_123287 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 826, 15), None, None, int_123286)
    # Getting the type of 'labels' (line 826)
    labels_123288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 15), 'labels')
    # Obtaining the member '__getitem__' of a type (line 826)
    getitem___123289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 15), labels_123288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 826)
    subscript_call_result_123290 = invoke(stypy.reporting.localization.Localization(__file__, 826, 15), getitem___123289, slice_123287)
    
    # Storing an element on a container (line 826)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 8), minpos_123285, (subscript_call_result_123290, subscript_call_result_123284))
    
    # Getting the type of 'result' (line 827)
    result_123291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'result')
    
    # Obtaining an instance of the builtin type 'list' (line 827)
    list_123292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 827)
    # Adding element type (line 827)
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 827)
    idxs_123293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 26), 'idxs')
    # Getting the type of 'minpos' (line 827)
    minpos_123294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 19), 'minpos')
    # Obtaining the member '__getitem__' of a type (line 827)
    getitem___123295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 19), minpos_123294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 827)
    subscript_call_result_123296 = invoke(stypy.reporting.localization.Localization(__file__, 827, 19), getitem___123295, idxs_123293)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 827, 18), list_123292, subscript_call_result_123296)
    
    # Applying the binary operator '+=' (line 827)
    result_iadd_123297 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 8), '+=', result_123291, list_123292)
    # Assigning a type to the variable 'result' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'result', result_iadd_123297)
    
    # SSA join for if statement (line 824)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'find_max' (line 828)
    find_max_123298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 7), 'find_max')
    # Testing the type of an if condition (line 828)
    if_condition_123299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 4), find_max_123298)
    # Assigning a type to the variable 'if_condition_123299' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'if_condition_123299', if_condition_123299)
    # SSA begins for if statement (line 828)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 829):
    
    # Assigning a Call to a Name (line 829):
    
    # Call to zeros(...): (line 829)
    # Processing the call arguments (line 829)
    
    # Call to max(...): (line 829)
    # Processing the call keyword arguments (line 829)
    kwargs_123304 = {}
    # Getting the type of 'labels' (line 829)
    labels_123302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 27), 'labels', False)
    # Obtaining the member 'max' of a type (line 829)
    max_123303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 27), labels_123302, 'max')
    # Calling max(args, kwargs) (line 829)
    max_call_result_123305 = invoke(stypy.reporting.localization.Localization(__file__, 829, 27), max_123303, *[], **kwargs_123304)
    
    int_123306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 42), 'int')
    # Applying the binary operator '+' (line 829)
    result_add_123307 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 27), '+', max_call_result_123305, int_123306)
    
    # Getting the type of 'input' (line 829)
    input_123308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 45), 'input', False)
    # Obtaining the member 'dtype' of a type (line 829)
    dtype_123309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 45), input_123308, 'dtype')
    # Processing the call keyword arguments (line 829)
    kwargs_123310 = {}
    # Getting the type of 'numpy' (line 829)
    numpy_123300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 15), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 829)
    zeros_123301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 15), numpy_123300, 'zeros')
    # Calling zeros(args, kwargs) (line 829)
    zeros_call_result_123311 = invoke(stypy.reporting.localization.Localization(__file__, 829, 15), zeros_123301, *[result_add_123307, dtype_123309], **kwargs_123310)
    
    # Assigning a type to the variable 'maxs' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'maxs', zeros_call_result_123311)
    
    # Assigning a Name to a Subscript (line 830):
    
    # Assigning a Name to a Subscript (line 830):
    # Getting the type of 'input' (line 830)
    input_123312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 23), 'input')
    # Getting the type of 'maxs' (line 830)
    maxs_123313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'maxs')
    # Getting the type of 'labels' (line 830)
    labels_123314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 13), 'labels')
    # Storing an element on a container (line 830)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 830, 8), maxs_123313, (labels_123314, input_123312))
    
    # Getting the type of 'result' (line 831)
    result_123315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'result')
    
    # Obtaining an instance of the builtin type 'list' (line 831)
    list_123316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 831)
    # Adding element type (line 831)
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 831)
    idxs_123317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 24), 'idxs')
    # Getting the type of 'maxs' (line 831)
    maxs_123318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 19), 'maxs')
    # Obtaining the member '__getitem__' of a type (line 831)
    getitem___123319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 19), maxs_123318, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 831)
    subscript_call_result_123320 = invoke(stypy.reporting.localization.Localization(__file__, 831, 19), getitem___123319, idxs_123317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 831, 18), list_123316, subscript_call_result_123320)
    
    # Applying the binary operator '+=' (line 831)
    result_iadd_123321 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 8), '+=', result_123315, list_123316)
    # Assigning a type to the variable 'result' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'result', result_iadd_123321)
    
    # SSA join for if statement (line 828)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'find_max_positions' (line 832)
    find_max_positions_123322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 7), 'find_max_positions')
    # Testing the type of an if condition (line 832)
    if_condition_123323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 4), find_max_positions_123322)
    # Assigning a type to the variable 'if_condition_123323' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'if_condition_123323', if_condition_123323)
    # SSA begins for if statement (line 832)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 833):
    
    # Assigning a Call to a Name (line 833):
    
    # Call to zeros(...): (line 833)
    # Processing the call arguments (line 833)
    
    # Call to max(...): (line 833)
    # Processing the call keyword arguments (line 833)
    kwargs_123328 = {}
    # Getting the type of 'labels' (line 833)
    labels_123326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 29), 'labels', False)
    # Obtaining the member 'max' of a type (line 833)
    max_123327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 29), labels_123326, 'max')
    # Calling max(args, kwargs) (line 833)
    max_call_result_123329 = invoke(stypy.reporting.localization.Localization(__file__, 833, 29), max_123327, *[], **kwargs_123328)
    
    int_123330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 44), 'int')
    # Applying the binary operator '+' (line 833)
    result_add_123331 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 29), '+', max_call_result_123329, int_123330)
    
    # Getting the type of 'int' (line 833)
    int_123332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 47), 'int', False)
    # Processing the call keyword arguments (line 833)
    kwargs_123333 = {}
    # Getting the type of 'numpy' (line 833)
    numpy_123324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 17), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 833)
    zeros_123325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 17), numpy_123324, 'zeros')
    # Calling zeros(args, kwargs) (line 833)
    zeros_call_result_123334 = invoke(stypy.reporting.localization.Localization(__file__, 833, 17), zeros_123325, *[result_add_123331, int_123332], **kwargs_123333)
    
    # Assigning a type to the variable 'maxpos' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'maxpos', zeros_call_result_123334)
    
    # Assigning a Name to a Subscript (line 834):
    
    # Assigning a Name to a Subscript (line 834):
    # Getting the type of 'positions' (line 834)
    positions_123335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 25), 'positions')
    # Getting the type of 'maxpos' (line 834)
    maxpos_123336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'maxpos')
    # Getting the type of 'labels' (line 834)
    labels_123337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 15), 'labels')
    # Storing an element on a container (line 834)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 8), maxpos_123336, (labels_123337, positions_123335))
    
    # Getting the type of 'result' (line 835)
    result_123338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'result')
    
    # Obtaining an instance of the builtin type 'list' (line 835)
    list_123339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 835)
    # Adding element type (line 835)
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 835)
    idxs_123340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 26), 'idxs')
    # Getting the type of 'maxpos' (line 835)
    maxpos_123341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 19), 'maxpos')
    # Obtaining the member '__getitem__' of a type (line 835)
    getitem___123342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 835, 19), maxpos_123341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 835)
    subscript_call_result_123343 = invoke(stypy.reporting.localization.Localization(__file__, 835, 19), getitem___123342, idxs_123340)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 835, 18), list_123339, subscript_call_result_123343)
    
    # Applying the binary operator '+=' (line 835)
    result_iadd_123344 = python_operator(stypy.reporting.localization.Localization(__file__, 835, 8), '+=', result_123338, list_123339)
    # Assigning a type to the variable 'result' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'result', result_iadd_123344)
    
    # SSA join for if statement (line 832)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'find_median' (line 836)
    find_median_123345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 7), 'find_median')
    # Testing the type of an if condition (line 836)
    if_condition_123346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 836, 4), find_median_123345)
    # Assigning a type to the variable 'if_condition_123346' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 4), 'if_condition_123346', if_condition_123346)
    # SSA begins for if statement (line 836)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 837):
    
    # Assigning a Call to a Name (line 837):
    
    # Call to arange(...): (line 837)
    # Processing the call arguments (line 837)
    
    # Call to len(...): (line 837)
    # Processing the call arguments (line 837)
    # Getting the type of 'labels' (line 837)
    labels_123350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 32), 'labels', False)
    # Processing the call keyword arguments (line 837)
    kwargs_123351 = {}
    # Getting the type of 'len' (line 837)
    len_123349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 28), 'len', False)
    # Calling len(args, kwargs) (line 837)
    len_call_result_123352 = invoke(stypy.reporting.localization.Localization(__file__, 837, 28), len_123349, *[labels_123350], **kwargs_123351)
    
    # Processing the call keyword arguments (line 837)
    kwargs_123353 = {}
    # Getting the type of 'numpy' (line 837)
    numpy_123347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 15), 'numpy', False)
    # Obtaining the member 'arange' of a type (line 837)
    arange_123348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 15), numpy_123347, 'arange')
    # Calling arange(args, kwargs) (line 837)
    arange_call_result_123354 = invoke(stypy.reporting.localization.Localization(__file__, 837, 15), arange_123348, *[len_call_result_123352], **kwargs_123353)
    
    # Assigning a type to the variable 'locs' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'locs', arange_call_result_123354)
    
    # Assigning a Call to a Name (line 838):
    
    # Assigning a Call to a Name (line 838):
    
    # Call to zeros(...): (line 838)
    # Processing the call arguments (line 838)
    
    # Call to max(...): (line 838)
    # Processing the call keyword arguments (line 838)
    kwargs_123359 = {}
    # Getting the type of 'labels' (line 838)
    labels_123357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 25), 'labels', False)
    # Obtaining the member 'max' of a type (line 838)
    max_123358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 25), labels_123357, 'max')
    # Calling max(args, kwargs) (line 838)
    max_call_result_123360 = invoke(stypy.reporting.localization.Localization(__file__, 838, 25), max_123358, *[], **kwargs_123359)
    
    int_123361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 40), 'int')
    # Applying the binary operator '+' (line 838)
    result_add_123362 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 25), '+', max_call_result_123360, int_123361)
    
    # Getting the type of 'numpy' (line 838)
    numpy_123363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 43), 'numpy', False)
    # Obtaining the member 'int' of a type (line 838)
    int_123364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 43), numpy_123363, 'int')
    # Processing the call keyword arguments (line 838)
    kwargs_123365 = {}
    # Getting the type of 'numpy' (line 838)
    numpy_123355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 838)
    zeros_123356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 13), numpy_123355, 'zeros')
    # Calling zeros(args, kwargs) (line 838)
    zeros_call_result_123366 = invoke(stypy.reporting.localization.Localization(__file__, 838, 13), zeros_123356, *[result_add_123362, int_123364], **kwargs_123365)
    
    # Assigning a type to the variable 'lo' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'lo', zeros_call_result_123366)
    
    # Assigning a Subscript to a Subscript (line 839):
    
    # Assigning a Subscript to a Subscript (line 839):
    
    # Obtaining the type of the subscript
    int_123367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 34), 'int')
    slice_123368 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 839, 27), None, None, int_123367)
    # Getting the type of 'locs' (line 839)
    locs_123369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 27), 'locs')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___123370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 27), locs_123369, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_123371 = invoke(stypy.reporting.localization.Localization(__file__, 839, 27), getitem___123370, slice_123368)
    
    # Getting the type of 'lo' (line 839)
    lo_123372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'lo')
    
    # Obtaining the type of the subscript
    int_123373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 20), 'int')
    slice_123374 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 839, 11), None, None, int_123373)
    # Getting the type of 'labels' (line 839)
    labels_123375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 11), 'labels')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___123376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 11), labels_123375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_123377 = invoke(stypy.reporting.localization.Localization(__file__, 839, 11), getitem___123376, slice_123374)
    
    # Storing an element on a container (line 839)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 8), lo_123372, (subscript_call_result_123377, subscript_call_result_123371))
    
    # Assigning a Call to a Name (line 840):
    
    # Assigning a Call to a Name (line 840):
    
    # Call to zeros(...): (line 840)
    # Processing the call arguments (line 840)
    
    # Call to max(...): (line 840)
    # Processing the call keyword arguments (line 840)
    kwargs_123382 = {}
    # Getting the type of 'labels' (line 840)
    labels_123380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 25), 'labels', False)
    # Obtaining the member 'max' of a type (line 840)
    max_123381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 25), labels_123380, 'max')
    # Calling max(args, kwargs) (line 840)
    max_call_result_123383 = invoke(stypy.reporting.localization.Localization(__file__, 840, 25), max_123381, *[], **kwargs_123382)
    
    int_123384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 40), 'int')
    # Applying the binary operator '+' (line 840)
    result_add_123385 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 25), '+', max_call_result_123383, int_123384)
    
    # Getting the type of 'numpy' (line 840)
    numpy_123386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 43), 'numpy', False)
    # Obtaining the member 'int' of a type (line 840)
    int_123387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 43), numpy_123386, 'int')
    # Processing the call keyword arguments (line 840)
    kwargs_123388 = {}
    # Getting the type of 'numpy' (line 840)
    numpy_123378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 840)
    zeros_123379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 13), numpy_123378, 'zeros')
    # Calling zeros(args, kwargs) (line 840)
    zeros_call_result_123389 = invoke(stypy.reporting.localization.Localization(__file__, 840, 13), zeros_123379, *[result_add_123385, int_123387], **kwargs_123388)
    
    # Assigning a type to the variable 'hi' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'hi', zeros_call_result_123389)
    
    # Assigning a Name to a Subscript (line 841):
    
    # Assigning a Name to a Subscript (line 841):
    # Getting the type of 'locs' (line 841)
    locs_123390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 21), 'locs')
    # Getting the type of 'hi' (line 841)
    hi_123391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'hi')
    # Getting the type of 'labels' (line 841)
    labels_123392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 11), 'labels')
    # Storing an element on a container (line 841)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 8), hi_123391, (labels_123392, locs_123390))
    
    # Assigning a Subscript to a Name (line 842):
    
    # Assigning a Subscript to a Name (line 842):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 842)
    idxs_123393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 16), 'idxs')
    # Getting the type of 'lo' (line 842)
    lo_123394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 13), 'lo')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___123395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 13), lo_123394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_123396 = invoke(stypy.reporting.localization.Localization(__file__, 842, 13), getitem___123395, idxs_123393)
    
    # Assigning a type to the variable 'lo' (line 842)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'lo', subscript_call_result_123396)
    
    # Assigning a Subscript to a Name (line 843):
    
    # Assigning a Subscript to a Name (line 843):
    
    # Obtaining the type of the subscript
    # Getting the type of 'idxs' (line 843)
    idxs_123397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 16), 'idxs')
    # Getting the type of 'hi' (line 843)
    hi_123398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 13), 'hi')
    # Obtaining the member '__getitem__' of a type (line 843)
    getitem___123399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 13), hi_123398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 843)
    subscript_call_result_123400 = invoke(stypy.reporting.localization.Localization(__file__, 843, 13), getitem___123399, idxs_123397)
    
    # Assigning a type to the variable 'hi' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 8), 'hi', subscript_call_result_123400)
    
    # Assigning a BinOp to a Name (line 848):
    
    # Assigning a BinOp to a Name (line 848):
    # Getting the type of 'hi' (line 848)
    hi_123401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 16), 'hi')
    # Getting the type of 'lo' (line 848)
    lo_123402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 21), 'lo')
    # Applying the binary operator '-' (line 848)
    result_sub_123403 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 16), '-', hi_123401, lo_123402)
    
    int_123404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 28), 'int')
    # Applying the binary operator '//' (line 848)
    result_floordiv_123405 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 15), '//', result_sub_123403, int_123404)
    
    # Assigning a type to the variable 'step' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'step', result_floordiv_123405)
    
    # Getting the type of 'lo' (line 849)
    lo_123406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'lo')
    # Getting the type of 'step' (line 849)
    step_123407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 14), 'step')
    # Applying the binary operator '+=' (line 849)
    result_iadd_123408 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 8), '+=', lo_123406, step_123407)
    # Assigning a type to the variable 'lo' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'lo', result_iadd_123408)
    
    
    # Getting the type of 'hi' (line 850)
    hi_123409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'hi')
    # Getting the type of 'step' (line 850)
    step_123410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 14), 'step')
    # Applying the binary operator '-=' (line 850)
    result_isub_123411 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 8), '-=', hi_123409, step_123410)
    # Assigning a type to the variable 'hi' (line 850)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 8), 'hi', result_isub_123411)
    
    
    # Getting the type of 'result' (line 851)
    result_123412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'result')
    
    # Obtaining an instance of the builtin type 'list' (line 851)
    list_123413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 851)
    # Adding element type (line 851)
    
    # Obtaining the type of the subscript
    # Getting the type of 'lo' (line 851)
    lo_123414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 26), 'lo')
    # Getting the type of 'input' (line 851)
    input_123415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 20), 'input')
    # Obtaining the member '__getitem__' of a type (line 851)
    getitem___123416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 20), input_123415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 851)
    subscript_call_result_123417 = invoke(stypy.reporting.localization.Localization(__file__, 851, 20), getitem___123416, lo_123414)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'hi' (line 851)
    hi_123418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 38), 'hi')
    # Getting the type of 'input' (line 851)
    input_123419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 32), 'input')
    # Obtaining the member '__getitem__' of a type (line 851)
    getitem___123420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 32), input_123419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 851)
    subscript_call_result_123421 = invoke(stypy.reporting.localization.Localization(__file__, 851, 32), getitem___123420, hi_123418)
    
    # Applying the binary operator '+' (line 851)
    result_add_123422 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 20), '+', subscript_call_result_123417, subscript_call_result_123421)
    
    float_123423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 45), 'float')
    # Applying the binary operator 'div' (line 851)
    result_div_123424 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 19), 'div', result_add_123422, float_123423)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 851, 18), list_123413, result_div_123424)
    
    # Applying the binary operator '+=' (line 851)
    result_iadd_123425 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 8), '+=', result_123412, list_123413)
    # Assigning a type to the variable 'result' (line 851)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'result', result_iadd_123425)
    
    # SSA join for if statement (line 836)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 853)
    result_123426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 853)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 4), 'stypy_return_type', result_123426)
    
    # ################# End of '_select(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_select' in the type store
    # Getting the type of 'stypy_return_type' (line 745)
    stypy_return_type_123427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_select'
    return stypy_return_type_123427

# Assigning a type to the variable '_select' (line 745)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 0), '_select', _select)

@norecursion
def minimum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 856)
    None_123428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 26), 'None')
    # Getting the type of 'None' (line 856)
    None_123429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 38), 'None')
    defaults = [None_123428, None_123429]
    # Create a new context for function 'minimum'
    module_type_store = module_type_store.open_function_context('minimum', 856, 0, False)
    
    # Passed parameters checking function
    minimum.stypy_localization = localization
    minimum.stypy_type_of_self = None
    minimum.stypy_type_store = module_type_store
    minimum.stypy_function_name = 'minimum'
    minimum.stypy_param_names_list = ['input', 'labels', 'index']
    minimum.stypy_varargs_param_name = None
    minimum.stypy_kwargs_param_name = None
    minimum.stypy_call_defaults = defaults
    minimum.stypy_call_varargs = varargs
    minimum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimum', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimum', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimum(...)' code ##################

    str_123430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, (-1)), 'str', '\n    Calculate the minimum of the values of an array over labeled regions.\n\n    Parameters\n    ----------\n    input : array_like\n        Array_like of values. For each region specified by `labels`, the\n        minimal values of `input` over the region is computed.\n    labels : array_like, optional\n        An array_like of integers marking different regions over which the\n        minimum value of `input` is to be computed. `labels` must have the\n        same shape as `input`. If `labels` is not specified, the minimum\n        over the whole array is returned.\n    index : array_like, optional\n        A list of region labels that are taken into account for computing the\n        minima. If index is None, the minimum over all elements where `labels`\n        is non-zero is returned.\n\n    Returns\n    -------\n    minimum : float or list of floats\n        List of minima of `input` over the regions determined by `labels` and\n        whose index is in `index`. If `index` or `labels` are not specified, a\n        float is returned: the minimal value of `input` if `labels` is None,\n        and the minimal value of elements where `labels` is greater than zero\n        if `index` is None.\n\n    See also\n    --------\n    label, maximum, median, minimum_position, extrema, sum, mean, variance,\n    standard_deviation\n\n    Notes\n    -----\n    The function returns a Python list and not a Numpy array, use\n    `np.array` to convert the list to an array.\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> labels, labels_nb = ndimage.label(a)\n    >>> labels\n    array([[1, 1, 0, 0],\n           [1, 1, 0, 2],\n           [0, 0, 0, 2],\n           [3, 3, 0, 0]])\n    >>> ndimage.minimum(a, labels=labels, index=np.arange(1, labels_nb + 1))\n    [1.0, 4.0, 3.0]\n    >>> ndimage.minimum(a)\n    0.0\n    >>> ndimage.minimum(a, labels=labels)\n    1.0\n\n    ')
    
    # Obtaining the type of the subscript
    int_123431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 56), 'int')
    
    # Call to _select(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'input' (line 915)
    input_123433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 19), 'input', False)
    # Getting the type of 'labels' (line 915)
    labels_123434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 26), 'labels', False)
    # Getting the type of 'index' (line 915)
    index_123435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 34), 'index', False)
    # Processing the call keyword arguments (line 915)
    # Getting the type of 'True' (line 915)
    True_123436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'True', False)
    keyword_123437 = True_123436
    kwargs_123438 = {'find_min': keyword_123437}
    # Getting the type of '_select' (line 915)
    _select_123432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 11), '_select', False)
    # Calling _select(args, kwargs) (line 915)
    _select_call_result_123439 = invoke(stypy.reporting.localization.Localization(__file__, 915, 11), _select_123432, *[input_123433, labels_123434, index_123435], **kwargs_123438)
    
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___123440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 11), _select_call_result_123439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_123441 = invoke(stypy.reporting.localization.Localization(__file__, 915, 11), getitem___123440, int_123431)
    
    # Assigning a type to the variable 'stypy_return_type' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'stypy_return_type', subscript_call_result_123441)
    
    # ################# End of 'minimum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimum' in the type store
    # Getting the type of 'stypy_return_type' (line 856)
    stypy_return_type_123442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimum'
    return stypy_return_type_123442

# Assigning a type to the variable 'minimum' (line 856)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 0), 'minimum', minimum)

@norecursion
def maximum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 918)
    None_123443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 26), 'None')
    # Getting the type of 'None' (line 918)
    None_123444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 38), 'None')
    defaults = [None_123443, None_123444]
    # Create a new context for function 'maximum'
    module_type_store = module_type_store.open_function_context('maximum', 918, 0, False)
    
    # Passed parameters checking function
    maximum.stypy_localization = localization
    maximum.stypy_type_of_self = None
    maximum.stypy_type_store = module_type_store
    maximum.stypy_function_name = 'maximum'
    maximum.stypy_param_names_list = ['input', 'labels', 'index']
    maximum.stypy_varargs_param_name = None
    maximum.stypy_kwargs_param_name = None
    maximum.stypy_call_defaults = defaults
    maximum.stypy_call_varargs = varargs
    maximum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'maximum', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'maximum', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'maximum(...)' code ##################

    str_123445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, (-1)), 'str', '\n    Calculate the maximum of the values of an array over labeled regions.\n\n    Parameters\n    ----------\n    input : array_like\n        Array_like of values. For each region specified by `labels`, the\n        maximal values of `input` over the region is computed.\n    labels : array_like, optional\n        An array of integers marking different regions over which the\n        maximum value of `input` is to be computed. `labels` must have the\n        same shape as `input`. If `labels` is not specified, the maximum\n        over the whole array is returned.\n    index : array_like, optional\n        A list of region labels that are taken into account for computing the\n        maxima. If index is None, the maximum over all elements where `labels`\n        is non-zero is returned.\n\n    Returns\n    -------\n    output : float or list of floats\n        List of maxima of `input` over the regions determined by `labels` and\n        whose index is in `index`. If `index` or `labels` are not specified, a\n        float is returned: the maximal value of `input` if `labels` is None,\n        and the maximal value of elements where `labels` is greater than zero\n        if `index` is None.\n\n    See also\n    --------\n    label, minimum, median, maximum_position, extrema, sum, mean, variance,\n    standard_deviation\n\n    Notes\n    -----\n    The function returns a Python list and not a Numpy array, use\n    `np.array` to convert the list to an array.\n\n    Examples\n    --------\n    >>> a = np.arange(16).reshape((4,4))\n    >>> a\n    array([[ 0,  1,  2,  3],\n           [ 4,  5,  6,  7],\n           [ 8,  9, 10, 11],\n           [12, 13, 14, 15]])\n    >>> labels = np.zeros_like(a)\n    >>> labels[:2,:2] = 1\n    >>> labels[2:, 1:3] = 2\n    >>> labels\n    array([[1, 1, 0, 0],\n           [1, 1, 0, 0],\n           [0, 2, 2, 0],\n           [0, 2, 2, 0]])\n    >>> from scipy import ndimage\n    >>> ndimage.maximum(a)\n    15.0\n    >>> ndimage.maximum(a, labels=labels, index=[1,2])\n    [5.0, 14.0]\n    >>> ndimage.maximum(a, labels=labels)\n    14.0\n\n    >>> b = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> labels, labels_nb = ndimage.label(b)\n    >>> labels\n    array([[1, 1, 0, 0],\n           [1, 1, 0, 2],\n           [0, 0, 0, 2],\n           [3, 3, 0, 0]])\n    >>> ndimage.maximum(b, labels=labels, index=np.arange(1, labels_nb + 1))\n    [5.0, 7.0, 9.0]\n\n    ')
    
    # Obtaining the type of the subscript
    int_123446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 994, 56), 'int')
    
    # Call to _select(...): (line 994)
    # Processing the call arguments (line 994)
    # Getting the type of 'input' (line 994)
    input_123448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 19), 'input', False)
    # Getting the type of 'labels' (line 994)
    labels_123449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 26), 'labels', False)
    # Getting the type of 'index' (line 994)
    index_123450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 34), 'index', False)
    # Processing the call keyword arguments (line 994)
    # Getting the type of 'True' (line 994)
    True_123451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 50), 'True', False)
    keyword_123452 = True_123451
    kwargs_123453 = {'find_max': keyword_123452}
    # Getting the type of '_select' (line 994)
    _select_123447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 11), '_select', False)
    # Calling _select(args, kwargs) (line 994)
    _select_call_result_123454 = invoke(stypy.reporting.localization.Localization(__file__, 994, 11), _select_123447, *[input_123448, labels_123449, index_123450], **kwargs_123453)
    
    # Obtaining the member '__getitem__' of a type (line 994)
    getitem___123455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 994, 11), _select_call_result_123454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 994)
    subscript_call_result_123456 = invoke(stypy.reporting.localization.Localization(__file__, 994, 11), getitem___123455, int_123446)
    
    # Assigning a type to the variable 'stypy_return_type' (line 994)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 4), 'stypy_return_type', subscript_call_result_123456)
    
    # ################# End of 'maximum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'maximum' in the type store
    # Getting the type of 'stypy_return_type' (line 918)
    stypy_return_type_123457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123457)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'maximum'
    return stypy_return_type_123457

# Assigning a type to the variable 'maximum' (line 918)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 0), 'maximum', maximum)

@norecursion
def median(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 997)
    None_123458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 25), 'None')
    # Getting the type of 'None' (line 997)
    None_123459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 37), 'None')
    defaults = [None_123458, None_123459]
    # Create a new context for function 'median'
    module_type_store = module_type_store.open_function_context('median', 997, 0, False)
    
    # Passed parameters checking function
    median.stypy_localization = localization
    median.stypy_type_of_self = None
    median.stypy_type_store = module_type_store
    median.stypy_function_name = 'median'
    median.stypy_param_names_list = ['input', 'labels', 'index']
    median.stypy_varargs_param_name = None
    median.stypy_kwargs_param_name = None
    median.stypy_call_defaults = defaults
    median.stypy_call_varargs = varargs
    median.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'median', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'median', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'median(...)' code ##################

    str_123460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1054, (-1)), 'str', '\n    Calculate the median of the values of an array over labeled regions.\n\n    Parameters\n    ----------\n    input : array_like\n        Array_like of values. For each region specified by `labels`, the\n        median value of `input` over the region is computed.\n    labels : array_like, optional\n        An array_like of integers marking different regions over which the\n        median value of `input` is to be computed. `labels` must have the\n        same shape as `input`. If `labels` is not specified, the median\n        over the whole array is returned.\n    index : array_like, optional\n        A list of region labels that are taken into account for computing the\n        medians. If index is None, the median over all elements where `labels`\n        is non-zero is returned.\n\n    Returns\n    -------\n    median : float or list of floats\n        List of medians of `input` over the regions determined by `labels` and\n        whose index is in `index`. If `index` or `labels` are not specified, a\n        float is returned: the median value of `input` if `labels` is None,\n        and the median value of elements where `labels` is greater than zero\n        if `index` is None.\n\n    See also\n    --------\n    label, minimum, maximum, extrema, sum, mean, variance, standard_deviation\n\n    Notes\n    -----\n    The function returns a Python list and not a Numpy array, use\n    `np.array` to convert the list to an array.\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.array([[1, 2, 0, 1],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> labels, labels_nb = ndimage.label(a)\n    >>> labels\n    array([[1, 1, 0, 2],\n           [1, 1, 0, 2],\n           [0, 0, 0, 2],\n           [3, 3, 0, 0]])\n    >>> ndimage.median(a, labels=labels, index=np.arange(1, labels_nb + 1))\n    [2.5, 4.0, 6.0]\n    >>> ndimage.median(a)\n    1.0\n    >>> ndimage.median(a, labels=labels)\n    3.0\n\n    ')
    
    # Obtaining the type of the subscript
    int_123461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1055, 59), 'int')
    
    # Call to _select(...): (line 1055)
    # Processing the call arguments (line 1055)
    # Getting the type of 'input' (line 1055)
    input_123463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 19), 'input', False)
    # Getting the type of 'labels' (line 1055)
    labels_123464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 26), 'labels', False)
    # Getting the type of 'index' (line 1055)
    index_123465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 34), 'index', False)
    # Processing the call keyword arguments (line 1055)
    # Getting the type of 'True' (line 1055)
    True_123466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 53), 'True', False)
    keyword_123467 = True_123466
    kwargs_123468 = {'find_median': keyword_123467}
    # Getting the type of '_select' (line 1055)
    _select_123462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 11), '_select', False)
    # Calling _select(args, kwargs) (line 1055)
    _select_call_result_123469 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), _select_123462, *[input_123463, labels_123464, index_123465], **kwargs_123468)
    
    # Obtaining the member '__getitem__' of a type (line 1055)
    getitem___123470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 11), _select_call_result_123469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1055)
    subscript_call_result_123471 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 11), getitem___123470, int_123461)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'stypy_return_type', subscript_call_result_123471)
    
    # ################# End of 'median(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'median' in the type store
    # Getting the type of 'stypy_return_type' (line 997)
    stypy_return_type_123472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'median'
    return stypy_return_type_123472

# Assigning a type to the variable 'median' (line 997)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 0), 'median', median)

@norecursion
def minimum_position(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1058)
    None_123473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 35), 'None')
    # Getting the type of 'None' (line 1058)
    None_123474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 47), 'None')
    defaults = [None_123473, None_123474]
    # Create a new context for function 'minimum_position'
    module_type_store = module_type_store.open_function_context('minimum_position', 1058, 0, False)
    
    # Passed parameters checking function
    minimum_position.stypy_localization = localization
    minimum_position.stypy_type_of_self = None
    minimum_position.stypy_type_store = module_type_store
    minimum_position.stypy_function_name = 'minimum_position'
    minimum_position.stypy_param_names_list = ['input', 'labels', 'index']
    minimum_position.stypy_varargs_param_name = None
    minimum_position.stypy_kwargs_param_name = None
    minimum_position.stypy_call_defaults = defaults
    minimum_position.stypy_call_varargs = varargs
    minimum_position.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimum_position', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimum_position', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimum_position(...)' code ##################

    str_123475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, (-1)), 'str', '\n    Find the positions of the minimums of the values of an array at labels.\n\n    Parameters\n    ----------\n    input : array_like\n        Array_like of values.\n    labels : array_like, optional\n        An array of integers marking different regions over which the\n        position of the minimum value of `input` is to be computed.\n        `labels` must have the same shape as `input`. If `labels` is not\n        specified, the location of the first minimum over the whole\n        array is returned.\n\n        The `labels` argument only works when `index` is specified.\n    index : array_like, optional\n        A list of region labels that are taken into account for finding the\n        location of the minima. If `index` is None, the ``first`` minimum\n        over all elements where `labels` is non-zero is returned.\n\n        The `index` argument only works when `labels` is specified.\n\n    Returns\n    -------\n    output : list of tuples of ints\n        Tuple of ints or list of tuples of ints that specify the location\n        of minima of `input` over the regions determined by `labels` and\n        whose index is in `index`.\n\n        If `index` or `labels` are not specified, a tuple of ints is\n        returned specifying the location of the first minimal value of `input`.\n\n    See also\n    --------\n    label, minimum, median, maximum_position, extrema, sum, mean, variance,\n    standard_deviation\n\n    ')
    
    # Assigning a Call to a Name (line 1097):
    
    # Assigning a Call to a Name (line 1097):
    
    # Call to array(...): (line 1097)
    # Processing the call arguments (line 1097)
    
    # Call to asarray(...): (line 1097)
    # Processing the call arguments (line 1097)
    # Getting the type of 'input' (line 1097)
    input_123480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 37), 'input', False)
    # Processing the call keyword arguments (line 1097)
    kwargs_123481 = {}
    # Getting the type of 'numpy' (line 1097)
    numpy_123478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 23), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1097)
    asarray_123479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 23), numpy_123478, 'asarray')
    # Calling asarray(args, kwargs) (line 1097)
    asarray_call_result_123482 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 23), asarray_123479, *[input_123480], **kwargs_123481)
    
    # Obtaining the member 'shape' of a type (line 1097)
    shape_123483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 23), asarray_call_result_123482, 'shape')
    # Processing the call keyword arguments (line 1097)
    kwargs_123484 = {}
    # Getting the type of 'numpy' (line 1097)
    numpy_123476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 11), 'numpy', False)
    # Obtaining the member 'array' of a type (line 1097)
    array_123477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 11), numpy_123476, 'array')
    # Calling array(args, kwargs) (line 1097)
    array_call_result_123485 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 11), array_123477, *[shape_123483], **kwargs_123484)
    
    # Assigning a type to the variable 'dims' (line 1097)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 4), 'dims', array_call_result_123485)
    
    # Assigning a Subscript to a Name (line 1099):
    
    # Assigning a Subscript to a Name (line 1099):
    
    # Obtaining the type of the subscript
    int_123486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 56), 'int')
    slice_123487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1099, 15), None, None, int_123486)
    
    # Call to cumprod(...): (line 1099)
    # Processing the call arguments (line 1099)
    
    # Obtaining an instance of the builtin type 'list' (line 1099)
    list_123490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1099)
    # Adding element type (line 1099)
    int_123491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1099, 29), list_123490, int_123491)
    
    
    # Call to list(...): (line 1099)
    # Processing the call arguments (line 1099)
    
    # Obtaining the type of the subscript
    int_123493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 46), 'int')
    int_123494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 48), 'int')
    slice_123495 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1099, 40), None, int_123493, int_123494)
    # Getting the type of 'dims' (line 1099)
    dims_123496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 40), 'dims', False)
    # Obtaining the member '__getitem__' of a type (line 1099)
    getitem___123497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 40), dims_123496, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1099)
    subscript_call_result_123498 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 40), getitem___123497, slice_123495)
    
    # Processing the call keyword arguments (line 1099)
    kwargs_123499 = {}
    # Getting the type of 'list' (line 1099)
    list_123492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 35), 'list', False)
    # Calling list(args, kwargs) (line 1099)
    list_call_result_123500 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 35), list_123492, *[subscript_call_result_123498], **kwargs_123499)
    
    # Applying the binary operator '+' (line 1099)
    result_add_123501 = python_operator(stypy.reporting.localization.Localization(__file__, 1099, 29), '+', list_123490, list_call_result_123500)
    
    # Processing the call keyword arguments (line 1099)
    kwargs_123502 = {}
    # Getting the type of 'numpy' (line 1099)
    numpy_123488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 15), 'numpy', False)
    # Obtaining the member 'cumprod' of a type (line 1099)
    cumprod_123489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 15), numpy_123488, 'cumprod')
    # Calling cumprod(args, kwargs) (line 1099)
    cumprod_call_result_123503 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 15), cumprod_123489, *[result_add_123501], **kwargs_123502)
    
    # Obtaining the member '__getitem__' of a type (line 1099)
    getitem___123504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 15), cumprod_call_result_123503, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1099)
    subscript_call_result_123505 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 15), getitem___123504, slice_123487)
    
    # Assigning a type to the variable 'dim_prod' (line 1099)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 4), 'dim_prod', subscript_call_result_123505)
    
    # Assigning a Subscript to a Name (line 1101):
    
    # Assigning a Subscript to a Name (line 1101):
    
    # Obtaining the type of the subscript
    int_123506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, 68), 'int')
    
    # Call to _select(...): (line 1101)
    # Processing the call arguments (line 1101)
    # Getting the type of 'input' (line 1101)
    input_123508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 21), 'input', False)
    # Getting the type of 'labels' (line 1101)
    labels_123509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 28), 'labels', False)
    # Getting the type of 'index' (line 1101)
    index_123510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 36), 'index', False)
    # Processing the call keyword arguments (line 1101)
    # Getting the type of 'True' (line 1101)
    True_123511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 62), 'True', False)
    keyword_123512 = True_123511
    kwargs_123513 = {'find_min_positions': keyword_123512}
    # Getting the type of '_select' (line 1101)
    _select_123507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 13), '_select', False)
    # Calling _select(args, kwargs) (line 1101)
    _select_call_result_123514 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 13), _select_123507, *[input_123508, labels_123509, index_123510], **kwargs_123513)
    
    # Obtaining the member '__getitem__' of a type (line 1101)
    getitem___123515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 13), _select_call_result_123514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1101)
    subscript_call_result_123516 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 13), getitem___123515, int_123506)
    
    # Assigning a type to the variable 'result' (line 1101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 4), 'result', subscript_call_result_123516)
    
    
    # Call to isscalar(...): (line 1103)
    # Processing the call arguments (line 1103)
    # Getting the type of 'result' (line 1103)
    result_123519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 22), 'result', False)
    # Processing the call keyword arguments (line 1103)
    kwargs_123520 = {}
    # Getting the type of 'numpy' (line 1103)
    numpy_123517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 1103)
    isscalar_123518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1103, 7), numpy_123517, 'isscalar')
    # Calling isscalar(args, kwargs) (line 1103)
    isscalar_call_result_123521 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 7), isscalar_123518, *[result_123519], **kwargs_123520)
    
    # Testing the type of an if condition (line 1103)
    if_condition_123522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1103, 4), isscalar_call_result_123521)
    # Assigning a type to the variable 'if_condition_123522' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'if_condition_123522', if_condition_123522)
    # SSA begins for if statement (line 1103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 1104)
    # Processing the call arguments (line 1104)
    # Getting the type of 'result' (line 1104)
    result_123524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 22), 'result', False)
    # Getting the type of 'dim_prod' (line 1104)
    dim_prod_123525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 32), 'dim_prod', False)
    # Applying the binary operator '//' (line 1104)
    result_floordiv_123526 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 22), '//', result_123524, dim_prod_123525)
    
    # Getting the type of 'dims' (line 1104)
    dims_123527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 44), 'dims', False)
    # Applying the binary operator '%' (line 1104)
    result_mod_123528 = python_operator(stypy.reporting.localization.Localization(__file__, 1104, 21), '%', result_floordiv_123526, dims_123527)
    
    # Processing the call keyword arguments (line 1104)
    kwargs_123529 = {}
    # Getting the type of 'tuple' (line 1104)
    tuple_123523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1104)
    tuple_call_result_123530 = invoke(stypy.reporting.localization.Localization(__file__, 1104, 15), tuple_123523, *[result_mod_123528], **kwargs_123529)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 8), 'stypy_return_type', tuple_call_result_123530)
    # SSA join for if statement (line 1103)
    module_type_store = module_type_store.join_ssa_context()
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reshape(...): (line 1106)
    # Processing the call arguments (line 1106)
    int_123537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 46), 'int')
    int_123538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 50), 'int')
    # Processing the call keyword arguments (line 1106)
    kwargs_123539 = {}
    # Getting the type of 'result' (line 1106)
    result_123535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 31), 'result', False)
    # Obtaining the member 'reshape' of a type (line 1106)
    reshape_123536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 31), result_123535, 'reshape')
    # Calling reshape(args, kwargs) (line 1106)
    reshape_call_result_123540 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 31), reshape_123536, *[int_123537, int_123538], **kwargs_123539)
    
    # Getting the type of 'dim_prod' (line 1106)
    dim_prod_123541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 56), 'dim_prod')
    # Applying the binary operator '//' (line 1106)
    result_floordiv_123542 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 31), '//', reshape_call_result_123540, dim_prod_123541)
    
    # Getting the type of 'dims' (line 1106)
    dims_123543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 68), 'dims')
    # Applying the binary operator '%' (line 1106)
    result_mod_123544 = python_operator(stypy.reporting.localization.Localization(__file__, 1106, 30), '%', result_floordiv_123542, dims_123543)
    
    comprehension_123545 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 12), result_mod_123544)
    # Assigning a type to the variable 'v' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 12), 'v', comprehension_123545)
    
    # Call to tuple(...): (line 1106)
    # Processing the call arguments (line 1106)
    # Getting the type of 'v' (line 1106)
    v_123532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 18), 'v', False)
    # Processing the call keyword arguments (line 1106)
    kwargs_123533 = {}
    # Getting the type of 'tuple' (line 1106)
    tuple_123531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1106)
    tuple_call_result_123534 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 12), tuple_123531, *[v_123532], **kwargs_123533)
    
    list_123546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 12), list_123546, tuple_call_result_123534)
    # Assigning a type to the variable 'stypy_return_type' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 4), 'stypy_return_type', list_123546)
    
    # ################# End of 'minimum_position(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimum_position' in the type store
    # Getting the type of 'stypy_return_type' (line 1058)
    stypy_return_type_123547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123547)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimum_position'
    return stypy_return_type_123547

# Assigning a type to the variable 'minimum_position' (line 1058)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 0), 'minimum_position', minimum_position)

@norecursion
def maximum_position(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1109)
    None_123548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 35), 'None')
    # Getting the type of 'None' (line 1109)
    None_123549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 47), 'None')
    defaults = [None_123548, None_123549]
    # Create a new context for function 'maximum_position'
    module_type_store = module_type_store.open_function_context('maximum_position', 1109, 0, False)
    
    # Passed parameters checking function
    maximum_position.stypy_localization = localization
    maximum_position.stypy_type_of_self = None
    maximum_position.stypy_type_store = module_type_store
    maximum_position.stypy_function_name = 'maximum_position'
    maximum_position.stypy_param_names_list = ['input', 'labels', 'index']
    maximum_position.stypy_varargs_param_name = None
    maximum_position.stypy_kwargs_param_name = None
    maximum_position.stypy_call_defaults = defaults
    maximum_position.stypy_call_varargs = varargs
    maximum_position.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'maximum_position', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'maximum_position', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'maximum_position(...)' code ##################

    str_123550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, (-1)), 'str', '\n    Find the positions of the maximums of the values of an array at labels.\n\n    For each region specified by `labels`, the position of the maximum\n    value of `input` within the region is returned.\n\n    Parameters\n    ----------\n    input : array_like\n        Array_like of values.\n    labels : array_like, optional\n        An array of integers marking different regions over which the\n        position of the maximum value of `input` is to be computed.\n        `labels` must have the same shape as `input`. If `labels` is not\n        specified, the location of the first maximum over the whole\n        array is returned.\n\n        The `labels` argument only works when `index` is specified.\n    index : array_like, optional\n        A list of region labels that are taken into account for finding the\n        location of the maxima.  If `index` is None, the first maximum\n        over all elements where `labels` is non-zero is returned.\n\n        The `index` argument only works when `labels` is specified.\n\n    Returns\n    -------\n    output : list of tuples of ints\n        List of tuples of ints that specify the location of maxima of\n        `input` over the regions determined by `labels` and whose index\n        is in `index`.\n\n        If `index` or `labels` are not specified, a tuple of ints is\n        returned specifying the location of the ``first`` maximal value\n        of `input`.\n\n    See also\n    --------\n    label, minimum, median, maximum_position, extrema, sum, mean, variance,\n    standard_deviation\n\n    ')
    
    # Assigning a Call to a Name (line 1152):
    
    # Assigning a Call to a Name (line 1152):
    
    # Call to array(...): (line 1152)
    # Processing the call arguments (line 1152)
    
    # Call to asarray(...): (line 1152)
    # Processing the call arguments (line 1152)
    # Getting the type of 'input' (line 1152)
    input_123555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 37), 'input', False)
    # Processing the call keyword arguments (line 1152)
    kwargs_123556 = {}
    # Getting the type of 'numpy' (line 1152)
    numpy_123553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 23), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1152)
    asarray_123554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 23), numpy_123553, 'asarray')
    # Calling asarray(args, kwargs) (line 1152)
    asarray_call_result_123557 = invoke(stypy.reporting.localization.Localization(__file__, 1152, 23), asarray_123554, *[input_123555], **kwargs_123556)
    
    # Obtaining the member 'shape' of a type (line 1152)
    shape_123558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 23), asarray_call_result_123557, 'shape')
    # Processing the call keyword arguments (line 1152)
    kwargs_123559 = {}
    # Getting the type of 'numpy' (line 1152)
    numpy_123551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 11), 'numpy', False)
    # Obtaining the member 'array' of a type (line 1152)
    array_123552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1152, 11), numpy_123551, 'array')
    # Calling array(args, kwargs) (line 1152)
    array_call_result_123560 = invoke(stypy.reporting.localization.Localization(__file__, 1152, 11), array_123552, *[shape_123558], **kwargs_123559)
    
    # Assigning a type to the variable 'dims' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 4), 'dims', array_call_result_123560)
    
    # Assigning a Subscript to a Name (line 1154):
    
    # Assigning a Subscript to a Name (line 1154):
    
    # Obtaining the type of the subscript
    int_123561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 56), 'int')
    slice_123562 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1154, 15), None, None, int_123561)
    
    # Call to cumprod(...): (line 1154)
    # Processing the call arguments (line 1154)
    
    # Obtaining an instance of the builtin type 'list' (line 1154)
    list_123565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1154)
    # Adding element type (line 1154)
    int_123566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1154, 29), list_123565, int_123566)
    
    
    # Call to list(...): (line 1154)
    # Processing the call arguments (line 1154)
    
    # Obtaining the type of the subscript
    int_123568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 46), 'int')
    int_123569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, 48), 'int')
    slice_123570 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1154, 40), None, int_123568, int_123569)
    # Getting the type of 'dims' (line 1154)
    dims_123571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 40), 'dims', False)
    # Obtaining the member '__getitem__' of a type (line 1154)
    getitem___123572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 40), dims_123571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1154)
    subscript_call_result_123573 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 40), getitem___123572, slice_123570)
    
    # Processing the call keyword arguments (line 1154)
    kwargs_123574 = {}
    # Getting the type of 'list' (line 1154)
    list_123567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 35), 'list', False)
    # Calling list(args, kwargs) (line 1154)
    list_call_result_123575 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 35), list_123567, *[subscript_call_result_123573], **kwargs_123574)
    
    # Applying the binary operator '+' (line 1154)
    result_add_123576 = python_operator(stypy.reporting.localization.Localization(__file__, 1154, 29), '+', list_123565, list_call_result_123575)
    
    # Processing the call keyword arguments (line 1154)
    kwargs_123577 = {}
    # Getting the type of 'numpy' (line 1154)
    numpy_123563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 15), 'numpy', False)
    # Obtaining the member 'cumprod' of a type (line 1154)
    cumprod_123564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 15), numpy_123563, 'cumprod')
    # Calling cumprod(args, kwargs) (line 1154)
    cumprod_call_result_123578 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 15), cumprod_123564, *[result_add_123576], **kwargs_123577)
    
    # Obtaining the member '__getitem__' of a type (line 1154)
    getitem___123579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 15), cumprod_call_result_123578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1154)
    subscript_call_result_123580 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 15), getitem___123579, slice_123562)
    
    # Assigning a type to the variable 'dim_prod' (line 1154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1154, 4), 'dim_prod', subscript_call_result_123580)
    
    # Assigning a Subscript to a Name (line 1156):
    
    # Assigning a Subscript to a Name (line 1156):
    
    # Obtaining the type of the subscript
    int_123581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1156, 68), 'int')
    
    # Call to _select(...): (line 1156)
    # Processing the call arguments (line 1156)
    # Getting the type of 'input' (line 1156)
    input_123583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 21), 'input', False)
    # Getting the type of 'labels' (line 1156)
    labels_123584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 28), 'labels', False)
    # Getting the type of 'index' (line 1156)
    index_123585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 36), 'index', False)
    # Processing the call keyword arguments (line 1156)
    # Getting the type of 'True' (line 1156)
    True_123586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 62), 'True', False)
    keyword_123587 = True_123586
    kwargs_123588 = {'find_max_positions': keyword_123587}
    # Getting the type of '_select' (line 1156)
    _select_123582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 13), '_select', False)
    # Calling _select(args, kwargs) (line 1156)
    _select_call_result_123589 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 13), _select_123582, *[input_123583, labels_123584, index_123585], **kwargs_123588)
    
    # Obtaining the member '__getitem__' of a type (line 1156)
    getitem___123590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1156, 13), _select_call_result_123589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1156)
    subscript_call_result_123591 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 13), getitem___123590, int_123581)
    
    # Assigning a type to the variable 'result' (line 1156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'result', subscript_call_result_123591)
    
    
    # Call to isscalar(...): (line 1158)
    # Processing the call arguments (line 1158)
    # Getting the type of 'result' (line 1158)
    result_123594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 22), 'result', False)
    # Processing the call keyword arguments (line 1158)
    kwargs_123595 = {}
    # Getting the type of 'numpy' (line 1158)
    numpy_123592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 1158)
    isscalar_123593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1158, 7), numpy_123592, 'isscalar')
    # Calling isscalar(args, kwargs) (line 1158)
    isscalar_call_result_123596 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 7), isscalar_123593, *[result_123594], **kwargs_123595)
    
    # Testing the type of an if condition (line 1158)
    if_condition_123597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1158, 4), isscalar_call_result_123596)
    # Assigning a type to the variable 'if_condition_123597' (line 1158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1158, 4), 'if_condition_123597', if_condition_123597)
    # SSA begins for if statement (line 1158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 1159)
    # Processing the call arguments (line 1159)
    # Getting the type of 'result' (line 1159)
    result_123599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 22), 'result', False)
    # Getting the type of 'dim_prod' (line 1159)
    dim_prod_123600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 32), 'dim_prod', False)
    # Applying the binary operator '//' (line 1159)
    result_floordiv_123601 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 22), '//', result_123599, dim_prod_123600)
    
    # Getting the type of 'dims' (line 1159)
    dims_123602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 44), 'dims', False)
    # Applying the binary operator '%' (line 1159)
    result_mod_123603 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 21), '%', result_floordiv_123601, dims_123602)
    
    # Processing the call keyword arguments (line 1159)
    kwargs_123604 = {}
    # Getting the type of 'tuple' (line 1159)
    tuple_123598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1159)
    tuple_call_result_123605 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 15), tuple_123598, *[result_mod_123603], **kwargs_123604)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 8), 'stypy_return_type', tuple_call_result_123605)
    # SSA join for if statement (line 1158)
    module_type_store = module_type_store.join_ssa_context()
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reshape(...): (line 1161)
    # Processing the call arguments (line 1161)
    int_123612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 46), 'int')
    int_123613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 50), 'int')
    # Processing the call keyword arguments (line 1161)
    kwargs_123614 = {}
    # Getting the type of 'result' (line 1161)
    result_123610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 31), 'result', False)
    # Obtaining the member 'reshape' of a type (line 1161)
    reshape_123611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 31), result_123610, 'reshape')
    # Calling reshape(args, kwargs) (line 1161)
    reshape_call_result_123615 = invoke(stypy.reporting.localization.Localization(__file__, 1161, 31), reshape_123611, *[int_123612, int_123613], **kwargs_123614)
    
    # Getting the type of 'dim_prod' (line 1161)
    dim_prod_123616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 56), 'dim_prod')
    # Applying the binary operator '//' (line 1161)
    result_floordiv_123617 = python_operator(stypy.reporting.localization.Localization(__file__, 1161, 31), '//', reshape_call_result_123615, dim_prod_123616)
    
    # Getting the type of 'dims' (line 1161)
    dims_123618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 68), 'dims')
    # Applying the binary operator '%' (line 1161)
    result_mod_123619 = python_operator(stypy.reporting.localization.Localization(__file__, 1161, 30), '%', result_floordiv_123617, dims_123618)
    
    comprehension_123620 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1161, 12), result_mod_123619)
    # Assigning a type to the variable 'v' (line 1161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 12), 'v', comprehension_123620)
    
    # Call to tuple(...): (line 1161)
    # Processing the call arguments (line 1161)
    # Getting the type of 'v' (line 1161)
    v_123607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 18), 'v', False)
    # Processing the call keyword arguments (line 1161)
    kwargs_123608 = {}
    # Getting the type of 'tuple' (line 1161)
    tuple_123606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1161)
    tuple_call_result_123609 = invoke(stypy.reporting.localization.Localization(__file__, 1161, 12), tuple_123606, *[v_123607], **kwargs_123608)
    
    list_123621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1161, 12), list_123621, tuple_call_result_123609)
    # Assigning a type to the variable 'stypy_return_type' (line 1161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 4), 'stypy_return_type', list_123621)
    
    # ################# End of 'maximum_position(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'maximum_position' in the type store
    # Getting the type of 'stypy_return_type' (line 1109)
    stypy_return_type_123622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'maximum_position'
    return stypy_return_type_123622

# Assigning a type to the variable 'maximum_position' (line 1109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 0), 'maximum_position', maximum_position)

@norecursion
def extrema(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1164)
    None_123623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 26), 'None')
    # Getting the type of 'None' (line 1164)
    None_123624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 38), 'None')
    defaults = [None_123623, None_123624]
    # Create a new context for function 'extrema'
    module_type_store = module_type_store.open_function_context('extrema', 1164, 0, False)
    
    # Passed parameters checking function
    extrema.stypy_localization = localization
    extrema.stypy_type_of_self = None
    extrema.stypy_type_store = module_type_store
    extrema.stypy_function_name = 'extrema'
    extrema.stypy_param_names_list = ['input', 'labels', 'index']
    extrema.stypy_varargs_param_name = None
    extrema.stypy_kwargs_param_name = None
    extrema.stypy_call_defaults = defaults
    extrema.stypy_call_varargs = varargs
    extrema.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'extrema', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'extrema', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'extrema(...)' code ##################

    str_123625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, (-1)), 'str', '\n    Calculate the minimums and maximums of the values of an array\n    at labels, along with their positions.\n\n    Parameters\n    ----------\n    input : ndarray\n        Nd-image data to process.\n    labels : ndarray, optional\n        Labels of features in input.\n        If not None, must be same shape as `input`.\n    index : int or sequence of ints, optional\n        Labels to include in output.  If None (default), all values where\n        non-zero `labels` are used.\n\n    Returns\n    -------\n    minimums, maximums : int or ndarray\n        Values of minimums and maximums in each feature.\n    min_positions, max_positions : tuple or list of tuples\n        Each tuple gives the n-D coordinates of the corresponding minimum\n        or maximum.\n\n    See Also\n    --------\n    maximum, minimum, maximum_position, minimum_position, center_of_mass\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> from scipy import ndimage\n    >>> ndimage.extrema(a)\n    (0, 9, (0, 2), (3, 0))\n\n    Features to process can be specified using `labels` and `index`:\n\n    >>> lbl, nlbl = ndimage.label(a)\n    >>> ndimage.extrema(a, lbl, index=np.arange(1, nlbl+1))\n    (array([1, 4, 3]),\n     array([5, 7, 9]),\n     [(0, 0), (1, 3), (3, 1)],\n     [(1, 0), (2, 3), (3, 0)])\n\n    If no index is given, non-zero `labels` are processed:\n\n    >>> ndimage.extrema(a, lbl)\n    (1, 9, (0, 0), (3, 0))\n\n    ')
    
    # Assigning a Call to a Name (line 1217):
    
    # Assigning a Call to a Name (line 1217):
    
    # Call to array(...): (line 1217)
    # Processing the call arguments (line 1217)
    
    # Call to asarray(...): (line 1217)
    # Processing the call arguments (line 1217)
    # Getting the type of 'input' (line 1217)
    input_123630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 37), 'input', False)
    # Processing the call keyword arguments (line 1217)
    kwargs_123631 = {}
    # Getting the type of 'numpy' (line 1217)
    numpy_123628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 23), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1217)
    asarray_123629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 23), numpy_123628, 'asarray')
    # Calling asarray(args, kwargs) (line 1217)
    asarray_call_result_123632 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 23), asarray_123629, *[input_123630], **kwargs_123631)
    
    # Obtaining the member 'shape' of a type (line 1217)
    shape_123633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 23), asarray_call_result_123632, 'shape')
    # Processing the call keyword arguments (line 1217)
    kwargs_123634 = {}
    # Getting the type of 'numpy' (line 1217)
    numpy_123626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 11), 'numpy', False)
    # Obtaining the member 'array' of a type (line 1217)
    array_123627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 11), numpy_123626, 'array')
    # Calling array(args, kwargs) (line 1217)
    array_call_result_123635 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 11), array_123627, *[shape_123633], **kwargs_123634)
    
    # Assigning a type to the variable 'dims' (line 1217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 4), 'dims', array_call_result_123635)
    
    # Assigning a Subscript to a Name (line 1219):
    
    # Assigning a Subscript to a Name (line 1219):
    
    # Obtaining the type of the subscript
    int_123636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 56), 'int')
    slice_123637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1219, 15), None, None, int_123636)
    
    # Call to cumprod(...): (line 1219)
    # Processing the call arguments (line 1219)
    
    # Obtaining an instance of the builtin type 'list' (line 1219)
    list_123640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1219)
    # Adding element type (line 1219)
    int_123641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1219, 29), list_123640, int_123641)
    
    
    # Call to list(...): (line 1219)
    # Processing the call arguments (line 1219)
    
    # Obtaining the type of the subscript
    int_123643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 46), 'int')
    int_123644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 48), 'int')
    slice_123645 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1219, 40), None, int_123643, int_123644)
    # Getting the type of 'dims' (line 1219)
    dims_123646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 40), 'dims', False)
    # Obtaining the member '__getitem__' of a type (line 1219)
    getitem___123647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 40), dims_123646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1219)
    subscript_call_result_123648 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 40), getitem___123647, slice_123645)
    
    # Processing the call keyword arguments (line 1219)
    kwargs_123649 = {}
    # Getting the type of 'list' (line 1219)
    list_123642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 35), 'list', False)
    # Calling list(args, kwargs) (line 1219)
    list_call_result_123650 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 35), list_123642, *[subscript_call_result_123648], **kwargs_123649)
    
    # Applying the binary operator '+' (line 1219)
    result_add_123651 = python_operator(stypy.reporting.localization.Localization(__file__, 1219, 29), '+', list_123640, list_call_result_123650)
    
    # Processing the call keyword arguments (line 1219)
    kwargs_123652 = {}
    # Getting the type of 'numpy' (line 1219)
    numpy_123638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 15), 'numpy', False)
    # Obtaining the member 'cumprod' of a type (line 1219)
    cumprod_123639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 15), numpy_123638, 'cumprod')
    # Calling cumprod(args, kwargs) (line 1219)
    cumprod_call_result_123653 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 15), cumprod_123639, *[result_add_123651], **kwargs_123652)
    
    # Obtaining the member '__getitem__' of a type (line 1219)
    getitem___123654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1219, 15), cumprod_call_result_123653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1219)
    subscript_call_result_123655 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 15), getitem___123654, slice_123637)
    
    # Assigning a type to the variable 'dim_prod' (line 1219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 4), 'dim_prod', subscript_call_result_123655)
    
    # Assigning a Call to a Tuple (line 1221):
    
    # Assigning a Subscript to a Name (line 1221):
    
    # Obtaining the type of the subscript
    int_123656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 4), 'int')
    
    # Call to _select(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'input' (line 1221)
    input_123658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 63), 'input', False)
    # Getting the type of 'labels' (line 1221)
    labels_123659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 70), 'labels', False)
    # Getting the type of 'index' (line 1222)
    index_123660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 63), 'index', False)
    # Processing the call keyword arguments (line 1221)
    # Getting the type of 'True' (line 1223)
    True_123661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 72), 'True', False)
    keyword_123662 = True_123661
    # Getting the type of 'True' (line 1224)
    True_123663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 72), 'True', False)
    keyword_123664 = True_123663
    # Getting the type of 'True' (line 1225)
    True_123665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 82), 'True', False)
    keyword_123666 = True_123665
    # Getting the type of 'True' (line 1226)
    True_123667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 82), 'True', False)
    keyword_123668 = True_123667
    kwargs_123669 = {'find_min': keyword_123662, 'find_min_positions': keyword_123666, 'find_max_positions': keyword_123668, 'find_max': keyword_123664}
    # Getting the type of '_select' (line 1221)
    _select_123657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 55), '_select', False)
    # Calling _select(args, kwargs) (line 1221)
    _select_call_result_123670 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 55), _select_123657, *[input_123658, labels_123659, index_123660], **kwargs_123669)
    
    # Obtaining the member '__getitem__' of a type (line 1221)
    getitem___123671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 4), _select_call_result_123670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1221)
    subscript_call_result_123672 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 4), getitem___123671, int_123656)
    
    # Assigning a type to the variable 'tuple_var_assignment_121919' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121919', subscript_call_result_123672)
    
    # Assigning a Subscript to a Name (line 1221):
    
    # Obtaining the type of the subscript
    int_123673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 4), 'int')
    
    # Call to _select(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'input' (line 1221)
    input_123675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 63), 'input', False)
    # Getting the type of 'labels' (line 1221)
    labels_123676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 70), 'labels', False)
    # Getting the type of 'index' (line 1222)
    index_123677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 63), 'index', False)
    # Processing the call keyword arguments (line 1221)
    # Getting the type of 'True' (line 1223)
    True_123678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 72), 'True', False)
    keyword_123679 = True_123678
    # Getting the type of 'True' (line 1224)
    True_123680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 72), 'True', False)
    keyword_123681 = True_123680
    # Getting the type of 'True' (line 1225)
    True_123682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 82), 'True', False)
    keyword_123683 = True_123682
    # Getting the type of 'True' (line 1226)
    True_123684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 82), 'True', False)
    keyword_123685 = True_123684
    kwargs_123686 = {'find_min': keyword_123679, 'find_min_positions': keyword_123683, 'find_max_positions': keyword_123685, 'find_max': keyword_123681}
    # Getting the type of '_select' (line 1221)
    _select_123674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 55), '_select', False)
    # Calling _select(args, kwargs) (line 1221)
    _select_call_result_123687 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 55), _select_123674, *[input_123675, labels_123676, index_123677], **kwargs_123686)
    
    # Obtaining the member '__getitem__' of a type (line 1221)
    getitem___123688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 4), _select_call_result_123687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1221)
    subscript_call_result_123689 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 4), getitem___123688, int_123673)
    
    # Assigning a type to the variable 'tuple_var_assignment_121920' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121920', subscript_call_result_123689)
    
    # Assigning a Subscript to a Name (line 1221):
    
    # Obtaining the type of the subscript
    int_123690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 4), 'int')
    
    # Call to _select(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'input' (line 1221)
    input_123692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 63), 'input', False)
    # Getting the type of 'labels' (line 1221)
    labels_123693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 70), 'labels', False)
    # Getting the type of 'index' (line 1222)
    index_123694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 63), 'index', False)
    # Processing the call keyword arguments (line 1221)
    # Getting the type of 'True' (line 1223)
    True_123695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 72), 'True', False)
    keyword_123696 = True_123695
    # Getting the type of 'True' (line 1224)
    True_123697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 72), 'True', False)
    keyword_123698 = True_123697
    # Getting the type of 'True' (line 1225)
    True_123699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 82), 'True', False)
    keyword_123700 = True_123699
    # Getting the type of 'True' (line 1226)
    True_123701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 82), 'True', False)
    keyword_123702 = True_123701
    kwargs_123703 = {'find_min': keyword_123696, 'find_min_positions': keyword_123700, 'find_max_positions': keyword_123702, 'find_max': keyword_123698}
    # Getting the type of '_select' (line 1221)
    _select_123691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 55), '_select', False)
    # Calling _select(args, kwargs) (line 1221)
    _select_call_result_123704 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 55), _select_123691, *[input_123692, labels_123693, index_123694], **kwargs_123703)
    
    # Obtaining the member '__getitem__' of a type (line 1221)
    getitem___123705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 4), _select_call_result_123704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1221)
    subscript_call_result_123706 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 4), getitem___123705, int_123690)
    
    # Assigning a type to the variable 'tuple_var_assignment_121921' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121921', subscript_call_result_123706)
    
    # Assigning a Subscript to a Name (line 1221):
    
    # Obtaining the type of the subscript
    int_123707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 4), 'int')
    
    # Call to _select(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'input' (line 1221)
    input_123709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 63), 'input', False)
    # Getting the type of 'labels' (line 1221)
    labels_123710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 70), 'labels', False)
    # Getting the type of 'index' (line 1222)
    index_123711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 63), 'index', False)
    # Processing the call keyword arguments (line 1221)
    # Getting the type of 'True' (line 1223)
    True_123712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 72), 'True', False)
    keyword_123713 = True_123712
    # Getting the type of 'True' (line 1224)
    True_123714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 72), 'True', False)
    keyword_123715 = True_123714
    # Getting the type of 'True' (line 1225)
    True_123716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 82), 'True', False)
    keyword_123717 = True_123716
    # Getting the type of 'True' (line 1226)
    True_123718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 82), 'True', False)
    keyword_123719 = True_123718
    kwargs_123720 = {'find_min': keyword_123713, 'find_min_positions': keyword_123717, 'find_max_positions': keyword_123719, 'find_max': keyword_123715}
    # Getting the type of '_select' (line 1221)
    _select_123708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 55), '_select', False)
    # Calling _select(args, kwargs) (line 1221)
    _select_call_result_123721 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 55), _select_123708, *[input_123709, labels_123710, index_123711], **kwargs_123720)
    
    # Obtaining the member '__getitem__' of a type (line 1221)
    getitem___123722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 4), _select_call_result_123721, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1221)
    subscript_call_result_123723 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 4), getitem___123722, int_123707)
    
    # Assigning a type to the variable 'tuple_var_assignment_121922' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121922', subscript_call_result_123723)
    
    # Assigning a Name to a Name (line 1221):
    # Getting the type of 'tuple_var_assignment_121919' (line 1221)
    tuple_var_assignment_121919_123724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121919')
    # Assigning a type to the variable 'minimums' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'minimums', tuple_var_assignment_121919_123724)
    
    # Assigning a Name to a Name (line 1221):
    # Getting the type of 'tuple_var_assignment_121920' (line 1221)
    tuple_var_assignment_121920_123725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121920')
    # Assigning a type to the variable 'min_positions' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 14), 'min_positions', tuple_var_assignment_121920_123725)
    
    # Assigning a Name to a Name (line 1221):
    # Getting the type of 'tuple_var_assignment_121921' (line 1221)
    tuple_var_assignment_121921_123726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121921')
    # Assigning a type to the variable 'maximums' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 29), 'maximums', tuple_var_assignment_121921_123726)
    
    # Assigning a Name to a Name (line 1221):
    # Getting the type of 'tuple_var_assignment_121922' (line 1221)
    tuple_var_assignment_121922_123727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'tuple_var_assignment_121922')
    # Assigning a type to the variable 'max_positions' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 39), 'max_positions', tuple_var_assignment_121922_123727)
    
    
    # Call to isscalar(...): (line 1228)
    # Processing the call arguments (line 1228)
    # Getting the type of 'minimums' (line 1228)
    minimums_123730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 22), 'minimums', False)
    # Processing the call keyword arguments (line 1228)
    kwargs_123731 = {}
    # Getting the type of 'numpy' (line 1228)
    numpy_123728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 1228)
    isscalar_123729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 7), numpy_123728, 'isscalar')
    # Calling isscalar(args, kwargs) (line 1228)
    isscalar_call_result_123732 = invoke(stypy.reporting.localization.Localization(__file__, 1228, 7), isscalar_123729, *[minimums_123730], **kwargs_123731)
    
    # Testing the type of an if condition (line 1228)
    if_condition_123733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1228, 4), isscalar_call_result_123732)
    # Assigning a type to the variable 'if_condition_123733' (line 1228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 4), 'if_condition_123733', if_condition_123733)
    # SSA begins for if statement (line 1228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1229)
    tuple_123734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1229)
    # Adding element type (line 1229)
    # Getting the type of 'minimums' (line 1229)
    minimums_123735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 16), 'minimums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1229, 16), tuple_123734, minimums_123735)
    # Adding element type (line 1229)
    # Getting the type of 'maximums' (line 1229)
    maximums_123736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 26), 'maximums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1229, 16), tuple_123734, maximums_123736)
    # Adding element type (line 1229)
    
    # Call to tuple(...): (line 1229)
    # Processing the call arguments (line 1229)
    # Getting the type of 'min_positions' (line 1229)
    min_positions_123738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 43), 'min_positions', False)
    # Getting the type of 'dim_prod' (line 1229)
    dim_prod_123739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 60), 'dim_prod', False)
    # Applying the binary operator '//' (line 1229)
    result_floordiv_123740 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 43), '//', min_positions_123738, dim_prod_123739)
    
    # Getting the type of 'dims' (line 1229)
    dims_123741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 72), 'dims', False)
    # Applying the binary operator '%' (line 1229)
    result_mod_123742 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 42), '%', result_floordiv_123740, dims_123741)
    
    # Processing the call keyword arguments (line 1229)
    kwargs_123743 = {}
    # Getting the type of 'tuple' (line 1229)
    tuple_123737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 36), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1229)
    tuple_call_result_123744 = invoke(stypy.reporting.localization.Localization(__file__, 1229, 36), tuple_123737, *[result_mod_123742], **kwargs_123743)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1229, 16), tuple_123734, tuple_call_result_123744)
    # Adding element type (line 1229)
    
    # Call to tuple(...): (line 1230)
    # Processing the call arguments (line 1230)
    # Getting the type of 'max_positions' (line 1230)
    max_positions_123746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 23), 'max_positions', False)
    # Getting the type of 'dim_prod' (line 1230)
    dim_prod_123747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 40), 'dim_prod', False)
    # Applying the binary operator '//' (line 1230)
    result_floordiv_123748 = python_operator(stypy.reporting.localization.Localization(__file__, 1230, 23), '//', max_positions_123746, dim_prod_123747)
    
    # Getting the type of 'dims' (line 1230)
    dims_123749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 52), 'dims', False)
    # Applying the binary operator '%' (line 1230)
    result_mod_123750 = python_operator(stypy.reporting.localization.Localization(__file__, 1230, 22), '%', result_floordiv_123748, dims_123749)
    
    # Processing the call keyword arguments (line 1230)
    kwargs_123751 = {}
    # Getting the type of 'tuple' (line 1230)
    tuple_123745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1230)
    tuple_call_result_123752 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 16), tuple_123745, *[result_mod_123750], **kwargs_123751)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1229, 16), tuple_123734, tuple_call_result_123752)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 8), 'stypy_return_type', tuple_123734)
    # SSA join for if statement (line 1228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 1232):
    
    # Assigning a ListComp to a Name (line 1232):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reshape(...): (line 1232)
    # Processing the call arguments (line 1232)
    int_123759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 62), 'int')
    int_123760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 66), 'int')
    # Processing the call keyword arguments (line 1232)
    kwargs_123761 = {}
    # Getting the type of 'min_positions' (line 1232)
    min_positions_123757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 40), 'min_positions', False)
    # Obtaining the member 'reshape' of a type (line 1232)
    reshape_123758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1232, 40), min_positions_123757, 'reshape')
    # Calling reshape(args, kwargs) (line 1232)
    reshape_call_result_123762 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 40), reshape_123758, *[int_123759, int_123760], **kwargs_123761)
    
    # Getting the type of 'dim_prod' (line 1232)
    dim_prod_123763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 72), 'dim_prod')
    # Applying the binary operator '//' (line 1232)
    result_floordiv_123764 = python_operator(stypy.reporting.localization.Localization(__file__, 1232, 40), '//', reshape_call_result_123762, dim_prod_123763)
    
    # Getting the type of 'dims' (line 1232)
    dims_123765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 84), 'dims')
    # Applying the binary operator '%' (line 1232)
    result_mod_123766 = python_operator(stypy.reporting.localization.Localization(__file__, 1232, 39), '%', result_floordiv_123764, dims_123765)
    
    comprehension_123767 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 21), result_mod_123766)
    # Assigning a type to the variable 'v' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 21), 'v', comprehension_123767)
    
    # Call to tuple(...): (line 1232)
    # Processing the call arguments (line 1232)
    # Getting the type of 'v' (line 1232)
    v_123754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 27), 'v', False)
    # Processing the call keyword arguments (line 1232)
    kwargs_123755 = {}
    # Getting the type of 'tuple' (line 1232)
    tuple_123753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 21), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1232)
    tuple_call_result_123756 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 21), tuple_123753, *[v_123754], **kwargs_123755)
    
    list_123768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1232, 21), list_123768, tuple_call_result_123756)
    # Assigning a type to the variable 'min_positions' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 4), 'min_positions', list_123768)
    
    # Assigning a ListComp to a Name (line 1233):
    
    # Assigning a ListComp to a Name (line 1233):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to reshape(...): (line 1233)
    # Processing the call arguments (line 1233)
    int_123775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 62), 'int')
    int_123776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 66), 'int')
    # Processing the call keyword arguments (line 1233)
    kwargs_123777 = {}
    # Getting the type of 'max_positions' (line 1233)
    max_positions_123773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 40), 'max_positions', False)
    # Obtaining the member 'reshape' of a type (line 1233)
    reshape_123774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 40), max_positions_123773, 'reshape')
    # Calling reshape(args, kwargs) (line 1233)
    reshape_call_result_123778 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 40), reshape_123774, *[int_123775, int_123776], **kwargs_123777)
    
    # Getting the type of 'dim_prod' (line 1233)
    dim_prod_123779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 72), 'dim_prod')
    # Applying the binary operator '//' (line 1233)
    result_floordiv_123780 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 40), '//', reshape_call_result_123778, dim_prod_123779)
    
    # Getting the type of 'dims' (line 1233)
    dims_123781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 84), 'dims')
    # Applying the binary operator '%' (line 1233)
    result_mod_123782 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 39), '%', result_floordiv_123780, dims_123781)
    
    comprehension_123783 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1233, 21), result_mod_123782)
    # Assigning a type to the variable 'v' (line 1233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 21), 'v', comprehension_123783)
    
    # Call to tuple(...): (line 1233)
    # Processing the call arguments (line 1233)
    # Getting the type of 'v' (line 1233)
    v_123770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 27), 'v', False)
    # Processing the call keyword arguments (line 1233)
    kwargs_123771 = {}
    # Getting the type of 'tuple' (line 1233)
    tuple_123769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 21), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1233)
    tuple_call_result_123772 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 21), tuple_123769, *[v_123770], **kwargs_123771)
    
    list_123784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1233, 21), list_123784, tuple_call_result_123772)
    # Assigning a type to the variable 'max_positions' (line 1233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 4), 'max_positions', list_123784)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1235)
    tuple_123785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1235)
    # Adding element type (line 1235)
    # Getting the type of 'minimums' (line 1235)
    minimums_123786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 11), 'minimums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 11), tuple_123785, minimums_123786)
    # Adding element type (line 1235)
    # Getting the type of 'maximums' (line 1235)
    maximums_123787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 21), 'maximums')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 11), tuple_123785, maximums_123787)
    # Adding element type (line 1235)
    # Getting the type of 'min_positions' (line 1235)
    min_positions_123788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 31), 'min_positions')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 11), tuple_123785, min_positions_123788)
    # Adding element type (line 1235)
    # Getting the type of 'max_positions' (line 1235)
    max_positions_123789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 46), 'max_positions')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 11), tuple_123785, max_positions_123789)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 4), 'stypy_return_type', tuple_123785)
    
    # ################# End of 'extrema(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'extrema' in the type store
    # Getting the type of 'stypy_return_type' (line 1164)
    stypy_return_type_123790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123790)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'extrema'
    return stypy_return_type_123790

# Assigning a type to the variable 'extrema' (line 1164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 0), 'extrema', extrema)

@norecursion
def center_of_mass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1238)
    None_123791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 33), 'None')
    # Getting the type of 'None' (line 1238)
    None_123792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 45), 'None')
    defaults = [None_123791, None_123792]
    # Create a new context for function 'center_of_mass'
    module_type_store = module_type_store.open_function_context('center_of_mass', 1238, 0, False)
    
    # Passed parameters checking function
    center_of_mass.stypy_localization = localization
    center_of_mass.stypy_type_of_self = None
    center_of_mass.stypy_type_store = module_type_store
    center_of_mass.stypy_function_name = 'center_of_mass'
    center_of_mass.stypy_param_names_list = ['input', 'labels', 'index']
    center_of_mass.stypy_varargs_param_name = None
    center_of_mass.stypy_kwargs_param_name = None
    center_of_mass.stypy_call_defaults = defaults
    center_of_mass.stypy_call_varargs = varargs
    center_of_mass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'center_of_mass', ['input', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'center_of_mass', localization, ['input', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'center_of_mass(...)' code ##################

    str_123793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1296, (-1)), 'str', '\n    Calculate the center of mass of the values of an array at labels.\n\n    Parameters\n    ----------\n    input : ndarray\n        Data from which to calculate center-of-mass. The masses can either\n        be positive or negative.\n    labels : ndarray, optional\n        Labels for objects in `input`, as generated by `ndimage.label`.\n        Only used with `index`.  Dimensions must be the same as `input`.\n    index : int or sequence of ints, optional\n        Labels for which to calculate centers-of-mass. If not specified,\n        all labels greater than zero are used.  Only used with `labels`.\n\n    Returns\n    -------\n    center_of_mass : tuple, or list of tuples\n        Coordinates of centers-of-mass.\n\n    Examples\n    --------\n    >>> a = np.array(([0,0,0,0],\n    ...               [0,1,1,0],\n    ...               [0,1,1,0],\n    ...               [0,1,1,0]))\n    >>> from scipy import ndimage\n    >>> ndimage.measurements.center_of_mass(a)\n    (2.0, 1.5)\n\n    Calculation of multiple objects in an image\n\n    >>> b = np.array(([0,1,1,0],\n    ...               [0,1,0,0],\n    ...               [0,0,0,0],\n    ...               [0,0,1,1],\n    ...               [0,0,1,1]))\n    >>> lbl = ndimage.label(b)[0]\n    >>> ndimage.measurements.center_of_mass(b, lbl, [1,2])\n    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]\n\n    Negative masses are also accepted, which can occur for example when\n    bias is removed from measured data due to random noise.\n\n    >>> c = np.array(([-1,0,0,0],\n    ...               [0,-1,-1,0],\n    ...               [0,1,-1,0],\n    ...               [0,1,1,0]))\n    >>> ndimage.measurements.center_of_mass(c)\n    (-4.0, 1.0)\n\n    If there are division by zero issues, the function does not raise an\n    error but rather issues a RuntimeWarning before returning inf and/or NaN.\n\n    >>> d = np.array([-1, 1])\n    >>> ndimage.measurements.center_of_mass(d)\n    (inf,)\n    ')
    
    # Assigning a Call to a Name (line 1297):
    
    # Assigning a Call to a Name (line 1297):
    
    # Call to sum(...): (line 1297)
    # Processing the call arguments (line 1297)
    # Getting the type of 'input' (line 1297)
    input_123795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 21), 'input', False)
    # Getting the type of 'labels' (line 1297)
    labels_123796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 28), 'labels', False)
    # Getting the type of 'index' (line 1297)
    index_123797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 36), 'index', False)
    # Processing the call keyword arguments (line 1297)
    kwargs_123798 = {}
    # Getting the type of 'sum' (line 1297)
    sum_123794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 17), 'sum', False)
    # Calling sum(args, kwargs) (line 1297)
    sum_call_result_123799 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 17), sum_123794, *[input_123795, labels_123796, index_123797], **kwargs_123798)
    
    # Assigning a type to the variable 'normalizer' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'normalizer', sum_call_result_123799)
    
    # Assigning a Subscript to a Name (line 1298):
    
    # Assigning a Subscript to a Name (line 1298):
    
    # Obtaining the type of the subscript
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'input' (line 1298)
    input_123805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 46), 'input')
    # Obtaining the member 'shape' of a type (line 1298)
    shape_123806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 46), input_123805, 'shape')
    comprehension_123807 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1298, 25), shape_123806)
    # Assigning a type to the variable 'i' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 25), 'i', comprehension_123807)
    
    # Call to slice(...): (line 1298)
    # Processing the call arguments (line 1298)
    int_123801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1298, 31), 'int')
    # Getting the type of 'i' (line 1298)
    i_123802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 34), 'i', False)
    # Processing the call keyword arguments (line 1298)
    kwargs_123803 = {}
    # Getting the type of 'slice' (line 1298)
    slice_123800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 25), 'slice', False)
    # Calling slice(args, kwargs) (line 1298)
    slice_call_result_123804 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 25), slice_123800, *[int_123801, i_123802], **kwargs_123803)
    
    list_123808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1298, 25), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1298, 25), list_123808, slice_call_result_123804)
    # Getting the type of 'numpy' (line 1298)
    numpy_123809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 12), 'numpy')
    # Obtaining the member 'ogrid' of a type (line 1298)
    ogrid_123810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 12), numpy_123809, 'ogrid')
    # Obtaining the member '__getitem__' of a type (line 1298)
    getitem___123811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1298, 12), ogrid_123810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1298)
    subscript_call_result_123812 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 12), getitem___123811, list_123808)
    
    # Assigning a type to the variable 'grids' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 4), 'grids', subscript_call_result_123812)
    
    # Assigning a ListComp to a Name (line 1300):
    
    # Assigning a ListComp to a Name (line 1300):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 1301)
    # Processing the call arguments (line 1301)
    # Getting the type of 'input' (line 1301)
    input_123831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 32), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1301)
    ndim_123832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 32), input_123831, 'ndim')
    # Processing the call keyword arguments (line 1301)
    kwargs_123833 = {}
    # Getting the type of 'range' (line 1301)
    range_123830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 26), 'range', False)
    # Calling range(args, kwargs) (line 1301)
    range_call_result_123834 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 26), range_123830, *[ndim_123832], **kwargs_123833)
    
    comprehension_123835 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1300, 15), range_call_result_123834)
    # Assigning a type to the variable 'dir' (line 1300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 15), 'dir', comprehension_123835)
    
    # Call to sum(...): (line 1300)
    # Processing the call arguments (line 1300)
    # Getting the type of 'input' (line 1300)
    input_123814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 19), 'input', False)
    
    # Call to astype(...): (line 1300)
    # Processing the call arguments (line 1300)
    # Getting the type of 'float' (line 1300)
    float_123820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 45), 'float', False)
    # Processing the call keyword arguments (line 1300)
    kwargs_123821 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 1300)
    dir_123815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 33), 'dir', False)
    # Getting the type of 'grids' (line 1300)
    grids_123816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 27), 'grids', False)
    # Obtaining the member '__getitem__' of a type (line 1300)
    getitem___123817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 27), grids_123816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1300)
    subscript_call_result_123818 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 27), getitem___123817, dir_123815)
    
    # Obtaining the member 'astype' of a type (line 1300)
    astype_123819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 27), subscript_call_result_123818, 'astype')
    # Calling astype(args, kwargs) (line 1300)
    astype_call_result_123822 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 27), astype_123819, *[float_123820], **kwargs_123821)
    
    # Applying the binary operator '*' (line 1300)
    result_mul_123823 = python_operator(stypy.reporting.localization.Localization(__file__, 1300, 19), '*', input_123814, astype_call_result_123822)
    
    # Getting the type of 'labels' (line 1300)
    labels_123824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 53), 'labels', False)
    # Getting the type of 'index' (line 1300)
    index_123825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 61), 'index', False)
    # Processing the call keyword arguments (line 1300)
    kwargs_123826 = {}
    # Getting the type of 'sum' (line 1300)
    sum_123813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 15), 'sum', False)
    # Calling sum(args, kwargs) (line 1300)
    sum_call_result_123827 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 15), sum_123813, *[result_mul_123823, labels_123824, index_123825], **kwargs_123826)
    
    # Getting the type of 'normalizer' (line 1300)
    normalizer_123828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 70), 'normalizer')
    # Applying the binary operator 'div' (line 1300)
    result_div_123829 = python_operator(stypy.reporting.localization.Localization(__file__, 1300, 15), 'div', sum_call_result_123827, normalizer_123828)
    
    list_123836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1300, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1300, 15), list_123836, result_div_123829)
    # Assigning a type to the variable 'results' (line 1300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 4), 'results', list_123836)
    
    
    # Call to isscalar(...): (line 1303)
    # Processing the call arguments (line 1303)
    
    # Obtaining the type of the subscript
    int_123839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 30), 'int')
    # Getting the type of 'results' (line 1303)
    results_123840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 22), 'results', False)
    # Obtaining the member '__getitem__' of a type (line 1303)
    getitem___123841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 22), results_123840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1303)
    subscript_call_result_123842 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 22), getitem___123841, int_123839)
    
    # Processing the call keyword arguments (line 1303)
    kwargs_123843 = {}
    # Getting the type of 'numpy' (line 1303)
    numpy_123837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 7), 'numpy', False)
    # Obtaining the member 'isscalar' of a type (line 1303)
    isscalar_123838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 7), numpy_123837, 'isscalar')
    # Calling isscalar(args, kwargs) (line 1303)
    isscalar_call_result_123844 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 7), isscalar_123838, *[subscript_call_result_123842], **kwargs_123843)
    
    # Testing the type of an if condition (line 1303)
    if_condition_123845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1303, 4), isscalar_call_result_123844)
    # Assigning a type to the variable 'if_condition_123845' (line 1303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 4), 'if_condition_123845', if_condition_123845)
    # SSA begins for if statement (line 1303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to tuple(...): (line 1304)
    # Processing the call arguments (line 1304)
    # Getting the type of 'results' (line 1304)
    results_123847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 21), 'results', False)
    # Processing the call keyword arguments (line 1304)
    kwargs_123848 = {}
    # Getting the type of 'tuple' (line 1304)
    tuple_123846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1304)
    tuple_call_result_123849 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 15), tuple_123846, *[results_123847], **kwargs_123848)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 8), 'stypy_return_type', tuple_call_result_123849)
    # SSA join for if statement (line 1303)
    module_type_store = module_type_store.join_ssa_context()
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to array(...): (line 1306)
    # Processing the call arguments (line 1306)
    # Getting the type of 'results' (line 1306)
    results_123856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 42), 'results', False)
    # Processing the call keyword arguments (line 1306)
    kwargs_123857 = {}
    # Getting the type of 'numpy' (line 1306)
    numpy_123854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 30), 'numpy', False)
    # Obtaining the member 'array' of a type (line 1306)
    array_123855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 30), numpy_123854, 'array')
    # Calling array(args, kwargs) (line 1306)
    array_call_result_123858 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 30), array_123855, *[results_123856], **kwargs_123857)
    
    # Obtaining the member 'T' of a type (line 1306)
    T_123859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1306, 30), array_call_result_123858, 'T')
    comprehension_123860 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 12), T_123859)
    # Assigning a type to the variable 'v' (line 1306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 12), 'v', comprehension_123860)
    
    # Call to tuple(...): (line 1306)
    # Processing the call arguments (line 1306)
    # Getting the type of 'v' (line 1306)
    v_123851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 18), 'v', False)
    # Processing the call keyword arguments (line 1306)
    kwargs_123852 = {}
    # Getting the type of 'tuple' (line 1306)
    tuple_123850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 1306)
    tuple_call_result_123853 = invoke(stypy.reporting.localization.Localization(__file__, 1306, 12), tuple_123850, *[v_123851], **kwargs_123852)
    
    list_123861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 12), list_123861, tuple_call_result_123853)
    # Assigning a type to the variable 'stypy_return_type' (line 1306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 4), 'stypy_return_type', list_123861)
    
    # ################# End of 'center_of_mass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'center_of_mass' in the type store
    # Getting the type of 'stypy_return_type' (line 1238)
    stypy_return_type_123862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123862)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'center_of_mass'
    return stypy_return_type_123862

# Assigning a type to the variable 'center_of_mass' (line 1238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 0), 'center_of_mass', center_of_mass)

@norecursion
def histogram(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1309)
    None_123863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 44), 'None')
    # Getting the type of 'None' (line 1309)
    None_123864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 56), 'None')
    defaults = [None_123863, None_123864]
    # Create a new context for function 'histogram'
    module_type_store = module_type_store.open_function_context('histogram', 1309, 0, False)
    
    # Passed parameters checking function
    histogram.stypy_localization = localization
    histogram.stypy_type_of_self = None
    histogram.stypy_type_store = module_type_store
    histogram.stypy_function_name = 'histogram'
    histogram.stypy_param_names_list = ['input', 'min', 'max', 'bins', 'labels', 'index']
    histogram.stypy_varargs_param_name = None
    histogram.stypy_kwargs_param_name = None
    histogram.stypy_call_defaults = defaults
    histogram.stypy_call_varargs = varargs
    histogram.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'histogram', ['input', 'min', 'max', 'bins', 'labels', 'index'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'histogram', localization, ['input', 'min', 'max', 'bins', 'labels', 'index'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'histogram(...)' code ##################

    str_123865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1360, (-1)), 'str', '\n    Calculate the histogram of the values of an array, optionally at labels.\n\n    Histogram calculates the frequency of values in an array within bins\n    determined by `min`, `max`, and `bins`. The `labels` and `index`\n    keywords can limit the scope of the histogram to specified sub-regions\n    within the array.\n\n    Parameters\n    ----------\n    input : array_like\n        Data for which to calculate histogram.\n    min, max : int\n        Minimum and maximum values of range of histogram bins.\n    bins : int\n        Number of bins.\n    labels : array_like, optional\n        Labels for objects in `input`.\n        If not None, must be same shape as `input`.\n    index : int or sequence of ints, optional\n        Label or labels for which to calculate histogram. If None, all values\n        where label is greater than zero are used\n\n    Returns\n    -------\n    hist : ndarray\n        Histogram counts.\n\n    Examples\n    --------\n    >>> a = np.array([[ 0.    ,  0.2146,  0.5962,  0.    ],\n    ...               [ 0.    ,  0.7778,  0.    ,  0.    ],\n    ...               [ 0.    ,  0.    ,  0.    ,  0.    ],\n    ...               [ 0.    ,  0.    ,  0.7181,  0.2787],\n    ...               [ 0.    ,  0.    ,  0.6573,  0.3094]])\n    >>> from scipy import ndimage\n    >>> ndimage.measurements.histogram(a, 0, 1, 10)\n    array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])\n\n    With labels and no indices, non-zero elements are counted:\n\n    >>> lbl, nlbl = ndimage.label(a)\n    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)\n    array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])\n\n    Indices can be used to count only certain objects:\n\n    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)\n    array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])\n\n    ')
    
    # Assigning a Call to a Name (line 1361):
    
    # Assigning a Call to a Name (line 1361):
    
    # Call to linspace(...): (line 1361)
    # Processing the call arguments (line 1361)
    # Getting the type of 'min' (line 1361)
    min_123868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 27), 'min', False)
    # Getting the type of 'max' (line 1361)
    max_123869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 32), 'max', False)
    # Getting the type of 'bins' (line 1361)
    bins_123870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 37), 'bins', False)
    int_123871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1361, 44), 'int')
    # Applying the binary operator '+' (line 1361)
    result_add_123872 = python_operator(stypy.reporting.localization.Localization(__file__, 1361, 37), '+', bins_123870, int_123871)
    
    # Processing the call keyword arguments (line 1361)
    kwargs_123873 = {}
    # Getting the type of 'numpy' (line 1361)
    numpy_123866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 12), 'numpy', False)
    # Obtaining the member 'linspace' of a type (line 1361)
    linspace_123867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 12), numpy_123866, 'linspace')
    # Calling linspace(args, kwargs) (line 1361)
    linspace_call_result_123874 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 12), linspace_123867, *[min_123868, max_123869, result_add_123872], **kwargs_123873)
    
    # Assigning a type to the variable '_bins' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), '_bins', linspace_call_result_123874)

    @norecursion
    def _hist(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_hist'
        module_type_store = module_type_store.open_function_context('_hist', 1363, 4, False)
        
        # Passed parameters checking function
        _hist.stypy_localization = localization
        _hist.stypy_type_of_self = None
        _hist.stypy_type_store = module_type_store
        _hist.stypy_function_name = '_hist'
        _hist.stypy_param_names_list = ['vals']
        _hist.stypy_varargs_param_name = None
        _hist.stypy_kwargs_param_name = None
        _hist.stypy_call_defaults = defaults
        _hist.stypy_call_varargs = varargs
        _hist.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_hist', ['vals'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_hist', localization, ['vals'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_hist(...)' code ##################

        
        # Obtaining the type of the subscript
        int_123875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1364, 44), 'int')
        
        # Call to histogram(...): (line 1364)
        # Processing the call arguments (line 1364)
        # Getting the type of 'vals' (line 1364)
        vals_123878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 31), 'vals', False)
        # Getting the type of '_bins' (line 1364)
        _bins_123879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 37), '_bins', False)
        # Processing the call keyword arguments (line 1364)
        kwargs_123880 = {}
        # Getting the type of 'numpy' (line 1364)
        numpy_123876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 15), 'numpy', False)
        # Obtaining the member 'histogram' of a type (line 1364)
        histogram_123877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 15), numpy_123876, 'histogram')
        # Calling histogram(args, kwargs) (line 1364)
        histogram_call_result_123881 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 15), histogram_123877, *[vals_123878, _bins_123879], **kwargs_123880)
        
        # Obtaining the member '__getitem__' of a type (line 1364)
        getitem___123882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 15), histogram_call_result_123881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
        subscript_call_result_123883 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 15), getitem___123882, int_123875)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 8), 'stypy_return_type', subscript_call_result_123883)
        
        # ################# End of '_hist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_hist' in the type store
        # Getting the type of 'stypy_return_type' (line 1363)
        stypy_return_type_123884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_123884)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_hist'
        return stypy_return_type_123884

    # Assigning a type to the variable '_hist' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), '_hist', _hist)
    
    # Call to labeled_comprehension(...): (line 1366)
    # Processing the call arguments (line 1366)
    # Getting the type of 'input' (line 1366)
    input_123886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 33), 'input', False)
    # Getting the type of 'labels' (line 1366)
    labels_123887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 40), 'labels', False)
    # Getting the type of 'index' (line 1366)
    index_123888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 48), 'index', False)
    # Getting the type of '_hist' (line 1366)
    _hist_123889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 55), '_hist', False)
    # Getting the type of 'object' (line 1366)
    object_123890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 62), 'object', False)
    # Getting the type of 'None' (line 1366)
    None_123891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 70), 'None', False)
    # Processing the call keyword arguments (line 1366)
    # Getting the type of 'False' (line 1367)
    False_123892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 48), 'False', False)
    keyword_123893 = False_123892
    kwargs_123894 = {'pass_positions': keyword_123893}
    # Getting the type of 'labeled_comprehension' (line 1366)
    labeled_comprehension_123885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 11), 'labeled_comprehension', False)
    # Calling labeled_comprehension(args, kwargs) (line 1366)
    labeled_comprehension_call_result_123895 = invoke(stypy.reporting.localization.Localization(__file__, 1366, 11), labeled_comprehension_123885, *[input_123886, labels_123887, index_123888, _hist_123889, object_123890, None_123891], **kwargs_123894)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1366, 4), 'stypy_return_type', labeled_comprehension_call_result_123895)
    
    # ################# End of 'histogram(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'histogram' in the type store
    # Getting the type of 'stypy_return_type' (line 1309)
    stypy_return_type_123896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'histogram'
    return stypy_return_type_123896

# Assigning a type to the variable 'histogram' (line 1309)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1309, 0), 'histogram', histogram)

@norecursion
def watershed_ift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1370)
    None_123897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 44), 'None')
    # Getting the type of 'None' (line 1370)
    None_123898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 57), 'None')
    defaults = [None_123897, None_123898]
    # Create a new context for function 'watershed_ift'
    module_type_store = module_type_store.open_function_context('watershed_ift', 1370, 0, False)
    
    # Passed parameters checking function
    watershed_ift.stypy_localization = localization
    watershed_ift.stypy_type_of_self = None
    watershed_ift.stypy_type_store = module_type_store
    watershed_ift.stypy_function_name = 'watershed_ift'
    watershed_ift.stypy_param_names_list = ['input', 'markers', 'structure', 'output']
    watershed_ift.stypy_varargs_param_name = None
    watershed_ift.stypy_kwargs_param_name = None
    watershed_ift.stypy_call_defaults = defaults
    watershed_ift.stypy_call_varargs = varargs
    watershed_ift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'watershed_ift', ['input', 'markers', 'structure', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'watershed_ift', localization, ['input', 'markers', 'structure', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'watershed_ift(...)' code ##################

    str_123899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1400, (-1)), 'str', '\n    Apply watershed from markers using image foresting transform algorithm.\n\n    Parameters\n    ----------\n    input : array_like\n        Input.\n    markers : array_like\n        Markers are points within each watershed that form the beginning\n        of the process.  Negative markers are considered background markers\n        which are processed after the other markers.\n    structure : structure element, optional\n        A structuring element defining the connectivity of the object can be\n        provided. If None, an element is generated with a squared\n        connectivity equal to one.\n    output : ndarray, optional\n        An output array can optionally be provided.  The same shape as input.\n\n    Returns\n    -------\n    watershed_ift : ndarray\n        Output.  Same shape as `input`.\n\n    References\n    ----------\n    .. [1] A.X. Falcao, J. Stolfi and R. de Alencar Lotufo, "The image\n           foresting transform: theory, algorithms, and applications",\n           Pattern Analysis and Machine Intelligence, vol. 26, pp. 19-29, 2004.\n\n    ')
    
    # Assigning a Call to a Name (line 1401):
    
    # Assigning a Call to a Name (line 1401):
    
    # Call to asarray(...): (line 1401)
    # Processing the call arguments (line 1401)
    # Getting the type of 'input' (line 1401)
    input_123902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 26), 'input', False)
    # Processing the call keyword arguments (line 1401)
    kwargs_123903 = {}
    # Getting the type of 'numpy' (line 1401)
    numpy_123900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1401, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1401)
    asarray_123901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1401, 12), numpy_123900, 'asarray')
    # Calling asarray(args, kwargs) (line 1401)
    asarray_call_result_123904 = invoke(stypy.reporting.localization.Localization(__file__, 1401, 12), asarray_123901, *[input_123902], **kwargs_123903)
    
    # Assigning a type to the variable 'input' (line 1401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1401, 4), 'input', asarray_call_result_123904)
    
    
    # Getting the type of 'input' (line 1402)
    input_123905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1402, 7), 'input')
    # Obtaining the member 'dtype' of a type (line 1402)
    dtype_123906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1402, 7), input_123905, 'dtype')
    # Obtaining the member 'type' of a type (line 1402)
    type_123907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1402, 7), dtype_123906, 'type')
    
    # Obtaining an instance of the builtin type 'list' (line 1402)
    list_123908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1402, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1402)
    # Adding element type (line 1402)
    # Getting the type of 'numpy' (line 1402)
    numpy_123909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1402, 32), 'numpy')
    # Obtaining the member 'uint8' of a type (line 1402)
    uint8_123910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1402, 32), numpy_123909, 'uint8')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 31), list_123908, uint8_123910)
    # Adding element type (line 1402)
    # Getting the type of 'numpy' (line 1402)
    numpy_123911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1402, 45), 'numpy')
    # Obtaining the member 'uint16' of a type (line 1402)
    uint16_123912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1402, 45), numpy_123911, 'uint16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1402, 31), list_123908, uint16_123912)
    
    # Applying the binary operator 'notin' (line 1402)
    result_contains_123913 = python_operator(stypy.reporting.localization.Localization(__file__, 1402, 7), 'notin', type_123907, list_123908)
    
    # Testing the type of an if condition (line 1402)
    if_condition_123914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1402, 4), result_contains_123913)
    # Assigning a type to the variable 'if_condition_123914' (line 1402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1402, 4), 'if_condition_123914', if_condition_123914)
    # SSA begins for if statement (line 1402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1403)
    # Processing the call arguments (line 1403)
    str_123916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1403, 24), 'str', 'only 8 and 16 unsigned inputs are supported')
    # Processing the call keyword arguments (line 1403)
    kwargs_123917 = {}
    # Getting the type of 'TypeError' (line 1403)
    TypeError_123915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1403, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1403)
    TypeError_call_result_123918 = invoke(stypy.reporting.localization.Localization(__file__, 1403, 14), TypeError_123915, *[str_123916], **kwargs_123917)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1403, 8), TypeError_call_result_123918, 'raise parameter', BaseException)
    # SSA join for if statement (line 1402)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1405)
    # Getting the type of 'structure' (line 1405)
    structure_123919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 7), 'structure')
    # Getting the type of 'None' (line 1405)
    None_123920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 20), 'None')
    
    (may_be_123921, more_types_in_union_123922) = may_be_none(structure_123919, None_123920)

    if may_be_123921:

        if more_types_in_union_123922:
            # Runtime conditional SSA (line 1405)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1406):
        
        # Assigning a Call to a Name (line 1406):
        
        # Call to generate_binary_structure(...): (line 1406)
        # Processing the call arguments (line 1406)
        # Getting the type of 'input' (line 1406)
        input_123925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1406, 57), 'input', False)
        # Obtaining the member 'ndim' of a type (line 1406)
        ndim_123926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1406, 57), input_123925, 'ndim')
        int_123927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1406, 69), 'int')
        # Processing the call keyword arguments (line 1406)
        kwargs_123928 = {}
        # Getting the type of 'morphology' (line 1406)
        morphology_123923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1406, 20), 'morphology', False)
        # Obtaining the member 'generate_binary_structure' of a type (line 1406)
        generate_binary_structure_123924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1406, 20), morphology_123923, 'generate_binary_structure')
        # Calling generate_binary_structure(args, kwargs) (line 1406)
        generate_binary_structure_call_result_123929 = invoke(stypy.reporting.localization.Localization(__file__, 1406, 20), generate_binary_structure_123924, *[ndim_123926, int_123927], **kwargs_123928)
        
        # Assigning a type to the variable 'structure' (line 1406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1406, 8), 'structure', generate_binary_structure_call_result_123929)

        if more_types_in_union_123922:
            # SSA join for if statement (line 1405)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1407):
    
    # Assigning a Call to a Name (line 1407):
    
    # Call to asarray(...): (line 1407)
    # Processing the call arguments (line 1407)
    # Getting the type of 'structure' (line 1407)
    structure_123932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 30), 'structure', False)
    # Processing the call keyword arguments (line 1407)
    # Getting the type of 'bool' (line 1407)
    bool_123933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 47), 'bool', False)
    keyword_123934 = bool_123933
    kwargs_123935 = {'dtype': keyword_123934}
    # Getting the type of 'numpy' (line 1407)
    numpy_123930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 16), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1407)
    asarray_123931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1407, 16), numpy_123930, 'asarray')
    # Calling asarray(args, kwargs) (line 1407)
    asarray_call_result_123936 = invoke(stypy.reporting.localization.Localization(__file__, 1407, 16), asarray_123931, *[structure_123932], **kwargs_123935)
    
    # Assigning a type to the variable 'structure' (line 1407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1407, 4), 'structure', asarray_call_result_123936)
    
    
    # Getting the type of 'structure' (line 1408)
    structure_123937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 7), 'structure')
    # Obtaining the member 'ndim' of a type (line 1408)
    ndim_123938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1408, 7), structure_123937, 'ndim')
    # Getting the type of 'input' (line 1408)
    input_123939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 25), 'input')
    # Obtaining the member 'ndim' of a type (line 1408)
    ndim_123940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1408, 25), input_123939, 'ndim')
    # Applying the binary operator '!=' (line 1408)
    result_ne_123941 = python_operator(stypy.reporting.localization.Localization(__file__, 1408, 7), '!=', ndim_123938, ndim_123940)
    
    # Testing the type of an if condition (line 1408)
    if_condition_123942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1408, 4), result_ne_123941)
    # Assigning a type to the variable 'if_condition_123942' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 4), 'if_condition_123942', if_condition_123942)
    # SSA begins for if statement (line 1408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1409)
    # Processing the call arguments (line 1409)
    str_123944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1409, 27), 'str', 'structure and input must have equal rank')
    # Processing the call keyword arguments (line 1409)
    kwargs_123945 = {}
    # Getting the type of 'RuntimeError' (line 1409)
    RuntimeError_123943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1409, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1409)
    RuntimeError_call_result_123946 = invoke(stypy.reporting.localization.Localization(__file__, 1409, 14), RuntimeError_123943, *[str_123944], **kwargs_123945)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1409, 8), RuntimeError_call_result_123946, 'raise parameter', BaseException)
    # SSA join for if statement (line 1408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'structure' (line 1410)
    structure_123947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 14), 'structure')
    # Obtaining the member 'shape' of a type (line 1410)
    shape_123948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1410, 14), structure_123947, 'shape')
    # Testing the type of a for loop iterable (line 1410)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1410, 4), shape_123948)
    # Getting the type of the for loop variable (line 1410)
    for_loop_var_123949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1410, 4), shape_123948)
    # Assigning a type to the variable 'ii' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 4), 'ii', for_loop_var_123949)
    # SSA begins for a for statement (line 1410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ii' (line 1411)
    ii_123950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 11), 'ii')
    int_123951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1411, 17), 'int')
    # Applying the binary operator '!=' (line 1411)
    result_ne_123952 = python_operator(stypy.reporting.localization.Localization(__file__, 1411, 11), '!=', ii_123950, int_123951)
    
    # Testing the type of an if condition (line 1411)
    if_condition_123953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1411, 8), result_ne_123952)
    # Assigning a type to the variable 'if_condition_123953' (line 1411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1411, 8), 'if_condition_123953', if_condition_123953)
    # SSA begins for if statement (line 1411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1412)
    # Processing the call arguments (line 1412)
    str_123955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 31), 'str', 'structure dimensions must be equal to 3')
    # Processing the call keyword arguments (line 1412)
    kwargs_123956 = {}
    # Getting the type of 'RuntimeError' (line 1412)
    RuntimeError_123954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1412)
    RuntimeError_call_result_123957 = invoke(stypy.reporting.localization.Localization(__file__, 1412, 18), RuntimeError_123954, *[str_123955], **kwargs_123956)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1412, 12), RuntimeError_call_result_123957, 'raise parameter', BaseException)
    # SSA join for if statement (line 1411)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'structure' (line 1414)
    structure_123958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 11), 'structure')
    # Obtaining the member 'flags' of a type (line 1414)
    flags_123959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 11), structure_123958, 'flags')
    # Obtaining the member 'contiguous' of a type (line 1414)
    contiguous_123960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 11), flags_123959, 'contiguous')
    # Applying the 'not' unary operator (line 1414)
    result_not__123961 = python_operator(stypy.reporting.localization.Localization(__file__, 1414, 7), 'not', contiguous_123960)
    
    # Testing the type of an if condition (line 1414)
    if_condition_123962 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1414, 4), result_not__123961)
    # Assigning a type to the variable 'if_condition_123962' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'if_condition_123962', if_condition_123962)
    # SSA begins for if statement (line 1414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1415):
    
    # Assigning a Call to a Name (line 1415):
    
    # Call to copy(...): (line 1415)
    # Processing the call keyword arguments (line 1415)
    kwargs_123965 = {}
    # Getting the type of 'structure' (line 1415)
    structure_123963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 20), 'structure', False)
    # Obtaining the member 'copy' of a type (line 1415)
    copy_123964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1415, 20), structure_123963, 'copy')
    # Calling copy(args, kwargs) (line 1415)
    copy_call_result_123966 = invoke(stypy.reporting.localization.Localization(__file__, 1415, 20), copy_123964, *[], **kwargs_123965)
    
    # Assigning a type to the variable 'structure' (line 1415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1415, 8), 'structure', copy_call_result_123966)
    # SSA join for if statement (line 1414)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1416):
    
    # Assigning a Call to a Name (line 1416):
    
    # Call to asarray(...): (line 1416)
    # Processing the call arguments (line 1416)
    # Getting the type of 'markers' (line 1416)
    markers_123969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 28), 'markers', False)
    # Processing the call keyword arguments (line 1416)
    kwargs_123970 = {}
    # Getting the type of 'numpy' (line 1416)
    numpy_123967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 14), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1416)
    asarray_123968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1416, 14), numpy_123967, 'asarray')
    # Calling asarray(args, kwargs) (line 1416)
    asarray_call_result_123971 = invoke(stypy.reporting.localization.Localization(__file__, 1416, 14), asarray_123968, *[markers_123969], **kwargs_123970)
    
    # Assigning a type to the variable 'markers' (line 1416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1416, 4), 'markers', asarray_call_result_123971)
    
    
    # Getting the type of 'input' (line 1417)
    input_123972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 7), 'input')
    # Obtaining the member 'shape' of a type (line 1417)
    shape_123973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 7), input_123972, 'shape')
    # Getting the type of 'markers' (line 1417)
    markers_123974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 22), 'markers')
    # Obtaining the member 'shape' of a type (line 1417)
    shape_123975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 22), markers_123974, 'shape')
    # Applying the binary operator '!=' (line 1417)
    result_ne_123976 = python_operator(stypy.reporting.localization.Localization(__file__, 1417, 7), '!=', shape_123973, shape_123975)
    
    # Testing the type of an if condition (line 1417)
    if_condition_123977 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1417, 4), result_ne_123976)
    # Assigning a type to the variable 'if_condition_123977' (line 1417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1417, 4), 'if_condition_123977', if_condition_123977)
    # SSA begins for if statement (line 1417)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1418)
    # Processing the call arguments (line 1418)
    str_123979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1418, 27), 'str', 'input and markers must have equal shape')
    # Processing the call keyword arguments (line 1418)
    kwargs_123980 = {}
    # Getting the type of 'RuntimeError' (line 1418)
    RuntimeError_123978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1418)
    RuntimeError_call_result_123981 = invoke(stypy.reporting.localization.Localization(__file__, 1418, 14), RuntimeError_123978, *[str_123979], **kwargs_123980)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1418, 8), RuntimeError_call_result_123981, 'raise parameter', BaseException)
    # SSA join for if statement (line 1417)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 1420):
    
    # Assigning a List to a Name (line 1420):
    
    # Obtaining an instance of the builtin type 'list' (line 1420)
    list_123982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1420, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1420)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1420)
    numpy_123983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1420, 22), 'numpy')
    # Obtaining the member 'int0' of a type (line 1420)
    int0_123984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1420, 22), numpy_123983, 'int0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int0_123984)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1421)
    numpy_123985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 22), 'numpy')
    # Obtaining the member 'int8' of a type (line 1421)
    int8_123986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 22), numpy_123985, 'int8')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int8_123986)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1422)
    numpy_123987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 22), 'numpy')
    # Obtaining the member 'int16' of a type (line 1422)
    int16_123988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1422, 22), numpy_123987, 'int16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int16_123988)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1423)
    numpy_123989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 22), 'numpy')
    # Obtaining the member 'int32' of a type (line 1423)
    int32_123990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1423, 22), numpy_123989, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int32_123990)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1424)
    numpy_123991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 22), 'numpy')
    # Obtaining the member 'int_' of a type (line 1424)
    int__123992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1424, 22), numpy_123991, 'int_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int__123992)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1425)
    numpy_123993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 22), 'numpy')
    # Obtaining the member 'int64' of a type (line 1425)
    int64_123994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1425, 22), numpy_123993, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, int64_123994)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1426)
    numpy_123995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 22), 'numpy')
    # Obtaining the member 'intc' of a type (line 1426)
    intc_123996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1426, 22), numpy_123995, 'intc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, intc_123996)
    # Adding element type (line 1420)
    # Getting the type of 'numpy' (line 1427)
    numpy_123997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 22), 'numpy')
    # Obtaining the member 'intp' of a type (line 1427)
    intp_123998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1427, 22), numpy_123997, 'intp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1420, 21), list_123982, intp_123998)
    
    # Assigning a type to the variable 'integral_types' (line 1420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1420, 4), 'integral_types', list_123982)
    
    
    # Getting the type of 'markers' (line 1429)
    markers_123999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 7), 'markers')
    # Obtaining the member 'dtype' of a type (line 1429)
    dtype_124000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1429, 7), markers_123999, 'dtype')
    # Obtaining the member 'type' of a type (line 1429)
    type_124001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1429, 7), dtype_124000, 'type')
    # Getting the type of 'integral_types' (line 1429)
    integral_types_124002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 33), 'integral_types')
    # Applying the binary operator 'notin' (line 1429)
    result_contains_124003 = python_operator(stypy.reporting.localization.Localization(__file__, 1429, 7), 'notin', type_124001, integral_types_124002)
    
    # Testing the type of an if condition (line 1429)
    if_condition_124004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1429, 4), result_contains_124003)
    # Assigning a type to the variable 'if_condition_124004' (line 1429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1429, 4), 'if_condition_124004', if_condition_124004)
    # SSA begins for if statement (line 1429)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1430)
    # Processing the call arguments (line 1430)
    str_124006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1430, 27), 'str', 'marker should be of integer type')
    # Processing the call keyword arguments (line 1430)
    kwargs_124007 = {}
    # Getting the type of 'RuntimeError' (line 1430)
    RuntimeError_124005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1430, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1430)
    RuntimeError_call_result_124008 = invoke(stypy.reporting.localization.Localization(__file__, 1430, 14), RuntimeError_124005, *[str_124006], **kwargs_124007)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1430, 8), RuntimeError_call_result_124008, 'raise parameter', BaseException)
    # SSA join for if statement (line 1429)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 1432)
    # Processing the call arguments (line 1432)
    # Getting the type of 'output' (line 1432)
    output_124010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 18), 'output', False)
    # Getting the type of 'numpy' (line 1432)
    numpy_124011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 26), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 1432)
    ndarray_124012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1432, 26), numpy_124011, 'ndarray')
    # Processing the call keyword arguments (line 1432)
    kwargs_124013 = {}
    # Getting the type of 'isinstance' (line 1432)
    isinstance_124009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1432)
    isinstance_call_result_124014 = invoke(stypy.reporting.localization.Localization(__file__, 1432, 7), isinstance_124009, *[output_124010, ndarray_124012], **kwargs_124013)
    
    # Testing the type of an if condition (line 1432)
    if_condition_124015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1432, 4), isinstance_call_result_124014)
    # Assigning a type to the variable 'if_condition_124015' (line 1432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1432, 4), 'if_condition_124015', if_condition_124015)
    # SSA begins for if statement (line 1432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'output' (line 1433)
    output_124016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1433, 11), 'output')
    # Obtaining the member 'dtype' of a type (line 1433)
    dtype_124017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1433, 11), output_124016, 'dtype')
    # Obtaining the member 'type' of a type (line 1433)
    type_124018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1433, 11), dtype_124017, 'type')
    # Getting the type of 'integral_types' (line 1433)
    integral_types_124019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1433, 36), 'integral_types')
    # Applying the binary operator 'notin' (line 1433)
    result_contains_124020 = python_operator(stypy.reporting.localization.Localization(__file__, 1433, 11), 'notin', type_124018, integral_types_124019)
    
    # Testing the type of an if condition (line 1433)
    if_condition_124021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1433, 8), result_contains_124020)
    # Assigning a type to the variable 'if_condition_124021' (line 1433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1433, 8), 'if_condition_124021', if_condition_124021)
    # SSA begins for if statement (line 1433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1434)
    # Processing the call arguments (line 1434)
    str_124023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1434, 31), 'str', 'output should be of integer type')
    # Processing the call keyword arguments (line 1434)
    kwargs_124024 = {}
    # Getting the type of 'RuntimeError' (line 1434)
    RuntimeError_124022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1434, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1434)
    RuntimeError_call_result_124025 = invoke(stypy.reporting.localization.Localization(__file__, 1434, 18), RuntimeError_124022, *[str_124023], **kwargs_124024)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1434, 12), RuntimeError_call_result_124025, 'raise parameter', BaseException)
    # SSA join for if statement (line 1433)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1432)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 1436):
    
    # Assigning a Attribute to a Name (line 1436):
    # Getting the type of 'markers' (line 1436)
    markers_124026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1436, 17), 'markers')
    # Obtaining the member 'dtype' of a type (line 1436)
    dtype_124027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1436, 17), markers_124026, 'dtype')
    # Assigning a type to the variable 'output' (line 1436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1436, 8), 'output', dtype_124027)
    # SSA join for if statement (line 1432)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1438):
    
    # Assigning a Subscript to a Name (line 1438):
    
    # Obtaining the type of the subscript
    int_124028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1438, 4), 'int')
    
    # Call to _get_output(...): (line 1438)
    # Processing the call arguments (line 1438)
    # Getting the type of 'output' (line 1438)
    output_124031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 51), 'output', False)
    # Getting the type of 'input' (line 1438)
    input_124032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 59), 'input', False)
    # Processing the call keyword arguments (line 1438)
    kwargs_124033 = {}
    # Getting the type of '_ni_support' (line 1438)
    _ni_support_124029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1438)
    _get_output_124030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1438, 27), _ni_support_124029, '_get_output')
    # Calling _get_output(args, kwargs) (line 1438)
    _get_output_call_result_124034 = invoke(stypy.reporting.localization.Localization(__file__, 1438, 27), _get_output_124030, *[output_124031, input_124032], **kwargs_124033)
    
    # Obtaining the member '__getitem__' of a type (line 1438)
    getitem___124035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1438, 4), _get_output_call_result_124034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1438)
    subscript_call_result_124036 = invoke(stypy.reporting.localization.Localization(__file__, 1438, 4), getitem___124035, int_124028)
    
    # Assigning a type to the variable 'tuple_var_assignment_121923' (line 1438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1438, 4), 'tuple_var_assignment_121923', subscript_call_result_124036)
    
    # Assigning a Subscript to a Name (line 1438):
    
    # Obtaining the type of the subscript
    int_124037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1438, 4), 'int')
    
    # Call to _get_output(...): (line 1438)
    # Processing the call arguments (line 1438)
    # Getting the type of 'output' (line 1438)
    output_124040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 51), 'output', False)
    # Getting the type of 'input' (line 1438)
    input_124041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 59), 'input', False)
    # Processing the call keyword arguments (line 1438)
    kwargs_124042 = {}
    # Getting the type of '_ni_support' (line 1438)
    _ni_support_124038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1438)
    _get_output_124039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1438, 27), _ni_support_124038, '_get_output')
    # Calling _get_output(args, kwargs) (line 1438)
    _get_output_call_result_124043 = invoke(stypy.reporting.localization.Localization(__file__, 1438, 27), _get_output_124039, *[output_124040, input_124041], **kwargs_124042)
    
    # Obtaining the member '__getitem__' of a type (line 1438)
    getitem___124044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1438, 4), _get_output_call_result_124043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1438)
    subscript_call_result_124045 = invoke(stypy.reporting.localization.Localization(__file__, 1438, 4), getitem___124044, int_124037)
    
    # Assigning a type to the variable 'tuple_var_assignment_121924' (line 1438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1438, 4), 'tuple_var_assignment_121924', subscript_call_result_124045)
    
    # Assigning a Name to a Name (line 1438):
    # Getting the type of 'tuple_var_assignment_121923' (line 1438)
    tuple_var_assignment_121923_124046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 4), 'tuple_var_assignment_121923')
    # Assigning a type to the variable 'output' (line 1438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1438, 4), 'output', tuple_var_assignment_121923_124046)
    
    # Assigning a Name to a Name (line 1438):
    # Getting the type of 'tuple_var_assignment_121924' (line 1438)
    tuple_var_assignment_121924_124047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 4), 'tuple_var_assignment_121924')
    # Assigning a type to the variable 'return_value' (line 1438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1438, 12), 'return_value', tuple_var_assignment_121924_124047)
    
    # Call to watershed_ift(...): (line 1439)
    # Processing the call arguments (line 1439)
    # Getting the type of 'input' (line 1439)
    input_124050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1439, 28), 'input', False)
    # Getting the type of 'markers' (line 1439)
    markers_124051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1439, 35), 'markers', False)
    # Getting the type of 'structure' (line 1439)
    structure_124052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1439, 44), 'structure', False)
    # Getting the type of 'output' (line 1439)
    output_124053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1439, 55), 'output', False)
    # Processing the call keyword arguments (line 1439)
    kwargs_124054 = {}
    # Getting the type of '_nd_image' (line 1439)
    _nd_image_124048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1439, 4), '_nd_image', False)
    # Obtaining the member 'watershed_ift' of a type (line 1439)
    watershed_ift_124049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1439, 4), _nd_image_124048, 'watershed_ift')
    # Calling watershed_ift(args, kwargs) (line 1439)
    watershed_ift_call_result_124055 = invoke(stypy.reporting.localization.Localization(__file__, 1439, 4), watershed_ift_124049, *[input_124050, markers_124051, structure_124052, output_124053], **kwargs_124054)
    
    # Getting the type of 'return_value' (line 1440)
    return_value_124056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1440, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1440, 4), 'stypy_return_type', return_value_124056)
    
    # ################# End of 'watershed_ift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'watershed_ift' in the type store
    # Getting the type of 'stypy_return_type' (line 1370)
    stypy_return_type_124057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_124057)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'watershed_ift'
    return stypy_return_type_124057

# Assigning a type to the variable 'watershed_ift' (line 1370)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1370, 0), 'watershed_ift', watershed_ift)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
