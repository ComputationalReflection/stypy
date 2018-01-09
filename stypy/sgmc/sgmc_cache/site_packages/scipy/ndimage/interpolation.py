
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
33: import math
34: import numpy
35: from . import _ni_support
36: from . import _nd_image
37: from functools import wraps
38: 
39: import warnings
40: 
41: __all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform',
42:            'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate']
43: 
44: 
45: def _extend_mode_to_code(mode):
46:     mode = _ni_support._extend_mode_to_code(mode)
47:     return mode
48: 
49: 
50: def spline_filter1d(input, order=3, axis=-1, output=numpy.float64):
51:     '''
52:     Calculates a one-dimensional spline filter along the given axis.
53: 
54:     The lines of the array along the given axis are filtered by a
55:     spline filter. The order of the spline must be >= 2 and <= 5.
56: 
57:     Parameters
58:     ----------
59:     input : array_like
60:         The input array.
61:     order : int, optional
62:         The order of the spline, default is 3.
63:     axis : int, optional
64:         The axis along which the spline filter is applied. Default is the last
65:         axis.
66:     output : ndarray or dtype, optional
67:         The array in which to place the output, or the dtype of the returned
68:         array. Default is `numpy.float64`.
69: 
70:     Returns
71:     -------
72:     spline_filter1d : ndarray or None
73:         The filtered input. If `output` is given as a parameter, None is
74:         returned.
75: 
76:     '''
77:     if order < 0 or order > 5:
78:         raise RuntimeError('spline order not supported')
79:     input = numpy.asarray(input)
80:     if numpy.iscomplexobj(input):
81:         raise TypeError('Complex type not supported')
82:     output, return_value = _ni_support._get_output(output, input)
83:     if order in [0, 1]:
84:         output[...] = numpy.array(input)
85:     else:
86:         axis = _ni_support._check_axis(axis, input.ndim)
87:         _nd_image.spline_filter1d(input, order, axis, output)
88:     return return_value
89: 
90: 
91: def spline_filter(input, order=3, output=numpy.float64):
92:     '''
93:     Multi-dimensional spline filter.
94: 
95:     For more details, see `spline_filter1d`.
96: 
97:     See Also
98:     --------
99:     spline_filter1d
100: 
101:     Notes
102:     -----
103:     The multi-dimensional filter is implemented as a sequence of
104:     one-dimensional spline filters. The intermediate arrays are stored
105:     in the same data type as the output. Therefore, for output types
106:     with a limited precision, the results may be imprecise because
107:     intermediate results may be stored with insufficient precision.
108: 
109:     '''
110:     if order < 2 or order > 5:
111:         raise RuntimeError('spline order not supported')
112:     input = numpy.asarray(input)
113:     if numpy.iscomplexobj(input):
114:         raise TypeError('Complex type not supported')
115:     output, return_value = _ni_support._get_output(output, input)
116:     if order not in [0, 1] and input.ndim > 0:
117:         for axis in range(input.ndim):
118:             spline_filter1d(input, order, axis, output=output)
119:             input = output
120:     else:
121:         output[...] = input[...]
122:     return return_value
123: 
124: 
125: def geometric_transform(input, mapping, output_shape=None,
126:                         output=None, order=3,
127:                         mode='constant', cval=0.0, prefilter=True,
128:                         extra_arguments=(), extra_keywords={}):
129:     '''
130:     Apply an arbitrary geometric transform.
131: 
132:     The given mapping function is used to find, for each point in the
133:     output, the corresponding coordinates in the input. The value of the
134:     input at those coordinates is determined by spline interpolation of
135:     the requested order.
136: 
137:     Parameters
138:     ----------
139:     input : array_like
140:         The input array.
141:     mapping : {callable, scipy.LowLevelCallable}
142:         A callable object that accepts a tuple of length equal to the output
143:         array rank, and returns the corresponding input coordinates as a tuple
144:         of length equal to the input array rank.
145:     output_shape : tuple of ints, optional
146:         Shape tuple.
147:     output : ndarray or dtype, optional
148:         The array in which to place the output, or the dtype of the returned
149:         array.
150:     order : int, optional
151:         The order of the spline interpolation, default is 3.
152:         The order has to be in the range 0-5.
153:     mode : str, optional
154:         Points outside the boundaries of the input are filled according
155:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
156:         Default is 'constant'.
157:     cval : scalar, optional
158:         Value used for points outside the boundaries of the input if
159:         ``mode='constant'``. Default is 0.0
160:     prefilter : bool, optional
161:         The parameter prefilter determines if the input is pre-filtered with
162:         `spline_filter` before interpolation (necessary for spline
163:         interpolation of order > 1).  If False, it is assumed that the input is
164:         already filtered. Default is True.
165:     extra_arguments : tuple, optional
166:         Extra arguments passed to `mapping`.
167:     extra_keywords : dict, optional
168:         Extra keywords passed to `mapping`.
169: 
170:     Returns
171:     -------
172:     return_value : ndarray or None
173:         The filtered input. If `output` is given as a parameter, None is
174:         returned.
175: 
176:     See Also
177:     --------
178:     map_coordinates, affine_transform, spline_filter1d
179: 
180: 
181:     Notes
182:     -----
183:     This function also accepts low-level callback functions with one
184:     the following signatures and wrapped in `scipy.LowLevelCallable`:
185: 
186:     .. code:: c
187: 
188:        int mapping(npy_intp *output_coordinates, double *input_coordinates,
189:                    int output_rank, int input_rank, void *user_data)
190:        int mapping(intptr_t *output_coordinates, double *input_coordinates,
191:                    int output_rank, int input_rank, void *user_data)
192: 
193:     The calling function iterates over the elements of the output array,
194:     calling the callback function at each element. The coordinates of the
195:     current output element are passed through ``output_coordinates``. The
196:     callback function must return the coordinates at which the input must
197:     be interpolated in ``input_coordinates``. The rank of the input and
198:     output arrays are given by ``input_rank`` and ``output_rank``
199:     respectively.  ``user_data`` is the data pointer provided
200:     to `scipy.LowLevelCallable` as-is.
201: 
202:     The callback function must return an integer error status that is zero
203:     if something went wrong and one otherwise. If an error occurs, you should
204:     normally set the python error status with an informative message
205:     before returning, otherwise a default error message is set by the
206:     calling function.
207: 
208:     In addition, some other low-level function pointer specifications
209:     are accepted, but these are for backward compatibility only and should
210:     not be used in new code.
211: 
212:     Examples
213:     --------
214:     >>> from scipy import ndimage
215:     >>> a = np.arange(12.).reshape((4, 3))
216:     >>> def shift_func(output_coords):
217:     ...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)
218:     ...
219:     >>> ndimage.geometric_transform(a, shift_func)
220:     array([[ 0.   ,  0.   ,  0.   ],
221:            [ 0.   ,  1.362,  2.738],
222:            [ 0.   ,  4.812,  6.187],
223:            [ 0.   ,  8.263,  9.637]])
224: 
225:     '''
226:     if order < 0 or order > 5:
227:         raise RuntimeError('spline order not supported')
228:     input = numpy.asarray(input)
229:     if numpy.iscomplexobj(input):
230:         raise TypeError('Complex type not supported')
231:     if output_shape is None:
232:         output_shape = input.shape
233:     if input.ndim < 1 or len(output_shape) < 1:
234:         raise RuntimeError('input and output rank must be > 0')
235:     mode = _extend_mode_to_code(mode)
236:     if prefilter and order > 1:
237:         filtered = spline_filter(input, order, output=numpy.float64)
238:     else:
239:         filtered = input
240:     output, return_value = _ni_support._get_output(output, input,
241:                                                    shape=output_shape)
242:     _nd_image.geometric_transform(filtered, mapping, None, None, None, output,
243:                                   order, mode, cval, extra_arguments,
244:                                   extra_keywords)
245:     return return_value
246: 
247: 
248: def map_coordinates(input, coordinates, output=None, order=3,
249:                     mode='constant', cval=0.0, prefilter=True):
250:     '''
251:     Map the input array to new coordinates by interpolation.
252: 
253:     The array of coordinates is used to find, for each point in the output,
254:     the corresponding coordinates in the input. The value of the input at
255:     those coordinates is determined by spline interpolation of the
256:     requested order.
257: 
258:     The shape of the output is derived from that of the coordinate
259:     array by dropping the first axis. The values of the array along
260:     the first axis are the coordinates in the input array at which the
261:     output value is found.
262: 
263:     Parameters
264:     ----------
265:     input : ndarray
266:         The input array.
267:     coordinates : array_like
268:         The coordinates at which `input` is evaluated.
269:     output : ndarray or dtype, optional
270:         The array in which to place the output, or the dtype of the returned
271:         array.
272:     order : int, optional
273:         The order of the spline interpolation, default is 3.
274:         The order has to be in the range 0-5.
275:     mode : str, optional
276:         Points outside the boundaries of the input are filled according
277:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
278:         Default is 'constant'.
279:     cval : scalar, optional
280:         Value used for points outside the boundaries of the input if
281:         ``mode='constant'``. Default is 0.0
282:     prefilter : bool, optional
283:         The parameter prefilter determines if the input is pre-filtered with
284:         `spline_filter` before interpolation (necessary for spline
285:         interpolation of order > 1).  If False, it is assumed that the input is
286:         already filtered. Default is True.
287: 
288:     Returns
289:     -------
290:     map_coordinates : ndarray
291:         The result of transforming the input. The shape of the output is
292:         derived from that of `coordinates` by dropping the first axis.
293: 
294:     See Also
295:     --------
296:     spline_filter, geometric_transform, scipy.interpolate
297: 
298:     Examples
299:     --------
300:     >>> from scipy import ndimage
301:     >>> a = np.arange(12.).reshape((4, 3))
302:     >>> a
303:     array([[  0.,   1.,   2.],
304:            [  3.,   4.,   5.],
305:            [  6.,   7.,   8.],
306:            [  9.,  10.,  11.]])
307:     >>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
308:     array([ 2.,  7.])
309: 
310:     Above, the interpolated value of a[0.5, 0.5] gives output[0], while
311:     a[2, 1] is output[1].
312: 
313:     >>> inds = np.array([[0.5, 2], [0.5, 4]])
314:     >>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)
315:     array([  2. , -33.3])
316:     >>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')
317:     array([ 2.,  8.])
318:     >>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
319:     array([ True, False], dtype=bool)
320: 
321:     '''
322:     if order < 0 or order > 5:
323:         raise RuntimeError('spline order not supported')
324:     input = numpy.asarray(input)
325:     if numpy.iscomplexobj(input):
326:         raise TypeError('Complex type not supported')
327:     coordinates = numpy.asarray(coordinates)
328:     if numpy.iscomplexobj(coordinates):
329:         raise TypeError('Complex type not supported')
330:     output_shape = coordinates.shape[1:]
331:     if input.ndim < 1 or len(output_shape) < 1:
332:         raise RuntimeError('input and output rank must be > 0')
333:     if coordinates.shape[0] != input.ndim:
334:         raise RuntimeError('invalid shape for coordinate array')
335:     mode = _extend_mode_to_code(mode)
336:     if prefilter and order > 1:
337:         filtered = spline_filter(input, order, output=numpy.float64)
338:     else:
339:         filtered = input
340:     output, return_value = _ni_support._get_output(output, input,
341:                                                    shape=output_shape)
342:     _nd_image.geometric_transform(filtered, None, coordinates, None, None,
343:                                   output, order, mode, cval, None, None)
344:     return return_value
345: 
346: 
347: def affine_transform(input, matrix, offset=0.0, output_shape=None,
348:                      output=None, order=3,
349:                      mode='constant', cval=0.0, prefilter=True):
350:     '''
351:     Apply an affine transformation.
352: 
353:     Given an output image pixel index vector ``o``, the pixel value
354:     is determined from the input image at position
355:     ``np.dot(matrix, o) + offset``.
356: 
357:     Parameters
358:     ----------
359:     input : ndarray
360:         The input array.
361:     matrix : ndarray
362:         The inverse coordinate transformation matrix, mapping output
363:         coordinates to input coordinates. If ``ndim`` is the number of
364:         dimensions of ``input``, the given matrix must have one of the
365:         following shapes:
366: 
367:             - ``(ndim, ndim)``: the linear transformation matrix for each
368:               output coordinate.
369:             - ``(ndim,)``: assume that the 2D transformation matrix is
370:               diagonal, with the diagonal specified by the given value. A more
371:               efficient algorithm is then used that exploits the separability
372:               of the problem.
373:             - ``(ndim + 1, ndim + 1)``: assume that the transformation is
374:               specified using homogeneous coordinates [1]_. In this case, any
375:               value passed to ``offset`` is ignored.
376:             - ``(ndim, ndim + 1)``: as above, but the bottom row of a
377:               homogeneous transformation matrix is always ``[0, 0, ..., 1]``,
378:               and may be omitted.
379: 
380:     offset : float or sequence, optional
381:         The offset into the array where the transform is applied. If a float,
382:         `offset` is the same for each axis. If a sequence, `offset` should
383:         contain one value for each axis.
384:     output_shape : tuple of ints, optional
385:         Shape tuple.
386:     output : ndarray or dtype, optional
387:         The array in which to place the output, or the dtype of the returned
388:         array.
389:     order : int, optional
390:         The order of the spline interpolation, default is 3.
391:         The order has to be in the range 0-5.
392:     mode : str, optional
393:         Points outside the boundaries of the input are filled according
394:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or
395:         'wrap').
396:         Default is 'constant'.
397:     cval : scalar, optional
398:         Value used for points outside the boundaries of the input if
399:         ``mode='constant'``. Default is 0.0
400:     prefilter : bool, optional
401:         The parameter prefilter determines if the input is pre-filtered with
402:         `spline_filter` before interpolation (necessary for spline
403:         interpolation of order > 1).  If False, it is assumed that the input is
404:         already filtered. Default is True.
405: 
406:     Returns
407:     -------
408:     affine_transform : ndarray or None
409:         The transformed input. If `output` is given as a parameter, None is
410:         returned.
411: 
412:     Notes
413:     -----
414:     The given matrix and offset are used to find for each point in the
415:     output the corresponding coordinates in the input by an affine
416:     transformation. The value of the input at those coordinates is
417:     determined by spline interpolation of the requested order. Points
418:     outside the boundaries of the input are filled according to the given
419:     mode.
420: 
421:     .. versionchanged:: 0.18.0
422:         Previously, the exact interpretation of the affine transformation
423:         depended on whether the matrix was supplied as a one-dimensional or
424:         two-dimensional array. If a one-dimensional array was supplied
425:         to the matrix parameter, the output pixel value at index ``o``
426:         was determined from the input image at position
427:         ``matrix * (o + offset)``.
428: 
429:     References
430:     ----------
431:     .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates
432:     '''
433:     if order < 0 or order > 5:
434:         raise RuntimeError('spline order not supported')
435:     input = numpy.asarray(input)
436:     if numpy.iscomplexobj(input):
437:         raise TypeError('Complex type not supported')
438:     if output_shape is None:
439:         output_shape = input.shape
440:     if input.ndim < 1 or len(output_shape) < 1:
441:         raise RuntimeError('input and output rank must be > 0')
442:     mode = _extend_mode_to_code(mode)
443:     if prefilter and order > 1:
444:         filtered = spline_filter(input, order, output=numpy.float64)
445:     else:
446:         filtered = input
447:     output, return_value = _ni_support._get_output(output, input,
448:                                                    shape=output_shape)
449:     matrix = numpy.asarray(matrix, dtype=numpy.float64)
450:     if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
451:         raise RuntimeError('no proper affine matrix provided')
452:     if (matrix.ndim == 2 and matrix.shape[1] == input.ndim + 1 and
453:             (matrix.shape[0] in [input.ndim, input.ndim + 1])):
454:         if matrix.shape[0] == input.ndim + 1:
455:             exptd = [0] * input.ndim + [1]
456:             if not numpy.all(matrix[input.ndim] == exptd):
457:                 msg = ('Expected homogeneous transformation matrix with '
458:                        'shape %s for image shape %s, but bottom row was '
459:                        'not equal to %s' % (matrix.shape, input.shape, exptd))
460:                 raise ValueError(msg)
461:         # assume input is homogeneous coordinate transformation matrix
462:         offset = matrix[:input.ndim, input.ndim]
463:         matrix = matrix[:input.ndim, :input.ndim]
464:     if matrix.shape[0] != input.ndim:
465:         raise RuntimeError('affine matrix has wrong number of rows')
466:     if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
467:         raise RuntimeError('affine matrix has wrong number of columns')
468:     if not matrix.flags.contiguous:
469:         matrix = matrix.copy()
470:     offset = _ni_support._normalize_sequence(offset, input.ndim)
471:     offset = numpy.asarray(offset, dtype=numpy.float64)
472:     if offset.ndim != 1 or offset.shape[0] < 1:
473:         raise RuntimeError('no proper offset provided')
474:     if not offset.flags.contiguous:
475:         offset = offset.copy()
476:     if matrix.ndim == 1:
477:         warnings.warn(
478:             "The behaviour of affine_transform with a one-dimensional "
479:             "array supplied for the matrix parameter has changed in "
480:             "scipy 0.18.0."
481:         )
482:         _nd_image.zoom_shift(filtered, matrix, offset/matrix, output, order,
483:                              mode, cval)
484:     else:
485:         _nd_image.geometric_transform(filtered, None, None, matrix, offset,
486:                                       output, order, mode, cval, None, None)
487:     return return_value
488: 
489: 
490: def shift(input, shift, output=None, order=3, mode='constant', cval=0.0,
491:           prefilter=True):
492:     '''
493:     Shift an array.
494: 
495:     The array is shifted using spline interpolation of the requested order.
496:     Points outside the boundaries of the input are filled according to the
497:     given mode.
498: 
499:     Parameters
500:     ----------
501:     input : ndarray
502:         The input array.
503:     shift : float or sequence
504:         The shift along the axes. If a float, `shift` is the same for each
505:         axis. If a sequence, `shift` should contain one value for each axis.
506:     output : ndarray or dtype, optional
507:         The array in which to place the output, or the dtype of the returned
508:         array.
509:     order : int, optional
510:         The order of the spline interpolation, default is 3.
511:         The order has to be in the range 0-5.
512:     mode : str, optional
513:         Points outside the boundaries of the input are filled according
514:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
515:         Default is 'constant'.
516:     cval : scalar, optional
517:         Value used for points outside the boundaries of the input if
518:         ``mode='constant'``. Default is 0.0
519:     prefilter : bool, optional
520:         The parameter prefilter determines if the input is pre-filtered with
521:         `spline_filter` before interpolation (necessary for spline
522:         interpolation of order > 1).  If False, it is assumed that the input is
523:         already filtered. Default is True.
524: 
525:     Returns
526:     -------
527:     shift : ndarray or None
528:         The shifted input. If `output` is given as a parameter, None is
529:         returned.
530: 
531:     '''
532:     if order < 0 or order > 5:
533:         raise RuntimeError('spline order not supported')
534:     input = numpy.asarray(input)
535:     if numpy.iscomplexobj(input):
536:         raise TypeError('Complex type not supported')
537:     if input.ndim < 1:
538:         raise RuntimeError('input and output rank must be > 0')
539:     mode = _extend_mode_to_code(mode)
540:     if prefilter and order > 1:
541:         filtered = spline_filter(input, order, output=numpy.float64)
542:     else:
543:         filtered = input
544:     output, return_value = _ni_support._get_output(output, input)
545:     shift = _ni_support._normalize_sequence(shift, input.ndim)
546:     shift = [-ii for ii in shift]
547:     shift = numpy.asarray(shift, dtype=numpy.float64)
548:     if not shift.flags.contiguous:
549:         shift = shift.copy()
550:     _nd_image.zoom_shift(filtered, None, shift, output, order, mode, cval)
551:     return return_value
552: 
553: 
554: def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
555:          prefilter=True):
556:     '''
557:     Zoom an array.
558: 
559:     The array is zoomed using spline interpolation of the requested order.
560: 
561:     Parameters
562:     ----------
563:     input : ndarray
564:         The input array.
565:     zoom : float or sequence
566:         The zoom factor along the axes. If a float, `zoom` is the same for each
567:         axis. If a sequence, `zoom` should contain one value for each axis.
568:     output : ndarray or dtype, optional
569:         The array in which to place the output, or the dtype of the returned
570:         array.
571:     order : int, optional
572:         The order of the spline interpolation, default is 3.
573:         The order has to be in the range 0-5.
574:     mode : str, optional
575:         Points outside the boundaries of the input are filled according
576:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
577:         Default is 'constant'.
578:     cval : scalar, optional
579:         Value used for points outside the boundaries of the input if
580:         ``mode='constant'``. Default is 0.0
581:     prefilter : bool, optional
582:         The parameter prefilter determines if the input is pre-filtered with
583:         `spline_filter` before interpolation (necessary for spline
584:         interpolation of order > 1).  If False, it is assumed that the input is
585:         already filtered. Default is True.
586: 
587:     Returns
588:     -------
589:     zoom : ndarray or None
590:         The zoomed input. If `output` is given as a parameter, None is
591:         returned.
592: 
593:     '''
594:     if order < 0 or order > 5:
595:         raise RuntimeError('spline order not supported')
596:     input = numpy.asarray(input)
597:     if numpy.iscomplexobj(input):
598:         raise TypeError('Complex type not supported')
599:     if input.ndim < 1:
600:         raise RuntimeError('input and output rank must be > 0')
601:     mode = _extend_mode_to_code(mode)
602:     if prefilter and order > 1:
603:         filtered = spline_filter(input, order, output=numpy.float64)
604:     else:
605:         filtered = input
606:     zoom = _ni_support._normalize_sequence(zoom, input.ndim)
607:     output_shape = tuple(
608:             [int(round(ii * jj)) for ii, jj in zip(input.shape, zoom)])
609: 
610:     output_shape_old = tuple(
611:             [int(ii * jj) for ii, jj in zip(input.shape, zoom)])
612:     if output_shape != output_shape_old:
613:         warnings.warn(
614:                 "From scipy 0.13.0, the output shape of zoom() is calculated "
615:                 "with round() instead of int() - for these inputs the size of "
616:                 "the returned array has changed.", UserWarning)
617: 
618:     zoom_div = numpy.array(output_shape, float) - 1
619:     # Zooming to infinite values is unpredictable, so just choose
620:     # zoom factor 1 instead
621:     zoom = numpy.divide(numpy.array(input.shape) - 1, zoom_div,
622:                         out=numpy.ones_like(input.shape, dtype=numpy.float64),
623:                         where=zoom_div != 0)
624: 
625:     output, return_value = _ni_support._get_output(output, input,
626:                                                    shape=output_shape)
627:     zoom = numpy.ascontiguousarray(zoom)
628:     _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval)
629:     return return_value
630: 
631: 
632: def _minmax(coor, minc, maxc):
633:     if coor[0] < minc[0]:
634:         minc[0] = coor[0]
635:     if coor[0] > maxc[0]:
636:         maxc[0] = coor[0]
637:     if coor[1] < minc[1]:
638:         minc[1] = coor[1]
639:     if coor[1] > maxc[1]:
640:         maxc[1] = coor[1]
641:     return minc, maxc
642: 
643: 
644: def rotate(input, angle, axes=(1, 0), reshape=True,
645:            output=None, order=3,
646:            mode='constant', cval=0.0, prefilter=True):
647:     '''
648:     Rotate an array.
649: 
650:     The array is rotated in the plane defined by the two axes given by the
651:     `axes` parameter using spline interpolation of the requested order.
652: 
653:     Parameters
654:     ----------
655:     input : ndarray
656:         The input array.
657:     angle : float
658:         The rotation angle in degrees.
659:     axes : tuple of 2 ints, optional
660:         The two axes that define the plane of rotation. Default is the first
661:         two axes.
662:     reshape : bool, optional
663:         If `reshape` is true, the output shape is adapted so that the input
664:         array is contained completely in the output. Default is True.
665:     output : ndarray or dtype, optional
666:         The array in which to place the output, or the dtype of the returned
667:         array.
668:     order : int, optional
669:         The order of the spline interpolation, default is 3.
670:         The order has to be in the range 0-5.
671:     mode : str, optional
672:         Points outside the boundaries of the input are filled according
673:         to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').
674:         Default is 'constant'.
675:     cval : scalar, optional
676:         Value used for points outside the boundaries of the input if
677:         ``mode='constant'``. Default is 0.0
678:     prefilter : bool, optional
679:         The parameter prefilter determines if the input is pre-filtered with
680:         `spline_filter` before interpolation (necessary for spline
681:         interpolation of order > 1).  If False, it is assumed that the input is
682:         already filtered. Default is True.
683: 
684:     Returns
685:     -------
686:     rotate : ndarray or None
687:         The rotated input. If `output` is given as a parameter, None is
688:         returned.
689: 
690:     '''
691:     input = numpy.asarray(input)
692:     axes = list(axes)
693:     rank = input.ndim
694:     if axes[0] < 0:
695:         axes[0] += rank
696:     if axes[1] < 0:
697:         axes[1] += rank
698:     if axes[0] < 0 or axes[1] < 0 or axes[0] > rank or axes[1] > rank:
699:         raise RuntimeError('invalid rotation plane specified')
700:     if axes[0] > axes[1]:
701:         axes = axes[1], axes[0]
702:     angle = numpy.pi / 180 * angle
703:     m11 = math.cos(angle)
704:     m12 = math.sin(angle)
705:     m21 = -math.sin(angle)
706:     m22 = math.cos(angle)
707:     matrix = numpy.array([[m11, m12],
708:                              [m21, m22]], dtype=numpy.float64)
709:     iy = input.shape[axes[0]]
710:     ix = input.shape[axes[1]]
711:     if reshape:
712:         mtrx = numpy.array([[m11, -m21],
713:                                [-m12, m22]], dtype=numpy.float64)
714:         minc = [0, 0]
715:         maxc = [0, 0]
716:         coor = numpy.dot(mtrx, [0, ix])
717:         minc, maxc = _minmax(coor, minc, maxc)
718:         coor = numpy.dot(mtrx, [iy, 0])
719:         minc, maxc = _minmax(coor, minc, maxc)
720:         coor = numpy.dot(mtrx, [iy, ix])
721:         minc, maxc = _minmax(coor, minc, maxc)
722:         oy = int(maxc[0] - minc[0] + 0.5)
723:         ox = int(maxc[1] - minc[1] + 0.5)
724:     else:
725:         oy = input.shape[axes[0]]
726:         ox = input.shape[axes[1]]
727:     offset = numpy.zeros((2,), dtype=numpy.float64)
728:     offset[0] = float(oy) / 2.0 - 0.5
729:     offset[1] = float(ox) / 2.0 - 0.5
730:     offset = numpy.dot(matrix, offset)
731:     tmp = numpy.zeros((2,), dtype=numpy.float64)
732:     tmp[0] = float(iy) / 2.0 - 0.5
733:     tmp[1] = float(ix) / 2.0 - 0.5
734:     offset = tmp - offset
735:     output_shape = list(input.shape)
736:     output_shape[axes[0]] = oy
737:     output_shape[axes[1]] = ox
738:     output_shape = tuple(output_shape)
739:     output, return_value = _ni_support._get_output(output, input,
740:                                                    shape=output_shape)
741:     if input.ndim <= 2:
742:         affine_transform(input, matrix, offset, output_shape, output,
743:                          order, mode, cval, prefilter)
744:     else:
745:         coordinates = []
746:         size = numpy.product(input.shape,axis=0)
747:         size //= input.shape[axes[0]]
748:         size //= input.shape[axes[1]]
749:         for ii in range(input.ndim):
750:             if ii not in axes:
751:                 coordinates.append(0)
752:             else:
753:                 coordinates.append(slice(None, None, None))
754:         iter_axes = list(range(input.ndim))
755:         iter_axes.reverse()
756:         iter_axes.remove(axes[0])
757:         iter_axes.remove(axes[1])
758:         os = (output_shape[axes[0]], output_shape[axes[1]])
759:         for ii in range(size):
760:             ia = input[tuple(coordinates)]
761:             oa = output[tuple(coordinates)]
762:             affine_transform(ia, matrix, offset, os, oa, order, mode,
763:                              cval, prefilter)
764:             for jj in iter_axes:
765:                 if coordinates[jj] < input.shape[jj] - 1:
766:                     coordinates[jj] += 1
767:                     break
768:                 else:
769:                     coordinates[jj] = 0
770:     return return_value
771: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'import math' statement (line 33)
import math

import_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'math', math, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import numpy' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_120093 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_120093) is not StypyTypeError):

    if (import_120093 != 'pyd_module'):
        __import__(import_120093)
        sys_modules_120094 = sys.modules[import_120093]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', sys_modules_120094.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_120093)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.ndimage import _ni_support' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_120095 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage')

if (type(import_120095) is not StypyTypeError):

    if (import_120095 != 'pyd_module'):
        __import__(import_120095)
        sys_modules_120096 = sys.modules[import_120095]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', sys_modules_120096.module_type_store, module_type_store, ['_ni_support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_120096, sys_modules_120096.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_support

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', None, module_type_store, ['_ni_support'], [_ni_support])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', import_120095)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.ndimage import _nd_image' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_120097 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage')

if (type(import_120097) is not StypyTypeError):

    if (import_120097 != 'pyd_module'):
        __import__(import_120097)
        sys_modules_120098 = sys.modules[import_120097]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', sys_modules_120098.module_type_store, module_type_store, ['_nd_image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_120098, sys_modules_120098.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _nd_image

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', None, module_type_store, ['_nd_image'], [_nd_image])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', import_120097)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from functools import wraps' statement (line 37)
try:
    from functools import wraps

except:
    wraps = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'functools', None, module_type_store, ['wraps'], [wraps])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'import warnings' statement (line 39)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'warnings', warnings, module_type_store)


# Assigning a List to a Name (line 41):

# Assigning a List to a Name (line 41):
__all__ = ['spline_filter1d', 'spline_filter', 'geometric_transform', 'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate']
module_type_store.set_exportable_members(['spline_filter1d', 'spline_filter', 'geometric_transform', 'map_coordinates', 'affine_transform', 'shift', 'zoom', 'rotate'])

# Obtaining an instance of the builtin type 'list' (line 41)
list_120099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_120100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'spline_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120100)
# Adding element type (line 41)
str_120101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'str', 'spline_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120101)
# Adding element type (line 41)
str_120102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 47), 'str', 'geometric_transform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120102)
# Adding element type (line 41)
str_120103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'map_coordinates')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120103)
# Adding element type (line 41)
str_120104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'str', 'affine_transform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120104)
# Adding element type (line 41)
str_120105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'str', 'shift')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120105)
# Adding element type (line 41)
str_120106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 59), 'str', 'zoom')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120106)
# Adding element type (line 41)
str_120107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 67), 'str', 'rotate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_120099, str_120107)

# Assigning a type to the variable '__all__' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '__all__', list_120099)

@norecursion
def _extend_mode_to_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_extend_mode_to_code'
    module_type_store = module_type_store.open_function_context('_extend_mode_to_code', 45, 0, False)
    
    # Passed parameters checking function
    _extend_mode_to_code.stypy_localization = localization
    _extend_mode_to_code.stypy_type_of_self = None
    _extend_mode_to_code.stypy_type_store = module_type_store
    _extend_mode_to_code.stypy_function_name = '_extend_mode_to_code'
    _extend_mode_to_code.stypy_param_names_list = ['mode']
    _extend_mode_to_code.stypy_varargs_param_name = None
    _extend_mode_to_code.stypy_kwargs_param_name = None
    _extend_mode_to_code.stypy_call_defaults = defaults
    _extend_mode_to_code.stypy_call_varargs = varargs
    _extend_mode_to_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_extend_mode_to_code', ['mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_extend_mode_to_code', localization, ['mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_extend_mode_to_code(...)' code ##################

    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to _extend_mode_to_code(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'mode' (line 46)
    mode_120110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'mode', False)
    # Processing the call keyword arguments (line 46)
    kwargs_120111 = {}
    # Getting the type of '_ni_support' (line 46)
    _ni_support_120108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 46)
    _extend_mode_to_code_120109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), _ni_support_120108, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 46)
    _extend_mode_to_code_call_result_120112 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), _extend_mode_to_code_120109, *[mode_120110], **kwargs_120111)
    
    # Assigning a type to the variable 'mode' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'mode', _extend_mode_to_code_call_result_120112)
    # Getting the type of 'mode' (line 47)
    mode_120113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'mode')
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type', mode_120113)
    
    # ################# End of '_extend_mode_to_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_extend_mode_to_code' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_120114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120114)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_extend_mode_to_code'
    return stypy_return_type_120114

# Assigning a type to the variable '_extend_mode_to_code' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '_extend_mode_to_code', _extend_mode_to_code)

@norecursion
def spline_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_120115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'int')
    int_120116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 41), 'int')
    # Getting the type of 'numpy' (line 50)
    numpy_120117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 52), 'numpy')
    # Obtaining the member 'float64' of a type (line 50)
    float64_120118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 52), numpy_120117, 'float64')
    defaults = [int_120115, int_120116, float64_120118]
    # Create a new context for function 'spline_filter1d'
    module_type_store = module_type_store.open_function_context('spline_filter1d', 50, 0, False)
    
    # Passed parameters checking function
    spline_filter1d.stypy_localization = localization
    spline_filter1d.stypy_type_of_self = None
    spline_filter1d.stypy_type_store = module_type_store
    spline_filter1d.stypy_function_name = 'spline_filter1d'
    spline_filter1d.stypy_param_names_list = ['input', 'order', 'axis', 'output']
    spline_filter1d.stypy_varargs_param_name = None
    spline_filter1d.stypy_kwargs_param_name = None
    spline_filter1d.stypy_call_defaults = defaults
    spline_filter1d.stypy_call_varargs = varargs
    spline_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spline_filter1d', ['input', 'order', 'axis', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spline_filter1d', localization, ['input', 'order', 'axis', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spline_filter1d(...)' code ##################

    str_120119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n    Calculates a one-dimensional spline filter along the given axis.\n\n    The lines of the array along the given axis are filtered by a\n    spline filter. The order of the spline must be >= 2 and <= 5.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    order : int, optional\n        The order of the spline, default is 3.\n    axis : int, optional\n        The axis along which the spline filter is applied. Default is the last\n        axis.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array. Default is `numpy.float64`.\n\n    Returns\n    -------\n    spline_filter1d : ndarray or None\n        The filtered input. If `output` is given as a parameter, None is\n        returned.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 77)
    order_120120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'order')
    int_120121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 15), 'int')
    # Applying the binary operator '<' (line 77)
    result_lt_120122 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), '<', order_120120, int_120121)
    
    
    # Getting the type of 'order' (line 77)
    order_120123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'order')
    int_120124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 28), 'int')
    # Applying the binary operator '>' (line 77)
    result_gt_120125 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 20), '>', order_120123, int_120124)
    
    # Applying the binary operator 'or' (line 77)
    result_or_keyword_120126 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), 'or', result_lt_120122, result_gt_120125)
    
    # Testing the type of an if condition (line 77)
    if_condition_120127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_or_keyword_120126)
    # Assigning a type to the variable 'if_condition_120127' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_120127', if_condition_120127)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 78)
    # Processing the call arguments (line 78)
    str_120129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 78)
    kwargs_120130 = {}
    # Getting the type of 'RuntimeError' (line 78)
    RuntimeError_120128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 78)
    RuntimeError_call_result_120131 = invoke(stypy.reporting.localization.Localization(__file__, 78, 14), RuntimeError_120128, *[str_120129], **kwargs_120130)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 8), RuntimeError_call_result_120131, 'raise parameter', BaseException)
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to asarray(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'input' (line 79)
    input_120134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'input', False)
    # Processing the call keyword arguments (line 79)
    kwargs_120135 = {}
    # Getting the type of 'numpy' (line 79)
    numpy_120132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 79)
    asarray_120133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), numpy_120132, 'asarray')
    # Calling asarray(args, kwargs) (line 79)
    asarray_call_result_120136 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), asarray_120133, *[input_120134], **kwargs_120135)
    
    # Assigning a type to the variable 'input' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'input', asarray_call_result_120136)
    
    
    # Call to iscomplexobj(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'input' (line 80)
    input_120139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'input', False)
    # Processing the call keyword arguments (line 80)
    kwargs_120140 = {}
    # Getting the type of 'numpy' (line 80)
    numpy_120137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 80)
    iscomplexobj_120138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 7), numpy_120137, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 80)
    iscomplexobj_call_result_120141 = invoke(stypy.reporting.localization.Localization(__file__, 80, 7), iscomplexobj_120138, *[input_120139], **kwargs_120140)
    
    # Testing the type of an if condition (line 80)
    if_condition_120142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), iscomplexobj_call_result_120141)
    # Assigning a type to the variable 'if_condition_120142' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_120142', if_condition_120142)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 81)
    # Processing the call arguments (line 81)
    str_120144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 81)
    kwargs_120145 = {}
    # Getting the type of 'TypeError' (line 81)
    TypeError_120143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 81)
    TypeError_call_result_120146 = invoke(stypy.reporting.localization.Localization(__file__, 81, 14), TypeError_120143, *[str_120144], **kwargs_120145)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 81, 8), TypeError_call_result_120146, 'raise parameter', BaseException)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 82):
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_120147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'int')
    
    # Call to _get_output(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'output' (line 82)
    output_120150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'output', False)
    # Getting the type of 'input' (line 82)
    input_120151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 59), 'input', False)
    # Processing the call keyword arguments (line 82)
    kwargs_120152 = {}
    # Getting the type of '_ni_support' (line 82)
    _ni_support_120148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 82)
    _get_output_120149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), _ni_support_120148, '_get_output')
    # Calling _get_output(args, kwargs) (line 82)
    _get_output_call_result_120153 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), _get_output_120149, *[output_120150, input_120151], **kwargs_120152)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___120154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), _get_output_call_result_120153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_120155 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), getitem___120154, int_120147)
    
    # Assigning a type to the variable 'tuple_var_assignment_120071' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_var_assignment_120071', subscript_call_result_120155)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_120156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'int')
    
    # Call to _get_output(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'output' (line 82)
    output_120159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'output', False)
    # Getting the type of 'input' (line 82)
    input_120160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 59), 'input', False)
    # Processing the call keyword arguments (line 82)
    kwargs_120161 = {}
    # Getting the type of '_ni_support' (line 82)
    _ni_support_120157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 82)
    _get_output_120158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), _ni_support_120157, '_get_output')
    # Calling _get_output(args, kwargs) (line 82)
    _get_output_call_result_120162 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), _get_output_120158, *[output_120159, input_120160], **kwargs_120161)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___120163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), _get_output_call_result_120162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_120164 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), getitem___120163, int_120156)
    
    # Assigning a type to the variable 'tuple_var_assignment_120072' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_var_assignment_120072', subscript_call_result_120164)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_120071' (line 82)
    tuple_var_assignment_120071_120165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_var_assignment_120071')
    # Assigning a type to the variable 'output' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'output', tuple_var_assignment_120071_120165)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_120072' (line 82)
    tuple_var_assignment_120072_120166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'tuple_var_assignment_120072')
    # Assigning a type to the variable 'return_value' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'return_value', tuple_var_assignment_120072_120166)
    
    
    # Getting the type of 'order' (line 83)
    order_120167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 7), 'order')
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_120168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    int_120169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), list_120168, int_120169)
    # Adding element type (line 83)
    int_120170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), list_120168, int_120170)
    
    # Applying the binary operator 'in' (line 83)
    result_contains_120171 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 7), 'in', order_120167, list_120168)
    
    # Testing the type of an if condition (line 83)
    if_condition_120172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 4), result_contains_120171)
    # Assigning a type to the variable 'if_condition_120172' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'if_condition_120172', if_condition_120172)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 84):
    
    # Assigning a Call to a Subscript (line 84):
    
    # Call to array(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'input' (line 84)
    input_120175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'input', False)
    # Processing the call keyword arguments (line 84)
    kwargs_120176 = {}
    # Getting the type of 'numpy' (line 84)
    numpy_120173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'numpy', False)
    # Obtaining the member 'array' of a type (line 84)
    array_120174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 22), numpy_120173, 'array')
    # Calling array(args, kwargs) (line 84)
    array_call_result_120177 = invoke(stypy.reporting.localization.Localization(__file__, 84, 22), array_120174, *[input_120175], **kwargs_120176)
    
    # Getting the type of 'output' (line 84)
    output_120178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'output')
    Ellipsis_120179 = Ellipsis
    # Storing an element on a container (line 84)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), output_120178, (Ellipsis_120179, array_call_result_120177))
    # SSA branch for the else part of an if statement (line 83)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to _check_axis(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'axis' (line 86)
    axis_120182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'axis', False)
    # Getting the type of 'input' (line 86)
    input_120183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 45), 'input', False)
    # Obtaining the member 'ndim' of a type (line 86)
    ndim_120184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 45), input_120183, 'ndim')
    # Processing the call keyword arguments (line 86)
    kwargs_120185 = {}
    # Getting the type of '_ni_support' (line 86)
    _ni_support_120180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 86)
    _check_axis_120181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), _ni_support_120180, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 86)
    _check_axis_call_result_120186 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), _check_axis_120181, *[axis_120182, ndim_120184], **kwargs_120185)
    
    # Assigning a type to the variable 'axis' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'axis', _check_axis_call_result_120186)
    
    # Call to spline_filter1d(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'input' (line 87)
    input_120189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'input', False)
    # Getting the type of 'order' (line 87)
    order_120190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'order', False)
    # Getting the type of 'axis' (line 87)
    axis_120191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 48), 'axis', False)
    # Getting the type of 'output' (line 87)
    output_120192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 54), 'output', False)
    # Processing the call keyword arguments (line 87)
    kwargs_120193 = {}
    # Getting the type of '_nd_image' (line 87)
    _nd_image_120187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), '_nd_image', False)
    # Obtaining the member 'spline_filter1d' of a type (line 87)
    spline_filter1d_120188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), _nd_image_120187, 'spline_filter1d')
    # Calling spline_filter1d(args, kwargs) (line 87)
    spline_filter1d_call_result_120194 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), spline_filter1d_120188, *[input_120189, order_120190, axis_120191, output_120192], **kwargs_120193)
    
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 88)
    return_value_120195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', return_value_120195)
    
    # ################# End of 'spline_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spline_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_120196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spline_filter1d'
    return stypy_return_type_120196

# Assigning a type to the variable 'spline_filter1d' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'spline_filter1d', spline_filter1d)

@norecursion
def spline_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_120197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'int')
    # Getting the type of 'numpy' (line 91)
    numpy_120198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 41), 'numpy')
    # Obtaining the member 'float64' of a type (line 91)
    float64_120199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 41), numpy_120198, 'float64')
    defaults = [int_120197, float64_120199]
    # Create a new context for function 'spline_filter'
    module_type_store = module_type_store.open_function_context('spline_filter', 91, 0, False)
    
    # Passed parameters checking function
    spline_filter.stypy_localization = localization
    spline_filter.stypy_type_of_self = None
    spline_filter.stypy_type_store = module_type_store
    spline_filter.stypy_function_name = 'spline_filter'
    spline_filter.stypy_param_names_list = ['input', 'order', 'output']
    spline_filter.stypy_varargs_param_name = None
    spline_filter.stypy_kwargs_param_name = None
    spline_filter.stypy_call_defaults = defaults
    spline_filter.stypy_call_varargs = varargs
    spline_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spline_filter', ['input', 'order', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spline_filter', localization, ['input', 'order', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spline_filter(...)' code ##################

    str_120200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Multi-dimensional spline filter.\n\n    For more details, see `spline_filter1d`.\n\n    See Also\n    --------\n    spline_filter1d\n\n    Notes\n    -----\n    The multi-dimensional filter is implemented as a sequence of\n    one-dimensional spline filters. The intermediate arrays are stored\n    in the same data type as the output. Therefore, for output types\n    with a limited precision, the results may be imprecise because\n    intermediate results may be stored with insufficient precision.\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 110)
    order_120201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'order')
    int_120202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
    # Applying the binary operator '<' (line 110)
    result_lt_120203 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), '<', order_120201, int_120202)
    
    
    # Getting the type of 'order' (line 110)
    order_120204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'order')
    int_120205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 28), 'int')
    # Applying the binary operator '>' (line 110)
    result_gt_120206 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 20), '>', order_120204, int_120205)
    
    # Applying the binary operator 'or' (line 110)
    result_or_keyword_120207 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), 'or', result_lt_120203, result_gt_120206)
    
    # Testing the type of an if condition (line 110)
    if_condition_120208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_or_keyword_120207)
    # Assigning a type to the variable 'if_condition_120208' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_120208', if_condition_120208)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 111)
    # Processing the call arguments (line 111)
    str_120210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 111)
    kwargs_120211 = {}
    # Getting the type of 'RuntimeError' (line 111)
    RuntimeError_120209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 111)
    RuntimeError_call_result_120212 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), RuntimeError_120209, *[str_120210], **kwargs_120211)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 8), RuntimeError_call_result_120212, 'raise parameter', BaseException)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to asarray(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'input' (line 112)
    input_120215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'input', False)
    # Processing the call keyword arguments (line 112)
    kwargs_120216 = {}
    # Getting the type of 'numpy' (line 112)
    numpy_120213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 112)
    asarray_120214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), numpy_120213, 'asarray')
    # Calling asarray(args, kwargs) (line 112)
    asarray_call_result_120217 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), asarray_120214, *[input_120215], **kwargs_120216)
    
    # Assigning a type to the variable 'input' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'input', asarray_call_result_120217)
    
    
    # Call to iscomplexobj(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'input' (line 113)
    input_120220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 26), 'input', False)
    # Processing the call keyword arguments (line 113)
    kwargs_120221 = {}
    # Getting the type of 'numpy' (line 113)
    numpy_120218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 113)
    iscomplexobj_120219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 7), numpy_120218, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 113)
    iscomplexobj_call_result_120222 = invoke(stypy.reporting.localization.Localization(__file__, 113, 7), iscomplexobj_120219, *[input_120220], **kwargs_120221)
    
    # Testing the type of an if condition (line 113)
    if_condition_120223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), iscomplexobj_call_result_120222)
    # Assigning a type to the variable 'if_condition_120223' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_120223', if_condition_120223)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 114)
    # Processing the call arguments (line 114)
    str_120225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 114)
    kwargs_120226 = {}
    # Getting the type of 'TypeError' (line 114)
    TypeError_120224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 114)
    TypeError_call_result_120227 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), TypeError_120224, *[str_120225], **kwargs_120226)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 8), TypeError_call_result_120227, 'raise parameter', BaseException)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 115):
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_120228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to _get_output(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'output' (line 115)
    output_120231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), 'output', False)
    # Getting the type of 'input' (line 115)
    input_120232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'input', False)
    # Processing the call keyword arguments (line 115)
    kwargs_120233 = {}
    # Getting the type of '_ni_support' (line 115)
    _ni_support_120229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 115)
    _get_output_120230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), _ni_support_120229, '_get_output')
    # Calling _get_output(args, kwargs) (line 115)
    _get_output_call_result_120234 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), _get_output_120230, *[output_120231, input_120232], **kwargs_120233)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___120235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), _get_output_call_result_120234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_120236 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___120235, int_120228)
    
    # Assigning a type to the variable 'tuple_var_assignment_120073' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_120073', subscript_call_result_120236)
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_120237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 4), 'int')
    
    # Call to _get_output(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'output' (line 115)
    output_120240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), 'output', False)
    # Getting the type of 'input' (line 115)
    input_120241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'input', False)
    # Processing the call keyword arguments (line 115)
    kwargs_120242 = {}
    # Getting the type of '_ni_support' (line 115)
    _ni_support_120238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 115)
    _get_output_120239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), _ni_support_120238, '_get_output')
    # Calling _get_output(args, kwargs) (line 115)
    _get_output_call_result_120243 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), _get_output_120239, *[output_120240, input_120241], **kwargs_120242)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___120244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), _get_output_call_result_120243, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_120245 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), getitem___120244, int_120237)
    
    # Assigning a type to the variable 'tuple_var_assignment_120074' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_120074', subscript_call_result_120245)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_120073' (line 115)
    tuple_var_assignment_120073_120246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_120073')
    # Assigning a type to the variable 'output' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'output', tuple_var_assignment_120073_120246)
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'tuple_var_assignment_120074' (line 115)
    tuple_var_assignment_120074_120247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'tuple_var_assignment_120074')
    # Assigning a type to the variable 'return_value' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'return_value', tuple_var_assignment_120074_120247)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 116)
    order_120248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'order')
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_120249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    int_120250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_120249, int_120250)
    # Adding element type (line 116)
    int_120251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 20), list_120249, int_120251)
    
    # Applying the binary operator 'notin' (line 116)
    result_contains_120252 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'notin', order_120248, list_120249)
    
    
    # Getting the type of 'input' (line 116)
    input_120253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'input')
    # Obtaining the member 'ndim' of a type (line 116)
    ndim_120254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 31), input_120253, 'ndim')
    int_120255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 44), 'int')
    # Applying the binary operator '>' (line 116)
    result_gt_120256 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), '>', ndim_120254, int_120255)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_120257 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'and', result_contains_120252, result_gt_120256)
    
    # Testing the type of an if condition (line 116)
    if_condition_120258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_and_keyword_120257)
    # Assigning a type to the variable 'if_condition_120258' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_120258', if_condition_120258)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'input' (line 117)
    input_120260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 26), 'input', False)
    # Obtaining the member 'ndim' of a type (line 117)
    ndim_120261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 26), input_120260, 'ndim')
    # Processing the call keyword arguments (line 117)
    kwargs_120262 = {}
    # Getting the type of 'range' (line 117)
    range_120259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'range', False)
    # Calling range(args, kwargs) (line 117)
    range_call_result_120263 = invoke(stypy.reporting.localization.Localization(__file__, 117, 20), range_120259, *[ndim_120261], **kwargs_120262)
    
    # Testing the type of a for loop iterable (line 117)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), range_call_result_120263)
    # Getting the type of the for loop variable (line 117)
    for_loop_var_120264 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), range_call_result_120263)
    # Assigning a type to the variable 'axis' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'axis', for_loop_var_120264)
    # SSA begins for a for statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to spline_filter1d(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'input' (line 118)
    input_120266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'input', False)
    # Getting the type of 'order' (line 118)
    order_120267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'order', False)
    # Getting the type of 'axis' (line 118)
    axis_120268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'axis', False)
    # Processing the call keyword arguments (line 118)
    # Getting the type of 'output' (line 118)
    output_120269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 55), 'output', False)
    keyword_120270 = output_120269
    kwargs_120271 = {'output': keyword_120270}
    # Getting the type of 'spline_filter1d' (line 118)
    spline_filter1d_120265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'spline_filter1d', False)
    # Calling spline_filter1d(args, kwargs) (line 118)
    spline_filter1d_call_result_120272 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), spline_filter1d_120265, *[input_120266, order_120267, axis_120268], **kwargs_120271)
    
    
    # Assigning a Name to a Name (line 119):
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'output' (line 119)
    output_120273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'output')
    # Assigning a type to the variable 'input' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'input', output_120273)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 116)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 121):
    
    # Assigning a Subscript to a Subscript (line 121):
    
    # Obtaining the type of the subscript
    Ellipsis_120274 = Ellipsis
    # Getting the type of 'input' (line 121)
    input_120275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___120276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 22), input_120275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_120277 = invoke(stypy.reporting.localization.Localization(__file__, 121, 22), getitem___120276, Ellipsis_120274)
    
    # Getting the type of 'output' (line 121)
    output_120278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'output')
    Ellipsis_120279 = Ellipsis
    # Storing an element on a container (line 121)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 8), output_120278, (Ellipsis_120279, subscript_call_result_120277))
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 122)
    return_value_120280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', return_value_120280)
    
    # ################# End of 'spline_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spline_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_120281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spline_filter'
    return stypy_return_type_120281

# Assigning a type to the variable 'spline_filter' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'spline_filter', spline_filter)

@norecursion
def geometric_transform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 125)
    None_120282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 53), 'None')
    # Getting the type of 'None' (line 126)
    None_120283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'None')
    int_120284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 43), 'int')
    str_120285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'str', 'constant')
    float_120286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 46), 'float')
    # Getting the type of 'True' (line 127)
    True_120287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 61), 'True')
    
    # Obtaining an instance of the builtin type 'tuple' (line 128)
    tuple_120288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 128)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 128)
    dict_120289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 59), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 128)
    
    defaults = [None_120282, None_120283, int_120284, str_120285, float_120286, True_120287, tuple_120288, dict_120289]
    # Create a new context for function 'geometric_transform'
    module_type_store = module_type_store.open_function_context('geometric_transform', 125, 0, False)
    
    # Passed parameters checking function
    geometric_transform.stypy_localization = localization
    geometric_transform.stypy_type_of_self = None
    geometric_transform.stypy_type_store = module_type_store
    geometric_transform.stypy_function_name = 'geometric_transform'
    geometric_transform.stypy_param_names_list = ['input', 'mapping', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter', 'extra_arguments', 'extra_keywords']
    geometric_transform.stypy_varargs_param_name = None
    geometric_transform.stypy_kwargs_param_name = None
    geometric_transform.stypy_call_defaults = defaults
    geometric_transform.stypy_call_varargs = varargs
    geometric_transform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'geometric_transform', ['input', 'mapping', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter', 'extra_arguments', 'extra_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'geometric_transform', localization, ['input', 'mapping', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter', 'extra_arguments', 'extra_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'geometric_transform(...)' code ##################

    str_120290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, (-1)), 'str', "\n    Apply an arbitrary geometric transform.\n\n    The given mapping function is used to find, for each point in the\n    output, the corresponding coordinates in the input. The value of the\n    input at those coordinates is determined by spline interpolation of\n    the requested order.\n\n    Parameters\n    ----------\n    input : array_like\n        The input array.\n    mapping : {callable, scipy.LowLevelCallable}\n        A callable object that accepts a tuple of length equal to the output\n        array rank, and returns the corresponding input coordinates as a tuple\n        of length equal to the input array rank.\n    output_shape : tuple of ints, optional\n        Shape tuple.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n    extra_arguments : tuple, optional\n        Extra arguments passed to `mapping`.\n    extra_keywords : dict, optional\n        Extra keywords passed to `mapping`.\n\n    Returns\n    -------\n    return_value : ndarray or None\n        The filtered input. If `output` is given as a parameter, None is\n        returned.\n\n    See Also\n    --------\n    map_coordinates, affine_transform, spline_filter1d\n\n\n    Notes\n    -----\n    This function also accepts low-level callback functions with one\n    the following signatures and wrapped in `scipy.LowLevelCallable`:\n\n    .. code:: c\n\n       int mapping(npy_intp *output_coordinates, double *input_coordinates,\n                   int output_rank, int input_rank, void *user_data)\n       int mapping(intptr_t *output_coordinates, double *input_coordinates,\n                   int output_rank, int input_rank, void *user_data)\n\n    The calling function iterates over the elements of the output array,\n    calling the callback function at each element. The coordinates of the\n    current output element are passed through ``output_coordinates``. The\n    callback function must return the coordinates at which the input must\n    be interpolated in ``input_coordinates``. The rank of the input and\n    output arrays are given by ``input_rank`` and ``output_rank``\n    respectively.  ``user_data`` is the data pointer provided\n    to `scipy.LowLevelCallable` as-is.\n\n    The callback function must return an integer error status that is zero\n    if something went wrong and one otherwise. If an error occurs, you should\n    normally set the python error status with an informative message\n    before returning, otherwise a default error message is set by the\n    calling function.\n\n    In addition, some other low-level function pointer specifications\n    are accepted, but these are for backward compatibility only and should\n    not be used in new code.\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.arange(12.).reshape((4, 3))\n    >>> def shift_func(output_coords):\n    ...     return (output_coords[0] - 0.5, output_coords[1] - 0.5)\n    ...\n    >>> ndimage.geometric_transform(a, shift_func)\n    array([[ 0.   ,  0.   ,  0.   ],\n           [ 0.   ,  1.362,  2.738],\n           [ 0.   ,  4.812,  6.187],\n           [ 0.   ,  8.263,  9.637]])\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 226)
    order_120291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'order')
    int_120292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'int')
    # Applying the binary operator '<' (line 226)
    result_lt_120293 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), '<', order_120291, int_120292)
    
    
    # Getting the type of 'order' (line 226)
    order_120294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'order')
    int_120295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'int')
    # Applying the binary operator '>' (line 226)
    result_gt_120296 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 20), '>', order_120294, int_120295)
    
    # Applying the binary operator 'or' (line 226)
    result_or_keyword_120297 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), 'or', result_lt_120293, result_gt_120296)
    
    # Testing the type of an if condition (line 226)
    if_condition_120298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), result_or_keyword_120297)
    # Assigning a type to the variable 'if_condition_120298' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_120298', if_condition_120298)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 227)
    # Processing the call arguments (line 227)
    str_120300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 227)
    kwargs_120301 = {}
    # Getting the type of 'RuntimeError' (line 227)
    RuntimeError_120299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 227)
    RuntimeError_call_result_120302 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), RuntimeError_120299, *[str_120300], **kwargs_120301)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 227, 8), RuntimeError_call_result_120302, 'raise parameter', BaseException)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to asarray(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'input' (line 228)
    input_120305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'input', False)
    # Processing the call keyword arguments (line 228)
    kwargs_120306 = {}
    # Getting the type of 'numpy' (line 228)
    numpy_120303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 228)
    asarray_120304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), numpy_120303, 'asarray')
    # Calling asarray(args, kwargs) (line 228)
    asarray_call_result_120307 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), asarray_120304, *[input_120305], **kwargs_120306)
    
    # Assigning a type to the variable 'input' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'input', asarray_call_result_120307)
    
    
    # Call to iscomplexobj(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'input' (line 229)
    input_120310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 26), 'input', False)
    # Processing the call keyword arguments (line 229)
    kwargs_120311 = {}
    # Getting the type of 'numpy' (line 229)
    numpy_120308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 229)
    iscomplexobj_120309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 7), numpy_120308, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 229)
    iscomplexobj_call_result_120312 = invoke(stypy.reporting.localization.Localization(__file__, 229, 7), iscomplexobj_120309, *[input_120310], **kwargs_120311)
    
    # Testing the type of an if condition (line 229)
    if_condition_120313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 4), iscomplexobj_call_result_120312)
    # Assigning a type to the variable 'if_condition_120313' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'if_condition_120313', if_condition_120313)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 230)
    # Processing the call arguments (line 230)
    str_120315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 230)
    kwargs_120316 = {}
    # Getting the type of 'TypeError' (line 230)
    TypeError_120314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 230)
    TypeError_call_result_120317 = invoke(stypy.reporting.localization.Localization(__file__, 230, 14), TypeError_120314, *[str_120315], **kwargs_120316)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 230, 8), TypeError_call_result_120317, 'raise parameter', BaseException)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 231)
    # Getting the type of 'output_shape' (line 231)
    output_shape_120318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'output_shape')
    # Getting the type of 'None' (line 231)
    None_120319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'None')
    
    (may_be_120320, more_types_in_union_120321) = may_be_none(output_shape_120318, None_120319)

    if may_be_120320:

        if more_types_in_union_120321:
            # Runtime conditional SSA (line 231)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 232):
        
        # Assigning a Attribute to a Name (line 232):
        # Getting the type of 'input' (line 232)
        input_120322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'input')
        # Obtaining the member 'shape' of a type (line 232)
        shape_120323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), input_120322, 'shape')
        # Assigning a type to the variable 'output_shape' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'output_shape', shape_120323)

        if more_types_in_union_120321:
            # SSA join for if statement (line 231)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'input' (line 233)
    input_120324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 233)
    ndim_120325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 7), input_120324, 'ndim')
    int_120326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
    # Applying the binary operator '<' (line 233)
    result_lt_120327 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 7), '<', ndim_120325, int_120326)
    
    
    
    # Call to len(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'output_shape' (line 233)
    output_shape_120329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 29), 'output_shape', False)
    # Processing the call keyword arguments (line 233)
    kwargs_120330 = {}
    # Getting the type of 'len' (line 233)
    len_120328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'len', False)
    # Calling len(args, kwargs) (line 233)
    len_call_result_120331 = invoke(stypy.reporting.localization.Localization(__file__, 233, 25), len_120328, *[output_shape_120329], **kwargs_120330)
    
    int_120332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 45), 'int')
    # Applying the binary operator '<' (line 233)
    result_lt_120333 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 25), '<', len_call_result_120331, int_120332)
    
    # Applying the binary operator 'or' (line 233)
    result_or_keyword_120334 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 7), 'or', result_lt_120327, result_lt_120333)
    
    # Testing the type of an if condition (line 233)
    if_condition_120335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 4), result_or_keyword_120334)
    # Assigning a type to the variable 'if_condition_120335' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'if_condition_120335', if_condition_120335)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 234)
    # Processing the call arguments (line 234)
    str_120337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 27), 'str', 'input and output rank must be > 0')
    # Processing the call keyword arguments (line 234)
    kwargs_120338 = {}
    # Getting the type of 'RuntimeError' (line 234)
    RuntimeError_120336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 234)
    RuntimeError_call_result_120339 = invoke(stypy.reporting.localization.Localization(__file__, 234, 14), RuntimeError_120336, *[str_120337], **kwargs_120338)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 234, 8), RuntimeError_call_result_120339, 'raise parameter', BaseException)
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to _extend_mode_to_code(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'mode' (line 235)
    mode_120341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'mode', False)
    # Processing the call keyword arguments (line 235)
    kwargs_120342 = {}
    # Getting the type of '_extend_mode_to_code' (line 235)
    _extend_mode_to_code_120340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), '_extend_mode_to_code', False)
    # Calling _extend_mode_to_code(args, kwargs) (line 235)
    _extend_mode_to_code_call_result_120343 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), _extend_mode_to_code_120340, *[mode_120341], **kwargs_120342)
    
    # Assigning a type to the variable 'mode' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'mode', _extend_mode_to_code_call_result_120343)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prefilter' (line 236)
    prefilter_120344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 7), 'prefilter')
    
    # Getting the type of 'order' (line 236)
    order_120345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 21), 'order')
    int_120346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'int')
    # Applying the binary operator '>' (line 236)
    result_gt_120347 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 21), '>', order_120345, int_120346)
    
    # Applying the binary operator 'and' (line 236)
    result_and_keyword_120348 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 7), 'and', prefilter_120344, result_gt_120347)
    
    # Testing the type of an if condition (line 236)
    if_condition_120349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 4), result_and_keyword_120348)
    # Assigning a type to the variable 'if_condition_120349' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'if_condition_120349', if_condition_120349)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to spline_filter(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'input' (line 237)
    input_120351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'input', False)
    # Getting the type of 'order' (line 237)
    order_120352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 40), 'order', False)
    # Processing the call keyword arguments (line 237)
    # Getting the type of 'numpy' (line 237)
    numpy_120353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 54), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 237)
    float64_120354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 54), numpy_120353, 'float64')
    keyword_120355 = float64_120354
    kwargs_120356 = {'output': keyword_120355}
    # Getting the type of 'spline_filter' (line 237)
    spline_filter_120350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'spline_filter', False)
    # Calling spline_filter(args, kwargs) (line 237)
    spline_filter_call_result_120357 = invoke(stypy.reporting.localization.Localization(__file__, 237, 19), spline_filter_120350, *[input_120351, order_120352], **kwargs_120356)
    
    # Assigning a type to the variable 'filtered' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'filtered', spline_filter_call_result_120357)
    # SSA branch for the else part of an if statement (line 236)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 239):
    
    # Assigning a Name to a Name (line 239):
    # Getting the type of 'input' (line 239)
    input_120358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'input')
    # Assigning a type to the variable 'filtered' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'filtered', input_120358)
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 240):
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_120359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to _get_output(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'output' (line 240)
    output_120362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 51), 'output', False)
    # Getting the type of 'input' (line 240)
    input_120363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'input', False)
    # Processing the call keyword arguments (line 240)
    # Getting the type of 'output_shape' (line 241)
    output_shape_120364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 57), 'output_shape', False)
    keyword_120365 = output_shape_120364
    kwargs_120366 = {'shape': keyword_120365}
    # Getting the type of '_ni_support' (line 240)
    _ni_support_120360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 240)
    _get_output_120361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), _ni_support_120360, '_get_output')
    # Calling _get_output(args, kwargs) (line 240)
    _get_output_call_result_120367 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), _get_output_120361, *[output_120362, input_120363], **kwargs_120366)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___120368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), _get_output_call_result_120367, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_120369 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___120368, int_120359)
    
    # Assigning a type to the variable 'tuple_var_assignment_120075' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_120075', subscript_call_result_120369)
    
    # Assigning a Subscript to a Name (line 240):
    
    # Obtaining the type of the subscript
    int_120370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 4), 'int')
    
    # Call to _get_output(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'output' (line 240)
    output_120373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 51), 'output', False)
    # Getting the type of 'input' (line 240)
    input_120374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'input', False)
    # Processing the call keyword arguments (line 240)
    # Getting the type of 'output_shape' (line 241)
    output_shape_120375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 57), 'output_shape', False)
    keyword_120376 = output_shape_120375
    kwargs_120377 = {'shape': keyword_120376}
    # Getting the type of '_ni_support' (line 240)
    _ni_support_120371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 240)
    _get_output_120372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), _ni_support_120371, '_get_output')
    # Calling _get_output(args, kwargs) (line 240)
    _get_output_call_result_120378 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), _get_output_120372, *[output_120373, input_120374], **kwargs_120377)
    
    # Obtaining the member '__getitem__' of a type (line 240)
    getitem___120379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 4), _get_output_call_result_120378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 240)
    subscript_call_result_120380 = invoke(stypy.reporting.localization.Localization(__file__, 240, 4), getitem___120379, int_120370)
    
    # Assigning a type to the variable 'tuple_var_assignment_120076' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_120076', subscript_call_result_120380)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_120075' (line 240)
    tuple_var_assignment_120075_120381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_120075')
    # Assigning a type to the variable 'output' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'output', tuple_var_assignment_120075_120381)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_var_assignment_120076' (line 240)
    tuple_var_assignment_120076_120382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'tuple_var_assignment_120076')
    # Assigning a type to the variable 'return_value' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'return_value', tuple_var_assignment_120076_120382)
    
    # Call to geometric_transform(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'filtered' (line 242)
    filtered_120385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'filtered', False)
    # Getting the type of 'mapping' (line 242)
    mapping_120386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 44), 'mapping', False)
    # Getting the type of 'None' (line 242)
    None_120387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 53), 'None', False)
    # Getting the type of 'None' (line 242)
    None_120388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 59), 'None', False)
    # Getting the type of 'None' (line 242)
    None_120389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 65), 'None', False)
    # Getting the type of 'output' (line 242)
    output_120390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 71), 'output', False)
    # Getting the type of 'order' (line 243)
    order_120391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 34), 'order', False)
    # Getting the type of 'mode' (line 243)
    mode_120392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'mode', False)
    # Getting the type of 'cval' (line 243)
    cval_120393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 47), 'cval', False)
    # Getting the type of 'extra_arguments' (line 243)
    extra_arguments_120394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 53), 'extra_arguments', False)
    # Getting the type of 'extra_keywords' (line 244)
    extra_keywords_120395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'extra_keywords', False)
    # Processing the call keyword arguments (line 242)
    kwargs_120396 = {}
    # Getting the type of '_nd_image' (line 242)
    _nd_image_120383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), '_nd_image', False)
    # Obtaining the member 'geometric_transform' of a type (line 242)
    geometric_transform_120384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 4), _nd_image_120383, 'geometric_transform')
    # Calling geometric_transform(args, kwargs) (line 242)
    geometric_transform_call_result_120397 = invoke(stypy.reporting.localization.Localization(__file__, 242, 4), geometric_transform_120384, *[filtered_120385, mapping_120386, None_120387, None_120388, None_120389, output_120390, order_120391, mode_120392, cval_120393, extra_arguments_120394, extra_keywords_120395], **kwargs_120396)
    
    # Getting the type of 'return_value' (line 245)
    return_value_120398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type', return_value_120398)
    
    # ################# End of 'geometric_transform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'geometric_transform' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_120399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'geometric_transform'
    return stypy_return_type_120399

# Assigning a type to the variable 'geometric_transform' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'geometric_transform', geometric_transform)

@norecursion
def map_coordinates(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 248)
    None_120400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'None')
    int_120401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 59), 'int')
    str_120402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'str', 'constant')
    float_120403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 42), 'float')
    # Getting the type of 'True' (line 249)
    True_120404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 57), 'True')
    defaults = [None_120400, int_120401, str_120402, float_120403, True_120404]
    # Create a new context for function 'map_coordinates'
    module_type_store = module_type_store.open_function_context('map_coordinates', 248, 0, False)
    
    # Passed parameters checking function
    map_coordinates.stypy_localization = localization
    map_coordinates.stypy_type_of_self = None
    map_coordinates.stypy_type_store = module_type_store
    map_coordinates.stypy_function_name = 'map_coordinates'
    map_coordinates.stypy_param_names_list = ['input', 'coordinates', 'output', 'order', 'mode', 'cval', 'prefilter']
    map_coordinates.stypy_varargs_param_name = None
    map_coordinates.stypy_kwargs_param_name = None
    map_coordinates.stypy_call_defaults = defaults
    map_coordinates.stypy_call_varargs = varargs
    map_coordinates.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'map_coordinates', ['input', 'coordinates', 'output', 'order', 'mode', 'cval', 'prefilter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'map_coordinates', localization, ['input', 'coordinates', 'output', 'order', 'mode', 'cval', 'prefilter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'map_coordinates(...)' code ##################

    str_120405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, (-1)), 'str', "\n    Map the input array to new coordinates by interpolation.\n\n    The array of coordinates is used to find, for each point in the output,\n    the corresponding coordinates in the input. The value of the input at\n    those coordinates is determined by spline interpolation of the\n    requested order.\n\n    The shape of the output is derived from that of the coordinate\n    array by dropping the first axis. The values of the array along\n    the first axis are the coordinates in the input array at which the\n    output value is found.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input array.\n    coordinates : array_like\n        The coordinates at which `input` is evaluated.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n\n    Returns\n    -------\n    map_coordinates : ndarray\n        The result of transforming the input. The shape of the output is\n        derived from that of `coordinates` by dropping the first axis.\n\n    See Also\n    --------\n    spline_filter, geometric_transform, scipy.interpolate\n\n    Examples\n    --------\n    >>> from scipy import ndimage\n    >>> a = np.arange(12.).reshape((4, 3))\n    >>> a\n    array([[  0.,   1.,   2.],\n           [  3.,   4.,   5.],\n           [  6.,   7.,   8.],\n           [  9.,  10.,  11.]])\n    >>> ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)\n    array([ 2.,  7.])\n\n    Above, the interpolated value of a[0.5, 0.5] gives output[0], while\n    a[2, 1] is output[1].\n\n    >>> inds = np.array([[0.5, 2], [0.5, 4]])\n    >>> ndimage.map_coordinates(a, inds, order=1, cval=-33.3)\n    array([  2. , -33.3])\n    >>> ndimage.map_coordinates(a, inds, order=1, mode='nearest')\n    array([ 2.,  8.])\n    >>> ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)\n    array([ True, False], dtype=bool)\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 322)
    order_120406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'order')
    int_120407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 15), 'int')
    # Applying the binary operator '<' (line 322)
    result_lt_120408 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), '<', order_120406, int_120407)
    
    
    # Getting the type of 'order' (line 322)
    order_120409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 20), 'order')
    int_120410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'int')
    # Applying the binary operator '>' (line 322)
    result_gt_120411 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 20), '>', order_120409, int_120410)
    
    # Applying the binary operator 'or' (line 322)
    result_or_keyword_120412 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), 'or', result_lt_120408, result_gt_120411)
    
    # Testing the type of an if condition (line 322)
    if_condition_120413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_or_keyword_120412)
    # Assigning a type to the variable 'if_condition_120413' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_120413', if_condition_120413)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 323)
    # Processing the call arguments (line 323)
    str_120415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 323)
    kwargs_120416 = {}
    # Getting the type of 'RuntimeError' (line 323)
    RuntimeError_120414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 323)
    RuntimeError_call_result_120417 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), RuntimeError_120414, *[str_120415], **kwargs_120416)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 323, 8), RuntimeError_call_result_120417, 'raise parameter', BaseException)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to asarray(...): (line 324)
    # Processing the call arguments (line 324)
    # Getting the type of 'input' (line 324)
    input_120420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'input', False)
    # Processing the call keyword arguments (line 324)
    kwargs_120421 = {}
    # Getting the type of 'numpy' (line 324)
    numpy_120418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 324)
    asarray_120419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), numpy_120418, 'asarray')
    # Calling asarray(args, kwargs) (line 324)
    asarray_call_result_120422 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), asarray_120419, *[input_120420], **kwargs_120421)
    
    # Assigning a type to the variable 'input' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'input', asarray_call_result_120422)
    
    
    # Call to iscomplexobj(...): (line 325)
    # Processing the call arguments (line 325)
    # Getting the type of 'input' (line 325)
    input_120425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 26), 'input', False)
    # Processing the call keyword arguments (line 325)
    kwargs_120426 = {}
    # Getting the type of 'numpy' (line 325)
    numpy_120423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 325)
    iscomplexobj_120424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 7), numpy_120423, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 325)
    iscomplexobj_call_result_120427 = invoke(stypy.reporting.localization.Localization(__file__, 325, 7), iscomplexobj_120424, *[input_120425], **kwargs_120426)
    
    # Testing the type of an if condition (line 325)
    if_condition_120428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 4), iscomplexobj_call_result_120427)
    # Assigning a type to the variable 'if_condition_120428' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'if_condition_120428', if_condition_120428)
    # SSA begins for if statement (line 325)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 326)
    # Processing the call arguments (line 326)
    str_120430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 326)
    kwargs_120431 = {}
    # Getting the type of 'TypeError' (line 326)
    TypeError_120429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 326)
    TypeError_call_result_120432 = invoke(stypy.reporting.localization.Localization(__file__, 326, 14), TypeError_120429, *[str_120430], **kwargs_120431)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 326, 8), TypeError_call_result_120432, 'raise parameter', BaseException)
    # SSA join for if statement (line 325)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 327):
    
    # Assigning a Call to a Name (line 327):
    
    # Call to asarray(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'coordinates' (line 327)
    coordinates_120435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'coordinates', False)
    # Processing the call keyword arguments (line 327)
    kwargs_120436 = {}
    # Getting the type of 'numpy' (line 327)
    numpy_120433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 18), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 327)
    asarray_120434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 18), numpy_120433, 'asarray')
    # Calling asarray(args, kwargs) (line 327)
    asarray_call_result_120437 = invoke(stypy.reporting.localization.Localization(__file__, 327, 18), asarray_120434, *[coordinates_120435], **kwargs_120436)
    
    # Assigning a type to the variable 'coordinates' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'coordinates', asarray_call_result_120437)
    
    
    # Call to iscomplexobj(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'coordinates' (line 328)
    coordinates_120440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 26), 'coordinates', False)
    # Processing the call keyword arguments (line 328)
    kwargs_120441 = {}
    # Getting the type of 'numpy' (line 328)
    numpy_120438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 328)
    iscomplexobj_120439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 7), numpy_120438, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 328)
    iscomplexobj_call_result_120442 = invoke(stypy.reporting.localization.Localization(__file__, 328, 7), iscomplexobj_120439, *[coordinates_120440], **kwargs_120441)
    
    # Testing the type of an if condition (line 328)
    if_condition_120443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 4), iscomplexobj_call_result_120442)
    # Assigning a type to the variable 'if_condition_120443' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'if_condition_120443', if_condition_120443)
    # SSA begins for if statement (line 328)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 329)
    # Processing the call arguments (line 329)
    str_120445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 329)
    kwargs_120446 = {}
    # Getting the type of 'TypeError' (line 329)
    TypeError_120444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 329)
    TypeError_call_result_120447 = invoke(stypy.reporting.localization.Localization(__file__, 329, 14), TypeError_120444, *[str_120445], **kwargs_120446)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 329, 8), TypeError_call_result_120447, 'raise parameter', BaseException)
    # SSA join for if statement (line 328)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 330):
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    int_120448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 37), 'int')
    slice_120449 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 330, 19), int_120448, None, None)
    # Getting the type of 'coordinates' (line 330)
    coordinates_120450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'coordinates')
    # Obtaining the member 'shape' of a type (line 330)
    shape_120451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), coordinates_120450, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___120452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), shape_120451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_120453 = invoke(stypy.reporting.localization.Localization(__file__, 330, 19), getitem___120452, slice_120449)
    
    # Assigning a type to the variable 'output_shape' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'output_shape', subscript_call_result_120453)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'input' (line 331)
    input_120454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 331)
    ndim_120455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 7), input_120454, 'ndim')
    int_120456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 20), 'int')
    # Applying the binary operator '<' (line 331)
    result_lt_120457 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 7), '<', ndim_120455, int_120456)
    
    
    
    # Call to len(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'output_shape' (line 331)
    output_shape_120459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'output_shape', False)
    # Processing the call keyword arguments (line 331)
    kwargs_120460 = {}
    # Getting the type of 'len' (line 331)
    len_120458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 25), 'len', False)
    # Calling len(args, kwargs) (line 331)
    len_call_result_120461 = invoke(stypy.reporting.localization.Localization(__file__, 331, 25), len_120458, *[output_shape_120459], **kwargs_120460)
    
    int_120462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 45), 'int')
    # Applying the binary operator '<' (line 331)
    result_lt_120463 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 25), '<', len_call_result_120461, int_120462)
    
    # Applying the binary operator 'or' (line 331)
    result_or_keyword_120464 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 7), 'or', result_lt_120457, result_lt_120463)
    
    # Testing the type of an if condition (line 331)
    if_condition_120465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 4), result_or_keyword_120464)
    # Assigning a type to the variable 'if_condition_120465' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'if_condition_120465', if_condition_120465)
    # SSA begins for if statement (line 331)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 332)
    # Processing the call arguments (line 332)
    str_120467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 27), 'str', 'input and output rank must be > 0')
    # Processing the call keyword arguments (line 332)
    kwargs_120468 = {}
    # Getting the type of 'RuntimeError' (line 332)
    RuntimeError_120466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 332)
    RuntimeError_call_result_120469 = invoke(stypy.reporting.localization.Localization(__file__, 332, 14), RuntimeError_120466, *[str_120467], **kwargs_120468)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 332, 8), RuntimeError_call_result_120469, 'raise parameter', BaseException)
    # SSA join for if statement (line 331)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_120470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 25), 'int')
    # Getting the type of 'coordinates' (line 333)
    coordinates_120471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 7), 'coordinates')
    # Obtaining the member 'shape' of a type (line 333)
    shape_120472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 7), coordinates_120471, 'shape')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___120473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 7), shape_120472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_120474 = invoke(stypy.reporting.localization.Localization(__file__, 333, 7), getitem___120473, int_120470)
    
    # Getting the type of 'input' (line 333)
    input_120475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 'input')
    # Obtaining the member 'ndim' of a type (line 333)
    ndim_120476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 31), input_120475, 'ndim')
    # Applying the binary operator '!=' (line 333)
    result_ne_120477 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 7), '!=', subscript_call_result_120474, ndim_120476)
    
    # Testing the type of an if condition (line 333)
    if_condition_120478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 4), result_ne_120477)
    # Assigning a type to the variable 'if_condition_120478' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'if_condition_120478', if_condition_120478)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 334)
    # Processing the call arguments (line 334)
    str_120480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 27), 'str', 'invalid shape for coordinate array')
    # Processing the call keyword arguments (line 334)
    kwargs_120481 = {}
    # Getting the type of 'RuntimeError' (line 334)
    RuntimeError_120479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 334)
    RuntimeError_call_result_120482 = invoke(stypy.reporting.localization.Localization(__file__, 334, 14), RuntimeError_120479, *[str_120480], **kwargs_120481)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 334, 8), RuntimeError_call_result_120482, 'raise parameter', BaseException)
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to _extend_mode_to_code(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'mode' (line 335)
    mode_120484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 32), 'mode', False)
    # Processing the call keyword arguments (line 335)
    kwargs_120485 = {}
    # Getting the type of '_extend_mode_to_code' (line 335)
    _extend_mode_to_code_120483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), '_extend_mode_to_code', False)
    # Calling _extend_mode_to_code(args, kwargs) (line 335)
    _extend_mode_to_code_call_result_120486 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), _extend_mode_to_code_120483, *[mode_120484], **kwargs_120485)
    
    # Assigning a type to the variable 'mode' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'mode', _extend_mode_to_code_call_result_120486)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prefilter' (line 336)
    prefilter_120487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 7), 'prefilter')
    
    # Getting the type of 'order' (line 336)
    order_120488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 21), 'order')
    int_120489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 29), 'int')
    # Applying the binary operator '>' (line 336)
    result_gt_120490 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 21), '>', order_120488, int_120489)
    
    # Applying the binary operator 'and' (line 336)
    result_and_keyword_120491 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 7), 'and', prefilter_120487, result_gt_120490)
    
    # Testing the type of an if condition (line 336)
    if_condition_120492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 4), result_and_keyword_120491)
    # Assigning a type to the variable 'if_condition_120492' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'if_condition_120492', if_condition_120492)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to spline_filter(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'input' (line 337)
    input_120494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'input', False)
    # Getting the type of 'order' (line 337)
    order_120495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 40), 'order', False)
    # Processing the call keyword arguments (line 337)
    # Getting the type of 'numpy' (line 337)
    numpy_120496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 54), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 337)
    float64_120497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 54), numpy_120496, 'float64')
    keyword_120498 = float64_120497
    kwargs_120499 = {'output': keyword_120498}
    # Getting the type of 'spline_filter' (line 337)
    spline_filter_120493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'spline_filter', False)
    # Calling spline_filter(args, kwargs) (line 337)
    spline_filter_call_result_120500 = invoke(stypy.reporting.localization.Localization(__file__, 337, 19), spline_filter_120493, *[input_120494, order_120495], **kwargs_120499)
    
    # Assigning a type to the variable 'filtered' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'filtered', spline_filter_call_result_120500)
    # SSA branch for the else part of an if statement (line 336)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 339):
    
    # Assigning a Name to a Name (line 339):
    # Getting the type of 'input' (line 339)
    input_120501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'input')
    # Assigning a type to the variable 'filtered' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'filtered', input_120501)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 340):
    
    # Assigning a Subscript to a Name (line 340):
    
    # Obtaining the type of the subscript
    int_120502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 4), 'int')
    
    # Call to _get_output(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'output' (line 340)
    output_120505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 51), 'output', False)
    # Getting the type of 'input' (line 340)
    input_120506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'input', False)
    # Processing the call keyword arguments (line 340)
    # Getting the type of 'output_shape' (line 341)
    output_shape_120507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 57), 'output_shape', False)
    keyword_120508 = output_shape_120507
    kwargs_120509 = {'shape': keyword_120508}
    # Getting the type of '_ni_support' (line 340)
    _ni_support_120503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 340)
    _get_output_120504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 27), _ni_support_120503, '_get_output')
    # Calling _get_output(args, kwargs) (line 340)
    _get_output_call_result_120510 = invoke(stypy.reporting.localization.Localization(__file__, 340, 27), _get_output_120504, *[output_120505, input_120506], **kwargs_120509)
    
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___120511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 4), _get_output_call_result_120510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_120512 = invoke(stypy.reporting.localization.Localization(__file__, 340, 4), getitem___120511, int_120502)
    
    # Assigning a type to the variable 'tuple_var_assignment_120077' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'tuple_var_assignment_120077', subscript_call_result_120512)
    
    # Assigning a Subscript to a Name (line 340):
    
    # Obtaining the type of the subscript
    int_120513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 4), 'int')
    
    # Call to _get_output(...): (line 340)
    # Processing the call arguments (line 340)
    # Getting the type of 'output' (line 340)
    output_120516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 51), 'output', False)
    # Getting the type of 'input' (line 340)
    input_120517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 59), 'input', False)
    # Processing the call keyword arguments (line 340)
    # Getting the type of 'output_shape' (line 341)
    output_shape_120518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 57), 'output_shape', False)
    keyword_120519 = output_shape_120518
    kwargs_120520 = {'shape': keyword_120519}
    # Getting the type of '_ni_support' (line 340)
    _ni_support_120514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 340)
    _get_output_120515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 27), _ni_support_120514, '_get_output')
    # Calling _get_output(args, kwargs) (line 340)
    _get_output_call_result_120521 = invoke(stypy.reporting.localization.Localization(__file__, 340, 27), _get_output_120515, *[output_120516, input_120517], **kwargs_120520)
    
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___120522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 4), _get_output_call_result_120521, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_120523 = invoke(stypy.reporting.localization.Localization(__file__, 340, 4), getitem___120522, int_120513)
    
    # Assigning a type to the variable 'tuple_var_assignment_120078' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'tuple_var_assignment_120078', subscript_call_result_120523)
    
    # Assigning a Name to a Name (line 340):
    # Getting the type of 'tuple_var_assignment_120077' (line 340)
    tuple_var_assignment_120077_120524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'tuple_var_assignment_120077')
    # Assigning a type to the variable 'output' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'output', tuple_var_assignment_120077_120524)
    
    # Assigning a Name to a Name (line 340):
    # Getting the type of 'tuple_var_assignment_120078' (line 340)
    tuple_var_assignment_120078_120525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'tuple_var_assignment_120078')
    # Assigning a type to the variable 'return_value' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'return_value', tuple_var_assignment_120078_120525)
    
    # Call to geometric_transform(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'filtered' (line 342)
    filtered_120528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 34), 'filtered', False)
    # Getting the type of 'None' (line 342)
    None_120529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 44), 'None', False)
    # Getting the type of 'coordinates' (line 342)
    coordinates_120530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 50), 'coordinates', False)
    # Getting the type of 'None' (line 342)
    None_120531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 63), 'None', False)
    # Getting the type of 'None' (line 342)
    None_120532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 69), 'None', False)
    # Getting the type of 'output' (line 343)
    output_120533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 34), 'output', False)
    # Getting the type of 'order' (line 343)
    order_120534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'order', False)
    # Getting the type of 'mode' (line 343)
    mode_120535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 49), 'mode', False)
    # Getting the type of 'cval' (line 343)
    cval_120536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 55), 'cval', False)
    # Getting the type of 'None' (line 343)
    None_120537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 61), 'None', False)
    # Getting the type of 'None' (line 343)
    None_120538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 67), 'None', False)
    # Processing the call keyword arguments (line 342)
    kwargs_120539 = {}
    # Getting the type of '_nd_image' (line 342)
    _nd_image_120526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), '_nd_image', False)
    # Obtaining the member 'geometric_transform' of a type (line 342)
    geometric_transform_120527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), _nd_image_120526, 'geometric_transform')
    # Calling geometric_transform(args, kwargs) (line 342)
    geometric_transform_call_result_120540 = invoke(stypy.reporting.localization.Localization(__file__, 342, 4), geometric_transform_120527, *[filtered_120528, None_120529, coordinates_120530, None_120531, None_120532, output_120533, order_120534, mode_120535, cval_120536, None_120537, None_120538], **kwargs_120539)
    
    # Getting the type of 'return_value' (line 344)
    return_value_120541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type', return_value_120541)
    
    # ################# End of 'map_coordinates(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'map_coordinates' in the type store
    # Getting the type of 'stypy_return_type' (line 248)
    stypy_return_type_120542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'map_coordinates'
    return stypy_return_type_120542

# Assigning a type to the variable 'map_coordinates' (line 248)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 0), 'map_coordinates', map_coordinates)

@norecursion
def affine_transform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_120543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 43), 'float')
    # Getting the type of 'None' (line 347)
    None_120544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 61), 'None')
    # Getting the type of 'None' (line 348)
    None_120545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 28), 'None')
    int_120546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 40), 'int')
    str_120547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 26), 'str', 'constant')
    float_120548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 43), 'float')
    # Getting the type of 'True' (line 349)
    True_120549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 58), 'True')
    defaults = [float_120543, None_120544, None_120545, int_120546, str_120547, float_120548, True_120549]
    # Create a new context for function 'affine_transform'
    module_type_store = module_type_store.open_function_context('affine_transform', 347, 0, False)
    
    # Passed parameters checking function
    affine_transform.stypy_localization = localization
    affine_transform.stypy_type_of_self = None
    affine_transform.stypy_type_store = module_type_store
    affine_transform.stypy_function_name = 'affine_transform'
    affine_transform.stypy_param_names_list = ['input', 'matrix', 'offset', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter']
    affine_transform.stypy_varargs_param_name = None
    affine_transform.stypy_kwargs_param_name = None
    affine_transform.stypy_call_defaults = defaults
    affine_transform.stypy_call_varargs = varargs
    affine_transform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'affine_transform', ['input', 'matrix', 'offset', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'affine_transform', localization, ['input', 'matrix', 'offset', 'output_shape', 'output', 'order', 'mode', 'cval', 'prefilter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'affine_transform(...)' code ##################

    str_120550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, (-1)), 'str', "\n    Apply an affine transformation.\n\n    Given an output image pixel index vector ``o``, the pixel value\n    is determined from the input image at position\n    ``np.dot(matrix, o) + offset``.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input array.\n    matrix : ndarray\n        The inverse coordinate transformation matrix, mapping output\n        coordinates to input coordinates. If ``ndim`` is the number of\n        dimensions of ``input``, the given matrix must have one of the\n        following shapes:\n\n            - ``(ndim, ndim)``: the linear transformation matrix for each\n              output coordinate.\n            - ``(ndim,)``: assume that the 2D transformation matrix is\n              diagonal, with the diagonal specified by the given value. A more\n              efficient algorithm is then used that exploits the separability\n              of the problem.\n            - ``(ndim + 1, ndim + 1)``: assume that the transformation is\n              specified using homogeneous coordinates [1]_. In this case, any\n              value passed to ``offset`` is ignored.\n            - ``(ndim, ndim + 1)``: as above, but the bottom row of a\n              homogeneous transformation matrix is always ``[0, 0, ..., 1]``,\n              and may be omitted.\n\n    offset : float or sequence, optional\n        The offset into the array where the transform is applied. If a float,\n        `offset` is the same for each axis. If a sequence, `offset` should\n        contain one value for each axis.\n    output_shape : tuple of ints, optional\n        Shape tuple.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or\n        'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n\n    Returns\n    -------\n    affine_transform : ndarray or None\n        The transformed input. If `output` is given as a parameter, None is\n        returned.\n\n    Notes\n    -----\n    The given matrix and offset are used to find for each point in the\n    output the corresponding coordinates in the input by an affine\n    transformation. The value of the input at those coordinates is\n    determined by spline interpolation of the requested order. Points\n    outside the boundaries of the input are filled according to the given\n    mode.\n\n    .. versionchanged:: 0.18.0\n        Previously, the exact interpretation of the affine transformation\n        depended on whether the matrix was supplied as a one-dimensional or\n        two-dimensional array. If a one-dimensional array was supplied\n        to the matrix parameter, the output pixel value at index ``o``\n        was determined from the input image at position\n        ``matrix * (o + offset)``.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 433)
    order_120551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 7), 'order')
    int_120552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 15), 'int')
    # Applying the binary operator '<' (line 433)
    result_lt_120553 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 7), '<', order_120551, int_120552)
    
    
    # Getting the type of 'order' (line 433)
    order_120554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'order')
    int_120555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 28), 'int')
    # Applying the binary operator '>' (line 433)
    result_gt_120556 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 20), '>', order_120554, int_120555)
    
    # Applying the binary operator 'or' (line 433)
    result_or_keyword_120557 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 7), 'or', result_lt_120553, result_gt_120556)
    
    # Testing the type of an if condition (line 433)
    if_condition_120558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 4), result_or_keyword_120557)
    # Assigning a type to the variable 'if_condition_120558' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'if_condition_120558', if_condition_120558)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 434)
    # Processing the call arguments (line 434)
    str_120560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 434)
    kwargs_120561 = {}
    # Getting the type of 'RuntimeError' (line 434)
    RuntimeError_120559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 434)
    RuntimeError_call_result_120562 = invoke(stypy.reporting.localization.Localization(__file__, 434, 14), RuntimeError_120559, *[str_120560], **kwargs_120561)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 434, 8), RuntimeError_call_result_120562, 'raise parameter', BaseException)
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to asarray(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'input' (line 435)
    input_120565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'input', False)
    # Processing the call keyword arguments (line 435)
    kwargs_120566 = {}
    # Getting the type of 'numpy' (line 435)
    numpy_120563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 435)
    asarray_120564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 12), numpy_120563, 'asarray')
    # Calling asarray(args, kwargs) (line 435)
    asarray_call_result_120567 = invoke(stypy.reporting.localization.Localization(__file__, 435, 12), asarray_120564, *[input_120565], **kwargs_120566)
    
    # Assigning a type to the variable 'input' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'input', asarray_call_result_120567)
    
    
    # Call to iscomplexobj(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'input' (line 436)
    input_120570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 26), 'input', False)
    # Processing the call keyword arguments (line 436)
    kwargs_120571 = {}
    # Getting the type of 'numpy' (line 436)
    numpy_120568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 436)
    iscomplexobj_120569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 7), numpy_120568, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 436)
    iscomplexobj_call_result_120572 = invoke(stypy.reporting.localization.Localization(__file__, 436, 7), iscomplexobj_120569, *[input_120570], **kwargs_120571)
    
    # Testing the type of an if condition (line 436)
    if_condition_120573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), iscomplexobj_call_result_120572)
    # Assigning a type to the variable 'if_condition_120573' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_120573', if_condition_120573)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 437)
    # Processing the call arguments (line 437)
    str_120575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 437)
    kwargs_120576 = {}
    # Getting the type of 'TypeError' (line 437)
    TypeError_120574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 437)
    TypeError_call_result_120577 = invoke(stypy.reporting.localization.Localization(__file__, 437, 14), TypeError_120574, *[str_120575], **kwargs_120576)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 437, 8), TypeError_call_result_120577, 'raise parameter', BaseException)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 438)
    # Getting the type of 'output_shape' (line 438)
    output_shape_120578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 7), 'output_shape')
    # Getting the type of 'None' (line 438)
    None_120579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'None')
    
    (may_be_120580, more_types_in_union_120581) = may_be_none(output_shape_120578, None_120579)

    if may_be_120580:

        if more_types_in_union_120581:
            # Runtime conditional SSA (line 438)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 439):
        
        # Assigning a Attribute to a Name (line 439):
        # Getting the type of 'input' (line 439)
        input_120582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'input')
        # Obtaining the member 'shape' of a type (line 439)
        shape_120583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), input_120582, 'shape')
        # Assigning a type to the variable 'output_shape' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'output_shape', shape_120583)

        if more_types_in_union_120581:
            # SSA join for if statement (line 438)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'input' (line 440)
    input_120584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 440)
    ndim_120585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 7), input_120584, 'ndim')
    int_120586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'int')
    # Applying the binary operator '<' (line 440)
    result_lt_120587 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 7), '<', ndim_120585, int_120586)
    
    
    
    # Call to len(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'output_shape' (line 440)
    output_shape_120589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 29), 'output_shape', False)
    # Processing the call keyword arguments (line 440)
    kwargs_120590 = {}
    # Getting the type of 'len' (line 440)
    len_120588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'len', False)
    # Calling len(args, kwargs) (line 440)
    len_call_result_120591 = invoke(stypy.reporting.localization.Localization(__file__, 440, 25), len_120588, *[output_shape_120589], **kwargs_120590)
    
    int_120592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 45), 'int')
    # Applying the binary operator '<' (line 440)
    result_lt_120593 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 25), '<', len_call_result_120591, int_120592)
    
    # Applying the binary operator 'or' (line 440)
    result_or_keyword_120594 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 7), 'or', result_lt_120587, result_lt_120593)
    
    # Testing the type of an if condition (line 440)
    if_condition_120595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 4), result_or_keyword_120594)
    # Assigning a type to the variable 'if_condition_120595' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'if_condition_120595', if_condition_120595)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 441)
    # Processing the call arguments (line 441)
    str_120597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 27), 'str', 'input and output rank must be > 0')
    # Processing the call keyword arguments (line 441)
    kwargs_120598 = {}
    # Getting the type of 'RuntimeError' (line 441)
    RuntimeError_120596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 441)
    RuntimeError_call_result_120599 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), RuntimeError_120596, *[str_120597], **kwargs_120598)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 441, 8), RuntimeError_call_result_120599, 'raise parameter', BaseException)
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to _extend_mode_to_code(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'mode' (line 442)
    mode_120601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'mode', False)
    # Processing the call keyword arguments (line 442)
    kwargs_120602 = {}
    # Getting the type of '_extend_mode_to_code' (line 442)
    _extend_mode_to_code_120600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), '_extend_mode_to_code', False)
    # Calling _extend_mode_to_code(args, kwargs) (line 442)
    _extend_mode_to_code_call_result_120603 = invoke(stypy.reporting.localization.Localization(__file__, 442, 11), _extend_mode_to_code_120600, *[mode_120601], **kwargs_120602)
    
    # Assigning a type to the variable 'mode' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'mode', _extend_mode_to_code_call_result_120603)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prefilter' (line 443)
    prefilter_120604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'prefilter')
    
    # Getting the type of 'order' (line 443)
    order_120605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 21), 'order')
    int_120606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 29), 'int')
    # Applying the binary operator '>' (line 443)
    result_gt_120607 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 21), '>', order_120605, int_120606)
    
    # Applying the binary operator 'and' (line 443)
    result_and_keyword_120608 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 7), 'and', prefilter_120604, result_gt_120607)
    
    # Testing the type of an if condition (line 443)
    if_condition_120609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), result_and_keyword_120608)
    # Assigning a type to the variable 'if_condition_120609' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_120609', if_condition_120609)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to spline_filter(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'input' (line 444)
    input_120611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 33), 'input', False)
    # Getting the type of 'order' (line 444)
    order_120612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 40), 'order', False)
    # Processing the call keyword arguments (line 444)
    # Getting the type of 'numpy' (line 444)
    numpy_120613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 54), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 444)
    float64_120614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 54), numpy_120613, 'float64')
    keyword_120615 = float64_120614
    kwargs_120616 = {'output': keyword_120615}
    # Getting the type of 'spline_filter' (line 444)
    spline_filter_120610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'spline_filter', False)
    # Calling spline_filter(args, kwargs) (line 444)
    spline_filter_call_result_120617 = invoke(stypy.reporting.localization.Localization(__file__, 444, 19), spline_filter_120610, *[input_120611, order_120612], **kwargs_120616)
    
    # Assigning a type to the variable 'filtered' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'filtered', spline_filter_call_result_120617)
    # SSA branch for the else part of an if statement (line 443)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 446):
    
    # Assigning a Name to a Name (line 446):
    # Getting the type of 'input' (line 446)
    input_120618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 'input')
    # Assigning a type to the variable 'filtered' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'filtered', input_120618)
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 447):
    
    # Assigning a Subscript to a Name (line 447):
    
    # Obtaining the type of the subscript
    int_120619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    
    # Call to _get_output(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'output' (line 447)
    output_120622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 51), 'output', False)
    # Getting the type of 'input' (line 447)
    input_120623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 59), 'input', False)
    # Processing the call keyword arguments (line 447)
    # Getting the type of 'output_shape' (line 448)
    output_shape_120624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 57), 'output_shape', False)
    keyword_120625 = output_shape_120624
    kwargs_120626 = {'shape': keyword_120625}
    # Getting the type of '_ni_support' (line 447)
    _ni_support_120620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 447)
    _get_output_120621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 27), _ni_support_120620, '_get_output')
    # Calling _get_output(args, kwargs) (line 447)
    _get_output_call_result_120627 = invoke(stypy.reporting.localization.Localization(__file__, 447, 27), _get_output_120621, *[output_120622, input_120623], **kwargs_120626)
    
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___120628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), _get_output_call_result_120627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_120629 = invoke(stypy.reporting.localization.Localization(__file__, 447, 4), getitem___120628, int_120619)
    
    # Assigning a type to the variable 'tuple_var_assignment_120079' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'tuple_var_assignment_120079', subscript_call_result_120629)
    
    # Assigning a Subscript to a Name (line 447):
    
    # Obtaining the type of the subscript
    int_120630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 4), 'int')
    
    # Call to _get_output(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'output' (line 447)
    output_120633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 51), 'output', False)
    # Getting the type of 'input' (line 447)
    input_120634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 59), 'input', False)
    # Processing the call keyword arguments (line 447)
    # Getting the type of 'output_shape' (line 448)
    output_shape_120635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 57), 'output_shape', False)
    keyword_120636 = output_shape_120635
    kwargs_120637 = {'shape': keyword_120636}
    # Getting the type of '_ni_support' (line 447)
    _ni_support_120631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 447)
    _get_output_120632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 27), _ni_support_120631, '_get_output')
    # Calling _get_output(args, kwargs) (line 447)
    _get_output_call_result_120638 = invoke(stypy.reporting.localization.Localization(__file__, 447, 27), _get_output_120632, *[output_120633, input_120634], **kwargs_120637)
    
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___120639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), _get_output_call_result_120638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_120640 = invoke(stypy.reporting.localization.Localization(__file__, 447, 4), getitem___120639, int_120630)
    
    # Assigning a type to the variable 'tuple_var_assignment_120080' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'tuple_var_assignment_120080', subscript_call_result_120640)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'tuple_var_assignment_120079' (line 447)
    tuple_var_assignment_120079_120641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'tuple_var_assignment_120079')
    # Assigning a type to the variable 'output' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'output', tuple_var_assignment_120079_120641)
    
    # Assigning a Name to a Name (line 447):
    # Getting the type of 'tuple_var_assignment_120080' (line 447)
    tuple_var_assignment_120080_120642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'tuple_var_assignment_120080')
    # Assigning a type to the variable 'return_value' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'return_value', tuple_var_assignment_120080_120642)
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to asarray(...): (line 449)
    # Processing the call arguments (line 449)
    # Getting the type of 'matrix' (line 449)
    matrix_120645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 27), 'matrix', False)
    # Processing the call keyword arguments (line 449)
    # Getting the type of 'numpy' (line 449)
    numpy_120646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 41), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 449)
    float64_120647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 41), numpy_120646, 'float64')
    keyword_120648 = float64_120647
    kwargs_120649 = {'dtype': keyword_120648}
    # Getting the type of 'numpy' (line 449)
    numpy_120643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 449)
    asarray_120644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 13), numpy_120643, 'asarray')
    # Calling asarray(args, kwargs) (line 449)
    asarray_call_result_120650 = invoke(stypy.reporting.localization.Localization(__file__, 449, 13), asarray_120644, *[matrix_120645], **kwargs_120649)
    
    # Assigning a type to the variable 'matrix' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'matrix', asarray_call_result_120650)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'matrix' (line 450)
    matrix_120651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 7), 'matrix')
    # Obtaining the member 'ndim' of a type (line 450)
    ndim_120652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 7), matrix_120651, 'ndim')
    
    # Obtaining an instance of the builtin type 'list' (line 450)
    list_120653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 450)
    # Adding element type (line 450)
    int_120654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 26), list_120653, int_120654)
    # Adding element type (line 450)
    int_120655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 450, 26), list_120653, int_120655)
    
    # Applying the binary operator 'notin' (line 450)
    result_contains_120656 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 7), 'notin', ndim_120652, list_120653)
    
    
    
    # Obtaining the type of the subscript
    int_120657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 49), 'int')
    # Getting the type of 'matrix' (line 450)
    matrix_120658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 36), 'matrix')
    # Obtaining the member 'shape' of a type (line 450)
    shape_120659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 36), matrix_120658, 'shape')
    # Obtaining the member '__getitem__' of a type (line 450)
    getitem___120660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 36), shape_120659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 450)
    subscript_call_result_120661 = invoke(stypy.reporting.localization.Localization(__file__, 450, 36), getitem___120660, int_120657)
    
    int_120662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 54), 'int')
    # Applying the binary operator '<' (line 450)
    result_lt_120663 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 36), '<', subscript_call_result_120661, int_120662)
    
    # Applying the binary operator 'or' (line 450)
    result_or_keyword_120664 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 7), 'or', result_contains_120656, result_lt_120663)
    
    # Testing the type of an if condition (line 450)
    if_condition_120665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 4), result_or_keyword_120664)
    # Assigning a type to the variable 'if_condition_120665' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'if_condition_120665', if_condition_120665)
    # SSA begins for if statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 451)
    # Processing the call arguments (line 451)
    str_120667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 27), 'str', 'no proper affine matrix provided')
    # Processing the call keyword arguments (line 451)
    kwargs_120668 = {}
    # Getting the type of 'RuntimeError' (line 451)
    RuntimeError_120666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 451)
    RuntimeError_call_result_120669 = invoke(stypy.reporting.localization.Localization(__file__, 451, 14), RuntimeError_120666, *[str_120667], **kwargs_120668)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 451, 8), RuntimeError_call_result_120669, 'raise parameter', BaseException)
    # SSA join for if statement (line 450)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'matrix' (line 452)
    matrix_120670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'matrix')
    # Obtaining the member 'ndim' of a type (line 452)
    ndim_120671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), matrix_120670, 'ndim')
    int_120672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 23), 'int')
    # Applying the binary operator '==' (line 452)
    result_eq_120673 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), '==', ndim_120671, int_120672)
    
    
    
    # Obtaining the type of the subscript
    int_120674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 42), 'int')
    # Getting the type of 'matrix' (line 452)
    matrix_120675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 29), 'matrix')
    # Obtaining the member 'shape' of a type (line 452)
    shape_120676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 29), matrix_120675, 'shape')
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___120677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 29), shape_120676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_120678 = invoke(stypy.reporting.localization.Localization(__file__, 452, 29), getitem___120677, int_120674)
    
    # Getting the type of 'input' (line 452)
    input_120679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 48), 'input')
    # Obtaining the member 'ndim' of a type (line 452)
    ndim_120680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 48), input_120679, 'ndim')
    int_120681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 61), 'int')
    # Applying the binary operator '+' (line 452)
    result_add_120682 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 48), '+', ndim_120680, int_120681)
    
    # Applying the binary operator '==' (line 452)
    result_eq_120683 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 29), '==', subscript_call_result_120678, result_add_120682)
    
    # Applying the binary operator 'and' (line 452)
    result_and_keyword_120684 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), 'and', result_eq_120673, result_eq_120683)
    
    
    # Obtaining the type of the subscript
    int_120685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 26), 'int')
    # Getting the type of 'matrix' (line 453)
    matrix_120686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 13), 'matrix')
    # Obtaining the member 'shape' of a type (line 453)
    shape_120687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 13), matrix_120686, 'shape')
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___120688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 13), shape_120687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_120689 = invoke(stypy.reporting.localization.Localization(__file__, 453, 13), getitem___120688, int_120685)
    
    
    # Obtaining an instance of the builtin type 'list' (line 453)
    list_120690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 453)
    # Adding element type (line 453)
    # Getting the type of 'input' (line 453)
    input_120691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 33), 'input')
    # Obtaining the member 'ndim' of a type (line 453)
    ndim_120692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 33), input_120691, 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 32), list_120690, ndim_120692)
    # Adding element type (line 453)
    # Getting the type of 'input' (line 453)
    input_120693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 45), 'input')
    # Obtaining the member 'ndim' of a type (line 453)
    ndim_120694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 45), input_120693, 'ndim')
    int_120695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 58), 'int')
    # Applying the binary operator '+' (line 453)
    result_add_120696 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 45), '+', ndim_120694, int_120695)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 453, 32), list_120690, result_add_120696)
    
    # Applying the binary operator 'in' (line 453)
    result_contains_120697 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 13), 'in', subscript_call_result_120689, list_120690)
    
    # Applying the binary operator 'and' (line 452)
    result_and_keyword_120698 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 8), 'and', result_and_keyword_120684, result_contains_120697)
    
    # Testing the type of an if condition (line 452)
    if_condition_120699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 452, 4), result_and_keyword_120698)
    # Assigning a type to the variable 'if_condition_120699' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'if_condition_120699', if_condition_120699)
    # SSA begins for if statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_120700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 24), 'int')
    # Getting the type of 'matrix' (line 454)
    matrix_120701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'matrix')
    # Obtaining the member 'shape' of a type (line 454)
    shape_120702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 11), matrix_120701, 'shape')
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___120703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 11), shape_120702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_120704 = invoke(stypy.reporting.localization.Localization(__file__, 454, 11), getitem___120703, int_120700)
    
    # Getting the type of 'input' (line 454)
    input_120705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 30), 'input')
    # Obtaining the member 'ndim' of a type (line 454)
    ndim_120706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 30), input_120705, 'ndim')
    int_120707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 43), 'int')
    # Applying the binary operator '+' (line 454)
    result_add_120708 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 30), '+', ndim_120706, int_120707)
    
    # Applying the binary operator '==' (line 454)
    result_eq_120709 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 11), '==', subscript_call_result_120704, result_add_120708)
    
    # Testing the type of an if condition (line 454)
    if_condition_120710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), result_eq_120709)
    # Assigning a type to the variable 'if_condition_120710' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_120710', if_condition_120710)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 455):
    
    # Assigning a BinOp to a Name (line 455):
    
    # Obtaining an instance of the builtin type 'list' (line 455)
    list_120711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 455)
    # Adding element type (line 455)
    int_120712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 20), list_120711, int_120712)
    
    # Getting the type of 'input' (line 455)
    input_120713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'input')
    # Obtaining the member 'ndim' of a type (line 455)
    ndim_120714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 26), input_120713, 'ndim')
    # Applying the binary operator '*' (line 455)
    result_mul_120715 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 20), '*', list_120711, ndim_120714)
    
    
    # Obtaining an instance of the builtin type 'list' (line 455)
    list_120716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 455)
    # Adding element type (line 455)
    int_120717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 39), list_120716, int_120717)
    
    # Applying the binary operator '+' (line 455)
    result_add_120718 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 20), '+', result_mul_120715, list_120716)
    
    # Assigning a type to the variable 'exptd' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'exptd', result_add_120718)
    
    
    
    # Call to all(...): (line 456)
    # Processing the call arguments (line 456)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'input' (line 456)
    input_120721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 36), 'input', False)
    # Obtaining the member 'ndim' of a type (line 456)
    ndim_120722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 36), input_120721, 'ndim')
    # Getting the type of 'matrix' (line 456)
    matrix_120723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'matrix', False)
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___120724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 29), matrix_120723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_120725 = invoke(stypy.reporting.localization.Localization(__file__, 456, 29), getitem___120724, ndim_120722)
    
    # Getting the type of 'exptd' (line 456)
    exptd_120726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'exptd', False)
    # Applying the binary operator '==' (line 456)
    result_eq_120727 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 29), '==', subscript_call_result_120725, exptd_120726)
    
    # Processing the call keyword arguments (line 456)
    kwargs_120728 = {}
    # Getting the type of 'numpy' (line 456)
    numpy_120719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 19), 'numpy', False)
    # Obtaining the member 'all' of a type (line 456)
    all_120720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 19), numpy_120719, 'all')
    # Calling all(args, kwargs) (line 456)
    all_call_result_120729 = invoke(stypy.reporting.localization.Localization(__file__, 456, 19), all_120720, *[result_eq_120727], **kwargs_120728)
    
    # Applying the 'not' unary operator (line 456)
    result_not__120730 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 15), 'not', all_call_result_120729)
    
    # Testing the type of an if condition (line 456)
    if_condition_120731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 12), result_not__120730)
    # Assigning a type to the variable 'if_condition_120731' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'if_condition_120731', if_condition_120731)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 457):
    
    # Assigning a BinOp to a Name (line 457):
    str_120732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 23), 'str', 'Expected homogeneous transformation matrix with shape %s for image shape %s, but bottom row was not equal to %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 459)
    tuple_120733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 459)
    # Adding element type (line 459)
    # Getting the type of 'matrix' (line 459)
    matrix_120734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 44), 'matrix')
    # Obtaining the member 'shape' of a type (line 459)
    shape_120735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 44), matrix_120734, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 44), tuple_120733, shape_120735)
    # Adding element type (line 459)
    # Getting the type of 'input' (line 459)
    input_120736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 58), 'input')
    # Obtaining the member 'shape' of a type (line 459)
    shape_120737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 58), input_120736, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 44), tuple_120733, shape_120737)
    # Adding element type (line 459)
    # Getting the type of 'exptd' (line 459)
    exptd_120738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 71), 'exptd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 44), tuple_120733, exptd_120738)
    
    # Applying the binary operator '%' (line 457)
    result_mod_120739 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 23), '%', str_120732, tuple_120733)
    
    # Assigning a type to the variable 'msg' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 16), 'msg', result_mod_120739)
    
    # Call to ValueError(...): (line 460)
    # Processing the call arguments (line 460)
    # Getting the type of 'msg' (line 460)
    msg_120741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 33), 'msg', False)
    # Processing the call keyword arguments (line 460)
    kwargs_120742 = {}
    # Getting the type of 'ValueError' (line 460)
    ValueError_120740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 460)
    ValueError_call_result_120743 = invoke(stypy.reporting.localization.Localization(__file__, 460, 22), ValueError_120740, *[msg_120741], **kwargs_120742)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 460, 16), ValueError_call_result_120743, 'raise parameter', BaseException)
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 462):
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    # Getting the type of 'input' (line 462)
    input_120744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 25), 'input')
    # Obtaining the member 'ndim' of a type (line 462)
    ndim_120745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 25), input_120744, 'ndim')
    slice_120746 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 462, 17), None, ndim_120745, None)
    # Getting the type of 'input' (line 462)
    input_120747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 37), 'input')
    # Obtaining the member 'ndim' of a type (line 462)
    ndim_120748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 37), input_120747, 'ndim')
    # Getting the type of 'matrix' (line 462)
    matrix_120749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'matrix')
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___120750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 17), matrix_120749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_120751 = invoke(stypy.reporting.localization.Localization(__file__, 462, 17), getitem___120750, (slice_120746, ndim_120748))
    
    # Assigning a type to the variable 'offset' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'offset', subscript_call_result_120751)
    
    # Assigning a Subscript to a Name (line 463):
    
    # Assigning a Subscript to a Name (line 463):
    
    # Obtaining the type of the subscript
    # Getting the type of 'input' (line 463)
    input_120752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'input')
    # Obtaining the member 'ndim' of a type (line 463)
    ndim_120753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 25), input_120752, 'ndim')
    slice_120754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 17), None, ndim_120753, None)
    # Getting the type of 'input' (line 463)
    input_120755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 38), 'input')
    # Obtaining the member 'ndim' of a type (line 463)
    ndim_120756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 38), input_120755, 'ndim')
    slice_120757 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 17), None, ndim_120756, None)
    # Getting the type of 'matrix' (line 463)
    matrix_120758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 17), 'matrix')
    # Obtaining the member '__getitem__' of a type (line 463)
    getitem___120759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 17), matrix_120758, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 463)
    subscript_call_result_120760 = invoke(stypy.reporting.localization.Localization(__file__, 463, 17), getitem___120759, (slice_120754, slice_120757))
    
    # Assigning a type to the variable 'matrix' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'matrix', subscript_call_result_120760)
    # SSA join for if statement (line 452)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_120761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 20), 'int')
    # Getting the type of 'matrix' (line 464)
    matrix_120762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 7), 'matrix')
    # Obtaining the member 'shape' of a type (line 464)
    shape_120763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 7), matrix_120762, 'shape')
    # Obtaining the member '__getitem__' of a type (line 464)
    getitem___120764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 7), shape_120763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 464)
    subscript_call_result_120765 = invoke(stypy.reporting.localization.Localization(__file__, 464, 7), getitem___120764, int_120761)
    
    # Getting the type of 'input' (line 464)
    input_120766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 26), 'input')
    # Obtaining the member 'ndim' of a type (line 464)
    ndim_120767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 26), input_120766, 'ndim')
    # Applying the binary operator '!=' (line 464)
    result_ne_120768 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 7), '!=', subscript_call_result_120765, ndim_120767)
    
    # Testing the type of an if condition (line 464)
    if_condition_120769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 4), result_ne_120768)
    # Assigning a type to the variable 'if_condition_120769' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'if_condition_120769', if_condition_120769)
    # SSA begins for if statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 465)
    # Processing the call arguments (line 465)
    str_120771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 27), 'str', 'affine matrix has wrong number of rows')
    # Processing the call keyword arguments (line 465)
    kwargs_120772 = {}
    # Getting the type of 'RuntimeError' (line 465)
    RuntimeError_120770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 465)
    RuntimeError_call_result_120773 = invoke(stypy.reporting.localization.Localization(__file__, 465, 14), RuntimeError_120770, *[str_120771], **kwargs_120772)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 465, 8), RuntimeError_call_result_120773, 'raise parameter', BaseException)
    # SSA join for if statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'matrix' (line 466)
    matrix_120774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'matrix')
    # Obtaining the member 'ndim' of a type (line 466)
    ndim_120775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 7), matrix_120774, 'ndim')
    int_120776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 22), 'int')
    # Applying the binary operator '==' (line 466)
    result_eq_120777 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), '==', ndim_120775, int_120776)
    
    
    
    # Obtaining the type of the subscript
    int_120778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 41), 'int')
    # Getting the type of 'matrix' (line 466)
    matrix_120779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 28), 'matrix')
    # Obtaining the member 'shape' of a type (line 466)
    shape_120780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 28), matrix_120779, 'shape')
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___120781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 28), shape_120780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_120782 = invoke(stypy.reporting.localization.Localization(__file__, 466, 28), getitem___120781, int_120778)
    
    # Getting the type of 'output' (line 466)
    output_120783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 47), 'output')
    # Obtaining the member 'ndim' of a type (line 466)
    ndim_120784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 47), output_120783, 'ndim')
    # Applying the binary operator '!=' (line 466)
    result_ne_120785 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 28), '!=', subscript_call_result_120782, ndim_120784)
    
    # Applying the binary operator 'and' (line 466)
    result_and_keyword_120786 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), 'and', result_eq_120777, result_ne_120785)
    
    # Testing the type of an if condition (line 466)
    if_condition_120787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 4), result_and_keyword_120786)
    # Assigning a type to the variable 'if_condition_120787' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'if_condition_120787', if_condition_120787)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 467)
    # Processing the call arguments (line 467)
    str_120789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 27), 'str', 'affine matrix has wrong number of columns')
    # Processing the call keyword arguments (line 467)
    kwargs_120790 = {}
    # Getting the type of 'RuntimeError' (line 467)
    RuntimeError_120788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 467)
    RuntimeError_call_result_120791 = invoke(stypy.reporting.localization.Localization(__file__, 467, 14), RuntimeError_120788, *[str_120789], **kwargs_120790)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 467, 8), RuntimeError_call_result_120791, 'raise parameter', BaseException)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'matrix' (line 468)
    matrix_120792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'matrix')
    # Obtaining the member 'flags' of a type (line 468)
    flags_120793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), matrix_120792, 'flags')
    # Obtaining the member 'contiguous' of a type (line 468)
    contiguous_120794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), flags_120793, 'contiguous')
    # Applying the 'not' unary operator (line 468)
    result_not__120795 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 7), 'not', contiguous_120794)
    
    # Testing the type of an if condition (line 468)
    if_condition_120796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 4), result_not__120795)
    # Assigning a type to the variable 'if_condition_120796' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'if_condition_120796', if_condition_120796)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to copy(...): (line 469)
    # Processing the call keyword arguments (line 469)
    kwargs_120799 = {}
    # Getting the type of 'matrix' (line 469)
    matrix_120797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 17), 'matrix', False)
    # Obtaining the member 'copy' of a type (line 469)
    copy_120798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 17), matrix_120797, 'copy')
    # Calling copy(args, kwargs) (line 469)
    copy_call_result_120800 = invoke(stypy.reporting.localization.Localization(__file__, 469, 17), copy_120798, *[], **kwargs_120799)
    
    # Assigning a type to the variable 'matrix' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'matrix', copy_call_result_120800)
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 470):
    
    # Assigning a Call to a Name (line 470):
    
    # Call to _normalize_sequence(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'offset' (line 470)
    offset_120803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 45), 'offset', False)
    # Getting the type of 'input' (line 470)
    input_120804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 53), 'input', False)
    # Obtaining the member 'ndim' of a type (line 470)
    ndim_120805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 53), input_120804, 'ndim')
    # Processing the call keyword arguments (line 470)
    kwargs_120806 = {}
    # Getting the type of '_ni_support' (line 470)
    _ni_support_120801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 470)
    _normalize_sequence_120802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 13), _ni_support_120801, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 470)
    _normalize_sequence_call_result_120807 = invoke(stypy.reporting.localization.Localization(__file__, 470, 13), _normalize_sequence_120802, *[offset_120803, ndim_120805], **kwargs_120806)
    
    # Assigning a type to the variable 'offset' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'offset', _normalize_sequence_call_result_120807)
    
    # Assigning a Call to a Name (line 471):
    
    # Assigning a Call to a Name (line 471):
    
    # Call to asarray(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'offset' (line 471)
    offset_120810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 27), 'offset', False)
    # Processing the call keyword arguments (line 471)
    # Getting the type of 'numpy' (line 471)
    numpy_120811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 471)
    float64_120812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 41), numpy_120811, 'float64')
    keyword_120813 = float64_120812
    kwargs_120814 = {'dtype': keyword_120813}
    # Getting the type of 'numpy' (line 471)
    numpy_120808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 471)
    asarray_120809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 13), numpy_120808, 'asarray')
    # Calling asarray(args, kwargs) (line 471)
    asarray_call_result_120815 = invoke(stypy.reporting.localization.Localization(__file__, 471, 13), asarray_120809, *[offset_120810], **kwargs_120814)
    
    # Assigning a type to the variable 'offset' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'offset', asarray_call_result_120815)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'offset' (line 472)
    offset_120816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 7), 'offset')
    # Obtaining the member 'ndim' of a type (line 472)
    ndim_120817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 7), offset_120816, 'ndim')
    int_120818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 22), 'int')
    # Applying the binary operator '!=' (line 472)
    result_ne_120819 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 7), '!=', ndim_120817, int_120818)
    
    
    
    # Obtaining the type of the subscript
    int_120820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 40), 'int')
    # Getting the type of 'offset' (line 472)
    offset_120821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 27), 'offset')
    # Obtaining the member 'shape' of a type (line 472)
    shape_120822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 27), offset_120821, 'shape')
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___120823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 27), shape_120822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_120824 = invoke(stypy.reporting.localization.Localization(__file__, 472, 27), getitem___120823, int_120820)
    
    int_120825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 45), 'int')
    # Applying the binary operator '<' (line 472)
    result_lt_120826 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 27), '<', subscript_call_result_120824, int_120825)
    
    # Applying the binary operator 'or' (line 472)
    result_or_keyword_120827 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 7), 'or', result_ne_120819, result_lt_120826)
    
    # Testing the type of an if condition (line 472)
    if_condition_120828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 4), result_or_keyword_120827)
    # Assigning a type to the variable 'if_condition_120828' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'if_condition_120828', if_condition_120828)
    # SSA begins for if statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 473)
    # Processing the call arguments (line 473)
    str_120830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 27), 'str', 'no proper offset provided')
    # Processing the call keyword arguments (line 473)
    kwargs_120831 = {}
    # Getting the type of 'RuntimeError' (line 473)
    RuntimeError_120829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 473)
    RuntimeError_call_result_120832 = invoke(stypy.reporting.localization.Localization(__file__, 473, 14), RuntimeError_120829, *[str_120830], **kwargs_120831)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 473, 8), RuntimeError_call_result_120832, 'raise parameter', BaseException)
    # SSA join for if statement (line 472)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'offset' (line 474)
    offset_120833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'offset')
    # Obtaining the member 'flags' of a type (line 474)
    flags_120834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), offset_120833, 'flags')
    # Obtaining the member 'contiguous' of a type (line 474)
    contiguous_120835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), flags_120834, 'contiguous')
    # Applying the 'not' unary operator (line 474)
    result_not__120836 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 7), 'not', contiguous_120835)
    
    # Testing the type of an if condition (line 474)
    if_condition_120837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 4), result_not__120836)
    # Assigning a type to the variable 'if_condition_120837' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'if_condition_120837', if_condition_120837)
    # SSA begins for if statement (line 474)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to copy(...): (line 475)
    # Processing the call keyword arguments (line 475)
    kwargs_120840 = {}
    # Getting the type of 'offset' (line 475)
    offset_120838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 17), 'offset', False)
    # Obtaining the member 'copy' of a type (line 475)
    copy_120839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 17), offset_120838, 'copy')
    # Calling copy(args, kwargs) (line 475)
    copy_call_result_120841 = invoke(stypy.reporting.localization.Localization(__file__, 475, 17), copy_120839, *[], **kwargs_120840)
    
    # Assigning a type to the variable 'offset' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'offset', copy_call_result_120841)
    # SSA join for if statement (line 474)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'matrix' (line 476)
    matrix_120842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 7), 'matrix')
    # Obtaining the member 'ndim' of a type (line 476)
    ndim_120843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 7), matrix_120842, 'ndim')
    int_120844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 22), 'int')
    # Applying the binary operator '==' (line 476)
    result_eq_120845 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 7), '==', ndim_120843, int_120844)
    
    # Testing the type of an if condition (line 476)
    if_condition_120846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 4), result_eq_120845)
    # Assigning a type to the variable 'if_condition_120846' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'if_condition_120846', if_condition_120846)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 477)
    # Processing the call arguments (line 477)
    str_120849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 12), 'str', 'The behaviour of affine_transform with a one-dimensional array supplied for the matrix parameter has changed in scipy 0.18.0.')
    # Processing the call keyword arguments (line 477)
    kwargs_120850 = {}
    # Getting the type of 'warnings' (line 477)
    warnings_120847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 477)
    warn_120848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), warnings_120847, 'warn')
    # Calling warn(args, kwargs) (line 477)
    warn_call_result_120851 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), warn_120848, *[str_120849], **kwargs_120850)
    
    
    # Call to zoom_shift(...): (line 482)
    # Processing the call arguments (line 482)
    # Getting the type of 'filtered' (line 482)
    filtered_120854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 29), 'filtered', False)
    # Getting the type of 'matrix' (line 482)
    matrix_120855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 39), 'matrix', False)
    # Getting the type of 'offset' (line 482)
    offset_120856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 47), 'offset', False)
    # Getting the type of 'matrix' (line 482)
    matrix_120857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 54), 'matrix', False)
    # Applying the binary operator 'div' (line 482)
    result_div_120858 = python_operator(stypy.reporting.localization.Localization(__file__, 482, 47), 'div', offset_120856, matrix_120857)
    
    # Getting the type of 'output' (line 482)
    output_120859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 62), 'output', False)
    # Getting the type of 'order' (line 482)
    order_120860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 70), 'order', False)
    # Getting the type of 'mode' (line 483)
    mode_120861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 29), 'mode', False)
    # Getting the type of 'cval' (line 483)
    cval_120862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 35), 'cval', False)
    # Processing the call keyword arguments (line 482)
    kwargs_120863 = {}
    # Getting the type of '_nd_image' (line 482)
    _nd_image_120852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), '_nd_image', False)
    # Obtaining the member 'zoom_shift' of a type (line 482)
    zoom_shift_120853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), _nd_image_120852, 'zoom_shift')
    # Calling zoom_shift(args, kwargs) (line 482)
    zoom_shift_call_result_120864 = invoke(stypy.reporting.localization.Localization(__file__, 482, 8), zoom_shift_120853, *[filtered_120854, matrix_120855, result_div_120858, output_120859, order_120860, mode_120861, cval_120862], **kwargs_120863)
    
    # SSA branch for the else part of an if statement (line 476)
    module_type_store.open_ssa_branch('else')
    
    # Call to geometric_transform(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'filtered' (line 485)
    filtered_120867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 38), 'filtered', False)
    # Getting the type of 'None' (line 485)
    None_120868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 48), 'None', False)
    # Getting the type of 'None' (line 485)
    None_120869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 54), 'None', False)
    # Getting the type of 'matrix' (line 485)
    matrix_120870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 60), 'matrix', False)
    # Getting the type of 'offset' (line 485)
    offset_120871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 68), 'offset', False)
    # Getting the type of 'output' (line 486)
    output_120872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 38), 'output', False)
    # Getting the type of 'order' (line 486)
    order_120873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 46), 'order', False)
    # Getting the type of 'mode' (line 486)
    mode_120874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 53), 'mode', False)
    # Getting the type of 'cval' (line 486)
    cval_120875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 59), 'cval', False)
    # Getting the type of 'None' (line 486)
    None_120876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 65), 'None', False)
    # Getting the type of 'None' (line 486)
    None_120877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 71), 'None', False)
    # Processing the call keyword arguments (line 485)
    kwargs_120878 = {}
    # Getting the type of '_nd_image' (line 485)
    _nd_image_120865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), '_nd_image', False)
    # Obtaining the member 'geometric_transform' of a type (line 485)
    geometric_transform_120866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), _nd_image_120865, 'geometric_transform')
    # Calling geometric_transform(args, kwargs) (line 485)
    geometric_transform_call_result_120879 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), geometric_transform_120866, *[filtered_120867, None_120868, None_120869, matrix_120870, offset_120871, output_120872, order_120873, mode_120874, cval_120875, None_120876, None_120877], **kwargs_120878)
    
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 487)
    return_value_120880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 487)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'stypy_return_type', return_value_120880)
    
    # ################# End of 'affine_transform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'affine_transform' in the type store
    # Getting the type of 'stypy_return_type' (line 347)
    stypy_return_type_120881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_120881)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'affine_transform'
    return stypy_return_type_120881

# Assigning a type to the variable 'affine_transform' (line 347)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'affine_transform', affine_transform)

@norecursion
def shift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 490)
    None_120882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 31), 'None')
    int_120883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 43), 'int')
    str_120884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 51), 'str', 'constant')
    float_120885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 68), 'float')
    # Getting the type of 'True' (line 491)
    True_120886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 20), 'True')
    defaults = [None_120882, int_120883, str_120884, float_120885, True_120886]
    # Create a new context for function 'shift'
    module_type_store = module_type_store.open_function_context('shift', 490, 0, False)
    
    # Passed parameters checking function
    shift.stypy_localization = localization
    shift.stypy_type_of_self = None
    shift.stypy_type_store = module_type_store
    shift.stypy_function_name = 'shift'
    shift.stypy_param_names_list = ['input', 'shift', 'output', 'order', 'mode', 'cval', 'prefilter']
    shift.stypy_varargs_param_name = None
    shift.stypy_kwargs_param_name = None
    shift.stypy_call_defaults = defaults
    shift.stypy_call_varargs = varargs
    shift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'shift', ['input', 'shift', 'output', 'order', 'mode', 'cval', 'prefilter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'shift', localization, ['input', 'shift', 'output', 'order', 'mode', 'cval', 'prefilter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'shift(...)' code ##################

    str_120887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, (-1)), 'str', "\n    Shift an array.\n\n    The array is shifted using spline interpolation of the requested order.\n    Points outside the boundaries of the input are filled according to the\n    given mode.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input array.\n    shift : float or sequence\n        The shift along the axes. If a float, `shift` is the same for each\n        axis. If a sequence, `shift` should contain one value for each axis.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n\n    Returns\n    -------\n    shift : ndarray or None\n        The shifted input. If `output` is given as a parameter, None is\n        returned.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 532)
    order_120888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 7), 'order')
    int_120889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 15), 'int')
    # Applying the binary operator '<' (line 532)
    result_lt_120890 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 7), '<', order_120888, int_120889)
    
    
    # Getting the type of 'order' (line 532)
    order_120891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 20), 'order')
    int_120892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 28), 'int')
    # Applying the binary operator '>' (line 532)
    result_gt_120893 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 20), '>', order_120891, int_120892)
    
    # Applying the binary operator 'or' (line 532)
    result_or_keyword_120894 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 7), 'or', result_lt_120890, result_gt_120893)
    
    # Testing the type of an if condition (line 532)
    if_condition_120895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 4), result_or_keyword_120894)
    # Assigning a type to the variable 'if_condition_120895' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'if_condition_120895', if_condition_120895)
    # SSA begins for if statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 533)
    # Processing the call arguments (line 533)
    str_120897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 533)
    kwargs_120898 = {}
    # Getting the type of 'RuntimeError' (line 533)
    RuntimeError_120896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 533)
    RuntimeError_call_result_120899 = invoke(stypy.reporting.localization.Localization(__file__, 533, 14), RuntimeError_120896, *[str_120897], **kwargs_120898)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 533, 8), RuntimeError_call_result_120899, 'raise parameter', BaseException)
    # SSA join for if statement (line 532)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 534):
    
    # Assigning a Call to a Name (line 534):
    
    # Call to asarray(...): (line 534)
    # Processing the call arguments (line 534)
    # Getting the type of 'input' (line 534)
    input_120902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 26), 'input', False)
    # Processing the call keyword arguments (line 534)
    kwargs_120903 = {}
    # Getting the type of 'numpy' (line 534)
    numpy_120900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 534)
    asarray_120901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 12), numpy_120900, 'asarray')
    # Calling asarray(args, kwargs) (line 534)
    asarray_call_result_120904 = invoke(stypy.reporting.localization.Localization(__file__, 534, 12), asarray_120901, *[input_120902], **kwargs_120903)
    
    # Assigning a type to the variable 'input' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'input', asarray_call_result_120904)
    
    
    # Call to iscomplexobj(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'input' (line 535)
    input_120907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 26), 'input', False)
    # Processing the call keyword arguments (line 535)
    kwargs_120908 = {}
    # Getting the type of 'numpy' (line 535)
    numpy_120905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 535)
    iscomplexobj_120906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 7), numpy_120905, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 535)
    iscomplexobj_call_result_120909 = invoke(stypy.reporting.localization.Localization(__file__, 535, 7), iscomplexobj_120906, *[input_120907], **kwargs_120908)
    
    # Testing the type of an if condition (line 535)
    if_condition_120910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 4), iscomplexobj_call_result_120909)
    # Assigning a type to the variable 'if_condition_120910' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'if_condition_120910', if_condition_120910)
    # SSA begins for if statement (line 535)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 536)
    # Processing the call arguments (line 536)
    str_120912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 536)
    kwargs_120913 = {}
    # Getting the type of 'TypeError' (line 536)
    TypeError_120911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 536)
    TypeError_call_result_120914 = invoke(stypy.reporting.localization.Localization(__file__, 536, 14), TypeError_120911, *[str_120912], **kwargs_120913)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 536, 8), TypeError_call_result_120914, 'raise parameter', BaseException)
    # SSA join for if statement (line 535)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'input' (line 537)
    input_120915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 537)
    ndim_120916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 7), input_120915, 'ndim')
    int_120917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 20), 'int')
    # Applying the binary operator '<' (line 537)
    result_lt_120918 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 7), '<', ndim_120916, int_120917)
    
    # Testing the type of an if condition (line 537)
    if_condition_120919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 4), result_lt_120918)
    # Assigning a type to the variable 'if_condition_120919' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'if_condition_120919', if_condition_120919)
    # SSA begins for if statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 538)
    # Processing the call arguments (line 538)
    str_120921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 27), 'str', 'input and output rank must be > 0')
    # Processing the call keyword arguments (line 538)
    kwargs_120922 = {}
    # Getting the type of 'RuntimeError' (line 538)
    RuntimeError_120920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 538)
    RuntimeError_call_result_120923 = invoke(stypy.reporting.localization.Localization(__file__, 538, 14), RuntimeError_120920, *[str_120921], **kwargs_120922)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 538, 8), RuntimeError_call_result_120923, 'raise parameter', BaseException)
    # SSA join for if statement (line 537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 539):
    
    # Assigning a Call to a Name (line 539):
    
    # Call to _extend_mode_to_code(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'mode' (line 539)
    mode_120925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 32), 'mode', False)
    # Processing the call keyword arguments (line 539)
    kwargs_120926 = {}
    # Getting the type of '_extend_mode_to_code' (line 539)
    _extend_mode_to_code_120924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), '_extend_mode_to_code', False)
    # Calling _extend_mode_to_code(args, kwargs) (line 539)
    _extend_mode_to_code_call_result_120927 = invoke(stypy.reporting.localization.Localization(__file__, 539, 11), _extend_mode_to_code_120924, *[mode_120925], **kwargs_120926)
    
    # Assigning a type to the variable 'mode' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'mode', _extend_mode_to_code_call_result_120927)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prefilter' (line 540)
    prefilter_120928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 7), 'prefilter')
    
    # Getting the type of 'order' (line 540)
    order_120929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), 'order')
    int_120930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 29), 'int')
    # Applying the binary operator '>' (line 540)
    result_gt_120931 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 21), '>', order_120929, int_120930)
    
    # Applying the binary operator 'and' (line 540)
    result_and_keyword_120932 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 7), 'and', prefilter_120928, result_gt_120931)
    
    # Testing the type of an if condition (line 540)
    if_condition_120933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 4), result_and_keyword_120932)
    # Assigning a type to the variable 'if_condition_120933' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'if_condition_120933', if_condition_120933)
    # SSA begins for if statement (line 540)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 541):
    
    # Assigning a Call to a Name (line 541):
    
    # Call to spline_filter(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'input' (line 541)
    input_120935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 33), 'input', False)
    # Getting the type of 'order' (line 541)
    order_120936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 40), 'order', False)
    # Processing the call keyword arguments (line 541)
    # Getting the type of 'numpy' (line 541)
    numpy_120937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 54), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 541)
    float64_120938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 54), numpy_120937, 'float64')
    keyword_120939 = float64_120938
    kwargs_120940 = {'output': keyword_120939}
    # Getting the type of 'spline_filter' (line 541)
    spline_filter_120934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'spline_filter', False)
    # Calling spline_filter(args, kwargs) (line 541)
    spline_filter_call_result_120941 = invoke(stypy.reporting.localization.Localization(__file__, 541, 19), spline_filter_120934, *[input_120935, order_120936], **kwargs_120940)
    
    # Assigning a type to the variable 'filtered' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'filtered', spline_filter_call_result_120941)
    # SSA branch for the else part of an if statement (line 540)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 543):
    
    # Assigning a Name to a Name (line 543):
    # Getting the type of 'input' (line 543)
    input_120942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 19), 'input')
    # Assigning a type to the variable 'filtered' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 8), 'filtered', input_120942)
    # SSA join for if statement (line 540)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 544):
    
    # Assigning a Subscript to a Name (line 544):
    
    # Obtaining the type of the subscript
    int_120943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 4), 'int')
    
    # Call to _get_output(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'output' (line 544)
    output_120946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 51), 'output', False)
    # Getting the type of 'input' (line 544)
    input_120947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 59), 'input', False)
    # Processing the call keyword arguments (line 544)
    kwargs_120948 = {}
    # Getting the type of '_ni_support' (line 544)
    _ni_support_120944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 544)
    _get_output_120945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 27), _ni_support_120944, '_get_output')
    # Calling _get_output(args, kwargs) (line 544)
    _get_output_call_result_120949 = invoke(stypy.reporting.localization.Localization(__file__, 544, 27), _get_output_120945, *[output_120946, input_120947], **kwargs_120948)
    
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___120950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 4), _get_output_call_result_120949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_120951 = invoke(stypy.reporting.localization.Localization(__file__, 544, 4), getitem___120950, int_120943)
    
    # Assigning a type to the variable 'tuple_var_assignment_120081' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'tuple_var_assignment_120081', subscript_call_result_120951)
    
    # Assigning a Subscript to a Name (line 544):
    
    # Obtaining the type of the subscript
    int_120952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 4), 'int')
    
    # Call to _get_output(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'output' (line 544)
    output_120955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 51), 'output', False)
    # Getting the type of 'input' (line 544)
    input_120956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 59), 'input', False)
    # Processing the call keyword arguments (line 544)
    kwargs_120957 = {}
    # Getting the type of '_ni_support' (line 544)
    _ni_support_120953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 544)
    _get_output_120954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 27), _ni_support_120953, '_get_output')
    # Calling _get_output(args, kwargs) (line 544)
    _get_output_call_result_120958 = invoke(stypy.reporting.localization.Localization(__file__, 544, 27), _get_output_120954, *[output_120955, input_120956], **kwargs_120957)
    
    # Obtaining the member '__getitem__' of a type (line 544)
    getitem___120959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 4), _get_output_call_result_120958, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 544)
    subscript_call_result_120960 = invoke(stypy.reporting.localization.Localization(__file__, 544, 4), getitem___120959, int_120952)
    
    # Assigning a type to the variable 'tuple_var_assignment_120082' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'tuple_var_assignment_120082', subscript_call_result_120960)
    
    # Assigning a Name to a Name (line 544):
    # Getting the type of 'tuple_var_assignment_120081' (line 544)
    tuple_var_assignment_120081_120961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'tuple_var_assignment_120081')
    # Assigning a type to the variable 'output' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'output', tuple_var_assignment_120081_120961)
    
    # Assigning a Name to a Name (line 544):
    # Getting the type of 'tuple_var_assignment_120082' (line 544)
    tuple_var_assignment_120082_120962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'tuple_var_assignment_120082')
    # Assigning a type to the variable 'return_value' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'return_value', tuple_var_assignment_120082_120962)
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to _normalize_sequence(...): (line 545)
    # Processing the call arguments (line 545)
    # Getting the type of 'shift' (line 545)
    shift_120965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 44), 'shift', False)
    # Getting the type of 'input' (line 545)
    input_120966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 51), 'input', False)
    # Obtaining the member 'ndim' of a type (line 545)
    ndim_120967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 51), input_120966, 'ndim')
    # Processing the call keyword arguments (line 545)
    kwargs_120968 = {}
    # Getting the type of '_ni_support' (line 545)
    _ni_support_120963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 545)
    _normalize_sequence_120964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 12), _ni_support_120963, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 545)
    _normalize_sequence_call_result_120969 = invoke(stypy.reporting.localization.Localization(__file__, 545, 12), _normalize_sequence_120964, *[shift_120965, ndim_120967], **kwargs_120968)
    
    # Assigning a type to the variable 'shift' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'shift', _normalize_sequence_call_result_120969)
    
    # Assigning a ListComp to a Name (line 546):
    
    # Assigning a ListComp to a Name (line 546):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'shift' (line 546)
    shift_120972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 27), 'shift')
    comprehension_120973 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 13), shift_120972)
    # Assigning a type to the variable 'ii' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 13), 'ii', comprehension_120973)
    
    # Getting the type of 'ii' (line 546)
    ii_120970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 14), 'ii')
    # Applying the 'usub' unary operator (line 546)
    result___neg___120971 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 13), 'usub', ii_120970)
    
    list_120974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 13), list_120974, result___neg___120971)
    # Assigning a type to the variable 'shift' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'shift', list_120974)
    
    # Assigning a Call to a Name (line 547):
    
    # Assigning a Call to a Name (line 547):
    
    # Call to asarray(...): (line 547)
    # Processing the call arguments (line 547)
    # Getting the type of 'shift' (line 547)
    shift_120977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 26), 'shift', False)
    # Processing the call keyword arguments (line 547)
    # Getting the type of 'numpy' (line 547)
    numpy_120978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 39), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 547)
    float64_120979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 39), numpy_120978, 'float64')
    keyword_120980 = float64_120979
    kwargs_120981 = {'dtype': keyword_120980}
    # Getting the type of 'numpy' (line 547)
    numpy_120975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 547)
    asarray_120976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), numpy_120975, 'asarray')
    # Calling asarray(args, kwargs) (line 547)
    asarray_call_result_120982 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), asarray_120976, *[shift_120977], **kwargs_120981)
    
    # Assigning a type to the variable 'shift' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 4), 'shift', asarray_call_result_120982)
    
    
    # Getting the type of 'shift' (line 548)
    shift_120983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 11), 'shift')
    # Obtaining the member 'flags' of a type (line 548)
    flags_120984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), shift_120983, 'flags')
    # Obtaining the member 'contiguous' of a type (line 548)
    contiguous_120985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 11), flags_120984, 'contiguous')
    # Applying the 'not' unary operator (line 548)
    result_not__120986 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 7), 'not', contiguous_120985)
    
    # Testing the type of an if condition (line 548)
    if_condition_120987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 4), result_not__120986)
    # Assigning a type to the variable 'if_condition_120987' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'if_condition_120987', if_condition_120987)
    # SSA begins for if statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 549):
    
    # Assigning a Call to a Name (line 549):
    
    # Call to copy(...): (line 549)
    # Processing the call keyword arguments (line 549)
    kwargs_120990 = {}
    # Getting the type of 'shift' (line 549)
    shift_120988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'shift', False)
    # Obtaining the member 'copy' of a type (line 549)
    copy_120989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 16), shift_120988, 'copy')
    # Calling copy(args, kwargs) (line 549)
    copy_call_result_120991 = invoke(stypy.reporting.localization.Localization(__file__, 549, 16), copy_120989, *[], **kwargs_120990)
    
    # Assigning a type to the variable 'shift' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'shift', copy_call_result_120991)
    # SSA join for if statement (line 548)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to zoom_shift(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'filtered' (line 550)
    filtered_120994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 25), 'filtered', False)
    # Getting the type of 'None' (line 550)
    None_120995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'None', False)
    # Getting the type of 'shift' (line 550)
    shift_120996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 41), 'shift', False)
    # Getting the type of 'output' (line 550)
    output_120997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 48), 'output', False)
    # Getting the type of 'order' (line 550)
    order_120998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 56), 'order', False)
    # Getting the type of 'mode' (line 550)
    mode_120999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 63), 'mode', False)
    # Getting the type of 'cval' (line 550)
    cval_121000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 69), 'cval', False)
    # Processing the call keyword arguments (line 550)
    kwargs_121001 = {}
    # Getting the type of '_nd_image' (line 550)
    _nd_image_120992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), '_nd_image', False)
    # Obtaining the member 'zoom_shift' of a type (line 550)
    zoom_shift_120993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 4), _nd_image_120992, 'zoom_shift')
    # Calling zoom_shift(args, kwargs) (line 550)
    zoom_shift_call_result_121002 = invoke(stypy.reporting.localization.Localization(__file__, 550, 4), zoom_shift_120993, *[filtered_120994, None_120995, shift_120996, output_120997, order_120998, mode_120999, cval_121000], **kwargs_121001)
    
    # Getting the type of 'return_value' (line 551)
    return_value_121003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'stypy_return_type', return_value_121003)
    
    # ################# End of 'shift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'shift' in the type store
    # Getting the type of 'stypy_return_type' (line 490)
    stypy_return_type_121004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'shift'
    return stypy_return_type_121004

# Assigning a type to the variable 'shift' (line 490)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'shift', shift)

@norecursion
def zoom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 554)
    None_121005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 29), 'None')
    int_121006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 41), 'int')
    str_121007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 49), 'str', 'constant')
    float_121008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 66), 'float')
    # Getting the type of 'True' (line 555)
    True_121009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 'True')
    defaults = [None_121005, int_121006, str_121007, float_121008, True_121009]
    # Create a new context for function 'zoom'
    module_type_store = module_type_store.open_function_context('zoom', 554, 0, False)
    
    # Passed parameters checking function
    zoom.stypy_localization = localization
    zoom.stypy_type_of_self = None
    zoom.stypy_type_store = module_type_store
    zoom.stypy_function_name = 'zoom'
    zoom.stypy_param_names_list = ['input', 'zoom', 'output', 'order', 'mode', 'cval', 'prefilter']
    zoom.stypy_varargs_param_name = None
    zoom.stypy_kwargs_param_name = None
    zoom.stypy_call_defaults = defaults
    zoom.stypy_call_varargs = varargs
    zoom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zoom', ['input', 'zoom', 'output', 'order', 'mode', 'cval', 'prefilter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zoom', localization, ['input', 'zoom', 'output', 'order', 'mode', 'cval', 'prefilter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zoom(...)' code ##################

    str_121010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, (-1)), 'str', "\n    Zoom an array.\n\n    The array is zoomed using spline interpolation of the requested order.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input array.\n    zoom : float or sequence\n        The zoom factor along the axes. If a float, `zoom` is the same for each\n        axis. If a sequence, `zoom` should contain one value for each axis.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n\n    Returns\n    -------\n    zoom : ndarray or None\n        The zoomed input. If `output` is given as a parameter, None is\n        returned.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'order' (line 594)
    order_121011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 7), 'order')
    int_121012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 15), 'int')
    # Applying the binary operator '<' (line 594)
    result_lt_121013 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 7), '<', order_121011, int_121012)
    
    
    # Getting the type of 'order' (line 594)
    order_121014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 20), 'order')
    int_121015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 28), 'int')
    # Applying the binary operator '>' (line 594)
    result_gt_121016 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 20), '>', order_121014, int_121015)
    
    # Applying the binary operator 'or' (line 594)
    result_or_keyword_121017 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 7), 'or', result_lt_121013, result_gt_121016)
    
    # Testing the type of an if condition (line 594)
    if_condition_121018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 4), result_or_keyword_121017)
    # Assigning a type to the variable 'if_condition_121018' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'if_condition_121018', if_condition_121018)
    # SSA begins for if statement (line 594)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 595)
    # Processing the call arguments (line 595)
    str_121020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 27), 'str', 'spline order not supported')
    # Processing the call keyword arguments (line 595)
    kwargs_121021 = {}
    # Getting the type of 'RuntimeError' (line 595)
    RuntimeError_121019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 595)
    RuntimeError_call_result_121022 = invoke(stypy.reporting.localization.Localization(__file__, 595, 14), RuntimeError_121019, *[str_121020], **kwargs_121021)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 595, 8), RuntimeError_call_result_121022, 'raise parameter', BaseException)
    # SSA join for if statement (line 594)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 596):
    
    # Assigning a Call to a Name (line 596):
    
    # Call to asarray(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'input' (line 596)
    input_121025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 26), 'input', False)
    # Processing the call keyword arguments (line 596)
    kwargs_121026 = {}
    # Getting the type of 'numpy' (line 596)
    numpy_121023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 596)
    asarray_121024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 12), numpy_121023, 'asarray')
    # Calling asarray(args, kwargs) (line 596)
    asarray_call_result_121027 = invoke(stypy.reporting.localization.Localization(__file__, 596, 12), asarray_121024, *[input_121025], **kwargs_121026)
    
    # Assigning a type to the variable 'input' (line 596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'input', asarray_call_result_121027)
    
    
    # Call to iscomplexobj(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'input' (line 597)
    input_121030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 26), 'input', False)
    # Processing the call keyword arguments (line 597)
    kwargs_121031 = {}
    # Getting the type of 'numpy' (line 597)
    numpy_121028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 597)
    iscomplexobj_121029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 7), numpy_121028, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 597)
    iscomplexobj_call_result_121032 = invoke(stypy.reporting.localization.Localization(__file__, 597, 7), iscomplexobj_121029, *[input_121030], **kwargs_121031)
    
    # Testing the type of an if condition (line 597)
    if_condition_121033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 4), iscomplexobj_call_result_121032)
    # Assigning a type to the variable 'if_condition_121033' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'if_condition_121033', if_condition_121033)
    # SSA begins for if statement (line 597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 598)
    # Processing the call arguments (line 598)
    str_121035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 598)
    kwargs_121036 = {}
    # Getting the type of 'TypeError' (line 598)
    TypeError_121034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 598)
    TypeError_call_result_121037 = invoke(stypy.reporting.localization.Localization(__file__, 598, 14), TypeError_121034, *[str_121035], **kwargs_121036)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 598, 8), TypeError_call_result_121037, 'raise parameter', BaseException)
    # SSA join for if statement (line 597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'input' (line 599)
    input_121038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 599)
    ndim_121039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 7), input_121038, 'ndim')
    int_121040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 20), 'int')
    # Applying the binary operator '<' (line 599)
    result_lt_121041 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 7), '<', ndim_121039, int_121040)
    
    # Testing the type of an if condition (line 599)
    if_condition_121042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 4), result_lt_121041)
    # Assigning a type to the variable 'if_condition_121042' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'if_condition_121042', if_condition_121042)
    # SSA begins for if statement (line 599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 600)
    # Processing the call arguments (line 600)
    str_121044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'str', 'input and output rank must be > 0')
    # Processing the call keyword arguments (line 600)
    kwargs_121045 = {}
    # Getting the type of 'RuntimeError' (line 600)
    RuntimeError_121043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 600)
    RuntimeError_call_result_121046 = invoke(stypy.reporting.localization.Localization(__file__, 600, 14), RuntimeError_121043, *[str_121044], **kwargs_121045)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 600, 8), RuntimeError_call_result_121046, 'raise parameter', BaseException)
    # SSA join for if statement (line 599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 601):
    
    # Assigning a Call to a Name (line 601):
    
    # Call to _extend_mode_to_code(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'mode' (line 601)
    mode_121048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 32), 'mode', False)
    # Processing the call keyword arguments (line 601)
    kwargs_121049 = {}
    # Getting the type of '_extend_mode_to_code' (line 601)
    _extend_mode_to_code_121047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 11), '_extend_mode_to_code', False)
    # Calling _extend_mode_to_code(args, kwargs) (line 601)
    _extend_mode_to_code_call_result_121050 = invoke(stypy.reporting.localization.Localization(__file__, 601, 11), _extend_mode_to_code_121047, *[mode_121048], **kwargs_121049)
    
    # Assigning a type to the variable 'mode' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'mode', _extend_mode_to_code_call_result_121050)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'prefilter' (line 602)
    prefilter_121051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 7), 'prefilter')
    
    # Getting the type of 'order' (line 602)
    order_121052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'order')
    int_121053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 29), 'int')
    # Applying the binary operator '>' (line 602)
    result_gt_121054 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 21), '>', order_121052, int_121053)
    
    # Applying the binary operator 'and' (line 602)
    result_and_keyword_121055 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 7), 'and', prefilter_121051, result_gt_121054)
    
    # Testing the type of an if condition (line 602)
    if_condition_121056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 602, 4), result_and_keyword_121055)
    # Assigning a type to the variable 'if_condition_121056' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'if_condition_121056', if_condition_121056)
    # SSA begins for if statement (line 602)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to spline_filter(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'input' (line 603)
    input_121058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 33), 'input', False)
    # Getting the type of 'order' (line 603)
    order_121059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 40), 'order', False)
    # Processing the call keyword arguments (line 603)
    # Getting the type of 'numpy' (line 603)
    numpy_121060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 54), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 603)
    float64_121061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 54), numpy_121060, 'float64')
    keyword_121062 = float64_121061
    kwargs_121063 = {'output': keyword_121062}
    # Getting the type of 'spline_filter' (line 603)
    spline_filter_121057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 19), 'spline_filter', False)
    # Calling spline_filter(args, kwargs) (line 603)
    spline_filter_call_result_121064 = invoke(stypy.reporting.localization.Localization(__file__, 603, 19), spline_filter_121057, *[input_121058, order_121059], **kwargs_121063)
    
    # Assigning a type to the variable 'filtered' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'filtered', spline_filter_call_result_121064)
    # SSA branch for the else part of an if statement (line 602)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 605):
    
    # Assigning a Name to a Name (line 605):
    # Getting the type of 'input' (line 605)
    input_121065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 19), 'input')
    # Assigning a type to the variable 'filtered' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'filtered', input_121065)
    # SSA join for if statement (line 602)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 606):
    
    # Assigning a Call to a Name (line 606):
    
    # Call to _normalize_sequence(...): (line 606)
    # Processing the call arguments (line 606)
    # Getting the type of 'zoom' (line 606)
    zoom_121068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 43), 'zoom', False)
    # Getting the type of 'input' (line 606)
    input_121069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 49), 'input', False)
    # Obtaining the member 'ndim' of a type (line 606)
    ndim_121070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 49), input_121069, 'ndim')
    # Processing the call keyword arguments (line 606)
    kwargs_121071 = {}
    # Getting the type of '_ni_support' (line 606)
    _ni_support_121066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 11), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 606)
    _normalize_sequence_121067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 11), _ni_support_121066, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 606)
    _normalize_sequence_call_result_121072 = invoke(stypy.reporting.localization.Localization(__file__, 606, 11), _normalize_sequence_121067, *[zoom_121068, ndim_121070], **kwargs_121071)
    
    # Assigning a type to the variable 'zoom' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 4), 'zoom', _normalize_sequence_call_result_121072)
    
    # Assigning a Call to a Name (line 607):
    
    # Assigning a Call to a Name (line 607):
    
    # Call to tuple(...): (line 607)
    # Processing the call arguments (line 607)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'input' (line 608)
    input_121084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 51), 'input', False)
    # Obtaining the member 'shape' of a type (line 608)
    shape_121085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 51), input_121084, 'shape')
    # Getting the type of 'zoom' (line 608)
    zoom_121086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 64), 'zoom', False)
    # Processing the call keyword arguments (line 608)
    kwargs_121087 = {}
    # Getting the type of 'zip' (line 608)
    zip_121083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 47), 'zip', False)
    # Calling zip(args, kwargs) (line 608)
    zip_call_result_121088 = invoke(stypy.reporting.localization.Localization(__file__, 608, 47), zip_121083, *[shape_121085, zoom_121086], **kwargs_121087)
    
    comprehension_121089 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 13), zip_call_result_121088)
    # Assigning a type to the variable 'ii' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 13), 'ii', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 13), comprehension_121089))
    # Assigning a type to the variable 'jj' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 13), 'jj', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 13), comprehension_121089))
    
    # Call to int(...): (line 608)
    # Processing the call arguments (line 608)
    
    # Call to round(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'ii' (line 608)
    ii_121076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 23), 'ii', False)
    # Getting the type of 'jj' (line 608)
    jj_121077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 28), 'jj', False)
    # Applying the binary operator '*' (line 608)
    result_mul_121078 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 23), '*', ii_121076, jj_121077)
    
    # Processing the call keyword arguments (line 608)
    kwargs_121079 = {}
    # Getting the type of 'round' (line 608)
    round_121075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 17), 'round', False)
    # Calling round(args, kwargs) (line 608)
    round_call_result_121080 = invoke(stypy.reporting.localization.Localization(__file__, 608, 17), round_121075, *[result_mul_121078], **kwargs_121079)
    
    # Processing the call keyword arguments (line 608)
    kwargs_121081 = {}
    # Getting the type of 'int' (line 608)
    int_121074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 13), 'int', False)
    # Calling int(args, kwargs) (line 608)
    int_call_result_121082 = invoke(stypy.reporting.localization.Localization(__file__, 608, 13), int_121074, *[round_call_result_121080], **kwargs_121081)
    
    list_121090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 608, 13), list_121090, int_call_result_121082)
    # Processing the call keyword arguments (line 607)
    kwargs_121091 = {}
    # Getting the type of 'tuple' (line 607)
    tuple_121073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 19), 'tuple', False)
    # Calling tuple(args, kwargs) (line 607)
    tuple_call_result_121092 = invoke(stypy.reporting.localization.Localization(__file__, 607, 19), tuple_121073, *[list_121090], **kwargs_121091)
    
    # Assigning a type to the variable 'output_shape' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'output_shape', tuple_call_result_121092)
    
    # Assigning a Call to a Name (line 610):
    
    # Assigning a Call to a Name (line 610):
    
    # Call to tuple(...): (line 610)
    # Processing the call arguments (line 610)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'input' (line 611)
    input_121101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 44), 'input', False)
    # Obtaining the member 'shape' of a type (line 611)
    shape_121102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 44), input_121101, 'shape')
    # Getting the type of 'zoom' (line 611)
    zoom_121103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 57), 'zoom', False)
    # Processing the call keyword arguments (line 611)
    kwargs_121104 = {}
    # Getting the type of 'zip' (line 611)
    zip_121100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 40), 'zip', False)
    # Calling zip(args, kwargs) (line 611)
    zip_call_result_121105 = invoke(stypy.reporting.localization.Localization(__file__, 611, 40), zip_121100, *[shape_121102, zoom_121103], **kwargs_121104)
    
    comprehension_121106 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 13), zip_call_result_121105)
    # Assigning a type to the variable 'ii' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'ii', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 13), comprehension_121106))
    # Assigning a type to the variable 'jj' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'jj', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 13), comprehension_121106))
    
    # Call to int(...): (line 611)
    # Processing the call arguments (line 611)
    # Getting the type of 'ii' (line 611)
    ii_121095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 17), 'ii', False)
    # Getting the type of 'jj' (line 611)
    jj_121096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 22), 'jj', False)
    # Applying the binary operator '*' (line 611)
    result_mul_121097 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 17), '*', ii_121095, jj_121096)
    
    # Processing the call keyword arguments (line 611)
    kwargs_121098 = {}
    # Getting the type of 'int' (line 611)
    int_121094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 13), 'int', False)
    # Calling int(args, kwargs) (line 611)
    int_call_result_121099 = invoke(stypy.reporting.localization.Localization(__file__, 611, 13), int_121094, *[result_mul_121097], **kwargs_121098)
    
    list_121107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 13), list_121107, int_call_result_121099)
    # Processing the call keyword arguments (line 610)
    kwargs_121108 = {}
    # Getting the type of 'tuple' (line 610)
    tuple_121093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 23), 'tuple', False)
    # Calling tuple(args, kwargs) (line 610)
    tuple_call_result_121109 = invoke(stypy.reporting.localization.Localization(__file__, 610, 23), tuple_121093, *[list_121107], **kwargs_121108)
    
    # Assigning a type to the variable 'output_shape_old' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'output_shape_old', tuple_call_result_121109)
    
    
    # Getting the type of 'output_shape' (line 612)
    output_shape_121110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 7), 'output_shape')
    # Getting the type of 'output_shape_old' (line 612)
    output_shape_old_121111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 23), 'output_shape_old')
    # Applying the binary operator '!=' (line 612)
    result_ne_121112 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 7), '!=', output_shape_121110, output_shape_old_121111)
    
    # Testing the type of an if condition (line 612)
    if_condition_121113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 4), result_ne_121112)
    # Assigning a type to the variable 'if_condition_121113' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'if_condition_121113', if_condition_121113)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 613)
    # Processing the call arguments (line 613)
    str_121116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 16), 'str', 'From scipy 0.13.0, the output shape of zoom() is calculated with round() instead of int() - for these inputs the size of the returned array has changed.')
    # Getting the type of 'UserWarning' (line 616)
    UserWarning_121117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 51), 'UserWarning', False)
    # Processing the call keyword arguments (line 613)
    kwargs_121118 = {}
    # Getting the type of 'warnings' (line 613)
    warnings_121114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 613)
    warn_121115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), warnings_121114, 'warn')
    # Calling warn(args, kwargs) (line 613)
    warn_call_result_121119 = invoke(stypy.reporting.localization.Localization(__file__, 613, 8), warn_121115, *[str_121116, UserWarning_121117], **kwargs_121118)
    
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 618):
    
    # Assigning a BinOp to a Name (line 618):
    
    # Call to array(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'output_shape' (line 618)
    output_shape_121122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 27), 'output_shape', False)
    # Getting the type of 'float' (line 618)
    float_121123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 41), 'float', False)
    # Processing the call keyword arguments (line 618)
    kwargs_121124 = {}
    # Getting the type of 'numpy' (line 618)
    numpy_121120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 15), 'numpy', False)
    # Obtaining the member 'array' of a type (line 618)
    array_121121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 15), numpy_121120, 'array')
    # Calling array(args, kwargs) (line 618)
    array_call_result_121125 = invoke(stypy.reporting.localization.Localization(__file__, 618, 15), array_121121, *[output_shape_121122, float_121123], **kwargs_121124)
    
    int_121126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 50), 'int')
    # Applying the binary operator '-' (line 618)
    result_sub_121127 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 15), '-', array_call_result_121125, int_121126)
    
    # Assigning a type to the variable 'zoom_div' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'zoom_div', result_sub_121127)
    
    # Assigning a Call to a Name (line 621):
    
    # Assigning a Call to a Name (line 621):
    
    # Call to divide(...): (line 621)
    # Processing the call arguments (line 621)
    
    # Call to array(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'input' (line 621)
    input_121132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 36), 'input', False)
    # Obtaining the member 'shape' of a type (line 621)
    shape_121133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 36), input_121132, 'shape')
    # Processing the call keyword arguments (line 621)
    kwargs_121134 = {}
    # Getting the type of 'numpy' (line 621)
    numpy_121130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'numpy', False)
    # Obtaining the member 'array' of a type (line 621)
    array_121131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 24), numpy_121130, 'array')
    # Calling array(args, kwargs) (line 621)
    array_call_result_121135 = invoke(stypy.reporting.localization.Localization(__file__, 621, 24), array_121131, *[shape_121133], **kwargs_121134)
    
    int_121136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 51), 'int')
    # Applying the binary operator '-' (line 621)
    result_sub_121137 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 24), '-', array_call_result_121135, int_121136)
    
    # Getting the type of 'zoom_div' (line 621)
    zoom_div_121138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 54), 'zoom_div', False)
    # Processing the call keyword arguments (line 621)
    
    # Call to ones_like(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'input' (line 622)
    input_121141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 44), 'input', False)
    # Obtaining the member 'shape' of a type (line 622)
    shape_121142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 44), input_121141, 'shape')
    # Processing the call keyword arguments (line 622)
    # Getting the type of 'numpy' (line 622)
    numpy_121143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 63), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 622)
    float64_121144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 63), numpy_121143, 'float64')
    keyword_121145 = float64_121144
    kwargs_121146 = {'dtype': keyword_121145}
    # Getting the type of 'numpy' (line 622)
    numpy_121139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 28), 'numpy', False)
    # Obtaining the member 'ones_like' of a type (line 622)
    ones_like_121140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 28), numpy_121139, 'ones_like')
    # Calling ones_like(args, kwargs) (line 622)
    ones_like_call_result_121147 = invoke(stypy.reporting.localization.Localization(__file__, 622, 28), ones_like_121140, *[shape_121142], **kwargs_121146)
    
    keyword_121148 = ones_like_call_result_121147
    
    # Getting the type of 'zoom_div' (line 623)
    zoom_div_121149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 30), 'zoom_div', False)
    int_121150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 42), 'int')
    # Applying the binary operator '!=' (line 623)
    result_ne_121151 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 30), '!=', zoom_div_121149, int_121150)
    
    keyword_121152 = result_ne_121151
    kwargs_121153 = {'where': keyword_121152, 'out': keyword_121148}
    # Getting the type of 'numpy' (line 621)
    numpy_121128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'numpy', False)
    # Obtaining the member 'divide' of a type (line 621)
    divide_121129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 11), numpy_121128, 'divide')
    # Calling divide(args, kwargs) (line 621)
    divide_call_result_121154 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), divide_121129, *[result_sub_121137, zoom_div_121138], **kwargs_121153)
    
    # Assigning a type to the variable 'zoom' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'zoom', divide_call_result_121154)
    
    # Assigning a Call to a Tuple (line 625):
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_121155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'int')
    
    # Call to _get_output(...): (line 625)
    # Processing the call arguments (line 625)
    # Getting the type of 'output' (line 625)
    output_121158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 51), 'output', False)
    # Getting the type of 'input' (line 625)
    input_121159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 59), 'input', False)
    # Processing the call keyword arguments (line 625)
    # Getting the type of 'output_shape' (line 626)
    output_shape_121160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 57), 'output_shape', False)
    keyword_121161 = output_shape_121160
    kwargs_121162 = {'shape': keyword_121161}
    # Getting the type of '_ni_support' (line 625)
    _ni_support_121156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 625)
    _get_output_121157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 27), _ni_support_121156, '_get_output')
    # Calling _get_output(args, kwargs) (line 625)
    _get_output_call_result_121163 = invoke(stypy.reporting.localization.Localization(__file__, 625, 27), _get_output_121157, *[output_121158, input_121159], **kwargs_121162)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___121164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 4), _get_output_call_result_121163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_121165 = invoke(stypy.reporting.localization.Localization(__file__, 625, 4), getitem___121164, int_121155)
    
    # Assigning a type to the variable 'tuple_var_assignment_120083' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'tuple_var_assignment_120083', subscript_call_result_121165)
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    int_121166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 4), 'int')
    
    # Call to _get_output(...): (line 625)
    # Processing the call arguments (line 625)
    # Getting the type of 'output' (line 625)
    output_121169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 51), 'output', False)
    # Getting the type of 'input' (line 625)
    input_121170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 59), 'input', False)
    # Processing the call keyword arguments (line 625)
    # Getting the type of 'output_shape' (line 626)
    output_shape_121171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 57), 'output_shape', False)
    keyword_121172 = output_shape_121171
    kwargs_121173 = {'shape': keyword_121172}
    # Getting the type of '_ni_support' (line 625)
    _ni_support_121167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 625)
    _get_output_121168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 27), _ni_support_121167, '_get_output')
    # Calling _get_output(args, kwargs) (line 625)
    _get_output_call_result_121174 = invoke(stypy.reporting.localization.Localization(__file__, 625, 27), _get_output_121168, *[output_121169, input_121170], **kwargs_121173)
    
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___121175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 4), _get_output_call_result_121174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_121176 = invoke(stypy.reporting.localization.Localization(__file__, 625, 4), getitem___121175, int_121166)
    
    # Assigning a type to the variable 'tuple_var_assignment_120084' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'tuple_var_assignment_120084', subscript_call_result_121176)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_120083' (line 625)
    tuple_var_assignment_120083_121177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'tuple_var_assignment_120083')
    # Assigning a type to the variable 'output' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'output', tuple_var_assignment_120083_121177)
    
    # Assigning a Name to a Name (line 625):
    # Getting the type of 'tuple_var_assignment_120084' (line 625)
    tuple_var_assignment_120084_121178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'tuple_var_assignment_120084')
    # Assigning a type to the variable 'return_value' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'return_value', tuple_var_assignment_120084_121178)
    
    # Assigning a Call to a Name (line 627):
    
    # Assigning a Call to a Name (line 627):
    
    # Call to ascontiguousarray(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'zoom' (line 627)
    zoom_121181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 35), 'zoom', False)
    # Processing the call keyword arguments (line 627)
    kwargs_121182 = {}
    # Getting the type of 'numpy' (line 627)
    numpy_121179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 'numpy', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 627)
    ascontiguousarray_121180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 11), numpy_121179, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 627)
    ascontiguousarray_call_result_121183 = invoke(stypy.reporting.localization.Localization(__file__, 627, 11), ascontiguousarray_121180, *[zoom_121181], **kwargs_121182)
    
    # Assigning a type to the variable 'zoom' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'zoom', ascontiguousarray_call_result_121183)
    
    # Call to zoom_shift(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'filtered' (line 628)
    filtered_121186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 25), 'filtered', False)
    # Getting the type of 'zoom' (line 628)
    zoom_121187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 35), 'zoom', False)
    # Getting the type of 'None' (line 628)
    None_121188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 41), 'None', False)
    # Getting the type of 'output' (line 628)
    output_121189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 47), 'output', False)
    # Getting the type of 'order' (line 628)
    order_121190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 55), 'order', False)
    # Getting the type of 'mode' (line 628)
    mode_121191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 62), 'mode', False)
    # Getting the type of 'cval' (line 628)
    cval_121192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 68), 'cval', False)
    # Processing the call keyword arguments (line 628)
    kwargs_121193 = {}
    # Getting the type of '_nd_image' (line 628)
    _nd_image_121184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), '_nd_image', False)
    # Obtaining the member 'zoom_shift' of a type (line 628)
    zoom_shift_121185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 4), _nd_image_121184, 'zoom_shift')
    # Calling zoom_shift(args, kwargs) (line 628)
    zoom_shift_call_result_121194 = invoke(stypy.reporting.localization.Localization(__file__, 628, 4), zoom_shift_121185, *[filtered_121186, zoom_121187, None_121188, output_121189, order_121190, mode_121191, cval_121192], **kwargs_121193)
    
    # Getting the type of 'return_value' (line 629)
    return_value_121195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'stypy_return_type', return_value_121195)
    
    # ################# End of 'zoom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zoom' in the type store
    # Getting the type of 'stypy_return_type' (line 554)
    stypy_return_type_121196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zoom'
    return stypy_return_type_121196

# Assigning a type to the variable 'zoom' (line 554)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 0), 'zoom', zoom)

@norecursion
def _minmax(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_minmax'
    module_type_store = module_type_store.open_function_context('_minmax', 632, 0, False)
    
    # Passed parameters checking function
    _minmax.stypy_localization = localization
    _minmax.stypy_type_of_self = None
    _minmax.stypy_type_store = module_type_store
    _minmax.stypy_function_name = '_minmax'
    _minmax.stypy_param_names_list = ['coor', 'minc', 'maxc']
    _minmax.stypy_varargs_param_name = None
    _minmax.stypy_kwargs_param_name = None
    _minmax.stypy_call_defaults = defaults
    _minmax.stypy_call_varargs = varargs
    _minmax.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_minmax', ['coor', 'minc', 'maxc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_minmax', localization, ['coor', 'minc', 'maxc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_minmax(...)' code ##################

    
    
    
    # Obtaining the type of the subscript
    int_121197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 12), 'int')
    # Getting the type of 'coor' (line 633)
    coor_121198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 7), 'coor')
    # Obtaining the member '__getitem__' of a type (line 633)
    getitem___121199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 7), coor_121198, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 633)
    subscript_call_result_121200 = invoke(stypy.reporting.localization.Localization(__file__, 633, 7), getitem___121199, int_121197)
    
    
    # Obtaining the type of the subscript
    int_121201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 22), 'int')
    # Getting the type of 'minc' (line 633)
    minc_121202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 17), 'minc')
    # Obtaining the member '__getitem__' of a type (line 633)
    getitem___121203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 17), minc_121202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 633)
    subscript_call_result_121204 = invoke(stypy.reporting.localization.Localization(__file__, 633, 17), getitem___121203, int_121201)
    
    # Applying the binary operator '<' (line 633)
    result_lt_121205 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 7), '<', subscript_call_result_121200, subscript_call_result_121204)
    
    # Testing the type of an if condition (line 633)
    if_condition_121206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 633, 4), result_lt_121205)
    # Assigning a type to the variable 'if_condition_121206' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'if_condition_121206', if_condition_121206)
    # SSA begins for if statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 634):
    
    # Assigning a Subscript to a Subscript (line 634):
    
    # Obtaining the type of the subscript
    int_121207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 23), 'int')
    # Getting the type of 'coor' (line 634)
    coor_121208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 18), 'coor')
    # Obtaining the member '__getitem__' of a type (line 634)
    getitem___121209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 18), coor_121208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 634)
    subscript_call_result_121210 = invoke(stypy.reporting.localization.Localization(__file__, 634, 18), getitem___121209, int_121207)
    
    # Getting the type of 'minc' (line 634)
    minc_121211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'minc')
    int_121212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 13), 'int')
    # Storing an element on a container (line 634)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), minc_121211, (int_121212, subscript_call_result_121210))
    # SSA join for if statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 12), 'int')
    # Getting the type of 'coor' (line 635)
    coor_121214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 7), 'coor')
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___121215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 7), coor_121214, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_121216 = invoke(stypy.reporting.localization.Localization(__file__, 635, 7), getitem___121215, int_121213)
    
    
    # Obtaining the type of the subscript
    int_121217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 22), 'int')
    # Getting the type of 'maxc' (line 635)
    maxc_121218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 17), 'maxc')
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___121219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 17), maxc_121218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_121220 = invoke(stypy.reporting.localization.Localization(__file__, 635, 17), getitem___121219, int_121217)
    
    # Applying the binary operator '>' (line 635)
    result_gt_121221 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 7), '>', subscript_call_result_121216, subscript_call_result_121220)
    
    # Testing the type of an if condition (line 635)
    if_condition_121222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 4), result_gt_121221)
    # Assigning a type to the variable 'if_condition_121222' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'if_condition_121222', if_condition_121222)
    # SSA begins for if statement (line 635)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 636):
    
    # Assigning a Subscript to a Subscript (line 636):
    
    # Obtaining the type of the subscript
    int_121223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 23), 'int')
    # Getting the type of 'coor' (line 636)
    coor_121224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 18), 'coor')
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___121225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 18), coor_121224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_121226 = invoke(stypy.reporting.localization.Localization(__file__, 636, 18), getitem___121225, int_121223)
    
    # Getting the type of 'maxc' (line 636)
    maxc_121227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'maxc')
    int_121228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 13), 'int')
    # Storing an element on a container (line 636)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 8), maxc_121227, (int_121228, subscript_call_result_121226))
    # SSA join for if statement (line 635)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 12), 'int')
    # Getting the type of 'coor' (line 637)
    coor_121230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 7), 'coor')
    # Obtaining the member '__getitem__' of a type (line 637)
    getitem___121231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 7), coor_121230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 637)
    subscript_call_result_121232 = invoke(stypy.reporting.localization.Localization(__file__, 637, 7), getitem___121231, int_121229)
    
    
    # Obtaining the type of the subscript
    int_121233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 22), 'int')
    # Getting the type of 'minc' (line 637)
    minc_121234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 17), 'minc')
    # Obtaining the member '__getitem__' of a type (line 637)
    getitem___121235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 17), minc_121234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 637)
    subscript_call_result_121236 = invoke(stypy.reporting.localization.Localization(__file__, 637, 17), getitem___121235, int_121233)
    
    # Applying the binary operator '<' (line 637)
    result_lt_121237 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 7), '<', subscript_call_result_121232, subscript_call_result_121236)
    
    # Testing the type of an if condition (line 637)
    if_condition_121238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 637, 4), result_lt_121237)
    # Assigning a type to the variable 'if_condition_121238' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'if_condition_121238', if_condition_121238)
    # SSA begins for if statement (line 637)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 638):
    
    # Assigning a Subscript to a Subscript (line 638):
    
    # Obtaining the type of the subscript
    int_121239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 23), 'int')
    # Getting the type of 'coor' (line 638)
    coor_121240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 18), 'coor')
    # Obtaining the member '__getitem__' of a type (line 638)
    getitem___121241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 18), coor_121240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 638)
    subscript_call_result_121242 = invoke(stypy.reporting.localization.Localization(__file__, 638, 18), getitem___121241, int_121239)
    
    # Getting the type of 'minc' (line 638)
    minc_121243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 8), 'minc')
    int_121244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 13), 'int')
    # Storing an element on a container (line 638)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 8), minc_121243, (int_121244, subscript_call_result_121242))
    # SSA join for if statement (line 637)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 12), 'int')
    # Getting the type of 'coor' (line 639)
    coor_121246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 7), 'coor')
    # Obtaining the member '__getitem__' of a type (line 639)
    getitem___121247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 7), coor_121246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 639)
    subscript_call_result_121248 = invoke(stypy.reporting.localization.Localization(__file__, 639, 7), getitem___121247, int_121245)
    
    
    # Obtaining the type of the subscript
    int_121249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 22), 'int')
    # Getting the type of 'maxc' (line 639)
    maxc_121250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 17), 'maxc')
    # Obtaining the member '__getitem__' of a type (line 639)
    getitem___121251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 639, 17), maxc_121250, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 639)
    subscript_call_result_121252 = invoke(stypy.reporting.localization.Localization(__file__, 639, 17), getitem___121251, int_121249)
    
    # Applying the binary operator '>' (line 639)
    result_gt_121253 = python_operator(stypy.reporting.localization.Localization(__file__, 639, 7), '>', subscript_call_result_121248, subscript_call_result_121252)
    
    # Testing the type of an if condition (line 639)
    if_condition_121254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 639, 4), result_gt_121253)
    # Assigning a type to the variable 'if_condition_121254' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'if_condition_121254', if_condition_121254)
    # SSA begins for if statement (line 639)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 640):
    
    # Assigning a Subscript to a Subscript (line 640):
    
    # Obtaining the type of the subscript
    int_121255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 23), 'int')
    # Getting the type of 'coor' (line 640)
    coor_121256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 18), 'coor')
    # Obtaining the member '__getitem__' of a type (line 640)
    getitem___121257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 18), coor_121256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 640)
    subscript_call_result_121258 = invoke(stypy.reporting.localization.Localization(__file__, 640, 18), getitem___121257, int_121255)
    
    # Getting the type of 'maxc' (line 640)
    maxc_121259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'maxc')
    int_121260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 13), 'int')
    # Storing an element on a container (line 640)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 640, 8), maxc_121259, (int_121260, subscript_call_result_121258))
    # SSA join for if statement (line 639)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 641)
    tuple_121261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 641)
    # Adding element type (line 641)
    # Getting the type of 'minc' (line 641)
    minc_121262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 11), 'minc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 11), tuple_121261, minc_121262)
    # Adding element type (line 641)
    # Getting the type of 'maxc' (line 641)
    maxc_121263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 17), 'maxc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 11), tuple_121261, maxc_121263)
    
    # Assigning a type to the variable 'stypy_return_type' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'stypy_return_type', tuple_121261)
    
    # ################# End of '_minmax(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_minmax' in the type store
    # Getting the type of 'stypy_return_type' (line 632)
    stypy_return_type_121264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121264)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_minmax'
    return stypy_return_type_121264

# Assigning a type to the variable '_minmax' (line 632)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 0), '_minmax', _minmax)

@norecursion
def rotate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 644)
    tuple_121265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 644)
    # Adding element type (line 644)
    int_121266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 31), tuple_121265, int_121266)
    # Adding element type (line 644)
    int_121267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 644, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 644, 31), tuple_121265, int_121267)
    
    # Getting the type of 'True' (line 644)
    True_121268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 46), 'True')
    # Getting the type of 'None' (line 645)
    None_121269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 18), 'None')
    int_121270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 645, 30), 'int')
    str_121271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 16), 'str', 'constant')
    float_121272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 33), 'float')
    # Getting the type of 'True' (line 646)
    True_121273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 48), 'True')
    defaults = [tuple_121265, True_121268, None_121269, int_121270, str_121271, float_121272, True_121273]
    # Create a new context for function 'rotate'
    module_type_store = module_type_store.open_function_context('rotate', 644, 0, False)
    
    # Passed parameters checking function
    rotate.stypy_localization = localization
    rotate.stypy_type_of_self = None
    rotate.stypy_type_store = module_type_store
    rotate.stypy_function_name = 'rotate'
    rotate.stypy_param_names_list = ['input', 'angle', 'axes', 'reshape', 'output', 'order', 'mode', 'cval', 'prefilter']
    rotate.stypy_varargs_param_name = None
    rotate.stypy_kwargs_param_name = None
    rotate.stypy_call_defaults = defaults
    rotate.stypy_call_varargs = varargs
    rotate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rotate', ['input', 'angle', 'axes', 'reshape', 'output', 'order', 'mode', 'cval', 'prefilter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rotate', localization, ['input', 'angle', 'axes', 'reshape', 'output', 'order', 'mode', 'cval', 'prefilter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rotate(...)' code ##################

    str_121274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, (-1)), 'str', "\n    Rotate an array.\n\n    The array is rotated in the plane defined by the two axes given by the\n    `axes` parameter using spline interpolation of the requested order.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input array.\n    angle : float\n        The rotation angle in degrees.\n    axes : tuple of 2 ints, optional\n        The two axes that define the plane of rotation. Default is the first\n        two axes.\n    reshape : bool, optional\n        If `reshape` is true, the output shape is adapted so that the input\n        array is contained completely in the output. Default is True.\n    output : ndarray or dtype, optional\n        The array in which to place the output, or the dtype of the returned\n        array.\n    order : int, optional\n        The order of the spline interpolation, default is 3.\n        The order has to be in the range 0-5.\n    mode : str, optional\n        Points outside the boundaries of the input are filled according\n        to the given mode ('constant', 'nearest', 'reflect', 'mirror' or 'wrap').\n        Default is 'constant'.\n    cval : scalar, optional\n        Value used for points outside the boundaries of the input if\n        ``mode='constant'``. Default is 0.0\n    prefilter : bool, optional\n        The parameter prefilter determines if the input is pre-filtered with\n        `spline_filter` before interpolation (necessary for spline\n        interpolation of order > 1).  If False, it is assumed that the input is\n        already filtered. Default is True.\n\n    Returns\n    -------\n    rotate : ndarray or None\n        The rotated input. If `output` is given as a parameter, None is\n        returned.\n\n    ")
    
    # Assigning a Call to a Name (line 691):
    
    # Assigning a Call to a Name (line 691):
    
    # Call to asarray(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'input' (line 691)
    input_121277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 26), 'input', False)
    # Processing the call keyword arguments (line 691)
    kwargs_121278 = {}
    # Getting the type of 'numpy' (line 691)
    numpy_121275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 691)
    asarray_121276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 12), numpy_121275, 'asarray')
    # Calling asarray(args, kwargs) (line 691)
    asarray_call_result_121279 = invoke(stypy.reporting.localization.Localization(__file__, 691, 12), asarray_121276, *[input_121277], **kwargs_121278)
    
    # Assigning a type to the variable 'input' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'input', asarray_call_result_121279)
    
    # Assigning a Call to a Name (line 692):
    
    # Assigning a Call to a Name (line 692):
    
    # Call to list(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'axes' (line 692)
    axes_121281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'axes', False)
    # Processing the call keyword arguments (line 692)
    kwargs_121282 = {}
    # Getting the type of 'list' (line 692)
    list_121280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 11), 'list', False)
    # Calling list(args, kwargs) (line 692)
    list_call_result_121283 = invoke(stypy.reporting.localization.Localization(__file__, 692, 11), list_121280, *[axes_121281], **kwargs_121282)
    
    # Assigning a type to the variable 'axes' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'axes', list_call_result_121283)
    
    # Assigning a Attribute to a Name (line 693):
    
    # Assigning a Attribute to a Name (line 693):
    # Getting the type of 'input' (line 693)
    input_121284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 11), 'input')
    # Obtaining the member 'ndim' of a type (line 693)
    ndim_121285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 11), input_121284, 'ndim')
    # Assigning a type to the variable 'rank' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'rank', ndim_121285)
    
    
    
    # Obtaining the type of the subscript
    int_121286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 12), 'int')
    # Getting the type of 'axes' (line 694)
    axes_121287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), 'axes')
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___121288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 7), axes_121287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_121289 = invoke(stypy.reporting.localization.Localization(__file__, 694, 7), getitem___121288, int_121286)
    
    int_121290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 17), 'int')
    # Applying the binary operator '<' (line 694)
    result_lt_121291 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 7), '<', subscript_call_result_121289, int_121290)
    
    # Testing the type of an if condition (line 694)
    if_condition_121292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 4), result_lt_121291)
    # Assigning a type to the variable 'if_condition_121292' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'if_condition_121292', if_condition_121292)
    # SSA begins for if statement (line 694)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 695)
    axes_121293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'axes')
    
    # Obtaining the type of the subscript
    int_121294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 13), 'int')
    # Getting the type of 'axes' (line 695)
    axes_121295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'axes')
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___121296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 8), axes_121295, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_121297 = invoke(stypy.reporting.localization.Localization(__file__, 695, 8), getitem___121296, int_121294)
    
    # Getting the type of 'rank' (line 695)
    rank_121298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 19), 'rank')
    # Applying the binary operator '+=' (line 695)
    result_iadd_121299 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 8), '+=', subscript_call_result_121297, rank_121298)
    # Getting the type of 'axes' (line 695)
    axes_121300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'axes')
    int_121301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 13), 'int')
    # Storing an element on a container (line 695)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 8), axes_121300, (int_121301, result_iadd_121299))
    
    # SSA join for if statement (line 694)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 12), 'int')
    # Getting the type of 'axes' (line 696)
    axes_121303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 7), 'axes')
    # Obtaining the member '__getitem__' of a type (line 696)
    getitem___121304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 7), axes_121303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 696)
    subscript_call_result_121305 = invoke(stypy.reporting.localization.Localization(__file__, 696, 7), getitem___121304, int_121302)
    
    int_121306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 17), 'int')
    # Applying the binary operator '<' (line 696)
    result_lt_121307 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 7), '<', subscript_call_result_121305, int_121306)
    
    # Testing the type of an if condition (line 696)
    if_condition_121308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 4), result_lt_121307)
    # Assigning a type to the variable 'if_condition_121308' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'if_condition_121308', if_condition_121308)
    # SSA begins for if statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 697)
    axes_121309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'axes')
    
    # Obtaining the type of the subscript
    int_121310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 13), 'int')
    # Getting the type of 'axes' (line 697)
    axes_121311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'axes')
    # Obtaining the member '__getitem__' of a type (line 697)
    getitem___121312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), axes_121311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 697)
    subscript_call_result_121313 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), getitem___121312, int_121310)
    
    # Getting the type of 'rank' (line 697)
    rank_121314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 19), 'rank')
    # Applying the binary operator '+=' (line 697)
    result_iadd_121315 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 8), '+=', subscript_call_result_121313, rank_121314)
    # Getting the type of 'axes' (line 697)
    axes_121316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'axes')
    int_121317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 13), 'int')
    # Storing an element on a container (line 697)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 8), axes_121316, (int_121317, result_iadd_121315))
    
    # SSA join for if statement (line 696)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_121318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 12), 'int')
    # Getting the type of 'axes' (line 698)
    axes_121319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 7), 'axes')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___121320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 7), axes_121319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_121321 = invoke(stypy.reporting.localization.Localization(__file__, 698, 7), getitem___121320, int_121318)
    
    int_121322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 17), 'int')
    # Applying the binary operator '<' (line 698)
    result_lt_121323 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), '<', subscript_call_result_121321, int_121322)
    
    
    
    # Obtaining the type of the subscript
    int_121324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 27), 'int')
    # Getting the type of 'axes' (line 698)
    axes_121325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 22), 'axes')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___121326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 22), axes_121325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_121327 = invoke(stypy.reporting.localization.Localization(__file__, 698, 22), getitem___121326, int_121324)
    
    int_121328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 32), 'int')
    # Applying the binary operator '<' (line 698)
    result_lt_121329 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 22), '<', subscript_call_result_121327, int_121328)
    
    # Applying the binary operator 'or' (line 698)
    result_or_keyword_121330 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), 'or', result_lt_121323, result_lt_121329)
    
    
    # Obtaining the type of the subscript
    int_121331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 42), 'int')
    # Getting the type of 'axes' (line 698)
    axes_121332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 37), 'axes')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___121333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 37), axes_121332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_121334 = invoke(stypy.reporting.localization.Localization(__file__, 698, 37), getitem___121333, int_121331)
    
    # Getting the type of 'rank' (line 698)
    rank_121335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'rank')
    # Applying the binary operator '>' (line 698)
    result_gt_121336 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 37), '>', subscript_call_result_121334, rank_121335)
    
    # Applying the binary operator 'or' (line 698)
    result_or_keyword_121337 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), 'or', result_or_keyword_121330, result_gt_121336)
    
    
    # Obtaining the type of the subscript
    int_121338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 60), 'int')
    # Getting the type of 'axes' (line 698)
    axes_121339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 55), 'axes')
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___121340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 55), axes_121339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_121341 = invoke(stypy.reporting.localization.Localization(__file__, 698, 55), getitem___121340, int_121338)
    
    # Getting the type of 'rank' (line 698)
    rank_121342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 65), 'rank')
    # Applying the binary operator '>' (line 698)
    result_gt_121343 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 55), '>', subscript_call_result_121341, rank_121342)
    
    # Applying the binary operator 'or' (line 698)
    result_or_keyword_121344 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), 'or', result_or_keyword_121337, result_gt_121343)
    
    # Testing the type of an if condition (line 698)
    if_condition_121345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 4), result_or_keyword_121344)
    # Assigning a type to the variable 'if_condition_121345' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'if_condition_121345', if_condition_121345)
    # SSA begins for if statement (line 698)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 699)
    # Processing the call arguments (line 699)
    str_121347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 27), 'str', 'invalid rotation plane specified')
    # Processing the call keyword arguments (line 699)
    kwargs_121348 = {}
    # Getting the type of 'RuntimeError' (line 699)
    RuntimeError_121346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 699)
    RuntimeError_call_result_121349 = invoke(stypy.reporting.localization.Localization(__file__, 699, 14), RuntimeError_121346, *[str_121347], **kwargs_121348)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 699, 8), RuntimeError_call_result_121349, 'raise parameter', BaseException)
    # SSA join for if statement (line 698)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_121350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 12), 'int')
    # Getting the type of 'axes' (line 700)
    axes_121351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 7), 'axes')
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___121352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 7), axes_121351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_121353 = invoke(stypy.reporting.localization.Localization(__file__, 700, 7), getitem___121352, int_121350)
    
    
    # Obtaining the type of the subscript
    int_121354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 22), 'int')
    # Getting the type of 'axes' (line 700)
    axes_121355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 17), 'axes')
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___121356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 17), axes_121355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_121357 = invoke(stypy.reporting.localization.Localization(__file__, 700, 17), getitem___121356, int_121354)
    
    # Applying the binary operator '>' (line 700)
    result_gt_121358 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 7), '>', subscript_call_result_121353, subscript_call_result_121357)
    
    # Testing the type of an if condition (line 700)
    if_condition_121359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 4), result_gt_121358)
    # Assigning a type to the variable 'if_condition_121359' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'if_condition_121359', if_condition_121359)
    # SSA begins for if statement (line 700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 701):
    
    # Assigning a Tuple to a Name (line 701):
    
    # Obtaining an instance of the builtin type 'tuple' (line 701)
    tuple_121360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 701)
    # Adding element type (line 701)
    
    # Obtaining the type of the subscript
    int_121361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 20), 'int')
    # Getting the type of 'axes' (line 701)
    axes_121362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 15), 'axes')
    # Obtaining the member '__getitem__' of a type (line 701)
    getitem___121363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 15), axes_121362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 701)
    subscript_call_result_121364 = invoke(stypy.reporting.localization.Localization(__file__, 701, 15), getitem___121363, int_121361)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 15), tuple_121360, subscript_call_result_121364)
    # Adding element type (line 701)
    
    # Obtaining the type of the subscript
    int_121365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 29), 'int')
    # Getting the type of 'axes' (line 701)
    axes_121366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 24), 'axes')
    # Obtaining the member '__getitem__' of a type (line 701)
    getitem___121367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 701, 24), axes_121366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 701)
    subscript_call_result_121368 = invoke(stypy.reporting.localization.Localization(__file__, 701, 24), getitem___121367, int_121365)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 15), tuple_121360, subscript_call_result_121368)
    
    # Assigning a type to the variable 'axes' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 8), 'axes', tuple_121360)
    # SSA join for if statement (line 700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 702):
    
    # Assigning a BinOp to a Name (line 702):
    # Getting the type of 'numpy' (line 702)
    numpy_121369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'numpy')
    # Obtaining the member 'pi' of a type (line 702)
    pi_121370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 12), numpy_121369, 'pi')
    int_121371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 23), 'int')
    # Applying the binary operator 'div' (line 702)
    result_div_121372 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 12), 'div', pi_121370, int_121371)
    
    # Getting the type of 'angle' (line 702)
    angle_121373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 29), 'angle')
    # Applying the binary operator '*' (line 702)
    result_mul_121374 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 27), '*', result_div_121372, angle_121373)
    
    # Assigning a type to the variable 'angle' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'angle', result_mul_121374)
    
    # Assigning a Call to a Name (line 703):
    
    # Assigning a Call to a Name (line 703):
    
    # Call to cos(...): (line 703)
    # Processing the call arguments (line 703)
    # Getting the type of 'angle' (line 703)
    angle_121377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 19), 'angle', False)
    # Processing the call keyword arguments (line 703)
    kwargs_121378 = {}
    # Getting the type of 'math' (line 703)
    math_121375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 703)
    cos_121376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 10), math_121375, 'cos')
    # Calling cos(args, kwargs) (line 703)
    cos_call_result_121379 = invoke(stypy.reporting.localization.Localization(__file__, 703, 10), cos_121376, *[angle_121377], **kwargs_121378)
    
    # Assigning a type to the variable 'm11' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'm11', cos_call_result_121379)
    
    # Assigning a Call to a Name (line 704):
    
    # Assigning a Call to a Name (line 704):
    
    # Call to sin(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'angle' (line 704)
    angle_121382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 19), 'angle', False)
    # Processing the call keyword arguments (line 704)
    kwargs_121383 = {}
    # Getting the type of 'math' (line 704)
    math_121380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 10), 'math', False)
    # Obtaining the member 'sin' of a type (line 704)
    sin_121381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 10), math_121380, 'sin')
    # Calling sin(args, kwargs) (line 704)
    sin_call_result_121384 = invoke(stypy.reporting.localization.Localization(__file__, 704, 10), sin_121381, *[angle_121382], **kwargs_121383)
    
    # Assigning a type to the variable 'm12' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'm12', sin_call_result_121384)
    
    # Assigning a UnaryOp to a Name (line 705):
    
    # Assigning a UnaryOp to a Name (line 705):
    
    
    # Call to sin(...): (line 705)
    # Processing the call arguments (line 705)
    # Getting the type of 'angle' (line 705)
    angle_121387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 20), 'angle', False)
    # Processing the call keyword arguments (line 705)
    kwargs_121388 = {}
    # Getting the type of 'math' (line 705)
    math_121385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 11), 'math', False)
    # Obtaining the member 'sin' of a type (line 705)
    sin_121386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 11), math_121385, 'sin')
    # Calling sin(args, kwargs) (line 705)
    sin_call_result_121389 = invoke(stypy.reporting.localization.Localization(__file__, 705, 11), sin_121386, *[angle_121387], **kwargs_121388)
    
    # Applying the 'usub' unary operator (line 705)
    result___neg___121390 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 10), 'usub', sin_call_result_121389)
    
    # Assigning a type to the variable 'm21' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'm21', result___neg___121390)
    
    # Assigning a Call to a Name (line 706):
    
    # Assigning a Call to a Name (line 706):
    
    # Call to cos(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'angle' (line 706)
    angle_121393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 19), 'angle', False)
    # Processing the call keyword arguments (line 706)
    kwargs_121394 = {}
    # Getting the type of 'math' (line 706)
    math_121391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 706)
    cos_121392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 10), math_121391, 'cos')
    # Calling cos(args, kwargs) (line 706)
    cos_call_result_121395 = invoke(stypy.reporting.localization.Localization(__file__, 706, 10), cos_121392, *[angle_121393], **kwargs_121394)
    
    # Assigning a type to the variable 'm22' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'm22', cos_call_result_121395)
    
    # Assigning a Call to a Name (line 707):
    
    # Assigning a Call to a Name (line 707):
    
    # Call to array(...): (line 707)
    # Processing the call arguments (line 707)
    
    # Obtaining an instance of the builtin type 'list' (line 707)
    list_121398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 707)
    # Adding element type (line 707)
    
    # Obtaining an instance of the builtin type 'list' (line 707)
    list_121399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 707)
    # Adding element type (line 707)
    # Getting the type of 'm11' (line 707)
    m11_121400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 27), 'm11', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 26), list_121399, m11_121400)
    # Adding element type (line 707)
    # Getting the type of 'm12' (line 707)
    m12_121401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 32), 'm12', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 26), list_121399, m12_121401)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 25), list_121398, list_121399)
    # Adding element type (line 707)
    
    # Obtaining an instance of the builtin type 'list' (line 708)
    list_121402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 708)
    # Adding element type (line 708)
    # Getting the type of 'm21' (line 708)
    m21_121403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 30), 'm21', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 29), list_121402, m21_121403)
    # Adding element type (line 708)
    # Getting the type of 'm22' (line 708)
    m22_121404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 35), 'm22', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 29), list_121402, m22_121404)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 707, 25), list_121398, list_121402)
    
    # Processing the call keyword arguments (line 707)
    # Getting the type of 'numpy' (line 708)
    numpy_121405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 48), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 708)
    float64_121406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 48), numpy_121405, 'float64')
    keyword_121407 = float64_121406
    kwargs_121408 = {'dtype': keyword_121407}
    # Getting the type of 'numpy' (line 707)
    numpy_121396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 13), 'numpy', False)
    # Obtaining the member 'array' of a type (line 707)
    array_121397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 13), numpy_121396, 'array')
    # Calling array(args, kwargs) (line 707)
    array_call_result_121409 = invoke(stypy.reporting.localization.Localization(__file__, 707, 13), array_121397, *[list_121398], **kwargs_121408)
    
    # Assigning a type to the variable 'matrix' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'matrix', array_call_result_121409)
    
    # Assigning a Subscript to a Name (line 709):
    
    # Assigning a Subscript to a Name (line 709):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 26), 'int')
    # Getting the type of 'axes' (line 709)
    axes_121411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 21), 'axes')
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___121412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 21), axes_121411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_121413 = invoke(stypy.reporting.localization.Localization(__file__, 709, 21), getitem___121412, int_121410)
    
    # Getting the type of 'input' (line 709)
    input_121414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 9), 'input')
    # Obtaining the member 'shape' of a type (line 709)
    shape_121415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 9), input_121414, 'shape')
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___121416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 9), shape_121415, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_121417 = invoke(stypy.reporting.localization.Localization(__file__, 709, 9), getitem___121416, subscript_call_result_121413)
    
    # Assigning a type to the variable 'iy' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'iy', subscript_call_result_121417)
    
    # Assigning a Subscript to a Name (line 710):
    
    # Assigning a Subscript to a Name (line 710):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 26), 'int')
    # Getting the type of 'axes' (line 710)
    axes_121419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 21), 'axes')
    # Obtaining the member '__getitem__' of a type (line 710)
    getitem___121420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 21), axes_121419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 710)
    subscript_call_result_121421 = invoke(stypy.reporting.localization.Localization(__file__, 710, 21), getitem___121420, int_121418)
    
    # Getting the type of 'input' (line 710)
    input_121422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 9), 'input')
    # Obtaining the member 'shape' of a type (line 710)
    shape_121423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 9), input_121422, 'shape')
    # Obtaining the member '__getitem__' of a type (line 710)
    getitem___121424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 9), shape_121423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 710)
    subscript_call_result_121425 = invoke(stypy.reporting.localization.Localization(__file__, 710, 9), getitem___121424, subscript_call_result_121421)
    
    # Assigning a type to the variable 'ix' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'ix', subscript_call_result_121425)
    
    # Getting the type of 'reshape' (line 711)
    reshape_121426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 7), 'reshape')
    # Testing the type of an if condition (line 711)
    if_condition_121427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 4), reshape_121426)
    # Assigning a type to the variable 'if_condition_121427' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'if_condition_121427', if_condition_121427)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 712):
    
    # Assigning a Call to a Name (line 712):
    
    # Call to array(...): (line 712)
    # Processing the call arguments (line 712)
    
    # Obtaining an instance of the builtin type 'list' (line 712)
    list_121430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 712)
    # Adding element type (line 712)
    
    # Obtaining an instance of the builtin type 'list' (line 712)
    list_121431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 712)
    # Adding element type (line 712)
    # Getting the type of 'm11' (line 712)
    m11_121432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 29), 'm11', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 28), list_121431, m11_121432)
    # Adding element type (line 712)
    
    # Getting the type of 'm21' (line 712)
    m21_121433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 35), 'm21', False)
    # Applying the 'usub' unary operator (line 712)
    result___neg___121434 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 34), 'usub', m21_121433)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 28), list_121431, result___neg___121434)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 27), list_121430, list_121431)
    # Adding element type (line 712)
    
    # Obtaining an instance of the builtin type 'list' (line 713)
    list_121435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 713)
    # Adding element type (line 713)
    
    # Getting the type of 'm12' (line 713)
    m12_121436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 33), 'm12', False)
    # Applying the 'usub' unary operator (line 713)
    result___neg___121437 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 32), 'usub', m12_121436)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 31), list_121435, result___neg___121437)
    # Adding element type (line 713)
    # Getting the type of 'm22' (line 713)
    m22_121438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 38), 'm22', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 31), list_121435, m22_121438)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 712, 27), list_121430, list_121435)
    
    # Processing the call keyword arguments (line 712)
    # Getting the type of 'numpy' (line 713)
    numpy_121439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 51), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 713)
    float64_121440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 51), numpy_121439, 'float64')
    keyword_121441 = float64_121440
    kwargs_121442 = {'dtype': keyword_121441}
    # Getting the type of 'numpy' (line 712)
    numpy_121428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 15), 'numpy', False)
    # Obtaining the member 'array' of a type (line 712)
    array_121429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 15), numpy_121428, 'array')
    # Calling array(args, kwargs) (line 712)
    array_call_result_121443 = invoke(stypy.reporting.localization.Localization(__file__, 712, 15), array_121429, *[list_121430], **kwargs_121442)
    
    # Assigning a type to the variable 'mtrx' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'mtrx', array_call_result_121443)
    
    # Assigning a List to a Name (line 714):
    
    # Assigning a List to a Name (line 714):
    
    # Obtaining an instance of the builtin type 'list' (line 714)
    list_121444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 714)
    # Adding element type (line 714)
    int_121445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 15), list_121444, int_121445)
    # Adding element type (line 714)
    int_121446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 15), list_121444, int_121446)
    
    # Assigning a type to the variable 'minc' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'minc', list_121444)
    
    # Assigning a List to a Name (line 715):
    
    # Assigning a List to a Name (line 715):
    
    # Obtaining an instance of the builtin type 'list' (line 715)
    list_121447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 715)
    # Adding element type (line 715)
    int_121448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), list_121447, int_121448)
    # Adding element type (line 715)
    int_121449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 15), list_121447, int_121449)
    
    # Assigning a type to the variable 'maxc' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'maxc', list_121447)
    
    # Assigning a Call to a Name (line 716):
    
    # Assigning a Call to a Name (line 716):
    
    # Call to dot(...): (line 716)
    # Processing the call arguments (line 716)
    # Getting the type of 'mtrx' (line 716)
    mtrx_121452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 25), 'mtrx', False)
    
    # Obtaining an instance of the builtin type 'list' (line 716)
    list_121453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 716)
    # Adding element type (line 716)
    int_121454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 31), list_121453, int_121454)
    # Adding element type (line 716)
    # Getting the type of 'ix' (line 716)
    ix_121455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 35), 'ix', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 31), list_121453, ix_121455)
    
    # Processing the call keyword arguments (line 716)
    kwargs_121456 = {}
    # Getting the type of 'numpy' (line 716)
    numpy_121450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 15), 'numpy', False)
    # Obtaining the member 'dot' of a type (line 716)
    dot_121451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 15), numpy_121450, 'dot')
    # Calling dot(args, kwargs) (line 716)
    dot_call_result_121457 = invoke(stypy.reporting.localization.Localization(__file__, 716, 15), dot_121451, *[mtrx_121452, list_121453], **kwargs_121456)
    
    # Assigning a type to the variable 'coor' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'coor', dot_call_result_121457)
    
    # Assigning a Call to a Tuple (line 717):
    
    # Assigning a Subscript to a Name (line 717):
    
    # Obtaining the type of the subscript
    int_121458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 8), 'int')
    
    # Call to _minmax(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'coor' (line 717)
    coor_121460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 29), 'coor', False)
    # Getting the type of 'minc' (line 717)
    minc_121461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 35), 'minc', False)
    # Getting the type of 'maxc' (line 717)
    maxc_121462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 41), 'maxc', False)
    # Processing the call keyword arguments (line 717)
    kwargs_121463 = {}
    # Getting the type of '_minmax' (line 717)
    _minmax_121459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 717)
    _minmax_call_result_121464 = invoke(stypy.reporting.localization.Localization(__file__, 717, 21), _minmax_121459, *[coor_121460, minc_121461, maxc_121462], **kwargs_121463)
    
    # Obtaining the member '__getitem__' of a type (line 717)
    getitem___121465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), _minmax_call_result_121464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 717)
    subscript_call_result_121466 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), getitem___121465, int_121458)
    
    # Assigning a type to the variable 'tuple_var_assignment_120085' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'tuple_var_assignment_120085', subscript_call_result_121466)
    
    # Assigning a Subscript to a Name (line 717):
    
    # Obtaining the type of the subscript
    int_121467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 8), 'int')
    
    # Call to _minmax(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'coor' (line 717)
    coor_121469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 29), 'coor', False)
    # Getting the type of 'minc' (line 717)
    minc_121470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 35), 'minc', False)
    # Getting the type of 'maxc' (line 717)
    maxc_121471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 41), 'maxc', False)
    # Processing the call keyword arguments (line 717)
    kwargs_121472 = {}
    # Getting the type of '_minmax' (line 717)
    _minmax_121468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 717)
    _minmax_call_result_121473 = invoke(stypy.reporting.localization.Localization(__file__, 717, 21), _minmax_121468, *[coor_121469, minc_121470, maxc_121471], **kwargs_121472)
    
    # Obtaining the member '__getitem__' of a type (line 717)
    getitem___121474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 8), _minmax_call_result_121473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 717)
    subscript_call_result_121475 = invoke(stypy.reporting.localization.Localization(__file__, 717, 8), getitem___121474, int_121467)
    
    # Assigning a type to the variable 'tuple_var_assignment_120086' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'tuple_var_assignment_120086', subscript_call_result_121475)
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'tuple_var_assignment_120085' (line 717)
    tuple_var_assignment_120085_121476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'tuple_var_assignment_120085')
    # Assigning a type to the variable 'minc' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'minc', tuple_var_assignment_120085_121476)
    
    # Assigning a Name to a Name (line 717):
    # Getting the type of 'tuple_var_assignment_120086' (line 717)
    tuple_var_assignment_120086_121477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'tuple_var_assignment_120086')
    # Assigning a type to the variable 'maxc' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 14), 'maxc', tuple_var_assignment_120086_121477)
    
    # Assigning a Call to a Name (line 718):
    
    # Assigning a Call to a Name (line 718):
    
    # Call to dot(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'mtrx' (line 718)
    mtrx_121480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 25), 'mtrx', False)
    
    # Obtaining an instance of the builtin type 'list' (line 718)
    list_121481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 718)
    # Adding element type (line 718)
    # Getting the type of 'iy' (line 718)
    iy_121482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 32), 'iy', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 31), list_121481, iy_121482)
    # Adding element type (line 718)
    int_121483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 31), list_121481, int_121483)
    
    # Processing the call keyword arguments (line 718)
    kwargs_121484 = {}
    # Getting the type of 'numpy' (line 718)
    numpy_121478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 15), 'numpy', False)
    # Obtaining the member 'dot' of a type (line 718)
    dot_121479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 15), numpy_121478, 'dot')
    # Calling dot(args, kwargs) (line 718)
    dot_call_result_121485 = invoke(stypy.reporting.localization.Localization(__file__, 718, 15), dot_121479, *[mtrx_121480, list_121481], **kwargs_121484)
    
    # Assigning a type to the variable 'coor' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 8), 'coor', dot_call_result_121485)
    
    # Assigning a Call to a Tuple (line 719):
    
    # Assigning a Subscript to a Name (line 719):
    
    # Obtaining the type of the subscript
    int_121486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 8), 'int')
    
    # Call to _minmax(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'coor' (line 719)
    coor_121488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 29), 'coor', False)
    # Getting the type of 'minc' (line 719)
    minc_121489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'minc', False)
    # Getting the type of 'maxc' (line 719)
    maxc_121490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 41), 'maxc', False)
    # Processing the call keyword arguments (line 719)
    kwargs_121491 = {}
    # Getting the type of '_minmax' (line 719)
    _minmax_121487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 719)
    _minmax_call_result_121492 = invoke(stypy.reporting.localization.Localization(__file__, 719, 21), _minmax_121487, *[coor_121488, minc_121489, maxc_121490], **kwargs_121491)
    
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___121493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 8), _minmax_call_result_121492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_121494 = invoke(stypy.reporting.localization.Localization(__file__, 719, 8), getitem___121493, int_121486)
    
    # Assigning a type to the variable 'tuple_var_assignment_120087' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'tuple_var_assignment_120087', subscript_call_result_121494)
    
    # Assigning a Subscript to a Name (line 719):
    
    # Obtaining the type of the subscript
    int_121495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 8), 'int')
    
    # Call to _minmax(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'coor' (line 719)
    coor_121497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 29), 'coor', False)
    # Getting the type of 'minc' (line 719)
    minc_121498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 35), 'minc', False)
    # Getting the type of 'maxc' (line 719)
    maxc_121499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 41), 'maxc', False)
    # Processing the call keyword arguments (line 719)
    kwargs_121500 = {}
    # Getting the type of '_minmax' (line 719)
    _minmax_121496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 719)
    _minmax_call_result_121501 = invoke(stypy.reporting.localization.Localization(__file__, 719, 21), _minmax_121496, *[coor_121497, minc_121498, maxc_121499], **kwargs_121500)
    
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___121502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 8), _minmax_call_result_121501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_121503 = invoke(stypy.reporting.localization.Localization(__file__, 719, 8), getitem___121502, int_121495)
    
    # Assigning a type to the variable 'tuple_var_assignment_120088' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'tuple_var_assignment_120088', subscript_call_result_121503)
    
    # Assigning a Name to a Name (line 719):
    # Getting the type of 'tuple_var_assignment_120087' (line 719)
    tuple_var_assignment_120087_121504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'tuple_var_assignment_120087')
    # Assigning a type to the variable 'minc' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'minc', tuple_var_assignment_120087_121504)
    
    # Assigning a Name to a Name (line 719):
    # Getting the type of 'tuple_var_assignment_120088' (line 719)
    tuple_var_assignment_120088_121505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'tuple_var_assignment_120088')
    # Assigning a type to the variable 'maxc' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 14), 'maxc', tuple_var_assignment_120088_121505)
    
    # Assigning a Call to a Name (line 720):
    
    # Assigning a Call to a Name (line 720):
    
    # Call to dot(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'mtrx' (line 720)
    mtrx_121508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 25), 'mtrx', False)
    
    # Obtaining an instance of the builtin type 'list' (line 720)
    list_121509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 720)
    # Adding element type (line 720)
    # Getting the type of 'iy' (line 720)
    iy_121510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 32), 'iy', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 31), list_121509, iy_121510)
    # Adding element type (line 720)
    # Getting the type of 'ix' (line 720)
    ix_121511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 36), 'ix', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 31), list_121509, ix_121511)
    
    # Processing the call keyword arguments (line 720)
    kwargs_121512 = {}
    # Getting the type of 'numpy' (line 720)
    numpy_121506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 15), 'numpy', False)
    # Obtaining the member 'dot' of a type (line 720)
    dot_121507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 15), numpy_121506, 'dot')
    # Calling dot(args, kwargs) (line 720)
    dot_call_result_121513 = invoke(stypy.reporting.localization.Localization(__file__, 720, 15), dot_121507, *[mtrx_121508, list_121509], **kwargs_121512)
    
    # Assigning a type to the variable 'coor' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 8), 'coor', dot_call_result_121513)
    
    # Assigning a Call to a Tuple (line 721):
    
    # Assigning a Subscript to a Name (line 721):
    
    # Obtaining the type of the subscript
    int_121514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 8), 'int')
    
    # Call to _minmax(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 'coor' (line 721)
    coor_121516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 29), 'coor', False)
    # Getting the type of 'minc' (line 721)
    minc_121517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 35), 'minc', False)
    # Getting the type of 'maxc' (line 721)
    maxc_121518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 41), 'maxc', False)
    # Processing the call keyword arguments (line 721)
    kwargs_121519 = {}
    # Getting the type of '_minmax' (line 721)
    _minmax_121515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 721)
    _minmax_call_result_121520 = invoke(stypy.reporting.localization.Localization(__file__, 721, 21), _minmax_121515, *[coor_121516, minc_121517, maxc_121518], **kwargs_121519)
    
    # Obtaining the member '__getitem__' of a type (line 721)
    getitem___121521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), _minmax_call_result_121520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 721)
    subscript_call_result_121522 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), getitem___121521, int_121514)
    
    # Assigning a type to the variable 'tuple_var_assignment_120089' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'tuple_var_assignment_120089', subscript_call_result_121522)
    
    # Assigning a Subscript to a Name (line 721):
    
    # Obtaining the type of the subscript
    int_121523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 8), 'int')
    
    # Call to _minmax(...): (line 721)
    # Processing the call arguments (line 721)
    # Getting the type of 'coor' (line 721)
    coor_121525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 29), 'coor', False)
    # Getting the type of 'minc' (line 721)
    minc_121526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 35), 'minc', False)
    # Getting the type of 'maxc' (line 721)
    maxc_121527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 41), 'maxc', False)
    # Processing the call keyword arguments (line 721)
    kwargs_121528 = {}
    # Getting the type of '_minmax' (line 721)
    _minmax_121524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 21), '_minmax', False)
    # Calling _minmax(args, kwargs) (line 721)
    _minmax_call_result_121529 = invoke(stypy.reporting.localization.Localization(__file__, 721, 21), _minmax_121524, *[coor_121525, minc_121526, maxc_121527], **kwargs_121528)
    
    # Obtaining the member '__getitem__' of a type (line 721)
    getitem___121530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 8), _minmax_call_result_121529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 721)
    subscript_call_result_121531 = invoke(stypy.reporting.localization.Localization(__file__, 721, 8), getitem___121530, int_121523)
    
    # Assigning a type to the variable 'tuple_var_assignment_120090' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'tuple_var_assignment_120090', subscript_call_result_121531)
    
    # Assigning a Name to a Name (line 721):
    # Getting the type of 'tuple_var_assignment_120089' (line 721)
    tuple_var_assignment_120089_121532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'tuple_var_assignment_120089')
    # Assigning a type to the variable 'minc' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'minc', tuple_var_assignment_120089_121532)
    
    # Assigning a Name to a Name (line 721):
    # Getting the type of 'tuple_var_assignment_120090' (line 721)
    tuple_var_assignment_120090_121533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 8), 'tuple_var_assignment_120090')
    # Assigning a type to the variable 'maxc' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 14), 'maxc', tuple_var_assignment_120090_121533)
    
    # Assigning a Call to a Name (line 722):
    
    # Assigning a Call to a Name (line 722):
    
    # Call to int(...): (line 722)
    # Processing the call arguments (line 722)
    
    # Obtaining the type of the subscript
    int_121535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 22), 'int')
    # Getting the type of 'maxc' (line 722)
    maxc_121536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 17), 'maxc', False)
    # Obtaining the member '__getitem__' of a type (line 722)
    getitem___121537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 17), maxc_121536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 722)
    subscript_call_result_121538 = invoke(stypy.reporting.localization.Localization(__file__, 722, 17), getitem___121537, int_121535)
    
    
    # Obtaining the type of the subscript
    int_121539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 32), 'int')
    # Getting the type of 'minc' (line 722)
    minc_121540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 27), 'minc', False)
    # Obtaining the member '__getitem__' of a type (line 722)
    getitem___121541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 27), minc_121540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 722)
    subscript_call_result_121542 = invoke(stypy.reporting.localization.Localization(__file__, 722, 27), getitem___121541, int_121539)
    
    # Applying the binary operator '-' (line 722)
    result_sub_121543 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 17), '-', subscript_call_result_121538, subscript_call_result_121542)
    
    float_121544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 37), 'float')
    # Applying the binary operator '+' (line 722)
    result_add_121545 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 35), '+', result_sub_121543, float_121544)
    
    # Processing the call keyword arguments (line 722)
    kwargs_121546 = {}
    # Getting the type of 'int' (line 722)
    int_121534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 13), 'int', False)
    # Calling int(args, kwargs) (line 722)
    int_call_result_121547 = invoke(stypy.reporting.localization.Localization(__file__, 722, 13), int_121534, *[result_add_121545], **kwargs_121546)
    
    # Assigning a type to the variable 'oy' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'oy', int_call_result_121547)
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Call to int(...): (line 723)
    # Processing the call arguments (line 723)
    
    # Obtaining the type of the subscript
    int_121549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 22), 'int')
    # Getting the type of 'maxc' (line 723)
    maxc_121550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 17), 'maxc', False)
    # Obtaining the member '__getitem__' of a type (line 723)
    getitem___121551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 17), maxc_121550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 723)
    subscript_call_result_121552 = invoke(stypy.reporting.localization.Localization(__file__, 723, 17), getitem___121551, int_121549)
    
    
    # Obtaining the type of the subscript
    int_121553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 32), 'int')
    # Getting the type of 'minc' (line 723)
    minc_121554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 27), 'minc', False)
    # Obtaining the member '__getitem__' of a type (line 723)
    getitem___121555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 27), minc_121554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 723)
    subscript_call_result_121556 = invoke(stypy.reporting.localization.Localization(__file__, 723, 27), getitem___121555, int_121553)
    
    # Applying the binary operator '-' (line 723)
    result_sub_121557 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 17), '-', subscript_call_result_121552, subscript_call_result_121556)
    
    float_121558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 37), 'float')
    # Applying the binary operator '+' (line 723)
    result_add_121559 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 35), '+', result_sub_121557, float_121558)
    
    # Processing the call keyword arguments (line 723)
    kwargs_121560 = {}
    # Getting the type of 'int' (line 723)
    int_121548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 13), 'int', False)
    # Calling int(args, kwargs) (line 723)
    int_call_result_121561 = invoke(stypy.reporting.localization.Localization(__file__, 723, 13), int_121548, *[result_add_121559], **kwargs_121560)
    
    # Assigning a type to the variable 'ox' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'ox', int_call_result_121561)
    # SSA branch for the else part of an if statement (line 711)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 725):
    
    # Assigning a Subscript to a Name (line 725):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 30), 'int')
    # Getting the type of 'axes' (line 725)
    axes_121563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 25), 'axes')
    # Obtaining the member '__getitem__' of a type (line 725)
    getitem___121564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 25), axes_121563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 725)
    subscript_call_result_121565 = invoke(stypy.reporting.localization.Localization(__file__, 725, 25), getitem___121564, int_121562)
    
    # Getting the type of 'input' (line 725)
    input_121566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 13), 'input')
    # Obtaining the member 'shape' of a type (line 725)
    shape_121567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 13), input_121566, 'shape')
    # Obtaining the member '__getitem__' of a type (line 725)
    getitem___121568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 13), shape_121567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 725)
    subscript_call_result_121569 = invoke(stypy.reporting.localization.Localization(__file__, 725, 13), getitem___121568, subscript_call_result_121565)
    
    # Assigning a type to the variable 'oy' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'oy', subscript_call_result_121569)
    
    # Assigning a Subscript to a Name (line 726):
    
    # Assigning a Subscript to a Name (line 726):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 30), 'int')
    # Getting the type of 'axes' (line 726)
    axes_121571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 25), 'axes')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___121572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 25), axes_121571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_121573 = invoke(stypy.reporting.localization.Localization(__file__, 726, 25), getitem___121572, int_121570)
    
    # Getting the type of 'input' (line 726)
    input_121574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 13), 'input')
    # Obtaining the member 'shape' of a type (line 726)
    shape_121575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 13), input_121574, 'shape')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___121576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 13), shape_121575, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_121577 = invoke(stypy.reporting.localization.Localization(__file__, 726, 13), getitem___121576, subscript_call_result_121573)
    
    # Assigning a type to the variable 'ox' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'ox', subscript_call_result_121577)
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 727):
    
    # Assigning a Call to a Name (line 727):
    
    # Call to zeros(...): (line 727)
    # Processing the call arguments (line 727)
    
    # Obtaining an instance of the builtin type 'tuple' (line 727)
    tuple_121580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 727)
    # Adding element type (line 727)
    int_121581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 727, 26), tuple_121580, int_121581)
    
    # Processing the call keyword arguments (line 727)
    # Getting the type of 'numpy' (line 727)
    numpy_121582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 37), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 727)
    float64_121583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 37), numpy_121582, 'float64')
    keyword_121584 = float64_121583
    kwargs_121585 = {'dtype': keyword_121584}
    # Getting the type of 'numpy' (line 727)
    numpy_121578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 13), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 727)
    zeros_121579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 13), numpy_121578, 'zeros')
    # Calling zeros(args, kwargs) (line 727)
    zeros_call_result_121586 = invoke(stypy.reporting.localization.Localization(__file__, 727, 13), zeros_121579, *[tuple_121580], **kwargs_121585)
    
    # Assigning a type to the variable 'offset' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'offset', zeros_call_result_121586)
    
    # Assigning a BinOp to a Subscript (line 728):
    
    # Assigning a BinOp to a Subscript (line 728):
    
    # Call to float(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'oy' (line 728)
    oy_121588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 22), 'oy', False)
    # Processing the call keyword arguments (line 728)
    kwargs_121589 = {}
    # Getting the type of 'float' (line 728)
    float_121587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), 'float', False)
    # Calling float(args, kwargs) (line 728)
    float_call_result_121590 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), float_121587, *[oy_121588], **kwargs_121589)
    
    float_121591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 28), 'float')
    # Applying the binary operator 'div' (line 728)
    result_div_121592 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 16), 'div', float_call_result_121590, float_121591)
    
    float_121593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 34), 'float')
    # Applying the binary operator '-' (line 728)
    result_sub_121594 = python_operator(stypy.reporting.localization.Localization(__file__, 728, 16), '-', result_div_121592, float_121593)
    
    # Getting the type of 'offset' (line 728)
    offset_121595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'offset')
    int_121596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 11), 'int')
    # Storing an element on a container (line 728)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 728, 4), offset_121595, (int_121596, result_sub_121594))
    
    # Assigning a BinOp to a Subscript (line 729):
    
    # Assigning a BinOp to a Subscript (line 729):
    
    # Call to float(...): (line 729)
    # Processing the call arguments (line 729)
    # Getting the type of 'ox' (line 729)
    ox_121598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 22), 'ox', False)
    # Processing the call keyword arguments (line 729)
    kwargs_121599 = {}
    # Getting the type of 'float' (line 729)
    float_121597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 16), 'float', False)
    # Calling float(args, kwargs) (line 729)
    float_call_result_121600 = invoke(stypy.reporting.localization.Localization(__file__, 729, 16), float_121597, *[ox_121598], **kwargs_121599)
    
    float_121601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 28), 'float')
    # Applying the binary operator 'div' (line 729)
    result_div_121602 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 16), 'div', float_call_result_121600, float_121601)
    
    float_121603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 34), 'float')
    # Applying the binary operator '-' (line 729)
    result_sub_121604 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 16), '-', result_div_121602, float_121603)
    
    # Getting the type of 'offset' (line 729)
    offset_121605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'offset')
    int_121606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, 11), 'int')
    # Storing an element on a container (line 729)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 729, 4), offset_121605, (int_121606, result_sub_121604))
    
    # Assigning a Call to a Name (line 730):
    
    # Assigning a Call to a Name (line 730):
    
    # Call to dot(...): (line 730)
    # Processing the call arguments (line 730)
    # Getting the type of 'matrix' (line 730)
    matrix_121609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 23), 'matrix', False)
    # Getting the type of 'offset' (line 730)
    offset_121610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 31), 'offset', False)
    # Processing the call keyword arguments (line 730)
    kwargs_121611 = {}
    # Getting the type of 'numpy' (line 730)
    numpy_121607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 13), 'numpy', False)
    # Obtaining the member 'dot' of a type (line 730)
    dot_121608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 13), numpy_121607, 'dot')
    # Calling dot(args, kwargs) (line 730)
    dot_call_result_121612 = invoke(stypy.reporting.localization.Localization(__file__, 730, 13), dot_121608, *[matrix_121609, offset_121610], **kwargs_121611)
    
    # Assigning a type to the variable 'offset' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 4), 'offset', dot_call_result_121612)
    
    # Assigning a Call to a Name (line 731):
    
    # Assigning a Call to a Name (line 731):
    
    # Call to zeros(...): (line 731)
    # Processing the call arguments (line 731)
    
    # Obtaining an instance of the builtin type 'tuple' (line 731)
    tuple_121615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 731)
    # Adding element type (line 731)
    int_121616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 731, 23), tuple_121615, int_121616)
    
    # Processing the call keyword arguments (line 731)
    # Getting the type of 'numpy' (line 731)
    numpy_121617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 34), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 731)
    float64_121618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 34), numpy_121617, 'float64')
    keyword_121619 = float64_121618
    kwargs_121620 = {'dtype': keyword_121619}
    # Getting the type of 'numpy' (line 731)
    numpy_121613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 10), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 731)
    zeros_121614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 731, 10), numpy_121613, 'zeros')
    # Calling zeros(args, kwargs) (line 731)
    zeros_call_result_121621 = invoke(stypy.reporting.localization.Localization(__file__, 731, 10), zeros_121614, *[tuple_121615], **kwargs_121620)
    
    # Assigning a type to the variable 'tmp' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'tmp', zeros_call_result_121621)
    
    # Assigning a BinOp to a Subscript (line 732):
    
    # Assigning a BinOp to a Subscript (line 732):
    
    # Call to float(...): (line 732)
    # Processing the call arguments (line 732)
    # Getting the type of 'iy' (line 732)
    iy_121623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 19), 'iy', False)
    # Processing the call keyword arguments (line 732)
    kwargs_121624 = {}
    # Getting the type of 'float' (line 732)
    float_121622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 13), 'float', False)
    # Calling float(args, kwargs) (line 732)
    float_call_result_121625 = invoke(stypy.reporting.localization.Localization(__file__, 732, 13), float_121622, *[iy_121623], **kwargs_121624)
    
    float_121626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 25), 'float')
    # Applying the binary operator 'div' (line 732)
    result_div_121627 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 13), 'div', float_call_result_121625, float_121626)
    
    float_121628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 31), 'float')
    # Applying the binary operator '-' (line 732)
    result_sub_121629 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 13), '-', result_div_121627, float_121628)
    
    # Getting the type of 'tmp' (line 732)
    tmp_121630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'tmp')
    int_121631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 8), 'int')
    # Storing an element on a container (line 732)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 4), tmp_121630, (int_121631, result_sub_121629))
    
    # Assigning a BinOp to a Subscript (line 733):
    
    # Assigning a BinOp to a Subscript (line 733):
    
    # Call to float(...): (line 733)
    # Processing the call arguments (line 733)
    # Getting the type of 'ix' (line 733)
    ix_121633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 19), 'ix', False)
    # Processing the call keyword arguments (line 733)
    kwargs_121634 = {}
    # Getting the type of 'float' (line 733)
    float_121632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 13), 'float', False)
    # Calling float(args, kwargs) (line 733)
    float_call_result_121635 = invoke(stypy.reporting.localization.Localization(__file__, 733, 13), float_121632, *[ix_121633], **kwargs_121634)
    
    float_121636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 25), 'float')
    # Applying the binary operator 'div' (line 733)
    result_div_121637 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 13), 'div', float_call_result_121635, float_121636)
    
    float_121638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 31), 'float')
    # Applying the binary operator '-' (line 733)
    result_sub_121639 = python_operator(stypy.reporting.localization.Localization(__file__, 733, 13), '-', result_div_121637, float_121638)
    
    # Getting the type of 'tmp' (line 733)
    tmp_121640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'tmp')
    int_121641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 8), 'int')
    # Storing an element on a container (line 733)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 4), tmp_121640, (int_121641, result_sub_121639))
    
    # Assigning a BinOp to a Name (line 734):
    
    # Assigning a BinOp to a Name (line 734):
    # Getting the type of 'tmp' (line 734)
    tmp_121642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 13), 'tmp')
    # Getting the type of 'offset' (line 734)
    offset_121643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 19), 'offset')
    # Applying the binary operator '-' (line 734)
    result_sub_121644 = python_operator(stypy.reporting.localization.Localization(__file__, 734, 13), '-', tmp_121642, offset_121643)
    
    # Assigning a type to the variable 'offset' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'offset', result_sub_121644)
    
    # Assigning a Call to a Name (line 735):
    
    # Assigning a Call to a Name (line 735):
    
    # Call to list(...): (line 735)
    # Processing the call arguments (line 735)
    # Getting the type of 'input' (line 735)
    input_121646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 24), 'input', False)
    # Obtaining the member 'shape' of a type (line 735)
    shape_121647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 735, 24), input_121646, 'shape')
    # Processing the call keyword arguments (line 735)
    kwargs_121648 = {}
    # Getting the type of 'list' (line 735)
    list_121645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 19), 'list', False)
    # Calling list(args, kwargs) (line 735)
    list_call_result_121649 = invoke(stypy.reporting.localization.Localization(__file__, 735, 19), list_121645, *[shape_121647], **kwargs_121648)
    
    # Assigning a type to the variable 'output_shape' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'output_shape', list_call_result_121649)
    
    # Assigning a Name to a Subscript (line 736):
    
    # Assigning a Name to a Subscript (line 736):
    # Getting the type of 'oy' (line 736)
    oy_121650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 28), 'oy')
    # Getting the type of 'output_shape' (line 736)
    output_shape_121651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'output_shape')
    
    # Obtaining the type of the subscript
    int_121652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 22), 'int')
    # Getting the type of 'axes' (line 736)
    axes_121653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 17), 'axes')
    # Obtaining the member '__getitem__' of a type (line 736)
    getitem___121654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 736, 17), axes_121653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 736)
    subscript_call_result_121655 = invoke(stypy.reporting.localization.Localization(__file__, 736, 17), getitem___121654, int_121652)
    
    # Storing an element on a container (line 736)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 736, 4), output_shape_121651, (subscript_call_result_121655, oy_121650))
    
    # Assigning a Name to a Subscript (line 737):
    
    # Assigning a Name to a Subscript (line 737):
    # Getting the type of 'ox' (line 737)
    ox_121656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 28), 'ox')
    # Getting the type of 'output_shape' (line 737)
    output_shape_121657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'output_shape')
    
    # Obtaining the type of the subscript
    int_121658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 22), 'int')
    # Getting the type of 'axes' (line 737)
    axes_121659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 17), 'axes')
    # Obtaining the member '__getitem__' of a type (line 737)
    getitem___121660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 737, 17), axes_121659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 737)
    subscript_call_result_121661 = invoke(stypy.reporting.localization.Localization(__file__, 737, 17), getitem___121660, int_121658)
    
    # Storing an element on a container (line 737)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 737, 4), output_shape_121657, (subscript_call_result_121661, ox_121656))
    
    # Assigning a Call to a Name (line 738):
    
    # Assigning a Call to a Name (line 738):
    
    # Call to tuple(...): (line 738)
    # Processing the call arguments (line 738)
    # Getting the type of 'output_shape' (line 738)
    output_shape_121663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 25), 'output_shape', False)
    # Processing the call keyword arguments (line 738)
    kwargs_121664 = {}
    # Getting the type of 'tuple' (line 738)
    tuple_121662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 19), 'tuple', False)
    # Calling tuple(args, kwargs) (line 738)
    tuple_call_result_121665 = invoke(stypy.reporting.localization.Localization(__file__, 738, 19), tuple_121662, *[output_shape_121663], **kwargs_121664)
    
    # Assigning a type to the variable 'output_shape' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'output_shape', tuple_call_result_121665)
    
    # Assigning a Call to a Tuple (line 739):
    
    # Assigning a Subscript to a Name (line 739):
    
    # Obtaining the type of the subscript
    int_121666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 4), 'int')
    
    # Call to _get_output(...): (line 739)
    # Processing the call arguments (line 739)
    # Getting the type of 'output' (line 739)
    output_121669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 51), 'output', False)
    # Getting the type of 'input' (line 739)
    input_121670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 59), 'input', False)
    # Processing the call keyword arguments (line 739)
    # Getting the type of 'output_shape' (line 740)
    output_shape_121671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 57), 'output_shape', False)
    keyword_121672 = output_shape_121671
    kwargs_121673 = {'shape': keyword_121672}
    # Getting the type of '_ni_support' (line 739)
    _ni_support_121667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 739)
    _get_output_121668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 27), _ni_support_121667, '_get_output')
    # Calling _get_output(args, kwargs) (line 739)
    _get_output_call_result_121674 = invoke(stypy.reporting.localization.Localization(__file__, 739, 27), _get_output_121668, *[output_121669, input_121670], **kwargs_121673)
    
    # Obtaining the member '__getitem__' of a type (line 739)
    getitem___121675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 4), _get_output_call_result_121674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 739)
    subscript_call_result_121676 = invoke(stypy.reporting.localization.Localization(__file__, 739, 4), getitem___121675, int_121666)
    
    # Assigning a type to the variable 'tuple_var_assignment_120091' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'tuple_var_assignment_120091', subscript_call_result_121676)
    
    # Assigning a Subscript to a Name (line 739):
    
    # Obtaining the type of the subscript
    int_121677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 4), 'int')
    
    # Call to _get_output(...): (line 739)
    # Processing the call arguments (line 739)
    # Getting the type of 'output' (line 739)
    output_121680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 51), 'output', False)
    # Getting the type of 'input' (line 739)
    input_121681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 59), 'input', False)
    # Processing the call keyword arguments (line 739)
    # Getting the type of 'output_shape' (line 740)
    output_shape_121682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 57), 'output_shape', False)
    keyword_121683 = output_shape_121682
    kwargs_121684 = {'shape': keyword_121683}
    # Getting the type of '_ni_support' (line 739)
    _ni_support_121678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 739)
    _get_output_121679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 27), _ni_support_121678, '_get_output')
    # Calling _get_output(args, kwargs) (line 739)
    _get_output_call_result_121685 = invoke(stypy.reporting.localization.Localization(__file__, 739, 27), _get_output_121679, *[output_121680, input_121681], **kwargs_121684)
    
    # Obtaining the member '__getitem__' of a type (line 739)
    getitem___121686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 739, 4), _get_output_call_result_121685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 739)
    subscript_call_result_121687 = invoke(stypy.reporting.localization.Localization(__file__, 739, 4), getitem___121686, int_121677)
    
    # Assigning a type to the variable 'tuple_var_assignment_120092' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'tuple_var_assignment_120092', subscript_call_result_121687)
    
    # Assigning a Name to a Name (line 739):
    # Getting the type of 'tuple_var_assignment_120091' (line 739)
    tuple_var_assignment_120091_121688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'tuple_var_assignment_120091')
    # Assigning a type to the variable 'output' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'output', tuple_var_assignment_120091_121688)
    
    # Assigning a Name to a Name (line 739):
    # Getting the type of 'tuple_var_assignment_120092' (line 739)
    tuple_var_assignment_120092_121689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 4), 'tuple_var_assignment_120092')
    # Assigning a type to the variable 'return_value' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 12), 'return_value', tuple_var_assignment_120092_121689)
    
    
    # Getting the type of 'input' (line 741)
    input_121690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 7), 'input')
    # Obtaining the member 'ndim' of a type (line 741)
    ndim_121691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 7), input_121690, 'ndim')
    int_121692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 741, 21), 'int')
    # Applying the binary operator '<=' (line 741)
    result_le_121693 = python_operator(stypy.reporting.localization.Localization(__file__, 741, 7), '<=', ndim_121691, int_121692)
    
    # Testing the type of an if condition (line 741)
    if_condition_121694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 741, 4), result_le_121693)
    # Assigning a type to the variable 'if_condition_121694' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'if_condition_121694', if_condition_121694)
    # SSA begins for if statement (line 741)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to affine_transform(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'input' (line 742)
    input_121696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 25), 'input', False)
    # Getting the type of 'matrix' (line 742)
    matrix_121697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 32), 'matrix', False)
    # Getting the type of 'offset' (line 742)
    offset_121698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 40), 'offset', False)
    # Getting the type of 'output_shape' (line 742)
    output_shape_121699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 48), 'output_shape', False)
    # Getting the type of 'output' (line 742)
    output_121700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 62), 'output', False)
    # Getting the type of 'order' (line 743)
    order_121701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 25), 'order', False)
    # Getting the type of 'mode' (line 743)
    mode_121702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 32), 'mode', False)
    # Getting the type of 'cval' (line 743)
    cval_121703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 38), 'cval', False)
    # Getting the type of 'prefilter' (line 743)
    prefilter_121704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 44), 'prefilter', False)
    # Processing the call keyword arguments (line 742)
    kwargs_121705 = {}
    # Getting the type of 'affine_transform' (line 742)
    affine_transform_121695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'affine_transform', False)
    # Calling affine_transform(args, kwargs) (line 742)
    affine_transform_call_result_121706 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), affine_transform_121695, *[input_121696, matrix_121697, offset_121698, output_shape_121699, output_121700, order_121701, mode_121702, cval_121703, prefilter_121704], **kwargs_121705)
    
    # SSA branch for the else part of an if statement (line 741)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 745):
    
    # Assigning a List to a Name (line 745):
    
    # Obtaining an instance of the builtin type 'list' (line 745)
    list_121707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 745)
    
    # Assigning a type to the variable 'coordinates' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'coordinates', list_121707)
    
    # Assigning a Call to a Name (line 746):
    
    # Assigning a Call to a Name (line 746):
    
    # Call to product(...): (line 746)
    # Processing the call arguments (line 746)
    # Getting the type of 'input' (line 746)
    input_121710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 29), 'input', False)
    # Obtaining the member 'shape' of a type (line 746)
    shape_121711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 29), input_121710, 'shape')
    # Processing the call keyword arguments (line 746)
    int_121712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 46), 'int')
    keyword_121713 = int_121712
    kwargs_121714 = {'axis': keyword_121713}
    # Getting the type of 'numpy' (line 746)
    numpy_121708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 15), 'numpy', False)
    # Obtaining the member 'product' of a type (line 746)
    product_121709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 15), numpy_121708, 'product')
    # Calling product(args, kwargs) (line 746)
    product_call_result_121715 = invoke(stypy.reporting.localization.Localization(__file__, 746, 15), product_121709, *[shape_121711], **kwargs_121714)
    
    # Assigning a type to the variable 'size' (line 746)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'size', product_call_result_121715)
    
    # Getting the type of 'size' (line 747)
    size_121716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'size')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 34), 'int')
    # Getting the type of 'axes' (line 747)
    axes_121718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 29), 'axes')
    # Obtaining the member '__getitem__' of a type (line 747)
    getitem___121719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 29), axes_121718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 747)
    subscript_call_result_121720 = invoke(stypy.reporting.localization.Localization(__file__, 747, 29), getitem___121719, int_121717)
    
    # Getting the type of 'input' (line 747)
    input_121721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 17), 'input')
    # Obtaining the member 'shape' of a type (line 747)
    shape_121722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 17), input_121721, 'shape')
    # Obtaining the member '__getitem__' of a type (line 747)
    getitem___121723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 17), shape_121722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 747)
    subscript_call_result_121724 = invoke(stypy.reporting.localization.Localization(__file__, 747, 17), getitem___121723, subscript_call_result_121720)
    
    # Applying the binary operator '//=' (line 747)
    result_ifloordiv_121725 = python_operator(stypy.reporting.localization.Localization(__file__, 747, 8), '//=', size_121716, subscript_call_result_121724)
    # Assigning a type to the variable 'size' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 8), 'size', result_ifloordiv_121725)
    
    
    # Getting the type of 'size' (line 748)
    size_121726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'size')
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 34), 'int')
    # Getting the type of 'axes' (line 748)
    axes_121728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 29), 'axes')
    # Obtaining the member '__getitem__' of a type (line 748)
    getitem___121729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 29), axes_121728, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 748)
    subscript_call_result_121730 = invoke(stypy.reporting.localization.Localization(__file__, 748, 29), getitem___121729, int_121727)
    
    # Getting the type of 'input' (line 748)
    input_121731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 17), 'input')
    # Obtaining the member 'shape' of a type (line 748)
    shape_121732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 17), input_121731, 'shape')
    # Obtaining the member '__getitem__' of a type (line 748)
    getitem___121733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 17), shape_121732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 748)
    subscript_call_result_121734 = invoke(stypy.reporting.localization.Localization(__file__, 748, 17), getitem___121733, subscript_call_result_121730)
    
    # Applying the binary operator '//=' (line 748)
    result_ifloordiv_121735 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 8), '//=', size_121726, subscript_call_result_121734)
    # Assigning a type to the variable 'size' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 8), 'size', result_ifloordiv_121735)
    
    
    
    # Call to range(...): (line 749)
    # Processing the call arguments (line 749)
    # Getting the type of 'input' (line 749)
    input_121737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 24), 'input', False)
    # Obtaining the member 'ndim' of a type (line 749)
    ndim_121738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 24), input_121737, 'ndim')
    # Processing the call keyword arguments (line 749)
    kwargs_121739 = {}
    # Getting the type of 'range' (line 749)
    range_121736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 18), 'range', False)
    # Calling range(args, kwargs) (line 749)
    range_call_result_121740 = invoke(stypy.reporting.localization.Localization(__file__, 749, 18), range_121736, *[ndim_121738], **kwargs_121739)
    
    # Testing the type of a for loop iterable (line 749)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 749, 8), range_call_result_121740)
    # Getting the type of the for loop variable (line 749)
    for_loop_var_121741 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 749, 8), range_call_result_121740)
    # Assigning a type to the variable 'ii' (line 749)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'ii', for_loop_var_121741)
    # SSA begins for a for statement (line 749)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ii' (line 750)
    ii_121742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 15), 'ii')
    # Getting the type of 'axes' (line 750)
    axes_121743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 25), 'axes')
    # Applying the binary operator 'notin' (line 750)
    result_contains_121744 = python_operator(stypy.reporting.localization.Localization(__file__, 750, 15), 'notin', ii_121742, axes_121743)
    
    # Testing the type of an if condition (line 750)
    if_condition_121745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 750, 12), result_contains_121744)
    # Assigning a type to the variable 'if_condition_121745' (line 750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'if_condition_121745', if_condition_121745)
    # SSA begins for if statement (line 750)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 751)
    # Processing the call arguments (line 751)
    int_121748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 35), 'int')
    # Processing the call keyword arguments (line 751)
    kwargs_121749 = {}
    # Getting the type of 'coordinates' (line 751)
    coordinates_121746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'coordinates', False)
    # Obtaining the member 'append' of a type (line 751)
    append_121747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 16), coordinates_121746, 'append')
    # Calling append(args, kwargs) (line 751)
    append_call_result_121750 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), append_121747, *[int_121748], **kwargs_121749)
    
    # SSA branch for the else part of an if statement (line 750)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 753)
    # Processing the call arguments (line 753)
    
    # Call to slice(...): (line 753)
    # Processing the call arguments (line 753)
    # Getting the type of 'None' (line 753)
    None_121754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 41), 'None', False)
    # Getting the type of 'None' (line 753)
    None_121755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 47), 'None', False)
    # Getting the type of 'None' (line 753)
    None_121756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 53), 'None', False)
    # Processing the call keyword arguments (line 753)
    kwargs_121757 = {}
    # Getting the type of 'slice' (line 753)
    slice_121753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 35), 'slice', False)
    # Calling slice(args, kwargs) (line 753)
    slice_call_result_121758 = invoke(stypy.reporting.localization.Localization(__file__, 753, 35), slice_121753, *[None_121754, None_121755, None_121756], **kwargs_121757)
    
    # Processing the call keyword arguments (line 753)
    kwargs_121759 = {}
    # Getting the type of 'coordinates' (line 753)
    coordinates_121751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 16), 'coordinates', False)
    # Obtaining the member 'append' of a type (line 753)
    append_121752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 753, 16), coordinates_121751, 'append')
    # Calling append(args, kwargs) (line 753)
    append_call_result_121760 = invoke(stypy.reporting.localization.Localization(__file__, 753, 16), append_121752, *[slice_call_result_121758], **kwargs_121759)
    
    # SSA join for if statement (line 750)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 754):
    
    # Assigning a Call to a Name (line 754):
    
    # Call to list(...): (line 754)
    # Processing the call arguments (line 754)
    
    # Call to range(...): (line 754)
    # Processing the call arguments (line 754)
    # Getting the type of 'input' (line 754)
    input_121763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 31), 'input', False)
    # Obtaining the member 'ndim' of a type (line 754)
    ndim_121764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 31), input_121763, 'ndim')
    # Processing the call keyword arguments (line 754)
    kwargs_121765 = {}
    # Getting the type of 'range' (line 754)
    range_121762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 25), 'range', False)
    # Calling range(args, kwargs) (line 754)
    range_call_result_121766 = invoke(stypy.reporting.localization.Localization(__file__, 754, 25), range_121762, *[ndim_121764], **kwargs_121765)
    
    # Processing the call keyword arguments (line 754)
    kwargs_121767 = {}
    # Getting the type of 'list' (line 754)
    list_121761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 20), 'list', False)
    # Calling list(args, kwargs) (line 754)
    list_call_result_121768 = invoke(stypy.reporting.localization.Localization(__file__, 754, 20), list_121761, *[range_call_result_121766], **kwargs_121767)
    
    # Assigning a type to the variable 'iter_axes' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 8), 'iter_axes', list_call_result_121768)
    
    # Call to reverse(...): (line 755)
    # Processing the call keyword arguments (line 755)
    kwargs_121771 = {}
    # Getting the type of 'iter_axes' (line 755)
    iter_axes_121769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'iter_axes', False)
    # Obtaining the member 'reverse' of a type (line 755)
    reverse_121770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 8), iter_axes_121769, 'reverse')
    # Calling reverse(args, kwargs) (line 755)
    reverse_call_result_121772 = invoke(stypy.reporting.localization.Localization(__file__, 755, 8), reverse_121770, *[], **kwargs_121771)
    
    
    # Call to remove(...): (line 756)
    # Processing the call arguments (line 756)
    
    # Obtaining the type of the subscript
    int_121775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 30), 'int')
    # Getting the type of 'axes' (line 756)
    axes_121776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 25), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___121777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 25), axes_121776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_121778 = invoke(stypy.reporting.localization.Localization(__file__, 756, 25), getitem___121777, int_121775)
    
    # Processing the call keyword arguments (line 756)
    kwargs_121779 = {}
    # Getting the type of 'iter_axes' (line 756)
    iter_axes_121773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 8), 'iter_axes', False)
    # Obtaining the member 'remove' of a type (line 756)
    remove_121774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 8), iter_axes_121773, 'remove')
    # Calling remove(args, kwargs) (line 756)
    remove_call_result_121780 = invoke(stypy.reporting.localization.Localization(__file__, 756, 8), remove_121774, *[subscript_call_result_121778], **kwargs_121779)
    
    
    # Call to remove(...): (line 757)
    # Processing the call arguments (line 757)
    
    # Obtaining the type of the subscript
    int_121783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 30), 'int')
    # Getting the type of 'axes' (line 757)
    axes_121784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 25), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 757)
    getitem___121785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 25), axes_121784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 757)
    subscript_call_result_121786 = invoke(stypy.reporting.localization.Localization(__file__, 757, 25), getitem___121785, int_121783)
    
    # Processing the call keyword arguments (line 757)
    kwargs_121787 = {}
    # Getting the type of 'iter_axes' (line 757)
    iter_axes_121781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 8), 'iter_axes', False)
    # Obtaining the member 'remove' of a type (line 757)
    remove_121782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 757, 8), iter_axes_121781, 'remove')
    # Calling remove(args, kwargs) (line 757)
    remove_call_result_121788 = invoke(stypy.reporting.localization.Localization(__file__, 757, 8), remove_121782, *[subscript_call_result_121786], **kwargs_121787)
    
    
    # Assigning a Tuple to a Name (line 758):
    
    # Assigning a Tuple to a Name (line 758):
    
    # Obtaining an instance of the builtin type 'tuple' (line 758)
    tuple_121789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 758)
    # Adding element type (line 758)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 32), 'int')
    # Getting the type of 'axes' (line 758)
    axes_121791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 27), 'axes')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___121792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 27), axes_121791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_121793 = invoke(stypy.reporting.localization.Localization(__file__, 758, 27), getitem___121792, int_121790)
    
    # Getting the type of 'output_shape' (line 758)
    output_shape_121794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 14), 'output_shape')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___121795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 14), output_shape_121794, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_121796 = invoke(stypy.reporting.localization.Localization(__file__, 758, 14), getitem___121795, subscript_call_result_121793)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 14), tuple_121789, subscript_call_result_121796)
    # Adding element type (line 758)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_121797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 55), 'int')
    # Getting the type of 'axes' (line 758)
    axes_121798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 50), 'axes')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___121799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 50), axes_121798, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_121800 = invoke(stypy.reporting.localization.Localization(__file__, 758, 50), getitem___121799, int_121797)
    
    # Getting the type of 'output_shape' (line 758)
    output_shape_121801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 37), 'output_shape')
    # Obtaining the member '__getitem__' of a type (line 758)
    getitem___121802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 37), output_shape_121801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 758)
    subscript_call_result_121803 = invoke(stypy.reporting.localization.Localization(__file__, 758, 37), getitem___121802, subscript_call_result_121800)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 14), tuple_121789, subscript_call_result_121803)
    
    # Assigning a type to the variable 'os' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 8), 'os', tuple_121789)
    
    
    # Call to range(...): (line 759)
    # Processing the call arguments (line 759)
    # Getting the type of 'size' (line 759)
    size_121805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 24), 'size', False)
    # Processing the call keyword arguments (line 759)
    kwargs_121806 = {}
    # Getting the type of 'range' (line 759)
    range_121804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 18), 'range', False)
    # Calling range(args, kwargs) (line 759)
    range_call_result_121807 = invoke(stypy.reporting.localization.Localization(__file__, 759, 18), range_121804, *[size_121805], **kwargs_121806)
    
    # Testing the type of a for loop iterable (line 759)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 759, 8), range_call_result_121807)
    # Getting the type of the for loop variable (line 759)
    for_loop_var_121808 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 759, 8), range_call_result_121807)
    # Assigning a type to the variable 'ii' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 8), 'ii', for_loop_var_121808)
    # SSA begins for a for statement (line 759)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 760):
    
    # Assigning a Subscript to a Name (line 760):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 760)
    # Processing the call arguments (line 760)
    # Getting the type of 'coordinates' (line 760)
    coordinates_121810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 29), 'coordinates', False)
    # Processing the call keyword arguments (line 760)
    kwargs_121811 = {}
    # Getting the type of 'tuple' (line 760)
    tuple_121809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 23), 'tuple', False)
    # Calling tuple(args, kwargs) (line 760)
    tuple_call_result_121812 = invoke(stypy.reporting.localization.Localization(__file__, 760, 23), tuple_121809, *[coordinates_121810], **kwargs_121811)
    
    # Getting the type of 'input' (line 760)
    input_121813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 17), 'input')
    # Obtaining the member '__getitem__' of a type (line 760)
    getitem___121814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 17), input_121813, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 760)
    subscript_call_result_121815 = invoke(stypy.reporting.localization.Localization(__file__, 760, 17), getitem___121814, tuple_call_result_121812)
    
    # Assigning a type to the variable 'ia' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'ia', subscript_call_result_121815)
    
    # Assigning a Subscript to a Name (line 761):
    
    # Assigning a Subscript to a Name (line 761):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 761)
    # Processing the call arguments (line 761)
    # Getting the type of 'coordinates' (line 761)
    coordinates_121817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'coordinates', False)
    # Processing the call keyword arguments (line 761)
    kwargs_121818 = {}
    # Getting the type of 'tuple' (line 761)
    tuple_121816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 24), 'tuple', False)
    # Calling tuple(args, kwargs) (line 761)
    tuple_call_result_121819 = invoke(stypy.reporting.localization.Localization(__file__, 761, 24), tuple_121816, *[coordinates_121817], **kwargs_121818)
    
    # Getting the type of 'output' (line 761)
    output_121820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 17), 'output')
    # Obtaining the member '__getitem__' of a type (line 761)
    getitem___121821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 761, 17), output_121820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 761)
    subscript_call_result_121822 = invoke(stypy.reporting.localization.Localization(__file__, 761, 17), getitem___121821, tuple_call_result_121819)
    
    # Assigning a type to the variable 'oa' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'oa', subscript_call_result_121822)
    
    # Call to affine_transform(...): (line 762)
    # Processing the call arguments (line 762)
    # Getting the type of 'ia' (line 762)
    ia_121824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 29), 'ia', False)
    # Getting the type of 'matrix' (line 762)
    matrix_121825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 33), 'matrix', False)
    # Getting the type of 'offset' (line 762)
    offset_121826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 41), 'offset', False)
    # Getting the type of 'os' (line 762)
    os_121827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 49), 'os', False)
    # Getting the type of 'oa' (line 762)
    oa_121828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 53), 'oa', False)
    # Getting the type of 'order' (line 762)
    order_121829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 57), 'order', False)
    # Getting the type of 'mode' (line 762)
    mode_121830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 64), 'mode', False)
    # Getting the type of 'cval' (line 763)
    cval_121831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 29), 'cval', False)
    # Getting the type of 'prefilter' (line 763)
    prefilter_121832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 35), 'prefilter', False)
    # Processing the call keyword arguments (line 762)
    kwargs_121833 = {}
    # Getting the type of 'affine_transform' (line 762)
    affine_transform_121823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'affine_transform', False)
    # Calling affine_transform(args, kwargs) (line 762)
    affine_transform_call_result_121834 = invoke(stypy.reporting.localization.Localization(__file__, 762, 12), affine_transform_121823, *[ia_121824, matrix_121825, offset_121826, os_121827, oa_121828, order_121829, mode_121830, cval_121831, prefilter_121832], **kwargs_121833)
    
    
    # Getting the type of 'iter_axes' (line 764)
    iter_axes_121835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 22), 'iter_axes')
    # Testing the type of a for loop iterable (line 764)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 764, 12), iter_axes_121835)
    # Getting the type of the for loop variable (line 764)
    for_loop_var_121836 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 764, 12), iter_axes_121835)
    # Assigning a type to the variable 'jj' (line 764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 764, 12), 'jj', for_loop_var_121836)
    # SSA begins for a for statement (line 764)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'jj' (line 765)
    jj_121837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 31), 'jj')
    # Getting the type of 'coordinates' (line 765)
    coordinates_121838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 19), 'coordinates')
    # Obtaining the member '__getitem__' of a type (line 765)
    getitem___121839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 19), coordinates_121838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 765)
    subscript_call_result_121840 = invoke(stypy.reporting.localization.Localization(__file__, 765, 19), getitem___121839, jj_121837)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'jj' (line 765)
    jj_121841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 49), 'jj')
    # Getting the type of 'input' (line 765)
    input_121842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 37), 'input')
    # Obtaining the member 'shape' of a type (line 765)
    shape_121843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 37), input_121842, 'shape')
    # Obtaining the member '__getitem__' of a type (line 765)
    getitem___121844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 37), shape_121843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 765)
    subscript_call_result_121845 = invoke(stypy.reporting.localization.Localization(__file__, 765, 37), getitem___121844, jj_121841)
    
    int_121846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 765, 55), 'int')
    # Applying the binary operator '-' (line 765)
    result_sub_121847 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 37), '-', subscript_call_result_121845, int_121846)
    
    # Applying the binary operator '<' (line 765)
    result_lt_121848 = python_operator(stypy.reporting.localization.Localization(__file__, 765, 19), '<', subscript_call_result_121840, result_sub_121847)
    
    # Testing the type of an if condition (line 765)
    if_condition_121849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 765, 16), result_lt_121848)
    # Assigning a type to the variable 'if_condition_121849' (line 765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 765, 16), 'if_condition_121849', if_condition_121849)
    # SSA begins for if statement (line 765)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'coordinates' (line 766)
    coordinates_121850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 20), 'coordinates')
    
    # Obtaining the type of the subscript
    # Getting the type of 'jj' (line 766)
    jj_121851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 32), 'jj')
    # Getting the type of 'coordinates' (line 766)
    coordinates_121852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 20), 'coordinates')
    # Obtaining the member '__getitem__' of a type (line 766)
    getitem___121853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 20), coordinates_121852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 766)
    subscript_call_result_121854 = invoke(stypy.reporting.localization.Localization(__file__, 766, 20), getitem___121853, jj_121851)
    
    int_121855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, 39), 'int')
    # Applying the binary operator '+=' (line 766)
    result_iadd_121856 = python_operator(stypy.reporting.localization.Localization(__file__, 766, 20), '+=', subscript_call_result_121854, int_121855)
    # Getting the type of 'coordinates' (line 766)
    coordinates_121857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 20), 'coordinates')
    # Getting the type of 'jj' (line 766)
    jj_121858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 32), 'jj')
    # Storing an element on a container (line 766)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 766, 20), coordinates_121857, (jj_121858, result_iadd_121856))
    
    # SSA branch for the else part of an if statement (line 765)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 769):
    
    # Assigning a Num to a Subscript (line 769):
    int_121859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 769, 38), 'int')
    # Getting the type of 'coordinates' (line 769)
    coordinates_121860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 20), 'coordinates')
    # Getting the type of 'jj' (line 769)
    jj_121861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 32), 'jj')
    # Storing an element on a container (line 769)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 769, 20), coordinates_121860, (jj_121861, int_121859))
    # SSA join for if statement (line 765)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 741)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 770)
    return_value_121862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'stypy_return_type', return_value_121862)
    
    # ################# End of 'rotate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rotate' in the type store
    # Getting the type of 'stypy_return_type' (line 644)
    stypy_return_type_121863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_121863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rotate'
    return stypy_return_type_121863

# Assigning a type to the variable 'rotate' (line 644)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 0), 'rotate', rotate)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
