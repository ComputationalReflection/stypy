
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
37: from scipy.misc import doccer
38: from scipy._lib._version import NumpyVersion
39: 
40: __all__ = ['correlate1d', 'convolve1d', 'gaussian_filter1d', 'gaussian_filter',
41:            'prewitt', 'sobel', 'generic_laplace', 'laplace',
42:            'gaussian_laplace', 'generic_gradient_magnitude',
43:            'gaussian_gradient_magnitude', 'correlate', 'convolve',
44:            'uniform_filter1d', 'uniform_filter', 'minimum_filter1d',
45:            'maximum_filter1d', 'minimum_filter', 'maximum_filter',
46:            'rank_filter', 'median_filter', 'percentile_filter',
47:            'generic_filter1d', 'generic_filter']
48: 
49: 
50: _input_doc = \
51: '''input : array_like
52:     Input array to filter.'''
53: _axis_doc = \
54: '''axis : int, optional
55:     The axis of `input` along which to calculate. Default is -1.'''
56: _output_doc = \
57: '''output : array, optional
58:     The `output` parameter passes an array in which to store the
59:     filter output. Output array should have different name as compared
60:     to input array to avoid aliasing errors.'''
61: _size_foot_doc = \
62: '''size : scalar or tuple, optional
63:     See footprint, below
64: footprint : array, optional
65:     Either `size` or `footprint` must be defined.  `size` gives
66:     the shape that is taken from the input array, at every element
67:     position, to define the input to the filter function.
68:     `footprint` is a boolean array that specifies (implicitly) a
69:     shape, but also which of the elements within this shape will get
70:     passed to the filter function.  Thus ``size=(n,m)`` is equivalent
71:     to ``footprint=np.ones((n,m))``.  We adjust `size` to the number
72:     of dimensions of the input array, so that, if the input array is
73:     shape (10,10,10), and `size` is 2, then the actual size used is
74:     (2,2,2).
75: '''
76: _mode_doc = \
77: '''mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
78:     The `mode` parameter determines how the array borders are
79:     handled, where `cval` is the value when mode is equal to
80:     'constant'. Default is 'reflect''''
81: _mode_multiple_doc = \
82: '''mode : str or sequence, optional
83:     The `mode` parameter determines how the array borders are
84:     handled. Valid modes are {'reflect', 'constant', 'nearest',
85:     'mirror', 'wrap'}. `cval` is the value used when mode is equal to
86:     'constant'. A list of modes with length equal to the number of
87:     axes can be provided to specify different modes for different
88:     axes. Default is 'reflect''''
89: _cval_doc = \
90: '''cval : scalar, optional
91:     Value to fill past edges of input if `mode` is 'constant'. Default
92:     is 0.0'''
93: _origin_doc = \
94: '''origin : scalar, optional
95:     The `origin` parameter controls the placement of the filter.
96:     Default 0.0.'''
97: _extra_arguments_doc = \
98: '''extra_arguments : sequence, optional
99:     Sequence of extra positional arguments to pass to passed function'''
100: _extra_keywords_doc = \
101: '''extra_keywords : dict, optional
102:     dict of extra keyword arguments to pass to passed function'''
103: 
104: docdict = {
105:     'input': _input_doc,
106:     'axis': _axis_doc,
107:     'output': _output_doc,
108:     'size_foot': _size_foot_doc,
109:     'mode': _mode_doc,
110:     'mode_multiple': _mode_multiple_doc,
111:     'cval': _cval_doc,
112:     'origin': _origin_doc,
113:     'extra_arguments': _extra_arguments_doc,
114:     'extra_keywords': _extra_keywords_doc,
115:     }
116: 
117: docfiller = doccer.filldoc(docdict)
118: 
119: 
120: @docfiller
121: def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
122:                 cval=0.0, origin=0):
123:     '''Calculate a one-dimensional correlation along the given axis.
124: 
125:     The lines of the array along the given axis are correlated with the
126:     given weights.
127: 
128:     Parameters
129:     ----------
130:     %(input)s
131:     weights : array
132:         One-dimensional sequence of numbers.
133:     %(axis)s
134:     %(output)s
135:     %(mode)s
136:     %(cval)s
137:     %(origin)s
138: 
139:     Examples
140:     --------
141:     >>> from scipy.ndimage import correlate1d
142:     >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
143:     array([ 8, 26,  8, 12,  7, 28, 36,  9])
144:     '''
145:     input = numpy.asarray(input)
146:     if numpy.iscomplexobj(input):
147:         raise TypeError('Complex type not supported')
148:     output, return_value = _ni_support._get_output(output, input)
149:     weights = numpy.asarray(weights, dtype=numpy.float64)
150:     if weights.ndim != 1 or weights.shape[0] < 1:
151:         raise RuntimeError('no filter weights given')
152:     if not weights.flags.contiguous:
153:         weights = weights.copy()
154:     axis = _ni_support._check_axis(axis, input.ndim)
155:     if (len(weights) // 2 + origin < 0) or (len(weights) // 2 +
156:                                             origin > len(weights)):
157:         raise ValueError('invalid origin')
158:     mode = _ni_support._extend_mode_to_code(mode)
159:     _nd_image.correlate1d(input, weights, axis, output, mode, cval,
160:                           origin)
161:     return return_value
162: 
163: 
164: @docfiller
165: def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
166:                cval=0.0, origin=0):
167:     '''Calculate a one-dimensional convolution along the given axis.
168: 
169:     The lines of the array along the given axis are convolved with the
170:     given weights.
171: 
172:     Parameters
173:     ----------
174:     %(input)s
175:     weights : ndarray
176:         One-dimensional sequence of numbers.
177:     %(axis)s
178:     %(output)s
179:     %(mode)s
180:     %(cval)s
181:     %(origin)s
182: 
183:     Returns
184:     -------
185:     convolve1d : ndarray
186:         Convolved array with same shape as input
187: 
188:     Examples
189:     --------
190:     >>> from scipy.ndimage import convolve1d
191:     >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
192:     array([14, 24,  4, 13, 12, 36, 27,  0])
193:     '''
194:     weights = weights[::-1]
195:     origin = -origin
196:     if not len(weights) & 1:
197:         origin -= 1
198:     return correlate1d(input, weights, axis, output, mode, cval, origin)
199: 
200: 
201: def _gaussian_kernel1d(sigma, order, radius):
202:     '''
203:     Computes a 1D Gaussian convolution kernel.
204:     '''
205:     if order < 0:
206:         raise ValueError('order must be non-negative')
207:     p = numpy.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
208:     x = numpy.arange(-radius, radius + 1)
209:     phi_x = numpy.exp(p(x), dtype=numpy.double)
210:     phi_x /= phi_x.sum()
211:     if order > 0:
212:         q = numpy.polynomial.Polynomial([1])
213:         p_deriv = p.deriv()
214:         for _ in range(order):
215:             # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
216:             # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
217:             q = q.deriv() + q * p_deriv
218:         phi_x *= q(x)
219:     return phi_x
220: 
221: 
222: @docfiller
223: def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
224:                       mode="reflect", cval=0.0, truncate=4.0):
225:     '''One-dimensional Gaussian filter.
226: 
227:     Parameters
228:     ----------
229:     %(input)s
230:     sigma : scalar
231:         standard deviation for Gaussian kernel
232:     %(axis)s
233:     order : int, optional
234:         An order of 0 corresponds to convolution with a Gaussian
235:         kernel. A positive order corresponds to convolution with
236:         that derivative of a Gaussian.
237:     %(output)s
238:     %(mode)s
239:     %(cval)s
240:     truncate : float, optional
241:         Truncate the filter at this many standard deviations.
242:         Default is 4.0.
243: 
244:     Returns
245:     -------
246:     gaussian_filter1d : ndarray
247: 
248:     Examples
249:     --------
250:     >>> from scipy.ndimage import gaussian_filter1d
251:     >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
252:     array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
253:     >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
254:     array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
255:     >>> import matplotlib.pyplot as plt
256:     >>> np.random.seed(280490)
257:     >>> x = np.random.randn(101).cumsum()
258:     >>> y3 = gaussian_filter1d(x, 3)
259:     >>> y6 = gaussian_filter1d(x, 6)
260:     >>> plt.plot(x, 'k', label='original data')
261:     >>> plt.plot(y3, '--', label='filtered, sigma=3')
262:     >>> plt.plot(y6, ':', label='filtered, sigma=6')
263:     >>> plt.legend()
264:     >>> plt.grid()
265:     >>> plt.show()
266:     '''
267:     sd = float(sigma)
268:     # make the radius of the filter equal to truncate standard deviations
269:     lw = int(truncate * sd + 0.5)
270:     # Since we are calling correlate, not convolve, revert the kernel
271:     weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
272:     return correlate1d(input, weights, axis, output, mode, cval, 0)
273: 
274: 
275: @docfiller
276: def gaussian_filter(input, sigma, order=0, output=None,
277:                     mode="reflect", cval=0.0, truncate=4.0):
278:     '''Multidimensional Gaussian filter.
279: 
280:     Parameters
281:     ----------
282:     %(input)s
283:     sigma : scalar or sequence of scalars
284:         Standard deviation for Gaussian kernel. The standard
285:         deviations of the Gaussian filter are given for each axis as a
286:         sequence, or as a single number, in which case it is equal for
287:         all axes.
288:     order : int or sequence of ints, optional
289:         The order of the filter along each axis is given as a sequence
290:         of integers, or as a single number.  An order of 0 corresponds
291:         to convolution with a Gaussian kernel. A positive order
292:         corresponds to convolution with that derivative of a Gaussian.
293:     %(output)s
294:     %(mode_multiple)s
295:     %(cval)s
296:     truncate : float
297:         Truncate the filter at this many standard deviations.
298:         Default is 4.0.
299: 
300:     Returns
301:     -------
302:     gaussian_filter : ndarray
303:         Returned array of same shape as `input`.
304: 
305:     Notes
306:     -----
307:     The multidimensional filter is implemented as a sequence of
308:     one-dimensional convolution filters. The intermediate arrays are
309:     stored in the same data type as the output. Therefore, for output
310:     types with a limited precision, the results may be imprecise
311:     because intermediate results may be stored with insufficient
312:     precision.
313: 
314:     Examples
315:     --------
316:     >>> from scipy.ndimage import gaussian_filter
317:     >>> a = np.arange(50, step=2).reshape((5,5))
318:     >>> a
319:     array([[ 0,  2,  4,  6,  8],
320:            [10, 12, 14, 16, 18],
321:            [20, 22, 24, 26, 28],
322:            [30, 32, 34, 36, 38],
323:            [40, 42, 44, 46, 48]])
324:     >>> gaussian_filter(a, sigma=1)
325:     array([[ 4,  6,  8,  9, 11],
326:            [10, 12, 14, 15, 17],
327:            [20, 22, 24, 25, 27],
328:            [29, 31, 33, 34, 36],
329:            [35, 37, 39, 40, 42]])
330: 
331:     >>> from scipy import misc
332:     >>> import matplotlib.pyplot as plt
333:     >>> fig = plt.figure()
334:     >>> plt.gray()  # show the filtered result in grayscale
335:     >>> ax1 = fig.add_subplot(121)  # left side
336:     >>> ax2 = fig.add_subplot(122)  # right side
337:     >>> ascent = misc.ascent()
338:     >>> result = gaussian_filter(ascent, sigma=5)
339:     >>> ax1.imshow(ascent)
340:     >>> ax2.imshow(result)
341:     >>> plt.show()
342:     '''
343:     input = numpy.asarray(input)
344:     output, return_value = _ni_support._get_output(output, input)
345:     orders = _ni_support._normalize_sequence(order, input.ndim)
346:     sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
347:     modes = _ni_support._normalize_sequence(mode, input.ndim)
348:     axes = list(range(input.ndim))
349:     axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii])
350:             for ii in range(len(axes)) if sigmas[ii] > 1e-15]
351:     if len(axes) > 0:
352:         for axis, sigma, order, mode in axes:
353:             gaussian_filter1d(input, sigma, axis, order, output,
354:                               mode, cval, truncate)
355:             input = output
356:     else:
357:         output[...] = input[...]
358:     return return_value
359: 
360: 
361: @docfiller
362: def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
363:     '''Calculate a Prewitt filter.
364: 
365:     Parameters
366:     ----------
367:     %(input)s
368:     %(axis)s
369:     %(output)s
370:     %(mode_multiple)s
371:     %(cval)s
372: 
373:     Examples
374:     --------
375:     >>> from scipy import ndimage, misc
376:     >>> import matplotlib.pyplot as plt
377:     >>> fig = plt.figure()
378:     >>> plt.gray()  # show the filtered result in grayscale
379:     >>> ax1 = fig.add_subplot(121)  # left side
380:     >>> ax2 = fig.add_subplot(122)  # right side
381:     >>> ascent = misc.ascent()
382:     >>> result = ndimage.prewitt(ascent)
383:     >>> ax1.imshow(ascent)
384:     >>> ax2.imshow(result)
385:     >>> plt.show()
386:     '''
387:     input = numpy.asarray(input)
388:     axis = _ni_support._check_axis(axis, input.ndim)
389:     output, return_value = _ni_support._get_output(output, input)
390:     modes = _ni_support._normalize_sequence(mode, input.ndim)
391:     correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
392:     axes = [ii for ii in range(input.ndim) if ii != axis]
393:     for ii in axes:
394:         correlate1d(output, [1, 1, 1], ii, output, modes[ii], cval, 0,)
395:     return return_value
396: 
397: 
398: @docfiller
399: def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
400:     '''Calculate a Sobel filter.
401: 
402:     Parameters
403:     ----------
404:     %(input)s
405:     %(axis)s
406:     %(output)s
407:     %(mode_multiple)s
408:     %(cval)s
409: 
410:     Examples
411:     --------
412:     >>> from scipy import ndimage, misc
413:     >>> import matplotlib.pyplot as plt
414:     >>> fig = plt.figure()
415:     >>> plt.gray()  # show the filtered result in grayscale
416:     >>> ax1 = fig.add_subplot(121)  # left side
417:     >>> ax2 = fig.add_subplot(122)  # right side
418:     >>> ascent = misc.ascent()
419:     >>> result = ndimage.sobel(ascent)
420:     >>> ax1.imshow(ascent)
421:     >>> ax2.imshow(result)
422:     >>> plt.show()
423:     '''
424:     input = numpy.asarray(input)
425:     axis = _ni_support._check_axis(axis, input.ndim)
426:     output, return_value = _ni_support._get_output(output, input)
427:     modes = _ni_support._normalize_sequence(mode, input.ndim)
428:     correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
429:     axes = [ii for ii in range(input.ndim) if ii != axis]
430:     for ii in axes:
431:         correlate1d(output, [1, 2, 1], ii, output, modes[ii], cval, 0)
432:     return return_value
433: 
434: 
435: @docfiller
436: def generic_laplace(input, derivative2, output=None, mode="reflect",
437:                     cval=0.0,
438:                     extra_arguments=(),
439:                     extra_keywords=None):
440:     '''
441:     N-dimensional Laplace filter using a provided second derivative function.
442: 
443:     Parameters
444:     ----------
445:     %(input)s
446:     derivative2 : callable
447:         Callable with the following signature::
448: 
449:             derivative2(input, axis, output, mode, cval,
450:                         *extra_arguments, **extra_keywords)
451: 
452:         See `extra_arguments`, `extra_keywords` below.
453:     %(output)s
454:     %(mode_multiple)s
455:     %(cval)s
456:     %(extra_keywords)s
457:     %(extra_arguments)s
458:     '''
459:     if extra_keywords is None:
460:         extra_keywords = {}
461:     input = numpy.asarray(input)
462:     output, return_value = _ni_support._get_output(output, input)
463:     axes = list(range(input.ndim))
464:     if len(axes) > 0:
465:         modes = _ni_support._normalize_sequence(mode, len(axes))
466:         derivative2(input, axes[0], output, modes[0], cval,
467:                     *extra_arguments, **extra_keywords)
468:         for ii in range(1, len(axes)):
469:             tmp = derivative2(input, axes[ii], output.dtype, modes[ii], cval,
470:                               *extra_arguments, **extra_keywords)
471:             output += tmp
472:     else:
473:         output[...] = input[...]
474:     return return_value
475: 
476: 
477: @docfiller
478: def laplace(input, output=None, mode="reflect", cval=0.0):
479:     '''N-dimensional Laplace filter based on approximate second derivatives.
480: 
481:     Parameters
482:     ----------
483:     %(input)s
484:     %(output)s
485:     %(mode_multiple)s
486:     %(cval)s
487: 
488:     Examples
489:     --------
490:     >>> from scipy import ndimage, misc
491:     >>> import matplotlib.pyplot as plt
492:     >>> fig = plt.figure()
493:     >>> plt.gray()  # show the filtered result in grayscale
494:     >>> ax1 = fig.add_subplot(121)  # left side
495:     >>> ax2 = fig.add_subplot(122)  # right side
496:     >>> ascent = misc.ascent()
497:     >>> result = ndimage.laplace(ascent)
498:     >>> ax1.imshow(ascent)
499:     >>> ax2.imshow(result)
500:     >>> plt.show()
501:     '''
502:     def derivative2(input, axis, output, mode, cval):
503:         return correlate1d(input, [1, -2, 1], axis, output, mode, cval, 0)
504:     return generic_laplace(input, derivative2, output, mode, cval)
505: 
506: 
507: @docfiller
508: def gaussian_laplace(input, sigma, output=None, mode="reflect",
509:                      cval=0.0, **kwargs):
510:     '''Multidimensional Laplace filter using gaussian second derivatives.
511: 
512:     Parameters
513:     ----------
514:     %(input)s
515:     sigma : scalar or sequence of scalars
516:         The standard deviations of the Gaussian filter are given for
517:         each axis as a sequence, or as a single number, in which case
518:         it is equal for all axes.
519:     %(output)s
520:     %(mode_multiple)s
521:     %(cval)s
522:     Extra keyword arguments will be passed to gaussian_filter().
523: 
524:     Examples
525:     --------
526:     >>> from scipy import ndimage, misc
527:     >>> import matplotlib.pyplot as plt
528:     >>> ascent = misc.ascent()
529: 
530:     >>> fig = plt.figure()
531:     >>> plt.gray()  # show the filtered result in grayscale
532:     >>> ax1 = fig.add_subplot(121)  # left side
533:     >>> ax2 = fig.add_subplot(122)  # right side
534: 
535:     >>> result = ndimage.gaussian_laplace(ascent, sigma=1)
536:     >>> ax1.imshow(result)
537: 
538:     >>> result = ndimage.gaussian_laplace(ascent, sigma=3)
539:     >>> ax2.imshow(result)
540:     >>> plt.show()
541:     '''
542:     input = numpy.asarray(input)
543: 
544:     def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
545:         order = [0] * input.ndim
546:         order[axis] = 2
547:         return gaussian_filter(input, sigma, order, output, mode, cval,
548:                                **kwargs)
549: 
550:     return generic_laplace(input, derivative2, output, mode, cval,
551:                            extra_arguments=(sigma,),
552:                            extra_keywords=kwargs)
553: 
554: 
555: @docfiller
556: def generic_gradient_magnitude(input, derivative, output=None,
557:                                mode="reflect", cval=0.0,
558:                                extra_arguments=(), extra_keywords=None):
559:     '''Gradient magnitude using a provided gradient function.
560: 
561:     Parameters
562:     ----------
563:     %(input)s
564:     derivative : callable
565:         Callable with the following signature::
566: 
567:             derivative(input, axis, output, mode, cval,
568:                        *extra_arguments, **extra_keywords)
569: 
570:         See `extra_arguments`, `extra_keywords` below.
571:         `derivative` can assume that `input` and `output` are ndarrays.
572:         Note that the output from `derivative` is modified inplace;
573:         be careful to copy important inputs before returning them.
574:     %(output)s
575:     %(mode_multiple)s
576:     %(cval)s
577:     %(extra_keywords)s
578:     %(extra_arguments)s
579:     '''
580:     if extra_keywords is None:
581:         extra_keywords = {}
582:     input = numpy.asarray(input)
583:     output, return_value = _ni_support._get_output(output, input)
584:     axes = list(range(input.ndim))
585:     if len(axes) > 0:
586:         modes = _ni_support._normalize_sequence(mode, len(axes))
587:         derivative(input, axes[0], output, modes[0], cval,
588:                    *extra_arguments, **extra_keywords)
589:         numpy.multiply(output, output, output)
590:         for ii in range(1, len(axes)):
591:             tmp = derivative(input, axes[ii], output.dtype, modes[ii], cval,
592:                              *extra_arguments, **extra_keywords)
593:             numpy.multiply(tmp, tmp, tmp)
594:             output += tmp
595:         # This allows the sqrt to work with a different default casting
596:         numpy.sqrt(output, output, casting='unsafe')
597:     else:
598:         output[...] = input[...]
599:     return return_value
600: 
601: 
602: @docfiller
603: def gaussian_gradient_magnitude(input, sigma, output=None,
604:                                 mode="reflect", cval=0.0, **kwargs):
605:     '''Multidimensional gradient magnitude using Gaussian derivatives.
606: 
607:     Parameters
608:     ----------
609:     %(input)s
610:     sigma : scalar or sequence of scalars
611:         The standard deviations of the Gaussian filter are given for
612:         each axis as a sequence, or as a single number, in which case
613:         it is equal for all axes..
614:     %(output)s
615:     %(mode_multiple)s
616:     %(cval)s
617:     Extra keyword arguments will be passed to gaussian_filter().
618: 
619:     Returns
620:     -------
621:     gaussian_gradient_magnitude : ndarray
622:         Filtered array. Has the same shape as `input`.
623: 
624:     Examples
625:     --------
626:     >>> from scipy import ndimage, misc
627:     >>> import matplotlib.pyplot as plt
628:     >>> fig = plt.figure()
629:     >>> plt.gray()  # show the filtered result in grayscale
630:     >>> ax1 = fig.add_subplot(121)  # left side
631:     >>> ax2 = fig.add_subplot(122)  # right side
632:     >>> ascent = misc.ascent()
633:     >>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)
634:     >>> ax1.imshow(ascent)
635:     >>> ax2.imshow(result)
636:     >>> plt.show()
637:     '''
638:     input = numpy.asarray(input)
639: 
640:     def derivative(input, axis, output, mode, cval, sigma, **kwargs):
641:         order = [0] * input.ndim
642:         order[axis] = 1
643:         return gaussian_filter(input, sigma, order, output, mode,
644:                                cval, **kwargs)
645: 
646:     return generic_gradient_magnitude(input, derivative, output, mode,
647:                                       cval, extra_arguments=(sigma,),
648:                                       extra_keywords=kwargs)
649: 
650: 
651: def _correlate_or_convolve(input, weights, output, mode, cval, origin,
652:                            convolution):
653:     input = numpy.asarray(input)
654:     if numpy.iscomplexobj(input):
655:         raise TypeError('Complex type not supported')
656:     origins = _ni_support._normalize_sequence(origin, input.ndim)
657:     weights = numpy.asarray(weights, dtype=numpy.float64)
658:     wshape = [ii for ii in weights.shape if ii > 0]
659:     if len(wshape) != input.ndim:
660:         raise RuntimeError('filter weights array has incorrect shape.')
661:     if convolution:
662:         weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
663:         for ii in range(len(origins)):
664:             origins[ii] = -origins[ii]
665:             if not weights.shape[ii] & 1:
666:                 origins[ii] -= 1
667:     for origin, lenw in zip(origins, wshape):
668:         if (lenw // 2 + origin < 0) or (lenw // 2 + origin > lenw):
669:             raise ValueError('invalid origin')
670:     if not weights.flags.contiguous:
671:         weights = weights.copy()
672:     output, return_value = _ni_support._get_output(output, input)
673:     mode = _ni_support._extend_mode_to_code(mode)
674:     _nd_image.correlate(input, weights, output, mode, cval, origins)
675:     return return_value
676: 
677: 
678: @docfiller
679: def correlate(input, weights, output=None, mode='reflect', cval=0.0,
680:               origin=0):
681:     '''
682:     Multi-dimensional correlation.
683: 
684:     The array is correlated with the given kernel.
685: 
686:     Parameters
687:     ----------
688:     input : array-like
689:         input array to filter
690:     weights : ndarray
691:         array of weights, same number of dimensions as input
692:     output : array, optional
693:         The ``output`` parameter passes an array in which to store the
694:         filter output. Output array should have different name as
695:         compared to input array to avoid aliasing errors.
696:     mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
697:         The ``mode`` parameter determines how the array borders are
698:         handled, where ``cval`` is the value when mode is equal to
699:         'constant'. Default is 'reflect'
700:     cval : scalar, optional
701:         Value to fill past edges of input if ``mode`` is 'constant'. Default
702:         is 0.0
703:     origin : scalar, optional
704:         The ``origin`` parameter controls the placement of the filter.
705:         Default 0
706: 
707:     See Also
708:     --------
709:     convolve : Convolve an image with a kernel.
710:     '''
711:     return _correlate_or_convolve(input, weights, output, mode, cval,
712:                                   origin, False)
713: 
714: 
715: @docfiller
716: def convolve(input, weights, output=None, mode='reflect', cval=0.0,
717:              origin=0):
718:     '''
719:     Multidimensional convolution.
720: 
721:     The array is convolved with the given kernel.
722: 
723:     Parameters
724:     ----------
725:     input : array_like
726:         Input array to filter.
727:     weights : array_like
728:         Array of weights, same number of dimensions as input
729:     output : ndarray, optional
730:         The `output` parameter passes an array in which to store the
731:         filter output. Output array should have different name as
732:         compared to input array to avoid aliasing errors.
733:     mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
734:         the `mode` parameter determines how the array borders are
735:         handled. For 'constant' mode, values beyond borders are set to be
736:         `cval`. Default is 'reflect'.
737:     cval : scalar, optional
738:         Value to fill past edges of input if `mode` is 'constant'. Default
739:         is 0.0
740:     origin : array_like, optional
741:         The `origin` parameter controls the placement of the filter,
742:         relative to the centre of the current element of the input.
743:         Default of 0 is equivalent to ``(0,)*input.ndim``.
744: 
745:     Returns
746:     -------
747:     result : ndarray
748:         The result of convolution of `input` with `weights`.
749: 
750:     See Also
751:     --------
752:     correlate : Correlate an image with a kernel.
753: 
754:     Notes
755:     -----
756:     Each value in result is :math:`C_i = \\sum_j{I_{i+k-j} W_j}`, where
757:     W is the `weights` kernel,
758:     j is the n-D spatial index over :math:`W`,
759:     I is the `input` and k is the coordinate of the center of
760:     W, specified by `origin` in the input parameters.
761: 
762:     Examples
763:     --------
764:     Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,
765:     because in this case borders (i.e. where the `weights` kernel, centered
766:     on any one value, extends beyond an edge of `input`.
767: 
768:     >>> a = np.array([[1, 2, 0, 0],
769:     ...               [5, 3, 0, 4],
770:     ...               [0, 0, 0, 7],
771:     ...               [9, 3, 0, 0]])
772:     >>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])
773:     >>> from scipy import ndimage
774:     >>> ndimage.convolve(a, k, mode='constant', cval=0.0)
775:     array([[11, 10,  7,  4],
776:            [10,  3, 11, 11],
777:            [15, 12, 14,  7],
778:            [12,  3,  7,  0]])
779: 
780:     Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`
781:     with 1.0's (and then extracting only the original region of the result).
782: 
783:     >>> ndimage.convolve(a, k, mode='constant', cval=1.0)
784:     array([[13, 11,  8,  7],
785:            [11,  3, 11, 14],
786:            [16, 12, 14, 10],
787:            [15,  6, 10,  5]])
788: 
789:     With ``mode='reflect'`` (the default), outer values are reflected at the
790:     edge of `input` to fill in missing values.
791: 
792:     >>> b = np.array([[2, 0, 0],
793:     ...               [1, 0, 0],
794:     ...               [0, 0, 0]])
795:     >>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])
796:     >>> ndimage.convolve(b, k, mode='reflect')
797:     array([[5, 0, 0],
798:            [3, 0, 0],
799:            [1, 0, 0]])
800: 
801:     This includes diagonally at the corners.
802: 
803:     >>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])
804:     >>> ndimage.convolve(b, k)
805:     array([[4, 2, 0],
806:            [3, 2, 0],
807:            [1, 1, 0]])
808: 
809:     With ``mode='nearest'``, the single nearest value in to an edge in
810:     `input` is repeated as many times as needed to match the overlapping
811:     `weights`.
812: 
813:     >>> c = np.array([[2, 0, 1],
814:     ...               [1, 0, 0],
815:     ...               [0, 0, 0]])
816:     >>> k = np.array([[0, 1, 0],
817:     ...               [0, 1, 0],
818:     ...               [0, 1, 0],
819:     ...               [0, 1, 0],
820:     ...               [0, 1, 0]])
821:     >>> ndimage.convolve(c, k, mode='nearest')
822:     array([[7, 0, 3],
823:            [5, 0, 2],
824:            [3, 0, 1]])
825: 
826:     '''
827:     return _correlate_or_convolve(input, weights, output, mode, cval,
828:                                   origin, True)
829: 
830: 
831: @docfiller
832: def uniform_filter1d(input, size, axis=-1, output=None,
833:                      mode="reflect", cval=0.0, origin=0):
834:     '''Calculate a one-dimensional uniform filter along the given axis.
835: 
836:     The lines of the array along the given axis are filtered with a
837:     uniform filter of given size.
838: 
839:     Parameters
840:     ----------
841:     %(input)s
842:     size : int
843:         length of uniform filter
844:     %(axis)s
845:     %(output)s
846:     %(mode)s
847:     %(cval)s
848:     %(origin)s
849: 
850:     Examples
851:     --------
852:     >>> from scipy.ndimage import uniform_filter1d
853:     >>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
854:     array([4, 3, 4, 1, 4, 6, 6, 3])
855:     '''
856:     input = numpy.asarray(input)
857:     if numpy.iscomplexobj(input):
858:         raise TypeError('Complex type not supported')
859:     axis = _ni_support._check_axis(axis, input.ndim)
860:     if size < 1:
861:         raise RuntimeError('incorrect filter size')
862:     output, return_value = _ni_support._get_output(output, input)
863:     if (size // 2 + origin < 0) or (size // 2 + origin >= size):
864:         raise ValueError('invalid origin')
865:     mode = _ni_support._extend_mode_to_code(mode)
866:     _nd_image.uniform_filter1d(input, size, axis, output, mode, cval,
867:                                origin)
868:     return return_value
869: 
870: 
871: @docfiller
872: def uniform_filter(input, size=3, output=None, mode="reflect",
873:                    cval=0.0, origin=0):
874:     '''Multi-dimensional uniform filter.
875: 
876:     Parameters
877:     ----------
878:     %(input)s
879:     size : int or sequence of ints, optional
880:         The sizes of the uniform filter are given for each axis as a
881:         sequence, or as a single number, in which case the size is
882:         equal for all axes.
883:     %(output)s
884:     %(mode_multiple)s
885:     %(cval)s
886:     %(origin)s
887: 
888:     Returns
889:     -------
890:     uniform_filter : ndarray
891:         Filtered array. Has the same shape as `input`.
892: 
893:     Notes
894:     -----
895:     The multi-dimensional filter is implemented as a sequence of
896:     one-dimensional uniform filters. The intermediate arrays are stored
897:     in the same data type as the output. Therefore, for output types
898:     with a limited precision, the results may be imprecise because
899:     intermediate results may be stored with insufficient precision.
900: 
901:     Examples
902:     --------
903:     >>> from scipy import ndimage, misc
904:     >>> import matplotlib.pyplot as plt
905:     >>> fig = plt.figure()
906:     >>> plt.gray()  # show the filtered result in grayscale
907:     >>> ax1 = fig.add_subplot(121)  # left side
908:     >>> ax2 = fig.add_subplot(122)  # right side
909:     >>> ascent = misc.ascent()
910:     >>> result = ndimage.uniform_filter(ascent, size=20)
911:     >>> ax1.imshow(ascent)
912:     >>> ax2.imshow(result)
913:     >>> plt.show()
914:     '''
915:     input = numpy.asarray(input)
916:     output, return_value = _ni_support._get_output(output, input)
917:     sizes = _ni_support._normalize_sequence(size, input.ndim)
918:     origins = _ni_support._normalize_sequence(origin, input.ndim)
919:     modes = _ni_support._normalize_sequence(mode, input.ndim)
920:     axes = list(range(input.ndim))
921:     axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
922:             for ii in range(len(axes)) if sizes[ii] > 1]
923:     if len(axes) > 0:
924:         for axis, size, origin, mode in axes:
925:             uniform_filter1d(input, int(size), axis, output, mode,
926:                              cval, origin)
927:             input = output
928:     else:
929:         output[...] = input[...]
930:     return return_value
931: 
932: 
933: @docfiller
934: def minimum_filter1d(input, size, axis=-1, output=None,
935:                      mode="reflect", cval=0.0, origin=0):
936:     '''Calculate a one-dimensional minimum filter along the given axis.
937: 
938:     The lines of the array along the given axis are filtered with a
939:     minimum filter of given size.
940: 
941:     Parameters
942:     ----------
943:     %(input)s
944:     size : int
945:         length along which to calculate 1D minimum
946:     %(axis)s
947:     %(output)s
948:     %(mode)s
949:     %(cval)s
950:     %(origin)s
951: 
952:     Notes
953:     -----
954:     This function implements the MINLIST algorithm [1]_, as described by
955:     Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
956:     the `input` length, regardless of filter size.
957: 
958:     References
959:     ----------
960:     .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
961:     .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html
962: 
963: 
964:     Examples
965:     --------
966:     >>> from scipy.ndimage import minimum_filter1d
967:     >>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
968:     array([2, 0, 0, 0, 1, 1, 0, 0])
969:     '''
970:     input = numpy.asarray(input)
971:     if numpy.iscomplexobj(input):
972:         raise TypeError('Complex type not supported')
973:     axis = _ni_support._check_axis(axis, input.ndim)
974:     if size < 1:
975:         raise RuntimeError('incorrect filter size')
976:     output, return_value = _ni_support._get_output(output, input)
977:     if (size // 2 + origin < 0) or (size // 2 + origin >= size):
978:         raise ValueError('invalid origin')
979:     mode = _ni_support._extend_mode_to_code(mode)
980:     _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
981:                                   origin, 1)
982:     return return_value
983: 
984: 
985: @docfiller
986: def maximum_filter1d(input, size, axis=-1, output=None,
987:                      mode="reflect", cval=0.0, origin=0):
988:     '''Calculate a one-dimensional maximum filter along the given axis.
989: 
990:     The lines of the array along the given axis are filtered with a
991:     maximum filter of given size.
992: 
993:     Parameters
994:     ----------
995:     %(input)s
996:     size : int
997:         Length along which to calculate the 1-D maximum.
998:     %(axis)s
999:     %(output)s
1000:     %(mode)s
1001:     %(cval)s
1002:     %(origin)s
1003: 
1004:     Returns
1005:     -------
1006:     maximum1d : ndarray, None
1007:         Maximum-filtered array with same shape as input.
1008:         None if `output` is not None
1009: 
1010:     Notes
1011:     -----
1012:     This function implements the MAXLIST algorithm [1]_, as described by
1013:     Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being
1014:     the `input` length, regardless of filter size.
1015: 
1016:     References
1017:     ----------
1018:     .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777
1019:     .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html
1020: 
1021:     Examples
1022:     --------
1023:     >>> from scipy.ndimage import maximum_filter1d
1024:     >>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)
1025:     array([8, 8, 8, 4, 9, 9, 9, 9])
1026:     '''
1027:     input = numpy.asarray(input)
1028:     if numpy.iscomplexobj(input):
1029:         raise TypeError('Complex type not supported')
1030:     axis = _ni_support._check_axis(axis, input.ndim)
1031:     if size < 1:
1032:         raise RuntimeError('incorrect filter size')
1033:     output, return_value = _ni_support._get_output(output, input)
1034:     if (size // 2 + origin < 0) or (size // 2 + origin >= size):
1035:         raise ValueError('invalid origin')
1036:     mode = _ni_support._extend_mode_to_code(mode)
1037:     _nd_image.min_or_max_filter1d(input, size, axis, output, mode, cval,
1038:                                   origin, 0)
1039:     return return_value
1040: 
1041: 
1042: def _min_or_max_filter(input, size, footprint, structure, output, mode,
1043:                        cval, origin, minimum):
1044:     if structure is None:
1045:         if footprint is None:
1046:             if size is None:
1047:                 raise RuntimeError("no footprint provided")
1048:             separable = True
1049:         else:
1050:             footprint = numpy.asarray(footprint, dtype=bool)
1051:             if not footprint.any():
1052:                 raise ValueError("All-zero footprint is not supported.")
1053:             if footprint.all():
1054:                 size = footprint.shape
1055:                 footprint = None
1056:                 separable = True
1057:             else:
1058:                 separable = False
1059:     else:
1060:         structure = numpy.asarray(structure, dtype=numpy.float64)
1061:         separable = False
1062:         if footprint is None:
1063:             footprint = numpy.ones(structure.shape, bool)
1064:         else:
1065:             footprint = numpy.asarray(footprint, dtype=bool)
1066:     input = numpy.asarray(input)
1067:     if numpy.iscomplexobj(input):
1068:         raise TypeError('Complex type not supported')
1069:     output, return_value = _ni_support._get_output(output, input)
1070:     origins = _ni_support._normalize_sequence(origin, input.ndim)
1071:     if separable:
1072:         sizes = _ni_support._normalize_sequence(size, input.ndim)
1073:         modes = _ni_support._normalize_sequence(mode, input.ndim)
1074:         axes = list(range(input.ndim))
1075:         axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
1076:                 for ii in range(len(axes)) if sizes[ii] > 1]
1077:         if minimum:
1078:             filter_ = minimum_filter1d
1079:         else:
1080:             filter_ = maximum_filter1d
1081:         if len(axes) > 0:
1082:             for axis, size, origin, mode in axes:
1083:                 filter_(input, int(size), axis, output, mode, cval, origin)
1084:                 input = output
1085:         else:
1086:             output[...] = input[...]
1087:     else:
1088:         fshape = [ii for ii in footprint.shape if ii > 0]
1089:         if len(fshape) != input.ndim:
1090:             raise RuntimeError('footprint array has incorrect shape.')
1091:         for origin, lenf in zip(origins, fshape):
1092:             if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
1093:                 raise ValueError('invalid origin')
1094:         if not footprint.flags.contiguous:
1095:             footprint = footprint.copy()
1096:         if structure is not None:
1097:             if len(structure.shape) != input.ndim:
1098:                 raise RuntimeError('structure array has incorrect shape')
1099:             if not structure.flags.contiguous:
1100:                 structure = structure.copy()
1101:         mode = _ni_support._extend_mode_to_code(mode)
1102:         _nd_image.min_or_max_filter(input, footprint, structure, output,
1103:                                     mode, cval, origins, minimum)
1104:     return return_value
1105: 
1106: 
1107: @docfiller
1108: def minimum_filter(input, size=None, footprint=None, output=None,
1109:                    mode="reflect", cval=0.0, origin=0):
1110:     '''Calculate a multi-dimensional minimum filter.
1111: 
1112:     Parameters
1113:     ----------
1114:     %(input)s
1115:     %(size_foot)s
1116:     %(output)s
1117:     %(mode_multiple)s
1118:     %(cval)s
1119:     %(origin)s
1120: 
1121:     Returns
1122:     -------
1123:     minimum_filter : ndarray
1124:         Filtered array. Has the same shape as `input`.
1125: 
1126:     Examples
1127:     --------
1128:     >>> from scipy import ndimage, misc
1129:     >>> import matplotlib.pyplot as plt
1130:     >>> fig = plt.figure()
1131:     >>> plt.gray()  # show the filtered result in grayscale
1132:     >>> ax1 = fig.add_subplot(121)  # left side
1133:     >>> ax2 = fig.add_subplot(122)  # right side
1134:     >>> ascent = misc.ascent()
1135:     >>> result = ndimage.minimum_filter(ascent, size=20)
1136:     >>> ax1.imshow(ascent)
1137:     >>> ax2.imshow(result)
1138:     >>> plt.show()
1139:     '''
1140:     return _min_or_max_filter(input, size, footprint, None, output, mode,
1141:                               cval, origin, 1)
1142: 
1143: 
1144: @docfiller
1145: def maximum_filter(input, size=None, footprint=None, output=None,
1146:                    mode="reflect", cval=0.0, origin=0):
1147:     '''Calculate a multi-dimensional maximum filter.
1148: 
1149:     Parameters
1150:     ----------
1151:     %(input)s
1152:     %(size_foot)s
1153:     %(output)s
1154:     %(mode_multiple)s
1155:     %(cval)s
1156:     %(origin)s
1157: 
1158:     Returns
1159:     -------
1160:     maximum_filter : ndarray
1161:         Filtered array. Has the same shape as `input`.
1162: 
1163:     Examples
1164:     --------
1165:     >>> from scipy import ndimage, misc
1166:     >>> import matplotlib.pyplot as plt
1167:     >>> fig = plt.figure()
1168:     >>> plt.gray()  # show the filtered result in grayscale
1169:     >>> ax1 = fig.add_subplot(121)  # left side
1170:     >>> ax2 = fig.add_subplot(122)  # right side
1171:     >>> ascent = misc.ascent()
1172:     >>> result = ndimage.maximum_filter(ascent, size=20)
1173:     >>> ax1.imshow(ascent)
1174:     >>> ax2.imshow(result)
1175:     >>> plt.show()
1176:     '''
1177:     return _min_or_max_filter(input, size, footprint, None, output, mode,
1178:                               cval, origin, 0)
1179: 
1180: 
1181: @docfiller
1182: def _rank_filter(input, rank, size=None, footprint=None, output=None,
1183:                  mode="reflect", cval=0.0, origin=0, operation='rank'):
1184:     input = numpy.asarray(input)
1185:     if numpy.iscomplexobj(input):
1186:         raise TypeError('Complex type not supported')
1187:     origins = _ni_support._normalize_sequence(origin, input.ndim)
1188:     if footprint is None:
1189:         if size is None:
1190:             raise RuntimeError("no footprint or filter size provided")
1191:         sizes = _ni_support._normalize_sequence(size, input.ndim)
1192:         footprint = numpy.ones(sizes, dtype=bool)
1193:     else:
1194:         footprint = numpy.asarray(footprint, dtype=bool)
1195:     fshape = [ii for ii in footprint.shape if ii > 0]
1196:     if len(fshape) != input.ndim:
1197:         raise RuntimeError('filter footprint array has incorrect shape.')
1198:     for origin, lenf in zip(origins, fshape):
1199:         if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
1200:             raise ValueError('invalid origin')
1201:     if not footprint.flags.contiguous:
1202:         footprint = footprint.copy()
1203:     filter_size = numpy.where(footprint, 1, 0).sum()
1204:     if operation == 'median':
1205:         rank = filter_size // 2
1206:     elif operation == 'percentile':
1207:         percentile = rank
1208:         if percentile < 0.0:
1209:             percentile += 100.0
1210:         if percentile < 0 or percentile > 100:
1211:             raise RuntimeError('invalid percentile')
1212:         if percentile == 100.0:
1213:             rank = filter_size - 1
1214:         else:
1215:             rank = int(float(filter_size) * percentile / 100.0)
1216:     if rank < 0:
1217:         rank += filter_size
1218:     if rank < 0 or rank >= filter_size:
1219:         raise RuntimeError('rank not within filter footprint size')
1220:     if rank == 0:
1221:         return minimum_filter(input, None, footprint, output, mode, cval,
1222:                               origins)
1223:     elif rank == filter_size - 1:
1224:         return maximum_filter(input, None, footprint, output, mode, cval,
1225:                               origins)
1226:     else:
1227:         output, return_value = _ni_support._get_output(output, input)
1228:         mode = _ni_support._extend_mode_to_code(mode)
1229:         _nd_image.rank_filter(input, rank, footprint, output, mode, cval,
1230:                               origins)
1231:         return return_value
1232: 
1233: 
1234: @docfiller
1235: def rank_filter(input, rank, size=None, footprint=None, output=None,
1236:                 mode="reflect", cval=0.0, origin=0):
1237:     '''Calculate a multi-dimensional rank filter.
1238: 
1239:     Parameters
1240:     ----------
1241:     %(input)s
1242:     rank : int
1243:         The rank parameter may be less then zero, i.e., rank = -1
1244:         indicates the largest element.
1245:     %(size_foot)s
1246:     %(output)s
1247:     %(mode)s
1248:     %(cval)s
1249:     %(origin)s
1250: 
1251:     Returns
1252:     -------
1253:     rank_filter : ndarray
1254:         Filtered array. Has the same shape as `input`.
1255: 
1256:     Examples
1257:     --------
1258:     >>> from scipy import ndimage, misc
1259:     >>> import matplotlib.pyplot as plt
1260:     >>> fig = plt.figure()
1261:     >>> plt.gray()  # show the filtered result in grayscale
1262:     >>> ax1 = fig.add_subplot(121)  # left side
1263:     >>> ax2 = fig.add_subplot(122)  # right side
1264:     >>> ascent = misc.ascent()
1265:     >>> result = ndimage.rank_filter(ascent, rank=42, size=20)
1266:     >>> ax1.imshow(ascent)
1267:     >>> ax2.imshow(result)
1268:     >>> plt.show()
1269:     '''
1270:     return _rank_filter(input, rank, size, footprint, output, mode, cval,
1271:                         origin, 'rank')
1272: 
1273: 
1274: @docfiller
1275: def median_filter(input, size=None, footprint=None, output=None,
1276:                   mode="reflect", cval=0.0, origin=0):
1277:     '''
1278:     Calculate a multidimensional median filter.
1279: 
1280:     Parameters
1281:     ----------
1282:     %(input)s
1283:     %(size_foot)s
1284:     %(output)s
1285:     %(mode)s
1286:     %(cval)s
1287:     %(origin)s
1288: 
1289:     Returns
1290:     -------
1291:     median_filter : ndarray
1292:         Filtered array. Has the same shape as `input`.
1293: 
1294:     Examples
1295:     --------
1296:     >>> from scipy import ndimage, misc
1297:     >>> import matplotlib.pyplot as plt
1298:     >>> fig = plt.figure()
1299:     >>> plt.gray()  # show the filtered result in grayscale
1300:     >>> ax1 = fig.add_subplot(121)  # left side
1301:     >>> ax2 = fig.add_subplot(122)  # right side
1302:     >>> ascent = misc.ascent()
1303:     >>> result = ndimage.median_filter(ascent, size=20)
1304:     >>> ax1.imshow(ascent)
1305:     >>> ax2.imshow(result)
1306:     >>> plt.show()
1307:     '''
1308:     return _rank_filter(input, 0, size, footprint, output, mode, cval,
1309:                         origin, 'median')
1310: 
1311: 
1312: @docfiller
1313: def percentile_filter(input, percentile, size=None, footprint=None,
1314:                       output=None, mode="reflect", cval=0.0, origin=0):
1315:     '''Calculate a multi-dimensional percentile filter.
1316: 
1317:     Parameters
1318:     ----------
1319:     %(input)s
1320:     percentile : scalar
1321:         The percentile parameter may be less then zero, i.e.,
1322:         percentile = -20 equals percentile = 80
1323:     %(size_foot)s
1324:     %(output)s
1325:     %(mode)s
1326:     %(cval)s
1327:     %(origin)s
1328: 
1329:     Returns
1330:     -------
1331:     percentile_filter : ndarray
1332:         Filtered array. Has the same shape as `input`.
1333: 
1334:     Examples
1335:     --------
1336:     >>> from scipy import ndimage, misc
1337:     >>> import matplotlib.pyplot as plt
1338:     >>> fig = plt.figure()
1339:     >>> plt.gray()  # show the filtered result in grayscale
1340:     >>> ax1 = fig.add_subplot(121)  # left side
1341:     >>> ax2 = fig.add_subplot(122)  # right side
1342:     >>> ascent = misc.ascent()
1343:     >>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)
1344:     >>> ax1.imshow(ascent)
1345:     >>> ax2.imshow(result)
1346:     >>> plt.show()
1347:     '''
1348:     return _rank_filter(input, percentile, size, footprint, output, mode,
1349:                         cval, origin, 'percentile')
1350: 
1351: 
1352: @docfiller
1353: def generic_filter1d(input, function, filter_size, axis=-1,
1354:                      output=None, mode="reflect", cval=0.0, origin=0,
1355:                      extra_arguments=(), extra_keywords=None):
1356:     '''Calculate a one-dimensional filter along the given axis.
1357: 
1358:     `generic_filter1d` iterates over the lines of the array, calling the
1359:     given function at each line. The arguments of the line are the
1360:     input line, and the output line. The input and output lines are 1D
1361:     double arrays.  The input line is extended appropriately according
1362:     to the filter size and origin. The output line must be modified
1363:     in-place with the result.
1364: 
1365:     Parameters
1366:     ----------
1367:     %(input)s
1368:     function : {callable, scipy.LowLevelCallable}
1369:         Function to apply along given axis.
1370:     filter_size : scalar
1371:         Length of the filter.
1372:     %(axis)s
1373:     %(output)s
1374:     %(mode)s
1375:     %(cval)s
1376:     %(origin)s
1377:     %(extra_arguments)s
1378:     %(extra_keywords)s
1379: 
1380:     Notes
1381:     -----
1382:     This function also accepts low-level callback functions with one of
1383:     the following signatures and wrapped in `scipy.LowLevelCallable`:
1384: 
1385:     .. code:: c
1386: 
1387:        int function(double *input_line, npy_intp input_length,
1388:                     double *output_line, npy_intp output_length,
1389:                     void *user_data)
1390:        int function(double *input_line, intptr_t input_length,
1391:                     double *output_line, intptr_t output_length,
1392:                     void *user_data)
1393: 
1394:     The calling function iterates over the lines of the input and output
1395:     arrays, calling the callback function at each line. The current line
1396:     is extended according to the border conditions set by the calling
1397:     function, and the result is copied into the array that is passed
1398:     through ``input_line``. The length of the input line (after extension)
1399:     is passed through ``input_length``. The callback function should apply
1400:     the filter and store the result in the array passed through
1401:     ``output_line``. The length of the output line is passed through
1402:     ``output_length``. ``user_data`` is the data pointer provided
1403:     to `scipy.LowLevelCallable` as-is.
1404: 
1405:     The callback function must return an integer error status that is zero
1406:     if something went wrong and one otherwise. If an error occurs, you should
1407:     normally set the python error status with an informative message
1408:     before returning, otherwise a default error message is set by the
1409:     calling function.
1410: 
1411:     In addition, some other low-level function pointer specifications
1412:     are accepted, but these are for backward compatibility only and should
1413:     not be used in new code.
1414: 
1415:     '''
1416:     if extra_keywords is None:
1417:         extra_keywords = {}
1418:     input = numpy.asarray(input)
1419:     if numpy.iscomplexobj(input):
1420:         raise TypeError('Complex type not supported')
1421:     output, return_value = _ni_support._get_output(output, input)
1422:     if filter_size < 1:
1423:         raise RuntimeError('invalid filter size')
1424:     axis = _ni_support._check_axis(axis, input.ndim)
1425:     if (filter_size // 2 + origin < 0) or (filter_size // 2 + origin >=
1426:                                            filter_size):
1427:         raise ValueError('invalid origin')
1428:     mode = _ni_support._extend_mode_to_code(mode)
1429:     _nd_image.generic_filter1d(input, function, filter_size, axis, output,
1430:                                mode, cval, origin, extra_arguments,
1431:                                extra_keywords)
1432:     return return_value
1433: 
1434: 
1435: @docfiller
1436: def generic_filter(input, function, size=None, footprint=None,
1437:                    output=None, mode="reflect", cval=0.0, origin=0,
1438:                    extra_arguments=(), extra_keywords=None):
1439:     '''Calculate a multi-dimensional filter using the given function.
1440: 
1441:     At each element the provided function is called. The input values
1442:     within the filter footprint at that element are passed to the function
1443:     as a 1D array of double values.
1444: 
1445:     Parameters
1446:     ----------
1447:     %(input)s
1448:     function : {callable, scipy.LowLevelCallable}
1449:         Function to apply at each element.
1450:     %(size_foot)s
1451:     %(output)s
1452:     %(mode)s
1453:     %(cval)s
1454:     %(origin)s
1455:     %(extra_arguments)s
1456:     %(extra_keywords)s
1457: 
1458:     Notes
1459:     -----
1460:     This function also accepts low-level callback functions with one of
1461:     the following signatures and wrapped in `scipy.LowLevelCallable`:
1462: 
1463:     .. code:: c
1464: 
1465:        int callback(double *buffer, npy_intp filter_size,
1466:                     double *return_value, void *user_data)
1467:        int callback(double *buffer, intptr_t filter_size,
1468:                     double *return_value, void *user_data)
1469: 
1470:     The calling function iterates over the elements of the input and
1471:     output arrays, calling the callback function at each element. The
1472:     elements within the footprint of the filter at the current element are
1473:     passed through the ``buffer`` parameter, and the number of elements
1474:     within the footprint through ``filter_size``. The calculated value is
1475:     returned in ``return_value``. ``user_data`` is the data pointer provided
1476:     to `scipy.LowLevelCallable` as-is.
1477: 
1478:     The callback function must return an integer error status that is zero
1479:     if something went wrong and one otherwise. If an error occurs, you should
1480:     normally set the python error status with an informative message
1481:     before returning, otherwise a default error message is set by the
1482:     calling function.
1483: 
1484:     In addition, some other low-level function pointer specifications
1485:     are accepted, but these are for backward compatibility only and should
1486:     not be used in new code.
1487: 
1488:     '''
1489:     if extra_keywords is None:
1490:         extra_keywords = {}
1491:     input = numpy.asarray(input)
1492:     if numpy.iscomplexobj(input):
1493:         raise TypeError('Complex type not supported')
1494:     origins = _ni_support._normalize_sequence(origin, input.ndim)
1495:     if footprint is None:
1496:         if size is None:
1497:             raise RuntimeError("no footprint or filter size provided")
1498:         sizes = _ni_support._normalize_sequence(size, input.ndim)
1499:         footprint = numpy.ones(sizes, dtype=bool)
1500:     else:
1501:         footprint = numpy.asarray(footprint, dtype=bool)
1502:     fshape = [ii for ii in footprint.shape if ii > 0]
1503:     if len(fshape) != input.ndim:
1504:         raise RuntimeError('filter footprint array has incorrect shape.')
1505:     for origin, lenf in zip(origins, fshape):
1506:         if (lenf // 2 + origin < 0) or (lenf // 2 + origin >= lenf):
1507:             raise ValueError('invalid origin')
1508:     if not footprint.flags.contiguous:
1509:         footprint = footprint.copy()
1510:     output, return_value = _ni_support._get_output(output, input)
1511:     mode = _ni_support._extend_mode_to_code(mode)
1512:     _nd_image.generic_filter(input, function, footprint, output, mode,
1513:                              cval, origins, extra_arguments, extra_keywords)
1514:     return return_value
1515: 

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
import_117053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy')

if (type(import_117053) is not StypyTypeError):

    if (import_117053 != 'pyd_module'):
        __import__(import_117053)
        sys_modules_117054 = sys.modules[import_117053]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', sys_modules_117054.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'numpy', import_117053)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'from scipy.ndimage import _ni_support' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_117055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage')

if (type(import_117055) is not StypyTypeError):

    if (import_117055 != 'pyd_module'):
        __import__(import_117055)
        sys_modules_117056 = sys.modules[import_117055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', sys_modules_117056.module_type_store, module_type_store, ['_ni_support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 35, 0), __file__, sys_modules_117056, sys_modules_117056.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _ni_support

        import_from_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', None, module_type_store, ['_ni_support'], [_ni_support])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'scipy.ndimage', import_117055)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from scipy.ndimage import _nd_image' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_117057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage')

if (type(import_117057) is not StypyTypeError):

    if (import_117057 != 'pyd_module'):
        __import__(import_117057)
        sys_modules_117058 = sys.modules[import_117057]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', sys_modules_117058.module_type_store, module_type_store, ['_nd_image'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_117058, sys_modules_117058.module_type_store, module_type_store)
    else:
        from scipy.ndimage import _nd_image

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', None, module_type_store, ['_nd_image'], [_nd_image])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'scipy.ndimage', import_117057)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 37, 0))

# 'from scipy.misc import doccer' statement (line 37)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_117059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.misc')

if (type(import_117059) is not StypyTypeError):

    if (import_117059 != 'pyd_module'):
        __import__(import_117059)
        sys_modules_117060 = sys.modules[import_117059]
        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.misc', sys_modules_117060.module_type_store, module_type_store, ['doccer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 37, 0), __file__, sys_modules_117060, sys_modules_117060.module_type_store, module_type_store)
    else:
        from scipy.misc import doccer

        import_from_module(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.misc', None, module_type_store, ['doccer'], [doccer])

else:
    # Assigning a type to the variable 'scipy.misc' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'scipy.misc', import_117059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/ndimage/')
import_117061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy._lib._version')

if (type(import_117061) is not StypyTypeError):

    if (import_117061 != 'pyd_module'):
        __import__(import_117061)
        sys_modules_117062 = sys.modules[import_117061]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy._lib._version', sys_modules_117062.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_117062, sys_modules_117062.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'scipy._lib._version', import_117061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/ndimage/')


# Assigning a List to a Name (line 40):

# Assigning a List to a Name (line 40):
__all__ = ['correlate1d', 'convolve1d', 'gaussian_filter1d', 'gaussian_filter', 'prewitt', 'sobel', 'generic_laplace', 'laplace', 'gaussian_laplace', 'generic_gradient_magnitude', 'gaussian_gradient_magnitude', 'correlate', 'convolve', 'uniform_filter1d', 'uniform_filter', 'minimum_filter1d', 'maximum_filter1d', 'minimum_filter', 'maximum_filter', 'rank_filter', 'median_filter', 'percentile_filter', 'generic_filter1d', 'generic_filter']
module_type_store.set_exportable_members(['correlate1d', 'convolve1d', 'gaussian_filter1d', 'gaussian_filter', 'prewitt', 'sobel', 'generic_laplace', 'laplace', 'gaussian_laplace', 'generic_gradient_magnitude', 'gaussian_gradient_magnitude', 'correlate', 'convolve', 'uniform_filter1d', 'uniform_filter', 'minimum_filter1d', 'maximum_filter1d', 'minimum_filter', 'maximum_filter', 'rank_filter', 'median_filter', 'percentile_filter', 'generic_filter1d', 'generic_filter'])

# Obtaining an instance of the builtin type 'list' (line 40)
list_117063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_117064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'str', 'correlate1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117064)
# Adding element type (line 40)
str_117065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'str', 'convolve1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117065)
# Adding element type (line 40)
str_117066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'str', 'gaussian_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117066)
# Adding element type (line 40)
str_117067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'str', 'gaussian_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117067)
# Adding element type (line 40)
str_117068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'str', 'prewitt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117068)
# Adding element type (line 40)
str_117069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'str', 'sobel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117069)
# Adding element type (line 40)
str_117070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'str', 'generic_laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117070)
# Adding element type (line 40)
str_117071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'str', 'laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117071)
# Adding element type (line 40)
str_117072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'str', 'gaussian_laplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117072)
# Adding element type (line 40)
str_117073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'str', 'generic_gradient_magnitude')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117073)
# Adding element type (line 40)
str_117074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'gaussian_gradient_magnitude')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117074)
# Adding element type (line 40)
str_117075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'str', 'correlate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117075)
# Adding element type (line 40)
str_117076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 55), 'str', 'convolve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117076)
# Adding element type (line 40)
str_117077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', 'uniform_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117077)
# Adding element type (line 40)
str_117078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'str', 'uniform_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117078)
# Adding element type (line 40)
str_117079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 49), 'str', 'minimum_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117079)
# Adding element type (line 40)
str_117080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'str', 'maximum_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117080)
# Adding element type (line 40)
str_117081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'str', 'minimum_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117081)
# Adding element type (line 40)
str_117082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 49), 'str', 'maximum_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117082)
# Adding element type (line 40)
str_117083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'str', 'rank_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117083)
# Adding element type (line 40)
str_117084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'str', 'median_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117084)
# Adding element type (line 40)
str_117085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 43), 'str', 'percentile_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117085)
# Adding element type (line 40)
str_117086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'str', 'generic_filter1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117086)
# Adding element type (line 40)
str_117087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'str', 'generic_filter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 10), list_117063, str_117087)

# Assigning a type to the variable '__all__' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '__all__', list_117063)

# Assigning a Str to a Name (line 50):

# Assigning a Str to a Name (line 50):
str_117088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', 'input : array_like\n    Input array to filter.')
# Assigning a type to the variable '_input_doc' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_input_doc', str_117088)

# Assigning a Str to a Name (line 53):

# Assigning a Str to a Name (line 53):
str_117089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', 'axis : int, optional\n    The axis of `input` along which to calculate. Default is -1.')
# Assigning a type to the variable '_axis_doc' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '_axis_doc', str_117089)

# Assigning a Str to a Name (line 56):

# Assigning a Str to a Name (line 56):
str_117090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', 'output : array, optional\n    The `output` parameter passes an array in which to store the\n    filter output. Output array should have different name as compared\n    to input array to avoid aliasing errors.')
# Assigning a type to the variable '_output_doc' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), '_output_doc', str_117090)

# Assigning a Str to a Name (line 61):

# Assigning a Str to a Name (line 61):
str_117091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', 'size : scalar or tuple, optional\n    See footprint, below\nfootprint : array, optional\n    Either `size` or `footprint` must be defined.  `size` gives\n    the shape that is taken from the input array, at every element\n    position, to define the input to the filter function.\n    `footprint` is a boolean array that specifies (implicitly) a\n    shape, but also which of the elements within this shape will get\n    passed to the filter function.  Thus ``size=(n,m)`` is equivalent\n    to ``footprint=np.ones((n,m))``.  We adjust `size` to the number\n    of dimensions of the input array, so that, if the input array is\n    shape (10,10,10), and `size` is 2, then the actual size used is\n    (2,2,2).\n')
# Assigning a type to the variable '_size_foot_doc' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), '_size_foot_doc', str_117091)

# Assigning a Str to a Name (line 76):

# Assigning a Str to a Name (line 76):
str_117092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, (-1)), 'str', "mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional\n    The `mode` parameter determines how the array borders are\n    handled, where `cval` is the value when mode is equal to\n    'constant'. Default is 'reflect'")
# Assigning a type to the variable '_mode_doc' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), '_mode_doc', str_117092)

# Assigning a Str to a Name (line 81):

# Assigning a Str to a Name (line 81):
str_117093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', "mode : str or sequence, optional\n    The `mode` parameter determines how the array borders are\n    handled. Valid modes are {'reflect', 'constant', 'nearest',\n    'mirror', 'wrap'}. `cval` is the value used when mode is equal to\n    'constant'. A list of modes with length equal to the number of\n    axes can be provided to specify different modes for different\n    axes. Default is 'reflect'")
# Assigning a type to the variable '_mode_multiple_doc' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), '_mode_multiple_doc', str_117093)

# Assigning a Str to a Name (line 89):

# Assigning a Str to a Name (line 89):
str_117094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', "cval : scalar, optional\n    Value to fill past edges of input if `mode` is 'constant'. Default\n    is 0.0")
# Assigning a type to the variable '_cval_doc' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), '_cval_doc', str_117094)

# Assigning a Str to a Name (line 93):

# Assigning a Str to a Name (line 93):
str_117095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', 'origin : scalar, optional\n    The `origin` parameter controls the placement of the filter.\n    Default 0.0.')
# Assigning a type to the variable '_origin_doc' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), '_origin_doc', str_117095)

# Assigning a Str to a Name (line 97):

# Assigning a Str to a Name (line 97):
str_117096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', 'extra_arguments : sequence, optional\n    Sequence of extra positional arguments to pass to passed function')
# Assigning a type to the variable '_extra_arguments_doc' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), '_extra_arguments_doc', str_117096)

# Assigning a Str to a Name (line 100):

# Assigning a Str to a Name (line 100):
str_117097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', 'extra_keywords : dict, optional\n    dict of extra keyword arguments to pass to passed function')
# Assigning a type to the variable '_extra_keywords_doc' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), '_extra_keywords_doc', str_117097)

# Assigning a Dict to a Name (line 104):

# Assigning a Dict to a Name (line 104):

# Obtaining an instance of the builtin type 'dict' (line 104)
dict_117098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 104)
# Adding element type (key, value) (line 104)
str_117099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 4), 'str', 'input')
# Getting the type of '_input_doc' (line 105)
_input_doc_117100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), '_input_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117099, _input_doc_117100))
# Adding element type (key, value) (line 104)
str_117101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 4), 'str', 'axis')
# Getting the type of '_axis_doc' (line 106)
_axis_doc_117102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), '_axis_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117101, _axis_doc_117102))
# Adding element type (key, value) (line 104)
str_117103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 4), 'str', 'output')
# Getting the type of '_output_doc' (line 107)
_output_doc_117104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), '_output_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117103, _output_doc_117104))
# Adding element type (key, value) (line 104)
str_117105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 4), 'str', 'size_foot')
# Getting the type of '_size_foot_doc' (line 108)
_size_foot_doc_117106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), '_size_foot_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117105, _size_foot_doc_117106))
# Adding element type (key, value) (line 104)
str_117107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 4), 'str', 'mode')
# Getting the type of '_mode_doc' (line 109)
_mode_doc_117108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), '_mode_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117107, _mode_doc_117108))
# Adding element type (key, value) (line 104)
str_117109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'str', 'mode_multiple')
# Getting the type of '_mode_multiple_doc' (line 110)
_mode_multiple_doc_117110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), '_mode_multiple_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117109, _mode_multiple_doc_117110))
# Adding element type (key, value) (line 104)
str_117111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 4), 'str', 'cval')
# Getting the type of '_cval_doc' (line 111)
_cval_doc_117112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), '_cval_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117111, _cval_doc_117112))
# Adding element type (key, value) (line 104)
str_117113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'str', 'origin')
# Getting the type of '_origin_doc' (line 112)
_origin_doc_117114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), '_origin_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117113, _origin_doc_117114))
# Adding element type (key, value) (line 104)
str_117115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'str', 'extra_arguments')
# Getting the type of '_extra_arguments_doc' (line 113)
_extra_arguments_doc_117116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), '_extra_arguments_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117115, _extra_arguments_doc_117116))
# Adding element type (key, value) (line 104)
str_117117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 4), 'str', 'extra_keywords')
# Getting the type of '_extra_keywords_doc' (line 114)
_extra_keywords_doc_117118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), '_extra_keywords_doc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 10), dict_117098, (str_117117, _extra_keywords_doc_117118))

# Assigning a type to the variable 'docdict' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'docdict', dict_117098)

# Assigning a Call to a Name (line 117):

# Assigning a Call to a Name (line 117):

# Call to filldoc(...): (line 117)
# Processing the call arguments (line 117)
# Getting the type of 'docdict' (line 117)
docdict_117121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'docdict', False)
# Processing the call keyword arguments (line 117)
kwargs_117122 = {}
# Getting the type of 'doccer' (line 117)
doccer_117119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'doccer', False)
# Obtaining the member 'filldoc' of a type (line 117)
filldoc_117120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), doccer_117119, 'filldoc')
# Calling filldoc(args, kwargs) (line 117)
filldoc_call_result_117123 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), filldoc_117120, *[docdict_117121], **kwargs_117122)

# Assigning a type to the variable 'docfiller' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'docfiller', filldoc_call_result_117123)

@norecursion
def correlate1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 37), 'int')
    # Getting the type of 'None' (line 121)
    None_117125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'None')
    str_117126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 59), 'str', 'reflect')
    float_117127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'float')
    int_117128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'int')
    defaults = [int_117124, None_117125, str_117126, float_117127, int_117128]
    # Create a new context for function 'correlate1d'
    module_type_store = module_type_store.open_function_context('correlate1d', 120, 0, False)
    
    # Passed parameters checking function
    correlate1d.stypy_localization = localization
    correlate1d.stypy_type_of_self = None
    correlate1d.stypy_type_store = module_type_store
    correlate1d.stypy_function_name = 'correlate1d'
    correlate1d.stypy_param_names_list = ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin']
    correlate1d.stypy_varargs_param_name = None
    correlate1d.stypy_kwargs_param_name = None
    correlate1d.stypy_call_defaults = defaults
    correlate1d.stypy_call_varargs = varargs
    correlate1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'correlate1d', ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'correlate1d', localization, ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'correlate1d(...)' code ##################

    str_117129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', 'Calculate a one-dimensional correlation along the given axis.\n\n    The lines of the array along the given axis are correlated with the\n    given weights.\n\n    Parameters\n    ----------\n    %(input)s\n    weights : array\n        One-dimensional sequence of numbers.\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Examples\n    --------\n    >>> from scipy.ndimage import correlate1d\n    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])\n    array([ 8, 26,  8, 12,  7, 28, 36,  9])\n    ')
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to asarray(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'input' (line 145)
    input_117132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'input', False)
    # Processing the call keyword arguments (line 145)
    kwargs_117133 = {}
    # Getting the type of 'numpy' (line 145)
    numpy_117130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 145)
    asarray_117131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), numpy_117130, 'asarray')
    # Calling asarray(args, kwargs) (line 145)
    asarray_call_result_117134 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), asarray_117131, *[input_117132], **kwargs_117133)
    
    # Assigning a type to the variable 'input' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'input', asarray_call_result_117134)
    
    
    # Call to iscomplexobj(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'input' (line 146)
    input_117137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'input', False)
    # Processing the call keyword arguments (line 146)
    kwargs_117138 = {}
    # Getting the type of 'numpy' (line 146)
    numpy_117135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 146)
    iscomplexobj_117136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 7), numpy_117135, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 146)
    iscomplexobj_call_result_117139 = invoke(stypy.reporting.localization.Localization(__file__, 146, 7), iscomplexobj_117136, *[input_117137], **kwargs_117138)
    
    # Testing the type of an if condition (line 146)
    if_condition_117140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), iscomplexobj_call_result_117139)
    # Assigning a type to the variable 'if_condition_117140' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_117140', if_condition_117140)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 147)
    # Processing the call arguments (line 147)
    str_117142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 147)
    kwargs_117143 = {}
    # Getting the type of 'TypeError' (line 147)
    TypeError_117141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 147)
    TypeError_call_result_117144 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), TypeError_117141, *[str_117142], **kwargs_117143)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 147, 8), TypeError_call_result_117144, 'raise parameter', BaseException)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 148):
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_117145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _get_output(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'output' (line 148)
    output_117148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'output', False)
    # Getting the type of 'input' (line 148)
    input_117149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 59), 'input', False)
    # Processing the call keyword arguments (line 148)
    kwargs_117150 = {}
    # Getting the type of '_ni_support' (line 148)
    _ni_support_117146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 148)
    _get_output_117147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), _ni_support_117146, '_get_output')
    # Calling _get_output(args, kwargs) (line 148)
    _get_output_call_result_117151 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), _get_output_117147, *[output_117148, input_117149], **kwargs_117150)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___117152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _get_output_call_result_117151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_117153 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___117152, int_117145)
    
    # Assigning a type to the variable 'tuple_var_assignment_117023' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_117023', subscript_call_result_117153)
    
    # Assigning a Subscript to a Name (line 148):
    
    # Obtaining the type of the subscript
    int_117154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 4), 'int')
    
    # Call to _get_output(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'output' (line 148)
    output_117157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 51), 'output', False)
    # Getting the type of 'input' (line 148)
    input_117158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 59), 'input', False)
    # Processing the call keyword arguments (line 148)
    kwargs_117159 = {}
    # Getting the type of '_ni_support' (line 148)
    _ni_support_117155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 148)
    _get_output_117156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 27), _ni_support_117155, '_get_output')
    # Calling _get_output(args, kwargs) (line 148)
    _get_output_call_result_117160 = invoke(stypy.reporting.localization.Localization(__file__, 148, 27), _get_output_117156, *[output_117157, input_117158], **kwargs_117159)
    
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___117161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), _get_output_call_result_117160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_117162 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), getitem___117161, int_117154)
    
    # Assigning a type to the variable 'tuple_var_assignment_117024' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_117024', subscript_call_result_117162)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_117023' (line 148)
    tuple_var_assignment_117023_117163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_117023')
    # Assigning a type to the variable 'output' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'output', tuple_var_assignment_117023_117163)
    
    # Assigning a Name to a Name (line 148):
    # Getting the type of 'tuple_var_assignment_117024' (line 148)
    tuple_var_assignment_117024_117164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'tuple_var_assignment_117024')
    # Assigning a type to the variable 'return_value' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'return_value', tuple_var_assignment_117024_117164)
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 149):
    
    # Call to asarray(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'weights' (line 149)
    weights_117167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'weights', False)
    # Processing the call keyword arguments (line 149)
    # Getting the type of 'numpy' (line 149)
    numpy_117168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 43), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 149)
    float64_117169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 43), numpy_117168, 'float64')
    keyword_117170 = float64_117169
    kwargs_117171 = {'dtype': keyword_117170}
    # Getting the type of 'numpy' (line 149)
    numpy_117165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 149)
    asarray_117166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 14), numpy_117165, 'asarray')
    # Calling asarray(args, kwargs) (line 149)
    asarray_call_result_117172 = invoke(stypy.reporting.localization.Localization(__file__, 149, 14), asarray_117166, *[weights_117167], **kwargs_117171)
    
    # Assigning a type to the variable 'weights' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'weights', asarray_call_result_117172)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'weights' (line 150)
    weights_117173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'weights')
    # Obtaining the member 'ndim' of a type (line 150)
    ndim_117174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 7), weights_117173, 'ndim')
    int_117175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 23), 'int')
    # Applying the binary operator '!=' (line 150)
    result_ne_117176 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '!=', ndim_117174, int_117175)
    
    
    
    # Obtaining the type of the subscript
    int_117177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'int')
    # Getting the type of 'weights' (line 150)
    weights_117178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 28), 'weights')
    # Obtaining the member 'shape' of a type (line 150)
    shape_117179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 28), weights_117178, 'shape')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___117180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 28), shape_117179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_117181 = invoke(stypy.reporting.localization.Localization(__file__, 150, 28), getitem___117180, int_117177)
    
    int_117182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'int')
    # Applying the binary operator '<' (line 150)
    result_lt_117183 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 28), '<', subscript_call_result_117181, int_117182)
    
    # Applying the binary operator 'or' (line 150)
    result_or_keyword_117184 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), 'or', result_ne_117176, result_lt_117183)
    
    # Testing the type of an if condition (line 150)
    if_condition_117185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_or_keyword_117184)
    # Assigning a type to the variable 'if_condition_117185' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_117185', if_condition_117185)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 151)
    # Processing the call arguments (line 151)
    str_117187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 27), 'str', 'no filter weights given')
    # Processing the call keyword arguments (line 151)
    kwargs_117188 = {}
    # Getting the type of 'RuntimeError' (line 151)
    RuntimeError_117186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 151)
    RuntimeError_call_result_117189 = invoke(stypy.reporting.localization.Localization(__file__, 151, 14), RuntimeError_117186, *[str_117187], **kwargs_117188)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 8), RuntimeError_call_result_117189, 'raise parameter', BaseException)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'weights' (line 152)
    weights_117190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'weights')
    # Obtaining the member 'flags' of a type (line 152)
    flags_117191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), weights_117190, 'flags')
    # Obtaining the member 'contiguous' of a type (line 152)
    contiguous_117192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 11), flags_117191, 'contiguous')
    # Applying the 'not' unary operator (line 152)
    result_not__117193 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 7), 'not', contiguous_117192)
    
    # Testing the type of an if condition (line 152)
    if_condition_117194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 4), result_not__117193)
    # Assigning a type to the variable 'if_condition_117194' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'if_condition_117194', if_condition_117194)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 153):
    
    # Assigning a Call to a Name (line 153):
    
    # Call to copy(...): (line 153)
    # Processing the call keyword arguments (line 153)
    kwargs_117197 = {}
    # Getting the type of 'weights' (line 153)
    weights_117195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'weights', False)
    # Obtaining the member 'copy' of a type (line 153)
    copy_117196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 18), weights_117195, 'copy')
    # Calling copy(args, kwargs) (line 153)
    copy_call_result_117198 = invoke(stypy.reporting.localization.Localization(__file__, 153, 18), copy_117196, *[], **kwargs_117197)
    
    # Assigning a type to the variable 'weights' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'weights', copy_call_result_117198)
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to _check_axis(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'axis' (line 154)
    axis_117201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'axis', False)
    # Getting the type of 'input' (line 154)
    input_117202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 154)
    ndim_117203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 41), input_117202, 'ndim')
    # Processing the call keyword arguments (line 154)
    kwargs_117204 = {}
    # Getting the type of '_ni_support' (line 154)
    _ni_support_117199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 154)
    _check_axis_117200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), _ni_support_117199, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 154)
    _check_axis_call_result_117205 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), _check_axis_117200, *[axis_117201, ndim_117203], **kwargs_117204)
    
    # Assigning a type to the variable 'axis' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'axis', _check_axis_call_result_117205)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'weights' (line 155)
    weights_117207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'weights', False)
    # Processing the call keyword arguments (line 155)
    kwargs_117208 = {}
    # Getting the type of 'len' (line 155)
    len_117206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'len', False)
    # Calling len(args, kwargs) (line 155)
    len_call_result_117209 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), len_117206, *[weights_117207], **kwargs_117208)
    
    int_117210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
    # Applying the binary operator '//' (line 155)
    result_floordiv_117211 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 8), '//', len_call_result_117209, int_117210)
    
    # Getting the type of 'origin' (line 155)
    origin_117212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'origin')
    # Applying the binary operator '+' (line 155)
    result_add_117213 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 8), '+', result_floordiv_117211, origin_117212)
    
    int_117214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 37), 'int')
    # Applying the binary operator '<' (line 155)
    result_lt_117215 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 8), '<', result_add_117213, int_117214)
    
    
    
    # Call to len(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'weights' (line 155)
    weights_117217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 48), 'weights', False)
    # Processing the call keyword arguments (line 155)
    kwargs_117218 = {}
    # Getting the type of 'len' (line 155)
    len_117216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 44), 'len', False)
    # Calling len(args, kwargs) (line 155)
    len_call_result_117219 = invoke(stypy.reporting.localization.Localization(__file__, 155, 44), len_117216, *[weights_117217], **kwargs_117218)
    
    int_117220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 60), 'int')
    # Applying the binary operator '//' (line 155)
    result_floordiv_117221 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 44), '//', len_call_result_117219, int_117220)
    
    # Getting the type of 'origin' (line 156)
    origin_117222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 44), 'origin')
    # Applying the binary operator '+' (line 155)
    result_add_117223 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 44), '+', result_floordiv_117221, origin_117222)
    
    
    # Call to len(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'weights' (line 156)
    weights_117225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 57), 'weights', False)
    # Processing the call keyword arguments (line 156)
    kwargs_117226 = {}
    # Getting the type of 'len' (line 156)
    len_117224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'len', False)
    # Calling len(args, kwargs) (line 156)
    len_call_result_117227 = invoke(stypy.reporting.localization.Localization(__file__, 156, 53), len_117224, *[weights_117225], **kwargs_117226)
    
    # Applying the binary operator '>' (line 155)
    result_gt_117228 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 44), '>', result_add_117223, len_call_result_117227)
    
    # Applying the binary operator 'or' (line 155)
    result_or_keyword_117229 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 7), 'or', result_lt_117215, result_gt_117228)
    
    # Testing the type of an if condition (line 155)
    if_condition_117230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 4), result_or_keyword_117229)
    # Assigning a type to the variable 'if_condition_117230' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'if_condition_117230', if_condition_117230)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 157)
    # Processing the call arguments (line 157)
    str_117232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 157)
    kwargs_117233 = {}
    # Getting the type of 'ValueError' (line 157)
    ValueError_117231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 157)
    ValueError_call_result_117234 = invoke(stypy.reporting.localization.Localization(__file__, 157, 14), ValueError_117231, *[str_117232], **kwargs_117233)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 8), ValueError_call_result_117234, 'raise parameter', BaseException)
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to _extend_mode_to_code(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'mode' (line 158)
    mode_117237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 44), 'mode', False)
    # Processing the call keyword arguments (line 158)
    kwargs_117238 = {}
    # Getting the type of '_ni_support' (line 158)
    _ni_support_117235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 158)
    _extend_mode_to_code_117236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), _ni_support_117235, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 158)
    _extend_mode_to_code_call_result_117239 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), _extend_mode_to_code_117236, *[mode_117237], **kwargs_117238)
    
    # Assigning a type to the variable 'mode' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'mode', _extend_mode_to_code_call_result_117239)
    
    # Call to correlate1d(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'input' (line 159)
    input_117242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'input', False)
    # Getting the type of 'weights' (line 159)
    weights_117243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'weights', False)
    # Getting the type of 'axis' (line 159)
    axis_117244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'axis', False)
    # Getting the type of 'output' (line 159)
    output_117245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'output', False)
    # Getting the type of 'mode' (line 159)
    mode_117246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 56), 'mode', False)
    # Getting the type of 'cval' (line 159)
    cval_117247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 62), 'cval', False)
    # Getting the type of 'origin' (line 160)
    origin_117248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'origin', False)
    # Processing the call keyword arguments (line 159)
    kwargs_117249 = {}
    # Getting the type of '_nd_image' (line 159)
    _nd_image_117240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), '_nd_image', False)
    # Obtaining the member 'correlate1d' of a type (line 159)
    correlate1d_117241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), _nd_image_117240, 'correlate1d')
    # Calling correlate1d(args, kwargs) (line 159)
    correlate1d_call_result_117250 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), correlate1d_117241, *[input_117242, weights_117243, axis_117244, output_117245, mode_117246, cval_117247, origin_117248], **kwargs_117249)
    
    # Getting the type of 'return_value' (line 161)
    return_value_117251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', return_value_117251)
    
    # ################# End of 'correlate1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'correlate1d' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_117252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117252)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'correlate1d'
    return stypy_return_type_117252

# Assigning a type to the variable 'correlate1d' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'correlate1d', correlate1d)

@norecursion
def convolve1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'int')
    # Getting the type of 'None' (line 165)
    None_117254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 47), 'None')
    str_117255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 58), 'str', 'reflect')
    float_117256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'float')
    int_117257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 32), 'int')
    defaults = [int_117253, None_117254, str_117255, float_117256, int_117257]
    # Create a new context for function 'convolve1d'
    module_type_store = module_type_store.open_function_context('convolve1d', 164, 0, False)
    
    # Passed parameters checking function
    convolve1d.stypy_localization = localization
    convolve1d.stypy_type_of_self = None
    convolve1d.stypy_type_store = module_type_store
    convolve1d.stypy_function_name = 'convolve1d'
    convolve1d.stypy_param_names_list = ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin']
    convolve1d.stypy_varargs_param_name = None
    convolve1d.stypy_kwargs_param_name = None
    convolve1d.stypy_call_defaults = defaults
    convolve1d.stypy_call_varargs = varargs
    convolve1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convolve1d', ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convolve1d', localization, ['input', 'weights', 'axis', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convolve1d(...)' code ##################

    str_117258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, (-1)), 'str', 'Calculate a one-dimensional convolution along the given axis.\n\n    The lines of the array along the given axis are convolved with the\n    given weights.\n\n    Parameters\n    ----------\n    %(input)s\n    weights : ndarray\n        One-dimensional sequence of numbers.\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    convolve1d : ndarray\n        Convolved array with same shape as input\n\n    Examples\n    --------\n    >>> from scipy.ndimage import convolve1d\n    >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])\n    array([14, 24,  4, 13, 12, 36, 27,  0])\n    ')
    
    # Assigning a Subscript to a Name (line 194):
    
    # Assigning a Subscript to a Name (line 194):
    
    # Obtaining the type of the subscript
    int_117259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 24), 'int')
    slice_117260 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 194, 14), None, None, int_117259)
    # Getting the type of 'weights' (line 194)
    weights_117261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 14), 'weights')
    # Obtaining the member '__getitem__' of a type (line 194)
    getitem___117262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 14), weights_117261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 194)
    subscript_call_result_117263 = invoke(stypy.reporting.localization.Localization(__file__, 194, 14), getitem___117262, slice_117260)
    
    # Assigning a type to the variable 'weights' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'weights', subscript_call_result_117263)
    
    # Assigning a UnaryOp to a Name (line 195):
    
    # Assigning a UnaryOp to a Name (line 195):
    
    # Getting the type of 'origin' (line 195)
    origin_117264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'origin')
    # Applying the 'usub' unary operator (line 195)
    result___neg___117265 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 13), 'usub', origin_117264)
    
    # Assigning a type to the variable 'origin' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'origin', result___neg___117265)
    
    
    
    # Call to len(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'weights' (line 196)
    weights_117267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'weights', False)
    # Processing the call keyword arguments (line 196)
    kwargs_117268 = {}
    # Getting the type of 'len' (line 196)
    len_117266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'len', False)
    # Calling len(args, kwargs) (line 196)
    len_call_result_117269 = invoke(stypy.reporting.localization.Localization(__file__, 196, 11), len_117266, *[weights_117267], **kwargs_117268)
    
    int_117270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 26), 'int')
    # Applying the binary operator '&' (line 196)
    result_and__117271 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), '&', len_call_result_117269, int_117270)
    
    # Applying the 'not' unary operator (line 196)
    result_not__117272 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 7), 'not', result_and__117271)
    
    # Testing the type of an if condition (line 196)
    if_condition_117273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), result_not__117272)
    # Assigning a type to the variable 'if_condition_117273' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_117273', if_condition_117273)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'origin' (line 197)
    origin_117274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'origin')
    int_117275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'int')
    # Applying the binary operator '-=' (line 197)
    result_isub_117276 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 8), '-=', origin_117274, int_117275)
    # Assigning a type to the variable 'origin' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'origin', result_isub_117276)
    
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to correlate1d(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'input' (line 198)
    input_117278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'input', False)
    # Getting the type of 'weights' (line 198)
    weights_117279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'weights', False)
    # Getting the type of 'axis' (line 198)
    axis_117280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 39), 'axis', False)
    # Getting the type of 'output' (line 198)
    output_117281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 45), 'output', False)
    # Getting the type of 'mode' (line 198)
    mode_117282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 53), 'mode', False)
    # Getting the type of 'cval' (line 198)
    cval_117283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 59), 'cval', False)
    # Getting the type of 'origin' (line 198)
    origin_117284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 65), 'origin', False)
    # Processing the call keyword arguments (line 198)
    kwargs_117285 = {}
    # Getting the type of 'correlate1d' (line 198)
    correlate1d_117277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 198)
    correlate1d_call_result_117286 = invoke(stypy.reporting.localization.Localization(__file__, 198, 11), correlate1d_117277, *[input_117278, weights_117279, axis_117280, output_117281, mode_117282, cval_117283, origin_117284], **kwargs_117285)
    
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type', correlate1d_call_result_117286)
    
    # ################# End of 'convolve1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convolve1d' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_117287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117287)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convolve1d'
    return stypy_return_type_117287

# Assigning a type to the variable 'convolve1d' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'convolve1d', convolve1d)

@norecursion
def _gaussian_kernel1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_gaussian_kernel1d'
    module_type_store = module_type_store.open_function_context('_gaussian_kernel1d', 201, 0, False)
    
    # Passed parameters checking function
    _gaussian_kernel1d.stypy_localization = localization
    _gaussian_kernel1d.stypy_type_of_self = None
    _gaussian_kernel1d.stypy_type_store = module_type_store
    _gaussian_kernel1d.stypy_function_name = '_gaussian_kernel1d'
    _gaussian_kernel1d.stypy_param_names_list = ['sigma', 'order', 'radius']
    _gaussian_kernel1d.stypy_varargs_param_name = None
    _gaussian_kernel1d.stypy_kwargs_param_name = None
    _gaussian_kernel1d.stypy_call_defaults = defaults
    _gaussian_kernel1d.stypy_call_varargs = varargs
    _gaussian_kernel1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_gaussian_kernel1d', ['sigma', 'order', 'radius'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_gaussian_kernel1d', localization, ['sigma', 'order', 'radius'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_gaussian_kernel1d(...)' code ##################

    str_117288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, (-1)), 'str', '\n    Computes a 1D Gaussian convolution kernel.\n    ')
    
    
    # Getting the type of 'order' (line 205)
    order_117289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 7), 'order')
    int_117290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 15), 'int')
    # Applying the binary operator '<' (line 205)
    result_lt_117291 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 7), '<', order_117289, int_117290)
    
    # Testing the type of an if condition (line 205)
    if_condition_117292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 4), result_lt_117291)
    # Assigning a type to the variable 'if_condition_117292' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'if_condition_117292', if_condition_117292)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 206)
    # Processing the call arguments (line 206)
    str_117294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'str', 'order must be non-negative')
    # Processing the call keyword arguments (line 206)
    kwargs_117295 = {}
    # Getting the type of 'ValueError' (line 206)
    ValueError_117293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 206)
    ValueError_call_result_117296 = invoke(stypy.reporting.localization.Localization(__file__, 206, 14), ValueError_117293, *[str_117294], **kwargs_117295)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 206, 8), ValueError_call_result_117296, 'raise parameter', BaseException)
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to Polynomial(...): (line 207)
    # Processing the call arguments (line 207)
    
    # Obtaining an instance of the builtin type 'list' (line 207)
    list_117300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 207)
    # Adding element type (line 207)
    int_117301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_117300, int_117301)
    # Adding element type (line 207)
    int_117302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_117300, int_117302)
    # Adding element type (line 207)
    float_117303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 43), 'float')
    # Getting the type of 'sigma' (line 207)
    sigma_117304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 51), 'sigma', False)
    # Getting the type of 'sigma' (line 207)
    sigma_117305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 59), 'sigma', False)
    # Applying the binary operator '*' (line 207)
    result_mul_117306 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 51), '*', sigma_117304, sigma_117305)
    
    # Applying the binary operator 'div' (line 207)
    result_div_117307 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 43), 'div', float_117303, result_mul_117306)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_117300, result_div_117307)
    
    # Processing the call keyword arguments (line 207)
    kwargs_117308 = {}
    # Getting the type of 'numpy' (line 207)
    numpy_117297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'numpy', False)
    # Obtaining the member 'polynomial' of a type (line 207)
    polynomial_117298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), numpy_117297, 'polynomial')
    # Obtaining the member 'Polynomial' of a type (line 207)
    Polynomial_117299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), polynomial_117298, 'Polynomial')
    # Calling Polynomial(args, kwargs) (line 207)
    Polynomial_call_result_117309 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), Polynomial_117299, *[list_117300], **kwargs_117308)
    
    # Assigning a type to the variable 'p' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'p', Polynomial_call_result_117309)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to arange(...): (line 208)
    # Processing the call arguments (line 208)
    
    # Getting the type of 'radius' (line 208)
    radius_117312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'radius', False)
    # Applying the 'usub' unary operator (line 208)
    result___neg___117313 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 21), 'usub', radius_117312)
    
    # Getting the type of 'radius' (line 208)
    radius_117314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'radius', False)
    int_117315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 39), 'int')
    # Applying the binary operator '+' (line 208)
    result_add_117316 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 30), '+', radius_117314, int_117315)
    
    # Processing the call keyword arguments (line 208)
    kwargs_117317 = {}
    # Getting the type of 'numpy' (line 208)
    numpy_117310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'numpy', False)
    # Obtaining the member 'arange' of a type (line 208)
    arange_117311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), numpy_117310, 'arange')
    # Calling arange(args, kwargs) (line 208)
    arange_call_result_117318 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), arange_117311, *[result___neg___117313, result_add_117316], **kwargs_117317)
    
    # Assigning a type to the variable 'x' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'x', arange_call_result_117318)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to exp(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Call to p(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'x' (line 209)
    x_117322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'x', False)
    # Processing the call keyword arguments (line 209)
    kwargs_117323 = {}
    # Getting the type of 'p' (line 209)
    p_117321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'p', False)
    # Calling p(args, kwargs) (line 209)
    p_call_result_117324 = invoke(stypy.reporting.localization.Localization(__file__, 209, 22), p_117321, *[x_117322], **kwargs_117323)
    
    # Processing the call keyword arguments (line 209)
    # Getting the type of 'numpy' (line 209)
    numpy_117325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'numpy', False)
    # Obtaining the member 'double' of a type (line 209)
    double_117326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 34), numpy_117325, 'double')
    keyword_117327 = double_117326
    kwargs_117328 = {'dtype': keyword_117327}
    # Getting the type of 'numpy' (line 209)
    numpy_117319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'numpy', False)
    # Obtaining the member 'exp' of a type (line 209)
    exp_117320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), numpy_117319, 'exp')
    # Calling exp(args, kwargs) (line 209)
    exp_call_result_117329 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), exp_117320, *[p_call_result_117324], **kwargs_117328)
    
    # Assigning a type to the variable 'phi_x' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'phi_x', exp_call_result_117329)
    
    # Getting the type of 'phi_x' (line 210)
    phi_x_117330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'phi_x')
    
    # Call to sum(...): (line 210)
    # Processing the call keyword arguments (line 210)
    kwargs_117333 = {}
    # Getting the type of 'phi_x' (line 210)
    phi_x_117331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'phi_x', False)
    # Obtaining the member 'sum' of a type (line 210)
    sum_117332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), phi_x_117331, 'sum')
    # Calling sum(args, kwargs) (line 210)
    sum_call_result_117334 = invoke(stypy.reporting.localization.Localization(__file__, 210, 13), sum_117332, *[], **kwargs_117333)
    
    # Applying the binary operator 'div=' (line 210)
    result_div_117335 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 4), 'div=', phi_x_117330, sum_call_result_117334)
    # Assigning a type to the variable 'phi_x' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'phi_x', result_div_117335)
    
    
    
    # Getting the type of 'order' (line 211)
    order_117336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 7), 'order')
    int_117337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'int')
    # Applying the binary operator '>' (line 211)
    result_gt_117338 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 7), '>', order_117336, int_117337)
    
    # Testing the type of an if condition (line 211)
    if_condition_117339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 4), result_gt_117338)
    # Assigning a type to the variable 'if_condition_117339' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'if_condition_117339', if_condition_117339)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to Polynomial(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Obtaining an instance of the builtin type 'list' (line 212)
    list_117343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 212)
    # Adding element type (line 212)
    int_117344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 40), list_117343, int_117344)
    
    # Processing the call keyword arguments (line 212)
    kwargs_117345 = {}
    # Getting the type of 'numpy' (line 212)
    numpy_117340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'numpy', False)
    # Obtaining the member 'polynomial' of a type (line 212)
    polynomial_117341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), numpy_117340, 'polynomial')
    # Obtaining the member 'Polynomial' of a type (line 212)
    Polynomial_117342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), polynomial_117341, 'Polynomial')
    # Calling Polynomial(args, kwargs) (line 212)
    Polynomial_call_result_117346 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), Polynomial_117342, *[list_117343], **kwargs_117345)
    
    # Assigning a type to the variable 'q' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'q', Polynomial_call_result_117346)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to deriv(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_117349 = {}
    # Getting the type of 'p' (line 213)
    p_117347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'p', False)
    # Obtaining the member 'deriv' of a type (line 213)
    deriv_117348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 18), p_117347, 'deriv')
    # Calling deriv(args, kwargs) (line 213)
    deriv_call_result_117350 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), deriv_117348, *[], **kwargs_117349)
    
    # Assigning a type to the variable 'p_deriv' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'p_deriv', deriv_call_result_117350)
    
    
    # Call to range(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'order' (line 214)
    order_117352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'order', False)
    # Processing the call keyword arguments (line 214)
    kwargs_117353 = {}
    # Getting the type of 'range' (line 214)
    range_117351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'range', False)
    # Calling range(args, kwargs) (line 214)
    range_call_result_117354 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), range_117351, *[order_117352], **kwargs_117353)
    
    # Testing the type of a for loop iterable (line 214)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 214, 8), range_call_result_117354)
    # Getting the type of the for loop variable (line 214)
    for_loop_var_117355 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 214, 8), range_call_result_117354)
    # Assigning a type to the variable '_' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), '_', for_loop_var_117355)
    # SSA begins for a for statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 217):
    
    # Assigning a BinOp to a Name (line 217):
    
    # Call to deriv(...): (line 217)
    # Processing the call keyword arguments (line 217)
    kwargs_117358 = {}
    # Getting the type of 'q' (line 217)
    q_117356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'q', False)
    # Obtaining the member 'deriv' of a type (line 217)
    deriv_117357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), q_117356, 'deriv')
    # Calling deriv(args, kwargs) (line 217)
    deriv_call_result_117359 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), deriv_117357, *[], **kwargs_117358)
    
    # Getting the type of 'q' (line 217)
    q_117360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'q')
    # Getting the type of 'p_deriv' (line 217)
    p_deriv_117361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'p_deriv')
    # Applying the binary operator '*' (line 217)
    result_mul_117362 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 28), '*', q_117360, p_deriv_117361)
    
    # Applying the binary operator '+' (line 217)
    result_add_117363 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 16), '+', deriv_call_result_117359, result_mul_117362)
    
    # Assigning a type to the variable 'q' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'q', result_add_117363)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'phi_x' (line 218)
    phi_x_117364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'phi_x')
    
    # Call to q(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'x' (line 218)
    x_117366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'x', False)
    # Processing the call keyword arguments (line 218)
    kwargs_117367 = {}
    # Getting the type of 'q' (line 218)
    q_117365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'q', False)
    # Calling q(args, kwargs) (line 218)
    q_call_result_117368 = invoke(stypy.reporting.localization.Localization(__file__, 218, 17), q_117365, *[x_117366], **kwargs_117367)
    
    # Applying the binary operator '*=' (line 218)
    result_imul_117369 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 8), '*=', phi_x_117364, q_call_result_117368)
    # Assigning a type to the variable 'phi_x' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'phi_x', result_imul_117369)
    
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'phi_x' (line 219)
    phi_x_117370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'phi_x')
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type', phi_x_117370)
    
    # ################# End of '_gaussian_kernel1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_gaussian_kernel1d' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_117371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117371)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_gaussian_kernel1d'
    return stypy_return_type_117371

# Assigning a type to the variable '_gaussian_kernel1d' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), '_gaussian_kernel1d', _gaussian_kernel1d)

@norecursion
def gaussian_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 41), 'int')
    int_117373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 51), 'int')
    # Getting the type of 'None' (line 223)
    None_117374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 61), 'None')
    str_117375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 27), 'str', 'reflect')
    float_117376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 43), 'float')
    float_117377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 57), 'float')
    defaults = [int_117372, int_117373, None_117374, str_117375, float_117376, float_117377]
    # Create a new context for function 'gaussian_filter1d'
    module_type_store = module_type_store.open_function_context('gaussian_filter1d', 222, 0, False)
    
    # Passed parameters checking function
    gaussian_filter1d.stypy_localization = localization
    gaussian_filter1d.stypy_type_of_self = None
    gaussian_filter1d.stypy_type_store = module_type_store
    gaussian_filter1d.stypy_function_name = 'gaussian_filter1d'
    gaussian_filter1d.stypy_param_names_list = ['input', 'sigma', 'axis', 'order', 'output', 'mode', 'cval', 'truncate']
    gaussian_filter1d.stypy_varargs_param_name = None
    gaussian_filter1d.stypy_kwargs_param_name = None
    gaussian_filter1d.stypy_call_defaults = defaults
    gaussian_filter1d.stypy_call_varargs = varargs
    gaussian_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gaussian_filter1d', ['input', 'sigma', 'axis', 'order', 'output', 'mode', 'cval', 'truncate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gaussian_filter1d', localization, ['input', 'sigma', 'axis', 'order', 'output', 'mode', 'cval', 'truncate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gaussian_filter1d(...)' code ##################

    str_117378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, (-1)), 'str', "One-dimensional Gaussian filter.\n\n    Parameters\n    ----------\n    %(input)s\n    sigma : scalar\n        standard deviation for Gaussian kernel\n    %(axis)s\n    order : int, optional\n        An order of 0 corresponds to convolution with a Gaussian\n        kernel. A positive order corresponds to convolution with\n        that derivative of a Gaussian.\n    %(output)s\n    %(mode)s\n    %(cval)s\n    truncate : float, optional\n        Truncate the filter at this many standard deviations.\n        Default is 4.0.\n\n    Returns\n    -------\n    gaussian_filter1d : ndarray\n\n    Examples\n    --------\n    >>> from scipy.ndimage import gaussian_filter1d\n    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)\n    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])\n    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)\n    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])\n    >>> import matplotlib.pyplot as plt\n    >>> np.random.seed(280490)\n    >>> x = np.random.randn(101).cumsum()\n    >>> y3 = gaussian_filter1d(x, 3)\n    >>> y6 = gaussian_filter1d(x, 6)\n    >>> plt.plot(x, 'k', label='original data')\n    >>> plt.plot(y3, '--', label='filtered, sigma=3')\n    >>> plt.plot(y6, ':', label='filtered, sigma=6')\n    >>> plt.legend()\n    >>> plt.grid()\n    >>> plt.show()\n    ")
    
    # Assigning a Call to a Name (line 267):
    
    # Assigning a Call to a Name (line 267):
    
    # Call to float(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'sigma' (line 267)
    sigma_117380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 15), 'sigma', False)
    # Processing the call keyword arguments (line 267)
    kwargs_117381 = {}
    # Getting the type of 'float' (line 267)
    float_117379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 9), 'float', False)
    # Calling float(args, kwargs) (line 267)
    float_call_result_117382 = invoke(stypy.reporting.localization.Localization(__file__, 267, 9), float_117379, *[sigma_117380], **kwargs_117381)
    
    # Assigning a type to the variable 'sd' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'sd', float_call_result_117382)
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to int(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'truncate' (line 269)
    truncate_117384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), 'truncate', False)
    # Getting the type of 'sd' (line 269)
    sd_117385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'sd', False)
    # Applying the binary operator '*' (line 269)
    result_mul_117386 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 13), '*', truncate_117384, sd_117385)
    
    float_117387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 29), 'float')
    # Applying the binary operator '+' (line 269)
    result_add_117388 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 13), '+', result_mul_117386, float_117387)
    
    # Processing the call keyword arguments (line 269)
    kwargs_117389 = {}
    # Getting the type of 'int' (line 269)
    int_117383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 9), 'int', False)
    # Calling int(args, kwargs) (line 269)
    int_call_result_117390 = invoke(stypy.reporting.localization.Localization(__file__, 269, 9), int_117383, *[result_add_117388], **kwargs_117389)
    
    # Assigning a type to the variable 'lw' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'lw', int_call_result_117390)
    
    # Assigning a Subscript to a Name (line 271):
    
    # Assigning a Subscript to a Name (line 271):
    
    # Obtaining the type of the subscript
    int_117391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 53), 'int')
    slice_117392 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 14), None, None, int_117391)
    
    # Call to _gaussian_kernel1d(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'sigma' (line 271)
    sigma_117394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 33), 'sigma', False)
    # Getting the type of 'order' (line 271)
    order_117395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 40), 'order', False)
    # Getting the type of 'lw' (line 271)
    lw_117396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 47), 'lw', False)
    # Processing the call keyword arguments (line 271)
    kwargs_117397 = {}
    # Getting the type of '_gaussian_kernel1d' (line 271)
    _gaussian_kernel1d_117393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 14), '_gaussian_kernel1d', False)
    # Calling _gaussian_kernel1d(args, kwargs) (line 271)
    _gaussian_kernel1d_call_result_117398 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), _gaussian_kernel1d_117393, *[sigma_117394, order_117395, lw_117396], **kwargs_117397)
    
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___117399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 14), _gaussian_kernel1d_call_result_117398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_117400 = invoke(stypy.reporting.localization.Localization(__file__, 271, 14), getitem___117399, slice_117392)
    
    # Assigning a type to the variable 'weights' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'weights', subscript_call_result_117400)
    
    # Call to correlate1d(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'input' (line 272)
    input_117402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'input', False)
    # Getting the type of 'weights' (line 272)
    weights_117403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'weights', False)
    # Getting the type of 'axis' (line 272)
    axis_117404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 39), 'axis', False)
    # Getting the type of 'output' (line 272)
    output_117405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 45), 'output', False)
    # Getting the type of 'mode' (line 272)
    mode_117406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 53), 'mode', False)
    # Getting the type of 'cval' (line 272)
    cval_117407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 59), 'cval', False)
    int_117408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 65), 'int')
    # Processing the call keyword arguments (line 272)
    kwargs_117409 = {}
    # Getting the type of 'correlate1d' (line 272)
    correlate1d_117401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 272)
    correlate1d_call_result_117410 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), correlate1d_117401, *[input_117402, weights_117403, axis_117404, output_117405, mode_117406, cval_117407, int_117408], **kwargs_117409)
    
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', correlate1d_call_result_117410)
    
    # ################# End of 'gaussian_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gaussian_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 222)
    stypy_return_type_117411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117411)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gaussian_filter1d'
    return stypy_return_type_117411

# Assigning a type to the variable 'gaussian_filter1d' (line 222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 0), 'gaussian_filter1d', gaussian_filter1d)

@norecursion
def gaussian_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 40), 'int')
    # Getting the type of 'None' (line 276)
    None_117413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 50), 'None')
    str_117414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'str', 'reflect')
    float_117415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 41), 'float')
    float_117416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 55), 'float')
    defaults = [int_117412, None_117413, str_117414, float_117415, float_117416]
    # Create a new context for function 'gaussian_filter'
    module_type_store = module_type_store.open_function_context('gaussian_filter', 275, 0, False)
    
    # Passed parameters checking function
    gaussian_filter.stypy_localization = localization
    gaussian_filter.stypy_type_of_self = None
    gaussian_filter.stypy_type_store = module_type_store
    gaussian_filter.stypy_function_name = 'gaussian_filter'
    gaussian_filter.stypy_param_names_list = ['input', 'sigma', 'order', 'output', 'mode', 'cval', 'truncate']
    gaussian_filter.stypy_varargs_param_name = None
    gaussian_filter.stypy_kwargs_param_name = None
    gaussian_filter.stypy_call_defaults = defaults
    gaussian_filter.stypy_call_varargs = varargs
    gaussian_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gaussian_filter', ['input', 'sigma', 'order', 'output', 'mode', 'cval', 'truncate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gaussian_filter', localization, ['input', 'sigma', 'order', 'output', 'mode', 'cval', 'truncate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gaussian_filter(...)' code ##################

    str_117417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, (-1)), 'str', 'Multidimensional Gaussian filter.\n\n    Parameters\n    ----------\n    %(input)s\n    sigma : scalar or sequence of scalars\n        Standard deviation for Gaussian kernel. The standard\n        deviations of the Gaussian filter are given for each axis as a\n        sequence, or as a single number, in which case it is equal for\n        all axes.\n    order : int or sequence of ints, optional\n        The order of the filter along each axis is given as a sequence\n        of integers, or as a single number.  An order of 0 corresponds\n        to convolution with a Gaussian kernel. A positive order\n        corresponds to convolution with that derivative of a Gaussian.\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    truncate : float\n        Truncate the filter at this many standard deviations.\n        Default is 4.0.\n\n    Returns\n    -------\n    gaussian_filter : ndarray\n        Returned array of same shape as `input`.\n\n    Notes\n    -----\n    The multidimensional filter is implemented as a sequence of\n    one-dimensional convolution filters. The intermediate arrays are\n    stored in the same data type as the output. Therefore, for output\n    types with a limited precision, the results may be imprecise\n    because intermediate results may be stored with insufficient\n    precision.\n\n    Examples\n    --------\n    >>> from scipy.ndimage import gaussian_filter\n    >>> a = np.arange(50, step=2).reshape((5,5))\n    >>> a\n    array([[ 0,  2,  4,  6,  8],\n           [10, 12, 14, 16, 18],\n           [20, 22, 24, 26, 28],\n           [30, 32, 34, 36, 38],\n           [40, 42, 44, 46, 48]])\n    >>> gaussian_filter(a, sigma=1)\n    array([[ 4,  6,  8,  9, 11],\n           [10, 12, 14, 15, 17],\n           [20, 22, 24, 25, 27],\n           [29, 31, 33, 34, 36],\n           [35, 37, 39, 40, 42]])\n\n    >>> from scipy import misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = gaussian_filter(ascent, sigma=5)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 343):
    
    # Assigning a Call to a Name (line 343):
    
    # Call to asarray(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'input' (line 343)
    input_117420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'input', False)
    # Processing the call keyword arguments (line 343)
    kwargs_117421 = {}
    # Getting the type of 'numpy' (line 343)
    numpy_117418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 343)
    asarray_117419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), numpy_117418, 'asarray')
    # Calling asarray(args, kwargs) (line 343)
    asarray_call_result_117422 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), asarray_117419, *[input_117420], **kwargs_117421)
    
    # Assigning a type to the variable 'input' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'input', asarray_call_result_117422)
    
    # Assigning a Call to a Tuple (line 344):
    
    # Assigning a Subscript to a Name (line 344):
    
    # Obtaining the type of the subscript
    int_117423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 4), 'int')
    
    # Call to _get_output(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'output' (line 344)
    output_117426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'output', False)
    # Getting the type of 'input' (line 344)
    input_117427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 59), 'input', False)
    # Processing the call keyword arguments (line 344)
    kwargs_117428 = {}
    # Getting the type of '_ni_support' (line 344)
    _ni_support_117424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 344)
    _get_output_117425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 27), _ni_support_117424, '_get_output')
    # Calling _get_output(args, kwargs) (line 344)
    _get_output_call_result_117429 = invoke(stypy.reporting.localization.Localization(__file__, 344, 27), _get_output_117425, *[output_117426, input_117427], **kwargs_117428)
    
    # Obtaining the member '__getitem__' of a type (line 344)
    getitem___117430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 4), _get_output_call_result_117429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 344)
    subscript_call_result_117431 = invoke(stypy.reporting.localization.Localization(__file__, 344, 4), getitem___117430, int_117423)
    
    # Assigning a type to the variable 'tuple_var_assignment_117025' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'tuple_var_assignment_117025', subscript_call_result_117431)
    
    # Assigning a Subscript to a Name (line 344):
    
    # Obtaining the type of the subscript
    int_117432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 4), 'int')
    
    # Call to _get_output(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'output' (line 344)
    output_117435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 51), 'output', False)
    # Getting the type of 'input' (line 344)
    input_117436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 59), 'input', False)
    # Processing the call keyword arguments (line 344)
    kwargs_117437 = {}
    # Getting the type of '_ni_support' (line 344)
    _ni_support_117433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 344)
    _get_output_117434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 27), _ni_support_117433, '_get_output')
    # Calling _get_output(args, kwargs) (line 344)
    _get_output_call_result_117438 = invoke(stypy.reporting.localization.Localization(__file__, 344, 27), _get_output_117434, *[output_117435, input_117436], **kwargs_117437)
    
    # Obtaining the member '__getitem__' of a type (line 344)
    getitem___117439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 4), _get_output_call_result_117438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 344)
    subscript_call_result_117440 = invoke(stypy.reporting.localization.Localization(__file__, 344, 4), getitem___117439, int_117432)
    
    # Assigning a type to the variable 'tuple_var_assignment_117026' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'tuple_var_assignment_117026', subscript_call_result_117440)
    
    # Assigning a Name to a Name (line 344):
    # Getting the type of 'tuple_var_assignment_117025' (line 344)
    tuple_var_assignment_117025_117441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'tuple_var_assignment_117025')
    # Assigning a type to the variable 'output' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'output', tuple_var_assignment_117025_117441)
    
    # Assigning a Name to a Name (line 344):
    # Getting the type of 'tuple_var_assignment_117026' (line 344)
    tuple_var_assignment_117026_117442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'tuple_var_assignment_117026')
    # Assigning a type to the variable 'return_value' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'return_value', tuple_var_assignment_117026_117442)
    
    # Assigning a Call to a Name (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to _normalize_sequence(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'order' (line 345)
    order_117445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 45), 'order', False)
    # Getting the type of 'input' (line 345)
    input_117446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 52), 'input', False)
    # Obtaining the member 'ndim' of a type (line 345)
    ndim_117447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 52), input_117446, 'ndim')
    # Processing the call keyword arguments (line 345)
    kwargs_117448 = {}
    # Getting the type of '_ni_support' (line 345)
    _ni_support_117443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 345)
    _normalize_sequence_117444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 13), _ni_support_117443, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 345)
    _normalize_sequence_call_result_117449 = invoke(stypy.reporting.localization.Localization(__file__, 345, 13), _normalize_sequence_117444, *[order_117445, ndim_117447], **kwargs_117448)
    
    # Assigning a type to the variable 'orders' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'orders', _normalize_sequence_call_result_117449)
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to _normalize_sequence(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'sigma' (line 346)
    sigma_117452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 45), 'sigma', False)
    # Getting the type of 'input' (line 346)
    input_117453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 52), 'input', False)
    # Obtaining the member 'ndim' of a type (line 346)
    ndim_117454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 52), input_117453, 'ndim')
    # Processing the call keyword arguments (line 346)
    kwargs_117455 = {}
    # Getting the type of '_ni_support' (line 346)
    _ni_support_117450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 346)
    _normalize_sequence_117451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 13), _ni_support_117450, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 346)
    _normalize_sequence_call_result_117456 = invoke(stypy.reporting.localization.Localization(__file__, 346, 13), _normalize_sequence_117451, *[sigma_117452, ndim_117454], **kwargs_117455)
    
    # Assigning a type to the variable 'sigmas' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'sigmas', _normalize_sequence_call_result_117456)
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to _normalize_sequence(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'mode' (line 347)
    mode_117459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 44), 'mode', False)
    # Getting the type of 'input' (line 347)
    input_117460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 347)
    ndim_117461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 50), input_117460, 'ndim')
    # Processing the call keyword arguments (line 347)
    kwargs_117462 = {}
    # Getting the type of '_ni_support' (line 347)
    _ni_support_117457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 347)
    _normalize_sequence_117458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), _ni_support_117457, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 347)
    _normalize_sequence_call_result_117463 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), _normalize_sequence_117458, *[mode_117459, ndim_117461], **kwargs_117462)
    
    # Assigning a type to the variable 'modes' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'modes', _normalize_sequence_call_result_117463)
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to list(...): (line 348)
    # Processing the call arguments (line 348)
    
    # Call to range(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'input' (line 348)
    input_117466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 22), 'input', False)
    # Obtaining the member 'ndim' of a type (line 348)
    ndim_117467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 22), input_117466, 'ndim')
    # Processing the call keyword arguments (line 348)
    kwargs_117468 = {}
    # Getting the type of 'range' (line 348)
    range_117465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'range', False)
    # Calling range(args, kwargs) (line 348)
    range_call_result_117469 = invoke(stypy.reporting.localization.Localization(__file__, 348, 16), range_117465, *[ndim_117467], **kwargs_117468)
    
    # Processing the call keyword arguments (line 348)
    kwargs_117470 = {}
    # Getting the type of 'list' (line 348)
    list_117464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'list', False)
    # Calling list(args, kwargs) (line 348)
    list_call_result_117471 = invoke(stypy.reporting.localization.Localization(__file__, 348, 11), list_117464, *[range_call_result_117469], **kwargs_117470)
    
    # Assigning a type to the variable 'axes' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'axes', list_call_result_117471)
    
    # Assigning a ListComp to a Name (line 349):
    
    # Assigning a ListComp to a Name (line 349):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 350)
    # Processing the call arguments (line 350)
    
    # Call to len(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'axes' (line 350)
    axes_117497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 32), 'axes', False)
    # Processing the call keyword arguments (line 350)
    kwargs_117498 = {}
    # Getting the type of 'len' (line 350)
    len_117496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 28), 'len', False)
    # Calling len(args, kwargs) (line 350)
    len_call_result_117499 = invoke(stypy.reporting.localization.Localization(__file__, 350, 28), len_117496, *[axes_117497], **kwargs_117498)
    
    # Processing the call keyword arguments (line 350)
    kwargs_117500 = {}
    # Getting the type of 'range' (line 350)
    range_117495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'range', False)
    # Calling range(args, kwargs) (line 350)
    range_call_result_117501 = invoke(stypy.reporting.localization.Localization(__file__, 350, 22), range_117495, *[len_call_result_117499], **kwargs_117500)
    
    comprehension_117502 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 12), range_call_result_117501)
    # Assigning a type to the variable 'ii' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'ii', comprehension_117502)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 350)
    ii_117489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'ii')
    # Getting the type of 'sigmas' (line 350)
    sigmas_117490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 42), 'sigmas')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___117491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 42), sigmas_117490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_117492 = invoke(stypy.reporting.localization.Localization(__file__, 350, 42), getitem___117491, ii_117489)
    
    float_117493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 55), 'float')
    # Applying the binary operator '>' (line 350)
    result_gt_117494 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 42), '>', subscript_call_result_117492, float_117493)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 349)
    tuple_117472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 349)
    # Adding element type (line 349)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 349)
    ii_117473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'ii')
    # Getting the type of 'axes' (line 349)
    axes_117474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'axes')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___117475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 13), axes_117474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_117476 = invoke(stypy.reporting.localization.Localization(__file__, 349, 13), getitem___117475, ii_117473)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), tuple_117472, subscript_call_result_117476)
    # Adding element type (line 349)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 349)
    ii_117477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 30), 'ii')
    # Getting the type of 'sigmas' (line 349)
    sigmas_117478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'sigmas')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___117479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), sigmas_117478, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_117480 = invoke(stypy.reporting.localization.Localization(__file__, 349, 23), getitem___117479, ii_117477)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), tuple_117472, subscript_call_result_117480)
    # Adding element type (line 349)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 349)
    ii_117481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'ii')
    # Getting the type of 'orders' (line 349)
    orders_117482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'orders')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___117483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 35), orders_117482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_117484 = invoke(stypy.reporting.localization.Localization(__file__, 349, 35), getitem___117483, ii_117481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), tuple_117472, subscript_call_result_117484)
    # Adding element type (line 349)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 349)
    ii_117485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 53), 'ii')
    # Getting the type of 'modes' (line 349)
    modes_117486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 47), 'modes')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___117487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 47), modes_117486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_117488 = invoke(stypy.reporting.localization.Localization(__file__, 349, 47), getitem___117487, ii_117485)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 13), tuple_117472, subscript_call_result_117488)
    
    list_117503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 12), list_117503, tuple_117472)
    # Assigning a type to the variable 'axes' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'axes', list_117503)
    
    
    
    # Call to len(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'axes' (line 351)
    axes_117505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'axes', False)
    # Processing the call keyword arguments (line 351)
    kwargs_117506 = {}
    # Getting the type of 'len' (line 351)
    len_117504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 7), 'len', False)
    # Calling len(args, kwargs) (line 351)
    len_call_result_117507 = invoke(stypy.reporting.localization.Localization(__file__, 351, 7), len_117504, *[axes_117505], **kwargs_117506)
    
    int_117508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 19), 'int')
    # Applying the binary operator '>' (line 351)
    result_gt_117509 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 7), '>', len_call_result_117507, int_117508)
    
    # Testing the type of an if condition (line 351)
    if_condition_117510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 4), result_gt_117509)
    # Assigning a type to the variable 'if_condition_117510' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'if_condition_117510', if_condition_117510)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 352)
    axes_117511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 40), 'axes')
    # Testing the type of a for loop iterable (line 352)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 352, 8), axes_117511)
    # Getting the type of the for loop variable (line 352)
    for_loop_var_117512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 352, 8), axes_117511)
    # Assigning a type to the variable 'axis' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), for_loop_var_117512))
    # Assigning a type to the variable 'sigma' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'sigma', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), for_loop_var_117512))
    # Assigning a type to the variable 'order' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'order', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), for_loop_var_117512))
    # Assigning a type to the variable 'mode' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'mode', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 8), for_loop_var_117512))
    # SSA begins for a for statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to gaussian_filter1d(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'input' (line 353)
    input_117514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'input', False)
    # Getting the type of 'sigma' (line 353)
    sigma_117515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 37), 'sigma', False)
    # Getting the type of 'axis' (line 353)
    axis_117516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 44), 'axis', False)
    # Getting the type of 'order' (line 353)
    order_117517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 50), 'order', False)
    # Getting the type of 'output' (line 353)
    output_117518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 57), 'output', False)
    # Getting the type of 'mode' (line 354)
    mode_117519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 30), 'mode', False)
    # Getting the type of 'cval' (line 354)
    cval_117520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 36), 'cval', False)
    # Getting the type of 'truncate' (line 354)
    truncate_117521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 42), 'truncate', False)
    # Processing the call keyword arguments (line 353)
    kwargs_117522 = {}
    # Getting the type of 'gaussian_filter1d' (line 353)
    gaussian_filter1d_117513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'gaussian_filter1d', False)
    # Calling gaussian_filter1d(args, kwargs) (line 353)
    gaussian_filter1d_call_result_117523 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), gaussian_filter1d_117513, *[input_117514, sigma_117515, axis_117516, order_117517, output_117518, mode_117519, cval_117520, truncate_117521], **kwargs_117522)
    
    
    # Assigning a Name to a Name (line 355):
    
    # Assigning a Name to a Name (line 355):
    # Getting the type of 'output' (line 355)
    output_117524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'output')
    # Assigning a type to the variable 'input' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'input', output_117524)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 351)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 357):
    
    # Assigning a Subscript to a Subscript (line 357):
    
    # Obtaining the type of the subscript
    Ellipsis_117525 = Ellipsis
    # Getting the type of 'input' (line 357)
    input_117526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 357)
    getitem___117527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 22), input_117526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 357)
    subscript_call_result_117528 = invoke(stypy.reporting.localization.Localization(__file__, 357, 22), getitem___117527, Ellipsis_117525)
    
    # Getting the type of 'output' (line 357)
    output_117529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'output')
    Ellipsis_117530 = Ellipsis
    # Storing an element on a container (line 357)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 8), output_117529, (Ellipsis_117530, subscript_call_result_117528))
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 358)
    return_value_117531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type', return_value_117531)
    
    # ################# End of 'gaussian_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gaussian_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 275)
    stypy_return_type_117532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117532)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gaussian_filter'
    return stypy_return_type_117532

# Assigning a type to the variable 'gaussian_filter' (line 275)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'gaussian_filter', gaussian_filter)

@norecursion
def prewitt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'int')
    # Getting the type of 'None' (line 362)
    None_117534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'None')
    str_117535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 46), 'str', 'reflect')
    float_117536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 62), 'float')
    defaults = [int_117533, None_117534, str_117535, float_117536]
    # Create a new context for function 'prewitt'
    module_type_store = module_type_store.open_function_context('prewitt', 361, 0, False)
    
    # Passed parameters checking function
    prewitt.stypy_localization = localization
    prewitt.stypy_type_of_self = None
    prewitt.stypy_type_store = module_type_store
    prewitt.stypy_function_name = 'prewitt'
    prewitt.stypy_param_names_list = ['input', 'axis', 'output', 'mode', 'cval']
    prewitt.stypy_varargs_param_name = None
    prewitt.stypy_kwargs_param_name = None
    prewitt.stypy_call_defaults = defaults
    prewitt.stypy_call_varargs = varargs
    prewitt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'prewitt', ['input', 'axis', 'output', 'mode', 'cval'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'prewitt', localization, ['input', 'axis', 'output', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'prewitt(...)' code ##################

    str_117537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, (-1)), 'str', 'Calculate a Prewitt filter.\n\n    Parameters\n    ----------\n    %(input)s\n    %(axis)s\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.prewitt(ascent)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to asarray(...): (line 387)
    # Processing the call arguments (line 387)
    # Getting the type of 'input' (line 387)
    input_117540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 26), 'input', False)
    # Processing the call keyword arguments (line 387)
    kwargs_117541 = {}
    # Getting the type of 'numpy' (line 387)
    numpy_117538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 387)
    asarray_117539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), numpy_117538, 'asarray')
    # Calling asarray(args, kwargs) (line 387)
    asarray_call_result_117542 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), asarray_117539, *[input_117540], **kwargs_117541)
    
    # Assigning a type to the variable 'input' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'input', asarray_call_result_117542)
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to _check_axis(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'axis' (line 388)
    axis_117545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 'axis', False)
    # Getting the type of 'input' (line 388)
    input_117546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 388)
    ndim_117547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 41), input_117546, 'ndim')
    # Processing the call keyword arguments (line 388)
    kwargs_117548 = {}
    # Getting the type of '_ni_support' (line 388)
    _ni_support_117543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 388)
    _check_axis_117544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 11), _ni_support_117543, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 388)
    _check_axis_call_result_117549 = invoke(stypy.reporting.localization.Localization(__file__, 388, 11), _check_axis_117544, *[axis_117545, ndim_117547], **kwargs_117548)
    
    # Assigning a type to the variable 'axis' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'axis', _check_axis_call_result_117549)
    
    # Assigning a Call to a Tuple (line 389):
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_117550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'int')
    
    # Call to _get_output(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'output' (line 389)
    output_117553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 51), 'output', False)
    # Getting the type of 'input' (line 389)
    input_117554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 59), 'input', False)
    # Processing the call keyword arguments (line 389)
    kwargs_117555 = {}
    # Getting the type of '_ni_support' (line 389)
    _ni_support_117551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 389)
    _get_output_117552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 27), _ni_support_117551, '_get_output')
    # Calling _get_output(args, kwargs) (line 389)
    _get_output_call_result_117556 = invoke(stypy.reporting.localization.Localization(__file__, 389, 27), _get_output_117552, *[output_117553, input_117554], **kwargs_117555)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___117557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 4), _get_output_call_result_117556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_117558 = invoke(stypy.reporting.localization.Localization(__file__, 389, 4), getitem___117557, int_117550)
    
    # Assigning a type to the variable 'tuple_var_assignment_117027' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_117027', subscript_call_result_117558)
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_117559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 4), 'int')
    
    # Call to _get_output(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'output' (line 389)
    output_117562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 51), 'output', False)
    # Getting the type of 'input' (line 389)
    input_117563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 59), 'input', False)
    # Processing the call keyword arguments (line 389)
    kwargs_117564 = {}
    # Getting the type of '_ni_support' (line 389)
    _ni_support_117560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 389)
    _get_output_117561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 27), _ni_support_117560, '_get_output')
    # Calling _get_output(args, kwargs) (line 389)
    _get_output_call_result_117565 = invoke(stypy.reporting.localization.Localization(__file__, 389, 27), _get_output_117561, *[output_117562, input_117563], **kwargs_117564)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___117566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 4), _get_output_call_result_117565, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_117567 = invoke(stypy.reporting.localization.Localization(__file__, 389, 4), getitem___117566, int_117559)
    
    # Assigning a type to the variable 'tuple_var_assignment_117028' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_117028', subscript_call_result_117567)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_117027' (line 389)
    tuple_var_assignment_117027_117568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_117027')
    # Assigning a type to the variable 'output' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'output', tuple_var_assignment_117027_117568)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_117028' (line 389)
    tuple_var_assignment_117028_117569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'tuple_var_assignment_117028')
    # Assigning a type to the variable 'return_value' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'return_value', tuple_var_assignment_117028_117569)
    
    # Assigning a Call to a Name (line 390):
    
    # Assigning a Call to a Name (line 390):
    
    # Call to _normalize_sequence(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'mode' (line 390)
    mode_117572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 44), 'mode', False)
    # Getting the type of 'input' (line 390)
    input_117573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 390)
    ndim_117574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 50), input_117573, 'ndim')
    # Processing the call keyword arguments (line 390)
    kwargs_117575 = {}
    # Getting the type of '_ni_support' (line 390)
    _ni_support_117570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 390)
    _normalize_sequence_117571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), _ni_support_117570, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 390)
    _normalize_sequence_call_result_117576 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), _normalize_sequence_117571, *[mode_117572, ndim_117574], **kwargs_117575)
    
    # Assigning a type to the variable 'modes' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'modes', _normalize_sequence_call_result_117576)
    
    # Call to correlate1d(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'input' (line 391)
    input_117578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'input', False)
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_117579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    # Adding element type (line 391)
    int_117580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 23), list_117579, int_117580)
    # Adding element type (line 391)
    int_117581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 23), list_117579, int_117581)
    # Adding element type (line 391)
    int_117582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 23), list_117579, int_117582)
    
    # Getting the type of 'axis' (line 391)
    axis_117583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 35), 'axis', False)
    # Getting the type of 'output' (line 391)
    output_117584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 41), 'output', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 391)
    axis_117585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 55), 'axis', False)
    # Getting the type of 'modes' (line 391)
    modes_117586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 49), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___117587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 49), modes_117586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 391)
    subscript_call_result_117588 = invoke(stypy.reporting.localization.Localization(__file__, 391, 49), getitem___117587, axis_117585)
    
    # Getting the type of 'cval' (line 391)
    cval_117589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 62), 'cval', False)
    int_117590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 68), 'int')
    # Processing the call keyword arguments (line 391)
    kwargs_117591 = {}
    # Getting the type of 'correlate1d' (line 391)
    correlate1d_117577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 391)
    correlate1d_call_result_117592 = invoke(stypy.reporting.localization.Localization(__file__, 391, 4), correlate1d_117577, *[input_117578, list_117579, axis_117583, output_117584, subscript_call_result_117588, cval_117589, int_117590], **kwargs_117591)
    
    
    # Assigning a ListComp to a Name (line 392):
    
    # Assigning a ListComp to a Name (line 392):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'input' (line 392)
    input_117598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'input', False)
    # Obtaining the member 'ndim' of a type (line 392)
    ndim_117599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 31), input_117598, 'ndim')
    # Processing the call keyword arguments (line 392)
    kwargs_117600 = {}
    # Getting the type of 'range' (line 392)
    range_117597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'range', False)
    # Calling range(args, kwargs) (line 392)
    range_call_result_117601 = invoke(stypy.reporting.localization.Localization(__file__, 392, 25), range_117597, *[ndim_117599], **kwargs_117600)
    
    comprehension_117602 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 12), range_call_result_117601)
    # Assigning a type to the variable 'ii' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'ii', comprehension_117602)
    
    # Getting the type of 'ii' (line 392)
    ii_117594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 46), 'ii')
    # Getting the type of 'axis' (line 392)
    axis_117595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 52), 'axis')
    # Applying the binary operator '!=' (line 392)
    result_ne_117596 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 46), '!=', ii_117594, axis_117595)
    
    # Getting the type of 'ii' (line 392)
    ii_117593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'ii')
    list_117603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 12), list_117603, ii_117593)
    # Assigning a type to the variable 'axes' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'axes', list_117603)
    
    # Getting the type of 'axes' (line 393)
    axes_117604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 14), 'axes')
    # Testing the type of a for loop iterable (line 393)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 393, 4), axes_117604)
    # Getting the type of the for loop variable (line 393)
    for_loop_var_117605 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 393, 4), axes_117604)
    # Assigning a type to the variable 'ii' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'ii', for_loop_var_117605)
    # SSA begins for a for statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to correlate1d(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'output' (line 394)
    output_117607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 20), 'output', False)
    
    # Obtaining an instance of the builtin type 'list' (line 394)
    list_117608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 394)
    # Adding element type (line 394)
    int_117609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 28), list_117608, int_117609)
    # Adding element type (line 394)
    int_117610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 28), list_117608, int_117610)
    # Adding element type (line 394)
    int_117611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 28), list_117608, int_117611)
    
    # Getting the type of 'ii' (line 394)
    ii_117612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 39), 'ii', False)
    # Getting the type of 'output' (line 394)
    output_117613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 43), 'output', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 394)
    ii_117614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 57), 'ii', False)
    # Getting the type of 'modes' (line 394)
    modes_117615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 51), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___117616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 51), modes_117615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_117617 = invoke(stypy.reporting.localization.Localization(__file__, 394, 51), getitem___117616, ii_117614)
    
    # Getting the type of 'cval' (line 394)
    cval_117618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 62), 'cval', False)
    int_117619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 68), 'int')
    # Processing the call keyword arguments (line 394)
    kwargs_117620 = {}
    # Getting the type of 'correlate1d' (line 394)
    correlate1d_117606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 394)
    correlate1d_call_result_117621 = invoke(stypy.reporting.localization.Localization(__file__, 394, 8), correlate1d_117606, *[output_117607, list_117608, ii_117612, output_117613, subscript_call_result_117617, cval_117618, int_117619], **kwargs_117620)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 395)
    return_value_117622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type', return_value_117622)
    
    # ################# End of 'prewitt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'prewitt' in the type store
    # Getting the type of 'stypy_return_type' (line 361)
    stypy_return_type_117623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117623)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'prewitt'
    return stypy_return_type_117623

# Assigning a type to the variable 'prewitt' (line 361)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 0), 'prewitt', prewitt)

@norecursion
def sobel(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_117624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 22), 'int')
    # Getting the type of 'None' (line 399)
    None_117625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 33), 'None')
    str_117626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 44), 'str', 'reflect')
    float_117627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 60), 'float')
    defaults = [int_117624, None_117625, str_117626, float_117627]
    # Create a new context for function 'sobel'
    module_type_store = module_type_store.open_function_context('sobel', 398, 0, False)
    
    # Passed parameters checking function
    sobel.stypy_localization = localization
    sobel.stypy_type_of_self = None
    sobel.stypy_type_store = module_type_store
    sobel.stypy_function_name = 'sobel'
    sobel.stypy_param_names_list = ['input', 'axis', 'output', 'mode', 'cval']
    sobel.stypy_varargs_param_name = None
    sobel.stypy_kwargs_param_name = None
    sobel.stypy_call_defaults = defaults
    sobel.stypy_call_varargs = varargs
    sobel.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sobel', ['input', 'axis', 'output', 'mode', 'cval'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sobel', localization, ['input', 'axis', 'output', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sobel(...)' code ##################

    str_117628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, (-1)), 'str', 'Calculate a Sobel filter.\n\n    Parameters\n    ----------\n    %(input)s\n    %(axis)s\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.sobel(ascent)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to asarray(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'input' (line 424)
    input_117631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 26), 'input', False)
    # Processing the call keyword arguments (line 424)
    kwargs_117632 = {}
    # Getting the type of 'numpy' (line 424)
    numpy_117629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 424)
    asarray_117630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), numpy_117629, 'asarray')
    # Calling asarray(args, kwargs) (line 424)
    asarray_call_result_117633 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), asarray_117630, *[input_117631], **kwargs_117632)
    
    # Assigning a type to the variable 'input' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'input', asarray_call_result_117633)
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to _check_axis(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'axis' (line 425)
    axis_117636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 35), 'axis', False)
    # Getting the type of 'input' (line 425)
    input_117637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 425)
    ndim_117638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 41), input_117637, 'ndim')
    # Processing the call keyword arguments (line 425)
    kwargs_117639 = {}
    # Getting the type of '_ni_support' (line 425)
    _ni_support_117634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 425)
    _check_axis_117635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 11), _ni_support_117634, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 425)
    _check_axis_call_result_117640 = invoke(stypy.reporting.localization.Localization(__file__, 425, 11), _check_axis_117635, *[axis_117636, ndim_117638], **kwargs_117639)
    
    # Assigning a type to the variable 'axis' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'axis', _check_axis_call_result_117640)
    
    # Assigning a Call to a Tuple (line 426):
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_117641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 4), 'int')
    
    # Call to _get_output(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'output' (line 426)
    output_117644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 51), 'output', False)
    # Getting the type of 'input' (line 426)
    input_117645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 59), 'input', False)
    # Processing the call keyword arguments (line 426)
    kwargs_117646 = {}
    # Getting the type of '_ni_support' (line 426)
    _ni_support_117642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 426)
    _get_output_117643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 27), _ni_support_117642, '_get_output')
    # Calling _get_output(args, kwargs) (line 426)
    _get_output_call_result_117647 = invoke(stypy.reporting.localization.Localization(__file__, 426, 27), _get_output_117643, *[output_117644, input_117645], **kwargs_117646)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___117648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 4), _get_output_call_result_117647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_117649 = invoke(stypy.reporting.localization.Localization(__file__, 426, 4), getitem___117648, int_117641)
    
    # Assigning a type to the variable 'tuple_var_assignment_117029' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'tuple_var_assignment_117029', subscript_call_result_117649)
    
    # Assigning a Subscript to a Name (line 426):
    
    # Obtaining the type of the subscript
    int_117650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 4), 'int')
    
    # Call to _get_output(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'output' (line 426)
    output_117653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 51), 'output', False)
    # Getting the type of 'input' (line 426)
    input_117654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 59), 'input', False)
    # Processing the call keyword arguments (line 426)
    kwargs_117655 = {}
    # Getting the type of '_ni_support' (line 426)
    _ni_support_117651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 426)
    _get_output_117652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 27), _ni_support_117651, '_get_output')
    # Calling _get_output(args, kwargs) (line 426)
    _get_output_call_result_117656 = invoke(stypy.reporting.localization.Localization(__file__, 426, 27), _get_output_117652, *[output_117653, input_117654], **kwargs_117655)
    
    # Obtaining the member '__getitem__' of a type (line 426)
    getitem___117657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 4), _get_output_call_result_117656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 426)
    subscript_call_result_117658 = invoke(stypy.reporting.localization.Localization(__file__, 426, 4), getitem___117657, int_117650)
    
    # Assigning a type to the variable 'tuple_var_assignment_117030' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'tuple_var_assignment_117030', subscript_call_result_117658)
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'tuple_var_assignment_117029' (line 426)
    tuple_var_assignment_117029_117659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'tuple_var_assignment_117029')
    # Assigning a type to the variable 'output' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'output', tuple_var_assignment_117029_117659)
    
    # Assigning a Name to a Name (line 426):
    # Getting the type of 'tuple_var_assignment_117030' (line 426)
    tuple_var_assignment_117030_117660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'tuple_var_assignment_117030')
    # Assigning a type to the variable 'return_value' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'return_value', tuple_var_assignment_117030_117660)
    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to _normalize_sequence(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'mode' (line 427)
    mode_117663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 44), 'mode', False)
    # Getting the type of 'input' (line 427)
    input_117664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 427)
    ndim_117665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 50), input_117664, 'ndim')
    # Processing the call keyword arguments (line 427)
    kwargs_117666 = {}
    # Getting the type of '_ni_support' (line 427)
    _ni_support_117661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 427)
    _normalize_sequence_117662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 12), _ni_support_117661, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 427)
    _normalize_sequence_call_result_117667 = invoke(stypy.reporting.localization.Localization(__file__, 427, 12), _normalize_sequence_117662, *[mode_117663, ndim_117665], **kwargs_117666)
    
    # Assigning a type to the variable 'modes' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'modes', _normalize_sequence_call_result_117667)
    
    # Call to correlate1d(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'input' (line 428)
    input_117669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 16), 'input', False)
    
    # Obtaining an instance of the builtin type 'list' (line 428)
    list_117670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 428)
    # Adding element type (line 428)
    int_117671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 23), list_117670, int_117671)
    # Adding element type (line 428)
    int_117672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 23), list_117670, int_117672)
    # Adding element type (line 428)
    int_117673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 428, 23), list_117670, int_117673)
    
    # Getting the type of 'axis' (line 428)
    axis_117674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 35), 'axis', False)
    # Getting the type of 'output' (line 428)
    output_117675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 41), 'output', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 428)
    axis_117676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 55), 'axis', False)
    # Getting the type of 'modes' (line 428)
    modes_117677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 49), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___117678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 49), modes_117677, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_117679 = invoke(stypy.reporting.localization.Localization(__file__, 428, 49), getitem___117678, axis_117676)
    
    # Getting the type of 'cval' (line 428)
    cval_117680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 62), 'cval', False)
    int_117681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 68), 'int')
    # Processing the call keyword arguments (line 428)
    kwargs_117682 = {}
    # Getting the type of 'correlate1d' (line 428)
    correlate1d_117668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 428)
    correlate1d_call_result_117683 = invoke(stypy.reporting.localization.Localization(__file__, 428, 4), correlate1d_117668, *[input_117669, list_117670, axis_117674, output_117675, subscript_call_result_117679, cval_117680, int_117681], **kwargs_117682)
    
    
    # Assigning a ListComp to a Name (line 429):
    
    # Assigning a ListComp to a Name (line 429):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'input' (line 429)
    input_117689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 31), 'input', False)
    # Obtaining the member 'ndim' of a type (line 429)
    ndim_117690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 31), input_117689, 'ndim')
    # Processing the call keyword arguments (line 429)
    kwargs_117691 = {}
    # Getting the type of 'range' (line 429)
    range_117688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 25), 'range', False)
    # Calling range(args, kwargs) (line 429)
    range_call_result_117692 = invoke(stypy.reporting.localization.Localization(__file__, 429, 25), range_117688, *[ndim_117690], **kwargs_117691)
    
    comprehension_117693 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 12), range_call_result_117692)
    # Assigning a type to the variable 'ii' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'ii', comprehension_117693)
    
    # Getting the type of 'ii' (line 429)
    ii_117685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 46), 'ii')
    # Getting the type of 'axis' (line 429)
    axis_117686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 52), 'axis')
    # Applying the binary operator '!=' (line 429)
    result_ne_117687 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 46), '!=', ii_117685, axis_117686)
    
    # Getting the type of 'ii' (line 429)
    ii_117684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'ii')
    list_117694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 12), list_117694, ii_117684)
    # Assigning a type to the variable 'axes' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'axes', list_117694)
    
    # Getting the type of 'axes' (line 430)
    axes_117695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 14), 'axes')
    # Testing the type of a for loop iterable (line 430)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 430, 4), axes_117695)
    # Getting the type of the for loop variable (line 430)
    for_loop_var_117696 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 430, 4), axes_117695)
    # Assigning a type to the variable 'ii' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'ii', for_loop_var_117696)
    # SSA begins for a for statement (line 430)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to correlate1d(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'output' (line 431)
    output_117698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 20), 'output', False)
    
    # Obtaining an instance of the builtin type 'list' (line 431)
    list_117699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 431)
    # Adding element type (line 431)
    int_117700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 28), list_117699, int_117700)
    # Adding element type (line 431)
    int_117701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 28), list_117699, int_117701)
    # Adding element type (line 431)
    int_117702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 28), list_117699, int_117702)
    
    # Getting the type of 'ii' (line 431)
    ii_117703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 39), 'ii', False)
    # Getting the type of 'output' (line 431)
    output_117704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 43), 'output', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 431)
    ii_117705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 57), 'ii', False)
    # Getting the type of 'modes' (line 431)
    modes_117706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 51), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 431)
    getitem___117707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 51), modes_117706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 431)
    subscript_call_result_117708 = invoke(stypy.reporting.localization.Localization(__file__, 431, 51), getitem___117707, ii_117705)
    
    # Getting the type of 'cval' (line 431)
    cval_117709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 62), 'cval', False)
    int_117710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 68), 'int')
    # Processing the call keyword arguments (line 431)
    kwargs_117711 = {}
    # Getting the type of 'correlate1d' (line 431)
    correlate1d_117697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'correlate1d', False)
    # Calling correlate1d(args, kwargs) (line 431)
    correlate1d_call_result_117712 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), correlate1d_117697, *[output_117698, list_117699, ii_117703, output_117704, subscript_call_result_117708, cval_117709, int_117710], **kwargs_117711)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 432)
    return_value_117713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'stypy_return_type', return_value_117713)
    
    # ################# End of 'sobel(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sobel' in the type store
    # Getting the type of 'stypy_return_type' (line 398)
    stypy_return_type_117714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117714)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sobel'
    return stypy_return_type_117714

# Assigning a type to the variable 'sobel' (line 398)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 0), 'sobel', sobel)

@norecursion
def generic_laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 436)
    None_117715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 47), 'None')
    str_117716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 58), 'str', 'reflect')
    float_117717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 25), 'float')
    
    # Obtaining an instance of the builtin type 'tuple' (line 438)
    tuple_117718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 438)
    
    # Getting the type of 'None' (line 439)
    None_117719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 35), 'None')
    defaults = [None_117715, str_117716, float_117717, tuple_117718, None_117719]
    # Create a new context for function 'generic_laplace'
    module_type_store = module_type_store.open_function_context('generic_laplace', 435, 0, False)
    
    # Passed parameters checking function
    generic_laplace.stypy_localization = localization
    generic_laplace.stypy_type_of_self = None
    generic_laplace.stypy_type_store = module_type_store
    generic_laplace.stypy_function_name = 'generic_laplace'
    generic_laplace.stypy_param_names_list = ['input', 'derivative2', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords']
    generic_laplace.stypy_varargs_param_name = None
    generic_laplace.stypy_kwargs_param_name = None
    generic_laplace.stypy_call_defaults = defaults
    generic_laplace.stypy_call_varargs = varargs
    generic_laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generic_laplace', ['input', 'derivative2', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generic_laplace', localization, ['input', 'derivative2', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generic_laplace(...)' code ##################

    str_117720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, (-1)), 'str', '\n    N-dimensional Laplace filter using a provided second derivative function.\n\n    Parameters\n    ----------\n    %(input)s\n    derivative2 : callable\n        Callable with the following signature::\n\n            derivative2(input, axis, output, mode, cval,\n                        *extra_arguments, **extra_keywords)\n\n        See `extra_arguments`, `extra_keywords` below.\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    %(extra_keywords)s\n    %(extra_arguments)s\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 459)
    # Getting the type of 'extra_keywords' (line 459)
    extra_keywords_117721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 7), 'extra_keywords')
    # Getting the type of 'None' (line 459)
    None_117722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'None')
    
    (may_be_117723, more_types_in_union_117724) = may_be_none(extra_keywords_117721, None_117722)

    if may_be_117723:

        if more_types_in_union_117724:
            # Runtime conditional SSA (line 459)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 460):
        
        # Assigning a Dict to a Name (line 460):
        
        # Obtaining an instance of the builtin type 'dict' (line 460)
        dict_117725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 460)
        
        # Assigning a type to the variable 'extra_keywords' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'extra_keywords', dict_117725)

        if more_types_in_union_117724:
            # SSA join for if statement (line 459)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Call to asarray(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'input' (line 461)
    input_117728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 26), 'input', False)
    # Processing the call keyword arguments (line 461)
    kwargs_117729 = {}
    # Getting the type of 'numpy' (line 461)
    numpy_117726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 461)
    asarray_117727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), numpy_117726, 'asarray')
    # Calling asarray(args, kwargs) (line 461)
    asarray_call_result_117730 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), asarray_117727, *[input_117728], **kwargs_117729)
    
    # Assigning a type to the variable 'input' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'input', asarray_call_result_117730)
    
    # Assigning a Call to a Tuple (line 462):
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_117731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 4), 'int')
    
    # Call to _get_output(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'output' (line 462)
    output_117734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 51), 'output', False)
    # Getting the type of 'input' (line 462)
    input_117735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'input', False)
    # Processing the call keyword arguments (line 462)
    kwargs_117736 = {}
    # Getting the type of '_ni_support' (line 462)
    _ni_support_117732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 462)
    _get_output_117733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 27), _ni_support_117732, '_get_output')
    # Calling _get_output(args, kwargs) (line 462)
    _get_output_call_result_117737 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), _get_output_117733, *[output_117734, input_117735], **kwargs_117736)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___117738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 4), _get_output_call_result_117737, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_117739 = invoke(stypy.reporting.localization.Localization(__file__, 462, 4), getitem___117738, int_117731)
    
    # Assigning a type to the variable 'tuple_var_assignment_117031' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'tuple_var_assignment_117031', subscript_call_result_117739)
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_117740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 4), 'int')
    
    # Call to _get_output(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'output' (line 462)
    output_117743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 51), 'output', False)
    # Getting the type of 'input' (line 462)
    input_117744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'input', False)
    # Processing the call keyword arguments (line 462)
    kwargs_117745 = {}
    # Getting the type of '_ni_support' (line 462)
    _ni_support_117741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 462)
    _get_output_117742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 27), _ni_support_117741, '_get_output')
    # Calling _get_output(args, kwargs) (line 462)
    _get_output_call_result_117746 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), _get_output_117742, *[output_117743, input_117744], **kwargs_117745)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___117747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 4), _get_output_call_result_117746, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_117748 = invoke(stypy.reporting.localization.Localization(__file__, 462, 4), getitem___117747, int_117740)
    
    # Assigning a type to the variable 'tuple_var_assignment_117032' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'tuple_var_assignment_117032', subscript_call_result_117748)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_117031' (line 462)
    tuple_var_assignment_117031_117749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'tuple_var_assignment_117031')
    # Assigning a type to the variable 'output' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'output', tuple_var_assignment_117031_117749)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_117032' (line 462)
    tuple_var_assignment_117032_117750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'tuple_var_assignment_117032')
    # Assigning a type to the variable 'return_value' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'return_value', tuple_var_assignment_117032_117750)
    
    # Assigning a Call to a Name (line 463):
    
    # Assigning a Call to a Name (line 463):
    
    # Call to list(...): (line 463)
    # Processing the call arguments (line 463)
    
    # Call to range(...): (line 463)
    # Processing the call arguments (line 463)
    # Getting the type of 'input' (line 463)
    input_117753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 22), 'input', False)
    # Obtaining the member 'ndim' of a type (line 463)
    ndim_117754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 22), input_117753, 'ndim')
    # Processing the call keyword arguments (line 463)
    kwargs_117755 = {}
    # Getting the type of 'range' (line 463)
    range_117752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 16), 'range', False)
    # Calling range(args, kwargs) (line 463)
    range_call_result_117756 = invoke(stypy.reporting.localization.Localization(__file__, 463, 16), range_117752, *[ndim_117754], **kwargs_117755)
    
    # Processing the call keyword arguments (line 463)
    kwargs_117757 = {}
    # Getting the type of 'list' (line 463)
    list_117751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'list', False)
    # Calling list(args, kwargs) (line 463)
    list_call_result_117758 = invoke(stypy.reporting.localization.Localization(__file__, 463, 11), list_117751, *[range_call_result_117756], **kwargs_117757)
    
    # Assigning a type to the variable 'axes' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'axes', list_call_result_117758)
    
    
    
    # Call to len(...): (line 464)
    # Processing the call arguments (line 464)
    # Getting the type of 'axes' (line 464)
    axes_117760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'axes', False)
    # Processing the call keyword arguments (line 464)
    kwargs_117761 = {}
    # Getting the type of 'len' (line 464)
    len_117759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 7), 'len', False)
    # Calling len(args, kwargs) (line 464)
    len_call_result_117762 = invoke(stypy.reporting.localization.Localization(__file__, 464, 7), len_117759, *[axes_117760], **kwargs_117761)
    
    int_117763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 19), 'int')
    # Applying the binary operator '>' (line 464)
    result_gt_117764 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 7), '>', len_call_result_117762, int_117763)
    
    # Testing the type of an if condition (line 464)
    if_condition_117765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 4), result_gt_117764)
    # Assigning a type to the variable 'if_condition_117765' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'if_condition_117765', if_condition_117765)
    # SSA begins for if statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 465):
    
    # Assigning a Call to a Name (line 465):
    
    # Call to _normalize_sequence(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'mode' (line 465)
    mode_117768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 48), 'mode', False)
    
    # Call to len(...): (line 465)
    # Processing the call arguments (line 465)
    # Getting the type of 'axes' (line 465)
    axes_117770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 58), 'axes', False)
    # Processing the call keyword arguments (line 465)
    kwargs_117771 = {}
    # Getting the type of 'len' (line 465)
    len_117769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 54), 'len', False)
    # Calling len(args, kwargs) (line 465)
    len_call_result_117772 = invoke(stypy.reporting.localization.Localization(__file__, 465, 54), len_117769, *[axes_117770], **kwargs_117771)
    
    # Processing the call keyword arguments (line 465)
    kwargs_117773 = {}
    # Getting the type of '_ni_support' (line 465)
    _ni_support_117766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 16), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 465)
    _normalize_sequence_117767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 16), _ni_support_117766, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 465)
    _normalize_sequence_call_result_117774 = invoke(stypy.reporting.localization.Localization(__file__, 465, 16), _normalize_sequence_117767, *[mode_117768, len_call_result_117772], **kwargs_117773)
    
    # Assigning a type to the variable 'modes' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'modes', _normalize_sequence_call_result_117774)
    
    # Call to derivative2(...): (line 466)
    # Processing the call arguments (line 466)
    # Getting the type of 'input' (line 466)
    input_117776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'input', False)
    
    # Obtaining the type of the subscript
    int_117777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 32), 'int')
    # Getting the type of 'axes' (line 466)
    axes_117778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 27), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___117779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 27), axes_117778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_117780 = invoke(stypy.reporting.localization.Localization(__file__, 466, 27), getitem___117779, int_117777)
    
    # Getting the type of 'output' (line 466)
    output_117781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 36), 'output', False)
    
    # Obtaining the type of the subscript
    int_117782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 50), 'int')
    # Getting the type of 'modes' (line 466)
    modes_117783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 44), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 466)
    getitem___117784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 44), modes_117783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 466)
    subscript_call_result_117785 = invoke(stypy.reporting.localization.Localization(__file__, 466, 44), getitem___117784, int_117782)
    
    # Getting the type of 'cval' (line 466)
    cval_117786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 54), 'cval', False)
    # Getting the type of 'extra_arguments' (line 467)
    extra_arguments_117787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'extra_arguments', False)
    # Processing the call keyword arguments (line 466)
    # Getting the type of 'extra_keywords' (line 467)
    extra_keywords_117788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 40), 'extra_keywords', False)
    kwargs_117789 = {'extra_keywords_117788': extra_keywords_117788}
    # Getting the type of 'derivative2' (line 466)
    derivative2_117775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'derivative2', False)
    # Calling derivative2(args, kwargs) (line 466)
    derivative2_call_result_117790 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), derivative2_117775, *[input_117776, subscript_call_result_117780, output_117781, subscript_call_result_117785, cval_117786, extra_arguments_117787], **kwargs_117789)
    
    
    
    # Call to range(...): (line 468)
    # Processing the call arguments (line 468)
    int_117792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 24), 'int')
    
    # Call to len(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'axes' (line 468)
    axes_117794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 31), 'axes', False)
    # Processing the call keyword arguments (line 468)
    kwargs_117795 = {}
    # Getting the type of 'len' (line 468)
    len_117793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'len', False)
    # Calling len(args, kwargs) (line 468)
    len_call_result_117796 = invoke(stypy.reporting.localization.Localization(__file__, 468, 27), len_117793, *[axes_117794], **kwargs_117795)
    
    # Processing the call keyword arguments (line 468)
    kwargs_117797 = {}
    # Getting the type of 'range' (line 468)
    range_117791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 18), 'range', False)
    # Calling range(args, kwargs) (line 468)
    range_call_result_117798 = invoke(stypy.reporting.localization.Localization(__file__, 468, 18), range_117791, *[int_117792, len_call_result_117796], **kwargs_117797)
    
    # Testing the type of a for loop iterable (line 468)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 468, 8), range_call_result_117798)
    # Getting the type of the for loop variable (line 468)
    for_loop_var_117799 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 468, 8), range_call_result_117798)
    # Assigning a type to the variable 'ii' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'ii', for_loop_var_117799)
    # SSA begins for a for statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to derivative2(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'input' (line 469)
    input_117801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 30), 'input', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 469)
    ii_117802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 42), 'ii', False)
    # Getting the type of 'axes' (line 469)
    axes_117803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___117804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 37), axes_117803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 469)
    subscript_call_result_117805 = invoke(stypy.reporting.localization.Localization(__file__, 469, 37), getitem___117804, ii_117802)
    
    # Getting the type of 'output' (line 469)
    output_117806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 47), 'output', False)
    # Obtaining the member 'dtype' of a type (line 469)
    dtype_117807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 47), output_117806, 'dtype')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 469)
    ii_117808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 67), 'ii', False)
    # Getting the type of 'modes' (line 469)
    modes_117809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 61), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 469)
    getitem___117810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 61), modes_117809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 469)
    subscript_call_result_117811 = invoke(stypy.reporting.localization.Localization(__file__, 469, 61), getitem___117810, ii_117808)
    
    # Getting the type of 'cval' (line 469)
    cval_117812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 72), 'cval', False)
    # Getting the type of 'extra_arguments' (line 470)
    extra_arguments_117813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 31), 'extra_arguments', False)
    # Processing the call keyword arguments (line 469)
    # Getting the type of 'extra_keywords' (line 470)
    extra_keywords_117814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 50), 'extra_keywords', False)
    kwargs_117815 = {'extra_keywords_117814': extra_keywords_117814}
    # Getting the type of 'derivative2' (line 469)
    derivative2_117800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 18), 'derivative2', False)
    # Calling derivative2(args, kwargs) (line 469)
    derivative2_call_result_117816 = invoke(stypy.reporting.localization.Localization(__file__, 469, 18), derivative2_117800, *[input_117801, subscript_call_result_117805, dtype_117807, subscript_call_result_117811, cval_117812, extra_arguments_117813], **kwargs_117815)
    
    # Assigning a type to the variable 'tmp' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 12), 'tmp', derivative2_call_result_117816)
    
    # Getting the type of 'output' (line 471)
    output_117817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'output')
    # Getting the type of 'tmp' (line 471)
    tmp_117818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'tmp')
    # Applying the binary operator '+=' (line 471)
    result_iadd_117819 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 12), '+=', output_117817, tmp_117818)
    # Assigning a type to the variable 'output' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'output', result_iadd_117819)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 464)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 473):
    
    # Assigning a Subscript to a Subscript (line 473):
    
    # Obtaining the type of the subscript
    Ellipsis_117820 = Ellipsis
    # Getting the type of 'input' (line 473)
    input_117821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 473)
    getitem___117822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 22), input_117821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 473)
    subscript_call_result_117823 = invoke(stypy.reporting.localization.Localization(__file__, 473, 22), getitem___117822, Ellipsis_117820)
    
    # Getting the type of 'output' (line 473)
    output_117824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'output')
    Ellipsis_117825 = Ellipsis
    # Storing an element on a container (line 473)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 473, 8), output_117824, (Ellipsis_117825, subscript_call_result_117823))
    # SSA join for if statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 474)
    return_value_117826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type', return_value_117826)
    
    # ################# End of 'generic_laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generic_laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 435)
    stypy_return_type_117827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117827)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generic_laplace'
    return stypy_return_type_117827

# Assigning a type to the variable 'generic_laplace' (line 435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 0), 'generic_laplace', generic_laplace)

@norecursion
def laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 478)
    None_117828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 26), 'None')
    str_117829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 37), 'str', 'reflect')
    float_117830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 53), 'float')
    defaults = [None_117828, str_117829, float_117830]
    # Create a new context for function 'laplace'
    module_type_store = module_type_store.open_function_context('laplace', 477, 0, False)
    
    # Passed parameters checking function
    laplace.stypy_localization = localization
    laplace.stypy_type_of_self = None
    laplace.stypy_type_store = module_type_store
    laplace.stypy_function_name = 'laplace'
    laplace.stypy_param_names_list = ['input', 'output', 'mode', 'cval']
    laplace.stypy_varargs_param_name = None
    laplace.stypy_kwargs_param_name = None
    laplace.stypy_call_defaults = defaults
    laplace.stypy_call_varargs = varargs
    laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'laplace', ['input', 'output', 'mode', 'cval'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'laplace', localization, ['input', 'output', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'laplace(...)' code ##################

    str_117831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, (-1)), 'str', 'N-dimensional Laplace filter based on approximate second derivatives.\n\n    Parameters\n    ----------\n    %(input)s\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.laplace(ascent)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')

    @norecursion
    def derivative2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'derivative2'
        module_type_store = module_type_store.open_function_context('derivative2', 502, 4, False)
        
        # Passed parameters checking function
        derivative2.stypy_localization = localization
        derivative2.stypy_type_of_self = None
        derivative2.stypy_type_store = module_type_store
        derivative2.stypy_function_name = 'derivative2'
        derivative2.stypy_param_names_list = ['input', 'axis', 'output', 'mode', 'cval']
        derivative2.stypy_varargs_param_name = None
        derivative2.stypy_kwargs_param_name = None
        derivative2.stypy_call_defaults = defaults
        derivative2.stypy_call_varargs = varargs
        derivative2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'derivative2', ['input', 'axis', 'output', 'mode', 'cval'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivative2', localization, ['input', 'axis', 'output', 'mode', 'cval'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivative2(...)' code ##################

        
        # Call to correlate1d(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'input' (line 503)
        input_117833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 27), 'input', False)
        
        # Obtaining an instance of the builtin type 'list' (line 503)
        list_117834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 503)
        # Adding element type (line 503)
        int_117835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 34), list_117834, int_117835)
        # Adding element type (line 503)
        int_117836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 34), list_117834, int_117836)
        # Adding element type (line 503)
        int_117837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 34), list_117834, int_117837)
        
        # Getting the type of 'axis' (line 503)
        axis_117838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 46), 'axis', False)
        # Getting the type of 'output' (line 503)
        output_117839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 52), 'output', False)
        # Getting the type of 'mode' (line 503)
        mode_117840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 60), 'mode', False)
        # Getting the type of 'cval' (line 503)
        cval_117841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 66), 'cval', False)
        int_117842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 72), 'int')
        # Processing the call keyword arguments (line 503)
        kwargs_117843 = {}
        # Getting the type of 'correlate1d' (line 503)
        correlate1d_117832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 15), 'correlate1d', False)
        # Calling correlate1d(args, kwargs) (line 503)
        correlate1d_call_result_117844 = invoke(stypy.reporting.localization.Localization(__file__, 503, 15), correlate1d_117832, *[input_117833, list_117834, axis_117838, output_117839, mode_117840, cval_117841, int_117842], **kwargs_117843)
        
        # Assigning a type to the variable 'stypy_return_type' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'stypy_return_type', correlate1d_call_result_117844)
        
        # ################# End of 'derivative2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivative2' in the type store
        # Getting the type of 'stypy_return_type' (line 502)
        stypy_return_type_117845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_117845)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivative2'
        return stypy_return_type_117845

    # Assigning a type to the variable 'derivative2' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'derivative2', derivative2)
    
    # Call to generic_laplace(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'input' (line 504)
    input_117847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 27), 'input', False)
    # Getting the type of 'derivative2' (line 504)
    derivative2_117848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 34), 'derivative2', False)
    # Getting the type of 'output' (line 504)
    output_117849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 47), 'output', False)
    # Getting the type of 'mode' (line 504)
    mode_117850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 55), 'mode', False)
    # Getting the type of 'cval' (line 504)
    cval_117851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 61), 'cval', False)
    # Processing the call keyword arguments (line 504)
    kwargs_117852 = {}
    # Getting the type of 'generic_laplace' (line 504)
    generic_laplace_117846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), 'generic_laplace', False)
    # Calling generic_laplace(args, kwargs) (line 504)
    generic_laplace_call_result_117853 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), generic_laplace_117846, *[input_117847, derivative2_117848, output_117849, mode_117850, cval_117851], **kwargs_117852)
    
    # Assigning a type to the variable 'stypy_return_type' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type', generic_laplace_call_result_117853)
    
    # ################# End of 'laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 477)
    stypy_return_type_117854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117854)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'laplace'
    return stypy_return_type_117854

# Assigning a type to the variable 'laplace' (line 477)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'laplace', laplace)

@norecursion
def gaussian_laplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 508)
    None_117855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 42), 'None')
    str_117856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 53), 'str', 'reflect')
    float_117857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 26), 'float')
    defaults = [None_117855, str_117856, float_117857]
    # Create a new context for function 'gaussian_laplace'
    module_type_store = module_type_store.open_function_context('gaussian_laplace', 507, 0, False)
    
    # Passed parameters checking function
    gaussian_laplace.stypy_localization = localization
    gaussian_laplace.stypy_type_of_self = None
    gaussian_laplace.stypy_type_store = module_type_store
    gaussian_laplace.stypy_function_name = 'gaussian_laplace'
    gaussian_laplace.stypy_param_names_list = ['input', 'sigma', 'output', 'mode', 'cval']
    gaussian_laplace.stypy_varargs_param_name = None
    gaussian_laplace.stypy_kwargs_param_name = 'kwargs'
    gaussian_laplace.stypy_call_defaults = defaults
    gaussian_laplace.stypy_call_varargs = varargs
    gaussian_laplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gaussian_laplace', ['input', 'sigma', 'output', 'mode', 'cval'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gaussian_laplace', localization, ['input', 'sigma', 'output', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gaussian_laplace(...)' code ##################

    str_117858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, (-1)), 'str', 'Multidimensional Laplace filter using gaussian second derivatives.\n\n    Parameters\n    ----------\n    %(input)s\n    sigma : scalar or sequence of scalars\n        The standard deviations of the Gaussian filter are given for\n        each axis as a sequence, or as a single number, in which case\n        it is equal for all axes.\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    Extra keyword arguments will be passed to gaussian_filter().\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> ascent = misc.ascent()\n\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n\n    >>> result = ndimage.gaussian_laplace(ascent, sigma=1)\n    >>> ax1.imshow(result)\n\n    >>> result = ndimage.gaussian_laplace(ascent, sigma=3)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 542):
    
    # Assigning a Call to a Name (line 542):
    
    # Call to asarray(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 'input' (line 542)
    input_117861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 26), 'input', False)
    # Processing the call keyword arguments (line 542)
    kwargs_117862 = {}
    # Getting the type of 'numpy' (line 542)
    numpy_117859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 542)
    asarray_117860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 12), numpy_117859, 'asarray')
    # Calling asarray(args, kwargs) (line 542)
    asarray_call_result_117863 = invoke(stypy.reporting.localization.Localization(__file__, 542, 12), asarray_117860, *[input_117861], **kwargs_117862)
    
    # Assigning a type to the variable 'input' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'input', asarray_call_result_117863)

    @norecursion
    def derivative2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'derivative2'
        module_type_store = module_type_store.open_function_context('derivative2', 544, 4, False)
        
        # Passed parameters checking function
        derivative2.stypy_localization = localization
        derivative2.stypy_type_of_self = None
        derivative2.stypy_type_store = module_type_store
        derivative2.stypy_function_name = 'derivative2'
        derivative2.stypy_param_names_list = ['input', 'axis', 'output', 'mode', 'cval', 'sigma']
        derivative2.stypy_varargs_param_name = None
        derivative2.stypy_kwargs_param_name = 'kwargs'
        derivative2.stypy_call_defaults = defaults
        derivative2.stypy_call_varargs = varargs
        derivative2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'derivative2', ['input', 'axis', 'output', 'mode', 'cval', 'sigma'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivative2', localization, ['input', 'axis', 'output', 'mode', 'cval', 'sigma'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivative2(...)' code ##################

        
        # Assigning a BinOp to a Name (line 545):
        
        # Assigning a BinOp to a Name (line 545):
        
        # Obtaining an instance of the builtin type 'list' (line 545)
        list_117864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 545)
        # Adding element type (line 545)
        int_117865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 16), list_117864, int_117865)
        
        # Getting the type of 'input' (line 545)
        input_117866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 22), 'input')
        # Obtaining the member 'ndim' of a type (line 545)
        ndim_117867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 22), input_117866, 'ndim')
        # Applying the binary operator '*' (line 545)
        result_mul_117868 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 16), '*', list_117864, ndim_117867)
        
        # Assigning a type to the variable 'order' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'order', result_mul_117868)
        
        # Assigning a Num to a Subscript (line 546):
        
        # Assigning a Num to a Subscript (line 546):
        int_117869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 22), 'int')
        # Getting the type of 'order' (line 546)
        order_117870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'order')
        # Getting the type of 'axis' (line 546)
        axis_117871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 14), 'axis')
        # Storing an element on a container (line 546)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 8), order_117870, (axis_117871, int_117869))
        
        # Call to gaussian_filter(...): (line 547)
        # Processing the call arguments (line 547)
        # Getting the type of 'input' (line 547)
        input_117873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 31), 'input', False)
        # Getting the type of 'sigma' (line 547)
        sigma_117874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 38), 'sigma', False)
        # Getting the type of 'order' (line 547)
        order_117875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 45), 'order', False)
        # Getting the type of 'output' (line 547)
        output_117876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 52), 'output', False)
        # Getting the type of 'mode' (line 547)
        mode_117877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 60), 'mode', False)
        # Getting the type of 'cval' (line 547)
        cval_117878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 66), 'cval', False)
        # Processing the call keyword arguments (line 547)
        # Getting the type of 'kwargs' (line 548)
        kwargs_117879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 33), 'kwargs', False)
        kwargs_117880 = {'kwargs_117879': kwargs_117879}
        # Getting the type of 'gaussian_filter' (line 547)
        gaussian_filter_117872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'gaussian_filter', False)
        # Calling gaussian_filter(args, kwargs) (line 547)
        gaussian_filter_call_result_117881 = invoke(stypy.reporting.localization.Localization(__file__, 547, 15), gaussian_filter_117872, *[input_117873, sigma_117874, order_117875, output_117876, mode_117877, cval_117878], **kwargs_117880)
        
        # Assigning a type to the variable 'stypy_return_type' (line 547)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'stypy_return_type', gaussian_filter_call_result_117881)
        
        # ################# End of 'derivative2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivative2' in the type store
        # Getting the type of 'stypy_return_type' (line 544)
        stypy_return_type_117882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_117882)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivative2'
        return stypy_return_type_117882

    # Assigning a type to the variable 'derivative2' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'derivative2', derivative2)
    
    # Call to generic_laplace(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'input' (line 550)
    input_117884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 27), 'input', False)
    # Getting the type of 'derivative2' (line 550)
    derivative2_117885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 34), 'derivative2', False)
    # Getting the type of 'output' (line 550)
    output_117886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 47), 'output', False)
    # Getting the type of 'mode' (line 550)
    mode_117887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 55), 'mode', False)
    # Getting the type of 'cval' (line 550)
    cval_117888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 61), 'cval', False)
    # Processing the call keyword arguments (line 550)
    
    # Obtaining an instance of the builtin type 'tuple' (line 551)
    tuple_117889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 551)
    # Adding element type (line 551)
    # Getting the type of 'sigma' (line 551)
    sigma_117890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 44), 'sigma', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 551, 44), tuple_117889, sigma_117890)
    
    keyword_117891 = tuple_117889
    # Getting the type of 'kwargs' (line 552)
    kwargs_117892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 42), 'kwargs', False)
    keyword_117893 = kwargs_117892
    kwargs_117894 = {'extra_keywords': keyword_117893, 'extra_arguments': keyword_117891}
    # Getting the type of 'generic_laplace' (line 550)
    generic_laplace_117883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 11), 'generic_laplace', False)
    # Calling generic_laplace(args, kwargs) (line 550)
    generic_laplace_call_result_117895 = invoke(stypy.reporting.localization.Localization(__file__, 550, 11), generic_laplace_117883, *[input_117884, derivative2_117885, output_117886, mode_117887, cval_117888], **kwargs_117894)
    
    # Assigning a type to the variable 'stypy_return_type' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'stypy_return_type', generic_laplace_call_result_117895)
    
    # ################# End of 'gaussian_laplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gaussian_laplace' in the type store
    # Getting the type of 'stypy_return_type' (line 507)
    stypy_return_type_117896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_117896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gaussian_laplace'
    return stypy_return_type_117896

# Assigning a type to the variable 'gaussian_laplace' (line 507)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'gaussian_laplace', gaussian_laplace)

@norecursion
def generic_gradient_magnitude(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 556)
    None_117897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 57), 'None')
    str_117898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 36), 'str', 'reflect')
    float_117899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 52), 'float')
    
    # Obtaining an instance of the builtin type 'tuple' (line 558)
    tuple_117900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 558)
    
    # Getting the type of 'None' (line 558)
    None_117901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 66), 'None')
    defaults = [None_117897, str_117898, float_117899, tuple_117900, None_117901]
    # Create a new context for function 'generic_gradient_magnitude'
    module_type_store = module_type_store.open_function_context('generic_gradient_magnitude', 555, 0, False)
    
    # Passed parameters checking function
    generic_gradient_magnitude.stypy_localization = localization
    generic_gradient_magnitude.stypy_type_of_self = None
    generic_gradient_magnitude.stypy_type_store = module_type_store
    generic_gradient_magnitude.stypy_function_name = 'generic_gradient_magnitude'
    generic_gradient_magnitude.stypy_param_names_list = ['input', 'derivative', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords']
    generic_gradient_magnitude.stypy_varargs_param_name = None
    generic_gradient_magnitude.stypy_kwargs_param_name = None
    generic_gradient_magnitude.stypy_call_defaults = defaults
    generic_gradient_magnitude.stypy_call_varargs = varargs
    generic_gradient_magnitude.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generic_gradient_magnitude', ['input', 'derivative', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generic_gradient_magnitude', localization, ['input', 'derivative', 'output', 'mode', 'cval', 'extra_arguments', 'extra_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generic_gradient_magnitude(...)' code ##################

    str_117902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, (-1)), 'str', 'Gradient magnitude using a provided gradient function.\n\n    Parameters\n    ----------\n    %(input)s\n    derivative : callable\n        Callable with the following signature::\n\n            derivative(input, axis, output, mode, cval,\n                       *extra_arguments, **extra_keywords)\n\n        See `extra_arguments`, `extra_keywords` below.\n        `derivative` can assume that `input` and `output` are ndarrays.\n        Note that the output from `derivative` is modified inplace;\n        be careful to copy important inputs before returning them.\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    %(extra_keywords)s\n    %(extra_arguments)s\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 580)
    # Getting the type of 'extra_keywords' (line 580)
    extra_keywords_117903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 7), 'extra_keywords')
    # Getting the type of 'None' (line 580)
    None_117904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 25), 'None')
    
    (may_be_117905, more_types_in_union_117906) = may_be_none(extra_keywords_117903, None_117904)

    if may_be_117905:

        if more_types_in_union_117906:
            # Runtime conditional SSA (line 580)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 581):
        
        # Assigning a Dict to a Name (line 581):
        
        # Obtaining an instance of the builtin type 'dict' (line 581)
        dict_117907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 581)
        
        # Assigning a type to the variable 'extra_keywords' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'extra_keywords', dict_117907)

        if more_types_in_union_117906:
            # SSA join for if statement (line 580)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to asarray(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'input' (line 582)
    input_117910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 26), 'input', False)
    # Processing the call keyword arguments (line 582)
    kwargs_117911 = {}
    # Getting the type of 'numpy' (line 582)
    numpy_117908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 582)
    asarray_117909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 12), numpy_117908, 'asarray')
    # Calling asarray(args, kwargs) (line 582)
    asarray_call_result_117912 = invoke(stypy.reporting.localization.Localization(__file__, 582, 12), asarray_117909, *[input_117910], **kwargs_117911)
    
    # Assigning a type to the variable 'input' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'input', asarray_call_result_117912)
    
    # Assigning a Call to a Tuple (line 583):
    
    # Assigning a Subscript to a Name (line 583):
    
    # Obtaining the type of the subscript
    int_117913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    
    # Call to _get_output(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'output' (line 583)
    output_117916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 51), 'output', False)
    # Getting the type of 'input' (line 583)
    input_117917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 59), 'input', False)
    # Processing the call keyword arguments (line 583)
    kwargs_117918 = {}
    # Getting the type of '_ni_support' (line 583)
    _ni_support_117914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 583)
    _get_output_117915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 27), _ni_support_117914, '_get_output')
    # Calling _get_output(args, kwargs) (line 583)
    _get_output_call_result_117919 = invoke(stypy.reporting.localization.Localization(__file__, 583, 27), _get_output_117915, *[output_117916, input_117917], **kwargs_117918)
    
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___117920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), _get_output_call_result_117919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 583)
    subscript_call_result_117921 = invoke(stypy.reporting.localization.Localization(__file__, 583, 4), getitem___117920, int_117913)
    
    # Assigning a type to the variable 'tuple_var_assignment_117033' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'tuple_var_assignment_117033', subscript_call_result_117921)
    
    # Assigning a Subscript to a Name (line 583):
    
    # Obtaining the type of the subscript
    int_117922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 4), 'int')
    
    # Call to _get_output(...): (line 583)
    # Processing the call arguments (line 583)
    # Getting the type of 'output' (line 583)
    output_117925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 51), 'output', False)
    # Getting the type of 'input' (line 583)
    input_117926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 59), 'input', False)
    # Processing the call keyword arguments (line 583)
    kwargs_117927 = {}
    # Getting the type of '_ni_support' (line 583)
    _ni_support_117923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 583)
    _get_output_117924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 27), _ni_support_117923, '_get_output')
    # Calling _get_output(args, kwargs) (line 583)
    _get_output_call_result_117928 = invoke(stypy.reporting.localization.Localization(__file__, 583, 27), _get_output_117924, *[output_117925, input_117926], **kwargs_117927)
    
    # Obtaining the member '__getitem__' of a type (line 583)
    getitem___117929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 4), _get_output_call_result_117928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 583)
    subscript_call_result_117930 = invoke(stypy.reporting.localization.Localization(__file__, 583, 4), getitem___117929, int_117922)
    
    # Assigning a type to the variable 'tuple_var_assignment_117034' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'tuple_var_assignment_117034', subscript_call_result_117930)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'tuple_var_assignment_117033' (line 583)
    tuple_var_assignment_117033_117931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'tuple_var_assignment_117033')
    # Assigning a type to the variable 'output' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'output', tuple_var_assignment_117033_117931)
    
    # Assigning a Name to a Name (line 583):
    # Getting the type of 'tuple_var_assignment_117034' (line 583)
    tuple_var_assignment_117034_117932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'tuple_var_assignment_117034')
    # Assigning a type to the variable 'return_value' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'return_value', tuple_var_assignment_117034_117932)
    
    # Assigning a Call to a Name (line 584):
    
    # Assigning a Call to a Name (line 584):
    
    # Call to list(...): (line 584)
    # Processing the call arguments (line 584)
    
    # Call to range(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'input' (line 584)
    input_117935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 22), 'input', False)
    # Obtaining the member 'ndim' of a type (line 584)
    ndim_117936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 22), input_117935, 'ndim')
    # Processing the call keyword arguments (line 584)
    kwargs_117937 = {}
    # Getting the type of 'range' (line 584)
    range_117934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'range', False)
    # Calling range(args, kwargs) (line 584)
    range_call_result_117938 = invoke(stypy.reporting.localization.Localization(__file__, 584, 16), range_117934, *[ndim_117936], **kwargs_117937)
    
    # Processing the call keyword arguments (line 584)
    kwargs_117939 = {}
    # Getting the type of 'list' (line 584)
    list_117933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'list', False)
    # Calling list(args, kwargs) (line 584)
    list_call_result_117940 = invoke(stypy.reporting.localization.Localization(__file__, 584, 11), list_117933, *[range_call_result_117938], **kwargs_117939)
    
    # Assigning a type to the variable 'axes' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'axes', list_call_result_117940)
    
    
    
    # Call to len(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'axes' (line 585)
    axes_117942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'axes', False)
    # Processing the call keyword arguments (line 585)
    kwargs_117943 = {}
    # Getting the type of 'len' (line 585)
    len_117941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 7), 'len', False)
    # Calling len(args, kwargs) (line 585)
    len_call_result_117944 = invoke(stypy.reporting.localization.Localization(__file__, 585, 7), len_117941, *[axes_117942], **kwargs_117943)
    
    int_117945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 19), 'int')
    # Applying the binary operator '>' (line 585)
    result_gt_117946 = python_operator(stypy.reporting.localization.Localization(__file__, 585, 7), '>', len_call_result_117944, int_117945)
    
    # Testing the type of an if condition (line 585)
    if_condition_117947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 4), result_gt_117946)
    # Assigning a type to the variable 'if_condition_117947' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'if_condition_117947', if_condition_117947)
    # SSA begins for if statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 586):
    
    # Assigning a Call to a Name (line 586):
    
    # Call to _normalize_sequence(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'mode' (line 586)
    mode_117950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 48), 'mode', False)
    
    # Call to len(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'axes' (line 586)
    axes_117952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 58), 'axes', False)
    # Processing the call keyword arguments (line 586)
    kwargs_117953 = {}
    # Getting the type of 'len' (line 586)
    len_117951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 54), 'len', False)
    # Calling len(args, kwargs) (line 586)
    len_call_result_117954 = invoke(stypy.reporting.localization.Localization(__file__, 586, 54), len_117951, *[axes_117952], **kwargs_117953)
    
    # Processing the call keyword arguments (line 586)
    kwargs_117955 = {}
    # Getting the type of '_ni_support' (line 586)
    _ni_support_117948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 586)
    _normalize_sequence_117949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 16), _ni_support_117948, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 586)
    _normalize_sequence_call_result_117956 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), _normalize_sequence_117949, *[mode_117950, len_call_result_117954], **kwargs_117955)
    
    # Assigning a type to the variable 'modes' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'modes', _normalize_sequence_call_result_117956)
    
    # Call to derivative(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'input' (line 587)
    input_117958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 19), 'input', False)
    
    # Obtaining the type of the subscript
    int_117959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 31), 'int')
    # Getting the type of 'axes' (line 587)
    axes_117960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___117961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 26), axes_117960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_117962 = invoke(stypy.reporting.localization.Localization(__file__, 587, 26), getitem___117961, int_117959)
    
    # Getting the type of 'output' (line 587)
    output_117963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 35), 'output', False)
    
    # Obtaining the type of the subscript
    int_117964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 49), 'int')
    # Getting the type of 'modes' (line 587)
    modes_117965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 43), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___117966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 43), modes_117965, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_117967 = invoke(stypy.reporting.localization.Localization(__file__, 587, 43), getitem___117966, int_117964)
    
    # Getting the type of 'cval' (line 587)
    cval_117968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 53), 'cval', False)
    # Getting the type of 'extra_arguments' (line 588)
    extra_arguments_117969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 20), 'extra_arguments', False)
    # Processing the call keyword arguments (line 587)
    # Getting the type of 'extra_keywords' (line 588)
    extra_keywords_117970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 39), 'extra_keywords', False)
    kwargs_117971 = {'extra_keywords_117970': extra_keywords_117970}
    # Getting the type of 'derivative' (line 587)
    derivative_117957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'derivative', False)
    # Calling derivative(args, kwargs) (line 587)
    derivative_call_result_117972 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), derivative_117957, *[input_117958, subscript_call_result_117962, output_117963, subscript_call_result_117967, cval_117968, extra_arguments_117969], **kwargs_117971)
    
    
    # Call to multiply(...): (line 589)
    # Processing the call arguments (line 589)
    # Getting the type of 'output' (line 589)
    output_117975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 'output', False)
    # Getting the type of 'output' (line 589)
    output_117976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 31), 'output', False)
    # Getting the type of 'output' (line 589)
    output_117977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 39), 'output', False)
    # Processing the call keyword arguments (line 589)
    kwargs_117978 = {}
    # Getting the type of 'numpy' (line 589)
    numpy_117973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'numpy', False)
    # Obtaining the member 'multiply' of a type (line 589)
    multiply_117974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 8), numpy_117973, 'multiply')
    # Calling multiply(args, kwargs) (line 589)
    multiply_call_result_117979 = invoke(stypy.reporting.localization.Localization(__file__, 589, 8), multiply_117974, *[output_117975, output_117976, output_117977], **kwargs_117978)
    
    
    
    # Call to range(...): (line 590)
    # Processing the call arguments (line 590)
    int_117981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 24), 'int')
    
    # Call to len(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'axes' (line 590)
    axes_117983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 31), 'axes', False)
    # Processing the call keyword arguments (line 590)
    kwargs_117984 = {}
    # Getting the type of 'len' (line 590)
    len_117982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 27), 'len', False)
    # Calling len(args, kwargs) (line 590)
    len_call_result_117985 = invoke(stypy.reporting.localization.Localization(__file__, 590, 27), len_117982, *[axes_117983], **kwargs_117984)
    
    # Processing the call keyword arguments (line 590)
    kwargs_117986 = {}
    # Getting the type of 'range' (line 590)
    range_117980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 18), 'range', False)
    # Calling range(args, kwargs) (line 590)
    range_call_result_117987 = invoke(stypy.reporting.localization.Localization(__file__, 590, 18), range_117980, *[int_117981, len_call_result_117985], **kwargs_117986)
    
    # Testing the type of a for loop iterable (line 590)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 590, 8), range_call_result_117987)
    # Getting the type of the for loop variable (line 590)
    for_loop_var_117988 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 590, 8), range_call_result_117987)
    # Assigning a type to the variable 'ii' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'ii', for_loop_var_117988)
    # SSA begins for a for statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 591):
    
    # Assigning a Call to a Name (line 591):
    
    # Call to derivative(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'input' (line 591)
    input_117990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 29), 'input', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 591)
    ii_117991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 41), 'ii', False)
    # Getting the type of 'axes' (line 591)
    axes_117992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 36), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___117993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 36), axes_117992, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_117994 = invoke(stypy.reporting.localization.Localization(__file__, 591, 36), getitem___117993, ii_117991)
    
    # Getting the type of 'output' (line 591)
    output_117995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 46), 'output', False)
    # Obtaining the member 'dtype' of a type (line 591)
    dtype_117996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 46), output_117995, 'dtype')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 591)
    ii_117997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 66), 'ii', False)
    # Getting the type of 'modes' (line 591)
    modes_117998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 60), 'modes', False)
    # Obtaining the member '__getitem__' of a type (line 591)
    getitem___117999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 591, 60), modes_117998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 591)
    subscript_call_result_118000 = invoke(stypy.reporting.localization.Localization(__file__, 591, 60), getitem___117999, ii_117997)
    
    # Getting the type of 'cval' (line 591)
    cval_118001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 71), 'cval', False)
    # Getting the type of 'extra_arguments' (line 592)
    extra_arguments_118002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 30), 'extra_arguments', False)
    # Processing the call keyword arguments (line 591)
    # Getting the type of 'extra_keywords' (line 592)
    extra_keywords_118003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 49), 'extra_keywords', False)
    kwargs_118004 = {'extra_keywords_118003': extra_keywords_118003}
    # Getting the type of 'derivative' (line 591)
    derivative_117989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 18), 'derivative', False)
    # Calling derivative(args, kwargs) (line 591)
    derivative_call_result_118005 = invoke(stypy.reporting.localization.Localization(__file__, 591, 18), derivative_117989, *[input_117990, subscript_call_result_117994, dtype_117996, subscript_call_result_118000, cval_118001, extra_arguments_118002], **kwargs_118004)
    
    # Assigning a type to the variable 'tmp' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'tmp', derivative_call_result_118005)
    
    # Call to multiply(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'tmp' (line 593)
    tmp_118008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'tmp', False)
    # Getting the type of 'tmp' (line 593)
    tmp_118009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 32), 'tmp', False)
    # Getting the type of 'tmp' (line 593)
    tmp_118010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 37), 'tmp', False)
    # Processing the call keyword arguments (line 593)
    kwargs_118011 = {}
    # Getting the type of 'numpy' (line 593)
    numpy_118006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'numpy', False)
    # Obtaining the member 'multiply' of a type (line 593)
    multiply_118007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 12), numpy_118006, 'multiply')
    # Calling multiply(args, kwargs) (line 593)
    multiply_call_result_118012 = invoke(stypy.reporting.localization.Localization(__file__, 593, 12), multiply_118007, *[tmp_118008, tmp_118009, tmp_118010], **kwargs_118011)
    
    
    # Getting the type of 'output' (line 594)
    output_118013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'output')
    # Getting the type of 'tmp' (line 594)
    tmp_118014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 22), 'tmp')
    # Applying the binary operator '+=' (line 594)
    result_iadd_118015 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 12), '+=', output_118013, tmp_118014)
    # Assigning a type to the variable 'output' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 12), 'output', result_iadd_118015)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sqrt(...): (line 596)
    # Processing the call arguments (line 596)
    # Getting the type of 'output' (line 596)
    output_118018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 19), 'output', False)
    # Getting the type of 'output' (line 596)
    output_118019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 27), 'output', False)
    # Processing the call keyword arguments (line 596)
    str_118020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 43), 'str', 'unsafe')
    keyword_118021 = str_118020
    kwargs_118022 = {'casting': keyword_118021}
    # Getting the type of 'numpy' (line 596)
    numpy_118016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'numpy', False)
    # Obtaining the member 'sqrt' of a type (line 596)
    sqrt_118017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 8), numpy_118016, 'sqrt')
    # Calling sqrt(args, kwargs) (line 596)
    sqrt_call_result_118023 = invoke(stypy.reporting.localization.Localization(__file__, 596, 8), sqrt_118017, *[output_118018, output_118019], **kwargs_118022)
    
    # SSA branch for the else part of an if statement (line 585)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 598):
    
    # Assigning a Subscript to a Subscript (line 598):
    
    # Obtaining the type of the subscript
    Ellipsis_118024 = Ellipsis
    # Getting the type of 'input' (line 598)
    input_118025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 598)
    getitem___118026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 22), input_118025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 598)
    subscript_call_result_118027 = invoke(stypy.reporting.localization.Localization(__file__, 598, 22), getitem___118026, Ellipsis_118024)
    
    # Getting the type of 'output' (line 598)
    output_118028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'output')
    Ellipsis_118029 = Ellipsis
    # Storing an element on a container (line 598)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 8), output_118028, (Ellipsis_118029, subscript_call_result_118027))
    # SSA join for if statement (line 585)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 599)
    return_value_118030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 4), 'stypy_return_type', return_value_118030)
    
    # ################# End of 'generic_gradient_magnitude(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generic_gradient_magnitude' in the type store
    # Getting the type of 'stypy_return_type' (line 555)
    stypy_return_type_118031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118031)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generic_gradient_magnitude'
    return stypy_return_type_118031

# Assigning a type to the variable 'generic_gradient_magnitude' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'generic_gradient_magnitude', generic_gradient_magnitude)

@norecursion
def gaussian_gradient_magnitude(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 603)
    None_118032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 53), 'None')
    str_118033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 37), 'str', 'reflect')
    float_118034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 53), 'float')
    defaults = [None_118032, str_118033, float_118034]
    # Create a new context for function 'gaussian_gradient_magnitude'
    module_type_store = module_type_store.open_function_context('gaussian_gradient_magnitude', 602, 0, False)
    
    # Passed parameters checking function
    gaussian_gradient_magnitude.stypy_localization = localization
    gaussian_gradient_magnitude.stypy_type_of_self = None
    gaussian_gradient_magnitude.stypy_type_store = module_type_store
    gaussian_gradient_magnitude.stypy_function_name = 'gaussian_gradient_magnitude'
    gaussian_gradient_magnitude.stypy_param_names_list = ['input', 'sigma', 'output', 'mode', 'cval']
    gaussian_gradient_magnitude.stypy_varargs_param_name = None
    gaussian_gradient_magnitude.stypy_kwargs_param_name = 'kwargs'
    gaussian_gradient_magnitude.stypy_call_defaults = defaults
    gaussian_gradient_magnitude.stypy_call_varargs = varargs
    gaussian_gradient_magnitude.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gaussian_gradient_magnitude', ['input', 'sigma', 'output', 'mode', 'cval'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gaussian_gradient_magnitude', localization, ['input', 'sigma', 'output', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gaussian_gradient_magnitude(...)' code ##################

    str_118035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, (-1)), 'str', 'Multidimensional gradient magnitude using Gaussian derivatives.\n\n    Parameters\n    ----------\n    %(input)s\n    sigma : scalar or sequence of scalars\n        The standard deviations of the Gaussian filter are given for\n        each axis as a sequence, or as a single number, in which case\n        it is equal for all axes..\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    Extra keyword arguments will be passed to gaussian_filter().\n\n    Returns\n    -------\n    gaussian_gradient_magnitude : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.gaussian_gradient_magnitude(ascent, sigma=5)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 638):
    
    # Assigning a Call to a Name (line 638):
    
    # Call to asarray(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'input' (line 638)
    input_118038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 26), 'input', False)
    # Processing the call keyword arguments (line 638)
    kwargs_118039 = {}
    # Getting the type of 'numpy' (line 638)
    numpy_118036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 638)
    asarray_118037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 12), numpy_118036, 'asarray')
    # Calling asarray(args, kwargs) (line 638)
    asarray_call_result_118040 = invoke(stypy.reporting.localization.Localization(__file__, 638, 12), asarray_118037, *[input_118038], **kwargs_118039)
    
    # Assigning a type to the variable 'input' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'input', asarray_call_result_118040)

    @norecursion
    def derivative(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'derivative'
        module_type_store = module_type_store.open_function_context('derivative', 640, 4, False)
        
        # Passed parameters checking function
        derivative.stypy_localization = localization
        derivative.stypy_type_of_self = None
        derivative.stypy_type_store = module_type_store
        derivative.stypy_function_name = 'derivative'
        derivative.stypy_param_names_list = ['input', 'axis', 'output', 'mode', 'cval', 'sigma']
        derivative.stypy_varargs_param_name = None
        derivative.stypy_kwargs_param_name = 'kwargs'
        derivative.stypy_call_defaults = defaults
        derivative.stypy_call_varargs = varargs
        derivative.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'derivative', ['input', 'axis', 'output', 'mode', 'cval', 'sigma'], None, 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'derivative', localization, ['input', 'axis', 'output', 'mode', 'cval', 'sigma'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'derivative(...)' code ##################

        
        # Assigning a BinOp to a Name (line 641):
        
        # Assigning a BinOp to a Name (line 641):
        
        # Obtaining an instance of the builtin type 'list' (line 641)
        list_118041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 641)
        # Adding element type (line 641)
        int_118042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 641, 16), list_118041, int_118042)
        
        # Getting the type of 'input' (line 641)
        input_118043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 22), 'input')
        # Obtaining the member 'ndim' of a type (line 641)
        ndim_118044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 22), input_118043, 'ndim')
        # Applying the binary operator '*' (line 641)
        result_mul_118045 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 16), '*', list_118041, ndim_118044)
        
        # Assigning a type to the variable 'order' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'order', result_mul_118045)
        
        # Assigning a Num to a Subscript (line 642):
        
        # Assigning a Num to a Subscript (line 642):
        int_118046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 22), 'int')
        # Getting the type of 'order' (line 642)
        order_118047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'order')
        # Getting the type of 'axis' (line 642)
        axis_118048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 14), 'axis')
        # Storing an element on a container (line 642)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 642, 8), order_118047, (axis_118048, int_118046))
        
        # Call to gaussian_filter(...): (line 643)
        # Processing the call arguments (line 643)
        # Getting the type of 'input' (line 643)
        input_118050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 31), 'input', False)
        # Getting the type of 'sigma' (line 643)
        sigma_118051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 38), 'sigma', False)
        # Getting the type of 'order' (line 643)
        order_118052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 45), 'order', False)
        # Getting the type of 'output' (line 643)
        output_118053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 52), 'output', False)
        # Getting the type of 'mode' (line 643)
        mode_118054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 60), 'mode', False)
        # Getting the type of 'cval' (line 644)
        cval_118055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 31), 'cval', False)
        # Processing the call keyword arguments (line 643)
        # Getting the type of 'kwargs' (line 644)
        kwargs_118056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 39), 'kwargs', False)
        kwargs_118057 = {'kwargs_118056': kwargs_118056}
        # Getting the type of 'gaussian_filter' (line 643)
        gaussian_filter_118049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'gaussian_filter', False)
        # Calling gaussian_filter(args, kwargs) (line 643)
        gaussian_filter_call_result_118058 = invoke(stypy.reporting.localization.Localization(__file__, 643, 15), gaussian_filter_118049, *[input_118050, sigma_118051, order_118052, output_118053, mode_118054, cval_118055], **kwargs_118057)
        
        # Assigning a type to the variable 'stypy_return_type' (line 643)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'stypy_return_type', gaussian_filter_call_result_118058)
        
        # ################# End of 'derivative(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'derivative' in the type store
        # Getting the type of 'stypy_return_type' (line 640)
        stypy_return_type_118059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_118059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'derivative'
        return stypy_return_type_118059

    # Assigning a type to the variable 'derivative' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'derivative', derivative)
    
    # Call to generic_gradient_magnitude(...): (line 646)
    # Processing the call arguments (line 646)
    # Getting the type of 'input' (line 646)
    input_118061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 38), 'input', False)
    # Getting the type of 'derivative' (line 646)
    derivative_118062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 45), 'derivative', False)
    # Getting the type of 'output' (line 646)
    output_118063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 57), 'output', False)
    # Getting the type of 'mode' (line 646)
    mode_118064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 65), 'mode', False)
    # Getting the type of 'cval' (line 647)
    cval_118065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 38), 'cval', False)
    # Processing the call keyword arguments (line 646)
    
    # Obtaining an instance of the builtin type 'tuple' (line 647)
    tuple_118066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 647)
    # Adding element type (line 647)
    # Getting the type of 'sigma' (line 647)
    sigma_118067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 61), 'sigma', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 647, 61), tuple_118066, sigma_118067)
    
    keyword_118068 = tuple_118066
    # Getting the type of 'kwargs' (line 648)
    kwargs_118069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 53), 'kwargs', False)
    keyword_118070 = kwargs_118069
    kwargs_118071 = {'extra_keywords': keyword_118070, 'extra_arguments': keyword_118068}
    # Getting the type of 'generic_gradient_magnitude' (line 646)
    generic_gradient_magnitude_118060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'generic_gradient_magnitude', False)
    # Calling generic_gradient_magnitude(args, kwargs) (line 646)
    generic_gradient_magnitude_call_result_118072 = invoke(stypy.reporting.localization.Localization(__file__, 646, 11), generic_gradient_magnitude_118060, *[input_118061, derivative_118062, output_118063, mode_118064, cval_118065], **kwargs_118071)
    
    # Assigning a type to the variable 'stypy_return_type' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 4), 'stypy_return_type', generic_gradient_magnitude_call_result_118072)
    
    # ################# End of 'gaussian_gradient_magnitude(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gaussian_gradient_magnitude' in the type store
    # Getting the type of 'stypy_return_type' (line 602)
    stypy_return_type_118073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118073)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gaussian_gradient_magnitude'
    return stypy_return_type_118073

# Assigning a type to the variable 'gaussian_gradient_magnitude' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'gaussian_gradient_magnitude', gaussian_gradient_magnitude)

@norecursion
def _correlate_or_convolve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_correlate_or_convolve'
    module_type_store = module_type_store.open_function_context('_correlate_or_convolve', 651, 0, False)
    
    # Passed parameters checking function
    _correlate_or_convolve.stypy_localization = localization
    _correlate_or_convolve.stypy_type_of_self = None
    _correlate_or_convolve.stypy_type_store = module_type_store
    _correlate_or_convolve.stypy_function_name = '_correlate_or_convolve'
    _correlate_or_convolve.stypy_param_names_list = ['input', 'weights', 'output', 'mode', 'cval', 'origin', 'convolution']
    _correlate_or_convolve.stypy_varargs_param_name = None
    _correlate_or_convolve.stypy_kwargs_param_name = None
    _correlate_or_convolve.stypy_call_defaults = defaults
    _correlate_or_convolve.stypy_call_varargs = varargs
    _correlate_or_convolve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_correlate_or_convolve', ['input', 'weights', 'output', 'mode', 'cval', 'origin', 'convolution'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_correlate_or_convolve', localization, ['input', 'weights', 'output', 'mode', 'cval', 'origin', 'convolution'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_correlate_or_convolve(...)' code ##################

    
    # Assigning a Call to a Name (line 653):
    
    # Assigning a Call to a Name (line 653):
    
    # Call to asarray(...): (line 653)
    # Processing the call arguments (line 653)
    # Getting the type of 'input' (line 653)
    input_118076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 26), 'input', False)
    # Processing the call keyword arguments (line 653)
    kwargs_118077 = {}
    # Getting the type of 'numpy' (line 653)
    numpy_118074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 653)
    asarray_118075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 12), numpy_118074, 'asarray')
    # Calling asarray(args, kwargs) (line 653)
    asarray_call_result_118078 = invoke(stypy.reporting.localization.Localization(__file__, 653, 12), asarray_118075, *[input_118076], **kwargs_118077)
    
    # Assigning a type to the variable 'input' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 4), 'input', asarray_call_result_118078)
    
    
    # Call to iscomplexobj(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'input' (line 654)
    input_118081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 26), 'input', False)
    # Processing the call keyword arguments (line 654)
    kwargs_118082 = {}
    # Getting the type of 'numpy' (line 654)
    numpy_118079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 654)
    iscomplexobj_118080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 7), numpy_118079, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 654)
    iscomplexobj_call_result_118083 = invoke(stypy.reporting.localization.Localization(__file__, 654, 7), iscomplexobj_118080, *[input_118081], **kwargs_118082)
    
    # Testing the type of an if condition (line 654)
    if_condition_118084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 4), iscomplexobj_call_result_118083)
    # Assigning a type to the variable 'if_condition_118084' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 4), 'if_condition_118084', if_condition_118084)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 655)
    # Processing the call arguments (line 655)
    str_118086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 655)
    kwargs_118087 = {}
    # Getting the type of 'TypeError' (line 655)
    TypeError_118085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 655)
    TypeError_call_result_118088 = invoke(stypy.reporting.localization.Localization(__file__, 655, 14), TypeError_118085, *[str_118086], **kwargs_118087)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 655, 8), TypeError_call_result_118088, 'raise parameter', BaseException)
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 656):
    
    # Assigning a Call to a Name (line 656):
    
    # Call to _normalize_sequence(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of 'origin' (line 656)
    origin_118091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 46), 'origin', False)
    # Getting the type of 'input' (line 656)
    input_118092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 656)
    ndim_118093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 54), input_118092, 'ndim')
    # Processing the call keyword arguments (line 656)
    kwargs_118094 = {}
    # Getting the type of '_ni_support' (line 656)
    _ni_support_118089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 656)
    _normalize_sequence_118090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 14), _ni_support_118089, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 656)
    _normalize_sequence_call_result_118095 = invoke(stypy.reporting.localization.Localization(__file__, 656, 14), _normalize_sequence_118090, *[origin_118091, ndim_118093], **kwargs_118094)
    
    # Assigning a type to the variable 'origins' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'origins', _normalize_sequence_call_result_118095)
    
    # Assigning a Call to a Name (line 657):
    
    # Assigning a Call to a Name (line 657):
    
    # Call to asarray(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'weights' (line 657)
    weights_118098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 28), 'weights', False)
    # Processing the call keyword arguments (line 657)
    # Getting the type of 'numpy' (line 657)
    numpy_118099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 43), 'numpy', False)
    # Obtaining the member 'float64' of a type (line 657)
    float64_118100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 43), numpy_118099, 'float64')
    keyword_118101 = float64_118100
    kwargs_118102 = {'dtype': keyword_118101}
    # Getting the type of 'numpy' (line 657)
    numpy_118096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 14), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 657)
    asarray_118097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 14), numpy_118096, 'asarray')
    # Calling asarray(args, kwargs) (line 657)
    asarray_call_result_118103 = invoke(stypy.reporting.localization.Localization(__file__, 657, 14), asarray_118097, *[weights_118098], **kwargs_118102)
    
    # Assigning a type to the variable 'weights' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 4), 'weights', asarray_call_result_118103)
    
    # Assigning a ListComp to a Name (line 658):
    
    # Assigning a ListComp to a Name (line 658):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'weights' (line 658)
    weights_118108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 27), 'weights')
    # Obtaining the member 'shape' of a type (line 658)
    shape_118109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 27), weights_118108, 'shape')
    comprehension_118110 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 14), shape_118109)
    # Assigning a type to the variable 'ii' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 14), 'ii', comprehension_118110)
    
    # Getting the type of 'ii' (line 658)
    ii_118105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 44), 'ii')
    int_118106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 49), 'int')
    # Applying the binary operator '>' (line 658)
    result_gt_118107 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 44), '>', ii_118105, int_118106)
    
    # Getting the type of 'ii' (line 658)
    ii_118104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 14), 'ii')
    list_118111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 14), list_118111, ii_118104)
    # Assigning a type to the variable 'wshape' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 4), 'wshape', list_118111)
    
    
    
    # Call to len(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'wshape' (line 659)
    wshape_118113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 11), 'wshape', False)
    # Processing the call keyword arguments (line 659)
    kwargs_118114 = {}
    # Getting the type of 'len' (line 659)
    len_118112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 7), 'len', False)
    # Calling len(args, kwargs) (line 659)
    len_call_result_118115 = invoke(stypy.reporting.localization.Localization(__file__, 659, 7), len_118112, *[wshape_118113], **kwargs_118114)
    
    # Getting the type of 'input' (line 659)
    input_118116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 22), 'input')
    # Obtaining the member 'ndim' of a type (line 659)
    ndim_118117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 659, 22), input_118116, 'ndim')
    # Applying the binary operator '!=' (line 659)
    result_ne_118118 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 7), '!=', len_call_result_118115, ndim_118117)
    
    # Testing the type of an if condition (line 659)
    if_condition_118119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 659, 4), result_ne_118118)
    # Assigning a type to the variable 'if_condition_118119' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'if_condition_118119', if_condition_118119)
    # SSA begins for if statement (line 659)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 660)
    # Processing the call arguments (line 660)
    str_118121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 27), 'str', 'filter weights array has incorrect shape.')
    # Processing the call keyword arguments (line 660)
    kwargs_118122 = {}
    # Getting the type of 'RuntimeError' (line 660)
    RuntimeError_118120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 660)
    RuntimeError_call_result_118123 = invoke(stypy.reporting.localization.Localization(__file__, 660, 14), RuntimeError_118120, *[str_118121], **kwargs_118122)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 660, 8), RuntimeError_call_result_118123, 'raise parameter', BaseException)
    # SSA join for if statement (line 659)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'convolution' (line 661)
    convolution_118124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 7), 'convolution')
    # Testing the type of an if condition (line 661)
    if_condition_118125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 4), convolution_118124)
    # Assigning a type to the variable 'if_condition_118125' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'if_condition_118125', if_condition_118125)
    # SSA begins for if statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 662):
    
    # Assigning a Subscript to a Name (line 662):
    
    # Obtaining the type of the subscript
    
    # Call to tuple(...): (line 662)
    # Processing the call arguments (line 662)
    
    # Obtaining an instance of the builtin type 'list' (line 662)
    list_118127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 662)
    # Adding element type (line 662)
    
    # Call to slice(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'None' (line 662)
    None_118129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 39), 'None', False)
    # Getting the type of 'None' (line 662)
    None_118130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 45), 'None', False)
    int_118131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 51), 'int')
    # Processing the call keyword arguments (line 662)
    kwargs_118132 = {}
    # Getting the type of 'slice' (line 662)
    slice_118128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 33), 'slice', False)
    # Calling slice(args, kwargs) (line 662)
    slice_call_result_118133 = invoke(stypy.reporting.localization.Localization(__file__, 662, 33), slice_118128, *[None_118129, None_118130, int_118131], **kwargs_118132)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 662, 32), list_118127, slice_call_result_118133)
    
    # Getting the type of 'weights' (line 662)
    weights_118134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 58), 'weights', False)
    # Obtaining the member 'ndim' of a type (line 662)
    ndim_118135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 58), weights_118134, 'ndim')
    # Applying the binary operator '*' (line 662)
    result_mul_118136 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 32), '*', list_118127, ndim_118135)
    
    # Processing the call keyword arguments (line 662)
    kwargs_118137 = {}
    # Getting the type of 'tuple' (line 662)
    tuple_118126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 26), 'tuple', False)
    # Calling tuple(args, kwargs) (line 662)
    tuple_call_result_118138 = invoke(stypy.reporting.localization.Localization(__file__, 662, 26), tuple_118126, *[result_mul_118136], **kwargs_118137)
    
    # Getting the type of 'weights' (line 662)
    weights_118139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 18), 'weights')
    # Obtaining the member '__getitem__' of a type (line 662)
    getitem___118140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 18), weights_118139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 662)
    subscript_call_result_118141 = invoke(stypy.reporting.localization.Localization(__file__, 662, 18), getitem___118140, tuple_call_result_118138)
    
    # Assigning a type to the variable 'weights' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'weights', subscript_call_result_118141)
    
    
    # Call to range(...): (line 663)
    # Processing the call arguments (line 663)
    
    # Call to len(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'origins' (line 663)
    origins_118144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 28), 'origins', False)
    # Processing the call keyword arguments (line 663)
    kwargs_118145 = {}
    # Getting the type of 'len' (line 663)
    len_118143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 24), 'len', False)
    # Calling len(args, kwargs) (line 663)
    len_call_result_118146 = invoke(stypy.reporting.localization.Localization(__file__, 663, 24), len_118143, *[origins_118144], **kwargs_118145)
    
    # Processing the call keyword arguments (line 663)
    kwargs_118147 = {}
    # Getting the type of 'range' (line 663)
    range_118142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 18), 'range', False)
    # Calling range(args, kwargs) (line 663)
    range_call_result_118148 = invoke(stypy.reporting.localization.Localization(__file__, 663, 18), range_118142, *[len_call_result_118146], **kwargs_118147)
    
    # Testing the type of a for loop iterable (line 663)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 663, 8), range_call_result_118148)
    # Getting the type of the for loop variable (line 663)
    for_loop_var_118149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 663, 8), range_call_result_118148)
    # Assigning a type to the variable 'ii' (line 663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'ii', for_loop_var_118149)
    # SSA begins for a for statement (line 663)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a UnaryOp to a Subscript (line 664):
    
    # Assigning a UnaryOp to a Subscript (line 664):
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 664)
    ii_118150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 35), 'ii')
    # Getting the type of 'origins' (line 664)
    origins_118151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 27), 'origins')
    # Obtaining the member '__getitem__' of a type (line 664)
    getitem___118152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 27), origins_118151, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 664)
    subscript_call_result_118153 = invoke(stypy.reporting.localization.Localization(__file__, 664, 27), getitem___118152, ii_118150)
    
    # Applying the 'usub' unary operator (line 664)
    result___neg___118154 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 26), 'usub', subscript_call_result_118153)
    
    # Getting the type of 'origins' (line 664)
    origins_118155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'origins')
    # Getting the type of 'ii' (line 664)
    ii_118156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 20), 'ii')
    # Storing an element on a container (line 664)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 664, 12), origins_118155, (ii_118156, result___neg___118154))
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 665)
    ii_118157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 33), 'ii')
    # Getting the type of 'weights' (line 665)
    weights_118158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 19), 'weights')
    # Obtaining the member 'shape' of a type (line 665)
    shape_118159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 19), weights_118158, 'shape')
    # Obtaining the member '__getitem__' of a type (line 665)
    getitem___118160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 665, 19), shape_118159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 665)
    subscript_call_result_118161 = invoke(stypy.reporting.localization.Localization(__file__, 665, 19), getitem___118160, ii_118157)
    
    int_118162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 39), 'int')
    # Applying the binary operator '&' (line 665)
    result_and__118163 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 19), '&', subscript_call_result_118161, int_118162)
    
    # Applying the 'not' unary operator (line 665)
    result_not__118164 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 15), 'not', result_and__118163)
    
    # Testing the type of an if condition (line 665)
    if_condition_118165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 665, 12), result_not__118164)
    # Assigning a type to the variable 'if_condition_118165' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 12), 'if_condition_118165', if_condition_118165)
    # SSA begins for if statement (line 665)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'origins' (line 666)
    origins_118166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'origins')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 666)
    ii_118167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'ii')
    # Getting the type of 'origins' (line 666)
    origins_118168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'origins')
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___118169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 16), origins_118168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_118170 = invoke(stypy.reporting.localization.Localization(__file__, 666, 16), getitem___118169, ii_118167)
    
    int_118171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 31), 'int')
    # Applying the binary operator '-=' (line 666)
    result_isub_118172 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 16), '-=', subscript_call_result_118170, int_118171)
    # Getting the type of 'origins' (line 666)
    origins_118173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 16), 'origins')
    # Getting the type of 'ii' (line 666)
    ii_118174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'ii')
    # Storing an element on a container (line 666)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 16), origins_118173, (ii_118174, result_isub_118172))
    
    # SSA join for if statement (line 665)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to zip(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 'origins' (line 667)
    origins_118176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 28), 'origins', False)
    # Getting the type of 'wshape' (line 667)
    wshape_118177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 37), 'wshape', False)
    # Processing the call keyword arguments (line 667)
    kwargs_118178 = {}
    # Getting the type of 'zip' (line 667)
    zip_118175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 24), 'zip', False)
    # Calling zip(args, kwargs) (line 667)
    zip_call_result_118179 = invoke(stypy.reporting.localization.Localization(__file__, 667, 24), zip_118175, *[origins_118176, wshape_118177], **kwargs_118178)
    
    # Testing the type of a for loop iterable (line 667)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 667, 4), zip_call_result_118179)
    # Getting the type of the for loop variable (line 667)
    for_loop_var_118180 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 667, 4), zip_call_result_118179)
    # Assigning a type to the variable 'origin' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 4), for_loop_var_118180))
    # Assigning a type to the variable 'lenw' (line 667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 4), 'lenw', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 4), for_loop_var_118180))
    # SSA begins for a for statement (line 667)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lenw' (line 668)
    lenw_118181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'lenw')
    int_118182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 20), 'int')
    # Applying the binary operator '//' (line 668)
    result_floordiv_118183 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 12), '//', lenw_118181, int_118182)
    
    # Getting the type of 'origin' (line 668)
    origin_118184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 24), 'origin')
    # Applying the binary operator '+' (line 668)
    result_add_118185 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 12), '+', result_floordiv_118183, origin_118184)
    
    int_118186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 33), 'int')
    # Applying the binary operator '<' (line 668)
    result_lt_118187 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 12), '<', result_add_118185, int_118186)
    
    
    # Getting the type of 'lenw' (line 668)
    lenw_118188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 40), 'lenw')
    int_118189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 668, 48), 'int')
    # Applying the binary operator '//' (line 668)
    result_floordiv_118190 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 40), '//', lenw_118188, int_118189)
    
    # Getting the type of 'origin' (line 668)
    origin_118191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 52), 'origin')
    # Applying the binary operator '+' (line 668)
    result_add_118192 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 40), '+', result_floordiv_118190, origin_118191)
    
    # Getting the type of 'lenw' (line 668)
    lenw_118193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 61), 'lenw')
    # Applying the binary operator '>' (line 668)
    result_gt_118194 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 40), '>', result_add_118192, lenw_118193)
    
    # Applying the binary operator 'or' (line 668)
    result_or_keyword_118195 = python_operator(stypy.reporting.localization.Localization(__file__, 668, 11), 'or', result_lt_118187, result_gt_118194)
    
    # Testing the type of an if condition (line 668)
    if_condition_118196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 668, 8), result_or_keyword_118195)
    # Assigning a type to the variable 'if_condition_118196' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'if_condition_118196', if_condition_118196)
    # SSA begins for if statement (line 668)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 669)
    # Processing the call arguments (line 669)
    str_118198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 29), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 669)
    kwargs_118199 = {}
    # Getting the type of 'ValueError' (line 669)
    ValueError_118197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 669)
    ValueError_call_result_118200 = invoke(stypy.reporting.localization.Localization(__file__, 669, 18), ValueError_118197, *[str_118198], **kwargs_118199)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 669, 12), ValueError_call_result_118200, 'raise parameter', BaseException)
    # SSA join for if statement (line 668)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'weights' (line 670)
    weights_118201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 11), 'weights')
    # Obtaining the member 'flags' of a type (line 670)
    flags_118202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 11), weights_118201, 'flags')
    # Obtaining the member 'contiguous' of a type (line 670)
    contiguous_118203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 11), flags_118202, 'contiguous')
    # Applying the 'not' unary operator (line 670)
    result_not__118204 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 7), 'not', contiguous_118203)
    
    # Testing the type of an if condition (line 670)
    if_condition_118205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 4), result_not__118204)
    # Assigning a type to the variable 'if_condition_118205' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'if_condition_118205', if_condition_118205)
    # SSA begins for if statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 671):
    
    # Assigning a Call to a Name (line 671):
    
    # Call to copy(...): (line 671)
    # Processing the call keyword arguments (line 671)
    kwargs_118208 = {}
    # Getting the type of 'weights' (line 671)
    weights_118206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 18), 'weights', False)
    # Obtaining the member 'copy' of a type (line 671)
    copy_118207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 18), weights_118206, 'copy')
    # Calling copy(args, kwargs) (line 671)
    copy_call_result_118209 = invoke(stypy.reporting.localization.Localization(__file__, 671, 18), copy_118207, *[], **kwargs_118208)
    
    # Assigning a type to the variable 'weights' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'weights', copy_call_result_118209)
    # SSA join for if statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 672):
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_118210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 4), 'int')
    
    # Call to _get_output(...): (line 672)
    # Processing the call arguments (line 672)
    # Getting the type of 'output' (line 672)
    output_118213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 51), 'output', False)
    # Getting the type of 'input' (line 672)
    input_118214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 59), 'input', False)
    # Processing the call keyword arguments (line 672)
    kwargs_118215 = {}
    # Getting the type of '_ni_support' (line 672)
    _ni_support_118211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 672)
    _get_output_118212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 27), _ni_support_118211, '_get_output')
    # Calling _get_output(args, kwargs) (line 672)
    _get_output_call_result_118216 = invoke(stypy.reporting.localization.Localization(__file__, 672, 27), _get_output_118212, *[output_118213, input_118214], **kwargs_118215)
    
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___118217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 4), _get_output_call_result_118216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_118218 = invoke(stypy.reporting.localization.Localization(__file__, 672, 4), getitem___118217, int_118210)
    
    # Assigning a type to the variable 'tuple_var_assignment_117035' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'tuple_var_assignment_117035', subscript_call_result_118218)
    
    # Assigning a Subscript to a Name (line 672):
    
    # Obtaining the type of the subscript
    int_118219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 4), 'int')
    
    # Call to _get_output(...): (line 672)
    # Processing the call arguments (line 672)
    # Getting the type of 'output' (line 672)
    output_118222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 51), 'output', False)
    # Getting the type of 'input' (line 672)
    input_118223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 59), 'input', False)
    # Processing the call keyword arguments (line 672)
    kwargs_118224 = {}
    # Getting the type of '_ni_support' (line 672)
    _ni_support_118220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 672)
    _get_output_118221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 27), _ni_support_118220, '_get_output')
    # Calling _get_output(args, kwargs) (line 672)
    _get_output_call_result_118225 = invoke(stypy.reporting.localization.Localization(__file__, 672, 27), _get_output_118221, *[output_118222, input_118223], **kwargs_118224)
    
    # Obtaining the member '__getitem__' of a type (line 672)
    getitem___118226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 4), _get_output_call_result_118225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 672)
    subscript_call_result_118227 = invoke(stypy.reporting.localization.Localization(__file__, 672, 4), getitem___118226, int_118219)
    
    # Assigning a type to the variable 'tuple_var_assignment_117036' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'tuple_var_assignment_117036', subscript_call_result_118227)
    
    # Assigning a Name to a Name (line 672):
    # Getting the type of 'tuple_var_assignment_117035' (line 672)
    tuple_var_assignment_117035_118228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'tuple_var_assignment_117035')
    # Assigning a type to the variable 'output' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'output', tuple_var_assignment_117035_118228)
    
    # Assigning a Name to a Name (line 672):
    # Getting the type of 'tuple_var_assignment_117036' (line 672)
    tuple_var_assignment_117036_118229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 4), 'tuple_var_assignment_117036')
    # Assigning a type to the variable 'return_value' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 12), 'return_value', tuple_var_assignment_117036_118229)
    
    # Assigning a Call to a Name (line 673):
    
    # Assigning a Call to a Name (line 673):
    
    # Call to _extend_mode_to_code(...): (line 673)
    # Processing the call arguments (line 673)
    # Getting the type of 'mode' (line 673)
    mode_118232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 44), 'mode', False)
    # Processing the call keyword arguments (line 673)
    kwargs_118233 = {}
    # Getting the type of '_ni_support' (line 673)
    _ni_support_118230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 673)
    _extend_mode_to_code_118231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 673, 11), _ni_support_118230, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 673)
    _extend_mode_to_code_call_result_118234 = invoke(stypy.reporting.localization.Localization(__file__, 673, 11), _extend_mode_to_code_118231, *[mode_118232], **kwargs_118233)
    
    # Assigning a type to the variable 'mode' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'mode', _extend_mode_to_code_call_result_118234)
    
    # Call to correlate(...): (line 674)
    # Processing the call arguments (line 674)
    # Getting the type of 'input' (line 674)
    input_118237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 24), 'input', False)
    # Getting the type of 'weights' (line 674)
    weights_118238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 31), 'weights', False)
    # Getting the type of 'output' (line 674)
    output_118239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 40), 'output', False)
    # Getting the type of 'mode' (line 674)
    mode_118240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 48), 'mode', False)
    # Getting the type of 'cval' (line 674)
    cval_118241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 54), 'cval', False)
    # Getting the type of 'origins' (line 674)
    origins_118242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 60), 'origins', False)
    # Processing the call keyword arguments (line 674)
    kwargs_118243 = {}
    # Getting the type of '_nd_image' (line 674)
    _nd_image_118235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), '_nd_image', False)
    # Obtaining the member 'correlate' of a type (line 674)
    correlate_118236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 4), _nd_image_118235, 'correlate')
    # Calling correlate(args, kwargs) (line 674)
    correlate_call_result_118244 = invoke(stypy.reporting.localization.Localization(__file__, 674, 4), correlate_118236, *[input_118237, weights_118238, output_118239, mode_118240, cval_118241, origins_118242], **kwargs_118243)
    
    # Getting the type of 'return_value' (line 675)
    return_value_118245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'stypy_return_type', return_value_118245)
    
    # ################# End of '_correlate_or_convolve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_correlate_or_convolve' in the type store
    # Getting the type of 'stypy_return_type' (line 651)
    stypy_return_type_118246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118246)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_correlate_or_convolve'
    return stypy_return_type_118246

# Assigning a type to the variable '_correlate_or_convolve' (line 651)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 0), '_correlate_or_convolve', _correlate_or_convolve)

@norecursion
def correlate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 679)
    None_118247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 37), 'None')
    str_118248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 48), 'str', 'reflect')
    float_118249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 64), 'float')
    int_118250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 21), 'int')
    defaults = [None_118247, str_118248, float_118249, int_118250]
    # Create a new context for function 'correlate'
    module_type_store = module_type_store.open_function_context('correlate', 678, 0, False)
    
    # Passed parameters checking function
    correlate.stypy_localization = localization
    correlate.stypy_type_of_self = None
    correlate.stypy_type_store = module_type_store
    correlate.stypy_function_name = 'correlate'
    correlate.stypy_param_names_list = ['input', 'weights', 'output', 'mode', 'cval', 'origin']
    correlate.stypy_varargs_param_name = None
    correlate.stypy_kwargs_param_name = None
    correlate.stypy_call_defaults = defaults
    correlate.stypy_call_varargs = varargs
    correlate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'correlate', ['input', 'weights', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'correlate', localization, ['input', 'weights', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'correlate(...)' code ##################

    str_118251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, (-1)), 'str', "\n    Multi-dimensional correlation.\n\n    The array is correlated with the given kernel.\n\n    Parameters\n    ----------\n    input : array-like\n        input array to filter\n    weights : ndarray\n        array of weights, same number of dimensions as input\n    output : array, optional\n        The ``output`` parameter passes an array in which to store the\n        filter output. Output array should have different name as\n        compared to input array to avoid aliasing errors.\n    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional\n        The ``mode`` parameter determines how the array borders are\n        handled, where ``cval`` is the value when mode is equal to\n        'constant'. Default is 'reflect'\n    cval : scalar, optional\n        Value to fill past edges of input if ``mode`` is 'constant'. Default\n        is 0.0\n    origin : scalar, optional\n        The ``origin`` parameter controls the placement of the filter.\n        Default 0\n\n    See Also\n    --------\n    convolve : Convolve an image with a kernel.\n    ")
    
    # Call to _correlate_or_convolve(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'input' (line 711)
    input_118253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 34), 'input', False)
    # Getting the type of 'weights' (line 711)
    weights_118254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 41), 'weights', False)
    # Getting the type of 'output' (line 711)
    output_118255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 50), 'output', False)
    # Getting the type of 'mode' (line 711)
    mode_118256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 58), 'mode', False)
    # Getting the type of 'cval' (line 711)
    cval_118257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 64), 'cval', False)
    # Getting the type of 'origin' (line 712)
    origin_118258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 34), 'origin', False)
    # Getting the type of 'False' (line 712)
    False_118259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 42), 'False', False)
    # Processing the call keyword arguments (line 711)
    kwargs_118260 = {}
    # Getting the type of '_correlate_or_convolve' (line 711)
    _correlate_or_convolve_118252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 11), '_correlate_or_convolve', False)
    # Calling _correlate_or_convolve(args, kwargs) (line 711)
    _correlate_or_convolve_call_result_118261 = invoke(stypy.reporting.localization.Localization(__file__, 711, 11), _correlate_or_convolve_118252, *[input_118253, weights_118254, output_118255, mode_118256, cval_118257, origin_118258, False_118259], **kwargs_118260)
    
    # Assigning a type to the variable 'stypy_return_type' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'stypy_return_type', _correlate_or_convolve_call_result_118261)
    
    # ################# End of 'correlate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'correlate' in the type store
    # Getting the type of 'stypy_return_type' (line 678)
    stypy_return_type_118262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118262)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'correlate'
    return stypy_return_type_118262

# Assigning a type to the variable 'correlate' (line 678)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 0), 'correlate', correlate)

@norecursion
def convolve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 716)
    None_118263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 36), 'None')
    str_118264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 47), 'str', 'reflect')
    float_118265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 63), 'float')
    int_118266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 20), 'int')
    defaults = [None_118263, str_118264, float_118265, int_118266]
    # Create a new context for function 'convolve'
    module_type_store = module_type_store.open_function_context('convolve', 715, 0, False)
    
    # Passed parameters checking function
    convolve.stypy_localization = localization
    convolve.stypy_type_of_self = None
    convolve.stypy_type_store = module_type_store
    convolve.stypy_function_name = 'convolve'
    convolve.stypy_param_names_list = ['input', 'weights', 'output', 'mode', 'cval', 'origin']
    convolve.stypy_varargs_param_name = None
    convolve.stypy_kwargs_param_name = None
    convolve.stypy_call_defaults = defaults
    convolve.stypy_call_varargs = varargs
    convolve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'convolve', ['input', 'weights', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'convolve', localization, ['input', 'weights', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'convolve(...)' code ##################

    str_118267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, (-1)), 'str', "\n    Multidimensional convolution.\n\n    The array is convolved with the given kernel.\n\n    Parameters\n    ----------\n    input : array_like\n        Input array to filter.\n    weights : array_like\n        Array of weights, same number of dimensions as input\n    output : ndarray, optional\n        The `output` parameter passes an array in which to store the\n        filter output. Output array should have different name as\n        compared to input array to avoid aliasing errors.\n    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional\n        the `mode` parameter determines how the array borders are\n        handled. For 'constant' mode, values beyond borders are set to be\n        `cval`. Default is 'reflect'.\n    cval : scalar, optional\n        Value to fill past edges of input if `mode` is 'constant'. Default\n        is 0.0\n    origin : array_like, optional\n        The `origin` parameter controls the placement of the filter,\n        relative to the centre of the current element of the input.\n        Default of 0 is equivalent to ``(0,)*input.ndim``.\n\n    Returns\n    -------\n    result : ndarray\n        The result of convolution of `input` with `weights`.\n\n    See Also\n    --------\n    correlate : Correlate an image with a kernel.\n\n    Notes\n    -----\n    Each value in result is :math:`C_i = \\sum_j{I_{i+k-j} W_j}`, where\n    W is the `weights` kernel,\n    j is the n-D spatial index over :math:`W`,\n    I is the `input` and k is the coordinate of the center of\n    W, specified by `origin` in the input parameters.\n\n    Examples\n    --------\n    Perhaps the simplest case to understand is ``mode='constant', cval=0.0``,\n    because in this case borders (i.e. where the `weights` kernel, centered\n    on any one value, extends beyond an edge of `input`.\n\n    >>> a = np.array([[1, 2, 0, 0],\n    ...               [5, 3, 0, 4],\n    ...               [0, 0, 0, 7],\n    ...               [9, 3, 0, 0]])\n    >>> k = np.array([[1,1,1],[1,1,0],[1,0,0]])\n    >>> from scipy import ndimage\n    >>> ndimage.convolve(a, k, mode='constant', cval=0.0)\n    array([[11, 10,  7,  4],\n           [10,  3, 11, 11],\n           [15, 12, 14,  7],\n           [12,  3,  7,  0]])\n\n    Setting ``cval=1.0`` is equivalent to padding the outer edge of `input`\n    with 1.0's (and then extracting only the original region of the result).\n\n    >>> ndimage.convolve(a, k, mode='constant', cval=1.0)\n    array([[13, 11,  8,  7],\n           [11,  3, 11, 14],\n           [16, 12, 14, 10],\n           [15,  6, 10,  5]])\n\n    With ``mode='reflect'`` (the default), outer values are reflected at the\n    edge of `input` to fill in missing values.\n\n    >>> b = np.array([[2, 0, 0],\n    ...               [1, 0, 0],\n    ...               [0, 0, 0]])\n    >>> k = np.array([[0,1,0], [0,1,0], [0,1,0]])\n    >>> ndimage.convolve(b, k, mode='reflect')\n    array([[5, 0, 0],\n           [3, 0, 0],\n           [1, 0, 0]])\n\n    This includes diagonally at the corners.\n\n    >>> k = np.array([[1,0,0],[0,1,0],[0,0,1]])\n    >>> ndimage.convolve(b, k)\n    array([[4, 2, 0],\n           [3, 2, 0],\n           [1, 1, 0]])\n\n    With ``mode='nearest'``, the single nearest value in to an edge in\n    `input` is repeated as many times as needed to match the overlapping\n    `weights`.\n\n    >>> c = np.array([[2, 0, 1],\n    ...               [1, 0, 0],\n    ...               [0, 0, 0]])\n    >>> k = np.array([[0, 1, 0],\n    ...               [0, 1, 0],\n    ...               [0, 1, 0],\n    ...               [0, 1, 0],\n    ...               [0, 1, 0]])\n    >>> ndimage.convolve(c, k, mode='nearest')\n    array([[7, 0, 3],\n           [5, 0, 2],\n           [3, 0, 1]])\n\n    ")
    
    # Call to _correlate_or_convolve(...): (line 827)
    # Processing the call arguments (line 827)
    # Getting the type of 'input' (line 827)
    input_118269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 34), 'input', False)
    # Getting the type of 'weights' (line 827)
    weights_118270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 41), 'weights', False)
    # Getting the type of 'output' (line 827)
    output_118271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 50), 'output', False)
    # Getting the type of 'mode' (line 827)
    mode_118272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 58), 'mode', False)
    # Getting the type of 'cval' (line 827)
    cval_118273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 64), 'cval', False)
    # Getting the type of 'origin' (line 828)
    origin_118274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 34), 'origin', False)
    # Getting the type of 'True' (line 828)
    True_118275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 42), 'True', False)
    # Processing the call keyword arguments (line 827)
    kwargs_118276 = {}
    # Getting the type of '_correlate_or_convolve' (line 827)
    _correlate_or_convolve_118268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 11), '_correlate_or_convolve', False)
    # Calling _correlate_or_convolve(args, kwargs) (line 827)
    _correlate_or_convolve_call_result_118277 = invoke(stypy.reporting.localization.Localization(__file__, 827, 11), _correlate_or_convolve_118268, *[input_118269, weights_118270, output_118271, mode_118272, cval_118273, origin_118274, True_118275], **kwargs_118276)
    
    # Assigning a type to the variable 'stypy_return_type' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'stypy_return_type', _correlate_or_convolve_call_result_118277)
    
    # ################# End of 'convolve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'convolve' in the type store
    # Getting the type of 'stypy_return_type' (line 715)
    stypy_return_type_118278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'convolve'
    return stypy_return_type_118278

# Assigning a type to the variable 'convolve' (line 715)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 0), 'convolve', convolve)

@norecursion
def uniform_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_118279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 39), 'int')
    # Getting the type of 'None' (line 832)
    None_118280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 50), 'None')
    str_118281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 26), 'str', 'reflect')
    float_118282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 42), 'float')
    int_118283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 54), 'int')
    defaults = [int_118279, None_118280, str_118281, float_118282, int_118283]
    # Create a new context for function 'uniform_filter1d'
    module_type_store = module_type_store.open_function_context('uniform_filter1d', 831, 0, False)
    
    # Passed parameters checking function
    uniform_filter1d.stypy_localization = localization
    uniform_filter1d.stypy_type_of_self = None
    uniform_filter1d.stypy_type_store = module_type_store
    uniform_filter1d.stypy_function_name = 'uniform_filter1d'
    uniform_filter1d.stypy_param_names_list = ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin']
    uniform_filter1d.stypy_varargs_param_name = None
    uniform_filter1d.stypy_kwargs_param_name = None
    uniform_filter1d.stypy_call_defaults = defaults
    uniform_filter1d.stypy_call_varargs = varargs
    uniform_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uniform_filter1d', ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uniform_filter1d', localization, ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uniform_filter1d(...)' code ##################

    str_118284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, (-1)), 'str', 'Calculate a one-dimensional uniform filter along the given axis.\n\n    The lines of the array along the given axis are filtered with a\n    uniform filter of given size.\n\n    Parameters\n    ----------\n    %(input)s\n    size : int\n        length of uniform filter\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Examples\n    --------\n    >>> from scipy.ndimage import uniform_filter1d\n    >>> uniform_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)\n    array([4, 3, 4, 1, 4, 6, 6, 3])\n    ')
    
    # Assigning a Call to a Name (line 856):
    
    # Assigning a Call to a Name (line 856):
    
    # Call to asarray(...): (line 856)
    # Processing the call arguments (line 856)
    # Getting the type of 'input' (line 856)
    input_118287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 26), 'input', False)
    # Processing the call keyword arguments (line 856)
    kwargs_118288 = {}
    # Getting the type of 'numpy' (line 856)
    numpy_118285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 856)
    asarray_118286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 12), numpy_118285, 'asarray')
    # Calling asarray(args, kwargs) (line 856)
    asarray_call_result_118289 = invoke(stypy.reporting.localization.Localization(__file__, 856, 12), asarray_118286, *[input_118287], **kwargs_118288)
    
    # Assigning a type to the variable 'input' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'input', asarray_call_result_118289)
    
    
    # Call to iscomplexobj(...): (line 857)
    # Processing the call arguments (line 857)
    # Getting the type of 'input' (line 857)
    input_118292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 26), 'input', False)
    # Processing the call keyword arguments (line 857)
    kwargs_118293 = {}
    # Getting the type of 'numpy' (line 857)
    numpy_118290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 857)
    iscomplexobj_118291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 857, 7), numpy_118290, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 857)
    iscomplexobj_call_result_118294 = invoke(stypy.reporting.localization.Localization(__file__, 857, 7), iscomplexobj_118291, *[input_118292], **kwargs_118293)
    
    # Testing the type of an if condition (line 857)
    if_condition_118295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 857, 4), iscomplexobj_call_result_118294)
    # Assigning a type to the variable 'if_condition_118295' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'if_condition_118295', if_condition_118295)
    # SSA begins for if statement (line 857)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 858)
    # Processing the call arguments (line 858)
    str_118297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 858)
    kwargs_118298 = {}
    # Getting the type of 'TypeError' (line 858)
    TypeError_118296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 858)
    TypeError_call_result_118299 = invoke(stypy.reporting.localization.Localization(__file__, 858, 14), TypeError_118296, *[str_118297], **kwargs_118298)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 858, 8), TypeError_call_result_118299, 'raise parameter', BaseException)
    # SSA join for if statement (line 857)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 859):
    
    # Assigning a Call to a Name (line 859):
    
    # Call to _check_axis(...): (line 859)
    # Processing the call arguments (line 859)
    # Getting the type of 'axis' (line 859)
    axis_118302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 35), 'axis', False)
    # Getting the type of 'input' (line 859)
    input_118303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 859)
    ndim_118304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 41), input_118303, 'ndim')
    # Processing the call keyword arguments (line 859)
    kwargs_118305 = {}
    # Getting the type of '_ni_support' (line 859)
    _ni_support_118300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 859)
    _check_axis_118301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 11), _ni_support_118300, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 859)
    _check_axis_call_result_118306 = invoke(stypy.reporting.localization.Localization(__file__, 859, 11), _check_axis_118301, *[axis_118302, ndim_118304], **kwargs_118305)
    
    # Assigning a type to the variable 'axis' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'axis', _check_axis_call_result_118306)
    
    
    # Getting the type of 'size' (line 860)
    size_118307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 7), 'size')
    int_118308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 14), 'int')
    # Applying the binary operator '<' (line 860)
    result_lt_118309 = python_operator(stypy.reporting.localization.Localization(__file__, 860, 7), '<', size_118307, int_118308)
    
    # Testing the type of an if condition (line 860)
    if_condition_118310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 860, 4), result_lt_118309)
    # Assigning a type to the variable 'if_condition_118310' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'if_condition_118310', if_condition_118310)
    # SSA begins for if statement (line 860)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 861)
    # Processing the call arguments (line 861)
    str_118312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 27), 'str', 'incorrect filter size')
    # Processing the call keyword arguments (line 861)
    kwargs_118313 = {}
    # Getting the type of 'RuntimeError' (line 861)
    RuntimeError_118311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 861)
    RuntimeError_call_result_118314 = invoke(stypy.reporting.localization.Localization(__file__, 861, 14), RuntimeError_118311, *[str_118312], **kwargs_118313)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 861, 8), RuntimeError_call_result_118314, 'raise parameter', BaseException)
    # SSA join for if statement (line 860)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 862):
    
    # Assigning a Subscript to a Name (line 862):
    
    # Obtaining the type of the subscript
    int_118315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 4), 'int')
    
    # Call to _get_output(...): (line 862)
    # Processing the call arguments (line 862)
    # Getting the type of 'output' (line 862)
    output_118318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 51), 'output', False)
    # Getting the type of 'input' (line 862)
    input_118319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 59), 'input', False)
    # Processing the call keyword arguments (line 862)
    kwargs_118320 = {}
    # Getting the type of '_ni_support' (line 862)
    _ni_support_118316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 862)
    _get_output_118317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 27), _ni_support_118316, '_get_output')
    # Calling _get_output(args, kwargs) (line 862)
    _get_output_call_result_118321 = invoke(stypy.reporting.localization.Localization(__file__, 862, 27), _get_output_118317, *[output_118318, input_118319], **kwargs_118320)
    
    # Obtaining the member '__getitem__' of a type (line 862)
    getitem___118322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 4), _get_output_call_result_118321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 862)
    subscript_call_result_118323 = invoke(stypy.reporting.localization.Localization(__file__, 862, 4), getitem___118322, int_118315)
    
    # Assigning a type to the variable 'tuple_var_assignment_117037' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'tuple_var_assignment_117037', subscript_call_result_118323)
    
    # Assigning a Subscript to a Name (line 862):
    
    # Obtaining the type of the subscript
    int_118324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 4), 'int')
    
    # Call to _get_output(...): (line 862)
    # Processing the call arguments (line 862)
    # Getting the type of 'output' (line 862)
    output_118327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 51), 'output', False)
    # Getting the type of 'input' (line 862)
    input_118328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 59), 'input', False)
    # Processing the call keyword arguments (line 862)
    kwargs_118329 = {}
    # Getting the type of '_ni_support' (line 862)
    _ni_support_118325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 862)
    _get_output_118326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 27), _ni_support_118325, '_get_output')
    # Calling _get_output(args, kwargs) (line 862)
    _get_output_call_result_118330 = invoke(stypy.reporting.localization.Localization(__file__, 862, 27), _get_output_118326, *[output_118327, input_118328], **kwargs_118329)
    
    # Obtaining the member '__getitem__' of a type (line 862)
    getitem___118331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 4), _get_output_call_result_118330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 862)
    subscript_call_result_118332 = invoke(stypy.reporting.localization.Localization(__file__, 862, 4), getitem___118331, int_118324)
    
    # Assigning a type to the variable 'tuple_var_assignment_117038' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'tuple_var_assignment_117038', subscript_call_result_118332)
    
    # Assigning a Name to a Name (line 862):
    # Getting the type of 'tuple_var_assignment_117037' (line 862)
    tuple_var_assignment_117037_118333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'tuple_var_assignment_117037')
    # Assigning a type to the variable 'output' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'output', tuple_var_assignment_117037_118333)
    
    # Assigning a Name to a Name (line 862):
    # Getting the type of 'tuple_var_assignment_117038' (line 862)
    tuple_var_assignment_117038_118334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'tuple_var_assignment_117038')
    # Assigning a type to the variable 'return_value' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 12), 'return_value', tuple_var_assignment_117038_118334)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 863)
    size_118335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'size')
    int_118336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 16), 'int')
    # Applying the binary operator '//' (line 863)
    result_floordiv_118337 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 8), '//', size_118335, int_118336)
    
    # Getting the type of 'origin' (line 863)
    origin_118338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 20), 'origin')
    # Applying the binary operator '+' (line 863)
    result_add_118339 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 8), '+', result_floordiv_118337, origin_118338)
    
    int_118340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 29), 'int')
    # Applying the binary operator '<' (line 863)
    result_lt_118341 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 8), '<', result_add_118339, int_118340)
    
    
    # Getting the type of 'size' (line 863)
    size_118342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 36), 'size')
    int_118343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 44), 'int')
    # Applying the binary operator '//' (line 863)
    result_floordiv_118344 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 36), '//', size_118342, int_118343)
    
    # Getting the type of 'origin' (line 863)
    origin_118345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 48), 'origin')
    # Applying the binary operator '+' (line 863)
    result_add_118346 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 36), '+', result_floordiv_118344, origin_118345)
    
    # Getting the type of 'size' (line 863)
    size_118347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 58), 'size')
    # Applying the binary operator '>=' (line 863)
    result_ge_118348 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 36), '>=', result_add_118346, size_118347)
    
    # Applying the binary operator 'or' (line 863)
    result_or_keyword_118349 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 7), 'or', result_lt_118341, result_ge_118348)
    
    # Testing the type of an if condition (line 863)
    if_condition_118350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 863, 4), result_or_keyword_118349)
    # Assigning a type to the variable 'if_condition_118350' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'if_condition_118350', if_condition_118350)
    # SSA begins for if statement (line 863)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 864)
    # Processing the call arguments (line 864)
    str_118352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 25), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 864)
    kwargs_118353 = {}
    # Getting the type of 'ValueError' (line 864)
    ValueError_118351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 864)
    ValueError_call_result_118354 = invoke(stypy.reporting.localization.Localization(__file__, 864, 14), ValueError_118351, *[str_118352], **kwargs_118353)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 864, 8), ValueError_call_result_118354, 'raise parameter', BaseException)
    # SSA join for if statement (line 863)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 865):
    
    # Assigning a Call to a Name (line 865):
    
    # Call to _extend_mode_to_code(...): (line 865)
    # Processing the call arguments (line 865)
    # Getting the type of 'mode' (line 865)
    mode_118357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 44), 'mode', False)
    # Processing the call keyword arguments (line 865)
    kwargs_118358 = {}
    # Getting the type of '_ni_support' (line 865)
    _ni_support_118355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 865)
    _extend_mode_to_code_118356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 865, 11), _ni_support_118355, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 865)
    _extend_mode_to_code_call_result_118359 = invoke(stypy.reporting.localization.Localization(__file__, 865, 11), _extend_mode_to_code_118356, *[mode_118357], **kwargs_118358)
    
    # Assigning a type to the variable 'mode' (line 865)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'mode', _extend_mode_to_code_call_result_118359)
    
    # Call to uniform_filter1d(...): (line 866)
    # Processing the call arguments (line 866)
    # Getting the type of 'input' (line 866)
    input_118362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 31), 'input', False)
    # Getting the type of 'size' (line 866)
    size_118363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 38), 'size', False)
    # Getting the type of 'axis' (line 866)
    axis_118364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 44), 'axis', False)
    # Getting the type of 'output' (line 866)
    output_118365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 50), 'output', False)
    # Getting the type of 'mode' (line 866)
    mode_118366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 58), 'mode', False)
    # Getting the type of 'cval' (line 866)
    cval_118367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 64), 'cval', False)
    # Getting the type of 'origin' (line 867)
    origin_118368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 31), 'origin', False)
    # Processing the call keyword arguments (line 866)
    kwargs_118369 = {}
    # Getting the type of '_nd_image' (line 866)
    _nd_image_118360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), '_nd_image', False)
    # Obtaining the member 'uniform_filter1d' of a type (line 866)
    uniform_filter1d_118361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 4), _nd_image_118360, 'uniform_filter1d')
    # Calling uniform_filter1d(args, kwargs) (line 866)
    uniform_filter1d_call_result_118370 = invoke(stypy.reporting.localization.Localization(__file__, 866, 4), uniform_filter1d_118361, *[input_118362, size_118363, axis_118364, output_118365, mode_118366, cval_118367, origin_118368], **kwargs_118369)
    
    # Getting the type of 'return_value' (line 868)
    return_value_118371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'stypy_return_type', return_value_118371)
    
    # ################# End of 'uniform_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uniform_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 831)
    stypy_return_type_118372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118372)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uniform_filter1d'
    return stypy_return_type_118372

# Assigning a type to the variable 'uniform_filter1d' (line 831)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 0), 'uniform_filter1d', uniform_filter1d)

@norecursion
def uniform_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_118373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 31), 'int')
    # Getting the type of 'None' (line 872)
    None_118374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 41), 'None')
    str_118375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 52), 'str', 'reflect')
    float_118376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 24), 'float')
    int_118377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 36), 'int')
    defaults = [int_118373, None_118374, str_118375, float_118376, int_118377]
    # Create a new context for function 'uniform_filter'
    module_type_store = module_type_store.open_function_context('uniform_filter', 871, 0, False)
    
    # Passed parameters checking function
    uniform_filter.stypy_localization = localization
    uniform_filter.stypy_type_of_self = None
    uniform_filter.stypy_type_store = module_type_store
    uniform_filter.stypy_function_name = 'uniform_filter'
    uniform_filter.stypy_param_names_list = ['input', 'size', 'output', 'mode', 'cval', 'origin']
    uniform_filter.stypy_varargs_param_name = None
    uniform_filter.stypy_kwargs_param_name = None
    uniform_filter.stypy_call_defaults = defaults
    uniform_filter.stypy_call_varargs = varargs
    uniform_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'uniform_filter', ['input', 'size', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'uniform_filter', localization, ['input', 'size', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'uniform_filter(...)' code ##################

    str_118378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, (-1)), 'str', 'Multi-dimensional uniform filter.\n\n    Parameters\n    ----------\n    %(input)s\n    size : int or sequence of ints, optional\n        The sizes of the uniform filter are given for each axis as a\n        sequence, or as a single number, in which case the size is\n        equal for all axes.\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    uniform_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Notes\n    -----\n    The multi-dimensional filter is implemented as a sequence of\n    one-dimensional uniform filters. The intermediate arrays are stored\n    in the same data type as the output. Therefore, for output types\n    with a limited precision, the results may be imprecise because\n    intermediate results may be stored with insufficient precision.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.uniform_filter(ascent, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Name (line 915):
    
    # Assigning a Call to a Name (line 915):
    
    # Call to asarray(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'input' (line 915)
    input_118381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 26), 'input', False)
    # Processing the call keyword arguments (line 915)
    kwargs_118382 = {}
    # Getting the type of 'numpy' (line 915)
    numpy_118379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 915)
    asarray_118380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 12), numpy_118379, 'asarray')
    # Calling asarray(args, kwargs) (line 915)
    asarray_call_result_118383 = invoke(stypy.reporting.localization.Localization(__file__, 915, 12), asarray_118380, *[input_118381], **kwargs_118382)
    
    # Assigning a type to the variable 'input' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 4), 'input', asarray_call_result_118383)
    
    # Assigning a Call to a Tuple (line 916):
    
    # Assigning a Subscript to a Name (line 916):
    
    # Obtaining the type of the subscript
    int_118384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 4), 'int')
    
    # Call to _get_output(...): (line 916)
    # Processing the call arguments (line 916)
    # Getting the type of 'output' (line 916)
    output_118387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 51), 'output', False)
    # Getting the type of 'input' (line 916)
    input_118388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 59), 'input', False)
    # Processing the call keyword arguments (line 916)
    kwargs_118389 = {}
    # Getting the type of '_ni_support' (line 916)
    _ni_support_118385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 916)
    _get_output_118386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 27), _ni_support_118385, '_get_output')
    # Calling _get_output(args, kwargs) (line 916)
    _get_output_call_result_118390 = invoke(stypy.reporting.localization.Localization(__file__, 916, 27), _get_output_118386, *[output_118387, input_118388], **kwargs_118389)
    
    # Obtaining the member '__getitem__' of a type (line 916)
    getitem___118391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 4), _get_output_call_result_118390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 916)
    subscript_call_result_118392 = invoke(stypy.reporting.localization.Localization(__file__, 916, 4), getitem___118391, int_118384)
    
    # Assigning a type to the variable 'tuple_var_assignment_117039' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'tuple_var_assignment_117039', subscript_call_result_118392)
    
    # Assigning a Subscript to a Name (line 916):
    
    # Obtaining the type of the subscript
    int_118393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 4), 'int')
    
    # Call to _get_output(...): (line 916)
    # Processing the call arguments (line 916)
    # Getting the type of 'output' (line 916)
    output_118396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 51), 'output', False)
    # Getting the type of 'input' (line 916)
    input_118397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 59), 'input', False)
    # Processing the call keyword arguments (line 916)
    kwargs_118398 = {}
    # Getting the type of '_ni_support' (line 916)
    _ni_support_118394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 916)
    _get_output_118395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 27), _ni_support_118394, '_get_output')
    # Calling _get_output(args, kwargs) (line 916)
    _get_output_call_result_118399 = invoke(stypy.reporting.localization.Localization(__file__, 916, 27), _get_output_118395, *[output_118396, input_118397], **kwargs_118398)
    
    # Obtaining the member '__getitem__' of a type (line 916)
    getitem___118400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 916, 4), _get_output_call_result_118399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 916)
    subscript_call_result_118401 = invoke(stypy.reporting.localization.Localization(__file__, 916, 4), getitem___118400, int_118393)
    
    # Assigning a type to the variable 'tuple_var_assignment_117040' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'tuple_var_assignment_117040', subscript_call_result_118401)
    
    # Assigning a Name to a Name (line 916):
    # Getting the type of 'tuple_var_assignment_117039' (line 916)
    tuple_var_assignment_117039_118402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'tuple_var_assignment_117039')
    # Assigning a type to the variable 'output' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'output', tuple_var_assignment_117039_118402)
    
    # Assigning a Name to a Name (line 916):
    # Getting the type of 'tuple_var_assignment_117040' (line 916)
    tuple_var_assignment_117040_118403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'tuple_var_assignment_117040')
    # Assigning a type to the variable 'return_value' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 12), 'return_value', tuple_var_assignment_117040_118403)
    
    # Assigning a Call to a Name (line 917):
    
    # Assigning a Call to a Name (line 917):
    
    # Call to _normalize_sequence(...): (line 917)
    # Processing the call arguments (line 917)
    # Getting the type of 'size' (line 917)
    size_118406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 44), 'size', False)
    # Getting the type of 'input' (line 917)
    input_118407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 917)
    ndim_118408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 50), input_118407, 'ndim')
    # Processing the call keyword arguments (line 917)
    kwargs_118409 = {}
    # Getting the type of '_ni_support' (line 917)
    _ni_support_118404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 917)
    _normalize_sequence_118405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 12), _ni_support_118404, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 917)
    _normalize_sequence_call_result_118410 = invoke(stypy.reporting.localization.Localization(__file__, 917, 12), _normalize_sequence_118405, *[size_118406, ndim_118408], **kwargs_118409)
    
    # Assigning a type to the variable 'sizes' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 4), 'sizes', _normalize_sequence_call_result_118410)
    
    # Assigning a Call to a Name (line 918):
    
    # Assigning a Call to a Name (line 918):
    
    # Call to _normalize_sequence(...): (line 918)
    # Processing the call arguments (line 918)
    # Getting the type of 'origin' (line 918)
    origin_118413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 46), 'origin', False)
    # Getting the type of 'input' (line 918)
    input_118414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 918)
    ndim_118415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 54), input_118414, 'ndim')
    # Processing the call keyword arguments (line 918)
    kwargs_118416 = {}
    # Getting the type of '_ni_support' (line 918)
    _ni_support_118411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 918)
    _normalize_sequence_118412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 14), _ni_support_118411, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 918)
    _normalize_sequence_call_result_118417 = invoke(stypy.reporting.localization.Localization(__file__, 918, 14), _normalize_sequence_118412, *[origin_118413, ndim_118415], **kwargs_118416)
    
    # Assigning a type to the variable 'origins' (line 918)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 4), 'origins', _normalize_sequence_call_result_118417)
    
    # Assigning a Call to a Name (line 919):
    
    # Assigning a Call to a Name (line 919):
    
    # Call to _normalize_sequence(...): (line 919)
    # Processing the call arguments (line 919)
    # Getting the type of 'mode' (line 919)
    mode_118420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 44), 'mode', False)
    # Getting the type of 'input' (line 919)
    input_118421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 50), 'input', False)
    # Obtaining the member 'ndim' of a type (line 919)
    ndim_118422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 50), input_118421, 'ndim')
    # Processing the call keyword arguments (line 919)
    kwargs_118423 = {}
    # Getting the type of '_ni_support' (line 919)
    _ni_support_118418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 12), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 919)
    _normalize_sequence_118419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 12), _ni_support_118418, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 919)
    _normalize_sequence_call_result_118424 = invoke(stypy.reporting.localization.Localization(__file__, 919, 12), _normalize_sequence_118419, *[mode_118420, ndim_118422], **kwargs_118423)
    
    # Assigning a type to the variable 'modes' (line 919)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 4), 'modes', _normalize_sequence_call_result_118424)
    
    # Assigning a Call to a Name (line 920):
    
    # Assigning a Call to a Name (line 920):
    
    # Call to list(...): (line 920)
    # Processing the call arguments (line 920)
    
    # Call to range(...): (line 920)
    # Processing the call arguments (line 920)
    # Getting the type of 'input' (line 920)
    input_118427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 22), 'input', False)
    # Obtaining the member 'ndim' of a type (line 920)
    ndim_118428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 22), input_118427, 'ndim')
    # Processing the call keyword arguments (line 920)
    kwargs_118429 = {}
    # Getting the type of 'range' (line 920)
    range_118426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 16), 'range', False)
    # Calling range(args, kwargs) (line 920)
    range_call_result_118430 = invoke(stypy.reporting.localization.Localization(__file__, 920, 16), range_118426, *[ndim_118428], **kwargs_118429)
    
    # Processing the call keyword arguments (line 920)
    kwargs_118431 = {}
    # Getting the type of 'list' (line 920)
    list_118425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 11), 'list', False)
    # Calling list(args, kwargs) (line 920)
    list_call_result_118432 = invoke(stypy.reporting.localization.Localization(__file__, 920, 11), list_118425, *[range_call_result_118430], **kwargs_118431)
    
    # Assigning a type to the variable 'axes' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'axes', list_call_result_118432)
    
    # Assigning a ListComp to a Name (line 921):
    
    # Assigning a ListComp to a Name (line 921):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 922)
    # Processing the call arguments (line 922)
    
    # Call to len(...): (line 922)
    # Processing the call arguments (line 922)
    # Getting the type of 'axes' (line 922)
    axes_118458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 32), 'axes', False)
    # Processing the call keyword arguments (line 922)
    kwargs_118459 = {}
    # Getting the type of 'len' (line 922)
    len_118457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 28), 'len', False)
    # Calling len(args, kwargs) (line 922)
    len_call_result_118460 = invoke(stypy.reporting.localization.Localization(__file__, 922, 28), len_118457, *[axes_118458], **kwargs_118459)
    
    # Processing the call keyword arguments (line 922)
    kwargs_118461 = {}
    # Getting the type of 'range' (line 922)
    range_118456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 22), 'range', False)
    # Calling range(args, kwargs) (line 922)
    range_call_result_118462 = invoke(stypy.reporting.localization.Localization(__file__, 922, 22), range_118456, *[len_call_result_118460], **kwargs_118461)
    
    comprehension_118463 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 12), range_call_result_118462)
    # Assigning a type to the variable 'ii' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 12), 'ii', comprehension_118463)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 922)
    ii_118450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 48), 'ii')
    # Getting the type of 'sizes' (line 922)
    sizes_118451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 42), 'sizes')
    # Obtaining the member '__getitem__' of a type (line 922)
    getitem___118452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 42), sizes_118451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 922)
    subscript_call_result_118453 = invoke(stypy.reporting.localization.Localization(__file__, 922, 42), getitem___118452, ii_118450)
    
    int_118454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 54), 'int')
    # Applying the binary operator '>' (line 922)
    result_gt_118455 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 42), '>', subscript_call_result_118453, int_118454)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 921)
    tuple_118433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 921)
    # Adding element type (line 921)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 921)
    ii_118434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 18), 'ii')
    # Getting the type of 'axes' (line 921)
    axes_118435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 13), 'axes')
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___118436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 13), axes_118435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_118437 = invoke(stypy.reporting.localization.Localization(__file__, 921, 13), getitem___118436, ii_118434)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 13), tuple_118433, subscript_call_result_118437)
    # Adding element type (line 921)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 921)
    ii_118438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 29), 'ii')
    # Getting the type of 'sizes' (line 921)
    sizes_118439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 23), 'sizes')
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___118440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 23), sizes_118439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_118441 = invoke(stypy.reporting.localization.Localization(__file__, 921, 23), getitem___118440, ii_118438)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 13), tuple_118433, subscript_call_result_118441)
    # Adding element type (line 921)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 921)
    ii_118442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 42), 'ii')
    # Getting the type of 'origins' (line 921)
    origins_118443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 34), 'origins')
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___118444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 34), origins_118443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_118445 = invoke(stypy.reporting.localization.Localization(__file__, 921, 34), getitem___118444, ii_118442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 13), tuple_118433, subscript_call_result_118445)
    # Adding element type (line 921)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 921)
    ii_118446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 53), 'ii')
    # Getting the type of 'modes' (line 921)
    modes_118447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 47), 'modes')
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___118448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 47), modes_118447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_118449 = invoke(stypy.reporting.localization.Localization(__file__, 921, 47), getitem___118448, ii_118446)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 13), tuple_118433, subscript_call_result_118449)
    
    list_118464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 921, 12), list_118464, tuple_118433)
    # Assigning a type to the variable 'axes' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'axes', list_118464)
    
    
    
    # Call to len(...): (line 923)
    # Processing the call arguments (line 923)
    # Getting the type of 'axes' (line 923)
    axes_118466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 11), 'axes', False)
    # Processing the call keyword arguments (line 923)
    kwargs_118467 = {}
    # Getting the type of 'len' (line 923)
    len_118465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 7), 'len', False)
    # Calling len(args, kwargs) (line 923)
    len_call_result_118468 = invoke(stypy.reporting.localization.Localization(__file__, 923, 7), len_118465, *[axes_118466], **kwargs_118467)
    
    int_118469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 19), 'int')
    # Applying the binary operator '>' (line 923)
    result_gt_118470 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 7), '>', len_call_result_118468, int_118469)
    
    # Testing the type of an if condition (line 923)
    if_condition_118471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 4), result_gt_118470)
    # Assigning a type to the variable 'if_condition_118471' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'if_condition_118471', if_condition_118471)
    # SSA begins for if statement (line 923)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 924)
    axes_118472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 40), 'axes')
    # Testing the type of a for loop iterable (line 924)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 924, 8), axes_118472)
    # Getting the type of the for loop variable (line 924)
    for_loop_var_118473 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 924, 8), axes_118472)
    # Assigning a type to the variable 'axis' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 8), for_loop_var_118473))
    # Assigning a type to the variable 'size' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'size', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 8), for_loop_var_118473))
    # Assigning a type to the variable 'origin' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 8), for_loop_var_118473))
    # Assigning a type to the variable 'mode' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'mode', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 8), for_loop_var_118473))
    # SSA begins for a for statement (line 924)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to uniform_filter1d(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'input' (line 925)
    input_118475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 29), 'input', False)
    
    # Call to int(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'size' (line 925)
    size_118477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 40), 'size', False)
    # Processing the call keyword arguments (line 925)
    kwargs_118478 = {}
    # Getting the type of 'int' (line 925)
    int_118476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 36), 'int', False)
    # Calling int(args, kwargs) (line 925)
    int_call_result_118479 = invoke(stypy.reporting.localization.Localization(__file__, 925, 36), int_118476, *[size_118477], **kwargs_118478)
    
    # Getting the type of 'axis' (line 925)
    axis_118480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 47), 'axis', False)
    # Getting the type of 'output' (line 925)
    output_118481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 53), 'output', False)
    # Getting the type of 'mode' (line 925)
    mode_118482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 61), 'mode', False)
    # Getting the type of 'cval' (line 926)
    cval_118483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 29), 'cval', False)
    # Getting the type of 'origin' (line 926)
    origin_118484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 35), 'origin', False)
    # Processing the call keyword arguments (line 925)
    kwargs_118485 = {}
    # Getting the type of 'uniform_filter1d' (line 925)
    uniform_filter1d_118474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'uniform_filter1d', False)
    # Calling uniform_filter1d(args, kwargs) (line 925)
    uniform_filter1d_call_result_118486 = invoke(stypy.reporting.localization.Localization(__file__, 925, 12), uniform_filter1d_118474, *[input_118475, int_call_result_118479, axis_118480, output_118481, mode_118482, cval_118483, origin_118484], **kwargs_118485)
    
    
    # Assigning a Name to a Name (line 927):
    
    # Assigning a Name to a Name (line 927):
    # Getting the type of 'output' (line 927)
    output_118487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 20), 'output')
    # Assigning a type to the variable 'input' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 12), 'input', output_118487)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 923)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 929):
    
    # Assigning a Subscript to a Subscript (line 929):
    
    # Obtaining the type of the subscript
    Ellipsis_118488 = Ellipsis
    # Getting the type of 'input' (line 929)
    input_118489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 22), 'input')
    # Obtaining the member '__getitem__' of a type (line 929)
    getitem___118490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 22), input_118489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 929)
    subscript_call_result_118491 = invoke(stypy.reporting.localization.Localization(__file__, 929, 22), getitem___118490, Ellipsis_118488)
    
    # Getting the type of 'output' (line 929)
    output_118492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 8), 'output')
    Ellipsis_118493 = Ellipsis
    # Storing an element on a container (line 929)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 929, 8), output_118492, (Ellipsis_118493, subscript_call_result_118491))
    # SSA join for if statement (line 923)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 930)
    return_value_118494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'stypy_return_type', return_value_118494)
    
    # ################# End of 'uniform_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'uniform_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 871)
    stypy_return_type_118495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118495)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'uniform_filter'
    return stypy_return_type_118495

# Assigning a type to the variable 'uniform_filter' (line 871)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 0), 'uniform_filter', uniform_filter)

@norecursion
def minimum_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_118496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 39), 'int')
    # Getting the type of 'None' (line 934)
    None_118497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 50), 'None')
    str_118498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 26), 'str', 'reflect')
    float_118499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 42), 'float')
    int_118500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 54), 'int')
    defaults = [int_118496, None_118497, str_118498, float_118499, int_118500]
    # Create a new context for function 'minimum_filter1d'
    module_type_store = module_type_store.open_function_context('minimum_filter1d', 933, 0, False)
    
    # Passed parameters checking function
    minimum_filter1d.stypy_localization = localization
    minimum_filter1d.stypy_type_of_self = None
    minimum_filter1d.stypy_type_store = module_type_store
    minimum_filter1d.stypy_function_name = 'minimum_filter1d'
    minimum_filter1d.stypy_param_names_list = ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin']
    minimum_filter1d.stypy_varargs_param_name = None
    minimum_filter1d.stypy_kwargs_param_name = None
    minimum_filter1d.stypy_call_defaults = defaults
    minimum_filter1d.stypy_call_varargs = varargs
    minimum_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimum_filter1d', ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimum_filter1d', localization, ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimum_filter1d(...)' code ##################

    str_118501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 969, (-1)), 'str', 'Calculate a one-dimensional minimum filter along the given axis.\n\n    The lines of the array along the given axis are filtered with a\n    minimum filter of given size.\n\n    Parameters\n    ----------\n    %(input)s\n    size : int\n        length along which to calculate 1D minimum\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Notes\n    -----\n    This function implements the MINLIST algorithm [1]_, as described by\n    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being\n    the `input` length, regardless of filter size.\n\n    References\n    ----------\n    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777\n    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html\n\n\n    Examples\n    --------\n    >>> from scipy.ndimage import minimum_filter1d\n    >>> minimum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)\n    array([2, 0, 0, 0, 1, 1, 0, 0])\n    ')
    
    # Assigning a Call to a Name (line 970):
    
    # Assigning a Call to a Name (line 970):
    
    # Call to asarray(...): (line 970)
    # Processing the call arguments (line 970)
    # Getting the type of 'input' (line 970)
    input_118504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 26), 'input', False)
    # Processing the call keyword arguments (line 970)
    kwargs_118505 = {}
    # Getting the type of 'numpy' (line 970)
    numpy_118502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 970)
    asarray_118503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 12), numpy_118502, 'asarray')
    # Calling asarray(args, kwargs) (line 970)
    asarray_call_result_118506 = invoke(stypy.reporting.localization.Localization(__file__, 970, 12), asarray_118503, *[input_118504], **kwargs_118505)
    
    # Assigning a type to the variable 'input' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'input', asarray_call_result_118506)
    
    
    # Call to iscomplexobj(...): (line 971)
    # Processing the call arguments (line 971)
    # Getting the type of 'input' (line 971)
    input_118509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 26), 'input', False)
    # Processing the call keyword arguments (line 971)
    kwargs_118510 = {}
    # Getting the type of 'numpy' (line 971)
    numpy_118507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 971)
    iscomplexobj_118508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 971, 7), numpy_118507, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 971)
    iscomplexobj_call_result_118511 = invoke(stypy.reporting.localization.Localization(__file__, 971, 7), iscomplexobj_118508, *[input_118509], **kwargs_118510)
    
    # Testing the type of an if condition (line 971)
    if_condition_118512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 971, 4), iscomplexobj_call_result_118511)
    # Assigning a type to the variable 'if_condition_118512' (line 971)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 971, 4), 'if_condition_118512', if_condition_118512)
    # SSA begins for if statement (line 971)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 972)
    # Processing the call arguments (line 972)
    str_118514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 972)
    kwargs_118515 = {}
    # Getting the type of 'TypeError' (line 972)
    TypeError_118513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 972)
    TypeError_call_result_118516 = invoke(stypy.reporting.localization.Localization(__file__, 972, 14), TypeError_118513, *[str_118514], **kwargs_118515)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 972, 8), TypeError_call_result_118516, 'raise parameter', BaseException)
    # SSA join for if statement (line 971)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 973):
    
    # Assigning a Call to a Name (line 973):
    
    # Call to _check_axis(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'axis' (line 973)
    axis_118519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 35), 'axis', False)
    # Getting the type of 'input' (line 973)
    input_118520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 973)
    ndim_118521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 41), input_118520, 'ndim')
    # Processing the call keyword arguments (line 973)
    kwargs_118522 = {}
    # Getting the type of '_ni_support' (line 973)
    _ni_support_118517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 973)
    _check_axis_118518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 973, 11), _ni_support_118517, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 973)
    _check_axis_call_result_118523 = invoke(stypy.reporting.localization.Localization(__file__, 973, 11), _check_axis_118518, *[axis_118519, ndim_118521], **kwargs_118522)
    
    # Assigning a type to the variable 'axis' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 4), 'axis', _check_axis_call_result_118523)
    
    
    # Getting the type of 'size' (line 974)
    size_118524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 7), 'size')
    int_118525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 14), 'int')
    # Applying the binary operator '<' (line 974)
    result_lt_118526 = python_operator(stypy.reporting.localization.Localization(__file__, 974, 7), '<', size_118524, int_118525)
    
    # Testing the type of an if condition (line 974)
    if_condition_118527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 974, 4), result_lt_118526)
    # Assigning a type to the variable 'if_condition_118527' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'if_condition_118527', if_condition_118527)
    # SSA begins for if statement (line 974)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 975)
    # Processing the call arguments (line 975)
    str_118529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 27), 'str', 'incorrect filter size')
    # Processing the call keyword arguments (line 975)
    kwargs_118530 = {}
    # Getting the type of 'RuntimeError' (line 975)
    RuntimeError_118528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 975)
    RuntimeError_call_result_118531 = invoke(stypy.reporting.localization.Localization(__file__, 975, 14), RuntimeError_118528, *[str_118529], **kwargs_118530)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 975, 8), RuntimeError_call_result_118531, 'raise parameter', BaseException)
    # SSA join for if statement (line 974)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 976):
    
    # Assigning a Subscript to a Name (line 976):
    
    # Obtaining the type of the subscript
    int_118532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 4), 'int')
    
    # Call to _get_output(...): (line 976)
    # Processing the call arguments (line 976)
    # Getting the type of 'output' (line 976)
    output_118535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 51), 'output', False)
    # Getting the type of 'input' (line 976)
    input_118536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 59), 'input', False)
    # Processing the call keyword arguments (line 976)
    kwargs_118537 = {}
    # Getting the type of '_ni_support' (line 976)
    _ni_support_118533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 976)
    _get_output_118534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 27), _ni_support_118533, '_get_output')
    # Calling _get_output(args, kwargs) (line 976)
    _get_output_call_result_118538 = invoke(stypy.reporting.localization.Localization(__file__, 976, 27), _get_output_118534, *[output_118535, input_118536], **kwargs_118537)
    
    # Obtaining the member '__getitem__' of a type (line 976)
    getitem___118539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 4), _get_output_call_result_118538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 976)
    subscript_call_result_118540 = invoke(stypy.reporting.localization.Localization(__file__, 976, 4), getitem___118539, int_118532)
    
    # Assigning a type to the variable 'tuple_var_assignment_117041' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'tuple_var_assignment_117041', subscript_call_result_118540)
    
    # Assigning a Subscript to a Name (line 976):
    
    # Obtaining the type of the subscript
    int_118541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 4), 'int')
    
    # Call to _get_output(...): (line 976)
    # Processing the call arguments (line 976)
    # Getting the type of 'output' (line 976)
    output_118544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 51), 'output', False)
    # Getting the type of 'input' (line 976)
    input_118545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 59), 'input', False)
    # Processing the call keyword arguments (line 976)
    kwargs_118546 = {}
    # Getting the type of '_ni_support' (line 976)
    _ni_support_118542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 976)
    _get_output_118543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 27), _ni_support_118542, '_get_output')
    # Calling _get_output(args, kwargs) (line 976)
    _get_output_call_result_118547 = invoke(stypy.reporting.localization.Localization(__file__, 976, 27), _get_output_118543, *[output_118544, input_118545], **kwargs_118546)
    
    # Obtaining the member '__getitem__' of a type (line 976)
    getitem___118548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 976, 4), _get_output_call_result_118547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 976)
    subscript_call_result_118549 = invoke(stypy.reporting.localization.Localization(__file__, 976, 4), getitem___118548, int_118541)
    
    # Assigning a type to the variable 'tuple_var_assignment_117042' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'tuple_var_assignment_117042', subscript_call_result_118549)
    
    # Assigning a Name to a Name (line 976):
    # Getting the type of 'tuple_var_assignment_117041' (line 976)
    tuple_var_assignment_117041_118550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'tuple_var_assignment_117041')
    # Assigning a type to the variable 'output' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'output', tuple_var_assignment_117041_118550)
    
    # Assigning a Name to a Name (line 976):
    # Getting the type of 'tuple_var_assignment_117042' (line 976)
    tuple_var_assignment_117042_118551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'tuple_var_assignment_117042')
    # Assigning a type to the variable 'return_value' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 12), 'return_value', tuple_var_assignment_117042_118551)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 977)
    size_118552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 8), 'size')
    int_118553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 16), 'int')
    # Applying the binary operator '//' (line 977)
    result_floordiv_118554 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 8), '//', size_118552, int_118553)
    
    # Getting the type of 'origin' (line 977)
    origin_118555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 20), 'origin')
    # Applying the binary operator '+' (line 977)
    result_add_118556 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 8), '+', result_floordiv_118554, origin_118555)
    
    int_118557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 29), 'int')
    # Applying the binary operator '<' (line 977)
    result_lt_118558 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 8), '<', result_add_118556, int_118557)
    
    
    # Getting the type of 'size' (line 977)
    size_118559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 36), 'size')
    int_118560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 44), 'int')
    # Applying the binary operator '//' (line 977)
    result_floordiv_118561 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 36), '//', size_118559, int_118560)
    
    # Getting the type of 'origin' (line 977)
    origin_118562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 48), 'origin')
    # Applying the binary operator '+' (line 977)
    result_add_118563 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 36), '+', result_floordiv_118561, origin_118562)
    
    # Getting the type of 'size' (line 977)
    size_118564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 58), 'size')
    # Applying the binary operator '>=' (line 977)
    result_ge_118565 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 36), '>=', result_add_118563, size_118564)
    
    # Applying the binary operator 'or' (line 977)
    result_or_keyword_118566 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 7), 'or', result_lt_118558, result_ge_118565)
    
    # Testing the type of an if condition (line 977)
    if_condition_118567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 977, 4), result_or_keyword_118566)
    # Assigning a type to the variable 'if_condition_118567' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 4), 'if_condition_118567', if_condition_118567)
    # SSA begins for if statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 978)
    # Processing the call arguments (line 978)
    str_118569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 25), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 978)
    kwargs_118570 = {}
    # Getting the type of 'ValueError' (line 978)
    ValueError_118568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 978)
    ValueError_call_result_118571 = invoke(stypy.reporting.localization.Localization(__file__, 978, 14), ValueError_118568, *[str_118569], **kwargs_118570)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 978, 8), ValueError_call_result_118571, 'raise parameter', BaseException)
    # SSA join for if statement (line 977)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 979):
    
    # Assigning a Call to a Name (line 979):
    
    # Call to _extend_mode_to_code(...): (line 979)
    # Processing the call arguments (line 979)
    # Getting the type of 'mode' (line 979)
    mode_118574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 44), 'mode', False)
    # Processing the call keyword arguments (line 979)
    kwargs_118575 = {}
    # Getting the type of '_ni_support' (line 979)
    _ni_support_118572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 979)
    _extend_mode_to_code_118573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 979, 11), _ni_support_118572, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 979)
    _extend_mode_to_code_call_result_118576 = invoke(stypy.reporting.localization.Localization(__file__, 979, 11), _extend_mode_to_code_118573, *[mode_118574], **kwargs_118575)
    
    # Assigning a type to the variable 'mode' (line 979)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 4), 'mode', _extend_mode_to_code_call_result_118576)
    
    # Call to min_or_max_filter1d(...): (line 980)
    # Processing the call arguments (line 980)
    # Getting the type of 'input' (line 980)
    input_118579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 34), 'input', False)
    # Getting the type of 'size' (line 980)
    size_118580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 41), 'size', False)
    # Getting the type of 'axis' (line 980)
    axis_118581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 47), 'axis', False)
    # Getting the type of 'output' (line 980)
    output_118582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 53), 'output', False)
    # Getting the type of 'mode' (line 980)
    mode_118583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 61), 'mode', False)
    # Getting the type of 'cval' (line 980)
    cval_118584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 67), 'cval', False)
    # Getting the type of 'origin' (line 981)
    origin_118585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 34), 'origin', False)
    int_118586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 42), 'int')
    # Processing the call keyword arguments (line 980)
    kwargs_118587 = {}
    # Getting the type of '_nd_image' (line 980)
    _nd_image_118577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), '_nd_image', False)
    # Obtaining the member 'min_or_max_filter1d' of a type (line 980)
    min_or_max_filter1d_118578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 4), _nd_image_118577, 'min_or_max_filter1d')
    # Calling min_or_max_filter1d(args, kwargs) (line 980)
    min_or_max_filter1d_call_result_118588 = invoke(stypy.reporting.localization.Localization(__file__, 980, 4), min_or_max_filter1d_118578, *[input_118579, size_118580, axis_118581, output_118582, mode_118583, cval_118584, origin_118585, int_118586], **kwargs_118587)
    
    # Getting the type of 'return_value' (line 982)
    return_value_118589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 982)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 982, 4), 'stypy_return_type', return_value_118589)
    
    # ################# End of 'minimum_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimum_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 933)
    stypy_return_type_118590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimum_filter1d'
    return stypy_return_type_118590

# Assigning a type to the variable 'minimum_filter1d' (line 933)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 0), 'minimum_filter1d', minimum_filter1d)

@norecursion
def maximum_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_118591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 39), 'int')
    # Getting the type of 'None' (line 986)
    None_118592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 50), 'None')
    str_118593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 26), 'str', 'reflect')
    float_118594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 42), 'float')
    int_118595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 987, 54), 'int')
    defaults = [int_118591, None_118592, str_118593, float_118594, int_118595]
    # Create a new context for function 'maximum_filter1d'
    module_type_store = module_type_store.open_function_context('maximum_filter1d', 985, 0, False)
    
    # Passed parameters checking function
    maximum_filter1d.stypy_localization = localization
    maximum_filter1d.stypy_type_of_self = None
    maximum_filter1d.stypy_type_store = module_type_store
    maximum_filter1d.stypy_function_name = 'maximum_filter1d'
    maximum_filter1d.stypy_param_names_list = ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin']
    maximum_filter1d.stypy_varargs_param_name = None
    maximum_filter1d.stypy_kwargs_param_name = None
    maximum_filter1d.stypy_call_defaults = defaults
    maximum_filter1d.stypy_call_varargs = varargs
    maximum_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'maximum_filter1d', ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'maximum_filter1d', localization, ['input', 'size', 'axis', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'maximum_filter1d(...)' code ##################

    str_118596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, (-1)), 'str', 'Calculate a one-dimensional maximum filter along the given axis.\n\n    The lines of the array along the given axis are filtered with a\n    maximum filter of given size.\n\n    Parameters\n    ----------\n    %(input)s\n    size : int\n        Length along which to calculate the 1-D maximum.\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    maximum1d : ndarray, None\n        Maximum-filtered array with same shape as input.\n        None if `output` is not None\n\n    Notes\n    -----\n    This function implements the MAXLIST algorithm [1]_, as described by\n    Richard Harter [2]_, and has a guaranteed O(n) performance, `n` being\n    the `input` length, regardless of filter size.\n\n    References\n    ----------\n    .. [1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.2777\n    .. [2] http://www.richardhartersworld.com/cri/2001/slidingmin.html\n\n    Examples\n    --------\n    >>> from scipy.ndimage import maximum_filter1d\n    >>> maximum_filter1d([2, 8, 0, 4, 1, 9, 9, 0], size=3)\n    array([8, 8, 8, 4, 9, 9, 9, 9])\n    ')
    
    # Assigning a Call to a Name (line 1027):
    
    # Assigning a Call to a Name (line 1027):
    
    # Call to asarray(...): (line 1027)
    # Processing the call arguments (line 1027)
    # Getting the type of 'input' (line 1027)
    input_118599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 26), 'input', False)
    # Processing the call keyword arguments (line 1027)
    kwargs_118600 = {}
    # Getting the type of 'numpy' (line 1027)
    numpy_118597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1027)
    asarray_118598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 12), numpy_118597, 'asarray')
    # Calling asarray(args, kwargs) (line 1027)
    asarray_call_result_118601 = invoke(stypy.reporting.localization.Localization(__file__, 1027, 12), asarray_118598, *[input_118599], **kwargs_118600)
    
    # Assigning a type to the variable 'input' (line 1027)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 4), 'input', asarray_call_result_118601)
    
    
    # Call to iscomplexobj(...): (line 1028)
    # Processing the call arguments (line 1028)
    # Getting the type of 'input' (line 1028)
    input_118604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 26), 'input', False)
    # Processing the call keyword arguments (line 1028)
    kwargs_118605 = {}
    # Getting the type of 'numpy' (line 1028)
    numpy_118602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1028)
    iscomplexobj_118603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 7), numpy_118602, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1028)
    iscomplexobj_call_result_118606 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 7), iscomplexobj_118603, *[input_118604], **kwargs_118605)
    
    # Testing the type of an if condition (line 1028)
    if_condition_118607 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1028, 4), iscomplexobj_call_result_118606)
    # Assigning a type to the variable 'if_condition_118607' (line 1028)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1028, 4), 'if_condition_118607', if_condition_118607)
    # SSA begins for if statement (line 1028)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1029)
    # Processing the call arguments (line 1029)
    str_118609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 1029)
    kwargs_118610 = {}
    # Getting the type of 'TypeError' (line 1029)
    TypeError_118608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1029)
    TypeError_call_result_118611 = invoke(stypy.reporting.localization.Localization(__file__, 1029, 14), TypeError_118608, *[str_118609], **kwargs_118610)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1029, 8), TypeError_call_result_118611, 'raise parameter', BaseException)
    # SSA join for if statement (line 1028)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1030):
    
    # Assigning a Call to a Name (line 1030):
    
    # Call to _check_axis(...): (line 1030)
    # Processing the call arguments (line 1030)
    # Getting the type of 'axis' (line 1030)
    axis_118614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 35), 'axis', False)
    # Getting the type of 'input' (line 1030)
    input_118615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1030)
    ndim_118616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 41), input_118615, 'ndim')
    # Processing the call keyword arguments (line 1030)
    kwargs_118617 = {}
    # Getting the type of '_ni_support' (line 1030)
    _ni_support_118612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 1030)
    _check_axis_118613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 11), _ni_support_118612, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 1030)
    _check_axis_call_result_118618 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 11), _check_axis_118613, *[axis_118614, ndim_118616], **kwargs_118617)
    
    # Assigning a type to the variable 'axis' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 4), 'axis', _check_axis_call_result_118618)
    
    
    # Getting the type of 'size' (line 1031)
    size_118619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 7), 'size')
    int_118620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1031, 14), 'int')
    # Applying the binary operator '<' (line 1031)
    result_lt_118621 = python_operator(stypy.reporting.localization.Localization(__file__, 1031, 7), '<', size_118619, int_118620)
    
    # Testing the type of an if condition (line 1031)
    if_condition_118622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1031, 4), result_lt_118621)
    # Assigning a type to the variable 'if_condition_118622' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'if_condition_118622', if_condition_118622)
    # SSA begins for if statement (line 1031)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1032)
    # Processing the call arguments (line 1032)
    str_118624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 27), 'str', 'incorrect filter size')
    # Processing the call keyword arguments (line 1032)
    kwargs_118625 = {}
    # Getting the type of 'RuntimeError' (line 1032)
    RuntimeError_118623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1032)
    RuntimeError_call_result_118626 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 14), RuntimeError_118623, *[str_118624], **kwargs_118625)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1032, 8), RuntimeError_call_result_118626, 'raise parameter', BaseException)
    # SSA join for if statement (line 1031)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1033):
    
    # Assigning a Subscript to a Name (line 1033):
    
    # Obtaining the type of the subscript
    int_118627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 4), 'int')
    
    # Call to _get_output(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'output' (line 1033)
    output_118630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 51), 'output', False)
    # Getting the type of 'input' (line 1033)
    input_118631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 59), 'input', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_118632 = {}
    # Getting the type of '_ni_support' (line 1033)
    _ni_support_118628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1033)
    _get_output_118629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 27), _ni_support_118628, '_get_output')
    # Calling _get_output(args, kwargs) (line 1033)
    _get_output_call_result_118633 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 27), _get_output_118629, *[output_118630, input_118631], **kwargs_118632)
    
    # Obtaining the member '__getitem__' of a type (line 1033)
    getitem___118634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 4), _get_output_call_result_118633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
    subscript_call_result_118635 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 4), getitem___118634, int_118627)
    
    # Assigning a type to the variable 'tuple_var_assignment_117043' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'tuple_var_assignment_117043', subscript_call_result_118635)
    
    # Assigning a Subscript to a Name (line 1033):
    
    # Obtaining the type of the subscript
    int_118636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 4), 'int')
    
    # Call to _get_output(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'output' (line 1033)
    output_118639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 51), 'output', False)
    # Getting the type of 'input' (line 1033)
    input_118640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 59), 'input', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_118641 = {}
    # Getting the type of '_ni_support' (line 1033)
    _ni_support_118637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1033)
    _get_output_118638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 27), _ni_support_118637, '_get_output')
    # Calling _get_output(args, kwargs) (line 1033)
    _get_output_call_result_118642 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 27), _get_output_118638, *[output_118639, input_118640], **kwargs_118641)
    
    # Obtaining the member '__getitem__' of a type (line 1033)
    getitem___118643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 4), _get_output_call_result_118642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1033)
    subscript_call_result_118644 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 4), getitem___118643, int_118636)
    
    # Assigning a type to the variable 'tuple_var_assignment_117044' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'tuple_var_assignment_117044', subscript_call_result_118644)
    
    # Assigning a Name to a Name (line 1033):
    # Getting the type of 'tuple_var_assignment_117043' (line 1033)
    tuple_var_assignment_117043_118645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'tuple_var_assignment_117043')
    # Assigning a type to the variable 'output' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'output', tuple_var_assignment_117043_118645)
    
    # Assigning a Name to a Name (line 1033):
    # Getting the type of 'tuple_var_assignment_117044' (line 1033)
    tuple_var_assignment_117044_118646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'tuple_var_assignment_117044')
    # Assigning a type to the variable 'return_value' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 12), 'return_value', tuple_var_assignment_117044_118646)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'size' (line 1034)
    size_118647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'size')
    int_118648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 16), 'int')
    # Applying the binary operator '//' (line 1034)
    result_floordiv_118649 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 8), '//', size_118647, int_118648)
    
    # Getting the type of 'origin' (line 1034)
    origin_118650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 20), 'origin')
    # Applying the binary operator '+' (line 1034)
    result_add_118651 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 8), '+', result_floordiv_118649, origin_118650)
    
    int_118652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 29), 'int')
    # Applying the binary operator '<' (line 1034)
    result_lt_118653 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 8), '<', result_add_118651, int_118652)
    
    
    # Getting the type of 'size' (line 1034)
    size_118654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 36), 'size')
    int_118655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 44), 'int')
    # Applying the binary operator '//' (line 1034)
    result_floordiv_118656 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 36), '//', size_118654, int_118655)
    
    # Getting the type of 'origin' (line 1034)
    origin_118657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 48), 'origin')
    # Applying the binary operator '+' (line 1034)
    result_add_118658 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 36), '+', result_floordiv_118656, origin_118657)
    
    # Getting the type of 'size' (line 1034)
    size_118659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 58), 'size')
    # Applying the binary operator '>=' (line 1034)
    result_ge_118660 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 36), '>=', result_add_118658, size_118659)
    
    # Applying the binary operator 'or' (line 1034)
    result_or_keyword_118661 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 7), 'or', result_lt_118653, result_ge_118660)
    
    # Testing the type of an if condition (line 1034)
    if_condition_118662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1034, 4), result_or_keyword_118661)
    # Assigning a type to the variable 'if_condition_118662' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'if_condition_118662', if_condition_118662)
    # SSA begins for if statement (line 1034)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1035)
    # Processing the call arguments (line 1035)
    str_118664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 25), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 1035)
    kwargs_118665 = {}
    # Getting the type of 'ValueError' (line 1035)
    ValueError_118663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1035)
    ValueError_call_result_118666 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 14), ValueError_118663, *[str_118664], **kwargs_118665)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1035, 8), ValueError_call_result_118666, 'raise parameter', BaseException)
    # SSA join for if statement (line 1034)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1036):
    
    # Assigning a Call to a Name (line 1036):
    
    # Call to _extend_mode_to_code(...): (line 1036)
    # Processing the call arguments (line 1036)
    # Getting the type of 'mode' (line 1036)
    mode_118669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 44), 'mode', False)
    # Processing the call keyword arguments (line 1036)
    kwargs_118670 = {}
    # Getting the type of '_ni_support' (line 1036)
    _ni_support_118667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 1036)
    _extend_mode_to_code_118668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 11), _ni_support_118667, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 1036)
    _extend_mode_to_code_call_result_118671 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 11), _extend_mode_to_code_118668, *[mode_118669], **kwargs_118670)
    
    # Assigning a type to the variable 'mode' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'mode', _extend_mode_to_code_call_result_118671)
    
    # Call to min_or_max_filter1d(...): (line 1037)
    # Processing the call arguments (line 1037)
    # Getting the type of 'input' (line 1037)
    input_118674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 34), 'input', False)
    # Getting the type of 'size' (line 1037)
    size_118675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 41), 'size', False)
    # Getting the type of 'axis' (line 1037)
    axis_118676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 47), 'axis', False)
    # Getting the type of 'output' (line 1037)
    output_118677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 53), 'output', False)
    # Getting the type of 'mode' (line 1037)
    mode_118678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 61), 'mode', False)
    # Getting the type of 'cval' (line 1037)
    cval_118679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 67), 'cval', False)
    # Getting the type of 'origin' (line 1038)
    origin_118680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 34), 'origin', False)
    int_118681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 42), 'int')
    # Processing the call keyword arguments (line 1037)
    kwargs_118682 = {}
    # Getting the type of '_nd_image' (line 1037)
    _nd_image_118672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), '_nd_image', False)
    # Obtaining the member 'min_or_max_filter1d' of a type (line 1037)
    min_or_max_filter1d_118673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 4), _nd_image_118672, 'min_or_max_filter1d')
    # Calling min_or_max_filter1d(args, kwargs) (line 1037)
    min_or_max_filter1d_call_result_118683 = invoke(stypy.reporting.localization.Localization(__file__, 1037, 4), min_or_max_filter1d_118673, *[input_118674, size_118675, axis_118676, output_118677, mode_118678, cval_118679, origin_118680, int_118681], **kwargs_118682)
    
    # Getting the type of 'return_value' (line 1039)
    return_value_118684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'stypy_return_type', return_value_118684)
    
    # ################# End of 'maximum_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'maximum_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 985)
    stypy_return_type_118685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118685)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'maximum_filter1d'
    return stypy_return_type_118685

# Assigning a type to the variable 'maximum_filter1d' (line 985)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 0), 'maximum_filter1d', maximum_filter1d)

@norecursion
def _min_or_max_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_min_or_max_filter'
    module_type_store = module_type_store.open_function_context('_min_or_max_filter', 1042, 0, False)
    
    # Passed parameters checking function
    _min_or_max_filter.stypy_localization = localization
    _min_or_max_filter.stypy_type_of_self = None
    _min_or_max_filter.stypy_type_store = module_type_store
    _min_or_max_filter.stypy_function_name = '_min_or_max_filter'
    _min_or_max_filter.stypy_param_names_list = ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin', 'minimum']
    _min_or_max_filter.stypy_varargs_param_name = None
    _min_or_max_filter.stypy_kwargs_param_name = None
    _min_or_max_filter.stypy_call_defaults = defaults
    _min_or_max_filter.stypy_call_varargs = varargs
    _min_or_max_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_min_or_max_filter', ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin', 'minimum'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_min_or_max_filter', localization, ['input', 'size', 'footprint', 'structure', 'output', 'mode', 'cval', 'origin', 'minimum'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_min_or_max_filter(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 1044)
    # Getting the type of 'structure' (line 1044)
    structure_118686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 7), 'structure')
    # Getting the type of 'None' (line 1044)
    None_118687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 20), 'None')
    
    (may_be_118688, more_types_in_union_118689) = may_be_none(structure_118686, None_118687)

    if may_be_118688:

        if more_types_in_union_118689:
            # Runtime conditional SSA (line 1044)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 1045)
        # Getting the type of 'footprint' (line 1045)
        footprint_118690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 11), 'footprint')
        # Getting the type of 'None' (line 1045)
        None_118691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 24), 'None')
        
        (may_be_118692, more_types_in_union_118693) = may_be_none(footprint_118690, None_118691)

        if may_be_118692:

            if more_types_in_union_118693:
                # Runtime conditional SSA (line 1045)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 1046)
            # Getting the type of 'size' (line 1046)
            size_118694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 15), 'size')
            # Getting the type of 'None' (line 1046)
            None_118695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 23), 'None')
            
            (may_be_118696, more_types_in_union_118697) = may_be_none(size_118694, None_118695)

            if may_be_118696:

                if more_types_in_union_118697:
                    # Runtime conditional SSA (line 1046)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to RuntimeError(...): (line 1047)
                # Processing the call arguments (line 1047)
                str_118699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 35), 'str', 'no footprint provided')
                # Processing the call keyword arguments (line 1047)
                kwargs_118700 = {}
                # Getting the type of 'RuntimeError' (line 1047)
                RuntimeError_118698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 22), 'RuntimeError', False)
                # Calling RuntimeError(args, kwargs) (line 1047)
                RuntimeError_call_result_118701 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 22), RuntimeError_118698, *[str_118699], **kwargs_118700)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1047, 16), RuntimeError_call_result_118701, 'raise parameter', BaseException)

                if more_types_in_union_118697:
                    # SSA join for if statement (line 1046)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Name to a Name (line 1048):
            
            # Assigning a Name to a Name (line 1048):
            # Getting the type of 'True' (line 1048)
            True_118702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 24), 'True')
            # Assigning a type to the variable 'separable' (line 1048)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1048, 12), 'separable', True_118702)

            if more_types_in_union_118693:
                # Runtime conditional SSA for else branch (line 1045)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_118692) or more_types_in_union_118693):
            
            # Assigning a Call to a Name (line 1050):
            
            # Assigning a Call to a Name (line 1050):
            
            # Call to asarray(...): (line 1050)
            # Processing the call arguments (line 1050)
            # Getting the type of 'footprint' (line 1050)
            footprint_118705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 38), 'footprint', False)
            # Processing the call keyword arguments (line 1050)
            # Getting the type of 'bool' (line 1050)
            bool_118706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 55), 'bool', False)
            keyword_118707 = bool_118706
            kwargs_118708 = {'dtype': keyword_118707}
            # Getting the type of 'numpy' (line 1050)
            numpy_118703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 24), 'numpy', False)
            # Obtaining the member 'asarray' of a type (line 1050)
            asarray_118704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 24), numpy_118703, 'asarray')
            # Calling asarray(args, kwargs) (line 1050)
            asarray_call_result_118709 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 24), asarray_118704, *[footprint_118705], **kwargs_118708)
            
            # Assigning a type to the variable 'footprint' (line 1050)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 12), 'footprint', asarray_call_result_118709)
            
            
            
            # Call to any(...): (line 1051)
            # Processing the call keyword arguments (line 1051)
            kwargs_118712 = {}
            # Getting the type of 'footprint' (line 1051)
            footprint_118710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 19), 'footprint', False)
            # Obtaining the member 'any' of a type (line 1051)
            any_118711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1051, 19), footprint_118710, 'any')
            # Calling any(args, kwargs) (line 1051)
            any_call_result_118713 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 19), any_118711, *[], **kwargs_118712)
            
            # Applying the 'not' unary operator (line 1051)
            result_not__118714 = python_operator(stypy.reporting.localization.Localization(__file__, 1051, 15), 'not', any_call_result_118713)
            
            # Testing the type of an if condition (line 1051)
            if_condition_118715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1051, 12), result_not__118714)
            # Assigning a type to the variable 'if_condition_118715' (line 1051)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 12), 'if_condition_118715', if_condition_118715)
            # SSA begins for if statement (line 1051)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 1052)
            # Processing the call arguments (line 1052)
            str_118717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 33), 'str', 'All-zero footprint is not supported.')
            # Processing the call keyword arguments (line 1052)
            kwargs_118718 = {}
            # Getting the type of 'ValueError' (line 1052)
            ValueError_118716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 1052)
            ValueError_call_result_118719 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 22), ValueError_118716, *[str_118717], **kwargs_118718)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1052, 16), ValueError_call_result_118719, 'raise parameter', BaseException)
            # SSA join for if statement (line 1051)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Call to all(...): (line 1053)
            # Processing the call keyword arguments (line 1053)
            kwargs_118722 = {}
            # Getting the type of 'footprint' (line 1053)
            footprint_118720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 15), 'footprint', False)
            # Obtaining the member 'all' of a type (line 1053)
            all_118721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1053, 15), footprint_118720, 'all')
            # Calling all(args, kwargs) (line 1053)
            all_call_result_118723 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 15), all_118721, *[], **kwargs_118722)
            
            # Testing the type of an if condition (line 1053)
            if_condition_118724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1053, 12), all_call_result_118723)
            # Assigning a type to the variable 'if_condition_118724' (line 1053)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 12), 'if_condition_118724', if_condition_118724)
            # SSA begins for if statement (line 1053)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 1054):
            
            # Assigning a Attribute to a Name (line 1054):
            # Getting the type of 'footprint' (line 1054)
            footprint_118725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 23), 'footprint')
            # Obtaining the member 'shape' of a type (line 1054)
            shape_118726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 23), footprint_118725, 'shape')
            # Assigning a type to the variable 'size' (line 1054)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 16), 'size', shape_118726)
            
            # Assigning a Name to a Name (line 1055):
            
            # Assigning a Name to a Name (line 1055):
            # Getting the type of 'None' (line 1055)
            None_118727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 28), 'None')
            # Assigning a type to the variable 'footprint' (line 1055)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 16), 'footprint', None_118727)
            
            # Assigning a Name to a Name (line 1056):
            
            # Assigning a Name to a Name (line 1056):
            # Getting the type of 'True' (line 1056)
            True_118728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 28), 'True')
            # Assigning a type to the variable 'separable' (line 1056)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 16), 'separable', True_118728)
            # SSA branch for the else part of an if statement (line 1053)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 1058):
            
            # Assigning a Name to a Name (line 1058):
            # Getting the type of 'False' (line 1058)
            False_118729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 28), 'False')
            # Assigning a type to the variable 'separable' (line 1058)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 16), 'separable', False_118729)
            # SSA join for if statement (line 1053)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_118692 and more_types_in_union_118693):
                # SSA join for if statement (line 1045)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_118689:
            # Runtime conditional SSA for else branch (line 1044)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_118688) or more_types_in_union_118689):
        
        # Assigning a Call to a Name (line 1060):
        
        # Assigning a Call to a Name (line 1060):
        
        # Call to asarray(...): (line 1060)
        # Processing the call arguments (line 1060)
        # Getting the type of 'structure' (line 1060)
        structure_118732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 34), 'structure', False)
        # Processing the call keyword arguments (line 1060)
        # Getting the type of 'numpy' (line 1060)
        numpy_118733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 51), 'numpy', False)
        # Obtaining the member 'float64' of a type (line 1060)
        float64_118734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 51), numpy_118733, 'float64')
        keyword_118735 = float64_118734
        kwargs_118736 = {'dtype': keyword_118735}
        # Getting the type of 'numpy' (line 1060)
        numpy_118730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1060)
        asarray_118731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1060, 20), numpy_118730, 'asarray')
        # Calling asarray(args, kwargs) (line 1060)
        asarray_call_result_118737 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 20), asarray_118731, *[structure_118732], **kwargs_118736)
        
        # Assigning a type to the variable 'structure' (line 1060)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1060, 8), 'structure', asarray_call_result_118737)
        
        # Assigning a Name to a Name (line 1061):
        
        # Assigning a Name to a Name (line 1061):
        # Getting the type of 'False' (line 1061)
        False_118738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 20), 'False')
        # Assigning a type to the variable 'separable' (line 1061)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1061, 8), 'separable', False_118738)
        
        # Type idiom detected: calculating its left and rigth part (line 1062)
        # Getting the type of 'footprint' (line 1062)
        footprint_118739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 11), 'footprint')
        # Getting the type of 'None' (line 1062)
        None_118740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 24), 'None')
        
        (may_be_118741, more_types_in_union_118742) = may_be_none(footprint_118739, None_118740)

        if may_be_118741:

            if more_types_in_union_118742:
                # Runtime conditional SSA (line 1062)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 1063):
            
            # Assigning a Call to a Name (line 1063):
            
            # Call to ones(...): (line 1063)
            # Processing the call arguments (line 1063)
            # Getting the type of 'structure' (line 1063)
            structure_118745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 35), 'structure', False)
            # Obtaining the member 'shape' of a type (line 1063)
            shape_118746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1063, 35), structure_118745, 'shape')
            # Getting the type of 'bool' (line 1063)
            bool_118747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 52), 'bool', False)
            # Processing the call keyword arguments (line 1063)
            kwargs_118748 = {}
            # Getting the type of 'numpy' (line 1063)
            numpy_118743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 24), 'numpy', False)
            # Obtaining the member 'ones' of a type (line 1063)
            ones_118744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1063, 24), numpy_118743, 'ones')
            # Calling ones(args, kwargs) (line 1063)
            ones_call_result_118749 = invoke(stypy.reporting.localization.Localization(__file__, 1063, 24), ones_118744, *[shape_118746, bool_118747], **kwargs_118748)
            
            # Assigning a type to the variable 'footprint' (line 1063)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'footprint', ones_call_result_118749)

            if more_types_in_union_118742:
                # Runtime conditional SSA for else branch (line 1062)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_118741) or more_types_in_union_118742):
            
            # Assigning a Call to a Name (line 1065):
            
            # Assigning a Call to a Name (line 1065):
            
            # Call to asarray(...): (line 1065)
            # Processing the call arguments (line 1065)
            # Getting the type of 'footprint' (line 1065)
            footprint_118752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 38), 'footprint', False)
            # Processing the call keyword arguments (line 1065)
            # Getting the type of 'bool' (line 1065)
            bool_118753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 55), 'bool', False)
            keyword_118754 = bool_118753
            kwargs_118755 = {'dtype': keyword_118754}
            # Getting the type of 'numpy' (line 1065)
            numpy_118750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 24), 'numpy', False)
            # Obtaining the member 'asarray' of a type (line 1065)
            asarray_118751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1065, 24), numpy_118750, 'asarray')
            # Calling asarray(args, kwargs) (line 1065)
            asarray_call_result_118756 = invoke(stypy.reporting.localization.Localization(__file__, 1065, 24), asarray_118751, *[footprint_118752], **kwargs_118755)
            
            # Assigning a type to the variable 'footprint' (line 1065)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 12), 'footprint', asarray_call_result_118756)

            if (may_be_118741 and more_types_in_union_118742):
                # SSA join for if statement (line 1062)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_118688 and more_types_in_union_118689):
            # SSA join for if statement (line 1044)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1066):
    
    # Assigning a Call to a Name (line 1066):
    
    # Call to asarray(...): (line 1066)
    # Processing the call arguments (line 1066)
    # Getting the type of 'input' (line 1066)
    input_118759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 26), 'input', False)
    # Processing the call keyword arguments (line 1066)
    kwargs_118760 = {}
    # Getting the type of 'numpy' (line 1066)
    numpy_118757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1066)
    asarray_118758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 12), numpy_118757, 'asarray')
    # Calling asarray(args, kwargs) (line 1066)
    asarray_call_result_118761 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 12), asarray_118758, *[input_118759], **kwargs_118760)
    
    # Assigning a type to the variable 'input' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'input', asarray_call_result_118761)
    
    
    # Call to iscomplexobj(...): (line 1067)
    # Processing the call arguments (line 1067)
    # Getting the type of 'input' (line 1067)
    input_118764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 26), 'input', False)
    # Processing the call keyword arguments (line 1067)
    kwargs_118765 = {}
    # Getting the type of 'numpy' (line 1067)
    numpy_118762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1067)
    iscomplexobj_118763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1067, 7), numpy_118762, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1067)
    iscomplexobj_call_result_118766 = invoke(stypy.reporting.localization.Localization(__file__, 1067, 7), iscomplexobj_118763, *[input_118764], **kwargs_118765)
    
    # Testing the type of an if condition (line 1067)
    if_condition_118767 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1067, 4), iscomplexobj_call_result_118766)
    # Assigning a type to the variable 'if_condition_118767' (line 1067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 4), 'if_condition_118767', if_condition_118767)
    # SSA begins for if statement (line 1067)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1068)
    # Processing the call arguments (line 1068)
    str_118769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1068, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 1068)
    kwargs_118770 = {}
    # Getting the type of 'TypeError' (line 1068)
    TypeError_118768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1068)
    TypeError_call_result_118771 = invoke(stypy.reporting.localization.Localization(__file__, 1068, 14), TypeError_118768, *[str_118769], **kwargs_118770)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1068, 8), TypeError_call_result_118771, 'raise parameter', BaseException)
    # SSA join for if statement (line 1067)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1069):
    
    # Assigning a Subscript to a Name (line 1069):
    
    # Obtaining the type of the subscript
    int_118772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 4), 'int')
    
    # Call to _get_output(...): (line 1069)
    # Processing the call arguments (line 1069)
    # Getting the type of 'output' (line 1069)
    output_118775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 51), 'output', False)
    # Getting the type of 'input' (line 1069)
    input_118776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 59), 'input', False)
    # Processing the call keyword arguments (line 1069)
    kwargs_118777 = {}
    # Getting the type of '_ni_support' (line 1069)
    _ni_support_118773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1069)
    _get_output_118774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 27), _ni_support_118773, '_get_output')
    # Calling _get_output(args, kwargs) (line 1069)
    _get_output_call_result_118778 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 27), _get_output_118774, *[output_118775, input_118776], **kwargs_118777)
    
    # Obtaining the member '__getitem__' of a type (line 1069)
    getitem___118779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 4), _get_output_call_result_118778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1069)
    subscript_call_result_118780 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 4), getitem___118779, int_118772)
    
    # Assigning a type to the variable 'tuple_var_assignment_117045' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'tuple_var_assignment_117045', subscript_call_result_118780)
    
    # Assigning a Subscript to a Name (line 1069):
    
    # Obtaining the type of the subscript
    int_118781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 4), 'int')
    
    # Call to _get_output(...): (line 1069)
    # Processing the call arguments (line 1069)
    # Getting the type of 'output' (line 1069)
    output_118784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 51), 'output', False)
    # Getting the type of 'input' (line 1069)
    input_118785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 59), 'input', False)
    # Processing the call keyword arguments (line 1069)
    kwargs_118786 = {}
    # Getting the type of '_ni_support' (line 1069)
    _ni_support_118782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1069)
    _get_output_118783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 27), _ni_support_118782, '_get_output')
    # Calling _get_output(args, kwargs) (line 1069)
    _get_output_call_result_118787 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 27), _get_output_118783, *[output_118784, input_118785], **kwargs_118786)
    
    # Obtaining the member '__getitem__' of a type (line 1069)
    getitem___118788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 4), _get_output_call_result_118787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1069)
    subscript_call_result_118789 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 4), getitem___118788, int_118781)
    
    # Assigning a type to the variable 'tuple_var_assignment_117046' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'tuple_var_assignment_117046', subscript_call_result_118789)
    
    # Assigning a Name to a Name (line 1069):
    # Getting the type of 'tuple_var_assignment_117045' (line 1069)
    tuple_var_assignment_117045_118790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'tuple_var_assignment_117045')
    # Assigning a type to the variable 'output' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'output', tuple_var_assignment_117045_118790)
    
    # Assigning a Name to a Name (line 1069):
    # Getting the type of 'tuple_var_assignment_117046' (line 1069)
    tuple_var_assignment_117046_118791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 4), 'tuple_var_assignment_117046')
    # Assigning a type to the variable 'return_value' (line 1069)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1069, 12), 'return_value', tuple_var_assignment_117046_118791)
    
    # Assigning a Call to a Name (line 1070):
    
    # Assigning a Call to a Name (line 1070):
    
    # Call to _normalize_sequence(...): (line 1070)
    # Processing the call arguments (line 1070)
    # Getting the type of 'origin' (line 1070)
    origin_118794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 46), 'origin', False)
    # Getting the type of 'input' (line 1070)
    input_118795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1070)
    ndim_118796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 54), input_118795, 'ndim')
    # Processing the call keyword arguments (line 1070)
    kwargs_118797 = {}
    # Getting the type of '_ni_support' (line 1070)
    _ni_support_118792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1070)
    _normalize_sequence_118793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1070, 14), _ni_support_118792, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1070)
    _normalize_sequence_call_result_118798 = invoke(stypy.reporting.localization.Localization(__file__, 1070, 14), _normalize_sequence_118793, *[origin_118794, ndim_118796], **kwargs_118797)
    
    # Assigning a type to the variable 'origins' (line 1070)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1070, 4), 'origins', _normalize_sequence_call_result_118798)
    
    # Getting the type of 'separable' (line 1071)
    separable_118799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 7), 'separable')
    # Testing the type of an if condition (line 1071)
    if_condition_118800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1071, 4), separable_118799)
    # Assigning a type to the variable 'if_condition_118800' (line 1071)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'if_condition_118800', if_condition_118800)
    # SSA begins for if statement (line 1071)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1072):
    
    # Assigning a Call to a Name (line 1072):
    
    # Call to _normalize_sequence(...): (line 1072)
    # Processing the call arguments (line 1072)
    # Getting the type of 'size' (line 1072)
    size_118803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 48), 'size', False)
    # Getting the type of 'input' (line 1072)
    input_118804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1072)
    ndim_118805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 54), input_118804, 'ndim')
    # Processing the call keyword arguments (line 1072)
    kwargs_118806 = {}
    # Getting the type of '_ni_support' (line 1072)
    _ni_support_118801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 16), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1072)
    _normalize_sequence_118802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 16), _ni_support_118801, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1072)
    _normalize_sequence_call_result_118807 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 16), _normalize_sequence_118802, *[size_118803, ndim_118805], **kwargs_118806)
    
    # Assigning a type to the variable 'sizes' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 8), 'sizes', _normalize_sequence_call_result_118807)
    
    # Assigning a Call to a Name (line 1073):
    
    # Assigning a Call to a Name (line 1073):
    
    # Call to _normalize_sequence(...): (line 1073)
    # Processing the call arguments (line 1073)
    # Getting the type of 'mode' (line 1073)
    mode_118810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 48), 'mode', False)
    # Getting the type of 'input' (line 1073)
    input_118811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1073)
    ndim_118812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 54), input_118811, 'ndim')
    # Processing the call keyword arguments (line 1073)
    kwargs_118813 = {}
    # Getting the type of '_ni_support' (line 1073)
    _ni_support_118808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 16), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1073)
    _normalize_sequence_118809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1073, 16), _ni_support_118808, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1073)
    _normalize_sequence_call_result_118814 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 16), _normalize_sequence_118809, *[mode_118810, ndim_118812], **kwargs_118813)
    
    # Assigning a type to the variable 'modes' (line 1073)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'modes', _normalize_sequence_call_result_118814)
    
    # Assigning a Call to a Name (line 1074):
    
    # Assigning a Call to a Name (line 1074):
    
    # Call to list(...): (line 1074)
    # Processing the call arguments (line 1074)
    
    # Call to range(...): (line 1074)
    # Processing the call arguments (line 1074)
    # Getting the type of 'input' (line 1074)
    input_118817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 26), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1074)
    ndim_118818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 26), input_118817, 'ndim')
    # Processing the call keyword arguments (line 1074)
    kwargs_118819 = {}
    # Getting the type of 'range' (line 1074)
    range_118816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 20), 'range', False)
    # Calling range(args, kwargs) (line 1074)
    range_call_result_118820 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 20), range_118816, *[ndim_118818], **kwargs_118819)
    
    # Processing the call keyword arguments (line 1074)
    kwargs_118821 = {}
    # Getting the type of 'list' (line 1074)
    list_118815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 15), 'list', False)
    # Calling list(args, kwargs) (line 1074)
    list_call_result_118822 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 15), list_118815, *[range_call_result_118820], **kwargs_118821)
    
    # Assigning a type to the variable 'axes' (line 1074)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1074, 8), 'axes', list_call_result_118822)
    
    # Assigning a ListComp to a Name (line 1075):
    
    # Assigning a ListComp to a Name (line 1075):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 1076)
    # Processing the call arguments (line 1076)
    
    # Call to len(...): (line 1076)
    # Processing the call arguments (line 1076)
    # Getting the type of 'axes' (line 1076)
    axes_118848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 36), 'axes', False)
    # Processing the call keyword arguments (line 1076)
    kwargs_118849 = {}
    # Getting the type of 'len' (line 1076)
    len_118847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 32), 'len', False)
    # Calling len(args, kwargs) (line 1076)
    len_call_result_118850 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 32), len_118847, *[axes_118848], **kwargs_118849)
    
    # Processing the call keyword arguments (line 1076)
    kwargs_118851 = {}
    # Getting the type of 'range' (line 1076)
    range_118846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 26), 'range', False)
    # Calling range(args, kwargs) (line 1076)
    range_call_result_118852 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 26), range_118846, *[len_call_result_118850], **kwargs_118851)
    
    comprehension_118853 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 16), range_call_result_118852)
    # Assigning a type to the variable 'ii' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 16), 'ii', comprehension_118853)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1076)
    ii_118840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 52), 'ii')
    # Getting the type of 'sizes' (line 1076)
    sizes_118841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 46), 'sizes')
    # Obtaining the member '__getitem__' of a type (line 1076)
    getitem___118842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 46), sizes_118841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1076)
    subscript_call_result_118843 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 46), getitem___118842, ii_118840)
    
    int_118844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 58), 'int')
    # Applying the binary operator '>' (line 1076)
    result_gt_118845 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 46), '>', subscript_call_result_118843, int_118844)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1075)
    tuple_118823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1075)
    # Adding element type (line 1075)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1075)
    ii_118824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 22), 'ii')
    # Getting the type of 'axes' (line 1075)
    axes_118825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 17), 'axes')
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___118826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 17), axes_118825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_118827 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 17), getitem___118826, ii_118824)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 17), tuple_118823, subscript_call_result_118827)
    # Adding element type (line 1075)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1075)
    ii_118828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 33), 'ii')
    # Getting the type of 'sizes' (line 1075)
    sizes_118829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 27), 'sizes')
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___118830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 27), sizes_118829, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_118831 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 27), getitem___118830, ii_118828)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 17), tuple_118823, subscript_call_result_118831)
    # Adding element type (line 1075)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1075)
    ii_118832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 46), 'ii')
    # Getting the type of 'origins' (line 1075)
    origins_118833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 38), 'origins')
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___118834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 38), origins_118833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_118835 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 38), getitem___118834, ii_118832)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 17), tuple_118823, subscript_call_result_118835)
    # Adding element type (line 1075)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1075)
    ii_118836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 57), 'ii')
    # Getting the type of 'modes' (line 1075)
    modes_118837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 51), 'modes')
    # Obtaining the member '__getitem__' of a type (line 1075)
    getitem___118838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1075, 51), modes_118837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1075)
    subscript_call_result_118839 = invoke(stypy.reporting.localization.Localization(__file__, 1075, 51), getitem___118838, ii_118836)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 17), tuple_118823, subscript_call_result_118839)
    
    list_118854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1075, 16), list_118854, tuple_118823)
    # Assigning a type to the variable 'axes' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 8), 'axes', list_118854)
    
    # Getting the type of 'minimum' (line 1077)
    minimum_118855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 11), 'minimum')
    # Testing the type of an if condition (line 1077)
    if_condition_118856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1077, 8), minimum_118855)
    # Assigning a type to the variable 'if_condition_118856' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'if_condition_118856', if_condition_118856)
    # SSA begins for if statement (line 1077)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1078):
    
    # Assigning a Name to a Name (line 1078):
    # Getting the type of 'minimum_filter1d' (line 1078)
    minimum_filter1d_118857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 22), 'minimum_filter1d')
    # Assigning a type to the variable 'filter_' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 12), 'filter_', minimum_filter1d_118857)
    # SSA branch for the else part of an if statement (line 1077)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 1080):
    
    # Assigning a Name to a Name (line 1080):
    # Getting the type of 'maximum_filter1d' (line 1080)
    maximum_filter1d_118858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 22), 'maximum_filter1d')
    # Assigning a type to the variable 'filter_' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 12), 'filter_', maximum_filter1d_118858)
    # SSA join for if statement (line 1077)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1081)
    # Processing the call arguments (line 1081)
    # Getting the type of 'axes' (line 1081)
    axes_118860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 15), 'axes', False)
    # Processing the call keyword arguments (line 1081)
    kwargs_118861 = {}
    # Getting the type of 'len' (line 1081)
    len_118859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 11), 'len', False)
    # Calling len(args, kwargs) (line 1081)
    len_call_result_118862 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 11), len_118859, *[axes_118860], **kwargs_118861)
    
    int_118863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1081, 23), 'int')
    # Applying the binary operator '>' (line 1081)
    result_gt_118864 = python_operator(stypy.reporting.localization.Localization(__file__, 1081, 11), '>', len_call_result_118862, int_118863)
    
    # Testing the type of an if condition (line 1081)
    if_condition_118865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1081, 8), result_gt_118864)
    # Assigning a type to the variable 'if_condition_118865' (line 1081)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 8), 'if_condition_118865', if_condition_118865)
    # SSA begins for if statement (line 1081)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 1082)
    axes_118866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 44), 'axes')
    # Testing the type of a for loop iterable (line 1082)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1082, 12), axes_118866)
    # Getting the type of the for loop variable (line 1082)
    for_loop_var_118867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1082, 12), axes_118866)
    # Assigning a type to the variable 'axis' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 12), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1082, 12), for_loop_var_118867))
    # Assigning a type to the variable 'size' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 12), 'size', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1082, 12), for_loop_var_118867))
    # Assigning a type to the variable 'origin' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 12), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1082, 12), for_loop_var_118867))
    # Assigning a type to the variable 'mode' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 12), 'mode', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1082, 12), for_loop_var_118867))
    # SSA begins for a for statement (line 1082)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to filter_(...): (line 1083)
    # Processing the call arguments (line 1083)
    # Getting the type of 'input' (line 1083)
    input_118869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 24), 'input', False)
    
    # Call to int(...): (line 1083)
    # Processing the call arguments (line 1083)
    # Getting the type of 'size' (line 1083)
    size_118871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 35), 'size', False)
    # Processing the call keyword arguments (line 1083)
    kwargs_118872 = {}
    # Getting the type of 'int' (line 1083)
    int_118870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 31), 'int', False)
    # Calling int(args, kwargs) (line 1083)
    int_call_result_118873 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 31), int_118870, *[size_118871], **kwargs_118872)
    
    # Getting the type of 'axis' (line 1083)
    axis_118874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 42), 'axis', False)
    # Getting the type of 'output' (line 1083)
    output_118875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 48), 'output', False)
    # Getting the type of 'mode' (line 1083)
    mode_118876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 56), 'mode', False)
    # Getting the type of 'cval' (line 1083)
    cval_118877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 62), 'cval', False)
    # Getting the type of 'origin' (line 1083)
    origin_118878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 68), 'origin', False)
    # Processing the call keyword arguments (line 1083)
    kwargs_118879 = {}
    # Getting the type of 'filter_' (line 1083)
    filter__118868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1083, 16), 'filter_', False)
    # Calling filter_(args, kwargs) (line 1083)
    filter__call_result_118880 = invoke(stypy.reporting.localization.Localization(__file__, 1083, 16), filter__118868, *[input_118869, int_call_result_118873, axis_118874, output_118875, mode_118876, cval_118877, origin_118878], **kwargs_118879)
    
    
    # Assigning a Name to a Name (line 1084):
    
    # Assigning a Name to a Name (line 1084):
    # Getting the type of 'output' (line 1084)
    output_118881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 24), 'output')
    # Assigning a type to the variable 'input' (line 1084)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 16), 'input', output_118881)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1081)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 1086):
    
    # Assigning a Subscript to a Subscript (line 1086):
    
    # Obtaining the type of the subscript
    Ellipsis_118882 = Ellipsis
    # Getting the type of 'input' (line 1086)
    input_118883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 26), 'input')
    # Obtaining the member '__getitem__' of a type (line 1086)
    getitem___118884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1086, 26), input_118883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1086)
    subscript_call_result_118885 = invoke(stypy.reporting.localization.Localization(__file__, 1086, 26), getitem___118884, Ellipsis_118882)
    
    # Getting the type of 'output' (line 1086)
    output_118886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 12), 'output')
    Ellipsis_118887 = Ellipsis
    # Storing an element on a container (line 1086)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1086, 12), output_118886, (Ellipsis_118887, subscript_call_result_118885))
    # SSA join for if statement (line 1081)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1071)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 1088):
    
    # Assigning a ListComp to a Name (line 1088):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'footprint' (line 1088)
    footprint_118892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 31), 'footprint')
    # Obtaining the member 'shape' of a type (line 1088)
    shape_118893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1088, 31), footprint_118892, 'shape')
    comprehension_118894 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1088, 18), shape_118893)
    # Assigning a type to the variable 'ii' (line 1088)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 18), 'ii', comprehension_118894)
    
    # Getting the type of 'ii' (line 1088)
    ii_118889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 50), 'ii')
    int_118890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 55), 'int')
    # Applying the binary operator '>' (line 1088)
    result_gt_118891 = python_operator(stypy.reporting.localization.Localization(__file__, 1088, 50), '>', ii_118889, int_118890)
    
    # Getting the type of 'ii' (line 1088)
    ii_118888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 18), 'ii')
    list_118895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1088, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1088, 18), list_118895, ii_118888)
    # Assigning a type to the variable 'fshape' (line 1088)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1088, 8), 'fshape', list_118895)
    
    
    
    # Call to len(...): (line 1089)
    # Processing the call arguments (line 1089)
    # Getting the type of 'fshape' (line 1089)
    fshape_118897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 15), 'fshape', False)
    # Processing the call keyword arguments (line 1089)
    kwargs_118898 = {}
    # Getting the type of 'len' (line 1089)
    len_118896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 11), 'len', False)
    # Calling len(args, kwargs) (line 1089)
    len_call_result_118899 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 11), len_118896, *[fshape_118897], **kwargs_118898)
    
    # Getting the type of 'input' (line 1089)
    input_118900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 26), 'input')
    # Obtaining the member 'ndim' of a type (line 1089)
    ndim_118901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 26), input_118900, 'ndim')
    # Applying the binary operator '!=' (line 1089)
    result_ne_118902 = python_operator(stypy.reporting.localization.Localization(__file__, 1089, 11), '!=', len_call_result_118899, ndim_118901)
    
    # Testing the type of an if condition (line 1089)
    if_condition_118903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1089, 8), result_ne_118902)
    # Assigning a type to the variable 'if_condition_118903' (line 1089)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1089, 8), 'if_condition_118903', if_condition_118903)
    # SSA begins for if statement (line 1089)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1090)
    # Processing the call arguments (line 1090)
    str_118905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1090, 31), 'str', 'footprint array has incorrect shape.')
    # Processing the call keyword arguments (line 1090)
    kwargs_118906 = {}
    # Getting the type of 'RuntimeError' (line 1090)
    RuntimeError_118904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1090, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1090)
    RuntimeError_call_result_118907 = invoke(stypy.reporting.localization.Localization(__file__, 1090, 18), RuntimeError_118904, *[str_118905], **kwargs_118906)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1090, 12), RuntimeError_call_result_118907, 'raise parameter', BaseException)
    # SSA join for if statement (line 1089)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to zip(...): (line 1091)
    # Processing the call arguments (line 1091)
    # Getting the type of 'origins' (line 1091)
    origins_118909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 32), 'origins', False)
    # Getting the type of 'fshape' (line 1091)
    fshape_118910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 41), 'fshape', False)
    # Processing the call keyword arguments (line 1091)
    kwargs_118911 = {}
    # Getting the type of 'zip' (line 1091)
    zip_118908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 28), 'zip', False)
    # Calling zip(args, kwargs) (line 1091)
    zip_call_result_118912 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 28), zip_118908, *[origins_118909, fshape_118910], **kwargs_118911)
    
    # Testing the type of a for loop iterable (line 1091)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1091, 8), zip_call_result_118912)
    # Getting the type of the for loop variable (line 1091)
    for_loop_var_118913 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1091, 8), zip_call_result_118912)
    # Assigning a type to the variable 'origin' (line 1091)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1091, 8), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1091, 8), for_loop_var_118913))
    # Assigning a type to the variable 'lenf' (line 1091)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1091, 8), 'lenf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1091, 8), for_loop_var_118913))
    # SSA begins for a for statement (line 1091)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lenf' (line 1092)
    lenf_118914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 16), 'lenf')
    int_118915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 24), 'int')
    # Applying the binary operator '//' (line 1092)
    result_floordiv_118916 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 16), '//', lenf_118914, int_118915)
    
    # Getting the type of 'origin' (line 1092)
    origin_118917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 28), 'origin')
    # Applying the binary operator '+' (line 1092)
    result_add_118918 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 16), '+', result_floordiv_118916, origin_118917)
    
    int_118919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 37), 'int')
    # Applying the binary operator '<' (line 1092)
    result_lt_118920 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 16), '<', result_add_118918, int_118919)
    
    
    # Getting the type of 'lenf' (line 1092)
    lenf_118921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 44), 'lenf')
    int_118922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, 52), 'int')
    # Applying the binary operator '//' (line 1092)
    result_floordiv_118923 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 44), '//', lenf_118921, int_118922)
    
    # Getting the type of 'origin' (line 1092)
    origin_118924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 56), 'origin')
    # Applying the binary operator '+' (line 1092)
    result_add_118925 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 44), '+', result_floordiv_118923, origin_118924)
    
    # Getting the type of 'lenf' (line 1092)
    lenf_118926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 66), 'lenf')
    # Applying the binary operator '>=' (line 1092)
    result_ge_118927 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 44), '>=', result_add_118925, lenf_118926)
    
    # Applying the binary operator 'or' (line 1092)
    result_or_keyword_118928 = python_operator(stypy.reporting.localization.Localization(__file__, 1092, 15), 'or', result_lt_118920, result_ge_118927)
    
    # Testing the type of an if condition (line 1092)
    if_condition_118929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1092, 12), result_or_keyword_118928)
    # Assigning a type to the variable 'if_condition_118929' (line 1092)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1092, 12), 'if_condition_118929', if_condition_118929)
    # SSA begins for if statement (line 1092)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1093)
    # Processing the call arguments (line 1093)
    str_118931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, 33), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 1093)
    kwargs_118932 = {}
    # Getting the type of 'ValueError' (line 1093)
    ValueError_118930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1093)
    ValueError_call_result_118933 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 22), ValueError_118930, *[str_118931], **kwargs_118932)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1093, 16), ValueError_call_result_118933, 'raise parameter', BaseException)
    # SSA join for if statement (line 1092)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'footprint' (line 1094)
    footprint_118934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 15), 'footprint')
    # Obtaining the member 'flags' of a type (line 1094)
    flags_118935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 15), footprint_118934, 'flags')
    # Obtaining the member 'contiguous' of a type (line 1094)
    contiguous_118936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 15), flags_118935, 'contiguous')
    # Applying the 'not' unary operator (line 1094)
    result_not__118937 = python_operator(stypy.reporting.localization.Localization(__file__, 1094, 11), 'not', contiguous_118936)
    
    # Testing the type of an if condition (line 1094)
    if_condition_118938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1094, 8), result_not__118937)
    # Assigning a type to the variable 'if_condition_118938' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'if_condition_118938', if_condition_118938)
    # SSA begins for if statement (line 1094)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1095):
    
    # Assigning a Call to a Name (line 1095):
    
    # Call to copy(...): (line 1095)
    # Processing the call keyword arguments (line 1095)
    kwargs_118941 = {}
    # Getting the type of 'footprint' (line 1095)
    footprint_118939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 24), 'footprint', False)
    # Obtaining the member 'copy' of a type (line 1095)
    copy_118940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 24), footprint_118939, 'copy')
    # Calling copy(args, kwargs) (line 1095)
    copy_call_result_118942 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 24), copy_118940, *[], **kwargs_118941)
    
    # Assigning a type to the variable 'footprint' (line 1095)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 12), 'footprint', copy_call_result_118942)
    # SSA join for if statement (line 1094)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1096)
    # Getting the type of 'structure' (line 1096)
    structure_118943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 8), 'structure')
    # Getting the type of 'None' (line 1096)
    None_118944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 28), 'None')
    
    (may_be_118945, more_types_in_union_118946) = may_not_be_none(structure_118943, None_118944)

    if may_be_118945:

        if more_types_in_union_118946:
            # Runtime conditional SSA (line 1096)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 1097)
        # Processing the call arguments (line 1097)
        # Getting the type of 'structure' (line 1097)
        structure_118948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 19), 'structure', False)
        # Obtaining the member 'shape' of a type (line 1097)
        shape_118949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 19), structure_118948, 'shape')
        # Processing the call keyword arguments (line 1097)
        kwargs_118950 = {}
        # Getting the type of 'len' (line 1097)
        len_118947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 15), 'len', False)
        # Calling len(args, kwargs) (line 1097)
        len_call_result_118951 = invoke(stypy.reporting.localization.Localization(__file__, 1097, 15), len_118947, *[shape_118949], **kwargs_118950)
        
        # Getting the type of 'input' (line 1097)
        input_118952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 39), 'input')
        # Obtaining the member 'ndim' of a type (line 1097)
        ndim_118953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 39), input_118952, 'ndim')
        # Applying the binary operator '!=' (line 1097)
        result_ne_118954 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 15), '!=', len_call_result_118951, ndim_118953)
        
        # Testing the type of an if condition (line 1097)
        if_condition_118955 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1097, 12), result_ne_118954)
        # Assigning a type to the variable 'if_condition_118955' (line 1097)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 12), 'if_condition_118955', if_condition_118955)
        # SSA begins for if statement (line 1097)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 1098)
        # Processing the call arguments (line 1098)
        str_118957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 35), 'str', 'structure array has incorrect shape')
        # Processing the call keyword arguments (line 1098)
        kwargs_118958 = {}
        # Getting the type of 'RuntimeError' (line 1098)
        RuntimeError_118956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 1098)
        RuntimeError_call_result_118959 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 22), RuntimeError_118956, *[str_118957], **kwargs_118958)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1098, 16), RuntimeError_call_result_118959, 'raise parameter', BaseException)
        # SSA join for if statement (line 1097)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'structure' (line 1099)
        structure_118960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 19), 'structure')
        # Obtaining the member 'flags' of a type (line 1099)
        flags_118961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 19), structure_118960, 'flags')
        # Obtaining the member 'contiguous' of a type (line 1099)
        contiguous_118962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1099, 19), flags_118961, 'contiguous')
        # Applying the 'not' unary operator (line 1099)
        result_not__118963 = python_operator(stypy.reporting.localization.Localization(__file__, 1099, 15), 'not', contiguous_118962)
        
        # Testing the type of an if condition (line 1099)
        if_condition_118964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1099, 12), result_not__118963)
        # Assigning a type to the variable 'if_condition_118964' (line 1099)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 12), 'if_condition_118964', if_condition_118964)
        # SSA begins for if statement (line 1099)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1100):
        
        # Assigning a Call to a Name (line 1100):
        
        # Call to copy(...): (line 1100)
        # Processing the call keyword arguments (line 1100)
        kwargs_118967 = {}
        # Getting the type of 'structure' (line 1100)
        structure_118965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 28), 'structure', False)
        # Obtaining the member 'copy' of a type (line 1100)
        copy_118966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 28), structure_118965, 'copy')
        # Calling copy(args, kwargs) (line 1100)
        copy_call_result_118968 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 28), copy_118966, *[], **kwargs_118967)
        
        # Assigning a type to the variable 'structure' (line 1100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 16), 'structure', copy_call_result_118968)
        # SSA join for if statement (line 1099)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_118946:
            # SSA join for if statement (line 1096)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1101):
    
    # Assigning a Call to a Name (line 1101):
    
    # Call to _extend_mode_to_code(...): (line 1101)
    # Processing the call arguments (line 1101)
    # Getting the type of 'mode' (line 1101)
    mode_118971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 48), 'mode', False)
    # Processing the call keyword arguments (line 1101)
    kwargs_118972 = {}
    # Getting the type of '_ni_support' (line 1101)
    _ni_support_118969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 15), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 1101)
    _extend_mode_to_code_118970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 15), _ni_support_118969, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 1101)
    _extend_mode_to_code_call_result_118973 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 15), _extend_mode_to_code_118970, *[mode_118971], **kwargs_118972)
    
    # Assigning a type to the variable 'mode' (line 1101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 8), 'mode', _extend_mode_to_code_call_result_118973)
    
    # Call to min_or_max_filter(...): (line 1102)
    # Processing the call arguments (line 1102)
    # Getting the type of 'input' (line 1102)
    input_118976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 36), 'input', False)
    # Getting the type of 'footprint' (line 1102)
    footprint_118977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 43), 'footprint', False)
    # Getting the type of 'structure' (line 1102)
    structure_118978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 54), 'structure', False)
    # Getting the type of 'output' (line 1102)
    output_118979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 65), 'output', False)
    # Getting the type of 'mode' (line 1103)
    mode_118980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 36), 'mode', False)
    # Getting the type of 'cval' (line 1103)
    cval_118981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 42), 'cval', False)
    # Getting the type of 'origins' (line 1103)
    origins_118982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 48), 'origins', False)
    # Getting the type of 'minimum' (line 1103)
    minimum_118983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 57), 'minimum', False)
    # Processing the call keyword arguments (line 1102)
    kwargs_118984 = {}
    # Getting the type of '_nd_image' (line 1102)
    _nd_image_118974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 8), '_nd_image', False)
    # Obtaining the member 'min_or_max_filter' of a type (line 1102)
    min_or_max_filter_118975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1102, 8), _nd_image_118974, 'min_or_max_filter')
    # Calling min_or_max_filter(args, kwargs) (line 1102)
    min_or_max_filter_call_result_118985 = invoke(stypy.reporting.localization.Localization(__file__, 1102, 8), min_or_max_filter_118975, *[input_118976, footprint_118977, structure_118978, output_118979, mode_118980, cval_118981, origins_118982, minimum_118983], **kwargs_118984)
    
    # SSA join for if statement (line 1071)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'return_value' (line 1104)
    return_value_118986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1104, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1104, 4), 'stypy_return_type', return_value_118986)
    
    # ################# End of '_min_or_max_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_min_or_max_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1042)
    stypy_return_type_118987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_118987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_min_or_max_filter'
    return stypy_return_type_118987

# Assigning a type to the variable '_min_or_max_filter' (line 1042)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 0), '_min_or_max_filter', _min_or_max_filter)

@norecursion
def minimum_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1108)
    None_118988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 31), 'None')
    # Getting the type of 'None' (line 1108)
    None_118989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 47), 'None')
    # Getting the type of 'None' (line 1108)
    None_118990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 60), 'None')
    str_118991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 24), 'str', 'reflect')
    float_118992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 40), 'float')
    int_118993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1109, 52), 'int')
    defaults = [None_118988, None_118989, None_118990, str_118991, float_118992, int_118993]
    # Create a new context for function 'minimum_filter'
    module_type_store = module_type_store.open_function_context('minimum_filter', 1107, 0, False)
    
    # Passed parameters checking function
    minimum_filter.stypy_localization = localization
    minimum_filter.stypy_type_of_self = None
    minimum_filter.stypy_type_store = module_type_store
    minimum_filter.stypy_function_name = 'minimum_filter'
    minimum_filter.stypy_param_names_list = ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin']
    minimum_filter.stypy_varargs_param_name = None
    minimum_filter.stypy_kwargs_param_name = None
    minimum_filter.stypy_call_defaults = defaults
    minimum_filter.stypy_call_varargs = varargs
    minimum_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimum_filter', ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimum_filter', localization, ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimum_filter(...)' code ##################

    str_118994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, (-1)), 'str', 'Calculate a multi-dimensional minimum filter.\n\n    Parameters\n    ----------\n    %(input)s\n    %(size_foot)s\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    minimum_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.minimum_filter(ascent, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Call to _min_or_max_filter(...): (line 1140)
    # Processing the call arguments (line 1140)
    # Getting the type of 'input' (line 1140)
    input_118996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 30), 'input', False)
    # Getting the type of 'size' (line 1140)
    size_118997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 37), 'size', False)
    # Getting the type of 'footprint' (line 1140)
    footprint_118998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 43), 'footprint', False)
    # Getting the type of 'None' (line 1140)
    None_118999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 54), 'None', False)
    # Getting the type of 'output' (line 1140)
    output_119000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 60), 'output', False)
    # Getting the type of 'mode' (line 1140)
    mode_119001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 68), 'mode', False)
    # Getting the type of 'cval' (line 1141)
    cval_119002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 30), 'cval', False)
    # Getting the type of 'origin' (line 1141)
    origin_119003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 36), 'origin', False)
    int_119004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 44), 'int')
    # Processing the call keyword arguments (line 1140)
    kwargs_119005 = {}
    # Getting the type of '_min_or_max_filter' (line 1140)
    _min_or_max_filter_118995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 11), '_min_or_max_filter', False)
    # Calling _min_or_max_filter(args, kwargs) (line 1140)
    _min_or_max_filter_call_result_119006 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 11), _min_or_max_filter_118995, *[input_118996, size_118997, footprint_118998, None_118999, output_119000, mode_119001, cval_119002, origin_119003, int_119004], **kwargs_119005)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 4), 'stypy_return_type', _min_or_max_filter_call_result_119006)
    
    # ################# End of 'minimum_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimum_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1107)
    stypy_return_type_119007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimum_filter'
    return stypy_return_type_119007

# Assigning a type to the variable 'minimum_filter' (line 1107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1107, 0), 'minimum_filter', minimum_filter)

@norecursion
def maximum_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1145)
    None_119008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 31), 'None')
    # Getting the type of 'None' (line 1145)
    None_119009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 47), 'None')
    # Getting the type of 'None' (line 1145)
    None_119010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1145, 60), 'None')
    str_119011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 24), 'str', 'reflect')
    float_119012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 40), 'float')
    int_119013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1146, 52), 'int')
    defaults = [None_119008, None_119009, None_119010, str_119011, float_119012, int_119013]
    # Create a new context for function 'maximum_filter'
    module_type_store = module_type_store.open_function_context('maximum_filter', 1144, 0, False)
    
    # Passed parameters checking function
    maximum_filter.stypy_localization = localization
    maximum_filter.stypy_type_of_self = None
    maximum_filter.stypy_type_store = module_type_store
    maximum_filter.stypy_function_name = 'maximum_filter'
    maximum_filter.stypy_param_names_list = ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin']
    maximum_filter.stypy_varargs_param_name = None
    maximum_filter.stypy_kwargs_param_name = None
    maximum_filter.stypy_call_defaults = defaults
    maximum_filter.stypy_call_varargs = varargs
    maximum_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'maximum_filter', ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'maximum_filter', localization, ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'maximum_filter(...)' code ##################

    str_119014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1176, (-1)), 'str', 'Calculate a multi-dimensional maximum filter.\n\n    Parameters\n    ----------\n    %(input)s\n    %(size_foot)s\n    %(output)s\n    %(mode_multiple)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    maximum_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.maximum_filter(ascent, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Call to _min_or_max_filter(...): (line 1177)
    # Processing the call arguments (line 1177)
    # Getting the type of 'input' (line 1177)
    input_119016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 30), 'input', False)
    # Getting the type of 'size' (line 1177)
    size_119017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 37), 'size', False)
    # Getting the type of 'footprint' (line 1177)
    footprint_119018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 43), 'footprint', False)
    # Getting the type of 'None' (line 1177)
    None_119019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 54), 'None', False)
    # Getting the type of 'output' (line 1177)
    output_119020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 60), 'output', False)
    # Getting the type of 'mode' (line 1177)
    mode_119021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 68), 'mode', False)
    # Getting the type of 'cval' (line 1178)
    cval_119022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 30), 'cval', False)
    # Getting the type of 'origin' (line 1178)
    origin_119023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 36), 'origin', False)
    int_119024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 44), 'int')
    # Processing the call keyword arguments (line 1177)
    kwargs_119025 = {}
    # Getting the type of '_min_or_max_filter' (line 1177)
    _min_or_max_filter_119015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 11), '_min_or_max_filter', False)
    # Calling _min_or_max_filter(args, kwargs) (line 1177)
    _min_or_max_filter_call_result_119026 = invoke(stypy.reporting.localization.Localization(__file__, 1177, 11), _min_or_max_filter_119015, *[input_119016, size_119017, footprint_119018, None_119019, output_119020, mode_119021, cval_119022, origin_119023, int_119024], **kwargs_119025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 4), 'stypy_return_type', _min_or_max_filter_call_result_119026)
    
    # ################# End of 'maximum_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'maximum_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1144)
    stypy_return_type_119027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'maximum_filter'
    return stypy_return_type_119027

# Assigning a type to the variable 'maximum_filter' (line 1144)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 0), 'maximum_filter', maximum_filter)

@norecursion
def _rank_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1182)
    None_119028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 35), 'None')
    # Getting the type of 'None' (line 1182)
    None_119029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 51), 'None')
    # Getting the type of 'None' (line 1182)
    None_119030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 64), 'None')
    str_119031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 22), 'str', 'reflect')
    float_119032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 38), 'float')
    int_119033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 50), 'int')
    str_119034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 63), 'str', 'rank')
    defaults = [None_119028, None_119029, None_119030, str_119031, float_119032, int_119033, str_119034]
    # Create a new context for function '_rank_filter'
    module_type_store = module_type_store.open_function_context('_rank_filter', 1181, 0, False)
    
    # Passed parameters checking function
    _rank_filter.stypy_localization = localization
    _rank_filter.stypy_type_of_self = None
    _rank_filter.stypy_type_store = module_type_store
    _rank_filter.stypy_function_name = '_rank_filter'
    _rank_filter.stypy_param_names_list = ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'operation']
    _rank_filter.stypy_varargs_param_name = None
    _rank_filter.stypy_kwargs_param_name = None
    _rank_filter.stypy_call_defaults = defaults
    _rank_filter.stypy_call_varargs = varargs
    _rank_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_rank_filter', ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'operation'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_rank_filter', localization, ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'operation'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_rank_filter(...)' code ##################

    
    # Assigning a Call to a Name (line 1184):
    
    # Assigning a Call to a Name (line 1184):
    
    # Call to asarray(...): (line 1184)
    # Processing the call arguments (line 1184)
    # Getting the type of 'input' (line 1184)
    input_119037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 26), 'input', False)
    # Processing the call keyword arguments (line 1184)
    kwargs_119038 = {}
    # Getting the type of 'numpy' (line 1184)
    numpy_119035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1184)
    asarray_119036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1184, 12), numpy_119035, 'asarray')
    # Calling asarray(args, kwargs) (line 1184)
    asarray_call_result_119039 = invoke(stypy.reporting.localization.Localization(__file__, 1184, 12), asarray_119036, *[input_119037], **kwargs_119038)
    
    # Assigning a type to the variable 'input' (line 1184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 4), 'input', asarray_call_result_119039)
    
    
    # Call to iscomplexobj(...): (line 1185)
    # Processing the call arguments (line 1185)
    # Getting the type of 'input' (line 1185)
    input_119042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 26), 'input', False)
    # Processing the call keyword arguments (line 1185)
    kwargs_119043 = {}
    # Getting the type of 'numpy' (line 1185)
    numpy_119040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1185)
    iscomplexobj_119041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1185, 7), numpy_119040, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1185)
    iscomplexobj_call_result_119044 = invoke(stypy.reporting.localization.Localization(__file__, 1185, 7), iscomplexobj_119041, *[input_119042], **kwargs_119043)
    
    # Testing the type of an if condition (line 1185)
    if_condition_119045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1185, 4), iscomplexobj_call_result_119044)
    # Assigning a type to the variable 'if_condition_119045' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'if_condition_119045', if_condition_119045)
    # SSA begins for if statement (line 1185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1186)
    # Processing the call arguments (line 1186)
    str_119047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1186, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 1186)
    kwargs_119048 = {}
    # Getting the type of 'TypeError' (line 1186)
    TypeError_119046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1186)
    TypeError_call_result_119049 = invoke(stypy.reporting.localization.Localization(__file__, 1186, 14), TypeError_119046, *[str_119047], **kwargs_119048)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1186, 8), TypeError_call_result_119049, 'raise parameter', BaseException)
    # SSA join for if statement (line 1185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1187):
    
    # Assigning a Call to a Name (line 1187):
    
    # Call to _normalize_sequence(...): (line 1187)
    # Processing the call arguments (line 1187)
    # Getting the type of 'origin' (line 1187)
    origin_119052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 46), 'origin', False)
    # Getting the type of 'input' (line 1187)
    input_119053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1187)
    ndim_119054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1187, 54), input_119053, 'ndim')
    # Processing the call keyword arguments (line 1187)
    kwargs_119055 = {}
    # Getting the type of '_ni_support' (line 1187)
    _ni_support_119050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1187)
    _normalize_sequence_119051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1187, 14), _ni_support_119050, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1187)
    _normalize_sequence_call_result_119056 = invoke(stypy.reporting.localization.Localization(__file__, 1187, 14), _normalize_sequence_119051, *[origin_119052, ndim_119054], **kwargs_119055)
    
    # Assigning a type to the variable 'origins' (line 1187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1187, 4), 'origins', _normalize_sequence_call_result_119056)
    
    # Type idiom detected: calculating its left and rigth part (line 1188)
    # Getting the type of 'footprint' (line 1188)
    footprint_119057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 7), 'footprint')
    # Getting the type of 'None' (line 1188)
    None_119058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 20), 'None')
    
    (may_be_119059, more_types_in_union_119060) = may_be_none(footprint_119057, None_119058)

    if may_be_119059:

        if more_types_in_union_119060:
            # Runtime conditional SSA (line 1188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 1189)
        # Getting the type of 'size' (line 1189)
        size_119061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 11), 'size')
        # Getting the type of 'None' (line 1189)
        None_119062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 19), 'None')
        
        (may_be_119063, more_types_in_union_119064) = may_be_none(size_119061, None_119062)

        if may_be_119063:

            if more_types_in_union_119064:
                # Runtime conditional SSA (line 1189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to RuntimeError(...): (line 1190)
            # Processing the call arguments (line 1190)
            str_119066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1190, 31), 'str', 'no footprint or filter size provided')
            # Processing the call keyword arguments (line 1190)
            kwargs_119067 = {}
            # Getting the type of 'RuntimeError' (line 1190)
            RuntimeError_119065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 1190)
            RuntimeError_call_result_119068 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 18), RuntimeError_119065, *[str_119066], **kwargs_119067)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1190, 12), RuntimeError_call_result_119068, 'raise parameter', BaseException)

            if more_types_in_union_119064:
                # SSA join for if statement (line 1189)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1191):
        
        # Assigning a Call to a Name (line 1191):
        
        # Call to _normalize_sequence(...): (line 1191)
        # Processing the call arguments (line 1191)
        # Getting the type of 'size' (line 1191)
        size_119071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 48), 'size', False)
        # Getting the type of 'input' (line 1191)
        input_119072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 54), 'input', False)
        # Obtaining the member 'ndim' of a type (line 1191)
        ndim_119073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 54), input_119072, 'ndim')
        # Processing the call keyword arguments (line 1191)
        kwargs_119074 = {}
        # Getting the type of '_ni_support' (line 1191)
        _ni_support_119069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1191, 16), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 1191)
        _normalize_sequence_119070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1191, 16), _ni_support_119069, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 1191)
        _normalize_sequence_call_result_119075 = invoke(stypy.reporting.localization.Localization(__file__, 1191, 16), _normalize_sequence_119070, *[size_119071, ndim_119073], **kwargs_119074)
        
        # Assigning a type to the variable 'sizes' (line 1191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1191, 8), 'sizes', _normalize_sequence_call_result_119075)
        
        # Assigning a Call to a Name (line 1192):
        
        # Assigning a Call to a Name (line 1192):
        
        # Call to ones(...): (line 1192)
        # Processing the call arguments (line 1192)
        # Getting the type of 'sizes' (line 1192)
        sizes_119078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 31), 'sizes', False)
        # Processing the call keyword arguments (line 1192)
        # Getting the type of 'bool' (line 1192)
        bool_119079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 44), 'bool', False)
        keyword_119080 = bool_119079
        kwargs_119081 = {'dtype': keyword_119080}
        # Getting the type of 'numpy' (line 1192)
        numpy_119076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 20), 'numpy', False)
        # Obtaining the member 'ones' of a type (line 1192)
        ones_119077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 20), numpy_119076, 'ones')
        # Calling ones(args, kwargs) (line 1192)
        ones_call_result_119082 = invoke(stypy.reporting.localization.Localization(__file__, 1192, 20), ones_119077, *[sizes_119078], **kwargs_119081)
        
        # Assigning a type to the variable 'footprint' (line 1192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 8), 'footprint', ones_call_result_119082)

        if more_types_in_union_119060:
            # Runtime conditional SSA for else branch (line 1188)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_119059) or more_types_in_union_119060):
        
        # Assigning a Call to a Name (line 1194):
        
        # Assigning a Call to a Name (line 1194):
        
        # Call to asarray(...): (line 1194)
        # Processing the call arguments (line 1194)
        # Getting the type of 'footprint' (line 1194)
        footprint_119085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 34), 'footprint', False)
        # Processing the call keyword arguments (line 1194)
        # Getting the type of 'bool' (line 1194)
        bool_119086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 51), 'bool', False)
        keyword_119087 = bool_119086
        kwargs_119088 = {'dtype': keyword_119087}
        # Getting the type of 'numpy' (line 1194)
        numpy_119083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1194)
        asarray_119084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1194, 20), numpy_119083, 'asarray')
        # Calling asarray(args, kwargs) (line 1194)
        asarray_call_result_119089 = invoke(stypy.reporting.localization.Localization(__file__, 1194, 20), asarray_119084, *[footprint_119085], **kwargs_119088)
        
        # Assigning a type to the variable 'footprint' (line 1194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 8), 'footprint', asarray_call_result_119089)

        if (may_be_119059 and more_types_in_union_119060):
            # SSA join for if statement (line 1188)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a ListComp to a Name (line 1195):
    
    # Assigning a ListComp to a Name (line 1195):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'footprint' (line 1195)
    footprint_119094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 27), 'footprint')
    # Obtaining the member 'shape' of a type (line 1195)
    shape_119095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1195, 27), footprint_119094, 'shape')
    comprehension_119096 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1195, 14), shape_119095)
    # Assigning a type to the variable 'ii' (line 1195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 14), 'ii', comprehension_119096)
    
    # Getting the type of 'ii' (line 1195)
    ii_119091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 46), 'ii')
    int_119092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1195, 51), 'int')
    # Applying the binary operator '>' (line 1195)
    result_gt_119093 = python_operator(stypy.reporting.localization.Localization(__file__, 1195, 46), '>', ii_119091, int_119092)
    
    # Getting the type of 'ii' (line 1195)
    ii_119090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 14), 'ii')
    list_119097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1195, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1195, 14), list_119097, ii_119090)
    # Assigning a type to the variable 'fshape' (line 1195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 4), 'fshape', list_119097)
    
    
    
    # Call to len(...): (line 1196)
    # Processing the call arguments (line 1196)
    # Getting the type of 'fshape' (line 1196)
    fshape_119099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 11), 'fshape', False)
    # Processing the call keyword arguments (line 1196)
    kwargs_119100 = {}
    # Getting the type of 'len' (line 1196)
    len_119098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 7), 'len', False)
    # Calling len(args, kwargs) (line 1196)
    len_call_result_119101 = invoke(stypy.reporting.localization.Localization(__file__, 1196, 7), len_119098, *[fshape_119099], **kwargs_119100)
    
    # Getting the type of 'input' (line 1196)
    input_119102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 22), 'input')
    # Obtaining the member 'ndim' of a type (line 1196)
    ndim_119103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1196, 22), input_119102, 'ndim')
    # Applying the binary operator '!=' (line 1196)
    result_ne_119104 = python_operator(stypy.reporting.localization.Localization(__file__, 1196, 7), '!=', len_call_result_119101, ndim_119103)
    
    # Testing the type of an if condition (line 1196)
    if_condition_119105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1196, 4), result_ne_119104)
    # Assigning a type to the variable 'if_condition_119105' (line 1196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 4), 'if_condition_119105', if_condition_119105)
    # SSA begins for if statement (line 1196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1197)
    # Processing the call arguments (line 1197)
    str_119107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 27), 'str', 'filter footprint array has incorrect shape.')
    # Processing the call keyword arguments (line 1197)
    kwargs_119108 = {}
    # Getting the type of 'RuntimeError' (line 1197)
    RuntimeError_119106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1197)
    RuntimeError_call_result_119109 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 14), RuntimeError_119106, *[str_119107], **kwargs_119108)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1197, 8), RuntimeError_call_result_119109, 'raise parameter', BaseException)
    # SSA join for if statement (line 1196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to zip(...): (line 1198)
    # Processing the call arguments (line 1198)
    # Getting the type of 'origins' (line 1198)
    origins_119111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 28), 'origins', False)
    # Getting the type of 'fshape' (line 1198)
    fshape_119112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 37), 'fshape', False)
    # Processing the call keyword arguments (line 1198)
    kwargs_119113 = {}
    # Getting the type of 'zip' (line 1198)
    zip_119110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 24), 'zip', False)
    # Calling zip(args, kwargs) (line 1198)
    zip_call_result_119114 = invoke(stypy.reporting.localization.Localization(__file__, 1198, 24), zip_119110, *[origins_119111, fshape_119112], **kwargs_119113)
    
    # Testing the type of a for loop iterable (line 1198)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1198, 4), zip_call_result_119114)
    # Getting the type of the for loop variable (line 1198)
    for_loop_var_119115 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1198, 4), zip_call_result_119114)
    # Assigning a type to the variable 'origin' (line 1198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 4), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1198, 4), for_loop_var_119115))
    # Assigning a type to the variable 'lenf' (line 1198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 4), 'lenf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1198, 4), for_loop_var_119115))
    # SSA begins for a for statement (line 1198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lenf' (line 1199)
    lenf_119116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 12), 'lenf')
    int_119117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 20), 'int')
    # Applying the binary operator '//' (line 1199)
    result_floordiv_119118 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 12), '//', lenf_119116, int_119117)
    
    # Getting the type of 'origin' (line 1199)
    origin_119119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 24), 'origin')
    # Applying the binary operator '+' (line 1199)
    result_add_119120 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 12), '+', result_floordiv_119118, origin_119119)
    
    int_119121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 33), 'int')
    # Applying the binary operator '<' (line 1199)
    result_lt_119122 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 12), '<', result_add_119120, int_119121)
    
    
    # Getting the type of 'lenf' (line 1199)
    lenf_119123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 40), 'lenf')
    int_119124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1199, 48), 'int')
    # Applying the binary operator '//' (line 1199)
    result_floordiv_119125 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 40), '//', lenf_119123, int_119124)
    
    # Getting the type of 'origin' (line 1199)
    origin_119126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 52), 'origin')
    # Applying the binary operator '+' (line 1199)
    result_add_119127 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 40), '+', result_floordiv_119125, origin_119126)
    
    # Getting the type of 'lenf' (line 1199)
    lenf_119128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 62), 'lenf')
    # Applying the binary operator '>=' (line 1199)
    result_ge_119129 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 40), '>=', result_add_119127, lenf_119128)
    
    # Applying the binary operator 'or' (line 1199)
    result_or_keyword_119130 = python_operator(stypy.reporting.localization.Localization(__file__, 1199, 11), 'or', result_lt_119122, result_ge_119129)
    
    # Testing the type of an if condition (line 1199)
    if_condition_119131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1199, 8), result_or_keyword_119130)
    # Assigning a type to the variable 'if_condition_119131' (line 1199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1199, 8), 'if_condition_119131', if_condition_119131)
    # SSA begins for if statement (line 1199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1200)
    # Processing the call arguments (line 1200)
    str_119133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1200, 29), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 1200)
    kwargs_119134 = {}
    # Getting the type of 'ValueError' (line 1200)
    ValueError_119132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1200)
    ValueError_call_result_119135 = invoke(stypy.reporting.localization.Localization(__file__, 1200, 18), ValueError_119132, *[str_119133], **kwargs_119134)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1200, 12), ValueError_call_result_119135, 'raise parameter', BaseException)
    # SSA join for if statement (line 1199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'footprint' (line 1201)
    footprint_119136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 11), 'footprint')
    # Obtaining the member 'flags' of a type (line 1201)
    flags_119137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 11), footprint_119136, 'flags')
    # Obtaining the member 'contiguous' of a type (line 1201)
    contiguous_119138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1201, 11), flags_119137, 'contiguous')
    # Applying the 'not' unary operator (line 1201)
    result_not__119139 = python_operator(stypy.reporting.localization.Localization(__file__, 1201, 7), 'not', contiguous_119138)
    
    # Testing the type of an if condition (line 1201)
    if_condition_119140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1201, 4), result_not__119139)
    # Assigning a type to the variable 'if_condition_119140' (line 1201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1201, 4), 'if_condition_119140', if_condition_119140)
    # SSA begins for if statement (line 1201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1202):
    
    # Assigning a Call to a Name (line 1202):
    
    # Call to copy(...): (line 1202)
    # Processing the call keyword arguments (line 1202)
    kwargs_119143 = {}
    # Getting the type of 'footprint' (line 1202)
    footprint_119141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 20), 'footprint', False)
    # Obtaining the member 'copy' of a type (line 1202)
    copy_119142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1202, 20), footprint_119141, 'copy')
    # Calling copy(args, kwargs) (line 1202)
    copy_call_result_119144 = invoke(stypy.reporting.localization.Localization(__file__, 1202, 20), copy_119142, *[], **kwargs_119143)
    
    # Assigning a type to the variable 'footprint' (line 1202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 8), 'footprint', copy_call_result_119144)
    # SSA join for if statement (line 1201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1203):
    
    # Assigning a Call to a Name (line 1203):
    
    # Call to sum(...): (line 1203)
    # Processing the call keyword arguments (line 1203)
    kwargs_119153 = {}
    
    # Call to where(...): (line 1203)
    # Processing the call arguments (line 1203)
    # Getting the type of 'footprint' (line 1203)
    footprint_119147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 30), 'footprint', False)
    int_119148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 41), 'int')
    int_119149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 44), 'int')
    # Processing the call keyword arguments (line 1203)
    kwargs_119150 = {}
    # Getting the type of 'numpy' (line 1203)
    numpy_119145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 18), 'numpy', False)
    # Obtaining the member 'where' of a type (line 1203)
    where_119146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 18), numpy_119145, 'where')
    # Calling where(args, kwargs) (line 1203)
    where_call_result_119151 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 18), where_119146, *[footprint_119147, int_119148, int_119149], **kwargs_119150)
    
    # Obtaining the member 'sum' of a type (line 1203)
    sum_119152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1203, 18), where_call_result_119151, 'sum')
    # Calling sum(args, kwargs) (line 1203)
    sum_call_result_119154 = invoke(stypy.reporting.localization.Localization(__file__, 1203, 18), sum_119152, *[], **kwargs_119153)
    
    # Assigning a type to the variable 'filter_size' (line 1203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 4), 'filter_size', sum_call_result_119154)
    
    
    # Getting the type of 'operation' (line 1204)
    operation_119155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 7), 'operation')
    str_119156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1204, 20), 'str', 'median')
    # Applying the binary operator '==' (line 1204)
    result_eq_119157 = python_operator(stypy.reporting.localization.Localization(__file__, 1204, 7), '==', operation_119155, str_119156)
    
    # Testing the type of an if condition (line 1204)
    if_condition_119158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1204, 4), result_eq_119157)
    # Assigning a type to the variable 'if_condition_119158' (line 1204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 4), 'if_condition_119158', if_condition_119158)
    # SSA begins for if statement (line 1204)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1205):
    
    # Assigning a BinOp to a Name (line 1205):
    # Getting the type of 'filter_size' (line 1205)
    filter_size_119159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 15), 'filter_size')
    int_119160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 30), 'int')
    # Applying the binary operator '//' (line 1205)
    result_floordiv_119161 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 15), '//', filter_size_119159, int_119160)
    
    # Assigning a type to the variable 'rank' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 8), 'rank', result_floordiv_119161)
    # SSA branch for the else part of an if statement (line 1204)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'operation' (line 1206)
    operation_119162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 9), 'operation')
    str_119163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1206, 22), 'str', 'percentile')
    # Applying the binary operator '==' (line 1206)
    result_eq_119164 = python_operator(stypy.reporting.localization.Localization(__file__, 1206, 9), '==', operation_119162, str_119163)
    
    # Testing the type of an if condition (line 1206)
    if_condition_119165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1206, 9), result_eq_119164)
    # Assigning a type to the variable 'if_condition_119165' (line 1206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 9), 'if_condition_119165', if_condition_119165)
    # SSA begins for if statement (line 1206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1207):
    
    # Assigning a Name to a Name (line 1207):
    # Getting the type of 'rank' (line 1207)
    rank_119166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 21), 'rank')
    # Assigning a type to the variable 'percentile' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 8), 'percentile', rank_119166)
    
    
    # Getting the type of 'percentile' (line 1208)
    percentile_119167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 11), 'percentile')
    float_119168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1208, 24), 'float')
    # Applying the binary operator '<' (line 1208)
    result_lt_119169 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 11), '<', percentile_119167, float_119168)
    
    # Testing the type of an if condition (line 1208)
    if_condition_119170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1208, 8), result_lt_119169)
    # Assigning a type to the variable 'if_condition_119170' (line 1208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 8), 'if_condition_119170', if_condition_119170)
    # SSA begins for if statement (line 1208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'percentile' (line 1209)
    percentile_119171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 12), 'percentile')
    float_119172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1209, 26), 'float')
    # Applying the binary operator '+=' (line 1209)
    result_iadd_119173 = python_operator(stypy.reporting.localization.Localization(__file__, 1209, 12), '+=', percentile_119171, float_119172)
    # Assigning a type to the variable 'percentile' (line 1209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 12), 'percentile', result_iadd_119173)
    
    # SSA join for if statement (line 1208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'percentile' (line 1210)
    percentile_119174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 11), 'percentile')
    int_119175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 24), 'int')
    # Applying the binary operator '<' (line 1210)
    result_lt_119176 = python_operator(stypy.reporting.localization.Localization(__file__, 1210, 11), '<', percentile_119174, int_119175)
    
    
    # Getting the type of 'percentile' (line 1210)
    percentile_119177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 29), 'percentile')
    int_119178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 42), 'int')
    # Applying the binary operator '>' (line 1210)
    result_gt_119179 = python_operator(stypy.reporting.localization.Localization(__file__, 1210, 29), '>', percentile_119177, int_119178)
    
    # Applying the binary operator 'or' (line 1210)
    result_or_keyword_119180 = python_operator(stypy.reporting.localization.Localization(__file__, 1210, 11), 'or', result_lt_119176, result_gt_119179)
    
    # Testing the type of an if condition (line 1210)
    if_condition_119181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1210, 8), result_or_keyword_119180)
    # Assigning a type to the variable 'if_condition_119181' (line 1210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 8), 'if_condition_119181', if_condition_119181)
    # SSA begins for if statement (line 1210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1211)
    # Processing the call arguments (line 1211)
    str_119183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1211, 31), 'str', 'invalid percentile')
    # Processing the call keyword arguments (line 1211)
    kwargs_119184 = {}
    # Getting the type of 'RuntimeError' (line 1211)
    RuntimeError_119182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 18), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1211)
    RuntimeError_call_result_119185 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 18), RuntimeError_119182, *[str_119183], **kwargs_119184)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1211, 12), RuntimeError_call_result_119185, 'raise parameter', BaseException)
    # SSA join for if statement (line 1210)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'percentile' (line 1212)
    percentile_119186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 11), 'percentile')
    float_119187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 25), 'float')
    # Applying the binary operator '==' (line 1212)
    result_eq_119188 = python_operator(stypy.reporting.localization.Localization(__file__, 1212, 11), '==', percentile_119186, float_119187)
    
    # Testing the type of an if condition (line 1212)
    if_condition_119189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1212, 8), result_eq_119188)
    # Assigning a type to the variable 'if_condition_119189' (line 1212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 8), 'if_condition_119189', if_condition_119189)
    # SSA begins for if statement (line 1212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1213):
    
    # Assigning a BinOp to a Name (line 1213):
    # Getting the type of 'filter_size' (line 1213)
    filter_size_119190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 19), 'filter_size')
    int_119191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 33), 'int')
    # Applying the binary operator '-' (line 1213)
    result_sub_119192 = python_operator(stypy.reporting.localization.Localization(__file__, 1213, 19), '-', filter_size_119190, int_119191)
    
    # Assigning a type to the variable 'rank' (line 1213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 12), 'rank', result_sub_119192)
    # SSA branch for the else part of an if statement (line 1212)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1215):
    
    # Assigning a Call to a Name (line 1215):
    
    # Call to int(...): (line 1215)
    # Processing the call arguments (line 1215)
    
    # Call to float(...): (line 1215)
    # Processing the call arguments (line 1215)
    # Getting the type of 'filter_size' (line 1215)
    filter_size_119195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 29), 'filter_size', False)
    # Processing the call keyword arguments (line 1215)
    kwargs_119196 = {}
    # Getting the type of 'float' (line 1215)
    float_119194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 23), 'float', False)
    # Calling float(args, kwargs) (line 1215)
    float_call_result_119197 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 23), float_119194, *[filter_size_119195], **kwargs_119196)
    
    # Getting the type of 'percentile' (line 1215)
    percentile_119198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 44), 'percentile', False)
    # Applying the binary operator '*' (line 1215)
    result_mul_119199 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 23), '*', float_call_result_119197, percentile_119198)
    
    float_119200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1215, 57), 'float')
    # Applying the binary operator 'div' (line 1215)
    result_div_119201 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 55), 'div', result_mul_119199, float_119200)
    
    # Processing the call keyword arguments (line 1215)
    kwargs_119202 = {}
    # Getting the type of 'int' (line 1215)
    int_119193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 19), 'int', False)
    # Calling int(args, kwargs) (line 1215)
    int_call_result_119203 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 19), int_119193, *[result_div_119201], **kwargs_119202)
    
    # Assigning a type to the variable 'rank' (line 1215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 12), 'rank', int_call_result_119203)
    # SSA join for if statement (line 1212)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1206)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1204)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rank' (line 1216)
    rank_119204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 7), 'rank')
    int_119205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 14), 'int')
    # Applying the binary operator '<' (line 1216)
    result_lt_119206 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 7), '<', rank_119204, int_119205)
    
    # Testing the type of an if condition (line 1216)
    if_condition_119207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1216, 4), result_lt_119206)
    # Assigning a type to the variable 'if_condition_119207' (line 1216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 4), 'if_condition_119207', if_condition_119207)
    # SSA begins for if statement (line 1216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'rank' (line 1217)
    rank_119208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 8), 'rank')
    # Getting the type of 'filter_size' (line 1217)
    filter_size_119209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 16), 'filter_size')
    # Applying the binary operator '+=' (line 1217)
    result_iadd_119210 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 8), '+=', rank_119208, filter_size_119209)
    # Assigning a type to the variable 'rank' (line 1217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 8), 'rank', result_iadd_119210)
    
    # SSA join for if statement (line 1216)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1218)
    rank_119211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 7), 'rank')
    int_119212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, 14), 'int')
    # Applying the binary operator '<' (line 1218)
    result_lt_119213 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 7), '<', rank_119211, int_119212)
    
    
    # Getting the type of 'rank' (line 1218)
    rank_119214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 19), 'rank')
    # Getting the type of 'filter_size' (line 1218)
    filter_size_119215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 27), 'filter_size')
    # Applying the binary operator '>=' (line 1218)
    result_ge_119216 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 19), '>=', rank_119214, filter_size_119215)
    
    # Applying the binary operator 'or' (line 1218)
    result_or_keyword_119217 = python_operator(stypy.reporting.localization.Localization(__file__, 1218, 7), 'or', result_lt_119213, result_ge_119216)
    
    # Testing the type of an if condition (line 1218)
    if_condition_119218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1218, 4), result_or_keyword_119217)
    # Assigning a type to the variable 'if_condition_119218' (line 1218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 4), 'if_condition_119218', if_condition_119218)
    # SSA begins for if statement (line 1218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1219)
    # Processing the call arguments (line 1219)
    str_119220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1219, 27), 'str', 'rank not within filter footprint size')
    # Processing the call keyword arguments (line 1219)
    kwargs_119221 = {}
    # Getting the type of 'RuntimeError' (line 1219)
    RuntimeError_119219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1219)
    RuntimeError_call_result_119222 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 14), RuntimeError_119219, *[str_119220], **kwargs_119221)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1219, 8), RuntimeError_call_result_119222, 'raise parameter', BaseException)
    # SSA join for if statement (line 1218)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rank' (line 1220)
    rank_119223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 7), 'rank')
    int_119224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1220, 15), 'int')
    # Applying the binary operator '==' (line 1220)
    result_eq_119225 = python_operator(stypy.reporting.localization.Localization(__file__, 1220, 7), '==', rank_119223, int_119224)
    
    # Testing the type of an if condition (line 1220)
    if_condition_119226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1220, 4), result_eq_119225)
    # Assigning a type to the variable 'if_condition_119226' (line 1220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1220, 4), 'if_condition_119226', if_condition_119226)
    # SSA begins for if statement (line 1220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to minimum_filter(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'input' (line 1221)
    input_119228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 30), 'input', False)
    # Getting the type of 'None' (line 1221)
    None_119229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 37), 'None', False)
    # Getting the type of 'footprint' (line 1221)
    footprint_119230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 43), 'footprint', False)
    # Getting the type of 'output' (line 1221)
    output_119231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 54), 'output', False)
    # Getting the type of 'mode' (line 1221)
    mode_119232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 62), 'mode', False)
    # Getting the type of 'cval' (line 1221)
    cval_119233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 68), 'cval', False)
    # Getting the type of 'origins' (line 1222)
    origins_119234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 30), 'origins', False)
    # Processing the call keyword arguments (line 1221)
    kwargs_119235 = {}
    # Getting the type of 'minimum_filter' (line 1221)
    minimum_filter_119227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 15), 'minimum_filter', False)
    # Calling minimum_filter(args, kwargs) (line 1221)
    minimum_filter_call_result_119236 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 15), minimum_filter_119227, *[input_119228, None_119229, footprint_119230, output_119231, mode_119232, cval_119233, origins_119234], **kwargs_119235)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 8), 'stypy_return_type', minimum_filter_call_result_119236)
    # SSA branch for the else part of an if statement (line 1220)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'rank' (line 1223)
    rank_119237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 9), 'rank')
    # Getting the type of 'filter_size' (line 1223)
    filter_size_119238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 17), 'filter_size')
    int_119239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 31), 'int')
    # Applying the binary operator '-' (line 1223)
    result_sub_119240 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 17), '-', filter_size_119238, int_119239)
    
    # Applying the binary operator '==' (line 1223)
    result_eq_119241 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 9), '==', rank_119237, result_sub_119240)
    
    # Testing the type of an if condition (line 1223)
    if_condition_119242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1223, 9), result_eq_119241)
    # Assigning a type to the variable 'if_condition_119242' (line 1223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 9), 'if_condition_119242', if_condition_119242)
    # SSA begins for if statement (line 1223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to maximum_filter(...): (line 1224)
    # Processing the call arguments (line 1224)
    # Getting the type of 'input' (line 1224)
    input_119244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 30), 'input', False)
    # Getting the type of 'None' (line 1224)
    None_119245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 37), 'None', False)
    # Getting the type of 'footprint' (line 1224)
    footprint_119246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 43), 'footprint', False)
    # Getting the type of 'output' (line 1224)
    output_119247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 54), 'output', False)
    # Getting the type of 'mode' (line 1224)
    mode_119248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 62), 'mode', False)
    # Getting the type of 'cval' (line 1224)
    cval_119249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 68), 'cval', False)
    # Getting the type of 'origins' (line 1225)
    origins_119250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 30), 'origins', False)
    # Processing the call keyword arguments (line 1224)
    kwargs_119251 = {}
    # Getting the type of 'maximum_filter' (line 1224)
    maximum_filter_119243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 15), 'maximum_filter', False)
    # Calling maximum_filter(args, kwargs) (line 1224)
    maximum_filter_call_result_119252 = invoke(stypy.reporting.localization.Localization(__file__, 1224, 15), maximum_filter_119243, *[input_119244, None_119245, footprint_119246, output_119247, mode_119248, cval_119249, origins_119250], **kwargs_119251)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 8), 'stypy_return_type', maximum_filter_call_result_119252)
    # SSA branch for the else part of an if statement (line 1223)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 1227):
    
    # Assigning a Subscript to a Name (line 1227):
    
    # Obtaining the type of the subscript
    int_119253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 8), 'int')
    
    # Call to _get_output(...): (line 1227)
    # Processing the call arguments (line 1227)
    # Getting the type of 'output' (line 1227)
    output_119256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 55), 'output', False)
    # Getting the type of 'input' (line 1227)
    input_119257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 63), 'input', False)
    # Processing the call keyword arguments (line 1227)
    kwargs_119258 = {}
    # Getting the type of '_ni_support' (line 1227)
    _ni_support_119254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 31), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1227)
    _get_output_119255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 31), _ni_support_119254, '_get_output')
    # Calling _get_output(args, kwargs) (line 1227)
    _get_output_call_result_119259 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 31), _get_output_119255, *[output_119256, input_119257], **kwargs_119258)
    
    # Obtaining the member '__getitem__' of a type (line 1227)
    getitem___119260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 8), _get_output_call_result_119259, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1227)
    subscript_call_result_119261 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 8), getitem___119260, int_119253)
    
    # Assigning a type to the variable 'tuple_var_assignment_117047' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'tuple_var_assignment_117047', subscript_call_result_119261)
    
    # Assigning a Subscript to a Name (line 1227):
    
    # Obtaining the type of the subscript
    int_119262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 8), 'int')
    
    # Call to _get_output(...): (line 1227)
    # Processing the call arguments (line 1227)
    # Getting the type of 'output' (line 1227)
    output_119265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 55), 'output', False)
    # Getting the type of 'input' (line 1227)
    input_119266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 63), 'input', False)
    # Processing the call keyword arguments (line 1227)
    kwargs_119267 = {}
    # Getting the type of '_ni_support' (line 1227)
    _ni_support_119263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 31), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1227)
    _get_output_119264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 31), _ni_support_119263, '_get_output')
    # Calling _get_output(args, kwargs) (line 1227)
    _get_output_call_result_119268 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 31), _get_output_119264, *[output_119265, input_119266], **kwargs_119267)
    
    # Obtaining the member '__getitem__' of a type (line 1227)
    getitem___119269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 8), _get_output_call_result_119268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1227)
    subscript_call_result_119270 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 8), getitem___119269, int_119262)
    
    # Assigning a type to the variable 'tuple_var_assignment_117048' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'tuple_var_assignment_117048', subscript_call_result_119270)
    
    # Assigning a Name to a Name (line 1227):
    # Getting the type of 'tuple_var_assignment_117047' (line 1227)
    tuple_var_assignment_117047_119271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'tuple_var_assignment_117047')
    # Assigning a type to the variable 'output' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'output', tuple_var_assignment_117047_119271)
    
    # Assigning a Name to a Name (line 1227):
    # Getting the type of 'tuple_var_assignment_117048' (line 1227)
    tuple_var_assignment_117048_119272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'tuple_var_assignment_117048')
    # Assigning a type to the variable 'return_value' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 16), 'return_value', tuple_var_assignment_117048_119272)
    
    # Assigning a Call to a Name (line 1228):
    
    # Assigning a Call to a Name (line 1228):
    
    # Call to _extend_mode_to_code(...): (line 1228)
    # Processing the call arguments (line 1228)
    # Getting the type of 'mode' (line 1228)
    mode_119275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 48), 'mode', False)
    # Processing the call keyword arguments (line 1228)
    kwargs_119276 = {}
    # Getting the type of '_ni_support' (line 1228)
    _ni_support_119273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 15), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 1228)
    _extend_mode_to_code_119274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 15), _ni_support_119273, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 1228)
    _extend_mode_to_code_call_result_119277 = invoke(stypy.reporting.localization.Localization(__file__, 1228, 15), _extend_mode_to_code_119274, *[mode_119275], **kwargs_119276)
    
    # Assigning a type to the variable 'mode' (line 1228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 8), 'mode', _extend_mode_to_code_call_result_119277)
    
    # Call to rank_filter(...): (line 1229)
    # Processing the call arguments (line 1229)
    # Getting the type of 'input' (line 1229)
    input_119280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 30), 'input', False)
    # Getting the type of 'rank' (line 1229)
    rank_119281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 37), 'rank', False)
    # Getting the type of 'footprint' (line 1229)
    footprint_119282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 43), 'footprint', False)
    # Getting the type of 'output' (line 1229)
    output_119283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 54), 'output', False)
    # Getting the type of 'mode' (line 1229)
    mode_119284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 62), 'mode', False)
    # Getting the type of 'cval' (line 1229)
    cval_119285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 68), 'cval', False)
    # Getting the type of 'origins' (line 1230)
    origins_119286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 30), 'origins', False)
    # Processing the call keyword arguments (line 1229)
    kwargs_119287 = {}
    # Getting the type of '_nd_image' (line 1229)
    _nd_image_119278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 8), '_nd_image', False)
    # Obtaining the member 'rank_filter' of a type (line 1229)
    rank_filter_119279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1229, 8), _nd_image_119278, 'rank_filter')
    # Calling rank_filter(args, kwargs) (line 1229)
    rank_filter_call_result_119288 = invoke(stypy.reporting.localization.Localization(__file__, 1229, 8), rank_filter_119279, *[input_119280, rank_119281, footprint_119282, output_119283, mode_119284, cval_119285, origins_119286], **kwargs_119287)
    
    # Getting the type of 'return_value' (line 1231)
    return_value_119289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 15), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'stypy_return_type', return_value_119289)
    # SSA join for if statement (line 1223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1220)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_rank_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_rank_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1181)
    stypy_return_type_119290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119290)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_rank_filter'
    return stypy_return_type_119290

# Assigning a type to the variable '_rank_filter' (line 1181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1181, 0), '_rank_filter', _rank_filter)

@norecursion
def rank_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1235)
    None_119291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 34), 'None')
    # Getting the type of 'None' (line 1235)
    None_119292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 50), 'None')
    # Getting the type of 'None' (line 1235)
    None_119293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 63), 'None')
    str_119294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 21), 'str', 'reflect')
    float_119295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 37), 'float')
    int_119296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 49), 'int')
    defaults = [None_119291, None_119292, None_119293, str_119294, float_119295, int_119296]
    # Create a new context for function 'rank_filter'
    module_type_store = module_type_store.open_function_context('rank_filter', 1234, 0, False)
    
    # Passed parameters checking function
    rank_filter.stypy_localization = localization
    rank_filter.stypy_type_of_self = None
    rank_filter.stypy_type_store = module_type_store
    rank_filter.stypy_function_name = 'rank_filter'
    rank_filter.stypy_param_names_list = ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin']
    rank_filter.stypy_varargs_param_name = None
    rank_filter.stypy_kwargs_param_name = None
    rank_filter.stypy_call_defaults = defaults
    rank_filter.stypy_call_varargs = varargs
    rank_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rank_filter', ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rank_filter', localization, ['input', 'rank', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rank_filter(...)' code ##################

    str_119297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, (-1)), 'str', 'Calculate a multi-dimensional rank filter.\n\n    Parameters\n    ----------\n    %(input)s\n    rank : int\n        The rank parameter may be less then zero, i.e., rank = -1\n        indicates the largest element.\n    %(size_foot)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    rank_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.rank_filter(ascent, rank=42, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Call to _rank_filter(...): (line 1270)
    # Processing the call arguments (line 1270)
    # Getting the type of 'input' (line 1270)
    input_119299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 24), 'input', False)
    # Getting the type of 'rank' (line 1270)
    rank_119300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 31), 'rank', False)
    # Getting the type of 'size' (line 1270)
    size_119301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 37), 'size', False)
    # Getting the type of 'footprint' (line 1270)
    footprint_119302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 43), 'footprint', False)
    # Getting the type of 'output' (line 1270)
    output_119303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 54), 'output', False)
    # Getting the type of 'mode' (line 1270)
    mode_119304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 62), 'mode', False)
    # Getting the type of 'cval' (line 1270)
    cval_119305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 68), 'cval', False)
    # Getting the type of 'origin' (line 1271)
    origin_119306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 24), 'origin', False)
    str_119307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1271, 32), 'str', 'rank')
    # Processing the call keyword arguments (line 1270)
    kwargs_119308 = {}
    # Getting the type of '_rank_filter' (line 1270)
    _rank_filter_119298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 11), '_rank_filter', False)
    # Calling _rank_filter(args, kwargs) (line 1270)
    _rank_filter_call_result_119309 = invoke(stypy.reporting.localization.Localization(__file__, 1270, 11), _rank_filter_119298, *[input_119299, rank_119300, size_119301, footprint_119302, output_119303, mode_119304, cval_119305, origin_119306, str_119307], **kwargs_119308)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1270, 4), 'stypy_return_type', _rank_filter_call_result_119309)
    
    # ################# End of 'rank_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rank_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1234)
    stypy_return_type_119310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119310)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rank_filter'
    return stypy_return_type_119310

# Assigning a type to the variable 'rank_filter' (line 1234)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 0), 'rank_filter', rank_filter)

@norecursion
def median_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1275)
    None_119311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1275, 30), 'None')
    # Getting the type of 'None' (line 1275)
    None_119312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1275, 46), 'None')
    # Getting the type of 'None' (line 1275)
    None_119313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1275, 59), 'None')
    str_119314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1276, 23), 'str', 'reflect')
    float_119315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1276, 39), 'float')
    int_119316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1276, 51), 'int')
    defaults = [None_119311, None_119312, None_119313, str_119314, float_119315, int_119316]
    # Create a new context for function 'median_filter'
    module_type_store = module_type_store.open_function_context('median_filter', 1274, 0, False)
    
    # Passed parameters checking function
    median_filter.stypy_localization = localization
    median_filter.stypy_type_of_self = None
    median_filter.stypy_type_store = module_type_store
    median_filter.stypy_function_name = 'median_filter'
    median_filter.stypy_param_names_list = ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin']
    median_filter.stypy_varargs_param_name = None
    median_filter.stypy_kwargs_param_name = None
    median_filter.stypy_call_defaults = defaults
    median_filter.stypy_call_varargs = varargs
    median_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'median_filter', ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'median_filter', localization, ['input', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'median_filter(...)' code ##################

    str_119317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1307, (-1)), 'str', '\n    Calculate a multidimensional median filter.\n\n    Parameters\n    ----------\n    %(input)s\n    %(size_foot)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    median_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.median_filter(ascent, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Call to _rank_filter(...): (line 1308)
    # Processing the call arguments (line 1308)
    # Getting the type of 'input' (line 1308)
    input_119319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 24), 'input', False)
    int_119320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1308, 31), 'int')
    # Getting the type of 'size' (line 1308)
    size_119321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 34), 'size', False)
    # Getting the type of 'footprint' (line 1308)
    footprint_119322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 40), 'footprint', False)
    # Getting the type of 'output' (line 1308)
    output_119323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 51), 'output', False)
    # Getting the type of 'mode' (line 1308)
    mode_119324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 59), 'mode', False)
    # Getting the type of 'cval' (line 1308)
    cval_119325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 65), 'cval', False)
    # Getting the type of 'origin' (line 1309)
    origin_119326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 24), 'origin', False)
    str_119327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1309, 32), 'str', 'median')
    # Processing the call keyword arguments (line 1308)
    kwargs_119328 = {}
    # Getting the type of '_rank_filter' (line 1308)
    _rank_filter_119318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1308, 11), '_rank_filter', False)
    # Calling _rank_filter(args, kwargs) (line 1308)
    _rank_filter_call_result_119329 = invoke(stypy.reporting.localization.Localization(__file__, 1308, 11), _rank_filter_119318, *[input_119319, int_119320, size_119321, footprint_119322, output_119323, mode_119324, cval_119325, origin_119326, str_119327], **kwargs_119328)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1308, 4), 'stypy_return_type', _rank_filter_call_result_119329)
    
    # ################# End of 'median_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'median_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1274)
    stypy_return_type_119330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1274, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119330)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'median_filter'
    return stypy_return_type_119330

# Assigning a type to the variable 'median_filter' (line 1274)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1274, 0), 'median_filter', median_filter)

@norecursion
def percentile_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1313)
    None_119331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 46), 'None')
    # Getting the type of 'None' (line 1313)
    None_119332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1313, 62), 'None')
    # Getting the type of 'None' (line 1314)
    None_119333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1314, 29), 'None')
    str_119334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, 40), 'str', 'reflect')
    float_119335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, 56), 'float')
    int_119336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, 68), 'int')
    defaults = [None_119331, None_119332, None_119333, str_119334, float_119335, int_119336]
    # Create a new context for function 'percentile_filter'
    module_type_store = module_type_store.open_function_context('percentile_filter', 1312, 0, False)
    
    # Passed parameters checking function
    percentile_filter.stypy_localization = localization
    percentile_filter.stypy_type_of_self = None
    percentile_filter.stypy_type_store = module_type_store
    percentile_filter.stypy_function_name = 'percentile_filter'
    percentile_filter.stypy_param_names_list = ['input', 'percentile', 'size', 'footprint', 'output', 'mode', 'cval', 'origin']
    percentile_filter.stypy_varargs_param_name = None
    percentile_filter.stypy_kwargs_param_name = None
    percentile_filter.stypy_call_defaults = defaults
    percentile_filter.stypy_call_varargs = varargs
    percentile_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'percentile_filter', ['input', 'percentile', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'percentile_filter', localization, ['input', 'percentile', 'size', 'footprint', 'output', 'mode', 'cval', 'origin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'percentile_filter(...)' code ##################

    str_119337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1347, (-1)), 'str', 'Calculate a multi-dimensional percentile filter.\n\n    Parameters\n    ----------\n    %(input)s\n    percentile : scalar\n        The percentile parameter may be less then zero, i.e.,\n        percentile = -20 equals percentile = 80\n    %(size_foot)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n\n    Returns\n    -------\n    percentile_filter : ndarray\n        Filtered array. Has the same shape as `input`.\n\n    Examples\n    --------\n    >>> from scipy import ndimage, misc\n    >>> import matplotlib.pyplot as plt\n    >>> fig = plt.figure()\n    >>> plt.gray()  # show the filtered result in grayscale\n    >>> ax1 = fig.add_subplot(121)  # left side\n    >>> ax2 = fig.add_subplot(122)  # right side\n    >>> ascent = misc.ascent()\n    >>> result = ndimage.percentile_filter(ascent, percentile=20, size=20)\n    >>> ax1.imshow(ascent)\n    >>> ax2.imshow(result)\n    >>> plt.show()\n    ')
    
    # Call to _rank_filter(...): (line 1348)
    # Processing the call arguments (line 1348)
    # Getting the type of 'input' (line 1348)
    input_119339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 24), 'input', False)
    # Getting the type of 'percentile' (line 1348)
    percentile_119340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 31), 'percentile', False)
    # Getting the type of 'size' (line 1348)
    size_119341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 43), 'size', False)
    # Getting the type of 'footprint' (line 1348)
    footprint_119342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 49), 'footprint', False)
    # Getting the type of 'output' (line 1348)
    output_119343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 60), 'output', False)
    # Getting the type of 'mode' (line 1348)
    mode_119344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 68), 'mode', False)
    # Getting the type of 'cval' (line 1349)
    cval_119345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 24), 'cval', False)
    # Getting the type of 'origin' (line 1349)
    origin_119346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1349, 30), 'origin', False)
    str_119347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1349, 38), 'str', 'percentile')
    # Processing the call keyword arguments (line 1348)
    kwargs_119348 = {}
    # Getting the type of '_rank_filter' (line 1348)
    _rank_filter_119338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1348, 11), '_rank_filter', False)
    # Calling _rank_filter(args, kwargs) (line 1348)
    _rank_filter_call_result_119349 = invoke(stypy.reporting.localization.Localization(__file__, 1348, 11), _rank_filter_119338, *[input_119339, percentile_119340, size_119341, footprint_119342, output_119343, mode_119344, cval_119345, origin_119346, str_119347], **kwargs_119348)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1348, 4), 'stypy_return_type', _rank_filter_call_result_119349)
    
    # ################# End of 'percentile_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'percentile_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1312)
    stypy_return_type_119350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1312, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119350)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'percentile_filter'
    return stypy_return_type_119350

# Assigning a type to the variable 'percentile_filter' (line 1312)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1312, 0), 'percentile_filter', percentile_filter)

@norecursion
def generic_filter1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_119351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 56), 'int')
    # Getting the type of 'None' (line 1354)
    None_119352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 28), 'None')
    str_119353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 39), 'str', 'reflect')
    float_119354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 55), 'float')
    int_119355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 67), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1355)
    tuple_119356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1355)
    
    # Getting the type of 'None' (line 1355)
    None_119357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 56), 'None')
    defaults = [int_119351, None_119352, str_119353, float_119354, int_119355, tuple_119356, None_119357]
    # Create a new context for function 'generic_filter1d'
    module_type_store = module_type_store.open_function_context('generic_filter1d', 1352, 0, False)
    
    # Passed parameters checking function
    generic_filter1d.stypy_localization = localization
    generic_filter1d.stypy_type_of_self = None
    generic_filter1d.stypy_type_store = module_type_store
    generic_filter1d.stypy_function_name = 'generic_filter1d'
    generic_filter1d.stypy_param_names_list = ['input', 'function', 'filter_size', 'axis', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords']
    generic_filter1d.stypy_varargs_param_name = None
    generic_filter1d.stypy_kwargs_param_name = None
    generic_filter1d.stypy_call_defaults = defaults
    generic_filter1d.stypy_call_varargs = varargs
    generic_filter1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generic_filter1d', ['input', 'function', 'filter_size', 'axis', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generic_filter1d', localization, ['input', 'function', 'filter_size', 'axis', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generic_filter1d(...)' code ##################

    str_119358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1415, (-1)), 'str', 'Calculate a one-dimensional filter along the given axis.\n\n    `generic_filter1d` iterates over the lines of the array, calling the\n    given function at each line. The arguments of the line are the\n    input line, and the output line. The input and output lines are 1D\n    double arrays.  The input line is extended appropriately according\n    to the filter size and origin. The output line must be modified\n    in-place with the result.\n\n    Parameters\n    ----------\n    %(input)s\n    function : {callable, scipy.LowLevelCallable}\n        Function to apply along given axis.\n    filter_size : scalar\n        Length of the filter.\n    %(axis)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n    %(extra_arguments)s\n    %(extra_keywords)s\n\n    Notes\n    -----\n    This function also accepts low-level callback functions with one of\n    the following signatures and wrapped in `scipy.LowLevelCallable`:\n\n    .. code:: c\n\n       int function(double *input_line, npy_intp input_length,\n                    double *output_line, npy_intp output_length,\n                    void *user_data)\n       int function(double *input_line, intptr_t input_length,\n                    double *output_line, intptr_t output_length,\n                    void *user_data)\n\n    The calling function iterates over the lines of the input and output\n    arrays, calling the callback function at each line. The current line\n    is extended according to the border conditions set by the calling\n    function, and the result is copied into the array that is passed\n    through ``input_line``. The length of the input line (after extension)\n    is passed through ``input_length``. The callback function should apply\n    the filter and store the result in the array passed through\n    ``output_line``. The length of the output line is passed through\n    ``output_length``. ``user_data`` is the data pointer provided\n    to `scipy.LowLevelCallable` as-is.\n\n    The callback function must return an integer error status that is zero\n    if something went wrong and one otherwise. If an error occurs, you should\n    normally set the python error status with an informative message\n    before returning, otherwise a default error message is set by the\n    calling function.\n\n    In addition, some other low-level function pointer specifications\n    are accepted, but these are for backward compatibility only and should\n    not be used in new code.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 1416)
    # Getting the type of 'extra_keywords' (line 1416)
    extra_keywords_119359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 7), 'extra_keywords')
    # Getting the type of 'None' (line 1416)
    None_119360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 25), 'None')
    
    (may_be_119361, more_types_in_union_119362) = may_be_none(extra_keywords_119359, None_119360)

    if may_be_119361:

        if more_types_in_union_119362:
            # Runtime conditional SSA (line 1416)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 1417):
        
        # Assigning a Dict to a Name (line 1417):
        
        # Obtaining an instance of the builtin type 'dict' (line 1417)
        dict_119363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1417, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1417)
        
        # Assigning a type to the variable 'extra_keywords' (line 1417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1417, 8), 'extra_keywords', dict_119363)

        if more_types_in_union_119362:
            # SSA join for if statement (line 1416)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1418):
    
    # Assigning a Call to a Name (line 1418):
    
    # Call to asarray(...): (line 1418)
    # Processing the call arguments (line 1418)
    # Getting the type of 'input' (line 1418)
    input_119366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 26), 'input', False)
    # Processing the call keyword arguments (line 1418)
    kwargs_119367 = {}
    # Getting the type of 'numpy' (line 1418)
    numpy_119364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1418)
    asarray_119365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1418, 12), numpy_119364, 'asarray')
    # Calling asarray(args, kwargs) (line 1418)
    asarray_call_result_119368 = invoke(stypy.reporting.localization.Localization(__file__, 1418, 12), asarray_119365, *[input_119366], **kwargs_119367)
    
    # Assigning a type to the variable 'input' (line 1418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1418, 4), 'input', asarray_call_result_119368)
    
    
    # Call to iscomplexobj(...): (line 1419)
    # Processing the call arguments (line 1419)
    # Getting the type of 'input' (line 1419)
    input_119371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1419, 26), 'input', False)
    # Processing the call keyword arguments (line 1419)
    kwargs_119372 = {}
    # Getting the type of 'numpy' (line 1419)
    numpy_119369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1419, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1419)
    iscomplexobj_119370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1419, 7), numpy_119369, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1419)
    iscomplexobj_call_result_119373 = invoke(stypy.reporting.localization.Localization(__file__, 1419, 7), iscomplexobj_119370, *[input_119371], **kwargs_119372)
    
    # Testing the type of an if condition (line 1419)
    if_condition_119374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1419, 4), iscomplexobj_call_result_119373)
    # Assigning a type to the variable 'if_condition_119374' (line 1419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1419, 4), 'if_condition_119374', if_condition_119374)
    # SSA begins for if statement (line 1419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1420)
    # Processing the call arguments (line 1420)
    str_119376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1420, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 1420)
    kwargs_119377 = {}
    # Getting the type of 'TypeError' (line 1420)
    TypeError_119375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1420, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1420)
    TypeError_call_result_119378 = invoke(stypy.reporting.localization.Localization(__file__, 1420, 14), TypeError_119375, *[str_119376], **kwargs_119377)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1420, 8), TypeError_call_result_119378, 'raise parameter', BaseException)
    # SSA join for if statement (line 1419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1421):
    
    # Assigning a Subscript to a Name (line 1421):
    
    # Obtaining the type of the subscript
    int_119379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1421, 4), 'int')
    
    # Call to _get_output(...): (line 1421)
    # Processing the call arguments (line 1421)
    # Getting the type of 'output' (line 1421)
    output_119382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 51), 'output', False)
    # Getting the type of 'input' (line 1421)
    input_119383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 59), 'input', False)
    # Processing the call keyword arguments (line 1421)
    kwargs_119384 = {}
    # Getting the type of '_ni_support' (line 1421)
    _ni_support_119380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1421)
    _get_output_119381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 27), _ni_support_119380, '_get_output')
    # Calling _get_output(args, kwargs) (line 1421)
    _get_output_call_result_119385 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 27), _get_output_119381, *[output_119382, input_119383], **kwargs_119384)
    
    # Obtaining the member '__getitem__' of a type (line 1421)
    getitem___119386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 4), _get_output_call_result_119385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1421)
    subscript_call_result_119387 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 4), getitem___119386, int_119379)
    
    # Assigning a type to the variable 'tuple_var_assignment_117049' (line 1421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'tuple_var_assignment_117049', subscript_call_result_119387)
    
    # Assigning a Subscript to a Name (line 1421):
    
    # Obtaining the type of the subscript
    int_119388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1421, 4), 'int')
    
    # Call to _get_output(...): (line 1421)
    # Processing the call arguments (line 1421)
    # Getting the type of 'output' (line 1421)
    output_119391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 51), 'output', False)
    # Getting the type of 'input' (line 1421)
    input_119392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 59), 'input', False)
    # Processing the call keyword arguments (line 1421)
    kwargs_119393 = {}
    # Getting the type of '_ni_support' (line 1421)
    _ni_support_119389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1421)
    _get_output_119390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 27), _ni_support_119389, '_get_output')
    # Calling _get_output(args, kwargs) (line 1421)
    _get_output_call_result_119394 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 27), _get_output_119390, *[output_119391, input_119392], **kwargs_119393)
    
    # Obtaining the member '__getitem__' of a type (line 1421)
    getitem___119395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1421, 4), _get_output_call_result_119394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1421)
    subscript_call_result_119396 = invoke(stypy.reporting.localization.Localization(__file__, 1421, 4), getitem___119395, int_119388)
    
    # Assigning a type to the variable 'tuple_var_assignment_117050' (line 1421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'tuple_var_assignment_117050', subscript_call_result_119396)
    
    # Assigning a Name to a Name (line 1421):
    # Getting the type of 'tuple_var_assignment_117049' (line 1421)
    tuple_var_assignment_117049_119397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'tuple_var_assignment_117049')
    # Assigning a type to the variable 'output' (line 1421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'output', tuple_var_assignment_117049_119397)
    
    # Assigning a Name to a Name (line 1421):
    # Getting the type of 'tuple_var_assignment_117050' (line 1421)
    tuple_var_assignment_117050_119398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 4), 'tuple_var_assignment_117050')
    # Assigning a type to the variable 'return_value' (line 1421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 12), 'return_value', tuple_var_assignment_117050_119398)
    
    
    # Getting the type of 'filter_size' (line 1422)
    filter_size_119399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1422, 7), 'filter_size')
    int_119400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1422, 21), 'int')
    # Applying the binary operator '<' (line 1422)
    result_lt_119401 = python_operator(stypy.reporting.localization.Localization(__file__, 1422, 7), '<', filter_size_119399, int_119400)
    
    # Testing the type of an if condition (line 1422)
    if_condition_119402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1422, 4), result_lt_119401)
    # Assigning a type to the variable 'if_condition_119402' (line 1422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1422, 4), 'if_condition_119402', if_condition_119402)
    # SSA begins for if statement (line 1422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1423)
    # Processing the call arguments (line 1423)
    str_119404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1423, 27), 'str', 'invalid filter size')
    # Processing the call keyword arguments (line 1423)
    kwargs_119405 = {}
    # Getting the type of 'RuntimeError' (line 1423)
    RuntimeError_119403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1423, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1423)
    RuntimeError_call_result_119406 = invoke(stypy.reporting.localization.Localization(__file__, 1423, 14), RuntimeError_119403, *[str_119404], **kwargs_119405)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1423, 8), RuntimeError_call_result_119406, 'raise parameter', BaseException)
    # SSA join for if statement (line 1422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1424):
    
    # Assigning a Call to a Name (line 1424):
    
    # Call to _check_axis(...): (line 1424)
    # Processing the call arguments (line 1424)
    # Getting the type of 'axis' (line 1424)
    axis_119409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 35), 'axis', False)
    # Getting the type of 'input' (line 1424)
    input_119410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 41), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1424)
    ndim_119411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1424, 41), input_119410, 'ndim')
    # Processing the call keyword arguments (line 1424)
    kwargs_119412 = {}
    # Getting the type of '_ni_support' (line 1424)
    _ni_support_119407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1424, 11), '_ni_support', False)
    # Obtaining the member '_check_axis' of a type (line 1424)
    _check_axis_119408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1424, 11), _ni_support_119407, '_check_axis')
    # Calling _check_axis(args, kwargs) (line 1424)
    _check_axis_call_result_119413 = invoke(stypy.reporting.localization.Localization(__file__, 1424, 11), _check_axis_119408, *[axis_119409, ndim_119411], **kwargs_119412)
    
    # Assigning a type to the variable 'axis' (line 1424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1424, 4), 'axis', _check_axis_call_result_119413)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'filter_size' (line 1425)
    filter_size_119414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 8), 'filter_size')
    int_119415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, 23), 'int')
    # Applying the binary operator '//' (line 1425)
    result_floordiv_119416 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 8), '//', filter_size_119414, int_119415)
    
    # Getting the type of 'origin' (line 1425)
    origin_119417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 27), 'origin')
    # Applying the binary operator '+' (line 1425)
    result_add_119418 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 8), '+', result_floordiv_119416, origin_119417)
    
    int_119419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, 36), 'int')
    # Applying the binary operator '<' (line 1425)
    result_lt_119420 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 8), '<', result_add_119418, int_119419)
    
    
    # Getting the type of 'filter_size' (line 1425)
    filter_size_119421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 43), 'filter_size')
    int_119422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1425, 58), 'int')
    # Applying the binary operator '//' (line 1425)
    result_floordiv_119423 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 43), '//', filter_size_119421, int_119422)
    
    # Getting the type of 'origin' (line 1425)
    origin_119424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1425, 62), 'origin')
    # Applying the binary operator '+' (line 1425)
    result_add_119425 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 43), '+', result_floordiv_119423, origin_119424)
    
    # Getting the type of 'filter_size' (line 1426)
    filter_size_119426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1426, 43), 'filter_size')
    # Applying the binary operator '>=' (line 1425)
    result_ge_119427 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 43), '>=', result_add_119425, filter_size_119426)
    
    # Applying the binary operator 'or' (line 1425)
    result_or_keyword_119428 = python_operator(stypy.reporting.localization.Localization(__file__, 1425, 7), 'or', result_lt_119420, result_ge_119427)
    
    # Testing the type of an if condition (line 1425)
    if_condition_119429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1425, 4), result_or_keyword_119428)
    # Assigning a type to the variable 'if_condition_119429' (line 1425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1425, 4), 'if_condition_119429', if_condition_119429)
    # SSA begins for if statement (line 1425)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1427)
    # Processing the call arguments (line 1427)
    str_119431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1427, 25), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 1427)
    kwargs_119432 = {}
    # Getting the type of 'ValueError' (line 1427)
    ValueError_119430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1427, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1427)
    ValueError_call_result_119433 = invoke(stypy.reporting.localization.Localization(__file__, 1427, 14), ValueError_119430, *[str_119431], **kwargs_119432)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1427, 8), ValueError_call_result_119433, 'raise parameter', BaseException)
    # SSA join for if statement (line 1425)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1428):
    
    # Assigning a Call to a Name (line 1428):
    
    # Call to _extend_mode_to_code(...): (line 1428)
    # Processing the call arguments (line 1428)
    # Getting the type of 'mode' (line 1428)
    mode_119436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 44), 'mode', False)
    # Processing the call keyword arguments (line 1428)
    kwargs_119437 = {}
    # Getting the type of '_ni_support' (line 1428)
    _ni_support_119434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1428, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 1428)
    _extend_mode_to_code_119435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1428, 11), _ni_support_119434, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 1428)
    _extend_mode_to_code_call_result_119438 = invoke(stypy.reporting.localization.Localization(__file__, 1428, 11), _extend_mode_to_code_119435, *[mode_119436], **kwargs_119437)
    
    # Assigning a type to the variable 'mode' (line 1428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1428, 4), 'mode', _extend_mode_to_code_call_result_119438)
    
    # Call to generic_filter1d(...): (line 1429)
    # Processing the call arguments (line 1429)
    # Getting the type of 'input' (line 1429)
    input_119441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 31), 'input', False)
    # Getting the type of 'function' (line 1429)
    function_119442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 38), 'function', False)
    # Getting the type of 'filter_size' (line 1429)
    filter_size_119443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 48), 'filter_size', False)
    # Getting the type of 'axis' (line 1429)
    axis_119444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 61), 'axis', False)
    # Getting the type of 'output' (line 1429)
    output_119445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 67), 'output', False)
    # Getting the type of 'mode' (line 1430)
    mode_119446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1430, 31), 'mode', False)
    # Getting the type of 'cval' (line 1430)
    cval_119447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1430, 37), 'cval', False)
    # Getting the type of 'origin' (line 1430)
    origin_119448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1430, 43), 'origin', False)
    # Getting the type of 'extra_arguments' (line 1430)
    extra_arguments_119449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1430, 51), 'extra_arguments', False)
    # Getting the type of 'extra_keywords' (line 1431)
    extra_keywords_119450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1431, 31), 'extra_keywords', False)
    # Processing the call keyword arguments (line 1429)
    kwargs_119451 = {}
    # Getting the type of '_nd_image' (line 1429)
    _nd_image_119439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1429, 4), '_nd_image', False)
    # Obtaining the member 'generic_filter1d' of a type (line 1429)
    generic_filter1d_119440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1429, 4), _nd_image_119439, 'generic_filter1d')
    # Calling generic_filter1d(args, kwargs) (line 1429)
    generic_filter1d_call_result_119452 = invoke(stypy.reporting.localization.Localization(__file__, 1429, 4), generic_filter1d_119440, *[input_119441, function_119442, filter_size_119443, axis_119444, output_119445, mode_119446, cval_119447, origin_119448, extra_arguments_119449, extra_keywords_119450], **kwargs_119451)
    
    # Getting the type of 'return_value' (line 1432)
    return_value_119453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1432, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1432, 4), 'stypy_return_type', return_value_119453)
    
    # ################# End of 'generic_filter1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generic_filter1d' in the type store
    # Getting the type of 'stypy_return_type' (line 1352)
    stypy_return_type_119454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generic_filter1d'
    return stypy_return_type_119454

# Assigning a type to the variable 'generic_filter1d' (line 1352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 0), 'generic_filter1d', generic_filter1d)

@norecursion
def generic_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1436)
    None_119455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1436, 41), 'None')
    # Getting the type of 'None' (line 1436)
    None_119456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1436, 57), 'None')
    # Getting the type of 'None' (line 1437)
    None_119457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1437, 26), 'None')
    str_119458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1437, 37), 'str', 'reflect')
    float_119459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1437, 53), 'float')
    int_119460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1437, 65), 'int')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1438)
    tuple_119461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1438, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1438)
    
    # Getting the type of 'None' (line 1438)
    None_119462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1438, 54), 'None')
    defaults = [None_119455, None_119456, None_119457, str_119458, float_119459, int_119460, tuple_119461, None_119462]
    # Create a new context for function 'generic_filter'
    module_type_store = module_type_store.open_function_context('generic_filter', 1435, 0, False)
    
    # Passed parameters checking function
    generic_filter.stypy_localization = localization
    generic_filter.stypy_type_of_self = None
    generic_filter.stypy_type_store = module_type_store
    generic_filter.stypy_function_name = 'generic_filter'
    generic_filter.stypy_param_names_list = ['input', 'function', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords']
    generic_filter.stypy_varargs_param_name = None
    generic_filter.stypy_kwargs_param_name = None
    generic_filter.stypy_call_defaults = defaults
    generic_filter.stypy_call_varargs = varargs
    generic_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generic_filter', ['input', 'function', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generic_filter', localization, ['input', 'function', 'size', 'footprint', 'output', 'mode', 'cval', 'origin', 'extra_arguments', 'extra_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generic_filter(...)' code ##################

    str_119463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1488, (-1)), 'str', 'Calculate a multi-dimensional filter using the given function.\n\n    At each element the provided function is called. The input values\n    within the filter footprint at that element are passed to the function\n    as a 1D array of double values.\n\n    Parameters\n    ----------\n    %(input)s\n    function : {callable, scipy.LowLevelCallable}\n        Function to apply at each element.\n    %(size_foot)s\n    %(output)s\n    %(mode)s\n    %(cval)s\n    %(origin)s\n    %(extra_arguments)s\n    %(extra_keywords)s\n\n    Notes\n    -----\n    This function also accepts low-level callback functions with one of\n    the following signatures and wrapped in `scipy.LowLevelCallable`:\n\n    .. code:: c\n\n       int callback(double *buffer, npy_intp filter_size,\n                    double *return_value, void *user_data)\n       int callback(double *buffer, intptr_t filter_size,\n                    double *return_value, void *user_data)\n\n    The calling function iterates over the elements of the input and\n    output arrays, calling the callback function at each element. The\n    elements within the footprint of the filter at the current element are\n    passed through the ``buffer`` parameter, and the number of elements\n    within the footprint through ``filter_size``. The calculated value is\n    returned in ``return_value``. ``user_data`` is the data pointer provided\n    to `scipy.LowLevelCallable` as-is.\n\n    The callback function must return an integer error status that is zero\n    if something went wrong and one otherwise. If an error occurs, you should\n    normally set the python error status with an informative message\n    before returning, otherwise a default error message is set by the\n    calling function.\n\n    In addition, some other low-level function pointer specifications\n    are accepted, but these are for backward compatibility only and should\n    not be used in new code.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 1489)
    # Getting the type of 'extra_keywords' (line 1489)
    extra_keywords_119464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 7), 'extra_keywords')
    # Getting the type of 'None' (line 1489)
    None_119465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 25), 'None')
    
    (may_be_119466, more_types_in_union_119467) = may_be_none(extra_keywords_119464, None_119465)

    if may_be_119466:

        if more_types_in_union_119467:
            # Runtime conditional SSA (line 1489)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 1490):
        
        # Assigning a Dict to a Name (line 1490):
        
        # Obtaining an instance of the builtin type 'dict' (line 1490)
        dict_119468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 25), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 1490)
        
        # Assigning a type to the variable 'extra_keywords' (line 1490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 8), 'extra_keywords', dict_119468)

        if more_types_in_union_119467:
            # SSA join for if statement (line 1489)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1491):
    
    # Assigning a Call to a Name (line 1491):
    
    # Call to asarray(...): (line 1491)
    # Processing the call arguments (line 1491)
    # Getting the type of 'input' (line 1491)
    input_119471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 26), 'input', False)
    # Processing the call keyword arguments (line 1491)
    kwargs_119472 = {}
    # Getting the type of 'numpy' (line 1491)
    numpy_119469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 12), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 1491)
    asarray_119470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1491, 12), numpy_119469, 'asarray')
    # Calling asarray(args, kwargs) (line 1491)
    asarray_call_result_119473 = invoke(stypy.reporting.localization.Localization(__file__, 1491, 12), asarray_119470, *[input_119471], **kwargs_119472)
    
    # Assigning a type to the variable 'input' (line 1491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1491, 4), 'input', asarray_call_result_119473)
    
    
    # Call to iscomplexobj(...): (line 1492)
    # Processing the call arguments (line 1492)
    # Getting the type of 'input' (line 1492)
    input_119476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 26), 'input', False)
    # Processing the call keyword arguments (line 1492)
    kwargs_119477 = {}
    # Getting the type of 'numpy' (line 1492)
    numpy_119474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 7), 'numpy', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1492)
    iscomplexobj_119475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1492, 7), numpy_119474, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1492)
    iscomplexobj_call_result_119478 = invoke(stypy.reporting.localization.Localization(__file__, 1492, 7), iscomplexobj_119475, *[input_119476], **kwargs_119477)
    
    # Testing the type of an if condition (line 1492)
    if_condition_119479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1492, 4), iscomplexobj_call_result_119478)
    # Assigning a type to the variable 'if_condition_119479' (line 1492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1492, 4), 'if_condition_119479', if_condition_119479)
    # SSA begins for if statement (line 1492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1493)
    # Processing the call arguments (line 1493)
    str_119481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 24), 'str', 'Complex type not supported')
    # Processing the call keyword arguments (line 1493)
    kwargs_119482 = {}
    # Getting the type of 'TypeError' (line 1493)
    TypeError_119480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1493)
    TypeError_call_result_119483 = invoke(stypy.reporting.localization.Localization(__file__, 1493, 14), TypeError_119480, *[str_119481], **kwargs_119482)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1493, 8), TypeError_call_result_119483, 'raise parameter', BaseException)
    # SSA join for if statement (line 1492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1494):
    
    # Assigning a Call to a Name (line 1494):
    
    # Call to _normalize_sequence(...): (line 1494)
    # Processing the call arguments (line 1494)
    # Getting the type of 'origin' (line 1494)
    origin_119486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 46), 'origin', False)
    # Getting the type of 'input' (line 1494)
    input_119487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 54), 'input', False)
    # Obtaining the member 'ndim' of a type (line 1494)
    ndim_119488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 54), input_119487, 'ndim')
    # Processing the call keyword arguments (line 1494)
    kwargs_119489 = {}
    # Getting the type of '_ni_support' (line 1494)
    _ni_support_119484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 14), '_ni_support', False)
    # Obtaining the member '_normalize_sequence' of a type (line 1494)
    _normalize_sequence_119485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 14), _ni_support_119484, '_normalize_sequence')
    # Calling _normalize_sequence(args, kwargs) (line 1494)
    _normalize_sequence_call_result_119490 = invoke(stypy.reporting.localization.Localization(__file__, 1494, 14), _normalize_sequence_119485, *[origin_119486, ndim_119488], **kwargs_119489)
    
    # Assigning a type to the variable 'origins' (line 1494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1494, 4), 'origins', _normalize_sequence_call_result_119490)
    
    # Type idiom detected: calculating its left and rigth part (line 1495)
    # Getting the type of 'footprint' (line 1495)
    footprint_119491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 7), 'footprint')
    # Getting the type of 'None' (line 1495)
    None_119492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 20), 'None')
    
    (may_be_119493, more_types_in_union_119494) = may_be_none(footprint_119491, None_119492)

    if may_be_119493:

        if more_types_in_union_119494:
            # Runtime conditional SSA (line 1495)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 1496)
        # Getting the type of 'size' (line 1496)
        size_119495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 11), 'size')
        # Getting the type of 'None' (line 1496)
        None_119496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 19), 'None')
        
        (may_be_119497, more_types_in_union_119498) = may_be_none(size_119495, None_119496)

        if may_be_119497:

            if more_types_in_union_119498:
                # Runtime conditional SSA (line 1496)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to RuntimeError(...): (line 1497)
            # Processing the call arguments (line 1497)
            str_119500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 31), 'str', 'no footprint or filter size provided')
            # Processing the call keyword arguments (line 1497)
            kwargs_119501 = {}
            # Getting the type of 'RuntimeError' (line 1497)
            RuntimeError_119499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 18), 'RuntimeError', False)
            # Calling RuntimeError(args, kwargs) (line 1497)
            RuntimeError_call_result_119502 = invoke(stypy.reporting.localization.Localization(__file__, 1497, 18), RuntimeError_119499, *[str_119500], **kwargs_119501)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1497, 12), RuntimeError_call_result_119502, 'raise parameter', BaseException)

            if more_types_in_union_119498:
                # SSA join for if statement (line 1496)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 1498):
        
        # Assigning a Call to a Name (line 1498):
        
        # Call to _normalize_sequence(...): (line 1498)
        # Processing the call arguments (line 1498)
        # Getting the type of 'size' (line 1498)
        size_119505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 48), 'size', False)
        # Getting the type of 'input' (line 1498)
        input_119506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 54), 'input', False)
        # Obtaining the member 'ndim' of a type (line 1498)
        ndim_119507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1498, 54), input_119506, 'ndim')
        # Processing the call keyword arguments (line 1498)
        kwargs_119508 = {}
        # Getting the type of '_ni_support' (line 1498)
        _ni_support_119503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 16), '_ni_support', False)
        # Obtaining the member '_normalize_sequence' of a type (line 1498)
        _normalize_sequence_119504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1498, 16), _ni_support_119503, '_normalize_sequence')
        # Calling _normalize_sequence(args, kwargs) (line 1498)
        _normalize_sequence_call_result_119509 = invoke(stypy.reporting.localization.Localization(__file__, 1498, 16), _normalize_sequence_119504, *[size_119505, ndim_119507], **kwargs_119508)
        
        # Assigning a type to the variable 'sizes' (line 1498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1498, 8), 'sizes', _normalize_sequence_call_result_119509)
        
        # Assigning a Call to a Name (line 1499):
        
        # Assigning a Call to a Name (line 1499):
        
        # Call to ones(...): (line 1499)
        # Processing the call arguments (line 1499)
        # Getting the type of 'sizes' (line 1499)
        sizes_119512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 31), 'sizes', False)
        # Processing the call keyword arguments (line 1499)
        # Getting the type of 'bool' (line 1499)
        bool_119513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 44), 'bool', False)
        keyword_119514 = bool_119513
        kwargs_119515 = {'dtype': keyword_119514}
        # Getting the type of 'numpy' (line 1499)
        numpy_119510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 20), 'numpy', False)
        # Obtaining the member 'ones' of a type (line 1499)
        ones_119511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1499, 20), numpy_119510, 'ones')
        # Calling ones(args, kwargs) (line 1499)
        ones_call_result_119516 = invoke(stypy.reporting.localization.Localization(__file__, 1499, 20), ones_119511, *[sizes_119512], **kwargs_119515)
        
        # Assigning a type to the variable 'footprint' (line 1499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1499, 8), 'footprint', ones_call_result_119516)

        if more_types_in_union_119494:
            # Runtime conditional SSA for else branch (line 1495)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_119493) or more_types_in_union_119494):
        
        # Assigning a Call to a Name (line 1501):
        
        # Assigning a Call to a Name (line 1501):
        
        # Call to asarray(...): (line 1501)
        # Processing the call arguments (line 1501)
        # Getting the type of 'footprint' (line 1501)
        footprint_119519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 34), 'footprint', False)
        # Processing the call keyword arguments (line 1501)
        # Getting the type of 'bool' (line 1501)
        bool_119520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 51), 'bool', False)
        keyword_119521 = bool_119520
        kwargs_119522 = {'dtype': keyword_119521}
        # Getting the type of 'numpy' (line 1501)
        numpy_119517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 20), 'numpy', False)
        # Obtaining the member 'asarray' of a type (line 1501)
        asarray_119518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1501, 20), numpy_119517, 'asarray')
        # Calling asarray(args, kwargs) (line 1501)
        asarray_call_result_119523 = invoke(stypy.reporting.localization.Localization(__file__, 1501, 20), asarray_119518, *[footprint_119519], **kwargs_119522)
        
        # Assigning a type to the variable 'footprint' (line 1501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 8), 'footprint', asarray_call_result_119523)

        if (may_be_119493 and more_types_in_union_119494):
            # SSA join for if statement (line 1495)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a ListComp to a Name (line 1502):
    
    # Assigning a ListComp to a Name (line 1502):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'footprint' (line 1502)
    footprint_119528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 27), 'footprint')
    # Obtaining the member 'shape' of a type (line 1502)
    shape_119529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1502, 27), footprint_119528, 'shape')
    comprehension_119530 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1502, 14), shape_119529)
    # Assigning a type to the variable 'ii' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 14), 'ii', comprehension_119530)
    
    # Getting the type of 'ii' (line 1502)
    ii_119525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 46), 'ii')
    int_119526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 51), 'int')
    # Applying the binary operator '>' (line 1502)
    result_gt_119527 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 46), '>', ii_119525, int_119526)
    
    # Getting the type of 'ii' (line 1502)
    ii_119524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 14), 'ii')
    list_119531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1502, 14), list_119531, ii_119524)
    # Assigning a type to the variable 'fshape' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'fshape', list_119531)
    
    
    
    # Call to len(...): (line 1503)
    # Processing the call arguments (line 1503)
    # Getting the type of 'fshape' (line 1503)
    fshape_119533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 11), 'fshape', False)
    # Processing the call keyword arguments (line 1503)
    kwargs_119534 = {}
    # Getting the type of 'len' (line 1503)
    len_119532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 7), 'len', False)
    # Calling len(args, kwargs) (line 1503)
    len_call_result_119535 = invoke(stypy.reporting.localization.Localization(__file__, 1503, 7), len_119532, *[fshape_119533], **kwargs_119534)
    
    # Getting the type of 'input' (line 1503)
    input_119536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 22), 'input')
    # Obtaining the member 'ndim' of a type (line 1503)
    ndim_119537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1503, 22), input_119536, 'ndim')
    # Applying the binary operator '!=' (line 1503)
    result_ne_119538 = python_operator(stypy.reporting.localization.Localization(__file__, 1503, 7), '!=', len_call_result_119535, ndim_119537)
    
    # Testing the type of an if condition (line 1503)
    if_condition_119539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1503, 4), result_ne_119538)
    # Assigning a type to the variable 'if_condition_119539' (line 1503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1503, 4), 'if_condition_119539', if_condition_119539)
    # SSA begins for if statement (line 1503)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RuntimeError(...): (line 1504)
    # Processing the call arguments (line 1504)
    str_119541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1504, 27), 'str', 'filter footprint array has incorrect shape.')
    # Processing the call keyword arguments (line 1504)
    kwargs_119542 = {}
    # Getting the type of 'RuntimeError' (line 1504)
    RuntimeError_119540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 14), 'RuntimeError', False)
    # Calling RuntimeError(args, kwargs) (line 1504)
    RuntimeError_call_result_119543 = invoke(stypy.reporting.localization.Localization(__file__, 1504, 14), RuntimeError_119540, *[str_119541], **kwargs_119542)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1504, 8), RuntimeError_call_result_119543, 'raise parameter', BaseException)
    # SSA join for if statement (line 1503)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to zip(...): (line 1505)
    # Processing the call arguments (line 1505)
    # Getting the type of 'origins' (line 1505)
    origins_119545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 28), 'origins', False)
    # Getting the type of 'fshape' (line 1505)
    fshape_119546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 37), 'fshape', False)
    # Processing the call keyword arguments (line 1505)
    kwargs_119547 = {}
    # Getting the type of 'zip' (line 1505)
    zip_119544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 24), 'zip', False)
    # Calling zip(args, kwargs) (line 1505)
    zip_call_result_119548 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 24), zip_119544, *[origins_119545, fshape_119546], **kwargs_119547)
    
    # Testing the type of a for loop iterable (line 1505)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1505, 4), zip_call_result_119548)
    # Getting the type of the for loop variable (line 1505)
    for_loop_var_119549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1505, 4), zip_call_result_119548)
    # Assigning a type to the variable 'origin' (line 1505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 4), 'origin', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1505, 4), for_loop_var_119549))
    # Assigning a type to the variable 'lenf' (line 1505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 4), 'lenf', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1505, 4), for_loop_var_119549))
    # SSA begins for a for statement (line 1505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lenf' (line 1506)
    lenf_119550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 12), 'lenf')
    int_119551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1506, 20), 'int')
    # Applying the binary operator '//' (line 1506)
    result_floordiv_119552 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 12), '//', lenf_119550, int_119551)
    
    # Getting the type of 'origin' (line 1506)
    origin_119553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 24), 'origin')
    # Applying the binary operator '+' (line 1506)
    result_add_119554 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 12), '+', result_floordiv_119552, origin_119553)
    
    int_119555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1506, 33), 'int')
    # Applying the binary operator '<' (line 1506)
    result_lt_119556 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 12), '<', result_add_119554, int_119555)
    
    
    # Getting the type of 'lenf' (line 1506)
    lenf_119557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 40), 'lenf')
    int_119558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1506, 48), 'int')
    # Applying the binary operator '//' (line 1506)
    result_floordiv_119559 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 40), '//', lenf_119557, int_119558)
    
    # Getting the type of 'origin' (line 1506)
    origin_119560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 52), 'origin')
    # Applying the binary operator '+' (line 1506)
    result_add_119561 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 40), '+', result_floordiv_119559, origin_119560)
    
    # Getting the type of 'lenf' (line 1506)
    lenf_119562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 62), 'lenf')
    # Applying the binary operator '>=' (line 1506)
    result_ge_119563 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 40), '>=', result_add_119561, lenf_119562)
    
    # Applying the binary operator 'or' (line 1506)
    result_or_keyword_119564 = python_operator(stypy.reporting.localization.Localization(__file__, 1506, 11), 'or', result_lt_119556, result_ge_119563)
    
    # Testing the type of an if condition (line 1506)
    if_condition_119565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1506, 8), result_or_keyword_119564)
    # Assigning a type to the variable 'if_condition_119565' (line 1506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1506, 8), 'if_condition_119565', if_condition_119565)
    # SSA begins for if statement (line 1506)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1507)
    # Processing the call arguments (line 1507)
    str_119567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1507, 29), 'str', 'invalid origin')
    # Processing the call keyword arguments (line 1507)
    kwargs_119568 = {}
    # Getting the type of 'ValueError' (line 1507)
    ValueError_119566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1507)
    ValueError_call_result_119569 = invoke(stypy.reporting.localization.Localization(__file__, 1507, 18), ValueError_119566, *[str_119567], **kwargs_119568)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1507, 12), ValueError_call_result_119569, 'raise parameter', BaseException)
    # SSA join for if statement (line 1506)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'footprint' (line 1508)
    footprint_119570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 11), 'footprint')
    # Obtaining the member 'flags' of a type (line 1508)
    flags_119571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1508, 11), footprint_119570, 'flags')
    # Obtaining the member 'contiguous' of a type (line 1508)
    contiguous_119572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1508, 11), flags_119571, 'contiguous')
    # Applying the 'not' unary operator (line 1508)
    result_not__119573 = python_operator(stypy.reporting.localization.Localization(__file__, 1508, 7), 'not', contiguous_119572)
    
    # Testing the type of an if condition (line 1508)
    if_condition_119574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1508, 4), result_not__119573)
    # Assigning a type to the variable 'if_condition_119574' (line 1508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1508, 4), 'if_condition_119574', if_condition_119574)
    # SSA begins for if statement (line 1508)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1509):
    
    # Assigning a Call to a Name (line 1509):
    
    # Call to copy(...): (line 1509)
    # Processing the call keyword arguments (line 1509)
    kwargs_119577 = {}
    # Getting the type of 'footprint' (line 1509)
    footprint_119575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1509, 20), 'footprint', False)
    # Obtaining the member 'copy' of a type (line 1509)
    copy_119576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1509, 20), footprint_119575, 'copy')
    # Calling copy(args, kwargs) (line 1509)
    copy_call_result_119578 = invoke(stypy.reporting.localization.Localization(__file__, 1509, 20), copy_119576, *[], **kwargs_119577)
    
    # Assigning a type to the variable 'footprint' (line 1509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1509, 8), 'footprint', copy_call_result_119578)
    # SSA join for if statement (line 1508)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1510):
    
    # Assigning a Subscript to a Name (line 1510):
    
    # Obtaining the type of the subscript
    int_119579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1510, 4), 'int')
    
    # Call to _get_output(...): (line 1510)
    # Processing the call arguments (line 1510)
    # Getting the type of 'output' (line 1510)
    output_119582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 51), 'output', False)
    # Getting the type of 'input' (line 1510)
    input_119583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 59), 'input', False)
    # Processing the call keyword arguments (line 1510)
    kwargs_119584 = {}
    # Getting the type of '_ni_support' (line 1510)
    _ni_support_119580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1510)
    _get_output_119581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 27), _ni_support_119580, '_get_output')
    # Calling _get_output(args, kwargs) (line 1510)
    _get_output_call_result_119585 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 27), _get_output_119581, *[output_119582, input_119583], **kwargs_119584)
    
    # Obtaining the member '__getitem__' of a type (line 1510)
    getitem___119586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 4), _get_output_call_result_119585, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1510)
    subscript_call_result_119587 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 4), getitem___119586, int_119579)
    
    # Assigning a type to the variable 'tuple_var_assignment_117051' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'tuple_var_assignment_117051', subscript_call_result_119587)
    
    # Assigning a Subscript to a Name (line 1510):
    
    # Obtaining the type of the subscript
    int_119588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1510, 4), 'int')
    
    # Call to _get_output(...): (line 1510)
    # Processing the call arguments (line 1510)
    # Getting the type of 'output' (line 1510)
    output_119591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 51), 'output', False)
    # Getting the type of 'input' (line 1510)
    input_119592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 59), 'input', False)
    # Processing the call keyword arguments (line 1510)
    kwargs_119593 = {}
    # Getting the type of '_ni_support' (line 1510)
    _ni_support_119589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 27), '_ni_support', False)
    # Obtaining the member '_get_output' of a type (line 1510)
    _get_output_119590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 27), _ni_support_119589, '_get_output')
    # Calling _get_output(args, kwargs) (line 1510)
    _get_output_call_result_119594 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 27), _get_output_119590, *[output_119591, input_119592], **kwargs_119593)
    
    # Obtaining the member '__getitem__' of a type (line 1510)
    getitem___119595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 4), _get_output_call_result_119594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1510)
    subscript_call_result_119596 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 4), getitem___119595, int_119588)
    
    # Assigning a type to the variable 'tuple_var_assignment_117052' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'tuple_var_assignment_117052', subscript_call_result_119596)
    
    # Assigning a Name to a Name (line 1510):
    # Getting the type of 'tuple_var_assignment_117051' (line 1510)
    tuple_var_assignment_117051_119597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'tuple_var_assignment_117051')
    # Assigning a type to the variable 'output' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'output', tuple_var_assignment_117051_119597)
    
    # Assigning a Name to a Name (line 1510):
    # Getting the type of 'tuple_var_assignment_117052' (line 1510)
    tuple_var_assignment_117052_119598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'tuple_var_assignment_117052')
    # Assigning a type to the variable 'return_value' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 12), 'return_value', tuple_var_assignment_117052_119598)
    
    # Assigning a Call to a Name (line 1511):
    
    # Assigning a Call to a Name (line 1511):
    
    # Call to _extend_mode_to_code(...): (line 1511)
    # Processing the call arguments (line 1511)
    # Getting the type of 'mode' (line 1511)
    mode_119601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 44), 'mode', False)
    # Processing the call keyword arguments (line 1511)
    kwargs_119602 = {}
    # Getting the type of '_ni_support' (line 1511)
    _ni_support_119599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 11), '_ni_support', False)
    # Obtaining the member '_extend_mode_to_code' of a type (line 1511)
    _extend_mode_to_code_119600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1511, 11), _ni_support_119599, '_extend_mode_to_code')
    # Calling _extend_mode_to_code(args, kwargs) (line 1511)
    _extend_mode_to_code_call_result_119603 = invoke(stypy.reporting.localization.Localization(__file__, 1511, 11), _extend_mode_to_code_119600, *[mode_119601], **kwargs_119602)
    
    # Assigning a type to the variable 'mode' (line 1511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1511, 4), 'mode', _extend_mode_to_code_call_result_119603)
    
    # Call to generic_filter(...): (line 1512)
    # Processing the call arguments (line 1512)
    # Getting the type of 'input' (line 1512)
    input_119606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 29), 'input', False)
    # Getting the type of 'function' (line 1512)
    function_119607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 36), 'function', False)
    # Getting the type of 'footprint' (line 1512)
    footprint_119608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 46), 'footprint', False)
    # Getting the type of 'output' (line 1512)
    output_119609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 57), 'output', False)
    # Getting the type of 'mode' (line 1512)
    mode_119610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 65), 'mode', False)
    # Getting the type of 'cval' (line 1513)
    cval_119611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 29), 'cval', False)
    # Getting the type of 'origins' (line 1513)
    origins_119612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 35), 'origins', False)
    # Getting the type of 'extra_arguments' (line 1513)
    extra_arguments_119613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 44), 'extra_arguments', False)
    # Getting the type of 'extra_keywords' (line 1513)
    extra_keywords_119614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 61), 'extra_keywords', False)
    # Processing the call keyword arguments (line 1512)
    kwargs_119615 = {}
    # Getting the type of '_nd_image' (line 1512)
    _nd_image_119604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 4), '_nd_image', False)
    # Obtaining the member 'generic_filter' of a type (line 1512)
    generic_filter_119605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1512, 4), _nd_image_119604, 'generic_filter')
    # Calling generic_filter(args, kwargs) (line 1512)
    generic_filter_call_result_119616 = invoke(stypy.reporting.localization.Localization(__file__, 1512, 4), generic_filter_119605, *[input_119606, function_119607, footprint_119608, output_119609, mode_119610, cval_119611, origins_119612, extra_arguments_119613, extra_keywords_119614], **kwargs_119615)
    
    # Getting the type of 'return_value' (line 1514)
    return_value_119617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 11), 'return_value')
    # Assigning a type to the variable 'stypy_return_type' (line 1514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1514, 4), 'stypy_return_type', return_value_119617)
    
    # ################# End of 'generic_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generic_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 1435)
    stypy_return_type_119618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1435, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_119618)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generic_filter'
    return stypy_return_type_119618

# Assigning a type to the variable 'generic_filter' (line 1435)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1435, 0), 'generic_filter', generic_filter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
