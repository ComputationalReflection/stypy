
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Discrete Fourier Transforms - basic.py
3: '''
4: # Created by Pearu Peterson, August,September 2002
5: from __future__ import division, print_function, absolute_import
6: 
7: __all__ = ['fft','ifft','fftn','ifftn','rfft','irfft',
8:            'fft2','ifft2']
9: 
10: from numpy import zeros, swapaxes
11: import numpy
12: from . import _fftpack
13: 
14: import atexit
15: atexit.register(_fftpack.destroy_zfft_cache)
16: atexit.register(_fftpack.destroy_zfftnd_cache)
17: atexit.register(_fftpack.destroy_drfft_cache)
18: atexit.register(_fftpack.destroy_cfft_cache)
19: atexit.register(_fftpack.destroy_cfftnd_cache)
20: atexit.register(_fftpack.destroy_rfft_cache)
21: del atexit
22: 
23: 
24: def istype(arr, typeclass):
25:     return issubclass(arr.dtype.type, typeclass)
26: 
27: 
28: def _datacopied(arr, original):
29:     '''
30:     Strict check for `arr` not sharing any data with `original`,
31:     under the assumption that arr = asarray(original)
32: 
33:     '''
34:     if arr is original:
35:         return False
36:     if not isinstance(original, numpy.ndarray) and hasattr(original, '__array__'):
37:         return False
38:     return arr.base is None
39: 
40: # XXX: single precision FFTs partially disabled due to accuracy issues
41: #      for large prime-sized inputs.
42: #
43: #      See http://permalink.gmane.org/gmane.comp.python.scientific.devel/13834
44: #      ("fftpack test failures for 0.8.0b1", Ralf Gommers, 17 Jun 2010,
45: #       @ scipy-dev)
46: #
47: #      These should be re-enabled once the problems are resolved
48: 
49: 
50: def _is_safe_size(n):
51:     '''
52:     Is the size of FFT such that FFTPACK can handle it in single precision
53:     with sufficient accuracy?
54: 
55:     Composite numbers of 2, 3, and 5 are accepted, as FFTPACK has those
56:     '''
57:     n = int(n)
58: 
59:     if n == 0:
60:         return True
61: 
62:     # Divide by 3 until you can't, then by 5 until you can't
63:     for c in (3, 5):
64:         while n % c == 0:
65:             n //= c
66: 
67:     # Return True if the remainder is a power of 2
68:     return not n & (n-1)
69: 
70: 
71: def _fake_crfft(x, n, *a, **kw):
72:     if _is_safe_size(n):
73:         return _fftpack.crfft(x, n, *a, **kw)
74:     else:
75:         return _fftpack.zrfft(x, n, *a, **kw).astype(numpy.complex64)
76: 
77: 
78: def _fake_cfft(x, n, *a, **kw):
79:     if _is_safe_size(n):
80:         return _fftpack.cfft(x, n, *a, **kw)
81:     else:
82:         return _fftpack.zfft(x, n, *a, **kw).astype(numpy.complex64)
83: 
84: 
85: def _fake_rfft(x, n, *a, **kw):
86:     if _is_safe_size(n):
87:         return _fftpack.rfft(x, n, *a, **kw)
88:     else:
89:         return _fftpack.drfft(x, n, *a, **kw).astype(numpy.float32)
90: 
91: 
92: def _fake_cfftnd(x, shape, *a, **kw):
93:     if numpy.all(list(map(_is_safe_size, shape))):
94:         return _fftpack.cfftnd(x, shape, *a, **kw)
95:     else:
96:         return _fftpack.zfftnd(x, shape, *a, **kw).astype(numpy.complex64)
97: 
98: _DTYPE_TO_FFT = {
99: #        numpy.dtype(numpy.float32): _fftpack.crfft,
100:         numpy.dtype(numpy.float32): _fake_crfft,
101:         numpy.dtype(numpy.float64): _fftpack.zrfft,
102: #        numpy.dtype(numpy.complex64): _fftpack.cfft,
103:         numpy.dtype(numpy.complex64): _fake_cfft,
104:         numpy.dtype(numpy.complex128): _fftpack.zfft,
105: }
106: 
107: _DTYPE_TO_RFFT = {
108: #        numpy.dtype(numpy.float32): _fftpack.rfft,
109:         numpy.dtype(numpy.float32): _fake_rfft,
110:         numpy.dtype(numpy.float64): _fftpack.drfft,
111: }
112: 
113: _DTYPE_TO_FFTN = {
114: #        numpy.dtype(numpy.complex64): _fftpack.cfftnd,
115:         numpy.dtype(numpy.complex64): _fake_cfftnd,
116:         numpy.dtype(numpy.complex128): _fftpack.zfftnd,
117: #        numpy.dtype(numpy.float32): _fftpack.cfftnd,
118:         numpy.dtype(numpy.float32): _fake_cfftnd,
119:         numpy.dtype(numpy.float64): _fftpack.zfftnd,
120: }
121: 
122: 
123: def _asfarray(x):
124:     '''Like numpy asfarray, except that it does not modify x dtype if x is
125:     already an array with a float dtype, and do not cast complex types to
126:     real.'''
127:     if hasattr(x, "dtype") and x.dtype.char in numpy.typecodes["AllFloat"]:
128:         # 'dtype' attribute does not ensure that the
129:         # object is an ndarray (e.g. Series class
130:         # from the pandas library)
131:         if x.dtype == numpy.half:
132:             # no half-precision routines, so convert to single precision
133:             return numpy.asarray(x, dtype=numpy.float32)
134:         return numpy.asarray(x, dtype=x.dtype)
135:     else:
136:         # We cannot use asfarray directly because it converts sequences of
137:         # complex to sequence of real
138:         ret = numpy.asarray(x)
139:         if ret.dtype == numpy.half:
140:             return numpy.asarray(ret, dtype=numpy.float32)
141:         elif ret.dtype.char not in numpy.typecodes["AllFloat"]:
142:             return numpy.asfarray(x)
143:         return ret
144: 
145: 
146: def _fix_shape(x, n, axis):
147:     ''' Internal auxiliary function for _raw_fft, _raw_fftnd.'''
148:     s = list(x.shape)
149:     if s[axis] > n:
150:         index = [slice(None)]*len(s)
151:         index[axis] = slice(0,n)
152:         x = x[index]
153:         return x, False
154:     else:
155:         index = [slice(None)]*len(s)
156:         index[axis] = slice(0,s[axis])
157:         s[axis] = n
158:         z = zeros(s,x.dtype.char)
159:         z[index] = x
160:         return z, True
161: 
162: 
163: def _raw_fft(x, n, axis, direction, overwrite_x, work_function):
164:     ''' Internal auxiliary function for fft, ifft, rfft, irfft.'''
165:     if n is None:
166:         n = x.shape[axis]
167:     elif n != x.shape[axis]:
168:         x, copy_made = _fix_shape(x,n,axis)
169:         overwrite_x = overwrite_x or copy_made
170: 
171:     if n < 1:
172:         raise ValueError("Invalid number of FFT data points "
173:                          "(%d) specified." % n)
174: 
175:     if axis == -1 or axis == len(x.shape)-1:
176:         r = work_function(x,n,direction,overwrite_x=overwrite_x)
177:     else:
178:         x = swapaxes(x, axis, -1)
179:         r = work_function(x,n,direction,overwrite_x=overwrite_x)
180:         r = swapaxes(r, axis, -1)
181:     return r
182: 
183: 
184: def fft(x, n=None, axis=-1, overwrite_x=False):
185:     '''
186:     Return discrete Fourier transform of real or complex sequence.
187: 
188:     The returned complex array contains ``y(0), y(1),..., y(n-1)`` where
189: 
190:     ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.
191: 
192:     Parameters
193:     ----------
194:     x : array_like
195:         Array to Fourier transform.
196:     n : int, optional
197:         Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is
198:         truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The
199:         default results in ``n = x.shape[axis]``.
200:     axis : int, optional
201:         Axis along which the fft's are computed; the default is over the
202:         last axis (i.e., ``axis=-1``).
203:     overwrite_x : bool, optional
204:         If True, the contents of `x` can be destroyed; the default is False.
205: 
206:     Returns
207:     -------
208:     z : complex ndarray
209:         with the elements::
210: 
211:             [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even
212:             [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd
213: 
214:         where::
215: 
216:             y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1
217: 
218:     See Also
219:     --------
220:     ifft : Inverse FFT
221:     rfft : FFT of a real sequence
222: 
223:     Notes
224:     -----
225:     The packing of the result is "standard": If ``A = fft(a, n)``, then
226:     ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the
227:     positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency
228:     terms, in order of decreasingly negative frequency. So for an 8-point
229:     transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].
230:     To rearrange the fft output so that the zero-frequency component is
231:     centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.
232: 
233:     Both single and double precision routines are implemented.  Half precision
234:     inputs will be converted to single precision.  Non floating-point inputs
235:     will be converted to double precision.  Long-double precision inputs are
236:     not supported.
237: 
238:     This function is most efficient when `n` is a power of two, and least
239:     efficient when `n` is prime.
240: 
241:     Note that if ``x`` is real-valued then ``A[j] == A[n-j].conjugate()``.
242:     If ``x`` is real-valued and ``n`` is even then ``A[n/2]`` is real.
243: 
244:     If the data type of `x` is real, a "real FFT" algorithm is automatically
245:     used, which roughly halves the computation time.  To increase efficiency
246:     a little further, use `rfft`, which does the same calculation, but only
247:     outputs half of the symmetrical spectrum.  If the data is both real and
248:     symmetrical, the `dct` can again double the efficiency, by generating
249:     half of the spectrum from half of the signal.
250: 
251:     Examples
252:     --------
253:     >>> from scipy.fftpack import fft, ifft
254:     >>> x = np.arange(5)
255:     >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.
256:     True
257: 
258:     '''
259:     tmp = _asfarray(x)
260: 
261:     try:
262:         work_function = _DTYPE_TO_FFT[tmp.dtype]
263:     except KeyError:
264:         raise ValueError("type %s is not supported" % tmp.dtype)
265: 
266:     if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
267:         overwrite_x = 1
268: 
269:     overwrite_x = overwrite_x or _datacopied(tmp, x)
270: 
271:     if n is None:
272:         n = tmp.shape[axis]
273:     elif n != tmp.shape[axis]:
274:         tmp, copy_made = _fix_shape(tmp,n,axis)
275:         overwrite_x = overwrite_x or copy_made
276: 
277:     if n < 1:
278:         raise ValueError("Invalid number of FFT data points "
279:                          "(%d) specified." % n)
280: 
281:     if axis == -1 or axis == len(tmp.shape) - 1:
282:         return work_function(tmp,n,1,0,overwrite_x)
283: 
284:     tmp = swapaxes(tmp, axis, -1)
285:     tmp = work_function(tmp,n,1,0,overwrite_x)
286:     return swapaxes(tmp, axis, -1)
287: 
288: 
289: def ifft(x, n=None, axis=-1, overwrite_x=False):
290:     '''
291:     Return discrete inverse Fourier transform of real or complex sequence.
292: 
293:     The returned complex array contains ``y(0), y(1),..., y(n-1)`` where
294: 
295:     ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.
296: 
297:     Parameters
298:     ----------
299:     x : array_like
300:         Transformed data to invert.
301:     n : int, optional
302:         Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,
303:         `x` is truncated.  If ``n > x.shape[axis]``, `x` is zero-padded.
304:         The default results in ``n = x.shape[axis]``.
305:     axis : int, optional
306:         Axis along which the ifft's are computed; the default is over the
307:         last axis (i.e., ``axis=-1``).
308:     overwrite_x : bool, optional
309:         If True, the contents of `x` can be destroyed; the default is False.
310: 
311:     Returns
312:     -------
313:     ifft : ndarray of floats
314:         The inverse discrete Fourier transform.
315: 
316:     See Also
317:     --------
318:     fft : Forward FFT
319: 
320:     Notes
321:     -----
322:     Both single and double precision routines are implemented.  Half precision
323:     inputs will be converted to single precision.  Non floating-point inputs
324:     will be converted to double precision.  Long-double precision inputs are
325:     not supported.
326: 
327:     This function is most efficient when `n` is a power of two, and least
328:     efficient when `n` is prime.
329: 
330:     If the data type of `x` is real, a "real IFFT" algorithm is automatically
331:     used, which roughly halves the computation time.
332: 
333:     '''
334:     tmp = _asfarray(x)
335: 
336:     try:
337:         work_function = _DTYPE_TO_FFT[tmp.dtype]
338:     except KeyError:
339:         raise ValueError("type %s is not supported" % tmp.dtype)
340: 
341:     if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
342:         overwrite_x = 1
343: 
344:     overwrite_x = overwrite_x or _datacopied(tmp, x)
345: 
346:     if n is None:
347:         n = tmp.shape[axis]
348:     elif n != tmp.shape[axis]:
349:         tmp, copy_made = _fix_shape(tmp,n,axis)
350:         overwrite_x = overwrite_x or copy_made
351: 
352:     if n < 1:
353:         raise ValueError("Invalid number of FFT data points "
354:                          "(%d) specified." % n)
355: 
356:     if axis == -1 or axis == len(tmp.shape) - 1:
357:         return work_function(tmp,n,-1,1,overwrite_x)
358: 
359:     tmp = swapaxes(tmp, axis, -1)
360:     tmp = work_function(tmp,n,-1,1,overwrite_x)
361:     return swapaxes(tmp, axis, -1)
362: 
363: 
364: def rfft(x, n=None, axis=-1, overwrite_x=False):
365:     '''
366:     Discrete Fourier transform of a real sequence.
367: 
368:     Parameters
369:     ----------
370:     x : array_like, real-valued
371:         The data to transform.
372:     n : int, optional
373:         Defines the length of the Fourier transform.  If `n` is not specified
374:         (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,
375:         `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.
376:     axis : int, optional
377:         The axis along which the transform is applied.  The default is the
378:         last axis.
379:     overwrite_x : bool, optional
380:         If set to true, the contents of `x` can be overwritten. Default is
381:         False.
382: 
383:     Returns
384:     -------
385:     z : real ndarray
386:         The returned real array contains::
387: 
388:           [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even
389:           [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd
390: 
391:         where::
392: 
393:           y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)
394:           j = 0..n-1
395: 
396:     See Also
397:     --------
398:     fft, irfft, numpy.fft.rfft
399: 
400:     Notes
401:     -----
402:     Within numerical accuracy, ``y == rfft(irfft(y))``.
403: 
404:     Both single and double precision routines are implemented.  Half precision
405:     inputs will be converted to single precision.  Non floating-point inputs
406:     will be converted to double precision.  Long-double precision inputs are
407:     not supported.
408: 
409:     To get an output with a complex datatype, consider using the related
410:     function `numpy.fft.rfft`.
411: 
412:     Examples
413:     --------
414:     >>> from scipy.fftpack import fft, rfft
415:     >>> a = [9, -9, 1, 3]
416:     >>> fft(a)
417:     array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])
418:     >>> rfft(a)
419:     array([  4.,   8.,  12.,  16.])
420: 
421:     '''
422:     tmp = _asfarray(x)
423: 
424:     if not numpy.isrealobj(tmp):
425:         raise TypeError("1st argument must be real sequence")
426: 
427:     try:
428:         work_function = _DTYPE_TO_RFFT[tmp.dtype]
429:     except KeyError:
430:         raise ValueError("type %s is not supported" % tmp.dtype)
431: 
432:     overwrite_x = overwrite_x or _datacopied(tmp, x)
433: 
434:     return _raw_fft(tmp,n,axis,1,overwrite_x,work_function)
435: 
436: 
437: def irfft(x, n=None, axis=-1, overwrite_x=False):
438:     '''
439:     Return inverse discrete Fourier transform of real sequence x.
440: 
441:     The contents of `x` are interpreted as the output of the `rfft`
442:     function.
443: 
444:     Parameters
445:     ----------
446:     x : array_like
447:         Transformed data to invert.
448:     n : int, optional
449:         Length of the inverse Fourier transform.
450:         If n < x.shape[axis], x is truncated.
451:         If n > x.shape[axis], x is zero-padded.
452:         The default results in n = x.shape[axis].
453:     axis : int, optional
454:         Axis along which the ifft's are computed; the default is over
455:         the last axis (i.e., axis=-1).
456:     overwrite_x : bool, optional
457:         If True, the contents of `x` can be destroyed; the default is False.
458: 
459:     Returns
460:     -------
461:     irfft : ndarray of floats
462:         The inverse discrete Fourier transform.
463: 
464:     See Also
465:     --------
466:     rfft, ifft, numpy.fft.irfft
467: 
468:     Notes
469:     -----
470:     The returned real array contains::
471: 
472:         [y(0),y(1),...,y(n-1)]
473: 
474:     where for n is even::
475: 
476:         y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])
477:                                      * exp(sqrt(-1)*j*k* 2*pi/n)
478:                     + c.c. + x[0] + (-1)**(j) x[n-1])
479: 
480:     and for n is odd::
481: 
482:         y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])
483:                                      * exp(sqrt(-1)*j*k* 2*pi/n)
484:                     + c.c. + x[0])
485: 
486:     c.c. denotes complex conjugate of preceding expression.
487: 
488:     For details on input parameters, see `rfft`.
489: 
490:     To process (conjugate-symmetric) frequency-domain data with a complex
491:     datatype, consider using the related function `numpy.fft.irfft`.
492:     '''
493:     tmp = _asfarray(x)
494:     if not numpy.isrealobj(tmp):
495:         raise TypeError("1st argument must be real sequence")
496: 
497:     try:
498:         work_function = _DTYPE_TO_RFFT[tmp.dtype]
499:     except KeyError:
500:         raise ValueError("type %s is not supported" % tmp.dtype)
501: 
502:     overwrite_x = overwrite_x or _datacopied(tmp, x)
503: 
504:     return _raw_fft(tmp,n,axis,-1,overwrite_x,work_function)
505: 
506: 
507: def _raw_fftnd(x, s, axes, direction, overwrite_x, work_function):
508:     ''' Internal auxiliary function for fftnd, ifftnd.'''
509:     if s is None:
510:         if axes is None:
511:             s = x.shape
512:         else:
513:             s = numpy.take(x.shape, axes)
514: 
515:     s = tuple(s)
516:     if axes is None:
517:         noaxes = True
518:         axes = list(range(-x.ndim, 0))
519:     else:
520:         noaxes = False
521:     if len(axes) != len(s):
522:         raise ValueError("when given, axes and shape arguments "
523:                          "have to be of the same length")
524: 
525:     for dim in s:
526:         if dim < 1:
527:             raise ValueError("Invalid number of FFT data points "
528:                              "(%s) specified." % (s,))
529: 
530:     # No need to swap axes, array is in C order
531:     if noaxes:
532:         for i in axes:
533:             x, copy_made = _fix_shape(x, s[i], i)
534:             overwrite_x = overwrite_x or copy_made
535:         return work_function(x,s,direction,overwrite_x=overwrite_x)
536: 
537:     # We ordered axes, because the code below to push axes at the end of the
538:     # array assumes axes argument is in ascending order.
539:     a = numpy.array(axes, numpy.intc)
540:     abs_axes = numpy.where(a < 0, a + x.ndim, a)
541:     id_ = numpy.argsort(abs_axes)
542:     axes = [axes[i] for i in id_]
543:     s = [s[i] for i in id_]
544: 
545:     # Swap the request axes, last first (i.e. First swap the axis which ends up
546:     # at -1, then at -2, etc...), such as the request axes on which the
547:     # operation is carried become the last ones
548:     for i in range(1, len(axes)+1):
549:         x = numpy.swapaxes(x, axes[-i], -i)
550: 
551:     # We can now operate on the axes waxes, the p last axes (p = len(axes)), by
552:     # fixing the shape of the input array to 1 for any axis the fft is not
553:     # carried upon.
554:     waxes = list(range(x.ndim - len(axes), x.ndim))
555:     shape = numpy.ones(x.ndim)
556:     shape[waxes] = s
557: 
558:     for i in range(len(waxes)):
559:         x, copy_made = _fix_shape(x, s[i], waxes[i])
560:         overwrite_x = overwrite_x or copy_made
561: 
562:     r = work_function(x, shape, direction, overwrite_x=overwrite_x)
563: 
564:     # reswap in the reverse order (first axis first, etc...) to get original
565:     # order
566:     for i in range(len(axes), 0, -1):
567:         r = numpy.swapaxes(r, -i, axes[-i])
568: 
569:     return r
570: 
571: 
572: def fftn(x, shape=None, axes=None, overwrite_x=False):
573:     '''
574:     Return multidimensional discrete Fourier transform.
575: 
576:     The returned array contains::
577: 
578:       y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
579:          x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)
580: 
581:     where d = len(x.shape) and n = x.shape.
582: 
583:     Parameters
584:     ----------
585:     x : array_like
586:         The (n-dimensional) array to transform.
587:     shape : tuple of ints, optional
588:         The shape of the result.  If both `shape` and `axes` (see below) are
589:         None, `shape` is ``x.shape``; if `shape` is None but `axes` is
590:         not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.
591:         If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.
592:         If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to
593:         length ``shape[i]``.
594:     axes : array_like of ints, optional
595:         The axes of `x` (`y` if `shape` is not None) along which the
596:         transform is applied.
597:     overwrite_x : bool, optional
598:         If True, the contents of `x` can be destroyed.  Default is False.
599: 
600:     Returns
601:     -------
602:     y : complex-valued n-dimensional numpy array
603:         The (n-dimensional) DFT of the input array.
604: 
605:     See Also
606:     --------
607:     ifftn
608: 
609:     Notes
610:     -----
611:     If ``x`` is real-valued, then
612:     ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.
613: 
614:     Both single and double precision routines are implemented.  Half precision
615:     inputs will be converted to single precision.  Non floating-point inputs
616:     will be converted to double precision.  Long-double precision inputs are
617:     not supported.
618: 
619:     Examples
620:     --------
621:     >>> from scipy.fftpack import fftn, ifftn
622:     >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))
623:     >>> np.allclose(y, fftn(ifftn(y)))
624:     True
625: 
626:     '''
627:     return _raw_fftn_dispatch(x, shape, axes, overwrite_x, 1)
628: 
629: 
630: def _raw_fftn_dispatch(x, shape, axes, overwrite_x, direction):
631:     tmp = _asfarray(x)
632: 
633:     try:
634:         work_function = _DTYPE_TO_FFTN[tmp.dtype]
635:     except KeyError:
636:         raise ValueError("type %s is not supported" % tmp.dtype)
637: 
638:     if not (istype(tmp, numpy.complex64) or istype(tmp, numpy.complex128)):
639:         overwrite_x = 1
640: 
641:     overwrite_x = overwrite_x or _datacopied(tmp, x)
642:     return _raw_fftnd(tmp,shape,axes,direction,overwrite_x,work_function)
643: 
644: 
645: def ifftn(x, shape=None, axes=None, overwrite_x=False):
646:     '''
647:     Return inverse multi-dimensional discrete Fourier transform of
648:     arbitrary type sequence x.
649: 
650:     The returned array contains::
651: 
652:       y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]
653:          x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)
654: 
655:     where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.
656: 
657:     For description of parameters see `fftn`.
658: 
659:     See Also
660:     --------
661:     fftn : for detailed information.
662: 
663:     '''
664:     return _raw_fftn_dispatch(x, shape, axes, overwrite_x, -1)
665: 
666: 
667: def fft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
668:     '''
669:     2-D discrete Fourier transform.
670: 
671:     Return the two-dimensional discrete Fourier transform of the 2-D argument
672:     `x`.
673: 
674:     See Also
675:     --------
676:     fftn : for detailed information.
677: 
678:     '''
679:     return fftn(x,shape,axes,overwrite_x)
680: 
681: 
682: def ifft2(x, shape=None, axes=(-2,-1), overwrite_x=False):
683:     '''
684:     2-D discrete inverse Fourier transform of real or complex sequence.
685: 
686:     Return inverse two-dimensional discrete Fourier transform of
687:     arbitrary type sequence x.
688: 
689:     See `ifft` for more information.
690: 
691:     See also
692:     --------
693:     fft2, ifft
694: 
695:     '''
696:     return ifftn(x,shape,axes,overwrite_x)
697: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_14889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nDiscrete Fourier Transforms - basic.py\n')

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2']
module_type_store.set_exportable_members(['fft', 'ifft', 'fftn', 'ifftn', 'rfft', 'irfft', 'fft2', 'ifft2'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_14890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_14891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14891)
# Adding element type (line 7)
str_14892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14892)
# Adding element type (line 7)
str_14893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'str', 'fftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14893)
# Adding element type (line 7)
str_14894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 31), 'str', 'ifftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14894)
# Adding element type (line 7)
str_14895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 39), 'str', 'rfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14895)
# Adding element type (line 7)
str_14896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 46), 'str', 'irfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14896)
# Adding element type (line 7)
str_14897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'fft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14897)
# Adding element type (line 7)
str_14898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 18), 'str', 'ifft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_14890, str_14898)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_14890)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import zeros, swapaxes' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_14899 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_14899) is not StypyTypeError):

    if (import_14899 != 'pyd_module'):
        __import__(import_14899)
        sys_modules_14900 = sys.modules[import_14899]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_14900.module_type_store, module_type_store, ['zeros', 'swapaxes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_14900, sys_modules_14900.module_type_store, module_type_store)
    else:
        from numpy import zeros, swapaxes

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['zeros', 'swapaxes'], [zeros, swapaxes])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_14899)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_14901 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_14901) is not StypyTypeError):

    if (import_14901 != 'pyd_module'):
        __import__(import_14901)
        sys_modules_14902 = sys.modules[import_14901]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_14902.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_14901)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.fftpack import _fftpack' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/')
import_14903 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.fftpack')

if (type(import_14903) is not StypyTypeError):

    if (import_14903 != 'pyd_module'):
        __import__(import_14903)
        sys_modules_14904 = sys.modules[import_14903]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.fftpack', sys_modules_14904.module_type_store, module_type_store, ['_fftpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_14904, sys_modules_14904.module_type_store, module_type_store)
    else:
        from scipy.fftpack import _fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.fftpack', None, module_type_store, ['_fftpack'], [_fftpack])

else:
    # Assigning a type to the variable 'scipy.fftpack' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.fftpack', import_14903)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import atexit' statement (line 14)
import atexit

import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'atexit', atexit, module_type_store)


# Call to register(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of '_fftpack' (line 15)
_fftpack_14907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), '_fftpack', False)
# Obtaining the member 'destroy_zfft_cache' of a type (line 15)
destroy_zfft_cache_14908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 16), _fftpack_14907, 'destroy_zfft_cache')
# Processing the call keyword arguments (line 15)
kwargs_14909 = {}
# Getting the type of 'atexit' (line 15)
atexit_14905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 15)
register_14906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 0), atexit_14905, 'register')
# Calling register(args, kwargs) (line 15)
register_call_result_14910 = invoke(stypy.reporting.localization.Localization(__file__, 15, 0), register_14906, *[destroy_zfft_cache_14908], **kwargs_14909)


# Call to register(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of '_fftpack' (line 16)
_fftpack_14913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), '_fftpack', False)
# Obtaining the member 'destroy_zfftnd_cache' of a type (line 16)
destroy_zfftnd_cache_14914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), _fftpack_14913, 'destroy_zfftnd_cache')
# Processing the call keyword arguments (line 16)
kwargs_14915 = {}
# Getting the type of 'atexit' (line 16)
atexit_14911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 16)
register_14912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 0), atexit_14911, 'register')
# Calling register(args, kwargs) (line 16)
register_call_result_14916 = invoke(stypy.reporting.localization.Localization(__file__, 16, 0), register_14912, *[destroy_zfftnd_cache_14914], **kwargs_14915)


# Call to register(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of '_fftpack' (line 17)
_fftpack_14919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), '_fftpack', False)
# Obtaining the member 'destroy_drfft_cache' of a type (line 17)
destroy_drfft_cache_14920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), _fftpack_14919, 'destroy_drfft_cache')
# Processing the call keyword arguments (line 17)
kwargs_14921 = {}
# Getting the type of 'atexit' (line 17)
atexit_14917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 17)
register_14918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 0), atexit_14917, 'register')
# Calling register(args, kwargs) (line 17)
register_call_result_14922 = invoke(stypy.reporting.localization.Localization(__file__, 17, 0), register_14918, *[destroy_drfft_cache_14920], **kwargs_14921)


# Call to register(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of '_fftpack' (line 18)
_fftpack_14925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), '_fftpack', False)
# Obtaining the member 'destroy_cfft_cache' of a type (line 18)
destroy_cfft_cache_14926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), _fftpack_14925, 'destroy_cfft_cache')
# Processing the call keyword arguments (line 18)
kwargs_14927 = {}
# Getting the type of 'atexit' (line 18)
atexit_14923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 18)
register_14924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 0), atexit_14923, 'register')
# Calling register(args, kwargs) (line 18)
register_call_result_14928 = invoke(stypy.reporting.localization.Localization(__file__, 18, 0), register_14924, *[destroy_cfft_cache_14926], **kwargs_14927)


# Call to register(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of '_fftpack' (line 19)
_fftpack_14931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), '_fftpack', False)
# Obtaining the member 'destroy_cfftnd_cache' of a type (line 19)
destroy_cfftnd_cache_14932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), _fftpack_14931, 'destroy_cfftnd_cache')
# Processing the call keyword arguments (line 19)
kwargs_14933 = {}
# Getting the type of 'atexit' (line 19)
atexit_14929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 19)
register_14930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 0), atexit_14929, 'register')
# Calling register(args, kwargs) (line 19)
register_call_result_14934 = invoke(stypy.reporting.localization.Localization(__file__, 19, 0), register_14930, *[destroy_cfftnd_cache_14932], **kwargs_14933)


# Call to register(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of '_fftpack' (line 20)
_fftpack_14937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), '_fftpack', False)
# Obtaining the member 'destroy_rfft_cache' of a type (line 20)
destroy_rfft_cache_14938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), _fftpack_14937, 'destroy_rfft_cache')
# Processing the call keyword arguments (line 20)
kwargs_14939 = {}
# Getting the type of 'atexit' (line 20)
atexit_14935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'atexit', False)
# Obtaining the member 'register' of a type (line 20)
register_14936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 0), atexit_14935, 'register')
# Calling register(args, kwargs) (line 20)
register_call_result_14940 = invoke(stypy.reporting.localization.Localization(__file__, 20, 0), register_14936, *[destroy_rfft_cache_14938], **kwargs_14939)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 21, 0), module_type_store, 'atexit')

@norecursion
def istype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'istype'
    module_type_store = module_type_store.open_function_context('istype', 24, 0, False)
    
    # Passed parameters checking function
    istype.stypy_localization = localization
    istype.stypy_type_of_self = None
    istype.stypy_type_store = module_type_store
    istype.stypy_function_name = 'istype'
    istype.stypy_param_names_list = ['arr', 'typeclass']
    istype.stypy_varargs_param_name = None
    istype.stypy_kwargs_param_name = None
    istype.stypy_call_defaults = defaults
    istype.stypy_call_varargs = varargs
    istype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'istype', ['arr', 'typeclass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'istype', localization, ['arr', 'typeclass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'istype(...)' code ##################

    
    # Call to issubclass(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'arr' (line 25)
    arr_14942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 25)
    dtype_14943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 22), arr_14942, 'dtype')
    # Obtaining the member 'type' of a type (line 25)
    type_14944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 22), dtype_14943, 'type')
    # Getting the type of 'typeclass' (line 25)
    typeclass_14945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'typeclass', False)
    # Processing the call keyword arguments (line 25)
    kwargs_14946 = {}
    # Getting the type of 'issubclass' (line 25)
    issubclass_14941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 25)
    issubclass_call_result_14947 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), issubclass_14941, *[type_14944, typeclass_14945], **kwargs_14946)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', issubclass_call_result_14947)
    
    # ################# End of 'istype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'istype' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_14948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14948)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'istype'
    return stypy_return_type_14948

# Assigning a type to the variable 'istype' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'istype', istype)

@norecursion
def _datacopied(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_datacopied'
    module_type_store = module_type_store.open_function_context('_datacopied', 28, 0, False)
    
    # Passed parameters checking function
    _datacopied.stypy_localization = localization
    _datacopied.stypy_type_of_self = None
    _datacopied.stypy_type_store = module_type_store
    _datacopied.stypy_function_name = '_datacopied'
    _datacopied.stypy_param_names_list = ['arr', 'original']
    _datacopied.stypy_varargs_param_name = None
    _datacopied.stypy_kwargs_param_name = None
    _datacopied.stypy_call_defaults = defaults
    _datacopied.stypy_call_varargs = varargs
    _datacopied.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_datacopied', ['arr', 'original'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_datacopied', localization, ['arr', 'original'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_datacopied(...)' code ##################

    str_14949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n    Strict check for `arr` not sharing any data with `original`,\n    under the assumption that arr = asarray(original)\n\n    ')
    
    
    # Getting the type of 'arr' (line 34)
    arr_14950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'arr')
    # Getting the type of 'original' (line 34)
    original_14951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'original')
    # Applying the binary operator 'is' (line 34)
    result_is__14952 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'is', arr_14950, original_14951)
    
    # Testing the type of an if condition (line 34)
    if_condition_14953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_is__14952)
    # Assigning a type to the variable 'if_condition_14953' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_14953', if_condition_14953)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 35)
    False_14954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', False_14954)
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'original' (line 36)
    original_14956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'original', False)
    # Getting the type of 'numpy' (line 36)
    numpy_14957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'numpy', False)
    # Obtaining the member 'ndarray' of a type (line 36)
    ndarray_14958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 32), numpy_14957, 'ndarray')
    # Processing the call keyword arguments (line 36)
    kwargs_14959 = {}
    # Getting the type of 'isinstance' (line 36)
    isinstance_14955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 36)
    isinstance_call_result_14960 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), isinstance_14955, *[original_14956, ndarray_14958], **kwargs_14959)
    
    # Applying the 'not' unary operator (line 36)
    result_not__14961 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), 'not', isinstance_call_result_14960)
    
    
    # Call to hasattr(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'original' (line 36)
    original_14963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 59), 'original', False)
    str_14964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 69), 'str', '__array__')
    # Processing the call keyword arguments (line 36)
    kwargs_14965 = {}
    # Getting the type of 'hasattr' (line 36)
    hasattr_14962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 51), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 36)
    hasattr_call_result_14966 = invoke(stypy.reporting.localization.Localization(__file__, 36, 51), hasattr_14962, *[original_14963, str_14964], **kwargs_14965)
    
    # Applying the binary operator 'and' (line 36)
    result_and_keyword_14967 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), 'and', result_not__14961, hasattr_call_result_14966)
    
    # Testing the type of an if condition (line 36)
    if_condition_14968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_and_keyword_14967)
    # Assigning a type to the variable 'if_condition_14968' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_14968', if_condition_14968)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 37)
    False_14969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', False_14969)
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'arr' (line 38)
    arr_14970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'arr')
    # Obtaining the member 'base' of a type (line 38)
    base_14971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), arr_14970, 'base')
    # Getting the type of 'None' (line 38)
    None_14972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'None')
    # Applying the binary operator 'is' (line 38)
    result_is__14973 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), 'is', base_14971, None_14972)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', result_is__14973)
    
    # ################# End of '_datacopied(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_datacopied' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_14974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14974)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_datacopied'
    return stypy_return_type_14974

# Assigning a type to the variable '_datacopied' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), '_datacopied', _datacopied)

@norecursion
def _is_safe_size(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_safe_size'
    module_type_store = module_type_store.open_function_context('_is_safe_size', 50, 0, False)
    
    # Passed parameters checking function
    _is_safe_size.stypy_localization = localization
    _is_safe_size.stypy_type_of_self = None
    _is_safe_size.stypy_type_store = module_type_store
    _is_safe_size.stypy_function_name = '_is_safe_size'
    _is_safe_size.stypy_param_names_list = ['n']
    _is_safe_size.stypy_varargs_param_name = None
    _is_safe_size.stypy_kwargs_param_name = None
    _is_safe_size.stypy_call_defaults = defaults
    _is_safe_size.stypy_call_varargs = varargs
    _is_safe_size.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_safe_size', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_safe_size', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_safe_size(...)' code ##################

    str_14975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Is the size of FFT such that FFTPACK can handle it in single precision\n    with sufficient accuracy?\n\n    Composite numbers of 2, 3, and 5 are accepted, as FFTPACK has those\n    ')
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to int(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'n' (line 57)
    n_14977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'n', False)
    # Processing the call keyword arguments (line 57)
    kwargs_14978 = {}
    # Getting the type of 'int' (line 57)
    int_14976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'int', False)
    # Calling int(args, kwargs) (line 57)
    int_call_result_14979 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), int_14976, *[n_14977], **kwargs_14978)
    
    # Assigning a type to the variable 'n' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'n', int_call_result_14979)
    
    
    # Getting the type of 'n' (line 59)
    n_14980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'n')
    int_14981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'int')
    # Applying the binary operator '==' (line 59)
    result_eq_14982 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '==', n_14980, int_14981)
    
    # Testing the type of an if condition (line 59)
    if_condition_14983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), result_eq_14982)
    # Assigning a type to the variable 'if_condition_14983' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_14983', if_condition_14983)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 60)
    True_14984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', True_14984)
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_14985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    int_14986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 14), tuple_14985, int_14986)
    # Adding element type (line 63)
    int_14987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 14), tuple_14985, int_14987)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 4), tuple_14985)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_14988 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 4), tuple_14985)
    # Assigning a type to the variable 'c' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'c', for_loop_var_14988)
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'n' (line 64)
    n_14989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'n')
    # Getting the type of 'c' (line 64)
    c_14990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'c')
    # Applying the binary operator '%' (line 64)
    result_mod_14991 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 14), '%', n_14989, c_14990)
    
    int_14992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
    # Applying the binary operator '==' (line 64)
    result_eq_14993 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 14), '==', result_mod_14991, int_14992)
    
    # Testing the type of an if condition (line 64)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_14993)
    # SSA begins for while statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'n' (line 65)
    n_14994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'n')
    # Getting the type of 'c' (line 65)
    c_14995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'c')
    # Applying the binary operator '//=' (line 65)
    result_ifloordiv_14996 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 12), '//=', n_14994, c_14995)
    # Assigning a type to the variable 'n' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'n', result_ifloordiv_14996)
    
    # SSA join for while statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'n' (line 68)
    n_14997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'n')
    # Getting the type of 'n' (line 68)
    n_14998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'n')
    int_14999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
    # Applying the binary operator '-' (line 68)
    result_sub_15000 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 20), '-', n_14998, int_14999)
    
    # Applying the binary operator '&' (line 68)
    result_and__15001 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '&', n_14997, result_sub_15000)
    
    # Applying the 'not' unary operator (line 68)
    result_not__15002 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 11), 'not', result_and__15001)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', result_not__15002)
    
    # ################# End of '_is_safe_size(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_safe_size' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_15003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15003)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_safe_size'
    return stypy_return_type_15003

# Assigning a type to the variable '_is_safe_size' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '_is_safe_size', _is_safe_size)

@norecursion
def _fake_crfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fake_crfft'
    module_type_store = module_type_store.open_function_context('_fake_crfft', 71, 0, False)
    
    # Passed parameters checking function
    _fake_crfft.stypy_localization = localization
    _fake_crfft.stypy_type_of_self = None
    _fake_crfft.stypy_type_store = module_type_store
    _fake_crfft.stypy_function_name = '_fake_crfft'
    _fake_crfft.stypy_param_names_list = ['x', 'n']
    _fake_crfft.stypy_varargs_param_name = 'a'
    _fake_crfft.stypy_kwargs_param_name = 'kw'
    _fake_crfft.stypy_call_defaults = defaults
    _fake_crfft.stypy_call_varargs = varargs
    _fake_crfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fake_crfft', ['x', 'n'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fake_crfft', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fake_crfft(...)' code ##################

    
    
    # Call to _is_safe_size(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'n' (line 72)
    n_15005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'n', False)
    # Processing the call keyword arguments (line 72)
    kwargs_15006 = {}
    # Getting the type of '_is_safe_size' (line 72)
    _is_safe_size_15004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), '_is_safe_size', False)
    # Calling _is_safe_size(args, kwargs) (line 72)
    _is_safe_size_call_result_15007 = invoke(stypy.reporting.localization.Localization(__file__, 72, 7), _is_safe_size_15004, *[n_15005], **kwargs_15006)
    
    # Testing the type of an if condition (line 72)
    if_condition_15008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), _is_safe_size_call_result_15007)
    # Assigning a type to the variable 'if_condition_15008' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_15008', if_condition_15008)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to crfft(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'x' (line 73)
    x_15011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'x', False)
    # Getting the type of 'n' (line 73)
    n_15012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'n', False)
    # Getting the type of 'a' (line 73)
    a_15013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'a', False)
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'kw' (line 73)
    kw_15014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'kw', False)
    kwargs_15015 = {'kw_15014': kw_15014}
    # Getting the type of '_fftpack' (line 73)
    _fftpack_15009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), '_fftpack', False)
    # Obtaining the member 'crfft' of a type (line 73)
    crfft_15010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), _fftpack_15009, 'crfft')
    # Calling crfft(args, kwargs) (line 73)
    crfft_call_result_15016 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), crfft_15010, *[x_15011, n_15012, a_15013], **kwargs_15015)
    
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', crfft_call_result_15016)
    # SSA branch for the else part of an if statement (line 72)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'numpy' (line 75)
    numpy_15026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 53), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 75)
    complex64_15027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 53), numpy_15026, 'complex64')
    # Processing the call keyword arguments (line 75)
    kwargs_15028 = {}
    
    # Call to zrfft(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'x' (line 75)
    x_15019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'x', False)
    # Getting the type of 'n' (line 75)
    n_15020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 33), 'n', False)
    # Getting the type of 'a' (line 75)
    a_15021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'a', False)
    # Processing the call keyword arguments (line 75)
    # Getting the type of 'kw' (line 75)
    kw_15022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'kw', False)
    kwargs_15023 = {'kw_15022': kw_15022}
    # Getting the type of '_fftpack' (line 75)
    _fftpack_15017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), '_fftpack', False)
    # Obtaining the member 'zrfft' of a type (line 75)
    zrfft_15018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), _fftpack_15017, 'zrfft')
    # Calling zrfft(args, kwargs) (line 75)
    zrfft_call_result_15024 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), zrfft_15018, *[x_15019, n_15020, a_15021], **kwargs_15023)
    
    # Obtaining the member 'astype' of a type (line 75)
    astype_15025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), zrfft_call_result_15024, 'astype')
    # Calling astype(args, kwargs) (line 75)
    astype_call_result_15029 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), astype_15025, *[complex64_15027], **kwargs_15028)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type', astype_call_result_15029)
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fake_crfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fake_crfft' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_15030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15030)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fake_crfft'
    return stypy_return_type_15030

# Assigning a type to the variable '_fake_crfft' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), '_fake_crfft', _fake_crfft)

@norecursion
def _fake_cfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fake_cfft'
    module_type_store = module_type_store.open_function_context('_fake_cfft', 78, 0, False)
    
    # Passed parameters checking function
    _fake_cfft.stypy_localization = localization
    _fake_cfft.stypy_type_of_self = None
    _fake_cfft.stypy_type_store = module_type_store
    _fake_cfft.stypy_function_name = '_fake_cfft'
    _fake_cfft.stypy_param_names_list = ['x', 'n']
    _fake_cfft.stypy_varargs_param_name = 'a'
    _fake_cfft.stypy_kwargs_param_name = 'kw'
    _fake_cfft.stypy_call_defaults = defaults
    _fake_cfft.stypy_call_varargs = varargs
    _fake_cfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fake_cfft', ['x', 'n'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fake_cfft', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fake_cfft(...)' code ##################

    
    
    # Call to _is_safe_size(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'n' (line 79)
    n_15032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'n', False)
    # Processing the call keyword arguments (line 79)
    kwargs_15033 = {}
    # Getting the type of '_is_safe_size' (line 79)
    _is_safe_size_15031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), '_is_safe_size', False)
    # Calling _is_safe_size(args, kwargs) (line 79)
    _is_safe_size_call_result_15034 = invoke(stypy.reporting.localization.Localization(__file__, 79, 7), _is_safe_size_15031, *[n_15032], **kwargs_15033)
    
    # Testing the type of an if condition (line 79)
    if_condition_15035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), _is_safe_size_call_result_15034)
    # Assigning a type to the variable 'if_condition_15035' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_15035', if_condition_15035)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cfft(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'x' (line 80)
    x_15038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'x', False)
    # Getting the type of 'n' (line 80)
    n_15039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'n', False)
    # Getting the type of 'a' (line 80)
    a_15040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'a', False)
    # Processing the call keyword arguments (line 80)
    # Getting the type of 'kw' (line 80)
    kw_15041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 41), 'kw', False)
    kwargs_15042 = {'kw_15041': kw_15041}
    # Getting the type of '_fftpack' (line 80)
    _fftpack_15036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), '_fftpack', False)
    # Obtaining the member 'cfft' of a type (line 80)
    cfft_15037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 15), _fftpack_15036, 'cfft')
    # Calling cfft(args, kwargs) (line 80)
    cfft_call_result_15043 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), cfft_15037, *[x_15038, n_15039, a_15040], **kwargs_15042)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', cfft_call_result_15043)
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'numpy' (line 82)
    numpy_15053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 52), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 82)
    complex64_15054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 52), numpy_15053, 'complex64')
    # Processing the call keyword arguments (line 82)
    kwargs_15055 = {}
    
    # Call to zfft(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'x' (line 82)
    x_15046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'x', False)
    # Getting the type of 'n' (line 82)
    n_15047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'n', False)
    # Getting the type of 'a' (line 82)
    a_15048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 36), 'a', False)
    # Processing the call keyword arguments (line 82)
    # Getting the type of 'kw' (line 82)
    kw_15049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'kw', False)
    kwargs_15050 = {'kw_15049': kw_15049}
    # Getting the type of '_fftpack' (line 82)
    _fftpack_15044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), '_fftpack', False)
    # Obtaining the member 'zfft' of a type (line 82)
    zfft_15045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), _fftpack_15044, 'zfft')
    # Calling zfft(args, kwargs) (line 82)
    zfft_call_result_15051 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), zfft_15045, *[x_15046, n_15047, a_15048], **kwargs_15050)
    
    # Obtaining the member 'astype' of a type (line 82)
    astype_15052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), zfft_call_result_15051, 'astype')
    # Calling astype(args, kwargs) (line 82)
    astype_call_result_15056 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), astype_15052, *[complex64_15054], **kwargs_15055)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', astype_call_result_15056)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fake_cfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fake_cfft' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_15057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15057)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fake_cfft'
    return stypy_return_type_15057

# Assigning a type to the variable '_fake_cfft' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '_fake_cfft', _fake_cfft)

@norecursion
def _fake_rfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fake_rfft'
    module_type_store = module_type_store.open_function_context('_fake_rfft', 85, 0, False)
    
    # Passed parameters checking function
    _fake_rfft.stypy_localization = localization
    _fake_rfft.stypy_type_of_self = None
    _fake_rfft.stypy_type_store = module_type_store
    _fake_rfft.stypy_function_name = '_fake_rfft'
    _fake_rfft.stypy_param_names_list = ['x', 'n']
    _fake_rfft.stypy_varargs_param_name = 'a'
    _fake_rfft.stypy_kwargs_param_name = 'kw'
    _fake_rfft.stypy_call_defaults = defaults
    _fake_rfft.stypy_call_varargs = varargs
    _fake_rfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fake_rfft', ['x', 'n'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fake_rfft', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fake_rfft(...)' code ##################

    
    
    # Call to _is_safe_size(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'n' (line 86)
    n_15059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'n', False)
    # Processing the call keyword arguments (line 86)
    kwargs_15060 = {}
    # Getting the type of '_is_safe_size' (line 86)
    _is_safe_size_15058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), '_is_safe_size', False)
    # Calling _is_safe_size(args, kwargs) (line 86)
    _is_safe_size_call_result_15061 = invoke(stypy.reporting.localization.Localization(__file__, 86, 7), _is_safe_size_15058, *[n_15059], **kwargs_15060)
    
    # Testing the type of an if condition (line 86)
    if_condition_15062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), _is_safe_size_call_result_15061)
    # Assigning a type to the variable 'if_condition_15062' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_15062', if_condition_15062)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to rfft(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'x' (line 87)
    x_15065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'x', False)
    # Getting the type of 'n' (line 87)
    n_15066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'n', False)
    # Getting the type of 'a' (line 87)
    a_15067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'a', False)
    # Processing the call keyword arguments (line 87)
    # Getting the type of 'kw' (line 87)
    kw_15068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'kw', False)
    kwargs_15069 = {'kw_15068': kw_15068}
    # Getting the type of '_fftpack' (line 87)
    _fftpack_15063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), '_fftpack', False)
    # Obtaining the member 'rfft' of a type (line 87)
    rfft_15064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), _fftpack_15063, 'rfft')
    # Calling rfft(args, kwargs) (line 87)
    rfft_call_result_15070 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), rfft_15064, *[x_15065, n_15066, a_15067], **kwargs_15069)
    
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', rfft_call_result_15070)
    # SSA branch for the else part of an if statement (line 86)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'numpy' (line 89)
    numpy_15080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'numpy', False)
    # Obtaining the member 'float32' of a type (line 89)
    float32_15081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 53), numpy_15080, 'float32')
    # Processing the call keyword arguments (line 89)
    kwargs_15082 = {}
    
    # Call to drfft(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'x' (line 89)
    x_15073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'x', False)
    # Getting the type of 'n' (line 89)
    n_15074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'n', False)
    # Getting the type of 'a' (line 89)
    a_15075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'a', False)
    # Processing the call keyword arguments (line 89)
    # Getting the type of 'kw' (line 89)
    kw_15076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 42), 'kw', False)
    kwargs_15077 = {'kw_15076': kw_15076}
    # Getting the type of '_fftpack' (line 89)
    _fftpack_15071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), '_fftpack', False)
    # Obtaining the member 'drfft' of a type (line 89)
    drfft_15072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), _fftpack_15071, 'drfft')
    # Calling drfft(args, kwargs) (line 89)
    drfft_call_result_15078 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), drfft_15072, *[x_15073, n_15074, a_15075], **kwargs_15077)
    
    # Obtaining the member 'astype' of a type (line 89)
    astype_15079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), drfft_call_result_15078, 'astype')
    # Calling astype(args, kwargs) (line 89)
    astype_call_result_15083 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), astype_15079, *[float32_15081], **kwargs_15082)
    
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', astype_call_result_15083)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fake_rfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fake_rfft' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_15084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15084)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fake_rfft'
    return stypy_return_type_15084

# Assigning a type to the variable '_fake_rfft' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), '_fake_rfft', _fake_rfft)

@norecursion
def _fake_cfftnd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fake_cfftnd'
    module_type_store = module_type_store.open_function_context('_fake_cfftnd', 92, 0, False)
    
    # Passed parameters checking function
    _fake_cfftnd.stypy_localization = localization
    _fake_cfftnd.stypy_type_of_self = None
    _fake_cfftnd.stypy_type_store = module_type_store
    _fake_cfftnd.stypy_function_name = '_fake_cfftnd'
    _fake_cfftnd.stypy_param_names_list = ['x', 'shape']
    _fake_cfftnd.stypy_varargs_param_name = 'a'
    _fake_cfftnd.stypy_kwargs_param_name = 'kw'
    _fake_cfftnd.stypy_call_defaults = defaults
    _fake_cfftnd.stypy_call_varargs = varargs
    _fake_cfftnd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fake_cfftnd', ['x', 'shape'], 'a', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fake_cfftnd', localization, ['x', 'shape'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fake_cfftnd(...)' code ##################

    
    
    # Call to all(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Call to list(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Call to map(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of '_is_safe_size' (line 93)
    _is_safe_size_15089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), '_is_safe_size', False)
    # Getting the type of 'shape' (line 93)
    shape_15090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'shape', False)
    # Processing the call keyword arguments (line 93)
    kwargs_15091 = {}
    # Getting the type of 'map' (line 93)
    map_15088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'map', False)
    # Calling map(args, kwargs) (line 93)
    map_call_result_15092 = invoke(stypy.reporting.localization.Localization(__file__, 93, 22), map_15088, *[_is_safe_size_15089, shape_15090], **kwargs_15091)
    
    # Processing the call keyword arguments (line 93)
    kwargs_15093 = {}
    # Getting the type of 'list' (line 93)
    list_15087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'list', False)
    # Calling list(args, kwargs) (line 93)
    list_call_result_15094 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), list_15087, *[map_call_result_15092], **kwargs_15093)
    
    # Processing the call keyword arguments (line 93)
    kwargs_15095 = {}
    # Getting the type of 'numpy' (line 93)
    numpy_15085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'numpy', False)
    # Obtaining the member 'all' of a type (line 93)
    all_15086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 7), numpy_15085, 'all')
    # Calling all(args, kwargs) (line 93)
    all_call_result_15096 = invoke(stypy.reporting.localization.Localization(__file__, 93, 7), all_15086, *[list_call_result_15094], **kwargs_15095)
    
    # Testing the type of an if condition (line 93)
    if_condition_15097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), all_call_result_15096)
    # Assigning a type to the variable 'if_condition_15097' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_15097', if_condition_15097)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cfftnd(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'x' (line 94)
    x_15100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'x', False)
    # Getting the type of 'shape' (line 94)
    shape_15101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 34), 'shape', False)
    # Getting the type of 'a' (line 94)
    a_15102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'a', False)
    # Processing the call keyword arguments (line 94)
    # Getting the type of 'kw' (line 94)
    kw_15103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 47), 'kw', False)
    kwargs_15104 = {'kw_15103': kw_15103}
    # Getting the type of '_fftpack' (line 94)
    _fftpack_15098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), '_fftpack', False)
    # Obtaining the member 'cfftnd' of a type (line 94)
    cfftnd_15099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 15), _fftpack_15098, 'cfftnd')
    # Calling cfftnd(args, kwargs) (line 94)
    cfftnd_call_result_15105 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), cfftnd_15099, *[x_15100, shape_15101, a_15102], **kwargs_15104)
    
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'stypy_return_type', cfftnd_call_result_15105)
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    # Call to astype(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'numpy' (line 96)
    numpy_15115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 58), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 96)
    complex64_15116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 58), numpy_15115, 'complex64')
    # Processing the call keyword arguments (line 96)
    kwargs_15117 = {}
    
    # Call to zfftnd(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'x' (line 96)
    x_15108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'x', False)
    # Getting the type of 'shape' (line 96)
    shape_15109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'shape', False)
    # Getting the type of 'a' (line 96)
    a_15110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'a', False)
    # Processing the call keyword arguments (line 96)
    # Getting the type of 'kw' (line 96)
    kw_15111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'kw', False)
    kwargs_15112 = {'kw_15111': kw_15111}
    # Getting the type of '_fftpack' (line 96)
    _fftpack_15106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), '_fftpack', False)
    # Obtaining the member 'zfftnd' of a type (line 96)
    zfftnd_15107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), _fftpack_15106, 'zfftnd')
    # Calling zfftnd(args, kwargs) (line 96)
    zfftnd_call_result_15113 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), zfftnd_15107, *[x_15108, shape_15109, a_15110], **kwargs_15112)
    
    # Obtaining the member 'astype' of a type (line 96)
    astype_15114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), zfftnd_call_result_15113, 'astype')
    # Calling astype(args, kwargs) (line 96)
    astype_call_result_15118 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), astype_15114, *[complex64_15116], **kwargs_15117)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', astype_call_result_15118)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fake_cfftnd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fake_cfftnd' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_15119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fake_cfftnd'
    return stypy_return_type_15119

# Assigning a type to the variable '_fake_cfftnd' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), '_fake_cfftnd', _fake_cfftnd)

# Assigning a Dict to a Name (line 98):

# Assigning a Dict to a Name (line 98):

# Obtaining an instance of the builtin type 'dict' (line 98)
dict_15120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 98)
# Adding element type (key, value) (line 98)

# Call to dtype(...): (line 100)
# Processing the call arguments (line 100)
# Getting the type of 'numpy' (line 100)
numpy_15123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'numpy', False)
# Obtaining the member 'float32' of a type (line 100)
float32_15124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), numpy_15123, 'float32')
# Processing the call keyword arguments (line 100)
kwargs_15125 = {}
# Getting the type of 'numpy' (line 100)
numpy_15121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 100)
dtype_15122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), numpy_15121, 'dtype')
# Calling dtype(args, kwargs) (line 100)
dtype_call_result_15126 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), dtype_15122, *[float32_15124], **kwargs_15125)

# Getting the type of '_fake_crfft' (line 100)
_fake_crfft_15127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), '_fake_crfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), dict_15120, (dtype_call_result_15126, _fake_crfft_15127))
# Adding element type (key, value) (line 98)

# Call to dtype(...): (line 101)
# Processing the call arguments (line 101)
# Getting the type of 'numpy' (line 101)
numpy_15130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'numpy', False)
# Obtaining the member 'float64' of a type (line 101)
float64_15131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 20), numpy_15130, 'float64')
# Processing the call keyword arguments (line 101)
kwargs_15132 = {}
# Getting the type of 'numpy' (line 101)
numpy_15128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 101)
dtype_15129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), numpy_15128, 'dtype')
# Calling dtype(args, kwargs) (line 101)
dtype_call_result_15133 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), dtype_15129, *[float64_15131], **kwargs_15132)

# Getting the type of '_fftpack' (line 101)
_fftpack_15134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 36), '_fftpack')
# Obtaining the member 'zrfft' of a type (line 101)
zrfft_15135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 36), _fftpack_15134, 'zrfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), dict_15120, (dtype_call_result_15133, zrfft_15135))
# Adding element type (key, value) (line 98)

# Call to dtype(...): (line 103)
# Processing the call arguments (line 103)
# Getting the type of 'numpy' (line 103)
numpy_15138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'numpy', False)
# Obtaining the member 'complex64' of a type (line 103)
complex64_15139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 20), numpy_15138, 'complex64')
# Processing the call keyword arguments (line 103)
kwargs_15140 = {}
# Getting the type of 'numpy' (line 103)
numpy_15136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 103)
dtype_15137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), numpy_15136, 'dtype')
# Calling dtype(args, kwargs) (line 103)
dtype_call_result_15141 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), dtype_15137, *[complex64_15139], **kwargs_15140)

# Getting the type of '_fake_cfft' (line 103)
_fake_cfft_15142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), '_fake_cfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), dict_15120, (dtype_call_result_15141, _fake_cfft_15142))
# Adding element type (key, value) (line 98)

# Call to dtype(...): (line 104)
# Processing the call arguments (line 104)
# Getting the type of 'numpy' (line 104)
numpy_15145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'numpy', False)
# Obtaining the member 'complex128' of a type (line 104)
complex128_15146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), numpy_15145, 'complex128')
# Processing the call keyword arguments (line 104)
kwargs_15147 = {}
# Getting the type of 'numpy' (line 104)
numpy_15143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 104)
dtype_15144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), numpy_15143, 'dtype')
# Calling dtype(args, kwargs) (line 104)
dtype_call_result_15148 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), dtype_15144, *[complex128_15146], **kwargs_15147)

# Getting the type of '_fftpack' (line 104)
_fftpack_15149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), '_fftpack')
# Obtaining the member 'zfft' of a type (line 104)
zfft_15150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 39), _fftpack_15149, 'zfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), dict_15120, (dtype_call_result_15148, zfft_15150))

# Assigning a type to the variable '_DTYPE_TO_FFT' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), '_DTYPE_TO_FFT', dict_15120)

# Assigning a Dict to a Name (line 107):

# Assigning a Dict to a Name (line 107):

# Obtaining an instance of the builtin type 'dict' (line 107)
dict_15151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 107)
# Adding element type (key, value) (line 107)

# Call to dtype(...): (line 109)
# Processing the call arguments (line 109)
# Getting the type of 'numpy' (line 109)
numpy_15154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'numpy', False)
# Obtaining the member 'float32' of a type (line 109)
float32_15155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 20), numpy_15154, 'float32')
# Processing the call keyword arguments (line 109)
kwargs_15156 = {}
# Getting the type of 'numpy' (line 109)
numpy_15152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 109)
dtype_15153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), numpy_15152, 'dtype')
# Calling dtype(args, kwargs) (line 109)
dtype_call_result_15157 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), dtype_15153, *[float32_15155], **kwargs_15156)

# Getting the type of '_fake_rfft' (line 109)
_fake_rfft_15158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 36), '_fake_rfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), dict_15151, (dtype_call_result_15157, _fake_rfft_15158))
# Adding element type (key, value) (line 107)

# Call to dtype(...): (line 110)
# Processing the call arguments (line 110)
# Getting the type of 'numpy' (line 110)
numpy_15161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'numpy', False)
# Obtaining the member 'float64' of a type (line 110)
float64_15162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 20), numpy_15161, 'float64')
# Processing the call keyword arguments (line 110)
kwargs_15163 = {}
# Getting the type of 'numpy' (line 110)
numpy_15159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 110)
dtype_15160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), numpy_15159, 'dtype')
# Calling dtype(args, kwargs) (line 110)
dtype_call_result_15164 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), dtype_15160, *[float64_15162], **kwargs_15163)

# Getting the type of '_fftpack' (line 110)
_fftpack_15165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 36), '_fftpack')
# Obtaining the member 'drfft' of a type (line 110)
drfft_15166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 36), _fftpack_15165, 'drfft')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 17), dict_15151, (dtype_call_result_15164, drfft_15166))

# Assigning a type to the variable '_DTYPE_TO_RFFT' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), '_DTYPE_TO_RFFT', dict_15151)

# Assigning a Dict to a Name (line 113):

# Assigning a Dict to a Name (line 113):

# Obtaining an instance of the builtin type 'dict' (line 113)
dict_15167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 113)
# Adding element type (key, value) (line 113)

# Call to dtype(...): (line 115)
# Processing the call arguments (line 115)
# Getting the type of 'numpy' (line 115)
numpy_15170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'numpy', False)
# Obtaining the member 'complex64' of a type (line 115)
complex64_15171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), numpy_15170, 'complex64')
# Processing the call keyword arguments (line 115)
kwargs_15172 = {}
# Getting the type of 'numpy' (line 115)
numpy_15168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 115)
dtype_15169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), numpy_15168, 'dtype')
# Calling dtype(args, kwargs) (line 115)
dtype_call_result_15173 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), dtype_15169, *[complex64_15171], **kwargs_15172)

# Getting the type of '_fake_cfftnd' (line 115)
_fake_cfftnd_15174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), '_fake_cfftnd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 17), dict_15167, (dtype_call_result_15173, _fake_cfftnd_15174))
# Adding element type (key, value) (line 113)

# Call to dtype(...): (line 116)
# Processing the call arguments (line 116)
# Getting the type of 'numpy' (line 116)
numpy_15177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'numpy', False)
# Obtaining the member 'complex128' of a type (line 116)
complex128_15178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), numpy_15177, 'complex128')
# Processing the call keyword arguments (line 116)
kwargs_15179 = {}
# Getting the type of 'numpy' (line 116)
numpy_15175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 116)
dtype_15176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), numpy_15175, 'dtype')
# Calling dtype(args, kwargs) (line 116)
dtype_call_result_15180 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), dtype_15176, *[complex128_15178], **kwargs_15179)

# Getting the type of '_fftpack' (line 116)
_fftpack_15181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), '_fftpack')
# Obtaining the member 'zfftnd' of a type (line 116)
zfftnd_15182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 39), _fftpack_15181, 'zfftnd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 17), dict_15167, (dtype_call_result_15180, zfftnd_15182))
# Adding element type (key, value) (line 113)

# Call to dtype(...): (line 118)
# Processing the call arguments (line 118)
# Getting the type of 'numpy' (line 118)
numpy_15185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'numpy', False)
# Obtaining the member 'float32' of a type (line 118)
float32_15186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), numpy_15185, 'float32')
# Processing the call keyword arguments (line 118)
kwargs_15187 = {}
# Getting the type of 'numpy' (line 118)
numpy_15183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 118)
dtype_15184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), numpy_15183, 'dtype')
# Calling dtype(args, kwargs) (line 118)
dtype_call_result_15188 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), dtype_15184, *[float32_15186], **kwargs_15187)

# Getting the type of '_fake_cfftnd' (line 118)
_fake_cfftnd_15189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), '_fake_cfftnd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 17), dict_15167, (dtype_call_result_15188, _fake_cfftnd_15189))
# Adding element type (key, value) (line 113)

# Call to dtype(...): (line 119)
# Processing the call arguments (line 119)
# Getting the type of 'numpy' (line 119)
numpy_15192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'numpy', False)
# Obtaining the member 'float64' of a type (line 119)
float64_15193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), numpy_15192, 'float64')
# Processing the call keyword arguments (line 119)
kwargs_15194 = {}
# Getting the type of 'numpy' (line 119)
numpy_15190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'numpy', False)
# Obtaining the member 'dtype' of a type (line 119)
dtype_15191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), numpy_15190, 'dtype')
# Calling dtype(args, kwargs) (line 119)
dtype_call_result_15195 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), dtype_15191, *[float64_15193], **kwargs_15194)

# Getting the type of '_fftpack' (line 119)
_fftpack_15196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), '_fftpack')
# Obtaining the member 'zfftnd' of a type (line 119)
zfftnd_15197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 36), _fftpack_15196, 'zfftnd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 17), dict_15167, (dtype_call_result_15195, zfftnd_15197))

# Assigning a type to the variable '_DTYPE_TO_FFTN' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), '_DTYPE_TO_FFTN', dict_15167)

@norecursion
def _asfarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_asfarray'
    module_type_store = module_type_store.open_function_context('_asfarray', 123, 0, False)
    
    # Passed parameters checking function
    _asfarray.stypy_localization = localization
    _asfarray.stypy_type_of_self = None
    _asfarray.stypy_type_store = module_type_store
    _asfarray.stypy_function_name = '_asfarray'
    _asfarray.stypy_param_names_list = ['x']
    _asfarray.stypy_varargs_param_name = None
    _asfarray.stypy_kwargs_param_name = None
    _asfarray.stypy_call_defaults = defaults
    _asfarray.stypy_call_varargs = varargs
    _asfarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_asfarray', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_asfarray', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_asfarray(...)' code ##################

    str_15198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, (-1)), 'str', 'Like numpy asfarray, except that it does not modify x dtype if x is\n    already an array with a float dtype, and do not cast complex types to\n    real.')
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'x' (line 127)
    x_15200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'x', False)
    str_15201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'str', 'dtype')
    # Processing the call keyword arguments (line 127)
    kwargs_15202 = {}
    # Getting the type of 'hasattr' (line 127)
    hasattr_15199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 127)
    hasattr_call_result_15203 = invoke(stypy.reporting.localization.Localization(__file__, 127, 7), hasattr_15199, *[x_15200, str_15201], **kwargs_15202)
    
    
    # Getting the type of 'x' (line 127)
    x_15204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'x')
    # Obtaining the member 'dtype' of a type (line 127)
    dtype_15205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 31), x_15204, 'dtype')
    # Obtaining the member 'char' of a type (line 127)
    char_15206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 31), dtype_15205, 'char')
    
    # Obtaining the type of the subscript
    str_15207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 63), 'str', 'AllFloat')
    # Getting the type of 'numpy' (line 127)
    numpy_15208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 47), 'numpy')
    # Obtaining the member 'typecodes' of a type (line 127)
    typecodes_15209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 47), numpy_15208, 'typecodes')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___15210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 47), typecodes_15209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_15211 = invoke(stypy.reporting.localization.Localization(__file__, 127, 47), getitem___15210, str_15207)
    
    # Applying the binary operator 'in' (line 127)
    result_contains_15212 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 31), 'in', char_15206, subscript_call_result_15211)
    
    # Applying the binary operator 'and' (line 127)
    result_and_keyword_15213 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 7), 'and', hasattr_call_result_15203, result_contains_15212)
    
    # Testing the type of an if condition (line 127)
    if_condition_15214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 4), result_and_keyword_15213)
    # Assigning a type to the variable 'if_condition_15214' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'if_condition_15214', if_condition_15214)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'x' (line 131)
    x_15215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 131)
    dtype_15216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), x_15215, 'dtype')
    # Getting the type of 'numpy' (line 131)
    numpy_15217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'numpy')
    # Obtaining the member 'half' of a type (line 131)
    half_15218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 22), numpy_15217, 'half')
    # Applying the binary operator '==' (line 131)
    result_eq_15219 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 11), '==', dtype_15216, half_15218)
    
    # Testing the type of an if condition (line 131)
    if_condition_15220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 8), result_eq_15219)
    # Assigning a type to the variable 'if_condition_15220' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'if_condition_15220', if_condition_15220)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to asarray(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'x' (line 133)
    x_15223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'x', False)
    # Processing the call keyword arguments (line 133)
    # Getting the type of 'numpy' (line 133)
    numpy_15224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'numpy', False)
    # Obtaining the member 'float32' of a type (line 133)
    float32_15225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 42), numpy_15224, 'float32')
    keyword_15226 = float32_15225
    kwargs_15227 = {'dtype': keyword_15226}
    # Getting the type of 'numpy' (line 133)
    numpy_15221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 133)
    asarray_15222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 19), numpy_15221, 'asarray')
    # Calling asarray(args, kwargs) (line 133)
    asarray_call_result_15228 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), asarray_15222, *[x_15223], **kwargs_15227)
    
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'stypy_return_type', asarray_call_result_15228)
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to asarray(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'x' (line 134)
    x_15231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'x', False)
    # Processing the call keyword arguments (line 134)
    # Getting the type of 'x' (line 134)
    x_15232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'x', False)
    # Obtaining the member 'dtype' of a type (line 134)
    dtype_15233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 38), x_15232, 'dtype')
    keyword_15234 = dtype_15233
    kwargs_15235 = {'dtype': keyword_15234}
    # Getting the type of 'numpy' (line 134)
    numpy_15229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 134)
    asarray_15230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), numpy_15229, 'asarray')
    # Calling asarray(args, kwargs) (line 134)
    asarray_call_result_15236 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), asarray_15230, *[x_15231], **kwargs_15235)
    
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', asarray_call_result_15236)
    # SSA branch for the else part of an if statement (line 127)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to asarray(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'x' (line 138)
    x_15239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'x', False)
    # Processing the call keyword arguments (line 138)
    kwargs_15240 = {}
    # Getting the type of 'numpy' (line 138)
    numpy_15237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 138)
    asarray_15238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 14), numpy_15237, 'asarray')
    # Calling asarray(args, kwargs) (line 138)
    asarray_call_result_15241 = invoke(stypy.reporting.localization.Localization(__file__, 138, 14), asarray_15238, *[x_15239], **kwargs_15240)
    
    # Assigning a type to the variable 'ret' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'ret', asarray_call_result_15241)
    
    
    # Getting the type of 'ret' (line 139)
    ret_15242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'ret')
    # Obtaining the member 'dtype' of a type (line 139)
    dtype_15243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), ret_15242, 'dtype')
    # Getting the type of 'numpy' (line 139)
    numpy_15244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'numpy')
    # Obtaining the member 'half' of a type (line 139)
    half_15245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 24), numpy_15244, 'half')
    # Applying the binary operator '==' (line 139)
    result_eq_15246 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), '==', dtype_15243, half_15245)
    
    # Testing the type of an if condition (line 139)
    if_condition_15247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_eq_15246)
    # Assigning a type to the variable 'if_condition_15247' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_15247', if_condition_15247)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to asarray(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'ret' (line 140)
    ret_15250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 33), 'ret', False)
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'numpy' (line 140)
    numpy_15251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'numpy', False)
    # Obtaining the member 'float32' of a type (line 140)
    float32_15252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 44), numpy_15251, 'float32')
    keyword_15253 = float32_15252
    kwargs_15254 = {'dtype': keyword_15253}
    # Getting the type of 'numpy' (line 140)
    numpy_15248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 140)
    asarray_15249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), numpy_15248, 'asarray')
    # Calling asarray(args, kwargs) (line 140)
    asarray_call_result_15255 = invoke(stypy.reporting.localization.Localization(__file__, 140, 19), asarray_15249, *[ret_15250], **kwargs_15254)
    
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'stypy_return_type', asarray_call_result_15255)
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ret' (line 141)
    ret_15256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'ret')
    # Obtaining the member 'dtype' of a type (line 141)
    dtype_15257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), ret_15256, 'dtype')
    # Obtaining the member 'char' of a type (line 141)
    char_15258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), dtype_15257, 'char')
    
    # Obtaining the type of the subscript
    str_15259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 51), 'str', 'AllFloat')
    # Getting the type of 'numpy' (line 141)
    numpy_15260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 35), 'numpy')
    # Obtaining the member 'typecodes' of a type (line 141)
    typecodes_15261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 35), numpy_15260, 'typecodes')
    # Obtaining the member '__getitem__' of a type (line 141)
    getitem___15262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 35), typecodes_15261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 141)
    subscript_call_result_15263 = invoke(stypy.reporting.localization.Localization(__file__, 141, 35), getitem___15262, str_15259)
    
    # Applying the binary operator 'notin' (line 141)
    result_contains_15264 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 13), 'notin', char_15258, subscript_call_result_15263)
    
    # Testing the type of an if condition (line 141)
    if_condition_15265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 13), result_contains_15264)
    # Assigning a type to the variable 'if_condition_15265' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'if_condition_15265', if_condition_15265)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to asfarray(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'x' (line 142)
    x_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'x', False)
    # Processing the call keyword arguments (line 142)
    kwargs_15269 = {}
    # Getting the type of 'numpy' (line 142)
    numpy_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'numpy', False)
    # Obtaining the member 'asfarray' of a type (line 142)
    asfarray_15267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), numpy_15266, 'asfarray')
    # Calling asfarray(args, kwargs) (line 142)
    asfarray_call_result_15270 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), asfarray_15267, *[x_15268], **kwargs_15269)
    
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'stypy_return_type', asfarray_call_result_15270)
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 143)
    ret_15271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', ret_15271)
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_asfarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_asfarray' in the type store
    # Getting the type of 'stypy_return_type' (line 123)
    stypy_return_type_15272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15272)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_asfarray'
    return stypy_return_type_15272

# Assigning a type to the variable '_asfarray' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), '_asfarray', _asfarray)

@norecursion
def _fix_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fix_shape'
    module_type_store = module_type_store.open_function_context('_fix_shape', 146, 0, False)
    
    # Passed parameters checking function
    _fix_shape.stypy_localization = localization
    _fix_shape.stypy_type_of_self = None
    _fix_shape.stypy_type_store = module_type_store
    _fix_shape.stypy_function_name = '_fix_shape'
    _fix_shape.stypy_param_names_list = ['x', 'n', 'axis']
    _fix_shape.stypy_varargs_param_name = None
    _fix_shape.stypy_kwargs_param_name = None
    _fix_shape.stypy_call_defaults = defaults
    _fix_shape.stypy_call_varargs = varargs
    _fix_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fix_shape', ['x', 'n', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fix_shape', localization, ['x', 'n', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fix_shape(...)' code ##################

    str_15273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 4), 'str', ' Internal auxiliary function for _raw_fft, _raw_fftnd.')
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to list(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'x' (line 148)
    x_15275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), 'x', False)
    # Obtaining the member 'shape' of a type (line 148)
    shape_15276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 13), x_15275, 'shape')
    # Processing the call keyword arguments (line 148)
    kwargs_15277 = {}
    # Getting the type of 'list' (line 148)
    list_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'list', False)
    # Calling list(args, kwargs) (line 148)
    list_call_result_15278 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), list_15274, *[shape_15276], **kwargs_15277)
    
    # Assigning a type to the variable 's' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 's', list_call_result_15278)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 149)
    axis_15279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 9), 'axis')
    # Getting the type of 's' (line 149)
    s_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 's')
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___15281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 7), s_15280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_15282 = invoke(stypy.reporting.localization.Localization(__file__, 149, 7), getitem___15281, axis_15279)
    
    # Getting the type of 'n' (line 149)
    n_15283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'n')
    # Applying the binary operator '>' (line 149)
    result_gt_15284 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), '>', subscript_call_result_15282, n_15283)
    
    # Testing the type of an if condition (line 149)
    if_condition_15285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_gt_15284)
    # Assigning a type to the variable 'if_condition_15285' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_15285', if_condition_15285)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 150):
    
    # Assigning a BinOp to a Name (line 150):
    
    # Obtaining an instance of the builtin type 'list' (line 150)
    list_15286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 150)
    # Adding element type (line 150)
    
    # Call to slice(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'None' (line 150)
    None_15288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 23), 'None', False)
    # Processing the call keyword arguments (line 150)
    kwargs_15289 = {}
    # Getting the type of 'slice' (line 150)
    slice_15287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 150)
    slice_call_result_15290 = invoke(stypy.reporting.localization.Localization(__file__, 150, 17), slice_15287, *[None_15288], **kwargs_15289)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 16), list_15286, slice_call_result_15290)
    
    
    # Call to len(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 's' (line 150)
    s_15292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 's', False)
    # Processing the call keyword arguments (line 150)
    kwargs_15293 = {}
    # Getting the type of 'len' (line 150)
    len_15291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'len', False)
    # Calling len(args, kwargs) (line 150)
    len_call_result_15294 = invoke(stypy.reporting.localization.Localization(__file__, 150, 30), len_15291, *[s_15292], **kwargs_15293)
    
    # Applying the binary operator '*' (line 150)
    result_mul_15295 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), '*', list_15286, len_call_result_15294)
    
    # Assigning a type to the variable 'index' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'index', result_mul_15295)
    
    # Assigning a Call to a Subscript (line 151):
    
    # Assigning a Call to a Subscript (line 151):
    
    # Call to slice(...): (line 151)
    # Processing the call arguments (line 151)
    int_15297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'int')
    # Getting the type of 'n' (line 151)
    n_15298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'n', False)
    # Processing the call keyword arguments (line 151)
    kwargs_15299 = {}
    # Getting the type of 'slice' (line 151)
    slice_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 151)
    slice_call_result_15300 = invoke(stypy.reporting.localization.Localization(__file__, 151, 22), slice_15296, *[int_15297, n_15298], **kwargs_15299)
    
    # Getting the type of 'index' (line 151)
    index_15301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'index')
    # Getting the type of 'axis' (line 151)
    axis_15302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'axis')
    # Storing an element on a container (line 151)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 8), index_15301, (axis_15302, slice_call_result_15300))
    
    # Assigning a Subscript to a Name (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 152)
    index_15303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'index')
    # Getting the type of 'x' (line 152)
    x_15304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___15305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), x_15304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_15306 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), getitem___15305, index_15303)
    
    # Assigning a type to the variable 'x' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'x', subscript_call_result_15306)
    
    # Obtaining an instance of the builtin type 'tuple' (line 153)
    tuple_15307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 153)
    # Adding element type (line 153)
    # Getting the type of 'x' (line 153)
    x_15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), tuple_15307, x_15308)
    # Adding element type (line 153)
    # Getting the type of 'False' (line 153)
    False_15309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 15), tuple_15307, False_15309)
    
    # Assigning a type to the variable 'stypy_return_type' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', tuple_15307)
    # SSA branch for the else part of an if statement (line 149)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 155):
    
    # Assigning a BinOp to a Name (line 155):
    
    # Obtaining an instance of the builtin type 'list' (line 155)
    list_15310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 155)
    # Adding element type (line 155)
    
    # Call to slice(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'None' (line 155)
    None_15312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'None', False)
    # Processing the call keyword arguments (line 155)
    kwargs_15313 = {}
    # Getting the type of 'slice' (line 155)
    slice_15311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'slice', False)
    # Calling slice(args, kwargs) (line 155)
    slice_call_result_15314 = invoke(stypy.reporting.localization.Localization(__file__, 155, 17), slice_15311, *[None_15312], **kwargs_15313)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 16), list_15310, slice_call_result_15314)
    
    
    # Call to len(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 's' (line 155)
    s_15316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 's', False)
    # Processing the call keyword arguments (line 155)
    kwargs_15317 = {}
    # Getting the type of 'len' (line 155)
    len_15315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'len', False)
    # Calling len(args, kwargs) (line 155)
    len_call_result_15318 = invoke(stypy.reporting.localization.Localization(__file__, 155, 30), len_15315, *[s_15316], **kwargs_15317)
    
    # Applying the binary operator '*' (line 155)
    result_mul_15319 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 16), '*', list_15310, len_call_result_15318)
    
    # Assigning a type to the variable 'index' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'index', result_mul_15319)
    
    # Assigning a Call to a Subscript (line 156):
    
    # Assigning a Call to a Subscript (line 156):
    
    # Call to slice(...): (line 156)
    # Processing the call arguments (line 156)
    int_15321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 156)
    axis_15322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'axis', False)
    # Getting the type of 's' (line 156)
    s_15323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 's', False)
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___15324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 30), s_15323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_15325 = invoke(stypy.reporting.localization.Localization(__file__, 156, 30), getitem___15324, axis_15322)
    
    # Processing the call keyword arguments (line 156)
    kwargs_15326 = {}
    # Getting the type of 'slice' (line 156)
    slice_15320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'slice', False)
    # Calling slice(args, kwargs) (line 156)
    slice_call_result_15327 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), slice_15320, *[int_15321, subscript_call_result_15325], **kwargs_15326)
    
    # Getting the type of 'index' (line 156)
    index_15328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'index')
    # Getting the type of 'axis' (line 156)
    axis_15329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'axis')
    # Storing an element on a container (line 156)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 8), index_15328, (axis_15329, slice_call_result_15327))
    
    # Assigning a Name to a Subscript (line 157):
    
    # Assigning a Name to a Subscript (line 157):
    # Getting the type of 'n' (line 157)
    n_15330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'n')
    # Getting the type of 's' (line 157)
    s_15331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 's')
    # Getting the type of 'axis' (line 157)
    axis_15332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 10), 'axis')
    # Storing an element on a container (line 157)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), s_15331, (axis_15332, n_15330))
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to zeros(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 's' (line 158)
    s_15334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 's', False)
    # Getting the type of 'x' (line 158)
    x_15335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'x', False)
    # Obtaining the member 'dtype' of a type (line 158)
    dtype_15336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), x_15335, 'dtype')
    # Obtaining the member 'char' of a type (line 158)
    char_15337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 20), dtype_15336, 'char')
    # Processing the call keyword arguments (line 158)
    kwargs_15338 = {}
    # Getting the type of 'zeros' (line 158)
    zeros_15333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'zeros', False)
    # Calling zeros(args, kwargs) (line 158)
    zeros_call_result_15339 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), zeros_15333, *[s_15334, char_15337], **kwargs_15338)
    
    # Assigning a type to the variable 'z' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'z', zeros_call_result_15339)
    
    # Assigning a Name to a Subscript (line 159):
    
    # Assigning a Name to a Subscript (line 159):
    # Getting the type of 'x' (line 159)
    x_15340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'x')
    # Getting the type of 'z' (line 159)
    z_15341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'z')
    # Getting the type of 'index' (line 159)
    index_15342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 10), 'index')
    # Storing an element on a container (line 159)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 8), z_15341, (index_15342, x_15340))
    
    # Obtaining an instance of the builtin type 'tuple' (line 160)
    tuple_15343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 160)
    # Adding element type (line 160)
    # Getting the type of 'z' (line 160)
    z_15344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'z')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_15343, z_15344)
    # Adding element type (line 160)
    # Getting the type of 'True' (line 160)
    True_15345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 15), tuple_15343, True_15345)
    
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', tuple_15343)
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_fix_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fix_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_15346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15346)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fix_shape'
    return stypy_return_type_15346

# Assigning a type to the variable '_fix_shape' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), '_fix_shape', _fix_shape)

@norecursion
def _raw_fft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raw_fft'
    module_type_store = module_type_store.open_function_context('_raw_fft', 163, 0, False)
    
    # Passed parameters checking function
    _raw_fft.stypy_localization = localization
    _raw_fft.stypy_type_of_self = None
    _raw_fft.stypy_type_store = module_type_store
    _raw_fft.stypy_function_name = '_raw_fft'
    _raw_fft.stypy_param_names_list = ['x', 'n', 'axis', 'direction', 'overwrite_x', 'work_function']
    _raw_fft.stypy_varargs_param_name = None
    _raw_fft.stypy_kwargs_param_name = None
    _raw_fft.stypy_call_defaults = defaults
    _raw_fft.stypy_call_varargs = varargs
    _raw_fft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_fft', ['x', 'n', 'axis', 'direction', 'overwrite_x', 'work_function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_fft', localization, ['x', 'n', 'axis', 'direction', 'overwrite_x', 'work_function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_fft(...)' code ##################

    str_15347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 4), 'str', ' Internal auxiliary function for fft, ifft, rfft, irfft.')
    
    # Type idiom detected: calculating its left and rigth part (line 165)
    # Getting the type of 'n' (line 165)
    n_15348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 7), 'n')
    # Getting the type of 'None' (line 165)
    None_15349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'None')
    
    (may_be_15350, more_types_in_union_15351) = may_be_none(n_15348, None_15349)

    if may_be_15350:

        if more_types_in_union_15351:
            # Runtime conditional SSA (line 165)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 166):
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 166)
        axis_15352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'axis')
        # Getting the type of 'x' (line 166)
        x_15353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'x')
        # Obtaining the member 'shape' of a type (line 166)
        shape_15354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), x_15353, 'shape')
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___15355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), shape_15354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_15356 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), getitem___15355, axis_15352)
        
        # Assigning a type to the variable 'n' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'n', subscript_call_result_15356)

        if more_types_in_union_15351:
            # Runtime conditional SSA for else branch (line 165)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15350) or more_types_in_union_15351):
        
        
        # Getting the type of 'n' (line 167)
        n_15357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 9), 'n')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 167)
        axis_15358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'axis')
        # Getting the type of 'x' (line 167)
        x_15359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 14), 'x')
        # Obtaining the member 'shape' of a type (line 167)
        shape_15360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 14), x_15359, 'shape')
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___15361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 14), shape_15360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_15362 = invoke(stypy.reporting.localization.Localization(__file__, 167, 14), getitem___15361, axis_15358)
        
        # Applying the binary operator '!=' (line 167)
        result_ne_15363 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 9), '!=', n_15357, subscript_call_result_15362)
        
        # Testing the type of an if condition (line 167)
        if_condition_15364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 9), result_ne_15363)
        # Assigning a type to the variable 'if_condition_15364' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 9), 'if_condition_15364', if_condition_15364)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 168):
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_15365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to _fix_shape(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'x' (line 168)
        x_15367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'x', False)
        # Getting the type of 'n' (line 168)
        n_15368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'n', False)
        # Getting the type of 'axis' (line 168)
        axis_15369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'axis', False)
        # Processing the call keyword arguments (line 168)
        kwargs_15370 = {}
        # Getting the type of '_fix_shape' (line 168)
        _fix_shape_15366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 168)
        _fix_shape_call_result_15371 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), _fix_shape_15366, *[x_15367, n_15368, axis_15369], **kwargs_15370)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___15372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), _fix_shape_call_result_15371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_15373 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___15372, int_15365)
        
        # Assigning a type to the variable 'tuple_var_assignment_14879' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_14879', subscript_call_result_15373)
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_15374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to _fix_shape(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'x' (line 168)
        x_15376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'x', False)
        # Getting the type of 'n' (line 168)
        n_15377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'n', False)
        # Getting the type of 'axis' (line 168)
        axis_15378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'axis', False)
        # Processing the call keyword arguments (line 168)
        kwargs_15379 = {}
        # Getting the type of '_fix_shape' (line 168)
        _fix_shape_15375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 168)
        _fix_shape_call_result_15380 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), _fix_shape_15375, *[x_15376, n_15377, axis_15378], **kwargs_15379)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___15381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), _fix_shape_call_result_15380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_15382 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___15381, int_15374)
        
        # Assigning a type to the variable 'tuple_var_assignment_14880' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_14880', subscript_call_result_15382)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_14879' (line 168)
        tuple_var_assignment_14879_15383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_14879')
        # Assigning a type to the variable 'x' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'x', tuple_var_assignment_14879_15383)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_14880' (line 168)
        tuple_var_assignment_14880_15384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_14880')
        # Assigning a type to the variable 'copy_made' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'copy_made', tuple_var_assignment_14880_15384)
        
        # Assigning a BoolOp to a Name (line 169):
        
        # Assigning a BoolOp to a Name (line 169):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_x' (line 169)
        overwrite_x_15385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'overwrite_x')
        # Getting the type of 'copy_made' (line 169)
        copy_made_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'copy_made')
        # Applying the binary operator 'or' (line 169)
        result_or_keyword_15387 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 22), 'or', overwrite_x_15385, copy_made_15386)
        
        # Assigning a type to the variable 'overwrite_x' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'overwrite_x', result_or_keyword_15387)
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_15350 and more_types_in_union_15351):
            # SSA join for if statement (line 165)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'n' (line 171)
    n_15388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'n')
    int_15389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 11), 'int')
    # Applying the binary operator '<' (line 171)
    result_lt_15390 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '<', n_15388, int_15389)
    
    # Testing the type of an if condition (line 171)
    if_condition_15391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_lt_15390)
    # Assigning a type to the variable 'if_condition_15391' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_15391', if_condition_15391)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 172)
    # Processing the call arguments (line 172)
    str_15393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'str', 'Invalid number of FFT data points (%d) specified.')
    # Getting the type of 'n' (line 173)
    n_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 45), 'n', False)
    # Applying the binary operator '%' (line 172)
    result_mod_15395 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 25), '%', str_15393, n_15394)
    
    # Processing the call keyword arguments (line 172)
    kwargs_15396 = {}
    # Getting the type of 'ValueError' (line 172)
    ValueError_15392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 172)
    ValueError_call_result_15397 = invoke(stypy.reporting.localization.Localization(__file__, 172, 14), ValueError_15392, *[result_mod_15395], **kwargs_15396)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 172, 8), ValueError_call_result_15397, 'raise parameter', BaseException)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 175)
    axis_15398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'axis')
    int_15399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 15), 'int')
    # Applying the binary operator '==' (line 175)
    result_eq_15400 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 7), '==', axis_15398, int_15399)
    
    
    # Getting the type of 'axis' (line 175)
    axis_15401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'axis')
    
    # Call to len(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'x' (line 175)
    x_15403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 33), 'x', False)
    # Obtaining the member 'shape' of a type (line 175)
    shape_15404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 33), x_15403, 'shape')
    # Processing the call keyword arguments (line 175)
    kwargs_15405 = {}
    # Getting the type of 'len' (line 175)
    len_15402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), 'len', False)
    # Calling len(args, kwargs) (line 175)
    len_call_result_15406 = invoke(stypy.reporting.localization.Localization(__file__, 175, 29), len_15402, *[shape_15404], **kwargs_15405)
    
    int_15407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'int')
    # Applying the binary operator '-' (line 175)
    result_sub_15408 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 29), '-', len_call_result_15406, int_15407)
    
    # Applying the binary operator '==' (line 175)
    result_eq_15409 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 21), '==', axis_15401, result_sub_15408)
    
    # Applying the binary operator 'or' (line 175)
    result_or_keyword_15410 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 7), 'or', result_eq_15400, result_eq_15409)
    
    # Testing the type of an if condition (line 175)
    if_condition_15411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 4), result_or_keyword_15410)
    # Assigning a type to the variable 'if_condition_15411' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'if_condition_15411', if_condition_15411)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to work_function(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'x' (line 176)
    x_15413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'x', False)
    # Getting the type of 'n' (line 176)
    n_15414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'n', False)
    # Getting the type of 'direction' (line 176)
    direction_15415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'direction', False)
    # Processing the call keyword arguments (line 176)
    # Getting the type of 'overwrite_x' (line 176)
    overwrite_x_15416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 52), 'overwrite_x', False)
    keyword_15417 = overwrite_x_15416
    kwargs_15418 = {'overwrite_x': keyword_15417}
    # Getting the type of 'work_function' (line 176)
    work_function_15412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'work_function', False)
    # Calling work_function(args, kwargs) (line 176)
    work_function_call_result_15419 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), work_function_15412, *[x_15413, n_15414, direction_15415], **kwargs_15418)
    
    # Assigning a type to the variable 'r' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'r', work_function_call_result_15419)
    # SSA branch for the else part of an if statement (line 175)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to swapaxes(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'x' (line 178)
    x_15421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'x', False)
    # Getting the type of 'axis' (line 178)
    axis_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'axis', False)
    int_15423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'int')
    # Processing the call keyword arguments (line 178)
    kwargs_15424 = {}
    # Getting the type of 'swapaxes' (line 178)
    swapaxes_15420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 178)
    swapaxes_call_result_15425 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), swapaxes_15420, *[x_15421, axis_15422, int_15423], **kwargs_15424)
    
    # Assigning a type to the variable 'x' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'x', swapaxes_call_result_15425)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to work_function(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'x' (line 179)
    x_15427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 26), 'x', False)
    # Getting the type of 'n' (line 179)
    n_15428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'n', False)
    # Getting the type of 'direction' (line 179)
    direction_15429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'direction', False)
    # Processing the call keyword arguments (line 179)
    # Getting the type of 'overwrite_x' (line 179)
    overwrite_x_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 52), 'overwrite_x', False)
    keyword_15431 = overwrite_x_15430
    kwargs_15432 = {'overwrite_x': keyword_15431}
    # Getting the type of 'work_function' (line 179)
    work_function_15426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'work_function', False)
    # Calling work_function(args, kwargs) (line 179)
    work_function_call_result_15433 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), work_function_15426, *[x_15427, n_15428, direction_15429], **kwargs_15432)
    
    # Assigning a type to the variable 'r' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'r', work_function_call_result_15433)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to swapaxes(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'r' (line 180)
    r_15435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'r', False)
    # Getting the type of 'axis' (line 180)
    axis_15436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'axis', False)
    int_15437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 30), 'int')
    # Processing the call keyword arguments (line 180)
    kwargs_15438 = {}
    # Getting the type of 'swapaxes' (line 180)
    swapaxes_15434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 180)
    swapaxes_call_result_15439 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), swapaxes_15434, *[r_15435, axis_15436, int_15437], **kwargs_15438)
    
    # Assigning a type to the variable 'r' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'r', swapaxes_call_result_15439)
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 181)
    r_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type', r_15440)
    
    # ################# End of '_raw_fft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_fft' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_15441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_fft'
    return stypy_return_type_15441

# Assigning a type to the variable '_raw_fft' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), '_raw_fft', _raw_fft)

@norecursion
def fft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 184)
    None_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'None')
    int_15443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 24), 'int')
    # Getting the type of 'False' (line 184)
    False_15444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 40), 'False')
    defaults = [None_15442, int_15443, False_15444]
    # Create a new context for function 'fft'
    module_type_store = module_type_store.open_function_context('fft', 184, 0, False)
    
    # Passed parameters checking function
    fft.stypy_localization = localization
    fft.stypy_type_of_self = None
    fft.stypy_type_store = module_type_store
    fft.stypy_function_name = 'fft'
    fft.stypy_param_names_list = ['x', 'n', 'axis', 'overwrite_x']
    fft.stypy_varargs_param_name = None
    fft.stypy_kwargs_param_name = None
    fft.stypy_call_defaults = defaults
    fft.stypy_call_varargs = varargs
    fft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fft', ['x', 'n', 'axis', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fft', localization, ['x', 'n', 'axis', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fft(...)' code ##################

    str_15445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, (-1)), 'str', '\n    Return discrete Fourier transform of real or complex sequence.\n\n    The returned complex array contains ``y(0), y(1),..., y(n-1)`` where\n\n    ``y(j) = (x * exp(-2*pi*sqrt(-1)*j*np.arange(n)/n)).sum()``.\n\n    Parameters\n    ----------\n    x : array_like\n        Array to Fourier transform.\n    n : int, optional\n        Length of the Fourier transform.  If ``n < x.shape[axis]``, `x` is\n        truncated.  If ``n > x.shape[axis]``, `x` is zero-padded. The\n        default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the fft\'s are computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    z : complex ndarray\n        with the elements::\n\n            [y(0),y(1),..,y(n/2),y(1-n/2),...,y(-1)]        if n is even\n            [y(0),y(1),..,y((n-1)/2),y(-(n-1)/2),...,y(-1)]  if n is odd\n\n        where::\n\n            y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k* 2*pi/n), j = 0..n-1\n\n    See Also\n    --------\n    ifft : Inverse FFT\n    rfft : FFT of a real sequence\n\n    Notes\n    -----\n    The packing of the result is "standard": If ``A = fft(a, n)``, then\n    ``A[0]`` contains the zero-frequency term, ``A[1:n/2]`` contains the\n    positive-frequency terms, and ``A[n/2:]`` contains the negative-frequency\n    terms, in order of decreasingly negative frequency. So for an 8-point\n    transform, the frequencies of the result are [0, 1, 2, 3, -4, -3, -2, -1].\n    To rearrange the fft output so that the zero-frequency component is\n    centered, like [-4, -3, -2, -1,  0,  1,  2,  3], use `fftshift`.\n\n    Both single and double precision routines are implemented.  Half precision\n    inputs will be converted to single precision.  Non floating-point inputs\n    will be converted to double precision.  Long-double precision inputs are\n    not supported.\n\n    This function is most efficient when `n` is a power of two, and least\n    efficient when `n` is prime.\n\n    Note that if ``x`` is real-valued then ``A[j] == A[n-j].conjugate()``.\n    If ``x`` is real-valued and ``n`` is even then ``A[n/2]`` is real.\n\n    If the data type of `x` is real, a "real FFT" algorithm is automatically\n    used, which roughly halves the computation time.  To increase efficiency\n    a little further, use `rfft`, which does the same calculation, but only\n    outputs half of the symmetrical spectrum.  If the data is both real and\n    symmetrical, the `dct` can again double the efficiency, by generating\n    half of the spectrum from half of the signal.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import fft, ifft\n    >>> x = np.arange(5)\n    >>> np.allclose(fft(ifft(x)), x, atol=1e-15)  # within numerical accuracy.\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to _asfarray(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'x' (line 259)
    x_15447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'x', False)
    # Processing the call keyword arguments (line 259)
    kwargs_15448 = {}
    # Getting the type of '_asfarray' (line 259)
    _asfarray_15446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 259)
    _asfarray_call_result_15449 = invoke(stypy.reporting.localization.Localization(__file__, 259, 10), _asfarray_15446, *[x_15447], **kwargs_15448)
    
    # Assigning a type to the variable 'tmp' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'tmp', _asfarray_call_result_15449)
    
    
    # SSA begins for try-except statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 262):
    
    # Assigning a Subscript to a Name (line 262):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tmp' (line 262)
    tmp_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'tmp')
    # Obtaining the member 'dtype' of a type (line 262)
    dtype_15451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 38), tmp_15450, 'dtype')
    # Getting the type of '_DTYPE_TO_FFT' (line 262)
    _DTYPE_TO_FFT_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), '_DTYPE_TO_FFT')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 24), _DTYPE_TO_FFT_15452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_15454 = invoke(stypy.reporting.localization.Localization(__file__, 262, 24), getitem___15453, dtype_15451)
    
    # Assigning a type to the variable 'work_function' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'work_function', subscript_call_result_15454)
    # SSA branch for the except part of a try statement (line 261)
    # SSA branch for the except 'KeyError' branch of a try statement (line 261)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 264)
    # Processing the call arguments (line 264)
    str_15456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 25), 'str', 'type %s is not supported')
    # Getting the type of 'tmp' (line 264)
    tmp_15457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 54), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 264)
    dtype_15458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 54), tmp_15457, 'dtype')
    # Applying the binary operator '%' (line 264)
    result_mod_15459 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 25), '%', str_15456, dtype_15458)
    
    # Processing the call keyword arguments (line 264)
    kwargs_15460 = {}
    # Getting the type of 'ValueError' (line 264)
    ValueError_15455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 264)
    ValueError_call_result_15461 = invoke(stypy.reporting.localization.Localization(__file__, 264, 14), ValueError_15455, *[result_mod_15459], **kwargs_15460)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 264, 8), ValueError_call_result_15461, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to istype(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'tmp' (line 266)
    tmp_15463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'tmp', False)
    # Getting the type of 'numpy' (line 266)
    numpy_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 266)
    complex64_15465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 24), numpy_15464, 'complex64')
    # Processing the call keyword arguments (line 266)
    kwargs_15466 = {}
    # Getting the type of 'istype' (line 266)
    istype_15462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'istype', False)
    # Calling istype(args, kwargs) (line 266)
    istype_call_result_15467 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), istype_15462, *[tmp_15463, complex64_15465], **kwargs_15466)
    
    
    # Call to istype(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'tmp' (line 266)
    tmp_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 51), 'tmp', False)
    # Getting the type of 'numpy' (line 266)
    numpy_15470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 56), 'numpy', False)
    # Obtaining the member 'complex128' of a type (line 266)
    complex128_15471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 56), numpy_15470, 'complex128')
    # Processing the call keyword arguments (line 266)
    kwargs_15472 = {}
    # Getting the type of 'istype' (line 266)
    istype_15468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 44), 'istype', False)
    # Calling istype(args, kwargs) (line 266)
    istype_call_result_15473 = invoke(stypy.reporting.localization.Localization(__file__, 266, 44), istype_15468, *[tmp_15469, complex128_15471], **kwargs_15472)
    
    # Applying the binary operator 'or' (line 266)
    result_or_keyword_15474 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 12), 'or', istype_call_result_15467, istype_call_result_15473)
    
    # Applying the 'not' unary operator (line 266)
    result_not__15475 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 7), 'not', result_or_keyword_15474)
    
    # Testing the type of an if condition (line 266)
    if_condition_15476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 4), result_not__15475)
    # Assigning a type to the variable 'if_condition_15476' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'if_condition_15476', if_condition_15476)
    # SSA begins for if statement (line 266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 267):
    
    # Assigning a Num to a Name (line 267):
    int_15477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 22), 'int')
    # Assigning a type to the variable 'overwrite_x' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'overwrite_x', int_15477)
    # SSA join for if statement (line 266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 269):
    
    # Assigning a BoolOp to a Name (line 269):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 269)
    overwrite_x_15478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'overwrite_x')
    
    # Call to _datacopied(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'tmp' (line 269)
    tmp_15480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'tmp', False)
    # Getting the type of 'x' (line 269)
    x_15481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 50), 'x', False)
    # Processing the call keyword arguments (line 269)
    kwargs_15482 = {}
    # Getting the type of '_datacopied' (line 269)
    _datacopied_15479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 269)
    _datacopied_call_result_15483 = invoke(stypy.reporting.localization.Localization(__file__, 269, 33), _datacopied_15479, *[tmp_15480, x_15481], **kwargs_15482)
    
    # Applying the binary operator 'or' (line 269)
    result_or_keyword_15484 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 18), 'or', overwrite_x_15478, _datacopied_call_result_15483)
    
    # Assigning a type to the variable 'overwrite_x' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'overwrite_x', result_or_keyword_15484)
    
    # Type idiom detected: calculating its left and rigth part (line 271)
    # Getting the type of 'n' (line 271)
    n_15485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 7), 'n')
    # Getting the type of 'None' (line 271)
    None_15486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'None')
    
    (may_be_15487, more_types_in_union_15488) = may_be_none(n_15485, None_15486)

    if may_be_15487:

        if more_types_in_union_15488:
            # Runtime conditional SSA (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 272):
        
        # Assigning a Subscript to a Name (line 272):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 272)
        axis_15489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'axis')
        # Getting the type of 'tmp' (line 272)
        tmp_15490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'tmp')
        # Obtaining the member 'shape' of a type (line 272)
        shape_15491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), tmp_15490, 'shape')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___15492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), shape_15491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_15493 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), getitem___15492, axis_15489)
        
        # Assigning a type to the variable 'n' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'n', subscript_call_result_15493)

        if more_types_in_union_15488:
            # Runtime conditional SSA for else branch (line 271)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15487) or more_types_in_union_15488):
        
        
        # Getting the type of 'n' (line 273)
        n_15494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'n')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 273)
        axis_15495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'axis')
        # Getting the type of 'tmp' (line 273)
        tmp_15496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 14), 'tmp')
        # Obtaining the member 'shape' of a type (line 273)
        shape_15497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 14), tmp_15496, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___15498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 14), shape_15497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_15499 = invoke(stypy.reporting.localization.Localization(__file__, 273, 14), getitem___15498, axis_15495)
        
        # Applying the binary operator '!=' (line 273)
        result_ne_15500 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 9), '!=', n_15494, subscript_call_result_15499)
        
        # Testing the type of an if condition (line 273)
        if_condition_15501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 9), result_ne_15500)
        # Assigning a type to the variable 'if_condition_15501' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'if_condition_15501', if_condition_15501)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 274):
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_15502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Call to _fix_shape(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'tmp' (line 274)
        tmp_15504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'tmp', False)
        # Getting the type of 'n' (line 274)
        n_15505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 40), 'n', False)
        # Getting the type of 'axis' (line 274)
        axis_15506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'axis', False)
        # Processing the call keyword arguments (line 274)
        kwargs_15507 = {}
        # Getting the type of '_fix_shape' (line 274)
        _fix_shape_15503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 25), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 274)
        _fix_shape_call_result_15508 = invoke(stypy.reporting.localization.Localization(__file__, 274, 25), _fix_shape_15503, *[tmp_15504, n_15505, axis_15506], **kwargs_15507)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___15509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), _fix_shape_call_result_15508, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_15510 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___15509, int_15502)
        
        # Assigning a type to the variable 'tuple_var_assignment_14881' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_14881', subscript_call_result_15510)
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        int_15511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 8), 'int')
        
        # Call to _fix_shape(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'tmp' (line 274)
        tmp_15513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 36), 'tmp', False)
        # Getting the type of 'n' (line 274)
        n_15514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 40), 'n', False)
        # Getting the type of 'axis' (line 274)
        axis_15515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'axis', False)
        # Processing the call keyword arguments (line 274)
        kwargs_15516 = {}
        # Getting the type of '_fix_shape' (line 274)
        _fix_shape_15512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 25), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 274)
        _fix_shape_call_result_15517 = invoke(stypy.reporting.localization.Localization(__file__, 274, 25), _fix_shape_15512, *[tmp_15513, n_15514, axis_15515], **kwargs_15516)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___15518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), _fix_shape_call_result_15517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_15519 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), getitem___15518, int_15511)
        
        # Assigning a type to the variable 'tuple_var_assignment_14882' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_14882', subscript_call_result_15519)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_14881' (line 274)
        tuple_var_assignment_14881_15520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_14881')
        # Assigning a type to the variable 'tmp' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tmp', tuple_var_assignment_14881_15520)
        
        # Assigning a Name to a Name (line 274):
        # Getting the type of 'tuple_var_assignment_14882' (line 274)
        tuple_var_assignment_14882_15521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'tuple_var_assignment_14882')
        # Assigning a type to the variable 'copy_made' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'copy_made', tuple_var_assignment_14882_15521)
        
        # Assigning a BoolOp to a Name (line 275):
        
        # Assigning a BoolOp to a Name (line 275):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_x' (line 275)
        overwrite_x_15522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'overwrite_x')
        # Getting the type of 'copy_made' (line 275)
        copy_made_15523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 37), 'copy_made')
        # Applying the binary operator 'or' (line 275)
        result_or_keyword_15524 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 22), 'or', overwrite_x_15522, copy_made_15523)
        
        # Assigning a type to the variable 'overwrite_x' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'overwrite_x', result_or_keyword_15524)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_15487 and more_types_in_union_15488):
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'n' (line 277)
    n_15525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'n')
    int_15526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 11), 'int')
    # Applying the binary operator '<' (line 277)
    result_lt_15527 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 7), '<', n_15525, int_15526)
    
    # Testing the type of an if condition (line 277)
    if_condition_15528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), result_lt_15527)
    # Assigning a type to the variable 'if_condition_15528' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_15528', if_condition_15528)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 278)
    # Processing the call arguments (line 278)
    str_15530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'str', 'Invalid number of FFT data points (%d) specified.')
    # Getting the type of 'n' (line 279)
    n_15531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 45), 'n', False)
    # Applying the binary operator '%' (line 278)
    result_mod_15532 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 25), '%', str_15530, n_15531)
    
    # Processing the call keyword arguments (line 278)
    kwargs_15533 = {}
    # Getting the type of 'ValueError' (line 278)
    ValueError_15529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 278)
    ValueError_call_result_15534 = invoke(stypy.reporting.localization.Localization(__file__, 278, 14), ValueError_15529, *[result_mod_15532], **kwargs_15533)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 278, 8), ValueError_call_result_15534, 'raise parameter', BaseException)
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 281)
    axis_15535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'axis')
    int_15536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'int')
    # Applying the binary operator '==' (line 281)
    result_eq_15537 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), '==', axis_15535, int_15536)
    
    
    # Getting the type of 'axis' (line 281)
    axis_15538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'axis')
    
    # Call to len(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'tmp' (line 281)
    tmp_15540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 33), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 281)
    shape_15541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 33), tmp_15540, 'shape')
    # Processing the call keyword arguments (line 281)
    kwargs_15542 = {}
    # Getting the type of 'len' (line 281)
    len_15539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 29), 'len', False)
    # Calling len(args, kwargs) (line 281)
    len_call_result_15543 = invoke(stypy.reporting.localization.Localization(__file__, 281, 29), len_15539, *[shape_15541], **kwargs_15542)
    
    int_15544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 46), 'int')
    # Applying the binary operator '-' (line 281)
    result_sub_15545 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 29), '-', len_call_result_15543, int_15544)
    
    # Applying the binary operator '==' (line 281)
    result_eq_15546 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 21), '==', axis_15538, result_sub_15545)
    
    # Applying the binary operator 'or' (line 281)
    result_or_keyword_15547 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'or', result_eq_15537, result_eq_15546)
    
    # Testing the type of an if condition (line 281)
    if_condition_15548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), result_or_keyword_15547)
    # Assigning a type to the variable 'if_condition_15548' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_15548', if_condition_15548)
    # SSA begins for if statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to work_function(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'tmp' (line 282)
    tmp_15550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 29), 'tmp', False)
    # Getting the type of 'n' (line 282)
    n_15551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 33), 'n', False)
    int_15552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 35), 'int')
    int_15553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 37), 'int')
    # Getting the type of 'overwrite_x' (line 282)
    overwrite_x_15554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 39), 'overwrite_x', False)
    # Processing the call keyword arguments (line 282)
    kwargs_15555 = {}
    # Getting the type of 'work_function' (line 282)
    work_function_15549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'work_function', False)
    # Calling work_function(args, kwargs) (line 282)
    work_function_call_result_15556 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), work_function_15549, *[tmp_15550, n_15551, int_15552, int_15553, overwrite_x_15554], **kwargs_15555)
    
    # Assigning a type to the variable 'stypy_return_type' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', work_function_call_result_15556)
    # SSA join for if statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 284):
    
    # Assigning a Call to a Name (line 284):
    
    # Call to swapaxes(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'tmp' (line 284)
    tmp_15558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 19), 'tmp', False)
    # Getting the type of 'axis' (line 284)
    axis_15559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'axis', False)
    int_15560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'int')
    # Processing the call keyword arguments (line 284)
    kwargs_15561 = {}
    # Getting the type of 'swapaxes' (line 284)
    swapaxes_15557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 10), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 284)
    swapaxes_call_result_15562 = invoke(stypy.reporting.localization.Localization(__file__, 284, 10), swapaxes_15557, *[tmp_15558, axis_15559, int_15560], **kwargs_15561)
    
    # Assigning a type to the variable 'tmp' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'tmp', swapaxes_call_result_15562)
    
    # Assigning a Call to a Name (line 285):
    
    # Assigning a Call to a Name (line 285):
    
    # Call to work_function(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'tmp' (line 285)
    tmp_15564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 24), 'tmp', False)
    # Getting the type of 'n' (line 285)
    n_15565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 28), 'n', False)
    int_15566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 30), 'int')
    int_15567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 32), 'int')
    # Getting the type of 'overwrite_x' (line 285)
    overwrite_x_15568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 34), 'overwrite_x', False)
    # Processing the call keyword arguments (line 285)
    kwargs_15569 = {}
    # Getting the type of 'work_function' (line 285)
    work_function_15563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 10), 'work_function', False)
    # Calling work_function(args, kwargs) (line 285)
    work_function_call_result_15570 = invoke(stypy.reporting.localization.Localization(__file__, 285, 10), work_function_15563, *[tmp_15564, n_15565, int_15566, int_15567, overwrite_x_15568], **kwargs_15569)
    
    # Assigning a type to the variable 'tmp' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'tmp', work_function_call_result_15570)
    
    # Call to swapaxes(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'tmp' (line 286)
    tmp_15572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 20), 'tmp', False)
    # Getting the type of 'axis' (line 286)
    axis_15573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 'axis', False)
    int_15574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'int')
    # Processing the call keyword arguments (line 286)
    kwargs_15575 = {}
    # Getting the type of 'swapaxes' (line 286)
    swapaxes_15571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 286)
    swapaxes_call_result_15576 = invoke(stypy.reporting.localization.Localization(__file__, 286, 11), swapaxes_15571, *[tmp_15572, axis_15573, int_15574], **kwargs_15575)
    
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type', swapaxes_call_result_15576)
    
    # ################# End of 'fft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fft' in the type store
    # Getting the type of 'stypy_return_type' (line 184)
    stypy_return_type_15577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15577)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fft'
    return stypy_return_type_15577

# Assigning a type to the variable 'fft' (line 184)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 0), 'fft', fft)

@norecursion
def ifft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 289)
    None_15578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'None')
    int_15579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 25), 'int')
    # Getting the type of 'False' (line 289)
    False_15580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 41), 'False')
    defaults = [None_15578, int_15579, False_15580]
    # Create a new context for function 'ifft'
    module_type_store = module_type_store.open_function_context('ifft', 289, 0, False)
    
    # Passed parameters checking function
    ifft.stypy_localization = localization
    ifft.stypy_type_of_self = None
    ifft.stypy_type_store = module_type_store
    ifft.stypy_function_name = 'ifft'
    ifft.stypy_param_names_list = ['x', 'n', 'axis', 'overwrite_x']
    ifft.stypy_varargs_param_name = None
    ifft.stypy_kwargs_param_name = None
    ifft.stypy_call_defaults = defaults
    ifft.stypy_call_varargs = varargs
    ifft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifft', ['x', 'n', 'axis', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifft', localization, ['x', 'n', 'axis', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifft(...)' code ##################

    str_15581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, (-1)), 'str', '\n    Return discrete inverse Fourier transform of real or complex sequence.\n\n    The returned complex array contains ``y(0), y(1),..., y(n-1)`` where\n\n    ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.\n\n    Parameters\n    ----------\n    x : array_like\n        Transformed data to invert.\n    n : int, optional\n        Length of the inverse Fourier transform.  If ``n < x.shape[axis]``,\n        `x` is truncated.  If ``n > x.shape[axis]``, `x` is zero-padded.\n        The default results in ``n = x.shape[axis]``.\n    axis : int, optional\n        Axis along which the ifft\'s are computed; the default is over the\n        last axis (i.e., ``axis=-1``).\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    ifft : ndarray of floats\n        The inverse discrete Fourier transform.\n\n    See Also\n    --------\n    fft : Forward FFT\n\n    Notes\n    -----\n    Both single and double precision routines are implemented.  Half precision\n    inputs will be converted to single precision.  Non floating-point inputs\n    will be converted to double precision.  Long-double precision inputs are\n    not supported.\n\n    This function is most efficient when `n` is a power of two, and least\n    efficient when `n` is prime.\n\n    If the data type of `x` is real, a "real IFFT" algorithm is automatically\n    used, which roughly halves the computation time.\n\n    ')
    
    # Assigning a Call to a Name (line 334):
    
    # Assigning a Call to a Name (line 334):
    
    # Call to _asfarray(...): (line 334)
    # Processing the call arguments (line 334)
    # Getting the type of 'x' (line 334)
    x_15583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 20), 'x', False)
    # Processing the call keyword arguments (line 334)
    kwargs_15584 = {}
    # Getting the type of '_asfarray' (line 334)
    _asfarray_15582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 334)
    _asfarray_call_result_15585 = invoke(stypy.reporting.localization.Localization(__file__, 334, 10), _asfarray_15582, *[x_15583], **kwargs_15584)
    
    # Assigning a type to the variable 'tmp' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'tmp', _asfarray_call_result_15585)
    
    
    # SSA begins for try-except statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 337):
    
    # Assigning a Subscript to a Name (line 337):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tmp' (line 337)
    tmp_15586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'tmp')
    # Obtaining the member 'dtype' of a type (line 337)
    dtype_15587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 38), tmp_15586, 'dtype')
    # Getting the type of '_DTYPE_TO_FFT' (line 337)
    _DTYPE_TO_FFT_15588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), '_DTYPE_TO_FFT')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___15589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), _DTYPE_TO_FFT_15588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_15590 = invoke(stypy.reporting.localization.Localization(__file__, 337, 24), getitem___15589, dtype_15587)
    
    # Assigning a type to the variable 'work_function' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'work_function', subscript_call_result_15590)
    # SSA branch for the except part of a try statement (line 336)
    # SSA branch for the except 'KeyError' branch of a try statement (line 336)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 339)
    # Processing the call arguments (line 339)
    str_15592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'str', 'type %s is not supported')
    # Getting the type of 'tmp' (line 339)
    tmp_15593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 54), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 339)
    dtype_15594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 54), tmp_15593, 'dtype')
    # Applying the binary operator '%' (line 339)
    result_mod_15595 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 25), '%', str_15592, dtype_15594)
    
    # Processing the call keyword arguments (line 339)
    kwargs_15596 = {}
    # Getting the type of 'ValueError' (line 339)
    ValueError_15591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 339)
    ValueError_call_result_15597 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), ValueError_15591, *[result_mod_15595], **kwargs_15596)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 339, 8), ValueError_call_result_15597, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to istype(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'tmp' (line 341)
    tmp_15599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'tmp', False)
    # Getting the type of 'numpy' (line 341)
    numpy_15600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 341)
    complex64_15601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 24), numpy_15600, 'complex64')
    # Processing the call keyword arguments (line 341)
    kwargs_15602 = {}
    # Getting the type of 'istype' (line 341)
    istype_15598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'istype', False)
    # Calling istype(args, kwargs) (line 341)
    istype_call_result_15603 = invoke(stypy.reporting.localization.Localization(__file__, 341, 12), istype_15598, *[tmp_15599, complex64_15601], **kwargs_15602)
    
    
    # Call to istype(...): (line 341)
    # Processing the call arguments (line 341)
    # Getting the type of 'tmp' (line 341)
    tmp_15605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 51), 'tmp', False)
    # Getting the type of 'numpy' (line 341)
    numpy_15606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 56), 'numpy', False)
    # Obtaining the member 'complex128' of a type (line 341)
    complex128_15607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 56), numpy_15606, 'complex128')
    # Processing the call keyword arguments (line 341)
    kwargs_15608 = {}
    # Getting the type of 'istype' (line 341)
    istype_15604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 44), 'istype', False)
    # Calling istype(args, kwargs) (line 341)
    istype_call_result_15609 = invoke(stypy.reporting.localization.Localization(__file__, 341, 44), istype_15604, *[tmp_15605, complex128_15607], **kwargs_15608)
    
    # Applying the binary operator 'or' (line 341)
    result_or_keyword_15610 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 12), 'or', istype_call_result_15603, istype_call_result_15609)
    
    # Applying the 'not' unary operator (line 341)
    result_not__15611 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 7), 'not', result_or_keyword_15610)
    
    # Testing the type of an if condition (line 341)
    if_condition_15612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 4), result_not__15611)
    # Assigning a type to the variable 'if_condition_15612' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'if_condition_15612', if_condition_15612)
    # SSA begins for if statement (line 341)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 342):
    
    # Assigning a Num to a Name (line 342):
    int_15613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 22), 'int')
    # Assigning a type to the variable 'overwrite_x' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'overwrite_x', int_15613)
    # SSA join for if statement (line 341)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 344):
    
    # Assigning a BoolOp to a Name (line 344):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 344)
    overwrite_x_15614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'overwrite_x')
    
    # Call to _datacopied(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'tmp' (line 344)
    tmp_15616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 45), 'tmp', False)
    # Getting the type of 'x' (line 344)
    x_15617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 50), 'x', False)
    # Processing the call keyword arguments (line 344)
    kwargs_15618 = {}
    # Getting the type of '_datacopied' (line 344)
    _datacopied_15615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 344)
    _datacopied_call_result_15619 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), _datacopied_15615, *[tmp_15616, x_15617], **kwargs_15618)
    
    # Applying the binary operator 'or' (line 344)
    result_or_keyword_15620 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 18), 'or', overwrite_x_15614, _datacopied_call_result_15619)
    
    # Assigning a type to the variable 'overwrite_x' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'overwrite_x', result_or_keyword_15620)
    
    # Type idiom detected: calculating its left and rigth part (line 346)
    # Getting the type of 'n' (line 346)
    n_15621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 7), 'n')
    # Getting the type of 'None' (line 346)
    None_15622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'None')
    
    (may_be_15623, more_types_in_union_15624) = may_be_none(n_15621, None_15622)

    if may_be_15623:

        if more_types_in_union_15624:
            # Runtime conditional SSA (line 346)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 347):
        
        # Assigning a Subscript to a Name (line 347):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 347)
        axis_15625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'axis')
        # Getting the type of 'tmp' (line 347)
        tmp_15626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'tmp')
        # Obtaining the member 'shape' of a type (line 347)
        shape_15627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), tmp_15626, 'shape')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___15628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), shape_15627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_15629 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), getitem___15628, axis_15625)
        
        # Assigning a type to the variable 'n' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'n', subscript_call_result_15629)

        if more_types_in_union_15624:
            # Runtime conditional SSA for else branch (line 346)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15623) or more_types_in_union_15624):
        
        
        # Getting the type of 'n' (line 348)
        n_15630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 9), 'n')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 348)
        axis_15631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 24), 'axis')
        # Getting the type of 'tmp' (line 348)
        tmp_15632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'tmp')
        # Obtaining the member 'shape' of a type (line 348)
        shape_15633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 14), tmp_15632, 'shape')
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___15634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 14), shape_15633, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_15635 = invoke(stypy.reporting.localization.Localization(__file__, 348, 14), getitem___15634, axis_15631)
        
        # Applying the binary operator '!=' (line 348)
        result_ne_15636 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 9), '!=', n_15630, subscript_call_result_15635)
        
        # Testing the type of an if condition (line 348)
        if_condition_15637 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 9), result_ne_15636)
        # Assigning a type to the variable 'if_condition_15637' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 9), 'if_condition_15637', if_condition_15637)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 349):
        
        # Assigning a Subscript to a Name (line 349):
        
        # Obtaining the type of the subscript
        int_15638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        
        # Call to _fix_shape(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'tmp' (line 349)
        tmp_15640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 36), 'tmp', False)
        # Getting the type of 'n' (line 349)
        n_15641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 40), 'n', False)
        # Getting the type of 'axis' (line 349)
        axis_15642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'axis', False)
        # Processing the call keyword arguments (line 349)
        kwargs_15643 = {}
        # Getting the type of '_fix_shape' (line 349)
        _fix_shape_15639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 349)
        _fix_shape_call_result_15644 = invoke(stypy.reporting.localization.Localization(__file__, 349, 25), _fix_shape_15639, *[tmp_15640, n_15641, axis_15642], **kwargs_15643)
        
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___15645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), _fix_shape_call_result_15644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_15646 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), getitem___15645, int_15638)
        
        # Assigning a type to the variable 'tuple_var_assignment_14883' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_14883', subscript_call_result_15646)
        
        # Assigning a Subscript to a Name (line 349):
        
        # Obtaining the type of the subscript
        int_15647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        
        # Call to _fix_shape(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'tmp' (line 349)
        tmp_15649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 36), 'tmp', False)
        # Getting the type of 'n' (line 349)
        n_15650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 40), 'n', False)
        # Getting the type of 'axis' (line 349)
        axis_15651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'axis', False)
        # Processing the call keyword arguments (line 349)
        kwargs_15652 = {}
        # Getting the type of '_fix_shape' (line 349)
        _fix_shape_15648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 25), '_fix_shape', False)
        # Calling _fix_shape(args, kwargs) (line 349)
        _fix_shape_call_result_15653 = invoke(stypy.reporting.localization.Localization(__file__, 349, 25), _fix_shape_15648, *[tmp_15649, n_15650, axis_15651], **kwargs_15652)
        
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___15654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), _fix_shape_call_result_15653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_15655 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), getitem___15654, int_15647)
        
        # Assigning a type to the variable 'tuple_var_assignment_14884' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_14884', subscript_call_result_15655)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'tuple_var_assignment_14883' (line 349)
        tuple_var_assignment_14883_15656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_14883')
        # Assigning a type to the variable 'tmp' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tmp', tuple_var_assignment_14883_15656)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'tuple_var_assignment_14884' (line 349)
        tuple_var_assignment_14884_15657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_14884')
        # Assigning a type to the variable 'copy_made' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'copy_made', tuple_var_assignment_14884_15657)
        
        # Assigning a BoolOp to a Name (line 350):
        
        # Assigning a BoolOp to a Name (line 350):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_x' (line 350)
        overwrite_x_15658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'overwrite_x')
        # Getting the type of 'copy_made' (line 350)
        copy_made_15659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 37), 'copy_made')
        # Applying the binary operator 'or' (line 350)
        result_or_keyword_15660 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 22), 'or', overwrite_x_15658, copy_made_15659)
        
        # Assigning a type to the variable 'overwrite_x' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'overwrite_x', result_or_keyword_15660)
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_15623 and more_types_in_union_15624):
            # SSA join for if statement (line 346)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'n' (line 352)
    n_15661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 7), 'n')
    int_15662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 11), 'int')
    # Applying the binary operator '<' (line 352)
    result_lt_15663 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 7), '<', n_15661, int_15662)
    
    # Testing the type of an if condition (line 352)
    if_condition_15664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 4), result_lt_15663)
    # Assigning a type to the variable 'if_condition_15664' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'if_condition_15664', if_condition_15664)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 353)
    # Processing the call arguments (line 353)
    str_15666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 25), 'str', 'Invalid number of FFT data points (%d) specified.')
    # Getting the type of 'n' (line 354)
    n_15667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 45), 'n', False)
    # Applying the binary operator '%' (line 353)
    result_mod_15668 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 25), '%', str_15666, n_15667)
    
    # Processing the call keyword arguments (line 353)
    kwargs_15669 = {}
    # Getting the type of 'ValueError' (line 353)
    ValueError_15665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 353)
    ValueError_call_result_15670 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), ValueError_15665, *[result_mod_15668], **kwargs_15669)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 8), ValueError_call_result_15670, 'raise parameter', BaseException)
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 356)
    axis_15671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 7), 'axis')
    int_15672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 15), 'int')
    # Applying the binary operator '==' (line 356)
    result_eq_15673 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 7), '==', axis_15671, int_15672)
    
    
    # Getting the type of 'axis' (line 356)
    axis_15674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 21), 'axis')
    
    # Call to len(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'tmp' (line 356)
    tmp_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'tmp', False)
    # Obtaining the member 'shape' of a type (line 356)
    shape_15677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 33), tmp_15676, 'shape')
    # Processing the call keyword arguments (line 356)
    kwargs_15678 = {}
    # Getting the type of 'len' (line 356)
    len_15675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 29), 'len', False)
    # Calling len(args, kwargs) (line 356)
    len_call_result_15679 = invoke(stypy.reporting.localization.Localization(__file__, 356, 29), len_15675, *[shape_15677], **kwargs_15678)
    
    int_15680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 46), 'int')
    # Applying the binary operator '-' (line 356)
    result_sub_15681 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 29), '-', len_call_result_15679, int_15680)
    
    # Applying the binary operator '==' (line 356)
    result_eq_15682 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 21), '==', axis_15674, result_sub_15681)
    
    # Applying the binary operator 'or' (line 356)
    result_or_keyword_15683 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 7), 'or', result_eq_15673, result_eq_15682)
    
    # Testing the type of an if condition (line 356)
    if_condition_15684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 4), result_or_keyword_15683)
    # Assigning a type to the variable 'if_condition_15684' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'if_condition_15684', if_condition_15684)
    # SSA begins for if statement (line 356)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to work_function(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'tmp' (line 357)
    tmp_15686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 29), 'tmp', False)
    # Getting the type of 'n' (line 357)
    n_15687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 33), 'n', False)
    int_15688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 35), 'int')
    int_15689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 38), 'int')
    # Getting the type of 'overwrite_x' (line 357)
    overwrite_x_15690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 40), 'overwrite_x', False)
    # Processing the call keyword arguments (line 357)
    kwargs_15691 = {}
    # Getting the type of 'work_function' (line 357)
    work_function_15685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 15), 'work_function', False)
    # Calling work_function(args, kwargs) (line 357)
    work_function_call_result_15692 = invoke(stypy.reporting.localization.Localization(__file__, 357, 15), work_function_15685, *[tmp_15686, n_15687, int_15688, int_15689, overwrite_x_15690], **kwargs_15691)
    
    # Assigning a type to the variable 'stypy_return_type' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'stypy_return_type', work_function_call_result_15692)
    # SSA join for if statement (line 356)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to swapaxes(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'tmp' (line 359)
    tmp_15694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'tmp', False)
    # Getting the type of 'axis' (line 359)
    axis_15695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'axis', False)
    int_15696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'int')
    # Processing the call keyword arguments (line 359)
    kwargs_15697 = {}
    # Getting the type of 'swapaxes' (line 359)
    swapaxes_15693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 10), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 359)
    swapaxes_call_result_15698 = invoke(stypy.reporting.localization.Localization(__file__, 359, 10), swapaxes_15693, *[tmp_15694, axis_15695, int_15696], **kwargs_15697)
    
    # Assigning a type to the variable 'tmp' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'tmp', swapaxes_call_result_15698)
    
    # Assigning a Call to a Name (line 360):
    
    # Assigning a Call to a Name (line 360):
    
    # Call to work_function(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'tmp' (line 360)
    tmp_15700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'tmp', False)
    # Getting the type of 'n' (line 360)
    n_15701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'n', False)
    int_15702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 30), 'int')
    int_15703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 33), 'int')
    # Getting the type of 'overwrite_x' (line 360)
    overwrite_x_15704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 35), 'overwrite_x', False)
    # Processing the call keyword arguments (line 360)
    kwargs_15705 = {}
    # Getting the type of 'work_function' (line 360)
    work_function_15699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 10), 'work_function', False)
    # Calling work_function(args, kwargs) (line 360)
    work_function_call_result_15706 = invoke(stypy.reporting.localization.Localization(__file__, 360, 10), work_function_15699, *[tmp_15700, n_15701, int_15702, int_15703, overwrite_x_15704], **kwargs_15705)
    
    # Assigning a type to the variable 'tmp' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'tmp', work_function_call_result_15706)
    
    # Call to swapaxes(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'tmp' (line 361)
    tmp_15708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'tmp', False)
    # Getting the type of 'axis' (line 361)
    axis_15709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 'axis', False)
    int_15710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 31), 'int')
    # Processing the call keyword arguments (line 361)
    kwargs_15711 = {}
    # Getting the type of 'swapaxes' (line 361)
    swapaxes_15707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 361)
    swapaxes_call_result_15712 = invoke(stypy.reporting.localization.Localization(__file__, 361, 11), swapaxes_15707, *[tmp_15708, axis_15709, int_15710], **kwargs_15711)
    
    # Assigning a type to the variable 'stypy_return_type' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'stypy_return_type', swapaxes_call_result_15712)
    
    # ################# End of 'ifft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifft' in the type store
    # Getting the type of 'stypy_return_type' (line 289)
    stypy_return_type_15713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifft'
    return stypy_return_type_15713

# Assigning a type to the variable 'ifft' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'ifft', ifft)

@norecursion
def rfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 364)
    None_15714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 14), 'None')
    int_15715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 25), 'int')
    # Getting the type of 'False' (line 364)
    False_15716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 41), 'False')
    defaults = [None_15714, int_15715, False_15716]
    # Create a new context for function 'rfft'
    module_type_store = module_type_store.open_function_context('rfft', 364, 0, False)
    
    # Passed parameters checking function
    rfft.stypy_localization = localization
    rfft.stypy_type_of_self = None
    rfft.stypy_type_store = module_type_store
    rfft.stypy_function_name = 'rfft'
    rfft.stypy_param_names_list = ['x', 'n', 'axis', 'overwrite_x']
    rfft.stypy_varargs_param_name = None
    rfft.stypy_kwargs_param_name = None
    rfft.stypy_call_defaults = defaults
    rfft.stypy_call_varargs = varargs
    rfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfft', ['x', 'n', 'axis', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfft', localization, ['x', 'n', 'axis', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfft(...)' code ##################

    str_15717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, (-1)), 'str', '\n    Discrete Fourier transform of a real sequence.\n\n    Parameters\n    ----------\n    x : array_like, real-valued\n        The data to transform.\n    n : int, optional\n        Defines the length of the Fourier transform.  If `n` is not specified\n        (the default) then ``n = x.shape[axis]``.  If ``n < x.shape[axis]``,\n        `x` is truncated, if ``n > x.shape[axis]``, `x` is zero-padded.\n    axis : int, optional\n        The axis along which the transform is applied.  The default is the\n        last axis.\n    overwrite_x : bool, optional\n        If set to true, the contents of `x` can be overwritten. Default is\n        False.\n\n    Returns\n    -------\n    z : real ndarray\n        The returned real array contains::\n\n          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]              if n is even\n          [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2)),Im(y(n/2))]   if n is odd\n\n        where::\n\n          y(j) = sum[k=0..n-1] x[k] * exp(-sqrt(-1)*j*k*2*pi/n)\n          j = 0..n-1\n\n    See Also\n    --------\n    fft, irfft, numpy.fft.rfft\n\n    Notes\n    -----\n    Within numerical accuracy, ``y == rfft(irfft(y))``.\n\n    Both single and double precision routines are implemented.  Half precision\n    inputs will be converted to single precision.  Non floating-point inputs\n    will be converted to double precision.  Long-double precision inputs are\n    not supported.\n\n    To get an output with a complex datatype, consider using the related\n    function `numpy.fft.rfft`.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import fft, rfft\n    >>> a = [9, -9, 1, 3]\n    >>> fft(a)\n    array([  4. +0.j,   8.+12.j,  16. +0.j,   8.-12.j])\n    >>> rfft(a)\n    array([  4.,   8.,  12.,  16.])\n\n    ')
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to _asfarray(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'x' (line 422)
    x_15719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'x', False)
    # Processing the call keyword arguments (line 422)
    kwargs_15720 = {}
    # Getting the type of '_asfarray' (line 422)
    _asfarray_15718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 422)
    _asfarray_call_result_15721 = invoke(stypy.reporting.localization.Localization(__file__, 422, 10), _asfarray_15718, *[x_15719], **kwargs_15720)
    
    # Assigning a type to the variable 'tmp' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'tmp', _asfarray_call_result_15721)
    
    
    
    # Call to isrealobj(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'tmp' (line 424)
    tmp_15724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'tmp', False)
    # Processing the call keyword arguments (line 424)
    kwargs_15725 = {}
    # Getting the type of 'numpy' (line 424)
    numpy_15722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'numpy', False)
    # Obtaining the member 'isrealobj' of a type (line 424)
    isrealobj_15723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 11), numpy_15722, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 424)
    isrealobj_call_result_15726 = invoke(stypy.reporting.localization.Localization(__file__, 424, 11), isrealobj_15723, *[tmp_15724], **kwargs_15725)
    
    # Applying the 'not' unary operator (line 424)
    result_not__15727 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 7), 'not', isrealobj_call_result_15726)
    
    # Testing the type of an if condition (line 424)
    if_condition_15728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 4), result_not__15727)
    # Assigning a type to the variable 'if_condition_15728' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'if_condition_15728', if_condition_15728)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 425)
    # Processing the call arguments (line 425)
    str_15730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 24), 'str', '1st argument must be real sequence')
    # Processing the call keyword arguments (line 425)
    kwargs_15731 = {}
    # Getting the type of 'TypeError' (line 425)
    TypeError_15729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 425)
    TypeError_call_result_15732 = invoke(stypy.reporting.localization.Localization(__file__, 425, 14), TypeError_15729, *[str_15730], **kwargs_15731)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 425, 8), TypeError_call_result_15732, 'raise parameter', BaseException)
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 427)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 428):
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tmp' (line 428)
    tmp_15733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'tmp')
    # Obtaining the member 'dtype' of a type (line 428)
    dtype_15734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 39), tmp_15733, 'dtype')
    # Getting the type of '_DTYPE_TO_RFFT' (line 428)
    _DTYPE_TO_RFFT_15735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 24), '_DTYPE_TO_RFFT')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___15736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 24), _DTYPE_TO_RFFT_15735, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_15737 = invoke(stypy.reporting.localization.Localization(__file__, 428, 24), getitem___15736, dtype_15734)
    
    # Assigning a type to the variable 'work_function' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'work_function', subscript_call_result_15737)
    # SSA branch for the except part of a try statement (line 427)
    # SSA branch for the except 'KeyError' branch of a try statement (line 427)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 430)
    # Processing the call arguments (line 430)
    str_15739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 25), 'str', 'type %s is not supported')
    # Getting the type of 'tmp' (line 430)
    tmp_15740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 54), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 430)
    dtype_15741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 54), tmp_15740, 'dtype')
    # Applying the binary operator '%' (line 430)
    result_mod_15742 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 25), '%', str_15739, dtype_15741)
    
    # Processing the call keyword arguments (line 430)
    kwargs_15743 = {}
    # Getting the type of 'ValueError' (line 430)
    ValueError_15738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 430)
    ValueError_call_result_15744 = invoke(stypy.reporting.localization.Localization(__file__, 430, 14), ValueError_15738, *[result_mod_15742], **kwargs_15743)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 430, 8), ValueError_call_result_15744, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 427)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 432):
    
    # Assigning a BoolOp to a Name (line 432):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 432)
    overwrite_x_15745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'overwrite_x')
    
    # Call to _datacopied(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'tmp' (line 432)
    tmp_15747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 45), 'tmp', False)
    # Getting the type of 'x' (line 432)
    x_15748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'x', False)
    # Processing the call keyword arguments (line 432)
    kwargs_15749 = {}
    # Getting the type of '_datacopied' (line 432)
    _datacopied_15746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 432)
    _datacopied_call_result_15750 = invoke(stypy.reporting.localization.Localization(__file__, 432, 33), _datacopied_15746, *[tmp_15747, x_15748], **kwargs_15749)
    
    # Applying the binary operator 'or' (line 432)
    result_or_keyword_15751 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 18), 'or', overwrite_x_15745, _datacopied_call_result_15750)
    
    # Assigning a type to the variable 'overwrite_x' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'overwrite_x', result_or_keyword_15751)
    
    # Call to _raw_fft(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'tmp' (line 434)
    tmp_15753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 20), 'tmp', False)
    # Getting the type of 'n' (line 434)
    n_15754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'n', False)
    # Getting the type of 'axis' (line 434)
    axis_15755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'axis', False)
    int_15756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 31), 'int')
    # Getting the type of 'overwrite_x' (line 434)
    overwrite_x_15757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 33), 'overwrite_x', False)
    # Getting the type of 'work_function' (line 434)
    work_function_15758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'work_function', False)
    # Processing the call keyword arguments (line 434)
    kwargs_15759 = {}
    # Getting the type of '_raw_fft' (line 434)
    _raw_fft_15752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 11), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 434)
    _raw_fft_call_result_15760 = invoke(stypy.reporting.localization.Localization(__file__, 434, 11), _raw_fft_15752, *[tmp_15753, n_15754, axis_15755, int_15756, overwrite_x_15757, work_function_15758], **kwargs_15759)
    
    # Assigning a type to the variable 'stypy_return_type' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'stypy_return_type', _raw_fft_call_result_15760)
    
    # ################# End of 'rfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfft' in the type store
    # Getting the type of 'stypy_return_type' (line 364)
    stypy_return_type_15761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15761)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfft'
    return stypy_return_type_15761

# Assigning a type to the variable 'rfft' (line 364)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 0), 'rfft', rfft)

@norecursion
def irfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 437)
    None_15762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'None')
    int_15763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 26), 'int')
    # Getting the type of 'False' (line 437)
    False_15764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 42), 'False')
    defaults = [None_15762, int_15763, False_15764]
    # Create a new context for function 'irfft'
    module_type_store = module_type_store.open_function_context('irfft', 437, 0, False)
    
    # Passed parameters checking function
    irfft.stypy_localization = localization
    irfft.stypy_type_of_self = None
    irfft.stypy_type_store = module_type_store
    irfft.stypy_function_name = 'irfft'
    irfft.stypy_param_names_list = ['x', 'n', 'axis', 'overwrite_x']
    irfft.stypy_varargs_param_name = None
    irfft.stypy_kwargs_param_name = None
    irfft.stypy_call_defaults = defaults
    irfft.stypy_call_varargs = varargs
    irfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'irfft', ['x', 'n', 'axis', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'irfft', localization, ['x', 'n', 'axis', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'irfft(...)' code ##################

    str_15765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, (-1)), 'str', "\n    Return inverse discrete Fourier transform of real sequence x.\n\n    The contents of `x` are interpreted as the output of the `rfft`\n    function.\n\n    Parameters\n    ----------\n    x : array_like\n        Transformed data to invert.\n    n : int, optional\n        Length of the inverse Fourier transform.\n        If n < x.shape[axis], x is truncated.\n        If n > x.shape[axis], x is zero-padded.\n        The default results in n = x.shape[axis].\n    axis : int, optional\n        Axis along which the ifft's are computed; the default is over\n        the last axis (i.e., axis=-1).\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed; the default is False.\n\n    Returns\n    -------\n    irfft : ndarray of floats\n        The inverse discrete Fourier transform.\n\n    See Also\n    --------\n    rfft, ifft, numpy.fft.irfft\n\n    Notes\n    -----\n    The returned real array contains::\n\n        [y(0),y(1),...,y(n-1)]\n\n    where for n is even::\n\n        y(j) = 1/n (sum[k=1..n/2-1] (x[2*k-1]+sqrt(-1)*x[2*k])\n                                     * exp(sqrt(-1)*j*k* 2*pi/n)\n                    + c.c. + x[0] + (-1)**(j) x[n-1])\n\n    and for n is odd::\n\n        y(j) = 1/n (sum[k=1..(n-1)/2] (x[2*k-1]+sqrt(-1)*x[2*k])\n                                     * exp(sqrt(-1)*j*k* 2*pi/n)\n                    + c.c. + x[0])\n\n    c.c. denotes complex conjugate of preceding expression.\n\n    For details on input parameters, see `rfft`.\n\n    To process (conjugate-symmetric) frequency-domain data with a complex\n    datatype, consider using the related function `numpy.fft.irfft`.\n    ")
    
    # Assigning a Call to a Name (line 493):
    
    # Assigning a Call to a Name (line 493):
    
    # Call to _asfarray(...): (line 493)
    # Processing the call arguments (line 493)
    # Getting the type of 'x' (line 493)
    x_15767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 20), 'x', False)
    # Processing the call keyword arguments (line 493)
    kwargs_15768 = {}
    # Getting the type of '_asfarray' (line 493)
    _asfarray_15766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 493)
    _asfarray_call_result_15769 = invoke(stypy.reporting.localization.Localization(__file__, 493, 10), _asfarray_15766, *[x_15767], **kwargs_15768)
    
    # Assigning a type to the variable 'tmp' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'tmp', _asfarray_call_result_15769)
    
    
    
    # Call to isrealobj(...): (line 494)
    # Processing the call arguments (line 494)
    # Getting the type of 'tmp' (line 494)
    tmp_15772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 27), 'tmp', False)
    # Processing the call keyword arguments (line 494)
    kwargs_15773 = {}
    # Getting the type of 'numpy' (line 494)
    numpy_15770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'numpy', False)
    # Obtaining the member 'isrealobj' of a type (line 494)
    isrealobj_15771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 11), numpy_15770, 'isrealobj')
    # Calling isrealobj(args, kwargs) (line 494)
    isrealobj_call_result_15774 = invoke(stypy.reporting.localization.Localization(__file__, 494, 11), isrealobj_15771, *[tmp_15772], **kwargs_15773)
    
    # Applying the 'not' unary operator (line 494)
    result_not__15775 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 7), 'not', isrealobj_call_result_15774)
    
    # Testing the type of an if condition (line 494)
    if_condition_15776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 4), result_not__15775)
    # Assigning a type to the variable 'if_condition_15776' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 4), 'if_condition_15776', if_condition_15776)
    # SSA begins for if statement (line 494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 495)
    # Processing the call arguments (line 495)
    str_15778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 24), 'str', '1st argument must be real sequence')
    # Processing the call keyword arguments (line 495)
    kwargs_15779 = {}
    # Getting the type of 'TypeError' (line 495)
    TypeError_15777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 495)
    TypeError_call_result_15780 = invoke(stypy.reporting.localization.Localization(__file__, 495, 14), TypeError_15777, *[str_15778], **kwargs_15779)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 495, 8), TypeError_call_result_15780, 'raise parameter', BaseException)
    # SSA join for if statement (line 494)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 498):
    
    # Assigning a Subscript to a Name (line 498):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tmp' (line 498)
    tmp_15781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 39), 'tmp')
    # Obtaining the member 'dtype' of a type (line 498)
    dtype_15782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 39), tmp_15781, 'dtype')
    # Getting the type of '_DTYPE_TO_RFFT' (line 498)
    _DTYPE_TO_RFFT_15783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), '_DTYPE_TO_RFFT')
    # Obtaining the member '__getitem__' of a type (line 498)
    getitem___15784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 24), _DTYPE_TO_RFFT_15783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 498)
    subscript_call_result_15785 = invoke(stypy.reporting.localization.Localization(__file__, 498, 24), getitem___15784, dtype_15782)
    
    # Assigning a type to the variable 'work_function' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'work_function', subscript_call_result_15785)
    # SSA branch for the except part of a try statement (line 497)
    # SSA branch for the except 'KeyError' branch of a try statement (line 497)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 500)
    # Processing the call arguments (line 500)
    str_15787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 25), 'str', 'type %s is not supported')
    # Getting the type of 'tmp' (line 500)
    tmp_15788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 54), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 500)
    dtype_15789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 54), tmp_15788, 'dtype')
    # Applying the binary operator '%' (line 500)
    result_mod_15790 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 25), '%', str_15787, dtype_15789)
    
    # Processing the call keyword arguments (line 500)
    kwargs_15791 = {}
    # Getting the type of 'ValueError' (line 500)
    ValueError_15786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 500)
    ValueError_call_result_15792 = invoke(stypy.reporting.localization.Localization(__file__, 500, 14), ValueError_15786, *[result_mod_15790], **kwargs_15791)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 500, 8), ValueError_call_result_15792, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 502):
    
    # Assigning a BoolOp to a Name (line 502):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 502)
    overwrite_x_15793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 18), 'overwrite_x')
    
    # Call to _datacopied(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'tmp' (line 502)
    tmp_15795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 45), 'tmp', False)
    # Getting the type of 'x' (line 502)
    x_15796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 50), 'x', False)
    # Processing the call keyword arguments (line 502)
    kwargs_15797 = {}
    # Getting the type of '_datacopied' (line 502)
    _datacopied_15794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 502)
    _datacopied_call_result_15798 = invoke(stypy.reporting.localization.Localization(__file__, 502, 33), _datacopied_15794, *[tmp_15795, x_15796], **kwargs_15797)
    
    # Applying the binary operator 'or' (line 502)
    result_or_keyword_15799 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 18), 'or', overwrite_x_15793, _datacopied_call_result_15798)
    
    # Assigning a type to the variable 'overwrite_x' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 4), 'overwrite_x', result_or_keyword_15799)
    
    # Call to _raw_fft(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'tmp' (line 504)
    tmp_15801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'tmp', False)
    # Getting the type of 'n' (line 504)
    n_15802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 24), 'n', False)
    # Getting the type of 'axis' (line 504)
    axis_15803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 26), 'axis', False)
    int_15804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 31), 'int')
    # Getting the type of 'overwrite_x' (line 504)
    overwrite_x_15805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 34), 'overwrite_x', False)
    # Getting the type of 'work_function' (line 504)
    work_function_15806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 46), 'work_function', False)
    # Processing the call keyword arguments (line 504)
    kwargs_15807 = {}
    # Getting the type of '_raw_fft' (line 504)
    _raw_fft_15800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 11), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 504)
    _raw_fft_call_result_15808 = invoke(stypy.reporting.localization.Localization(__file__, 504, 11), _raw_fft_15800, *[tmp_15801, n_15802, axis_15803, int_15804, overwrite_x_15805, work_function_15806], **kwargs_15807)
    
    # Assigning a type to the variable 'stypy_return_type' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 4), 'stypy_return_type', _raw_fft_call_result_15808)
    
    # ################# End of 'irfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'irfft' in the type store
    # Getting the type of 'stypy_return_type' (line 437)
    stypy_return_type_15809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'irfft'
    return stypy_return_type_15809

# Assigning a type to the variable 'irfft' (line 437)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 0), 'irfft', irfft)

@norecursion
def _raw_fftnd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raw_fftnd'
    module_type_store = module_type_store.open_function_context('_raw_fftnd', 507, 0, False)
    
    # Passed parameters checking function
    _raw_fftnd.stypy_localization = localization
    _raw_fftnd.stypy_type_of_self = None
    _raw_fftnd.stypy_type_store = module_type_store
    _raw_fftnd.stypy_function_name = '_raw_fftnd'
    _raw_fftnd.stypy_param_names_list = ['x', 's', 'axes', 'direction', 'overwrite_x', 'work_function']
    _raw_fftnd.stypy_varargs_param_name = None
    _raw_fftnd.stypy_kwargs_param_name = None
    _raw_fftnd.stypy_call_defaults = defaults
    _raw_fftnd.stypy_call_varargs = varargs
    _raw_fftnd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_fftnd', ['x', 's', 'axes', 'direction', 'overwrite_x', 'work_function'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_fftnd', localization, ['x', 's', 'axes', 'direction', 'overwrite_x', 'work_function'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_fftnd(...)' code ##################

    str_15810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 4), 'str', ' Internal auxiliary function for fftnd, ifftnd.')
    
    # Type idiom detected: calculating its left and rigth part (line 509)
    # Getting the type of 's' (line 509)
    s_15811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 7), 's')
    # Getting the type of 'None' (line 509)
    None_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'None')
    
    (may_be_15813, more_types_in_union_15814) = may_be_none(s_15811, None_15812)

    if may_be_15813:

        if more_types_in_union_15814:
            # Runtime conditional SSA (line 509)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 510)
        # Getting the type of 'axes' (line 510)
        axes_15815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 11), 'axes')
        # Getting the type of 'None' (line 510)
        None_15816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), 'None')
        
        (may_be_15817, more_types_in_union_15818) = may_be_none(axes_15815, None_15816)

        if may_be_15817:

            if more_types_in_union_15818:
                # Runtime conditional SSA (line 510)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 511):
            
            # Assigning a Attribute to a Name (line 511):
            # Getting the type of 'x' (line 511)
            x_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 16), 'x')
            # Obtaining the member 'shape' of a type (line 511)
            shape_15820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 16), x_15819, 'shape')
            # Assigning a type to the variable 's' (line 511)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 's', shape_15820)

            if more_types_in_union_15818:
                # Runtime conditional SSA for else branch (line 510)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15817) or more_types_in_union_15818):
            
            # Assigning a Call to a Name (line 513):
            
            # Assigning a Call to a Name (line 513):
            
            # Call to take(...): (line 513)
            # Processing the call arguments (line 513)
            # Getting the type of 'x' (line 513)
            x_15823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 27), 'x', False)
            # Obtaining the member 'shape' of a type (line 513)
            shape_15824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 27), x_15823, 'shape')
            # Getting the type of 'axes' (line 513)
            axes_15825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 36), 'axes', False)
            # Processing the call keyword arguments (line 513)
            kwargs_15826 = {}
            # Getting the type of 'numpy' (line 513)
            numpy_15821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'numpy', False)
            # Obtaining the member 'take' of a type (line 513)
            take_15822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 16), numpy_15821, 'take')
            # Calling take(args, kwargs) (line 513)
            take_call_result_15827 = invoke(stypy.reporting.localization.Localization(__file__, 513, 16), take_15822, *[shape_15824, axes_15825], **kwargs_15826)
            
            # Assigning a type to the variable 's' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 's', take_call_result_15827)

            if (may_be_15817 and more_types_in_union_15818):
                # SSA join for if statement (line 510)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_15814:
            # SSA join for if statement (line 509)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 515):
    
    # Assigning a Call to a Name (line 515):
    
    # Call to tuple(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 's' (line 515)
    s_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 14), 's', False)
    # Processing the call keyword arguments (line 515)
    kwargs_15830 = {}
    # Getting the type of 'tuple' (line 515)
    tuple_15828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'tuple', False)
    # Calling tuple(args, kwargs) (line 515)
    tuple_call_result_15831 = invoke(stypy.reporting.localization.Localization(__file__, 515, 8), tuple_15828, *[s_15829], **kwargs_15830)
    
    # Assigning a type to the variable 's' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 's', tuple_call_result_15831)
    
    # Type idiom detected: calculating its left and rigth part (line 516)
    # Getting the type of 'axes' (line 516)
    axes_15832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 7), 'axes')
    # Getting the type of 'None' (line 516)
    None_15833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 15), 'None')
    
    (may_be_15834, more_types_in_union_15835) = may_be_none(axes_15832, None_15833)

    if may_be_15834:

        if more_types_in_union_15835:
            # Runtime conditional SSA (line 516)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 517):
        
        # Assigning a Name to a Name (line 517):
        # Getting the type of 'True' (line 517)
        True_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 17), 'True')
        # Assigning a type to the variable 'noaxes' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'noaxes', True_15836)
        
        # Assigning a Call to a Name (line 518):
        
        # Assigning a Call to a Name (line 518):
        
        # Call to list(...): (line 518)
        # Processing the call arguments (line 518)
        
        # Call to range(...): (line 518)
        # Processing the call arguments (line 518)
        
        # Getting the type of 'x' (line 518)
        x_15839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 27), 'x', False)
        # Obtaining the member 'ndim' of a type (line 518)
        ndim_15840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 27), x_15839, 'ndim')
        # Applying the 'usub' unary operator (line 518)
        result___neg___15841 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 26), 'usub', ndim_15840)
        
        int_15842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 35), 'int')
        # Processing the call keyword arguments (line 518)
        kwargs_15843 = {}
        # Getting the type of 'range' (line 518)
        range_15838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'range', False)
        # Calling range(args, kwargs) (line 518)
        range_call_result_15844 = invoke(stypy.reporting.localization.Localization(__file__, 518, 20), range_15838, *[result___neg___15841, int_15842], **kwargs_15843)
        
        # Processing the call keyword arguments (line 518)
        kwargs_15845 = {}
        # Getting the type of 'list' (line 518)
        list_15837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 15), 'list', False)
        # Calling list(args, kwargs) (line 518)
        list_call_result_15846 = invoke(stypy.reporting.localization.Localization(__file__, 518, 15), list_15837, *[range_call_result_15844], **kwargs_15845)
        
        # Assigning a type to the variable 'axes' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'axes', list_call_result_15846)

        if more_types_in_union_15835:
            # Runtime conditional SSA for else branch (line 516)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15834) or more_types_in_union_15835):
        
        # Assigning a Name to a Name (line 520):
        
        # Assigning a Name to a Name (line 520):
        # Getting the type of 'False' (line 520)
        False_15847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 17), 'False')
        # Assigning a type to the variable 'noaxes' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'noaxes', False_15847)

        if (may_be_15834 and more_types_in_union_15835):
            # SSA join for if statement (line 516)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 'axes' (line 521)
    axes_15849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'axes', False)
    # Processing the call keyword arguments (line 521)
    kwargs_15850 = {}
    # Getting the type of 'len' (line 521)
    len_15848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 7), 'len', False)
    # Calling len(args, kwargs) (line 521)
    len_call_result_15851 = invoke(stypy.reporting.localization.Localization(__file__, 521, 7), len_15848, *[axes_15849], **kwargs_15850)
    
    
    # Call to len(...): (line 521)
    # Processing the call arguments (line 521)
    # Getting the type of 's' (line 521)
    s_15853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 24), 's', False)
    # Processing the call keyword arguments (line 521)
    kwargs_15854 = {}
    # Getting the type of 'len' (line 521)
    len_15852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'len', False)
    # Calling len(args, kwargs) (line 521)
    len_call_result_15855 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), len_15852, *[s_15853], **kwargs_15854)
    
    # Applying the binary operator '!=' (line 521)
    result_ne_15856 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 7), '!=', len_call_result_15851, len_call_result_15855)
    
    # Testing the type of an if condition (line 521)
    if_condition_15857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 4), result_ne_15856)
    # Assigning a type to the variable 'if_condition_15857' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'if_condition_15857', if_condition_15857)
    # SSA begins for if statement (line 521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 522)
    # Processing the call arguments (line 522)
    str_15859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 25), 'str', 'when given, axes and shape arguments have to be of the same length')
    # Processing the call keyword arguments (line 522)
    kwargs_15860 = {}
    # Getting the type of 'ValueError' (line 522)
    ValueError_15858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 522)
    ValueError_call_result_15861 = invoke(stypy.reporting.localization.Localization(__file__, 522, 14), ValueError_15858, *[str_15859], **kwargs_15860)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 522, 8), ValueError_call_result_15861, 'raise parameter', BaseException)
    # SSA join for if statement (line 521)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 's' (line 525)
    s_15862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 's')
    # Testing the type of a for loop iterable (line 525)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 525, 4), s_15862)
    # Getting the type of the for loop variable (line 525)
    for_loop_var_15863 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 525, 4), s_15862)
    # Assigning a type to the variable 'dim' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'dim', for_loop_var_15863)
    # SSA begins for a for statement (line 525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'dim' (line 526)
    dim_15864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 11), 'dim')
    int_15865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 17), 'int')
    # Applying the binary operator '<' (line 526)
    result_lt_15866 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 11), '<', dim_15864, int_15865)
    
    # Testing the type of an if condition (line 526)
    if_condition_15867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 526, 8), result_lt_15866)
    # Assigning a type to the variable 'if_condition_15867' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'if_condition_15867', if_condition_15867)
    # SSA begins for if statement (line 526)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 527)
    # Processing the call arguments (line 527)
    str_15869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 29), 'str', 'Invalid number of FFT data points (%s) specified.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 528)
    tuple_15870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 528)
    # Adding element type (line 528)
    # Getting the type of 's' (line 528)
    s_15871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 50), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 528, 50), tuple_15870, s_15871)
    
    # Applying the binary operator '%' (line 527)
    result_mod_15872 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 29), '%', str_15869, tuple_15870)
    
    # Processing the call keyword arguments (line 527)
    kwargs_15873 = {}
    # Getting the type of 'ValueError' (line 527)
    ValueError_15868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 527)
    ValueError_call_result_15874 = invoke(stypy.reporting.localization.Localization(__file__, 527, 18), ValueError_15868, *[result_mod_15872], **kwargs_15873)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 527, 12), ValueError_call_result_15874, 'raise parameter', BaseException)
    # SSA join for if statement (line 526)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'noaxes' (line 531)
    noaxes_15875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 7), 'noaxes')
    # Testing the type of an if condition (line 531)
    if_condition_15876 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 4), noaxes_15875)
    # Assigning a type to the variable 'if_condition_15876' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 4), 'if_condition_15876', if_condition_15876)
    # SSA begins for if statement (line 531)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axes' (line 532)
    axes_15877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'axes')
    # Testing the type of a for loop iterable (line 532)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 532, 8), axes_15877)
    # Getting the type of the for loop variable (line 532)
    for_loop_var_15878 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 532, 8), axes_15877)
    # Assigning a type to the variable 'i' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'i', for_loop_var_15878)
    # SSA begins for a for statement (line 532)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 533):
    
    # Assigning a Subscript to a Name (line 533):
    
    # Obtaining the type of the subscript
    int_15879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'int')
    
    # Call to _fix_shape(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'x' (line 533)
    x_15881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 38), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 533)
    i_15882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 43), 'i', False)
    # Getting the type of 's' (line 533)
    s_15883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 41), 's', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___15884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 41), s_15883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_15885 = invoke(stypy.reporting.localization.Localization(__file__, 533, 41), getitem___15884, i_15882)
    
    # Getting the type of 'i' (line 533)
    i_15886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 47), 'i', False)
    # Processing the call keyword arguments (line 533)
    kwargs_15887 = {}
    # Getting the type of '_fix_shape' (line 533)
    _fix_shape_15880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 27), '_fix_shape', False)
    # Calling _fix_shape(args, kwargs) (line 533)
    _fix_shape_call_result_15888 = invoke(stypy.reporting.localization.Localization(__file__, 533, 27), _fix_shape_15880, *[x_15881, subscript_call_result_15885, i_15886], **kwargs_15887)
    
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___15889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), _fix_shape_call_result_15888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_15890 = invoke(stypy.reporting.localization.Localization(__file__, 533, 12), getitem___15889, int_15879)
    
    # Assigning a type to the variable 'tuple_var_assignment_14885' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'tuple_var_assignment_14885', subscript_call_result_15890)
    
    # Assigning a Subscript to a Name (line 533):
    
    # Obtaining the type of the subscript
    int_15891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'int')
    
    # Call to _fix_shape(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'x' (line 533)
    x_15893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 38), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 533)
    i_15894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 43), 'i', False)
    # Getting the type of 's' (line 533)
    s_15895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 41), 's', False)
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___15896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 41), s_15895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_15897 = invoke(stypy.reporting.localization.Localization(__file__, 533, 41), getitem___15896, i_15894)
    
    # Getting the type of 'i' (line 533)
    i_15898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 47), 'i', False)
    # Processing the call keyword arguments (line 533)
    kwargs_15899 = {}
    # Getting the type of '_fix_shape' (line 533)
    _fix_shape_15892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 27), '_fix_shape', False)
    # Calling _fix_shape(args, kwargs) (line 533)
    _fix_shape_call_result_15900 = invoke(stypy.reporting.localization.Localization(__file__, 533, 27), _fix_shape_15892, *[x_15893, subscript_call_result_15897, i_15898], **kwargs_15899)
    
    # Obtaining the member '__getitem__' of a type (line 533)
    getitem___15901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 12), _fix_shape_call_result_15900, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 533)
    subscript_call_result_15902 = invoke(stypy.reporting.localization.Localization(__file__, 533, 12), getitem___15901, int_15891)
    
    # Assigning a type to the variable 'tuple_var_assignment_14886' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'tuple_var_assignment_14886', subscript_call_result_15902)
    
    # Assigning a Name to a Name (line 533):
    # Getting the type of 'tuple_var_assignment_14885' (line 533)
    tuple_var_assignment_14885_15903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'tuple_var_assignment_14885')
    # Assigning a type to the variable 'x' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'x', tuple_var_assignment_14885_15903)
    
    # Assigning a Name to a Name (line 533):
    # Getting the type of 'tuple_var_assignment_14886' (line 533)
    tuple_var_assignment_14886_15904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'tuple_var_assignment_14886')
    # Assigning a type to the variable 'copy_made' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'copy_made', tuple_var_assignment_14886_15904)
    
    # Assigning a BoolOp to a Name (line 534):
    
    # Assigning a BoolOp to a Name (line 534):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 534)
    overwrite_x_15905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 26), 'overwrite_x')
    # Getting the type of 'copy_made' (line 534)
    copy_made_15906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 41), 'copy_made')
    # Applying the binary operator 'or' (line 534)
    result_or_keyword_15907 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 26), 'or', overwrite_x_15905, copy_made_15906)
    
    # Assigning a type to the variable 'overwrite_x' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 12), 'overwrite_x', result_or_keyword_15907)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to work_function(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'x' (line 535)
    x_15909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 29), 'x', False)
    # Getting the type of 's' (line 535)
    s_15910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 31), 's', False)
    # Getting the type of 'direction' (line 535)
    direction_15911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 33), 'direction', False)
    # Processing the call keyword arguments (line 535)
    # Getting the type of 'overwrite_x' (line 535)
    overwrite_x_15912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 55), 'overwrite_x', False)
    keyword_15913 = overwrite_x_15912
    kwargs_15914 = {'overwrite_x': keyword_15913}
    # Getting the type of 'work_function' (line 535)
    work_function_15908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'work_function', False)
    # Calling work_function(args, kwargs) (line 535)
    work_function_call_result_15915 = invoke(stypy.reporting.localization.Localization(__file__, 535, 15), work_function_15908, *[x_15909, s_15910, direction_15911], **kwargs_15914)
    
    # Assigning a type to the variable 'stypy_return_type' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'stypy_return_type', work_function_call_result_15915)
    # SSA join for if statement (line 531)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 539):
    
    # Assigning a Call to a Name (line 539):
    
    # Call to array(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'axes' (line 539)
    axes_15918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'axes', False)
    # Getting the type of 'numpy' (line 539)
    numpy_15919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 26), 'numpy', False)
    # Obtaining the member 'intc' of a type (line 539)
    intc_15920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 26), numpy_15919, 'intc')
    # Processing the call keyword arguments (line 539)
    kwargs_15921 = {}
    # Getting the type of 'numpy' (line 539)
    numpy_15916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'numpy', False)
    # Obtaining the member 'array' of a type (line 539)
    array_15917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), numpy_15916, 'array')
    # Calling array(args, kwargs) (line 539)
    array_call_result_15922 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), array_15917, *[axes_15918, intc_15920], **kwargs_15921)
    
    # Assigning a type to the variable 'a' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'a', array_call_result_15922)
    
    # Assigning a Call to a Name (line 540):
    
    # Assigning a Call to a Name (line 540):
    
    # Call to where(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Getting the type of 'a' (line 540)
    a_15925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'a', False)
    int_15926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 31), 'int')
    # Applying the binary operator '<' (line 540)
    result_lt_15927 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 27), '<', a_15925, int_15926)
    
    # Getting the type of 'a' (line 540)
    a_15928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 34), 'a', False)
    # Getting the type of 'x' (line 540)
    x_15929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 38), 'x', False)
    # Obtaining the member 'ndim' of a type (line 540)
    ndim_15930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 38), x_15929, 'ndim')
    # Applying the binary operator '+' (line 540)
    result_add_15931 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 34), '+', a_15928, ndim_15930)
    
    # Getting the type of 'a' (line 540)
    a_15932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 46), 'a', False)
    # Processing the call keyword arguments (line 540)
    kwargs_15933 = {}
    # Getting the type of 'numpy' (line 540)
    numpy_15923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'numpy', False)
    # Obtaining the member 'where' of a type (line 540)
    where_15924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 15), numpy_15923, 'where')
    # Calling where(args, kwargs) (line 540)
    where_call_result_15934 = invoke(stypy.reporting.localization.Localization(__file__, 540, 15), where_15924, *[result_lt_15927, result_add_15931, a_15932], **kwargs_15933)
    
    # Assigning a type to the variable 'abs_axes' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'abs_axes', where_call_result_15934)
    
    # Assigning a Call to a Name (line 541):
    
    # Assigning a Call to a Name (line 541):
    
    # Call to argsort(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'abs_axes' (line 541)
    abs_axes_15937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 24), 'abs_axes', False)
    # Processing the call keyword arguments (line 541)
    kwargs_15938 = {}
    # Getting the type of 'numpy' (line 541)
    numpy_15935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 10), 'numpy', False)
    # Obtaining the member 'argsort' of a type (line 541)
    argsort_15936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 10), numpy_15935, 'argsort')
    # Calling argsort(args, kwargs) (line 541)
    argsort_call_result_15939 = invoke(stypy.reporting.localization.Localization(__file__, 541, 10), argsort_15936, *[abs_axes_15937], **kwargs_15938)
    
    # Assigning a type to the variable 'id_' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'id_', argsort_call_result_15939)
    
    # Assigning a ListComp to a Name (line 542):
    
    # Assigning a ListComp to a Name (line 542):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'id_' (line 542)
    id__15944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 29), 'id_')
    comprehension_15945 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 12), id__15944)
    # Assigning a type to the variable 'i' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'i', comprehension_15945)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 542)
    i_15940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 17), 'i')
    # Getting the type of 'axes' (line 542)
    axes_15941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'axes')
    # Obtaining the member '__getitem__' of a type (line 542)
    getitem___15942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 12), axes_15941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 542)
    subscript_call_result_15943 = invoke(stypy.reporting.localization.Localization(__file__, 542, 12), getitem___15942, i_15940)
    
    list_15946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 12), list_15946, subscript_call_result_15943)
    # Assigning a type to the variable 'axes' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'axes', list_15946)
    
    # Assigning a ListComp to a Name (line 543):
    
    # Assigning a ListComp to a Name (line 543):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'id_' (line 543)
    id__15951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 23), 'id_')
    comprehension_15952 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 9), id__15951)
    # Assigning a type to the variable 'i' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 9), 'i', comprehension_15952)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 543)
    i_15947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 11), 'i')
    # Getting the type of 's' (line 543)
    s_15948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 9), 's')
    # Obtaining the member '__getitem__' of a type (line 543)
    getitem___15949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 9), s_15948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 543)
    subscript_call_result_15950 = invoke(stypy.reporting.localization.Localization(__file__, 543, 9), getitem___15949, i_15947)
    
    list_15953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 543, 9), list_15953, subscript_call_result_15950)
    # Assigning a type to the variable 's' (line 543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 's', list_15953)
    
    
    # Call to range(...): (line 548)
    # Processing the call arguments (line 548)
    int_15955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 19), 'int')
    
    # Call to len(...): (line 548)
    # Processing the call arguments (line 548)
    # Getting the type of 'axes' (line 548)
    axes_15957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 26), 'axes', False)
    # Processing the call keyword arguments (line 548)
    kwargs_15958 = {}
    # Getting the type of 'len' (line 548)
    len_15956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 22), 'len', False)
    # Calling len(args, kwargs) (line 548)
    len_call_result_15959 = invoke(stypy.reporting.localization.Localization(__file__, 548, 22), len_15956, *[axes_15957], **kwargs_15958)
    
    int_15960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 32), 'int')
    # Applying the binary operator '+' (line 548)
    result_add_15961 = python_operator(stypy.reporting.localization.Localization(__file__, 548, 22), '+', len_call_result_15959, int_15960)
    
    # Processing the call keyword arguments (line 548)
    kwargs_15962 = {}
    # Getting the type of 'range' (line 548)
    range_15954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 13), 'range', False)
    # Calling range(args, kwargs) (line 548)
    range_call_result_15963 = invoke(stypy.reporting.localization.Localization(__file__, 548, 13), range_15954, *[int_15955, result_add_15961], **kwargs_15962)
    
    # Testing the type of a for loop iterable (line 548)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 548, 4), range_call_result_15963)
    # Getting the type of the for loop variable (line 548)
    for_loop_var_15964 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 548, 4), range_call_result_15963)
    # Assigning a type to the variable 'i' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'i', for_loop_var_15964)
    # SSA begins for a for statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 549):
    
    # Assigning a Call to a Name (line 549):
    
    # Call to swapaxes(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'x' (line 549)
    x_15967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 27), 'x', False)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 549)
    i_15968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 36), 'i', False)
    # Applying the 'usub' unary operator (line 549)
    result___neg___15969 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 35), 'usub', i_15968)
    
    # Getting the type of 'axes' (line 549)
    axes_15970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 30), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___15971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 30), axes_15970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_15972 = invoke(stypy.reporting.localization.Localization(__file__, 549, 30), getitem___15971, result___neg___15969)
    
    
    # Getting the type of 'i' (line 549)
    i_15973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 41), 'i', False)
    # Applying the 'usub' unary operator (line 549)
    result___neg___15974 = python_operator(stypy.reporting.localization.Localization(__file__, 549, 40), 'usub', i_15973)
    
    # Processing the call keyword arguments (line 549)
    kwargs_15975 = {}
    # Getting the type of 'numpy' (line 549)
    numpy_15965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'numpy', False)
    # Obtaining the member 'swapaxes' of a type (line 549)
    swapaxes_15966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 12), numpy_15965, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 549)
    swapaxes_call_result_15976 = invoke(stypy.reporting.localization.Localization(__file__, 549, 12), swapaxes_15966, *[x_15967, subscript_call_result_15972, result___neg___15974], **kwargs_15975)
    
    # Assigning a type to the variable 'x' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'x', swapaxes_call_result_15976)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 554):
    
    # Assigning a Call to a Name (line 554):
    
    # Call to list(...): (line 554)
    # Processing the call arguments (line 554)
    
    # Call to range(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'x' (line 554)
    x_15979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'x', False)
    # Obtaining the member 'ndim' of a type (line 554)
    ndim_15980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 23), x_15979, 'ndim')
    
    # Call to len(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'axes' (line 554)
    axes_15982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 36), 'axes', False)
    # Processing the call keyword arguments (line 554)
    kwargs_15983 = {}
    # Getting the type of 'len' (line 554)
    len_15981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 32), 'len', False)
    # Calling len(args, kwargs) (line 554)
    len_call_result_15984 = invoke(stypy.reporting.localization.Localization(__file__, 554, 32), len_15981, *[axes_15982], **kwargs_15983)
    
    # Applying the binary operator '-' (line 554)
    result_sub_15985 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 23), '-', ndim_15980, len_call_result_15984)
    
    # Getting the type of 'x' (line 554)
    x_15986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 43), 'x', False)
    # Obtaining the member 'ndim' of a type (line 554)
    ndim_15987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 43), x_15986, 'ndim')
    # Processing the call keyword arguments (line 554)
    kwargs_15988 = {}
    # Getting the type of 'range' (line 554)
    range_15978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 17), 'range', False)
    # Calling range(args, kwargs) (line 554)
    range_call_result_15989 = invoke(stypy.reporting.localization.Localization(__file__, 554, 17), range_15978, *[result_sub_15985, ndim_15987], **kwargs_15988)
    
    # Processing the call keyword arguments (line 554)
    kwargs_15990 = {}
    # Getting the type of 'list' (line 554)
    list_15977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'list', False)
    # Calling list(args, kwargs) (line 554)
    list_call_result_15991 = invoke(stypy.reporting.localization.Localization(__file__, 554, 12), list_15977, *[range_call_result_15989], **kwargs_15990)
    
    # Assigning a type to the variable 'waxes' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'waxes', list_call_result_15991)
    
    # Assigning a Call to a Name (line 555):
    
    # Assigning a Call to a Name (line 555):
    
    # Call to ones(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'x' (line 555)
    x_15994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'x', False)
    # Obtaining the member 'ndim' of a type (line 555)
    ndim_15995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 23), x_15994, 'ndim')
    # Processing the call keyword arguments (line 555)
    kwargs_15996 = {}
    # Getting the type of 'numpy' (line 555)
    numpy_15992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 12), 'numpy', False)
    # Obtaining the member 'ones' of a type (line 555)
    ones_15993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 12), numpy_15992, 'ones')
    # Calling ones(args, kwargs) (line 555)
    ones_call_result_15997 = invoke(stypy.reporting.localization.Localization(__file__, 555, 12), ones_15993, *[ndim_15995], **kwargs_15996)
    
    # Assigning a type to the variable 'shape' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'shape', ones_call_result_15997)
    
    # Assigning a Name to a Subscript (line 556):
    
    # Assigning a Name to a Subscript (line 556):
    # Getting the type of 's' (line 556)
    s_15998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 's')
    # Getting the type of 'shape' (line 556)
    shape_15999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'shape')
    # Getting the type of 'waxes' (line 556)
    waxes_16000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 10), 'waxes')
    # Storing an element on a container (line 556)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 556, 4), shape_15999, (waxes_16000, s_15998))
    
    
    # Call to range(...): (line 558)
    # Processing the call arguments (line 558)
    
    # Call to len(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'waxes' (line 558)
    waxes_16003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 'waxes', False)
    # Processing the call keyword arguments (line 558)
    kwargs_16004 = {}
    # Getting the type of 'len' (line 558)
    len_16002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 19), 'len', False)
    # Calling len(args, kwargs) (line 558)
    len_call_result_16005 = invoke(stypy.reporting.localization.Localization(__file__, 558, 19), len_16002, *[waxes_16003], **kwargs_16004)
    
    # Processing the call keyword arguments (line 558)
    kwargs_16006 = {}
    # Getting the type of 'range' (line 558)
    range_16001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 13), 'range', False)
    # Calling range(args, kwargs) (line 558)
    range_call_result_16007 = invoke(stypy.reporting.localization.Localization(__file__, 558, 13), range_16001, *[len_call_result_16005], **kwargs_16006)
    
    # Testing the type of a for loop iterable (line 558)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 558, 4), range_call_result_16007)
    # Getting the type of the for loop variable (line 558)
    for_loop_var_16008 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 558, 4), range_call_result_16007)
    # Assigning a type to the variable 'i' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'i', for_loop_var_16008)
    # SSA begins for a for statement (line 558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 559):
    
    # Assigning a Subscript to a Name (line 559):
    
    # Obtaining the type of the subscript
    int_16009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 8), 'int')
    
    # Call to _fix_shape(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'x' (line 559)
    x_16011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 34), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 559)
    i_16012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 39), 'i', False)
    # Getting the type of 's' (line 559)
    s_16013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 37), 's', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 37), s_16013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16015 = invoke(stypy.reporting.localization.Localization(__file__, 559, 37), getitem___16014, i_16012)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 559)
    i_16016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 49), 'i', False)
    # Getting the type of 'waxes' (line 559)
    waxes_16017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 43), 'waxes', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 43), waxes_16017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16019 = invoke(stypy.reporting.localization.Localization(__file__, 559, 43), getitem___16018, i_16016)
    
    # Processing the call keyword arguments (line 559)
    kwargs_16020 = {}
    # Getting the type of '_fix_shape' (line 559)
    _fix_shape_16010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), '_fix_shape', False)
    # Calling _fix_shape(args, kwargs) (line 559)
    _fix_shape_call_result_16021 = invoke(stypy.reporting.localization.Localization(__file__, 559, 23), _fix_shape_16010, *[x_16011, subscript_call_result_16015, subscript_call_result_16019], **kwargs_16020)
    
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), _fix_shape_call_result_16021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16023 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), getitem___16022, int_16009)
    
    # Assigning a type to the variable 'tuple_var_assignment_14887' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_14887', subscript_call_result_16023)
    
    # Assigning a Subscript to a Name (line 559):
    
    # Obtaining the type of the subscript
    int_16024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 8), 'int')
    
    # Call to _fix_shape(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'x' (line 559)
    x_16026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 34), 'x', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 559)
    i_16027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 39), 'i', False)
    # Getting the type of 's' (line 559)
    s_16028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 37), 's', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 37), s_16028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16030 = invoke(stypy.reporting.localization.Localization(__file__, 559, 37), getitem___16029, i_16027)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 559)
    i_16031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 49), 'i', False)
    # Getting the type of 'waxes' (line 559)
    waxes_16032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 43), 'waxes', False)
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 43), waxes_16032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16034 = invoke(stypy.reporting.localization.Localization(__file__, 559, 43), getitem___16033, i_16031)
    
    # Processing the call keyword arguments (line 559)
    kwargs_16035 = {}
    # Getting the type of '_fix_shape' (line 559)
    _fix_shape_16025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 23), '_fix_shape', False)
    # Calling _fix_shape(args, kwargs) (line 559)
    _fix_shape_call_result_16036 = invoke(stypy.reporting.localization.Localization(__file__, 559, 23), _fix_shape_16025, *[x_16026, subscript_call_result_16030, subscript_call_result_16034], **kwargs_16035)
    
    # Obtaining the member '__getitem__' of a type (line 559)
    getitem___16037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 8), _fix_shape_call_result_16036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 559)
    subscript_call_result_16038 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), getitem___16037, int_16024)
    
    # Assigning a type to the variable 'tuple_var_assignment_14888' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_14888', subscript_call_result_16038)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'tuple_var_assignment_14887' (line 559)
    tuple_var_assignment_14887_16039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_14887')
    # Assigning a type to the variable 'x' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'x', tuple_var_assignment_14887_16039)
    
    # Assigning a Name to a Name (line 559):
    # Getting the type of 'tuple_var_assignment_14888' (line 559)
    tuple_var_assignment_14888_16040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'tuple_var_assignment_14888')
    # Assigning a type to the variable 'copy_made' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), 'copy_made', tuple_var_assignment_14888_16040)
    
    # Assigning a BoolOp to a Name (line 560):
    
    # Assigning a BoolOp to a Name (line 560):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 560)
    overwrite_x_16041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 22), 'overwrite_x')
    # Getting the type of 'copy_made' (line 560)
    copy_made_16042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 37), 'copy_made')
    # Applying the binary operator 'or' (line 560)
    result_or_keyword_16043 = python_operator(stypy.reporting.localization.Localization(__file__, 560, 22), 'or', overwrite_x_16041, copy_made_16042)
    
    # Assigning a type to the variable 'overwrite_x' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 8), 'overwrite_x', result_or_keyword_16043)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 562):
    
    # Assigning a Call to a Name (line 562):
    
    # Call to work_function(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'x' (line 562)
    x_16045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 22), 'x', False)
    # Getting the type of 'shape' (line 562)
    shape_16046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'shape', False)
    # Getting the type of 'direction' (line 562)
    direction_16047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'direction', False)
    # Processing the call keyword arguments (line 562)
    # Getting the type of 'overwrite_x' (line 562)
    overwrite_x_16048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 55), 'overwrite_x', False)
    keyword_16049 = overwrite_x_16048
    kwargs_16050 = {'overwrite_x': keyword_16049}
    # Getting the type of 'work_function' (line 562)
    work_function_16044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'work_function', False)
    # Calling work_function(args, kwargs) (line 562)
    work_function_call_result_16051 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), work_function_16044, *[x_16045, shape_16046, direction_16047], **kwargs_16050)
    
    # Assigning a type to the variable 'r' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'r', work_function_call_result_16051)
    
    
    # Call to range(...): (line 566)
    # Processing the call arguments (line 566)
    
    # Call to len(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'axes' (line 566)
    axes_16054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 23), 'axes', False)
    # Processing the call keyword arguments (line 566)
    kwargs_16055 = {}
    # Getting the type of 'len' (line 566)
    len_16053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'len', False)
    # Calling len(args, kwargs) (line 566)
    len_call_result_16056 = invoke(stypy.reporting.localization.Localization(__file__, 566, 19), len_16053, *[axes_16054], **kwargs_16055)
    
    int_16057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 30), 'int')
    int_16058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 33), 'int')
    # Processing the call keyword arguments (line 566)
    kwargs_16059 = {}
    # Getting the type of 'range' (line 566)
    range_16052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 13), 'range', False)
    # Calling range(args, kwargs) (line 566)
    range_call_result_16060 = invoke(stypy.reporting.localization.Localization(__file__, 566, 13), range_16052, *[len_call_result_16056, int_16057, int_16058], **kwargs_16059)
    
    # Testing the type of a for loop iterable (line 566)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 566, 4), range_call_result_16060)
    # Getting the type of the for loop variable (line 566)
    for_loop_var_16061 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 566, 4), range_call_result_16060)
    # Assigning a type to the variable 'i' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'i', for_loop_var_16061)
    # SSA begins for a for statement (line 566)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to swapaxes(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'r' (line 567)
    r_16064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 27), 'r', False)
    
    # Getting the type of 'i' (line 567)
    i_16065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 31), 'i', False)
    # Applying the 'usub' unary operator (line 567)
    result___neg___16066 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 30), 'usub', i_16065)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 567)
    i_16067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 40), 'i', False)
    # Applying the 'usub' unary operator (line 567)
    result___neg___16068 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 39), 'usub', i_16067)
    
    # Getting the type of 'axes' (line 567)
    axes_16069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 34), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___16070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 34), axes_16069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_16071 = invoke(stypy.reporting.localization.Localization(__file__, 567, 34), getitem___16070, result___neg___16068)
    
    # Processing the call keyword arguments (line 567)
    kwargs_16072 = {}
    # Getting the type of 'numpy' (line 567)
    numpy_16062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'numpy', False)
    # Obtaining the member 'swapaxes' of a type (line 567)
    swapaxes_16063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 12), numpy_16062, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 567)
    swapaxes_call_result_16073 = invoke(stypy.reporting.localization.Localization(__file__, 567, 12), swapaxes_16063, *[r_16064, result___neg___16066, subscript_call_result_16071], **kwargs_16072)
    
    # Assigning a type to the variable 'r' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'r', swapaxes_call_result_16073)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 569)
    r_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'stypy_return_type', r_16074)
    
    # ################# End of '_raw_fftnd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_fftnd' in the type store
    # Getting the type of 'stypy_return_type' (line 507)
    stypy_return_type_16075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_fftnd'
    return stypy_return_type_16075

# Assigning a type to the variable '_raw_fftnd' (line 507)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 0), '_raw_fftnd', _raw_fftnd)

@norecursion
def fftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 572)
    None_16076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 18), 'None')
    # Getting the type of 'None' (line 572)
    None_16077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 29), 'None')
    # Getting the type of 'False' (line 572)
    False_16078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 47), 'False')
    defaults = [None_16076, None_16077, False_16078]
    # Create a new context for function 'fftn'
    module_type_store = module_type_store.open_function_context('fftn', 572, 0, False)
    
    # Passed parameters checking function
    fftn.stypy_localization = localization
    fftn.stypy_type_of_self = None
    fftn.stypy_type_store = module_type_store
    fftn.stypy_function_name = 'fftn'
    fftn.stypy_param_names_list = ['x', 'shape', 'axes', 'overwrite_x']
    fftn.stypy_varargs_param_name = None
    fftn.stypy_kwargs_param_name = None
    fftn.stypy_call_defaults = defaults
    fftn.stypy_call_varargs = varargs
    fftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftn', ['x', 'shape', 'axes', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftn', localization, ['x', 'shape', 'axes', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftn(...)' code ##################

    str_16079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, (-1)), 'str', '\n    Return multidimensional discrete Fourier transform.\n\n    The returned array contains::\n\n      y[j_1,..,j_d] = sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]\n         x[k_1,..,k_d] * prod[i=1..d] exp(-sqrt(-1)*2*pi/n_i * j_i * k_i)\n\n    where d = len(x.shape) and n = x.shape.\n\n    Parameters\n    ----------\n    x : array_like\n        The (n-dimensional) array to transform.\n    shape : tuple of ints, optional\n        The shape of the result.  If both `shape` and `axes` (see below) are\n        None, `shape` is ``x.shape``; if `shape` is None but `axes` is\n        not None, then `shape` is ``scipy.take(x.shape, axes, axis=0)``.\n        If ``shape[i] > x.shape[i]``, the i-th dimension is padded with zeros.\n        If ``shape[i] < x.shape[i]``, the i-th dimension is truncated to\n        length ``shape[i]``.\n    axes : array_like of ints, optional\n        The axes of `x` (`y` if `shape` is not None) along which the\n        transform is applied.\n    overwrite_x : bool, optional\n        If True, the contents of `x` can be destroyed.  Default is False.\n\n    Returns\n    -------\n    y : complex-valued n-dimensional numpy array\n        The (n-dimensional) DFT of the input array.\n\n    See Also\n    --------\n    ifftn\n\n    Notes\n    -----\n    If ``x`` is real-valued, then\n    ``y[..., j_i, ...] == y[..., n_i-j_i, ...].conjugate()``.\n\n    Both single and double precision routines are implemented.  Half precision\n    inputs will be converted to single precision.  Non floating-point inputs\n    will be converted to double precision.  Long-double precision inputs are\n    not supported.\n\n    Examples\n    --------\n    >>> from scipy.fftpack import fftn, ifftn\n    >>> y = (-np.arange(16), 8 - np.arange(16), np.arange(16))\n    >>> np.allclose(y, fftn(ifftn(y)))\n    True\n\n    ')
    
    # Call to _raw_fftn_dispatch(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'x' (line 627)
    x_16081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 30), 'x', False)
    # Getting the type of 'shape' (line 627)
    shape_16082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 33), 'shape', False)
    # Getting the type of 'axes' (line 627)
    axes_16083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 40), 'axes', False)
    # Getting the type of 'overwrite_x' (line 627)
    overwrite_x_16084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 46), 'overwrite_x', False)
    int_16085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 59), 'int')
    # Processing the call keyword arguments (line 627)
    kwargs_16086 = {}
    # Getting the type of '_raw_fftn_dispatch' (line 627)
    _raw_fftn_dispatch_16080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), '_raw_fftn_dispatch', False)
    # Calling _raw_fftn_dispatch(args, kwargs) (line 627)
    _raw_fftn_dispatch_call_result_16087 = invoke(stypy.reporting.localization.Localization(__file__, 627, 11), _raw_fftn_dispatch_16080, *[x_16081, shape_16082, axes_16083, overwrite_x_16084, int_16085], **kwargs_16086)
    
    # Assigning a type to the variable 'stypy_return_type' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'stypy_return_type', _raw_fftn_dispatch_call_result_16087)
    
    # ################# End of 'fftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftn' in the type store
    # Getting the type of 'stypy_return_type' (line 572)
    stypy_return_type_16088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16088)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftn'
    return stypy_return_type_16088

# Assigning a type to the variable 'fftn' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'fftn', fftn)

@norecursion
def _raw_fftn_dispatch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_raw_fftn_dispatch'
    module_type_store = module_type_store.open_function_context('_raw_fftn_dispatch', 630, 0, False)
    
    # Passed parameters checking function
    _raw_fftn_dispatch.stypy_localization = localization
    _raw_fftn_dispatch.stypy_type_of_self = None
    _raw_fftn_dispatch.stypy_type_store = module_type_store
    _raw_fftn_dispatch.stypy_function_name = '_raw_fftn_dispatch'
    _raw_fftn_dispatch.stypy_param_names_list = ['x', 'shape', 'axes', 'overwrite_x', 'direction']
    _raw_fftn_dispatch.stypy_varargs_param_name = None
    _raw_fftn_dispatch.stypy_kwargs_param_name = None
    _raw_fftn_dispatch.stypy_call_defaults = defaults
    _raw_fftn_dispatch.stypy_call_varargs = varargs
    _raw_fftn_dispatch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_fftn_dispatch', ['x', 'shape', 'axes', 'overwrite_x', 'direction'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_fftn_dispatch', localization, ['x', 'shape', 'axes', 'overwrite_x', 'direction'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_fftn_dispatch(...)' code ##################

    
    # Assigning a Call to a Name (line 631):
    
    # Assigning a Call to a Name (line 631):
    
    # Call to _asfarray(...): (line 631)
    # Processing the call arguments (line 631)
    # Getting the type of 'x' (line 631)
    x_16090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 20), 'x', False)
    # Processing the call keyword arguments (line 631)
    kwargs_16091 = {}
    # Getting the type of '_asfarray' (line 631)
    _asfarray_16089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 10), '_asfarray', False)
    # Calling _asfarray(args, kwargs) (line 631)
    _asfarray_call_result_16092 = invoke(stypy.reporting.localization.Localization(__file__, 631, 10), _asfarray_16089, *[x_16090], **kwargs_16091)
    
    # Assigning a type to the variable 'tmp' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'tmp', _asfarray_call_result_16092)
    
    
    # SSA begins for try-except statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 634):
    
    # Assigning a Subscript to a Name (line 634):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tmp' (line 634)
    tmp_16093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 39), 'tmp')
    # Obtaining the member 'dtype' of a type (line 634)
    dtype_16094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 39), tmp_16093, 'dtype')
    # Getting the type of '_DTYPE_TO_FFTN' (line 634)
    _DTYPE_TO_FFTN_16095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 24), '_DTYPE_TO_FFTN')
    # Obtaining the member '__getitem__' of a type (line 634)
    getitem___16096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 24), _DTYPE_TO_FFTN_16095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 634)
    subscript_call_result_16097 = invoke(stypy.reporting.localization.Localization(__file__, 634, 24), getitem___16096, dtype_16094)
    
    # Assigning a type to the variable 'work_function' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'work_function', subscript_call_result_16097)
    # SSA branch for the except part of a try statement (line 633)
    # SSA branch for the except 'KeyError' branch of a try statement (line 633)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 636)
    # Processing the call arguments (line 636)
    str_16099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 25), 'str', 'type %s is not supported')
    # Getting the type of 'tmp' (line 636)
    tmp_16100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 54), 'tmp', False)
    # Obtaining the member 'dtype' of a type (line 636)
    dtype_16101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 54), tmp_16100, 'dtype')
    # Applying the binary operator '%' (line 636)
    result_mod_16102 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 25), '%', str_16099, dtype_16101)
    
    # Processing the call keyword arguments (line 636)
    kwargs_16103 = {}
    # Getting the type of 'ValueError' (line 636)
    ValueError_16098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 636)
    ValueError_call_result_16104 = invoke(stypy.reporting.localization.Localization(__file__, 636, 14), ValueError_16098, *[result_mod_16102], **kwargs_16103)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 636, 8), ValueError_call_result_16104, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    
    # Call to istype(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'tmp' (line 638)
    tmp_16106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 19), 'tmp', False)
    # Getting the type of 'numpy' (line 638)
    numpy_16107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 24), 'numpy', False)
    # Obtaining the member 'complex64' of a type (line 638)
    complex64_16108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 24), numpy_16107, 'complex64')
    # Processing the call keyword arguments (line 638)
    kwargs_16109 = {}
    # Getting the type of 'istype' (line 638)
    istype_16105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'istype', False)
    # Calling istype(args, kwargs) (line 638)
    istype_call_result_16110 = invoke(stypy.reporting.localization.Localization(__file__, 638, 12), istype_16105, *[tmp_16106, complex64_16108], **kwargs_16109)
    
    
    # Call to istype(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'tmp' (line 638)
    tmp_16112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 51), 'tmp', False)
    # Getting the type of 'numpy' (line 638)
    numpy_16113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 56), 'numpy', False)
    # Obtaining the member 'complex128' of a type (line 638)
    complex128_16114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 638, 56), numpy_16113, 'complex128')
    # Processing the call keyword arguments (line 638)
    kwargs_16115 = {}
    # Getting the type of 'istype' (line 638)
    istype_16111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 44), 'istype', False)
    # Calling istype(args, kwargs) (line 638)
    istype_call_result_16116 = invoke(stypy.reporting.localization.Localization(__file__, 638, 44), istype_16111, *[tmp_16112, complex128_16114], **kwargs_16115)
    
    # Applying the binary operator 'or' (line 638)
    result_or_keyword_16117 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 12), 'or', istype_call_result_16110, istype_call_result_16116)
    
    # Applying the 'not' unary operator (line 638)
    result_not__16118 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 7), 'not', result_or_keyword_16117)
    
    # Testing the type of an if condition (line 638)
    if_condition_16119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 638, 4), result_not__16118)
    # Assigning a type to the variable 'if_condition_16119' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'if_condition_16119', if_condition_16119)
    # SSA begins for if statement (line 638)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 639):
    
    # Assigning a Num to a Name (line 639):
    int_16120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 22), 'int')
    # Assigning a type to the variable 'overwrite_x' (line 639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 8), 'overwrite_x', int_16120)
    # SSA join for if statement (line 638)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 641):
    
    # Assigning a BoolOp to a Name (line 641):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_x' (line 641)
    overwrite_x_16121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 18), 'overwrite_x')
    
    # Call to _datacopied(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'tmp' (line 641)
    tmp_16123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 45), 'tmp', False)
    # Getting the type of 'x' (line 641)
    x_16124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 50), 'x', False)
    # Processing the call keyword arguments (line 641)
    kwargs_16125 = {}
    # Getting the type of '_datacopied' (line 641)
    _datacopied_16122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 641)
    _datacopied_call_result_16126 = invoke(stypy.reporting.localization.Localization(__file__, 641, 33), _datacopied_16122, *[tmp_16123, x_16124], **kwargs_16125)
    
    # Applying the binary operator 'or' (line 641)
    result_or_keyword_16127 = python_operator(stypy.reporting.localization.Localization(__file__, 641, 18), 'or', overwrite_x_16121, _datacopied_call_result_16126)
    
    # Assigning a type to the variable 'overwrite_x' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'overwrite_x', result_or_keyword_16127)
    
    # Call to _raw_fftnd(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'tmp' (line 642)
    tmp_16129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 22), 'tmp', False)
    # Getting the type of 'shape' (line 642)
    shape_16130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 26), 'shape', False)
    # Getting the type of 'axes' (line 642)
    axes_16131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 32), 'axes', False)
    # Getting the type of 'direction' (line 642)
    direction_16132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 37), 'direction', False)
    # Getting the type of 'overwrite_x' (line 642)
    overwrite_x_16133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 47), 'overwrite_x', False)
    # Getting the type of 'work_function' (line 642)
    work_function_16134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 59), 'work_function', False)
    # Processing the call keyword arguments (line 642)
    kwargs_16135 = {}
    # Getting the type of '_raw_fftnd' (line 642)
    _raw_fftnd_16128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 11), '_raw_fftnd', False)
    # Calling _raw_fftnd(args, kwargs) (line 642)
    _raw_fftnd_call_result_16136 = invoke(stypy.reporting.localization.Localization(__file__, 642, 11), _raw_fftnd_16128, *[tmp_16129, shape_16130, axes_16131, direction_16132, overwrite_x_16133, work_function_16134], **kwargs_16135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'stypy_return_type', _raw_fftnd_call_result_16136)
    
    # ################# End of '_raw_fftn_dispatch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_fftn_dispatch' in the type store
    # Getting the type of 'stypy_return_type' (line 630)
    stypy_return_type_16137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16137)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_fftn_dispatch'
    return stypy_return_type_16137

# Assigning a type to the variable '_raw_fftn_dispatch' (line 630)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 0), '_raw_fftn_dispatch', _raw_fftn_dispatch)

@norecursion
def ifftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 645)
    None_16138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 19), 'None')
    # Getting the type of 'None' (line 645)
    None_16139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 30), 'None')
    # Getting the type of 'False' (line 645)
    False_16140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 48), 'False')
    defaults = [None_16138, None_16139, False_16140]
    # Create a new context for function 'ifftn'
    module_type_store = module_type_store.open_function_context('ifftn', 645, 0, False)
    
    # Passed parameters checking function
    ifftn.stypy_localization = localization
    ifftn.stypy_type_of_self = None
    ifftn.stypy_type_store = module_type_store
    ifftn.stypy_function_name = 'ifftn'
    ifftn.stypy_param_names_list = ['x', 'shape', 'axes', 'overwrite_x']
    ifftn.stypy_varargs_param_name = None
    ifftn.stypy_kwargs_param_name = None
    ifftn.stypy_call_defaults = defaults
    ifftn.stypy_call_varargs = varargs
    ifftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifftn', ['x', 'shape', 'axes', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifftn', localization, ['x', 'shape', 'axes', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifftn(...)' code ##################

    str_16141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, (-1)), 'str', '\n    Return inverse multi-dimensional discrete Fourier transform of\n    arbitrary type sequence x.\n\n    The returned array contains::\n\n      y[j_1,..,j_d] = 1/p * sum[k_1=0..n_1-1, ..., k_d=0..n_d-1]\n         x[k_1,..,k_d] * prod[i=1..d] exp(sqrt(-1)*2*pi/n_i * j_i * k_i)\n\n    where ``d = len(x.shape)``, ``n = x.shape``, and ``p = prod[i=1..d] n_i``.\n\n    For description of parameters see `fftn`.\n\n    See Also\n    --------\n    fftn : for detailed information.\n\n    ')
    
    # Call to _raw_fftn_dispatch(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'x' (line 664)
    x_16143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 30), 'x', False)
    # Getting the type of 'shape' (line 664)
    shape_16144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 33), 'shape', False)
    # Getting the type of 'axes' (line 664)
    axes_16145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 40), 'axes', False)
    # Getting the type of 'overwrite_x' (line 664)
    overwrite_x_16146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 46), 'overwrite_x', False)
    int_16147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 59), 'int')
    # Processing the call keyword arguments (line 664)
    kwargs_16148 = {}
    # Getting the type of '_raw_fftn_dispatch' (line 664)
    _raw_fftn_dispatch_16142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 11), '_raw_fftn_dispatch', False)
    # Calling _raw_fftn_dispatch(args, kwargs) (line 664)
    _raw_fftn_dispatch_call_result_16149 = invoke(stypy.reporting.localization.Localization(__file__, 664, 11), _raw_fftn_dispatch_16142, *[x_16143, shape_16144, axes_16145, overwrite_x_16146, int_16147], **kwargs_16148)
    
    # Assigning a type to the variable 'stypy_return_type' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 4), 'stypy_return_type', _raw_fftn_dispatch_call_result_16149)
    
    # ################# End of 'ifftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifftn' in the type store
    # Getting the type of 'stypy_return_type' (line 645)
    stypy_return_type_16150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16150)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifftn'
    return stypy_return_type_16150

# Assigning a type to the variable 'ifftn' (line 645)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 0), 'ifftn', ifftn)

@norecursion
def fft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 667)
    None_16151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 18), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 667)
    tuple_16152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 667)
    # Adding element type (line 667)
    int_16153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 30), tuple_16152, int_16153)
    # Adding element type (line 667)
    int_16154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 667, 30), tuple_16152, int_16154)
    
    # Getting the type of 'False' (line 667)
    False_16155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 50), 'False')
    defaults = [None_16151, tuple_16152, False_16155]
    # Create a new context for function 'fft2'
    module_type_store = module_type_store.open_function_context('fft2', 667, 0, False)
    
    # Passed parameters checking function
    fft2.stypy_localization = localization
    fft2.stypy_type_of_self = None
    fft2.stypy_type_store = module_type_store
    fft2.stypy_function_name = 'fft2'
    fft2.stypy_param_names_list = ['x', 'shape', 'axes', 'overwrite_x']
    fft2.stypy_varargs_param_name = None
    fft2.stypy_kwargs_param_name = None
    fft2.stypy_call_defaults = defaults
    fft2.stypy_call_varargs = varargs
    fft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fft2', ['x', 'shape', 'axes', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fft2', localization, ['x', 'shape', 'axes', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fft2(...)' code ##################

    str_16156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, (-1)), 'str', '\n    2-D discrete Fourier transform.\n\n    Return the two-dimensional discrete Fourier transform of the 2-D argument\n    `x`.\n\n    See Also\n    --------\n    fftn : for detailed information.\n\n    ')
    
    # Call to fftn(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'x' (line 679)
    x_16158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'x', False)
    # Getting the type of 'shape' (line 679)
    shape_16159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 18), 'shape', False)
    # Getting the type of 'axes' (line 679)
    axes_16160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 24), 'axes', False)
    # Getting the type of 'overwrite_x' (line 679)
    overwrite_x_16161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 29), 'overwrite_x', False)
    # Processing the call keyword arguments (line 679)
    kwargs_16162 = {}
    # Getting the type of 'fftn' (line 679)
    fftn_16157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 11), 'fftn', False)
    # Calling fftn(args, kwargs) (line 679)
    fftn_call_result_16163 = invoke(stypy.reporting.localization.Localization(__file__, 679, 11), fftn_16157, *[x_16158, shape_16159, axes_16160, overwrite_x_16161], **kwargs_16162)
    
    # Assigning a type to the variable 'stypy_return_type' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'stypy_return_type', fftn_call_result_16163)
    
    # ################# End of 'fft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fft2' in the type store
    # Getting the type of 'stypy_return_type' (line 667)
    stypy_return_type_16164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16164)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fft2'
    return stypy_return_type_16164

# Assigning a type to the variable 'fft2' (line 667)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 0), 'fft2', fft2)

@norecursion
def ifft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 682)
    None_16165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 19), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 682)
    tuple_16166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 682)
    # Adding element type (line 682)
    int_16167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 31), tuple_16166, int_16167)
    # Adding element type (line 682)
    int_16168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 31), tuple_16166, int_16168)
    
    # Getting the type of 'False' (line 682)
    False_16169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 51), 'False')
    defaults = [None_16165, tuple_16166, False_16169]
    # Create a new context for function 'ifft2'
    module_type_store = module_type_store.open_function_context('ifft2', 682, 0, False)
    
    # Passed parameters checking function
    ifft2.stypy_localization = localization
    ifft2.stypy_type_of_self = None
    ifft2.stypy_type_store = module_type_store
    ifft2.stypy_function_name = 'ifft2'
    ifft2.stypy_param_names_list = ['x', 'shape', 'axes', 'overwrite_x']
    ifft2.stypy_varargs_param_name = None
    ifft2.stypy_kwargs_param_name = None
    ifft2.stypy_call_defaults = defaults
    ifft2.stypy_call_varargs = varargs
    ifft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifft2', ['x', 'shape', 'axes', 'overwrite_x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifft2', localization, ['x', 'shape', 'axes', 'overwrite_x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifft2(...)' code ##################

    str_16170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, (-1)), 'str', '\n    2-D discrete inverse Fourier transform of real or complex sequence.\n\n    Return inverse two-dimensional discrete Fourier transform of\n    arbitrary type sequence x.\n\n    See `ifft` for more information.\n\n    See also\n    --------\n    fft2, ifft\n\n    ')
    
    # Call to ifftn(...): (line 696)
    # Processing the call arguments (line 696)
    # Getting the type of 'x' (line 696)
    x_16172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 17), 'x', False)
    # Getting the type of 'shape' (line 696)
    shape_16173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 19), 'shape', False)
    # Getting the type of 'axes' (line 696)
    axes_16174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 25), 'axes', False)
    # Getting the type of 'overwrite_x' (line 696)
    overwrite_x_16175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 30), 'overwrite_x', False)
    # Processing the call keyword arguments (line 696)
    kwargs_16176 = {}
    # Getting the type of 'ifftn' (line 696)
    ifftn_16171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 11), 'ifftn', False)
    # Calling ifftn(args, kwargs) (line 696)
    ifftn_call_result_16177 = invoke(stypy.reporting.localization.Localization(__file__, 696, 11), ifftn_16171, *[x_16172, shape_16173, axes_16174, overwrite_x_16175], **kwargs_16176)
    
    # Assigning a type to the variable 'stypy_return_type' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'stypy_return_type', ifftn_call_result_16177)
    
    # ################# End of 'ifft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifft2' in the type store
    # Getting the type of 'stypy_return_type' (line 682)
    stypy_return_type_16178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16178)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifft2'
    return stypy_return_type_16178

# Assigning a type to the variable 'ifft2' (line 682)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 0), 'ifft2', ifft2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
