
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Discrete Fourier Transforms
3: 
4: Routines in this module:
5: 
6: fft(a, n=None, axis=-1)
7: ifft(a, n=None, axis=-1)
8: rfft(a, n=None, axis=-1)
9: irfft(a, n=None, axis=-1)
10: hfft(a, n=None, axis=-1)
11: ihfft(a, n=None, axis=-1)
12: fftn(a, s=None, axes=None)
13: ifftn(a, s=None, axes=None)
14: rfftn(a, s=None, axes=None)
15: irfftn(a, s=None, axes=None)
16: fft2(a, s=None, axes=(-2,-1))
17: ifft2(a, s=None, axes=(-2, -1))
18: rfft2(a, s=None, axes=(-2,-1))
19: irfft2(a, s=None, axes=(-2, -1))
20: 
21: i = inverse transform
22: r = transform of purely real data
23: h = Hermite transform
24: n = n-dimensional transform
25: 2 = 2-dimensional transform
26: (Note: 2D routines are just nD routines with different default
27: behavior.)
28: 
29: The underlying code for these functions is an f2c-translated and modified
30: version of the FFTPACK routines.
31: 
32: '''
33: from __future__ import division, absolute_import, print_function
34: 
35: __all__ = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn',
36:            'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']
37: 
38: from numpy.core import (array, asarray, zeros, swapaxes, shape, conjugate,
39:                         take, sqrt)
40: from . import fftpack_lite as fftpack
41: 
42: _fft_cache = {}
43: _real_fft_cache = {}
44: 
45: 
46: def _raw_fft(a, n=None, axis=-1, init_function=fftpack.cffti,
47:              work_function=fftpack.cfftf, fft_cache=_fft_cache):
48:     a = asarray(a)
49: 
50:     if n is None:
51:         n = a.shape[axis]
52: 
53:     if n < 1:
54:         raise ValueError("Invalid number of FFT data points (%d) specified."
55:                          % n)
56: 
57:     try:
58:         # Thread-safety note: We rely on list.pop() here to atomically
59:         # retrieve-and-remove a wsave from the cache.  This ensures that no
60:         # other thread can get the same wsave while we're using it.
61:         wsave = fft_cache.setdefault(n, []).pop()
62:     except (IndexError):
63:         wsave = init_function(n)
64: 
65:     if a.shape[axis] != n:
66:         s = list(a.shape)
67:         if s[axis] > n:
68:             index = [slice(None)]*len(s)
69:             index[axis] = slice(0, n)
70:             a = a[index]
71:         else:
72:             index = [slice(None)]*len(s)
73:             index[axis] = slice(0, s[axis])
74:             s[axis] = n
75:             z = zeros(s, a.dtype.char)
76:             z[index] = a
77:             a = z
78: 
79:     if axis != -1:
80:         a = swapaxes(a, axis, -1)
81:     r = work_function(a, wsave)
82:     if axis != -1:
83:         r = swapaxes(r, axis, -1)
84: 
85:     # As soon as we put wsave back into the cache, another thread could pick it
86:     # up and start using it, so we must not do this until after we're
87:     # completely done using it ourselves.
88:     fft_cache[n].append(wsave)
89: 
90:     return r
91: 
92: 
93: def _unitary(norm):
94:     if norm not in (None, "ortho"):
95:         raise ValueError("Invalid norm value %s, should be None or \"ortho\"."
96:                          % norm)
97:     return norm is not None
98: 
99: 
100: def fft(a, n=None, axis=-1, norm=None):
101:     '''
102:     Compute the one-dimensional discrete Fourier Transform.
103: 
104:     This function computes the one-dimensional *n*-point discrete Fourier
105:     Transform (DFT) with the efficient Fast Fourier Transform (FFT)
106:     algorithm [CT].
107: 
108:     Parameters
109:     ----------
110:     a : array_like
111:         Input array, can be complex.
112:     n : int, optional
113:         Length of the transformed axis of the output.
114:         If `n` is smaller than the length of the input, the input is cropped.
115:         If it is larger, the input is padded with zeros.  If `n` is not given,
116:         the length of the input along the axis specified by `axis` is used.
117:     axis : int, optional
118:         Axis over which to compute the FFT.  If not given, the last axis is
119:         used.
120:     norm : {None, "ortho"}, optional
121:         .. versionadded:: 1.10.0
122:         Normalization mode (see `numpy.fft`). Default is None.
123: 
124:     Returns
125:     -------
126:     out : complex ndarray
127:         The truncated or zero-padded input, transformed along the axis
128:         indicated by `axis`, or the last one if `axis` is not specified.
129: 
130:     Raises
131:     ------
132:     IndexError
133:         if `axes` is larger than the last axis of `a`.
134: 
135:     See Also
136:     --------
137:     numpy.fft : for definition of the DFT and conventions used.
138:     ifft : The inverse of `fft`.
139:     fft2 : The two-dimensional FFT.
140:     fftn : The *n*-dimensional FFT.
141:     rfftn : The *n*-dimensional FFT of real input.
142:     fftfreq : Frequency bins for given FFT parameters.
143: 
144:     Notes
145:     -----
146:     FFT (Fast Fourier Transform) refers to a way the discrete Fourier
147:     Transform (DFT) can be calculated efficiently, by using symmetries in the
148:     calculated terms.  The symmetry is highest when `n` is a power of 2, and
149:     the transform is therefore most efficient for these sizes.
150: 
151:     The DFT is defined, with the conventions used in this implementation, in
152:     the documentation for the `numpy.fft` module.
153: 
154:     References
155:     ----------
156:     .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
157:             machine calculation of complex Fourier series," *Math. Comput.*
158:             19: 297-301.
159: 
160:     Examples
161:     --------
162:     >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
163:     array([ -3.44505240e-16 +1.14383329e-17j,
164:              8.00000000e+00 -5.71092652e-15j,
165:              2.33482938e-16 +1.22460635e-16j,
166:              1.64863782e-15 +1.77635684e-15j,
167:              9.95839695e-17 +2.33482938e-16j,
168:              0.00000000e+00 +1.66837030e-15j,
169:              1.14383329e-17 +1.22460635e-16j,
170:              -1.64863782e-15 +1.77635684e-15j])
171: 
172:     >>> import matplotlib.pyplot as plt
173:     >>> t = np.arange(256)
174:     >>> sp = np.fft.fft(np.sin(t))
175:     >>> freq = np.fft.fftfreq(t.shape[-1])
176:     >>> plt.plot(freq, sp.real, freq, sp.imag)
177:     [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
178:     >>> plt.show()
179: 
180:     In this example, real input has an FFT which is Hermitian, i.e., symmetric
181:     in the real part and anti-symmetric in the imaginary part, as described in
182:     the `numpy.fft` documentation.
183: 
184:     '''
185: 
186:     a = asarray(a).astype(complex, copy=False)
187:     if n is None:
188:         n = a.shape[axis]
189:     output = _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftf, _fft_cache)
190:     if _unitary(norm):
191:         output *= 1 / sqrt(n)
192:     return output
193: 
194: 
195: def ifft(a, n=None, axis=-1, norm=None):
196:     '''
197:     Compute the one-dimensional inverse discrete Fourier Transform.
198: 
199:     This function computes the inverse of the one-dimensional *n*-point
200:     discrete Fourier transform computed by `fft`.  In other words,
201:     ``ifft(fft(a)) == a`` to within numerical accuracy.
202:     For a general description of the algorithm and definitions,
203:     see `numpy.fft`.
204: 
205:     The input should be ordered in the same way as is returned by `fft`,
206:     i.e.,
207: 
208:     * ``a[0]`` should contain the zero frequency term,
209:     * ``a[1:n//2]`` should contain the positive-frequency terms,
210:     * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
211:       increasing order starting from the most negative frequency.
212: 
213:     For an even number of input points, ``A[n//2]`` represents the sum of
214:     the values at the positive and negative Nyquist frequencies, as the two
215:     are aliased together. See `numpy.fft` for details.
216: 
217:     Parameters
218:     ----------
219:     a : array_like
220:         Input array, can be complex.
221:     n : int, optional
222:         Length of the transformed axis of the output.
223:         If `n` is smaller than the length of the input, the input is cropped.
224:         If it is larger, the input is padded with zeros.  If `n` is not given,
225:         the length of the input along the axis specified by `axis` is used.
226:         See notes about padding issues.
227:     axis : int, optional
228:         Axis over which to compute the inverse DFT.  If not given, the last
229:         axis is used.
230:     norm : {None, "ortho"}, optional
231:         .. versionadded:: 1.10.0
232:         Normalization mode (see `numpy.fft`). Default is None.
233: 
234:     Returns
235:     -------
236:     out : complex ndarray
237:         The truncated or zero-padded input, transformed along the axis
238:         indicated by `axis`, or the last one if `axis` is not specified.
239: 
240:     Raises
241:     ------
242:     IndexError
243:         If `axes` is larger than the last axis of `a`.
244: 
245:     See Also
246:     --------
247:     numpy.fft : An introduction, with definitions and general explanations.
248:     fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse
249:     ifft2 : The two-dimensional inverse FFT.
250:     ifftn : The n-dimensional inverse FFT.
251: 
252:     Notes
253:     -----
254:     If the input parameter `n` is larger than the size of the input, the input
255:     is padded by appending zeros at the end.  Even though this is the common
256:     approach, it might lead to surprising results.  If a different padding is
257:     desired, it must be performed before calling `ifft`.
258: 
259:     Examples
260:     --------
261:     >>> np.fft.ifft([0, 4, 0, 0])
262:     array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j])
263: 
264:     Create and plot a band-limited signal with random phases:
265: 
266:     >>> import matplotlib.pyplot as plt
267:     >>> t = np.arange(400)
268:     >>> n = np.zeros((400,), dtype=complex)
269:     >>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))
270:     >>> s = np.fft.ifft(n)
271:     >>> plt.plot(t, s.real, 'b-', t, s.imag, 'r--')
272:     ...
273:     >>> plt.legend(('real', 'imaginary'))
274:     ...
275:     >>> plt.show()
276: 
277:     '''
278:     # The copy may be required for multithreading.
279:     a = array(a, copy=True, dtype=complex)
280:     if n is None:
281:         n = a.shape[axis]
282:     unitary = _unitary(norm)
283:     output = _raw_fft(a, n, axis, fftpack.cffti, fftpack.cfftb, _fft_cache)
284:     return output * (1 / (sqrt(n) if unitary else n))
285: 
286: 
287: def rfft(a, n=None, axis=-1, norm=None):
288:     '''
289:     Compute the one-dimensional discrete Fourier Transform for real input.
290: 
291:     This function computes the one-dimensional *n*-point discrete Fourier
292:     Transform (DFT) of a real-valued array by means of an efficient algorithm
293:     called the Fast Fourier Transform (FFT).
294: 
295:     Parameters
296:     ----------
297:     a : array_like
298:         Input array
299:     n : int, optional
300:         Number of points along transformation axis in the input to use.
301:         If `n` is smaller than the length of the input, the input is cropped.
302:         If it is larger, the input is padded with zeros. If `n` is not given,
303:         the length of the input along the axis specified by `axis` is used.
304:     axis : int, optional
305:         Axis over which to compute the FFT. If not given, the last axis is
306:         used.
307:     norm : {None, "ortho"}, optional
308:         .. versionadded:: 1.10.0
309:         Normalization mode (see `numpy.fft`). Default is None.
310: 
311:     Returns
312:     -------
313:     out : complex ndarray
314:         The truncated or zero-padded input, transformed along the axis
315:         indicated by `axis`, or the last one if `axis` is not specified.
316:         If `n` is even, the length of the transformed axis is ``(n/2)+1``.
317:         If `n` is odd, the length is ``(n+1)/2``.
318: 
319:     Raises
320:     ------
321:     IndexError
322:         If `axis` is larger than the last axis of `a`.
323: 
324:     See Also
325:     --------
326:     numpy.fft : For definition of the DFT and conventions used.
327:     irfft : The inverse of `rfft`.
328:     fft : The one-dimensional FFT of general (complex) input.
329:     fftn : The *n*-dimensional FFT.
330:     rfftn : The *n*-dimensional FFT of real input.
331: 
332:     Notes
333:     -----
334:     When the DFT is computed for purely real input, the output is
335:     Hermitian-symmetric, i.e. the negative frequency terms are just the complex
336:     conjugates of the corresponding positive-frequency terms, and the
337:     negative-frequency terms are therefore redundant.  This function does not
338:     compute the negative frequency terms, and the length of the transformed
339:     axis of the output is therefore ``n//2 + 1``.
340: 
341:     When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains
342:     the zero-frequency term 0*fs, which is real due to Hermitian symmetry.
343: 
344:     If `n` is even, ``A[-1]`` contains the term representing both positive
345:     and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
346:     real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
347:     the largest positive frequency (fs/2*(n-1)/n), and is complex in the
348:     general case.
349: 
350:     If the input `a` contains an imaginary part, it is silently discarded.
351: 
352:     Examples
353:     --------
354:     >>> np.fft.fft([0, 1, 0, 0])
355:     array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j])
356:     >>> np.fft.rfft([0, 1, 0, 0])
357:     array([ 1.+0.j,  0.-1.j, -1.+0.j])
358: 
359:     Notice how the final element of the `fft` output is the complex conjugate
360:     of the second element, for real input. For `rfft`, this symmetry is
361:     exploited to compute only the non-negative frequency terms.
362: 
363:     '''
364:     # The copy may be required for multithreading.
365:     a = array(a, copy=True, dtype=float)
366:     output = _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftf,
367:                       _real_fft_cache)
368:     if _unitary(norm):
369:         output *= 1 / sqrt(a.shape[axis])
370:     return output
371: 
372: 
373: def irfft(a, n=None, axis=-1, norm=None):
374:     '''
375:     Compute the inverse of the n-point DFT for real input.
376: 
377:     This function computes the inverse of the one-dimensional *n*-point
378:     discrete Fourier Transform of real input computed by `rfft`.
379:     In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical
380:     accuracy. (See Notes below for why ``len(a)`` is necessary here.)
381: 
382:     The input is expected to be in the form returned by `rfft`, i.e. the
383:     real zero-frequency term followed by the complex positive frequency terms
384:     in order of increasing frequency.  Since the discrete Fourier Transform of
385:     real input is Hermitian-symmetric, the negative frequency terms are taken
386:     to be the complex conjugates of the corresponding positive frequency terms.
387: 
388:     Parameters
389:     ----------
390:     a : array_like
391:         The input array.
392:     n : int, optional
393:         Length of the transformed axis of the output.
394:         For `n` output points, ``n//2+1`` input points are necessary.  If the
395:         input is longer than this, it is cropped.  If it is shorter than this,
396:         it is padded with zeros.  If `n` is not given, it is determined from
397:         the length of the input along the axis specified by `axis`.
398:     axis : int, optional
399:         Axis over which to compute the inverse FFT. If not given, the last
400:         axis is used.
401:     norm : {None, "ortho"}, optional
402:         .. versionadded:: 1.10.0
403:         Normalization mode (see `numpy.fft`). Default is None.
404: 
405:     Returns
406:     -------
407:     out : ndarray
408:         The truncated or zero-padded input, transformed along the axis
409:         indicated by `axis`, or the last one if `axis` is not specified.
410:         The length of the transformed axis is `n`, or, if `n` is not given,
411:         ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
412:         input. To get an odd number of output points, `n` must be specified.
413: 
414:     Raises
415:     ------
416:     IndexError
417:         If `axis` is larger than the last axis of `a`.
418: 
419:     See Also
420:     --------
421:     numpy.fft : For definition of the DFT and conventions used.
422:     rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.
423:     fft : The one-dimensional FFT.
424:     irfft2 : The inverse of the two-dimensional FFT of real input.
425:     irfftn : The inverse of the *n*-dimensional FFT of real input.
426: 
427:     Notes
428:     -----
429:     Returns the real valued `n`-point inverse discrete Fourier transform
430:     of `a`, where `a` contains the non-negative frequency terms of a
431:     Hermitian-symmetric sequence. `n` is the length of the result, not the
432:     input.
433: 
434:     If you specify an `n` such that `a` must be zero-padded or truncated, the
435:     extra/removed values will be added/removed at high frequencies. One can
436:     thus resample a series to `m` points via Fourier interpolation by:
437:     ``a_resamp = irfft(rfft(a), m)``.
438: 
439:     Examples
440:     --------
441:     >>> np.fft.ifft([1, -1j, -1, 1j])
442:     array([ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j])
443:     >>> np.fft.irfft([1, -1j, -1])
444:     array([ 0.,  1.,  0.,  0.])
445: 
446:     Notice how the last term in the input to the ordinary `ifft` is the
447:     complex conjugate of the second term, and the output has zero imaginary
448:     part everywhere.  When calling `irfft`, the negative frequencies are not
449:     specified, and the output array is purely real.
450: 
451:     '''
452:     # The copy may be required for multithreading.
453:     a = array(a, copy=True, dtype=complex)
454:     if n is None:
455:         n = (a.shape[axis] - 1) * 2
456:     unitary = _unitary(norm)
457:     output = _raw_fft(a, n, axis, fftpack.rffti, fftpack.rfftb,
458:                       _real_fft_cache)
459:     return output * (1 / (sqrt(n) if unitary else n))
460: 
461: 
462: def hfft(a, n=None, axis=-1, norm=None):
463:     '''
464:     Compute the FFT of a signal which has Hermitian symmetry (real spectrum).
465: 
466:     Parameters
467:     ----------
468:     a : array_like
469:         The input array.
470:     n : int, optional
471:         Length of the transformed axis of the output.
472:         For `n` output points, ``n//2+1`` input points are necessary.  If the
473:         input is longer than this, it is cropped.  If it is shorter than this,
474:         it is padded with zeros.  If `n` is not given, it is determined from
475:         the length of the input along the axis specified by `axis`.
476:     axis : int, optional
477:         Axis over which to compute the FFT. If not given, the last
478:         axis is used.
479:     norm : {None, "ortho"}, optional
480:         .. versionadded:: 1.10.0
481:         Normalization mode (see `numpy.fft`). Default is None.
482: 
483:     Returns
484:     -------
485:     out : ndarray
486:         The truncated or zero-padded input, transformed along the axis
487:         indicated by `axis`, or the last one if `axis` is not specified.
488:         The length of the transformed axis is `n`, or, if `n` is not given,
489:         ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
490:         input. To get an odd number of output points, `n` must be specified.
491: 
492:     Raises
493:     ------
494:     IndexError
495:         If `axis` is larger than the last axis of `a`.
496: 
497:     See also
498:     --------
499:     rfft : Compute the one-dimensional FFT for real input.
500:     ihfft : The inverse of `hfft`.
501: 
502:     Notes
503:     -----
504:     `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
505:     opposite case: here the signal has Hermitian symmetry in the time domain
506:     and is real in the frequency domain. So here it's `hfft` for which
507:     you must supply the length of the result if it is to be odd:
508:     ``ihfft(hfft(a), len(a)) == a``, within numerical accuracy.
509: 
510:     Examples
511:     --------
512:     >>> signal = np.array([1, 2, 3, 4, 3, 2])
513:     >>> np.fft.fft(signal)
514:     array([ 15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j])
515:     >>> np.fft.hfft(signal[:4]) # Input first half of signal
516:     array([ 15.,  -4.,   0.,  -1.,   0.,  -4.])
517:     >>> np.fft.hfft(signal, 6)  # Input entire signal and truncate
518:     array([ 15.,  -4.,   0.,  -1.,   0.,  -4.])
519: 
520: 
521:     >>> signal = np.array([[1, 1.j], [-1.j, 2]])
522:     >>> np.conj(signal.T) - signal   # check Hermitian symmetry
523:     array([[ 0.-0.j,  0.+0.j],
524:            [ 0.+0.j,  0.-0.j]])
525:     >>> freq_spectrum = np.fft.hfft(signal)
526:     >>> freq_spectrum
527:     array([[ 1.,  1.],
528:            [ 2., -2.]])
529: 
530:     '''
531:     # The copy may be required for multithreading.
532:     a = array(a, copy=True, dtype=complex)
533:     if n is None:
534:         n = (a.shape[axis] - 1) * 2
535:     unitary = _unitary(norm)
536:     return irfft(conjugate(a), n, axis) * (sqrt(n) if unitary else n)
537: 
538: 
539: def ihfft(a, n=None, axis=-1, norm=None):
540:     '''
541:     Compute the inverse FFT of a signal which has Hermitian symmetry.
542: 
543:     Parameters
544:     ----------
545:     a : array_like
546:         Input array.
547:     n : int, optional
548:         Length of the inverse FFT.
549:         Number of points along transformation axis in the input to use.
550:         If `n` is smaller than the length of the input, the input is cropped.
551:         If it is larger, the input is padded with zeros. If `n` is not given,
552:         the length of the input along the axis specified by `axis` is used.
553:     axis : int, optional
554:         Axis over which to compute the inverse FFT. If not given, the last
555:         axis is used.
556:     norm : {None, "ortho"}, optional
557:         .. versionadded:: 1.10.0
558:         Normalization mode (see `numpy.fft`). Default is None.
559: 
560:     Returns
561:     -------
562:     out : complex ndarray
563:         The truncated or zero-padded input, transformed along the axis
564:         indicated by `axis`, or the last one if `axis` is not specified.
565:         If `n` is even, the length of the transformed axis is ``(n/2)+1``.
566:         If `n` is odd, the length is ``(n+1)/2``.
567: 
568:     See also
569:     --------
570:     hfft, irfft
571: 
572:     Notes
573:     -----
574:     `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the
575:     opposite case: here the signal has Hermitian symmetry in the time domain
576:     and is real in the frequency domain. So here it's `hfft` for which
577:     you must supply the length of the result if it is to be odd:
578:     ``ihfft(hfft(a), len(a)) == a``, within numerical accuracy.
579: 
580:     Examples
581:     --------
582:     >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])
583:     >>> np.fft.ifft(spectrum)
584:     array([ 1.+0.j,  2.-0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.-0.j])
585:     >>> np.fft.ihfft(spectrum)
586:     array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j])
587: 
588:     '''
589:     # The copy may be required for multithreading.
590:     a = array(a, copy=True, dtype=float)
591:     if n is None:
592:         n = a.shape[axis]
593:     unitary = _unitary(norm)
594:     output = conjugate(rfft(a, n, axis))
595:     return output * (1 / (sqrt(n) if unitary else n))
596: 
597: 
598: def _cook_nd_args(a, s=None, axes=None, invreal=0):
599:     if s is None:
600:         shapeless = 1
601:         if axes is None:
602:             s = list(a.shape)
603:         else:
604:             s = take(a.shape, axes)
605:     else:
606:         shapeless = 0
607:     s = list(s)
608:     if axes is None:
609:         axes = list(range(-len(s), 0))
610:     if len(s) != len(axes):
611:         raise ValueError("Shape and axes have different lengths.")
612:     if invreal and shapeless:
613:         s[-1] = (a.shape[axes[-1]] - 1) * 2
614:     return s, axes
615: 
616: 
617: def _raw_fftnd(a, s=None, axes=None, function=fft, norm=None):
618:     a = asarray(a)
619:     s, axes = _cook_nd_args(a, s, axes)
620:     itl = list(range(len(axes)))
621:     itl.reverse()
622:     for ii in itl:
623:         a = function(a, n=s[ii], axis=axes[ii], norm=norm)
624:     return a
625: 
626: 
627: def fftn(a, s=None, axes=None, norm=None):
628:     '''
629:     Compute the N-dimensional discrete Fourier Transform.
630: 
631:     This function computes the *N*-dimensional discrete Fourier Transform over
632:     any number of axes in an *M*-dimensional array by means of the Fast Fourier
633:     Transform (FFT).
634: 
635:     Parameters
636:     ----------
637:     a : array_like
638:         Input array, can be complex.
639:     s : sequence of ints, optional
640:         Shape (length of each transformed axis) of the output
641:         (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).
642:         This corresponds to `n` for `fft(x, n)`.
643:         Along any axis, if the given shape is smaller than that of the input,
644:         the input is cropped.  If it is larger, the input is padded with zeros.
645:         if `s` is not given, the shape of the input along the axes specified
646:         by `axes` is used.
647:     axes : sequence of ints, optional
648:         Axes over which to compute the FFT.  If not given, the last ``len(s)``
649:         axes are used, or all axes if `s` is also not specified.
650:         Repeated indices in `axes` means that the transform over that axis is
651:         performed multiple times.
652:     norm : {None, "ortho"}, optional
653:         .. versionadded:: 1.10.0
654:         Normalization mode (see `numpy.fft`). Default is None.
655: 
656:     Returns
657:     -------
658:     out : complex ndarray
659:         The truncated or zero-padded input, transformed along the axes
660:         indicated by `axes`, or by a combination of `s` and `a`,
661:         as explained in the parameters section above.
662: 
663:     Raises
664:     ------
665:     ValueError
666:         If `s` and `axes` have different length.
667:     IndexError
668:         If an element of `axes` is larger than than the number of axes of `a`.
669: 
670:     See Also
671:     --------
672:     numpy.fft : Overall view of discrete Fourier transforms, with definitions
673:         and conventions used.
674:     ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.
675:     fft : The one-dimensional FFT, with definitions and conventions used.
676:     rfftn : The *n*-dimensional FFT of real input.
677:     fft2 : The two-dimensional FFT.
678:     fftshift : Shifts zero-frequency terms to centre of array
679: 
680:     Notes
681:     -----
682:     The output, analogously to `fft`, contains the term for zero frequency in
683:     the low-order corner of all axes, the positive frequency terms in the
684:     first half of all axes, the term for the Nyquist frequency in the middle
685:     of all axes and the negative frequency terms in the second half of all
686:     axes, in order of decreasingly negative frequency.
687: 
688:     See `numpy.fft` for details, definitions and conventions used.
689: 
690:     Examples
691:     --------
692:     >>> a = np.mgrid[:3, :3, :3][0]
693:     >>> np.fft.fftn(a, axes=(1, 2))
694:     array([[[  0.+0.j,   0.+0.j,   0.+0.j],
695:             [  0.+0.j,   0.+0.j,   0.+0.j],
696:             [  0.+0.j,   0.+0.j,   0.+0.j]],
697:            [[  9.+0.j,   0.+0.j,   0.+0.j],
698:             [  0.+0.j,   0.+0.j,   0.+0.j],
699:             [  0.+0.j,   0.+0.j,   0.+0.j]],
700:            [[ 18.+0.j,   0.+0.j,   0.+0.j],
701:             [  0.+0.j,   0.+0.j,   0.+0.j],
702:             [  0.+0.j,   0.+0.j,   0.+0.j]]])
703:     >>> np.fft.fftn(a, (2, 2), axes=(0, 1))
704:     array([[[ 2.+0.j,  2.+0.j,  2.+0.j],
705:             [ 0.+0.j,  0.+0.j,  0.+0.j]],
706:            [[-2.+0.j, -2.+0.j, -2.+0.j],
707:             [ 0.+0.j,  0.+0.j,  0.+0.j]]])
708: 
709:     >>> import matplotlib.pyplot as plt
710:     >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,
711:     ...                      2 * np.pi * np.arange(200) / 34)
712:     >>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)
713:     >>> FS = np.fft.fftn(S)
714:     >>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
715:     <matplotlib.image.AxesImage object at 0x...>
716:     >>> plt.show()
717: 
718:     '''
719: 
720:     return _raw_fftnd(a, s, axes, fft, norm)
721: 
722: 
723: def ifftn(a, s=None, axes=None, norm=None):
724:     '''
725:     Compute the N-dimensional inverse discrete Fourier Transform.
726: 
727:     This function computes the inverse of the N-dimensional discrete
728:     Fourier Transform over any number of axes in an M-dimensional array by
729:     means of the Fast Fourier Transform (FFT).  In other words,
730:     ``ifftn(fftn(a)) == a`` to within numerical accuracy.
731:     For a description of the definitions and conventions used, see `numpy.fft`.
732: 
733:     The input, analogously to `ifft`, should be ordered in the same way as is
734:     returned by `fftn`, i.e. it should have the term for zero frequency
735:     in all axes in the low-order corner, the positive frequency terms in the
736:     first half of all axes, the term for the Nyquist frequency in the middle
737:     of all axes and the negative frequency terms in the second half of all
738:     axes, in order of decreasingly negative frequency.
739: 
740:     Parameters
741:     ----------
742:     a : array_like
743:         Input array, can be complex.
744:     s : sequence of ints, optional
745:         Shape (length of each transformed axis) of the output
746:         (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
747:         This corresponds to ``n`` for ``ifft(x, n)``.
748:         Along any axis, if the given shape is smaller than that of the input,
749:         the input is cropped.  If it is larger, the input is padded with zeros.
750:         if `s` is not given, the shape of the input along the axes specified
751:         by `axes` is used.  See notes for issue on `ifft` zero padding.
752:     axes : sequence of ints, optional
753:         Axes over which to compute the IFFT.  If not given, the last ``len(s)``
754:         axes are used, or all axes if `s` is also not specified.
755:         Repeated indices in `axes` means that the inverse transform over that
756:         axis is performed multiple times.
757:     norm : {None, "ortho"}, optional
758:         .. versionadded:: 1.10.0
759:         Normalization mode (see `numpy.fft`). Default is None.
760: 
761:     Returns
762:     -------
763:     out : complex ndarray
764:         The truncated or zero-padded input, transformed along the axes
765:         indicated by `axes`, or by a combination of `s` or `a`,
766:         as explained in the parameters section above.
767: 
768:     Raises
769:     ------
770:     ValueError
771:         If `s` and `axes` have different length.
772:     IndexError
773:         If an element of `axes` is larger than than the number of axes of `a`.
774: 
775:     See Also
776:     --------
777:     numpy.fft : Overall view of discrete Fourier transforms, with definitions
778:          and conventions used.
779:     fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.
780:     ifft : The one-dimensional inverse FFT.
781:     ifft2 : The two-dimensional inverse FFT.
782:     ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning
783:         of array.
784: 
785:     Notes
786:     -----
787:     See `numpy.fft` for definitions and conventions used.
788: 
789:     Zero-padding, analogously with `ifft`, is performed by appending zeros to
790:     the input along the specified dimension.  Although this is the common
791:     approach, it might lead to surprising results.  If another form of zero
792:     padding is desired, it must be performed before `ifftn` is called.
793: 
794:     Examples
795:     --------
796:     >>> a = np.eye(4)
797:     >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))
798:     array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
799:            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
800:            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
801:            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])
802: 
803: 
804:     Create and plot an image with band-limited frequency content:
805: 
806:     >>> import matplotlib.pyplot as plt
807:     >>> n = np.zeros((200,200), dtype=complex)
808:     >>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))
809:     >>> im = np.fft.ifftn(n).real
810:     >>> plt.imshow(im)
811:     <matplotlib.image.AxesImage object at 0x...>
812:     >>> plt.show()
813: 
814:     '''
815: 
816:     return _raw_fftnd(a, s, axes, ifft, norm)
817: 
818: 
819: def fft2(a, s=None, axes=(-2, -1), norm=None):
820:     '''
821:     Compute the 2-dimensional discrete Fourier Transform
822: 
823:     This function computes the *n*-dimensional discrete Fourier Transform
824:     over any axes in an *M*-dimensional array by means of the
825:     Fast Fourier Transform (FFT).  By default, the transform is computed over
826:     the last two axes of the input array, i.e., a 2-dimensional FFT.
827: 
828:     Parameters
829:     ----------
830:     a : array_like
831:         Input array, can be complex
832:     s : sequence of ints, optional
833:         Shape (length of each transformed axis) of the output
834:         (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).
835:         This corresponds to `n` for `fft(x, n)`.
836:         Along each axis, if the given shape is smaller than that of the input,
837:         the input is cropped.  If it is larger, the input is padded with zeros.
838:         if `s` is not given, the shape of the input along the axes specified
839:         by `axes` is used.
840:     axes : sequence of ints, optional
841:         Axes over which to compute the FFT.  If not given, the last two
842:         axes are used.  A repeated index in `axes` means the transform over
843:         that axis is performed multiple times.  A one-element sequence means
844:         that a one-dimensional FFT is performed.
845:     norm : {None, "ortho"}, optional
846:         .. versionadded:: 1.10.0
847:         Normalization mode (see `numpy.fft`). Default is None.
848: 
849:     Returns
850:     -------
851:     out : complex ndarray
852:         The truncated or zero-padded input, transformed along the axes
853:         indicated by `axes`, or the last two axes if `axes` is not given.
854: 
855:     Raises
856:     ------
857:     ValueError
858:         If `s` and `axes` have different length, or `axes` not given and
859:         ``len(s) != 2``.
860:     IndexError
861:         If an element of `axes` is larger than than the number of axes of `a`.
862: 
863:     See Also
864:     --------
865:     numpy.fft : Overall view of discrete Fourier transforms, with definitions
866:          and conventions used.
867:     ifft2 : The inverse two-dimensional FFT.
868:     fft : The one-dimensional FFT.
869:     fftn : The *n*-dimensional FFT.
870:     fftshift : Shifts zero-frequency terms to the center of the array.
871:         For two-dimensional input, swaps first and third quadrants, and second
872:         and fourth quadrants.
873: 
874:     Notes
875:     -----
876:     `fft2` is just `fftn` with a different default for `axes`.
877: 
878:     The output, analogously to `fft`, contains the term for zero frequency in
879:     the low-order corner of the transformed axes, the positive frequency terms
880:     in the first half of these axes, the term for the Nyquist frequency in the
881:     middle of the axes and the negative frequency terms in the second half of
882:     the axes, in order of decreasingly negative frequency.
883: 
884:     See `fftn` for details and a plotting example, and `numpy.fft` for
885:     definitions and conventions used.
886: 
887: 
888:     Examples
889:     --------
890:     >>> a = np.mgrid[:5, :5][0]
891:     >>> np.fft.fft2(a)
892:     array([[ 50.0 +0.j        ,   0.0 +0.j        ,   0.0 +0.j        ,
893:               0.0 +0.j        ,   0.0 +0.j        ],
894:            [-12.5+17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
895:               0.0 +0.j        ,   0.0 +0.j        ],
896:            [-12.5 +4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
897:               0.0 +0.j        ,   0.0 +0.j        ],
898:            [-12.5 -4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,
899:                 0.0 +0.j        ,   0.0 +0.j        ],
900:            [-12.5-17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,
901:               0.0 +0.j        ,   0.0 +0.j        ]])
902: 
903:     '''
904: 
905:     return _raw_fftnd(a, s, axes, fft, norm)
906: 
907: 
908: def ifft2(a, s=None, axes=(-2, -1), norm=None):
909:     '''
910:     Compute the 2-dimensional inverse discrete Fourier Transform.
911: 
912:     This function computes the inverse of the 2-dimensional discrete Fourier
913:     Transform over any number of axes in an M-dimensional array by means of
914:     the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``
915:     to within numerical accuracy.  By default, the inverse transform is
916:     computed over the last two axes of the input array.
917: 
918:     The input, analogously to `ifft`, should be ordered in the same way as is
919:     returned by `fft2`, i.e. it should have the term for zero frequency
920:     in the low-order corner of the two axes, the positive frequency terms in
921:     the first half of these axes, the term for the Nyquist frequency in the
922:     middle of the axes and the negative frequency terms in the second half of
923:     both axes, in order of decreasingly negative frequency.
924: 
925:     Parameters
926:     ----------
927:     a : array_like
928:         Input array, can be complex.
929:     s : sequence of ints, optional
930:         Shape (length of each axis) of the output (``s[0]`` refers to axis 0,
931:         ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.
932:         Along each axis, if the given shape is smaller than that of the input,
933:         the input is cropped.  If it is larger, the input is padded with zeros.
934:         if `s` is not given, the shape of the input along the axes specified
935:         by `axes` is used.  See notes for issue on `ifft` zero padding.
936:     axes : sequence of ints, optional
937:         Axes over which to compute the FFT.  If not given, the last two
938:         axes are used.  A repeated index in `axes` means the transform over
939:         that axis is performed multiple times.  A one-element sequence means
940:         that a one-dimensional FFT is performed.
941:     norm : {None, "ortho"}, optional
942:         .. versionadded:: 1.10.0
943:         Normalization mode (see `numpy.fft`). Default is None.
944: 
945:     Returns
946:     -------
947:     out : complex ndarray
948:         The truncated or zero-padded input, transformed along the axes
949:         indicated by `axes`, or the last two axes if `axes` is not given.
950: 
951:     Raises
952:     ------
953:     ValueError
954:         If `s` and `axes` have different length, or `axes` not given and
955:         ``len(s) != 2``.
956:     IndexError
957:         If an element of `axes` is larger than than the number of axes of `a`.
958: 
959:     See Also
960:     --------
961:     numpy.fft : Overall view of discrete Fourier transforms, with definitions
962:          and conventions used.
963:     fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.
964:     ifftn : The inverse of the *n*-dimensional FFT.
965:     fft : The one-dimensional FFT.
966:     ifft : The one-dimensional inverse FFT.
967: 
968:     Notes
969:     -----
970:     `ifft2` is just `ifftn` with a different default for `axes`.
971: 
972:     See `ifftn` for details and a plotting example, and `numpy.fft` for
973:     definition and conventions used.
974: 
975:     Zero-padding, analogously with `ifft`, is performed by appending zeros to
976:     the input along the specified dimension.  Although this is the common
977:     approach, it might lead to surprising results.  If another form of zero
978:     padding is desired, it must be performed before `ifft2` is called.
979: 
980:     Examples
981:     --------
982:     >>> a = 4 * np.eye(4)
983:     >>> np.fft.ifft2(a)
984:     array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
985:            [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],
986:            [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],
987:            [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])
988: 
989:     '''
990: 
991:     return _raw_fftnd(a, s, axes, ifft, norm)
992: 
993: 
994: def rfftn(a, s=None, axes=None, norm=None):
995:     '''
996:     Compute the N-dimensional discrete Fourier Transform for real input.
997: 
998:     This function computes the N-dimensional discrete Fourier Transform over
999:     any number of axes in an M-dimensional real array by means of the Fast
1000:     Fourier Transform (FFT).  By default, all axes are transformed, with the
1001:     real transform performed over the last axis, while the remaining
1002:     transforms are complex.
1003: 
1004:     Parameters
1005:     ----------
1006:     a : array_like
1007:         Input array, taken to be real.
1008:     s : sequence of ints, optional
1009:         Shape (length along each transformed axis) to use from the input.
1010:         (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).
1011:         The final element of `s` corresponds to `n` for ``rfft(x, n)``, while
1012:         for the remaining axes, it corresponds to `n` for ``fft(x, n)``.
1013:         Along any axis, if the given shape is smaller than that of the input,
1014:         the input is cropped.  If it is larger, the input is padded with zeros.
1015:         if `s` is not given, the shape of the input along the axes specified
1016:         by `axes` is used.
1017:     axes : sequence of ints, optional
1018:         Axes over which to compute the FFT.  If not given, the last ``len(s)``
1019:         axes are used, or all axes if `s` is also not specified.
1020:     norm : {None, "ortho"}, optional
1021:         .. versionadded:: 1.10.0
1022:         Normalization mode (see `numpy.fft`). Default is None.
1023: 
1024:     Returns
1025:     -------
1026:     out : complex ndarray
1027:         The truncated or zero-padded input, transformed along the axes
1028:         indicated by `axes`, or by a combination of `s` and `a`,
1029:         as explained in the parameters section above.
1030:         The length of the last axis transformed will be ``s[-1]//2+1``,
1031:         while the remaining transformed axes will have lengths according to
1032:         `s`, or unchanged from the input.
1033: 
1034:     Raises
1035:     ------
1036:     ValueError
1037:         If `s` and `axes` have different length.
1038:     IndexError
1039:         If an element of `axes` is larger than than the number of axes of `a`.
1040: 
1041:     See Also
1042:     --------
1043:     irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT
1044:          of real input.
1045:     fft : The one-dimensional FFT, with definitions and conventions used.
1046:     rfft : The one-dimensional FFT of real input.
1047:     fftn : The n-dimensional FFT.
1048:     rfft2 : The two-dimensional FFT of real input.
1049: 
1050:     Notes
1051:     -----
1052:     The transform for real input is performed over the last transformation
1053:     axis, as by `rfft`, then the transform over the remaining axes is
1054:     performed as by `fftn`.  The order of the output is as for `rfft` for the
1055:     final transformation axis, and as for `fftn` for the remaining
1056:     transformation axes.
1057: 
1058:     See `fft` for details, definitions and conventions used.
1059: 
1060:     Examples
1061:     --------
1062:     >>> a = np.ones((2, 2, 2))
1063:     >>> np.fft.rfftn(a)
1064:     array([[[ 8.+0.j,  0.+0.j],
1065:             [ 0.+0.j,  0.+0.j]],
1066:            [[ 0.+0.j,  0.+0.j],
1067:             [ 0.+0.j,  0.+0.j]]])
1068: 
1069:     >>> np.fft.rfftn(a, axes=(2, 0))
1070:     array([[[ 4.+0.j,  0.+0.j],
1071:             [ 4.+0.j,  0.+0.j]],
1072:            [[ 0.+0.j,  0.+0.j],
1073:             [ 0.+0.j,  0.+0.j]]])
1074: 
1075:     '''
1076:     # The copy may be required for multithreading.
1077:     a = array(a, copy=True, dtype=float)
1078:     s, axes = _cook_nd_args(a, s, axes)
1079:     a = rfft(a, s[-1], axes[-1], norm)
1080:     for ii in range(len(axes)-1):
1081:         a = fft(a, s[ii], axes[ii], norm)
1082:     return a
1083: 
1084: 
1085: def rfft2(a, s=None, axes=(-2, -1), norm=None):
1086:     '''
1087:     Compute the 2-dimensional FFT of a real array.
1088: 
1089:     Parameters
1090:     ----------
1091:     a : array
1092:         Input array, taken to be real.
1093:     s : sequence of ints, optional
1094:         Shape of the FFT.
1095:     axes : sequence of ints, optional
1096:         Axes over which to compute the FFT.
1097:     norm : {None, "ortho"}, optional
1098:         .. versionadded:: 1.10.0
1099:         Normalization mode (see `numpy.fft`). Default is None.
1100: 
1101:     Returns
1102:     -------
1103:     out : ndarray
1104:         The result of the real 2-D FFT.
1105: 
1106:     See Also
1107:     --------
1108:     rfftn : Compute the N-dimensional discrete Fourier Transform for real
1109:             input.
1110: 
1111:     Notes
1112:     -----
1113:     This is really just `rfftn` with different default behavior.
1114:     For more details see `rfftn`.
1115: 
1116:     '''
1117: 
1118:     return rfftn(a, s, axes, norm)
1119: 
1120: 
1121: def irfftn(a, s=None, axes=None, norm=None):
1122:     '''
1123:     Compute the inverse of the N-dimensional FFT of real input.
1124: 
1125:     This function computes the inverse of the N-dimensional discrete
1126:     Fourier Transform for real input over any number of axes in an
1127:     M-dimensional array by means of the Fast Fourier Transform (FFT).  In
1128:     other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical
1129:     accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,
1130:     and for the same reason.)
1131: 
1132:     The input should be ordered in the same way as is returned by `rfftn`,
1133:     i.e. as for `irfft` for the final transformation axis, and as for `ifftn`
1134:     along all the other axes.
1135: 
1136:     Parameters
1137:     ----------
1138:     a : array_like
1139:         Input array.
1140:     s : sequence of ints, optional
1141:         Shape (length of each transformed axis) of the output
1142:         (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the
1143:         number of input points used along this axis, except for the last axis,
1144:         where ``s[-1]//2+1`` points of the input are used.
1145:         Along any axis, if the shape indicated by `s` is smaller than that of
1146:         the input, the input is cropped.  If it is larger, the input is padded
1147:         with zeros. If `s` is not given, the shape of the input along the
1148:         axes specified by `axes` is used.
1149:     axes : sequence of ints, optional
1150:         Axes over which to compute the inverse FFT. If not given, the last
1151:         `len(s)` axes are used, or all axes if `s` is also not specified.
1152:         Repeated indices in `axes` means that the inverse transform over that
1153:         axis is performed multiple times.
1154:     norm : {None, "ortho"}, optional
1155:         .. versionadded:: 1.10.0
1156:         Normalization mode (see `numpy.fft`). Default is None.
1157: 
1158:     Returns
1159:     -------
1160:     out : ndarray
1161:         The truncated or zero-padded input, transformed along the axes
1162:         indicated by `axes`, or by a combination of `s` or `a`,
1163:         as explained in the parameters section above.
1164:         The length of each transformed axis is as given by the corresponding
1165:         element of `s`, or the length of the input in every axis except for the
1166:         last one if `s` is not given.  In the final transformed axis the length
1167:         of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the
1168:         length of the final transformed axis of the input.  To get an odd
1169:         number of output points in the final axis, `s` must be specified.
1170: 
1171:     Raises
1172:     ------
1173:     ValueError
1174:         If `s` and `axes` have different length.
1175:     IndexError
1176:         If an element of `axes` is larger than than the number of axes of `a`.
1177: 
1178:     See Also
1179:     --------
1180:     rfftn : The forward n-dimensional FFT of real input,
1181:             of which `ifftn` is the inverse.
1182:     fft : The one-dimensional FFT, with definitions and conventions used.
1183:     irfft : The inverse of the one-dimensional FFT of real input.
1184:     irfft2 : The inverse of the two-dimensional FFT of real input.
1185: 
1186:     Notes
1187:     -----
1188:     See `fft` for definitions and conventions used.
1189: 
1190:     See `rfft` for definitions and conventions used for real input.
1191: 
1192:     Examples
1193:     --------
1194:     >>> a = np.zeros((3, 2, 2))
1195:     >>> a[0, 0, 0] = 3 * 2 * 2
1196:     >>> np.fft.irfftn(a)
1197:     array([[[ 1.,  1.],
1198:             [ 1.,  1.]],
1199:            [[ 1.,  1.],
1200:             [ 1.,  1.]],
1201:            [[ 1.,  1.],
1202:             [ 1.,  1.]]])
1203: 
1204:     '''
1205:     # The copy may be required for multithreading.
1206:     a = array(a, copy=True, dtype=complex)
1207:     s, axes = _cook_nd_args(a, s, axes, invreal=1)
1208:     for ii in range(len(axes)-1):
1209:         a = ifft(a, s[ii], axes[ii], norm)
1210:     a = irfft(a, s[-1], axes[-1], norm)
1211:     return a
1212: 
1213: 
1214: def irfft2(a, s=None, axes=(-2, -1), norm=None):
1215:     '''
1216:     Compute the 2-dimensional inverse FFT of a real array.
1217: 
1218:     Parameters
1219:     ----------
1220:     a : array_like
1221:         The input array
1222:     s : sequence of ints, optional
1223:         Shape of the inverse FFT.
1224:     axes : sequence of ints, optional
1225:         The axes over which to compute the inverse fft.
1226:         Default is the last two axes.
1227:     norm : {None, "ortho"}, optional
1228:         .. versionadded:: 1.10.0
1229:         Normalization mode (see `numpy.fft`). Default is None.
1230: 
1231:     Returns
1232:     -------
1233:     out : ndarray
1234:         The result of the inverse real 2-D FFT.
1235: 
1236:     See Also
1237:     --------
1238:     irfftn : Compute the inverse of the N-dimensional FFT of real input.
1239: 
1240:     Notes
1241:     -----
1242:     This is really `irfftn` with different defaults.
1243:     For more details see `irfftn`.
1244: 
1245:     '''
1246: 
1247:     return irfftn(a, s, axes, norm)
1248: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_100007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\nDiscrete Fourier Transforms\n\nRoutines in this module:\n\nfft(a, n=None, axis=-1)\nifft(a, n=None, axis=-1)\nrfft(a, n=None, axis=-1)\nirfft(a, n=None, axis=-1)\nhfft(a, n=None, axis=-1)\nihfft(a, n=None, axis=-1)\nfftn(a, s=None, axes=None)\nifftn(a, s=None, axes=None)\nrfftn(a, s=None, axes=None)\nirfftn(a, s=None, axes=None)\nfft2(a, s=None, axes=(-2,-1))\nifft2(a, s=None, axes=(-2, -1))\nrfft2(a, s=None, axes=(-2,-1))\nirfft2(a, s=None, axes=(-2, -1))\n\ni = inverse transform\nr = transform of purely real data\nh = Hermite transform\nn = n-dimensional transform\n2 = 2-dimensional transform\n(Note: 2D routines are just nD routines with different default\nbehavior.)\n\nThe underlying code for these functions is an f2c-translated and modified\nversion of the FFTPACK routines.\n\n')

# Assigning a List to a Name (line 35):

# Assigning a List to a Name (line 35):
__all__ = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn', 'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']
module_type_store.set_exportable_members(['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn', 'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn'])

# Obtaining an instance of the builtin type 'list' (line 35)
list_100008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 35)
# Adding element type (line 35)
str_100009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100009)
# Adding element type (line 35)
str_100010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'str', 'ifft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100010)
# Adding element type (line 35)
str_100011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'str', 'rfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100011)
# Adding element type (line 35)
str_100012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'str', 'irfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100012)
# Adding element type (line 35)
str_100013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 43), 'str', 'hfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100013)
# Adding element type (line 35)
str_100014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 51), 'str', 'ihfft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100014)
# Adding element type (line 35)
str_100015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 60), 'str', 'rfftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100015)
# Adding element type (line 35)
str_100016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'str', 'irfftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100016)
# Adding element type (line 35)
str_100017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'str', 'rfft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100017)
# Adding element type (line 35)
str_100018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'str', 'irfft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100018)
# Adding element type (line 35)
str_100019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 40), 'str', 'fft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100019)
# Adding element type (line 35)
str_100020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 48), 'str', 'ifft2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100020)
# Adding element type (line 35)
str_100021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 57), 'str', 'fftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100021)
# Adding element type (line 35)
str_100022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 65), 'str', 'ifftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 10), list_100008, str_100022)

# Assigning a type to the variable '__all__' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '__all__', list_100008)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from numpy.core import array, asarray, zeros, swapaxes, shape, conjugate, take, sqrt' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_100023 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core')

if (type(import_100023) is not StypyTypeError):

    if (import_100023 != 'pyd_module'):
        __import__(import_100023)
        sys_modules_100024 = sys.modules[import_100023]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', sys_modules_100024.module_type_store, module_type_store, ['array', 'asarray', 'zeros', 'swapaxes', 'shape', 'conjugate', 'take', 'sqrt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_100024, sys_modules_100024.module_type_store, module_type_store)
    else:
        from numpy.core import array, asarray, zeros, swapaxes, shape, conjugate, take, sqrt

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', None, module_type_store, ['array', 'asarray', 'zeros', 'swapaxes', 'shape', 'conjugate', 'take', 'sqrt'], [array, asarray, zeros, swapaxes, shape, conjugate, take, sqrt])

else:
    # Assigning a type to the variable 'numpy.core' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.core', import_100023)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from numpy.fft import fftpack' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
import_100025 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.fft')

if (type(import_100025) is not StypyTypeError):

    if (import_100025 != 'pyd_module'):
        __import__(import_100025)
        sys_modules_100026 = sys.modules[import_100025]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.fft', sys_modules_100026.module_type_store, module_type_store, ['fftpack_lite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_100026, sys_modules_100026.module_type_store, module_type_store)
    else:
        from numpy.fft import fftpack_lite as fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.fft', None, module_type_store, ['fftpack_lite'], [fftpack])

else:
    # Assigning a type to the variable 'numpy.fft' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.fft', import_100025)

# Adding an alias
module_type_store.add_alias('fftpack', 'fftpack_lite')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')


# Assigning a Dict to a Name (line 42):

# Assigning a Dict to a Name (line 42):

# Obtaining an instance of the builtin type 'dict' (line 42)
dict_100027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 42)

# Assigning a type to the variable '_fft_cache' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_fft_cache', dict_100027)

# Assigning a Dict to a Name (line 43):

# Assigning a Dict to a Name (line 43):

# Obtaining an instance of the builtin type 'dict' (line 43)
dict_100028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 43)

# Assigning a type to the variable '_real_fft_cache' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), '_real_fft_cache', dict_100028)

@norecursion
def _raw_fft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 46)
    None_100029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'None')
    int_100030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'int')
    # Getting the type of 'fftpack' (line 46)
    fftpack_100031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 47), 'fftpack')
    # Obtaining the member 'cffti' of a type (line 46)
    cffti_100032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 47), fftpack_100031, 'cffti')
    # Getting the type of 'fftpack' (line 47)
    fftpack_100033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'fftpack')
    # Obtaining the member 'cfftf' of a type (line 47)
    cfftf_100034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), fftpack_100033, 'cfftf')
    # Getting the type of '_fft_cache' (line 47)
    _fft_cache_100035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 52), '_fft_cache')
    defaults = [None_100029, int_100030, cffti_100032, cfftf_100034, _fft_cache_100035]
    # Create a new context for function '_raw_fft'
    module_type_store = module_type_store.open_function_context('_raw_fft', 46, 0, False)
    
    # Passed parameters checking function
    _raw_fft.stypy_localization = localization
    _raw_fft.stypy_type_of_self = None
    _raw_fft.stypy_type_store = module_type_store
    _raw_fft.stypy_function_name = '_raw_fft'
    _raw_fft.stypy_param_names_list = ['a', 'n', 'axis', 'init_function', 'work_function', 'fft_cache']
    _raw_fft.stypy_varargs_param_name = None
    _raw_fft.stypy_kwargs_param_name = None
    _raw_fft.stypy_call_defaults = defaults
    _raw_fft.stypy_call_varargs = varargs
    _raw_fft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_fft', ['a', 'n', 'axis', 'init_function', 'work_function', 'fft_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_fft', localization, ['a', 'n', 'axis', 'init_function', 'work_function', 'fft_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_fft(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to asarray(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'a' (line 48)
    a_100037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'a', False)
    # Processing the call keyword arguments (line 48)
    kwargs_100038 = {}
    # Getting the type of 'asarray' (line 48)
    asarray_100036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 48)
    asarray_call_result_100039 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), asarray_100036, *[a_100037], **kwargs_100038)
    
    # Assigning a type to the variable 'a' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'a', asarray_call_result_100039)
    
    # Type idiom detected: calculating its left and rigth part (line 50)
    # Getting the type of 'n' (line 50)
    n_100040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'n')
    # Getting the type of 'None' (line 50)
    None_100041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'None')
    
    (may_be_100042, more_types_in_union_100043) = may_be_none(n_100040, None_100041)

    if may_be_100042:

        if more_types_in_union_100043:
            # Runtime conditional SSA (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 51):
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 51)
        axis_100044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'axis')
        # Getting the type of 'a' (line 51)
        a_100045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'a')
        # Obtaining the member 'shape' of a type (line 51)
        shape_100046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), a_100045, 'shape')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___100047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), shape_100046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_100048 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), getitem___100047, axis_100044)
        
        # Assigning a type to the variable 'n' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'n', subscript_call_result_100048)

        if more_types_in_union_100043:
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'n' (line 53)
    n_100049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 7), 'n')
    int_100050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 11), 'int')
    # Applying the binary operator '<' (line 53)
    result_lt_100051 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 7), '<', n_100049, int_100050)
    
    # Testing the type of an if condition (line 53)
    if_condition_100052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 4), result_lt_100051)
    # Assigning a type to the variable 'if_condition_100052' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'if_condition_100052', if_condition_100052)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 54)
    # Processing the call arguments (line 54)
    str_100054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'str', 'Invalid number of FFT data points (%d) specified.')
    # Getting the type of 'n' (line 55)
    n_100055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'n', False)
    # Applying the binary operator '%' (line 54)
    result_mod_100056 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 25), '%', str_100054, n_100055)
    
    # Processing the call keyword arguments (line 54)
    kwargs_100057 = {}
    # Getting the type of 'ValueError' (line 54)
    ValueError_100053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 54)
    ValueError_call_result_100058 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), ValueError_100053, *[result_mod_100056], **kwargs_100057)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 54, 8), ValueError_call_result_100058, 'raise parameter', BaseException)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to pop(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_100066 = {}
    
    # Call to setdefault(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'n' (line 61)
    n_100061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 37), 'n', False)
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_100062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    
    # Processing the call keyword arguments (line 61)
    kwargs_100063 = {}
    # Getting the type of 'fft_cache' (line 61)
    fft_cache_100059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'fft_cache', False)
    # Obtaining the member 'setdefault' of a type (line 61)
    setdefault_100060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), fft_cache_100059, 'setdefault')
    # Calling setdefault(args, kwargs) (line 61)
    setdefault_call_result_100064 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), setdefault_100060, *[n_100061, list_100062], **kwargs_100063)
    
    # Obtaining the member 'pop' of a type (line 61)
    pop_100065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), setdefault_call_result_100064, 'pop')
    # Calling pop(args, kwargs) (line 61)
    pop_call_result_100067 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), pop_100065, *[], **kwargs_100066)
    
    # Assigning a type to the variable 'wsave' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'wsave', pop_call_result_100067)
    # SSA branch for the except part of a try statement (line 57)
    # SSA branch for the except 'IndexError' branch of a try statement (line 57)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to init_function(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'n' (line 63)
    n_100069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'n', False)
    # Processing the call keyword arguments (line 63)
    kwargs_100070 = {}
    # Getting the type of 'init_function' (line 63)
    init_function_100068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'init_function', False)
    # Calling init_function(args, kwargs) (line 63)
    init_function_call_result_100071 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), init_function_100068, *[n_100069], **kwargs_100070)
    
    # Assigning a type to the variable 'wsave' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'wsave', init_function_call_result_100071)
    # SSA join for try-except statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 65)
    axis_100072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'axis')
    # Getting the type of 'a' (line 65)
    a_100073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'a')
    # Obtaining the member 'shape' of a type (line 65)
    shape_100074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), a_100073, 'shape')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___100075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), shape_100074, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_100076 = invoke(stypy.reporting.localization.Localization(__file__, 65, 7), getitem___100075, axis_100072)
    
    # Getting the type of 'n' (line 65)
    n_100077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'n')
    # Applying the binary operator '!=' (line 65)
    result_ne_100078 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 7), '!=', subscript_call_result_100076, n_100077)
    
    # Testing the type of an if condition (line 65)
    if_condition_100079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 4), result_ne_100078)
    # Assigning a type to the variable 'if_condition_100079' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'if_condition_100079', if_condition_100079)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 66):
    
    # Assigning a Call to a Name (line 66):
    
    # Call to list(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'a' (line 66)
    a_100081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'a', False)
    # Obtaining the member 'shape' of a type (line 66)
    shape_100082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 17), a_100081, 'shape')
    # Processing the call keyword arguments (line 66)
    kwargs_100083 = {}
    # Getting the type of 'list' (line 66)
    list_100080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'list', False)
    # Calling list(args, kwargs) (line 66)
    list_call_result_100084 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), list_100080, *[shape_100082], **kwargs_100083)
    
    # Assigning a type to the variable 's' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 's', list_call_result_100084)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 67)
    axis_100085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'axis')
    # Getting the type of 's' (line 67)
    s_100086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 's')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___100087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), s_100086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_100088 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), getitem___100087, axis_100085)
    
    # Getting the type of 'n' (line 67)
    n_100089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'n')
    # Applying the binary operator '>' (line 67)
    result_gt_100090 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '>', subscript_call_result_100088, n_100089)
    
    # Testing the type of an if condition (line 67)
    if_condition_100091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_gt_100090)
    # Assigning a type to the variable 'if_condition_100091' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_100091', if_condition_100091)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_100092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    
    # Call to slice(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'None' (line 68)
    None_100094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'None', False)
    # Processing the call keyword arguments (line 68)
    kwargs_100095 = {}
    # Getting the type of 'slice' (line 68)
    slice_100093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'slice', False)
    # Calling slice(args, kwargs) (line 68)
    slice_call_result_100096 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), slice_100093, *[None_100094], **kwargs_100095)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 20), list_100092, slice_call_result_100096)
    
    
    # Call to len(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 's' (line 68)
    s_100098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 's', False)
    # Processing the call keyword arguments (line 68)
    kwargs_100099 = {}
    # Getting the type of 'len' (line 68)
    len_100097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'len', False)
    # Calling len(args, kwargs) (line 68)
    len_call_result_100100 = invoke(stypy.reporting.localization.Localization(__file__, 68, 34), len_100097, *[s_100098], **kwargs_100099)
    
    # Applying the binary operator '*' (line 68)
    result_mul_100101 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 20), '*', list_100092, len_call_result_100100)
    
    # Assigning a type to the variable 'index' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'index', result_mul_100101)
    
    # Assigning a Call to a Subscript (line 69):
    
    # Assigning a Call to a Subscript (line 69):
    
    # Call to slice(...): (line 69)
    # Processing the call arguments (line 69)
    int_100103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'int')
    # Getting the type of 'n' (line 69)
    n_100104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'n', False)
    # Processing the call keyword arguments (line 69)
    kwargs_100105 = {}
    # Getting the type of 'slice' (line 69)
    slice_100102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'slice', False)
    # Calling slice(args, kwargs) (line 69)
    slice_call_result_100106 = invoke(stypy.reporting.localization.Localization(__file__, 69, 26), slice_100102, *[int_100103, n_100104], **kwargs_100105)
    
    # Getting the type of 'index' (line 69)
    index_100107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'index')
    # Getting the type of 'axis' (line 69)
    axis_100108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'axis')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 12), index_100107, (axis_100108, slice_call_result_100106))
    
    # Assigning a Subscript to a Name (line 70):
    
    # Assigning a Subscript to a Name (line 70):
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 70)
    index_100109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'index')
    # Getting the type of 'a' (line 70)
    a_100110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'a')
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___100111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), a_100110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_100112 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), getitem___100111, index_100109)
    
    # Assigning a type to the variable 'a' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'a', subscript_call_result_100112)
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 72):
    
    # Assigning a BinOp to a Name (line 72):
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_100113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    # Adding element type (line 72)
    
    # Call to slice(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'None' (line 72)
    None_100115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'None', False)
    # Processing the call keyword arguments (line 72)
    kwargs_100116 = {}
    # Getting the type of 'slice' (line 72)
    slice_100114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'slice', False)
    # Calling slice(args, kwargs) (line 72)
    slice_call_result_100117 = invoke(stypy.reporting.localization.Localization(__file__, 72, 21), slice_100114, *[None_100115], **kwargs_100116)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), list_100113, slice_call_result_100117)
    
    
    # Call to len(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 's' (line 72)
    s_100119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 's', False)
    # Processing the call keyword arguments (line 72)
    kwargs_100120 = {}
    # Getting the type of 'len' (line 72)
    len_100118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'len', False)
    # Calling len(args, kwargs) (line 72)
    len_call_result_100121 = invoke(stypy.reporting.localization.Localization(__file__, 72, 34), len_100118, *[s_100119], **kwargs_100120)
    
    # Applying the binary operator '*' (line 72)
    result_mul_100122 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 20), '*', list_100113, len_call_result_100121)
    
    # Assigning a type to the variable 'index' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'index', result_mul_100122)
    
    # Assigning a Call to a Subscript (line 73):
    
    # Assigning a Call to a Subscript (line 73):
    
    # Call to slice(...): (line 73)
    # Processing the call arguments (line 73)
    int_100124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 73)
    axis_100125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'axis', False)
    # Getting the type of 's' (line 73)
    s_100126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 's', False)
    # Obtaining the member '__getitem__' of a type (line 73)
    getitem___100127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 35), s_100126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
    subscript_call_result_100128 = invoke(stypy.reporting.localization.Localization(__file__, 73, 35), getitem___100127, axis_100125)
    
    # Processing the call keyword arguments (line 73)
    kwargs_100129 = {}
    # Getting the type of 'slice' (line 73)
    slice_100123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'slice', False)
    # Calling slice(args, kwargs) (line 73)
    slice_call_result_100130 = invoke(stypy.reporting.localization.Localization(__file__, 73, 26), slice_100123, *[int_100124, subscript_call_result_100128], **kwargs_100129)
    
    # Getting the type of 'index' (line 73)
    index_100131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'index')
    # Getting the type of 'axis' (line 73)
    axis_100132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'axis')
    # Storing an element on a container (line 73)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), index_100131, (axis_100132, slice_call_result_100130))
    
    # Assigning a Name to a Subscript (line 74):
    
    # Assigning a Name to a Subscript (line 74):
    # Getting the type of 'n' (line 74)
    n_100133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'n')
    # Getting the type of 's' (line 74)
    s_100134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 's')
    # Getting the type of 'axis' (line 74)
    axis_100135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'axis')
    # Storing an element on a container (line 74)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 12), s_100134, (axis_100135, n_100133))
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to zeros(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 's' (line 75)
    s_100137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 's', False)
    # Getting the type of 'a' (line 75)
    a_100138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'a', False)
    # Obtaining the member 'dtype' of a type (line 75)
    dtype_100139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), a_100138, 'dtype')
    # Obtaining the member 'char' of a type (line 75)
    char_100140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), dtype_100139, 'char')
    # Processing the call keyword arguments (line 75)
    kwargs_100141 = {}
    # Getting the type of 'zeros' (line 75)
    zeros_100136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'zeros', False)
    # Calling zeros(args, kwargs) (line 75)
    zeros_call_result_100142 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), zeros_100136, *[s_100137, char_100140], **kwargs_100141)
    
    # Assigning a type to the variable 'z' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'z', zeros_call_result_100142)
    
    # Assigning a Name to a Subscript (line 76):
    
    # Assigning a Name to a Subscript (line 76):
    # Getting the type of 'a' (line 76)
    a_100143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'a')
    # Getting the type of 'z' (line 76)
    z_100144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'z')
    # Getting the type of 'index' (line 76)
    index_100145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'index')
    # Storing an element on a container (line 76)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 12), z_100144, (index_100145, a_100143))
    
    # Assigning a Name to a Name (line 77):
    
    # Assigning a Name to a Name (line 77):
    # Getting the type of 'z' (line 77)
    z_100146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'z')
    # Assigning a type to the variable 'a' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'a', z_100146)
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'axis' (line 79)
    axis_100147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'axis')
    int_100148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
    # Applying the binary operator '!=' (line 79)
    result_ne_100149 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), '!=', axis_100147, int_100148)
    
    # Testing the type of an if condition (line 79)
    if_condition_100150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_ne_100149)
    # Assigning a type to the variable 'if_condition_100150' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_100150', if_condition_100150)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to swapaxes(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'a' (line 80)
    a_100152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'a', False)
    # Getting the type of 'axis' (line 80)
    axis_100153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'axis', False)
    int_100154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_100155 = {}
    # Getting the type of 'swapaxes' (line 80)
    swapaxes_100151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 80)
    swapaxes_call_result_100156 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), swapaxes_100151, *[a_100152, axis_100153, int_100154], **kwargs_100155)
    
    # Assigning a type to the variable 'a' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'a', swapaxes_call_result_100156)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to work_function(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'a' (line 81)
    a_100158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'a', False)
    # Getting the type of 'wsave' (line 81)
    wsave_100159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'wsave', False)
    # Processing the call keyword arguments (line 81)
    kwargs_100160 = {}
    # Getting the type of 'work_function' (line 81)
    work_function_100157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'work_function', False)
    # Calling work_function(args, kwargs) (line 81)
    work_function_call_result_100161 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), work_function_100157, *[a_100158, wsave_100159], **kwargs_100160)
    
    # Assigning a type to the variable 'r' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'r', work_function_call_result_100161)
    
    
    # Getting the type of 'axis' (line 82)
    axis_100162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'axis')
    int_100163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'int')
    # Applying the binary operator '!=' (line 82)
    result_ne_100164 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 7), '!=', axis_100162, int_100163)
    
    # Testing the type of an if condition (line 82)
    if_condition_100165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_ne_100164)
    # Assigning a type to the variable 'if_condition_100165' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_100165', if_condition_100165)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to swapaxes(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'r' (line 83)
    r_100167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 21), 'r', False)
    # Getting the type of 'axis' (line 83)
    axis_100168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'axis', False)
    int_100169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
    # Processing the call keyword arguments (line 83)
    kwargs_100170 = {}
    # Getting the type of 'swapaxes' (line 83)
    swapaxes_100166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'swapaxes', False)
    # Calling swapaxes(args, kwargs) (line 83)
    swapaxes_call_result_100171 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), swapaxes_100166, *[r_100167, axis_100168, int_100169], **kwargs_100170)
    
    # Assigning a type to the variable 'r' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'r', swapaxes_call_result_100171)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'wsave' (line 88)
    wsave_100177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'wsave', False)
    # Processing the call keyword arguments (line 88)
    kwargs_100178 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 88)
    n_100172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'n', False)
    # Getting the type of 'fft_cache' (line 88)
    fft_cache_100173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'fft_cache', False)
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___100174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), fft_cache_100173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_100175 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), getitem___100174, n_100172)
    
    # Obtaining the member 'append' of a type (line 88)
    append_100176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), subscript_call_result_100175, 'append')
    # Calling append(args, kwargs) (line 88)
    append_call_result_100179 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), append_100176, *[wsave_100177], **kwargs_100178)
    
    # Getting the type of 'r' (line 90)
    r_100180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', r_100180)
    
    # ################# End of '_raw_fft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_fft' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_100181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100181)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_fft'
    return stypy_return_type_100181

# Assigning a type to the variable '_raw_fft' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_raw_fft', _raw_fft)

@norecursion
def _unitary(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unitary'
    module_type_store = module_type_store.open_function_context('_unitary', 93, 0, False)
    
    # Passed parameters checking function
    _unitary.stypy_localization = localization
    _unitary.stypy_type_of_self = None
    _unitary.stypy_type_store = module_type_store
    _unitary.stypy_function_name = '_unitary'
    _unitary.stypy_param_names_list = ['norm']
    _unitary.stypy_varargs_param_name = None
    _unitary.stypy_kwargs_param_name = None
    _unitary.stypy_call_defaults = defaults
    _unitary.stypy_call_varargs = varargs
    _unitary.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unitary', ['norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unitary', localization, ['norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unitary(...)' code ##################

    
    
    # Getting the type of 'norm' (line 94)
    norm_100182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 7), 'norm')
    
    # Obtaining an instance of the builtin type 'tuple' (line 94)
    tuple_100183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 94)
    # Adding element type (line 94)
    # Getting the type of 'None' (line 94)
    None_100184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 20), tuple_100183, None_100184)
    # Adding element type (line 94)
    str_100185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'str', 'ortho')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 20), tuple_100183, str_100185)
    
    # Applying the binary operator 'notin' (line 94)
    result_contains_100186 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), 'notin', norm_100182, tuple_100183)
    
    # Testing the type of an if condition (line 94)
    if_condition_100187 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_contains_100186)
    # Assigning a type to the variable 'if_condition_100187' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_100187', if_condition_100187)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 95)
    # Processing the call arguments (line 95)
    str_100189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'str', 'Invalid norm value %s, should be None or "ortho".')
    # Getting the type of 'norm' (line 96)
    norm_100190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'norm', False)
    # Applying the binary operator '%' (line 95)
    result_mod_100191 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 25), '%', str_100189, norm_100190)
    
    # Processing the call keyword arguments (line 95)
    kwargs_100192 = {}
    # Getting the type of 'ValueError' (line 95)
    ValueError_100188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 95)
    ValueError_call_result_100193 = invoke(stypy.reporting.localization.Localization(__file__, 95, 14), ValueError_100188, *[result_mod_100191], **kwargs_100192)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 95, 8), ValueError_call_result_100193, 'raise parameter', BaseException)
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'norm' (line 97)
    norm_100194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'norm')
    # Getting the type of 'None' (line 97)
    None_100195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'None')
    # Applying the binary operator 'isnot' (line 97)
    result_is_not_100196 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'isnot', norm_100194, None_100195)
    
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', result_is_not_100196)
    
    # ################# End of '_unitary(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unitary' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_100197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unitary'
    return stypy_return_type_100197

# Assigning a type to the variable '_unitary' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), '_unitary', _unitary)

@norecursion
def fft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 100)
    None_100198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'None')
    int_100199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'int')
    # Getting the type of 'None' (line 100)
    None_100200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'None')
    defaults = [None_100198, int_100199, None_100200]
    # Create a new context for function 'fft'
    module_type_store = module_type_store.open_function_context('fft', 100, 0, False)
    
    # Passed parameters checking function
    fft.stypy_localization = localization
    fft.stypy_type_of_self = None
    fft.stypy_type_store = module_type_store
    fft.stypy_function_name = 'fft'
    fft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    fft.stypy_varargs_param_name = None
    fft.stypy_kwargs_param_name = None
    fft.stypy_call_defaults = defaults
    fft.stypy_call_varargs = varargs
    fft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fft(...)' code ##################

    str_100201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', '\n    Compute the one-dimensional discrete Fourier Transform.\n\n    This function computes the one-dimensional *n*-point discrete Fourier\n    Transform (DFT) with the efficient Fast Fourier Transform (FFT)\n    algorithm [CT].\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    n : int, optional\n        Length of the transformed axis of the output.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros.  If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the FFT.  If not given, the last axis is\n        used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n\n    Raises\n    ------\n    IndexError\n        if `axes` is larger than the last axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : for definition of the DFT and conventions used.\n    ifft : The inverse of `fft`.\n    fft2 : The two-dimensional FFT.\n    fftn : The *n*-dimensional FFT.\n    rfftn : The *n*-dimensional FFT of real input.\n    fftfreq : Frequency bins for given FFT parameters.\n\n    Notes\n    -----\n    FFT (Fast Fourier Transform) refers to a way the discrete Fourier\n    Transform (DFT) can be calculated efficiently, by using symmetries in the\n    calculated terms.  The symmetry is highest when `n` is a power of 2, and\n    the transform is therefore most efficient for these sizes.\n\n    The DFT is defined, with the conventions used in this implementation, in\n    the documentation for the `numpy.fft` module.\n\n    References\n    ----------\n    .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the\n            machine calculation of complex Fourier series," *Math. Comput.*\n            19: 297-301.\n\n    Examples\n    --------\n    >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))\n    array([ -3.44505240e-16 +1.14383329e-17j,\n             8.00000000e+00 -5.71092652e-15j,\n             2.33482938e-16 +1.22460635e-16j,\n             1.64863782e-15 +1.77635684e-15j,\n             9.95839695e-17 +2.33482938e-16j,\n             0.00000000e+00 +1.66837030e-15j,\n             1.14383329e-17 +1.22460635e-16j,\n             -1.64863782e-15 +1.77635684e-15j])\n\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.arange(256)\n    >>> sp = np.fft.fft(np.sin(t))\n    >>> freq = np.fft.fftfreq(t.shape[-1])\n    >>> plt.plot(freq, sp.real, freq, sp.imag)\n    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]\n    >>> plt.show()\n\n    In this example, real input has an FFT which is Hermitian, i.e., symmetric\n    in the real part and anti-symmetric in the imaginary part, as described in\n    the `numpy.fft` documentation.\n\n    ')
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to astype(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'complex' (line 186)
    complex_100207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 26), 'complex', False)
    # Processing the call keyword arguments (line 186)
    # Getting the type of 'False' (line 186)
    False_100208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'False', False)
    keyword_100209 = False_100208
    kwargs_100210 = {'copy': keyword_100209}
    
    # Call to asarray(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'a' (line 186)
    a_100203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'a', False)
    # Processing the call keyword arguments (line 186)
    kwargs_100204 = {}
    # Getting the type of 'asarray' (line 186)
    asarray_100202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 186)
    asarray_call_result_100205 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), asarray_100202, *[a_100203], **kwargs_100204)
    
    # Obtaining the member 'astype' of a type (line 186)
    astype_100206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), asarray_call_result_100205, 'astype')
    # Calling astype(args, kwargs) (line 186)
    astype_call_result_100211 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), astype_100206, *[complex_100207], **kwargs_100210)
    
    # Assigning a type to the variable 'a' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'a', astype_call_result_100211)
    
    # Type idiom detected: calculating its left and rigth part (line 187)
    # Getting the type of 'n' (line 187)
    n_100212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 7), 'n')
    # Getting the type of 'None' (line 187)
    None_100213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'None')
    
    (may_be_100214, more_types_in_union_100215) = may_be_none(n_100212, None_100213)

    if may_be_100214:

        if more_types_in_union_100215:
            # Runtime conditional SSA (line 187)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 188):
        
        # Assigning a Subscript to a Name (line 188):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 188)
        axis_100216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'axis')
        # Getting the type of 'a' (line 188)
        a_100217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'a')
        # Obtaining the member 'shape' of a type (line 188)
        shape_100218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), a_100217, 'shape')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___100219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), shape_100218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_100220 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), getitem___100219, axis_100216)
        
        # Assigning a type to the variable 'n' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'n', subscript_call_result_100220)

        if more_types_in_union_100215:
            # SSA join for if statement (line 187)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 189):
    
    # Assigning a Call to a Name (line 189):
    
    # Call to _raw_fft(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'a' (line 189)
    a_100222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'a', False)
    # Getting the type of 'n' (line 189)
    n_100223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'n', False)
    # Getting the type of 'axis' (line 189)
    axis_100224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 28), 'axis', False)
    # Getting the type of 'fftpack' (line 189)
    fftpack_100225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 34), 'fftpack', False)
    # Obtaining the member 'cffti' of a type (line 189)
    cffti_100226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 34), fftpack_100225, 'cffti')
    # Getting the type of 'fftpack' (line 189)
    fftpack_100227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 49), 'fftpack', False)
    # Obtaining the member 'cfftf' of a type (line 189)
    cfftf_100228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 49), fftpack_100227, 'cfftf')
    # Getting the type of '_fft_cache' (line 189)
    _fft_cache_100229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 64), '_fft_cache', False)
    # Processing the call keyword arguments (line 189)
    kwargs_100230 = {}
    # Getting the type of '_raw_fft' (line 189)
    _raw_fft_100221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 189)
    _raw_fft_call_result_100231 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), _raw_fft_100221, *[a_100222, n_100223, axis_100224, cffti_100226, cfftf_100228, _fft_cache_100229], **kwargs_100230)
    
    # Assigning a type to the variable 'output' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'output', _raw_fft_call_result_100231)
    
    
    # Call to _unitary(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'norm' (line 190)
    norm_100233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'norm', False)
    # Processing the call keyword arguments (line 190)
    kwargs_100234 = {}
    # Getting the type of '_unitary' (line 190)
    _unitary_100232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 190)
    _unitary_call_result_100235 = invoke(stypy.reporting.localization.Localization(__file__, 190, 7), _unitary_100232, *[norm_100233], **kwargs_100234)
    
    # Testing the type of an if condition (line 190)
    if_condition_100236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), _unitary_call_result_100235)
    # Assigning a type to the variable 'if_condition_100236' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_100236', if_condition_100236)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output' (line 191)
    output_100237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'output')
    int_100238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 18), 'int')
    
    # Call to sqrt(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'n' (line 191)
    n_100240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'n', False)
    # Processing the call keyword arguments (line 191)
    kwargs_100241 = {}
    # Getting the type of 'sqrt' (line 191)
    sqrt_100239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 191)
    sqrt_call_result_100242 = invoke(stypy.reporting.localization.Localization(__file__, 191, 22), sqrt_100239, *[n_100240], **kwargs_100241)
    
    # Applying the binary operator 'div' (line 191)
    result_div_100243 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 18), 'div', int_100238, sqrt_call_result_100242)
    
    # Applying the binary operator '*=' (line 191)
    result_imul_100244 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 8), '*=', output_100237, result_div_100243)
    # Assigning a type to the variable 'output' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'output', result_imul_100244)
    
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 192)
    output_100245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type', output_100245)
    
    # ################# End of 'fft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fft' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_100246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100246)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fft'
    return stypy_return_type_100246

# Assigning a type to the variable 'fft' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'fft', fft)

@norecursion
def ifft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 195)
    None_100247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'None')
    int_100248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 25), 'int')
    # Getting the type of 'None' (line 195)
    None_100249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 34), 'None')
    defaults = [None_100247, int_100248, None_100249]
    # Create a new context for function 'ifft'
    module_type_store = module_type_store.open_function_context('ifft', 195, 0, False)
    
    # Passed parameters checking function
    ifft.stypy_localization = localization
    ifft.stypy_type_of_self = None
    ifft.stypy_type_store = module_type_store
    ifft.stypy_function_name = 'ifft'
    ifft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    ifft.stypy_varargs_param_name = None
    ifft.stypy_kwargs_param_name = None
    ifft.stypy_call_defaults = defaults
    ifft.stypy_call_varargs = varargs
    ifft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifft(...)' code ##################

    str_100250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', '\n    Compute the one-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the one-dimensional *n*-point\n    discrete Fourier transform computed by `fft`.  In other words,\n    ``ifft(fft(a)) == a`` to within numerical accuracy.\n    For a general description of the algorithm and definitions,\n    see `numpy.fft`.\n\n    The input should be ordered in the same way as is returned by `fft`,\n    i.e.,\n\n    * ``a[0]`` should contain the zero frequency term,\n    * ``a[1:n//2]`` should contain the positive-frequency terms,\n    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in\n      increasing order starting from the most negative frequency.\n\n    For an even number of input points, ``A[n//2]`` represents the sum of\n    the values at the positive and negative Nyquist frequencies, as the two\n    are aliased together. See `numpy.fft` for details.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    n : int, optional\n        Length of the transformed axis of the output.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros.  If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n        See notes about padding issues.\n    axis : int, optional\n        Axis over which to compute the inverse DFT.  If not given, the last\n        axis is used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n\n    Raises\n    ------\n    IndexError\n        If `axes` is larger than the last axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : An introduction, with definitions and general explanations.\n    fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse\n    ifft2 : The two-dimensional inverse FFT.\n    ifftn : The n-dimensional inverse FFT.\n\n    Notes\n    -----\n    If the input parameter `n` is larger than the size of the input, the input\n    is padded by appending zeros at the end.  Even though this is the common\n    approach, it might lead to surprising results.  If a different padding is\n    desired, it must be performed before calling `ifft`.\n\n    Examples\n    --------\n    >>> np.fft.ifft([0, 4, 0, 0])\n    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j])\n\n    Create and plot a band-limited signal with random phases:\n\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.arange(400)\n    >>> n = np.zeros((400,), dtype=complex)\n    >>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))\n    >>> s = np.fft.ifft(n)\n    >>> plt.plot(t, s.real, \'b-\', t, s.imag, \'r--\')\n    ...\n    >>> plt.legend((\'real\', \'imaginary\'))\n    ...\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 279):
    
    # Assigning a Call to a Name (line 279):
    
    # Call to array(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'a' (line 279)
    a_100252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 14), 'a', False)
    # Processing the call keyword arguments (line 279)
    # Getting the type of 'True' (line 279)
    True_100253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'True', False)
    keyword_100254 = True_100253
    # Getting the type of 'complex' (line 279)
    complex_100255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 34), 'complex', False)
    keyword_100256 = complex_100255
    kwargs_100257 = {'dtype': keyword_100256, 'copy': keyword_100254}
    # Getting the type of 'array' (line 279)
    array_100251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'array', False)
    # Calling array(args, kwargs) (line 279)
    array_call_result_100258 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), array_100251, *[a_100252], **kwargs_100257)
    
    # Assigning a type to the variable 'a' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'a', array_call_result_100258)
    
    # Type idiom detected: calculating its left and rigth part (line 280)
    # Getting the type of 'n' (line 280)
    n_100259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 7), 'n')
    # Getting the type of 'None' (line 280)
    None_100260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'None')
    
    (may_be_100261, more_types_in_union_100262) = may_be_none(n_100259, None_100260)

    if may_be_100261:

        if more_types_in_union_100262:
            # Runtime conditional SSA (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 281)
        axis_100263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 20), 'axis')
        # Getting the type of 'a' (line 281)
        a_100264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'a')
        # Obtaining the member 'shape' of a type (line 281)
        shape_100265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), a_100264, 'shape')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___100266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), shape_100265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_100267 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), getitem___100266, axis_100263)
        
        # Assigning a type to the variable 'n' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'n', subscript_call_result_100267)

        if more_types_in_union_100262:
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 282):
    
    # Assigning a Call to a Name (line 282):
    
    # Call to _unitary(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'norm' (line 282)
    norm_100269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 23), 'norm', False)
    # Processing the call keyword arguments (line 282)
    kwargs_100270 = {}
    # Getting the type of '_unitary' (line 282)
    _unitary_100268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 282)
    _unitary_call_result_100271 = invoke(stypy.reporting.localization.Localization(__file__, 282, 14), _unitary_100268, *[norm_100269], **kwargs_100270)
    
    # Assigning a type to the variable 'unitary' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'unitary', _unitary_call_result_100271)
    
    # Assigning a Call to a Name (line 283):
    
    # Assigning a Call to a Name (line 283):
    
    # Call to _raw_fft(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'a' (line 283)
    a_100273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 22), 'a', False)
    # Getting the type of 'n' (line 283)
    n_100274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'n', False)
    # Getting the type of 'axis' (line 283)
    axis_100275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'axis', False)
    # Getting the type of 'fftpack' (line 283)
    fftpack_100276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 34), 'fftpack', False)
    # Obtaining the member 'cffti' of a type (line 283)
    cffti_100277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 34), fftpack_100276, 'cffti')
    # Getting the type of 'fftpack' (line 283)
    fftpack_100278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 49), 'fftpack', False)
    # Obtaining the member 'cfftb' of a type (line 283)
    cfftb_100279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 49), fftpack_100278, 'cfftb')
    # Getting the type of '_fft_cache' (line 283)
    _fft_cache_100280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 64), '_fft_cache', False)
    # Processing the call keyword arguments (line 283)
    kwargs_100281 = {}
    # Getting the type of '_raw_fft' (line 283)
    _raw_fft_100272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 13), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 283)
    _raw_fft_call_result_100282 = invoke(stypy.reporting.localization.Localization(__file__, 283, 13), _raw_fft_100272, *[a_100273, n_100274, axis_100275, cffti_100277, cfftb_100279, _fft_cache_100280], **kwargs_100281)
    
    # Assigning a type to the variable 'output' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'output', _raw_fft_call_result_100282)
    # Getting the type of 'output' (line 284)
    output_100283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 11), 'output')
    int_100284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 21), 'int')
    
    # Getting the type of 'unitary' (line 284)
    unitary_100285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 37), 'unitary')
    # Testing the type of an if expression (line 284)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 284, 26), unitary_100285)
    # SSA begins for if expression (line 284)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to sqrt(...): (line 284)
    # Processing the call arguments (line 284)
    # Getting the type of 'n' (line 284)
    n_100287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'n', False)
    # Processing the call keyword arguments (line 284)
    kwargs_100288 = {}
    # Getting the type of 'sqrt' (line 284)
    sqrt_100286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 284)
    sqrt_call_result_100289 = invoke(stypy.reporting.localization.Localization(__file__, 284, 26), sqrt_100286, *[n_100287], **kwargs_100288)
    
    # SSA branch for the else part of an if expression (line 284)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 284)
    n_100290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 50), 'n')
    # SSA join for if expression (line 284)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_100291 = union_type.UnionType.add(sqrt_call_result_100289, n_100290)
    
    # Applying the binary operator 'div' (line 284)
    result_div_100292 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 21), 'div', int_100284, if_exp_100291)
    
    # Applying the binary operator '*' (line 284)
    result_mul_100293 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 11), '*', output_100283, result_div_100292)
    
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', result_mul_100293)
    
    # ################# End of 'ifft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifft' in the type store
    # Getting the type of 'stypy_return_type' (line 195)
    stypy_return_type_100294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifft'
    return stypy_return_type_100294

# Assigning a type to the variable 'ifft' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'ifft', ifft)

@norecursion
def rfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 287)
    None_100295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 14), 'None')
    int_100296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'int')
    # Getting the type of 'None' (line 287)
    None_100297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'None')
    defaults = [None_100295, int_100296, None_100297]
    # Create a new context for function 'rfft'
    module_type_store = module_type_store.open_function_context('rfft', 287, 0, False)
    
    # Passed parameters checking function
    rfft.stypy_localization = localization
    rfft.stypy_type_of_self = None
    rfft.stypy_type_store = module_type_store
    rfft.stypy_function_name = 'rfft'
    rfft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    rfft.stypy_varargs_param_name = None
    rfft.stypy_kwargs_param_name = None
    rfft.stypy_call_defaults = defaults
    rfft.stypy_call_varargs = varargs
    rfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfft(...)' code ##################

    str_100298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, (-1)), 'str', '\n    Compute the one-dimensional discrete Fourier Transform for real input.\n\n    This function computes the one-dimensional *n*-point discrete Fourier\n    Transform (DFT) of a real-valued array by means of an efficient algorithm\n    called the Fast Fourier Transform (FFT).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array\n    n : int, optional\n        Number of points along transformation axis in the input to use.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros. If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the FFT. If not given, the last axis is\n        used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        If `n` is even, the length of the transformed axis is ``(n/2)+1``.\n        If `n` is odd, the length is ``(n+1)/2``.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is larger than the last axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : For definition of the DFT and conventions used.\n    irfft : The inverse of `rfft`.\n    fft : The one-dimensional FFT of general (complex) input.\n    fftn : The *n*-dimensional FFT.\n    rfftn : The *n*-dimensional FFT of real input.\n\n    Notes\n    -----\n    When the DFT is computed for purely real input, the output is\n    Hermitian-symmetric, i.e. the negative frequency terms are just the complex\n    conjugates of the corresponding positive-frequency terms, and the\n    negative-frequency terms are therefore redundant.  This function does not\n    compute the negative frequency terms, and the length of the transformed\n    axis of the output is therefore ``n//2 + 1``.\n\n    When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains\n    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.\n\n    If `n` is even, ``A[-1]`` contains the term representing both positive\n    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely\n    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains\n    the largest positive frequency (fs/2*(n-1)/n), and is complex in the\n    general case.\n\n    If the input `a` contains an imaginary part, it is silently discarded.\n\n    Examples\n    --------\n    >>> np.fft.fft([0, 1, 0, 0])\n    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j])\n    >>> np.fft.rfft([0, 1, 0, 0])\n    array([ 1.+0.j,  0.-1.j, -1.+0.j])\n\n    Notice how the final element of the `fft` output is the complex conjugate\n    of the second element, for real input. For `rfft`, this symmetry is\n    exploited to compute only the non-negative frequency terms.\n\n    ')
    
    # Assigning a Call to a Name (line 365):
    
    # Assigning a Call to a Name (line 365):
    
    # Call to array(...): (line 365)
    # Processing the call arguments (line 365)
    # Getting the type of 'a' (line 365)
    a_100300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 14), 'a', False)
    # Processing the call keyword arguments (line 365)
    # Getting the type of 'True' (line 365)
    True_100301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'True', False)
    keyword_100302 = True_100301
    # Getting the type of 'float' (line 365)
    float_100303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'float', False)
    keyword_100304 = float_100303
    kwargs_100305 = {'dtype': keyword_100304, 'copy': keyword_100302}
    # Getting the type of 'array' (line 365)
    array_100299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'array', False)
    # Calling array(args, kwargs) (line 365)
    array_call_result_100306 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), array_100299, *[a_100300], **kwargs_100305)
    
    # Assigning a type to the variable 'a' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'a', array_call_result_100306)
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to _raw_fft(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'a' (line 366)
    a_100308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 22), 'a', False)
    # Getting the type of 'n' (line 366)
    n_100309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 25), 'n', False)
    # Getting the type of 'axis' (line 366)
    axis_100310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 28), 'axis', False)
    # Getting the type of 'fftpack' (line 366)
    fftpack_100311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 34), 'fftpack', False)
    # Obtaining the member 'rffti' of a type (line 366)
    rffti_100312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 34), fftpack_100311, 'rffti')
    # Getting the type of 'fftpack' (line 366)
    fftpack_100313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 49), 'fftpack', False)
    # Obtaining the member 'rfftf' of a type (line 366)
    rfftf_100314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 49), fftpack_100313, 'rfftf')
    # Getting the type of '_real_fft_cache' (line 367)
    _real_fft_cache_100315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), '_real_fft_cache', False)
    # Processing the call keyword arguments (line 366)
    kwargs_100316 = {}
    # Getting the type of '_raw_fft' (line 366)
    _raw_fft_100307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 366)
    _raw_fft_call_result_100317 = invoke(stypy.reporting.localization.Localization(__file__, 366, 13), _raw_fft_100307, *[a_100308, n_100309, axis_100310, rffti_100312, rfftf_100314, _real_fft_cache_100315], **kwargs_100316)
    
    # Assigning a type to the variable 'output' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'output', _raw_fft_call_result_100317)
    
    
    # Call to _unitary(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'norm' (line 368)
    norm_100319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'norm', False)
    # Processing the call keyword arguments (line 368)
    kwargs_100320 = {}
    # Getting the type of '_unitary' (line 368)
    _unitary_100318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 7), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 368)
    _unitary_call_result_100321 = invoke(stypy.reporting.localization.Localization(__file__, 368, 7), _unitary_100318, *[norm_100319], **kwargs_100320)
    
    # Testing the type of an if condition (line 368)
    if_condition_100322 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 4), _unitary_call_result_100321)
    # Assigning a type to the variable 'if_condition_100322' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'if_condition_100322', if_condition_100322)
    # SSA begins for if statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output' (line 369)
    output_100323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'output')
    int_100324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 18), 'int')
    
    # Call to sqrt(...): (line 369)
    # Processing the call arguments (line 369)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 369)
    axis_100326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 35), 'axis', False)
    # Getting the type of 'a' (line 369)
    a_100327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'a', False)
    # Obtaining the member 'shape' of a type (line 369)
    shape_100328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 27), a_100327, 'shape')
    # Obtaining the member '__getitem__' of a type (line 369)
    getitem___100329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 27), shape_100328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 369)
    subscript_call_result_100330 = invoke(stypy.reporting.localization.Localization(__file__, 369, 27), getitem___100329, axis_100326)
    
    # Processing the call keyword arguments (line 369)
    kwargs_100331 = {}
    # Getting the type of 'sqrt' (line 369)
    sqrt_100325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 22), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 369)
    sqrt_call_result_100332 = invoke(stypy.reporting.localization.Localization(__file__, 369, 22), sqrt_100325, *[subscript_call_result_100330], **kwargs_100331)
    
    # Applying the binary operator 'div' (line 369)
    result_div_100333 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 18), 'div', int_100324, sqrt_call_result_100332)
    
    # Applying the binary operator '*=' (line 369)
    result_imul_100334 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 8), '*=', output_100323, result_div_100333)
    # Assigning a type to the variable 'output' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'output', result_imul_100334)
    
    # SSA join for if statement (line 368)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'output' (line 370)
    output_100335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'output')
    # Assigning a type to the variable 'stypy_return_type' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type', output_100335)
    
    # ################# End of 'rfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfft' in the type store
    # Getting the type of 'stypy_return_type' (line 287)
    stypy_return_type_100336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfft'
    return stypy_return_type_100336

# Assigning a type to the variable 'rfft' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'rfft', rfft)

@norecursion
def irfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 373)
    None_100337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'None')
    int_100338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 26), 'int')
    # Getting the type of 'None' (line 373)
    None_100339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 35), 'None')
    defaults = [None_100337, int_100338, None_100339]
    # Create a new context for function 'irfft'
    module_type_store = module_type_store.open_function_context('irfft', 373, 0, False)
    
    # Passed parameters checking function
    irfft.stypy_localization = localization
    irfft.stypy_type_of_self = None
    irfft.stypy_type_store = module_type_store
    irfft.stypy_function_name = 'irfft'
    irfft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    irfft.stypy_varargs_param_name = None
    irfft.stypy_kwargs_param_name = None
    irfft.stypy_call_defaults = defaults
    irfft.stypy_call_varargs = varargs
    irfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'irfft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'irfft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'irfft(...)' code ##################

    str_100340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, (-1)), 'str', '\n    Compute the inverse of the n-point DFT for real input.\n\n    This function computes the inverse of the one-dimensional *n*-point\n    discrete Fourier Transform of real input computed by `rfft`.\n    In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical\n    accuracy. (See Notes below for why ``len(a)`` is necessary here.)\n\n    The input is expected to be in the form returned by `rfft`, i.e. the\n    real zero-frequency term followed by the complex positive frequency terms\n    in order of increasing frequency.  Since the discrete Fourier Transform of\n    real input is Hermitian-symmetric, the negative frequency terms are taken\n    to be the complex conjugates of the corresponding positive frequency terms.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array.\n    n : int, optional\n        Length of the transformed axis of the output.\n        For `n` output points, ``n//2+1`` input points are necessary.  If the\n        input is longer than this, it is cropped.  If it is shorter than this,\n        it is padded with zeros.  If `n` is not given, it is determined from\n        the length of the input along the axis specified by `axis`.\n    axis : int, optional\n        Axis over which to compute the inverse FFT. If not given, the last\n        axis is used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        The length of the transformed axis is `n`, or, if `n` is not given,\n        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the\n        input. To get an odd number of output points, `n` must be specified.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is larger than the last axis of `a`.\n\n    See Also\n    --------\n    numpy.fft : For definition of the DFT and conventions used.\n    rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.\n    fft : The one-dimensional FFT.\n    irfft2 : The inverse of the two-dimensional FFT of real input.\n    irfftn : The inverse of the *n*-dimensional FFT of real input.\n\n    Notes\n    -----\n    Returns the real valued `n`-point inverse discrete Fourier transform\n    of `a`, where `a` contains the non-negative frequency terms of a\n    Hermitian-symmetric sequence. `n` is the length of the result, not the\n    input.\n\n    If you specify an `n` such that `a` must be zero-padded or truncated, the\n    extra/removed values will be added/removed at high frequencies. One can\n    thus resample a series to `m` points via Fourier interpolation by:\n    ``a_resamp = irfft(rfft(a), m)``.\n\n    Examples\n    --------\n    >>> np.fft.ifft([1, -1j, -1, 1j])\n    array([ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j])\n    >>> np.fft.irfft([1, -1j, -1])\n    array([ 0.,  1.,  0.,  0.])\n\n    Notice how the last term in the input to the ordinary `ifft` is the\n    complex conjugate of the second term, and the output has zero imaginary\n    part everywhere.  When calling `irfft`, the negative frequencies are not\n    specified, and the output array is purely real.\n\n    ')
    
    # Assigning a Call to a Name (line 453):
    
    # Assigning a Call to a Name (line 453):
    
    # Call to array(...): (line 453)
    # Processing the call arguments (line 453)
    # Getting the type of 'a' (line 453)
    a_100342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 14), 'a', False)
    # Processing the call keyword arguments (line 453)
    # Getting the type of 'True' (line 453)
    True_100343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 22), 'True', False)
    keyword_100344 = True_100343
    # Getting the type of 'complex' (line 453)
    complex_100345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'complex', False)
    keyword_100346 = complex_100345
    kwargs_100347 = {'dtype': keyword_100346, 'copy': keyword_100344}
    # Getting the type of 'array' (line 453)
    array_100341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'array', False)
    # Calling array(args, kwargs) (line 453)
    array_call_result_100348 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), array_100341, *[a_100342], **kwargs_100347)
    
    # Assigning a type to the variable 'a' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'a', array_call_result_100348)
    
    # Type idiom detected: calculating its left and rigth part (line 454)
    # Getting the type of 'n' (line 454)
    n_100349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 7), 'n')
    # Getting the type of 'None' (line 454)
    None_100350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'None')
    
    (may_be_100351, more_types_in_union_100352) = may_be_none(n_100349, None_100350)

    if may_be_100351:

        if more_types_in_union_100352:
            # Runtime conditional SSA (line 454)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 455):
        
        # Assigning a BinOp to a Name (line 455):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 455)
        axis_100353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), 'axis')
        # Getting the type of 'a' (line 455)
        a_100354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'a')
        # Obtaining the member 'shape' of a type (line 455)
        shape_100355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), a_100354, 'shape')
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___100356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), shape_100355, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_100357 = invoke(stypy.reporting.localization.Localization(__file__, 455, 13), getitem___100356, axis_100353)
        
        int_100358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 29), 'int')
        # Applying the binary operator '-' (line 455)
        result_sub_100359 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 13), '-', subscript_call_result_100357, int_100358)
        
        int_100360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 34), 'int')
        # Applying the binary operator '*' (line 455)
        result_mul_100361 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 12), '*', result_sub_100359, int_100360)
        
        # Assigning a type to the variable 'n' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'n', result_mul_100361)

        if more_types_in_union_100352:
            # SSA join for if statement (line 454)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 456):
    
    # Assigning a Call to a Name (line 456):
    
    # Call to _unitary(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'norm' (line 456)
    norm_100363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'norm', False)
    # Processing the call keyword arguments (line 456)
    kwargs_100364 = {}
    # Getting the type of '_unitary' (line 456)
    _unitary_100362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 456)
    _unitary_call_result_100365 = invoke(stypy.reporting.localization.Localization(__file__, 456, 14), _unitary_100362, *[norm_100363], **kwargs_100364)
    
    # Assigning a type to the variable 'unitary' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'unitary', _unitary_call_result_100365)
    
    # Assigning a Call to a Name (line 457):
    
    # Assigning a Call to a Name (line 457):
    
    # Call to _raw_fft(...): (line 457)
    # Processing the call arguments (line 457)
    # Getting the type of 'a' (line 457)
    a_100367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'a', False)
    # Getting the type of 'n' (line 457)
    n_100368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 25), 'n', False)
    # Getting the type of 'axis' (line 457)
    axis_100369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 28), 'axis', False)
    # Getting the type of 'fftpack' (line 457)
    fftpack_100370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 34), 'fftpack', False)
    # Obtaining the member 'rffti' of a type (line 457)
    rffti_100371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 34), fftpack_100370, 'rffti')
    # Getting the type of 'fftpack' (line 457)
    fftpack_100372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'fftpack', False)
    # Obtaining the member 'rfftb' of a type (line 457)
    rfftb_100373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 49), fftpack_100372, 'rfftb')
    # Getting the type of '_real_fft_cache' (line 458)
    _real_fft_cache_100374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 22), '_real_fft_cache', False)
    # Processing the call keyword arguments (line 457)
    kwargs_100375 = {}
    # Getting the type of '_raw_fft' (line 457)
    _raw_fft_100366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 13), '_raw_fft', False)
    # Calling _raw_fft(args, kwargs) (line 457)
    _raw_fft_call_result_100376 = invoke(stypy.reporting.localization.Localization(__file__, 457, 13), _raw_fft_100366, *[a_100367, n_100368, axis_100369, rffti_100371, rfftb_100373, _real_fft_cache_100374], **kwargs_100375)
    
    # Assigning a type to the variable 'output' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'output', _raw_fft_call_result_100376)
    # Getting the type of 'output' (line 459)
    output_100377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 11), 'output')
    int_100378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 21), 'int')
    
    # Getting the type of 'unitary' (line 459)
    unitary_100379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'unitary')
    # Testing the type of an if expression (line 459)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 459, 26), unitary_100379)
    # SSA begins for if expression (line 459)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to sqrt(...): (line 459)
    # Processing the call arguments (line 459)
    # Getting the type of 'n' (line 459)
    n_100381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 31), 'n', False)
    # Processing the call keyword arguments (line 459)
    kwargs_100382 = {}
    # Getting the type of 'sqrt' (line 459)
    sqrt_100380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 26), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 459)
    sqrt_call_result_100383 = invoke(stypy.reporting.localization.Localization(__file__, 459, 26), sqrt_100380, *[n_100381], **kwargs_100382)
    
    # SSA branch for the else part of an if expression (line 459)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 459)
    n_100384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 50), 'n')
    # SSA join for if expression (line 459)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_100385 = union_type.UnionType.add(sqrt_call_result_100383, n_100384)
    
    # Applying the binary operator 'div' (line 459)
    result_div_100386 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 21), 'div', int_100378, if_exp_100385)
    
    # Applying the binary operator '*' (line 459)
    result_mul_100387 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 11), '*', output_100377, result_div_100386)
    
    # Assigning a type to the variable 'stypy_return_type' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type', result_mul_100387)
    
    # ################# End of 'irfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'irfft' in the type store
    # Getting the type of 'stypy_return_type' (line 373)
    stypy_return_type_100388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100388)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'irfft'
    return stypy_return_type_100388

# Assigning a type to the variable 'irfft' (line 373)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 0), 'irfft', irfft)

@norecursion
def hfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 462)
    None_100389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 14), 'None')
    int_100390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 25), 'int')
    # Getting the type of 'None' (line 462)
    None_100391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 34), 'None')
    defaults = [None_100389, int_100390, None_100391]
    # Create a new context for function 'hfft'
    module_type_store = module_type_store.open_function_context('hfft', 462, 0, False)
    
    # Passed parameters checking function
    hfft.stypy_localization = localization
    hfft.stypy_type_of_self = None
    hfft.stypy_type_store = module_type_store
    hfft.stypy_function_name = 'hfft'
    hfft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    hfft.stypy_varargs_param_name = None
    hfft.stypy_kwargs_param_name = None
    hfft.stypy_call_defaults = defaults
    hfft.stypy_call_varargs = varargs
    hfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hfft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hfft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hfft(...)' code ##################

    str_100392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, (-1)), 'str', '\n    Compute the FFT of a signal which has Hermitian symmetry (real spectrum).\n\n    Parameters\n    ----------\n    a : array_like\n        The input array.\n    n : int, optional\n        Length of the transformed axis of the output.\n        For `n` output points, ``n//2+1`` input points are necessary.  If the\n        input is longer than this, it is cropped.  If it is shorter than this,\n        it is padded with zeros.  If `n` is not given, it is determined from\n        the length of the input along the axis specified by `axis`.\n    axis : int, optional\n        Axis over which to compute the FFT. If not given, the last\n        axis is used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        The length of the transformed axis is `n`, or, if `n` is not given,\n        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the\n        input. To get an odd number of output points, `n` must be specified.\n\n    Raises\n    ------\n    IndexError\n        If `axis` is larger than the last axis of `a`.\n\n    See also\n    --------\n    rfft : Compute the one-dimensional FFT for real input.\n    ihfft : The inverse of `hfft`.\n\n    Notes\n    -----\n    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the\n    opposite case: here the signal has Hermitian symmetry in the time domain\n    and is real in the frequency domain. So here it\'s `hfft` for which\n    you must supply the length of the result if it is to be odd:\n    ``ihfft(hfft(a), len(a)) == a``, within numerical accuracy.\n\n    Examples\n    --------\n    >>> signal = np.array([1, 2, 3, 4, 3, 2])\n    >>> np.fft.fft(signal)\n    array([ 15.+0.j,  -4.+0.j,   0.+0.j,  -1.-0.j,   0.+0.j,  -4.+0.j])\n    >>> np.fft.hfft(signal[:4]) # Input first half of signal\n    array([ 15.,  -4.,   0.,  -1.,   0.,  -4.])\n    >>> np.fft.hfft(signal, 6)  # Input entire signal and truncate\n    array([ 15.,  -4.,   0.,  -1.,   0.,  -4.])\n\n\n    >>> signal = np.array([[1, 1.j], [-1.j, 2]])\n    >>> np.conj(signal.T) - signal   # check Hermitian symmetry\n    array([[ 0.-0.j,  0.+0.j],\n           [ 0.+0.j,  0.-0.j]])\n    >>> freq_spectrum = np.fft.hfft(signal)\n    >>> freq_spectrum\n    array([[ 1.,  1.],\n           [ 2., -2.]])\n\n    ')
    
    # Assigning a Call to a Name (line 532):
    
    # Assigning a Call to a Name (line 532):
    
    # Call to array(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'a' (line 532)
    a_100394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 14), 'a', False)
    # Processing the call keyword arguments (line 532)
    # Getting the type of 'True' (line 532)
    True_100395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 22), 'True', False)
    keyword_100396 = True_100395
    # Getting the type of 'complex' (line 532)
    complex_100397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 34), 'complex', False)
    keyword_100398 = complex_100397
    kwargs_100399 = {'dtype': keyword_100398, 'copy': keyword_100396}
    # Getting the type of 'array' (line 532)
    array_100393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'array', False)
    # Calling array(args, kwargs) (line 532)
    array_call_result_100400 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), array_100393, *[a_100394], **kwargs_100399)
    
    # Assigning a type to the variable 'a' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'a', array_call_result_100400)
    
    # Type idiom detected: calculating its left and rigth part (line 533)
    # Getting the type of 'n' (line 533)
    n_100401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 7), 'n')
    # Getting the type of 'None' (line 533)
    None_100402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'None')
    
    (may_be_100403, more_types_in_union_100404) = may_be_none(n_100401, None_100402)

    if may_be_100403:

        if more_types_in_union_100404:
            # Runtime conditional SSA (line 533)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 534):
        
        # Assigning a BinOp to a Name (line 534):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 534)
        axis_100405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 21), 'axis')
        # Getting the type of 'a' (line 534)
        a_100406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'a')
        # Obtaining the member 'shape' of a type (line 534)
        shape_100407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), a_100406, 'shape')
        # Obtaining the member '__getitem__' of a type (line 534)
        getitem___100408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), shape_100407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 534)
        subscript_call_result_100409 = invoke(stypy.reporting.localization.Localization(__file__, 534, 13), getitem___100408, axis_100405)
        
        int_100410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 29), 'int')
        # Applying the binary operator '-' (line 534)
        result_sub_100411 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 13), '-', subscript_call_result_100409, int_100410)
        
        int_100412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 34), 'int')
        # Applying the binary operator '*' (line 534)
        result_mul_100413 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 12), '*', result_sub_100411, int_100412)
        
        # Assigning a type to the variable 'n' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'n', result_mul_100413)

        if more_types_in_union_100404:
            # SSA join for if statement (line 533)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 535):
    
    # Assigning a Call to a Name (line 535):
    
    # Call to _unitary(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'norm' (line 535)
    norm_100415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 23), 'norm', False)
    # Processing the call keyword arguments (line 535)
    kwargs_100416 = {}
    # Getting the type of '_unitary' (line 535)
    _unitary_100414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 14), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 535)
    _unitary_call_result_100417 = invoke(stypy.reporting.localization.Localization(__file__, 535, 14), _unitary_100414, *[norm_100415], **kwargs_100416)
    
    # Assigning a type to the variable 'unitary' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'unitary', _unitary_call_result_100417)
    
    # Call to irfft(...): (line 536)
    # Processing the call arguments (line 536)
    
    # Call to conjugate(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'a' (line 536)
    a_100420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 27), 'a', False)
    # Processing the call keyword arguments (line 536)
    kwargs_100421 = {}
    # Getting the type of 'conjugate' (line 536)
    conjugate_100419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 17), 'conjugate', False)
    # Calling conjugate(args, kwargs) (line 536)
    conjugate_call_result_100422 = invoke(stypy.reporting.localization.Localization(__file__, 536, 17), conjugate_100419, *[a_100420], **kwargs_100421)
    
    # Getting the type of 'n' (line 536)
    n_100423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 31), 'n', False)
    # Getting the type of 'axis' (line 536)
    axis_100424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 34), 'axis', False)
    # Processing the call keyword arguments (line 536)
    kwargs_100425 = {}
    # Getting the type of 'irfft' (line 536)
    irfft_100418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'irfft', False)
    # Calling irfft(args, kwargs) (line 536)
    irfft_call_result_100426 = invoke(stypy.reporting.localization.Localization(__file__, 536, 11), irfft_100418, *[conjugate_call_result_100422, n_100423, axis_100424], **kwargs_100425)
    
    
    # Getting the type of 'unitary' (line 536)
    unitary_100427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 54), 'unitary')
    # Testing the type of an if expression (line 536)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 536, 43), unitary_100427)
    # SSA begins for if expression (line 536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to sqrt(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'n' (line 536)
    n_100429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 48), 'n', False)
    # Processing the call keyword arguments (line 536)
    kwargs_100430 = {}
    # Getting the type of 'sqrt' (line 536)
    sqrt_100428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 43), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 536)
    sqrt_call_result_100431 = invoke(stypy.reporting.localization.Localization(__file__, 536, 43), sqrt_100428, *[n_100429], **kwargs_100430)
    
    # SSA branch for the else part of an if expression (line 536)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 536)
    n_100432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 67), 'n')
    # SSA join for if expression (line 536)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_100433 = union_type.UnionType.add(sqrt_call_result_100431, n_100432)
    
    # Applying the binary operator '*' (line 536)
    result_mul_100434 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), '*', irfft_call_result_100426, if_exp_100433)
    
    # Assigning a type to the variable 'stypy_return_type' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type', result_mul_100434)
    
    # ################# End of 'hfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hfft' in the type store
    # Getting the type of 'stypy_return_type' (line 462)
    stypy_return_type_100435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100435)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hfft'
    return stypy_return_type_100435

# Assigning a type to the variable 'hfft' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'hfft', hfft)

@norecursion
def ihfft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 539)
    None_100436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'None')
    int_100437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 26), 'int')
    # Getting the type of 'None' (line 539)
    None_100438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 35), 'None')
    defaults = [None_100436, int_100437, None_100438]
    # Create a new context for function 'ihfft'
    module_type_store = module_type_store.open_function_context('ihfft', 539, 0, False)
    
    # Passed parameters checking function
    ihfft.stypy_localization = localization
    ihfft.stypy_type_of_self = None
    ihfft.stypy_type_store = module_type_store
    ihfft.stypy_function_name = 'ihfft'
    ihfft.stypy_param_names_list = ['a', 'n', 'axis', 'norm']
    ihfft.stypy_varargs_param_name = None
    ihfft.stypy_kwargs_param_name = None
    ihfft.stypy_call_defaults = defaults
    ihfft.stypy_call_varargs = varargs
    ihfft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ihfft', ['a', 'n', 'axis', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ihfft', localization, ['a', 'n', 'axis', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ihfft(...)' code ##################

    str_100439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, (-1)), 'str', '\n    Compute the inverse FFT of a signal which has Hermitian symmetry.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    n : int, optional\n        Length of the inverse FFT.\n        Number of points along transformation axis in the input to use.\n        If `n` is smaller than the length of the input, the input is cropped.\n        If it is larger, the input is padded with zeros. If `n` is not given,\n        the length of the input along the axis specified by `axis` is used.\n    axis : int, optional\n        Axis over which to compute the inverse FFT. If not given, the last\n        axis is used.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axis\n        indicated by `axis`, or the last one if `axis` is not specified.\n        If `n` is even, the length of the transformed axis is ``(n/2)+1``.\n        If `n` is odd, the length is ``(n+1)/2``.\n\n    See also\n    --------\n    hfft, irfft\n\n    Notes\n    -----\n    `hfft`/`ihfft` are a pair analogous to `rfft`/`irfft`, but for the\n    opposite case: here the signal has Hermitian symmetry in the time domain\n    and is real in the frequency domain. So here it\'s `hfft` for which\n    you must supply the length of the result if it is to be odd:\n    ``ihfft(hfft(a), len(a)) == a``, within numerical accuracy.\n\n    Examples\n    --------\n    >>> spectrum = np.array([ 15, -4, 0, -1, 0, -4])\n    >>> np.fft.ifft(spectrum)\n    array([ 1.+0.j,  2.-0.j,  3.+0.j,  4.+0.j,  3.+0.j,  2.-0.j])\n    >>> np.fft.ihfft(spectrum)\n    array([ 1.-0.j,  2.-0.j,  3.-0.j,  4.-0.j])\n\n    ')
    
    # Assigning a Call to a Name (line 590):
    
    # Assigning a Call to a Name (line 590):
    
    # Call to array(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'a' (line 590)
    a_100441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 14), 'a', False)
    # Processing the call keyword arguments (line 590)
    # Getting the type of 'True' (line 590)
    True_100442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 22), 'True', False)
    keyword_100443 = True_100442
    # Getting the type of 'float' (line 590)
    float_100444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 34), 'float', False)
    keyword_100445 = float_100444
    kwargs_100446 = {'dtype': keyword_100445, 'copy': keyword_100443}
    # Getting the type of 'array' (line 590)
    array_100440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'array', False)
    # Calling array(args, kwargs) (line 590)
    array_call_result_100447 = invoke(stypy.reporting.localization.Localization(__file__, 590, 8), array_100440, *[a_100441], **kwargs_100446)
    
    # Assigning a type to the variable 'a' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'a', array_call_result_100447)
    
    # Type idiom detected: calculating its left and rigth part (line 591)
    # Getting the type of 'n' (line 591)
    n_100448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 7), 'n')
    # Getting the type of 'None' (line 591)
    None_100449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'None')
    
    (may_be_100450, more_types_in_union_100451) = may_be_none(n_100448, None_100449)

    if may_be_100450:

        if more_types_in_union_100451:
            # Runtime conditional SSA (line 591)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 592):
        
        # Assigning a Subscript to a Name (line 592):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 592)
        axis_100452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 20), 'axis')
        # Getting the type of 'a' (line 592)
        a_100453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'a')
        # Obtaining the member 'shape' of a type (line 592)
        shape_100454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 12), a_100453, 'shape')
        # Obtaining the member '__getitem__' of a type (line 592)
        getitem___100455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 12), shape_100454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 592)
        subscript_call_result_100456 = invoke(stypy.reporting.localization.Localization(__file__, 592, 12), getitem___100455, axis_100452)
        
        # Assigning a type to the variable 'n' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'n', subscript_call_result_100456)

        if more_types_in_union_100451:
            # SSA join for if statement (line 591)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 593):
    
    # Assigning a Call to a Name (line 593):
    
    # Call to _unitary(...): (line 593)
    # Processing the call arguments (line 593)
    # Getting the type of 'norm' (line 593)
    norm_100458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'norm', False)
    # Processing the call keyword arguments (line 593)
    kwargs_100459 = {}
    # Getting the type of '_unitary' (line 593)
    _unitary_100457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 14), '_unitary', False)
    # Calling _unitary(args, kwargs) (line 593)
    _unitary_call_result_100460 = invoke(stypy.reporting.localization.Localization(__file__, 593, 14), _unitary_100457, *[norm_100458], **kwargs_100459)
    
    # Assigning a type to the variable 'unitary' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 4), 'unitary', _unitary_call_result_100460)
    
    # Assigning a Call to a Name (line 594):
    
    # Assigning a Call to a Name (line 594):
    
    # Call to conjugate(...): (line 594)
    # Processing the call arguments (line 594)
    
    # Call to rfft(...): (line 594)
    # Processing the call arguments (line 594)
    # Getting the type of 'a' (line 594)
    a_100463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 28), 'a', False)
    # Getting the type of 'n' (line 594)
    n_100464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 31), 'n', False)
    # Getting the type of 'axis' (line 594)
    axis_100465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 34), 'axis', False)
    # Processing the call keyword arguments (line 594)
    kwargs_100466 = {}
    # Getting the type of 'rfft' (line 594)
    rfft_100462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 23), 'rfft', False)
    # Calling rfft(args, kwargs) (line 594)
    rfft_call_result_100467 = invoke(stypy.reporting.localization.Localization(__file__, 594, 23), rfft_100462, *[a_100463, n_100464, axis_100465], **kwargs_100466)
    
    # Processing the call keyword arguments (line 594)
    kwargs_100468 = {}
    # Getting the type of 'conjugate' (line 594)
    conjugate_100461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 13), 'conjugate', False)
    # Calling conjugate(args, kwargs) (line 594)
    conjugate_call_result_100469 = invoke(stypy.reporting.localization.Localization(__file__, 594, 13), conjugate_100461, *[rfft_call_result_100467], **kwargs_100468)
    
    # Assigning a type to the variable 'output' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 4), 'output', conjugate_call_result_100469)
    # Getting the type of 'output' (line 595)
    output_100470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'output')
    int_100471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 21), 'int')
    
    # Getting the type of 'unitary' (line 595)
    unitary_100472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 37), 'unitary')
    # Testing the type of an if expression (line 595)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 26), unitary_100472)
    # SSA begins for if expression (line 595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to sqrt(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'n' (line 595)
    n_100474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 31), 'n', False)
    # Processing the call keyword arguments (line 595)
    kwargs_100475 = {}
    # Getting the type of 'sqrt' (line 595)
    sqrt_100473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 595)
    sqrt_call_result_100476 = invoke(stypy.reporting.localization.Localization(__file__, 595, 26), sqrt_100473, *[n_100474], **kwargs_100475)
    
    # SSA branch for the else part of an if expression (line 595)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 595)
    n_100477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 50), 'n')
    # SSA join for if expression (line 595)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_100478 = union_type.UnionType.add(sqrt_call_result_100476, n_100477)
    
    # Applying the binary operator 'div' (line 595)
    result_div_100479 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 21), 'div', int_100471, if_exp_100478)
    
    # Applying the binary operator '*' (line 595)
    result_mul_100480 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 11), '*', output_100470, result_div_100479)
    
    # Assigning a type to the variable 'stypy_return_type' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type', result_mul_100480)
    
    # ################# End of 'ihfft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ihfft' in the type store
    # Getting the type of 'stypy_return_type' (line 539)
    stypy_return_type_100481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100481)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ihfft'
    return stypy_return_type_100481

# Assigning a type to the variable 'ihfft' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'ihfft', ihfft)

@norecursion
def _cook_nd_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 598)
    None_100482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 23), 'None')
    # Getting the type of 'None' (line 598)
    None_100483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 34), 'None')
    int_100484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 48), 'int')
    defaults = [None_100482, None_100483, int_100484]
    # Create a new context for function '_cook_nd_args'
    module_type_store = module_type_store.open_function_context('_cook_nd_args', 598, 0, False)
    
    # Passed parameters checking function
    _cook_nd_args.stypy_localization = localization
    _cook_nd_args.stypy_type_of_self = None
    _cook_nd_args.stypy_type_store = module_type_store
    _cook_nd_args.stypy_function_name = '_cook_nd_args'
    _cook_nd_args.stypy_param_names_list = ['a', 's', 'axes', 'invreal']
    _cook_nd_args.stypy_varargs_param_name = None
    _cook_nd_args.stypy_kwargs_param_name = None
    _cook_nd_args.stypy_call_defaults = defaults
    _cook_nd_args.stypy_call_varargs = varargs
    _cook_nd_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cook_nd_args', ['a', 's', 'axes', 'invreal'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cook_nd_args', localization, ['a', 's', 'axes', 'invreal'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cook_nd_args(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 599)
    # Getting the type of 's' (line 599)
    s_100485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 7), 's')
    # Getting the type of 'None' (line 599)
    None_100486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'None')
    
    (may_be_100487, more_types_in_union_100488) = may_be_none(s_100485, None_100486)

    if may_be_100487:

        if more_types_in_union_100488:
            # Runtime conditional SSA (line 599)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 600):
        
        # Assigning a Num to a Name (line 600):
        int_100489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 20), 'int')
        # Assigning a type to the variable 'shapeless' (line 600)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 8), 'shapeless', int_100489)
        
        # Type idiom detected: calculating its left and rigth part (line 601)
        # Getting the type of 'axes' (line 601)
        axes_100490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 11), 'axes')
        # Getting the type of 'None' (line 601)
        None_100491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 19), 'None')
        
        (may_be_100492, more_types_in_union_100493) = may_be_none(axes_100490, None_100491)

        if may_be_100492:

            if more_types_in_union_100493:
                # Runtime conditional SSA (line 601)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 602):
            
            # Assigning a Call to a Name (line 602):
            
            # Call to list(...): (line 602)
            # Processing the call arguments (line 602)
            # Getting the type of 'a' (line 602)
            a_100495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'a', False)
            # Obtaining the member 'shape' of a type (line 602)
            shape_100496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 21), a_100495, 'shape')
            # Processing the call keyword arguments (line 602)
            kwargs_100497 = {}
            # Getting the type of 'list' (line 602)
            list_100494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 16), 'list', False)
            # Calling list(args, kwargs) (line 602)
            list_call_result_100498 = invoke(stypy.reporting.localization.Localization(__file__, 602, 16), list_100494, *[shape_100496], **kwargs_100497)
            
            # Assigning a type to the variable 's' (line 602)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 's', list_call_result_100498)

            if more_types_in_union_100493:
                # Runtime conditional SSA for else branch (line 601)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_100492) or more_types_in_union_100493):
            
            # Assigning a Call to a Name (line 604):
            
            # Assigning a Call to a Name (line 604):
            
            # Call to take(...): (line 604)
            # Processing the call arguments (line 604)
            # Getting the type of 'a' (line 604)
            a_100500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 21), 'a', False)
            # Obtaining the member 'shape' of a type (line 604)
            shape_100501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 21), a_100500, 'shape')
            # Getting the type of 'axes' (line 604)
            axes_100502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'axes', False)
            # Processing the call keyword arguments (line 604)
            kwargs_100503 = {}
            # Getting the type of 'take' (line 604)
            take_100499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 16), 'take', False)
            # Calling take(args, kwargs) (line 604)
            take_call_result_100504 = invoke(stypy.reporting.localization.Localization(__file__, 604, 16), take_100499, *[shape_100501, axes_100502], **kwargs_100503)
            
            # Assigning a type to the variable 's' (line 604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 's', take_call_result_100504)

            if (may_be_100492 and more_types_in_union_100493):
                # SSA join for if statement (line 601)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_100488:
            # Runtime conditional SSA for else branch (line 599)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_100487) or more_types_in_union_100488):
        
        # Assigning a Num to a Name (line 606):
        
        # Assigning a Num to a Name (line 606):
        int_100505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 20), 'int')
        # Assigning a type to the variable 'shapeless' (line 606)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 8), 'shapeless', int_100505)

        if (may_be_100487 and more_types_in_union_100488):
            # SSA join for if statement (line 599)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 607):
    
    # Assigning a Call to a Name (line 607):
    
    # Call to list(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of 's' (line 607)
    s_100507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 13), 's', False)
    # Processing the call keyword arguments (line 607)
    kwargs_100508 = {}
    # Getting the type of 'list' (line 607)
    list_100506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'list', False)
    # Calling list(args, kwargs) (line 607)
    list_call_result_100509 = invoke(stypy.reporting.localization.Localization(__file__, 607, 8), list_100506, *[s_100507], **kwargs_100508)
    
    # Assigning a type to the variable 's' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 's', list_call_result_100509)
    
    # Type idiom detected: calculating its left and rigth part (line 608)
    # Getting the type of 'axes' (line 608)
    axes_100510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 7), 'axes')
    # Getting the type of 'None' (line 608)
    None_100511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 15), 'None')
    
    (may_be_100512, more_types_in_union_100513) = may_be_none(axes_100510, None_100511)

    if may_be_100512:

        if more_types_in_union_100513:
            # Runtime conditional SSA (line 608)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 609):
        
        # Assigning a Call to a Name (line 609):
        
        # Call to list(...): (line 609)
        # Processing the call arguments (line 609)
        
        # Call to range(...): (line 609)
        # Processing the call arguments (line 609)
        
        
        # Call to len(...): (line 609)
        # Processing the call arguments (line 609)
        # Getting the type of 's' (line 609)
        s_100517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 31), 's', False)
        # Processing the call keyword arguments (line 609)
        kwargs_100518 = {}
        # Getting the type of 'len' (line 609)
        len_100516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 27), 'len', False)
        # Calling len(args, kwargs) (line 609)
        len_call_result_100519 = invoke(stypy.reporting.localization.Localization(__file__, 609, 27), len_100516, *[s_100517], **kwargs_100518)
        
        # Applying the 'usub' unary operator (line 609)
        result___neg___100520 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 26), 'usub', len_call_result_100519)
        
        int_100521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 35), 'int')
        # Processing the call keyword arguments (line 609)
        kwargs_100522 = {}
        # Getting the type of 'range' (line 609)
        range_100515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 20), 'range', False)
        # Calling range(args, kwargs) (line 609)
        range_call_result_100523 = invoke(stypy.reporting.localization.Localization(__file__, 609, 20), range_100515, *[result___neg___100520, int_100521], **kwargs_100522)
        
        # Processing the call keyword arguments (line 609)
        kwargs_100524 = {}
        # Getting the type of 'list' (line 609)
        list_100514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), 'list', False)
        # Calling list(args, kwargs) (line 609)
        list_call_result_100525 = invoke(stypy.reporting.localization.Localization(__file__, 609, 15), list_100514, *[range_call_result_100523], **kwargs_100524)
        
        # Assigning a type to the variable 'axes' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'axes', list_call_result_100525)

        if more_types_in_union_100513:
            # SSA join for if statement (line 608)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 's' (line 610)
    s_100527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 's', False)
    # Processing the call keyword arguments (line 610)
    kwargs_100528 = {}
    # Getting the type of 'len' (line 610)
    len_100526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 7), 'len', False)
    # Calling len(args, kwargs) (line 610)
    len_call_result_100529 = invoke(stypy.reporting.localization.Localization(__file__, 610, 7), len_100526, *[s_100527], **kwargs_100528)
    
    
    # Call to len(...): (line 610)
    # Processing the call arguments (line 610)
    # Getting the type of 'axes' (line 610)
    axes_100531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 21), 'axes', False)
    # Processing the call keyword arguments (line 610)
    kwargs_100532 = {}
    # Getting the type of 'len' (line 610)
    len_100530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 17), 'len', False)
    # Calling len(args, kwargs) (line 610)
    len_call_result_100533 = invoke(stypy.reporting.localization.Localization(__file__, 610, 17), len_100530, *[axes_100531], **kwargs_100532)
    
    # Applying the binary operator '!=' (line 610)
    result_ne_100534 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 7), '!=', len_call_result_100529, len_call_result_100533)
    
    # Testing the type of an if condition (line 610)
    if_condition_100535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 4), result_ne_100534)
    # Assigning a type to the variable 'if_condition_100535' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'if_condition_100535', if_condition_100535)
    # SSA begins for if statement (line 610)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 611)
    # Processing the call arguments (line 611)
    str_100537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 25), 'str', 'Shape and axes have different lengths.')
    # Processing the call keyword arguments (line 611)
    kwargs_100538 = {}
    # Getting the type of 'ValueError' (line 611)
    ValueError_100536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 611)
    ValueError_call_result_100539 = invoke(stypy.reporting.localization.Localization(__file__, 611, 14), ValueError_100536, *[str_100537], **kwargs_100538)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 611, 8), ValueError_call_result_100539, 'raise parameter', BaseException)
    # SSA join for if statement (line 610)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'invreal' (line 612)
    invreal_100540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 7), 'invreal')
    # Getting the type of 'shapeless' (line 612)
    shapeless_100541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 19), 'shapeless')
    # Applying the binary operator 'and' (line 612)
    result_and_keyword_100542 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 7), 'and', invreal_100540, shapeless_100541)
    
    # Testing the type of an if condition (line 612)
    if_condition_100543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 612, 4), result_and_keyword_100542)
    # Assigning a type to the variable 'if_condition_100543' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'if_condition_100543', if_condition_100543)
    # SSA begins for if statement (line 612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 613):
    
    # Assigning a BinOp to a Subscript (line 613):
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_100544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 30), 'int')
    # Getting the type of 'axes' (line 613)
    axes_100545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 25), 'axes')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___100546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 25), axes_100545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_100547 = invoke(stypy.reporting.localization.Localization(__file__, 613, 25), getitem___100546, int_100544)
    
    # Getting the type of 'a' (line 613)
    a_100548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), 'a')
    # Obtaining the member 'shape' of a type (line 613)
    shape_100549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 17), a_100548, 'shape')
    # Obtaining the member '__getitem__' of a type (line 613)
    getitem___100550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 17), shape_100549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 613)
    subscript_call_result_100551 = invoke(stypy.reporting.localization.Localization(__file__, 613, 17), getitem___100550, subscript_call_result_100547)
    
    int_100552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 37), 'int')
    # Applying the binary operator '-' (line 613)
    result_sub_100553 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 17), '-', subscript_call_result_100551, int_100552)
    
    int_100554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 42), 'int')
    # Applying the binary operator '*' (line 613)
    result_mul_100555 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 16), '*', result_sub_100553, int_100554)
    
    # Getting the type of 's' (line 613)
    s_100556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 's')
    int_100557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 10), 'int')
    # Storing an element on a container (line 613)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 613, 8), s_100556, (int_100557, result_mul_100555))
    # SSA join for if statement (line 612)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 614)
    tuple_100558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 614)
    # Adding element type (line 614)
    # Getting the type of 's' (line 614)
    s_100559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 11), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 11), tuple_100558, s_100559)
    # Adding element type (line 614)
    # Getting the type of 'axes' (line 614)
    axes_100560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 14), 'axes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 614, 11), tuple_100558, axes_100560)
    
    # Assigning a type to the variable 'stypy_return_type' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'stypy_return_type', tuple_100558)
    
    # ################# End of '_cook_nd_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cook_nd_args' in the type store
    # Getting the type of 'stypy_return_type' (line 598)
    stypy_return_type_100561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100561)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cook_nd_args'
    return stypy_return_type_100561

# Assigning a type to the variable '_cook_nd_args' (line 598)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 0), '_cook_nd_args', _cook_nd_args)

@norecursion
def _raw_fftnd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 617)
    None_100562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'None')
    # Getting the type of 'None' (line 617)
    None_100563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 31), 'None')
    # Getting the type of 'fft' (line 617)
    fft_100564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 46), 'fft')
    # Getting the type of 'None' (line 617)
    None_100565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 56), 'None')
    defaults = [None_100562, None_100563, fft_100564, None_100565]
    # Create a new context for function '_raw_fftnd'
    module_type_store = module_type_store.open_function_context('_raw_fftnd', 617, 0, False)
    
    # Passed parameters checking function
    _raw_fftnd.stypy_localization = localization
    _raw_fftnd.stypy_type_of_self = None
    _raw_fftnd.stypy_type_store = module_type_store
    _raw_fftnd.stypy_function_name = '_raw_fftnd'
    _raw_fftnd.stypy_param_names_list = ['a', 's', 'axes', 'function', 'norm']
    _raw_fftnd.stypy_varargs_param_name = None
    _raw_fftnd.stypy_kwargs_param_name = None
    _raw_fftnd.stypy_call_defaults = defaults
    _raw_fftnd.stypy_call_varargs = varargs
    _raw_fftnd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_raw_fftnd', ['a', 's', 'axes', 'function', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_raw_fftnd', localization, ['a', 's', 'axes', 'function', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_raw_fftnd(...)' code ##################

    
    # Assigning a Call to a Name (line 618):
    
    # Assigning a Call to a Name (line 618):
    
    # Call to asarray(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a' (line 618)
    a_100567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'a', False)
    # Processing the call keyword arguments (line 618)
    kwargs_100568 = {}
    # Getting the type of 'asarray' (line 618)
    asarray_100566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 618)
    asarray_call_result_100569 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), asarray_100566, *[a_100567], **kwargs_100568)
    
    # Assigning a type to the variable 'a' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'a', asarray_call_result_100569)
    
    # Assigning a Call to a Tuple (line 619):
    
    # Assigning a Call to a Name:
    
    # Call to _cook_nd_args(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'a' (line 619)
    a_100571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'a', False)
    # Getting the type of 's' (line 619)
    s_100572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 31), 's', False)
    # Getting the type of 'axes' (line 619)
    axes_100573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 34), 'axes', False)
    # Processing the call keyword arguments (line 619)
    kwargs_100574 = {}
    # Getting the type of '_cook_nd_args' (line 619)
    _cook_nd_args_100570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 14), '_cook_nd_args', False)
    # Calling _cook_nd_args(args, kwargs) (line 619)
    _cook_nd_args_call_result_100575 = invoke(stypy.reporting.localization.Localization(__file__, 619, 14), _cook_nd_args_100570, *[a_100571, s_100572, axes_100573], **kwargs_100574)
    
    # Assigning a type to the variable 'call_assignment_99998' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_99998', _cook_nd_args_call_result_100575)
    
    # Assigning a Call to a Name (line 619):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100579 = {}
    # Getting the type of 'call_assignment_99998' (line 619)
    call_assignment_99998_100576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_99998', False)
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___100577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 4), call_assignment_99998_100576, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100580 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100577, *[int_100578], **kwargs_100579)
    
    # Assigning a type to the variable 'call_assignment_99999' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_99999', getitem___call_result_100580)
    
    # Assigning a Name to a Name (line 619):
    # Getting the type of 'call_assignment_99999' (line 619)
    call_assignment_99999_100581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_99999')
    # Assigning a type to the variable 's' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 's', call_assignment_99999_100581)
    
    # Assigning a Call to a Name (line 619):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100585 = {}
    # Getting the type of 'call_assignment_99998' (line 619)
    call_assignment_99998_100582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_99998', False)
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___100583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 4), call_assignment_99998_100582, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100586 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100583, *[int_100584], **kwargs_100585)
    
    # Assigning a type to the variable 'call_assignment_100000' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_100000', getitem___call_result_100586)
    
    # Assigning a Name to a Name (line 619):
    # Getting the type of 'call_assignment_100000' (line 619)
    call_assignment_100000_100587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'call_assignment_100000')
    # Assigning a type to the variable 'axes' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 7), 'axes', call_assignment_100000_100587)
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 620):
    
    # Call to list(...): (line 620)
    # Processing the call arguments (line 620)
    
    # Call to range(...): (line 620)
    # Processing the call arguments (line 620)
    
    # Call to len(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'axes' (line 620)
    axes_100591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 25), 'axes', False)
    # Processing the call keyword arguments (line 620)
    kwargs_100592 = {}
    # Getting the type of 'len' (line 620)
    len_100590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'len', False)
    # Calling len(args, kwargs) (line 620)
    len_call_result_100593 = invoke(stypy.reporting.localization.Localization(__file__, 620, 21), len_100590, *[axes_100591], **kwargs_100592)
    
    # Processing the call keyword arguments (line 620)
    kwargs_100594 = {}
    # Getting the type of 'range' (line 620)
    range_100589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'range', False)
    # Calling range(args, kwargs) (line 620)
    range_call_result_100595 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), range_100589, *[len_call_result_100593], **kwargs_100594)
    
    # Processing the call keyword arguments (line 620)
    kwargs_100596 = {}
    # Getting the type of 'list' (line 620)
    list_100588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 10), 'list', False)
    # Calling list(args, kwargs) (line 620)
    list_call_result_100597 = invoke(stypy.reporting.localization.Localization(__file__, 620, 10), list_100588, *[range_call_result_100595], **kwargs_100596)
    
    # Assigning a type to the variable 'itl' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'itl', list_call_result_100597)
    
    # Call to reverse(...): (line 621)
    # Processing the call keyword arguments (line 621)
    kwargs_100600 = {}
    # Getting the type of 'itl' (line 621)
    itl_100598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'itl', False)
    # Obtaining the member 'reverse' of a type (line 621)
    reverse_100599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 4), itl_100598, 'reverse')
    # Calling reverse(args, kwargs) (line 621)
    reverse_call_result_100601 = invoke(stypy.reporting.localization.Localization(__file__, 621, 4), reverse_100599, *[], **kwargs_100600)
    
    
    # Getting the type of 'itl' (line 622)
    itl_100602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 14), 'itl')
    # Testing the type of a for loop iterable (line 622)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 622, 4), itl_100602)
    # Getting the type of the for loop variable (line 622)
    for_loop_var_100603 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 622, 4), itl_100602)
    # Assigning a type to the variable 'ii' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'ii', for_loop_var_100603)
    # SSA begins for a for statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 623):
    
    # Assigning a Call to a Name (line 623):
    
    # Call to function(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'a' (line 623)
    a_100605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), 'a', False)
    # Processing the call keyword arguments (line 623)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 623)
    ii_100606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'ii', False)
    # Getting the type of 's' (line 623)
    s_100607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 26), 's', False)
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___100608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 26), s_100607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_100609 = invoke(stypy.reporting.localization.Localization(__file__, 623, 26), getitem___100608, ii_100606)
    
    keyword_100610 = subscript_call_result_100609
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 623)
    ii_100611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 43), 'ii', False)
    # Getting the type of 'axes' (line 623)
    axes_100612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 38), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___100613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 38), axes_100612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_100614 = invoke(stypy.reporting.localization.Localization(__file__, 623, 38), getitem___100613, ii_100611)
    
    keyword_100615 = subscript_call_result_100614
    # Getting the type of 'norm' (line 623)
    norm_100616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 53), 'norm', False)
    keyword_100617 = norm_100616
    kwargs_100618 = {'axis': keyword_100615, 'norm': keyword_100617, 'n': keyword_100610}
    # Getting the type of 'function' (line 623)
    function_100604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'function', False)
    # Calling function(args, kwargs) (line 623)
    function_call_result_100619 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), function_100604, *[a_100605], **kwargs_100618)
    
    # Assigning a type to the variable 'a' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'a', function_call_result_100619)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 624)
    a_100620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'stypy_return_type', a_100620)
    
    # ################# End of '_raw_fftnd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_raw_fftnd' in the type store
    # Getting the type of 'stypy_return_type' (line 617)
    stypy_return_type_100621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_raw_fftnd'
    return stypy_return_type_100621

# Assigning a type to the variable '_raw_fftnd' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), '_raw_fftnd', _raw_fftnd)

@norecursion
def fftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 627)
    None_100622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 14), 'None')
    # Getting the type of 'None' (line 627)
    None_100623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 25), 'None')
    # Getting the type of 'None' (line 627)
    None_100624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 36), 'None')
    defaults = [None_100622, None_100623, None_100624]
    # Create a new context for function 'fftn'
    module_type_store = module_type_store.open_function_context('fftn', 627, 0, False)
    
    # Passed parameters checking function
    fftn.stypy_localization = localization
    fftn.stypy_type_of_self = None
    fftn.stypy_type_store = module_type_store
    fftn.stypy_function_name = 'fftn'
    fftn.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    fftn.stypy_varargs_param_name = None
    fftn.stypy_kwargs_param_name = None
    fftn.stypy_call_defaults = defaults
    fftn.stypy_call_varargs = varargs
    fftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fftn', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fftn', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fftn(...)' code ##################

    str_100625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, (-1)), 'str', '\n    Compute the N-dimensional discrete Fourier Transform.\n\n    This function computes the *N*-dimensional discrete Fourier Transform over\n    any number of axes in an *M*-dimensional array by means of the Fast Fourier\n    Transform (FFT).\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).\n        This corresponds to `n` for `fft(x, n)`.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the transform over that axis is\n        performed multiple times.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` and `a`,\n        as explained in the parameters section above.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n        and conventions used.\n    ifftn : The inverse of `fftn`, the inverse *n*-dimensional FFT.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    rfftn : The *n*-dimensional FFT of real input.\n    fft2 : The two-dimensional FFT.\n    fftshift : Shifts zero-frequency terms to centre of array\n\n    Notes\n    -----\n    The output, analogously to `fft`, contains the term for zero frequency in\n    the low-order corner of all axes, the positive frequency terms in the\n    first half of all axes, the term for the Nyquist frequency in the middle\n    of all axes and the negative frequency terms in the second half of all\n    axes, in order of decreasingly negative frequency.\n\n    See `numpy.fft` for details, definitions and conventions used.\n\n    Examples\n    --------\n    >>> a = np.mgrid[:3, :3, :3][0]\n    >>> np.fft.fftn(a, axes=(1, 2))\n    array([[[  0.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j]],\n           [[  9.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j]],\n           [[ 18.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j],\n            [  0.+0.j,   0.+0.j,   0.+0.j]]])\n    >>> np.fft.fftn(a, (2, 2), axes=(0, 1))\n    array([[[ 2.+0.j,  2.+0.j,  2.+0.j],\n            [ 0.+0.j,  0.+0.j,  0.+0.j]],\n           [[-2.+0.j, -2.+0.j, -2.+0.j],\n            [ 0.+0.j,  0.+0.j,  0.+0.j]]])\n\n    >>> import matplotlib.pyplot as plt\n    >>> [X, Y] = np.meshgrid(2 * np.pi * np.arange(200) / 12,\n    ...                      2 * np.pi * np.arange(200) / 34)\n    >>> S = np.sin(X) + np.cos(Y) + np.random.uniform(0, 1, X.shape)\n    >>> FS = np.fft.fftn(S)\n    >>> plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))\n    <matplotlib.image.AxesImage object at 0x...>\n    >>> plt.show()\n\n    ')
    
    # Call to _raw_fftnd(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'a' (line 720)
    a_100627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 22), 'a', False)
    # Getting the type of 's' (line 720)
    s_100628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 25), 's', False)
    # Getting the type of 'axes' (line 720)
    axes_100629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 28), 'axes', False)
    # Getting the type of 'fft' (line 720)
    fft_100630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 34), 'fft', False)
    # Getting the type of 'norm' (line 720)
    norm_100631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 39), 'norm', False)
    # Processing the call keyword arguments (line 720)
    kwargs_100632 = {}
    # Getting the type of '_raw_fftnd' (line 720)
    _raw_fftnd_100626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 11), '_raw_fftnd', False)
    # Calling _raw_fftnd(args, kwargs) (line 720)
    _raw_fftnd_call_result_100633 = invoke(stypy.reporting.localization.Localization(__file__, 720, 11), _raw_fftnd_100626, *[a_100627, s_100628, axes_100629, fft_100630, norm_100631], **kwargs_100632)
    
    # Assigning a type to the variable 'stypy_return_type' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'stypy_return_type', _raw_fftnd_call_result_100633)
    
    # ################# End of 'fftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fftn' in the type store
    # Getting the type of 'stypy_return_type' (line 627)
    stypy_return_type_100634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100634)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fftn'
    return stypy_return_type_100634

# Assigning a type to the variable 'fftn' (line 627)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'fftn', fftn)

@norecursion
def ifftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 723)
    None_100635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 15), 'None')
    # Getting the type of 'None' (line 723)
    None_100636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 26), 'None')
    # Getting the type of 'None' (line 723)
    None_100637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 37), 'None')
    defaults = [None_100635, None_100636, None_100637]
    # Create a new context for function 'ifftn'
    module_type_store = module_type_store.open_function_context('ifftn', 723, 0, False)
    
    # Passed parameters checking function
    ifftn.stypy_localization = localization
    ifftn.stypy_type_of_self = None
    ifftn.stypy_type_store = module_type_store
    ifftn.stypy_function_name = 'ifftn'
    ifftn.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    ifftn.stypy_varargs_param_name = None
    ifftn.stypy_kwargs_param_name = None
    ifftn.stypy_call_defaults = defaults
    ifftn.stypy_call_varargs = varargs
    ifftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifftn', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifftn', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifftn(...)' code ##################

    str_100638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, (-1)), 'str', '\n    Compute the N-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the N-dimensional discrete\n    Fourier Transform over any number of axes in an M-dimensional array by\n    means of the Fast Fourier Transform (FFT).  In other words,\n    ``ifftn(fftn(a)) == a`` to within numerical accuracy.\n    For a description of the definitions and conventions used, see `numpy.fft`.\n\n    The input, analogously to `ifft`, should be ordered in the same way as is\n    returned by `fftn`, i.e. it should have the term for zero frequency\n    in all axes in the low-order corner, the positive frequency terms in the\n    first half of all axes, the term for the Nyquist frequency in the middle\n    of all axes and the negative frequency terms in the second half of all\n    axes, in order of decreasingly negative frequency.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        This corresponds to ``n`` for ``ifft(x, n)``.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.  See notes for issue on `ifft` zero padding.\n    axes : sequence of ints, optional\n        Axes over which to compute the IFFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the inverse transform over that\n        axis is performed multiple times.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` or `a`,\n        as explained in the parameters section above.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    fftn : The forward *n*-dimensional FFT, of which `ifftn` is the inverse.\n    ifft : The one-dimensional inverse FFT.\n    ifft2 : The two-dimensional inverse FFT.\n    ifftshift : Undoes `fftshift`, shifts zero-frequency terms to beginning\n        of array.\n\n    Notes\n    -----\n    See `numpy.fft` for definitions and conventions used.\n\n    Zero-padding, analogously with `ifft`, is performed by appending zeros to\n    the input along the specified dimension.  Although this is the common\n    approach, it might lead to surprising results.  If another form of zero\n    padding is desired, it must be performed before `ifftn` is called.\n\n    Examples\n    --------\n    >>> a = np.eye(4)\n    >>> np.fft.ifftn(np.fft.fftn(a, axes=(0,)), axes=(1,))\n    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],\n           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],\n           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],\n           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j]])\n\n\n    Create and plot an image with band-limited frequency content:\n\n    >>> import matplotlib.pyplot as plt\n    >>> n = np.zeros((200,200), dtype=complex)\n    >>> n[60:80, 20:40] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20, 20)))\n    >>> im = np.fft.ifftn(n).real\n    >>> plt.imshow(im)\n    <matplotlib.image.AxesImage object at 0x...>\n    >>> plt.show()\n\n    ')
    
    # Call to _raw_fftnd(...): (line 816)
    # Processing the call arguments (line 816)
    # Getting the type of 'a' (line 816)
    a_100640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 22), 'a', False)
    # Getting the type of 's' (line 816)
    s_100641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 25), 's', False)
    # Getting the type of 'axes' (line 816)
    axes_100642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 28), 'axes', False)
    # Getting the type of 'ifft' (line 816)
    ifft_100643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 34), 'ifft', False)
    # Getting the type of 'norm' (line 816)
    norm_100644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 40), 'norm', False)
    # Processing the call keyword arguments (line 816)
    kwargs_100645 = {}
    # Getting the type of '_raw_fftnd' (line 816)
    _raw_fftnd_100639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 11), '_raw_fftnd', False)
    # Calling _raw_fftnd(args, kwargs) (line 816)
    _raw_fftnd_call_result_100646 = invoke(stypy.reporting.localization.Localization(__file__, 816, 11), _raw_fftnd_100639, *[a_100640, s_100641, axes_100642, ifft_100643, norm_100644], **kwargs_100645)
    
    # Assigning a type to the variable 'stypy_return_type' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'stypy_return_type', _raw_fftnd_call_result_100646)
    
    # ################# End of 'ifftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifftn' in the type store
    # Getting the type of 'stypy_return_type' (line 723)
    stypy_return_type_100647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifftn'
    return stypy_return_type_100647

# Assigning a type to the variable 'ifftn' (line 723)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), 'ifftn', ifftn)

@norecursion
def fft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 819)
    None_100648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 14), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 819)
    tuple_100649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 819)
    # Adding element type (line 819)
    int_100650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 26), tuple_100649, int_100650)
    # Adding element type (line 819)
    int_100651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 819, 26), tuple_100649, int_100651)
    
    # Getting the type of 'None' (line 819)
    None_100652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 40), 'None')
    defaults = [None_100648, tuple_100649, None_100652]
    # Create a new context for function 'fft2'
    module_type_store = module_type_store.open_function_context('fft2', 819, 0, False)
    
    # Passed parameters checking function
    fft2.stypy_localization = localization
    fft2.stypy_type_of_self = None
    fft2.stypy_type_store = module_type_store
    fft2.stypy_function_name = 'fft2'
    fft2.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    fft2.stypy_varargs_param_name = None
    fft2.stypy_kwargs_param_name = None
    fft2.stypy_call_defaults = defaults
    fft2.stypy_call_varargs = varargs
    fft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fft2', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fft2', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fft2(...)' code ##################

    str_100653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, (-1)), 'str', '\n    Compute the 2-dimensional discrete Fourier Transform\n\n    This function computes the *n*-dimensional discrete Fourier Transform\n    over any axes in an *M*-dimensional array by means of the\n    Fast Fourier Transform (FFT).  By default, the transform is computed over\n    the last two axes of the input array, i.e., a 2-dimensional FFT.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (`s[0]` refers to axis 0, `s[1]` to axis 1, etc.).\n        This corresponds to `n` for `fft(x, n)`.\n        Along each axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last two\n        axes are used.  A repeated index in `axes` means the transform over\n        that axis is performed multiple times.  A one-element sequence means\n        that a one-dimensional FFT is performed.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or the last two axes if `axes` is not given.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length, or `axes` not given and\n        ``len(s) != 2``.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    ifft2 : The inverse two-dimensional FFT.\n    fft : The one-dimensional FFT.\n    fftn : The *n*-dimensional FFT.\n    fftshift : Shifts zero-frequency terms to the center of the array.\n        For two-dimensional input, swaps first and third quadrants, and second\n        and fourth quadrants.\n\n    Notes\n    -----\n    `fft2` is just `fftn` with a different default for `axes`.\n\n    The output, analogously to `fft`, contains the term for zero frequency in\n    the low-order corner of the transformed axes, the positive frequency terms\n    in the first half of these axes, the term for the Nyquist frequency in the\n    middle of the axes and the negative frequency terms in the second half of\n    the axes, in order of decreasingly negative frequency.\n\n    See `fftn` for details and a plotting example, and `numpy.fft` for\n    definitions and conventions used.\n\n\n    Examples\n    --------\n    >>> a = np.mgrid[:5, :5][0]\n    >>> np.fft.fft2(a)\n    array([[ 50.0 +0.j        ,   0.0 +0.j        ,   0.0 +0.j        ,\n              0.0 +0.j        ,   0.0 +0.j        ],\n           [-12.5+17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,\n              0.0 +0.j        ,   0.0 +0.j        ],\n           [-12.5 +4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,\n              0.0 +0.j        ,   0.0 +0.j        ],\n           [-12.5 -4.0614962j ,   0.0 +0.j        ,   0.0 +0.j        ,\n                0.0 +0.j        ,   0.0 +0.j        ],\n           [-12.5-17.20477401j,   0.0 +0.j        ,   0.0 +0.j        ,\n              0.0 +0.j        ,   0.0 +0.j        ]])\n\n    ')
    
    # Call to _raw_fftnd(...): (line 905)
    # Processing the call arguments (line 905)
    # Getting the type of 'a' (line 905)
    a_100655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 22), 'a', False)
    # Getting the type of 's' (line 905)
    s_100656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 25), 's', False)
    # Getting the type of 'axes' (line 905)
    axes_100657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 28), 'axes', False)
    # Getting the type of 'fft' (line 905)
    fft_100658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 34), 'fft', False)
    # Getting the type of 'norm' (line 905)
    norm_100659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 39), 'norm', False)
    # Processing the call keyword arguments (line 905)
    kwargs_100660 = {}
    # Getting the type of '_raw_fftnd' (line 905)
    _raw_fftnd_100654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 11), '_raw_fftnd', False)
    # Calling _raw_fftnd(args, kwargs) (line 905)
    _raw_fftnd_call_result_100661 = invoke(stypy.reporting.localization.Localization(__file__, 905, 11), _raw_fftnd_100654, *[a_100655, s_100656, axes_100657, fft_100658, norm_100659], **kwargs_100660)
    
    # Assigning a type to the variable 'stypy_return_type' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'stypy_return_type', _raw_fftnd_call_result_100661)
    
    # ################# End of 'fft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fft2' in the type store
    # Getting the type of 'stypy_return_type' (line 819)
    stypy_return_type_100662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100662)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fft2'
    return stypy_return_type_100662

# Assigning a type to the variable 'fft2' (line 819)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'fft2', fft2)

@norecursion
def ifft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 908)
    None_100663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 15), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 908)
    tuple_100664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 908)
    # Adding element type (line 908)
    int_100665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 27), tuple_100664, int_100665)
    # Adding element type (line 908)
    int_100666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 908, 27), tuple_100664, int_100666)
    
    # Getting the type of 'None' (line 908)
    None_100667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 41), 'None')
    defaults = [None_100663, tuple_100664, None_100667]
    # Create a new context for function 'ifft2'
    module_type_store = module_type_store.open_function_context('ifft2', 908, 0, False)
    
    # Passed parameters checking function
    ifft2.stypy_localization = localization
    ifft2.stypy_type_of_self = None
    ifft2.stypy_type_store = module_type_store
    ifft2.stypy_function_name = 'ifft2'
    ifft2.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    ifft2.stypy_varargs_param_name = None
    ifft2.stypy_kwargs_param_name = None
    ifft2.stypy_call_defaults = defaults
    ifft2.stypy_call_varargs = varargs
    ifft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ifft2', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ifft2', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ifft2(...)' code ##################

    str_100668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, (-1)), 'str', '\n    Compute the 2-dimensional inverse discrete Fourier Transform.\n\n    This function computes the inverse of the 2-dimensional discrete Fourier\n    Transform over any number of axes in an M-dimensional array by means of\n    the Fast Fourier Transform (FFT).  In other words, ``ifft2(fft2(a)) == a``\n    to within numerical accuracy.  By default, the inverse transform is\n    computed over the last two axes of the input array.\n\n    The input, analogously to `ifft`, should be ordered in the same way as is\n    returned by `fft2`, i.e. it should have the term for zero frequency\n    in the low-order corner of the two axes, the positive frequency terms in\n    the first half of these axes, the term for the Nyquist frequency in the\n    middle of the axes and the negative frequency terms in the second half of\n    both axes, in order of decreasingly negative frequency.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, can be complex.\n    s : sequence of ints, optional\n        Shape (length of each axis) of the output (``s[0]`` refers to axis 0,\n        ``s[1]`` to axis 1, etc.).  This corresponds to `n` for ``ifft(x, n)``.\n        Along each axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.  See notes for issue on `ifft` zero padding.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last two\n        axes are used.  A repeated index in `axes` means the transform over\n        that axis is performed multiple times.  A one-element sequence means\n        that a one-dimensional FFT is performed.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or the last two axes if `axes` is not given.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length, or `axes` not given and\n        ``len(s) != 2``.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    numpy.fft : Overall view of discrete Fourier transforms, with definitions\n         and conventions used.\n    fft2 : The forward 2-dimensional FFT, of which `ifft2` is the inverse.\n    ifftn : The inverse of the *n*-dimensional FFT.\n    fft : The one-dimensional FFT.\n    ifft : The one-dimensional inverse FFT.\n\n    Notes\n    -----\n    `ifft2` is just `ifftn` with a different default for `axes`.\n\n    See `ifftn` for details and a plotting example, and `numpy.fft` for\n    definition and conventions used.\n\n    Zero-padding, analogously with `ifft`, is performed by appending zeros to\n    the input along the specified dimension.  Although this is the common\n    approach, it might lead to surprising results.  If another form of zero\n    padding is desired, it must be performed before `ifft2` is called.\n\n    Examples\n    --------\n    >>> a = 4 * np.eye(4)\n    >>> np.fft.ifft2(a)\n    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],\n           [ 0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j],\n           [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j],\n           [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]])\n\n    ')
    
    # Call to _raw_fftnd(...): (line 991)
    # Processing the call arguments (line 991)
    # Getting the type of 'a' (line 991)
    a_100670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 22), 'a', False)
    # Getting the type of 's' (line 991)
    s_100671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 25), 's', False)
    # Getting the type of 'axes' (line 991)
    axes_100672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 28), 'axes', False)
    # Getting the type of 'ifft' (line 991)
    ifft_100673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 34), 'ifft', False)
    # Getting the type of 'norm' (line 991)
    norm_100674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 40), 'norm', False)
    # Processing the call keyword arguments (line 991)
    kwargs_100675 = {}
    # Getting the type of '_raw_fftnd' (line 991)
    _raw_fftnd_100669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 11), '_raw_fftnd', False)
    # Calling _raw_fftnd(args, kwargs) (line 991)
    _raw_fftnd_call_result_100676 = invoke(stypy.reporting.localization.Localization(__file__, 991, 11), _raw_fftnd_100669, *[a_100670, s_100671, axes_100672, ifft_100673, norm_100674], **kwargs_100675)
    
    # Assigning a type to the variable 'stypy_return_type' (line 991)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 4), 'stypy_return_type', _raw_fftnd_call_result_100676)
    
    # ################# End of 'ifft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ifft2' in the type store
    # Getting the type of 'stypy_return_type' (line 908)
    stypy_return_type_100677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100677)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ifft2'
    return stypy_return_type_100677

# Assigning a type to the variable 'ifft2' (line 908)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 0), 'ifft2', ifft2)

@norecursion
def rfftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 994)
    None_100678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 15), 'None')
    # Getting the type of 'None' (line 994)
    None_100679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 26), 'None')
    # Getting the type of 'None' (line 994)
    None_100680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 37), 'None')
    defaults = [None_100678, None_100679, None_100680]
    # Create a new context for function 'rfftn'
    module_type_store = module_type_store.open_function_context('rfftn', 994, 0, False)
    
    # Passed parameters checking function
    rfftn.stypy_localization = localization
    rfftn.stypy_type_of_self = None
    rfftn.stypy_type_store = module_type_store
    rfftn.stypy_function_name = 'rfftn'
    rfftn.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    rfftn.stypy_varargs_param_name = None
    rfftn.stypy_kwargs_param_name = None
    rfftn.stypy_call_defaults = defaults
    rfftn.stypy_call_varargs = varargs
    rfftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfftn', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfftn', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfftn(...)' code ##################

    str_100681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, (-1)), 'str', '\n    Compute the N-dimensional discrete Fourier Transform for real input.\n\n    This function computes the N-dimensional discrete Fourier Transform over\n    any number of axes in an M-dimensional real array by means of the Fast\n    Fourier Transform (FFT).  By default, all axes are transformed, with the\n    real transform performed over the last axis, while the remaining\n    transforms are complex.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array, taken to be real.\n    s : sequence of ints, optional\n        Shape (length along each transformed axis) to use from the input.\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.).\n        The final element of `s` corresponds to `n` for ``rfft(x, n)``, while\n        for the remaining axes, it corresponds to `n` for ``fft(x, n)``.\n        Along any axis, if the given shape is smaller than that of the input,\n        the input is cropped.  If it is larger, the input is padded with zeros.\n        if `s` is not given, the shape of the input along the axes specified\n        by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.  If not given, the last ``len(s)``\n        axes are used, or all axes if `s` is also not specified.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : complex ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` and `a`,\n        as explained in the parameters section above.\n        The length of the last axis transformed will be ``s[-1]//2+1``,\n        while the remaining transformed axes will have lengths according to\n        `s`, or unchanged from the input.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    irfftn : The inverse of `rfftn`, i.e. the inverse of the n-dimensional FFT\n         of real input.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    rfft : The one-dimensional FFT of real input.\n    fftn : The n-dimensional FFT.\n    rfft2 : The two-dimensional FFT of real input.\n\n    Notes\n    -----\n    The transform for real input is performed over the last transformation\n    axis, as by `rfft`, then the transform over the remaining axes is\n    performed as by `fftn`.  The order of the output is as for `rfft` for the\n    final transformation axis, and as for `fftn` for the remaining\n    transformation axes.\n\n    See `fft` for details, definitions and conventions used.\n\n    Examples\n    --------\n    >>> a = np.ones((2, 2, 2))\n    >>> np.fft.rfftn(a)\n    array([[[ 8.+0.j,  0.+0.j],\n            [ 0.+0.j,  0.+0.j]],\n           [[ 0.+0.j,  0.+0.j],\n            [ 0.+0.j,  0.+0.j]]])\n\n    >>> np.fft.rfftn(a, axes=(2, 0))\n    array([[[ 4.+0.j,  0.+0.j],\n            [ 4.+0.j,  0.+0.j]],\n           [[ 0.+0.j,  0.+0.j],\n            [ 0.+0.j,  0.+0.j]]])\n\n    ')
    
    # Assigning a Call to a Name (line 1077):
    
    # Assigning a Call to a Name (line 1077):
    
    # Call to array(...): (line 1077)
    # Processing the call arguments (line 1077)
    # Getting the type of 'a' (line 1077)
    a_100683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 14), 'a', False)
    # Processing the call keyword arguments (line 1077)
    # Getting the type of 'True' (line 1077)
    True_100684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 22), 'True', False)
    keyword_100685 = True_100684
    # Getting the type of 'float' (line 1077)
    float_100686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 34), 'float', False)
    keyword_100687 = float_100686
    kwargs_100688 = {'dtype': keyword_100687, 'copy': keyword_100685}
    # Getting the type of 'array' (line 1077)
    array_100682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1077, 8), 'array', False)
    # Calling array(args, kwargs) (line 1077)
    array_call_result_100689 = invoke(stypy.reporting.localization.Localization(__file__, 1077, 8), array_100682, *[a_100683], **kwargs_100688)
    
    # Assigning a type to the variable 'a' (line 1077)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1077, 4), 'a', array_call_result_100689)
    
    # Assigning a Call to a Tuple (line 1078):
    
    # Assigning a Call to a Name:
    
    # Call to _cook_nd_args(...): (line 1078)
    # Processing the call arguments (line 1078)
    # Getting the type of 'a' (line 1078)
    a_100691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 28), 'a', False)
    # Getting the type of 's' (line 1078)
    s_100692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 31), 's', False)
    # Getting the type of 'axes' (line 1078)
    axes_100693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 34), 'axes', False)
    # Processing the call keyword arguments (line 1078)
    kwargs_100694 = {}
    # Getting the type of '_cook_nd_args' (line 1078)
    _cook_nd_args_100690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 14), '_cook_nd_args', False)
    # Calling _cook_nd_args(args, kwargs) (line 1078)
    _cook_nd_args_call_result_100695 = invoke(stypy.reporting.localization.Localization(__file__, 1078, 14), _cook_nd_args_100690, *[a_100691, s_100692, axes_100693], **kwargs_100694)
    
    # Assigning a type to the variable 'call_assignment_100001' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100001', _cook_nd_args_call_result_100695)
    
    # Assigning a Call to a Name (line 1078):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100699 = {}
    # Getting the type of 'call_assignment_100001' (line 1078)
    call_assignment_100001_100696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100001', False)
    # Obtaining the member '__getitem__' of a type (line 1078)
    getitem___100697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 4), call_assignment_100001_100696, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100700 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100697, *[int_100698], **kwargs_100699)
    
    # Assigning a type to the variable 'call_assignment_100002' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100002', getitem___call_result_100700)
    
    # Assigning a Name to a Name (line 1078):
    # Getting the type of 'call_assignment_100002' (line 1078)
    call_assignment_100002_100701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100002')
    # Assigning a type to the variable 's' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 's', call_assignment_100002_100701)
    
    # Assigning a Call to a Name (line 1078):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1078, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100705 = {}
    # Getting the type of 'call_assignment_100001' (line 1078)
    call_assignment_100001_100702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100001', False)
    # Obtaining the member '__getitem__' of a type (line 1078)
    getitem___100703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1078, 4), call_assignment_100001_100702, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100706 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100703, *[int_100704], **kwargs_100705)
    
    # Assigning a type to the variable 'call_assignment_100003' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100003', getitem___call_result_100706)
    
    # Assigning a Name to a Name (line 1078):
    # Getting the type of 'call_assignment_100003' (line 1078)
    call_assignment_100003_100707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1078, 4), 'call_assignment_100003')
    # Assigning a type to the variable 'axes' (line 1078)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1078, 7), 'axes', call_assignment_100003_100707)
    
    # Assigning a Call to a Name (line 1079):
    
    # Assigning a Call to a Name (line 1079):
    
    # Call to rfft(...): (line 1079)
    # Processing the call arguments (line 1079)
    # Getting the type of 'a' (line 1079)
    a_100709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 13), 'a', False)
    
    # Obtaining the type of the subscript
    int_100710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 18), 'int')
    # Getting the type of 's' (line 1079)
    s_100711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 16), 's', False)
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___100712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 16), s_100711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_100713 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 16), getitem___100712, int_100710)
    
    
    # Obtaining the type of the subscript
    int_100714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1079, 28), 'int')
    # Getting the type of 'axes' (line 1079)
    axes_100715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 23), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 1079)
    getitem___100716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1079, 23), axes_100715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1079)
    subscript_call_result_100717 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 23), getitem___100716, int_100714)
    
    # Getting the type of 'norm' (line 1079)
    norm_100718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 33), 'norm', False)
    # Processing the call keyword arguments (line 1079)
    kwargs_100719 = {}
    # Getting the type of 'rfft' (line 1079)
    rfft_100708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 8), 'rfft', False)
    # Calling rfft(args, kwargs) (line 1079)
    rfft_call_result_100720 = invoke(stypy.reporting.localization.Localization(__file__, 1079, 8), rfft_100708, *[a_100709, subscript_call_result_100713, subscript_call_result_100717, norm_100718], **kwargs_100719)
    
    # Assigning a type to the variable 'a' (line 1079)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 4), 'a', rfft_call_result_100720)
    
    
    # Call to range(...): (line 1080)
    # Processing the call arguments (line 1080)
    
    # Call to len(...): (line 1080)
    # Processing the call arguments (line 1080)
    # Getting the type of 'axes' (line 1080)
    axes_100723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 24), 'axes', False)
    # Processing the call keyword arguments (line 1080)
    kwargs_100724 = {}
    # Getting the type of 'len' (line 1080)
    len_100722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 20), 'len', False)
    # Calling len(args, kwargs) (line 1080)
    len_call_result_100725 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 20), len_100722, *[axes_100723], **kwargs_100724)
    
    int_100726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, 30), 'int')
    # Applying the binary operator '-' (line 1080)
    result_sub_100727 = python_operator(stypy.reporting.localization.Localization(__file__, 1080, 20), '-', len_call_result_100725, int_100726)
    
    # Processing the call keyword arguments (line 1080)
    kwargs_100728 = {}
    # Getting the type of 'range' (line 1080)
    range_100721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1080, 14), 'range', False)
    # Calling range(args, kwargs) (line 1080)
    range_call_result_100729 = invoke(stypy.reporting.localization.Localization(__file__, 1080, 14), range_100721, *[result_sub_100727], **kwargs_100728)
    
    # Testing the type of a for loop iterable (line 1080)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1080, 4), range_call_result_100729)
    # Getting the type of the for loop variable (line 1080)
    for_loop_var_100730 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1080, 4), range_call_result_100729)
    # Assigning a type to the variable 'ii' (line 1080)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1080, 4), 'ii', for_loop_var_100730)
    # SSA begins for a for statement (line 1080)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1081):
    
    # Assigning a Call to a Name (line 1081):
    
    # Call to fft(...): (line 1081)
    # Processing the call arguments (line 1081)
    # Getting the type of 'a' (line 1081)
    a_100732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 16), 'a', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1081)
    ii_100733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 21), 'ii', False)
    # Getting the type of 's' (line 1081)
    s_100734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 19), 's', False)
    # Obtaining the member '__getitem__' of a type (line 1081)
    getitem___100735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 19), s_100734, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
    subscript_call_result_100736 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 19), getitem___100735, ii_100733)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1081)
    ii_100737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 31), 'ii', False)
    # Getting the type of 'axes' (line 1081)
    axes_100738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 26), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 1081)
    getitem___100739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1081, 26), axes_100738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1081)
    subscript_call_result_100740 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 26), getitem___100739, ii_100737)
    
    # Getting the type of 'norm' (line 1081)
    norm_100741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 36), 'norm', False)
    # Processing the call keyword arguments (line 1081)
    kwargs_100742 = {}
    # Getting the type of 'fft' (line 1081)
    fft_100731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1081, 12), 'fft', False)
    # Calling fft(args, kwargs) (line 1081)
    fft_call_result_100743 = invoke(stypy.reporting.localization.Localization(__file__, 1081, 12), fft_100731, *[a_100732, subscript_call_result_100736, subscript_call_result_100740, norm_100741], **kwargs_100742)
    
    # Assigning a type to the variable 'a' (line 1081)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1081, 8), 'a', fft_call_result_100743)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 1082)
    a_100744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 1082)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1082, 4), 'stypy_return_type', a_100744)
    
    # ################# End of 'rfftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfftn' in the type store
    # Getting the type of 'stypy_return_type' (line 994)
    stypy_return_type_100745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100745)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfftn'
    return stypy_return_type_100745

# Assigning a type to the variable 'rfftn' (line 994)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 994, 0), 'rfftn', rfftn)

@norecursion
def rfft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1085)
    None_100746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 15), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1085)
    tuple_100747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1085)
    # Adding element type (line 1085)
    int_100748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1085, 27), tuple_100747, int_100748)
    # Adding element type (line 1085)
    int_100749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1085, 27), tuple_100747, int_100749)
    
    # Getting the type of 'None' (line 1085)
    None_100750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 41), 'None')
    defaults = [None_100746, tuple_100747, None_100750]
    # Create a new context for function 'rfft2'
    module_type_store = module_type_store.open_function_context('rfft2', 1085, 0, False)
    
    # Passed parameters checking function
    rfft2.stypy_localization = localization
    rfft2.stypy_type_of_self = None
    rfft2.stypy_type_store = module_type_store
    rfft2.stypy_function_name = 'rfft2'
    rfft2.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    rfft2.stypy_varargs_param_name = None
    rfft2.stypy_kwargs_param_name = None
    rfft2.stypy_call_defaults = defaults
    rfft2.stypy_call_varargs = varargs
    rfft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rfft2', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rfft2', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rfft2(...)' code ##################

    str_100751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, (-1)), 'str', '\n    Compute the 2-dimensional FFT of a real array.\n\n    Parameters\n    ----------\n    a : array\n        Input array, taken to be real.\n    s : sequence of ints, optional\n        Shape of the FFT.\n    axes : sequence of ints, optional\n        Axes over which to compute the FFT.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : ndarray\n        The result of the real 2-D FFT.\n\n    See Also\n    --------\n    rfftn : Compute the N-dimensional discrete Fourier Transform for real\n            input.\n\n    Notes\n    -----\n    This is really just `rfftn` with different default behavior.\n    For more details see `rfftn`.\n\n    ')
    
    # Call to rfftn(...): (line 1118)
    # Processing the call arguments (line 1118)
    # Getting the type of 'a' (line 1118)
    a_100753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 17), 'a', False)
    # Getting the type of 's' (line 1118)
    s_100754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 20), 's', False)
    # Getting the type of 'axes' (line 1118)
    axes_100755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 23), 'axes', False)
    # Getting the type of 'norm' (line 1118)
    norm_100756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 29), 'norm', False)
    # Processing the call keyword arguments (line 1118)
    kwargs_100757 = {}
    # Getting the type of 'rfftn' (line 1118)
    rfftn_100752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 11), 'rfftn', False)
    # Calling rfftn(args, kwargs) (line 1118)
    rfftn_call_result_100758 = invoke(stypy.reporting.localization.Localization(__file__, 1118, 11), rfftn_100752, *[a_100753, s_100754, axes_100755, norm_100756], **kwargs_100757)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 4), 'stypy_return_type', rfftn_call_result_100758)
    
    # ################# End of 'rfft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rfft2' in the type store
    # Getting the type of 'stypy_return_type' (line 1085)
    stypy_return_type_100759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100759)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rfft2'
    return stypy_return_type_100759

# Assigning a type to the variable 'rfft2' (line 1085)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 0), 'rfft2', rfft2)

@norecursion
def irfftn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1121)
    None_100760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 16), 'None')
    # Getting the type of 'None' (line 1121)
    None_100761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 27), 'None')
    # Getting the type of 'None' (line 1121)
    None_100762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 38), 'None')
    defaults = [None_100760, None_100761, None_100762]
    # Create a new context for function 'irfftn'
    module_type_store = module_type_store.open_function_context('irfftn', 1121, 0, False)
    
    # Passed parameters checking function
    irfftn.stypy_localization = localization
    irfftn.stypy_type_of_self = None
    irfftn.stypy_type_store = module_type_store
    irfftn.stypy_function_name = 'irfftn'
    irfftn.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    irfftn.stypy_varargs_param_name = None
    irfftn.stypy_kwargs_param_name = None
    irfftn.stypy_call_defaults = defaults
    irfftn.stypy_call_varargs = varargs
    irfftn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'irfftn', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'irfftn', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'irfftn(...)' code ##################

    str_100763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1204, (-1)), 'str', '\n    Compute the inverse of the N-dimensional FFT of real input.\n\n    This function computes the inverse of the N-dimensional discrete\n    Fourier Transform for real input over any number of axes in an\n    M-dimensional array by means of the Fast Fourier Transform (FFT).  In\n    other words, ``irfftn(rfftn(a), a.shape) == a`` to within numerical\n    accuracy. (The ``a.shape`` is necessary like ``len(a)`` is for `irfft`,\n    and for the same reason.)\n\n    The input should be ordered in the same way as is returned by `rfftn`,\n    i.e. as for `irfft` for the final transformation axis, and as for `ifftn`\n    along all the other axes.\n\n    Parameters\n    ----------\n    a : array_like\n        Input array.\n    s : sequence of ints, optional\n        Shape (length of each transformed axis) of the output\n        (``s[0]`` refers to axis 0, ``s[1]`` to axis 1, etc.). `s` is also the\n        number of input points used along this axis, except for the last axis,\n        where ``s[-1]//2+1`` points of the input are used.\n        Along any axis, if the shape indicated by `s` is smaller than that of\n        the input, the input is cropped.  If it is larger, the input is padded\n        with zeros. If `s` is not given, the shape of the input along the\n        axes specified by `axes` is used.\n    axes : sequence of ints, optional\n        Axes over which to compute the inverse FFT. If not given, the last\n        `len(s)` axes are used, or all axes if `s` is also not specified.\n        Repeated indices in `axes` means that the inverse transform over that\n        axis is performed multiple times.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : ndarray\n        The truncated or zero-padded input, transformed along the axes\n        indicated by `axes`, or by a combination of `s` or `a`,\n        as explained in the parameters section above.\n        The length of each transformed axis is as given by the corresponding\n        element of `s`, or the length of the input in every axis except for the\n        last one if `s` is not given.  In the final transformed axis the length\n        of the output when `s` is not given is ``2*(m-1)`` where ``m`` is the\n        length of the final transformed axis of the input.  To get an odd\n        number of output points in the final axis, `s` must be specified.\n\n    Raises\n    ------\n    ValueError\n        If `s` and `axes` have different length.\n    IndexError\n        If an element of `axes` is larger than than the number of axes of `a`.\n\n    See Also\n    --------\n    rfftn : The forward n-dimensional FFT of real input,\n            of which `ifftn` is the inverse.\n    fft : The one-dimensional FFT, with definitions and conventions used.\n    irfft : The inverse of the one-dimensional FFT of real input.\n    irfft2 : The inverse of the two-dimensional FFT of real input.\n\n    Notes\n    -----\n    See `fft` for definitions and conventions used.\n\n    See `rfft` for definitions and conventions used for real input.\n\n    Examples\n    --------\n    >>> a = np.zeros((3, 2, 2))\n    >>> a[0, 0, 0] = 3 * 2 * 2\n    >>> np.fft.irfftn(a)\n    array([[[ 1.,  1.],\n            [ 1.,  1.]],\n           [[ 1.,  1.],\n            [ 1.,  1.]],\n           [[ 1.,  1.],\n            [ 1.,  1.]]])\n\n    ')
    
    # Assigning a Call to a Name (line 1206):
    
    # Assigning a Call to a Name (line 1206):
    
    # Call to array(...): (line 1206)
    # Processing the call arguments (line 1206)
    # Getting the type of 'a' (line 1206)
    a_100765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 14), 'a', False)
    # Processing the call keyword arguments (line 1206)
    # Getting the type of 'True' (line 1206)
    True_100766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 22), 'True', False)
    keyword_100767 = True_100766
    # Getting the type of 'complex' (line 1206)
    complex_100768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 34), 'complex', False)
    keyword_100769 = complex_100768
    kwargs_100770 = {'dtype': keyword_100769, 'copy': keyword_100767}
    # Getting the type of 'array' (line 1206)
    array_100764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 8), 'array', False)
    # Calling array(args, kwargs) (line 1206)
    array_call_result_100771 = invoke(stypy.reporting.localization.Localization(__file__, 1206, 8), array_100764, *[a_100765], **kwargs_100770)
    
    # Assigning a type to the variable 'a' (line 1206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1206, 4), 'a', array_call_result_100771)
    
    # Assigning a Call to a Tuple (line 1207):
    
    # Assigning a Call to a Name:
    
    # Call to _cook_nd_args(...): (line 1207)
    # Processing the call arguments (line 1207)
    # Getting the type of 'a' (line 1207)
    a_100773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 28), 'a', False)
    # Getting the type of 's' (line 1207)
    s_100774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 31), 's', False)
    # Getting the type of 'axes' (line 1207)
    axes_100775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 34), 'axes', False)
    # Processing the call keyword arguments (line 1207)
    int_100776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 48), 'int')
    keyword_100777 = int_100776
    kwargs_100778 = {'invreal': keyword_100777}
    # Getting the type of '_cook_nd_args' (line 1207)
    _cook_nd_args_100772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 14), '_cook_nd_args', False)
    # Calling _cook_nd_args(args, kwargs) (line 1207)
    _cook_nd_args_call_result_100779 = invoke(stypy.reporting.localization.Localization(__file__, 1207, 14), _cook_nd_args_100772, *[a_100773, s_100774, axes_100775], **kwargs_100778)
    
    # Assigning a type to the variable 'call_assignment_100004' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100004', _cook_nd_args_call_result_100779)
    
    # Assigning a Call to a Name (line 1207):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100783 = {}
    # Getting the type of 'call_assignment_100004' (line 1207)
    call_assignment_100004_100780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100004', False)
    # Obtaining the member '__getitem__' of a type (line 1207)
    getitem___100781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1207, 4), call_assignment_100004_100780, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100784 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100781, *[int_100782], **kwargs_100783)
    
    # Assigning a type to the variable 'call_assignment_100005' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100005', getitem___call_result_100784)
    
    # Assigning a Name to a Name (line 1207):
    # Getting the type of 'call_assignment_100005' (line 1207)
    call_assignment_100005_100785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100005')
    # Assigning a type to the variable 's' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 's', call_assignment_100005_100785)
    
    # Assigning a Call to a Name (line 1207):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_100788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 4), 'int')
    # Processing the call keyword arguments
    kwargs_100789 = {}
    # Getting the type of 'call_assignment_100004' (line 1207)
    call_assignment_100004_100786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100004', False)
    # Obtaining the member '__getitem__' of a type (line 1207)
    getitem___100787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1207, 4), call_assignment_100004_100786, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_100790 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___100787, *[int_100788], **kwargs_100789)
    
    # Assigning a type to the variable 'call_assignment_100006' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100006', getitem___call_result_100790)
    
    # Assigning a Name to a Name (line 1207):
    # Getting the type of 'call_assignment_100006' (line 1207)
    call_assignment_100006_100791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'call_assignment_100006')
    # Assigning a type to the variable 'axes' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 7), 'axes', call_assignment_100006_100791)
    
    
    # Call to range(...): (line 1208)
    # Processing the call arguments (line 1208)
    
    # Call to len(...): (line 1208)
    # Processing the call arguments (line 1208)
    # Getting the type of 'axes' (line 1208)
    axes_100794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 24), 'axes', False)
    # Processing the call keyword arguments (line 1208)
    kwargs_100795 = {}
    # Getting the type of 'len' (line 1208)
    len_100793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 20), 'len', False)
    # Calling len(args, kwargs) (line 1208)
    len_call_result_100796 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 20), len_100793, *[axes_100794], **kwargs_100795)
    
    int_100797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1208, 30), 'int')
    # Applying the binary operator '-' (line 1208)
    result_sub_100798 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 20), '-', len_call_result_100796, int_100797)
    
    # Processing the call keyword arguments (line 1208)
    kwargs_100799 = {}
    # Getting the type of 'range' (line 1208)
    range_100792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 14), 'range', False)
    # Calling range(args, kwargs) (line 1208)
    range_call_result_100800 = invoke(stypy.reporting.localization.Localization(__file__, 1208, 14), range_100792, *[result_sub_100798], **kwargs_100799)
    
    # Testing the type of a for loop iterable (line 1208)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1208, 4), range_call_result_100800)
    # Getting the type of the for loop variable (line 1208)
    for_loop_var_100801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1208, 4), range_call_result_100800)
    # Assigning a type to the variable 'ii' (line 1208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 4), 'ii', for_loop_var_100801)
    # SSA begins for a for statement (line 1208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1209):
    
    # Assigning a Call to a Name (line 1209):
    
    # Call to ifft(...): (line 1209)
    # Processing the call arguments (line 1209)
    # Getting the type of 'a' (line 1209)
    a_100803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 17), 'a', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1209)
    ii_100804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 22), 'ii', False)
    # Getting the type of 's' (line 1209)
    s_100805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 20), 's', False)
    # Obtaining the member '__getitem__' of a type (line 1209)
    getitem___100806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1209, 20), s_100805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1209)
    subscript_call_result_100807 = invoke(stypy.reporting.localization.Localization(__file__, 1209, 20), getitem___100806, ii_100804)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 1209)
    ii_100808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 32), 'ii', False)
    # Getting the type of 'axes' (line 1209)
    axes_100809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 27), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 1209)
    getitem___100810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1209, 27), axes_100809, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1209)
    subscript_call_result_100811 = invoke(stypy.reporting.localization.Localization(__file__, 1209, 27), getitem___100810, ii_100808)
    
    # Getting the type of 'norm' (line 1209)
    norm_100812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 37), 'norm', False)
    # Processing the call keyword arguments (line 1209)
    kwargs_100813 = {}
    # Getting the type of 'ifft' (line 1209)
    ifft_100802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 12), 'ifft', False)
    # Calling ifft(args, kwargs) (line 1209)
    ifft_call_result_100814 = invoke(stypy.reporting.localization.Localization(__file__, 1209, 12), ifft_100802, *[a_100803, subscript_call_result_100807, subscript_call_result_100811, norm_100812], **kwargs_100813)
    
    # Assigning a type to the variable 'a' (line 1209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 8), 'a', ifft_call_result_100814)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1210):
    
    # Assigning a Call to a Name (line 1210):
    
    # Call to irfft(...): (line 1210)
    # Processing the call arguments (line 1210)
    # Getting the type of 'a' (line 1210)
    a_100816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 14), 'a', False)
    
    # Obtaining the type of the subscript
    int_100817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 19), 'int')
    # Getting the type of 's' (line 1210)
    s_100818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 17), 's', False)
    # Obtaining the member '__getitem__' of a type (line 1210)
    getitem___100819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1210, 17), s_100818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1210)
    subscript_call_result_100820 = invoke(stypy.reporting.localization.Localization(__file__, 1210, 17), getitem___100819, int_100817)
    
    
    # Obtaining the type of the subscript
    int_100821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, 29), 'int')
    # Getting the type of 'axes' (line 1210)
    axes_100822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 24), 'axes', False)
    # Obtaining the member '__getitem__' of a type (line 1210)
    getitem___100823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1210, 24), axes_100822, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1210)
    subscript_call_result_100824 = invoke(stypy.reporting.localization.Localization(__file__, 1210, 24), getitem___100823, int_100821)
    
    # Getting the type of 'norm' (line 1210)
    norm_100825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 34), 'norm', False)
    # Processing the call keyword arguments (line 1210)
    kwargs_100826 = {}
    # Getting the type of 'irfft' (line 1210)
    irfft_100815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 8), 'irfft', False)
    # Calling irfft(args, kwargs) (line 1210)
    irfft_call_result_100827 = invoke(stypy.reporting.localization.Localization(__file__, 1210, 8), irfft_100815, *[a_100816, subscript_call_result_100820, subscript_call_result_100824, norm_100825], **kwargs_100826)
    
    # Assigning a type to the variable 'a' (line 1210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 4), 'a', irfft_call_result_100827)
    # Getting the type of 'a' (line 1211)
    a_100828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 4), 'stypy_return_type', a_100828)
    
    # ################# End of 'irfftn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'irfftn' in the type store
    # Getting the type of 'stypy_return_type' (line 1121)
    stypy_return_type_100829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'irfftn'
    return stypy_return_type_100829

# Assigning a type to the variable 'irfftn' (line 1121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 0), 'irfftn', irfftn)

@norecursion
def irfft2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1214)
    None_100830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 16), 'None')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1214)
    tuple_100831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1214)
    # Adding element type (line 1214)
    int_100832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 28), tuple_100831, int_100832)
    # Adding element type (line 1214)
    int_100833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 28), tuple_100831, int_100833)
    
    # Getting the type of 'None' (line 1214)
    None_100834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 42), 'None')
    defaults = [None_100830, tuple_100831, None_100834]
    # Create a new context for function 'irfft2'
    module_type_store = module_type_store.open_function_context('irfft2', 1214, 0, False)
    
    # Passed parameters checking function
    irfft2.stypy_localization = localization
    irfft2.stypy_type_of_self = None
    irfft2.stypy_type_store = module_type_store
    irfft2.stypy_function_name = 'irfft2'
    irfft2.stypy_param_names_list = ['a', 's', 'axes', 'norm']
    irfft2.stypy_varargs_param_name = None
    irfft2.stypy_kwargs_param_name = None
    irfft2.stypy_call_defaults = defaults
    irfft2.stypy_call_varargs = varargs
    irfft2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'irfft2', ['a', 's', 'axes', 'norm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'irfft2', localization, ['a', 's', 'axes', 'norm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'irfft2(...)' code ##################

    str_100835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1245, (-1)), 'str', '\n    Compute the 2-dimensional inverse FFT of a real array.\n\n    Parameters\n    ----------\n    a : array_like\n        The input array\n    s : sequence of ints, optional\n        Shape of the inverse FFT.\n    axes : sequence of ints, optional\n        The axes over which to compute the inverse fft.\n        Default is the last two axes.\n    norm : {None, "ortho"}, optional\n        .. versionadded:: 1.10.0\n        Normalization mode (see `numpy.fft`). Default is None.\n\n    Returns\n    -------\n    out : ndarray\n        The result of the inverse real 2-D FFT.\n\n    See Also\n    --------\n    irfftn : Compute the inverse of the N-dimensional FFT of real input.\n\n    Notes\n    -----\n    This is really `irfftn` with different defaults.\n    For more details see `irfftn`.\n\n    ')
    
    # Call to irfftn(...): (line 1247)
    # Processing the call arguments (line 1247)
    # Getting the type of 'a' (line 1247)
    a_100837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 18), 'a', False)
    # Getting the type of 's' (line 1247)
    s_100838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 21), 's', False)
    # Getting the type of 'axes' (line 1247)
    axes_100839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 24), 'axes', False)
    # Getting the type of 'norm' (line 1247)
    norm_100840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 30), 'norm', False)
    # Processing the call keyword arguments (line 1247)
    kwargs_100841 = {}
    # Getting the type of 'irfftn' (line 1247)
    irfftn_100836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 11), 'irfftn', False)
    # Calling irfftn(args, kwargs) (line 1247)
    irfftn_call_result_100842 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 11), irfftn_100836, *[a_100837, s_100838, axes_100839, norm_100840], **kwargs_100841)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1247, 4), 'stypy_return_type', irfftn_call_result_100842)
    
    # ################# End of 'irfft2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'irfft2' in the type store
    # Getting the type of 'stypy_return_type' (line 1214)
    stypy_return_type_100843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'irfft2'
    return stypy_return_type_100843

# Assigning a type to the variable 'irfft2' (line 1214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 0), 'irfft2', irfft2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
