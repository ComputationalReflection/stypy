
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''The suite of window functions.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import warnings
5: 
6: import numpy as np
7: from scipy import fftpack, linalg, special
8: from scipy._lib.six import string_types
9: 
10: __all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
11:            'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann',
12:            'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'chebwin',
13:            'slepian', 'cosine', 'hann', 'exponential', 'tukey', 'get_window']
14: 
15: 
16: def _len_guards(M):
17:     '''Handle small or incorrect window lengths'''
18:     if int(M) != M or M < 0:
19:         raise ValueError('Window length M must be a non-negative integer')
20:     return M <= 1
21: 
22: 
23: def _extend(M, sym):
24:     '''Extend window by 1 sample if needed for DFT-even symmetry'''
25:     if not sym:
26:         return M + 1, True
27:     else:
28:         return M, False
29: 
30: 
31: def _truncate(w, needed):
32:     '''Truncate window by 1 sample if needed for DFT-even symmetry'''
33:     if needed:
34:         return w[:-1]
35:     else:
36:         return w
37: 
38: 
39: def _cos_win(M, a, sym=True):
40:     r'''
41:     Generic weighted sum of cosine terms window
42: 
43:     Parameters
44:     ----------
45:     M : int
46:         Number of points in the output window
47:     a : array_like
48:         Sequence of weighting coefficients. This uses the convention of being
49:         centered on the origin, so these will typically all be positive
50:         numbers, not alternating sign.
51:     sym : bool, optional
52:         When True (default), generates a symmetric window, for use in filter
53:         design.
54:         When False, generates a periodic window, for use in spectral analysis.
55: 
56:     References
57:     ----------
58:     .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
59:            Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
60:            no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
61:     .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
62:            Discrete Fourier transform (DFT), including a comprehensive list of
63:            window functions and some new flat-top windows", February 15, 2002
64:            https://holometer.fnal.gov/GH_FFT.pdf
65: 
66:     Examples
67:     --------
68:     Heinzel describes a flat-top window named "HFT90D" with formula: [2]_
69: 
70:     .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
71:                - 0.440811 \cos(3z) + 0.043097 \cos(4z)
72: 
73:     where
74: 
75:     .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1
76: 
77:     Since this uses the convention of starting at the origin, to reproduce the
78:     window, we need to convert every other coefficient to a positive number:
79: 
80:     >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]
81: 
82:     The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
83:     Figure 42 by plotting the window and its frequency response, and confirm
84:     the sidelobe level in red:
85: 
86:     >>> from scipy import signal
87:     >>> from scipy.fftpack import fft, fftshift
88:     >>> import matplotlib.pyplot as plt
89: 
90:     >>> window = signal._cos_win(1000, HFT90D, sym=False)
91:     >>> plt.plot(window)
92:     >>> plt.title("HFT90D window")
93:     >>> plt.ylabel("Amplitude")
94:     >>> plt.xlabel("Sample")
95: 
96:     >>> plt.figure()
97:     >>> A = fft(window, 10000) / (len(window)/2.0)
98:     >>> freq = np.linspace(-0.5, 0.5, len(A))
99:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
100:     >>> plt.plot(freq, response)
101:     >>> plt.axis([-50/1000, 50/1000, -140, 0])
102:     >>> plt.title("Frequency response of the HFT90D window")
103:     >>> plt.ylabel("Normalized magnitude [dB]")
104:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
105:     >>> plt.axhline(-90.2, color='red')
106: 
107:     '''
108:     if _len_guards(M):
109:         return np.ones(M)
110:     M, needs_trunc = _extend(M, sym)
111: 
112:     fac = np.linspace(-np.pi, np.pi, M)
113:     w = np.zeros(M)
114:     for k in range(len(a)):
115:         w += a[k] * np.cos(k * fac)
116: 
117:     return _truncate(w, needs_trunc)
118: 
119: 
120: def boxcar(M, sym=True):
121:     '''Return a boxcar or rectangular window.
122: 
123:     Also known as a rectangular window or Dirichlet window, this is equivalent
124:     to no window at all.
125: 
126:     Parameters
127:     ----------
128:     M : int
129:         Number of points in the output window. If zero or less, an empty
130:         array is returned.
131:     sym : bool, optional
132:         Whether the window is symmetric. (Has no effect for boxcar.)
133: 
134:     Returns
135:     -------
136:     w : ndarray
137:         The window, with the maximum value normalized to 1.
138: 
139:     Examples
140:     --------
141:     Plot the window and its frequency response:
142: 
143:     >>> from scipy import signal
144:     >>> from scipy.fftpack import fft, fftshift
145:     >>> import matplotlib.pyplot as plt
146: 
147:     >>> window = signal.boxcar(51)
148:     >>> plt.plot(window)
149:     >>> plt.title("Boxcar window")
150:     >>> plt.ylabel("Amplitude")
151:     >>> plt.xlabel("Sample")
152: 
153:     >>> plt.figure()
154:     >>> A = fft(window, 2048) / (len(window)/2.0)
155:     >>> freq = np.linspace(-0.5, 0.5, len(A))
156:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
157:     >>> plt.plot(freq, response)
158:     >>> plt.axis([-0.5, 0.5, -120, 0])
159:     >>> plt.title("Frequency response of the boxcar window")
160:     >>> plt.ylabel("Normalized magnitude [dB]")
161:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
162: 
163:     '''
164:     if _len_guards(M):
165:         return np.ones(M)
166:     M, needs_trunc = _extend(M, sym)
167: 
168:     w = np.ones(M, float)
169: 
170:     return _truncate(w, needs_trunc)
171: 
172: 
173: def triang(M, sym=True):
174:     '''Return a triangular window.
175: 
176:     Parameters
177:     ----------
178:     M : int
179:         Number of points in the output window. If zero or less, an empty
180:         array is returned.
181:     sym : bool, optional
182:         When True (default), generates a symmetric window, for use in filter
183:         design.
184:         When False, generates a periodic window, for use in spectral analysis.
185: 
186:     Returns
187:     -------
188:     w : ndarray
189:         The window, with the maximum value normalized to 1 (though the value 1
190:         does not appear if `M` is even and `sym` is True).
191: 
192:     See Also
193:     --------
194:     bartlett : A triangular window that touches zero
195: 
196:     Examples
197:     --------
198:     Plot the window and its frequency response:
199: 
200:     >>> from scipy import signal
201:     >>> from scipy.fftpack import fft, fftshift
202:     >>> import matplotlib.pyplot as plt
203: 
204:     >>> window = signal.triang(51)
205:     >>> plt.plot(window)
206:     >>> plt.title("Triangular window")
207:     >>> plt.ylabel("Amplitude")
208:     >>> plt.xlabel("Sample")
209: 
210:     >>> plt.figure()
211:     >>> A = fft(window, 2048) / (len(window)/2.0)
212:     >>> freq = np.linspace(-0.5, 0.5, len(A))
213:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
214:     >>> plt.plot(freq, response)
215:     >>> plt.axis([-0.5, 0.5, -120, 0])
216:     >>> plt.title("Frequency response of the triangular window")
217:     >>> plt.ylabel("Normalized magnitude [dB]")
218:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
219: 
220:     '''
221:     if _len_guards(M):
222:         return np.ones(M)
223:     M, needs_trunc = _extend(M, sym)
224: 
225:     n = np.arange(1, (M + 1) // 2 + 1)
226:     if M % 2 == 0:
227:         w = (2 * n - 1.0) / M
228:         w = np.r_[w, w[::-1]]
229:     else:
230:         w = 2 * n / (M + 1.0)
231:         w = np.r_[w, w[-2::-1]]
232: 
233:     return _truncate(w, needs_trunc)
234: 
235: 
236: def parzen(M, sym=True):
237:     '''Return a Parzen window.
238: 
239:     Parameters
240:     ----------
241:     M : int
242:         Number of points in the output window. If zero or less, an empty
243:         array is returned.
244:     sym : bool, optional
245:         When True (default), generates a symmetric window, for use in filter
246:         design.
247:         When False, generates a periodic window, for use in spectral analysis.
248: 
249:     Returns
250:     -------
251:     w : ndarray
252:         The window, with the maximum value normalized to 1 (though the value 1
253:         does not appear if `M` is even and `sym` is True).
254: 
255:     References
256:     ----------
257:     .. [1] E. Parzen, "Mathematical Considerations in the Estimation of
258:            Spectra", Technometrics,  Vol. 3, No. 2 (May, 1961), pp. 167-190
259: 
260:     Examples
261:     --------
262:     Plot the window and its frequency response:
263: 
264:     >>> from scipy import signal
265:     >>> from scipy.fftpack import fft, fftshift
266:     >>> import matplotlib.pyplot as plt
267: 
268:     >>> window = signal.parzen(51)
269:     >>> plt.plot(window)
270:     >>> plt.title("Parzen window")
271:     >>> plt.ylabel("Amplitude")
272:     >>> plt.xlabel("Sample")
273: 
274:     >>> plt.figure()
275:     >>> A = fft(window, 2048) / (len(window)/2.0)
276:     >>> freq = np.linspace(-0.5, 0.5, len(A))
277:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
278:     >>> plt.plot(freq, response)
279:     >>> plt.axis([-0.5, 0.5, -120, 0])
280:     >>> plt.title("Frequency response of the Parzen window")
281:     >>> plt.ylabel("Normalized magnitude [dB]")
282:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
283: 
284:     '''
285:     if _len_guards(M):
286:         return np.ones(M)
287:     M, needs_trunc = _extend(M, sym)
288: 
289:     n = np.arange(-(M - 1) / 2.0, (M - 1) / 2.0 + 0.5, 1.0)
290:     na = np.extract(n < -(M - 1) / 4.0, n)
291:     nb = np.extract(abs(n) <= (M - 1) / 4.0, n)
292:     wa = 2 * (1 - np.abs(na) / (M / 2.0)) ** 3.0
293:     wb = (1 - 6 * (np.abs(nb) / (M / 2.0)) ** 2.0 +
294:           6 * (np.abs(nb) / (M / 2.0)) ** 3.0)
295:     w = np.r_[wa, wb, wa[::-1]]
296: 
297:     return _truncate(w, needs_trunc)
298: 
299: 
300: def bohman(M, sym=True):
301:     '''Return a Bohman window.
302: 
303:     Parameters
304:     ----------
305:     M : int
306:         Number of points in the output window. If zero or less, an empty
307:         array is returned.
308:     sym : bool, optional
309:         When True (default), generates a symmetric window, for use in filter
310:         design.
311:         When False, generates a periodic window, for use in spectral analysis.
312: 
313:     Returns
314:     -------
315:     w : ndarray
316:         The window, with the maximum value normalized to 1 (though the value 1
317:         does not appear if `M` is even and `sym` is True).
318: 
319:     Examples
320:     --------
321:     Plot the window and its frequency response:
322: 
323:     >>> from scipy import signal
324:     >>> from scipy.fftpack import fft, fftshift
325:     >>> import matplotlib.pyplot as plt
326: 
327:     >>> window = signal.bohman(51)
328:     >>> plt.plot(window)
329:     >>> plt.title("Bohman window")
330:     >>> plt.ylabel("Amplitude")
331:     >>> plt.xlabel("Sample")
332: 
333:     >>> plt.figure()
334:     >>> A = fft(window, 2048) / (len(window)/2.0)
335:     >>> freq = np.linspace(-0.5, 0.5, len(A))
336:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
337:     >>> plt.plot(freq, response)
338:     >>> plt.axis([-0.5, 0.5, -120, 0])
339:     >>> plt.title("Frequency response of the Bohman window")
340:     >>> plt.ylabel("Normalized magnitude [dB]")
341:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
342: 
343:     '''
344:     if _len_guards(M):
345:         return np.ones(M)
346:     M, needs_trunc = _extend(M, sym)
347: 
348:     fac = np.abs(np.linspace(-1, 1, M)[1:-1])
349:     w = (1 - fac) * np.cos(np.pi * fac) + 1.0 / np.pi * np.sin(np.pi * fac)
350:     w = np.r_[0, w, 0]
351: 
352:     return _truncate(w, needs_trunc)
353: 
354: 
355: def blackman(M, sym=True):
356:     r'''
357:     Return a Blackman window.
358: 
359:     The Blackman window is a taper formed by using the first three terms of
360:     a summation of cosines. It was designed to have close to the minimal
361:     leakage possible.  It is close to optimal, only slightly worse than a
362:     Kaiser window.
363: 
364:     Parameters
365:     ----------
366:     M : int
367:         Number of points in the output window. If zero or less, an empty
368:         array is returned.
369:     sym : bool, optional
370:         When True (default), generates a symmetric window, for use in filter
371:         design.
372:         When False, generates a periodic window, for use in spectral analysis.
373: 
374:     Returns
375:     -------
376:     w : ndarray
377:         The window, with the maximum value normalized to 1 (though the value 1
378:         does not appear if `M` is even and `sym` is True).
379: 
380:     Notes
381:     -----
382:     The Blackman window is defined as
383: 
384:     .. math::  w(n) = 0.42 - 0.5 \cos(2\pi n/M) + 0.08 \cos(4\pi n/M)
385: 
386:     The "exact Blackman" window was designed to null out the third and fourth
387:     sidelobes, but has discontinuities at the boundaries, resulting in a
388:     6 dB/oct fall-off.  This window is an approximation of the "exact" window,
389:     which does not null the sidelobes as well, but is smooth at the edges,
390:     improving the fall-off rate to 18 dB/oct. [3]_
391: 
392:     Most references to the Blackman window come from the signal processing
393:     literature, where it is used as one of many windowing functions for
394:     smoothing values.  It is also known as an apodization (which means
395:     "removing the foot", i.e. smoothing discontinuities at the beginning
396:     and end of the sampled signal) or tapering function. It is known as a
397:     "near optimal" tapering function, almost as good (by some measures)
398:     as the Kaiser window.
399: 
400:     References
401:     ----------
402:     .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
403:            spectra, Dover Publications, New York.
404:     .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
405:            Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.
406:     .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
407:            Analysis with the Discrete Fourier Transform". Proceedings of the
408:            IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.
409: 
410:     Examples
411:     --------
412:     Plot the window and its frequency response:
413: 
414:     >>> from scipy import signal
415:     >>> from scipy.fftpack import fft, fftshift
416:     >>> import matplotlib.pyplot as plt
417: 
418:     >>> window = signal.blackman(51)
419:     >>> plt.plot(window)
420:     >>> plt.title("Blackman window")
421:     >>> plt.ylabel("Amplitude")
422:     >>> plt.xlabel("Sample")
423: 
424:     >>> plt.figure()
425:     >>> A = fft(window, 2048) / (len(window)/2.0)
426:     >>> freq = np.linspace(-0.5, 0.5, len(A))
427:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
428:     >>> plt.plot(freq, response)
429:     >>> plt.axis([-0.5, 0.5, -120, 0])
430:     >>> plt.title("Frequency response of the Blackman window")
431:     >>> plt.ylabel("Normalized magnitude [dB]")
432:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
433: 
434:     '''
435:     # Docstring adapted from NumPy's blackman function
436:     if _len_guards(M):
437:         return np.ones(M)
438:     M, needs_trunc = _extend(M, sym)
439: 
440:     w = _cos_win(M, [0.42, 0.50, 0.08])
441: 
442:     return _truncate(w, needs_trunc)
443: 
444: 
445: def nuttall(M, sym=True):
446:     '''Return a minimum 4-term Blackman-Harris window according to Nuttall.
447: 
448:     This variation is called "Nuttall4c" by Heinzel. [2]_
449: 
450:     Parameters
451:     ----------
452:     M : int
453:         Number of points in the output window. If zero or less, an empty
454:         array is returned.
455:     sym : bool, optional
456:         When True (default), generates a symmetric window, for use in filter
457:         design.
458:         When False, generates a periodic window, for use in spectral analysis.
459: 
460:     Returns
461:     -------
462:     w : ndarray
463:         The window, with the maximum value normalized to 1 (though the value 1
464:         does not appear if `M` is even and `sym` is True).
465: 
466:     References
467:     ----------
468:     .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
469:            Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
470:            no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
471:     .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
472:            Discrete Fourier transform (DFT), including a comprehensive list of
473:            window functions and some new flat-top windows", February 15, 2002
474:            https://holometer.fnal.gov/GH_FFT.pdf
475: 
476:     Examples
477:     --------
478:     Plot the window and its frequency response:
479: 
480:     >>> from scipy import signal
481:     >>> from scipy.fftpack import fft, fftshift
482:     >>> import matplotlib.pyplot as plt
483: 
484:     >>> window = signal.nuttall(51)
485:     >>> plt.plot(window)
486:     >>> plt.title("Nuttall window")
487:     >>> plt.ylabel("Amplitude")
488:     >>> plt.xlabel("Sample")
489: 
490:     >>> plt.figure()
491:     >>> A = fft(window, 2048) / (len(window)/2.0)
492:     >>> freq = np.linspace(-0.5, 0.5, len(A))
493:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
494:     >>> plt.plot(freq, response)
495:     >>> plt.axis([-0.5, 0.5, -120, 0])
496:     >>> plt.title("Frequency response of the Nuttall window")
497:     >>> plt.ylabel("Normalized magnitude [dB]")
498:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
499: 
500:     '''
501:     if _len_guards(M):
502:         return np.ones(M)
503:     M, needs_trunc = _extend(M, sym)
504: 
505:     w = _cos_win(M, [0.3635819, 0.4891775, 0.1365995, 0.0106411])
506: 
507:     return _truncate(w, needs_trunc)
508: 
509: 
510: def blackmanharris(M, sym=True):
511:     '''Return a minimum 4-term Blackman-Harris window.
512: 
513:     Parameters
514:     ----------
515:     M : int
516:         Number of points in the output window. If zero or less, an empty
517:         array is returned.
518:     sym : bool, optional
519:         When True (default), generates a symmetric window, for use in filter
520:         design.
521:         When False, generates a periodic window, for use in spectral analysis.
522: 
523:     Returns
524:     -------
525:     w : ndarray
526:         The window, with the maximum value normalized to 1 (though the value 1
527:         does not appear if `M` is even and `sym` is True).
528: 
529:     Examples
530:     --------
531:     Plot the window and its frequency response:
532: 
533:     >>> from scipy import signal
534:     >>> from scipy.fftpack import fft, fftshift
535:     >>> import matplotlib.pyplot as plt
536: 
537:     >>> window = signal.blackmanharris(51)
538:     >>> plt.plot(window)
539:     >>> plt.title("Blackman-Harris window")
540:     >>> plt.ylabel("Amplitude")
541:     >>> plt.xlabel("Sample")
542: 
543:     >>> plt.figure()
544:     >>> A = fft(window, 2048) / (len(window)/2.0)
545:     >>> freq = np.linspace(-0.5, 0.5, len(A))
546:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
547:     >>> plt.plot(freq, response)
548:     >>> plt.axis([-0.5, 0.5, -120, 0])
549:     >>> plt.title("Frequency response of the Blackman-Harris window")
550:     >>> plt.ylabel("Normalized magnitude [dB]")
551:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
552: 
553:     '''
554:     if _len_guards(M):
555:         return np.ones(M)
556:     M, needs_trunc = _extend(M, sym)
557: 
558:     w = _cos_win(M, [0.35875, 0.48829, 0.14128, 0.01168])
559: 
560:     return _truncate(w, needs_trunc)
561: 
562: 
563: def flattop(M, sym=True):
564:     '''Return a flat top window.
565: 
566:     Parameters
567:     ----------
568:     M : int
569:         Number of points in the output window. If zero or less, an empty
570:         array is returned.
571:     sym : bool, optional
572:         When True (default), generates a symmetric window, for use in filter
573:         design.
574:         When False, generates a periodic window, for use in spectral analysis.
575: 
576:     Returns
577:     -------
578:     w : ndarray
579:         The window, with the maximum value normalized to 1 (though the value 1
580:         does not appear if `M` is even and `sym` is True).
581: 
582:     Notes
583:     -----
584:     Flat top windows are used for taking accurate measurements of signal
585:     amplitude in the frequency domain, with minimal scalloping error from the
586:     center of a frequency bin to its edges, compared to others.  This is a
587:     5th-order cosine window, with the 5 terms optimized to make the main lobe
588:     maximally flat. [1]_
589: 
590:     References
591:     ----------
592:     .. [1] D'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for
593:            Measurement Systems", Springer Media, 2006, p. 70
594:            :doi:`10.1007/0-387-28666-7`.
595: 
596:     Examples
597:     --------
598:     Plot the window and its frequency response:
599: 
600:     >>> from scipy import signal
601:     >>> from scipy.fftpack import fft, fftshift
602:     >>> import matplotlib.pyplot as plt
603: 
604:     >>> window = signal.flattop(51)
605:     >>> plt.plot(window)
606:     >>> plt.title("Flat top window")
607:     >>> plt.ylabel("Amplitude")
608:     >>> plt.xlabel("Sample")
609: 
610:     >>> plt.figure()
611:     >>> A = fft(window, 2048) / (len(window)/2.0)
612:     >>> freq = np.linspace(-0.5, 0.5, len(A))
613:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
614:     >>> plt.plot(freq, response)
615:     >>> plt.axis([-0.5, 0.5, -120, 0])
616:     >>> plt.title("Frequency response of the flat top window")
617:     >>> plt.ylabel("Normalized magnitude [dB]")
618:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
619: 
620:     '''
621:     if _len_guards(M):
622:         return np.ones(M)
623:     M, needs_trunc = _extend(M, sym)
624: 
625:     a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
626:     w = _cos_win(M, a)
627: 
628:     return _truncate(w, needs_trunc)
629: 
630: 
631: def bartlett(M, sym=True):
632:     r'''
633:     Return a Bartlett window.
634: 
635:     The Bartlett window is very similar to a triangular window, except
636:     that the end points are at zero.  It is often used in signal
637:     processing for tapering a signal, without generating too much
638:     ripple in the frequency domain.
639: 
640:     Parameters
641:     ----------
642:     M : int
643:         Number of points in the output window. If zero or less, an empty
644:         array is returned.
645:     sym : bool, optional
646:         When True (default), generates a symmetric window, for use in filter
647:         design.
648:         When False, generates a periodic window, for use in spectral analysis.
649: 
650:     Returns
651:     -------
652:     w : ndarray
653:         The triangular window, with the first and last samples equal to zero
654:         and the maximum value normalized to 1 (though the value 1 does not
655:         appear if `M` is even and `sym` is True).
656: 
657:     See Also
658:     --------
659:     triang : A triangular window that does not touch zero at the ends
660: 
661:     Notes
662:     -----
663:     The Bartlett window is defined as
664: 
665:     .. math:: w(n) = \frac{2}{M-1} \left(
666:               \frac{M-1}{2} - \left|n - \frac{M-1}{2}\right|
667:               \right)
668: 
669:     Most references to the Bartlett window come from the signal
670:     processing literature, where it is used as one of many windowing
671:     functions for smoothing values.  Note that convolution with this
672:     window produces linear interpolation.  It is also known as an
673:     apodization (which means"removing the foot", i.e. smoothing
674:     discontinuities at the beginning and end of the sampled signal) or
675:     tapering function. The Fourier transform of the Bartlett is the product
676:     of two sinc functions.
677:     Note the excellent discussion in Kanasewich. [2]_
678: 
679:     References
680:     ----------
681:     .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
682:            Biometrika 37, 1-16, 1950.
683:     .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
684:            The University of Alberta Press, 1975, pp. 109-110.
685:     .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
686:            Processing", Prentice-Hall, 1999, pp. 468-471.
687:     .. [4] Wikipedia, "Window function",
688:            http://en.wikipedia.org/wiki/Window_function
689:     .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
690:            "Numerical Recipes", Cambridge University Press, 1986, page 429.
691: 
692:     Examples
693:     --------
694:     Plot the window and its frequency response:
695: 
696:     >>> from scipy import signal
697:     >>> from scipy.fftpack import fft, fftshift
698:     >>> import matplotlib.pyplot as plt
699: 
700:     >>> window = signal.bartlett(51)
701:     >>> plt.plot(window)
702:     >>> plt.title("Bartlett window")
703:     >>> plt.ylabel("Amplitude")
704:     >>> plt.xlabel("Sample")
705: 
706:     >>> plt.figure()
707:     >>> A = fft(window, 2048) / (len(window)/2.0)
708:     >>> freq = np.linspace(-0.5, 0.5, len(A))
709:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
710:     >>> plt.plot(freq, response)
711:     >>> plt.axis([-0.5, 0.5, -120, 0])
712:     >>> plt.title("Frequency response of the Bartlett window")
713:     >>> plt.ylabel("Normalized magnitude [dB]")
714:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
715: 
716:     '''
717:     # Docstring adapted from NumPy's bartlett function
718:     if _len_guards(M):
719:         return np.ones(M)
720:     M, needs_trunc = _extend(M, sym)
721: 
722:     n = np.arange(0, M)
723:     w = np.where(np.less_equal(n, (M - 1) / 2.0),
724:                  2.0 * n / (M - 1), 2.0 - 2.0 * n / (M - 1))
725: 
726:     return _truncate(w, needs_trunc)
727: 
728: 
729: def hann(M, sym=True):
730:     r'''
731:     Return a Hann window.
732: 
733:     The Hann window is a taper formed by using a raised cosine or sine-squared
734:     with ends that touch zero.
735: 
736:     Parameters
737:     ----------
738:     M : int
739:         Number of points in the output window. If zero or less, an empty
740:         array is returned.
741:     sym : bool, optional
742:         When True (default), generates a symmetric window, for use in filter
743:         design.
744:         When False, generates a periodic window, for use in spectral analysis.
745: 
746:     Returns
747:     -------
748:     w : ndarray
749:         The window, with the maximum value normalized to 1 (though the value 1
750:         does not appear if `M` is even and `sym` is True).
751: 
752:     Notes
753:     -----
754:     The Hann window is defined as
755: 
756:     .. math::  w(n) = 0.5 - 0.5 \cos\left(\frac{2\pi{n}}{M-1}\right)
757:                \qquad 0 \leq n \leq M-1
758: 
759:     The window was named for Julius von Hann, an Austrian meteorologist. It is
760:     also known as the Cosine Bell. It is sometimes erroneously referred to as
761:     the "Hanning" window, from the use of "hann" as a verb in the original
762:     paper and confusion with the very similar Hamming window.
763: 
764:     Most references to the Hann window come from the signal processing
765:     literature, where it is used as one of many windowing functions for
766:     smoothing values.  It is also known as an apodization (which means
767:     "removing the foot", i.e. smoothing discontinuities at the beginning
768:     and end of the sampled signal) or tapering function.
769: 
770:     References
771:     ----------
772:     .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
773:            spectra, Dover Publications, New York.
774:     .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
775:            The University of Alberta Press, 1975, pp. 106-108.
776:     .. [3] Wikipedia, "Window function",
777:            http://en.wikipedia.org/wiki/Window_function
778:     .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
779:            "Numerical Recipes", Cambridge University Press, 1986, page 425.
780: 
781:     Examples
782:     --------
783:     Plot the window and its frequency response:
784: 
785:     >>> from scipy import signal
786:     >>> from scipy.fftpack import fft, fftshift
787:     >>> import matplotlib.pyplot as plt
788: 
789:     >>> window = signal.hann(51)
790:     >>> plt.plot(window)
791:     >>> plt.title("Hann window")
792:     >>> plt.ylabel("Amplitude")
793:     >>> plt.xlabel("Sample")
794: 
795:     >>> plt.figure()
796:     >>> A = fft(window, 2048) / (len(window)/2.0)
797:     >>> freq = np.linspace(-0.5, 0.5, len(A))
798:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
799:     >>> plt.plot(freq, response)
800:     >>> plt.axis([-0.5, 0.5, -120, 0])
801:     >>> plt.title("Frequency response of the Hann window")
802:     >>> plt.ylabel("Normalized magnitude [dB]")
803:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
804: 
805:     '''
806:     # Docstring adapted from NumPy's hanning function
807:     if _len_guards(M):
808:         return np.ones(M)
809:     M, needs_trunc = _extend(M, sym)
810: 
811:     w = _cos_win(M, [0.5, 0.5])
812: 
813:     return _truncate(w, needs_trunc)
814: 
815: 
816: hanning = hann
817: 
818: 
819: def tukey(M, alpha=0.5, sym=True):
820:     r'''Return a Tukey window, also known as a tapered cosine window.
821: 
822:     Parameters
823:     ----------
824:     M : int
825:         Number of points in the output window. If zero or less, an empty
826:         array is returned.
827:     alpha : float, optional
828:         Shape parameter of the Tukey window, representing the fraction of the
829:         window inside the cosine tapered region.
830:         If zero, the Tukey window is equivalent to a rectangular window.
831:         If one, the Tukey window is equivalent to a Hann window.
832:     sym : bool, optional
833:         When True (default), generates a symmetric window, for use in filter
834:         design.
835:         When False, generates a periodic window, for use in spectral analysis.
836: 
837:     Returns
838:     -------
839:     w : ndarray
840:         The window, with the maximum value normalized to 1 (though the value 1
841:         does not appear if `M` is even and `sym` is True).
842: 
843:     References
844:     ----------
845:     .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic
846:            Analysis with the Discrete Fourier Transform". Proceedings of the
847:            IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`
848:     .. [2] Wikipedia, "Window function",
849:            http://en.wikipedia.org/wiki/Window_function#Tukey_window
850: 
851:     Examples
852:     --------
853:     Plot the window and its frequency response:
854: 
855:     >>> from scipy import signal
856:     >>> from scipy.fftpack import fft, fftshift
857:     >>> import matplotlib.pyplot as plt
858: 
859:     >>> window = signal.tukey(51)
860:     >>> plt.plot(window)
861:     >>> plt.title("Tukey window")
862:     >>> plt.ylabel("Amplitude")
863:     >>> plt.xlabel("Sample")
864:     >>> plt.ylim([0, 1.1])
865: 
866:     >>> plt.figure()
867:     >>> A = fft(window, 2048) / (len(window)/2.0)
868:     >>> freq = np.linspace(-0.5, 0.5, len(A))
869:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
870:     >>> plt.plot(freq, response)
871:     >>> plt.axis([-0.5, 0.5, -120, 0])
872:     >>> plt.title("Frequency response of the Tukey window")
873:     >>> plt.ylabel("Normalized magnitude [dB]")
874:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
875: 
876:     '''
877:     if _len_guards(M):
878:         return np.ones(M)
879: 
880:     if alpha <= 0:
881:         return np.ones(M, 'd')
882:     elif alpha >= 1.0:
883:         return hann(M, sym=sym)
884: 
885:     M, needs_trunc = _extend(M, sym)
886: 
887:     n = np.arange(0, M)
888:     width = int(np.floor(alpha*(M-1)/2.0))
889:     n1 = n[0:width+1]
890:     n2 = n[width+1:M-width-1]
891:     n3 = n[M-width-1:]
892: 
893:     w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(M-1))))
894:     w2 = np.ones(n2.shape)
895:     w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
896: 
897:     w = np.concatenate((w1, w2, w3))
898: 
899:     return _truncate(w, needs_trunc)
900: 
901: 
902: def barthann(M, sym=True):
903:     '''Return a modified Bartlett-Hann window.
904: 
905:     Parameters
906:     ----------
907:     M : int
908:         Number of points in the output window. If zero or less, an empty
909:         array is returned.
910:     sym : bool, optional
911:         When True (default), generates a symmetric window, for use in filter
912:         design.
913:         When False, generates a periodic window, for use in spectral analysis.
914: 
915:     Returns
916:     -------
917:     w : ndarray
918:         The window, with the maximum value normalized to 1 (though the value 1
919:         does not appear if `M` is even and `sym` is True).
920: 
921:     Examples
922:     --------
923:     Plot the window and its frequency response:
924: 
925:     >>> from scipy import signal
926:     >>> from scipy.fftpack import fft, fftshift
927:     >>> import matplotlib.pyplot as plt
928: 
929:     >>> window = signal.barthann(51)
930:     >>> plt.plot(window)
931:     >>> plt.title("Bartlett-Hann window")
932:     >>> plt.ylabel("Amplitude")
933:     >>> plt.xlabel("Sample")
934: 
935:     >>> plt.figure()
936:     >>> A = fft(window, 2048) / (len(window)/2.0)
937:     >>> freq = np.linspace(-0.5, 0.5, len(A))
938:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
939:     >>> plt.plot(freq, response)
940:     >>> plt.axis([-0.5, 0.5, -120, 0])
941:     >>> plt.title("Frequency response of the Bartlett-Hann window")
942:     >>> plt.ylabel("Normalized magnitude [dB]")
943:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
944: 
945:     '''
946:     if _len_guards(M):
947:         return np.ones(M)
948:     M, needs_trunc = _extend(M, sym)
949: 
950:     n = np.arange(0, M)
951:     fac = np.abs(n / (M - 1.0) - 0.5)
952:     w = 0.62 - 0.48 * fac + 0.38 * np.cos(2 * np.pi * fac)
953: 
954:     return _truncate(w, needs_trunc)
955: 
956: 
957: def hamming(M, sym=True):
958:     r'''Return a Hamming window.
959: 
960:     The Hamming window is a taper formed by using a raised cosine with
961:     non-zero endpoints, optimized to minimize the nearest side lobe.
962: 
963:     Parameters
964:     ----------
965:     M : int
966:         Number of points in the output window. If zero or less, an empty
967:         array is returned.
968:     sym : bool, optional
969:         When True (default), generates a symmetric window, for use in filter
970:         design.
971:         When False, generates a periodic window, for use in spectral analysis.
972: 
973:     Returns
974:     -------
975:     w : ndarray
976:         The window, with the maximum value normalized to 1 (though the value 1
977:         does not appear if `M` is even and `sym` is True).
978: 
979:     Notes
980:     -----
981:     The Hamming window is defined as
982: 
983:     .. math::  w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{M-1}\right)
984:                \qquad 0 \leq n \leq M-1
985: 
986:     The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and
987:     is described in Blackman and Tukey. It was recommended for smoothing the
988:     truncated autocovariance function in the time domain.
989:     Most references to the Hamming window come from the signal processing
990:     literature, where it is used as one of many windowing functions for
991:     smoothing values.  It is also known as an apodization (which means
992:     "removing the foot", i.e. smoothing discontinuities at the beginning
993:     and end of the sampled signal) or tapering function.
994: 
995:     References
996:     ----------
997:     .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
998:            spectra, Dover Publications, New York.
999:     .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
1000:            University of Alberta Press, 1975, pp. 109-110.
1001:     .. [3] Wikipedia, "Window function",
1002:            http://en.wikipedia.org/wiki/Window_function
1003:     .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
1004:            "Numerical Recipes", Cambridge University Press, 1986, page 425.
1005: 
1006:     Examples
1007:     --------
1008:     Plot the window and its frequency response:
1009: 
1010:     >>> from scipy import signal
1011:     >>> from scipy.fftpack import fft, fftshift
1012:     >>> import matplotlib.pyplot as plt
1013: 
1014:     >>> window = signal.hamming(51)
1015:     >>> plt.plot(window)
1016:     >>> plt.title("Hamming window")
1017:     >>> plt.ylabel("Amplitude")
1018:     >>> plt.xlabel("Sample")
1019: 
1020:     >>> plt.figure()
1021:     >>> A = fft(window, 2048) / (len(window)/2.0)
1022:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1023:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1024:     >>> plt.plot(freq, response)
1025:     >>> plt.axis([-0.5, 0.5, -120, 0])
1026:     >>> plt.title("Frequency response of the Hamming window")
1027:     >>> plt.ylabel("Normalized magnitude [dB]")
1028:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1029: 
1030:     '''
1031:     # Docstring adapted from NumPy's hamming function
1032:     if _len_guards(M):
1033:         return np.ones(M)
1034:     M, needs_trunc = _extend(M, sym)
1035: 
1036:     w = _cos_win(M, [0.54, 0.46])
1037: 
1038:     return _truncate(w, needs_trunc)
1039: 
1040: 
1041: def kaiser(M, beta, sym=True):
1042:     r'''Return a Kaiser window.
1043: 
1044:     The Kaiser window is a taper formed by using a Bessel function.
1045: 
1046:     Parameters
1047:     ----------
1048:     M : int
1049:         Number of points in the output window. If zero or less, an empty
1050:         array is returned.
1051:     beta : float
1052:         Shape parameter, determines trade-off between main-lobe width and
1053:         side lobe level. As beta gets large, the window narrows.
1054:     sym : bool, optional
1055:         When True (default), generates a symmetric window, for use in filter
1056:         design.
1057:         When False, generates a periodic window, for use in spectral analysis.
1058: 
1059:     Returns
1060:     -------
1061:     w : ndarray
1062:         The window, with the maximum value normalized to 1 (though the value 1
1063:         does not appear if `M` is even and `sym` is True).
1064: 
1065:     Notes
1066:     -----
1067:     The Kaiser window is defined as
1068: 
1069:     .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
1070:                \right)/I_0(\beta)
1071: 
1072:     with
1073: 
1074:     .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2},
1075: 
1076:     where :math:`I_0` is the modified zeroth-order Bessel function.
1077: 
1078:     The Kaiser was named for Jim Kaiser, who discovered a simple approximation
1079:     to the DPSS window based on Bessel functions.
1080:     The Kaiser window is a very good approximation to the Digital Prolate
1081:     Spheroidal Sequence, or Slepian window, which is the transform which
1082:     maximizes the energy in the main lobe of the window relative to total
1083:     energy.
1084: 
1085:     The Kaiser can approximate other windows by varying the beta parameter.
1086:     (Some literature uses alpha = beta/pi.) [4]_
1087: 
1088:     ====  =======================
1089:     beta  Window shape
1090:     ====  =======================
1091:     0     Rectangular
1092:     5     Similar to a Hamming
1093:     6     Similar to a Hann
1094:     8.6   Similar to a Blackman
1095:     ====  =======================
1096: 
1097:     A beta value of 14 is probably a good starting point. Note that as beta
1098:     gets large, the window narrows, and so the number of samples needs to be
1099:     large enough to sample the increasingly narrow spike, otherwise NaNs will
1100:     be returned.
1101: 
1102:     Most references to the Kaiser window come from the signal processing
1103:     literature, where it is used as one of many windowing functions for
1104:     smoothing values.  It is also known as an apodization (which means
1105:     "removing the foot", i.e. smoothing discontinuities at the beginning
1106:     and end of the sampled signal) or tapering function.
1107: 
1108:     References
1109:     ----------
1110:     .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
1111:            digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
1112:            John Wiley and Sons, New York, (1966).
1113:     .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
1114:            University of Alberta Press, 1975, pp. 177-178.
1115:     .. [3] Wikipedia, "Window function",
1116:            http://en.wikipedia.org/wiki/Window_function
1117:     .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the
1118:            discrete Fourier transform," Proceedings of the IEEE, vol. 66,
1119:            no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.
1120: 
1121:     Examples
1122:     --------
1123:     Plot the window and its frequency response:
1124: 
1125:     >>> from scipy import signal
1126:     >>> from scipy.fftpack import fft, fftshift
1127:     >>> import matplotlib.pyplot as plt
1128: 
1129:     >>> window = signal.kaiser(51, beta=14)
1130:     >>> plt.plot(window)
1131:     >>> plt.title(r"Kaiser window ($\beta$=14)")
1132:     >>> plt.ylabel("Amplitude")
1133:     >>> plt.xlabel("Sample")
1134: 
1135:     >>> plt.figure()
1136:     >>> A = fft(window, 2048) / (len(window)/2.0)
1137:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1138:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1139:     >>> plt.plot(freq, response)
1140:     >>> plt.axis([-0.5, 0.5, -120, 0])
1141:     >>> plt.title(r"Frequency response of the Kaiser window ($\beta$=14)")
1142:     >>> plt.ylabel("Normalized magnitude [dB]")
1143:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1144: 
1145:     '''
1146:     # Docstring adapted from NumPy's kaiser function
1147:     if _len_guards(M):
1148:         return np.ones(M)
1149:     M, needs_trunc = _extend(M, sym)
1150: 
1151:     n = np.arange(0, M)
1152:     alpha = (M - 1) / 2.0
1153:     w = (special.i0(beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)) /
1154:          special.i0(beta))
1155: 
1156:     return _truncate(w, needs_trunc)
1157: 
1158: 
1159: def gaussian(M, std, sym=True):
1160:     r'''Return a Gaussian window.
1161: 
1162:     Parameters
1163:     ----------
1164:     M : int
1165:         Number of points in the output window. If zero or less, an empty
1166:         array is returned.
1167:     std : float
1168:         The standard deviation, sigma.
1169:     sym : bool, optional
1170:         When True (default), generates a symmetric window, for use in filter
1171:         design.
1172:         When False, generates a periodic window, for use in spectral analysis.
1173: 
1174:     Returns
1175:     -------
1176:     w : ndarray
1177:         The window, with the maximum value normalized to 1 (though the value 1
1178:         does not appear if `M` is even and `sym` is True).
1179: 
1180:     Notes
1181:     -----
1182:     The Gaussian window is defined as
1183: 
1184:     .. math::  w(n) = e^{ -\frac{1}{2}\left(\frac{n}{\sigma}\right)^2 }
1185: 
1186:     Examples
1187:     --------
1188:     Plot the window and its frequency response:
1189: 
1190:     >>> from scipy import signal
1191:     >>> from scipy.fftpack import fft, fftshift
1192:     >>> import matplotlib.pyplot as plt
1193: 
1194:     >>> window = signal.gaussian(51, std=7)
1195:     >>> plt.plot(window)
1196:     >>> plt.title(r"Gaussian window ($\sigma$=7)")
1197:     >>> plt.ylabel("Amplitude")
1198:     >>> plt.xlabel("Sample")
1199: 
1200:     >>> plt.figure()
1201:     >>> A = fft(window, 2048) / (len(window)/2.0)
1202:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1203:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1204:     >>> plt.plot(freq, response)
1205:     >>> plt.axis([-0.5, 0.5, -120, 0])
1206:     >>> plt.title(r"Frequency response of the Gaussian window ($\sigma$=7)")
1207:     >>> plt.ylabel("Normalized magnitude [dB]")
1208:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1209: 
1210:     '''
1211:     if _len_guards(M):
1212:         return np.ones(M)
1213:     M, needs_trunc = _extend(M, sym)
1214: 
1215:     n = np.arange(0, M) - (M - 1.0) / 2.0
1216:     sig2 = 2 * std * std
1217:     w = np.exp(-n ** 2 / sig2)
1218: 
1219:     return _truncate(w, needs_trunc)
1220: 
1221: 
1222: def general_gaussian(M, p, sig, sym=True):
1223:     r'''Return a window with a generalized Gaussian shape.
1224: 
1225:     Parameters
1226:     ----------
1227:     M : int
1228:         Number of points in the output window. If zero or less, an empty
1229:         array is returned.
1230:     p : float
1231:         Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is
1232:         the same shape as the Laplace distribution.
1233:     sig : float
1234:         The standard deviation, sigma.
1235:     sym : bool, optional
1236:         When True (default), generates a symmetric window, for use in filter
1237:         design.
1238:         When False, generates a periodic window, for use in spectral analysis.
1239: 
1240:     Returns
1241:     -------
1242:     w : ndarray
1243:         The window, with the maximum value normalized to 1 (though the value 1
1244:         does not appear if `M` is even and `sym` is True).
1245: 
1246:     Notes
1247:     -----
1248:     The generalized Gaussian window is defined as
1249: 
1250:     .. math::  w(n) = e^{ -\frac{1}{2}\left|\frac{n}{\sigma}\right|^{2p} }
1251: 
1252:     the half-power point is at
1253: 
1254:     .. math::  (2 \log(2))^{1/(2 p)} \sigma
1255: 
1256:     Examples
1257:     --------
1258:     Plot the window and its frequency response:
1259: 
1260:     >>> from scipy import signal
1261:     >>> from scipy.fftpack import fft, fftshift
1262:     >>> import matplotlib.pyplot as plt
1263: 
1264:     >>> window = signal.general_gaussian(51, p=1.5, sig=7)
1265:     >>> plt.plot(window)
1266:     >>> plt.title(r"Generalized Gaussian window (p=1.5, $\sigma$=7)")
1267:     >>> plt.ylabel("Amplitude")
1268:     >>> plt.xlabel("Sample")
1269: 
1270:     >>> plt.figure()
1271:     >>> A = fft(window, 2048) / (len(window)/2.0)
1272:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1273:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1274:     >>> plt.plot(freq, response)
1275:     >>> plt.axis([-0.5, 0.5, -120, 0])
1276:     >>> plt.title(r"Freq. resp. of the gen. Gaussian "
1277:     ...           "window (p=1.5, $\sigma$=7)")
1278:     >>> plt.ylabel("Normalized magnitude [dB]")
1279:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1280: 
1281:     '''
1282:     if _len_guards(M):
1283:         return np.ones(M)
1284:     M, needs_trunc = _extend(M, sym)
1285: 
1286:     n = np.arange(0, M) - (M - 1.0) / 2.0
1287:     w = np.exp(-0.5 * np.abs(n / sig) ** (2 * p))
1288: 
1289:     return _truncate(w, needs_trunc)
1290: 
1291: 
1292: # `chebwin` contributed by Kumar Appaiah.
1293: def chebwin(M, at, sym=True):
1294:     r'''Return a Dolph-Chebyshev window.
1295: 
1296:     Parameters
1297:     ----------
1298:     M : int
1299:         Number of points in the output window. If zero or less, an empty
1300:         array is returned.
1301:     at : float
1302:         Attenuation (in dB).
1303:     sym : bool, optional
1304:         When True (default), generates a symmetric window, for use in filter
1305:         design.
1306:         When False, generates a periodic window, for use in spectral analysis.
1307: 
1308:     Returns
1309:     -------
1310:     w : ndarray
1311:         The window, with the maximum value always normalized to 1
1312: 
1313:     Notes
1314:     -----
1315:     This window optimizes for the narrowest main lobe width for a given order
1316:     `M` and sidelobe equiripple attenuation `at`, using Chebyshev
1317:     polynomials.  It was originally developed by Dolph to optimize the
1318:     directionality of radio antenna arrays.
1319: 
1320:     Unlike most windows, the Dolph-Chebyshev is defined in terms of its
1321:     frequency response:
1322: 
1323:     .. math:: W(k) = \frac
1324:               {\cos\{M \cos^{-1}[\beta \cos(\frac{\pi k}{M})]\}}
1325:               {\cosh[M \cosh^{-1}(\beta)]}
1326: 
1327:     where
1328: 
1329:     .. math:: \beta = \cosh \left [\frac{1}{M}
1330:               \cosh^{-1}(10^\frac{A}{20}) \right ]
1331: 
1332:     and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).
1333: 
1334:     The time domain window is then generated using the IFFT, so
1335:     power-of-two `M` are the fastest to generate, and prime number `M` are
1336:     the slowest.
1337: 
1338:     The equiripple condition in the frequency domain creates impulses in the
1339:     time domain, which appear at the ends of the window.
1340: 
1341:     References
1342:     ----------
1343:     .. [1] C. Dolph, "A current distribution for broadside arrays which
1344:            optimizes the relationship between beam width and side-lobe level",
1345:            Proceedings of the IEEE, Vol. 34, Issue 6
1346:     .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",
1347:            American Meteorological Society (April 1997)
1348:            http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf
1349:     .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the
1350:            discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,
1351:            No. 1, January 1978
1352: 
1353:     Examples
1354:     --------
1355:     Plot the window and its frequency response:
1356: 
1357:     >>> from scipy import signal
1358:     >>> from scipy.fftpack import fft, fftshift
1359:     >>> import matplotlib.pyplot as plt
1360: 
1361:     >>> window = signal.chebwin(51, at=100)
1362:     >>> plt.plot(window)
1363:     >>> plt.title("Dolph-Chebyshev window (100 dB)")
1364:     >>> plt.ylabel("Amplitude")
1365:     >>> plt.xlabel("Sample")
1366: 
1367:     >>> plt.figure()
1368:     >>> A = fft(window, 2048) / (len(window)/2.0)
1369:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1370:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1371:     >>> plt.plot(freq, response)
1372:     >>> plt.axis([-0.5, 0.5, -120, 0])
1373:     >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")
1374:     >>> plt.ylabel("Normalized magnitude [dB]")
1375:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1376: 
1377:     '''
1378:     if np.abs(at) < 45:
1379:         warnings.warn("This window is not suitable for spectral analysis "
1380:                       "for attenuation values lower than about 45dB because "
1381:                       "the equivalent noise bandwidth of a Chebyshev window "
1382:                       "does not grow monotonically with increasing sidelobe "
1383:                       "attenuation when the attenuation is smaller than "
1384:                       "about 45 dB.")
1385:     if _len_guards(M):
1386:         return np.ones(M)
1387:     M, needs_trunc = _extend(M, sym)
1388: 
1389:     # compute the parameter beta
1390:     order = M - 1.0
1391:     beta = np.cosh(1.0 / order * np.arccosh(10 ** (np.abs(at) / 20.)))
1392:     k = np.r_[0:M] * 1.0
1393:     x = beta * np.cos(np.pi * k / M)
1394:     # Find the window's DFT coefficients
1395:     # Use analytic definition of Chebyshev polynomial instead of expansion
1396:     # from scipy.special. Using the expansion in scipy.special leads to errors.
1397:     p = np.zeros(x.shape)
1398:     p[x > 1] = np.cosh(order * np.arccosh(x[x > 1]))
1399:     p[x < -1] = (1 - 2 * (order % 2)) * np.cosh(order * np.arccosh(-x[x < -1]))
1400:     p[np.abs(x) <= 1] = np.cos(order * np.arccos(x[np.abs(x) <= 1]))
1401: 
1402:     # Appropriate IDFT and filling up
1403:     # depending on even/odd M
1404:     if M % 2:
1405:         w = np.real(fftpack.fft(p))
1406:         n = (M + 1) // 2
1407:         w = w[:n]
1408:         w = np.concatenate((w[n - 1:0:-1], w))
1409:     else:
1410:         p = p * np.exp(1.j * np.pi / M * np.r_[0:M])
1411:         w = np.real(fftpack.fft(p))
1412:         n = M // 2 + 1
1413:         w = np.concatenate((w[n - 1:0:-1], w[1:n]))
1414:     w = w / max(w)
1415: 
1416:     return _truncate(w, needs_trunc)
1417: 
1418: 
1419: def slepian(M, width, sym=True):
1420:     '''Return a digital Slepian (DPSS) window.
1421: 
1422:     Used to maximize the energy concentration in the main lobe.  Also called
1423:     the digital prolate spheroidal sequence (DPSS).
1424: 
1425:     Parameters
1426:     ----------
1427:     M : int
1428:         Number of points in the output window. If zero or less, an empty
1429:         array is returned.
1430:     width : float
1431:         Bandwidth
1432:     sym : bool, optional
1433:         When True (default), generates a symmetric window, for use in filter
1434:         design.
1435:         When False, generates a periodic window, for use in spectral analysis.
1436: 
1437:     Returns
1438:     -------
1439:     w : ndarray
1440:         The window, with the maximum value always normalized to 1
1441: 
1442:     References
1443:     ----------
1444:     .. [1] D. Slepian & H. O. Pollak: "Prolate spheroidal wave functions,
1445:            Fourier analysis and uncertainty-I," Bell Syst. Tech. J., vol.40,
1446:            pp.43-63, 1961. https://archive.org/details/bstj40-1-43
1447:     .. [2] H. J. Landau & H. O. Pollak: "Prolate spheroidal wave functions,
1448:            Fourier analysis and uncertainty-II," Bell Syst. Tech. J. , vol.40,
1449:            pp.65-83, 1961. https://archive.org/details/bstj40-1-65
1450: 
1451:     Examples
1452:     --------
1453:     Plot the window and its frequency response:
1454: 
1455:     >>> from scipy import signal
1456:     >>> from scipy.fftpack import fft, fftshift
1457:     >>> import matplotlib.pyplot as plt
1458: 
1459:     >>> window = signal.slepian(51, width=0.3)
1460:     >>> plt.plot(window)
1461:     >>> plt.title("Slepian (DPSS) window (BW=0.3)")
1462:     >>> plt.ylabel("Amplitude")
1463:     >>> plt.xlabel("Sample")
1464: 
1465:     >>> plt.figure()
1466:     >>> A = fft(window, 2048) / (len(window)/2.0)
1467:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1468:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1469:     >>> plt.plot(freq, response)
1470:     >>> plt.axis([-0.5, 0.5, -120, 0])
1471:     >>> plt.title("Frequency response of the Slepian window (BW=0.3)")
1472:     >>> plt.ylabel("Normalized magnitude [dB]")
1473:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1474: 
1475:     '''
1476:     if _len_guards(M):
1477:         return np.ones(M)
1478:     M, needs_trunc = _extend(M, sym)
1479: 
1480:     # our width is the full bandwidth
1481:     width = width / 2
1482:     # to match the old version
1483:     width = width / 2
1484:     m = np.arange(M, dtype='d')
1485:     H = np.zeros((2, M))
1486:     H[0, 1:] = m[1:] * (M - m[1:]) / 2
1487:     H[1, :] = ((M - 1 - 2 * m) / 2)**2 * np.cos(2 * np.pi * width)
1488: 
1489:     _, win = linalg.eig_banded(H, select='i', select_range=(M-1, M-1))
1490:     win = win.ravel() / win.max()
1491: 
1492:     return _truncate(win, needs_trunc)
1493: 
1494: 
1495: def cosine(M, sym=True):
1496:     '''Return a window with a simple cosine shape.
1497: 
1498:     Parameters
1499:     ----------
1500:     M : int
1501:         Number of points in the output window. If zero or less, an empty
1502:         array is returned.
1503:     sym : bool, optional
1504:         When True (default), generates a symmetric window, for use in filter
1505:         design.
1506:         When False, generates a periodic window, for use in spectral analysis.
1507: 
1508:     Returns
1509:     -------
1510:     w : ndarray
1511:         The window, with the maximum value normalized to 1 (though the value 1
1512:         does not appear if `M` is even and `sym` is True).
1513: 
1514:     Notes
1515:     -----
1516: 
1517:     .. versionadded:: 0.13.0
1518: 
1519:     Examples
1520:     --------
1521:     Plot the window and its frequency response:
1522: 
1523:     >>> from scipy import signal
1524:     >>> from scipy.fftpack import fft, fftshift
1525:     >>> import matplotlib.pyplot as plt
1526: 
1527:     >>> window = signal.cosine(51)
1528:     >>> plt.plot(window)
1529:     >>> plt.title("Cosine window")
1530:     >>> plt.ylabel("Amplitude")
1531:     >>> plt.xlabel("Sample")
1532: 
1533:     >>> plt.figure()
1534:     >>> A = fft(window, 2048) / (len(window)/2.0)
1535:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1536:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1537:     >>> plt.plot(freq, response)
1538:     >>> plt.axis([-0.5, 0.5, -120, 0])
1539:     >>> plt.title("Frequency response of the cosine window")
1540:     >>> plt.ylabel("Normalized magnitude [dB]")
1541:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1542:     >>> plt.show()
1543: 
1544:     '''
1545:     if _len_guards(M):
1546:         return np.ones(M)
1547:     M, needs_trunc = _extend(M, sym)
1548: 
1549:     w = np.sin(np.pi / M * (np.arange(0, M) + .5))
1550: 
1551:     return _truncate(w, needs_trunc)
1552: 
1553: 
1554: def exponential(M, center=None, tau=1., sym=True):
1555:     r'''Return an exponential (or Poisson) window.
1556: 
1557:     Parameters
1558:     ----------
1559:     M : int
1560:         Number of points in the output window. If zero or less, an empty
1561:         array is returned.
1562:     center : float, optional
1563:         Parameter defining the center location of the window function.
1564:         The default value if not given is ``center = (M-1) / 2``.  This
1565:         parameter must take its default value for symmetric windows.
1566:     tau : float, optional
1567:         Parameter defining the decay.  For ``center = 0`` use
1568:         ``tau = -(M-1) / ln(x)`` if ``x`` is the fraction of the window
1569:         remaining at the end.
1570:     sym : bool, optional
1571:         When True (default), generates a symmetric window, for use in filter
1572:         design.
1573:         When False, generates a periodic window, for use in spectral analysis.
1574: 
1575:     Returns
1576:     -------
1577:     w : ndarray
1578:         The window, with the maximum value normalized to 1 (though the value 1
1579:         does not appear if `M` is even and `sym` is True).
1580: 
1581:     Notes
1582:     -----
1583:     The Exponential window is defined as
1584: 
1585:     .. math::  w(n) = e^{-|n-center| / \tau}
1586: 
1587:     References
1588:     ----------
1589:     S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",
1590:     Technical Review 3, Bruel & Kjaer, 1987.
1591: 
1592:     Examples
1593:     --------
1594:     Plot the symmetric window and its frequency response:
1595: 
1596:     >>> from scipy import signal
1597:     >>> from scipy.fftpack import fft, fftshift
1598:     >>> import matplotlib.pyplot as plt
1599: 
1600:     >>> M = 51
1601:     >>> tau = 3.0
1602:     >>> window = signal.exponential(M, tau=tau)
1603:     >>> plt.plot(window)
1604:     >>> plt.title("Exponential Window (tau=3.0)")
1605:     >>> plt.ylabel("Amplitude")
1606:     >>> plt.xlabel("Sample")
1607: 
1608:     >>> plt.figure()
1609:     >>> A = fft(window, 2048) / (len(window)/2.0)
1610:     >>> freq = np.linspace(-0.5, 0.5, len(A))
1611:     >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
1612:     >>> plt.plot(freq, response)
1613:     >>> plt.axis([-0.5, 0.5, -35, 0])
1614:     >>> plt.title("Frequency response of the Exponential window (tau=3.0)")
1615:     >>> plt.ylabel("Normalized magnitude [dB]")
1616:     >>> plt.xlabel("Normalized frequency [cycles per sample]")
1617: 
1618:     This function can also generate non-symmetric windows:
1619: 
1620:     >>> tau2 = -(M-1) / np.log(0.01)
1621:     >>> window2 = signal.exponential(M, 0, tau2, False)
1622:     >>> plt.figure()
1623:     >>> plt.plot(window2)
1624:     >>> plt.ylabel("Amplitude")
1625:     >>> plt.xlabel("Sample")
1626:     '''
1627:     if sym and center is not None:
1628:         raise ValueError("If sym==True, center must be None.")
1629:     if _len_guards(M):
1630:         return np.ones(M)
1631:     M, needs_trunc = _extend(M, sym)
1632: 
1633:     if center is None:
1634:         center = (M-1) / 2
1635: 
1636:     n = np.arange(0, M)
1637:     w = np.exp(-np.abs(n-center) / tau)
1638: 
1639:     return _truncate(w, needs_trunc)
1640: 
1641: 
1642: _win_equiv_raw = {
1643:     ('barthann', 'brthan', 'bth'): (barthann, False),
1644:     ('bartlett', 'bart', 'brt'): (bartlett, False),
1645:     ('blackman', 'black', 'blk'): (blackman, False),
1646:     ('blackmanharris', 'blackharr', 'bkh'): (blackmanharris, False),
1647:     ('bohman', 'bman', 'bmn'): (bohman, False),
1648:     ('boxcar', 'box', 'ones',
1649:         'rect', 'rectangular'): (boxcar, False),
1650:     ('chebwin', 'cheb'): (chebwin, True),
1651:     ('cosine', 'halfcosine'): (cosine, False),
1652:     ('exponential', 'poisson'): (exponential, True),
1653:     ('flattop', 'flat', 'flt'): (flattop, False),
1654:     ('gaussian', 'gauss', 'gss'): (gaussian, True),
1655:     ('general gaussian', 'general_gaussian',
1656:         'general gauss', 'general_gauss', 'ggs'): (general_gaussian, True),
1657:     ('hamming', 'hamm', 'ham'): (hamming, False),
1658:     ('hanning', 'hann', 'han'): (hann, False),
1659:     ('kaiser', 'ksr'): (kaiser, True),
1660:     ('nuttall', 'nutl', 'nut'): (nuttall, False),
1661:     ('parzen', 'parz', 'par'): (parzen, False),
1662:     ('slepian', 'slep', 'optimal', 'dpss', 'dss'): (slepian, True),
1663:     ('triangle', 'triang', 'tri'): (triang, False),
1664:     ('tukey', 'tuk'): (tukey, True),
1665: }
1666: 
1667: # Fill dict with all valid window name strings
1668: _win_equiv = {}
1669: for k, v in _win_equiv_raw.items():
1670:     for key in k:
1671:         _win_equiv[key] = v[0]
1672: 
1673: # Keep track of which windows need additional parameters
1674: _needs_param = set()
1675: for k, v in _win_equiv_raw.items():
1676:     if v[1]:
1677:         _needs_param.update(k)
1678: 
1679: 
1680: def get_window(window, Nx, fftbins=True):
1681:     '''
1682:     Return a window.
1683: 
1684:     Parameters
1685:     ----------
1686:     window : string, float, or tuple
1687:         The type of window to create. See below for more details.
1688:     Nx : int
1689:         The number of samples in the window.
1690:     fftbins : bool, optional
1691:         If True (default), create a "periodic" window, ready to use with
1692:         `ifftshift` and be multiplied by the result of an FFT (see also
1693:         `fftpack.fftfreq`).
1694:         If False, create a "symmetric" window, for use in filter design.
1695: 
1696:     Returns
1697:     -------
1698:     get_window : ndarray
1699:         Returns a window of length `Nx` and type `window`
1700: 
1701:     Notes
1702:     -----
1703:     Window types:
1704: 
1705:         `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
1706:         `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,
1707:         `barthann`, `kaiser` (needs beta), `gaussian` (needs standard
1708:         deviation), `general_gaussian` (needs power, width), `slepian`
1709:         (needs width), `chebwin` (needs attenuation), `exponential`
1710:         (needs decay scale), `tukey` (needs taper fraction)
1711: 
1712:     If the window requires no parameters, then `window` can be a string.
1713: 
1714:     If the window requires parameters, then `window` must be a tuple
1715:     with the first argument the string name of the window, and the next
1716:     arguments the needed parameters.
1717: 
1718:     If `window` is a floating point number, it is interpreted as the beta
1719:     parameter of the `kaiser` window.
1720: 
1721:     Each of the window types listed above is also the name of
1722:     a function that can be called directly to create a window of
1723:     that type.
1724: 
1725:     Examples
1726:     --------
1727:     >>> from scipy import signal
1728:     >>> signal.get_window('triang', 7)
1729:     array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])
1730:     >>> signal.get_window(('kaiser', 4.0), 9)
1731:     array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
1732:             0.97885093,  0.82160913,  0.56437221,  0.29425961])
1733:     >>> signal.get_window(4.0, 9)
1734:     array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,
1735:             0.97885093,  0.82160913,  0.56437221,  0.29425961])
1736: 
1737:     '''
1738:     sym = not fftbins
1739:     try:
1740:         beta = float(window)
1741:     except (TypeError, ValueError):
1742:         args = ()
1743:         if isinstance(window, tuple):
1744:             winstr = window[0]
1745:             if len(window) > 1:
1746:                 args = window[1:]
1747:         elif isinstance(window, string_types):
1748:             if window in _needs_param:
1749:                 raise ValueError("The '" + window + "' window needs one or "
1750:                                  "more parameters -- pass a tuple.")
1751:             else:
1752:                 winstr = window
1753:         else:
1754:             raise ValueError("%s as window type is not supported." %
1755:                              str(type(window)))
1756: 
1757:         try:
1758:             winfunc = _win_equiv[winstr]
1759:         except KeyError:
1760:             raise ValueError("Unknown window type.")
1761: 
1762:         params = (Nx,) + args + (sym,)
1763:     else:
1764:         winfunc = kaiser
1765:         params = (Nx, beta, sym)
1766: 
1767:     return winfunc(*params)
1768: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_284712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'The suite of window functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import warnings' statement (line 4)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_284713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_284713) is not StypyTypeError):

    if (import_284713 != 'pyd_module'):
        __import__(import_284713)
        sys_modules_284714 = sys.modules[import_284713]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_284714.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_284713)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy import fftpack, linalg, special' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_284715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy')

if (type(import_284715) is not StypyTypeError):

    if (import_284715 != 'pyd_module'):
        __import__(import_284715)
        sys_modules_284716 = sys.modules[import_284715]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', sys_modules_284716.module_type_store, module_type_store, ['fftpack', 'linalg', 'special'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_284716, sys_modules_284716.module_type_store, module_type_store)
    else:
        from scipy import fftpack, linalg, special

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', None, module_type_store, ['fftpack', 'linalg', 'special'], [fftpack, linalg, special])

else:
    # Assigning a type to the variable 'scipy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', import_284715)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib.six import string_types' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_284717 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six')

if (type(import_284717) is not StypyTypeError):

    if (import_284717 != 'pyd_module'):
        __import__(import_284717)
        sys_modules_284718 = sys.modules[import_284717]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', sys_modules_284718.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_284718, sys_modules_284718.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib.six', import_284717)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall', 'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann', 'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'chebwin', 'slepian', 'cosine', 'hann', 'exponential', 'tukey', 'get_window']
module_type_store.set_exportable_members(['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall', 'blackmanharris', 'flattop', 'bartlett', 'hanning', 'barthann', 'hamming', 'kaiser', 'gaussian', 'general_gaussian', 'chebwin', 'slepian', 'cosine', 'hann', 'exponential', 'tukey', 'get_window'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_284719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_284720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'boxcar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284720)
# Adding element type (line 10)
str_284721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', 'triang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284721)
# Adding element type (line 10)
str_284722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'str', 'parzen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284722)
# Adding element type (line 10)
str_284723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'str', 'bohman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284723)
# Adding element type (line 10)
str_284724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 51), 'str', 'blackman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284724)
# Adding element type (line 10)
str_284725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 63), 'str', 'nuttall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284725)
# Adding element type (line 10)
str_284726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'str', 'blackmanharris')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284726)
# Adding element type (line 10)
str_284727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'str', 'flattop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284727)
# Adding element type (line 10)
str_284728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 40), 'str', 'bartlett')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284728)
# Adding element type (line 10)
str_284729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 52), 'str', 'hanning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284729)
# Adding element type (line 10)
str_284730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 63), 'str', 'barthann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284730)
# Adding element type (line 10)
str_284731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'hamming')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284731)
# Adding element type (line 10)
str_284732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'str', 'kaiser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284732)
# Adding element type (line 10)
str_284733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'str', 'gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284733)
# Adding element type (line 10)
str_284734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 44), 'str', 'general_gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284734)
# Adding element type (line 10)
str_284735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 64), 'str', 'chebwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284735)
# Adding element type (line 10)
str_284736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'slepian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284736)
# Adding element type (line 10)
str_284737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284737)
# Adding element type (line 10)
str_284738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'str', 'hann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284738)
# Adding element type (line 10)
str_284739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 40), 'str', 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284739)
# Adding element type (line 10)
str_284740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 55), 'str', 'tukey')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284740)
# Adding element type (line 10)
str_284741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 64), 'str', 'get_window')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_284719, str_284741)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_284719)

@norecursion
def _len_guards(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_len_guards'
    module_type_store = module_type_store.open_function_context('_len_guards', 16, 0, False)
    
    # Passed parameters checking function
    _len_guards.stypy_localization = localization
    _len_guards.stypy_type_of_self = None
    _len_guards.stypy_type_store = module_type_store
    _len_guards.stypy_function_name = '_len_guards'
    _len_guards.stypy_param_names_list = ['M']
    _len_guards.stypy_varargs_param_name = None
    _len_guards.stypy_kwargs_param_name = None
    _len_guards.stypy_call_defaults = defaults
    _len_guards.stypy_call_varargs = varargs
    _len_guards.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_len_guards', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_len_guards', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_len_guards(...)' code ##################

    str_284742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Handle small or incorrect window lengths')
    
    
    # Evaluating a boolean operation
    
    
    # Call to int(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'M' (line 18)
    M_284744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'M', False)
    # Processing the call keyword arguments (line 18)
    kwargs_284745 = {}
    # Getting the type of 'int' (line 18)
    int_284743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'int', False)
    # Calling int(args, kwargs) (line 18)
    int_call_result_284746 = invoke(stypy.reporting.localization.Localization(__file__, 18, 7), int_284743, *[M_284744], **kwargs_284745)
    
    # Getting the type of 'M' (line 18)
    M_284747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'M')
    # Applying the binary operator '!=' (line 18)
    result_ne_284748 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), '!=', int_call_result_284746, M_284747)
    
    
    # Getting the type of 'M' (line 18)
    M_284749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'M')
    int_284750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'int')
    # Applying the binary operator '<' (line 18)
    result_lt_284751 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 22), '<', M_284749, int_284750)
    
    # Applying the binary operator 'or' (line 18)
    result_or_keyword_284752 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), 'or', result_ne_284748, result_lt_284751)
    
    # Testing the type of an if condition (line 18)
    if_condition_284753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_or_keyword_284752)
    # Assigning a type to the variable 'if_condition_284753' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_284753', if_condition_284753)
    # SSA begins for if statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 19)
    # Processing the call arguments (line 19)
    str_284755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', 'Window length M must be a non-negative integer')
    # Processing the call keyword arguments (line 19)
    kwargs_284756 = {}
    # Getting the type of 'ValueError' (line 19)
    ValueError_284754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 19)
    ValueError_call_result_284757 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), ValueError_284754, *[str_284755], **kwargs_284756)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 19, 8), ValueError_call_result_284757, 'raise parameter', BaseException)
    # SSA join for if statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'M' (line 20)
    M_284758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'M')
    int_284759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
    # Applying the binary operator '<=' (line 20)
    result_le_284760 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), '<=', M_284758, int_284759)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', result_le_284760)
    
    # ################# End of '_len_guards(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_len_guards' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_284761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284761)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_len_guards'
    return stypy_return_type_284761

# Assigning a type to the variable '_len_guards' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_len_guards', _len_guards)

@norecursion
def _extend(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_extend'
    module_type_store = module_type_store.open_function_context('_extend', 23, 0, False)
    
    # Passed parameters checking function
    _extend.stypy_localization = localization
    _extend.stypy_type_of_self = None
    _extend.stypy_type_store = module_type_store
    _extend.stypy_function_name = '_extend'
    _extend.stypy_param_names_list = ['M', 'sym']
    _extend.stypy_varargs_param_name = None
    _extend.stypy_kwargs_param_name = None
    _extend.stypy_call_defaults = defaults
    _extend.stypy_call_varargs = varargs
    _extend.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_extend', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_extend', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_extend(...)' code ##################

    str_284762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', 'Extend window by 1 sample if needed for DFT-even symmetry')
    
    
    # Getting the type of 'sym' (line 25)
    sym_284763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'sym')
    # Applying the 'not' unary operator (line 25)
    result_not__284764 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), 'not', sym_284763)
    
    # Testing the type of an if condition (line 25)
    if_condition_284765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_not__284764)
    # Assigning a type to the variable 'if_condition_284765' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_284765', if_condition_284765)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_284766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'M' (line 26)
    M_284767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'M')
    int_284768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    # Applying the binary operator '+' (line 26)
    result_add_284769 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '+', M_284767, int_284768)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_284766, result_add_284769)
    # Adding element type (line 26)
    # Getting the type of 'True' (line 26)
    True_284770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 15), tuple_284766, True_284770)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', tuple_284766)
    # SSA branch for the else part of an if statement (line 25)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_284771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    # Getting the type of 'M' (line 28)
    M_284772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'M')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), tuple_284771, M_284772)
    # Adding element type (line 28)
    # Getting the type of 'False' (line 28)
    False_284773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), tuple_284771, False_284773)
    
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', tuple_284771)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_extend(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_extend' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_284774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284774)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_extend'
    return stypy_return_type_284774

# Assigning a type to the variable '_extend' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '_extend', _extend)

@norecursion
def _truncate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_truncate'
    module_type_store = module_type_store.open_function_context('_truncate', 31, 0, False)
    
    # Passed parameters checking function
    _truncate.stypy_localization = localization
    _truncate.stypy_type_of_self = None
    _truncate.stypy_type_store = module_type_store
    _truncate.stypy_function_name = '_truncate'
    _truncate.stypy_param_names_list = ['w', 'needed']
    _truncate.stypy_varargs_param_name = None
    _truncate.stypy_kwargs_param_name = None
    _truncate.stypy_call_defaults = defaults
    _truncate.stypy_call_varargs = varargs
    _truncate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_truncate', ['w', 'needed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_truncate', localization, ['w', 'needed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_truncate(...)' code ##################

    str_284775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'Truncate window by 1 sample if needed for DFT-even symmetry')
    
    # Getting the type of 'needed' (line 33)
    needed_284776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'needed')
    # Testing the type of an if condition (line 33)
    if_condition_284777 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), needed_284776)
    # Assigning a type to the variable 'if_condition_284777' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_284777', if_condition_284777)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_284778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'int')
    slice_284779 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 34, 15), None, int_284778, None)
    # Getting the type of 'w' (line 34)
    w_284780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'w')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___284781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), w_284780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_284782 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), getitem___284781, slice_284779)
    
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', subscript_call_result_284782)
    # SSA branch for the else part of an if statement (line 33)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'w' (line 36)
    w_284783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', w_284783)
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_truncate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_truncate' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_284784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284784)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_truncate'
    return stypy_return_type_284784

# Assigning a type to the variable '_truncate' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), '_truncate', _truncate)

@norecursion
def _cos_win(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 39)
    True_284785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'True')
    defaults = [True_284785]
    # Create a new context for function '_cos_win'
    module_type_store = module_type_store.open_function_context('_cos_win', 39, 0, False)
    
    # Passed parameters checking function
    _cos_win.stypy_localization = localization
    _cos_win.stypy_type_of_self = None
    _cos_win.stypy_type_store = module_type_store
    _cos_win.stypy_function_name = '_cos_win'
    _cos_win.stypy_param_names_list = ['M', 'a', 'sym']
    _cos_win.stypy_varargs_param_name = None
    _cos_win.stypy_kwargs_param_name = None
    _cos_win.stypy_call_defaults = defaults
    _cos_win.stypy_call_varargs = varargs
    _cos_win.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_cos_win', ['M', 'a', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_cos_win', localization, ['M', 'a', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_cos_win(...)' code ##################

    str_284786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', '\n    Generic weighted sum of cosine terms window\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window\n    a : array_like\n        Sequence of weighting coefficients. This uses the convention of being\n        centered on the origin, so these will typically all be positive\n        numbers, not alternating sign.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    References\n    ----------\n    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE\n           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,\n           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.\n    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the\n           Discrete Fourier transform (DFT), including a comprehensive list of\n           window functions and some new flat-top windows", February 15, 2002\n           https://holometer.fnal.gov/GH_FFT.pdf\n\n    Examples\n    --------\n    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_\n\n    .. math::  w_j = 1 - 1.942604 \\cos(z) + 1.340318 \\cos(2z)\n               - 0.440811 \\cos(3z) + 0.043097 \\cos(4z)\n\n    where\n\n    .. math::  z = \\frac{2 \\pi j}{N}, j = 0...N - 1\n\n    Since this uses the convention of starting at the origin, to reproduce the\n    window, we need to convert every other coefficient to a positive number:\n\n    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]\n\n    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce\n    Figure 42 by plotting the window and its frequency response, and confirm\n    the sidelobe level in red:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal._cos_win(1000, HFT90D, sym=False)\n    >>> plt.plot(window)\n    >>> plt.title("HFT90D window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 10000) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-50/1000, 50/1000, -140, 0])\n    >>> plt.title("Frequency response of the HFT90D window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n    >>> plt.axhline(-90.2, color=\'red\')\n\n    ')
    
    
    # Call to _len_guards(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'M' (line 108)
    M_284788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'M', False)
    # Processing the call keyword arguments (line 108)
    kwargs_284789 = {}
    # Getting the type of '_len_guards' (line 108)
    _len_guards_284787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 108)
    _len_guards_call_result_284790 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), _len_guards_284787, *[M_284788], **kwargs_284789)
    
    # Testing the type of an if condition (line 108)
    if_condition_284791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), _len_guards_call_result_284790)
    # Assigning a type to the variable 'if_condition_284791' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_284791', if_condition_284791)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'M' (line 109)
    M_284794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'M', False)
    # Processing the call keyword arguments (line 109)
    kwargs_284795 = {}
    # Getting the type of 'np' (line 109)
    np_284792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 109)
    ones_284793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), np_284792, 'ones')
    # Calling ones(args, kwargs) (line 109)
    ones_call_result_284796 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), ones_284793, *[M_284794], **kwargs_284795)
    
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', ones_call_result_284796)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 110):
    
    # Assigning a Subscript to a Name (line 110):
    
    # Obtaining the type of the subscript
    int_284797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'int')
    
    # Call to _extend(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'M' (line 110)
    M_284799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'M', False)
    # Getting the type of 'sym' (line 110)
    sym_284800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'sym', False)
    # Processing the call keyword arguments (line 110)
    kwargs_284801 = {}
    # Getting the type of '_extend' (line 110)
    _extend_284798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 110)
    _extend_call_result_284802 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), _extend_284798, *[M_284799, sym_284800], **kwargs_284801)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___284803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), _extend_call_result_284802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_284804 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), getitem___284803, int_284797)
    
    # Assigning a type to the variable 'tuple_var_assignment_284668' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_var_assignment_284668', subscript_call_result_284804)
    
    # Assigning a Subscript to a Name (line 110):
    
    # Obtaining the type of the subscript
    int_284805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 4), 'int')
    
    # Call to _extend(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'M' (line 110)
    M_284807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'M', False)
    # Getting the type of 'sym' (line 110)
    sym_284808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'sym', False)
    # Processing the call keyword arguments (line 110)
    kwargs_284809 = {}
    # Getting the type of '_extend' (line 110)
    _extend_284806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 110)
    _extend_call_result_284810 = invoke(stypy.reporting.localization.Localization(__file__, 110, 21), _extend_284806, *[M_284807, sym_284808], **kwargs_284809)
    
    # Obtaining the member '__getitem__' of a type (line 110)
    getitem___284811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), _extend_call_result_284810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 110)
    subscript_call_result_284812 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), getitem___284811, int_284805)
    
    # Assigning a type to the variable 'tuple_var_assignment_284669' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_var_assignment_284669', subscript_call_result_284812)
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'tuple_var_assignment_284668' (line 110)
    tuple_var_assignment_284668_284813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_var_assignment_284668')
    # Assigning a type to the variable 'M' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'M', tuple_var_assignment_284668_284813)
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'tuple_var_assignment_284669' (line 110)
    tuple_var_assignment_284669_284814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'tuple_var_assignment_284669')
    # Assigning a type to the variable 'needs_trunc' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'needs_trunc', tuple_var_assignment_284669_284814)
    
    # Assigning a Call to a Name (line 112):
    
    # Assigning a Call to a Name (line 112):
    
    # Call to linspace(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Getting the type of 'np' (line 112)
    np_284817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'np', False)
    # Obtaining the member 'pi' of a type (line 112)
    pi_284818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 23), np_284817, 'pi')
    # Applying the 'usub' unary operator (line 112)
    result___neg___284819 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 22), 'usub', pi_284818)
    
    # Getting the type of 'np' (line 112)
    np_284820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'np', False)
    # Obtaining the member 'pi' of a type (line 112)
    pi_284821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), np_284820, 'pi')
    # Getting the type of 'M' (line 112)
    M_284822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'M', False)
    # Processing the call keyword arguments (line 112)
    kwargs_284823 = {}
    # Getting the type of 'np' (line 112)
    np_284815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 10), 'np', False)
    # Obtaining the member 'linspace' of a type (line 112)
    linspace_284816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 10), np_284815, 'linspace')
    # Calling linspace(args, kwargs) (line 112)
    linspace_call_result_284824 = invoke(stypy.reporting.localization.Localization(__file__, 112, 10), linspace_284816, *[result___neg___284819, pi_284821, M_284822], **kwargs_284823)
    
    # Assigning a type to the variable 'fac' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'fac', linspace_call_result_284824)
    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to zeros(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'M' (line 113)
    M_284827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'M', False)
    # Processing the call keyword arguments (line 113)
    kwargs_284828 = {}
    # Getting the type of 'np' (line 113)
    np_284825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 113)
    zeros_284826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), np_284825, 'zeros')
    # Calling zeros(args, kwargs) (line 113)
    zeros_call_result_284829 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), zeros_284826, *[M_284827], **kwargs_284828)
    
    # Assigning a type to the variable 'w' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'w', zeros_call_result_284829)
    
    
    # Call to range(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to len(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'a' (line 114)
    a_284832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'a', False)
    # Processing the call keyword arguments (line 114)
    kwargs_284833 = {}
    # Getting the type of 'len' (line 114)
    len_284831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'len', False)
    # Calling len(args, kwargs) (line 114)
    len_call_result_284834 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), len_284831, *[a_284832], **kwargs_284833)
    
    # Processing the call keyword arguments (line 114)
    kwargs_284835 = {}
    # Getting the type of 'range' (line 114)
    range_284830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'range', False)
    # Calling range(args, kwargs) (line 114)
    range_call_result_284836 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), range_284830, *[len_call_result_284834], **kwargs_284835)
    
    # Testing the type of a for loop iterable (line 114)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_284836)
    # Getting the type of the for loop variable (line 114)
    for_loop_var_284837 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_284836)
    # Assigning a type to the variable 'k' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'k', for_loop_var_284837)
    # SSA begins for a for statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'w' (line 115)
    w_284838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'w')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 115)
    k_284839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'k')
    # Getting the type of 'a' (line 115)
    a_284840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'a')
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___284841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 13), a_284840, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_284842 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), getitem___284841, k_284839)
    
    
    # Call to cos(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'k' (line 115)
    k_284845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'k', False)
    # Getting the type of 'fac' (line 115)
    fac_284846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'fac', False)
    # Applying the binary operator '*' (line 115)
    result_mul_284847 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 27), '*', k_284845, fac_284846)
    
    # Processing the call keyword arguments (line 115)
    kwargs_284848 = {}
    # Getting the type of 'np' (line 115)
    np_284843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 20), 'np', False)
    # Obtaining the member 'cos' of a type (line 115)
    cos_284844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 20), np_284843, 'cos')
    # Calling cos(args, kwargs) (line 115)
    cos_call_result_284849 = invoke(stypy.reporting.localization.Localization(__file__, 115, 20), cos_284844, *[result_mul_284847], **kwargs_284848)
    
    # Applying the binary operator '*' (line 115)
    result_mul_284850 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 13), '*', subscript_call_result_284842, cos_call_result_284849)
    
    # Applying the binary operator '+=' (line 115)
    result_iadd_284851 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 8), '+=', w_284838, result_mul_284850)
    # Assigning a type to the variable 'w' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'w', result_iadd_284851)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _truncate(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'w' (line 117)
    w_284853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 117)
    needs_trunc_284854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 117)
    kwargs_284855 = {}
    # Getting the type of '_truncate' (line 117)
    _truncate_284852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 117)
    _truncate_call_result_284856 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), _truncate_284852, *[w_284853, needs_trunc_284854], **kwargs_284855)
    
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type', _truncate_call_result_284856)
    
    # ################# End of '_cos_win(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_cos_win' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_284857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_cos_win'
    return stypy_return_type_284857

# Assigning a type to the variable '_cos_win' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_cos_win', _cos_win)

@norecursion
def boxcar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 120)
    True_284858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'True')
    defaults = [True_284858]
    # Create a new context for function 'boxcar'
    module_type_store = module_type_store.open_function_context('boxcar', 120, 0, False)
    
    # Passed parameters checking function
    boxcar.stypy_localization = localization
    boxcar.stypy_type_of_self = None
    boxcar.stypy_type_store = module_type_store
    boxcar.stypy_function_name = 'boxcar'
    boxcar.stypy_param_names_list = ['M', 'sym']
    boxcar.stypy_varargs_param_name = None
    boxcar.stypy_kwargs_param_name = None
    boxcar.stypy_call_defaults = defaults
    boxcar.stypy_call_varargs = varargs
    boxcar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'boxcar', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'boxcar', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'boxcar(...)' code ##################

    str_284859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', 'Return a boxcar or rectangular window.\n\n    Also known as a rectangular window or Dirichlet window, this is equivalent\n    to no window at all.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        Whether the window is symmetric. (Has no effect for boxcar.)\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.boxcar(51)\n    >>> plt.plot(window)\n    >>> plt.title("Boxcar window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the boxcar window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'M' (line 164)
    M_284861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 19), 'M', False)
    # Processing the call keyword arguments (line 164)
    kwargs_284862 = {}
    # Getting the type of '_len_guards' (line 164)
    _len_guards_284860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 164)
    _len_guards_call_result_284863 = invoke(stypy.reporting.localization.Localization(__file__, 164, 7), _len_guards_284860, *[M_284861], **kwargs_284862)
    
    # Testing the type of an if condition (line 164)
    if_condition_284864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), _len_guards_call_result_284863)
    # Assigning a type to the variable 'if_condition_284864' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'if_condition_284864', if_condition_284864)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'M' (line 165)
    M_284867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'M', False)
    # Processing the call keyword arguments (line 165)
    kwargs_284868 = {}
    # Getting the type of 'np' (line 165)
    np_284865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 165)
    ones_284866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), np_284865, 'ones')
    # Calling ones(args, kwargs) (line 165)
    ones_call_result_284869 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), ones_284866, *[M_284867], **kwargs_284868)
    
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', ones_call_result_284869)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 166):
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_284870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'int')
    
    # Call to _extend(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'M' (line 166)
    M_284872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'M', False)
    # Getting the type of 'sym' (line 166)
    sym_284873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'sym', False)
    # Processing the call keyword arguments (line 166)
    kwargs_284874 = {}
    # Getting the type of '_extend' (line 166)
    _extend_284871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 166)
    _extend_call_result_284875 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), _extend_284871, *[M_284872, sym_284873], **kwargs_284874)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___284876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 4), _extend_call_result_284875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_284877 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), getitem___284876, int_284870)
    
    # Assigning a type to the variable 'tuple_var_assignment_284670' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'tuple_var_assignment_284670', subscript_call_result_284877)
    
    # Assigning a Subscript to a Name (line 166):
    
    # Obtaining the type of the subscript
    int_284878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'int')
    
    # Call to _extend(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'M' (line 166)
    M_284880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'M', False)
    # Getting the type of 'sym' (line 166)
    sym_284881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'sym', False)
    # Processing the call keyword arguments (line 166)
    kwargs_284882 = {}
    # Getting the type of '_extend' (line 166)
    _extend_284879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 166)
    _extend_call_result_284883 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), _extend_284879, *[M_284880, sym_284881], **kwargs_284882)
    
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___284884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 4), _extend_call_result_284883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 166)
    subscript_call_result_284885 = invoke(stypy.reporting.localization.Localization(__file__, 166, 4), getitem___284884, int_284878)
    
    # Assigning a type to the variable 'tuple_var_assignment_284671' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'tuple_var_assignment_284671', subscript_call_result_284885)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_284670' (line 166)
    tuple_var_assignment_284670_284886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'tuple_var_assignment_284670')
    # Assigning a type to the variable 'M' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'M', tuple_var_assignment_284670_284886)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'tuple_var_assignment_284671' (line 166)
    tuple_var_assignment_284671_284887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'tuple_var_assignment_284671')
    # Assigning a type to the variable 'needs_trunc' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'needs_trunc', tuple_var_assignment_284671_284887)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to ones(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'M' (line 168)
    M_284890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'M', False)
    # Getting the type of 'float' (line 168)
    float_284891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'float', False)
    # Processing the call keyword arguments (line 168)
    kwargs_284892 = {}
    # Getting the type of 'np' (line 168)
    np_284888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'np', False)
    # Obtaining the member 'ones' of a type (line 168)
    ones_284889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), np_284888, 'ones')
    # Calling ones(args, kwargs) (line 168)
    ones_call_result_284893 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), ones_284889, *[M_284890, float_284891], **kwargs_284892)
    
    # Assigning a type to the variable 'w' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'w', ones_call_result_284893)
    
    # Call to _truncate(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'w' (line 170)
    w_284895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 170)
    needs_trunc_284896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 170)
    kwargs_284897 = {}
    # Getting the type of '_truncate' (line 170)
    _truncate_284894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 170)
    _truncate_call_result_284898 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), _truncate_284894, *[w_284895, needs_trunc_284896], **kwargs_284897)
    
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', _truncate_call_result_284898)
    
    # ################# End of 'boxcar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'boxcar' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_284899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'boxcar'
    return stypy_return_type_284899

# Assigning a type to the variable 'boxcar' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 0), 'boxcar', boxcar)

@norecursion
def triang(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 173)
    True_284900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'True')
    defaults = [True_284900]
    # Create a new context for function 'triang'
    module_type_store = module_type_store.open_function_context('triang', 173, 0, False)
    
    # Passed parameters checking function
    triang.stypy_localization = localization
    triang.stypy_type_of_self = None
    triang.stypy_type_store = module_type_store
    triang.stypy_function_name = 'triang'
    triang.stypy_param_names_list = ['M', 'sym']
    triang.stypy_varargs_param_name = None
    triang.stypy_kwargs_param_name = None
    triang.stypy_call_defaults = defaults
    triang.stypy_call_varargs = varargs
    triang.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'triang', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'triang', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'triang(...)' code ##################

    str_284901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, (-1)), 'str', 'Return a triangular window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    See Also\n    --------\n    bartlett : A triangular window that touches zero\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.triang(51)\n    >>> plt.plot(window)\n    >>> plt.title("Triangular window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the triangular window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'M' (line 221)
    M_284903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'M', False)
    # Processing the call keyword arguments (line 221)
    kwargs_284904 = {}
    # Getting the type of '_len_guards' (line 221)
    _len_guards_284902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 221)
    _len_guards_call_result_284905 = invoke(stypy.reporting.localization.Localization(__file__, 221, 7), _len_guards_284902, *[M_284903], **kwargs_284904)
    
    # Testing the type of an if condition (line 221)
    if_condition_284906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), _len_guards_call_result_284905)
    # Assigning a type to the variable 'if_condition_284906' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_284906', if_condition_284906)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'M' (line 222)
    M_284909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'M', False)
    # Processing the call keyword arguments (line 222)
    kwargs_284910 = {}
    # Getting the type of 'np' (line 222)
    np_284907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 222)
    ones_284908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 15), np_284907, 'ones')
    # Calling ones(args, kwargs) (line 222)
    ones_call_result_284911 = invoke(stypy.reporting.localization.Localization(__file__, 222, 15), ones_284908, *[M_284909], **kwargs_284910)
    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'stypy_return_type', ones_call_result_284911)
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 223):
    
    # Assigning a Subscript to a Name (line 223):
    
    # Obtaining the type of the subscript
    int_284912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'int')
    
    # Call to _extend(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'M' (line 223)
    M_284914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'M', False)
    # Getting the type of 'sym' (line 223)
    sym_284915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'sym', False)
    # Processing the call keyword arguments (line 223)
    kwargs_284916 = {}
    # Getting the type of '_extend' (line 223)
    _extend_284913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 223)
    _extend_call_result_284917 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), _extend_284913, *[M_284914, sym_284915], **kwargs_284916)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___284918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), _extend_call_result_284917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_284919 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), getitem___284918, int_284912)
    
    # Assigning a type to the variable 'tuple_var_assignment_284672' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_284672', subscript_call_result_284919)
    
    # Assigning a Subscript to a Name (line 223):
    
    # Obtaining the type of the subscript
    int_284920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 4), 'int')
    
    # Call to _extend(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'M' (line 223)
    M_284922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'M', False)
    # Getting the type of 'sym' (line 223)
    sym_284923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'sym', False)
    # Processing the call keyword arguments (line 223)
    kwargs_284924 = {}
    # Getting the type of '_extend' (line 223)
    _extend_284921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 223)
    _extend_call_result_284925 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), _extend_284921, *[M_284922, sym_284923], **kwargs_284924)
    
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___284926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), _extend_call_result_284925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_284927 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), getitem___284926, int_284920)
    
    # Assigning a type to the variable 'tuple_var_assignment_284673' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_284673', subscript_call_result_284927)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_var_assignment_284672' (line 223)
    tuple_var_assignment_284672_284928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_284672')
    # Assigning a type to the variable 'M' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'M', tuple_var_assignment_284672_284928)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'tuple_var_assignment_284673' (line 223)
    tuple_var_assignment_284673_284929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'tuple_var_assignment_284673')
    # Assigning a type to the variable 'needs_trunc' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 7), 'needs_trunc', tuple_var_assignment_284673_284929)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to arange(...): (line 225)
    # Processing the call arguments (line 225)
    int_284932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 18), 'int')
    # Getting the type of 'M' (line 225)
    M_284933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'M', False)
    int_284934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 26), 'int')
    # Applying the binary operator '+' (line 225)
    result_add_284935 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 22), '+', M_284933, int_284934)
    
    int_284936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 32), 'int')
    # Applying the binary operator '//' (line 225)
    result_floordiv_284937 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 21), '//', result_add_284935, int_284936)
    
    int_284938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 36), 'int')
    # Applying the binary operator '+' (line 225)
    result_add_284939 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 21), '+', result_floordiv_284937, int_284938)
    
    # Processing the call keyword arguments (line 225)
    kwargs_284940 = {}
    # Getting the type of 'np' (line 225)
    np_284930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 225)
    arange_284931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), np_284930, 'arange')
    # Calling arange(args, kwargs) (line 225)
    arange_call_result_284941 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), arange_284931, *[int_284932, result_add_284939], **kwargs_284940)
    
    # Assigning a type to the variable 'n' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'n', arange_call_result_284941)
    
    
    # Getting the type of 'M' (line 226)
    M_284942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'M')
    int_284943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 11), 'int')
    # Applying the binary operator '%' (line 226)
    result_mod_284944 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), '%', M_284942, int_284943)
    
    int_284945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 16), 'int')
    # Applying the binary operator '==' (line 226)
    result_eq_284946 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), '==', result_mod_284944, int_284945)
    
    # Testing the type of an if condition (line 226)
    if_condition_284947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), result_eq_284946)
    # Assigning a type to the variable 'if_condition_284947' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_284947', if_condition_284947)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 227):
    
    # Assigning a BinOp to a Name (line 227):
    int_284948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 13), 'int')
    # Getting the type of 'n' (line 227)
    n_284949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'n')
    # Applying the binary operator '*' (line 227)
    result_mul_284950 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 13), '*', int_284948, n_284949)
    
    float_284951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'float')
    # Applying the binary operator '-' (line 227)
    result_sub_284952 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 13), '-', result_mul_284950, float_284951)
    
    # Getting the type of 'M' (line 227)
    M_284953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'M')
    # Applying the binary operator 'div' (line 227)
    result_div_284954 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 12), 'div', result_sub_284952, M_284953)
    
    # Assigning a type to the variable 'w' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'w', result_div_284954)
    
    # Assigning a Subscript to a Name (line 228):
    
    # Assigning a Subscript to a Name (line 228):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 228)
    tuple_284955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 228)
    # Adding element type (line 228)
    # Getting the type of 'w' (line 228)
    w_284956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 18), tuple_284955, w_284956)
    # Adding element type (line 228)
    
    # Obtaining the type of the subscript
    int_284957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 25), 'int')
    slice_284958 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 228, 21), None, None, int_284957)
    # Getting the type of 'w' (line 228)
    w_284959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 21), 'w')
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___284960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 21), w_284959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_284961 = invoke(stypy.reporting.localization.Localization(__file__, 228, 21), getitem___284960, slice_284958)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 18), tuple_284955, subscript_call_result_284961)
    
    # Getting the type of 'np' (line 228)
    np_284962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'np')
    # Obtaining the member 'r_' of a type (line 228)
    r__284963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), np_284962, 'r_')
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___284964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), r__284963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_284965 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___284964, tuple_284955)
    
    # Assigning a type to the variable 'w' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'w', subscript_call_result_284965)
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 230):
    
    # Assigning a BinOp to a Name (line 230):
    int_284966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 12), 'int')
    # Getting the type of 'n' (line 230)
    n_284967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'n')
    # Applying the binary operator '*' (line 230)
    result_mul_284968 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 12), '*', int_284966, n_284967)
    
    # Getting the type of 'M' (line 230)
    M_284969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'M')
    float_284970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 25), 'float')
    # Applying the binary operator '+' (line 230)
    result_add_284971 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 21), '+', M_284969, float_284970)
    
    # Applying the binary operator 'div' (line 230)
    result_div_284972 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 18), 'div', result_mul_284968, result_add_284971)
    
    # Assigning a type to the variable 'w' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'w', result_div_284972)
    
    # Assigning a Subscript to a Name (line 231):
    
    # Assigning a Subscript to a Name (line 231):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_284973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    # Getting the type of 'w' (line 231)
    w_284974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 18), tuple_284973, w_284974)
    # Adding element type (line 231)
    
    # Obtaining the type of the subscript
    int_284975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 23), 'int')
    int_284976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'int')
    slice_284977 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 231, 21), int_284975, None, int_284976)
    # Getting the type of 'w' (line 231)
    w_284978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'w')
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___284979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 21), w_284978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_284980 = invoke(stypy.reporting.localization.Localization(__file__, 231, 21), getitem___284979, slice_284977)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 18), tuple_284973, subscript_call_result_284980)
    
    # Getting the type of 'np' (line 231)
    np_284981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'np')
    # Obtaining the member 'r_' of a type (line 231)
    r__284982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), np_284981, 'r_')
    # Obtaining the member '__getitem__' of a type (line 231)
    getitem___284983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), r__284982, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
    subscript_call_result_284984 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), getitem___284983, tuple_284973)
    
    # Assigning a type to the variable 'w' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'w', subscript_call_result_284984)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _truncate(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'w' (line 233)
    w_284986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 233)
    needs_trunc_284987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 233)
    kwargs_284988 = {}
    # Getting the type of '_truncate' (line 233)
    _truncate_284985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 233)
    _truncate_call_result_284989 = invoke(stypy.reporting.localization.Localization(__file__, 233, 11), _truncate_284985, *[w_284986, needs_trunc_284987], **kwargs_284988)
    
    # Assigning a type to the variable 'stypy_return_type' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type', _truncate_call_result_284989)
    
    # ################# End of 'triang(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'triang' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_284990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_284990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'triang'
    return stypy_return_type_284990

# Assigning a type to the variable 'triang' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'triang', triang)

@norecursion
def parzen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 236)
    True_284991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'True')
    defaults = [True_284991]
    # Create a new context for function 'parzen'
    module_type_store = module_type_store.open_function_context('parzen', 236, 0, False)
    
    # Passed parameters checking function
    parzen.stypy_localization = localization
    parzen.stypy_type_of_self = None
    parzen.stypy_type_store = module_type_store
    parzen.stypy_function_name = 'parzen'
    parzen.stypy_param_names_list = ['M', 'sym']
    parzen.stypy_varargs_param_name = None
    parzen.stypy_kwargs_param_name = None
    parzen.stypy_call_defaults = defaults
    parzen.stypy_call_varargs = varargs
    parzen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parzen', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parzen', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parzen(...)' code ##################

    str_284992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, (-1)), 'str', 'Return a Parzen window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    References\n    ----------\n    .. [1] E. Parzen, "Mathematical Considerations in the Estimation of\n           Spectra", Technometrics,  Vol. 3, No. 2 (May, 1961), pp. 167-190\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.parzen(51)\n    >>> plt.plot(window)\n    >>> plt.title("Parzen window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Parzen window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'M' (line 285)
    M_284994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 19), 'M', False)
    # Processing the call keyword arguments (line 285)
    kwargs_284995 = {}
    # Getting the type of '_len_guards' (line 285)
    _len_guards_284993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 285)
    _len_guards_call_result_284996 = invoke(stypy.reporting.localization.Localization(__file__, 285, 7), _len_guards_284993, *[M_284994], **kwargs_284995)
    
    # Testing the type of an if condition (line 285)
    if_condition_284997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), _len_guards_call_result_284996)
    # Assigning a type to the variable 'if_condition_284997' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_284997', if_condition_284997)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'M' (line 286)
    M_285000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'M', False)
    # Processing the call keyword arguments (line 286)
    kwargs_285001 = {}
    # Getting the type of 'np' (line 286)
    np_284998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 286)
    ones_284999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 15), np_284998, 'ones')
    # Calling ones(args, kwargs) (line 286)
    ones_call_result_285002 = invoke(stypy.reporting.localization.Localization(__file__, 286, 15), ones_284999, *[M_285000], **kwargs_285001)
    
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'stypy_return_type', ones_call_result_285002)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_285003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to _extend(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'M' (line 287)
    M_285005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 29), 'M', False)
    # Getting the type of 'sym' (line 287)
    sym_285006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'sym', False)
    # Processing the call keyword arguments (line 287)
    kwargs_285007 = {}
    # Getting the type of '_extend' (line 287)
    _extend_285004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 287)
    _extend_call_result_285008 = invoke(stypy.reporting.localization.Localization(__file__, 287, 21), _extend_285004, *[M_285005, sym_285006], **kwargs_285007)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___285009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), _extend_call_result_285008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_285010 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___285009, int_285003)
    
    # Assigning a type to the variable 'tuple_var_assignment_284674' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_284674', subscript_call_result_285010)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_285011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to _extend(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'M' (line 287)
    M_285013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 29), 'M', False)
    # Getting the type of 'sym' (line 287)
    sym_285014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'sym', False)
    # Processing the call keyword arguments (line 287)
    kwargs_285015 = {}
    # Getting the type of '_extend' (line 287)
    _extend_285012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 287)
    _extend_call_result_285016 = invoke(stypy.reporting.localization.Localization(__file__, 287, 21), _extend_285012, *[M_285013, sym_285014], **kwargs_285015)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___285017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), _extend_call_result_285016, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_285018 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___285017, int_285011)
    
    # Assigning a type to the variable 'tuple_var_assignment_284675' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_284675', subscript_call_result_285018)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_284674' (line 287)
    tuple_var_assignment_284674_285019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_284674')
    # Assigning a type to the variable 'M' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'M', tuple_var_assignment_284674_285019)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_284675' (line 287)
    tuple_var_assignment_284675_285020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_284675')
    # Assigning a type to the variable 'needs_trunc' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 'needs_trunc', tuple_var_assignment_284675_285020)
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to arange(...): (line 289)
    # Processing the call arguments (line 289)
    
    # Getting the type of 'M' (line 289)
    M_285023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'M', False)
    int_285024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 24), 'int')
    # Applying the binary operator '-' (line 289)
    result_sub_285025 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 20), '-', M_285023, int_285024)
    
    # Applying the 'usub' unary operator (line 289)
    result___neg___285026 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 18), 'usub', result_sub_285025)
    
    float_285027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'float')
    # Applying the binary operator 'div' (line 289)
    result_div_285028 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 18), 'div', result___neg___285026, float_285027)
    
    # Getting the type of 'M' (line 289)
    M_285029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'M', False)
    int_285030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 39), 'int')
    # Applying the binary operator '-' (line 289)
    result_sub_285031 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 35), '-', M_285029, int_285030)
    
    float_285032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 44), 'float')
    # Applying the binary operator 'div' (line 289)
    result_div_285033 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 34), 'div', result_sub_285031, float_285032)
    
    float_285034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 50), 'float')
    # Applying the binary operator '+' (line 289)
    result_add_285035 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 34), '+', result_div_285033, float_285034)
    
    float_285036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 55), 'float')
    # Processing the call keyword arguments (line 289)
    kwargs_285037 = {}
    # Getting the type of 'np' (line 289)
    np_285021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 289)
    arange_285022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), np_285021, 'arange')
    # Calling arange(args, kwargs) (line 289)
    arange_call_result_285038 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), arange_285022, *[result_div_285028, result_add_285035, float_285036], **kwargs_285037)
    
    # Assigning a type to the variable 'n' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'n', arange_call_result_285038)
    
    # Assigning a Call to a Name (line 290):
    
    # Assigning a Call to a Name (line 290):
    
    # Call to extract(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Getting the type of 'n' (line 290)
    n_285041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'n', False)
    
    # Getting the type of 'M' (line 290)
    M_285042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 26), 'M', False)
    int_285043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 30), 'int')
    # Applying the binary operator '-' (line 290)
    result_sub_285044 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 26), '-', M_285042, int_285043)
    
    # Applying the 'usub' unary operator (line 290)
    result___neg___285045 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 24), 'usub', result_sub_285044)
    
    float_285046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 35), 'float')
    # Applying the binary operator 'div' (line 290)
    result_div_285047 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 24), 'div', result___neg___285045, float_285046)
    
    # Applying the binary operator '<' (line 290)
    result_lt_285048 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 20), '<', n_285041, result_div_285047)
    
    # Getting the type of 'n' (line 290)
    n_285049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 40), 'n', False)
    # Processing the call keyword arguments (line 290)
    kwargs_285050 = {}
    # Getting the type of 'np' (line 290)
    np_285039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 9), 'np', False)
    # Obtaining the member 'extract' of a type (line 290)
    extract_285040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 9), np_285039, 'extract')
    # Calling extract(args, kwargs) (line 290)
    extract_call_result_285051 = invoke(stypy.reporting.localization.Localization(__file__, 290, 9), extract_285040, *[result_lt_285048, n_285049], **kwargs_285050)
    
    # Assigning a type to the variable 'na' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'na', extract_call_result_285051)
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to extract(...): (line 291)
    # Processing the call arguments (line 291)
    
    
    # Call to abs(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'n' (line 291)
    n_285055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'n', False)
    # Processing the call keyword arguments (line 291)
    kwargs_285056 = {}
    # Getting the type of 'abs' (line 291)
    abs_285054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'abs', False)
    # Calling abs(args, kwargs) (line 291)
    abs_call_result_285057 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), abs_285054, *[n_285055], **kwargs_285056)
    
    # Getting the type of 'M' (line 291)
    M_285058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'M', False)
    int_285059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'int')
    # Applying the binary operator '-' (line 291)
    result_sub_285060 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 31), '-', M_285058, int_285059)
    
    float_285061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 40), 'float')
    # Applying the binary operator 'div' (line 291)
    result_div_285062 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 30), 'div', result_sub_285060, float_285061)
    
    # Applying the binary operator '<=' (line 291)
    result_le_285063 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 20), '<=', abs_call_result_285057, result_div_285062)
    
    # Getting the type of 'n' (line 291)
    n_285064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 45), 'n', False)
    # Processing the call keyword arguments (line 291)
    kwargs_285065 = {}
    # Getting the type of 'np' (line 291)
    np_285052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 9), 'np', False)
    # Obtaining the member 'extract' of a type (line 291)
    extract_285053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 9), np_285052, 'extract')
    # Calling extract(args, kwargs) (line 291)
    extract_call_result_285066 = invoke(stypy.reporting.localization.Localization(__file__, 291, 9), extract_285053, *[result_le_285063, n_285064], **kwargs_285065)
    
    # Assigning a type to the variable 'nb' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'nb', extract_call_result_285066)
    
    # Assigning a BinOp to a Name (line 292):
    
    # Assigning a BinOp to a Name (line 292):
    int_285067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 9), 'int')
    int_285068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 14), 'int')
    
    # Call to abs(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'na' (line 292)
    na_285071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 25), 'na', False)
    # Processing the call keyword arguments (line 292)
    kwargs_285072 = {}
    # Getting the type of 'np' (line 292)
    np_285069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 292)
    abs_285070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 18), np_285069, 'abs')
    # Calling abs(args, kwargs) (line 292)
    abs_call_result_285073 = invoke(stypy.reporting.localization.Localization(__file__, 292, 18), abs_285070, *[na_285071], **kwargs_285072)
    
    # Getting the type of 'M' (line 292)
    M_285074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 32), 'M')
    float_285075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 36), 'float')
    # Applying the binary operator 'div' (line 292)
    result_div_285076 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 32), 'div', M_285074, float_285075)
    
    # Applying the binary operator 'div' (line 292)
    result_div_285077 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 18), 'div', abs_call_result_285073, result_div_285076)
    
    # Applying the binary operator '-' (line 292)
    result_sub_285078 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 14), '-', int_285068, result_div_285077)
    
    float_285079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 45), 'float')
    # Applying the binary operator '**' (line 292)
    result_pow_285080 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 13), '**', result_sub_285078, float_285079)
    
    # Applying the binary operator '*' (line 292)
    result_mul_285081 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 9), '*', int_285067, result_pow_285080)
    
    # Assigning a type to the variable 'wa' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'wa', result_mul_285081)
    
    # Assigning a BinOp to a Name (line 293):
    
    # Assigning a BinOp to a Name (line 293):
    int_285082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 10), 'int')
    int_285083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 14), 'int')
    
    # Call to abs(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'nb' (line 293)
    nb_285086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'nb', False)
    # Processing the call keyword arguments (line 293)
    kwargs_285087 = {}
    # Getting the type of 'np' (line 293)
    np_285084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'np', False)
    # Obtaining the member 'abs' of a type (line 293)
    abs_285085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 19), np_285084, 'abs')
    # Calling abs(args, kwargs) (line 293)
    abs_call_result_285088 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), abs_285085, *[nb_285086], **kwargs_285087)
    
    # Getting the type of 'M' (line 293)
    M_285089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 33), 'M')
    float_285090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 37), 'float')
    # Applying the binary operator 'div' (line 293)
    result_div_285091 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 33), 'div', M_285089, float_285090)
    
    # Applying the binary operator 'div' (line 293)
    result_div_285092 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 19), 'div', abs_call_result_285088, result_div_285091)
    
    float_285093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 46), 'float')
    # Applying the binary operator '**' (line 293)
    result_pow_285094 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 18), '**', result_div_285092, float_285093)
    
    # Applying the binary operator '*' (line 293)
    result_mul_285095 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 14), '*', int_285083, result_pow_285094)
    
    # Applying the binary operator '-' (line 293)
    result_sub_285096 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 10), '-', int_285082, result_mul_285095)
    
    int_285097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 10), 'int')
    
    # Call to abs(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'nb' (line 294)
    nb_285100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'nb', False)
    # Processing the call keyword arguments (line 294)
    kwargs_285101 = {}
    # Getting the type of 'np' (line 294)
    np_285098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 'np', False)
    # Obtaining the member 'abs' of a type (line 294)
    abs_285099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 15), np_285098, 'abs')
    # Calling abs(args, kwargs) (line 294)
    abs_call_result_285102 = invoke(stypy.reporting.localization.Localization(__file__, 294, 15), abs_285099, *[nb_285100], **kwargs_285101)
    
    # Getting the type of 'M' (line 294)
    M_285103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 29), 'M')
    float_285104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 33), 'float')
    # Applying the binary operator 'div' (line 294)
    result_div_285105 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 29), 'div', M_285103, float_285104)
    
    # Applying the binary operator 'div' (line 294)
    result_div_285106 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), 'div', abs_call_result_285102, result_div_285105)
    
    float_285107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 42), 'float')
    # Applying the binary operator '**' (line 294)
    result_pow_285108 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 14), '**', result_div_285106, float_285107)
    
    # Applying the binary operator '*' (line 294)
    result_mul_285109 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 10), '*', int_285097, result_pow_285108)
    
    # Applying the binary operator '+' (line 293)
    result_add_285110 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 50), '+', result_sub_285096, result_mul_285109)
    
    # Assigning a type to the variable 'wb' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'wb', result_add_285110)
    
    # Assigning a Subscript to a Name (line 295):
    
    # Assigning a Subscript to a Name (line 295):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 295)
    tuple_285111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 295)
    # Adding element type (line 295)
    # Getting the type of 'wa' (line 295)
    wa_285112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'wa')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 14), tuple_285111, wa_285112)
    # Adding element type (line 295)
    # Getting the type of 'wb' (line 295)
    wb_285113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'wb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 14), tuple_285111, wb_285113)
    # Adding element type (line 295)
    
    # Obtaining the type of the subscript
    int_285114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 27), 'int')
    slice_285115 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 295, 22), None, None, int_285114)
    # Getting the type of 'wa' (line 295)
    wa_285116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'wa')
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___285117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 22), wa_285116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_285118 = invoke(stypy.reporting.localization.Localization(__file__, 295, 22), getitem___285117, slice_285115)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 14), tuple_285111, subscript_call_result_285118)
    
    # Getting the type of 'np' (line 295)
    np_285119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'np')
    # Obtaining the member 'r_' of a type (line 295)
    r__285120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), np_285119, 'r_')
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___285121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), r__285120, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_285122 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), getitem___285121, tuple_285111)
    
    # Assigning a type to the variable 'w' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'w', subscript_call_result_285122)
    
    # Call to _truncate(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'w' (line 297)
    w_285124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 297)
    needs_trunc_285125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 297)
    kwargs_285126 = {}
    # Getting the type of '_truncate' (line 297)
    _truncate_285123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 297)
    _truncate_call_result_285127 = invoke(stypy.reporting.localization.Localization(__file__, 297, 11), _truncate_285123, *[w_285124, needs_trunc_285125], **kwargs_285126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'stypy_return_type', _truncate_call_result_285127)
    
    # ################# End of 'parzen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parzen' in the type store
    # Getting the type of 'stypy_return_type' (line 236)
    stypy_return_type_285128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parzen'
    return stypy_return_type_285128

# Assigning a type to the variable 'parzen' (line 236)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 0), 'parzen', parzen)

@norecursion
def bohman(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 300)
    True_285129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 18), 'True')
    defaults = [True_285129]
    # Create a new context for function 'bohman'
    module_type_store = module_type_store.open_function_context('bohman', 300, 0, False)
    
    # Passed parameters checking function
    bohman.stypy_localization = localization
    bohman.stypy_type_of_self = None
    bohman.stypy_type_store = module_type_store
    bohman.stypy_function_name = 'bohman'
    bohman.stypy_param_names_list = ['M', 'sym']
    bohman.stypy_varargs_param_name = None
    bohman.stypy_kwargs_param_name = None
    bohman.stypy_call_defaults = defaults
    bohman.stypy_call_varargs = varargs
    bohman.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bohman', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bohman', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bohman(...)' code ##################

    str_285130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, (-1)), 'str', 'Return a Bohman window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.bohman(51)\n    >>> plt.plot(window)\n    >>> plt.title("Bohman window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Bohman window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'M' (line 344)
    M_285132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'M', False)
    # Processing the call keyword arguments (line 344)
    kwargs_285133 = {}
    # Getting the type of '_len_guards' (line 344)
    _len_guards_285131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 344)
    _len_guards_call_result_285134 = invoke(stypy.reporting.localization.Localization(__file__, 344, 7), _len_guards_285131, *[M_285132], **kwargs_285133)
    
    # Testing the type of an if condition (line 344)
    if_condition_285135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 4), _len_guards_call_result_285134)
    # Assigning a type to the variable 'if_condition_285135' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'if_condition_285135', if_condition_285135)
    # SSA begins for if statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'M' (line 345)
    M_285138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'M', False)
    # Processing the call keyword arguments (line 345)
    kwargs_285139 = {}
    # Getting the type of 'np' (line 345)
    np_285136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 345)
    ones_285137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), np_285136, 'ones')
    # Calling ones(args, kwargs) (line 345)
    ones_call_result_285140 = invoke(stypy.reporting.localization.Localization(__file__, 345, 15), ones_285137, *[M_285138], **kwargs_285139)
    
    # Assigning a type to the variable 'stypy_return_type' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'stypy_return_type', ones_call_result_285140)
    # SSA join for if statement (line 344)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 346):
    
    # Assigning a Subscript to a Name (line 346):
    
    # Obtaining the type of the subscript
    int_285141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 4), 'int')
    
    # Call to _extend(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'M' (line 346)
    M_285143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'M', False)
    # Getting the type of 'sym' (line 346)
    sym_285144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 32), 'sym', False)
    # Processing the call keyword arguments (line 346)
    kwargs_285145 = {}
    # Getting the type of '_extend' (line 346)
    _extend_285142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 346)
    _extend_call_result_285146 = invoke(stypy.reporting.localization.Localization(__file__, 346, 21), _extend_285142, *[M_285143, sym_285144], **kwargs_285145)
    
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___285147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 4), _extend_call_result_285146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_285148 = invoke(stypy.reporting.localization.Localization(__file__, 346, 4), getitem___285147, int_285141)
    
    # Assigning a type to the variable 'tuple_var_assignment_284676' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_284676', subscript_call_result_285148)
    
    # Assigning a Subscript to a Name (line 346):
    
    # Obtaining the type of the subscript
    int_285149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 4), 'int')
    
    # Call to _extend(...): (line 346)
    # Processing the call arguments (line 346)
    # Getting the type of 'M' (line 346)
    M_285151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 29), 'M', False)
    # Getting the type of 'sym' (line 346)
    sym_285152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 32), 'sym', False)
    # Processing the call keyword arguments (line 346)
    kwargs_285153 = {}
    # Getting the type of '_extend' (line 346)
    _extend_285150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 346)
    _extend_call_result_285154 = invoke(stypy.reporting.localization.Localization(__file__, 346, 21), _extend_285150, *[M_285151, sym_285152], **kwargs_285153)
    
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___285155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 4), _extend_call_result_285154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_285156 = invoke(stypy.reporting.localization.Localization(__file__, 346, 4), getitem___285155, int_285149)
    
    # Assigning a type to the variable 'tuple_var_assignment_284677' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_284677', subscript_call_result_285156)
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'tuple_var_assignment_284676' (line 346)
    tuple_var_assignment_284676_285157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_284676')
    # Assigning a type to the variable 'M' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'M', tuple_var_assignment_284676_285157)
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'tuple_var_assignment_284677' (line 346)
    tuple_var_assignment_284677_285158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_284677')
    # Assigning a type to the variable 'needs_trunc' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 7), 'needs_trunc', tuple_var_assignment_284677_285158)
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to abs(...): (line 348)
    # Processing the call arguments (line 348)
    
    # Obtaining the type of the subscript
    int_285161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 39), 'int')
    int_285162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 41), 'int')
    slice_285163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 17), int_285161, int_285162, None)
    
    # Call to linspace(...): (line 348)
    # Processing the call arguments (line 348)
    int_285166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'int')
    int_285167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 33), 'int')
    # Getting the type of 'M' (line 348)
    M_285168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 36), 'M', False)
    # Processing the call keyword arguments (line 348)
    kwargs_285169 = {}
    # Getting the type of 'np' (line 348)
    np_285164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'np', False)
    # Obtaining the member 'linspace' of a type (line 348)
    linspace_285165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), np_285164, 'linspace')
    # Calling linspace(args, kwargs) (line 348)
    linspace_call_result_285170 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), linspace_285165, *[int_285166, int_285167, M_285168], **kwargs_285169)
    
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___285171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 17), linspace_call_result_285170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_285172 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), getitem___285171, slice_285163)
    
    # Processing the call keyword arguments (line 348)
    kwargs_285173 = {}
    # Getting the type of 'np' (line 348)
    np_285159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 348)
    abs_285160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 10), np_285159, 'abs')
    # Calling abs(args, kwargs) (line 348)
    abs_call_result_285174 = invoke(stypy.reporting.localization.Localization(__file__, 348, 10), abs_285160, *[subscript_call_result_285172], **kwargs_285173)
    
    # Assigning a type to the variable 'fac' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'fac', abs_call_result_285174)
    
    # Assigning a BinOp to a Name (line 349):
    
    # Assigning a BinOp to a Name (line 349):
    int_285175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 9), 'int')
    # Getting the type of 'fac' (line 349)
    fac_285176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'fac')
    # Applying the binary operator '-' (line 349)
    result_sub_285177 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 9), '-', int_285175, fac_285176)
    
    
    # Call to cos(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'np' (line 349)
    np_285180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'np', False)
    # Obtaining the member 'pi' of a type (line 349)
    pi_285181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 27), np_285180, 'pi')
    # Getting the type of 'fac' (line 349)
    fac_285182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'fac', False)
    # Applying the binary operator '*' (line 349)
    result_mul_285183 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 27), '*', pi_285181, fac_285182)
    
    # Processing the call keyword arguments (line 349)
    kwargs_285184 = {}
    # Getting the type of 'np' (line 349)
    np_285178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 20), 'np', False)
    # Obtaining the member 'cos' of a type (line 349)
    cos_285179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 20), np_285178, 'cos')
    # Calling cos(args, kwargs) (line 349)
    cos_call_result_285185 = invoke(stypy.reporting.localization.Localization(__file__, 349, 20), cos_285179, *[result_mul_285183], **kwargs_285184)
    
    # Applying the binary operator '*' (line 349)
    result_mul_285186 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 8), '*', result_sub_285177, cos_call_result_285185)
    
    float_285187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 42), 'float')
    # Getting the type of 'np' (line 349)
    np_285188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 48), 'np')
    # Obtaining the member 'pi' of a type (line 349)
    pi_285189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 48), np_285188, 'pi')
    # Applying the binary operator 'div' (line 349)
    result_div_285190 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 42), 'div', float_285187, pi_285189)
    
    
    # Call to sin(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'np' (line 349)
    np_285193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 63), 'np', False)
    # Obtaining the member 'pi' of a type (line 349)
    pi_285194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 63), np_285193, 'pi')
    # Getting the type of 'fac' (line 349)
    fac_285195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 71), 'fac', False)
    # Applying the binary operator '*' (line 349)
    result_mul_285196 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 63), '*', pi_285194, fac_285195)
    
    # Processing the call keyword arguments (line 349)
    kwargs_285197 = {}
    # Getting the type of 'np' (line 349)
    np_285191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 56), 'np', False)
    # Obtaining the member 'sin' of a type (line 349)
    sin_285192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 56), np_285191, 'sin')
    # Calling sin(args, kwargs) (line 349)
    sin_call_result_285198 = invoke(stypy.reporting.localization.Localization(__file__, 349, 56), sin_285192, *[result_mul_285196], **kwargs_285197)
    
    # Applying the binary operator '*' (line 349)
    result_mul_285199 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 54), '*', result_div_285190, sin_call_result_285198)
    
    # Applying the binary operator '+' (line 349)
    result_add_285200 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 8), '+', result_mul_285186, result_mul_285199)
    
    # Assigning a type to the variable 'w' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'w', result_add_285200)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 350)
    tuple_285201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 350)
    # Adding element type (line 350)
    int_285202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 14), tuple_285201, int_285202)
    # Adding element type (line 350)
    # Getting the type of 'w' (line 350)
    w_285203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 17), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 14), tuple_285201, w_285203)
    # Adding element type (line 350)
    int_285204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 14), tuple_285201, int_285204)
    
    # Getting the type of 'np' (line 350)
    np_285205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'np')
    # Obtaining the member 'r_' of a type (line 350)
    r__285206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), np_285205, 'r_')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___285207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), r__285206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_285208 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___285207, tuple_285201)
    
    # Assigning a type to the variable 'w' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'w', subscript_call_result_285208)
    
    # Call to _truncate(...): (line 352)
    # Processing the call arguments (line 352)
    # Getting the type of 'w' (line 352)
    w_285210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 352)
    needs_trunc_285211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 352)
    kwargs_285212 = {}
    # Getting the type of '_truncate' (line 352)
    _truncate_285209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 352)
    _truncate_call_result_285213 = invoke(stypy.reporting.localization.Localization(__file__, 352, 11), _truncate_285209, *[w_285210, needs_trunc_285211], **kwargs_285212)
    
    # Assigning a type to the variable 'stypy_return_type' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type', _truncate_call_result_285213)
    
    # ################# End of 'bohman(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bohman' in the type store
    # Getting the type of 'stypy_return_type' (line 300)
    stypy_return_type_285214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285214)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bohman'
    return stypy_return_type_285214

# Assigning a type to the variable 'bohman' (line 300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'bohman', bohman)

@norecursion
def blackman(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 355)
    True_285215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 20), 'True')
    defaults = [True_285215]
    # Create a new context for function 'blackman'
    module_type_store = module_type_store.open_function_context('blackman', 355, 0, False)
    
    # Passed parameters checking function
    blackman.stypy_localization = localization
    blackman.stypy_type_of_self = None
    blackman.stypy_type_store = module_type_store
    blackman.stypy_function_name = 'blackman'
    blackman.stypy_param_names_list = ['M', 'sym']
    blackman.stypy_varargs_param_name = None
    blackman.stypy_kwargs_param_name = None
    blackman.stypy_call_defaults = defaults
    blackman.stypy_call_varargs = varargs
    blackman.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'blackman', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'blackman', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'blackman(...)' code ##################

    str_285216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', '\n    Return a Blackman window.\n\n    The Blackman window is a taper formed by using the first three terms of\n    a summation of cosines. It was designed to have close to the minimal\n    leakage possible.  It is close to optimal, only slightly worse than a\n    Kaiser window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Blackman window is defined as\n\n    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)\n\n    The "exact Blackman" window was designed to null out the third and fourth\n    sidelobes, but has discontinuities at the boundaries, resulting in a\n    6 dB/oct fall-off.  This window is an approximation of the "exact" window,\n    which does not null the sidelobes as well, but is smooth at the edges,\n    improving the fall-off rate to 18 dB/oct. [3]_\n\n    Most references to the Blackman window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function. It is known as a\n    "near optimal" tapering function, almost as good (by some measures)\n    as the Kaiser window.\n\n    References\n    ----------\n    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power\n           spectra, Dover Publications, New York.\n    .. [2] Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.\n           Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.\n    .. [3] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic\n           Analysis with the Discrete Fourier Transform". Proceedings of the\n           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.blackman(51)\n    >>> plt.plot(window)\n    >>> plt.title("Blackman window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Blackman window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'M' (line 436)
    M_285218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 19), 'M', False)
    # Processing the call keyword arguments (line 436)
    kwargs_285219 = {}
    # Getting the type of '_len_guards' (line 436)
    _len_guards_285217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 436)
    _len_guards_call_result_285220 = invoke(stypy.reporting.localization.Localization(__file__, 436, 7), _len_guards_285217, *[M_285218], **kwargs_285219)
    
    # Testing the type of an if condition (line 436)
    if_condition_285221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), _len_guards_call_result_285220)
    # Assigning a type to the variable 'if_condition_285221' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_285221', if_condition_285221)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'M' (line 437)
    M_285224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 23), 'M', False)
    # Processing the call keyword arguments (line 437)
    kwargs_285225 = {}
    # Getting the type of 'np' (line 437)
    np_285222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 437)
    ones_285223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 15), np_285222, 'ones')
    # Calling ones(args, kwargs) (line 437)
    ones_call_result_285226 = invoke(stypy.reporting.localization.Localization(__file__, 437, 15), ones_285223, *[M_285224], **kwargs_285225)
    
    # Assigning a type to the variable 'stypy_return_type' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', ones_call_result_285226)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 438):
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_285227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 4), 'int')
    
    # Call to _extend(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'M' (line 438)
    M_285229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'M', False)
    # Getting the type of 'sym' (line 438)
    sym_285230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'sym', False)
    # Processing the call keyword arguments (line 438)
    kwargs_285231 = {}
    # Getting the type of '_extend' (line 438)
    _extend_285228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 438)
    _extend_call_result_285232 = invoke(stypy.reporting.localization.Localization(__file__, 438, 21), _extend_285228, *[M_285229, sym_285230], **kwargs_285231)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___285233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 4), _extend_call_result_285232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_285234 = invoke(stypy.reporting.localization.Localization(__file__, 438, 4), getitem___285233, int_285227)
    
    # Assigning a type to the variable 'tuple_var_assignment_284678' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'tuple_var_assignment_284678', subscript_call_result_285234)
    
    # Assigning a Subscript to a Name (line 438):
    
    # Obtaining the type of the subscript
    int_285235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 4), 'int')
    
    # Call to _extend(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'M' (line 438)
    M_285237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 29), 'M', False)
    # Getting the type of 'sym' (line 438)
    sym_285238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 32), 'sym', False)
    # Processing the call keyword arguments (line 438)
    kwargs_285239 = {}
    # Getting the type of '_extend' (line 438)
    _extend_285236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 438)
    _extend_call_result_285240 = invoke(stypy.reporting.localization.Localization(__file__, 438, 21), _extend_285236, *[M_285237, sym_285238], **kwargs_285239)
    
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___285241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 4), _extend_call_result_285240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_285242 = invoke(stypy.reporting.localization.Localization(__file__, 438, 4), getitem___285241, int_285235)
    
    # Assigning a type to the variable 'tuple_var_assignment_284679' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'tuple_var_assignment_284679', subscript_call_result_285242)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_284678' (line 438)
    tuple_var_assignment_284678_285243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'tuple_var_assignment_284678')
    # Assigning a type to the variable 'M' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'M', tuple_var_assignment_284678_285243)
    
    # Assigning a Name to a Name (line 438):
    # Getting the type of 'tuple_var_assignment_284679' (line 438)
    tuple_var_assignment_284679_285244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'tuple_var_assignment_284679')
    # Assigning a type to the variable 'needs_trunc' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 7), 'needs_trunc', tuple_var_assignment_284679_285244)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to _cos_win(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'M' (line 440)
    M_285246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'M', False)
    
    # Obtaining an instance of the builtin type 'list' (line 440)
    list_285247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 440)
    # Adding element type (line 440)
    float_285248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 20), list_285247, float_285248)
    # Adding element type (line 440)
    float_285249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 20), list_285247, float_285249)
    # Adding element type (line 440)
    float_285250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 20), list_285247, float_285250)
    
    # Processing the call keyword arguments (line 440)
    kwargs_285251 = {}
    # Getting the type of '_cos_win' (line 440)
    _cos_win_285245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 440)
    _cos_win_call_result_285252 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), _cos_win_285245, *[M_285246, list_285247], **kwargs_285251)
    
    # Assigning a type to the variable 'w' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'w', _cos_win_call_result_285252)
    
    # Call to _truncate(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'w' (line 442)
    w_285254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 442)
    needs_trunc_285255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 442)
    kwargs_285256 = {}
    # Getting the type of '_truncate' (line 442)
    _truncate_285253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 442)
    _truncate_call_result_285257 = invoke(stypy.reporting.localization.Localization(__file__, 442, 11), _truncate_285253, *[w_285254, needs_trunc_285255], **kwargs_285256)
    
    # Assigning a type to the variable 'stypy_return_type' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type', _truncate_call_result_285257)
    
    # ################# End of 'blackman(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'blackman' in the type store
    # Getting the type of 'stypy_return_type' (line 355)
    stypy_return_type_285258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'blackman'
    return stypy_return_type_285258

# Assigning a type to the variable 'blackman' (line 355)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'blackman', blackman)

@norecursion
def nuttall(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 445)
    True_285259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 19), 'True')
    defaults = [True_285259]
    # Create a new context for function 'nuttall'
    module_type_store = module_type_store.open_function_context('nuttall', 445, 0, False)
    
    # Passed parameters checking function
    nuttall.stypy_localization = localization
    nuttall.stypy_type_of_self = None
    nuttall.stypy_type_store = module_type_store
    nuttall.stypy_function_name = 'nuttall'
    nuttall.stypy_param_names_list = ['M', 'sym']
    nuttall.stypy_varargs_param_name = None
    nuttall.stypy_kwargs_param_name = None
    nuttall.stypy_call_defaults = defaults
    nuttall.stypy_call_varargs = varargs
    nuttall.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nuttall', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nuttall', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nuttall(...)' code ##################

    str_285260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, (-1)), 'str', 'Return a minimum 4-term Blackman-Harris window according to Nuttall.\n\n    This variation is called "Nuttall4c" by Heinzel. [2]_\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    References\n    ----------\n    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE\n           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,\n           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.\n    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the\n           Discrete Fourier transform (DFT), including a comprehensive list of\n           window functions and some new flat-top windows", February 15, 2002\n           https://holometer.fnal.gov/GH_FFT.pdf\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.nuttall(51)\n    >>> plt.plot(window)\n    >>> plt.title("Nuttall window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Nuttall window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'M' (line 501)
    M_285262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'M', False)
    # Processing the call keyword arguments (line 501)
    kwargs_285263 = {}
    # Getting the type of '_len_guards' (line 501)
    _len_guards_285261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 501)
    _len_guards_call_result_285264 = invoke(stypy.reporting.localization.Localization(__file__, 501, 7), _len_guards_285261, *[M_285262], **kwargs_285263)
    
    # Testing the type of an if condition (line 501)
    if_condition_285265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), _len_guards_call_result_285264)
    # Assigning a type to the variable 'if_condition_285265' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_285265', if_condition_285265)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'M' (line 502)
    M_285268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 23), 'M', False)
    # Processing the call keyword arguments (line 502)
    kwargs_285269 = {}
    # Getting the type of 'np' (line 502)
    np_285266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 502)
    ones_285267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 15), np_285266, 'ones')
    # Calling ones(args, kwargs) (line 502)
    ones_call_result_285270 = invoke(stypy.reporting.localization.Localization(__file__, 502, 15), ones_285267, *[M_285268], **kwargs_285269)
    
    # Assigning a type to the variable 'stypy_return_type' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'stypy_return_type', ones_call_result_285270)
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 503):
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_285271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 4), 'int')
    
    # Call to _extend(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'M' (line 503)
    M_285273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 29), 'M', False)
    # Getting the type of 'sym' (line 503)
    sym_285274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'sym', False)
    # Processing the call keyword arguments (line 503)
    kwargs_285275 = {}
    # Getting the type of '_extend' (line 503)
    _extend_285272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 503)
    _extend_call_result_285276 = invoke(stypy.reporting.localization.Localization(__file__, 503, 21), _extend_285272, *[M_285273, sym_285274], **kwargs_285275)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___285277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 4), _extend_call_result_285276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_285278 = invoke(stypy.reporting.localization.Localization(__file__, 503, 4), getitem___285277, int_285271)
    
    # Assigning a type to the variable 'tuple_var_assignment_284680' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'tuple_var_assignment_284680', subscript_call_result_285278)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_285279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 4), 'int')
    
    # Call to _extend(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'M' (line 503)
    M_285281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 29), 'M', False)
    # Getting the type of 'sym' (line 503)
    sym_285282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 32), 'sym', False)
    # Processing the call keyword arguments (line 503)
    kwargs_285283 = {}
    # Getting the type of '_extend' (line 503)
    _extend_285280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 503)
    _extend_call_result_285284 = invoke(stypy.reporting.localization.Localization(__file__, 503, 21), _extend_285280, *[M_285281, sym_285282], **kwargs_285283)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___285285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 4), _extend_call_result_285284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_285286 = invoke(stypy.reporting.localization.Localization(__file__, 503, 4), getitem___285285, int_285279)
    
    # Assigning a type to the variable 'tuple_var_assignment_284681' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'tuple_var_assignment_284681', subscript_call_result_285286)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_284680' (line 503)
    tuple_var_assignment_284680_285287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'tuple_var_assignment_284680')
    # Assigning a type to the variable 'M' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'M', tuple_var_assignment_284680_285287)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_284681' (line 503)
    tuple_var_assignment_284681_285288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 4), 'tuple_var_assignment_284681')
    # Assigning a type to the variable 'needs_trunc' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 7), 'needs_trunc', tuple_var_assignment_284681_285288)
    
    # Assigning a Call to a Name (line 505):
    
    # Assigning a Call to a Name (line 505):
    
    # Call to _cos_win(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'M' (line 505)
    M_285290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'M', False)
    
    # Obtaining an instance of the builtin type 'list' (line 505)
    list_285291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 505)
    # Adding element type (line 505)
    float_285292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 20), list_285291, float_285292)
    # Adding element type (line 505)
    float_285293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 20), list_285291, float_285293)
    # Adding element type (line 505)
    float_285294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 20), list_285291, float_285294)
    # Adding element type (line 505)
    float_285295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 54), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 20), list_285291, float_285295)
    
    # Processing the call keyword arguments (line 505)
    kwargs_285296 = {}
    # Getting the type of '_cos_win' (line 505)
    _cos_win_285289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 505)
    _cos_win_call_result_285297 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), _cos_win_285289, *[M_285290, list_285291], **kwargs_285296)
    
    # Assigning a type to the variable 'w' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'w', _cos_win_call_result_285297)
    
    # Call to _truncate(...): (line 507)
    # Processing the call arguments (line 507)
    # Getting the type of 'w' (line 507)
    w_285299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 507)
    needs_trunc_285300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 507)
    kwargs_285301 = {}
    # Getting the type of '_truncate' (line 507)
    _truncate_285298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 507)
    _truncate_call_result_285302 = invoke(stypy.reporting.localization.Localization(__file__, 507, 11), _truncate_285298, *[w_285299, needs_trunc_285300], **kwargs_285301)
    
    # Assigning a type to the variable 'stypy_return_type' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type', _truncate_call_result_285302)
    
    # ################# End of 'nuttall(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nuttall' in the type store
    # Getting the type of 'stypy_return_type' (line 445)
    stypy_return_type_285303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285303)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nuttall'
    return stypy_return_type_285303

# Assigning a type to the variable 'nuttall' (line 445)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 0), 'nuttall', nuttall)

@norecursion
def blackmanharris(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 510)
    True_285304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'True')
    defaults = [True_285304]
    # Create a new context for function 'blackmanharris'
    module_type_store = module_type_store.open_function_context('blackmanharris', 510, 0, False)
    
    # Passed parameters checking function
    blackmanharris.stypy_localization = localization
    blackmanharris.stypy_type_of_self = None
    blackmanharris.stypy_type_store = module_type_store
    blackmanharris.stypy_function_name = 'blackmanharris'
    blackmanharris.stypy_param_names_list = ['M', 'sym']
    blackmanharris.stypy_varargs_param_name = None
    blackmanharris.stypy_kwargs_param_name = None
    blackmanharris.stypy_call_defaults = defaults
    blackmanharris.stypy_call_varargs = varargs
    blackmanharris.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'blackmanharris', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'blackmanharris', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'blackmanharris(...)' code ##################

    str_285305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, (-1)), 'str', 'Return a minimum 4-term Blackman-Harris window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.blackmanharris(51)\n    >>> plt.plot(window)\n    >>> plt.title("Blackman-Harris window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Blackman-Harris window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'M' (line 554)
    M_285307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'M', False)
    # Processing the call keyword arguments (line 554)
    kwargs_285308 = {}
    # Getting the type of '_len_guards' (line 554)
    _len_guards_285306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 554)
    _len_guards_call_result_285309 = invoke(stypy.reporting.localization.Localization(__file__, 554, 7), _len_guards_285306, *[M_285307], **kwargs_285308)
    
    # Testing the type of an if condition (line 554)
    if_condition_285310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), _len_guards_call_result_285309)
    # Assigning a type to the variable 'if_condition_285310' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_285310', if_condition_285310)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'M' (line 555)
    M_285313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 23), 'M', False)
    # Processing the call keyword arguments (line 555)
    kwargs_285314 = {}
    # Getting the type of 'np' (line 555)
    np_285311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 555)
    ones_285312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 15), np_285311, 'ones')
    # Calling ones(args, kwargs) (line 555)
    ones_call_result_285315 = invoke(stypy.reporting.localization.Localization(__file__, 555, 15), ones_285312, *[M_285313], **kwargs_285314)
    
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'stypy_return_type', ones_call_result_285315)
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 556):
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_285316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 4), 'int')
    
    # Call to _extend(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'M' (line 556)
    M_285318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 29), 'M', False)
    # Getting the type of 'sym' (line 556)
    sym_285319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 32), 'sym', False)
    # Processing the call keyword arguments (line 556)
    kwargs_285320 = {}
    # Getting the type of '_extend' (line 556)
    _extend_285317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 556)
    _extend_call_result_285321 = invoke(stypy.reporting.localization.Localization(__file__, 556, 21), _extend_285317, *[M_285318, sym_285319], **kwargs_285320)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___285322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 4), _extend_call_result_285321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_285323 = invoke(stypy.reporting.localization.Localization(__file__, 556, 4), getitem___285322, int_285316)
    
    # Assigning a type to the variable 'tuple_var_assignment_284682' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'tuple_var_assignment_284682', subscript_call_result_285323)
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_285324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 4), 'int')
    
    # Call to _extend(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'M' (line 556)
    M_285326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 29), 'M', False)
    # Getting the type of 'sym' (line 556)
    sym_285327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 32), 'sym', False)
    # Processing the call keyword arguments (line 556)
    kwargs_285328 = {}
    # Getting the type of '_extend' (line 556)
    _extend_285325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 556)
    _extend_call_result_285329 = invoke(stypy.reporting.localization.Localization(__file__, 556, 21), _extend_285325, *[M_285326, sym_285327], **kwargs_285328)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___285330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 4), _extend_call_result_285329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_285331 = invoke(stypy.reporting.localization.Localization(__file__, 556, 4), getitem___285330, int_285324)
    
    # Assigning a type to the variable 'tuple_var_assignment_284683' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'tuple_var_assignment_284683', subscript_call_result_285331)
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'tuple_var_assignment_284682' (line 556)
    tuple_var_assignment_284682_285332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'tuple_var_assignment_284682')
    # Assigning a type to the variable 'M' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'M', tuple_var_assignment_284682_285332)
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'tuple_var_assignment_284683' (line 556)
    tuple_var_assignment_284683_285333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'tuple_var_assignment_284683')
    # Assigning a type to the variable 'needs_trunc' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 7), 'needs_trunc', tuple_var_assignment_284683_285333)
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to _cos_win(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'M' (line 558)
    M_285335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 17), 'M', False)
    
    # Obtaining an instance of the builtin type 'list' (line 558)
    list_285336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 558)
    # Adding element type (line 558)
    float_285337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 20), list_285336, float_285337)
    # Adding element type (line 558)
    float_285338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 20), list_285336, float_285338)
    # Adding element type (line 558)
    float_285339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 20), list_285336, float_285339)
    # Adding element type (line 558)
    float_285340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 20), list_285336, float_285340)
    
    # Processing the call keyword arguments (line 558)
    kwargs_285341 = {}
    # Getting the type of '_cos_win' (line 558)
    _cos_win_285334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 558)
    _cos_win_call_result_285342 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), _cos_win_285334, *[M_285335, list_285336], **kwargs_285341)
    
    # Assigning a type to the variable 'w' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'w', _cos_win_call_result_285342)
    
    # Call to _truncate(...): (line 560)
    # Processing the call arguments (line 560)
    # Getting the type of 'w' (line 560)
    w_285344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 560)
    needs_trunc_285345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 560)
    kwargs_285346 = {}
    # Getting the type of '_truncate' (line 560)
    _truncate_285343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 560)
    _truncate_call_result_285347 = invoke(stypy.reporting.localization.Localization(__file__, 560, 11), _truncate_285343, *[w_285344, needs_trunc_285345], **kwargs_285346)
    
    # Assigning a type to the variable 'stypy_return_type' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type', _truncate_call_result_285347)
    
    # ################# End of 'blackmanharris(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'blackmanharris' in the type store
    # Getting the type of 'stypy_return_type' (line 510)
    stypy_return_type_285348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285348)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'blackmanharris'
    return stypy_return_type_285348

# Assigning a type to the variable 'blackmanharris' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'blackmanharris', blackmanharris)

@norecursion
def flattop(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 563)
    True_285349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 19), 'True')
    defaults = [True_285349]
    # Create a new context for function 'flattop'
    module_type_store = module_type_store.open_function_context('flattop', 563, 0, False)
    
    # Passed parameters checking function
    flattop.stypy_localization = localization
    flattop.stypy_type_of_self = None
    flattop.stypy_type_store = module_type_store
    flattop.stypy_function_name = 'flattop'
    flattop.stypy_param_names_list = ['M', 'sym']
    flattop.stypy_varargs_param_name = None
    flattop.stypy_kwargs_param_name = None
    flattop.stypy_call_defaults = defaults
    flattop.stypy_call_varargs = varargs
    flattop.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flattop', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flattop', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flattop(...)' code ##################

    str_285350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, (-1)), 'str', 'Return a flat top window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    Flat top windows are used for taking accurate measurements of signal\n    amplitude in the frequency domain, with minimal scalloping error from the\n    center of a frequency bin to its edges, compared to others.  This is a\n    5th-order cosine window, with the 5 terms optimized to make the main lobe\n    maximally flat. [1]_\n\n    References\n    ----------\n    .. [1] D\'Antona, Gabriele, and A. Ferrero, "Digital Signal Processing for\n           Measurement Systems", Springer Media, 2006, p. 70\n           :doi:`10.1007/0-387-28666-7`.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.flattop(51)\n    >>> plt.plot(window)\n    >>> plt.title("Flat top window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the flat top window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'M' (line 621)
    M_285352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'M', False)
    # Processing the call keyword arguments (line 621)
    kwargs_285353 = {}
    # Getting the type of '_len_guards' (line 621)
    _len_guards_285351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 621)
    _len_guards_call_result_285354 = invoke(stypy.reporting.localization.Localization(__file__, 621, 7), _len_guards_285351, *[M_285352], **kwargs_285353)
    
    # Testing the type of an if condition (line 621)
    if_condition_285355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 4), _len_guards_call_result_285354)
    # Assigning a type to the variable 'if_condition_285355' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'if_condition_285355', if_condition_285355)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'M' (line 622)
    M_285358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 23), 'M', False)
    # Processing the call keyword arguments (line 622)
    kwargs_285359 = {}
    # Getting the type of 'np' (line 622)
    np_285356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 622)
    ones_285357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 15), np_285356, 'ones')
    # Calling ones(args, kwargs) (line 622)
    ones_call_result_285360 = invoke(stypy.reporting.localization.Localization(__file__, 622, 15), ones_285357, *[M_285358], **kwargs_285359)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'stypy_return_type', ones_call_result_285360)
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 623):
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_285361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 4), 'int')
    
    # Call to _extend(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'M' (line 623)
    M_285363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 29), 'M', False)
    # Getting the type of 'sym' (line 623)
    sym_285364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 32), 'sym', False)
    # Processing the call keyword arguments (line 623)
    kwargs_285365 = {}
    # Getting the type of '_extend' (line 623)
    _extend_285362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 623)
    _extend_call_result_285366 = invoke(stypy.reporting.localization.Localization(__file__, 623, 21), _extend_285362, *[M_285363, sym_285364], **kwargs_285365)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___285367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 4), _extend_call_result_285366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_285368 = invoke(stypy.reporting.localization.Localization(__file__, 623, 4), getitem___285367, int_285361)
    
    # Assigning a type to the variable 'tuple_var_assignment_284684' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'tuple_var_assignment_284684', subscript_call_result_285368)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    int_285369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 4), 'int')
    
    # Call to _extend(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'M' (line 623)
    M_285371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 29), 'M', False)
    # Getting the type of 'sym' (line 623)
    sym_285372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 32), 'sym', False)
    # Processing the call keyword arguments (line 623)
    kwargs_285373 = {}
    # Getting the type of '_extend' (line 623)
    _extend_285370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 623)
    _extend_call_result_285374 = invoke(stypy.reporting.localization.Localization(__file__, 623, 21), _extend_285370, *[M_285371, sym_285372], **kwargs_285373)
    
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___285375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 4), _extend_call_result_285374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_285376 = invoke(stypy.reporting.localization.Localization(__file__, 623, 4), getitem___285375, int_285369)
    
    # Assigning a type to the variable 'tuple_var_assignment_284685' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'tuple_var_assignment_284685', subscript_call_result_285376)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_284684' (line 623)
    tuple_var_assignment_284684_285377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'tuple_var_assignment_284684')
    # Assigning a type to the variable 'M' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'M', tuple_var_assignment_284684_285377)
    
    # Assigning a Name to a Name (line 623):
    # Getting the type of 'tuple_var_assignment_284685' (line 623)
    tuple_var_assignment_284685_285378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 4), 'tuple_var_assignment_284685')
    # Assigning a type to the variable 'needs_trunc' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 7), 'needs_trunc', tuple_var_assignment_284685_285378)
    
    # Assigning a List to a Name (line 625):
    
    # Assigning a List to a Name (line 625):
    
    # Obtaining an instance of the builtin type 'list' (line 625)
    list_285379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 625)
    # Adding element type (line 625)
    float_285380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 8), list_285379, float_285380)
    # Adding element type (line 625)
    float_285381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 8), list_285379, float_285381)
    # Adding element type (line 625)
    float_285382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 8), list_285379, float_285382)
    # Adding element type (line 625)
    float_285383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 8), list_285379, float_285383)
    # Adding element type (line 625)
    float_285384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 59), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 8), list_285379, float_285384)
    
    # Assigning a type to the variable 'a' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'a', list_285379)
    
    # Assigning a Call to a Name (line 626):
    
    # Assigning a Call to a Name (line 626):
    
    # Call to _cos_win(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'M' (line 626)
    M_285386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 17), 'M', False)
    # Getting the type of 'a' (line 626)
    a_285387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 20), 'a', False)
    # Processing the call keyword arguments (line 626)
    kwargs_285388 = {}
    # Getting the type of '_cos_win' (line 626)
    _cos_win_285385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 626)
    _cos_win_call_result_285389 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), _cos_win_285385, *[M_285386, a_285387], **kwargs_285388)
    
    # Assigning a type to the variable 'w' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'w', _cos_win_call_result_285389)
    
    # Call to _truncate(...): (line 628)
    # Processing the call arguments (line 628)
    # Getting the type of 'w' (line 628)
    w_285391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 628)
    needs_trunc_285392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 628)
    kwargs_285393 = {}
    # Getting the type of '_truncate' (line 628)
    _truncate_285390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 628)
    _truncate_call_result_285394 = invoke(stypy.reporting.localization.Localization(__file__, 628, 11), _truncate_285390, *[w_285391, needs_trunc_285392], **kwargs_285393)
    
    # Assigning a type to the variable 'stypy_return_type' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'stypy_return_type', _truncate_call_result_285394)
    
    # ################# End of 'flattop(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flattop' in the type store
    # Getting the type of 'stypy_return_type' (line 563)
    stypy_return_type_285395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285395)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flattop'
    return stypy_return_type_285395

# Assigning a type to the variable 'flattop' (line 563)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'flattop', flattop)

@norecursion
def bartlett(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 631)
    True_285396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 20), 'True')
    defaults = [True_285396]
    # Create a new context for function 'bartlett'
    module_type_store = module_type_store.open_function_context('bartlett', 631, 0, False)
    
    # Passed parameters checking function
    bartlett.stypy_localization = localization
    bartlett.stypy_type_of_self = None
    bartlett.stypy_type_store = module_type_store
    bartlett.stypy_function_name = 'bartlett'
    bartlett.stypy_param_names_list = ['M', 'sym']
    bartlett.stypy_varargs_param_name = None
    bartlett.stypy_kwargs_param_name = None
    bartlett.stypy_call_defaults = defaults
    bartlett.stypy_call_varargs = varargs
    bartlett.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bartlett', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bartlett', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bartlett(...)' code ##################

    str_285397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, (-1)), 'str', '\n    Return a Bartlett window.\n\n    The Bartlett window is very similar to a triangular window, except\n    that the end points are at zero.  It is often used in signal\n    processing for tapering a signal, without generating too much\n    ripple in the frequency domain.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The triangular window, with the first and last samples equal to zero\n        and the maximum value normalized to 1 (though the value 1 does not\n        appear if `M` is even and `sym` is True).\n\n    See Also\n    --------\n    triang : A triangular window that does not touch zero at the ends\n\n    Notes\n    -----\n    The Bartlett window is defined as\n\n    .. math:: w(n) = \\frac{2}{M-1} \\left(\n              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|\n              \\right)\n\n    Most references to the Bartlett window come from the signal\n    processing literature, where it is used as one of many windowing\n    functions for smoothing values.  Note that convolution with this\n    window produces linear interpolation.  It is also known as an\n    apodization (which means"removing the foot", i.e. smoothing\n    discontinuities at the beginning and end of the sampled signal) or\n    tapering function. The Fourier transform of the Bartlett is the product\n    of two sinc functions.\n    Note the excellent discussion in Kanasewich. [2]_\n\n    References\n    ----------\n    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",\n           Biometrika 37, 1-16, 1950.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",\n           The University of Alberta Press, 1975, pp. 109-110.\n    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal\n           Processing", Prentice-Hall, 1999, pp. 468-471.\n    .. [4] Wikipedia, "Window function",\n           http://en.wikipedia.org/wiki/Window_function\n    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 429.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.bartlett(51)\n    >>> plt.plot(window)\n    >>> plt.title("Bartlett window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Bartlett window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 718)
    # Processing the call arguments (line 718)
    # Getting the type of 'M' (line 718)
    M_285399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 19), 'M', False)
    # Processing the call keyword arguments (line 718)
    kwargs_285400 = {}
    # Getting the type of '_len_guards' (line 718)
    _len_guards_285398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 718)
    _len_guards_call_result_285401 = invoke(stypy.reporting.localization.Localization(__file__, 718, 7), _len_guards_285398, *[M_285399], **kwargs_285400)
    
    # Testing the type of an if condition (line 718)
    if_condition_285402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 718, 4), _len_guards_call_result_285401)
    # Assigning a type to the variable 'if_condition_285402' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'if_condition_285402', if_condition_285402)
    # SSA begins for if statement (line 718)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'M' (line 719)
    M_285405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 23), 'M', False)
    # Processing the call keyword arguments (line 719)
    kwargs_285406 = {}
    # Getting the type of 'np' (line 719)
    np_285403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 719)
    ones_285404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 15), np_285403, 'ones')
    # Calling ones(args, kwargs) (line 719)
    ones_call_result_285407 = invoke(stypy.reporting.localization.Localization(__file__, 719, 15), ones_285404, *[M_285405], **kwargs_285406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 8), 'stypy_return_type', ones_call_result_285407)
    # SSA join for if statement (line 718)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 720):
    
    # Assigning a Subscript to a Name (line 720):
    
    # Obtaining the type of the subscript
    int_285408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 4), 'int')
    
    # Call to _extend(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'M' (line 720)
    M_285410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 29), 'M', False)
    # Getting the type of 'sym' (line 720)
    sym_285411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 32), 'sym', False)
    # Processing the call keyword arguments (line 720)
    kwargs_285412 = {}
    # Getting the type of '_extend' (line 720)
    _extend_285409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 720)
    _extend_call_result_285413 = invoke(stypy.reporting.localization.Localization(__file__, 720, 21), _extend_285409, *[M_285410, sym_285411], **kwargs_285412)
    
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___285414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 4), _extend_call_result_285413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_285415 = invoke(stypy.reporting.localization.Localization(__file__, 720, 4), getitem___285414, int_285408)
    
    # Assigning a type to the variable 'tuple_var_assignment_284686' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'tuple_var_assignment_284686', subscript_call_result_285415)
    
    # Assigning a Subscript to a Name (line 720):
    
    # Obtaining the type of the subscript
    int_285416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 4), 'int')
    
    # Call to _extend(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'M' (line 720)
    M_285418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 29), 'M', False)
    # Getting the type of 'sym' (line 720)
    sym_285419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 32), 'sym', False)
    # Processing the call keyword arguments (line 720)
    kwargs_285420 = {}
    # Getting the type of '_extend' (line 720)
    _extend_285417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 720)
    _extend_call_result_285421 = invoke(stypy.reporting.localization.Localization(__file__, 720, 21), _extend_285417, *[M_285418, sym_285419], **kwargs_285420)
    
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___285422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 4), _extend_call_result_285421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_285423 = invoke(stypy.reporting.localization.Localization(__file__, 720, 4), getitem___285422, int_285416)
    
    # Assigning a type to the variable 'tuple_var_assignment_284687' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'tuple_var_assignment_284687', subscript_call_result_285423)
    
    # Assigning a Name to a Name (line 720):
    # Getting the type of 'tuple_var_assignment_284686' (line 720)
    tuple_var_assignment_284686_285424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'tuple_var_assignment_284686')
    # Assigning a type to the variable 'M' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'M', tuple_var_assignment_284686_285424)
    
    # Assigning a Name to a Name (line 720):
    # Getting the type of 'tuple_var_assignment_284687' (line 720)
    tuple_var_assignment_284687_285425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'tuple_var_assignment_284687')
    # Assigning a type to the variable 'needs_trunc' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 7), 'needs_trunc', tuple_var_assignment_284687_285425)
    
    # Assigning a Call to a Name (line 722):
    
    # Assigning a Call to a Name (line 722):
    
    # Call to arange(...): (line 722)
    # Processing the call arguments (line 722)
    int_285428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 18), 'int')
    # Getting the type of 'M' (line 722)
    M_285429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 21), 'M', False)
    # Processing the call keyword arguments (line 722)
    kwargs_285430 = {}
    # Getting the type of 'np' (line 722)
    np_285426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 722)
    arange_285427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 8), np_285426, 'arange')
    # Calling arange(args, kwargs) (line 722)
    arange_call_result_285431 = invoke(stypy.reporting.localization.Localization(__file__, 722, 8), arange_285427, *[int_285428, M_285429], **kwargs_285430)
    
    # Assigning a type to the variable 'n' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'n', arange_call_result_285431)
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Call to where(...): (line 723)
    # Processing the call arguments (line 723)
    
    # Call to less_equal(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 'n' (line 723)
    n_285436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 31), 'n', False)
    # Getting the type of 'M' (line 723)
    M_285437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 35), 'M', False)
    int_285438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 39), 'int')
    # Applying the binary operator '-' (line 723)
    result_sub_285439 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 35), '-', M_285437, int_285438)
    
    float_285440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 44), 'float')
    # Applying the binary operator 'div' (line 723)
    result_div_285441 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 34), 'div', result_sub_285439, float_285440)
    
    # Processing the call keyword arguments (line 723)
    kwargs_285442 = {}
    # Getting the type of 'np' (line 723)
    np_285434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 17), 'np', False)
    # Obtaining the member 'less_equal' of a type (line 723)
    less_equal_285435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 17), np_285434, 'less_equal')
    # Calling less_equal(args, kwargs) (line 723)
    less_equal_call_result_285443 = invoke(stypy.reporting.localization.Localization(__file__, 723, 17), less_equal_285435, *[n_285436, result_div_285441], **kwargs_285442)
    
    float_285444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 17), 'float')
    # Getting the type of 'n' (line 724)
    n_285445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 23), 'n', False)
    # Applying the binary operator '*' (line 724)
    result_mul_285446 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 17), '*', float_285444, n_285445)
    
    # Getting the type of 'M' (line 724)
    M_285447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 28), 'M', False)
    int_285448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 32), 'int')
    # Applying the binary operator '-' (line 724)
    result_sub_285449 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 28), '-', M_285447, int_285448)
    
    # Applying the binary operator 'div' (line 724)
    result_div_285450 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 25), 'div', result_mul_285446, result_sub_285449)
    
    float_285451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 36), 'float')
    float_285452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 42), 'float')
    # Getting the type of 'n' (line 724)
    n_285453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 48), 'n', False)
    # Applying the binary operator '*' (line 724)
    result_mul_285454 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 42), '*', float_285452, n_285453)
    
    # Getting the type of 'M' (line 724)
    M_285455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 53), 'M', False)
    int_285456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 57), 'int')
    # Applying the binary operator '-' (line 724)
    result_sub_285457 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 53), '-', M_285455, int_285456)
    
    # Applying the binary operator 'div' (line 724)
    result_div_285458 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 50), 'div', result_mul_285454, result_sub_285457)
    
    # Applying the binary operator '-' (line 724)
    result_sub_285459 = python_operator(stypy.reporting.localization.Localization(__file__, 724, 36), '-', float_285451, result_div_285458)
    
    # Processing the call keyword arguments (line 723)
    kwargs_285460 = {}
    # Getting the type of 'np' (line 723)
    np_285432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'np', False)
    # Obtaining the member 'where' of a type (line 723)
    where_285433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), np_285432, 'where')
    # Calling where(args, kwargs) (line 723)
    where_call_result_285461 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), where_285433, *[less_equal_call_result_285443, result_div_285450, result_sub_285459], **kwargs_285460)
    
    # Assigning a type to the variable 'w' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'w', where_call_result_285461)
    
    # Call to _truncate(...): (line 726)
    # Processing the call arguments (line 726)
    # Getting the type of 'w' (line 726)
    w_285463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 726)
    needs_trunc_285464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 726)
    kwargs_285465 = {}
    # Getting the type of '_truncate' (line 726)
    _truncate_285462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 726)
    _truncate_call_result_285466 = invoke(stypy.reporting.localization.Localization(__file__, 726, 11), _truncate_285462, *[w_285463, needs_trunc_285464], **kwargs_285465)
    
    # Assigning a type to the variable 'stypy_return_type' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'stypy_return_type', _truncate_call_result_285466)
    
    # ################# End of 'bartlett(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bartlett' in the type store
    # Getting the type of 'stypy_return_type' (line 631)
    stypy_return_type_285467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bartlett'
    return stypy_return_type_285467

# Assigning a type to the variable 'bartlett' (line 631)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 0), 'bartlett', bartlett)

@norecursion
def hann(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 729)
    True_285468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 16), 'True')
    defaults = [True_285468]
    # Create a new context for function 'hann'
    module_type_store = module_type_store.open_function_context('hann', 729, 0, False)
    
    # Passed parameters checking function
    hann.stypy_localization = localization
    hann.stypy_type_of_self = None
    hann.stypy_type_store = module_type_store
    hann.stypy_function_name = 'hann'
    hann.stypy_param_names_list = ['M', 'sym']
    hann.stypy_varargs_param_name = None
    hann.stypy_kwargs_param_name = None
    hann.stypy_call_defaults = defaults
    hann.stypy_call_varargs = varargs
    hann.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hann', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hann', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hann(...)' code ##################

    str_285469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, (-1)), 'str', '\n    Return a Hann window.\n\n    The Hann window is a taper formed by using a raised cosine or sine-squared\n    with ends that touch zero.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Hann window is defined as\n\n    .. math::  w(n) = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)\n               \\qquad 0 \\leq n \\leq M-1\n\n    The window was named for Julius von Hann, an Austrian meteorologist. It is\n    also known as the Cosine Bell. It is sometimes erroneously referred to as\n    the "Hanning" window, from the use of "hann" as a verb in the original\n    paper and confusion with the very similar Hamming window.\n\n    Most references to the Hann window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power\n           spectra, Dover Publications, New York.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",\n           The University of Alberta Press, 1975, pp. 106-108.\n    .. [3] Wikipedia, "Window function",\n           http://en.wikipedia.org/wiki/Window_function\n    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 425.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.hann(51)\n    >>> plt.plot(window)\n    >>> plt.title("Hann window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Hann window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 807)
    # Processing the call arguments (line 807)
    # Getting the type of 'M' (line 807)
    M_285471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 19), 'M', False)
    # Processing the call keyword arguments (line 807)
    kwargs_285472 = {}
    # Getting the type of '_len_guards' (line 807)
    _len_guards_285470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 807)
    _len_guards_call_result_285473 = invoke(stypy.reporting.localization.Localization(__file__, 807, 7), _len_guards_285470, *[M_285471], **kwargs_285472)
    
    # Testing the type of an if condition (line 807)
    if_condition_285474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 807, 4), _len_guards_call_result_285473)
    # Assigning a type to the variable 'if_condition_285474' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 4), 'if_condition_285474', if_condition_285474)
    # SSA begins for if statement (line 807)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 808)
    # Processing the call arguments (line 808)
    # Getting the type of 'M' (line 808)
    M_285477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 23), 'M', False)
    # Processing the call keyword arguments (line 808)
    kwargs_285478 = {}
    # Getting the type of 'np' (line 808)
    np_285475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 808)
    ones_285476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 15), np_285475, 'ones')
    # Calling ones(args, kwargs) (line 808)
    ones_call_result_285479 = invoke(stypy.reporting.localization.Localization(__file__, 808, 15), ones_285476, *[M_285477], **kwargs_285478)
    
    # Assigning a type to the variable 'stypy_return_type' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'stypy_return_type', ones_call_result_285479)
    # SSA join for if statement (line 807)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 809):
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_285480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to _extend(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'M' (line 809)
    M_285482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 29), 'M', False)
    # Getting the type of 'sym' (line 809)
    sym_285483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'sym', False)
    # Processing the call keyword arguments (line 809)
    kwargs_285484 = {}
    # Getting the type of '_extend' (line 809)
    _extend_285481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 809)
    _extend_call_result_285485 = invoke(stypy.reporting.localization.Localization(__file__, 809, 21), _extend_285481, *[M_285482, sym_285483], **kwargs_285484)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___285486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), _extend_call_result_285485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_285487 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___285486, int_285480)
    
    # Assigning a type to the variable 'tuple_var_assignment_284688' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_284688', subscript_call_result_285487)
    
    # Assigning a Subscript to a Name (line 809):
    
    # Obtaining the type of the subscript
    int_285488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 4), 'int')
    
    # Call to _extend(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'M' (line 809)
    M_285490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 29), 'M', False)
    # Getting the type of 'sym' (line 809)
    sym_285491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'sym', False)
    # Processing the call keyword arguments (line 809)
    kwargs_285492 = {}
    # Getting the type of '_extend' (line 809)
    _extend_285489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 809)
    _extend_call_result_285493 = invoke(stypy.reporting.localization.Localization(__file__, 809, 21), _extend_285489, *[M_285490, sym_285491], **kwargs_285492)
    
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___285494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 4), _extend_call_result_285493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_285495 = invoke(stypy.reporting.localization.Localization(__file__, 809, 4), getitem___285494, int_285488)
    
    # Assigning a type to the variable 'tuple_var_assignment_284689' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_284689', subscript_call_result_285495)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_284688' (line 809)
    tuple_var_assignment_284688_285496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_284688')
    # Assigning a type to the variable 'M' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'M', tuple_var_assignment_284688_285496)
    
    # Assigning a Name to a Name (line 809):
    # Getting the type of 'tuple_var_assignment_284689' (line 809)
    tuple_var_assignment_284689_285497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'tuple_var_assignment_284689')
    # Assigning a type to the variable 'needs_trunc' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 7), 'needs_trunc', tuple_var_assignment_284689_285497)
    
    # Assigning a Call to a Name (line 811):
    
    # Assigning a Call to a Name (line 811):
    
    # Call to _cos_win(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'M' (line 811)
    M_285499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 17), 'M', False)
    
    # Obtaining an instance of the builtin type 'list' (line 811)
    list_285500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 811)
    # Adding element type (line 811)
    float_285501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 20), list_285500, float_285501)
    # Adding element type (line 811)
    float_285502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 26), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 811, 20), list_285500, float_285502)
    
    # Processing the call keyword arguments (line 811)
    kwargs_285503 = {}
    # Getting the type of '_cos_win' (line 811)
    _cos_win_285498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 811)
    _cos_win_call_result_285504 = invoke(stypy.reporting.localization.Localization(__file__, 811, 8), _cos_win_285498, *[M_285499, list_285500], **kwargs_285503)
    
    # Assigning a type to the variable 'w' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'w', _cos_win_call_result_285504)
    
    # Call to _truncate(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'w' (line 813)
    w_285506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 813)
    needs_trunc_285507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 813)
    kwargs_285508 = {}
    # Getting the type of '_truncate' (line 813)
    _truncate_285505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 813)
    _truncate_call_result_285509 = invoke(stypy.reporting.localization.Localization(__file__, 813, 11), _truncate_285505, *[w_285506, needs_trunc_285507], **kwargs_285508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'stypy_return_type', _truncate_call_result_285509)
    
    # ################# End of 'hann(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hann' in the type store
    # Getting the type of 'stypy_return_type' (line 729)
    stypy_return_type_285510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hann'
    return stypy_return_type_285510

# Assigning a type to the variable 'hann' (line 729)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 0), 'hann', hann)

# Assigning a Name to a Name (line 816):

# Assigning a Name to a Name (line 816):
# Getting the type of 'hann' (line 816)
hann_285511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 10), 'hann')
# Assigning a type to the variable 'hanning' (line 816)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 0), 'hanning', hann_285511)

@norecursion
def tukey(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_285512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 19), 'float')
    # Getting the type of 'True' (line 819)
    True_285513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 28), 'True')
    defaults = [float_285512, True_285513]
    # Create a new context for function 'tukey'
    module_type_store = module_type_store.open_function_context('tukey', 819, 0, False)
    
    # Passed parameters checking function
    tukey.stypy_localization = localization
    tukey.stypy_type_of_self = None
    tukey.stypy_type_store = module_type_store
    tukey.stypy_function_name = 'tukey'
    tukey.stypy_param_names_list = ['M', 'alpha', 'sym']
    tukey.stypy_varargs_param_name = None
    tukey.stypy_kwargs_param_name = None
    tukey.stypy_call_defaults = defaults
    tukey.stypy_call_varargs = varargs
    tukey.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tukey', ['M', 'alpha', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tukey', localization, ['M', 'alpha', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tukey(...)' code ##################

    str_285514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 876, (-1)), 'str', 'Return a Tukey window, also known as a tapered cosine window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    alpha : float, optional\n        Shape parameter of the Tukey window, representing the fraction of the\n        window inside the cosine tapered region.\n        If zero, the Tukey window is equivalent to a rectangular window.\n        If one, the Tukey window is equivalent to a Hann window.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    References\n    ----------\n    .. [1] Harris, Fredric J. (Jan 1978). "On the use of Windows for Harmonic\n           Analysis with the Discrete Fourier Transform". Proceedings of the\n           IEEE 66 (1): 51-83. :doi:`10.1109/PROC.1978.10837`\n    .. [2] Wikipedia, "Window function",\n           http://en.wikipedia.org/wiki/Window_function#Tukey_window\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.tukey(51)\n    >>> plt.plot(window)\n    >>> plt.title("Tukey window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n    >>> plt.ylim([0, 1.1])\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Tukey window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'M' (line 877)
    M_285516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 19), 'M', False)
    # Processing the call keyword arguments (line 877)
    kwargs_285517 = {}
    # Getting the type of '_len_guards' (line 877)
    _len_guards_285515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 877)
    _len_guards_call_result_285518 = invoke(stypy.reporting.localization.Localization(__file__, 877, 7), _len_guards_285515, *[M_285516], **kwargs_285517)
    
    # Testing the type of an if condition (line 877)
    if_condition_285519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 877, 4), _len_guards_call_result_285518)
    # Assigning a type to the variable 'if_condition_285519' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 4), 'if_condition_285519', if_condition_285519)
    # SSA begins for if statement (line 877)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'M' (line 878)
    M_285522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 23), 'M', False)
    # Processing the call keyword arguments (line 878)
    kwargs_285523 = {}
    # Getting the type of 'np' (line 878)
    np_285520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 878)
    ones_285521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 15), np_285520, 'ones')
    # Calling ones(args, kwargs) (line 878)
    ones_call_result_285524 = invoke(stypy.reporting.localization.Localization(__file__, 878, 15), ones_285521, *[M_285522], **kwargs_285523)
    
    # Assigning a type to the variable 'stypy_return_type' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 8), 'stypy_return_type', ones_call_result_285524)
    # SSA join for if statement (line 877)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'alpha' (line 880)
    alpha_285525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 7), 'alpha')
    int_285526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 16), 'int')
    # Applying the binary operator '<=' (line 880)
    result_le_285527 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 7), '<=', alpha_285525, int_285526)
    
    # Testing the type of an if condition (line 880)
    if_condition_285528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 880, 4), result_le_285527)
    # Assigning a type to the variable 'if_condition_285528' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'if_condition_285528', if_condition_285528)
    # SSA begins for if statement (line 880)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 881)
    # Processing the call arguments (line 881)
    # Getting the type of 'M' (line 881)
    M_285531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 23), 'M', False)
    str_285532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 26), 'str', 'd')
    # Processing the call keyword arguments (line 881)
    kwargs_285533 = {}
    # Getting the type of 'np' (line 881)
    np_285529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 881)
    ones_285530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 881, 15), np_285529, 'ones')
    # Calling ones(args, kwargs) (line 881)
    ones_call_result_285534 = invoke(stypy.reporting.localization.Localization(__file__, 881, 15), ones_285530, *[M_285531, str_285532], **kwargs_285533)
    
    # Assigning a type to the variable 'stypy_return_type' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 8), 'stypy_return_type', ones_call_result_285534)
    # SSA branch for the else part of an if statement (line 880)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'alpha' (line 882)
    alpha_285535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 9), 'alpha')
    float_285536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 18), 'float')
    # Applying the binary operator '>=' (line 882)
    result_ge_285537 = python_operator(stypy.reporting.localization.Localization(__file__, 882, 9), '>=', alpha_285535, float_285536)
    
    # Testing the type of an if condition (line 882)
    if_condition_285538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 882, 9), result_ge_285537)
    # Assigning a type to the variable 'if_condition_285538' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 9), 'if_condition_285538', if_condition_285538)
    # SSA begins for if statement (line 882)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to hann(...): (line 883)
    # Processing the call arguments (line 883)
    # Getting the type of 'M' (line 883)
    M_285540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'M', False)
    # Processing the call keyword arguments (line 883)
    # Getting the type of 'sym' (line 883)
    sym_285541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 27), 'sym', False)
    keyword_285542 = sym_285541
    kwargs_285543 = {'sym': keyword_285542}
    # Getting the type of 'hann' (line 883)
    hann_285539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 15), 'hann', False)
    # Calling hann(args, kwargs) (line 883)
    hann_call_result_285544 = invoke(stypy.reporting.localization.Localization(__file__, 883, 15), hann_285539, *[M_285540], **kwargs_285543)
    
    # Assigning a type to the variable 'stypy_return_type' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 8), 'stypy_return_type', hann_call_result_285544)
    # SSA join for if statement (line 882)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 880)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 885):
    
    # Assigning a Subscript to a Name (line 885):
    
    # Obtaining the type of the subscript
    int_285545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 4), 'int')
    
    # Call to _extend(...): (line 885)
    # Processing the call arguments (line 885)
    # Getting the type of 'M' (line 885)
    M_285547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 29), 'M', False)
    # Getting the type of 'sym' (line 885)
    sym_285548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 32), 'sym', False)
    # Processing the call keyword arguments (line 885)
    kwargs_285549 = {}
    # Getting the type of '_extend' (line 885)
    _extend_285546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 885)
    _extend_call_result_285550 = invoke(stypy.reporting.localization.Localization(__file__, 885, 21), _extend_285546, *[M_285547, sym_285548], **kwargs_285549)
    
    # Obtaining the member '__getitem__' of a type (line 885)
    getitem___285551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 4), _extend_call_result_285550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 885)
    subscript_call_result_285552 = invoke(stypy.reporting.localization.Localization(__file__, 885, 4), getitem___285551, int_285545)
    
    # Assigning a type to the variable 'tuple_var_assignment_284690' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'tuple_var_assignment_284690', subscript_call_result_285552)
    
    # Assigning a Subscript to a Name (line 885):
    
    # Obtaining the type of the subscript
    int_285553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 4), 'int')
    
    # Call to _extend(...): (line 885)
    # Processing the call arguments (line 885)
    # Getting the type of 'M' (line 885)
    M_285555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 29), 'M', False)
    # Getting the type of 'sym' (line 885)
    sym_285556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 32), 'sym', False)
    # Processing the call keyword arguments (line 885)
    kwargs_285557 = {}
    # Getting the type of '_extend' (line 885)
    _extend_285554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 885)
    _extend_call_result_285558 = invoke(stypy.reporting.localization.Localization(__file__, 885, 21), _extend_285554, *[M_285555, sym_285556], **kwargs_285557)
    
    # Obtaining the member '__getitem__' of a type (line 885)
    getitem___285559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 885, 4), _extend_call_result_285558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 885)
    subscript_call_result_285560 = invoke(stypy.reporting.localization.Localization(__file__, 885, 4), getitem___285559, int_285553)
    
    # Assigning a type to the variable 'tuple_var_assignment_284691' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'tuple_var_assignment_284691', subscript_call_result_285560)
    
    # Assigning a Name to a Name (line 885):
    # Getting the type of 'tuple_var_assignment_284690' (line 885)
    tuple_var_assignment_284690_285561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'tuple_var_assignment_284690')
    # Assigning a type to the variable 'M' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'M', tuple_var_assignment_284690_285561)
    
    # Assigning a Name to a Name (line 885):
    # Getting the type of 'tuple_var_assignment_284691' (line 885)
    tuple_var_assignment_284691_285562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'tuple_var_assignment_284691')
    # Assigning a type to the variable 'needs_trunc' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 7), 'needs_trunc', tuple_var_assignment_284691_285562)
    
    # Assigning a Call to a Name (line 887):
    
    # Assigning a Call to a Name (line 887):
    
    # Call to arange(...): (line 887)
    # Processing the call arguments (line 887)
    int_285565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 18), 'int')
    # Getting the type of 'M' (line 887)
    M_285566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 21), 'M', False)
    # Processing the call keyword arguments (line 887)
    kwargs_285567 = {}
    # Getting the type of 'np' (line 887)
    np_285563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 887)
    arange_285564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), np_285563, 'arange')
    # Calling arange(args, kwargs) (line 887)
    arange_call_result_285568 = invoke(stypy.reporting.localization.Localization(__file__, 887, 8), arange_285564, *[int_285565, M_285566], **kwargs_285567)
    
    # Assigning a type to the variable 'n' (line 887)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 4), 'n', arange_call_result_285568)
    
    # Assigning a Call to a Name (line 888):
    
    # Assigning a Call to a Name (line 888):
    
    # Call to int(...): (line 888)
    # Processing the call arguments (line 888)
    
    # Call to floor(...): (line 888)
    # Processing the call arguments (line 888)
    # Getting the type of 'alpha' (line 888)
    alpha_285572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 25), 'alpha', False)
    # Getting the type of 'M' (line 888)
    M_285573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 32), 'M', False)
    int_285574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 34), 'int')
    # Applying the binary operator '-' (line 888)
    result_sub_285575 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 32), '-', M_285573, int_285574)
    
    # Applying the binary operator '*' (line 888)
    result_mul_285576 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 25), '*', alpha_285572, result_sub_285575)
    
    float_285577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 37), 'float')
    # Applying the binary operator 'div' (line 888)
    result_div_285578 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 36), 'div', result_mul_285576, float_285577)
    
    # Processing the call keyword arguments (line 888)
    kwargs_285579 = {}
    # Getting the type of 'np' (line 888)
    np_285570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 16), 'np', False)
    # Obtaining the member 'floor' of a type (line 888)
    floor_285571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 16), np_285570, 'floor')
    # Calling floor(args, kwargs) (line 888)
    floor_call_result_285580 = invoke(stypy.reporting.localization.Localization(__file__, 888, 16), floor_285571, *[result_div_285578], **kwargs_285579)
    
    # Processing the call keyword arguments (line 888)
    kwargs_285581 = {}
    # Getting the type of 'int' (line 888)
    int_285569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 12), 'int', False)
    # Calling int(args, kwargs) (line 888)
    int_call_result_285582 = invoke(stypy.reporting.localization.Localization(__file__, 888, 12), int_285569, *[floor_call_result_285580], **kwargs_285581)
    
    # Assigning a type to the variable 'width' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'width', int_call_result_285582)
    
    # Assigning a Subscript to a Name (line 889):
    
    # Assigning a Subscript to a Name (line 889):
    
    # Obtaining the type of the subscript
    int_285583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 11), 'int')
    # Getting the type of 'width' (line 889)
    width_285584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 13), 'width')
    int_285585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 19), 'int')
    # Applying the binary operator '+' (line 889)
    result_add_285586 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 13), '+', width_285584, int_285585)
    
    slice_285587 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 889, 9), int_285583, result_add_285586, None)
    # Getting the type of 'n' (line 889)
    n_285588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 9), 'n')
    # Obtaining the member '__getitem__' of a type (line 889)
    getitem___285589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 9), n_285588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 889)
    subscript_call_result_285590 = invoke(stypy.reporting.localization.Localization(__file__, 889, 9), getitem___285589, slice_285587)
    
    # Assigning a type to the variable 'n1' (line 889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 4), 'n1', subscript_call_result_285590)
    
    # Assigning a Subscript to a Name (line 890):
    
    # Assigning a Subscript to a Name (line 890):
    
    # Obtaining the type of the subscript
    # Getting the type of 'width' (line 890)
    width_285591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 11), 'width')
    int_285592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 17), 'int')
    # Applying the binary operator '+' (line 890)
    result_add_285593 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 11), '+', width_285591, int_285592)
    
    # Getting the type of 'M' (line 890)
    M_285594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 19), 'M')
    # Getting the type of 'width' (line 890)
    width_285595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 21), 'width')
    # Applying the binary operator '-' (line 890)
    result_sub_285596 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 19), '-', M_285594, width_285595)
    
    int_285597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 27), 'int')
    # Applying the binary operator '-' (line 890)
    result_sub_285598 = python_operator(stypy.reporting.localization.Localization(__file__, 890, 26), '-', result_sub_285596, int_285597)
    
    slice_285599 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 890, 9), result_add_285593, result_sub_285598, None)
    # Getting the type of 'n' (line 890)
    n_285600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 9), 'n')
    # Obtaining the member '__getitem__' of a type (line 890)
    getitem___285601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 9), n_285600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 890)
    subscript_call_result_285602 = invoke(stypy.reporting.localization.Localization(__file__, 890, 9), getitem___285601, slice_285599)
    
    # Assigning a type to the variable 'n2' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'n2', subscript_call_result_285602)
    
    # Assigning a Subscript to a Name (line 891):
    
    # Assigning a Subscript to a Name (line 891):
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 891)
    M_285603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 11), 'M')
    # Getting the type of 'width' (line 891)
    width_285604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 13), 'width')
    # Applying the binary operator '-' (line 891)
    result_sub_285605 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 11), '-', M_285603, width_285604)
    
    int_285606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, 19), 'int')
    # Applying the binary operator '-' (line 891)
    result_sub_285607 = python_operator(stypy.reporting.localization.Localization(__file__, 891, 18), '-', result_sub_285605, int_285606)
    
    slice_285608 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 891, 9), result_sub_285607, None, None)
    # Getting the type of 'n' (line 891)
    n_285609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 9), 'n')
    # Obtaining the member '__getitem__' of a type (line 891)
    getitem___285610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 891, 9), n_285609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 891)
    subscript_call_result_285611 = invoke(stypy.reporting.localization.Localization(__file__, 891, 9), getitem___285610, slice_285608)
    
    # Assigning a type to the variable 'n3' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 4), 'n3', subscript_call_result_285611)
    
    # Assigning a BinOp to a Name (line 893):
    
    # Assigning a BinOp to a Name (line 893):
    float_285612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 9), 'float')
    int_285613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 16), 'int')
    
    # Call to cos(...): (line 893)
    # Processing the call arguments (line 893)
    # Getting the type of 'np' (line 893)
    np_285616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 27), 'np', False)
    # Obtaining the member 'pi' of a type (line 893)
    pi_285617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 27), np_285616, 'pi')
    int_285618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 36), 'int')
    float_285619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 41), 'float')
    # Getting the type of 'n1' (line 893)
    n1_285620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 45), 'n1', False)
    # Applying the binary operator '*' (line 893)
    result_mul_285621 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 41), '*', float_285619, n1_285620)
    
    # Getting the type of 'alpha' (line 893)
    alpha_285622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 48), 'alpha', False)
    # Applying the binary operator 'div' (line 893)
    result_div_285623 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 47), 'div', result_mul_285621, alpha_285622)
    
    # Getting the type of 'M' (line 893)
    M_285624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 55), 'M', False)
    int_285625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 57), 'int')
    # Applying the binary operator '-' (line 893)
    result_sub_285626 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 55), '-', M_285624, int_285625)
    
    # Applying the binary operator 'div' (line 893)
    result_div_285627 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 53), 'div', result_div_285623, result_sub_285626)
    
    # Applying the binary operator '+' (line 893)
    result_add_285628 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 36), '+', int_285618, result_div_285627)
    
    # Applying the binary operator '*' (line 893)
    result_mul_285629 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 27), '*', pi_285617, result_add_285628)
    
    # Processing the call keyword arguments (line 893)
    kwargs_285630 = {}
    # Getting the type of 'np' (line 893)
    np_285614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 20), 'np', False)
    # Obtaining the member 'cos' of a type (line 893)
    cos_285615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 20), np_285614, 'cos')
    # Calling cos(args, kwargs) (line 893)
    cos_call_result_285631 = invoke(stypy.reporting.localization.Localization(__file__, 893, 20), cos_285615, *[result_mul_285629], **kwargs_285630)
    
    # Applying the binary operator '+' (line 893)
    result_add_285632 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 16), '+', int_285613, cos_call_result_285631)
    
    # Applying the binary operator '*' (line 893)
    result_mul_285633 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 9), '*', float_285612, result_add_285632)
    
    # Assigning a type to the variable 'w1' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'w1', result_mul_285633)
    
    # Assigning a Call to a Name (line 894):
    
    # Assigning a Call to a Name (line 894):
    
    # Call to ones(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'n2' (line 894)
    n2_285636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 17), 'n2', False)
    # Obtaining the member 'shape' of a type (line 894)
    shape_285637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 17), n2_285636, 'shape')
    # Processing the call keyword arguments (line 894)
    kwargs_285638 = {}
    # Getting the type of 'np' (line 894)
    np_285634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 9), 'np', False)
    # Obtaining the member 'ones' of a type (line 894)
    ones_285635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 9), np_285634, 'ones')
    # Calling ones(args, kwargs) (line 894)
    ones_call_result_285639 = invoke(stypy.reporting.localization.Localization(__file__, 894, 9), ones_285635, *[shape_285637], **kwargs_285638)
    
    # Assigning a type to the variable 'w2' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'w2', ones_call_result_285639)
    
    # Assigning a BinOp to a Name (line 895):
    
    # Assigning a BinOp to a Name (line 895):
    float_285640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 9), 'float')
    int_285641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 16), 'int')
    
    # Call to cos(...): (line 895)
    # Processing the call arguments (line 895)
    # Getting the type of 'np' (line 895)
    np_285644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 27), 'np', False)
    # Obtaining the member 'pi' of a type (line 895)
    pi_285645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 27), np_285644, 'pi')
    float_285646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 36), 'float')
    # Getting the type of 'alpha' (line 895)
    alpha_285647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 41), 'alpha', False)
    # Applying the binary operator 'div' (line 895)
    result_div_285648 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 36), 'div', float_285646, alpha_285647)
    
    int_285649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 49), 'int')
    # Applying the binary operator '+' (line 895)
    result_add_285650 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 36), '+', result_div_285648, int_285649)
    
    float_285651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 53), 'float')
    # Getting the type of 'n3' (line 895)
    n3_285652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 57), 'n3', False)
    # Applying the binary operator '*' (line 895)
    result_mul_285653 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 53), '*', float_285651, n3_285652)
    
    # Getting the type of 'alpha' (line 895)
    alpha_285654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 60), 'alpha', False)
    # Applying the binary operator 'div' (line 895)
    result_div_285655 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 59), 'div', result_mul_285653, alpha_285654)
    
    # Getting the type of 'M' (line 895)
    M_285656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 67), 'M', False)
    int_285657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 69), 'int')
    # Applying the binary operator '-' (line 895)
    result_sub_285658 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 67), '-', M_285656, int_285657)
    
    # Applying the binary operator 'div' (line 895)
    result_div_285659 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 65), 'div', result_div_285655, result_sub_285658)
    
    # Applying the binary operator '+' (line 895)
    result_add_285660 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 51), '+', result_add_285650, result_div_285659)
    
    # Applying the binary operator '*' (line 895)
    result_mul_285661 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 27), '*', pi_285645, result_add_285660)
    
    # Processing the call keyword arguments (line 895)
    kwargs_285662 = {}
    # Getting the type of 'np' (line 895)
    np_285642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 20), 'np', False)
    # Obtaining the member 'cos' of a type (line 895)
    cos_285643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 895, 20), np_285642, 'cos')
    # Calling cos(args, kwargs) (line 895)
    cos_call_result_285663 = invoke(stypy.reporting.localization.Localization(__file__, 895, 20), cos_285643, *[result_mul_285661], **kwargs_285662)
    
    # Applying the binary operator '+' (line 895)
    result_add_285664 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 16), '+', int_285641, cos_call_result_285663)
    
    # Applying the binary operator '*' (line 895)
    result_mul_285665 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 9), '*', float_285640, result_add_285664)
    
    # Assigning a type to the variable 'w3' (line 895)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 895, 4), 'w3', result_mul_285665)
    
    # Assigning a Call to a Name (line 897):
    
    # Assigning a Call to a Name (line 897):
    
    # Call to concatenate(...): (line 897)
    # Processing the call arguments (line 897)
    
    # Obtaining an instance of the builtin type 'tuple' (line 897)
    tuple_285668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 897)
    # Adding element type (line 897)
    # Getting the type of 'w1' (line 897)
    w1_285669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 24), 'w1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 24), tuple_285668, w1_285669)
    # Adding element type (line 897)
    # Getting the type of 'w2' (line 897)
    w2_285670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 28), 'w2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 24), tuple_285668, w2_285670)
    # Adding element type (line 897)
    # Getting the type of 'w3' (line 897)
    w3_285671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 32), 'w3', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 24), tuple_285668, w3_285671)
    
    # Processing the call keyword arguments (line 897)
    kwargs_285672 = {}
    # Getting the type of 'np' (line 897)
    np_285666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 897)
    concatenate_285667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 8), np_285666, 'concatenate')
    # Calling concatenate(args, kwargs) (line 897)
    concatenate_call_result_285673 = invoke(stypy.reporting.localization.Localization(__file__, 897, 8), concatenate_285667, *[tuple_285668], **kwargs_285672)
    
    # Assigning a type to the variable 'w' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'w', concatenate_call_result_285673)
    
    # Call to _truncate(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'w' (line 899)
    w_285675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 899)
    needs_trunc_285676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 899)
    kwargs_285677 = {}
    # Getting the type of '_truncate' (line 899)
    _truncate_285674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 899)
    _truncate_call_result_285678 = invoke(stypy.reporting.localization.Localization(__file__, 899, 11), _truncate_285674, *[w_285675, needs_trunc_285676], **kwargs_285677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 4), 'stypy_return_type', _truncate_call_result_285678)
    
    # ################# End of 'tukey(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tukey' in the type store
    # Getting the type of 'stypy_return_type' (line 819)
    stypy_return_type_285679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285679)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tukey'
    return stypy_return_type_285679

# Assigning a type to the variable 'tukey' (line 819)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 0), 'tukey', tukey)

@norecursion
def barthann(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 902)
    True_285680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 20), 'True')
    defaults = [True_285680]
    # Create a new context for function 'barthann'
    module_type_store = module_type_store.open_function_context('barthann', 902, 0, False)
    
    # Passed parameters checking function
    barthann.stypy_localization = localization
    barthann.stypy_type_of_self = None
    barthann.stypy_type_store = module_type_store
    barthann.stypy_function_name = 'barthann'
    barthann.stypy_param_names_list = ['M', 'sym']
    barthann.stypy_varargs_param_name = None
    barthann.stypy_kwargs_param_name = None
    barthann.stypy_call_defaults = defaults
    barthann.stypy_call_varargs = varargs
    barthann.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'barthann', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'barthann', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'barthann(...)' code ##################

    str_285681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 945, (-1)), 'str', 'Return a modified Bartlett-Hann window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.barthann(51)\n    >>> plt.plot(window)\n    >>> plt.title("Bartlett-Hann window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Bartlett-Hann window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 946)
    # Processing the call arguments (line 946)
    # Getting the type of 'M' (line 946)
    M_285683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 19), 'M', False)
    # Processing the call keyword arguments (line 946)
    kwargs_285684 = {}
    # Getting the type of '_len_guards' (line 946)
    _len_guards_285682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 946)
    _len_guards_call_result_285685 = invoke(stypy.reporting.localization.Localization(__file__, 946, 7), _len_guards_285682, *[M_285683], **kwargs_285684)
    
    # Testing the type of an if condition (line 946)
    if_condition_285686 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 946, 4), _len_guards_call_result_285685)
    # Assigning a type to the variable 'if_condition_285686' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'if_condition_285686', if_condition_285686)
    # SSA begins for if statement (line 946)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'M' (line 947)
    M_285689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 23), 'M', False)
    # Processing the call keyword arguments (line 947)
    kwargs_285690 = {}
    # Getting the type of 'np' (line 947)
    np_285687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 947)
    ones_285688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 947, 15), np_285687, 'ones')
    # Calling ones(args, kwargs) (line 947)
    ones_call_result_285691 = invoke(stypy.reporting.localization.Localization(__file__, 947, 15), ones_285688, *[M_285689], **kwargs_285690)
    
    # Assigning a type to the variable 'stypy_return_type' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 8), 'stypy_return_type', ones_call_result_285691)
    # SSA join for if statement (line 946)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 948):
    
    # Assigning a Subscript to a Name (line 948):
    
    # Obtaining the type of the subscript
    int_285692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 4), 'int')
    
    # Call to _extend(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'M' (line 948)
    M_285694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 29), 'M', False)
    # Getting the type of 'sym' (line 948)
    sym_285695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 32), 'sym', False)
    # Processing the call keyword arguments (line 948)
    kwargs_285696 = {}
    # Getting the type of '_extend' (line 948)
    _extend_285693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 948)
    _extend_call_result_285697 = invoke(stypy.reporting.localization.Localization(__file__, 948, 21), _extend_285693, *[M_285694, sym_285695], **kwargs_285696)
    
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___285698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 4), _extend_call_result_285697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_285699 = invoke(stypy.reporting.localization.Localization(__file__, 948, 4), getitem___285698, int_285692)
    
    # Assigning a type to the variable 'tuple_var_assignment_284692' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'tuple_var_assignment_284692', subscript_call_result_285699)
    
    # Assigning a Subscript to a Name (line 948):
    
    # Obtaining the type of the subscript
    int_285700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 4), 'int')
    
    # Call to _extend(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'M' (line 948)
    M_285702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 29), 'M', False)
    # Getting the type of 'sym' (line 948)
    sym_285703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 32), 'sym', False)
    # Processing the call keyword arguments (line 948)
    kwargs_285704 = {}
    # Getting the type of '_extend' (line 948)
    _extend_285701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 948)
    _extend_call_result_285705 = invoke(stypy.reporting.localization.Localization(__file__, 948, 21), _extend_285701, *[M_285702, sym_285703], **kwargs_285704)
    
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___285706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 4), _extend_call_result_285705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_285707 = invoke(stypy.reporting.localization.Localization(__file__, 948, 4), getitem___285706, int_285700)
    
    # Assigning a type to the variable 'tuple_var_assignment_284693' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'tuple_var_assignment_284693', subscript_call_result_285707)
    
    # Assigning a Name to a Name (line 948):
    # Getting the type of 'tuple_var_assignment_284692' (line 948)
    tuple_var_assignment_284692_285708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'tuple_var_assignment_284692')
    # Assigning a type to the variable 'M' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'M', tuple_var_assignment_284692_285708)
    
    # Assigning a Name to a Name (line 948):
    # Getting the type of 'tuple_var_assignment_284693' (line 948)
    tuple_var_assignment_284693_285709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'tuple_var_assignment_284693')
    # Assigning a type to the variable 'needs_trunc' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 7), 'needs_trunc', tuple_var_assignment_284693_285709)
    
    # Assigning a Call to a Name (line 950):
    
    # Assigning a Call to a Name (line 950):
    
    # Call to arange(...): (line 950)
    # Processing the call arguments (line 950)
    int_285712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 18), 'int')
    # Getting the type of 'M' (line 950)
    M_285713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 21), 'M', False)
    # Processing the call keyword arguments (line 950)
    kwargs_285714 = {}
    # Getting the type of 'np' (line 950)
    np_285710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 950)
    arange_285711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 8), np_285710, 'arange')
    # Calling arange(args, kwargs) (line 950)
    arange_call_result_285715 = invoke(stypy.reporting.localization.Localization(__file__, 950, 8), arange_285711, *[int_285712, M_285713], **kwargs_285714)
    
    # Assigning a type to the variable 'n' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'n', arange_call_result_285715)
    
    # Assigning a Call to a Name (line 951):
    
    # Assigning a Call to a Name (line 951):
    
    # Call to abs(...): (line 951)
    # Processing the call arguments (line 951)
    # Getting the type of 'n' (line 951)
    n_285718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 17), 'n', False)
    # Getting the type of 'M' (line 951)
    M_285719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 22), 'M', False)
    float_285720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 26), 'float')
    # Applying the binary operator '-' (line 951)
    result_sub_285721 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 22), '-', M_285719, float_285720)
    
    # Applying the binary operator 'div' (line 951)
    result_div_285722 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 17), 'div', n_285718, result_sub_285721)
    
    float_285723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 33), 'float')
    # Applying the binary operator '-' (line 951)
    result_sub_285724 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 17), '-', result_div_285722, float_285723)
    
    # Processing the call keyword arguments (line 951)
    kwargs_285725 = {}
    # Getting the type of 'np' (line 951)
    np_285716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 951)
    abs_285717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 10), np_285716, 'abs')
    # Calling abs(args, kwargs) (line 951)
    abs_call_result_285726 = invoke(stypy.reporting.localization.Localization(__file__, 951, 10), abs_285717, *[result_sub_285724], **kwargs_285725)
    
    # Assigning a type to the variable 'fac' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'fac', abs_call_result_285726)
    
    # Assigning a BinOp to a Name (line 952):
    
    # Assigning a BinOp to a Name (line 952):
    float_285727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 8), 'float')
    float_285728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 15), 'float')
    # Getting the type of 'fac' (line 952)
    fac_285729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 22), 'fac')
    # Applying the binary operator '*' (line 952)
    result_mul_285730 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 15), '*', float_285728, fac_285729)
    
    # Applying the binary operator '-' (line 952)
    result_sub_285731 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 8), '-', float_285727, result_mul_285730)
    
    float_285732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 28), 'float')
    
    # Call to cos(...): (line 952)
    # Processing the call arguments (line 952)
    int_285735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 42), 'int')
    # Getting the type of 'np' (line 952)
    np_285736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 46), 'np', False)
    # Obtaining the member 'pi' of a type (line 952)
    pi_285737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 46), np_285736, 'pi')
    # Applying the binary operator '*' (line 952)
    result_mul_285738 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 42), '*', int_285735, pi_285737)
    
    # Getting the type of 'fac' (line 952)
    fac_285739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 54), 'fac', False)
    # Applying the binary operator '*' (line 952)
    result_mul_285740 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 52), '*', result_mul_285738, fac_285739)
    
    # Processing the call keyword arguments (line 952)
    kwargs_285741 = {}
    # Getting the type of 'np' (line 952)
    np_285733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 35), 'np', False)
    # Obtaining the member 'cos' of a type (line 952)
    cos_285734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 35), np_285733, 'cos')
    # Calling cos(args, kwargs) (line 952)
    cos_call_result_285742 = invoke(stypy.reporting.localization.Localization(__file__, 952, 35), cos_285734, *[result_mul_285740], **kwargs_285741)
    
    # Applying the binary operator '*' (line 952)
    result_mul_285743 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 28), '*', float_285732, cos_call_result_285742)
    
    # Applying the binary operator '+' (line 952)
    result_add_285744 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 26), '+', result_sub_285731, result_mul_285743)
    
    # Assigning a type to the variable 'w' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 4), 'w', result_add_285744)
    
    # Call to _truncate(...): (line 954)
    # Processing the call arguments (line 954)
    # Getting the type of 'w' (line 954)
    w_285746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 954)
    needs_trunc_285747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 954)
    kwargs_285748 = {}
    # Getting the type of '_truncate' (line 954)
    _truncate_285745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 954)
    _truncate_call_result_285749 = invoke(stypy.reporting.localization.Localization(__file__, 954, 11), _truncate_285745, *[w_285746, needs_trunc_285747], **kwargs_285748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'stypy_return_type', _truncate_call_result_285749)
    
    # ################# End of 'barthann(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'barthann' in the type store
    # Getting the type of 'stypy_return_type' (line 902)
    stypy_return_type_285750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'barthann'
    return stypy_return_type_285750

# Assigning a type to the variable 'barthann' (line 902)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 0), 'barthann', barthann)

@norecursion
def hamming(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 957)
    True_285751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 19), 'True')
    defaults = [True_285751]
    # Create a new context for function 'hamming'
    module_type_store = module_type_store.open_function_context('hamming', 957, 0, False)
    
    # Passed parameters checking function
    hamming.stypy_localization = localization
    hamming.stypy_type_of_self = None
    hamming.stypy_type_store = module_type_store
    hamming.stypy_function_name = 'hamming'
    hamming.stypy_param_names_list = ['M', 'sym']
    hamming.stypy_varargs_param_name = None
    hamming.stypy_kwargs_param_name = None
    hamming.stypy_call_defaults = defaults
    hamming.stypy_call_varargs = varargs
    hamming.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hamming', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hamming', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hamming(...)' code ##################

    str_285752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'str', 'Return a Hamming window.\n\n    The Hamming window is a taper formed by using a raised cosine with\n    non-zero endpoints, optimized to minimize the nearest side lobe.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Hamming window is defined as\n\n    .. math::  w(n) = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)\n               \\qquad 0 \\leq n \\leq M-1\n\n    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey and\n    is described in Blackman and Tukey. It was recommended for smoothing the\n    truncated autocovariance function in the time domain.\n    Most references to the Hamming window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power\n           spectra, Dover Publications, New York.\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The\n           University of Alberta Press, 1975, pp. 109-110.\n    .. [3] Wikipedia, "Window function",\n           http://en.wikipedia.org/wiki/Window_function\n    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,\n           "Numerical Recipes", Cambridge University Press, 1986, page 425.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.hamming(51)\n    >>> plt.plot(window)\n    >>> plt.title("Hamming window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Hamming window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 1032)
    # Processing the call arguments (line 1032)
    # Getting the type of 'M' (line 1032)
    M_285754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 19), 'M', False)
    # Processing the call keyword arguments (line 1032)
    kwargs_285755 = {}
    # Getting the type of '_len_guards' (line 1032)
    _len_guards_285753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1032)
    _len_guards_call_result_285756 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 7), _len_guards_285753, *[M_285754], **kwargs_285755)
    
    # Testing the type of an if condition (line 1032)
    if_condition_285757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1032, 4), _len_guards_call_result_285756)
    # Assigning a type to the variable 'if_condition_285757' (line 1032)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 4), 'if_condition_285757', if_condition_285757)
    # SSA begins for if statement (line 1032)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'M' (line 1033)
    M_285760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 23), 'M', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_285761 = {}
    # Getting the type of 'np' (line 1033)
    np_285758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1033)
    ones_285759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1033, 15), np_285758, 'ones')
    # Calling ones(args, kwargs) (line 1033)
    ones_call_result_285762 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 15), ones_285759, *[M_285760], **kwargs_285761)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'stypy_return_type', ones_call_result_285762)
    # SSA join for if statement (line 1032)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1034):
    
    # Assigning a Subscript to a Name (line 1034):
    
    # Obtaining the type of the subscript
    int_285763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 4), 'int')
    
    # Call to _extend(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'M' (line 1034)
    M_285765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 29), 'M', False)
    # Getting the type of 'sym' (line 1034)
    sym_285766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 32), 'sym', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_285767 = {}
    # Getting the type of '_extend' (line 1034)
    _extend_285764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1034)
    _extend_call_result_285768 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 21), _extend_285764, *[M_285765, sym_285766], **kwargs_285767)
    
    # Obtaining the member '__getitem__' of a type (line 1034)
    getitem___285769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 4), _extend_call_result_285768, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1034)
    subscript_call_result_285770 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 4), getitem___285769, int_285763)
    
    # Assigning a type to the variable 'tuple_var_assignment_284694' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'tuple_var_assignment_284694', subscript_call_result_285770)
    
    # Assigning a Subscript to a Name (line 1034):
    
    # Obtaining the type of the subscript
    int_285771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 4), 'int')
    
    # Call to _extend(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'M' (line 1034)
    M_285773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 29), 'M', False)
    # Getting the type of 'sym' (line 1034)
    sym_285774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 32), 'sym', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_285775 = {}
    # Getting the type of '_extend' (line 1034)
    _extend_285772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1034)
    _extend_call_result_285776 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 21), _extend_285772, *[M_285773, sym_285774], **kwargs_285775)
    
    # Obtaining the member '__getitem__' of a type (line 1034)
    getitem___285777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 4), _extend_call_result_285776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1034)
    subscript_call_result_285778 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 4), getitem___285777, int_285771)
    
    # Assigning a type to the variable 'tuple_var_assignment_284695' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'tuple_var_assignment_284695', subscript_call_result_285778)
    
    # Assigning a Name to a Name (line 1034):
    # Getting the type of 'tuple_var_assignment_284694' (line 1034)
    tuple_var_assignment_284694_285779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'tuple_var_assignment_284694')
    # Assigning a type to the variable 'M' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'M', tuple_var_assignment_284694_285779)
    
    # Assigning a Name to a Name (line 1034):
    # Getting the type of 'tuple_var_assignment_284695' (line 1034)
    tuple_var_assignment_284695_285780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'tuple_var_assignment_284695')
    # Assigning a type to the variable 'needs_trunc' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 7), 'needs_trunc', tuple_var_assignment_284695_285780)
    
    # Assigning a Call to a Name (line 1036):
    
    # Assigning a Call to a Name (line 1036):
    
    # Call to _cos_win(...): (line 1036)
    # Processing the call arguments (line 1036)
    # Getting the type of 'M' (line 1036)
    M_285782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 17), 'M', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1036)
    list_285783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1036)
    # Adding element type (line 1036)
    float_285784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1036, 20), list_285783, float_285784)
    # Adding element type (line 1036)
    float_285785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1036, 20), list_285783, float_285785)
    
    # Processing the call keyword arguments (line 1036)
    kwargs_285786 = {}
    # Getting the type of '_cos_win' (line 1036)
    _cos_win_285781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 8), '_cos_win', False)
    # Calling _cos_win(args, kwargs) (line 1036)
    _cos_win_call_result_285787 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 8), _cos_win_285781, *[M_285782, list_285783], **kwargs_285786)
    
    # Assigning a type to the variable 'w' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'w', _cos_win_call_result_285787)
    
    # Call to _truncate(...): (line 1038)
    # Processing the call arguments (line 1038)
    # Getting the type of 'w' (line 1038)
    w_285789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1038)
    needs_trunc_285790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1038)
    kwargs_285791 = {}
    # Getting the type of '_truncate' (line 1038)
    _truncate_285788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1038)
    _truncate_call_result_285792 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 11), _truncate_285788, *[w_285789, needs_trunc_285790], **kwargs_285791)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 4), 'stypy_return_type', _truncate_call_result_285792)
    
    # ################# End of 'hamming(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hamming' in the type store
    # Getting the type of 'stypy_return_type' (line 957)
    stypy_return_type_285793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285793)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hamming'
    return stypy_return_type_285793

# Assigning a type to the variable 'hamming' (line 957)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 0), 'hamming', hamming)

@norecursion
def kaiser(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1041)
    True_285794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 24), 'True')
    defaults = [True_285794]
    # Create a new context for function 'kaiser'
    module_type_store = module_type_store.open_function_context('kaiser', 1041, 0, False)
    
    # Passed parameters checking function
    kaiser.stypy_localization = localization
    kaiser.stypy_type_of_self = None
    kaiser.stypy_type_store = module_type_store
    kaiser.stypy_function_name = 'kaiser'
    kaiser.stypy_param_names_list = ['M', 'beta', 'sym']
    kaiser.stypy_varargs_param_name = None
    kaiser.stypy_kwargs_param_name = None
    kaiser.stypy_call_defaults = defaults
    kaiser.stypy_call_varargs = varargs
    kaiser.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kaiser', ['M', 'beta', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kaiser', localization, ['M', 'beta', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kaiser(...)' code ##################

    str_285795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1145, (-1)), 'str', 'Return a Kaiser window.\n\n    The Kaiser window is a taper formed by using a Bessel function.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    beta : float\n        Shape parameter, determines trade-off between main-lobe width and\n        side lobe level. As beta gets large, the window narrows.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Kaiser window is defined as\n\n    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}\n               \\right)/I_0(\\beta)\n\n    with\n\n    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},\n\n    where :math:`I_0` is the modified zeroth-order Bessel function.\n\n    The Kaiser was named for Jim Kaiser, who discovered a simple approximation\n    to the DPSS window based on Bessel functions.\n    The Kaiser window is a very good approximation to the Digital Prolate\n    Spheroidal Sequence, or Slepian window, which is the transform which\n    maximizes the energy in the main lobe of the window relative to total\n    energy.\n\n    The Kaiser can approximate other windows by varying the beta parameter.\n    (Some literature uses alpha = beta/pi.) [4]_\n\n    ====  =======================\n    beta  Window shape\n    ====  =======================\n    0     Rectangular\n    5     Similar to a Hamming\n    6     Similar to a Hann\n    8.6   Similar to a Blackman\n    ====  =======================\n\n    A beta value of 14 is probably a good starting point. Note that as beta\n    gets large, the window narrows, and so the number of samples needs to be\n    large enough to sample the increasingly narrow spike, otherwise NaNs will\n    be returned.\n\n    Most references to the Kaiser window come from the signal processing\n    literature, where it is used as one of many windowing functions for\n    smoothing values.  It is also known as an apodization (which means\n    "removing the foot", i.e. smoothing discontinuities at the beginning\n    and end of the sampled signal) or tapering function.\n\n    References\n    ----------\n    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by\n           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.\n           John Wiley and Sons, New York, (1966).\n    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The\n           University of Alberta Press, 1975, pp. 177-178.\n    .. [3] Wikipedia, "Window function",\n           http://en.wikipedia.org/wiki/Window_function\n    .. [4] F. J. Harris, "On the use of windows for harmonic analysis with the\n           discrete Fourier transform," Proceedings of the IEEE, vol. 66,\n           no. 1, pp. 51-83, Jan. 1978. :doi:`10.1109/PROC.1978.10837`.\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.kaiser(51, beta=14)\n    >>> plt.plot(window)\n    >>> plt.title(r"Kaiser window ($\\beta$=14)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title(r"Frequency response of the Kaiser window ($\\beta$=14)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 1147)
    # Processing the call arguments (line 1147)
    # Getting the type of 'M' (line 1147)
    M_285797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 19), 'M', False)
    # Processing the call keyword arguments (line 1147)
    kwargs_285798 = {}
    # Getting the type of '_len_guards' (line 1147)
    _len_guards_285796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1147, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1147)
    _len_guards_call_result_285799 = invoke(stypy.reporting.localization.Localization(__file__, 1147, 7), _len_guards_285796, *[M_285797], **kwargs_285798)
    
    # Testing the type of an if condition (line 1147)
    if_condition_285800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1147, 4), _len_guards_call_result_285799)
    # Assigning a type to the variable 'if_condition_285800' (line 1147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1147, 4), 'if_condition_285800', if_condition_285800)
    # SSA begins for if statement (line 1147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1148)
    # Processing the call arguments (line 1148)
    # Getting the type of 'M' (line 1148)
    M_285803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 23), 'M', False)
    # Processing the call keyword arguments (line 1148)
    kwargs_285804 = {}
    # Getting the type of 'np' (line 1148)
    np_285801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1148, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1148)
    ones_285802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1148, 15), np_285801, 'ones')
    # Calling ones(args, kwargs) (line 1148)
    ones_call_result_285805 = invoke(stypy.reporting.localization.Localization(__file__, 1148, 15), ones_285802, *[M_285803], **kwargs_285804)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1148, 8), 'stypy_return_type', ones_call_result_285805)
    # SSA join for if statement (line 1147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1149):
    
    # Assigning a Subscript to a Name (line 1149):
    
    # Obtaining the type of the subscript
    int_285806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1149, 4), 'int')
    
    # Call to _extend(...): (line 1149)
    # Processing the call arguments (line 1149)
    # Getting the type of 'M' (line 1149)
    M_285808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 29), 'M', False)
    # Getting the type of 'sym' (line 1149)
    sym_285809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 32), 'sym', False)
    # Processing the call keyword arguments (line 1149)
    kwargs_285810 = {}
    # Getting the type of '_extend' (line 1149)
    _extend_285807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1149)
    _extend_call_result_285811 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 21), _extend_285807, *[M_285808, sym_285809], **kwargs_285810)
    
    # Obtaining the member '__getitem__' of a type (line 1149)
    getitem___285812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 4), _extend_call_result_285811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1149)
    subscript_call_result_285813 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 4), getitem___285812, int_285806)
    
    # Assigning a type to the variable 'tuple_var_assignment_284696' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'tuple_var_assignment_284696', subscript_call_result_285813)
    
    # Assigning a Subscript to a Name (line 1149):
    
    # Obtaining the type of the subscript
    int_285814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1149, 4), 'int')
    
    # Call to _extend(...): (line 1149)
    # Processing the call arguments (line 1149)
    # Getting the type of 'M' (line 1149)
    M_285816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 29), 'M', False)
    # Getting the type of 'sym' (line 1149)
    sym_285817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 32), 'sym', False)
    # Processing the call keyword arguments (line 1149)
    kwargs_285818 = {}
    # Getting the type of '_extend' (line 1149)
    _extend_285815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1149)
    _extend_call_result_285819 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 21), _extend_285815, *[M_285816, sym_285817], **kwargs_285818)
    
    # Obtaining the member '__getitem__' of a type (line 1149)
    getitem___285820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1149, 4), _extend_call_result_285819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1149)
    subscript_call_result_285821 = invoke(stypy.reporting.localization.Localization(__file__, 1149, 4), getitem___285820, int_285814)
    
    # Assigning a type to the variable 'tuple_var_assignment_284697' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'tuple_var_assignment_284697', subscript_call_result_285821)
    
    # Assigning a Name to a Name (line 1149):
    # Getting the type of 'tuple_var_assignment_284696' (line 1149)
    tuple_var_assignment_284696_285822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'tuple_var_assignment_284696')
    # Assigning a type to the variable 'M' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'M', tuple_var_assignment_284696_285822)
    
    # Assigning a Name to a Name (line 1149):
    # Getting the type of 'tuple_var_assignment_284697' (line 1149)
    tuple_var_assignment_284697_285823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1149, 4), 'tuple_var_assignment_284697')
    # Assigning a type to the variable 'needs_trunc' (line 1149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1149, 7), 'needs_trunc', tuple_var_assignment_284697_285823)
    
    # Assigning a Call to a Name (line 1151):
    
    # Assigning a Call to a Name (line 1151):
    
    # Call to arange(...): (line 1151)
    # Processing the call arguments (line 1151)
    int_285826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1151, 18), 'int')
    # Getting the type of 'M' (line 1151)
    M_285827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 21), 'M', False)
    # Processing the call keyword arguments (line 1151)
    kwargs_285828 = {}
    # Getting the type of 'np' (line 1151)
    np_285824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1151, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 1151)
    arange_285825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1151, 8), np_285824, 'arange')
    # Calling arange(args, kwargs) (line 1151)
    arange_call_result_285829 = invoke(stypy.reporting.localization.Localization(__file__, 1151, 8), arange_285825, *[int_285826, M_285827], **kwargs_285828)
    
    # Assigning a type to the variable 'n' (line 1151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1151, 4), 'n', arange_call_result_285829)
    
    # Assigning a BinOp to a Name (line 1152):
    
    # Assigning a BinOp to a Name (line 1152):
    # Getting the type of 'M' (line 1152)
    M_285830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1152, 13), 'M')
    int_285831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, 17), 'int')
    # Applying the binary operator '-' (line 1152)
    result_sub_285832 = python_operator(stypy.reporting.localization.Localization(__file__, 1152, 13), '-', M_285830, int_285831)
    
    float_285833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, 22), 'float')
    # Applying the binary operator 'div' (line 1152)
    result_div_285834 = python_operator(stypy.reporting.localization.Localization(__file__, 1152, 12), 'div', result_sub_285832, float_285833)
    
    # Assigning a type to the variable 'alpha' (line 1152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1152, 4), 'alpha', result_div_285834)
    
    # Assigning a BinOp to a Name (line 1153):
    
    # Assigning a BinOp to a Name (line 1153):
    
    # Call to i0(...): (line 1153)
    # Processing the call arguments (line 1153)
    # Getting the type of 'beta' (line 1153)
    beta_285837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 20), 'beta', False)
    
    # Call to sqrt(...): (line 1153)
    # Processing the call arguments (line 1153)
    int_285840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 35), 'int')
    # Getting the type of 'n' (line 1153)
    n_285841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 41), 'n', False)
    # Getting the type of 'alpha' (line 1153)
    alpha_285842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 45), 'alpha', False)
    # Applying the binary operator '-' (line 1153)
    result_sub_285843 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 41), '-', n_285841, alpha_285842)
    
    # Getting the type of 'alpha' (line 1153)
    alpha_285844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 54), 'alpha', False)
    # Applying the binary operator 'div' (line 1153)
    result_div_285845 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 40), 'div', result_sub_285843, alpha_285844)
    
    float_285846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1153, 64), 'float')
    # Applying the binary operator '**' (line 1153)
    result_pow_285847 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 39), '**', result_div_285845, float_285846)
    
    # Applying the binary operator '-' (line 1153)
    result_sub_285848 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 35), '-', int_285840, result_pow_285847)
    
    # Processing the call keyword arguments (line 1153)
    kwargs_285849 = {}
    # Getting the type of 'np' (line 1153)
    np_285838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 27), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1153)
    sqrt_285839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 27), np_285838, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1153)
    sqrt_call_result_285850 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 27), sqrt_285839, *[result_sub_285848], **kwargs_285849)
    
    # Applying the binary operator '*' (line 1153)
    result_mul_285851 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 20), '*', beta_285837, sqrt_call_result_285850)
    
    # Processing the call keyword arguments (line 1153)
    kwargs_285852 = {}
    # Getting the type of 'special' (line 1153)
    special_285835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1153, 9), 'special', False)
    # Obtaining the member 'i0' of a type (line 1153)
    i0_285836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1153, 9), special_285835, 'i0')
    # Calling i0(args, kwargs) (line 1153)
    i0_call_result_285853 = invoke(stypy.reporting.localization.Localization(__file__, 1153, 9), i0_285836, *[result_mul_285851], **kwargs_285852)
    
    
    # Call to i0(...): (line 1154)
    # Processing the call arguments (line 1154)
    # Getting the type of 'beta' (line 1154)
    beta_285856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 20), 'beta', False)
    # Processing the call keyword arguments (line 1154)
    kwargs_285857 = {}
    # Getting the type of 'special' (line 1154)
    special_285854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1154, 9), 'special', False)
    # Obtaining the member 'i0' of a type (line 1154)
    i0_285855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1154, 9), special_285854, 'i0')
    # Calling i0(args, kwargs) (line 1154)
    i0_call_result_285858 = invoke(stypy.reporting.localization.Localization(__file__, 1154, 9), i0_285855, *[beta_285856], **kwargs_285857)
    
    # Applying the binary operator 'div' (line 1153)
    result_div_285859 = python_operator(stypy.reporting.localization.Localization(__file__, 1153, 9), 'div', i0_call_result_285853, i0_call_result_285858)
    
    # Assigning a type to the variable 'w' (line 1153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1153, 4), 'w', result_div_285859)
    
    # Call to _truncate(...): (line 1156)
    # Processing the call arguments (line 1156)
    # Getting the type of 'w' (line 1156)
    w_285861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1156)
    needs_trunc_285862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1156)
    kwargs_285863 = {}
    # Getting the type of '_truncate' (line 1156)
    _truncate_285860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1156)
    _truncate_call_result_285864 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 11), _truncate_285860, *[w_285861, needs_trunc_285862], **kwargs_285863)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'stypy_return_type', _truncate_call_result_285864)
    
    # ################# End of 'kaiser(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kaiser' in the type store
    # Getting the type of 'stypy_return_type' (line 1041)
    stypy_return_type_285865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285865)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kaiser'
    return stypy_return_type_285865

# Assigning a type to the variable 'kaiser' (line 1041)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1041, 0), 'kaiser', kaiser)

@norecursion
def gaussian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1159)
    True_285866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 25), 'True')
    defaults = [True_285866]
    # Create a new context for function 'gaussian'
    module_type_store = module_type_store.open_function_context('gaussian', 1159, 0, False)
    
    # Passed parameters checking function
    gaussian.stypy_localization = localization
    gaussian.stypy_type_of_self = None
    gaussian.stypy_type_store = module_type_store
    gaussian.stypy_function_name = 'gaussian'
    gaussian.stypy_param_names_list = ['M', 'std', 'sym']
    gaussian.stypy_varargs_param_name = None
    gaussian.stypy_kwargs_param_name = None
    gaussian.stypy_call_defaults = defaults
    gaussian.stypy_call_varargs = varargs
    gaussian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gaussian', ['M', 'std', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gaussian', localization, ['M', 'std', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gaussian(...)' code ##################

    str_285867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1210, (-1)), 'str', 'Return a Gaussian window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    std : float\n        The standard deviation, sigma.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Gaussian window is defined as\n\n    .. math::  w(n) = e^{ -\\frac{1}{2}\\left(\\frac{n}{\\sigma}\\right)^2 }\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.gaussian(51, std=7)\n    >>> plt.plot(window)\n    >>> plt.title(r"Gaussian window ($\\sigma$=7)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title(r"Frequency response of the Gaussian window ($\\sigma$=7)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 1211)
    # Processing the call arguments (line 1211)
    # Getting the type of 'M' (line 1211)
    M_285869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 19), 'M', False)
    # Processing the call keyword arguments (line 1211)
    kwargs_285870 = {}
    # Getting the type of '_len_guards' (line 1211)
    _len_guards_285868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1211)
    _len_guards_call_result_285871 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 7), _len_guards_285868, *[M_285869], **kwargs_285870)
    
    # Testing the type of an if condition (line 1211)
    if_condition_285872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1211, 4), _len_guards_call_result_285871)
    # Assigning a type to the variable 'if_condition_285872' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 4), 'if_condition_285872', if_condition_285872)
    # SSA begins for if statement (line 1211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1212)
    # Processing the call arguments (line 1212)
    # Getting the type of 'M' (line 1212)
    M_285875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 23), 'M', False)
    # Processing the call keyword arguments (line 1212)
    kwargs_285876 = {}
    # Getting the type of 'np' (line 1212)
    np_285873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1212)
    ones_285874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1212, 15), np_285873, 'ones')
    # Calling ones(args, kwargs) (line 1212)
    ones_call_result_285877 = invoke(stypy.reporting.localization.Localization(__file__, 1212, 15), ones_285874, *[M_285875], **kwargs_285876)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 8), 'stypy_return_type', ones_call_result_285877)
    # SSA join for if statement (line 1211)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1213):
    
    # Assigning a Subscript to a Name (line 1213):
    
    # Obtaining the type of the subscript
    int_285878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 4), 'int')
    
    # Call to _extend(...): (line 1213)
    # Processing the call arguments (line 1213)
    # Getting the type of 'M' (line 1213)
    M_285880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 29), 'M', False)
    # Getting the type of 'sym' (line 1213)
    sym_285881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 32), 'sym', False)
    # Processing the call keyword arguments (line 1213)
    kwargs_285882 = {}
    # Getting the type of '_extend' (line 1213)
    _extend_285879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1213)
    _extend_call_result_285883 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 21), _extend_285879, *[M_285880, sym_285881], **kwargs_285882)
    
    # Obtaining the member '__getitem__' of a type (line 1213)
    getitem___285884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 4), _extend_call_result_285883, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1213)
    subscript_call_result_285885 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 4), getitem___285884, int_285878)
    
    # Assigning a type to the variable 'tuple_var_assignment_284698' (line 1213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'tuple_var_assignment_284698', subscript_call_result_285885)
    
    # Assigning a Subscript to a Name (line 1213):
    
    # Obtaining the type of the subscript
    int_285886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1213, 4), 'int')
    
    # Call to _extend(...): (line 1213)
    # Processing the call arguments (line 1213)
    # Getting the type of 'M' (line 1213)
    M_285888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 29), 'M', False)
    # Getting the type of 'sym' (line 1213)
    sym_285889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 32), 'sym', False)
    # Processing the call keyword arguments (line 1213)
    kwargs_285890 = {}
    # Getting the type of '_extend' (line 1213)
    _extend_285887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1213)
    _extend_call_result_285891 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 21), _extend_285887, *[M_285888, sym_285889], **kwargs_285890)
    
    # Obtaining the member '__getitem__' of a type (line 1213)
    getitem___285892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 4), _extend_call_result_285891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1213)
    subscript_call_result_285893 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 4), getitem___285892, int_285886)
    
    # Assigning a type to the variable 'tuple_var_assignment_284699' (line 1213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'tuple_var_assignment_284699', subscript_call_result_285893)
    
    # Assigning a Name to a Name (line 1213):
    # Getting the type of 'tuple_var_assignment_284698' (line 1213)
    tuple_var_assignment_284698_285894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'tuple_var_assignment_284698')
    # Assigning a type to the variable 'M' (line 1213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'M', tuple_var_assignment_284698_285894)
    
    # Assigning a Name to a Name (line 1213):
    # Getting the type of 'tuple_var_assignment_284699' (line 1213)
    tuple_var_assignment_284699_285895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 4), 'tuple_var_assignment_284699')
    # Assigning a type to the variable 'needs_trunc' (line 1213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1213, 7), 'needs_trunc', tuple_var_assignment_284699_285895)
    
    # Assigning a BinOp to a Name (line 1215):
    
    # Assigning a BinOp to a Name (line 1215):
    
    # Call to arange(...): (line 1215)
    # Processing the call arguments (line 1215)
    int_285898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1215, 18), 'int')
    # Getting the type of 'M' (line 1215)
    M_285899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 21), 'M', False)
    # Processing the call keyword arguments (line 1215)
    kwargs_285900 = {}
    # Getting the type of 'np' (line 1215)
    np_285896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 1215)
    arange_285897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1215, 8), np_285896, 'arange')
    # Calling arange(args, kwargs) (line 1215)
    arange_call_result_285901 = invoke(stypy.reporting.localization.Localization(__file__, 1215, 8), arange_285897, *[int_285898, M_285899], **kwargs_285900)
    
    # Getting the type of 'M' (line 1215)
    M_285902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1215, 27), 'M')
    float_285903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1215, 31), 'float')
    # Applying the binary operator '-' (line 1215)
    result_sub_285904 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 27), '-', M_285902, float_285903)
    
    float_285905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1215, 38), 'float')
    # Applying the binary operator 'div' (line 1215)
    result_div_285906 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 26), 'div', result_sub_285904, float_285905)
    
    # Applying the binary operator '-' (line 1215)
    result_sub_285907 = python_operator(stypy.reporting.localization.Localization(__file__, 1215, 8), '-', arange_call_result_285901, result_div_285906)
    
    # Assigning a type to the variable 'n' (line 1215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1215, 4), 'n', result_sub_285907)
    
    # Assigning a BinOp to a Name (line 1216):
    
    # Assigning a BinOp to a Name (line 1216):
    int_285908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 11), 'int')
    # Getting the type of 'std' (line 1216)
    std_285909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 15), 'std')
    # Applying the binary operator '*' (line 1216)
    result_mul_285910 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 11), '*', int_285908, std_285909)
    
    # Getting the type of 'std' (line 1216)
    std_285911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1216, 21), 'std')
    # Applying the binary operator '*' (line 1216)
    result_mul_285912 = python_operator(stypy.reporting.localization.Localization(__file__, 1216, 19), '*', result_mul_285910, std_285911)
    
    # Assigning a type to the variable 'sig2' (line 1216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 4), 'sig2', result_mul_285912)
    
    # Assigning a Call to a Name (line 1217):
    
    # Assigning a Call to a Name (line 1217):
    
    # Call to exp(...): (line 1217)
    # Processing the call arguments (line 1217)
    
    # Getting the type of 'n' (line 1217)
    n_285915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 16), 'n', False)
    int_285916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1217, 21), 'int')
    # Applying the binary operator '**' (line 1217)
    result_pow_285917 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 16), '**', n_285915, int_285916)
    
    # Applying the 'usub' unary operator (line 1217)
    result___neg___285918 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 15), 'usub', result_pow_285917)
    
    # Getting the type of 'sig2' (line 1217)
    sig2_285919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 25), 'sig2', False)
    # Applying the binary operator 'div' (line 1217)
    result_div_285920 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 15), 'div', result___neg___285918, sig2_285919)
    
    # Processing the call keyword arguments (line 1217)
    kwargs_285921 = {}
    # Getting the type of 'np' (line 1217)
    np_285913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1217)
    exp_285914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1217, 8), np_285913, 'exp')
    # Calling exp(args, kwargs) (line 1217)
    exp_call_result_285922 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 8), exp_285914, *[result_div_285920], **kwargs_285921)
    
    # Assigning a type to the variable 'w' (line 1217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 4), 'w', exp_call_result_285922)
    
    # Call to _truncate(...): (line 1219)
    # Processing the call arguments (line 1219)
    # Getting the type of 'w' (line 1219)
    w_285924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1219)
    needs_trunc_285925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1219)
    kwargs_285926 = {}
    # Getting the type of '_truncate' (line 1219)
    _truncate_285923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1219)
    _truncate_call_result_285927 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 11), _truncate_285923, *[w_285924, needs_trunc_285925], **kwargs_285926)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 4), 'stypy_return_type', _truncate_call_result_285927)
    
    # ################# End of 'gaussian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gaussian' in the type store
    # Getting the type of 'stypy_return_type' (line 1159)
    stypy_return_type_285928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gaussian'
    return stypy_return_type_285928

# Assigning a type to the variable 'gaussian' (line 1159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 0), 'gaussian', gaussian)

@norecursion
def general_gaussian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1222)
    True_285929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 36), 'True')
    defaults = [True_285929]
    # Create a new context for function 'general_gaussian'
    module_type_store = module_type_store.open_function_context('general_gaussian', 1222, 0, False)
    
    # Passed parameters checking function
    general_gaussian.stypy_localization = localization
    general_gaussian.stypy_type_of_self = None
    general_gaussian.stypy_type_store = module_type_store
    general_gaussian.stypy_function_name = 'general_gaussian'
    general_gaussian.stypy_param_names_list = ['M', 'p', 'sig', 'sym']
    general_gaussian.stypy_varargs_param_name = None
    general_gaussian.stypy_kwargs_param_name = None
    general_gaussian.stypy_call_defaults = defaults
    general_gaussian.stypy_call_varargs = varargs
    general_gaussian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'general_gaussian', ['M', 'p', 'sig', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'general_gaussian', localization, ['M', 'p', 'sig', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'general_gaussian(...)' code ##################

    str_285930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1281, (-1)), 'str', 'Return a window with a generalized Gaussian shape.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    p : float\n        Shape parameter.  p = 1 is identical to `gaussian`, p = 0.5 is\n        the same shape as the Laplace distribution.\n    sig : float\n        The standard deviation, sigma.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The generalized Gaussian window is defined as\n\n    .. math::  w(n) = e^{ -\\frac{1}{2}\\left|\\frac{n}{\\sigma}\\right|^{2p} }\n\n    the half-power point is at\n\n    .. math::  (2 \\log(2))^{1/(2 p)} \\sigma\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.general_gaussian(51, p=1.5, sig=7)\n    >>> plt.plot(window)\n    >>> plt.title(r"Generalized Gaussian window (p=1.5, $\\sigma$=7)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title(r"Freq. resp. of the gen. Gaussian "\n    ...           "window (p=1.5, $\\sigma$=7)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 1282)
    # Processing the call arguments (line 1282)
    # Getting the type of 'M' (line 1282)
    M_285932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 19), 'M', False)
    # Processing the call keyword arguments (line 1282)
    kwargs_285933 = {}
    # Getting the type of '_len_guards' (line 1282)
    _len_guards_285931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1282, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1282)
    _len_guards_call_result_285934 = invoke(stypy.reporting.localization.Localization(__file__, 1282, 7), _len_guards_285931, *[M_285932], **kwargs_285933)
    
    # Testing the type of an if condition (line 1282)
    if_condition_285935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1282, 4), _len_guards_call_result_285934)
    # Assigning a type to the variable 'if_condition_285935' (line 1282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1282, 4), 'if_condition_285935', if_condition_285935)
    # SSA begins for if statement (line 1282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1283)
    # Processing the call arguments (line 1283)
    # Getting the type of 'M' (line 1283)
    M_285938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 23), 'M', False)
    # Processing the call keyword arguments (line 1283)
    kwargs_285939 = {}
    # Getting the type of 'np' (line 1283)
    np_285936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1283, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1283)
    ones_285937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1283, 15), np_285936, 'ones')
    # Calling ones(args, kwargs) (line 1283)
    ones_call_result_285940 = invoke(stypy.reporting.localization.Localization(__file__, 1283, 15), ones_285937, *[M_285938], **kwargs_285939)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1283, 8), 'stypy_return_type', ones_call_result_285940)
    # SSA join for if statement (line 1282)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1284):
    
    # Assigning a Subscript to a Name (line 1284):
    
    # Obtaining the type of the subscript
    int_285941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 4), 'int')
    
    # Call to _extend(...): (line 1284)
    # Processing the call arguments (line 1284)
    # Getting the type of 'M' (line 1284)
    M_285943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 29), 'M', False)
    # Getting the type of 'sym' (line 1284)
    sym_285944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 32), 'sym', False)
    # Processing the call keyword arguments (line 1284)
    kwargs_285945 = {}
    # Getting the type of '_extend' (line 1284)
    _extend_285942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1284)
    _extend_call_result_285946 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 21), _extend_285942, *[M_285943, sym_285944], **kwargs_285945)
    
    # Obtaining the member '__getitem__' of a type (line 1284)
    getitem___285947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1284, 4), _extend_call_result_285946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1284)
    subscript_call_result_285948 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 4), getitem___285947, int_285941)
    
    # Assigning a type to the variable 'tuple_var_assignment_284700' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'tuple_var_assignment_284700', subscript_call_result_285948)
    
    # Assigning a Subscript to a Name (line 1284):
    
    # Obtaining the type of the subscript
    int_285949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1284, 4), 'int')
    
    # Call to _extend(...): (line 1284)
    # Processing the call arguments (line 1284)
    # Getting the type of 'M' (line 1284)
    M_285951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 29), 'M', False)
    # Getting the type of 'sym' (line 1284)
    sym_285952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 32), 'sym', False)
    # Processing the call keyword arguments (line 1284)
    kwargs_285953 = {}
    # Getting the type of '_extend' (line 1284)
    _extend_285950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1284)
    _extend_call_result_285954 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 21), _extend_285950, *[M_285951, sym_285952], **kwargs_285953)
    
    # Obtaining the member '__getitem__' of a type (line 1284)
    getitem___285955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1284, 4), _extend_call_result_285954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1284)
    subscript_call_result_285956 = invoke(stypy.reporting.localization.Localization(__file__, 1284, 4), getitem___285955, int_285949)
    
    # Assigning a type to the variable 'tuple_var_assignment_284701' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'tuple_var_assignment_284701', subscript_call_result_285956)
    
    # Assigning a Name to a Name (line 1284):
    # Getting the type of 'tuple_var_assignment_284700' (line 1284)
    tuple_var_assignment_284700_285957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'tuple_var_assignment_284700')
    # Assigning a type to the variable 'M' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'M', tuple_var_assignment_284700_285957)
    
    # Assigning a Name to a Name (line 1284):
    # Getting the type of 'tuple_var_assignment_284701' (line 1284)
    tuple_var_assignment_284701_285958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1284, 4), 'tuple_var_assignment_284701')
    # Assigning a type to the variable 'needs_trunc' (line 1284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1284, 7), 'needs_trunc', tuple_var_assignment_284701_285958)
    
    # Assigning a BinOp to a Name (line 1286):
    
    # Assigning a BinOp to a Name (line 1286):
    
    # Call to arange(...): (line 1286)
    # Processing the call arguments (line 1286)
    int_285961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, 18), 'int')
    # Getting the type of 'M' (line 1286)
    M_285962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 21), 'M', False)
    # Processing the call keyword arguments (line 1286)
    kwargs_285963 = {}
    # Getting the type of 'np' (line 1286)
    np_285959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 1286)
    arange_285960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1286, 8), np_285959, 'arange')
    # Calling arange(args, kwargs) (line 1286)
    arange_call_result_285964 = invoke(stypy.reporting.localization.Localization(__file__, 1286, 8), arange_285960, *[int_285961, M_285962], **kwargs_285963)
    
    # Getting the type of 'M' (line 1286)
    M_285965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1286, 27), 'M')
    float_285966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, 31), 'float')
    # Applying the binary operator '-' (line 1286)
    result_sub_285967 = python_operator(stypy.reporting.localization.Localization(__file__, 1286, 27), '-', M_285965, float_285966)
    
    float_285968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, 38), 'float')
    # Applying the binary operator 'div' (line 1286)
    result_div_285969 = python_operator(stypy.reporting.localization.Localization(__file__, 1286, 26), 'div', result_sub_285967, float_285968)
    
    # Applying the binary operator '-' (line 1286)
    result_sub_285970 = python_operator(stypy.reporting.localization.Localization(__file__, 1286, 8), '-', arange_call_result_285964, result_div_285969)
    
    # Assigning a type to the variable 'n' (line 1286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1286, 4), 'n', result_sub_285970)
    
    # Assigning a Call to a Name (line 1287):
    
    # Assigning a Call to a Name (line 1287):
    
    # Call to exp(...): (line 1287)
    # Processing the call arguments (line 1287)
    float_285973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 15), 'float')
    
    # Call to abs(...): (line 1287)
    # Processing the call arguments (line 1287)
    # Getting the type of 'n' (line 1287)
    n_285976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 29), 'n', False)
    # Getting the type of 'sig' (line 1287)
    sig_285977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 33), 'sig', False)
    # Applying the binary operator 'div' (line 1287)
    result_div_285978 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 29), 'div', n_285976, sig_285977)
    
    # Processing the call keyword arguments (line 1287)
    kwargs_285979 = {}
    # Getting the type of 'np' (line 1287)
    np_285974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 1287)
    abs_285975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 22), np_285974, 'abs')
    # Calling abs(args, kwargs) (line 1287)
    abs_call_result_285980 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 22), abs_285975, *[result_div_285978], **kwargs_285979)
    
    int_285981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 42), 'int')
    # Getting the type of 'p' (line 1287)
    p_285982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 46), 'p', False)
    # Applying the binary operator '*' (line 1287)
    result_mul_285983 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 42), '*', int_285981, p_285982)
    
    # Applying the binary operator '**' (line 1287)
    result_pow_285984 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 22), '**', abs_call_result_285980, result_mul_285983)
    
    # Applying the binary operator '*' (line 1287)
    result_mul_285985 = python_operator(stypy.reporting.localization.Localization(__file__, 1287, 15), '*', float_285973, result_pow_285984)
    
    # Processing the call keyword arguments (line 1287)
    kwargs_285986 = {}
    # Getting the type of 'np' (line 1287)
    np_285971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1287)
    exp_285972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1287, 8), np_285971, 'exp')
    # Calling exp(args, kwargs) (line 1287)
    exp_call_result_285987 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 8), exp_285972, *[result_mul_285985], **kwargs_285986)
    
    # Assigning a type to the variable 'w' (line 1287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 4), 'w', exp_call_result_285987)
    
    # Call to _truncate(...): (line 1289)
    # Processing the call arguments (line 1289)
    # Getting the type of 'w' (line 1289)
    w_285989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1289)
    needs_trunc_285990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1289)
    kwargs_285991 = {}
    # Getting the type of '_truncate' (line 1289)
    _truncate_285988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1289)
    _truncate_call_result_285992 = invoke(stypy.reporting.localization.Localization(__file__, 1289, 11), _truncate_285988, *[w_285989, needs_trunc_285990], **kwargs_285991)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'stypy_return_type', _truncate_call_result_285992)
    
    # ################# End of 'general_gaussian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'general_gaussian' in the type store
    # Getting the type of 'stypy_return_type' (line 1222)
    stypy_return_type_285993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285993)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'general_gaussian'
    return stypy_return_type_285993

# Assigning a type to the variable 'general_gaussian' (line 1222)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 0), 'general_gaussian', general_gaussian)

@norecursion
def chebwin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1293)
    True_285994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 23), 'True')
    defaults = [True_285994]
    # Create a new context for function 'chebwin'
    module_type_store = module_type_store.open_function_context('chebwin', 1293, 0, False)
    
    # Passed parameters checking function
    chebwin.stypy_localization = localization
    chebwin.stypy_type_of_self = None
    chebwin.stypy_type_store = module_type_store
    chebwin.stypy_function_name = 'chebwin'
    chebwin.stypy_param_names_list = ['M', 'at', 'sym']
    chebwin.stypy_varargs_param_name = None
    chebwin.stypy_kwargs_param_name = None
    chebwin.stypy_call_defaults = defaults
    chebwin.stypy_call_varargs = varargs
    chebwin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chebwin', ['M', 'at', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chebwin', localization, ['M', 'at', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chebwin(...)' code ##################

    str_285995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1377, (-1)), 'str', 'Return a Dolph-Chebyshev window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    at : float\n        Attenuation (in dB).\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value always normalized to 1\n\n    Notes\n    -----\n    This window optimizes for the narrowest main lobe width for a given order\n    `M` and sidelobe equiripple attenuation `at`, using Chebyshev\n    polynomials.  It was originally developed by Dolph to optimize the\n    directionality of radio antenna arrays.\n\n    Unlike most windows, the Dolph-Chebyshev is defined in terms of its\n    frequency response:\n\n    .. math:: W(k) = \\frac\n              {\\cos\\{M \\cos^{-1}[\\beta \\cos(\\frac{\\pi k}{M})]\\}}\n              {\\cosh[M \\cosh^{-1}(\\beta)]}\n\n    where\n\n    .. math:: \\beta = \\cosh \\left [\\frac{1}{M}\n              \\cosh^{-1}(10^\\frac{A}{20}) \\right ]\n\n    and 0 <= abs(k) <= M-1. A is the attenuation in decibels (`at`).\n\n    The time domain window is then generated using the IFFT, so\n    power-of-two `M` are the fastest to generate, and prime number `M` are\n    the slowest.\n\n    The equiripple condition in the frequency domain creates impulses in the\n    time domain, which appear at the ends of the window.\n\n    References\n    ----------\n    .. [1] C. Dolph, "A current distribution for broadside arrays which\n           optimizes the relationship between beam width and side-lobe level",\n           Proceedings of the IEEE, Vol. 34, Issue 6\n    .. [2] Peter Lynch, "The Dolph-Chebyshev Window: A Simple Optimal Filter",\n           American Meteorological Society (April 1997)\n           http://mathsci.ucd.ie/~plynch/Publications/Dolph.pdf\n    .. [3] F. J. Harris, "On the use of windows for harmonic analysis with the\n           discrete Fourier transforms", Proceedings of the IEEE, Vol. 66,\n           No. 1, January 1978\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.chebwin(51, at=100)\n    >>> plt.plot(window)\n    >>> plt.title("Dolph-Chebyshev window (100 dB)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Dolph-Chebyshev window (100 dB)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    
    # Call to abs(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'at' (line 1378)
    at_285998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 14), 'at', False)
    # Processing the call keyword arguments (line 1378)
    kwargs_285999 = {}
    # Getting the type of 'np' (line 1378)
    np_285996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 7), 'np', False)
    # Obtaining the member 'abs' of a type (line 1378)
    abs_285997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1378, 7), np_285996, 'abs')
    # Calling abs(args, kwargs) (line 1378)
    abs_call_result_286000 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 7), abs_285997, *[at_285998], **kwargs_285999)
    
    int_286001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 20), 'int')
    # Applying the binary operator '<' (line 1378)
    result_lt_286002 = python_operator(stypy.reporting.localization.Localization(__file__, 1378, 7), '<', abs_call_result_286000, int_286001)
    
    # Testing the type of an if condition (line 1378)
    if_condition_286003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1378, 4), result_lt_286002)
    # Assigning a type to the variable 'if_condition_286003' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'if_condition_286003', if_condition_286003)
    # SSA begins for if statement (line 1378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1379)
    # Processing the call arguments (line 1379)
    str_286006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1379, 22), 'str', 'This window is not suitable for spectral analysis for attenuation values lower than about 45dB because the equivalent noise bandwidth of a Chebyshev window does not grow monotonically with increasing sidelobe attenuation when the attenuation is smaller than about 45 dB.')
    # Processing the call keyword arguments (line 1379)
    kwargs_286007 = {}
    # Getting the type of 'warnings' (line 1379)
    warnings_286004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1379)
    warn_286005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1379, 8), warnings_286004, 'warn')
    # Calling warn(args, kwargs) (line 1379)
    warn_call_result_286008 = invoke(stypy.reporting.localization.Localization(__file__, 1379, 8), warn_286005, *[str_286006], **kwargs_286007)
    
    # SSA join for if statement (line 1378)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to _len_guards(...): (line 1385)
    # Processing the call arguments (line 1385)
    # Getting the type of 'M' (line 1385)
    M_286010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 19), 'M', False)
    # Processing the call keyword arguments (line 1385)
    kwargs_286011 = {}
    # Getting the type of '_len_guards' (line 1385)
    _len_guards_286009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1385)
    _len_guards_call_result_286012 = invoke(stypy.reporting.localization.Localization(__file__, 1385, 7), _len_guards_286009, *[M_286010], **kwargs_286011)
    
    # Testing the type of an if condition (line 1385)
    if_condition_286013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1385, 4), _len_guards_call_result_286012)
    # Assigning a type to the variable 'if_condition_286013' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'if_condition_286013', if_condition_286013)
    # SSA begins for if statement (line 1385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1386)
    # Processing the call arguments (line 1386)
    # Getting the type of 'M' (line 1386)
    M_286016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 23), 'M', False)
    # Processing the call keyword arguments (line 1386)
    kwargs_286017 = {}
    # Getting the type of 'np' (line 1386)
    np_286014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1386, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1386)
    ones_286015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1386, 15), np_286014, 'ones')
    # Calling ones(args, kwargs) (line 1386)
    ones_call_result_286018 = invoke(stypy.reporting.localization.Localization(__file__, 1386, 15), ones_286015, *[M_286016], **kwargs_286017)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1386, 8), 'stypy_return_type', ones_call_result_286018)
    # SSA join for if statement (line 1385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1387):
    
    # Assigning a Subscript to a Name (line 1387):
    
    # Obtaining the type of the subscript
    int_286019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 4), 'int')
    
    # Call to _extend(...): (line 1387)
    # Processing the call arguments (line 1387)
    # Getting the type of 'M' (line 1387)
    M_286021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 29), 'M', False)
    # Getting the type of 'sym' (line 1387)
    sym_286022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 32), 'sym', False)
    # Processing the call keyword arguments (line 1387)
    kwargs_286023 = {}
    # Getting the type of '_extend' (line 1387)
    _extend_286020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1387)
    _extend_call_result_286024 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 21), _extend_286020, *[M_286021, sym_286022], **kwargs_286023)
    
    # Obtaining the member '__getitem__' of a type (line 1387)
    getitem___286025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1387, 4), _extend_call_result_286024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1387)
    subscript_call_result_286026 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 4), getitem___286025, int_286019)
    
    # Assigning a type to the variable 'tuple_var_assignment_284702' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'tuple_var_assignment_284702', subscript_call_result_286026)
    
    # Assigning a Subscript to a Name (line 1387):
    
    # Obtaining the type of the subscript
    int_286027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1387, 4), 'int')
    
    # Call to _extend(...): (line 1387)
    # Processing the call arguments (line 1387)
    # Getting the type of 'M' (line 1387)
    M_286029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 29), 'M', False)
    # Getting the type of 'sym' (line 1387)
    sym_286030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 32), 'sym', False)
    # Processing the call keyword arguments (line 1387)
    kwargs_286031 = {}
    # Getting the type of '_extend' (line 1387)
    _extend_286028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1387)
    _extend_call_result_286032 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 21), _extend_286028, *[M_286029, sym_286030], **kwargs_286031)
    
    # Obtaining the member '__getitem__' of a type (line 1387)
    getitem___286033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1387, 4), _extend_call_result_286032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1387)
    subscript_call_result_286034 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 4), getitem___286033, int_286027)
    
    # Assigning a type to the variable 'tuple_var_assignment_284703' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'tuple_var_assignment_284703', subscript_call_result_286034)
    
    # Assigning a Name to a Name (line 1387):
    # Getting the type of 'tuple_var_assignment_284702' (line 1387)
    tuple_var_assignment_284702_286035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'tuple_var_assignment_284702')
    # Assigning a type to the variable 'M' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'M', tuple_var_assignment_284702_286035)
    
    # Assigning a Name to a Name (line 1387):
    # Getting the type of 'tuple_var_assignment_284703' (line 1387)
    tuple_var_assignment_284703_286036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'tuple_var_assignment_284703')
    # Assigning a type to the variable 'needs_trunc' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 7), 'needs_trunc', tuple_var_assignment_284703_286036)
    
    # Assigning a BinOp to a Name (line 1390):
    
    # Assigning a BinOp to a Name (line 1390):
    # Getting the type of 'M' (line 1390)
    M_286037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1390, 12), 'M')
    float_286038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1390, 16), 'float')
    # Applying the binary operator '-' (line 1390)
    result_sub_286039 = python_operator(stypy.reporting.localization.Localization(__file__, 1390, 12), '-', M_286037, float_286038)
    
    # Assigning a type to the variable 'order' (line 1390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1390, 4), 'order', result_sub_286039)
    
    # Assigning a Call to a Name (line 1391):
    
    # Assigning a Call to a Name (line 1391):
    
    # Call to cosh(...): (line 1391)
    # Processing the call arguments (line 1391)
    float_286042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1391, 19), 'float')
    # Getting the type of 'order' (line 1391)
    order_286043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 25), 'order', False)
    # Applying the binary operator 'div' (line 1391)
    result_div_286044 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 19), 'div', float_286042, order_286043)
    
    
    # Call to arccosh(...): (line 1391)
    # Processing the call arguments (line 1391)
    int_286047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1391, 44), 'int')
    
    # Call to abs(...): (line 1391)
    # Processing the call arguments (line 1391)
    # Getting the type of 'at' (line 1391)
    at_286050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 58), 'at', False)
    # Processing the call keyword arguments (line 1391)
    kwargs_286051 = {}
    # Getting the type of 'np' (line 1391)
    np_286048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 51), 'np', False)
    # Obtaining the member 'abs' of a type (line 1391)
    abs_286049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1391, 51), np_286048, 'abs')
    # Calling abs(args, kwargs) (line 1391)
    abs_call_result_286052 = invoke(stypy.reporting.localization.Localization(__file__, 1391, 51), abs_286049, *[at_286050], **kwargs_286051)
    
    float_286053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1391, 64), 'float')
    # Applying the binary operator 'div' (line 1391)
    result_div_286054 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 51), 'div', abs_call_result_286052, float_286053)
    
    # Applying the binary operator '**' (line 1391)
    result_pow_286055 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 44), '**', int_286047, result_div_286054)
    
    # Processing the call keyword arguments (line 1391)
    kwargs_286056 = {}
    # Getting the type of 'np' (line 1391)
    np_286045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 33), 'np', False)
    # Obtaining the member 'arccosh' of a type (line 1391)
    arccosh_286046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1391, 33), np_286045, 'arccosh')
    # Calling arccosh(args, kwargs) (line 1391)
    arccosh_call_result_286057 = invoke(stypy.reporting.localization.Localization(__file__, 1391, 33), arccosh_286046, *[result_pow_286055], **kwargs_286056)
    
    # Applying the binary operator '*' (line 1391)
    result_mul_286058 = python_operator(stypy.reporting.localization.Localization(__file__, 1391, 31), '*', result_div_286044, arccosh_call_result_286057)
    
    # Processing the call keyword arguments (line 1391)
    kwargs_286059 = {}
    # Getting the type of 'np' (line 1391)
    np_286040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 11), 'np', False)
    # Obtaining the member 'cosh' of a type (line 1391)
    cosh_286041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1391, 11), np_286040, 'cosh')
    # Calling cosh(args, kwargs) (line 1391)
    cosh_call_result_286060 = invoke(stypy.reporting.localization.Localization(__file__, 1391, 11), cosh_286041, *[result_mul_286058], **kwargs_286059)
    
    # Assigning a type to the variable 'beta' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 4), 'beta', cosh_call_result_286060)
    
    # Assigning a BinOp to a Name (line 1392):
    
    # Assigning a BinOp to a Name (line 1392):
    
    # Obtaining the type of the subscript
    int_286061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 14), 'int')
    # Getting the type of 'M' (line 1392)
    M_286062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 16), 'M')
    slice_286063 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1392, 8), int_286061, M_286062, None)
    # Getting the type of 'np' (line 1392)
    np_286064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 8), 'np')
    # Obtaining the member 'r_' of a type (line 1392)
    r__286065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 8), np_286064, 'r_')
    # Obtaining the member '__getitem__' of a type (line 1392)
    getitem___286066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1392, 8), r__286065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1392)
    subscript_call_result_286067 = invoke(stypy.reporting.localization.Localization(__file__, 1392, 8), getitem___286066, slice_286063)
    
    float_286068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 21), 'float')
    # Applying the binary operator '*' (line 1392)
    result_mul_286069 = python_operator(stypy.reporting.localization.Localization(__file__, 1392, 8), '*', subscript_call_result_286067, float_286068)
    
    # Assigning a type to the variable 'k' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 4), 'k', result_mul_286069)
    
    # Assigning a BinOp to a Name (line 1393):
    
    # Assigning a BinOp to a Name (line 1393):
    # Getting the type of 'beta' (line 1393)
    beta_286070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 8), 'beta')
    
    # Call to cos(...): (line 1393)
    # Processing the call arguments (line 1393)
    # Getting the type of 'np' (line 1393)
    np_286073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 22), 'np', False)
    # Obtaining the member 'pi' of a type (line 1393)
    pi_286074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1393, 22), np_286073, 'pi')
    # Getting the type of 'k' (line 1393)
    k_286075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 30), 'k', False)
    # Applying the binary operator '*' (line 1393)
    result_mul_286076 = python_operator(stypy.reporting.localization.Localization(__file__, 1393, 22), '*', pi_286074, k_286075)
    
    # Getting the type of 'M' (line 1393)
    M_286077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 34), 'M', False)
    # Applying the binary operator 'div' (line 1393)
    result_div_286078 = python_operator(stypy.reporting.localization.Localization(__file__, 1393, 32), 'div', result_mul_286076, M_286077)
    
    # Processing the call keyword arguments (line 1393)
    kwargs_286079 = {}
    # Getting the type of 'np' (line 1393)
    np_286071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1393, 15), 'np', False)
    # Obtaining the member 'cos' of a type (line 1393)
    cos_286072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1393, 15), np_286071, 'cos')
    # Calling cos(args, kwargs) (line 1393)
    cos_call_result_286080 = invoke(stypy.reporting.localization.Localization(__file__, 1393, 15), cos_286072, *[result_div_286078], **kwargs_286079)
    
    # Applying the binary operator '*' (line 1393)
    result_mul_286081 = python_operator(stypy.reporting.localization.Localization(__file__, 1393, 8), '*', beta_286070, cos_call_result_286080)
    
    # Assigning a type to the variable 'x' (line 1393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1393, 4), 'x', result_mul_286081)
    
    # Assigning a Call to a Name (line 1397):
    
    # Assigning a Call to a Name (line 1397):
    
    # Call to zeros(...): (line 1397)
    # Processing the call arguments (line 1397)
    # Getting the type of 'x' (line 1397)
    x_286084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 17), 'x', False)
    # Obtaining the member 'shape' of a type (line 1397)
    shape_286085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 17), x_286084, 'shape')
    # Processing the call keyword arguments (line 1397)
    kwargs_286086 = {}
    # Getting the type of 'np' (line 1397)
    np_286082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1397)
    zeros_286083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1397, 8), np_286082, 'zeros')
    # Calling zeros(args, kwargs) (line 1397)
    zeros_call_result_286087 = invoke(stypy.reporting.localization.Localization(__file__, 1397, 8), zeros_286083, *[shape_286085], **kwargs_286086)
    
    # Assigning a type to the variable 'p' (line 1397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 4), 'p', zeros_call_result_286087)
    
    # Assigning a Call to a Subscript (line 1398):
    
    # Assigning a Call to a Subscript (line 1398):
    
    # Call to cosh(...): (line 1398)
    # Processing the call arguments (line 1398)
    # Getting the type of 'order' (line 1398)
    order_286090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 23), 'order', False)
    
    # Call to arccosh(...): (line 1398)
    # Processing the call arguments (line 1398)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'x' (line 1398)
    x_286093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 44), 'x', False)
    int_286094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 48), 'int')
    # Applying the binary operator '>' (line 1398)
    result_gt_286095 = python_operator(stypy.reporting.localization.Localization(__file__, 1398, 44), '>', x_286093, int_286094)
    
    # Getting the type of 'x' (line 1398)
    x_286096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 42), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 1398)
    getitem___286097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 42), x_286096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1398)
    subscript_call_result_286098 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 42), getitem___286097, result_gt_286095)
    
    # Processing the call keyword arguments (line 1398)
    kwargs_286099 = {}
    # Getting the type of 'np' (line 1398)
    np_286091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 31), 'np', False)
    # Obtaining the member 'arccosh' of a type (line 1398)
    arccosh_286092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 31), np_286091, 'arccosh')
    # Calling arccosh(args, kwargs) (line 1398)
    arccosh_call_result_286100 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 31), arccosh_286092, *[subscript_call_result_286098], **kwargs_286099)
    
    # Applying the binary operator '*' (line 1398)
    result_mul_286101 = python_operator(stypy.reporting.localization.Localization(__file__, 1398, 23), '*', order_286090, arccosh_call_result_286100)
    
    # Processing the call keyword arguments (line 1398)
    kwargs_286102 = {}
    # Getting the type of 'np' (line 1398)
    np_286088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 15), 'np', False)
    # Obtaining the member 'cosh' of a type (line 1398)
    cosh_286089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1398, 15), np_286088, 'cosh')
    # Calling cosh(args, kwargs) (line 1398)
    cosh_call_result_286103 = invoke(stypy.reporting.localization.Localization(__file__, 1398, 15), cosh_286089, *[result_mul_286101], **kwargs_286102)
    
    # Getting the type of 'p' (line 1398)
    p_286104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 4), 'p')
    
    # Getting the type of 'x' (line 1398)
    x_286105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 6), 'x')
    int_286106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1398, 10), 'int')
    # Applying the binary operator '>' (line 1398)
    result_gt_286107 = python_operator(stypy.reporting.localization.Localization(__file__, 1398, 6), '>', x_286105, int_286106)
    
    # Storing an element on a container (line 1398)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1398, 4), p_286104, (result_gt_286107, cosh_call_result_286103))
    
    # Assigning a BinOp to a Subscript (line 1399):
    
    # Assigning a BinOp to a Subscript (line 1399):
    int_286108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 17), 'int')
    int_286109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 21), 'int')
    # Getting the type of 'order' (line 1399)
    order_286110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 26), 'order')
    int_286111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 34), 'int')
    # Applying the binary operator '%' (line 1399)
    result_mod_286112 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 26), '%', order_286110, int_286111)
    
    # Applying the binary operator '*' (line 1399)
    result_mul_286113 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 21), '*', int_286109, result_mod_286112)
    
    # Applying the binary operator '-' (line 1399)
    result_sub_286114 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 17), '-', int_286108, result_mul_286113)
    
    
    # Call to cosh(...): (line 1399)
    # Processing the call arguments (line 1399)
    # Getting the type of 'order' (line 1399)
    order_286117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 48), 'order', False)
    
    # Call to arccosh(...): (line 1399)
    # Processing the call arguments (line 1399)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'x' (line 1399)
    x_286120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 70), 'x', False)
    int_286121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 74), 'int')
    # Applying the binary operator '<' (line 1399)
    result_lt_286122 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 70), '<', x_286120, int_286121)
    
    # Getting the type of 'x' (line 1399)
    x_286123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 68), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 1399)
    getitem___286124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1399, 68), x_286123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1399)
    subscript_call_result_286125 = invoke(stypy.reporting.localization.Localization(__file__, 1399, 68), getitem___286124, result_lt_286122)
    
    # Applying the 'usub' unary operator (line 1399)
    result___neg___286126 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 67), 'usub', subscript_call_result_286125)
    
    # Processing the call keyword arguments (line 1399)
    kwargs_286127 = {}
    # Getting the type of 'np' (line 1399)
    np_286118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 56), 'np', False)
    # Obtaining the member 'arccosh' of a type (line 1399)
    arccosh_286119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1399, 56), np_286118, 'arccosh')
    # Calling arccosh(args, kwargs) (line 1399)
    arccosh_call_result_286128 = invoke(stypy.reporting.localization.Localization(__file__, 1399, 56), arccosh_286119, *[result___neg___286126], **kwargs_286127)
    
    # Applying the binary operator '*' (line 1399)
    result_mul_286129 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 48), '*', order_286117, arccosh_call_result_286128)
    
    # Processing the call keyword arguments (line 1399)
    kwargs_286130 = {}
    # Getting the type of 'np' (line 1399)
    np_286115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 40), 'np', False)
    # Obtaining the member 'cosh' of a type (line 1399)
    cosh_286116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1399, 40), np_286115, 'cosh')
    # Calling cosh(args, kwargs) (line 1399)
    cosh_call_result_286131 = invoke(stypy.reporting.localization.Localization(__file__, 1399, 40), cosh_286116, *[result_mul_286129], **kwargs_286130)
    
    # Applying the binary operator '*' (line 1399)
    result_mul_286132 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 16), '*', result_sub_286114, cosh_call_result_286131)
    
    # Getting the type of 'p' (line 1399)
    p_286133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 4), 'p')
    
    # Getting the type of 'x' (line 1399)
    x_286134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1399, 6), 'x')
    int_286135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1399, 10), 'int')
    # Applying the binary operator '<' (line 1399)
    result_lt_286136 = python_operator(stypy.reporting.localization.Localization(__file__, 1399, 6), '<', x_286134, int_286135)
    
    # Storing an element on a container (line 1399)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1399, 4), p_286133, (result_lt_286136, result_mul_286132))
    
    # Assigning a Call to a Subscript (line 1400):
    
    # Assigning a Call to a Subscript (line 1400):
    
    # Call to cos(...): (line 1400)
    # Processing the call arguments (line 1400)
    # Getting the type of 'order' (line 1400)
    order_286139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 31), 'order', False)
    
    # Call to arccos(...): (line 1400)
    # Processing the call arguments (line 1400)
    
    # Obtaining the type of the subscript
    
    
    # Call to abs(...): (line 1400)
    # Processing the call arguments (line 1400)
    # Getting the type of 'x' (line 1400)
    x_286144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 58), 'x', False)
    # Processing the call keyword arguments (line 1400)
    kwargs_286145 = {}
    # Getting the type of 'np' (line 1400)
    np_286142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 51), 'np', False)
    # Obtaining the member 'abs' of a type (line 1400)
    abs_286143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 51), np_286142, 'abs')
    # Calling abs(args, kwargs) (line 1400)
    abs_call_result_286146 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 51), abs_286143, *[x_286144], **kwargs_286145)
    
    int_286147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1400, 64), 'int')
    # Applying the binary operator '<=' (line 1400)
    result_le_286148 = python_operator(stypy.reporting.localization.Localization(__file__, 1400, 51), '<=', abs_call_result_286146, int_286147)
    
    # Getting the type of 'x' (line 1400)
    x_286149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 49), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 1400)
    getitem___286150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 49), x_286149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1400)
    subscript_call_result_286151 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 49), getitem___286150, result_le_286148)
    
    # Processing the call keyword arguments (line 1400)
    kwargs_286152 = {}
    # Getting the type of 'np' (line 1400)
    np_286140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 39), 'np', False)
    # Obtaining the member 'arccos' of a type (line 1400)
    arccos_286141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 39), np_286140, 'arccos')
    # Calling arccos(args, kwargs) (line 1400)
    arccos_call_result_286153 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 39), arccos_286141, *[subscript_call_result_286151], **kwargs_286152)
    
    # Applying the binary operator '*' (line 1400)
    result_mul_286154 = python_operator(stypy.reporting.localization.Localization(__file__, 1400, 31), '*', order_286139, arccos_call_result_286153)
    
    # Processing the call keyword arguments (line 1400)
    kwargs_286155 = {}
    # Getting the type of 'np' (line 1400)
    np_286137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 24), 'np', False)
    # Obtaining the member 'cos' of a type (line 1400)
    cos_286138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 24), np_286137, 'cos')
    # Calling cos(args, kwargs) (line 1400)
    cos_call_result_286156 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 24), cos_286138, *[result_mul_286154], **kwargs_286155)
    
    # Getting the type of 'p' (line 1400)
    p_286157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 4), 'p')
    
    
    # Call to abs(...): (line 1400)
    # Processing the call arguments (line 1400)
    # Getting the type of 'x' (line 1400)
    x_286160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 13), 'x', False)
    # Processing the call keyword arguments (line 1400)
    kwargs_286161 = {}
    # Getting the type of 'np' (line 1400)
    np_286158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1400, 6), 'np', False)
    # Obtaining the member 'abs' of a type (line 1400)
    abs_286159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1400, 6), np_286158, 'abs')
    # Calling abs(args, kwargs) (line 1400)
    abs_call_result_286162 = invoke(stypy.reporting.localization.Localization(__file__, 1400, 6), abs_286159, *[x_286160], **kwargs_286161)
    
    int_286163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1400, 19), 'int')
    # Applying the binary operator '<=' (line 1400)
    result_le_286164 = python_operator(stypy.reporting.localization.Localization(__file__, 1400, 6), '<=', abs_call_result_286162, int_286163)
    
    # Storing an element on a container (line 1400)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1400, 4), p_286157, (result_le_286164, cos_call_result_286156))
    
    # Getting the type of 'M' (line 1404)
    M_286165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1404, 7), 'M')
    int_286166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1404, 11), 'int')
    # Applying the binary operator '%' (line 1404)
    result_mod_286167 = python_operator(stypy.reporting.localization.Localization(__file__, 1404, 7), '%', M_286165, int_286166)
    
    # Testing the type of an if condition (line 1404)
    if_condition_286168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1404, 4), result_mod_286167)
    # Assigning a type to the variable 'if_condition_286168' (line 1404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1404, 4), 'if_condition_286168', if_condition_286168)
    # SSA begins for if statement (line 1404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1405):
    
    # Assigning a Call to a Name (line 1405):
    
    # Call to real(...): (line 1405)
    # Processing the call arguments (line 1405)
    
    # Call to fft(...): (line 1405)
    # Processing the call arguments (line 1405)
    # Getting the type of 'p' (line 1405)
    p_286173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 32), 'p', False)
    # Processing the call keyword arguments (line 1405)
    kwargs_286174 = {}
    # Getting the type of 'fftpack' (line 1405)
    fftpack_286171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 20), 'fftpack', False)
    # Obtaining the member 'fft' of a type (line 1405)
    fft_286172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1405, 20), fftpack_286171, 'fft')
    # Calling fft(args, kwargs) (line 1405)
    fft_call_result_286175 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 20), fft_286172, *[p_286173], **kwargs_286174)
    
    # Processing the call keyword arguments (line 1405)
    kwargs_286176 = {}
    # Getting the type of 'np' (line 1405)
    np_286169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1405, 12), 'np', False)
    # Obtaining the member 'real' of a type (line 1405)
    real_286170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1405, 12), np_286169, 'real')
    # Calling real(args, kwargs) (line 1405)
    real_call_result_286177 = invoke(stypy.reporting.localization.Localization(__file__, 1405, 12), real_286170, *[fft_call_result_286175], **kwargs_286176)
    
    # Assigning a type to the variable 'w' (line 1405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1405, 8), 'w', real_call_result_286177)
    
    # Assigning a BinOp to a Name (line 1406):
    
    # Assigning a BinOp to a Name (line 1406):
    # Getting the type of 'M' (line 1406)
    M_286178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1406, 13), 'M')
    int_286179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1406, 17), 'int')
    # Applying the binary operator '+' (line 1406)
    result_add_286180 = python_operator(stypy.reporting.localization.Localization(__file__, 1406, 13), '+', M_286178, int_286179)
    
    int_286181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1406, 23), 'int')
    # Applying the binary operator '//' (line 1406)
    result_floordiv_286182 = python_operator(stypy.reporting.localization.Localization(__file__, 1406, 12), '//', result_add_286180, int_286181)
    
    # Assigning a type to the variable 'n' (line 1406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1406, 8), 'n', result_floordiv_286182)
    
    # Assigning a Subscript to a Name (line 1407):
    
    # Assigning a Subscript to a Name (line 1407):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1407)
    n_286183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 15), 'n')
    slice_286184 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1407, 12), None, n_286183, None)
    # Getting the type of 'w' (line 1407)
    w_286185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1407, 12), 'w')
    # Obtaining the member '__getitem__' of a type (line 1407)
    getitem___286186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1407, 12), w_286185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1407)
    subscript_call_result_286187 = invoke(stypy.reporting.localization.Localization(__file__, 1407, 12), getitem___286186, slice_286184)
    
    # Assigning a type to the variable 'w' (line 1407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1407, 8), 'w', subscript_call_result_286187)
    
    # Assigning a Call to a Name (line 1408):
    
    # Assigning a Call to a Name (line 1408):
    
    # Call to concatenate(...): (line 1408)
    # Processing the call arguments (line 1408)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1408)
    tuple_286190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1408, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1408)
    # Adding element type (line 1408)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1408)
    n_286191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 30), 'n', False)
    int_286192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1408, 34), 'int')
    # Applying the binary operator '-' (line 1408)
    result_sub_286193 = python_operator(stypy.reporting.localization.Localization(__file__, 1408, 30), '-', n_286191, int_286192)
    
    int_286194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1408, 36), 'int')
    int_286195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1408, 38), 'int')
    slice_286196 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1408, 28), result_sub_286193, int_286194, int_286195)
    # Getting the type of 'w' (line 1408)
    w_286197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 28), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1408)
    getitem___286198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1408, 28), w_286197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1408)
    subscript_call_result_286199 = invoke(stypy.reporting.localization.Localization(__file__, 1408, 28), getitem___286198, slice_286196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 28), tuple_286190, subscript_call_result_286199)
    # Adding element type (line 1408)
    # Getting the type of 'w' (line 1408)
    w_286200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 43), 'w', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1408, 28), tuple_286190, w_286200)
    
    # Processing the call keyword arguments (line 1408)
    kwargs_286201 = {}
    # Getting the type of 'np' (line 1408)
    np_286188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1408, 12), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1408)
    concatenate_286189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1408, 12), np_286188, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1408)
    concatenate_call_result_286202 = invoke(stypy.reporting.localization.Localization(__file__, 1408, 12), concatenate_286189, *[tuple_286190], **kwargs_286201)
    
    # Assigning a type to the variable 'w' (line 1408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1408, 8), 'w', concatenate_call_result_286202)
    # SSA branch for the else part of an if statement (line 1404)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 1410):
    
    # Assigning a BinOp to a Name (line 1410):
    # Getting the type of 'p' (line 1410)
    p_286203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 12), 'p')
    
    # Call to exp(...): (line 1410)
    # Processing the call arguments (line 1410)
    complex_286206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1410, 23), 'complex')
    # Getting the type of 'np' (line 1410)
    np_286207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 1410)
    pi_286208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1410, 29), np_286207, 'pi')
    # Applying the binary operator '*' (line 1410)
    result_mul_286209 = python_operator(stypy.reporting.localization.Localization(__file__, 1410, 23), '*', complex_286206, pi_286208)
    
    # Getting the type of 'M' (line 1410)
    M_286210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 37), 'M', False)
    # Applying the binary operator 'div' (line 1410)
    result_div_286211 = python_operator(stypy.reporting.localization.Localization(__file__, 1410, 35), 'div', result_mul_286209, M_286210)
    
    
    # Obtaining the type of the subscript
    int_286212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1410, 47), 'int')
    # Getting the type of 'M' (line 1410)
    M_286213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 49), 'M', False)
    slice_286214 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1410, 41), int_286212, M_286213, None)
    # Getting the type of 'np' (line 1410)
    np_286215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 41), 'np', False)
    # Obtaining the member 'r_' of a type (line 1410)
    r__286216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1410, 41), np_286215, 'r_')
    # Obtaining the member '__getitem__' of a type (line 1410)
    getitem___286217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1410, 41), r__286216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1410)
    subscript_call_result_286218 = invoke(stypy.reporting.localization.Localization(__file__, 1410, 41), getitem___286217, slice_286214)
    
    # Applying the binary operator '*' (line 1410)
    result_mul_286219 = python_operator(stypy.reporting.localization.Localization(__file__, 1410, 39), '*', result_div_286211, subscript_call_result_286218)
    
    # Processing the call keyword arguments (line 1410)
    kwargs_286220 = {}
    # Getting the type of 'np' (line 1410)
    np_286204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1410, 16), 'np', False)
    # Obtaining the member 'exp' of a type (line 1410)
    exp_286205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1410, 16), np_286204, 'exp')
    # Calling exp(args, kwargs) (line 1410)
    exp_call_result_286221 = invoke(stypy.reporting.localization.Localization(__file__, 1410, 16), exp_286205, *[result_mul_286219], **kwargs_286220)
    
    # Applying the binary operator '*' (line 1410)
    result_mul_286222 = python_operator(stypy.reporting.localization.Localization(__file__, 1410, 12), '*', p_286203, exp_call_result_286221)
    
    # Assigning a type to the variable 'p' (line 1410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1410, 8), 'p', result_mul_286222)
    
    # Assigning a Call to a Name (line 1411):
    
    # Assigning a Call to a Name (line 1411):
    
    # Call to real(...): (line 1411)
    # Processing the call arguments (line 1411)
    
    # Call to fft(...): (line 1411)
    # Processing the call arguments (line 1411)
    # Getting the type of 'p' (line 1411)
    p_286227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 32), 'p', False)
    # Processing the call keyword arguments (line 1411)
    kwargs_286228 = {}
    # Getting the type of 'fftpack' (line 1411)
    fftpack_286225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 20), 'fftpack', False)
    # Obtaining the member 'fft' of a type (line 1411)
    fft_286226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1411, 20), fftpack_286225, 'fft')
    # Calling fft(args, kwargs) (line 1411)
    fft_call_result_286229 = invoke(stypy.reporting.localization.Localization(__file__, 1411, 20), fft_286226, *[p_286227], **kwargs_286228)
    
    # Processing the call keyword arguments (line 1411)
    kwargs_286230 = {}
    # Getting the type of 'np' (line 1411)
    np_286223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 12), 'np', False)
    # Obtaining the member 'real' of a type (line 1411)
    real_286224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1411, 12), np_286223, 'real')
    # Calling real(args, kwargs) (line 1411)
    real_call_result_286231 = invoke(stypy.reporting.localization.Localization(__file__, 1411, 12), real_286224, *[fft_call_result_286229], **kwargs_286230)
    
    # Assigning a type to the variable 'w' (line 1411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1411, 8), 'w', real_call_result_286231)
    
    # Assigning a BinOp to a Name (line 1412):
    
    # Assigning a BinOp to a Name (line 1412):
    # Getting the type of 'M' (line 1412)
    M_286232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 12), 'M')
    int_286233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 17), 'int')
    # Applying the binary operator '//' (line 1412)
    result_floordiv_286234 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 12), '//', M_286232, int_286233)
    
    int_286235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 21), 'int')
    # Applying the binary operator '+' (line 1412)
    result_add_286236 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 12), '+', result_floordiv_286234, int_286235)
    
    # Assigning a type to the variable 'n' (line 1412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1412, 8), 'n', result_add_286236)
    
    # Assigning a Call to a Name (line 1413):
    
    # Assigning a Call to a Name (line 1413):
    
    # Call to concatenate(...): (line 1413)
    # Processing the call arguments (line 1413)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1413)
    tuple_286239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1413)
    # Adding element type (line 1413)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1413)
    n_286240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 30), 'n', False)
    int_286241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 34), 'int')
    # Applying the binary operator '-' (line 1413)
    result_sub_286242 = python_operator(stypy.reporting.localization.Localization(__file__, 1413, 30), '-', n_286240, int_286241)
    
    int_286243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 36), 'int')
    int_286244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 38), 'int')
    slice_286245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1413, 28), result_sub_286242, int_286243, int_286244)
    # Getting the type of 'w' (line 1413)
    w_286246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 28), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1413)
    getitem___286247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1413, 28), w_286246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1413)
    subscript_call_result_286248 = invoke(stypy.reporting.localization.Localization(__file__, 1413, 28), getitem___286247, slice_286245)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1413, 28), tuple_286239, subscript_call_result_286248)
    # Adding element type (line 1413)
    
    # Obtaining the type of the subscript
    int_286249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 45), 'int')
    # Getting the type of 'n' (line 1413)
    n_286250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 47), 'n', False)
    slice_286251 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1413, 43), int_286249, n_286250, None)
    # Getting the type of 'w' (line 1413)
    w_286252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 43), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1413)
    getitem___286253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1413, 43), w_286252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1413)
    subscript_call_result_286254 = invoke(stypy.reporting.localization.Localization(__file__, 1413, 43), getitem___286253, slice_286251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1413, 28), tuple_286239, subscript_call_result_286254)
    
    # Processing the call keyword arguments (line 1413)
    kwargs_286255 = {}
    # Getting the type of 'np' (line 1413)
    np_286237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 12), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1413)
    concatenate_286238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1413, 12), np_286237, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1413)
    concatenate_call_result_286256 = invoke(stypy.reporting.localization.Localization(__file__, 1413, 12), concatenate_286238, *[tuple_286239], **kwargs_286255)
    
    # Assigning a type to the variable 'w' (line 1413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1413, 8), 'w', concatenate_call_result_286256)
    # SSA join for if statement (line 1404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1414):
    
    # Assigning a BinOp to a Name (line 1414):
    # Getting the type of 'w' (line 1414)
    w_286257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 8), 'w')
    
    # Call to max(...): (line 1414)
    # Processing the call arguments (line 1414)
    # Getting the type of 'w' (line 1414)
    w_286259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 16), 'w', False)
    # Processing the call keyword arguments (line 1414)
    kwargs_286260 = {}
    # Getting the type of 'max' (line 1414)
    max_286258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 12), 'max', False)
    # Calling max(args, kwargs) (line 1414)
    max_call_result_286261 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 12), max_286258, *[w_286259], **kwargs_286260)
    
    # Applying the binary operator 'div' (line 1414)
    result_div_286262 = python_operator(stypy.reporting.localization.Localization(__file__, 1414, 8), 'div', w_286257, max_call_result_286261)
    
    # Assigning a type to the variable 'w' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'w', result_div_286262)
    
    # Call to _truncate(...): (line 1416)
    # Processing the call arguments (line 1416)
    # Getting the type of 'w' (line 1416)
    w_286264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1416)
    needs_trunc_286265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1416)
    kwargs_286266 = {}
    # Getting the type of '_truncate' (line 1416)
    _truncate_286263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1416)
    _truncate_call_result_286267 = invoke(stypy.reporting.localization.Localization(__file__, 1416, 11), _truncate_286263, *[w_286264, needs_trunc_286265], **kwargs_286266)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1416, 4), 'stypy_return_type', _truncate_call_result_286267)
    
    # ################# End of 'chebwin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chebwin' in the type store
    # Getting the type of 'stypy_return_type' (line 1293)
    stypy_return_type_286268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chebwin'
    return stypy_return_type_286268

# Assigning a type to the variable 'chebwin' (line 1293)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1293, 0), 'chebwin', chebwin)

@norecursion
def slepian(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1419)
    True_286269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1419, 26), 'True')
    defaults = [True_286269]
    # Create a new context for function 'slepian'
    module_type_store = module_type_store.open_function_context('slepian', 1419, 0, False)
    
    # Passed parameters checking function
    slepian.stypy_localization = localization
    slepian.stypy_type_of_self = None
    slepian.stypy_type_store = module_type_store
    slepian.stypy_function_name = 'slepian'
    slepian.stypy_param_names_list = ['M', 'width', 'sym']
    slepian.stypy_varargs_param_name = None
    slepian.stypy_kwargs_param_name = None
    slepian.stypy_call_defaults = defaults
    slepian.stypy_call_varargs = varargs
    slepian.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'slepian', ['M', 'width', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'slepian', localization, ['M', 'width', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'slepian(...)' code ##################

    str_286270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1475, (-1)), 'str', 'Return a digital Slepian (DPSS) window.\n\n    Used to maximize the energy concentration in the main lobe.  Also called\n    the digital prolate spheroidal sequence (DPSS).\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    width : float\n        Bandwidth\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value always normalized to 1\n\n    References\n    ----------\n    .. [1] D. Slepian & H. O. Pollak: "Prolate spheroidal wave functions,\n           Fourier analysis and uncertainty-I," Bell Syst. Tech. J., vol.40,\n           pp.43-63, 1961. https://archive.org/details/bstj40-1-43\n    .. [2] H. J. Landau & H. O. Pollak: "Prolate spheroidal wave functions,\n           Fourier analysis and uncertainty-II," Bell Syst. Tech. J. , vol.40,\n           pp.65-83, 1961. https://archive.org/details/bstj40-1-65\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.slepian(51, width=0.3)\n    >>> plt.plot(window)\n    >>> plt.title("Slepian (DPSS) window (BW=0.3)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the Slepian window (BW=0.3)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    ')
    
    
    # Call to _len_guards(...): (line 1476)
    # Processing the call arguments (line 1476)
    # Getting the type of 'M' (line 1476)
    M_286272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 19), 'M', False)
    # Processing the call keyword arguments (line 1476)
    kwargs_286273 = {}
    # Getting the type of '_len_guards' (line 1476)
    _len_guards_286271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1476, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1476)
    _len_guards_call_result_286274 = invoke(stypy.reporting.localization.Localization(__file__, 1476, 7), _len_guards_286271, *[M_286272], **kwargs_286273)
    
    # Testing the type of an if condition (line 1476)
    if_condition_286275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1476, 4), _len_guards_call_result_286274)
    # Assigning a type to the variable 'if_condition_286275' (line 1476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1476, 4), 'if_condition_286275', if_condition_286275)
    # SSA begins for if statement (line 1476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1477)
    # Processing the call arguments (line 1477)
    # Getting the type of 'M' (line 1477)
    M_286278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 23), 'M', False)
    # Processing the call keyword arguments (line 1477)
    kwargs_286279 = {}
    # Getting the type of 'np' (line 1477)
    np_286276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1477, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1477)
    ones_286277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1477, 15), np_286276, 'ones')
    # Calling ones(args, kwargs) (line 1477)
    ones_call_result_286280 = invoke(stypy.reporting.localization.Localization(__file__, 1477, 15), ones_286277, *[M_286278], **kwargs_286279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1477, 8), 'stypy_return_type', ones_call_result_286280)
    # SSA join for if statement (line 1476)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1478):
    
    # Assigning a Subscript to a Name (line 1478):
    
    # Obtaining the type of the subscript
    int_286281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1478, 4), 'int')
    
    # Call to _extend(...): (line 1478)
    # Processing the call arguments (line 1478)
    # Getting the type of 'M' (line 1478)
    M_286283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 29), 'M', False)
    # Getting the type of 'sym' (line 1478)
    sym_286284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 32), 'sym', False)
    # Processing the call keyword arguments (line 1478)
    kwargs_286285 = {}
    # Getting the type of '_extend' (line 1478)
    _extend_286282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1478)
    _extend_call_result_286286 = invoke(stypy.reporting.localization.Localization(__file__, 1478, 21), _extend_286282, *[M_286283, sym_286284], **kwargs_286285)
    
    # Obtaining the member '__getitem__' of a type (line 1478)
    getitem___286287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1478, 4), _extend_call_result_286286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1478)
    subscript_call_result_286288 = invoke(stypy.reporting.localization.Localization(__file__, 1478, 4), getitem___286287, int_286281)
    
    # Assigning a type to the variable 'tuple_var_assignment_284704' (line 1478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1478, 4), 'tuple_var_assignment_284704', subscript_call_result_286288)
    
    # Assigning a Subscript to a Name (line 1478):
    
    # Obtaining the type of the subscript
    int_286289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1478, 4), 'int')
    
    # Call to _extend(...): (line 1478)
    # Processing the call arguments (line 1478)
    # Getting the type of 'M' (line 1478)
    M_286291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 29), 'M', False)
    # Getting the type of 'sym' (line 1478)
    sym_286292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 32), 'sym', False)
    # Processing the call keyword arguments (line 1478)
    kwargs_286293 = {}
    # Getting the type of '_extend' (line 1478)
    _extend_286290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1478)
    _extend_call_result_286294 = invoke(stypy.reporting.localization.Localization(__file__, 1478, 21), _extend_286290, *[M_286291, sym_286292], **kwargs_286293)
    
    # Obtaining the member '__getitem__' of a type (line 1478)
    getitem___286295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1478, 4), _extend_call_result_286294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1478)
    subscript_call_result_286296 = invoke(stypy.reporting.localization.Localization(__file__, 1478, 4), getitem___286295, int_286289)
    
    # Assigning a type to the variable 'tuple_var_assignment_284705' (line 1478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1478, 4), 'tuple_var_assignment_284705', subscript_call_result_286296)
    
    # Assigning a Name to a Name (line 1478):
    # Getting the type of 'tuple_var_assignment_284704' (line 1478)
    tuple_var_assignment_284704_286297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 4), 'tuple_var_assignment_284704')
    # Assigning a type to the variable 'M' (line 1478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1478, 4), 'M', tuple_var_assignment_284704_286297)
    
    # Assigning a Name to a Name (line 1478):
    # Getting the type of 'tuple_var_assignment_284705' (line 1478)
    tuple_var_assignment_284705_286298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1478, 4), 'tuple_var_assignment_284705')
    # Assigning a type to the variable 'needs_trunc' (line 1478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1478, 7), 'needs_trunc', tuple_var_assignment_284705_286298)
    
    # Assigning a BinOp to a Name (line 1481):
    
    # Assigning a BinOp to a Name (line 1481):
    # Getting the type of 'width' (line 1481)
    width_286299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1481, 12), 'width')
    int_286300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1481, 20), 'int')
    # Applying the binary operator 'div' (line 1481)
    result_div_286301 = python_operator(stypy.reporting.localization.Localization(__file__, 1481, 12), 'div', width_286299, int_286300)
    
    # Assigning a type to the variable 'width' (line 1481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1481, 4), 'width', result_div_286301)
    
    # Assigning a BinOp to a Name (line 1483):
    
    # Assigning a BinOp to a Name (line 1483):
    # Getting the type of 'width' (line 1483)
    width_286302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1483, 12), 'width')
    int_286303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1483, 20), 'int')
    # Applying the binary operator 'div' (line 1483)
    result_div_286304 = python_operator(stypy.reporting.localization.Localization(__file__, 1483, 12), 'div', width_286302, int_286303)
    
    # Assigning a type to the variable 'width' (line 1483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1483, 4), 'width', result_div_286304)
    
    # Assigning a Call to a Name (line 1484):
    
    # Assigning a Call to a Name (line 1484):
    
    # Call to arange(...): (line 1484)
    # Processing the call arguments (line 1484)
    # Getting the type of 'M' (line 1484)
    M_286307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 18), 'M', False)
    # Processing the call keyword arguments (line 1484)
    str_286308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1484, 27), 'str', 'd')
    keyword_286309 = str_286308
    kwargs_286310 = {'dtype': keyword_286309}
    # Getting the type of 'np' (line 1484)
    np_286305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1484, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 1484)
    arange_286306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1484, 8), np_286305, 'arange')
    # Calling arange(args, kwargs) (line 1484)
    arange_call_result_286311 = invoke(stypy.reporting.localization.Localization(__file__, 1484, 8), arange_286306, *[M_286307], **kwargs_286310)
    
    # Assigning a type to the variable 'm' (line 1484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1484, 4), 'm', arange_call_result_286311)
    
    # Assigning a Call to a Name (line 1485):
    
    # Assigning a Call to a Name (line 1485):
    
    # Call to zeros(...): (line 1485)
    # Processing the call arguments (line 1485)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1485)
    tuple_286314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1485, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1485)
    # Adding element type (line 1485)
    int_286315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1485, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1485, 18), tuple_286314, int_286315)
    # Adding element type (line 1485)
    # Getting the type of 'M' (line 1485)
    M_286316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 21), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1485, 18), tuple_286314, M_286316)
    
    # Processing the call keyword arguments (line 1485)
    kwargs_286317 = {}
    # Getting the type of 'np' (line 1485)
    np_286312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1485)
    zeros_286313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1485, 8), np_286312, 'zeros')
    # Calling zeros(args, kwargs) (line 1485)
    zeros_call_result_286318 = invoke(stypy.reporting.localization.Localization(__file__, 1485, 8), zeros_286313, *[tuple_286314], **kwargs_286317)
    
    # Assigning a type to the variable 'H' (line 1485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1485, 4), 'H', zeros_call_result_286318)
    
    # Assigning a BinOp to a Subscript (line 1486):
    
    # Assigning a BinOp to a Subscript (line 1486):
    
    # Obtaining the type of the subscript
    int_286319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 17), 'int')
    slice_286320 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1486, 15), int_286319, None, None)
    # Getting the type of 'm' (line 1486)
    m_286321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 15), 'm')
    # Obtaining the member '__getitem__' of a type (line 1486)
    getitem___286322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1486, 15), m_286321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1486)
    subscript_call_result_286323 = invoke(stypy.reporting.localization.Localization(__file__, 1486, 15), getitem___286322, slice_286320)
    
    # Getting the type of 'M' (line 1486)
    M_286324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 24), 'M')
    
    # Obtaining the type of the subscript
    int_286325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 30), 'int')
    slice_286326 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1486, 28), int_286325, None, None)
    # Getting the type of 'm' (line 1486)
    m_286327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 28), 'm')
    # Obtaining the member '__getitem__' of a type (line 1486)
    getitem___286328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1486, 28), m_286327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1486)
    subscript_call_result_286329 = invoke(stypy.reporting.localization.Localization(__file__, 1486, 28), getitem___286328, slice_286326)
    
    # Applying the binary operator '-' (line 1486)
    result_sub_286330 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 24), '-', M_286324, subscript_call_result_286329)
    
    # Applying the binary operator '*' (line 1486)
    result_mul_286331 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 15), '*', subscript_call_result_286323, result_sub_286330)
    
    int_286332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 37), 'int')
    # Applying the binary operator 'div' (line 1486)
    result_div_286333 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 35), 'div', result_mul_286331, int_286332)
    
    # Getting the type of 'H' (line 1486)
    H_286334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 4), 'H')
    int_286335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 6), 'int')
    int_286336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 9), 'int')
    slice_286337 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1486, 4), int_286336, None, None)
    # Storing an element on a container (line 1486)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1486, 4), H_286334, ((int_286335, slice_286337), result_div_286333))
    
    # Assigning a BinOp to a Subscript (line 1487):
    
    # Assigning a BinOp to a Subscript (line 1487):
    # Getting the type of 'M' (line 1487)
    M_286338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 16), 'M')
    int_286339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 20), 'int')
    # Applying the binary operator '-' (line 1487)
    result_sub_286340 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 16), '-', M_286338, int_286339)
    
    int_286341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 24), 'int')
    # Getting the type of 'm' (line 1487)
    m_286342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 28), 'm')
    # Applying the binary operator '*' (line 1487)
    result_mul_286343 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 24), '*', int_286341, m_286342)
    
    # Applying the binary operator '-' (line 1487)
    result_sub_286344 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 22), '-', result_sub_286340, result_mul_286343)
    
    int_286345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 33), 'int')
    # Applying the binary operator 'div' (line 1487)
    result_div_286346 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 15), 'div', result_sub_286344, int_286345)
    
    int_286347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 37), 'int')
    # Applying the binary operator '**' (line 1487)
    result_pow_286348 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 14), '**', result_div_286346, int_286347)
    
    
    # Call to cos(...): (line 1487)
    # Processing the call arguments (line 1487)
    int_286351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 48), 'int')
    # Getting the type of 'np' (line 1487)
    np_286352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 52), 'np', False)
    # Obtaining the member 'pi' of a type (line 1487)
    pi_286353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1487, 52), np_286352, 'pi')
    # Applying the binary operator '*' (line 1487)
    result_mul_286354 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 48), '*', int_286351, pi_286353)
    
    # Getting the type of 'width' (line 1487)
    width_286355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 60), 'width', False)
    # Applying the binary operator '*' (line 1487)
    result_mul_286356 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 58), '*', result_mul_286354, width_286355)
    
    # Processing the call keyword arguments (line 1487)
    kwargs_286357 = {}
    # Getting the type of 'np' (line 1487)
    np_286349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 41), 'np', False)
    # Obtaining the member 'cos' of a type (line 1487)
    cos_286350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1487, 41), np_286349, 'cos')
    # Calling cos(args, kwargs) (line 1487)
    cos_call_result_286358 = invoke(stypy.reporting.localization.Localization(__file__, 1487, 41), cos_286350, *[result_mul_286356], **kwargs_286357)
    
    # Applying the binary operator '*' (line 1487)
    result_mul_286359 = python_operator(stypy.reporting.localization.Localization(__file__, 1487, 14), '*', result_pow_286348, cos_call_result_286358)
    
    # Getting the type of 'H' (line 1487)
    H_286360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 4), 'H')
    int_286361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1487, 6), 'int')
    slice_286362 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1487, 4), None, None, None)
    # Storing an element on a container (line 1487)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1487, 4), H_286360, ((int_286361, slice_286362), result_mul_286359))
    
    # Assigning a Call to a Tuple (line 1489):
    
    # Assigning a Subscript to a Name (line 1489):
    
    # Obtaining the type of the subscript
    int_286363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 4), 'int')
    
    # Call to eig_banded(...): (line 1489)
    # Processing the call arguments (line 1489)
    # Getting the type of 'H' (line 1489)
    H_286366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 31), 'H', False)
    # Processing the call keyword arguments (line 1489)
    str_286367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 41), 'str', 'i')
    keyword_286368 = str_286367
    
    # Obtaining an instance of the builtin type 'tuple' (line 1489)
    tuple_286369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1489)
    # Adding element type (line 1489)
    # Getting the type of 'M' (line 1489)
    M_286370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 60), 'M', False)
    int_286371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 62), 'int')
    # Applying the binary operator '-' (line 1489)
    result_sub_286372 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 60), '-', M_286370, int_286371)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 60), tuple_286369, result_sub_286372)
    # Adding element type (line 1489)
    # Getting the type of 'M' (line 1489)
    M_286373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 65), 'M', False)
    int_286374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 67), 'int')
    # Applying the binary operator '-' (line 1489)
    result_sub_286375 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 65), '-', M_286373, int_286374)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 60), tuple_286369, result_sub_286375)
    
    keyword_286376 = tuple_286369
    kwargs_286377 = {'select_range': keyword_286376, 'select': keyword_286368}
    # Getting the type of 'linalg' (line 1489)
    linalg_286364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 13), 'linalg', False)
    # Obtaining the member 'eig_banded' of a type (line 1489)
    eig_banded_286365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1489, 13), linalg_286364, 'eig_banded')
    # Calling eig_banded(args, kwargs) (line 1489)
    eig_banded_call_result_286378 = invoke(stypy.reporting.localization.Localization(__file__, 1489, 13), eig_banded_286365, *[H_286366], **kwargs_286377)
    
    # Obtaining the member '__getitem__' of a type (line 1489)
    getitem___286379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1489, 4), eig_banded_call_result_286378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1489)
    subscript_call_result_286380 = invoke(stypy.reporting.localization.Localization(__file__, 1489, 4), getitem___286379, int_286363)
    
    # Assigning a type to the variable 'tuple_var_assignment_284706' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'tuple_var_assignment_284706', subscript_call_result_286380)
    
    # Assigning a Subscript to a Name (line 1489):
    
    # Obtaining the type of the subscript
    int_286381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 4), 'int')
    
    # Call to eig_banded(...): (line 1489)
    # Processing the call arguments (line 1489)
    # Getting the type of 'H' (line 1489)
    H_286384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 31), 'H', False)
    # Processing the call keyword arguments (line 1489)
    str_286385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 41), 'str', 'i')
    keyword_286386 = str_286385
    
    # Obtaining an instance of the builtin type 'tuple' (line 1489)
    tuple_286387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1489)
    # Adding element type (line 1489)
    # Getting the type of 'M' (line 1489)
    M_286388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 60), 'M', False)
    int_286389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 62), 'int')
    # Applying the binary operator '-' (line 1489)
    result_sub_286390 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 60), '-', M_286388, int_286389)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 60), tuple_286387, result_sub_286390)
    # Adding element type (line 1489)
    # Getting the type of 'M' (line 1489)
    M_286391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 65), 'M', False)
    int_286392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 67), 'int')
    # Applying the binary operator '-' (line 1489)
    result_sub_286393 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 65), '-', M_286391, int_286392)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 60), tuple_286387, result_sub_286393)
    
    keyword_286394 = tuple_286387
    kwargs_286395 = {'select_range': keyword_286394, 'select': keyword_286386}
    # Getting the type of 'linalg' (line 1489)
    linalg_286382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 13), 'linalg', False)
    # Obtaining the member 'eig_banded' of a type (line 1489)
    eig_banded_286383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1489, 13), linalg_286382, 'eig_banded')
    # Calling eig_banded(args, kwargs) (line 1489)
    eig_banded_call_result_286396 = invoke(stypy.reporting.localization.Localization(__file__, 1489, 13), eig_banded_286383, *[H_286384], **kwargs_286395)
    
    # Obtaining the member '__getitem__' of a type (line 1489)
    getitem___286397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1489, 4), eig_banded_call_result_286396, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1489)
    subscript_call_result_286398 = invoke(stypy.reporting.localization.Localization(__file__, 1489, 4), getitem___286397, int_286381)
    
    # Assigning a type to the variable 'tuple_var_assignment_284707' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'tuple_var_assignment_284707', subscript_call_result_286398)
    
    # Assigning a Name to a Name (line 1489):
    # Getting the type of 'tuple_var_assignment_284706' (line 1489)
    tuple_var_assignment_284706_286399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'tuple_var_assignment_284706')
    # Assigning a type to the variable '_' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), '_', tuple_var_assignment_284706_286399)
    
    # Assigning a Name to a Name (line 1489):
    # Getting the type of 'tuple_var_assignment_284707' (line 1489)
    tuple_var_assignment_284707_286400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'tuple_var_assignment_284707')
    # Assigning a type to the variable 'win' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 7), 'win', tuple_var_assignment_284707_286400)
    
    # Assigning a BinOp to a Name (line 1490):
    
    # Assigning a BinOp to a Name (line 1490):
    
    # Call to ravel(...): (line 1490)
    # Processing the call keyword arguments (line 1490)
    kwargs_286403 = {}
    # Getting the type of 'win' (line 1490)
    win_286401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 10), 'win', False)
    # Obtaining the member 'ravel' of a type (line 1490)
    ravel_286402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 10), win_286401, 'ravel')
    # Calling ravel(args, kwargs) (line 1490)
    ravel_call_result_286404 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 10), ravel_286402, *[], **kwargs_286403)
    
    
    # Call to max(...): (line 1490)
    # Processing the call keyword arguments (line 1490)
    kwargs_286407 = {}
    # Getting the type of 'win' (line 1490)
    win_286405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 24), 'win', False)
    # Obtaining the member 'max' of a type (line 1490)
    max_286406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 24), win_286405, 'max')
    # Calling max(args, kwargs) (line 1490)
    max_call_result_286408 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 24), max_286406, *[], **kwargs_286407)
    
    # Applying the binary operator 'div' (line 1490)
    result_div_286409 = python_operator(stypy.reporting.localization.Localization(__file__, 1490, 10), 'div', ravel_call_result_286404, max_call_result_286408)
    
    # Assigning a type to the variable 'win' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'win', result_div_286409)
    
    # Call to _truncate(...): (line 1492)
    # Processing the call arguments (line 1492)
    # Getting the type of 'win' (line 1492)
    win_286411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 21), 'win', False)
    # Getting the type of 'needs_trunc' (line 1492)
    needs_trunc_286412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 26), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1492)
    kwargs_286413 = {}
    # Getting the type of '_truncate' (line 1492)
    _truncate_286410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1492)
    _truncate_call_result_286414 = invoke(stypy.reporting.localization.Localization(__file__, 1492, 11), _truncate_286410, *[win_286411, needs_trunc_286412], **kwargs_286413)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1492, 4), 'stypy_return_type', _truncate_call_result_286414)
    
    # ################# End of 'slepian(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'slepian' in the type store
    # Getting the type of 'stypy_return_type' (line 1419)
    stypy_return_type_286415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1419, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'slepian'
    return stypy_return_type_286415

# Assigning a type to the variable 'slepian' (line 1419)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1419, 0), 'slepian', slepian)

@norecursion
def cosine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1495)
    True_286416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 18), 'True')
    defaults = [True_286416]
    # Create a new context for function 'cosine'
    module_type_store = module_type_store.open_function_context('cosine', 1495, 0, False)
    
    # Passed parameters checking function
    cosine.stypy_localization = localization
    cosine.stypy_type_of_self = None
    cosine.stypy_type_store = module_type_store
    cosine.stypy_function_name = 'cosine'
    cosine.stypy_param_names_list = ['M', 'sym']
    cosine.stypy_varargs_param_name = None
    cosine.stypy_kwargs_param_name = None
    cosine.stypy_call_defaults = defaults
    cosine.stypy_call_varargs = varargs
    cosine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cosine', ['M', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cosine', localization, ['M', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cosine(...)' code ##################

    str_286417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1544, (-1)), 'str', 'Return a window with a simple cosine shape.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n\n    .. versionadded:: 0.13.0\n\n    Examples\n    --------\n    Plot the window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> window = signal.cosine(51)\n    >>> plt.plot(window)\n    >>> plt.title("Cosine window")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -120, 0])\n    >>> plt.title("Frequency response of the cosine window")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n    >>> plt.show()\n\n    ')
    
    
    # Call to _len_guards(...): (line 1545)
    # Processing the call arguments (line 1545)
    # Getting the type of 'M' (line 1545)
    M_286419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 19), 'M', False)
    # Processing the call keyword arguments (line 1545)
    kwargs_286420 = {}
    # Getting the type of '_len_guards' (line 1545)
    _len_guards_286418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1545)
    _len_guards_call_result_286421 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 7), _len_guards_286418, *[M_286419], **kwargs_286420)
    
    # Testing the type of an if condition (line 1545)
    if_condition_286422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1545, 4), _len_guards_call_result_286421)
    # Assigning a type to the variable 'if_condition_286422' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 4), 'if_condition_286422', if_condition_286422)
    # SSA begins for if statement (line 1545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1546)
    # Processing the call arguments (line 1546)
    # Getting the type of 'M' (line 1546)
    M_286425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 23), 'M', False)
    # Processing the call keyword arguments (line 1546)
    kwargs_286426 = {}
    # Getting the type of 'np' (line 1546)
    np_286423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1546)
    ones_286424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 15), np_286423, 'ones')
    # Calling ones(args, kwargs) (line 1546)
    ones_call_result_286427 = invoke(stypy.reporting.localization.Localization(__file__, 1546, 15), ones_286424, *[M_286425], **kwargs_286426)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 8), 'stypy_return_type', ones_call_result_286427)
    # SSA join for if statement (line 1545)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1547):
    
    # Assigning a Subscript to a Name (line 1547):
    
    # Obtaining the type of the subscript
    int_286428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1547, 4), 'int')
    
    # Call to _extend(...): (line 1547)
    # Processing the call arguments (line 1547)
    # Getting the type of 'M' (line 1547)
    M_286430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 29), 'M', False)
    # Getting the type of 'sym' (line 1547)
    sym_286431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 32), 'sym', False)
    # Processing the call keyword arguments (line 1547)
    kwargs_286432 = {}
    # Getting the type of '_extend' (line 1547)
    _extend_286429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1547)
    _extend_call_result_286433 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 21), _extend_286429, *[M_286430, sym_286431], **kwargs_286432)
    
    # Obtaining the member '__getitem__' of a type (line 1547)
    getitem___286434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 4), _extend_call_result_286433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1547)
    subscript_call_result_286435 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 4), getitem___286434, int_286428)
    
    # Assigning a type to the variable 'tuple_var_assignment_284708' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'tuple_var_assignment_284708', subscript_call_result_286435)
    
    # Assigning a Subscript to a Name (line 1547):
    
    # Obtaining the type of the subscript
    int_286436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1547, 4), 'int')
    
    # Call to _extend(...): (line 1547)
    # Processing the call arguments (line 1547)
    # Getting the type of 'M' (line 1547)
    M_286438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 29), 'M', False)
    # Getting the type of 'sym' (line 1547)
    sym_286439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 32), 'sym', False)
    # Processing the call keyword arguments (line 1547)
    kwargs_286440 = {}
    # Getting the type of '_extend' (line 1547)
    _extend_286437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1547)
    _extend_call_result_286441 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 21), _extend_286437, *[M_286438, sym_286439], **kwargs_286440)
    
    # Obtaining the member '__getitem__' of a type (line 1547)
    getitem___286442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 4), _extend_call_result_286441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1547)
    subscript_call_result_286443 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 4), getitem___286442, int_286436)
    
    # Assigning a type to the variable 'tuple_var_assignment_284709' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'tuple_var_assignment_284709', subscript_call_result_286443)
    
    # Assigning a Name to a Name (line 1547):
    # Getting the type of 'tuple_var_assignment_284708' (line 1547)
    tuple_var_assignment_284708_286444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'tuple_var_assignment_284708')
    # Assigning a type to the variable 'M' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'M', tuple_var_assignment_284708_286444)
    
    # Assigning a Name to a Name (line 1547):
    # Getting the type of 'tuple_var_assignment_284709' (line 1547)
    tuple_var_assignment_284709_286445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'tuple_var_assignment_284709')
    # Assigning a type to the variable 'needs_trunc' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 7), 'needs_trunc', tuple_var_assignment_284709_286445)
    
    # Assigning a Call to a Name (line 1549):
    
    # Assigning a Call to a Name (line 1549):
    
    # Call to sin(...): (line 1549)
    # Processing the call arguments (line 1549)
    # Getting the type of 'np' (line 1549)
    np_286448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 15), 'np', False)
    # Obtaining the member 'pi' of a type (line 1549)
    pi_286449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 15), np_286448, 'pi')
    # Getting the type of 'M' (line 1549)
    M_286450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 23), 'M', False)
    # Applying the binary operator 'div' (line 1549)
    result_div_286451 = python_operator(stypy.reporting.localization.Localization(__file__, 1549, 15), 'div', pi_286449, M_286450)
    
    
    # Call to arange(...): (line 1549)
    # Processing the call arguments (line 1549)
    int_286454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 38), 'int')
    # Getting the type of 'M' (line 1549)
    M_286455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 41), 'M', False)
    # Processing the call keyword arguments (line 1549)
    kwargs_286456 = {}
    # Getting the type of 'np' (line 1549)
    np_286452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 28), 'np', False)
    # Obtaining the member 'arange' of a type (line 1549)
    arange_286453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 28), np_286452, 'arange')
    # Calling arange(args, kwargs) (line 1549)
    arange_call_result_286457 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 28), arange_286453, *[int_286454, M_286455], **kwargs_286456)
    
    float_286458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 46), 'float')
    # Applying the binary operator '+' (line 1549)
    result_add_286459 = python_operator(stypy.reporting.localization.Localization(__file__, 1549, 28), '+', arange_call_result_286457, float_286458)
    
    # Applying the binary operator '*' (line 1549)
    result_mul_286460 = python_operator(stypy.reporting.localization.Localization(__file__, 1549, 25), '*', result_div_286451, result_add_286459)
    
    # Processing the call keyword arguments (line 1549)
    kwargs_286461 = {}
    # Getting the type of 'np' (line 1549)
    np_286446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 8), 'np', False)
    # Obtaining the member 'sin' of a type (line 1549)
    sin_286447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 8), np_286446, 'sin')
    # Calling sin(args, kwargs) (line 1549)
    sin_call_result_286462 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 8), sin_286447, *[result_mul_286460], **kwargs_286461)
    
    # Assigning a type to the variable 'w' (line 1549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1549, 4), 'w', sin_call_result_286462)
    
    # Call to _truncate(...): (line 1551)
    # Processing the call arguments (line 1551)
    # Getting the type of 'w' (line 1551)
    w_286464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1551)
    needs_trunc_286465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1551)
    kwargs_286466 = {}
    # Getting the type of '_truncate' (line 1551)
    _truncate_286463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1551)
    _truncate_call_result_286467 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 11), _truncate_286463, *[w_286464, needs_trunc_286465], **kwargs_286466)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 4), 'stypy_return_type', _truncate_call_result_286467)
    
    # ################# End of 'cosine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cosine' in the type store
    # Getting the type of 'stypy_return_type' (line 1495)
    stypy_return_type_286468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286468)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cosine'
    return stypy_return_type_286468

# Assigning a type to the variable 'cosine' (line 1495)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1495, 0), 'cosine', cosine)

@norecursion
def exponential(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1554)
    None_286469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 26), 'None')
    float_286470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1554, 36), 'float')
    # Getting the type of 'True' (line 1554)
    True_286471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 44), 'True')
    defaults = [None_286469, float_286470, True_286471]
    # Create a new context for function 'exponential'
    module_type_store = module_type_store.open_function_context('exponential', 1554, 0, False)
    
    # Passed parameters checking function
    exponential.stypy_localization = localization
    exponential.stypy_type_of_self = None
    exponential.stypy_type_store = module_type_store
    exponential.stypy_function_name = 'exponential'
    exponential.stypy_param_names_list = ['M', 'center', 'tau', 'sym']
    exponential.stypy_varargs_param_name = None
    exponential.stypy_kwargs_param_name = None
    exponential.stypy_call_defaults = defaults
    exponential.stypy_call_varargs = varargs
    exponential.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exponential', ['M', 'center', 'tau', 'sym'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exponential', localization, ['M', 'center', 'tau', 'sym'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exponential(...)' code ##################

    str_286472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1626, (-1)), 'str', 'Return an exponential (or Poisson) window.\n\n    Parameters\n    ----------\n    M : int\n        Number of points in the output window. If zero or less, an empty\n        array is returned.\n    center : float, optional\n        Parameter defining the center location of the window function.\n        The default value if not given is ``center = (M-1) / 2``.  This\n        parameter must take its default value for symmetric windows.\n    tau : float, optional\n        Parameter defining the decay.  For ``center = 0`` use\n        ``tau = -(M-1) / ln(x)`` if ``x`` is the fraction of the window\n        remaining at the end.\n    sym : bool, optional\n        When True (default), generates a symmetric window, for use in filter\n        design.\n        When False, generates a periodic window, for use in spectral analysis.\n\n    Returns\n    -------\n    w : ndarray\n        The window, with the maximum value normalized to 1 (though the value 1\n        does not appear if `M` is even and `sym` is True).\n\n    Notes\n    -----\n    The Exponential window is defined as\n\n    .. math::  w(n) = e^{-|n-center| / \\tau}\n\n    References\n    ----------\n    S. Gade and H. Herlufsen, "Windows to FFT analysis (Part I)",\n    Technical Review 3, Bruel & Kjaer, 1987.\n\n    Examples\n    --------\n    Plot the symmetric window and its frequency response:\n\n    >>> from scipy import signal\n    >>> from scipy.fftpack import fft, fftshift\n    >>> import matplotlib.pyplot as plt\n\n    >>> M = 51\n    >>> tau = 3.0\n    >>> window = signal.exponential(M, tau=tau)\n    >>> plt.plot(window)\n    >>> plt.title("Exponential Window (tau=3.0)")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n\n    >>> plt.figure()\n    >>> A = fft(window, 2048) / (len(window)/2.0)\n    >>> freq = np.linspace(-0.5, 0.5, len(A))\n    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))\n    >>> plt.plot(freq, response)\n    >>> plt.axis([-0.5, 0.5, -35, 0])\n    >>> plt.title("Frequency response of the Exponential window (tau=3.0)")\n    >>> plt.ylabel("Normalized magnitude [dB]")\n    >>> plt.xlabel("Normalized frequency [cycles per sample]")\n\n    This function can also generate non-symmetric windows:\n\n    >>> tau2 = -(M-1) / np.log(0.01)\n    >>> window2 = signal.exponential(M, 0, tau2, False)\n    >>> plt.figure()\n    >>> plt.plot(window2)\n    >>> plt.ylabel("Amplitude")\n    >>> plt.xlabel("Sample")\n    ')
    
    
    # Evaluating a boolean operation
    # Getting the type of 'sym' (line 1627)
    sym_286473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 7), 'sym')
    
    # Getting the type of 'center' (line 1627)
    center_286474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 15), 'center')
    # Getting the type of 'None' (line 1627)
    None_286475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1627, 29), 'None')
    # Applying the binary operator 'isnot' (line 1627)
    result_is_not_286476 = python_operator(stypy.reporting.localization.Localization(__file__, 1627, 15), 'isnot', center_286474, None_286475)
    
    # Applying the binary operator 'and' (line 1627)
    result_and_keyword_286477 = python_operator(stypy.reporting.localization.Localization(__file__, 1627, 7), 'and', sym_286473, result_is_not_286476)
    
    # Testing the type of an if condition (line 1627)
    if_condition_286478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1627, 4), result_and_keyword_286477)
    # Assigning a type to the variable 'if_condition_286478' (line 1627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1627, 4), 'if_condition_286478', if_condition_286478)
    # SSA begins for if statement (line 1627)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1628)
    # Processing the call arguments (line 1628)
    str_286480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1628, 25), 'str', 'If sym==True, center must be None.')
    # Processing the call keyword arguments (line 1628)
    kwargs_286481 = {}
    # Getting the type of 'ValueError' (line 1628)
    ValueError_286479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1628, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1628)
    ValueError_call_result_286482 = invoke(stypy.reporting.localization.Localization(__file__, 1628, 14), ValueError_286479, *[str_286480], **kwargs_286481)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1628, 8), ValueError_call_result_286482, 'raise parameter', BaseException)
    # SSA join for if statement (line 1627)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to _len_guards(...): (line 1629)
    # Processing the call arguments (line 1629)
    # Getting the type of 'M' (line 1629)
    M_286484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 19), 'M', False)
    # Processing the call keyword arguments (line 1629)
    kwargs_286485 = {}
    # Getting the type of '_len_guards' (line 1629)
    _len_guards_286483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 7), '_len_guards', False)
    # Calling _len_guards(args, kwargs) (line 1629)
    _len_guards_call_result_286486 = invoke(stypy.reporting.localization.Localization(__file__, 1629, 7), _len_guards_286483, *[M_286484], **kwargs_286485)
    
    # Testing the type of an if condition (line 1629)
    if_condition_286487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1629, 4), _len_guards_call_result_286486)
    # Assigning a type to the variable 'if_condition_286487' (line 1629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1629, 4), 'if_condition_286487', if_condition_286487)
    # SSA begins for if statement (line 1629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1630)
    # Processing the call arguments (line 1630)
    # Getting the type of 'M' (line 1630)
    M_286490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 23), 'M', False)
    # Processing the call keyword arguments (line 1630)
    kwargs_286491 = {}
    # Getting the type of 'np' (line 1630)
    np_286488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1630, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1630)
    ones_286489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1630, 15), np_286488, 'ones')
    # Calling ones(args, kwargs) (line 1630)
    ones_call_result_286492 = invoke(stypy.reporting.localization.Localization(__file__, 1630, 15), ones_286489, *[M_286490], **kwargs_286491)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1630, 8), 'stypy_return_type', ones_call_result_286492)
    # SSA join for if statement (line 1629)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1631):
    
    # Assigning a Subscript to a Name (line 1631):
    
    # Obtaining the type of the subscript
    int_286493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 4), 'int')
    
    # Call to _extend(...): (line 1631)
    # Processing the call arguments (line 1631)
    # Getting the type of 'M' (line 1631)
    M_286495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 29), 'M', False)
    # Getting the type of 'sym' (line 1631)
    sym_286496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 32), 'sym', False)
    # Processing the call keyword arguments (line 1631)
    kwargs_286497 = {}
    # Getting the type of '_extend' (line 1631)
    _extend_286494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1631)
    _extend_call_result_286498 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 21), _extend_286494, *[M_286495, sym_286496], **kwargs_286497)
    
    # Obtaining the member '__getitem__' of a type (line 1631)
    getitem___286499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 4), _extend_call_result_286498, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1631)
    subscript_call_result_286500 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 4), getitem___286499, int_286493)
    
    # Assigning a type to the variable 'tuple_var_assignment_284710' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'tuple_var_assignment_284710', subscript_call_result_286500)
    
    # Assigning a Subscript to a Name (line 1631):
    
    # Obtaining the type of the subscript
    int_286501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 4), 'int')
    
    # Call to _extend(...): (line 1631)
    # Processing the call arguments (line 1631)
    # Getting the type of 'M' (line 1631)
    M_286503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 29), 'M', False)
    # Getting the type of 'sym' (line 1631)
    sym_286504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 32), 'sym', False)
    # Processing the call keyword arguments (line 1631)
    kwargs_286505 = {}
    # Getting the type of '_extend' (line 1631)
    _extend_286502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 21), '_extend', False)
    # Calling _extend(args, kwargs) (line 1631)
    _extend_call_result_286506 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 21), _extend_286502, *[M_286503, sym_286504], **kwargs_286505)
    
    # Obtaining the member '__getitem__' of a type (line 1631)
    getitem___286507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 4), _extend_call_result_286506, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1631)
    subscript_call_result_286508 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 4), getitem___286507, int_286501)
    
    # Assigning a type to the variable 'tuple_var_assignment_284711' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'tuple_var_assignment_284711', subscript_call_result_286508)
    
    # Assigning a Name to a Name (line 1631):
    # Getting the type of 'tuple_var_assignment_284710' (line 1631)
    tuple_var_assignment_284710_286509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'tuple_var_assignment_284710')
    # Assigning a type to the variable 'M' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'M', tuple_var_assignment_284710_286509)
    
    # Assigning a Name to a Name (line 1631):
    # Getting the type of 'tuple_var_assignment_284711' (line 1631)
    tuple_var_assignment_284711_286510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 4), 'tuple_var_assignment_284711')
    # Assigning a type to the variable 'needs_trunc' (line 1631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 7), 'needs_trunc', tuple_var_assignment_284711_286510)
    
    # Type idiom detected: calculating its left and rigth part (line 1633)
    # Getting the type of 'center' (line 1633)
    center_286511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 7), 'center')
    # Getting the type of 'None' (line 1633)
    None_286512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1633, 17), 'None')
    
    (may_be_286513, more_types_in_union_286514) = may_be_none(center_286511, None_286512)

    if may_be_286513:

        if more_types_in_union_286514:
            # Runtime conditional SSA (line 1633)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1634):
        
        # Assigning a BinOp to a Name (line 1634):
        # Getting the type of 'M' (line 1634)
        M_286515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1634, 18), 'M')
        int_286516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 20), 'int')
        # Applying the binary operator '-' (line 1634)
        result_sub_286517 = python_operator(stypy.reporting.localization.Localization(__file__, 1634, 18), '-', M_286515, int_286516)
        
        int_286518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 25), 'int')
        # Applying the binary operator 'div' (line 1634)
        result_div_286519 = python_operator(stypy.reporting.localization.Localization(__file__, 1634, 17), 'div', result_sub_286517, int_286518)
        
        # Assigning a type to the variable 'center' (line 1634)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1634, 8), 'center', result_div_286519)

        if more_types_in_union_286514:
            # SSA join for if statement (line 1633)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1636):
    
    # Assigning a Call to a Name (line 1636):
    
    # Call to arange(...): (line 1636)
    # Processing the call arguments (line 1636)
    int_286522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1636, 18), 'int')
    # Getting the type of 'M' (line 1636)
    M_286523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 21), 'M', False)
    # Processing the call keyword arguments (line 1636)
    kwargs_286524 = {}
    # Getting the type of 'np' (line 1636)
    np_286520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 1636)
    arange_286521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 8), np_286520, 'arange')
    # Calling arange(args, kwargs) (line 1636)
    arange_call_result_286525 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 8), arange_286521, *[int_286522, M_286523], **kwargs_286524)
    
    # Assigning a type to the variable 'n' (line 1636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1636, 4), 'n', arange_call_result_286525)
    
    # Assigning a Call to a Name (line 1637):
    
    # Assigning a Call to a Name (line 1637):
    
    # Call to exp(...): (line 1637)
    # Processing the call arguments (line 1637)
    
    
    # Call to abs(...): (line 1637)
    # Processing the call arguments (line 1637)
    # Getting the type of 'n' (line 1637)
    n_286530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 23), 'n', False)
    # Getting the type of 'center' (line 1637)
    center_286531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 25), 'center', False)
    # Applying the binary operator '-' (line 1637)
    result_sub_286532 = python_operator(stypy.reporting.localization.Localization(__file__, 1637, 23), '-', n_286530, center_286531)
    
    # Processing the call keyword arguments (line 1637)
    kwargs_286533 = {}
    # Getting the type of 'np' (line 1637)
    np_286528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 16), 'np', False)
    # Obtaining the member 'abs' of a type (line 1637)
    abs_286529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1637, 16), np_286528, 'abs')
    # Calling abs(args, kwargs) (line 1637)
    abs_call_result_286534 = invoke(stypy.reporting.localization.Localization(__file__, 1637, 16), abs_286529, *[result_sub_286532], **kwargs_286533)
    
    # Applying the 'usub' unary operator (line 1637)
    result___neg___286535 = python_operator(stypy.reporting.localization.Localization(__file__, 1637, 15), 'usub', abs_call_result_286534)
    
    # Getting the type of 'tau' (line 1637)
    tau_286536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 35), 'tau', False)
    # Applying the binary operator 'div' (line 1637)
    result_div_286537 = python_operator(stypy.reporting.localization.Localization(__file__, 1637, 15), 'div', result___neg___286535, tau_286536)
    
    # Processing the call keyword arguments (line 1637)
    kwargs_286538 = {}
    # Getting the type of 'np' (line 1637)
    np_286526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1637, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1637)
    exp_286527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1637, 8), np_286526, 'exp')
    # Calling exp(args, kwargs) (line 1637)
    exp_call_result_286539 = invoke(stypy.reporting.localization.Localization(__file__, 1637, 8), exp_286527, *[result_div_286537], **kwargs_286538)
    
    # Assigning a type to the variable 'w' (line 1637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1637, 4), 'w', exp_call_result_286539)
    
    # Call to _truncate(...): (line 1639)
    # Processing the call arguments (line 1639)
    # Getting the type of 'w' (line 1639)
    w_286541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1639, 21), 'w', False)
    # Getting the type of 'needs_trunc' (line 1639)
    needs_trunc_286542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1639, 24), 'needs_trunc', False)
    # Processing the call keyword arguments (line 1639)
    kwargs_286543 = {}
    # Getting the type of '_truncate' (line 1639)
    _truncate_286540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1639, 11), '_truncate', False)
    # Calling _truncate(args, kwargs) (line 1639)
    _truncate_call_result_286544 = invoke(stypy.reporting.localization.Localization(__file__, 1639, 11), _truncate_286540, *[w_286541, needs_trunc_286542], **kwargs_286543)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1639)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1639, 4), 'stypy_return_type', _truncate_call_result_286544)
    
    # ################# End of 'exponential(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exponential' in the type store
    # Getting the type of 'stypy_return_type' (line 1554)
    stypy_return_type_286545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286545)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exponential'
    return stypy_return_type_286545

# Assigning a type to the variable 'exponential' (line 1554)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1554, 0), 'exponential', exponential)

# Assigning a Dict to a Name (line 1642):

# Assigning a Dict to a Name (line 1642):

# Obtaining an instance of the builtin type 'dict' (line 1642)
dict_286546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1642, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1642)
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1643)
tuple_286547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1643)
# Adding element type (line 1643)
str_286548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 5), 'str', 'barthann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1643, 5), tuple_286547, str_286548)
# Adding element type (line 1643)
str_286549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 17), 'str', 'brthan')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1643, 5), tuple_286547, str_286549)
# Adding element type (line 1643)
str_286550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 27), 'str', 'bth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1643, 5), tuple_286547, str_286550)


# Obtaining an instance of the builtin type 'tuple' (line 1643)
tuple_286551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1643)
# Adding element type (line 1643)
# Getting the type of 'barthann' (line 1643)
barthann_286552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1643, 36), 'barthann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1643, 36), tuple_286551, barthann_286552)
# Adding element type (line 1643)
# Getting the type of 'False' (line 1643)
False_286553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1643, 46), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1643, 36), tuple_286551, False_286553)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286547, tuple_286551))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1644)
tuple_286554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1644)
# Adding element type (line 1644)
str_286555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 5), 'str', 'bartlett')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1644, 5), tuple_286554, str_286555)
# Adding element type (line 1644)
str_286556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 17), 'str', 'bart')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1644, 5), tuple_286554, str_286556)
# Adding element type (line 1644)
str_286557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 25), 'str', 'brt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1644, 5), tuple_286554, str_286557)


# Obtaining an instance of the builtin type 'tuple' (line 1644)
tuple_286558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1644)
# Adding element type (line 1644)
# Getting the type of 'bartlett' (line 1644)
bartlett_286559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1644, 34), 'bartlett')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1644, 34), tuple_286558, bartlett_286559)
# Adding element type (line 1644)
# Getting the type of 'False' (line 1644)
False_286560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1644, 44), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1644, 34), tuple_286558, False_286560)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286554, tuple_286558))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1645)
tuple_286561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1645)
# Adding element type (line 1645)
str_286562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 5), 'str', 'blackman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1645, 5), tuple_286561, str_286562)
# Adding element type (line 1645)
str_286563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 17), 'str', 'black')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1645, 5), tuple_286561, str_286563)
# Adding element type (line 1645)
str_286564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 26), 'str', 'blk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1645, 5), tuple_286561, str_286564)


# Obtaining an instance of the builtin type 'tuple' (line 1645)
tuple_286565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1645)
# Adding element type (line 1645)
# Getting the type of 'blackman' (line 1645)
blackman_286566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1645, 35), 'blackman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1645, 35), tuple_286565, blackman_286566)
# Adding element type (line 1645)
# Getting the type of 'False' (line 1645)
False_286567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1645, 45), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1645, 35), tuple_286565, False_286567)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286561, tuple_286565))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1646)
tuple_286568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1646)
# Adding element type (line 1646)
str_286569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 5), 'str', 'blackmanharris')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1646, 5), tuple_286568, str_286569)
# Adding element type (line 1646)
str_286570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 23), 'str', 'blackharr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1646, 5), tuple_286568, str_286570)
# Adding element type (line 1646)
str_286571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 36), 'str', 'bkh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1646, 5), tuple_286568, str_286571)


# Obtaining an instance of the builtin type 'tuple' (line 1646)
tuple_286572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 45), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1646)
# Adding element type (line 1646)
# Getting the type of 'blackmanharris' (line 1646)
blackmanharris_286573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1646, 45), 'blackmanharris')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1646, 45), tuple_286572, blackmanharris_286573)
# Adding element type (line 1646)
# Getting the type of 'False' (line 1646)
False_286574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1646, 61), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1646, 45), tuple_286572, False_286574)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286568, tuple_286572))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1647)
tuple_286575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1647, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1647)
# Adding element type (line 1647)
str_286576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1647, 5), 'str', 'bohman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1647, 5), tuple_286575, str_286576)
# Adding element type (line 1647)
str_286577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1647, 15), 'str', 'bman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1647, 5), tuple_286575, str_286577)
# Adding element type (line 1647)
str_286578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1647, 23), 'str', 'bmn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1647, 5), tuple_286575, str_286578)


# Obtaining an instance of the builtin type 'tuple' (line 1647)
tuple_286579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1647, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1647)
# Adding element type (line 1647)
# Getting the type of 'bohman' (line 1647)
bohman_286580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1647, 32), 'bohman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1647, 32), tuple_286579, bohman_286580)
# Adding element type (line 1647)
# Getting the type of 'False' (line 1647)
False_286581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1647, 40), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1647, 32), tuple_286579, False_286581)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286575, tuple_286579))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1648)
tuple_286582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1648)
# Adding element type (line 1648)
str_286583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 5), 'str', 'boxcar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1648, 5), tuple_286582, str_286583)
# Adding element type (line 1648)
str_286584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 15), 'str', 'box')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1648, 5), tuple_286582, str_286584)
# Adding element type (line 1648)
str_286585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 22), 'str', 'ones')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1648, 5), tuple_286582, str_286585)
# Adding element type (line 1648)
str_286586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1649, 8), 'str', 'rect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1648, 5), tuple_286582, str_286586)
# Adding element type (line 1648)
str_286587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1649, 16), 'str', 'rectangular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1648, 5), tuple_286582, str_286587)


# Obtaining an instance of the builtin type 'tuple' (line 1649)
tuple_286588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1649, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1649)
# Adding element type (line 1649)
# Getting the type of 'boxcar' (line 1649)
boxcar_286589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 33), 'boxcar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1649, 33), tuple_286588, boxcar_286589)
# Adding element type (line 1649)
# Getting the type of 'False' (line 1649)
False_286590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 41), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1649, 33), tuple_286588, False_286590)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286582, tuple_286588))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1650)
tuple_286591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1650, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1650)
# Adding element type (line 1650)
str_286592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1650, 5), 'str', 'chebwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1650, 5), tuple_286591, str_286592)
# Adding element type (line 1650)
str_286593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1650, 16), 'str', 'cheb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1650, 5), tuple_286591, str_286593)


# Obtaining an instance of the builtin type 'tuple' (line 1650)
tuple_286594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1650, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1650)
# Adding element type (line 1650)
# Getting the type of 'chebwin' (line 1650)
chebwin_286595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1650, 26), 'chebwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1650, 26), tuple_286594, chebwin_286595)
# Adding element type (line 1650)
# Getting the type of 'True' (line 1650)
True_286596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1650, 35), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1650, 26), tuple_286594, True_286596)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286591, tuple_286594))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1651)
tuple_286597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1651, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1651)
# Adding element type (line 1651)
str_286598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1651, 5), 'str', 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1651, 5), tuple_286597, str_286598)
# Adding element type (line 1651)
str_286599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1651, 15), 'str', 'halfcosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1651, 5), tuple_286597, str_286599)


# Obtaining an instance of the builtin type 'tuple' (line 1651)
tuple_286600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1651, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1651)
# Adding element type (line 1651)
# Getting the type of 'cosine' (line 1651)
cosine_286601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1651, 31), 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1651, 31), tuple_286600, cosine_286601)
# Adding element type (line 1651)
# Getting the type of 'False' (line 1651)
False_286602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1651, 39), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1651, 31), tuple_286600, False_286602)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286597, tuple_286600))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1652)
tuple_286603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1652, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1652)
# Adding element type (line 1652)
str_286604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1652, 5), 'str', 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1652, 5), tuple_286603, str_286604)
# Adding element type (line 1652)
str_286605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1652, 20), 'str', 'poisson')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1652, 5), tuple_286603, str_286605)


# Obtaining an instance of the builtin type 'tuple' (line 1652)
tuple_286606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1652, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1652)
# Adding element type (line 1652)
# Getting the type of 'exponential' (line 1652)
exponential_286607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1652, 33), 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1652, 33), tuple_286606, exponential_286607)
# Adding element type (line 1652)
# Getting the type of 'True' (line 1652)
True_286608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1652, 46), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1652, 33), tuple_286606, True_286608)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286603, tuple_286606))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1653)
tuple_286609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1653, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1653)
# Adding element type (line 1653)
str_286610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1653, 5), 'str', 'flattop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1653, 5), tuple_286609, str_286610)
# Adding element type (line 1653)
str_286611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1653, 16), 'str', 'flat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1653, 5), tuple_286609, str_286611)
# Adding element type (line 1653)
str_286612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1653, 24), 'str', 'flt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1653, 5), tuple_286609, str_286612)


# Obtaining an instance of the builtin type 'tuple' (line 1653)
tuple_286613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1653, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1653)
# Adding element type (line 1653)
# Getting the type of 'flattop' (line 1653)
flattop_286614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 33), 'flattop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1653, 33), tuple_286613, flattop_286614)
# Adding element type (line 1653)
# Getting the type of 'False' (line 1653)
False_286615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 42), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1653, 33), tuple_286613, False_286615)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286609, tuple_286613))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1654)
tuple_286616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1654, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1654)
# Adding element type (line 1654)
str_286617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1654, 5), 'str', 'gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1654, 5), tuple_286616, str_286617)
# Adding element type (line 1654)
str_286618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1654, 17), 'str', 'gauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1654, 5), tuple_286616, str_286618)
# Adding element type (line 1654)
str_286619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1654, 26), 'str', 'gss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1654, 5), tuple_286616, str_286619)


# Obtaining an instance of the builtin type 'tuple' (line 1654)
tuple_286620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1654, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1654)
# Adding element type (line 1654)
# Getting the type of 'gaussian' (line 1654)
gaussian_286621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1654, 35), 'gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1654, 35), tuple_286620, gaussian_286621)
# Adding element type (line 1654)
# Getting the type of 'True' (line 1654)
True_286622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1654, 45), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1654, 35), tuple_286620, True_286622)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286616, tuple_286620))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1655)
tuple_286623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1655, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1655)
# Adding element type (line 1655)
str_286624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1655, 5), 'str', 'general gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1655, 5), tuple_286623, str_286624)
# Adding element type (line 1655)
str_286625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1655, 25), 'str', 'general_gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1655, 5), tuple_286623, str_286625)
# Adding element type (line 1655)
str_286626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1656, 8), 'str', 'general gauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1655, 5), tuple_286623, str_286626)
# Adding element type (line 1655)
str_286627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1656, 25), 'str', 'general_gauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1655, 5), tuple_286623, str_286627)
# Adding element type (line 1655)
str_286628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1656, 42), 'str', 'ggs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1655, 5), tuple_286623, str_286628)


# Obtaining an instance of the builtin type 'tuple' (line 1656)
tuple_286629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1656, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1656)
# Adding element type (line 1656)
# Getting the type of 'general_gaussian' (line 1656)
general_gaussian_286630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1656, 51), 'general_gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1656, 51), tuple_286629, general_gaussian_286630)
# Adding element type (line 1656)
# Getting the type of 'True' (line 1656)
True_286631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1656, 69), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1656, 51), tuple_286629, True_286631)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286623, tuple_286629))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1657)
tuple_286632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1657)
# Adding element type (line 1657)
str_286633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, 5), 'str', 'hamming')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1657, 5), tuple_286632, str_286633)
# Adding element type (line 1657)
str_286634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, 16), 'str', 'hamm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1657, 5), tuple_286632, str_286634)
# Adding element type (line 1657)
str_286635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, 24), 'str', 'ham')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1657, 5), tuple_286632, str_286635)


# Obtaining an instance of the builtin type 'tuple' (line 1657)
tuple_286636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1657)
# Adding element type (line 1657)
# Getting the type of 'hamming' (line 1657)
hamming_286637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1657, 33), 'hamming')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1657, 33), tuple_286636, hamming_286637)
# Adding element type (line 1657)
# Getting the type of 'False' (line 1657)
False_286638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1657, 42), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1657, 33), tuple_286636, False_286638)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286632, tuple_286636))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1658)
tuple_286639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1658, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1658)
# Adding element type (line 1658)
str_286640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1658, 5), 'str', 'hanning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1658, 5), tuple_286639, str_286640)
# Adding element type (line 1658)
str_286641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1658, 16), 'str', 'hann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1658, 5), tuple_286639, str_286641)
# Adding element type (line 1658)
str_286642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1658, 24), 'str', 'han')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1658, 5), tuple_286639, str_286642)


# Obtaining an instance of the builtin type 'tuple' (line 1658)
tuple_286643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1658, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1658)
# Adding element type (line 1658)
# Getting the type of 'hann' (line 1658)
hann_286644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1658, 33), 'hann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1658, 33), tuple_286643, hann_286644)
# Adding element type (line 1658)
# Getting the type of 'False' (line 1658)
False_286645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1658, 39), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1658, 33), tuple_286643, False_286645)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286639, tuple_286643))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1659)
tuple_286646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1659)
# Adding element type (line 1659)
str_286647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 5), 'str', 'kaiser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 5), tuple_286646, str_286647)
# Adding element type (line 1659)
str_286648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 15), 'str', 'ksr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 5), tuple_286646, str_286648)


# Obtaining an instance of the builtin type 'tuple' (line 1659)
tuple_286649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1659)
# Adding element type (line 1659)
# Getting the type of 'kaiser' (line 1659)
kaiser_286650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 24), 'kaiser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 24), tuple_286649, kaiser_286650)
# Adding element type (line 1659)
# Getting the type of 'True' (line 1659)
True_286651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 32), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 24), tuple_286649, True_286651)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286646, tuple_286649))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1660)
tuple_286652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1660)
# Adding element type (line 1660)
str_286653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 5), 'str', 'nuttall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1660, 5), tuple_286652, str_286653)
# Adding element type (line 1660)
str_286654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 16), 'str', 'nutl')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1660, 5), tuple_286652, str_286654)
# Adding element type (line 1660)
str_286655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 24), 'str', 'nut')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1660, 5), tuple_286652, str_286655)


# Obtaining an instance of the builtin type 'tuple' (line 1660)
tuple_286656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1660)
# Adding element type (line 1660)
# Getting the type of 'nuttall' (line 1660)
nuttall_286657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 33), 'nuttall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1660, 33), tuple_286656, nuttall_286657)
# Adding element type (line 1660)
# Getting the type of 'False' (line 1660)
False_286658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 42), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1660, 33), tuple_286656, False_286658)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286652, tuple_286656))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1661)
tuple_286659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1661)
# Adding element type (line 1661)
str_286660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 5), 'str', 'parzen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 5), tuple_286659, str_286660)
# Adding element type (line 1661)
str_286661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 15), 'str', 'parz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 5), tuple_286659, str_286661)
# Adding element type (line 1661)
str_286662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 23), 'str', 'par')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 5), tuple_286659, str_286662)


# Obtaining an instance of the builtin type 'tuple' (line 1661)
tuple_286663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1661)
# Adding element type (line 1661)
# Getting the type of 'parzen' (line 1661)
parzen_286664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 32), 'parzen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 32), tuple_286663, parzen_286664)
# Adding element type (line 1661)
# Getting the type of 'False' (line 1661)
False_286665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 40), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 32), tuple_286663, False_286665)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286659, tuple_286663))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1662)
tuple_286666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1662)
# Adding element type (line 1662)
str_286667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 5), 'str', 'slepian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 5), tuple_286666, str_286667)
# Adding element type (line 1662)
str_286668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 16), 'str', 'slep')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 5), tuple_286666, str_286668)
# Adding element type (line 1662)
str_286669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 24), 'str', 'optimal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 5), tuple_286666, str_286669)
# Adding element type (line 1662)
str_286670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 35), 'str', 'dpss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 5), tuple_286666, str_286670)
# Adding element type (line 1662)
str_286671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 43), 'str', 'dss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 5), tuple_286666, str_286671)


# Obtaining an instance of the builtin type 'tuple' (line 1662)
tuple_286672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 52), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1662)
# Adding element type (line 1662)
# Getting the type of 'slepian' (line 1662)
slepian_286673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 52), 'slepian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 52), tuple_286672, slepian_286673)
# Adding element type (line 1662)
# Getting the type of 'True' (line 1662)
True_286674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 61), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 52), tuple_286672, True_286674)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286666, tuple_286672))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1663)
tuple_286675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1663)
# Adding element type (line 1663)
str_286676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 5), 'str', 'triangle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 5), tuple_286675, str_286676)
# Adding element type (line 1663)
str_286677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 17), 'str', 'triang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 5), tuple_286675, str_286677)
# Adding element type (line 1663)
str_286678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 27), 'str', 'tri')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 5), tuple_286675, str_286678)


# Obtaining an instance of the builtin type 'tuple' (line 1663)
tuple_286679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1663)
# Adding element type (line 1663)
# Getting the type of 'triang' (line 1663)
triang_286680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 36), 'triang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 36), tuple_286679, triang_286680)
# Adding element type (line 1663)
# Getting the type of 'False' (line 1663)
False_286681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 44), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 36), tuple_286679, False_286681)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286675, tuple_286679))
# Adding element type (key, value) (line 1642)

# Obtaining an instance of the builtin type 'tuple' (line 1664)
tuple_286682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1664, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1664)
# Adding element type (line 1664)
str_286683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1664, 5), 'str', 'tukey')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1664, 5), tuple_286682, str_286683)
# Adding element type (line 1664)
str_286684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1664, 14), 'str', 'tuk')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1664, 5), tuple_286682, str_286684)


# Obtaining an instance of the builtin type 'tuple' (line 1664)
tuple_286685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1664, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1664)
# Adding element type (line 1664)
# Getting the type of 'tukey' (line 1664)
tukey_286686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 23), 'tukey')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1664, 23), tuple_286685, tukey_286686)
# Adding element type (line 1664)
# Getting the type of 'True' (line 1664)
True_286687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 30), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1664, 23), tuple_286685, True_286687)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1642, 17), dict_286546, (tuple_286682, tuple_286685))

# Assigning a type to the variable '_win_equiv_raw' (line 1642)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1642, 0), '_win_equiv_raw', dict_286546)

# Assigning a Dict to a Name (line 1668):

# Assigning a Dict to a Name (line 1668):

# Obtaining an instance of the builtin type 'dict' (line 1668)
dict_286688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1668, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1668)

# Assigning a type to the variable '_win_equiv' (line 1668)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1668, 0), '_win_equiv', dict_286688)


# Call to items(...): (line 1669)
# Processing the call keyword arguments (line 1669)
kwargs_286691 = {}
# Getting the type of '_win_equiv_raw' (line 1669)
_win_equiv_raw_286689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 12), '_win_equiv_raw', False)
# Obtaining the member 'items' of a type (line 1669)
items_286690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1669, 12), _win_equiv_raw_286689, 'items')
# Calling items(args, kwargs) (line 1669)
items_call_result_286692 = invoke(stypy.reporting.localization.Localization(__file__, 1669, 12), items_286690, *[], **kwargs_286691)

# Testing the type of a for loop iterable (line 1669)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1669, 0), items_call_result_286692)
# Getting the type of the for loop variable (line 1669)
for_loop_var_286693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1669, 0), items_call_result_286692)
# Assigning a type to the variable 'k' (line 1669)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1669, 0), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1669, 0), for_loop_var_286693))
# Assigning a type to the variable 'v' (line 1669)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1669, 0), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1669, 0), for_loop_var_286693))
# SSA begins for a for statement (line 1669)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'k' (line 1670)
k_286694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1670, 15), 'k')
# Testing the type of a for loop iterable (line 1670)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1670, 4), k_286694)
# Getting the type of the for loop variable (line 1670)
for_loop_var_286695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1670, 4), k_286694)
# Assigning a type to the variable 'key' (line 1670)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1670, 4), 'key', for_loop_var_286695)
# SSA begins for a for statement (line 1670)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Subscript to a Subscript (line 1671):

# Assigning a Subscript to a Subscript (line 1671):

# Obtaining the type of the subscript
int_286696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1671, 28), 'int')
# Getting the type of 'v' (line 1671)
v_286697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1671, 26), 'v')
# Obtaining the member '__getitem__' of a type (line 1671)
getitem___286698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1671, 26), v_286697, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1671)
subscript_call_result_286699 = invoke(stypy.reporting.localization.Localization(__file__, 1671, 26), getitem___286698, int_286696)

# Getting the type of '_win_equiv' (line 1671)
_win_equiv_286700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1671, 8), '_win_equiv')
# Getting the type of 'key' (line 1671)
key_286701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1671, 19), 'key')
# Storing an element on a container (line 1671)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1671, 8), _win_equiv_286700, (key_286701, subscript_call_result_286699))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 1674):

# Assigning a Call to a Name (line 1674):

# Call to set(...): (line 1674)
# Processing the call keyword arguments (line 1674)
kwargs_286703 = {}
# Getting the type of 'set' (line 1674)
set_286702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1674, 15), 'set', False)
# Calling set(args, kwargs) (line 1674)
set_call_result_286704 = invoke(stypy.reporting.localization.Localization(__file__, 1674, 15), set_286702, *[], **kwargs_286703)

# Assigning a type to the variable '_needs_param' (line 1674)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1674, 0), '_needs_param', set_call_result_286704)


# Call to items(...): (line 1675)
# Processing the call keyword arguments (line 1675)
kwargs_286707 = {}
# Getting the type of '_win_equiv_raw' (line 1675)
_win_equiv_raw_286705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1675, 12), '_win_equiv_raw', False)
# Obtaining the member 'items' of a type (line 1675)
items_286706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1675, 12), _win_equiv_raw_286705, 'items')
# Calling items(args, kwargs) (line 1675)
items_call_result_286708 = invoke(stypy.reporting.localization.Localization(__file__, 1675, 12), items_286706, *[], **kwargs_286707)

# Testing the type of a for loop iterable (line 1675)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1675, 0), items_call_result_286708)
# Getting the type of the for loop variable (line 1675)
for_loop_var_286709 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1675, 0), items_call_result_286708)
# Assigning a type to the variable 'k' (line 1675)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1675, 0), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1675, 0), for_loop_var_286709))
# Assigning a type to the variable 'v' (line 1675)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1675, 0), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1675, 0), for_loop_var_286709))
# SSA begins for a for statement (line 1675)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# Obtaining the type of the subscript
int_286710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1676, 9), 'int')
# Getting the type of 'v' (line 1676)
v_286711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1676, 7), 'v')
# Obtaining the member '__getitem__' of a type (line 1676)
getitem___286712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1676, 7), v_286711, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 1676)
subscript_call_result_286713 = invoke(stypy.reporting.localization.Localization(__file__, 1676, 7), getitem___286712, int_286710)

# Testing the type of an if condition (line 1676)
if_condition_286714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1676, 4), subscript_call_result_286713)
# Assigning a type to the variable 'if_condition_286714' (line 1676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1676, 4), 'if_condition_286714', if_condition_286714)
# SSA begins for if statement (line 1676)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to update(...): (line 1677)
# Processing the call arguments (line 1677)
# Getting the type of 'k' (line 1677)
k_286717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1677, 28), 'k', False)
# Processing the call keyword arguments (line 1677)
kwargs_286718 = {}
# Getting the type of '_needs_param' (line 1677)
_needs_param_286715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1677, 8), '_needs_param', False)
# Obtaining the member 'update' of a type (line 1677)
update_286716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1677, 8), _needs_param_286715, 'update')
# Calling update(args, kwargs) (line 1677)
update_call_result_286719 = invoke(stypy.reporting.localization.Localization(__file__, 1677, 8), update_286716, *[k_286717], **kwargs_286718)

# SSA join for if statement (line 1676)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


@norecursion
def get_window(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1680)
    True_286720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1680, 35), 'True')
    defaults = [True_286720]
    # Create a new context for function 'get_window'
    module_type_store = module_type_store.open_function_context('get_window', 1680, 0, False)
    
    # Passed parameters checking function
    get_window.stypy_localization = localization
    get_window.stypy_type_of_self = None
    get_window.stypy_type_store = module_type_store
    get_window.stypy_function_name = 'get_window'
    get_window.stypy_param_names_list = ['window', 'Nx', 'fftbins']
    get_window.stypy_varargs_param_name = None
    get_window.stypy_kwargs_param_name = None
    get_window.stypy_call_defaults = defaults
    get_window.stypy_call_varargs = varargs
    get_window.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_window', ['window', 'Nx', 'fftbins'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_window', localization, ['window', 'Nx', 'fftbins'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_window(...)' code ##################

    str_286721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1737, (-1)), 'str', '\n    Return a window.\n\n    Parameters\n    ----------\n    window : string, float, or tuple\n        The type of window to create. See below for more details.\n    Nx : int\n        The number of samples in the window.\n    fftbins : bool, optional\n        If True (default), create a "periodic" window, ready to use with\n        `ifftshift` and be multiplied by the result of an FFT (see also\n        `fftpack.fftfreq`).\n        If False, create a "symmetric" window, for use in filter design.\n\n    Returns\n    -------\n    get_window : ndarray\n        Returns a window of length `Nx` and type `window`\n\n    Notes\n    -----\n    Window types:\n\n        `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,\n        `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,\n        `barthann`, `kaiser` (needs beta), `gaussian` (needs standard\n        deviation), `general_gaussian` (needs power, width), `slepian`\n        (needs width), `chebwin` (needs attenuation), `exponential`\n        (needs decay scale), `tukey` (needs taper fraction)\n\n    If the window requires no parameters, then `window` can be a string.\n\n    If the window requires parameters, then `window` must be a tuple\n    with the first argument the string name of the window, and the next\n    arguments the needed parameters.\n\n    If `window` is a floating point number, it is interpreted as the beta\n    parameter of the `kaiser` window.\n\n    Each of the window types listed above is also the name of\n    a function that can be called directly to create a window of\n    that type.\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> signal.get_window(\'triang\', 7)\n    array([ 0.125,  0.375,  0.625,  0.875,  0.875,  0.625,  0.375])\n    >>> signal.get_window((\'kaiser\', 4.0), 9)\n    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,\n            0.97885093,  0.82160913,  0.56437221,  0.29425961])\n    >>> signal.get_window(4.0, 9)\n    array([ 0.08848053,  0.29425961,  0.56437221,  0.82160913,  0.97885093,\n            0.97885093,  0.82160913,  0.56437221,  0.29425961])\n\n    ')
    
    # Assigning a UnaryOp to a Name (line 1738):
    
    # Assigning a UnaryOp to a Name (line 1738):
    
    # Getting the type of 'fftbins' (line 1738)
    fftbins_286722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 14), 'fftbins')
    # Applying the 'not' unary operator (line 1738)
    result_not__286723 = python_operator(stypy.reporting.localization.Localization(__file__, 1738, 10), 'not', fftbins_286722)
    
    # Assigning a type to the variable 'sym' (line 1738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1738, 4), 'sym', result_not__286723)
    
    
    # SSA begins for try-except statement (line 1739)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 1740):
    
    # Assigning a Call to a Name (line 1740):
    
    # Call to float(...): (line 1740)
    # Processing the call arguments (line 1740)
    # Getting the type of 'window' (line 1740)
    window_286725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 21), 'window', False)
    # Processing the call keyword arguments (line 1740)
    kwargs_286726 = {}
    # Getting the type of 'float' (line 1740)
    float_286724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 15), 'float', False)
    # Calling float(args, kwargs) (line 1740)
    float_call_result_286727 = invoke(stypy.reporting.localization.Localization(__file__, 1740, 15), float_286724, *[window_286725], **kwargs_286726)
    
    # Assigning a type to the variable 'beta' (line 1740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1740, 8), 'beta', float_call_result_286727)
    # SSA branch for the except part of a try statement (line 1739)
    # SSA branch for the except 'Tuple' branch of a try statement (line 1739)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Tuple to a Name (line 1742):
    
    # Assigning a Tuple to a Name (line 1742):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1742)
    tuple_286728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1742, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1742)
    
    # Assigning a type to the variable 'args' (line 1742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1742, 8), 'args', tuple_286728)
    
    # Type idiom detected: calculating its left and rigth part (line 1743)
    # Getting the type of 'tuple' (line 1743)
    tuple_286729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 30), 'tuple')
    # Getting the type of 'window' (line 1743)
    window_286730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1743, 22), 'window')
    
    (may_be_286731, more_types_in_union_286732) = may_be_subtype(tuple_286729, window_286730)

    if may_be_286731:

        if more_types_in_union_286732:
            # Runtime conditional SSA (line 1743)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'window' (line 1743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1743, 8), 'window', remove_not_subtype_from_union(window_286730, tuple))
        
        # Assigning a Subscript to a Name (line 1744):
        
        # Assigning a Subscript to a Name (line 1744):
        
        # Obtaining the type of the subscript
        int_286733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1744, 28), 'int')
        # Getting the type of 'window' (line 1744)
        window_286734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1744, 21), 'window')
        # Obtaining the member '__getitem__' of a type (line 1744)
        getitem___286735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1744, 21), window_286734, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1744)
        subscript_call_result_286736 = invoke(stypy.reporting.localization.Localization(__file__, 1744, 21), getitem___286735, int_286733)
        
        # Assigning a type to the variable 'winstr' (line 1744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1744, 12), 'winstr', subscript_call_result_286736)
        
        
        
        # Call to len(...): (line 1745)
        # Processing the call arguments (line 1745)
        # Getting the type of 'window' (line 1745)
        window_286738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 19), 'window', False)
        # Processing the call keyword arguments (line 1745)
        kwargs_286739 = {}
        # Getting the type of 'len' (line 1745)
        len_286737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1745, 15), 'len', False)
        # Calling len(args, kwargs) (line 1745)
        len_call_result_286740 = invoke(stypy.reporting.localization.Localization(__file__, 1745, 15), len_286737, *[window_286738], **kwargs_286739)
        
        int_286741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1745, 29), 'int')
        # Applying the binary operator '>' (line 1745)
        result_gt_286742 = python_operator(stypy.reporting.localization.Localization(__file__, 1745, 15), '>', len_call_result_286740, int_286741)
        
        # Testing the type of an if condition (line 1745)
        if_condition_286743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1745, 12), result_gt_286742)
        # Assigning a type to the variable 'if_condition_286743' (line 1745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1745, 12), 'if_condition_286743', if_condition_286743)
        # SSA begins for if statement (line 1745)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 1746):
        
        # Assigning a Subscript to a Name (line 1746):
        
        # Obtaining the type of the subscript
        int_286744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1746, 30), 'int')
        slice_286745 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1746, 23), int_286744, None, None)
        # Getting the type of 'window' (line 1746)
        window_286746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1746, 23), 'window')
        # Obtaining the member '__getitem__' of a type (line 1746)
        getitem___286747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1746, 23), window_286746, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1746)
        subscript_call_result_286748 = invoke(stypy.reporting.localization.Localization(__file__, 1746, 23), getitem___286747, slice_286745)
        
        # Assigning a type to the variable 'args' (line 1746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1746, 16), 'args', subscript_call_result_286748)
        # SSA join for if statement (line 1745)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_286732:
            # Runtime conditional SSA for else branch (line 1743)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_286731) or more_types_in_union_286732):
        # Assigning a type to the variable 'window' (line 1743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1743, 8), 'window', remove_subtype_from_union(window_286730, tuple))
        
        
        # Call to isinstance(...): (line 1747)
        # Processing the call arguments (line 1747)
        # Getting the type of 'window' (line 1747)
        window_286750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1747, 24), 'window', False)
        # Getting the type of 'string_types' (line 1747)
        string_types_286751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1747, 32), 'string_types', False)
        # Processing the call keyword arguments (line 1747)
        kwargs_286752 = {}
        # Getting the type of 'isinstance' (line 1747)
        isinstance_286749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1747, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1747)
        isinstance_call_result_286753 = invoke(stypy.reporting.localization.Localization(__file__, 1747, 13), isinstance_286749, *[window_286750, string_types_286751], **kwargs_286752)
        
        # Testing the type of an if condition (line 1747)
        if_condition_286754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1747, 13), isinstance_call_result_286753)
        # Assigning a type to the variable 'if_condition_286754' (line 1747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1747, 13), 'if_condition_286754', if_condition_286754)
        # SSA begins for if statement (line 1747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'window' (line 1748)
        window_286755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 15), 'window')
        # Getting the type of '_needs_param' (line 1748)
        _needs_param_286756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1748, 25), '_needs_param')
        # Applying the binary operator 'in' (line 1748)
        result_contains_286757 = python_operator(stypy.reporting.localization.Localization(__file__, 1748, 15), 'in', window_286755, _needs_param_286756)
        
        # Testing the type of an if condition (line 1748)
        if_condition_286758 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1748, 12), result_contains_286757)
        # Assigning a type to the variable 'if_condition_286758' (line 1748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1748, 12), 'if_condition_286758', if_condition_286758)
        # SSA begins for if statement (line 1748)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1749)
        # Processing the call arguments (line 1749)
        str_286760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1749, 33), 'str', "The '")
        # Getting the type of 'window' (line 1749)
        window_286761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 43), 'window', False)
        # Applying the binary operator '+' (line 1749)
        result_add_286762 = python_operator(stypy.reporting.localization.Localization(__file__, 1749, 33), '+', str_286760, window_286761)
        
        str_286763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1749, 52), 'str', "' window needs one or more parameters -- pass a tuple.")
        # Applying the binary operator '+' (line 1749)
        result_add_286764 = python_operator(stypy.reporting.localization.Localization(__file__, 1749, 50), '+', result_add_286762, str_286763)
        
        # Processing the call keyword arguments (line 1749)
        kwargs_286765 = {}
        # Getting the type of 'ValueError' (line 1749)
        ValueError_286759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1749, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1749)
        ValueError_call_result_286766 = invoke(stypy.reporting.localization.Localization(__file__, 1749, 22), ValueError_286759, *[result_add_286764], **kwargs_286765)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1749, 16), ValueError_call_result_286766, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 1748)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1752):
        
        # Assigning a Name to a Name (line 1752):
        # Getting the type of 'window' (line 1752)
        window_286767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 25), 'window')
        # Assigning a type to the variable 'winstr' (line 1752)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1752, 16), 'winstr', window_286767)
        # SSA join for if statement (line 1748)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 1747)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 1754)
        # Processing the call arguments (line 1754)
        str_286769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1754, 29), 'str', '%s as window type is not supported.')
        
        # Call to str(...): (line 1755)
        # Processing the call arguments (line 1755)
        
        # Call to type(...): (line 1755)
        # Processing the call arguments (line 1755)
        # Getting the type of 'window' (line 1755)
        window_286772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 38), 'window', False)
        # Processing the call keyword arguments (line 1755)
        kwargs_286773 = {}
        # Getting the type of 'type' (line 1755)
        type_286771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 33), 'type', False)
        # Calling type(args, kwargs) (line 1755)
        type_call_result_286774 = invoke(stypy.reporting.localization.Localization(__file__, 1755, 33), type_286771, *[window_286772], **kwargs_286773)
        
        # Processing the call keyword arguments (line 1755)
        kwargs_286775 = {}
        # Getting the type of 'str' (line 1755)
        str_286770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1755, 29), 'str', False)
        # Calling str(args, kwargs) (line 1755)
        str_call_result_286776 = invoke(stypy.reporting.localization.Localization(__file__, 1755, 29), str_286770, *[type_call_result_286774], **kwargs_286775)
        
        # Applying the binary operator '%' (line 1754)
        result_mod_286777 = python_operator(stypy.reporting.localization.Localization(__file__, 1754, 29), '%', str_286769, str_call_result_286776)
        
        # Processing the call keyword arguments (line 1754)
        kwargs_286778 = {}
        # Getting the type of 'ValueError' (line 1754)
        ValueError_286768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1754, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1754)
        ValueError_call_result_286779 = invoke(stypy.reporting.localization.Localization(__file__, 1754, 18), ValueError_286768, *[result_mod_286777], **kwargs_286778)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1754, 12), ValueError_call_result_286779, 'raise parameter', BaseException)
        # SSA join for if statement (line 1747)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_286731 and more_types_in_union_286732):
            # SSA join for if statement (line 1743)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 1757)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 1758):
    
    # Assigning a Subscript to a Name (line 1758):
    
    # Obtaining the type of the subscript
    # Getting the type of 'winstr' (line 1758)
    winstr_286780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 33), 'winstr')
    # Getting the type of '_win_equiv' (line 1758)
    _win_equiv_286781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 22), '_win_equiv')
    # Obtaining the member '__getitem__' of a type (line 1758)
    getitem___286782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1758, 22), _win_equiv_286781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1758)
    subscript_call_result_286783 = invoke(stypy.reporting.localization.Localization(__file__, 1758, 22), getitem___286782, winstr_286780)
    
    # Assigning a type to the variable 'winfunc' (line 1758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1758, 12), 'winfunc', subscript_call_result_286783)
    # SSA branch for the except part of a try statement (line 1757)
    # SSA branch for the except 'KeyError' branch of a try statement (line 1757)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1760)
    # Processing the call arguments (line 1760)
    str_286785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1760, 29), 'str', 'Unknown window type.')
    # Processing the call keyword arguments (line 1760)
    kwargs_286786 = {}
    # Getting the type of 'ValueError' (line 1760)
    ValueError_286784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1760)
    ValueError_call_result_286787 = invoke(stypy.reporting.localization.Localization(__file__, 1760, 18), ValueError_286784, *[str_286785], **kwargs_286786)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1760, 12), ValueError_call_result_286787, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1757)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1762):
    
    # Assigning a BinOp to a Name (line 1762):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1762)
    tuple_286788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1762, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1762)
    # Adding element type (line 1762)
    # Getting the type of 'Nx' (line 1762)
    Nx_286789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 18), 'Nx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1762, 18), tuple_286788, Nx_286789)
    
    # Getting the type of 'args' (line 1762)
    args_286790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 25), 'args')
    # Applying the binary operator '+' (line 1762)
    result_add_286791 = python_operator(stypy.reporting.localization.Localization(__file__, 1762, 17), '+', tuple_286788, args_286790)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1762)
    tuple_286792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1762, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1762)
    # Adding element type (line 1762)
    # Getting the type of 'sym' (line 1762)
    sym_286793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 33), 'sym')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1762, 33), tuple_286792, sym_286793)
    
    # Applying the binary operator '+' (line 1762)
    result_add_286794 = python_operator(stypy.reporting.localization.Localization(__file__, 1762, 30), '+', result_add_286791, tuple_286792)
    
    # Assigning a type to the variable 'params' (line 1762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1762, 8), 'params', result_add_286794)
    # SSA branch for the else branch of a try statement (line 1739)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Name to a Name (line 1764):
    
    # Assigning a Name to a Name (line 1764):
    # Getting the type of 'kaiser' (line 1764)
    kaiser_286795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 18), 'kaiser')
    # Assigning a type to the variable 'winfunc' (line 1764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1764, 8), 'winfunc', kaiser_286795)
    
    # Assigning a Tuple to a Name (line 1765):
    
    # Assigning a Tuple to a Name (line 1765):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1765)
    tuple_286796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1765, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1765)
    # Adding element type (line 1765)
    # Getting the type of 'Nx' (line 1765)
    Nx_286797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 18), 'Nx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1765, 18), tuple_286796, Nx_286797)
    # Adding element type (line 1765)
    # Getting the type of 'beta' (line 1765)
    beta_286798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 22), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1765, 18), tuple_286796, beta_286798)
    # Adding element type (line 1765)
    # Getting the type of 'sym' (line 1765)
    sym_286799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 28), 'sym')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1765, 18), tuple_286796, sym_286799)
    
    # Assigning a type to the variable 'params' (line 1765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1765, 8), 'params', tuple_286796)
    # SSA join for try-except statement (line 1739)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to winfunc(...): (line 1767)
    # Getting the type of 'params' (line 1767)
    params_286801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 20), 'params', False)
    # Processing the call keyword arguments (line 1767)
    kwargs_286802 = {}
    # Getting the type of 'winfunc' (line 1767)
    winfunc_286800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 11), 'winfunc', False)
    # Calling winfunc(args, kwargs) (line 1767)
    winfunc_call_result_286803 = invoke(stypy.reporting.localization.Localization(__file__, 1767, 11), winfunc_286800, *[params_286801], **kwargs_286802)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1767, 4), 'stypy_return_type', winfunc_call_result_286803)
    
    # ################# End of 'get_window(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_window' in the type store
    # Getting the type of 'stypy_return_type' (line 1680)
    stypy_return_type_286804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1680, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_286804)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_window'
    return stypy_return_type_286804

# Assigning a type to the variable 'get_window' (line 1680)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1680, 0), 'get_window', get_window)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
