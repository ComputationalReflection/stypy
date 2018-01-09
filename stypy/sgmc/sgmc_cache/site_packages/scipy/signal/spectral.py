
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tools for spectral analysis.
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from scipy import fftpack
8: from . import signaltools
9: from .windows import get_window
10: from ._spectral import _lombscargle
11: from ._arraytools import const_ext, even_ext, odd_ext, zero_ext
12: import warnings
13: 
14: from scipy._lib.six import string_types
15: 
16: __all__ = ['periodogram', 'welch', 'lombscargle', 'csd', 'coherence',
17:            'spectrogram', 'stft', 'istft', 'check_COLA']
18: 
19: 
20: def lombscargle(x,
21:                 y,
22:                 freqs,
23:                 precenter=False,
24:                 normalize=False):
25:     '''
26:     lombscargle(x, y, freqs)
27: 
28:     Computes the Lomb-Scargle periodogram.
29:     
30:     The Lomb-Scargle periodogram was developed by Lomb [1]_ and further
31:     extended by Scargle [2]_ to find, and test the significance of weak
32:     periodic signals with uneven temporal sampling.
33: 
34:     When *normalize* is False (default) the computed periodogram
35:     is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic
36:     signal with amplitude A for sufficiently large N.
37: 
38:     When *normalize* is True the computed periodogram is is normalized by
39:     the residuals of the data around a constant reference model (at zero).
40: 
41:     Input arrays should be one-dimensional and will be cast to float64.
42: 
43:     Parameters
44:     ----------
45:     x : array_like
46:         Sample times.
47:     y : array_like
48:         Measurement values.
49:     freqs : array_like
50:         Angular frequencies for output periodogram.
51:     precenter : bool, optional
52:         Pre-center amplitudes by subtracting the mean.
53:     normalize : bool, optional
54:         Compute normalized periodogram.
55: 
56:     Returns
57:     -------
58:     pgram : array_like
59:         Lomb-Scargle periodogram.
60: 
61:     Raises
62:     ------
63:     ValueError
64:         If the input arrays `x` and `y` do not have the same shape.
65: 
66:     Notes
67:     -----
68:     This subroutine calculates the periodogram using a slightly
69:     modified algorithm due to Townsend [3]_ which allows the
70:     periodogram to be calculated using only a single pass through
71:     the input arrays for each frequency.
72: 
73:     The algorithm running time scales roughly as O(x * freqs) or O(N^2)
74:     for a large number of samples and frequencies.
75: 
76:     References
77:     ----------
78:     .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced
79:            data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976
80: 
81:     .. [2] J.D. Scargle "Studies in astronomical time series analysis. II - 
82:            Statistical aspects of spectral analysis of unevenly spaced data",
83:            The Astrophysical Journal, vol 263, pp. 835-853, 1982
84: 
85:     .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle
86:            periodogram using graphics processing units.", The Astrophysical
87:            Journal Supplement Series, vol 191, pp. 247-253, 2010
88: 
89:     Examples
90:     --------
91:     >>> import scipy.signal
92:     >>> import matplotlib.pyplot as plt
93: 
94:     First define some input parameters for the signal:
95: 
96:     >>> A = 2.
97:     >>> w = 1.
98:     >>> phi = 0.5 * np.pi
99:     >>> nin = 1000
100:     >>> nout = 100000
101:     >>> frac_points = 0.9 # Fraction of points to select
102:      
103:     Randomly select a fraction of an array with timesteps:
104: 
105:     >>> r = np.random.rand(nin)
106:     >>> x = np.linspace(0.01, 10*np.pi, nin)
107:     >>> x = x[r >= frac_points]
108:      
109:     Plot a sine wave for the selected times:
110: 
111:     >>> y = A * np.sin(w*x+phi)
112: 
113:     Define the array of frequencies for which to compute the periodogram:
114:     
115:     >>> f = np.linspace(0.01, 10, nout)
116:      
117:     Calculate Lomb-Scargle periodogram:
118: 
119:     >>> import scipy.signal as signal
120:     >>> pgram = signal.lombscargle(x, y, f, normalize=True)
121: 
122:     Now make a plot of the input data:
123: 
124:     >>> plt.subplot(2, 1, 1)
125:     >>> plt.plot(x, y, 'b+')
126: 
127:     Then plot the normalized periodogram:
128: 
129:     >>> plt.subplot(2, 1, 2)
130:     >>> plt.plot(f, pgram)
131:     >>> plt.show()
132: 
133:     '''
134: 
135:     x = np.asarray(x, dtype=np.float64)
136:     y = np.asarray(y, dtype=np.float64)
137:     freqs = np.asarray(freqs, dtype=np.float64)
138: 
139:     assert x.ndim == 1
140:     assert y.ndim == 1
141:     assert freqs.ndim == 1
142: 
143:     if precenter:
144:         pgram = _lombscargle(x, y - y.mean(), freqs)
145:     else:
146:         pgram = _lombscargle(x, y, freqs)
147: 
148:     if normalize:
149:         pgram *= 2 / np.dot(y, y)
150: 
151:     return pgram
152: 
153: 
154: def periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant',
155:                 return_onesided=True, scaling='density', axis=-1):
156:     '''
157:     Estimate power spectral density using a periodogram.
158: 
159:     Parameters
160:     ----------
161:     x : array_like
162:         Time series of measurement values
163:     fs : float, optional
164:         Sampling frequency of the `x` time series. Defaults to 1.0.
165:     window : str or tuple or array_like, optional
166:         Desired window to use. If `window` is a string or tuple, it is
167:         passed to `get_window` to generate the window values, which are
168:         DFT-even by default. See `get_window` for a list of windows and
169:         required parameters. If `window` is array_like it will be used
170:         directly as the window and its length must be nperseg. Defaults
171:         to 'boxcar'.
172:     nfft : int, optional
173:         Length of the FFT used. If `None` the length of `x` will be
174:         used.
175:     detrend : str or function or `False`, optional
176:         Specifies how to detrend each segment. If `detrend` is a
177:         string, it is passed as the `type` argument to the `detrend`
178:         function. If it is a function, it takes a segment and returns a
179:         detrended segment. If `detrend` is `False`, no detrending is
180:         done. Defaults to 'constant'.
181:     return_onesided : bool, optional
182:         If `True`, return a one-sided spectrum for real data. If
183:         `False` return a two-sided spectrum. Note that for complex
184:         data, a two-sided spectrum is always returned.
185:     scaling : { 'density', 'spectrum' }, optional
186:         Selects between computing the power spectral density ('density')
187:         where `Pxx` has units of V**2/Hz and computing the power
188:         spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
189:         is measured in V and `fs` is measured in Hz. Defaults to
190:         'density'
191:     axis : int, optional
192:         Axis along which the periodogram is computed; the default is
193:         over the last axis (i.e. ``axis=-1``).
194: 
195:     Returns
196:     -------
197:     f : ndarray
198:         Array of sample frequencies.
199:     Pxx : ndarray
200:         Power spectral density or power spectrum of `x`.
201: 
202:     Notes
203:     -----
204:     .. versionadded:: 0.12.0
205: 
206:     See Also
207:     --------
208:     welch: Estimate power spectral density using Welch's method
209:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
210: 
211:     Examples
212:     --------
213:     >>> from scipy import signal
214:     >>> import matplotlib.pyplot as plt
215:     >>> np.random.seed(1234)
216: 
217:     Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
218:     0.001 V**2/Hz of white noise sampled at 10 kHz.
219: 
220:     >>> fs = 10e3
221:     >>> N = 1e5
222:     >>> amp = 2*np.sqrt(2)
223:     >>> freq = 1234.0
224:     >>> noise_power = 0.001 * fs / 2
225:     >>> time = np.arange(N) / fs
226:     >>> x = amp*np.sin(2*np.pi*freq*time)
227:     >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
228: 
229:     Compute and plot the power spectral density.
230: 
231:     >>> f, Pxx_den = signal.periodogram(x, fs)
232:     >>> plt.semilogy(f, Pxx_den)
233:     >>> plt.ylim([1e-7, 1e2])
234:     >>> plt.xlabel('frequency [Hz]')
235:     >>> plt.ylabel('PSD [V**2/Hz]')
236:     >>> plt.show()
237: 
238:     If we average the last half of the spectral density, to exclude the
239:     peak, we can recover the noise power on the signal.
240: 
241:     >>> np.mean(Pxx_den[25000:])
242:     0.00099728892368242854
243: 
244:     Now compute and plot the power spectrum.
245: 
246:     >>> f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')
247:     >>> plt.figure()
248:     >>> plt.semilogy(f, np.sqrt(Pxx_spec))
249:     >>> plt.ylim([1e-4, 1e1])
250:     >>> plt.xlabel('frequency [Hz]')
251:     >>> plt.ylabel('Linear spectrum [V RMS]')
252:     >>> plt.show()
253: 
254:     The peak height in the power spectrum is an estimate of the RMS
255:     amplitude.
256: 
257:     >>> np.sqrt(Pxx_spec.max())
258:     2.0077340678640727
259: 
260:     '''
261:     x = np.asarray(x)
262: 
263:     if x.size == 0:
264:         return np.empty(x.shape), np.empty(x.shape)
265: 
266:     if window is None:
267:         window = 'boxcar'
268: 
269:     if nfft is None:
270:         nperseg = x.shape[axis]
271:     elif nfft == x.shape[axis]:
272:         nperseg = nfft
273:     elif nfft > x.shape[axis]:
274:         nperseg = x.shape[axis]
275:     elif nfft < x.shape[axis]:
276:         s = [np.s_[:]]*len(x.shape)
277:         s[axis] = np.s_[:nfft]
278:         x = x[s]
279:         nperseg = nfft
280:         nfft = None
281: 
282:     return welch(x, fs, window, nperseg, 0, nfft, detrend, return_onesided,
283:                  scaling, axis)
284: 
285: 
286: def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
287:           detrend='constant', return_onesided=True, scaling='density',
288:           axis=-1):
289:     r'''
290:     Estimate power spectral density using Welch's method.
291: 
292:     Welch's method [1]_ computes an estimate of the power spectral
293:     density by dividing the data into overlapping segments, computing a
294:     modified periodogram for each segment and averaging the
295:     periodograms.
296: 
297:     Parameters
298:     ----------
299:     x : array_like
300:         Time series of measurement values
301:     fs : float, optional
302:         Sampling frequency of the `x` time series. Defaults to 1.0.
303:     window : str or tuple or array_like, optional
304:         Desired window to use. If `window` is a string or tuple, it is
305:         passed to `get_window` to generate the window values, which are
306:         DFT-even by default. See `get_window` for a list of windows and
307:         required parameters. If `window` is array_like it will be used
308:         directly as the window and its length must be nperseg. Defaults
309:         to a Hann window.
310:     nperseg : int, optional
311:         Length of each segment. Defaults to None, but if window is str or
312:         tuple, is set to 256, and if window is array_like, is set to the
313:         length of the window.
314:     noverlap : int, optional
315:         Number of points to overlap between segments. If `None`,
316:         ``noverlap = nperseg // 2``. Defaults to `None`.
317:     nfft : int, optional
318:         Length of the FFT used, if a zero padded FFT is desired. If
319:         `None`, the FFT length is `nperseg`. Defaults to `None`.
320:     detrend : str or function or `False`, optional
321:         Specifies how to detrend each segment. If `detrend` is a
322:         string, it is passed as the `type` argument to the `detrend`
323:         function. If it is a function, it takes a segment and returns a
324:         detrended segment. If `detrend` is `False`, no detrending is
325:         done. Defaults to 'constant'.
326:     return_onesided : bool, optional
327:         If `True`, return a one-sided spectrum for real data. If
328:         `False` return a two-sided spectrum. Note that for complex
329:         data, a two-sided spectrum is always returned.
330:     scaling : { 'density', 'spectrum' }, optional
331:         Selects between computing the power spectral density ('density')
332:         where `Pxx` has units of V**2/Hz and computing the power
333:         spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
334:         is measured in V and `fs` is measured in Hz. Defaults to
335:         'density'
336:     axis : int, optional
337:         Axis along which the periodogram is computed; the default is
338:         over the last axis (i.e. ``axis=-1``).
339: 
340:     Returns
341:     -------
342:     f : ndarray
343:         Array of sample frequencies.
344:     Pxx : ndarray
345:         Power spectral density or power spectrum of x.
346: 
347:     See Also
348:     --------
349:     periodogram: Simple, optionally modified periodogram
350:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
351: 
352:     Notes
353:     -----
354:     An appropriate amount of overlap will depend on the choice of window
355:     and on your requirements. For the default Hann window an overlap of
356:     50% is a reasonable trade off between accurately estimating the
357:     signal power, while not over counting any of the data. Narrower
358:     windows may require a larger overlap.
359: 
360:     If `noverlap` is 0, this method is equivalent to Bartlett's method
361:     [2]_.
362: 
363:     .. versionadded:: 0.12.0
364: 
365:     References
366:     ----------
367:     .. [1] P. Welch, "The use of the fast Fourier transform for the
368:            estimation of power spectra: A method based on time averaging
369:            over short, modified periodograms", IEEE Trans. Audio
370:            Electroacoust. vol. 15, pp. 70-73, 1967.
371:     .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
372:            Biometrika, vol. 37, pp. 1-16, 1950.
373: 
374:     Examples
375:     --------
376:     >>> from scipy import signal
377:     >>> import matplotlib.pyplot as plt
378:     >>> np.random.seed(1234)
379: 
380:     Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by
381:     0.001 V**2/Hz of white noise sampled at 10 kHz.
382: 
383:     >>> fs = 10e3
384:     >>> N = 1e5
385:     >>> amp = 2*np.sqrt(2)
386:     >>> freq = 1234.0
387:     >>> noise_power = 0.001 * fs / 2
388:     >>> time = np.arange(N) / fs
389:     >>> x = amp*np.sin(2*np.pi*freq*time)
390:     >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
391: 
392:     Compute and plot the power spectral density.
393: 
394:     >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)
395:     >>> plt.semilogy(f, Pxx_den)
396:     >>> plt.ylim([0.5e-3, 1])
397:     >>> plt.xlabel('frequency [Hz]')
398:     >>> plt.ylabel('PSD [V**2/Hz]')
399:     >>> plt.show()
400: 
401:     If we average the last half of the spectral density, to exclude the
402:     peak, we can recover the noise power on the signal.
403: 
404:     >>> np.mean(Pxx_den[256:])
405:     0.0009924865443739191
406: 
407:     Now compute and plot the power spectrum.
408: 
409:     >>> f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
410:     >>> plt.figure()
411:     >>> plt.semilogy(f, np.sqrt(Pxx_spec))
412:     >>> plt.xlabel('frequency [Hz]')
413:     >>> plt.ylabel('Linear spectrum [V RMS]')
414:     >>> plt.show()
415: 
416:     The peak height in the power spectrum is an estimate of the RMS
417:     amplitude.
418: 
419:     >>> np.sqrt(Pxx_spec.max())
420:     2.0077340678640727
421: 
422:     '''
423: 
424:     freqs, Pxx = csd(x, x, fs, window, nperseg, noverlap, nfft, detrend,
425:                      return_onesided, scaling, axis)
426: 
427:     return freqs, Pxx.real
428: 
429: 
430: def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
431:         detrend='constant', return_onesided=True, scaling='density', axis=-1):
432:     r'''
433:     Estimate the cross power spectral density, Pxy, using Welch's
434:     method.
435: 
436:     Parameters
437:     ----------
438:     x : array_like
439:         Time series of measurement values
440:     y : array_like
441:         Time series of measurement values
442:     fs : float, optional
443:         Sampling frequency of the `x` and `y` time series. Defaults
444:         to 1.0.
445:     window : str or tuple or array_like, optional
446:         Desired window to use. If `window` is a string or tuple, it is
447:         passed to `get_window` to generate the window values, which are
448:         DFT-even by default. See `get_window` for a list of windows and
449:         required parameters. If `window` is array_like it will be used
450:         directly as the window and its length must be nperseg. Defaults
451:         to a Hann window.
452:     nperseg : int, optional
453:         Length of each segment. Defaults to None, but if window is str or
454:         tuple, is set to 256, and if window is array_like, is set to the
455:         length of the window.
456:     noverlap: int, optional
457:         Number of points to overlap between segments. If `None`,
458:         ``noverlap = nperseg // 2``. Defaults to `None`.
459:     nfft : int, optional
460:         Length of the FFT used, if a zero padded FFT is desired. If
461:         `None`, the FFT length is `nperseg`. Defaults to `None`.
462:     detrend : str or function or `False`, optional
463:         Specifies how to detrend each segment. If `detrend` is a
464:         string, it is passed as the `type` argument to the `detrend`
465:         function. If it is a function, it takes a segment and returns a
466:         detrended segment. If `detrend` is `False`, no detrending is
467:         done. Defaults to 'constant'.
468:     return_onesided : bool, optional
469:         If `True`, return a one-sided spectrum for real data. If
470:         `False` return a two-sided spectrum. Note that for complex
471:         data, a two-sided spectrum is always returned.
472:     scaling : { 'density', 'spectrum' }, optional
473:         Selects between computing the cross spectral density ('density')
474:         where `Pxy` has units of V**2/Hz and computing the cross spectrum
475:         ('spectrum') where `Pxy` has units of V**2, if `x` and `y` are
476:         measured in V and `fs` is measured in Hz. Defaults to 'density'
477:     axis : int, optional
478:         Axis along which the CSD is computed for both inputs; the
479:         default is over the last axis (i.e. ``axis=-1``).
480: 
481:     Returns
482:     -------
483:     f : ndarray
484:         Array of sample frequencies.
485:     Pxy : ndarray
486:         Cross spectral density or cross power spectrum of x,y.
487: 
488:     See Also
489:     --------
490:     periodogram: Simple, optionally modified periodogram
491:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
492:     welch: Power spectral density by Welch's method. [Equivalent to
493:            csd(x,x)]
494:     coherence: Magnitude squared coherence by Welch's method.
495: 
496:     Notes
497:     --------
498:     By convention, Pxy is computed with the conjugate FFT of X
499:     multiplied by the FFT of Y.
500: 
501:     If the input series differ in length, the shorter series will be
502:     zero-padded to match.
503: 
504:     An appropriate amount of overlap will depend on the choice of window
505:     and on your requirements. For the default Hann window an overlap of
506:     50% is a reasonable trade off between accurately estimating the
507:     signal power, while not over counting any of the data. Narrower
508:     windows may require a larger overlap.
509: 
510:     .. versionadded:: 0.16.0
511: 
512:     References
513:     ----------
514:     .. [1] P. Welch, "The use of the fast Fourier transform for the
515:            estimation of power spectra: A method based on time averaging
516:            over short, modified periodograms", IEEE Trans. Audio
517:            Electroacoust. vol. 15, pp. 70-73, 1967.
518:     .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of
519:            Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975
520: 
521:     Examples
522:     --------
523:     >>> from scipy import signal
524:     >>> import matplotlib.pyplot as plt
525: 
526:     Generate two test signals with some common features.
527: 
528:     >>> fs = 10e3
529:     >>> N = 1e5
530:     >>> amp = 20
531:     >>> freq = 1234.0
532:     >>> noise_power = 0.001 * fs / 2
533:     >>> time = np.arange(N) / fs
534:     >>> b, a = signal.butter(2, 0.25, 'low')
535:     >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
536:     >>> y = signal.lfilter(b, a, x)
537:     >>> x += amp*np.sin(2*np.pi*freq*time)
538:     >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
539: 
540:     Compute and plot the magnitude of the cross spectral density.
541: 
542:     >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)
543:     >>> plt.semilogy(f, np.abs(Pxy))
544:     >>> plt.xlabel('frequency [Hz]')
545:     >>> plt.ylabel('CSD [V**2/Hz]')
546:     >>> plt.show()
547:     '''
548: 
549:     freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
550:                                      detrend, return_onesided, scaling, axis,
551:                                      mode='psd')
552: 
553:     # Average over windows.
554:     if len(Pxy.shape) >= 2 and Pxy.size > 0:
555:         if Pxy.shape[-1] > 1:
556:             Pxy = Pxy.mean(axis=-1)
557:         else:
558:             Pxy = np.reshape(Pxy, Pxy.shape[:-1])
559: 
560:     return freqs, Pxy
561: 
562: 
563: def spectrogram(x, fs=1.0, window=('tukey',.25), nperseg=None, noverlap=None,
564:                 nfft=None, detrend='constant', return_onesided=True,
565:                 scaling='density', axis=-1, mode='psd'):
566:     '''
567:     Compute a spectrogram with consecutive Fourier transforms.
568: 
569:     Spectrograms can be used as a way of visualizing the change of a
570:     nonstationary signal's frequency content over time.
571: 
572:     Parameters
573:     ----------
574:     x : array_like
575:         Time series of measurement values
576:     fs : float, optional
577:         Sampling frequency of the `x` time series. Defaults to 1.0.
578:     window : str or tuple or array_like, optional
579:         Desired window to use. If `window` is a string or tuple, it is
580:         passed to `get_window` to generate the window values, which are
581:         DFT-even by default. See `get_window` for a list of windows and
582:         required parameters. If `window` is array_like it will be used
583:         directly as the window and its length must be nperseg.
584:         Defaults to a Tukey window with shape parameter of 0.25.
585:     nperseg : int, optional
586:         Length of each segment. Defaults to None, but if window is str or
587:         tuple, is set to 256, and if window is array_like, is set to the
588:         length of the window.
589:     noverlap : int, optional
590:         Number of points to overlap between segments. If `None`,
591:         ``noverlap = nperseg // 8``. Defaults to `None`.
592:     nfft : int, optional
593:         Length of the FFT used, if a zero padded FFT is desired. If
594:         `None`, the FFT length is `nperseg`. Defaults to `None`.
595:     detrend : str or function or `False`, optional
596:         Specifies how to detrend each segment. If `detrend` is a
597:         string, it is passed as the `type` argument to the `detrend`
598:         function. If it is a function, it takes a segment and returns a
599:         detrended segment. If `detrend` is `False`, no detrending is
600:         done. Defaults to 'constant'.
601:     return_onesided : bool, optional
602:         If `True`, return a one-sided spectrum for real data. If
603:         `False` return a two-sided spectrum. Note that for complex
604:         data, a two-sided spectrum is always returned.
605:     scaling : { 'density', 'spectrum' }, optional
606:         Selects between computing the power spectral density ('density')
607:         where `Sxx` has units of V**2/Hz and computing the power
608:         spectrum ('spectrum') where `Sxx` has units of V**2, if `x`
609:         is measured in V and `fs` is measured in Hz. Defaults to
610:         'density'.
611:     axis : int, optional
612:         Axis along which the spectrogram is computed; the default is over
613:         the last axis (i.e. ``axis=-1``).
614:     mode : str, optional
615:         Defines what kind of return values are expected. Options are
616:         ['psd', 'complex', 'magnitude', 'angle', 'phase']. 'complex' is
617:         equivalent to the output of `stft` with no padding or boundary
618:         extension. 'magnitude' returns the absolute magnitude of the
619:         STFT. 'angle' and 'phase' return the complex angle of the STFT,
620:         with and without unwrapping, respectively.
621: 
622:     Returns
623:     -------
624:     f : ndarray
625:         Array of sample frequencies.
626:     t : ndarray
627:         Array of segment times.
628:     Sxx : ndarray
629:         Spectrogram of x. By default, the last axis of Sxx corresponds
630:         to the segment times.
631: 
632:     See Also
633:     --------
634:     periodogram: Simple, optionally modified periodogram
635:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
636:     welch: Power spectral density by Welch's method.
637:     csd: Cross spectral density by Welch's method.
638: 
639:     Notes
640:     -----
641:     An appropriate amount of overlap will depend on the choice of window
642:     and on your requirements. In contrast to welch's method, where the
643:     entire data stream is averaged over, one may wish to use a smaller
644:     overlap (or perhaps none at all) when computing a spectrogram, to
645:     maintain some statistical independence between individual segments.
646:     It is for this reason that the default window is a Tukey window with
647:     1/8th of a window's length overlap at each end.
648: 
649:     .. versionadded:: 0.16.0
650: 
651:     References
652:     ----------
653:     .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
654:            "Discrete-Time Signal Processing", Prentice Hall, 1999.
655: 
656:     Examples
657:     --------
658:     >>> from scipy import signal
659:     >>> import matplotlib.pyplot as plt
660: 
661:     Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
662:     modulated around 3kHz, corrupted by white noise of exponentially
663:     decreasing magnitude sampled at 10 kHz.
664: 
665:     >>> fs = 10e3
666:     >>> N = 1e5
667:     >>> amp = 2 * np.sqrt(2)
668:     >>> noise_power = 0.01 * fs / 2
669:     >>> time = np.arange(N) / float(fs)
670:     >>> mod = 500*np.cos(2*np.pi*0.25*time)
671:     >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
672:     >>> noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
673:     >>> noise *= np.exp(-time/5)
674:     >>> x = carrier + noise
675: 
676:     Compute and plot the spectrogram.
677: 
678:     >>> f, t, Sxx = signal.spectrogram(x, fs)
679:     >>> plt.pcolormesh(t, f, Sxx)
680:     >>> plt.ylabel('Frequency [Hz]')
681:     >>> plt.xlabel('Time [sec]')
682:     >>> plt.show()
683:     '''
684:     modelist = ['psd', 'complex', 'magnitude', 'angle', 'phase']
685:     if mode not in modelist:
686:         raise ValueError('unknown value for mode {}, must be one of {}'
687:                          .format(mode, modelist))
688: 
689:     # need to set default for nperseg before setting default for noverlap below
690:     window, nperseg = _triage_segments(window, nperseg,
691:                                        input_length=x.shape[axis])
692: 
693:     # Less overlap than welch, so samples are more statisically independent
694:     if noverlap is None:
695:         noverlap = nperseg // 8
696: 
697:     if mode == 'psd':
698:         freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
699:                                             noverlap, nfft, detrend,
700:                                             return_onesided, scaling, axis,
701:                                             mode='psd')
702: 
703:     else:
704:         freqs, time, Sxx = _spectral_helper(x, x, fs, window, nperseg,
705:                                             noverlap, nfft, detrend,
706:                                             return_onesided, scaling, axis,
707:                                             mode='stft')
708: 
709:         if mode == 'magnitude':
710:             Sxx = np.abs(Sxx)
711:         elif mode in ['angle', 'phase']:
712:             Sxx = np.angle(Sxx)
713:             if mode == 'phase':
714:                 # Sxx has one additional dimension for time strides
715:                 if axis < 0:
716:                     axis -= 1
717:                 Sxx = np.unwrap(Sxx, axis=axis)
718: 
719:         # mode =='complex' is same as `stft`, doesn't need modification
720: 
721:     return freqs, time, Sxx
722: 
723: 
724: def check_COLA(window, nperseg, noverlap, tol=1e-10):
725:     r'''
726:     Check whether the Constant OverLap Add (COLA) constraint is met
727: 
728:     Parameters
729:     ----------
730:     window : str or tuple or array_like
731:         Desired window to use. If `window` is a string or tuple, it is
732:         passed to `get_window` to generate the window values, which are
733:         DFT-even by default. See `get_window` for a list of windows and
734:         required parameters. If `window` is array_like it will be used
735:         directly as the window and its length must be nperseg.
736:     nperseg : int
737:         Length of each segment.
738:     noverlap : int
739:         Number of points to overlap between segments.
740:     tol : float, optional
741:         The allowed variance of a bin's weighted sum from the median bin
742:         sum.
743: 
744:     Returns
745:     -------
746:     verdict : bool
747:         `True` if chosen combination satisfies COLA within `tol`,
748:         `False` otherwise
749: 
750:     See Also
751:     --------
752:     stft: Short Time Fourier Transform
753:     istft: Inverse Short Time Fourier Transform
754: 
755:     Notes
756:     -----
757:     In order to enable inversion of an STFT via the inverse STFT in
758:     `istft`, the signal windowing must obey the constraint of "Constant
759:     OverLap Add" (COLA). This ensures that every point in the input data
760:     is equally weighted, thereby avoiding aliasing and allowing full
761:     reconstruction.
762: 
763:     Some examples of windows that satisfy COLA:
764:         - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...
765:         - Bartlett window at overlap of 1/2, 3/4, 5/6, ...
766:         - Hann window at 1/2, 2/3, 3/4, ...
767:         - Any Blackman family window at 2/3 overlap
768:         - Any window with ``noverlap = nperseg-1``
769: 
770:     A very comprehensive list of other windows may be found in [2]_,
771:     wherein the COLA condition is satisfied when the "Amplitude
772:     Flatness" is unity.
773: 
774:     .. versionadded:: 0.19.0
775: 
776:     References
777:     ----------
778:     .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K
779:            Publishing, 2011,ISBN 978-0-9745607-3-1.
780:     .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and
781:            spectral density estimation by the Discrete Fourier transform
782:            (DFT), including a comprehensive list of window functions and
783:            some new at-top windows", 2002,
784:            http://hdl.handle.net/11858/00-001M-0000-0013-557A-5
785: 
786:     Examples
787:     --------
788:     >>> from scipy import signal
789: 
790:     Confirm COLA condition for rectangular window of 75% (3/4) overlap:
791: 
792:     >>> signal.check_COLA(signal.boxcar(100), 100, 75)
793:     True
794: 
795:     COLA is not true for 25% (1/4) overlap, though:
796: 
797:     >>> signal.check_COLA(signal.boxcar(100), 100, 25)
798:     False
799: 
800:     "Symmetrical" Hann window (for filter design) is not COLA:
801: 
802:     >>> signal.check_COLA(signal.hann(120, sym=True), 120, 60)
803:     False
804: 
805:     "Periodic" or "DFT-even" Hann window (for FFT analysis) is COLA for
806:     overlap of 1/2, 2/3, 3/4, etc.:
807: 
808:     >>> signal.check_COLA(signal.hann(120, sym=False), 120, 60)
809:     True
810: 
811:     >>> signal.check_COLA(signal.hann(120, sym=False), 120, 80)
812:     True
813: 
814:     >>> signal.check_COLA(signal.hann(120, sym=False), 120, 90)
815:     True
816: 
817:     '''
818: 
819:     nperseg = int(nperseg)
820: 
821:     if nperseg < 1:
822:         raise ValueError('nperseg must be a positive integer')
823: 
824:     if noverlap >= nperseg:
825:         raise ValueError('noverlap must be less than nperseg.')
826:     noverlap = int(noverlap)
827: 
828:     if isinstance(window, string_types) or type(window) is tuple:
829:         win = get_window(window, nperseg)
830:     else:
831:         win = np.asarray(window)
832:         if len(win.shape) != 1:
833:             raise ValueError('window must be 1-D')
834:         if win.shape[0] != nperseg:
835:             raise ValueError('window must have length of nperseg')
836: 
837:     step = nperseg - noverlap
838:     binsums = np.sum((win[ii*step:(ii+1)*step] for ii in range(nperseg//step)),
839:                      axis=0)
840: 
841:     if nperseg % step != 0:
842:         binsums[:nperseg % step] += win[-(nperseg % step):]
843: 
844:     deviation = binsums - np.median(binsums)
845:     return np.max(np.abs(deviation)) < tol
846: 
847: 
848: def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
849:          detrend=False, return_onesided=True, boundary='zeros', padded=True,
850:          axis=-1):
851:     r'''
852:     Compute the Short Time Fourier Transform (STFT).
853: 
854:     STFTs can be used as a way of quantifying the change of a
855:     nonstationary signal's frequency and phase content over time.
856: 
857:     Parameters
858:     ----------
859:     x : array_like
860:         Time series of measurement values
861:     fs : float, optional
862:         Sampling frequency of the `x` time series. Defaults to 1.0.
863:     window : str or tuple or array_like, optional
864:         Desired window to use. If `window` is a string or tuple, it is
865:         passed to `get_window` to generate the window values, which are
866:         DFT-even by default. See `get_window` for a list of windows and
867:         required parameters. If `window` is array_like it will be used
868:         directly as the window and its length must be nperseg. Defaults
869:         to a Hann window.
870:     nperseg : int, optional
871:         Length of each segment. Defaults to 256.
872:     noverlap : int, optional
873:         Number of points to overlap between segments. If `None`,
874:         ``noverlap = nperseg // 2``. Defaults to `None`. When
875:         specified, the COLA constraint must be met (see Notes below).
876:     nfft : int, optional
877:         Length of the FFT used, if a zero padded FFT is desired. If
878:         `None`, the FFT length is `nperseg`. Defaults to `None`.
879:     detrend : str or function or `False`, optional
880:         Specifies how to detrend each segment. If `detrend` is a
881:         string, it is passed as the `type` argument to the `detrend`
882:         function. If it is a function, it takes a segment and returns a
883:         detrended segment. If `detrend` is `False`, no detrending is
884:         done. Defaults to `False`.
885:     return_onesided : bool, optional
886:         If `True`, return a one-sided spectrum for real data. If
887:         `False` return a two-sided spectrum. Note that for complex
888:         data, a two-sided spectrum is always returned. Defaults to
889:         `True`.
890:     boundary : str or None, optional
891:         Specifies whether the input signal is extended at both ends, and
892:         how to generate the new values, in order to center the first
893:         windowed segment on the first input point. This has the benefit
894:         of enabling reconstruction of the first input point when the
895:         employed window function starts at zero. Valid options are
896:         ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
897:         'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
898:         extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
899:     padded : bool, optional
900:         Specifies whether the input signal is zero-padded at the end to
901:         make the signal fit exactly into an integer number of window
902:         segments, so that all of the signal is included in the output.
903:         Defaults to `True`. Padding occurs after boundary extension, if
904:         `boundary` is not `None`, and `padded` is `True`, as is the
905:         default.
906:     axis : int, optional
907:         Axis along which the STFT is computed; the default is over the
908:         last axis (i.e. ``axis=-1``).
909: 
910:     Returns
911:     -------
912:     f : ndarray
913:         Array of sample frequencies.
914:     t : ndarray
915:         Array of segment times.
916:     Zxx : ndarray
917:         STFT of `x`. By default, the last axis of `Zxx` corresponds
918:         to the segment times.
919: 
920:     See Also
921:     --------
922:     istft: Inverse Short Time Fourier Transform
923:     check_COLA: Check whether the Constant OverLap Add (COLA) constraint
924:                 is met
925:     welch: Power spectral density by Welch's method.
926:     spectrogram: Spectrogram by Welch's method.
927:     csd: Cross spectral density by Welch's method.
928:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
929: 
930:     Notes
931:     -----
932:     In order to enable inversion of an STFT via the inverse STFT in
933:     `istft`, the signal windowing must obey the constraint of "Constant
934:     OverLap Add" (COLA), and the input signal must have complete
935:     windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
936:     (nperseg-noverlap) == 0``). The `padded` argument may be used to
937:     accomplish this.
938: 
939:     The COLA constraint ensures that every point in the input data is
940:     equally weighted, thereby avoiding aliasing and allowing full
941:     reconstruction. Whether a choice of `window`, `nperseg`, and
942:     `noverlap` satisfy this constraint can be tested with
943:     `check_COLA`.
944: 
945:     .. versionadded:: 0.19.0
946: 
947:     References
948:     ----------
949:     .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
950:            "Discrete-Time Signal Processing", Prentice Hall, 1999.
951:     .. [2] Daniel W. Griffin, Jae S. Limdt "Signal Estimation from
952:            Modified Short Fourier Transform", IEEE 1984,
953:            10.1109/TASSP.1984.1164317
954: 
955:     Examples
956:     --------
957:     >>> from scipy import signal
958:     >>> import matplotlib.pyplot as plt
959: 
960:     Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
961:     modulated around 3kHz, corrupted by white noise of exponentially
962:     decreasing magnitude sampled at 10 kHz.
963: 
964:     >>> fs = 10e3
965:     >>> N = 1e5
966:     >>> amp = 2 * np.sqrt(2)
967:     >>> noise_power = 0.01 * fs / 2
968:     >>> time = np.arange(N) / float(fs)
969:     >>> mod = 500*np.cos(2*np.pi*0.25*time)
970:     >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
971:     >>> noise = np.random.normal(scale=np.sqrt(noise_power),
972:     ...                          size=time.shape)
973:     >>> noise *= np.exp(-time/5)
974:     >>> x = carrier + noise
975: 
976:     Compute and plot the STFT's magnitude.
977: 
978:     >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)
979:     >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
980:     >>> plt.title('STFT Magnitude')
981:     >>> plt.ylabel('Frequency [Hz]')
982:     >>> plt.xlabel('Time [sec]')
983:     >>> plt.show()
984:     '''
985: 
986:     freqs, time, Zxx = _spectral_helper(x, x, fs, window, nperseg, noverlap,
987:                                         nfft, detrend, return_onesided,
988:                                         scaling='spectrum', axis=axis,
989:                                         mode='stft', boundary=boundary,
990:                                         padded=padded)
991: 
992:     return freqs, time, Zxx
993: 
994: 
995: def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
996:           input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2):
997:     r'''
998:     Perform the inverse Short Time Fourier transform (iSTFT).
999: 
1000:     Parameters
1001:     ----------
1002:     Zxx : array_like
1003:         STFT of the signal to be reconstructed. If a purely real array
1004:         is passed, it will be cast to a complex data type.
1005:     fs : float, optional
1006:         Sampling frequency of the time series. Defaults to 1.0.
1007:     window : str or tuple or array_like, optional
1008:         Desired window to use. If `window` is a string or tuple, it is
1009:         passed to `get_window` to generate the window values, which are
1010:         DFT-even by default. See `get_window` for a list of windows and
1011:         required parameters. If `window` is array_like it will be used
1012:         directly as the window and its length must be nperseg. Defaults
1013:         to a Hann window. Must match the window used to generate the
1014:         STFT for faithful inversion.
1015:     nperseg : int, optional
1016:         Number of data points corresponding to each STFT segment. This
1017:         parameter must be specified if the number of data points per
1018:         segment is odd, or if the STFT was padded via ``nfft >
1019:         nperseg``. If `None`, the value depends on the shape of
1020:         `Zxx` and `input_onesided`. If `input_onesided` is True,
1021:         ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,
1022:         ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.
1023:     noverlap : int, optional
1024:         Number of points to overlap between segments. If `None`, half
1025:         of the segment length. Defaults to `None`. When specified, the
1026:         COLA constraint must be met (see Notes below), and should match
1027:         the parameter used to generate the STFT. Defaults to `None`.
1028:     nfft : int, optional
1029:         Number of FFT points corresponding to each STFT segment. This
1030:         parameter must be specified if the STFT was padded via ``nfft >
1031:         nperseg``. If `None`, the default values are the same as for
1032:         `nperseg`, detailed above, with one exception: if
1033:         `input_onesided` is True and
1034:         ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on
1035:         that value. This case allows the proper inversion of an
1036:         odd-length unpadded STFT using ``nfft=None``. Defaults to
1037:         `None`.
1038:     input_onesided : bool, optional
1039:         If `True`, interpret the input array as one-sided FFTs, such
1040:         as is returned by `stft` with ``return_onesided=True`` and
1041:         `numpy.fft.rfft`. If `False`, interpret the input as a a
1042:         two-sided FFT. Defaults to `True`.
1043:     boundary : bool, optional
1044:         Specifies whether the input signal was extended at its
1045:         boundaries by supplying a non-`None` ``boundary`` argument to
1046:         `stft`. Defaults to `True`.
1047:     time_axis : int, optional
1048:         Where the time segments of the STFT is located; the default is
1049:         the last axis (i.e. ``axis=-1``).
1050:     freq_axis : int, optional
1051:         Where the frequency axis of the STFT is located; the default is
1052:         the penultimate axis (i.e. ``axis=-2``).
1053: 
1054:     Returns
1055:     -------
1056:     t : ndarray
1057:         Array of output data times.
1058:     x : ndarray
1059:         iSTFT of `Zxx`.
1060: 
1061:     See Also
1062:     --------
1063:     stft: Short Time Fourier Transform
1064:     check_COLA: Check whether the Constant OverLap Add (COLA) constraint
1065:                 is met
1066: 
1067:     Notes
1068:     -----
1069:     In order to enable inversion of an STFT via the inverse STFT with
1070:     `istft`, the signal windowing must obey the constraint of "Constant
1071:     OverLap Add" (COLA). This ensures that every point in the input data
1072:     is equally weighted, thereby avoiding aliasing and allowing full
1073:     reconstruction. Whether a choice of `window`, `nperseg`, and
1074:     `noverlap` satisfy this constraint can be tested with
1075:     `check_COLA`, by using ``nperseg = Zxx.shape[freq_axis]``.
1076: 
1077:     An STFT which has been modified (via masking or otherwise) is not
1078:     guaranteed to correspond to a exactly realizible signal. This
1079:     function implements the iSTFT via the least-squares esimation
1080:     algorithm detailed in [2]_, which produces a signal that minimizes
1081:     the mean squared error between the STFT of the returned signal and
1082:     the modified STFT.
1083: 
1084:     .. versionadded:: 0.19.0
1085: 
1086:     References
1087:     ----------
1088:     .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
1089:            "Discrete-Time Signal Processing", Prentice Hall, 1999.
1090:     .. [2] Daniel W. Griffin, Jae S. Limdt "Signal Estimation from
1091:            Modified Short Fourier Transform", IEEE 1984,
1092:            10.1109/TASSP.1984.1164317
1093: 
1094:     Examples
1095:     --------
1096:     >>> from scipy import signal
1097:     >>> import matplotlib.pyplot as plt
1098: 
1099:     Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by
1100:     0.001 V**2/Hz of white noise sampled at 1024 Hz.
1101: 
1102:     >>> fs = 1024
1103:     >>> N = 10*fs
1104:     >>> nperseg = 512
1105:     >>> amp = 2 * np.sqrt(2)
1106:     >>> noise_power = 0.001 * fs / 2
1107:     >>> time = np.arange(N) / float(fs)
1108:     >>> carrier = amp * np.sin(2*np.pi*50*time)
1109:     >>> noise = np.random.normal(scale=np.sqrt(noise_power),
1110:     ...                          size=time.shape)
1111:     >>> x = carrier + noise
1112: 
1113:     Compute the STFT, and plot its magnitude
1114: 
1115:     >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)
1116:     >>> plt.figure()
1117:     >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
1118:     >>> plt.ylim([f[1], f[-1]])
1119:     >>> plt.title('STFT Magnitude')
1120:     >>> plt.ylabel('Frequency [Hz]')
1121:     >>> plt.xlabel('Time [sec]')
1122:     >>> plt.yscale('log')
1123:     >>> plt.show()
1124: 
1125:     Zero the components that are 10% or less of the carrier magnitude,
1126:     then convert back to a time series via inverse STFT
1127: 
1128:     >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
1129:     >>> _, xrec = signal.istft(Zxx, fs)
1130: 
1131:     Compare the cleaned signal with the original and true carrier signals.
1132: 
1133:     >>> plt.figure()
1134:     >>> plt.plot(time, x, time, xrec, time, carrier)
1135:     >>> plt.xlim([2, 2.1])
1136:     >>> plt.xlabel('Time [sec]')
1137:     >>> plt.ylabel('Signal')
1138:     >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
1139:     >>> plt.show()
1140: 
1141:     Note that the cleaned signal does not start as abruptly as the original,
1142:     since some of the coefficients of the transient were also removed:
1143: 
1144:     >>> plt.figure()
1145:     >>> plt.plot(time, x, time, xrec, time, carrier)
1146:     >>> plt.xlim([0, 0.1])
1147:     >>> plt.xlabel('Time [sec]')
1148:     >>> plt.ylabel('Signal')
1149:     >>> plt.legend(['Carrier + Noise', 'Filtered via STFT', 'True Carrier'])
1150:     >>> plt.show()
1151: 
1152:     '''
1153: 
1154:     # Make sure input is an ndarray of appropriate complex dtype
1155:     Zxx = np.asarray(Zxx) + 0j
1156:     freq_axis = int(freq_axis)
1157:     time_axis = int(time_axis)
1158: 
1159:     if Zxx.ndim < 2:
1160:         raise ValueError('Input stft must be at least 2d!')
1161: 
1162:     if freq_axis == time_axis:
1163:         raise ValueError('Must specify differing time and frequency axes!')
1164: 
1165:     nseg = Zxx.shape[time_axis]
1166: 
1167:     if input_onesided:
1168:         # Assume even segment length
1169:         n_default = 2*(Zxx.shape[freq_axis] - 1)
1170:     else:
1171:         n_default = Zxx.shape[freq_axis]
1172: 
1173:     # Check windowing parameters
1174:     if nperseg is None:
1175:         nperseg = n_default
1176:     else:
1177:         nperseg = int(nperseg)
1178:         if nperseg < 1:
1179:             raise ValueError('nperseg must be a positive integer')
1180: 
1181:     if nfft is None:
1182:         if (input_onesided) and (nperseg == n_default + 1):
1183:             # Odd nperseg, no FFT padding
1184:             nfft = nperseg
1185:         else:
1186:             nfft = n_default
1187:     elif nfft < nperseg:
1188:         raise ValueError('nfft must be greater than or equal to nperseg.')
1189:     else:
1190:         nfft = int(nfft)
1191: 
1192:     if noverlap is None:
1193:         noverlap = nperseg//2
1194:     else:
1195:         noverlap = int(noverlap)
1196:     if noverlap >= nperseg:
1197:         raise ValueError('noverlap must be less than nperseg.')
1198:     nstep = nperseg - noverlap
1199: 
1200:     if not check_COLA(window, nperseg, noverlap):
1201:         raise ValueError('Window, STFT shape and noverlap do not satisfy the '
1202:                          'COLA constraint.')
1203: 
1204:     # Rearrange axes if neccessary
1205:     if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
1206:         # Turn negative indices to positive for the call to transpose
1207:         if freq_axis < 0:
1208:             freq_axis = Zxx.ndim + freq_axis
1209:         if time_axis < 0:
1210:             time_axis = Zxx.ndim + time_axis
1211:         zouter = list(range(Zxx.ndim))
1212:         for ax in sorted([time_axis, freq_axis], reverse=True):
1213:             zouter.pop(ax)
1214:         Zxx = np.transpose(Zxx, zouter+[freq_axis, time_axis])
1215: 
1216:     # Get window as array
1217:     if isinstance(window, string_types) or type(window) is tuple:
1218:         win = get_window(window, nperseg)
1219:     else:
1220:         win = np.asarray(window)
1221:         if len(win.shape) != 1:
1222:             raise ValueError('window must be 1-D')
1223:         if win.shape[0] != nperseg:
1224:             raise ValueError('window must have length of {0}'.format(nperseg))
1225: 
1226:     if input_onesided:
1227:         ifunc = np.fft.irfft
1228:     else:
1229:         ifunc = fftpack.ifft
1230: 
1231:     xsubs = ifunc(Zxx, axis=-2, n=nfft)[..., :nperseg, :]
1232: 
1233:     # Initialize output and normalization arrays
1234:     outputlength = nperseg + (nseg-1)*nstep
1235:     x = np.zeros(list(Zxx.shape[:-2])+[outputlength], dtype=xsubs.dtype)
1236:     norm = np.zeros(outputlength, dtype=xsubs.dtype)
1237: 
1238:     if np.result_type(win, xsubs) != xsubs.dtype:
1239:         win = win.astype(xsubs.dtype)
1240: 
1241:     xsubs *= win.sum()  # This takes care of the 'spectrum' scaling
1242: 
1243:     # Construct the output from the ifft segments
1244:     # This loop could perhaps be vectorized/strided somehow...
1245:     for ii in range(nseg):
1246:         # Window the ifft
1247:         x[..., ii*nstep:ii*nstep+nperseg] += xsubs[..., ii] * win
1248:         norm[..., ii*nstep:ii*nstep+nperseg] += win**2
1249: 
1250:     # Divide out normalization where non-tiny
1251:     x /= np.where(norm > 1e-10, norm, 1.0)
1252: 
1253:     # Remove extension points
1254:     if boundary:
1255:         x = x[..., nperseg//2:-(nperseg//2)]
1256: 
1257:     if input_onesided:
1258:         x = x.real
1259: 
1260:     # Put axes back
1261:     if x.ndim > 1:
1262:         if time_axis != Zxx.ndim-1:
1263:             if freq_axis < time_axis:
1264:                 time_axis -= 1
1265:             x = np.rollaxis(x, -1, time_axis)
1266: 
1267:     time = np.arange(x.shape[0])/float(fs)
1268:     return time, x
1269: 
1270: 
1271: def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
1272:               nfft=None, detrend='constant', axis=-1):
1273:     r'''
1274:     Estimate the magnitude squared coherence estimate, Cxy, of
1275:     discrete-time signals X and Y using Welch's method.
1276: 
1277:     ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power
1278:     spectral density estimates of X and Y, and `Pxy` is the cross
1279:     spectral density estimate of X and Y.
1280: 
1281:     Parameters
1282:     ----------
1283:     x : array_like
1284:         Time series of measurement values
1285:     y : array_like
1286:         Time series of measurement values
1287:     fs : float, optional
1288:         Sampling frequency of the `x` and `y` time series. Defaults
1289:         to 1.0.
1290:     window : str or tuple or array_like, optional
1291:         Desired window to use. If `window` is a string or tuple, it is
1292:         passed to `get_window` to generate the window values, which are
1293:         DFT-even by default. See `get_window` for a list of windows and
1294:         required parameters. If `window` is array_like it will be used
1295:         directly as the window and its length must be nperseg. Defaults
1296:         to a Hann window.
1297:     nperseg : int, optional
1298:         Length of each segment. Defaults to None, but if window is str or
1299:         tuple, is set to 256, and if window is array_like, is set to the
1300:         length of the window.
1301:     noverlap: int, optional
1302:         Number of points to overlap between segments. If `None`,
1303:         ``noverlap = nperseg // 2``. Defaults to `None`.
1304:     nfft : int, optional
1305:         Length of the FFT used, if a zero padded FFT is desired. If
1306:         `None`, the FFT length is `nperseg`. Defaults to `None`.
1307:     detrend : str or function or `False`, optional
1308:         Specifies how to detrend each segment. If `detrend` is a
1309:         string, it is passed as the `type` argument to the `detrend`
1310:         function. If it is a function, it takes a segment and returns a
1311:         detrended segment. If `detrend` is `False`, no detrending is
1312:         done. Defaults to 'constant'.
1313:     axis : int, optional
1314:         Axis along which the coherence is computed for both inputs; the
1315:         default is over the last axis (i.e. ``axis=-1``).
1316: 
1317:     Returns
1318:     -------
1319:     f : ndarray
1320:         Array of sample frequencies.
1321:     Cxy : ndarray
1322:         Magnitude squared coherence of x and y.
1323: 
1324:     See Also
1325:     --------
1326:     periodogram: Simple, optionally modified periodogram
1327:     lombscargle: Lomb-Scargle periodogram for unevenly sampled data
1328:     welch: Power spectral density by Welch's method.
1329:     csd: Cross spectral density by Welch's method.
1330: 
1331:     Notes
1332:     --------
1333:     An appropriate amount of overlap will depend on the choice of window
1334:     and on your requirements. For the default Hann window an overlap of
1335:     50% is a reasonable trade off between accurately estimating the
1336:     signal power, while not over counting any of the data. Narrower
1337:     windows may require a larger overlap.
1338: 
1339:     .. versionadded:: 0.16.0
1340: 
1341:     References
1342:     ----------
1343:     .. [1] P. Welch, "The use of the fast Fourier transform for the
1344:            estimation of power spectra: A method based on time averaging
1345:            over short, modified periodograms", IEEE Trans. Audio
1346:            Electroacoust. vol. 15, pp. 70-73, 1967.
1347:     .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of
1348:            Signals" Prentice Hall, 2005
1349: 
1350:     Examples
1351:     --------
1352:     >>> from scipy import signal
1353:     >>> import matplotlib.pyplot as plt
1354: 
1355:     Generate two test signals with some common features.
1356: 
1357:     >>> fs = 10e3
1358:     >>> N = 1e5
1359:     >>> amp = 20
1360:     >>> freq = 1234.0
1361:     >>> noise_power = 0.001 * fs / 2
1362:     >>> time = np.arange(N) / fs
1363:     >>> b, a = signal.butter(2, 0.25, 'low')
1364:     >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
1365:     >>> y = signal.lfilter(b, a, x)
1366:     >>> x += amp*np.sin(2*np.pi*freq*time)
1367:     >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)
1368: 
1369:     Compute and plot the coherence.
1370: 
1371:     >>> f, Cxy = signal.coherence(x, y, fs, nperseg=1024)
1372:     >>> plt.semilogy(f, Cxy)
1373:     >>> plt.xlabel('frequency [Hz]')
1374:     >>> plt.ylabel('Coherence')
1375:     >>> plt.show()
1376:     '''
1377: 
1378:     freqs, Pxx = welch(x, fs, window, nperseg, noverlap, nfft, detrend,
1379:                        axis=axis)
1380:     _, Pyy = welch(y, fs, window, nperseg, noverlap, nfft, detrend, axis=axis)
1381:     _, Pxy = csd(x, y, fs, window, nperseg, noverlap, nfft, detrend, axis=axis)
1382: 
1383:     Cxy = np.abs(Pxy)**2 / Pxx / Pyy
1384: 
1385:     return freqs, Cxy
1386: 
1387: 
1388: def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
1389:                      nfft=None, detrend='constant', return_onesided=True,
1390:                      scaling='spectrum', axis=-1, mode='psd', boundary=None,
1391:                      padded=False):
1392:     '''
1393:     Calculate various forms of windowed FFTs for PSD, CSD, etc.
1394: 
1395:     This is a helper function that implements the commonality between
1396:     the stft, psd, csd, and spectrogram functions. It is not designed to
1397:     be called externally. The windows are not averaged over; the result
1398:     from each window is returned.
1399: 
1400:     Parameters
1401:     ---------
1402:     x : array_like
1403:         Array or sequence containing the data to be analyzed.
1404:     y : array_like
1405:         Array or sequence containing the data to be analyzed. If this is
1406:         the same object in memory as `x` (i.e. ``_spectral_helper(x,
1407:         x, ...)``), the extra computations are spared.
1408:     fs : float, optional
1409:         Sampling frequency of the time series. Defaults to 1.0.
1410:     window : str or tuple or array_like, optional
1411:         Desired window to use. If `window` is a string or tuple, it is
1412:         passed to `get_window` to generate the window values, which are
1413:         DFT-even by default. See `get_window` for a list of windows and
1414:         required parameters. If `window` is array_like it will be used
1415:         directly as the window and its length must be nperseg. Defaults
1416:         to a Hann window.
1417:     nperseg : int, optional
1418:         Length of each segment. Defaults to None, but if window is str or
1419:         tuple, is set to 256, and if window is array_like, is set to the
1420:         length of the window.
1421:     noverlap : int, optional
1422:         Number of points to overlap between segments. If `None`,
1423:         ``noverlap = nperseg // 2``. Defaults to `None`.
1424:     nfft : int, optional
1425:         Length of the FFT used, if a zero padded FFT is desired. If
1426:         `None`, the FFT length is `nperseg`. Defaults to `None`.
1427:     detrend : str or function or `False`, optional
1428:         Specifies how to detrend each segment. If `detrend` is a
1429:         string, it is passed as the `type` argument to the `detrend`
1430:         function. If it is a function, it takes a segment and returns a
1431:         detrended segment. If `detrend` is `False`, no detrending is
1432:         done. Defaults to 'constant'.
1433:     return_onesided : bool, optional
1434:         If `True`, return a one-sided spectrum for real data. If
1435:         `False` return a two-sided spectrum. Note that for complex
1436:         data, a two-sided spectrum is always returned.
1437:     scaling : { 'density', 'spectrum' }, optional
1438:         Selects between computing the cross spectral density ('density')
1439:         where `Pxy` has units of V**2/Hz and computing the cross
1440:         spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
1441:         and `y` are measured in V and `fs` is measured in Hz.
1442:         Defaults to 'density'
1443:     axis : int, optional
1444:         Axis along which the FFTs are computed; the default is over the
1445:         last axis (i.e. ``axis=-1``).
1446:     mode: str {'psd', 'stft'}, optional
1447:         Defines what kind of return values are expected. Defaults to
1448:         'psd'.
1449:     boundary : str or None, optional
1450:         Specifies whether the input signal is extended at both ends, and
1451:         how to generate the new values, in order to center the first
1452:         windowed segment on the first input point. This has the benefit
1453:         of enabling reconstruction of the first input point when the
1454:         employed window function starts at zero. Valid options are
1455:         ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
1456:         `None`.
1457:     padded : bool, optional
1458:         Specifies whether the input signal is zero-padded at the end to
1459:         make the signal fit exactly into an integer number of window
1460:         segments, so that all of the signal is included in the output.
1461:         Defaults to `False`. Padding occurs after boundary extension, if
1462:         `boundary` is not `None`, and `padded` is `True`.
1463:     Returns
1464:     -------
1465:     freqs : ndarray
1466:         Array of sample frequencies.
1467:     t : ndarray
1468:         Array of times corresponding to each data segment
1469:     result : ndarray
1470:         Array of output data, contents dependant on *mode* kwarg.
1471: 
1472:     References
1473:     ----------
1474:     .. [1] Stack Overflow, "Rolling window for 1D arrays in Numpy?",
1475:            http://stackoverflow.com/a/6811241
1476:     .. [2] Stack Overflow, "Using strides for an efficient moving
1477:            average filter", http://stackoverflow.com/a/4947453
1478: 
1479:     Notes
1480:     -----
1481:     Adapted from matplotlib.mlab
1482: 
1483:     .. versionadded:: 0.16.0
1484:     '''
1485:     if mode not in ['psd', 'stft']:
1486:         raise ValueError("Unknown value for mode %s, must be one of: "
1487:                          "{'psd', 'stft'}" % mode)
1488: 
1489:     boundary_funcs = {'even': even_ext,
1490:                       'odd': odd_ext,
1491:                       'constant': const_ext,
1492:                       'zeros': zero_ext,
1493:                       None: None}
1494: 
1495:     if boundary not in boundary_funcs:
1496:         raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
1497:                           .format(boundary, list(boundary_funcs.keys())))
1498: 
1499:     # If x and y are the same object we can save ourselves some computation.
1500:     same_data = y is x
1501: 
1502:     if not same_data and mode != 'psd':
1503:         raise ValueError("x and y must be equal if mode is 'stft'")
1504: 
1505:     axis = int(axis)
1506: 
1507:     # Ensure we have np.arrays, get outdtype
1508:     x = np.asarray(x)
1509:     if not same_data:
1510:         y = np.asarray(y)
1511:         outdtype = np.result_type(x, y, np.complex64)
1512:     else:
1513:         outdtype = np.result_type(x, np.complex64)
1514: 
1515:     if not same_data:
1516:         # Check if we can broadcast the outer axes together
1517:         xouter = list(x.shape)
1518:         youter = list(y.shape)
1519:         xouter.pop(axis)
1520:         youter.pop(axis)
1521:         try:
1522:             outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
1523:         except ValueError:
1524:             raise ValueError('x and y cannot be broadcast together.')
1525: 
1526:     if same_data:
1527:         if x.size == 0:
1528:             return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
1529:     else:
1530:         if x.size == 0 or y.size == 0:
1531:             outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
1532:             emptyout = np.rollaxis(np.empty(outshape), -1, axis)
1533:             return emptyout, emptyout, emptyout
1534: 
1535:     if x.ndim > 1:
1536:         if axis != -1:
1537:             x = np.rollaxis(x, axis, len(x.shape))
1538:             if not same_data and y.ndim > 1:
1539:                 y = np.rollaxis(y, axis, len(y.shape))
1540: 
1541:     # Check if x and y are the same length, zero-pad if neccesary
1542:     if not same_data:
1543:         if x.shape[-1] != y.shape[-1]:
1544:             if x.shape[-1] < y.shape[-1]:
1545:                 pad_shape = list(x.shape)
1546:                 pad_shape[-1] = y.shape[-1] - x.shape[-1]
1547:                 x = np.concatenate((x, np.zeros(pad_shape)), -1)
1548:             else:
1549:                 pad_shape = list(y.shape)
1550:                 pad_shape[-1] = x.shape[-1] - y.shape[-1]
1551:                 y = np.concatenate((y, np.zeros(pad_shape)), -1)
1552: 
1553:     if nperseg is not None:  # if specified by user
1554:         nperseg = int(nperseg)
1555:         if nperseg < 1:
1556:             raise ValueError('nperseg must be a positive integer')
1557: 
1558:     # parse window; if array like, then set nperseg = win.shape
1559:     win, nperseg = _triage_segments(window, nperseg,input_length=x.shape[-1])
1560: 
1561:     if nfft is None:
1562:         nfft = nperseg
1563:     elif nfft < nperseg:
1564:         raise ValueError('nfft must be greater than or equal to nperseg.')
1565:     else:
1566:         nfft = int(nfft)
1567: 
1568:     if noverlap is None:
1569:         noverlap = nperseg//2
1570:     else:
1571:         noverlap = int(noverlap)
1572:     if noverlap >= nperseg:
1573:         raise ValueError('noverlap must be less than nperseg.')
1574:     nstep = nperseg - noverlap
1575: 
1576:     # Padding occurs after boundary extension, so that the extended signal ends
1577:     # in zeros, instead of introducing an impulse at the end.
1578:     # I.e. if x = [..., 3, 2]
1579:     # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
1580:     # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]
1581: 
1582:     if boundary is not None:
1583:         ext_func = boundary_funcs[boundary]
1584:         x = ext_func(x, nperseg//2, axis=-1)
1585:         if not same_data:
1586:             y = ext_func(y, nperseg//2, axis=-1)
1587: 
1588:     if padded:
1589:         # Pad to integer number of windowed segments
1590:         # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
1591:         nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
1592:         zeros_shape = list(x.shape[:-1]) + [nadd]
1593:         x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
1594:         if not same_data:
1595:             zeros_shape = list(y.shape[:-1]) + [nadd]
1596:             y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)
1597: 
1598:     # Handle detrending and window functions
1599:     if not detrend:
1600:         def detrend_func(d):
1601:             return d
1602:     elif not hasattr(detrend, '__call__'):
1603:         def detrend_func(d):
1604:             return signaltools.detrend(d, type=detrend, axis=-1)
1605:     elif axis != -1:
1606:         # Wrap this function so that it receives a shape that it could
1607:         # reasonably expect to receive.
1608:         def detrend_func(d):
1609:             d = np.rollaxis(d, -1, axis)
1610:             d = detrend(d)
1611:             return np.rollaxis(d, axis, len(d.shape))
1612:     else:
1613:         detrend_func = detrend
1614: 
1615:     if np.result_type(win,np.complex64) != outdtype:
1616:         win = win.astype(outdtype)
1617: 
1618:     if scaling == 'density':
1619:         scale = 1.0 / (fs * (win*win).sum())
1620:     elif scaling == 'spectrum':
1621:         scale = 1.0 / win.sum()**2
1622:     else:
1623:         raise ValueError('Unknown scaling: %r' % scaling)
1624: 
1625:     if mode == 'stft':
1626:         scale = np.sqrt(scale)
1627: 
1628:     if return_onesided:
1629:         if np.iscomplexobj(x):
1630:             sides = 'twosided'
1631:             warnings.warn('Input data is complex, switching to '
1632:                           'return_onesided=False')
1633:         else:
1634:             sides = 'onesided'
1635:             if not same_data:
1636:                 if np.iscomplexobj(y):
1637:                     sides = 'twosided'
1638:                     warnings.warn('Input data is complex, switching to '
1639:                                   'return_onesided=False')
1640:     else:
1641:         sides = 'twosided'
1642: 
1643:     if sides == 'twosided':
1644:         freqs = fftpack.fftfreq(nfft, 1/fs)
1645:     elif sides == 'onesided':
1646:         freqs = np.fft.rfftfreq(nfft, 1/fs)
1647: 
1648:     # Perform the windowed FFTs
1649:     result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)
1650: 
1651:     if not same_data:
1652:         # All the same operations on the y data
1653:         result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
1654:                                sides)
1655:         result = np.conjugate(result) * result_y
1656:     elif mode == 'psd':
1657:         result = np.conjugate(result) * result
1658: 
1659:     result *= scale
1660:     if sides == 'onesided' and mode == 'psd':
1661:         if nfft % 2:
1662:             result[..., 1:] *= 2
1663:         else:
1664:             # Last point is unpaired Nyquist freq point, don't double
1665:             result[..., 1:-1] *= 2
1666: 
1667:     time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
1668:                      nperseg - noverlap)/float(fs)
1669:     if boundary is not None:
1670:         time -= (nperseg/2) / fs
1671: 
1672:     result = result.astype(outdtype)
1673: 
1674:     # All imaginary parts are zero anyways
1675:     if same_data and mode != 'stft':
1676:         result = result.real
1677: 
1678:     # Output is going to have new last axis for time/window index, so a
1679:     # negative axis index shifts down one
1680:     if axis < 0:
1681:         axis -= 1
1682: 
1683:     # Roll frequency axis back to axis where the data came from
1684:     result = np.rollaxis(result, -1, axis)
1685: 
1686:     return freqs, time, result
1687: 
1688: 
1689: def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):
1690:     '''
1691:     Calculate windowed FFT, for internal use by
1692:     scipy.signal._spectral_helper
1693: 
1694:     This is a helper function that does the main FFT calculation for
1695:     `_spectral helper`. All input valdiation is performed there, and the
1696:     data axis is assumed to be the last axis of x. It is not designed to
1697:     be called externally. The windows are not averaged over; the result
1698:     from each window is returned.
1699: 
1700:     Returns
1701:     -------
1702:     result : ndarray
1703:         Array of FFT data
1704: 
1705:     References
1706:     ----------
1707:     .. [1] Stack Overflow, "Repeat NumPy array without replicating
1708:            data?", http://stackoverflow.com/a/5568169
1709: 
1710:     Notes
1711:     -----
1712:     Adapted from matplotlib.mlab
1713: 
1714:     .. versionadded:: 0.16.0
1715:     '''
1716:     # Created strided array of data segments
1717:     if nperseg == 1 and noverlap == 0:
1718:         result = x[..., np.newaxis]
1719:     else:
1720:         step = nperseg - noverlap
1721:         shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
1722:         strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
1723:         result = np.lib.stride_tricks.as_strided(x, shape=shape,
1724:                                                  strides=strides)
1725: 
1726:     # Detrend each data segment individually
1727:     result = detrend_func(result)
1728: 
1729:     # Apply window by multiplication
1730:     result = win * result
1731: 
1732:     # Perform the fft. Acts on last axis by default. Zero-pads automatically
1733:     if sides == 'twosided':
1734:         func = fftpack.fft
1735:     else:
1736:         result = result.real
1737:         func = np.fft.rfft
1738:     result = func(result, n=nfft)
1739: 
1740:     return result
1741: 
1742: def _triage_segments(window, nperseg,input_length):
1743:     '''
1744:     Parses window and nperseg arguments for spectrogram and _spectral_helper.
1745:     This is a helper function, not meant to be called externally.
1746: 
1747:     Parameters
1748:     ---------
1749:     window : string, tuple, or ndarray
1750:         If window is specified by a string or tuple and nperseg is not
1751:         specified, nperseg is set to the default of 256 and returns a window of
1752:         that length.
1753:         If instead the window is array_like and nperseg is not specified, then
1754:         nperseg is set to the length of the window. A ValueError is raised if
1755:         the user supplies both an array_like window and a value for nperseg but
1756:         nperseg does not equal the length of the window.
1757: 
1758:     nperseg : int
1759:         Length of each segment
1760: 
1761:     input_length: int
1762:         Length of input signal, i.e. x.shape[-1]. Used to test for errors.
1763: 
1764:     Returns
1765:     -------
1766:     win : ndarray
1767:         window. If function was called with string or tuple than this will hold
1768:         the actual array used as a window.
1769: 
1770:     nperseg : int
1771:         Length of each segment. If window is str or tuple, nperseg is set to
1772:         256. If window is array_like, nperseg is set to the length of the
1773:         6
1774:         window.
1775:     '''
1776: 
1777:     #parse window; if array like, then set nperseg = win.shape
1778:     if isinstance(window, string_types) or isinstance(window, tuple):
1779:         # if nperseg not specified
1780:         if nperseg is None:
1781:             nperseg = 256  # then change to default
1782:         if nperseg > input_length:
1783:             warnings.warn('nperseg = {0:d} is greater than input length '
1784:                               ' = {1:d}, using nperseg = {1:d}'
1785:                               .format(nperseg, input_length))
1786:             nperseg = input_length
1787:         win = get_window(window, nperseg)
1788:     else:
1789:         win = np.asarray(window)
1790:         if len(win.shape) != 1:
1791:             raise ValueError('window must be 1-D')
1792:         if input_length < win.shape[-1]:
1793:             raise ValueError('window is longer than input signal')
1794:         if nperseg is None:
1795:             nperseg = win.shape[0]
1796:         elif nperseg is not None:
1797:             if nperseg != win.shape[0]:
1798:                 raise ValueError("value specified for nperseg is different from"
1799:                                  " length of window")
1800:     return win, nperseg
1801: 
1802: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_280489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Tools for spectral analysis.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280490 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_280490) is not StypyTypeError):

    if (import_280490 != 'pyd_module'):
        __import__(import_280490)
        sys_modules_280491 = sys.modules[import_280490]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_280491.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_280490)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy import fftpack' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280492 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy')

if (type(import_280492) is not StypyTypeError):

    if (import_280492 != 'pyd_module'):
        __import__(import_280492)
        sys_modules_280493 = sys.modules[import_280492]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', sys_modules_280493.module_type_store, module_type_store, ['fftpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_280493, sys_modules_280493.module_type_store, module_type_store)
    else:
        from scipy import fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', None, module_type_store, ['fftpack'], [fftpack])

else:
    # Assigning a type to the variable 'scipy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy', import_280492)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.signal import signaltools' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280494 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal')

if (type(import_280494) is not StypyTypeError):

    if (import_280494 != 'pyd_module'):
        __import__(import_280494)
        sys_modules_280495 = sys.modules[import_280494]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', sys_modules_280495.module_type_store, module_type_store, ['signaltools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_280495, sys_modules_280495.module_type_store, module_type_store)
    else:
        from scipy.signal import signaltools

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', None, module_type_store, ['signaltools'], [signaltools])

else:
    # Assigning a type to the variable 'scipy.signal' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal', import_280494)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.signal.windows import get_window' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280496 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.windows')

if (type(import_280496) is not StypyTypeError):

    if (import_280496 != 'pyd_module'):
        __import__(import_280496)
        sys_modules_280497 = sys.modules[import_280496]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.windows', sys_modules_280497.module_type_store, module_type_store, ['get_window'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_280497, sys_modules_280497.module_type_store, module_type_store)
    else:
        from scipy.signal.windows import get_window

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.windows', None, module_type_store, ['get_window'], [get_window])

else:
    # Assigning a type to the variable 'scipy.signal.windows' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal.windows', import_280496)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.signal._spectral import _lombscargle' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280498 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal._spectral')

if (type(import_280498) is not StypyTypeError):

    if (import_280498 != 'pyd_module'):
        __import__(import_280498)
        sys_modules_280499 = sys.modules[import_280498]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal._spectral', sys_modules_280499.module_type_store, module_type_store, ['_lombscargle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_280499, sys_modules_280499.module_type_store, module_type_store)
    else:
        from scipy.signal._spectral import _lombscargle

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal._spectral', None, module_type_store, ['_lombscargle'], [_lombscargle])

else:
    # Assigning a type to the variable 'scipy.signal._spectral' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal._spectral', import_280498)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280500 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._arraytools')

if (type(import_280500) is not StypyTypeError):

    if (import_280500 != 'pyd_module'):
        __import__(import_280500)
        sys_modules_280501 = sys.modules[import_280500]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._arraytools', sys_modules_280501.module_type_store, module_type_store, ['const_ext', 'even_ext', 'odd_ext', 'zero_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_280501, sys_modules_280501.module_type_store, module_type_store)
    else:
        from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._arraytools', None, module_type_store, ['const_ext', 'even_ext', 'odd_ext', 'zero_ext'], [const_ext, even_ext, odd_ext, zero_ext])

else:
    # Assigning a type to the variable 'scipy.signal._arraytools' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal._arraytools', import_280500)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy._lib.six import string_types' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_280502 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six')

if (type(import_280502) is not StypyTypeError):

    if (import_280502 != 'pyd_module'):
        __import__(import_280502)
        sys_modules_280503 = sys.modules[import_280502]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', sys_modules_280503.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_280503, sys_modules_280503.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy._lib.six', import_280502)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = ['periodogram', 'welch', 'lombscargle', 'csd', 'coherence', 'spectrogram', 'stft', 'istft', 'check_COLA']
module_type_store.set_exportable_members(['periodogram', 'welch', 'lombscargle', 'csd', 'coherence', 'spectrogram', 'stft', 'istft', 'check_COLA'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_280504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_280505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'periodogram')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280505)
# Adding element type (line 16)
str_280506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'welch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280506)
# Adding element type (line 16)
str_280507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'str', 'lombscargle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280507)
# Adding element type (line 16)
str_280508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 50), 'str', 'csd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280508)
# Adding element type (line 16)
str_280509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 57), 'str', 'coherence')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280509)
# Adding element type (line 16)
str_280510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'spectrogram')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280510)
# Adding element type (line 16)
str_280511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'stft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280511)
# Adding element type (line 16)
str_280512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', 'istft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280512)
# Adding element type (line 16)
str_280513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 43), 'str', 'check_COLA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_280504, str_280513)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_280504)

@norecursion
def lombscargle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 23)
    False_280514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'False')
    # Getting the type of 'False' (line 24)
    False_280515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'False')
    defaults = [False_280514, False_280515]
    # Create a new context for function 'lombscargle'
    module_type_store = module_type_store.open_function_context('lombscargle', 20, 0, False)
    
    # Passed parameters checking function
    lombscargle.stypy_localization = localization
    lombscargle.stypy_type_of_self = None
    lombscargle.stypy_type_store = module_type_store
    lombscargle.stypy_function_name = 'lombscargle'
    lombscargle.stypy_param_names_list = ['x', 'y', 'freqs', 'precenter', 'normalize']
    lombscargle.stypy_varargs_param_name = None
    lombscargle.stypy_kwargs_param_name = None
    lombscargle.stypy_call_defaults = defaults
    lombscargle.stypy_call_varargs = varargs
    lombscargle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lombscargle', ['x', 'y', 'freqs', 'precenter', 'normalize'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lombscargle', localization, ['x', 'y', 'freqs', 'precenter', 'normalize'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lombscargle(...)' code ##################

    str_280516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\n    lombscargle(x, y, freqs)\n\n    Computes the Lomb-Scargle periodogram.\n    \n    The Lomb-Scargle periodogram was developed by Lomb [1]_ and further\n    extended by Scargle [2]_ to find, and test the significance of weak\n    periodic signals with uneven temporal sampling.\n\n    When *normalize* is False (default) the computed periodogram\n    is unnormalized, it takes the value ``(A**2) * N/4`` for a harmonic\n    signal with amplitude A for sufficiently large N.\n\n    When *normalize* is True the computed periodogram is is normalized by\n    the residuals of the data around a constant reference model (at zero).\n\n    Input arrays should be one-dimensional and will be cast to float64.\n\n    Parameters\n    ----------\n    x : array_like\n        Sample times.\n    y : array_like\n        Measurement values.\n    freqs : array_like\n        Angular frequencies for output periodogram.\n    precenter : bool, optional\n        Pre-center amplitudes by subtracting the mean.\n    normalize : bool, optional\n        Compute normalized periodogram.\n\n    Returns\n    -------\n    pgram : array_like\n        Lomb-Scargle periodogram.\n\n    Raises\n    ------\n    ValueError\n        If the input arrays `x` and `y` do not have the same shape.\n\n    Notes\n    -----\n    This subroutine calculates the periodogram using a slightly\n    modified algorithm due to Townsend [3]_ which allows the\n    periodogram to be calculated using only a single pass through\n    the input arrays for each frequency.\n\n    The algorithm running time scales roughly as O(x * freqs) or O(N^2)\n    for a large number of samples and frequencies.\n\n    References\n    ----------\n    .. [1] N.R. Lomb "Least-squares frequency analysis of unequally spaced\n           data", Astrophysics and Space Science, vol 39, pp. 447-462, 1976\n\n    .. [2] J.D. Scargle "Studies in astronomical time series analysis. II - \n           Statistical aspects of spectral analysis of unevenly spaced data",\n           The Astrophysical Journal, vol 263, pp. 835-853, 1982\n\n    .. [3] R.H.D. Townsend, "Fast calculation of the Lomb-Scargle\n           periodogram using graphics processing units.", The Astrophysical\n           Journal Supplement Series, vol 191, pp. 247-253, 2010\n\n    Examples\n    --------\n    >>> import scipy.signal\n    >>> import matplotlib.pyplot as plt\n\n    First define some input parameters for the signal:\n\n    >>> A = 2.\n    >>> w = 1.\n    >>> phi = 0.5 * np.pi\n    >>> nin = 1000\n    >>> nout = 100000\n    >>> frac_points = 0.9 # Fraction of points to select\n     \n    Randomly select a fraction of an array with timesteps:\n\n    >>> r = np.random.rand(nin)\n    >>> x = np.linspace(0.01, 10*np.pi, nin)\n    >>> x = x[r >= frac_points]\n     \n    Plot a sine wave for the selected times:\n\n    >>> y = A * np.sin(w*x+phi)\n\n    Define the array of frequencies for which to compute the periodogram:\n    \n    >>> f = np.linspace(0.01, 10, nout)\n     \n    Calculate Lomb-Scargle periodogram:\n\n    >>> import scipy.signal as signal\n    >>> pgram = signal.lombscargle(x, y, f, normalize=True)\n\n    Now make a plot of the input data:\n\n    >>> plt.subplot(2, 1, 1)\n    >>> plt.plot(x, y, \'b+\')\n\n    Then plot the normalized periodogram:\n\n    >>> plt.subplot(2, 1, 2)\n    >>> plt.plot(f, pgram)\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to asarray(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'x' (line 135)
    x_280519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'x', False)
    # Processing the call keyword arguments (line 135)
    # Getting the type of 'np' (line 135)
    np_280520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'np', False)
    # Obtaining the member 'float64' of a type (line 135)
    float64_280521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), np_280520, 'float64')
    keyword_280522 = float64_280521
    kwargs_280523 = {'dtype': keyword_280522}
    # Getting the type of 'np' (line 135)
    np_280517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 135)
    asarray_280518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), np_280517, 'asarray')
    # Calling asarray(args, kwargs) (line 135)
    asarray_call_result_280524 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), asarray_280518, *[x_280519], **kwargs_280523)
    
    # Assigning a type to the variable 'x' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'x', asarray_call_result_280524)
    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to asarray(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'y' (line 136)
    y_280527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'y', False)
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'np' (line 136)
    np_280528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'np', False)
    # Obtaining the member 'float64' of a type (line 136)
    float64_280529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 28), np_280528, 'float64')
    keyword_280530 = float64_280529
    kwargs_280531 = {'dtype': keyword_280530}
    # Getting the type of 'np' (line 136)
    np_280525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 136)
    asarray_280526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), np_280525, 'asarray')
    # Calling asarray(args, kwargs) (line 136)
    asarray_call_result_280532 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), asarray_280526, *[y_280527], **kwargs_280531)
    
    # Assigning a type to the variable 'y' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'y', asarray_call_result_280532)
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to asarray(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'freqs' (line 137)
    freqs_280535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'freqs', False)
    # Processing the call keyword arguments (line 137)
    # Getting the type of 'np' (line 137)
    np_280536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'np', False)
    # Obtaining the member 'float64' of a type (line 137)
    float64_280537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 36), np_280536, 'float64')
    keyword_280538 = float64_280537
    kwargs_280539 = {'dtype': keyword_280538}
    # Getting the type of 'np' (line 137)
    np_280533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 137)
    asarray_280534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), np_280533, 'asarray')
    # Calling asarray(args, kwargs) (line 137)
    asarray_call_result_280540 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), asarray_280534, *[freqs_280535], **kwargs_280539)
    
    # Assigning a type to the variable 'freqs' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'freqs', asarray_call_result_280540)
    # Evaluating assert statement condition
    
    # Getting the type of 'x' (line 139)
    x_280541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'x')
    # Obtaining the member 'ndim' of a type (line 139)
    ndim_280542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), x_280541, 'ndim')
    int_280543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'int')
    # Applying the binary operator '==' (line 139)
    result_eq_280544 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), '==', ndim_280542, int_280543)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'y' (line 140)
    y_280545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'y')
    # Obtaining the member 'ndim' of a type (line 140)
    ndim_280546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), y_280545, 'ndim')
    int_280547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'int')
    # Applying the binary operator '==' (line 140)
    result_eq_280548 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '==', ndim_280546, int_280547)
    
    # Evaluating assert statement condition
    
    # Getting the type of 'freqs' (line 141)
    freqs_280549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'freqs')
    # Obtaining the member 'ndim' of a type (line 141)
    ndim_280550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 11), freqs_280549, 'ndim')
    int_280551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'int')
    # Applying the binary operator '==' (line 141)
    result_eq_280552 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '==', ndim_280550, int_280551)
    
    
    # Getting the type of 'precenter' (line 143)
    precenter_280553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'precenter')
    # Testing the type of an if condition (line 143)
    if_condition_280554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), precenter_280553)
    # Assigning a type to the variable 'if_condition_280554' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_280554', if_condition_280554)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to _lombscargle(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'x' (line 144)
    x_280556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'x', False)
    # Getting the type of 'y' (line 144)
    y_280557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'y', False)
    
    # Call to mean(...): (line 144)
    # Processing the call keyword arguments (line 144)
    kwargs_280560 = {}
    # Getting the type of 'y' (line 144)
    y_280558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'y', False)
    # Obtaining the member 'mean' of a type (line 144)
    mean_280559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 36), y_280558, 'mean')
    # Calling mean(args, kwargs) (line 144)
    mean_call_result_280561 = invoke(stypy.reporting.localization.Localization(__file__, 144, 36), mean_280559, *[], **kwargs_280560)
    
    # Applying the binary operator '-' (line 144)
    result_sub_280562 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 32), '-', y_280557, mean_call_result_280561)
    
    # Getting the type of 'freqs' (line 144)
    freqs_280563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 46), 'freqs', False)
    # Processing the call keyword arguments (line 144)
    kwargs_280564 = {}
    # Getting the type of '_lombscargle' (line 144)
    _lombscargle_280555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), '_lombscargle', False)
    # Calling _lombscargle(args, kwargs) (line 144)
    _lombscargle_call_result_280565 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), _lombscargle_280555, *[x_280556, result_sub_280562, freqs_280563], **kwargs_280564)
    
    # Assigning a type to the variable 'pgram' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'pgram', _lombscargle_call_result_280565)
    # SSA branch for the else part of an if statement (line 143)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to _lombscargle(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'x' (line 146)
    x_280567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'x', False)
    # Getting the type of 'y' (line 146)
    y_280568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 32), 'y', False)
    # Getting the type of 'freqs' (line 146)
    freqs_280569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 35), 'freqs', False)
    # Processing the call keyword arguments (line 146)
    kwargs_280570 = {}
    # Getting the type of '_lombscargle' (line 146)
    _lombscargle_280566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), '_lombscargle', False)
    # Calling _lombscargle(args, kwargs) (line 146)
    _lombscargle_call_result_280571 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), _lombscargle_280566, *[x_280567, y_280568, freqs_280569], **kwargs_280570)
    
    # Assigning a type to the variable 'pgram' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'pgram', _lombscargle_call_result_280571)
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'normalize' (line 148)
    normalize_280572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'normalize')
    # Testing the type of an if condition (line 148)
    if_condition_280573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), normalize_280572)
    # Assigning a type to the variable 'if_condition_280573' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_280573', if_condition_280573)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'pgram' (line 149)
    pgram_280574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'pgram')
    int_280575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'int')
    
    # Call to dot(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'y' (line 149)
    y_280578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'y', False)
    # Getting the type of 'y' (line 149)
    y_280579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 31), 'y', False)
    # Processing the call keyword arguments (line 149)
    kwargs_280580 = {}
    # Getting the type of 'np' (line 149)
    np_280576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'np', False)
    # Obtaining the member 'dot' of a type (line 149)
    dot_280577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), np_280576, 'dot')
    # Calling dot(args, kwargs) (line 149)
    dot_call_result_280581 = invoke(stypy.reporting.localization.Localization(__file__, 149, 21), dot_280577, *[y_280578, y_280579], **kwargs_280580)
    
    # Applying the binary operator 'div' (line 149)
    result_div_280582 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 17), 'div', int_280575, dot_call_result_280581)
    
    # Applying the binary operator '*=' (line 149)
    result_imul_280583 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 8), '*=', pgram_280574, result_div_280582)
    # Assigning a type to the variable 'pgram' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'pgram', result_imul_280583)
    
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'pgram' (line 151)
    pgram_280584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'pgram')
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type', pgram_280584)
    
    # ################# End of 'lombscargle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lombscargle' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_280585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_280585)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lombscargle'
    return stypy_return_type_280585

# Assigning a type to the variable 'lombscargle' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'lombscargle', lombscargle)

@norecursion
def periodogram(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_280586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 22), 'float')
    str_280587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'str', 'boxcar')
    # Getting the type of 'None' (line 154)
    None_280588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'None')
    str_280589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 63), 'str', 'constant')
    # Getting the type of 'True' (line 155)
    True_280590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 32), 'True')
    str_280591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 46), 'str', 'density')
    int_280592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 62), 'int')
    defaults = [float_280586, str_280587, None_280588, str_280589, True_280590, str_280591, int_280592]
    # Create a new context for function 'periodogram'
    module_type_store = module_type_store.open_function_context('periodogram', 154, 0, False)
    
    # Passed parameters checking function
    periodogram.stypy_localization = localization
    periodogram.stypy_type_of_self = None
    periodogram.stypy_type_store = module_type_store
    periodogram.stypy_function_name = 'periodogram'
    periodogram.stypy_param_names_list = ['x', 'fs', 'window', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis']
    periodogram.stypy_varargs_param_name = None
    periodogram.stypy_kwargs_param_name = None
    periodogram.stypy_call_defaults = defaults
    periodogram.stypy_call_varargs = varargs
    periodogram.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'periodogram', ['x', 'fs', 'window', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'periodogram', localization, ['x', 'fs', 'window', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'periodogram(...)' code ##################

    str_280593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'str', "\n    Estimate power spectral density using a periodogram.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to 'boxcar'.\n    nfft : int, optional\n        Length of the FFT used. If `None` the length of `x` will be\n        used.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to 'constant'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned.\n    scaling : { 'density', 'spectrum' }, optional\n        Selects between computing the power spectral density ('density')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        'density'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of `x`.\n\n    Notes\n    -----\n    .. versionadded:: 0.12.0\n\n    See Also\n    --------\n    welch: Estimate power spectral density using Welch's method\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> np.random.seed(1234)\n\n    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by\n    0.001 V**2/Hz of white noise sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2*np.sqrt(2)\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> x = amp*np.sin(2*np.pi*freq*time)\n    >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the power spectral density.\n\n    >>> f, Pxx_den = signal.periodogram(x, fs)\n    >>> plt.semilogy(f, Pxx_den)\n    >>> plt.ylim([1e-7, 1e2])\n    >>> plt.xlabel('frequency [Hz]')\n    >>> plt.ylabel('PSD [V**2/Hz]')\n    >>> plt.show()\n\n    If we average the last half of the spectral density, to exclude the\n    peak, we can recover the noise power on the signal.\n\n    >>> np.mean(Pxx_den[25000:])\n    0.00099728892368242854\n\n    Now compute and plot the power spectrum.\n\n    >>> f, Pxx_spec = signal.periodogram(x, fs, 'flattop', scaling='spectrum')\n    >>> plt.figure()\n    >>> plt.semilogy(f, np.sqrt(Pxx_spec))\n    >>> plt.ylim([1e-4, 1e1])\n    >>> plt.xlabel('frequency [Hz]')\n    >>> plt.ylabel('Linear spectrum [V RMS]')\n    >>> plt.show()\n\n    The peak height in the power spectrum is an estimate of the RMS\n    amplitude.\n\n    >>> np.sqrt(Pxx_spec.max())\n    2.0077340678640727\n\n    ")
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to asarray(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'x' (line 261)
    x_280596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'x', False)
    # Processing the call keyword arguments (line 261)
    kwargs_280597 = {}
    # Getting the type of 'np' (line 261)
    np_280594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 261)
    asarray_280595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), np_280594, 'asarray')
    # Calling asarray(args, kwargs) (line 261)
    asarray_call_result_280598 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), asarray_280595, *[x_280596], **kwargs_280597)
    
    # Assigning a type to the variable 'x' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'x', asarray_call_result_280598)
    
    
    # Getting the type of 'x' (line 263)
    x_280599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 7), 'x')
    # Obtaining the member 'size' of a type (line 263)
    size_280600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 7), x_280599, 'size')
    int_280601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 17), 'int')
    # Applying the binary operator '==' (line 263)
    result_eq_280602 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 7), '==', size_280600, int_280601)
    
    # Testing the type of an if condition (line 263)
    if_condition_280603 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 4), result_eq_280602)
    # Assigning a type to the variable 'if_condition_280603' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'if_condition_280603', if_condition_280603)
    # SSA begins for if statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_280604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    
    # Call to empty(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_280607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'x', False)
    # Obtaining the member 'shape' of a type (line 264)
    shape_280608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), x_280607, 'shape')
    # Processing the call keyword arguments (line 264)
    kwargs_280609 = {}
    # Getting the type of 'np' (line 264)
    np_280605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'np', False)
    # Obtaining the member 'empty' of a type (line 264)
    empty_280606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), np_280605, 'empty')
    # Calling empty(args, kwargs) (line 264)
    empty_call_result_280610 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), empty_280606, *[shape_280608], **kwargs_280609)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 15), tuple_280604, empty_call_result_280610)
    # Adding element type (line 264)
    
    # Call to empty(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'x' (line 264)
    x_280613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'x', False)
    # Obtaining the member 'shape' of a type (line 264)
    shape_280614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 43), x_280613, 'shape')
    # Processing the call keyword arguments (line 264)
    kwargs_280615 = {}
    # Getting the type of 'np' (line 264)
    np_280611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'np', False)
    # Obtaining the member 'empty' of a type (line 264)
    empty_280612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 34), np_280611, 'empty')
    # Calling empty(args, kwargs) (line 264)
    empty_call_result_280616 = invoke(stypy.reporting.localization.Localization(__file__, 264, 34), empty_280612, *[shape_280614], **kwargs_280615)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 15), tuple_280604, empty_call_result_280616)
    
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type', tuple_280604)
    # SSA join for if statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 266)
    # Getting the type of 'window' (line 266)
    window_280617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 7), 'window')
    # Getting the type of 'None' (line 266)
    None_280618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'None')
    
    (may_be_280619, more_types_in_union_280620) = may_be_none(window_280617, None_280618)

    if may_be_280619:

        if more_types_in_union_280620:
            # Runtime conditional SSA (line 266)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 267):
        
        # Assigning a Str to a Name (line 267):
        str_280621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 17), 'str', 'boxcar')
        # Assigning a type to the variable 'window' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'window', str_280621)

        if more_types_in_union_280620:
            # SSA join for if statement (line 266)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 269)
    # Getting the type of 'nfft' (line 269)
    nfft_280622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 7), 'nfft')
    # Getting the type of 'None' (line 269)
    None_280623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'None')
    
    (may_be_280624, more_types_in_union_280625) = may_be_none(nfft_280622, None_280623)

    if may_be_280624:

        if more_types_in_union_280625:
            # Runtime conditional SSA (line 269)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 270):
        
        # Assigning a Subscript to a Name (line 270):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 270)
        axis_280626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'axis')
        # Getting the type of 'x' (line 270)
        x_280627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), 'x')
        # Obtaining the member 'shape' of a type (line 270)
        shape_280628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 18), x_280627, 'shape')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___280629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 18), shape_280628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_280630 = invoke(stypy.reporting.localization.Localization(__file__, 270, 18), getitem___280629, axis_280626)
        
        # Assigning a type to the variable 'nperseg' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'nperseg', subscript_call_result_280630)

        if more_types_in_union_280625:
            # Runtime conditional SSA for else branch (line 269)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_280624) or more_types_in_union_280625):
        
        
        # Getting the type of 'nfft' (line 271)
        nfft_280631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'nfft')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 271)
        axis_280632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'axis')
        # Getting the type of 'x' (line 271)
        x_280633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 17), 'x')
        # Obtaining the member 'shape' of a type (line 271)
        shape_280634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 17), x_280633, 'shape')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___280635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 17), shape_280634, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_280636 = invoke(stypy.reporting.localization.Localization(__file__, 271, 17), getitem___280635, axis_280632)
        
        # Applying the binary operator '==' (line 271)
        result_eq_280637 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 9), '==', nfft_280631, subscript_call_result_280636)
        
        # Testing the type of an if condition (line 271)
        if_condition_280638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 9), result_eq_280637)
        # Assigning a type to the variable 'if_condition_280638' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 9), 'if_condition_280638', if_condition_280638)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 272):
        
        # Assigning a Name to a Name (line 272):
        # Getting the type of 'nfft' (line 272)
        nfft_280639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 18), 'nfft')
        # Assigning a type to the variable 'nperseg' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'nperseg', nfft_280639)
        # SSA branch for the else part of an if statement (line 271)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'nfft' (line 273)
        nfft_280640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'nfft')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 273)
        axis_280641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'axis')
        # Getting the type of 'x' (line 273)
        x_280642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'x')
        # Obtaining the member 'shape' of a type (line 273)
        shape_280643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), x_280642, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___280644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), shape_280643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_280645 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), getitem___280644, axis_280641)
        
        # Applying the binary operator '>' (line 273)
        result_gt_280646 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 9), '>', nfft_280640, subscript_call_result_280645)
        
        # Testing the type of an if condition (line 273)
        if_condition_280647 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 9), result_gt_280646)
        # Assigning a type to the variable 'if_condition_280647' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'if_condition_280647', if_condition_280647)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 274):
        
        # Assigning a Subscript to a Name (line 274):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 274)
        axis_280648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 26), 'axis')
        # Getting the type of 'x' (line 274)
        x_280649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 18), 'x')
        # Obtaining the member 'shape' of a type (line 274)
        shape_280650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 18), x_280649, 'shape')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___280651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 18), shape_280650, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_280652 = invoke(stypy.reporting.localization.Localization(__file__, 274, 18), getitem___280651, axis_280648)
        
        # Assigning a type to the variable 'nperseg' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'nperseg', subscript_call_result_280652)
        # SSA branch for the else part of an if statement (line 273)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'nfft' (line 275)
        nfft_280653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'nfft')
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 275)
        axis_280654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'axis')
        # Getting the type of 'x' (line 275)
        x_280655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'x')
        # Obtaining the member 'shape' of a type (line 275)
        shape_280656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), x_280655, 'shape')
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___280657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), shape_280656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 275)
        subscript_call_result_280658 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), getitem___280657, axis_280654)
        
        # Applying the binary operator '<' (line 275)
        result_lt_280659 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 9), '<', nfft_280653, subscript_call_result_280658)
        
        # Testing the type of an if condition (line 275)
        if_condition_280660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 9), result_lt_280659)
        # Assigning a type to the variable 'if_condition_280660' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 9), 'if_condition_280660', if_condition_280660)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 276):
        
        # Assigning a BinOp to a Name (line 276):
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_280661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        
        # Obtaining the type of the subscript
        slice_280662 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 276, 13), None, None, None)
        # Getting the type of 'np' (line 276)
        np_280663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'np')
        # Obtaining the member 's_' of a type (line 276)
        s__280664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 13), np_280663, 's_')
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___280665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 13), s__280664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_280666 = invoke(stypy.reporting.localization.Localization(__file__, 276, 13), getitem___280665, slice_280662)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 12), list_280661, subscript_call_result_280666)
        
        
        # Call to len(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'x' (line 276)
        x_280668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'x', False)
        # Obtaining the member 'shape' of a type (line 276)
        shape_280669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 27), x_280668, 'shape')
        # Processing the call keyword arguments (line 276)
        kwargs_280670 = {}
        # Getting the type of 'len' (line 276)
        len_280667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'len', False)
        # Calling len(args, kwargs) (line 276)
        len_call_result_280671 = invoke(stypy.reporting.localization.Localization(__file__, 276, 23), len_280667, *[shape_280669], **kwargs_280670)
        
        # Applying the binary operator '*' (line 276)
        result_mul_280672 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '*', list_280661, len_call_result_280671)
        
        # Assigning a type to the variable 's' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 's', result_mul_280672)
        
        # Assigning a Subscript to a Subscript (line 277):
        
        # Assigning a Subscript to a Subscript (line 277):
        
        # Obtaining the type of the subscript
        # Getting the type of 'nfft' (line 277)
        nfft_280673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'nfft')
        slice_280674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 277, 18), None, nfft_280673, None)
        # Getting the type of 'np' (line 277)
        np_280675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 18), 'np')
        # Obtaining the member 's_' of a type (line 277)
        s__280676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 18), np_280675, 's_')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___280677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 18), s__280676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_280678 = invoke(stypy.reporting.localization.Localization(__file__, 277, 18), getitem___280677, slice_280674)
        
        # Getting the type of 's' (line 277)
        s_280679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 's')
        # Getting the type of 'axis' (line 277)
        axis_280680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 10), 'axis')
        # Storing an element on a container (line 277)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 8), s_280679, (axis_280680, subscript_call_result_280678))
        
        # Assigning a Subscript to a Name (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 278)
        s_280681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 's')
        # Getting the type of 'x' (line 278)
        x_280682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'x')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___280683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), x_280682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_280684 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), getitem___280683, s_280681)
        
        # Assigning a type to the variable 'x' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'x', subscript_call_result_280684)
        
        # Assigning a Name to a Name (line 279):
        
        # Assigning a Name to a Name (line 279):
        # Getting the type of 'nfft' (line 279)
        nfft_280685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'nfft')
        # Assigning a type to the variable 'nperseg' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'nperseg', nfft_280685)
        
        # Assigning a Name to a Name (line 280):
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'None' (line 280)
        None_280686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'None')
        # Assigning a type to the variable 'nfft' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'nfft', None_280686)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_280624 and more_types_in_union_280625):
            # SSA join for if statement (line 269)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to welch(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'x' (line 282)
    x_280688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'x', False)
    # Getting the type of 'fs' (line 282)
    fs_280689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'fs', False)
    # Getting the type of 'window' (line 282)
    window_280690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'window', False)
    # Getting the type of 'nperseg' (line 282)
    nperseg_280691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'nperseg', False)
    int_280692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 41), 'int')
    # Getting the type of 'nfft' (line 282)
    nfft_280693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 44), 'nfft', False)
    # Getting the type of 'detrend' (line 282)
    detrend_280694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 50), 'detrend', False)
    # Getting the type of 'return_onesided' (line 282)
    return_onesided_280695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 59), 'return_onesided', False)
    # Getting the type of 'scaling' (line 283)
    scaling_280696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 17), 'scaling', False)
    # Getting the type of 'axis' (line 283)
    axis_280697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'axis', False)
    # Processing the call keyword arguments (line 282)
    kwargs_280698 = {}
    # Getting the type of 'welch' (line 282)
    welch_280687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'welch', False)
    # Calling welch(args, kwargs) (line 282)
    welch_call_result_280699 = invoke(stypy.reporting.localization.Localization(__file__, 282, 11), welch_280687, *[x_280688, fs_280689, window_280690, nperseg_280691, int_280692, nfft_280693, detrend_280694, return_onesided_280695, scaling_280696, axis_280697], **kwargs_280698)
    
    # Assigning a type to the variable 'stypy_return_type' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type', welch_call_result_280699)
    
    # ################# End of 'periodogram(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'periodogram' in the type store
    # Getting the type of 'stypy_return_type' (line 154)
    stypy_return_type_280700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_280700)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'periodogram'
    return stypy_return_type_280700

# Assigning a type to the variable 'periodogram' (line 154)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 0), 'periodogram', periodogram)

@norecursion
def welch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_280701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'float')
    str_280702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 28), 'str', 'hann')
    # Getting the type of 'None' (line 286)
    None_280703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 44), 'None')
    # Getting the type of 'None' (line 286)
    None_280704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 59), 'None')
    # Getting the type of 'None' (line 286)
    None_280705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 70), 'None')
    str_280706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 18), 'str', 'constant')
    # Getting the type of 'True' (line 287)
    True_280707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 46), 'True')
    str_280708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 60), 'str', 'density')
    int_280709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'int')
    defaults = [float_280701, str_280702, None_280703, None_280704, None_280705, str_280706, True_280707, str_280708, int_280709]
    # Create a new context for function 'welch'
    module_type_store = module_type_store.open_function_context('welch', 286, 0, False)
    
    # Passed parameters checking function
    welch.stypy_localization = localization
    welch.stypy_type_of_self = None
    welch.stypy_type_store = module_type_store
    welch.stypy_function_name = 'welch'
    welch.stypy_param_names_list = ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis']
    welch.stypy_varargs_param_name = None
    welch.stypy_kwargs_param_name = None
    welch.stypy_call_defaults = defaults
    welch.stypy_call_varargs = varargs
    welch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'welch', ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'welch', localization, ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'welch(...)' code ##################

    str_280710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, (-1)), 'str', '\n    Estimate power spectral density using Welch\'s method.\n\n    Welch\'s method [1]_ computes an estimate of the power spectral\n    density by dividing the data into overlapping segments, computing a\n    modified periodogram for each segment and averaging the\n    periodograms.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Pxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Pxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'\n    axis : int, optional\n        Axis along which the periodogram is computed; the default is\n        over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxx : ndarray\n        Power spectral density or power spectrum of x.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    If `noverlap` is 0, this method is equivalent to Bartlett\'s method\n    [2]_.\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",\n           Biometrika, vol. 37, pp. 1-16, 1950.\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> np.random.seed(1234)\n\n    Generate a test signal, a 2 Vrms sine wave at 1234 Hz, corrupted by\n    0.001 V**2/Hz of white noise sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2*np.sqrt(2)\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> x = amp*np.sin(2*np.pi*freq*time)\n    >>> x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the power spectral density.\n\n    >>> f, Pxx_den = signal.welch(x, fs, nperseg=1024)\n    >>> plt.semilogy(f, Pxx_den)\n    >>> plt.ylim([0.5e-3, 1])\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'PSD [V**2/Hz]\')\n    >>> plt.show()\n\n    If we average the last half of the spectral density, to exclude the\n    peak, we can recover the noise power on the signal.\n\n    >>> np.mean(Pxx_den[256:])\n    0.0009924865443739191\n\n    Now compute and plot the power spectrum.\n\n    >>> f, Pxx_spec = signal.welch(x, fs, \'flattop\', 1024, scaling=\'spectrum\')\n    >>> plt.figure()\n    >>> plt.semilogy(f, np.sqrt(Pxx_spec))\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'Linear spectrum [V RMS]\')\n    >>> plt.show()\n\n    The peak height in the power spectrum is an estimate of the RMS\n    amplitude.\n\n    >>> np.sqrt(Pxx_spec.max())\n    2.0077340678640727\n\n    ')
    
    # Assigning a Call to a Tuple (line 424):
    
    # Assigning a Subscript to a Name (line 424):
    
    # Obtaining the type of the subscript
    int_280711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'int')
    
    # Call to csd(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'x' (line 424)
    x_280713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'x', False)
    # Getting the type of 'x' (line 424)
    x_280714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'x', False)
    # Getting the type of 'fs' (line 424)
    fs_280715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'fs', False)
    # Getting the type of 'window' (line 424)
    window_280716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 31), 'window', False)
    # Getting the type of 'nperseg' (line 424)
    nperseg_280717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 39), 'nperseg', False)
    # Getting the type of 'noverlap' (line 424)
    noverlap_280718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 48), 'noverlap', False)
    # Getting the type of 'nfft' (line 424)
    nfft_280719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 58), 'nfft', False)
    # Getting the type of 'detrend' (line 424)
    detrend_280720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 64), 'detrend', False)
    # Getting the type of 'return_onesided' (line 425)
    return_onesided_280721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 21), 'return_onesided', False)
    # Getting the type of 'scaling' (line 425)
    scaling_280722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 38), 'scaling', False)
    # Getting the type of 'axis' (line 425)
    axis_280723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 47), 'axis', False)
    # Processing the call keyword arguments (line 424)
    kwargs_280724 = {}
    # Getting the type of 'csd' (line 424)
    csd_280712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'csd', False)
    # Calling csd(args, kwargs) (line 424)
    csd_call_result_280725 = invoke(stypy.reporting.localization.Localization(__file__, 424, 17), csd_280712, *[x_280713, x_280714, fs_280715, window_280716, nperseg_280717, noverlap_280718, nfft_280719, detrend_280720, return_onesided_280721, scaling_280722, axis_280723], **kwargs_280724)
    
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___280726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 4), csd_call_result_280725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 424)
    subscript_call_result_280727 = invoke(stypy.reporting.localization.Localization(__file__, 424, 4), getitem___280726, int_280711)
    
    # Assigning a type to the variable 'tuple_var_assignment_280465' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'tuple_var_assignment_280465', subscript_call_result_280727)
    
    # Assigning a Subscript to a Name (line 424):
    
    # Obtaining the type of the subscript
    int_280728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 4), 'int')
    
    # Call to csd(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'x' (line 424)
    x_280730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'x', False)
    # Getting the type of 'x' (line 424)
    x_280731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'x', False)
    # Getting the type of 'fs' (line 424)
    fs_280732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'fs', False)
    # Getting the type of 'window' (line 424)
    window_280733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 31), 'window', False)
    # Getting the type of 'nperseg' (line 424)
    nperseg_280734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 39), 'nperseg', False)
    # Getting the type of 'noverlap' (line 424)
    noverlap_280735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 48), 'noverlap', False)
    # Getting the type of 'nfft' (line 424)
    nfft_280736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 58), 'nfft', False)
    # Getting the type of 'detrend' (line 424)
    detrend_280737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 64), 'detrend', False)
    # Getting the type of 'return_onesided' (line 425)
    return_onesided_280738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 21), 'return_onesided', False)
    # Getting the type of 'scaling' (line 425)
    scaling_280739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 38), 'scaling', False)
    # Getting the type of 'axis' (line 425)
    axis_280740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 47), 'axis', False)
    # Processing the call keyword arguments (line 424)
    kwargs_280741 = {}
    # Getting the type of 'csd' (line 424)
    csd_280729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 17), 'csd', False)
    # Calling csd(args, kwargs) (line 424)
    csd_call_result_280742 = invoke(stypy.reporting.localization.Localization(__file__, 424, 17), csd_280729, *[x_280730, x_280731, fs_280732, window_280733, nperseg_280734, noverlap_280735, nfft_280736, detrend_280737, return_onesided_280738, scaling_280739, axis_280740], **kwargs_280741)
    
    # Obtaining the member '__getitem__' of a type (line 424)
    getitem___280743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 4), csd_call_result_280742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 424)
    subscript_call_result_280744 = invoke(stypy.reporting.localization.Localization(__file__, 424, 4), getitem___280743, int_280728)
    
    # Assigning a type to the variable 'tuple_var_assignment_280466' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'tuple_var_assignment_280466', subscript_call_result_280744)
    
    # Assigning a Name to a Name (line 424):
    # Getting the type of 'tuple_var_assignment_280465' (line 424)
    tuple_var_assignment_280465_280745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'tuple_var_assignment_280465')
    # Assigning a type to the variable 'freqs' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'freqs', tuple_var_assignment_280465_280745)
    
    # Assigning a Name to a Name (line 424):
    # Getting the type of 'tuple_var_assignment_280466' (line 424)
    tuple_var_assignment_280466_280746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'tuple_var_assignment_280466')
    # Assigning a type to the variable 'Pxx' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'Pxx', tuple_var_assignment_280466_280746)
    
    # Obtaining an instance of the builtin type 'tuple' (line 427)
    tuple_280747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 427)
    # Adding element type (line 427)
    # Getting the type of 'freqs' (line 427)
    freqs_280748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 11), tuple_280747, freqs_280748)
    # Adding element type (line 427)
    # Getting the type of 'Pxx' (line 427)
    Pxx_280749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 18), 'Pxx')
    # Obtaining the member 'real' of a type (line 427)
    real_280750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 18), Pxx_280749, 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 11), tuple_280747, real_280750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'stypy_return_type', tuple_280747)
    
    # ################# End of 'welch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'welch' in the type store
    # Getting the type of 'stypy_return_type' (line 286)
    stypy_return_type_280751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_280751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'welch'
    return stypy_return_type_280751

# Assigning a type to the variable 'welch' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'welch', welch)

@norecursion
def csd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_280752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 17), 'float')
    str_280753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 29), 'str', 'hann')
    # Getting the type of 'None' (line 430)
    None_280754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 45), 'None')
    # Getting the type of 'None' (line 430)
    None_280755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 60), 'None')
    # Getting the type of 'None' (line 430)
    None_280756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 71), 'None')
    str_280757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 16), 'str', 'constant')
    # Getting the type of 'True' (line 431)
    True_280758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 44), 'True')
    str_280759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 58), 'str', 'density')
    int_280760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 74), 'int')
    defaults = [float_280752, str_280753, None_280754, None_280755, None_280756, str_280757, True_280758, str_280759, int_280760]
    # Create a new context for function 'csd'
    module_type_store = module_type_store.open_function_context('csd', 430, 0, False)
    
    # Passed parameters checking function
    csd.stypy_localization = localization
    csd.stypy_type_of_self = None
    csd.stypy_type_store = module_type_store
    csd.stypy_function_name = 'csd'
    csd.stypy_param_names_list = ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis']
    csd.stypy_varargs_param_name = None
    csd.stypy_kwargs_param_name = None
    csd.stypy_call_defaults = defaults
    csd.stypy_call_varargs = varargs
    csd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'csd', ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'csd', localization, ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'csd(...)' code ##################

    str_280761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, (-1)), 'str', '\n    Estimate the cross power spectral density, Pxy, using Welch\'s\n    method.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the cross spectral density (\'density\')\n        where `Pxy` has units of V**2/Hz and computing the cross spectrum\n        (\'spectrum\') where `Pxy` has units of V**2, if `x` and `y` are\n        measured in V and `fs` is measured in Hz. Defaults to \'density\'\n    axis : int, optional\n        Axis along which the CSD is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Pxy : ndarray\n        Cross spectral density or cross power spectrum of x,y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method. [Equivalent to\n           csd(x,x)]\n    coherence: Magnitude squared coherence by Welch\'s method.\n\n    Notes\n    --------\n    By convention, Pxy is computed with the conjugate FFT of X\n    multiplied by the FFT of Y.\n\n    If the input series differ in length, the shorter series will be\n    zero-padded to match.\n\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] Rabiner, Lawrence R., and B. Gold. "Theory and Application of\n           Digital Signal Processing" Prentice-Hall, pp. 414-419, 1975\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    Generate two test signals with some common features.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 20\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> b, a = signal.butter(2, 0.25, \'low\')\n    >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> y = signal.lfilter(b, a, x)\n    >>> x += amp*np.sin(2*np.pi*freq*time)\n    >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the magnitude of the cross spectral density.\n\n    >>> f, Pxy = signal.csd(x, y, fs, nperseg=1024)\n    >>> plt.semilogy(f, np.abs(Pxy))\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'CSD [V**2/Hz]\')\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Tuple (line 549):
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_280762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    
    # Call to _spectral_helper(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'x' (line 549)
    x_280764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 37), 'x', False)
    # Getting the type of 'y' (line 549)
    y_280765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 40), 'y', False)
    # Getting the type of 'fs' (line 549)
    fs_280766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 43), 'fs', False)
    # Getting the type of 'window' (line 549)
    window_280767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 47), 'window', False)
    # Getting the type of 'nperseg' (line 549)
    nperseg_280768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 55), 'nperseg', False)
    # Getting the type of 'noverlap' (line 549)
    noverlap_280769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 64), 'noverlap', False)
    # Getting the type of 'nfft' (line 549)
    nfft_280770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 74), 'nfft', False)
    # Getting the type of 'detrend' (line 550)
    detrend_280771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'detrend', False)
    # Getting the type of 'return_onesided' (line 550)
    return_onesided_280772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'return_onesided', False)
    # Getting the type of 'scaling' (line 550)
    scaling_280773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 63), 'scaling', False)
    # Getting the type of 'axis' (line 550)
    axis_280774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 72), 'axis', False)
    # Processing the call keyword arguments (line 549)
    str_280775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 42), 'str', 'psd')
    keyword_280776 = str_280775
    kwargs_280777 = {'mode': keyword_280776}
    # Getting the type of '_spectral_helper' (line 549)
    _spectral_helper_280763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 549)
    _spectral_helper_call_result_280778 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), _spectral_helper_280763, *[x_280764, y_280765, fs_280766, window_280767, nperseg_280768, noverlap_280769, nfft_280770, detrend_280771, return_onesided_280772, scaling_280773, axis_280774], **kwargs_280777)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___280779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), _spectral_helper_call_result_280778, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_280780 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___280779, int_280762)
    
    # Assigning a type to the variable 'tuple_var_assignment_280467' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280467', subscript_call_result_280780)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_280781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    
    # Call to _spectral_helper(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'x' (line 549)
    x_280783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 37), 'x', False)
    # Getting the type of 'y' (line 549)
    y_280784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 40), 'y', False)
    # Getting the type of 'fs' (line 549)
    fs_280785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 43), 'fs', False)
    # Getting the type of 'window' (line 549)
    window_280786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 47), 'window', False)
    # Getting the type of 'nperseg' (line 549)
    nperseg_280787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 55), 'nperseg', False)
    # Getting the type of 'noverlap' (line 549)
    noverlap_280788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 64), 'noverlap', False)
    # Getting the type of 'nfft' (line 549)
    nfft_280789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 74), 'nfft', False)
    # Getting the type of 'detrend' (line 550)
    detrend_280790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'detrend', False)
    # Getting the type of 'return_onesided' (line 550)
    return_onesided_280791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'return_onesided', False)
    # Getting the type of 'scaling' (line 550)
    scaling_280792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 63), 'scaling', False)
    # Getting the type of 'axis' (line 550)
    axis_280793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 72), 'axis', False)
    # Processing the call keyword arguments (line 549)
    str_280794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 42), 'str', 'psd')
    keyword_280795 = str_280794
    kwargs_280796 = {'mode': keyword_280795}
    # Getting the type of '_spectral_helper' (line 549)
    _spectral_helper_280782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 549)
    _spectral_helper_call_result_280797 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), _spectral_helper_280782, *[x_280783, y_280784, fs_280785, window_280786, nperseg_280787, noverlap_280788, nfft_280789, detrend_280790, return_onesided_280791, scaling_280792, axis_280793], **kwargs_280796)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___280798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), _spectral_helper_call_result_280797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_280799 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___280798, int_280781)
    
    # Assigning a type to the variable 'tuple_var_assignment_280468' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280468', subscript_call_result_280799)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_280800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    
    # Call to _spectral_helper(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'x' (line 549)
    x_280802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 37), 'x', False)
    # Getting the type of 'y' (line 549)
    y_280803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 40), 'y', False)
    # Getting the type of 'fs' (line 549)
    fs_280804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 43), 'fs', False)
    # Getting the type of 'window' (line 549)
    window_280805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 47), 'window', False)
    # Getting the type of 'nperseg' (line 549)
    nperseg_280806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 55), 'nperseg', False)
    # Getting the type of 'noverlap' (line 549)
    noverlap_280807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 64), 'noverlap', False)
    # Getting the type of 'nfft' (line 549)
    nfft_280808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 74), 'nfft', False)
    # Getting the type of 'detrend' (line 550)
    detrend_280809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 37), 'detrend', False)
    # Getting the type of 'return_onesided' (line 550)
    return_onesided_280810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'return_onesided', False)
    # Getting the type of 'scaling' (line 550)
    scaling_280811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 63), 'scaling', False)
    # Getting the type of 'axis' (line 550)
    axis_280812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 72), 'axis', False)
    # Processing the call keyword arguments (line 549)
    str_280813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 42), 'str', 'psd')
    keyword_280814 = str_280813
    kwargs_280815 = {'mode': keyword_280814}
    # Getting the type of '_spectral_helper' (line 549)
    _spectral_helper_280801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 549)
    _spectral_helper_call_result_280816 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), _spectral_helper_280801, *[x_280802, y_280803, fs_280804, window_280805, nperseg_280806, noverlap_280807, nfft_280808, detrend_280809, return_onesided_280810, scaling_280811, axis_280812], **kwargs_280815)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___280817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), _spectral_helper_call_result_280816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_280818 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___280817, int_280800)
    
    # Assigning a type to the variable 'tuple_var_assignment_280469' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280469', subscript_call_result_280818)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_280467' (line 549)
    tuple_var_assignment_280467_280819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280467')
    # Assigning a type to the variable 'freqs' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'freqs', tuple_var_assignment_280467_280819)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_280468' (line 549)
    tuple_var_assignment_280468_280820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280468')
    # Assigning a type to the variable '_' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 11), '_', tuple_var_assignment_280468_280820)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_280469' (line 549)
    tuple_var_assignment_280469_280821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_280469')
    # Assigning a type to the variable 'Pxy' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 14), 'Pxy', tuple_var_assignment_280469_280821)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'Pxy' (line 554)
    Pxy_280823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 11), 'Pxy', False)
    # Obtaining the member 'shape' of a type (line 554)
    shape_280824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 11), Pxy_280823, 'shape')
    # Processing the call keyword arguments (line 554)
    kwargs_280825 = {}
    # Getting the type of 'len' (line 554)
    len_280822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 7), 'len', False)
    # Calling len(args, kwargs) (line 554)
    len_call_result_280826 = invoke(stypy.reporting.localization.Localization(__file__, 554, 7), len_280822, *[shape_280824], **kwargs_280825)
    
    int_280827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 25), 'int')
    # Applying the binary operator '>=' (line 554)
    result_ge_280828 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 7), '>=', len_call_result_280826, int_280827)
    
    
    # Getting the type of 'Pxy' (line 554)
    Pxy_280829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 31), 'Pxy')
    # Obtaining the member 'size' of a type (line 554)
    size_280830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 31), Pxy_280829, 'size')
    int_280831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 42), 'int')
    # Applying the binary operator '>' (line 554)
    result_gt_280832 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 31), '>', size_280830, int_280831)
    
    # Applying the binary operator 'and' (line 554)
    result_and_keyword_280833 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 7), 'and', result_ge_280828, result_gt_280832)
    
    # Testing the type of an if condition (line 554)
    if_condition_280834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), result_and_keyword_280833)
    # Assigning a type to the variable 'if_condition_280834' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_280834', if_condition_280834)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_280835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 21), 'int')
    # Getting the type of 'Pxy' (line 555)
    Pxy_280836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'Pxy')
    # Obtaining the member 'shape' of a type (line 555)
    shape_280837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 11), Pxy_280836, 'shape')
    # Obtaining the member '__getitem__' of a type (line 555)
    getitem___280838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 11), shape_280837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 555)
    subscript_call_result_280839 = invoke(stypy.reporting.localization.Localization(__file__, 555, 11), getitem___280838, int_280835)
    
    int_280840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 27), 'int')
    # Applying the binary operator '>' (line 555)
    result_gt_280841 = python_operator(stypy.reporting.localization.Localization(__file__, 555, 11), '>', subscript_call_result_280839, int_280840)
    
    # Testing the type of an if condition (line 555)
    if_condition_280842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 555, 8), result_gt_280841)
    # Assigning a type to the variable 'if_condition_280842' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'if_condition_280842', if_condition_280842)
    # SSA begins for if statement (line 555)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 556):
    
    # Assigning a Call to a Name (line 556):
    
    # Call to mean(...): (line 556)
    # Processing the call keyword arguments (line 556)
    int_280845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 32), 'int')
    keyword_280846 = int_280845
    kwargs_280847 = {'axis': keyword_280846}
    # Getting the type of 'Pxy' (line 556)
    Pxy_280843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 18), 'Pxy', False)
    # Obtaining the member 'mean' of a type (line 556)
    mean_280844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 18), Pxy_280843, 'mean')
    # Calling mean(args, kwargs) (line 556)
    mean_call_result_280848 = invoke(stypy.reporting.localization.Localization(__file__, 556, 18), mean_280844, *[], **kwargs_280847)
    
    # Assigning a type to the variable 'Pxy' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'Pxy', mean_call_result_280848)
    # SSA branch for the else part of an if statement (line 555)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to reshape(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'Pxy' (line 558)
    Pxy_280851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 29), 'Pxy', False)
    
    # Obtaining the type of the subscript
    int_280852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 45), 'int')
    slice_280853 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 558, 34), None, int_280852, None)
    # Getting the type of 'Pxy' (line 558)
    Pxy_280854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 34), 'Pxy', False)
    # Obtaining the member 'shape' of a type (line 558)
    shape_280855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 34), Pxy_280854, 'shape')
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___280856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 34), shape_280855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_280857 = invoke(stypy.reporting.localization.Localization(__file__, 558, 34), getitem___280856, slice_280853)
    
    # Processing the call keyword arguments (line 558)
    kwargs_280858 = {}
    # Getting the type of 'np' (line 558)
    np_280849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'np', False)
    # Obtaining the member 'reshape' of a type (line 558)
    reshape_280850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 18), np_280849, 'reshape')
    # Calling reshape(args, kwargs) (line 558)
    reshape_call_result_280859 = invoke(stypy.reporting.localization.Localization(__file__, 558, 18), reshape_280850, *[Pxy_280851, subscript_call_result_280857], **kwargs_280858)
    
    # Assigning a type to the variable 'Pxy' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'Pxy', reshape_call_result_280859)
    # SSA join for if statement (line 555)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 560)
    tuple_280860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 560)
    # Adding element type (line 560)
    # Getting the type of 'freqs' (line 560)
    freqs_280861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 11), tuple_280860, freqs_280861)
    # Adding element type (line 560)
    # Getting the type of 'Pxy' (line 560)
    Pxy_280862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 18), 'Pxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 560, 11), tuple_280860, Pxy_280862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type', tuple_280860)
    
    # ################# End of 'csd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'csd' in the type store
    # Getting the type of 'stypy_return_type' (line 430)
    stypy_return_type_280863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_280863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'csd'
    return stypy_return_type_280863

# Assigning a type to the variable 'csd' (line 430)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'csd', csd)

@norecursion
def spectrogram(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_280864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 22), 'float')
    
    # Obtaining an instance of the builtin type 'tuple' (line 563)
    tuple_280865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 563)
    # Adding element type (line 563)
    str_280866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 35), 'str', 'tukey')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 35), tuple_280865, str_280866)
    # Adding element type (line 563)
    float_280867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 35), tuple_280865, float_280867)
    
    # Getting the type of 'None' (line 563)
    None_280868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 57), 'None')
    # Getting the type of 'None' (line 563)
    None_280869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 72), 'None')
    # Getting the type of 'None' (line 564)
    None_280870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 21), 'None')
    str_280871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 35), 'str', 'constant')
    # Getting the type of 'True' (line 564)
    True_280872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 63), 'True')
    str_280873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 24), 'str', 'density')
    int_280874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 40), 'int')
    str_280875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 49), 'str', 'psd')
    defaults = [float_280864, tuple_280865, None_280868, None_280869, None_280870, str_280871, True_280872, str_280873, int_280874, str_280875]
    # Create a new context for function 'spectrogram'
    module_type_store = module_type_store.open_function_context('spectrogram', 563, 0, False)
    
    # Passed parameters checking function
    spectrogram.stypy_localization = localization
    spectrogram.stypy_type_of_self = None
    spectrogram.stypy_type_store = module_type_store
    spectrogram.stypy_function_name = 'spectrogram'
    spectrogram.stypy_param_names_list = ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode']
    spectrogram.stypy_varargs_param_name = None
    spectrogram.stypy_kwargs_param_name = None
    spectrogram.stypy_call_defaults = defaults
    spectrogram.stypy_call_varargs = varargs
    spectrogram.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spectrogram', ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spectrogram', localization, ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spectrogram(...)' code ##################

    str_280876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, (-1)), 'str', '\n    Compute a spectrogram with consecutive Fourier transforms.\n\n    Spectrograms can be used as a way of visualizing the change of a\n    nonstationary signal\'s frequency content over time.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n        Defaults to a Tukey window with shape parameter of 0.25.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 8``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the power spectral density (\'density\')\n        where `Sxx` has units of V**2/Hz and computing the power\n        spectrum (\'spectrum\') where `Sxx` has units of V**2, if `x`\n        is measured in V and `fs` is measured in Hz. Defaults to\n        \'density\'.\n    axis : int, optional\n        Axis along which the spectrogram is computed; the default is over\n        the last axis (i.e. ``axis=-1``).\n    mode : str, optional\n        Defines what kind of return values are expected. Options are\n        [\'psd\', \'complex\', \'magnitude\', \'angle\', \'phase\']. \'complex\' is\n        equivalent to the output of `stft` with no padding or boundary\n        extension. \'magnitude\' returns the absolute magnitude of the\n        STFT. \'angle\' and \'phase\' return the complex angle of the STFT,\n        with and without unwrapping, respectively.\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Sxx : ndarray\n        Spectrogram of x. By default, the last axis of Sxx corresponds\n        to the segment times.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n\n    Notes\n    -----\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. In contrast to welch\'s method, where the\n    entire data stream is averaged over, one may wish to use a smaller\n    overlap (or perhaps none at all) when computing a spectrogram, to\n    maintain some statistical independence between individual segments.\n    It is for this reason that the default window is a Tukey window with\n    1/8th of a window\'s length overlap at each end.\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> mod = 500*np.cos(2*np.pi*0.25*time)\n    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)\n    >>> noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> noise *= np.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the spectrogram.\n\n    >>> f, t, Sxx = signal.spectrogram(x, fs)\n    >>> plt.pcolormesh(t, f, Sxx)\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n    ')
    
    # Assigning a List to a Name (line 684):
    
    # Assigning a List to a Name (line 684):
    
    # Obtaining an instance of the builtin type 'list' (line 684)
    list_280877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 684)
    # Adding element type (line 684)
    str_280878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 16), 'str', 'psd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_280877, str_280878)
    # Adding element type (line 684)
    str_280879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 23), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_280877, str_280879)
    # Adding element type (line 684)
    str_280880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 34), 'str', 'magnitude')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_280877, str_280880)
    # Adding element type (line 684)
    str_280881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 47), 'str', 'angle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_280877, str_280881)
    # Adding element type (line 684)
    str_280882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 56), 'str', 'phase')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 684, 15), list_280877, str_280882)
    
    # Assigning a type to the variable 'modelist' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'modelist', list_280877)
    
    
    # Getting the type of 'mode' (line 685)
    mode_280883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 7), 'mode')
    # Getting the type of 'modelist' (line 685)
    modelist_280884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 19), 'modelist')
    # Applying the binary operator 'notin' (line 685)
    result_contains_280885 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 7), 'notin', mode_280883, modelist_280884)
    
    # Testing the type of an if condition (line 685)
    if_condition_280886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 4), result_contains_280885)
    # Assigning a type to the variable 'if_condition_280886' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'if_condition_280886', if_condition_280886)
    # SSA begins for if statement (line 685)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 686)
    # Processing the call arguments (line 686)
    
    # Call to format(...): (line 686)
    # Processing the call arguments (line 686)
    # Getting the type of 'mode' (line 687)
    mode_280890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 33), 'mode', False)
    # Getting the type of 'modelist' (line 687)
    modelist_280891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 39), 'modelist', False)
    # Processing the call keyword arguments (line 686)
    kwargs_280892 = {}
    str_280888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 25), 'str', 'unknown value for mode {}, must be one of {}')
    # Obtaining the member 'format' of a type (line 686)
    format_280889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 25), str_280888, 'format')
    # Calling format(args, kwargs) (line 686)
    format_call_result_280893 = invoke(stypy.reporting.localization.Localization(__file__, 686, 25), format_280889, *[mode_280890, modelist_280891], **kwargs_280892)
    
    # Processing the call keyword arguments (line 686)
    kwargs_280894 = {}
    # Getting the type of 'ValueError' (line 686)
    ValueError_280887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 686)
    ValueError_call_result_280895 = invoke(stypy.reporting.localization.Localization(__file__, 686, 14), ValueError_280887, *[format_call_result_280893], **kwargs_280894)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 686, 8), ValueError_call_result_280895, 'raise parameter', BaseException)
    # SSA join for if statement (line 685)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 690):
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_280896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to _triage_segments(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'window' (line 690)
    window_280898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 39), 'window', False)
    # Getting the type of 'nperseg' (line 690)
    nperseg_280899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 47), 'nperseg', False)
    # Processing the call keyword arguments (line 690)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 691)
    axis_280900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 60), 'axis', False)
    # Getting the type of 'x' (line 691)
    x_280901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 52), 'x', False)
    # Obtaining the member 'shape' of a type (line 691)
    shape_280902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 52), x_280901, 'shape')
    # Obtaining the member '__getitem__' of a type (line 691)
    getitem___280903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 52), shape_280902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 691)
    subscript_call_result_280904 = invoke(stypy.reporting.localization.Localization(__file__, 691, 52), getitem___280903, axis_280900)
    
    keyword_280905 = subscript_call_result_280904
    kwargs_280906 = {'input_length': keyword_280905}
    # Getting the type of '_triage_segments' (line 690)
    _triage_segments_280897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 22), '_triage_segments', False)
    # Calling _triage_segments(args, kwargs) (line 690)
    _triage_segments_call_result_280907 = invoke(stypy.reporting.localization.Localization(__file__, 690, 22), _triage_segments_280897, *[window_280898, nperseg_280899], **kwargs_280906)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___280908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), _triage_segments_call_result_280907, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_280909 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___280908, int_280896)
    
    # Assigning a type to the variable 'tuple_var_assignment_280470' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_280470', subscript_call_result_280909)
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_280910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 4), 'int')
    
    # Call to _triage_segments(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'window' (line 690)
    window_280912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 39), 'window', False)
    # Getting the type of 'nperseg' (line 690)
    nperseg_280913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 47), 'nperseg', False)
    # Processing the call keyword arguments (line 690)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 691)
    axis_280914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 60), 'axis', False)
    # Getting the type of 'x' (line 691)
    x_280915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 52), 'x', False)
    # Obtaining the member 'shape' of a type (line 691)
    shape_280916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 52), x_280915, 'shape')
    # Obtaining the member '__getitem__' of a type (line 691)
    getitem___280917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 52), shape_280916, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 691)
    subscript_call_result_280918 = invoke(stypy.reporting.localization.Localization(__file__, 691, 52), getitem___280917, axis_280914)
    
    keyword_280919 = subscript_call_result_280918
    kwargs_280920 = {'input_length': keyword_280919}
    # Getting the type of '_triage_segments' (line 690)
    _triage_segments_280911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 22), '_triage_segments', False)
    # Calling _triage_segments(args, kwargs) (line 690)
    _triage_segments_call_result_280921 = invoke(stypy.reporting.localization.Localization(__file__, 690, 22), _triage_segments_280911, *[window_280912, nperseg_280913], **kwargs_280920)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___280922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 4), _triage_segments_call_result_280921, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_280923 = invoke(stypy.reporting.localization.Localization(__file__, 690, 4), getitem___280922, int_280910)
    
    # Assigning a type to the variable 'tuple_var_assignment_280471' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_280471', subscript_call_result_280923)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_280470' (line 690)
    tuple_var_assignment_280470_280924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_280470')
    # Assigning a type to the variable 'window' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'window', tuple_var_assignment_280470_280924)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_280471' (line 690)
    tuple_var_assignment_280471_280925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'tuple_var_assignment_280471')
    # Assigning a type to the variable 'nperseg' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 12), 'nperseg', tuple_var_assignment_280471_280925)
    
    # Type idiom detected: calculating its left and rigth part (line 694)
    # Getting the type of 'noverlap' (line 694)
    noverlap_280926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), 'noverlap')
    # Getting the type of 'None' (line 694)
    None_280927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'None')
    
    (may_be_280928, more_types_in_union_280929) = may_be_none(noverlap_280926, None_280927)

    if may_be_280928:

        if more_types_in_union_280929:
            # Runtime conditional SSA (line 694)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 695):
        
        # Assigning a BinOp to a Name (line 695):
        # Getting the type of 'nperseg' (line 695)
        nperseg_280930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 19), 'nperseg')
        int_280931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 30), 'int')
        # Applying the binary operator '//' (line 695)
        result_floordiv_280932 = python_operator(stypy.reporting.localization.Localization(__file__, 695, 19), '//', nperseg_280930, int_280931)
        
        # Assigning a type to the variable 'noverlap' (line 695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'noverlap', result_floordiv_280932)

        if more_types_in_union_280929:
            # SSA join for if statement (line 694)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'mode' (line 697)
    mode_280933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 7), 'mode')
    str_280934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 15), 'str', 'psd')
    # Applying the binary operator '==' (line 697)
    result_eq_280935 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 7), '==', mode_280933, str_280934)
    
    # Testing the type of an if condition (line 697)
    if_condition_280936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 4), result_eq_280935)
    # Assigning a type to the variable 'if_condition_280936' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'if_condition_280936', if_condition_280936)
    # SSA begins for if statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 698):
    
    # Assigning a Subscript to a Name (line 698):
    
    # Obtaining the type of the subscript
    int_280937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 8), 'int')
    
    # Call to _spectral_helper(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'x' (line 698)
    x_280939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 44), 'x', False)
    # Getting the type of 'x' (line 698)
    x_280940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'x', False)
    # Getting the type of 'fs' (line 698)
    fs_280941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 50), 'fs', False)
    # Getting the type of 'window' (line 698)
    window_280942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 54), 'window', False)
    # Getting the type of 'nperseg' (line 698)
    nperseg_280943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 699)
    noverlap_280944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 699)
    nfft_280945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 699)
    detrend_280946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 700)
    return_onesided_280947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 700)
    scaling_280948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 61), 'scaling', False)
    # Getting the type of 'axis' (line 700)
    axis_280949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 70), 'axis', False)
    # Processing the call keyword arguments (line 698)
    str_280950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 49), 'str', 'psd')
    keyword_280951 = str_280950
    kwargs_280952 = {'mode': keyword_280951}
    # Getting the type of '_spectral_helper' (line 698)
    _spectral_helper_280938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 698)
    _spectral_helper_call_result_280953 = invoke(stypy.reporting.localization.Localization(__file__, 698, 27), _spectral_helper_280938, *[x_280939, x_280940, fs_280941, window_280942, nperseg_280943, noverlap_280944, nfft_280945, detrend_280946, return_onesided_280947, scaling_280948, axis_280949], **kwargs_280952)
    
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___280954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), _spectral_helper_call_result_280953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_280955 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), getitem___280954, int_280937)
    
    # Assigning a type to the variable 'tuple_var_assignment_280472' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280472', subscript_call_result_280955)
    
    # Assigning a Subscript to a Name (line 698):
    
    # Obtaining the type of the subscript
    int_280956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 8), 'int')
    
    # Call to _spectral_helper(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'x' (line 698)
    x_280958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 44), 'x', False)
    # Getting the type of 'x' (line 698)
    x_280959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'x', False)
    # Getting the type of 'fs' (line 698)
    fs_280960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 50), 'fs', False)
    # Getting the type of 'window' (line 698)
    window_280961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 54), 'window', False)
    # Getting the type of 'nperseg' (line 698)
    nperseg_280962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 699)
    noverlap_280963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 699)
    nfft_280964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 699)
    detrend_280965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 700)
    return_onesided_280966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 700)
    scaling_280967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 61), 'scaling', False)
    # Getting the type of 'axis' (line 700)
    axis_280968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 70), 'axis', False)
    # Processing the call keyword arguments (line 698)
    str_280969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 49), 'str', 'psd')
    keyword_280970 = str_280969
    kwargs_280971 = {'mode': keyword_280970}
    # Getting the type of '_spectral_helper' (line 698)
    _spectral_helper_280957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 698)
    _spectral_helper_call_result_280972 = invoke(stypy.reporting.localization.Localization(__file__, 698, 27), _spectral_helper_280957, *[x_280958, x_280959, fs_280960, window_280961, nperseg_280962, noverlap_280963, nfft_280964, detrend_280965, return_onesided_280966, scaling_280967, axis_280968], **kwargs_280971)
    
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___280973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), _spectral_helper_call_result_280972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_280974 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), getitem___280973, int_280956)
    
    # Assigning a type to the variable 'tuple_var_assignment_280473' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280473', subscript_call_result_280974)
    
    # Assigning a Subscript to a Name (line 698):
    
    # Obtaining the type of the subscript
    int_280975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 8), 'int')
    
    # Call to _spectral_helper(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'x' (line 698)
    x_280977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 44), 'x', False)
    # Getting the type of 'x' (line 698)
    x_280978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 47), 'x', False)
    # Getting the type of 'fs' (line 698)
    fs_280979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 50), 'fs', False)
    # Getting the type of 'window' (line 698)
    window_280980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 54), 'window', False)
    # Getting the type of 'nperseg' (line 698)
    nperseg_280981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 699)
    noverlap_280982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 699)
    nfft_280983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 699)
    detrend_280984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 700)
    return_onesided_280985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 700)
    scaling_280986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 61), 'scaling', False)
    # Getting the type of 'axis' (line 700)
    axis_280987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 70), 'axis', False)
    # Processing the call keyword arguments (line 698)
    str_280988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 49), 'str', 'psd')
    keyword_280989 = str_280988
    kwargs_280990 = {'mode': keyword_280989}
    # Getting the type of '_spectral_helper' (line 698)
    _spectral_helper_280976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 698)
    _spectral_helper_call_result_280991 = invoke(stypy.reporting.localization.Localization(__file__, 698, 27), _spectral_helper_280976, *[x_280977, x_280978, fs_280979, window_280980, nperseg_280981, noverlap_280982, nfft_280983, detrend_280984, return_onesided_280985, scaling_280986, axis_280987], **kwargs_280990)
    
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___280992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), _spectral_helper_call_result_280991, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_280993 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), getitem___280992, int_280975)
    
    # Assigning a type to the variable 'tuple_var_assignment_280474' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280474', subscript_call_result_280993)
    
    # Assigning a Name to a Name (line 698):
    # Getting the type of 'tuple_var_assignment_280472' (line 698)
    tuple_var_assignment_280472_280994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280472')
    # Assigning a type to the variable 'freqs' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'freqs', tuple_var_assignment_280472_280994)
    
    # Assigning a Name to a Name (line 698):
    # Getting the type of 'tuple_var_assignment_280473' (line 698)
    tuple_var_assignment_280473_280995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280473')
    # Assigning a type to the variable 'time' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 15), 'time', tuple_var_assignment_280473_280995)
    
    # Assigning a Name to a Name (line 698):
    # Getting the type of 'tuple_var_assignment_280474' (line 698)
    tuple_var_assignment_280474_280996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'tuple_var_assignment_280474')
    # Assigning a type to the variable 'Sxx' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 21), 'Sxx', tuple_var_assignment_280474_280996)
    # SSA branch for the else part of an if statement (line 697)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 704):
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    int_280997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 8), 'int')
    
    # Call to _spectral_helper(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'x' (line 704)
    x_280999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 44), 'x', False)
    # Getting the type of 'x' (line 704)
    x_281000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 47), 'x', False)
    # Getting the type of 'fs' (line 704)
    fs_281001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 50), 'fs', False)
    # Getting the type of 'window' (line 704)
    window_281002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 54), 'window', False)
    # Getting the type of 'nperseg' (line 704)
    nperseg_281003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 705)
    noverlap_281004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 705)
    nfft_281005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 705)
    detrend_281006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 706)
    return_onesided_281007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 706)
    scaling_281008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 61), 'scaling', False)
    # Getting the type of 'axis' (line 706)
    axis_281009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 70), 'axis', False)
    # Processing the call keyword arguments (line 704)
    str_281010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 49), 'str', 'stft')
    keyword_281011 = str_281010
    kwargs_281012 = {'mode': keyword_281011}
    # Getting the type of '_spectral_helper' (line 704)
    _spectral_helper_280998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 704)
    _spectral_helper_call_result_281013 = invoke(stypy.reporting.localization.Localization(__file__, 704, 27), _spectral_helper_280998, *[x_280999, x_281000, fs_281001, window_281002, nperseg_281003, noverlap_281004, nfft_281005, detrend_281006, return_onesided_281007, scaling_281008, axis_281009], **kwargs_281012)
    
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___281014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), _spectral_helper_call_result_281013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_281015 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), getitem___281014, int_280997)
    
    # Assigning a type to the variable 'tuple_var_assignment_280475' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280475', subscript_call_result_281015)
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    int_281016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 8), 'int')
    
    # Call to _spectral_helper(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'x' (line 704)
    x_281018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 44), 'x', False)
    # Getting the type of 'x' (line 704)
    x_281019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 47), 'x', False)
    # Getting the type of 'fs' (line 704)
    fs_281020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 50), 'fs', False)
    # Getting the type of 'window' (line 704)
    window_281021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 54), 'window', False)
    # Getting the type of 'nperseg' (line 704)
    nperseg_281022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 705)
    noverlap_281023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 705)
    nfft_281024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 705)
    detrend_281025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 706)
    return_onesided_281026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 706)
    scaling_281027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 61), 'scaling', False)
    # Getting the type of 'axis' (line 706)
    axis_281028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 70), 'axis', False)
    # Processing the call keyword arguments (line 704)
    str_281029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 49), 'str', 'stft')
    keyword_281030 = str_281029
    kwargs_281031 = {'mode': keyword_281030}
    # Getting the type of '_spectral_helper' (line 704)
    _spectral_helper_281017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 704)
    _spectral_helper_call_result_281032 = invoke(stypy.reporting.localization.Localization(__file__, 704, 27), _spectral_helper_281017, *[x_281018, x_281019, fs_281020, window_281021, nperseg_281022, noverlap_281023, nfft_281024, detrend_281025, return_onesided_281026, scaling_281027, axis_281028], **kwargs_281031)
    
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___281033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), _spectral_helper_call_result_281032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_281034 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), getitem___281033, int_281016)
    
    # Assigning a type to the variable 'tuple_var_assignment_280476' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280476', subscript_call_result_281034)
    
    # Assigning a Subscript to a Name (line 704):
    
    # Obtaining the type of the subscript
    int_281035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 8), 'int')
    
    # Call to _spectral_helper(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'x' (line 704)
    x_281037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 44), 'x', False)
    # Getting the type of 'x' (line 704)
    x_281038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 47), 'x', False)
    # Getting the type of 'fs' (line 704)
    fs_281039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 50), 'fs', False)
    # Getting the type of 'window' (line 704)
    window_281040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 54), 'window', False)
    # Getting the type of 'nperseg' (line 704)
    nperseg_281041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 62), 'nperseg', False)
    # Getting the type of 'noverlap' (line 705)
    noverlap_281042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 705)
    nfft_281043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 705)
    detrend_281044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 60), 'detrend', False)
    # Getting the type of 'return_onesided' (line 706)
    return_onesided_281045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 44), 'return_onesided', False)
    # Getting the type of 'scaling' (line 706)
    scaling_281046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 61), 'scaling', False)
    # Getting the type of 'axis' (line 706)
    axis_281047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 70), 'axis', False)
    # Processing the call keyword arguments (line 704)
    str_281048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 49), 'str', 'stft')
    keyword_281049 = str_281048
    kwargs_281050 = {'mode': keyword_281049}
    # Getting the type of '_spectral_helper' (line 704)
    _spectral_helper_281036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 27), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 704)
    _spectral_helper_call_result_281051 = invoke(stypy.reporting.localization.Localization(__file__, 704, 27), _spectral_helper_281036, *[x_281037, x_281038, fs_281039, window_281040, nperseg_281041, noverlap_281042, nfft_281043, detrend_281044, return_onesided_281045, scaling_281046, axis_281047], **kwargs_281050)
    
    # Obtaining the member '__getitem__' of a type (line 704)
    getitem___281052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), _spectral_helper_call_result_281051, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 704)
    subscript_call_result_281053 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), getitem___281052, int_281035)
    
    # Assigning a type to the variable 'tuple_var_assignment_280477' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280477', subscript_call_result_281053)
    
    # Assigning a Name to a Name (line 704):
    # Getting the type of 'tuple_var_assignment_280475' (line 704)
    tuple_var_assignment_280475_281054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280475')
    # Assigning a type to the variable 'freqs' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'freqs', tuple_var_assignment_280475_281054)
    
    # Assigning a Name to a Name (line 704):
    # Getting the type of 'tuple_var_assignment_280476' (line 704)
    tuple_var_assignment_280476_281055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280476')
    # Assigning a type to the variable 'time' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 15), 'time', tuple_var_assignment_280476_281055)
    
    # Assigning a Name to a Name (line 704):
    # Getting the type of 'tuple_var_assignment_280477' (line 704)
    tuple_var_assignment_280477_281056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'tuple_var_assignment_280477')
    # Assigning a type to the variable 'Sxx' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 21), 'Sxx', tuple_var_assignment_280477_281056)
    
    
    # Getting the type of 'mode' (line 709)
    mode_281057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 11), 'mode')
    str_281058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 19), 'str', 'magnitude')
    # Applying the binary operator '==' (line 709)
    result_eq_281059 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 11), '==', mode_281057, str_281058)
    
    # Testing the type of an if condition (line 709)
    if_condition_281060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 8), result_eq_281059)
    # Assigning a type to the variable 'if_condition_281060' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'if_condition_281060', if_condition_281060)
    # SSA begins for if statement (line 709)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 710):
    
    # Assigning a Call to a Name (line 710):
    
    # Call to abs(...): (line 710)
    # Processing the call arguments (line 710)
    # Getting the type of 'Sxx' (line 710)
    Sxx_281063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 25), 'Sxx', False)
    # Processing the call keyword arguments (line 710)
    kwargs_281064 = {}
    # Getting the type of 'np' (line 710)
    np_281061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 710)
    abs_281062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 18), np_281061, 'abs')
    # Calling abs(args, kwargs) (line 710)
    abs_call_result_281065 = invoke(stypy.reporting.localization.Localization(__file__, 710, 18), abs_281062, *[Sxx_281063], **kwargs_281064)
    
    # Assigning a type to the variable 'Sxx' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 12), 'Sxx', abs_call_result_281065)
    # SSA branch for the else part of an if statement (line 709)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 711)
    mode_281066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 13), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 711)
    list_281067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 711)
    # Adding element type (line 711)
    str_281068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 22), 'str', 'angle')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 21), list_281067, str_281068)
    # Adding element type (line 711)
    str_281069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 31), 'str', 'phase')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 21), list_281067, str_281069)
    
    # Applying the binary operator 'in' (line 711)
    result_contains_281070 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 13), 'in', mode_281066, list_281067)
    
    # Testing the type of an if condition (line 711)
    if_condition_281071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 13), result_contains_281070)
    # Assigning a type to the variable 'if_condition_281071' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 13), 'if_condition_281071', if_condition_281071)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 712):
    
    # Assigning a Call to a Name (line 712):
    
    # Call to angle(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'Sxx' (line 712)
    Sxx_281074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 27), 'Sxx', False)
    # Processing the call keyword arguments (line 712)
    kwargs_281075 = {}
    # Getting the type of 'np' (line 712)
    np_281072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 18), 'np', False)
    # Obtaining the member 'angle' of a type (line 712)
    angle_281073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 18), np_281072, 'angle')
    # Calling angle(args, kwargs) (line 712)
    angle_call_result_281076 = invoke(stypy.reporting.localization.Localization(__file__, 712, 18), angle_281073, *[Sxx_281074], **kwargs_281075)
    
    # Assigning a type to the variable 'Sxx' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 12), 'Sxx', angle_call_result_281076)
    
    
    # Getting the type of 'mode' (line 713)
    mode_281077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'mode')
    str_281078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 23), 'str', 'phase')
    # Applying the binary operator '==' (line 713)
    result_eq_281079 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 15), '==', mode_281077, str_281078)
    
    # Testing the type of an if condition (line 713)
    if_condition_281080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 713, 12), result_eq_281079)
    # Assigning a type to the variable 'if_condition_281080' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 12), 'if_condition_281080', if_condition_281080)
    # SSA begins for if statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'axis' (line 715)
    axis_281081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 19), 'axis')
    int_281082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 26), 'int')
    # Applying the binary operator '<' (line 715)
    result_lt_281083 = python_operator(stypy.reporting.localization.Localization(__file__, 715, 19), '<', axis_281081, int_281082)
    
    # Testing the type of an if condition (line 715)
    if_condition_281084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 715, 16), result_lt_281083)
    # Assigning a type to the variable 'if_condition_281084' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'if_condition_281084', if_condition_281084)
    # SSA begins for if statement (line 715)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axis' (line 716)
    axis_281085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'axis')
    int_281086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 28), 'int')
    # Applying the binary operator '-=' (line 716)
    result_isub_281087 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 20), '-=', axis_281085, int_281086)
    # Assigning a type to the variable 'axis' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 20), 'axis', result_isub_281087)
    
    # SSA join for if statement (line 715)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 717):
    
    # Assigning a Call to a Name (line 717):
    
    # Call to unwrap(...): (line 717)
    # Processing the call arguments (line 717)
    # Getting the type of 'Sxx' (line 717)
    Sxx_281090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 32), 'Sxx', False)
    # Processing the call keyword arguments (line 717)
    # Getting the type of 'axis' (line 717)
    axis_281091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 42), 'axis', False)
    keyword_281092 = axis_281091
    kwargs_281093 = {'axis': keyword_281092}
    # Getting the type of 'np' (line 717)
    np_281088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 22), 'np', False)
    # Obtaining the member 'unwrap' of a type (line 717)
    unwrap_281089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 22), np_281088, 'unwrap')
    # Calling unwrap(args, kwargs) (line 717)
    unwrap_call_result_281094 = invoke(stypy.reporting.localization.Localization(__file__, 717, 22), unwrap_281089, *[Sxx_281090], **kwargs_281093)
    
    # Assigning a type to the variable 'Sxx' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 16), 'Sxx', unwrap_call_result_281094)
    # SSA join for if statement (line 713)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 709)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 721)
    tuple_281095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 721)
    # Adding element type (line 721)
    # Getting the type of 'freqs' (line 721)
    freqs_281096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 11), tuple_281095, freqs_281096)
    # Adding element type (line 721)
    # Getting the type of 'time' (line 721)
    time_281097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 18), 'time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 11), tuple_281095, time_281097)
    # Adding element type (line 721)
    # Getting the type of 'Sxx' (line 721)
    Sxx_281098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 24), 'Sxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 11), tuple_281095, Sxx_281098)
    
    # Assigning a type to the variable 'stypy_return_type' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'stypy_return_type', tuple_281095)
    
    # ################# End of 'spectrogram(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spectrogram' in the type store
    # Getting the type of 'stypy_return_type' (line 563)
    stypy_return_type_281099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_281099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spectrogram'
    return stypy_return_type_281099

# Assigning a type to the variable 'spectrogram' (line 563)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'spectrogram', spectrogram)

@norecursion
def check_COLA(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_281100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, 46), 'float')
    defaults = [float_281100]
    # Create a new context for function 'check_COLA'
    module_type_store = module_type_store.open_function_context('check_COLA', 724, 0, False)
    
    # Passed parameters checking function
    check_COLA.stypy_localization = localization
    check_COLA.stypy_type_of_self = None
    check_COLA.stypy_type_store = module_type_store
    check_COLA.stypy_function_name = 'check_COLA'
    check_COLA.stypy_param_names_list = ['window', 'nperseg', 'noverlap', 'tol']
    check_COLA.stypy_varargs_param_name = None
    check_COLA.stypy_kwargs_param_name = None
    check_COLA.stypy_call_defaults = defaults
    check_COLA.stypy_call_varargs = varargs
    check_COLA.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_COLA', ['window', 'nperseg', 'noverlap', 'tol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_COLA', localization, ['window', 'nperseg', 'noverlap', 'tol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_COLA(...)' code ##################

    str_281101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, (-1)), 'str', '\n    Check whether the Constant OverLap Add (COLA) constraint is met\n\n    Parameters\n    ----------\n    window : str or tuple or array_like\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg.\n    nperseg : int\n        Length of each segment.\n    noverlap : int\n        Number of points to overlap between segments.\n    tol : float, optional\n        The allowed variance of a bin\'s weighted sum from the median bin\n        sum.\n\n    Returns\n    -------\n    verdict : bool\n        `True` if chosen combination satisfies COLA within `tol`,\n        `False` otherwise\n\n    See Also\n    --------\n    stft: Short Time Fourier Transform\n    istft: Inverse Short Time Fourier Transform\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "Constant\n    OverLap Add" (COLA). This ensures that every point in the input data\n    is equally weighted, thereby avoiding aliasing and allowing full\n    reconstruction.\n\n    Some examples of windows that satisfy COLA:\n        - Rectangular window at overlap of 0, 1/2, 2/3, 3/4, ...\n        - Bartlett window at overlap of 1/2, 3/4, 5/6, ...\n        - Hann window at 1/2, 2/3, 3/4, ...\n        - Any Blackman family window at 2/3 overlap\n        - Any window with ``noverlap = nperseg-1``\n\n    A very comprehensive list of other windows may be found in [2]_,\n    wherein the COLA condition is satisfied when the "Amplitude\n    Flatness" is unity.\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Julius O. Smith III, "Spectral Audio Signal Processing", W3K\n           Publishing, 2011,ISBN 978-0-9745607-3-1.\n    .. [2] G. Heinzel, A. Ruediger and R. Schilling, "Spectrum and\n           spectral density estimation by the Discrete Fourier transform\n           (DFT), including a comprehensive list of window functions and\n           some new at-top windows", 2002,\n           http://hdl.handle.net/11858/00-001M-0000-0013-557A-5\n\n    Examples\n    --------\n    >>> from scipy import signal\n\n    Confirm COLA condition for rectangular window of 75% (3/4) overlap:\n\n    >>> signal.check_COLA(signal.boxcar(100), 100, 75)\n    True\n\n    COLA is not true for 25% (1/4) overlap, though:\n\n    >>> signal.check_COLA(signal.boxcar(100), 100, 25)\n    False\n\n    "Symmetrical" Hann window (for filter design) is not COLA:\n\n    >>> signal.check_COLA(signal.hann(120, sym=True), 120, 60)\n    False\n\n    "Periodic" or "DFT-even" Hann window (for FFT analysis) is COLA for\n    overlap of 1/2, 2/3, 3/4, etc.:\n\n    >>> signal.check_COLA(signal.hann(120, sym=False), 120, 60)\n    True\n\n    >>> signal.check_COLA(signal.hann(120, sym=False), 120, 80)\n    True\n\n    >>> signal.check_COLA(signal.hann(120, sym=False), 120, 90)\n    True\n\n    ')
    
    # Assigning a Call to a Name (line 819):
    
    # Assigning a Call to a Name (line 819):
    
    # Call to int(...): (line 819)
    # Processing the call arguments (line 819)
    # Getting the type of 'nperseg' (line 819)
    nperseg_281103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 18), 'nperseg', False)
    # Processing the call keyword arguments (line 819)
    kwargs_281104 = {}
    # Getting the type of 'int' (line 819)
    int_281102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 14), 'int', False)
    # Calling int(args, kwargs) (line 819)
    int_call_result_281105 = invoke(stypy.reporting.localization.Localization(__file__, 819, 14), int_281102, *[nperseg_281103], **kwargs_281104)
    
    # Assigning a type to the variable 'nperseg' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'nperseg', int_call_result_281105)
    
    
    # Getting the type of 'nperseg' (line 821)
    nperseg_281106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 7), 'nperseg')
    int_281107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 17), 'int')
    # Applying the binary operator '<' (line 821)
    result_lt_281108 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 7), '<', nperseg_281106, int_281107)
    
    # Testing the type of an if condition (line 821)
    if_condition_281109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 4), result_lt_281108)
    # Assigning a type to the variable 'if_condition_281109' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'if_condition_281109', if_condition_281109)
    # SSA begins for if statement (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 822)
    # Processing the call arguments (line 822)
    str_281111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 25), 'str', 'nperseg must be a positive integer')
    # Processing the call keyword arguments (line 822)
    kwargs_281112 = {}
    # Getting the type of 'ValueError' (line 822)
    ValueError_281110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 822)
    ValueError_call_result_281113 = invoke(stypy.reporting.localization.Localization(__file__, 822, 14), ValueError_281110, *[str_281111], **kwargs_281112)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 822, 8), ValueError_call_result_281113, 'raise parameter', BaseException)
    # SSA join for if statement (line 821)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'noverlap' (line 824)
    noverlap_281114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 7), 'noverlap')
    # Getting the type of 'nperseg' (line 824)
    nperseg_281115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 19), 'nperseg')
    # Applying the binary operator '>=' (line 824)
    result_ge_281116 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 7), '>=', noverlap_281114, nperseg_281115)
    
    # Testing the type of an if condition (line 824)
    if_condition_281117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 4), result_ge_281116)
    # Assigning a type to the variable 'if_condition_281117' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'if_condition_281117', if_condition_281117)
    # SSA begins for if statement (line 824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 825)
    # Processing the call arguments (line 825)
    str_281119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 25), 'str', 'noverlap must be less than nperseg.')
    # Processing the call keyword arguments (line 825)
    kwargs_281120 = {}
    # Getting the type of 'ValueError' (line 825)
    ValueError_281118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 825)
    ValueError_call_result_281121 = invoke(stypy.reporting.localization.Localization(__file__, 825, 14), ValueError_281118, *[str_281119], **kwargs_281120)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 825, 8), ValueError_call_result_281121, 'raise parameter', BaseException)
    # SSA join for if statement (line 824)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 826):
    
    # Assigning a Call to a Name (line 826):
    
    # Call to int(...): (line 826)
    # Processing the call arguments (line 826)
    # Getting the type of 'noverlap' (line 826)
    noverlap_281123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 19), 'noverlap', False)
    # Processing the call keyword arguments (line 826)
    kwargs_281124 = {}
    # Getting the type of 'int' (line 826)
    int_281122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 15), 'int', False)
    # Calling int(args, kwargs) (line 826)
    int_call_result_281125 = invoke(stypy.reporting.localization.Localization(__file__, 826, 15), int_281122, *[noverlap_281123], **kwargs_281124)
    
    # Assigning a type to the variable 'noverlap' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'noverlap', int_call_result_281125)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'window' (line 828)
    window_281127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 18), 'window', False)
    # Getting the type of 'string_types' (line 828)
    string_types_281128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 26), 'string_types', False)
    # Processing the call keyword arguments (line 828)
    kwargs_281129 = {}
    # Getting the type of 'isinstance' (line 828)
    isinstance_281126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 828)
    isinstance_call_result_281130 = invoke(stypy.reporting.localization.Localization(__file__, 828, 7), isinstance_281126, *[window_281127, string_types_281128], **kwargs_281129)
    
    
    
    # Call to type(...): (line 828)
    # Processing the call arguments (line 828)
    # Getting the type of 'window' (line 828)
    window_281132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 48), 'window', False)
    # Processing the call keyword arguments (line 828)
    kwargs_281133 = {}
    # Getting the type of 'type' (line 828)
    type_281131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 43), 'type', False)
    # Calling type(args, kwargs) (line 828)
    type_call_result_281134 = invoke(stypy.reporting.localization.Localization(__file__, 828, 43), type_281131, *[window_281132], **kwargs_281133)
    
    # Getting the type of 'tuple' (line 828)
    tuple_281135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 59), 'tuple')
    # Applying the binary operator 'is' (line 828)
    result_is__281136 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 43), 'is', type_call_result_281134, tuple_281135)
    
    # Applying the binary operator 'or' (line 828)
    result_or_keyword_281137 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 7), 'or', isinstance_call_result_281130, result_is__281136)
    
    # Testing the type of an if condition (line 828)
    if_condition_281138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 4), result_or_keyword_281137)
    # Assigning a type to the variable 'if_condition_281138' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 4), 'if_condition_281138', if_condition_281138)
    # SSA begins for if statement (line 828)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 829):
    
    # Assigning a Call to a Name (line 829):
    
    # Call to get_window(...): (line 829)
    # Processing the call arguments (line 829)
    # Getting the type of 'window' (line 829)
    window_281140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 25), 'window', False)
    # Getting the type of 'nperseg' (line 829)
    nperseg_281141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 33), 'nperseg', False)
    # Processing the call keyword arguments (line 829)
    kwargs_281142 = {}
    # Getting the type of 'get_window' (line 829)
    get_window_281139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 14), 'get_window', False)
    # Calling get_window(args, kwargs) (line 829)
    get_window_call_result_281143 = invoke(stypy.reporting.localization.Localization(__file__, 829, 14), get_window_281139, *[window_281140, nperseg_281141], **kwargs_281142)
    
    # Assigning a type to the variable 'win' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 8), 'win', get_window_call_result_281143)
    # SSA branch for the else part of an if statement (line 828)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 831):
    
    # Assigning a Call to a Name (line 831):
    
    # Call to asarray(...): (line 831)
    # Processing the call arguments (line 831)
    # Getting the type of 'window' (line 831)
    window_281146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 25), 'window', False)
    # Processing the call keyword arguments (line 831)
    kwargs_281147 = {}
    # Getting the type of 'np' (line 831)
    np_281144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 831)
    asarray_281145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 831, 14), np_281144, 'asarray')
    # Calling asarray(args, kwargs) (line 831)
    asarray_call_result_281148 = invoke(stypy.reporting.localization.Localization(__file__, 831, 14), asarray_281145, *[window_281146], **kwargs_281147)
    
    # Assigning a type to the variable 'win' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'win', asarray_call_result_281148)
    
    
    
    # Call to len(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'win' (line 832)
    win_281150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 15), 'win', False)
    # Obtaining the member 'shape' of a type (line 832)
    shape_281151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 15), win_281150, 'shape')
    # Processing the call keyword arguments (line 832)
    kwargs_281152 = {}
    # Getting the type of 'len' (line 832)
    len_281149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 11), 'len', False)
    # Calling len(args, kwargs) (line 832)
    len_call_result_281153 = invoke(stypy.reporting.localization.Localization(__file__, 832, 11), len_281149, *[shape_281151], **kwargs_281152)
    
    int_281154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 29), 'int')
    # Applying the binary operator '!=' (line 832)
    result_ne_281155 = python_operator(stypy.reporting.localization.Localization(__file__, 832, 11), '!=', len_call_result_281153, int_281154)
    
    # Testing the type of an if condition (line 832)
    if_condition_281156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 8), result_ne_281155)
    # Assigning a type to the variable 'if_condition_281156' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'if_condition_281156', if_condition_281156)
    # SSA begins for if statement (line 832)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 833)
    # Processing the call arguments (line 833)
    str_281158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 29), 'str', 'window must be 1-D')
    # Processing the call keyword arguments (line 833)
    kwargs_281159 = {}
    # Getting the type of 'ValueError' (line 833)
    ValueError_281157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 833)
    ValueError_call_result_281160 = invoke(stypy.reporting.localization.Localization(__file__, 833, 18), ValueError_281157, *[str_281158], **kwargs_281159)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 833, 12), ValueError_call_result_281160, 'raise parameter', BaseException)
    # SSA join for if statement (line 832)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_281161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 21), 'int')
    # Getting the type of 'win' (line 834)
    win_281162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 11), 'win')
    # Obtaining the member 'shape' of a type (line 834)
    shape_281163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 11), win_281162, 'shape')
    # Obtaining the member '__getitem__' of a type (line 834)
    getitem___281164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 11), shape_281163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 834)
    subscript_call_result_281165 = invoke(stypy.reporting.localization.Localization(__file__, 834, 11), getitem___281164, int_281161)
    
    # Getting the type of 'nperseg' (line 834)
    nperseg_281166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 27), 'nperseg')
    # Applying the binary operator '!=' (line 834)
    result_ne_281167 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 11), '!=', subscript_call_result_281165, nperseg_281166)
    
    # Testing the type of an if condition (line 834)
    if_condition_281168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 834, 8), result_ne_281167)
    # Assigning a type to the variable 'if_condition_281168' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'if_condition_281168', if_condition_281168)
    # SSA begins for if statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 835)
    # Processing the call arguments (line 835)
    str_281170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 835, 29), 'str', 'window must have length of nperseg')
    # Processing the call keyword arguments (line 835)
    kwargs_281171 = {}
    # Getting the type of 'ValueError' (line 835)
    ValueError_281169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 835)
    ValueError_call_result_281172 = invoke(stypy.reporting.localization.Localization(__file__, 835, 18), ValueError_281169, *[str_281170], **kwargs_281171)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 835, 12), ValueError_call_result_281172, 'raise parameter', BaseException)
    # SSA join for if statement (line 834)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 828)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 837):
    
    # Assigning a BinOp to a Name (line 837):
    # Getting the type of 'nperseg' (line 837)
    nperseg_281173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 11), 'nperseg')
    # Getting the type of 'noverlap' (line 837)
    noverlap_281174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 21), 'noverlap')
    # Applying the binary operator '-' (line 837)
    result_sub_281175 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 11), '-', nperseg_281173, noverlap_281174)
    
    # Assigning a type to the variable 'step' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'step', result_sub_281175)
    
    # Assigning a Call to a Name (line 838):
    
    # Assigning a Call to a Name (line 838):
    
    # Call to sum(...): (line 838)
    # Processing the call arguments (line 838)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 838, 22, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 838)
    # Processing the call arguments (line 838)
    # Getting the type of 'nperseg' (line 838)
    nperseg_281191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 63), 'nperseg', False)
    # Getting the type of 'step' (line 838)
    step_281192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 72), 'step', False)
    # Applying the binary operator '//' (line 838)
    result_floordiv_281193 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 63), '//', nperseg_281191, step_281192)
    
    # Processing the call keyword arguments (line 838)
    kwargs_281194 = {}
    # Getting the type of 'range' (line 838)
    range_281190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 57), 'range', False)
    # Calling range(args, kwargs) (line 838)
    range_call_result_281195 = invoke(stypy.reporting.localization.Localization(__file__, 838, 57), range_281190, *[result_floordiv_281193], **kwargs_281194)
    
    comprehension_281196 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 22), range_call_result_281195)
    # Assigning a type to the variable 'ii' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'ii', comprehension_281196)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ii' (line 838)
    ii_281178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 26), 'ii', False)
    # Getting the type of 'step' (line 838)
    step_281179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 29), 'step', False)
    # Applying the binary operator '*' (line 838)
    result_mul_281180 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 26), '*', ii_281178, step_281179)
    
    # Getting the type of 'ii' (line 838)
    ii_281181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 35), 'ii', False)
    int_281182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 38), 'int')
    # Applying the binary operator '+' (line 838)
    result_add_281183 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 35), '+', ii_281181, int_281182)
    
    # Getting the type of 'step' (line 838)
    step_281184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 41), 'step', False)
    # Applying the binary operator '*' (line 838)
    result_mul_281185 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 34), '*', result_add_281183, step_281184)
    
    slice_281186 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 838, 22), result_mul_281180, result_mul_281185, None)
    # Getting the type of 'win' (line 838)
    win_281187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'win', False)
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___281188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 22), win_281187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_281189 = invoke(stypy.reporting.localization.Localization(__file__, 838, 22), getitem___281188, slice_281186)
    
    list_281197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 22), list_281197, subscript_call_result_281189)
    # Processing the call keyword arguments (line 838)
    int_281198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 26), 'int')
    keyword_281199 = int_281198
    kwargs_281200 = {'axis': keyword_281199}
    # Getting the type of 'np' (line 838)
    np_281176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 14), 'np', False)
    # Obtaining the member 'sum' of a type (line 838)
    sum_281177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 14), np_281176, 'sum')
    # Calling sum(args, kwargs) (line 838)
    sum_call_result_281201 = invoke(stypy.reporting.localization.Localization(__file__, 838, 14), sum_281177, *[list_281197], **kwargs_281200)
    
    # Assigning a type to the variable 'binsums' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 4), 'binsums', sum_call_result_281201)
    
    
    # Getting the type of 'nperseg' (line 841)
    nperseg_281202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 7), 'nperseg')
    # Getting the type of 'step' (line 841)
    step_281203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 17), 'step')
    # Applying the binary operator '%' (line 841)
    result_mod_281204 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 7), '%', nperseg_281202, step_281203)
    
    int_281205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 25), 'int')
    # Applying the binary operator '!=' (line 841)
    result_ne_281206 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 7), '!=', result_mod_281204, int_281205)
    
    # Testing the type of an if condition (line 841)
    if_condition_281207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 841, 4), result_ne_281206)
    # Assigning a type to the variable 'if_condition_281207' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 4), 'if_condition_281207', if_condition_281207)
    # SSA begins for if statement (line 841)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'binsums' (line 842)
    binsums_281208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'binsums')
    
    # Obtaining the type of the subscript
    # Getting the type of 'nperseg' (line 842)
    nperseg_281209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 17), 'nperseg')
    # Getting the type of 'step' (line 842)
    step_281210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 27), 'step')
    # Applying the binary operator '%' (line 842)
    result_mod_281211 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 17), '%', nperseg_281209, step_281210)
    
    slice_281212 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 842, 8), None, result_mod_281211, None)
    # Getting the type of 'binsums' (line 842)
    binsums_281213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'binsums')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___281214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 8), binsums_281213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_281215 = invoke(stypy.reporting.localization.Localization(__file__, 842, 8), getitem___281214, slice_281212)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'nperseg' (line 842)
    nperseg_281216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 42), 'nperseg')
    # Getting the type of 'step' (line 842)
    step_281217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 52), 'step')
    # Applying the binary operator '%' (line 842)
    result_mod_281218 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 42), '%', nperseg_281216, step_281217)
    
    # Applying the 'usub' unary operator (line 842)
    result___neg___281219 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 40), 'usub', result_mod_281218)
    
    slice_281220 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 842, 36), result___neg___281219, None, None)
    # Getting the type of 'win' (line 842)
    win_281221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 36), 'win')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___281222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 36), win_281221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_281223 = invoke(stypy.reporting.localization.Localization(__file__, 842, 36), getitem___281222, slice_281220)
    
    # Applying the binary operator '+=' (line 842)
    result_iadd_281224 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 8), '+=', subscript_call_result_281215, subscript_call_result_281223)
    # Getting the type of 'binsums' (line 842)
    binsums_281225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 8), 'binsums')
    # Getting the type of 'nperseg' (line 842)
    nperseg_281226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 17), 'nperseg')
    # Getting the type of 'step' (line 842)
    step_281227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 27), 'step')
    # Applying the binary operator '%' (line 842)
    result_mod_281228 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 17), '%', nperseg_281226, step_281227)
    
    slice_281229 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 842, 8), None, result_mod_281228, None)
    # Storing an element on a container (line 842)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 8), binsums_281225, (slice_281229, result_iadd_281224))
    
    # SSA join for if statement (line 841)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 844):
    
    # Assigning a BinOp to a Name (line 844):
    # Getting the type of 'binsums' (line 844)
    binsums_281230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 16), 'binsums')
    
    # Call to median(...): (line 844)
    # Processing the call arguments (line 844)
    # Getting the type of 'binsums' (line 844)
    binsums_281233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 36), 'binsums', False)
    # Processing the call keyword arguments (line 844)
    kwargs_281234 = {}
    # Getting the type of 'np' (line 844)
    np_281231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 26), 'np', False)
    # Obtaining the member 'median' of a type (line 844)
    median_281232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 26), np_281231, 'median')
    # Calling median(args, kwargs) (line 844)
    median_call_result_281235 = invoke(stypy.reporting.localization.Localization(__file__, 844, 26), median_281232, *[binsums_281233], **kwargs_281234)
    
    # Applying the binary operator '-' (line 844)
    result_sub_281236 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 16), '-', binsums_281230, median_call_result_281235)
    
    # Assigning a type to the variable 'deviation' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 4), 'deviation', result_sub_281236)
    
    
    # Call to max(...): (line 845)
    # Processing the call arguments (line 845)
    
    # Call to abs(...): (line 845)
    # Processing the call arguments (line 845)
    # Getting the type of 'deviation' (line 845)
    deviation_281241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 25), 'deviation', False)
    # Processing the call keyword arguments (line 845)
    kwargs_281242 = {}
    # Getting the type of 'np' (line 845)
    np_281239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 18), 'np', False)
    # Obtaining the member 'abs' of a type (line 845)
    abs_281240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 18), np_281239, 'abs')
    # Calling abs(args, kwargs) (line 845)
    abs_call_result_281243 = invoke(stypy.reporting.localization.Localization(__file__, 845, 18), abs_281240, *[deviation_281241], **kwargs_281242)
    
    # Processing the call keyword arguments (line 845)
    kwargs_281244 = {}
    # Getting the type of 'np' (line 845)
    np_281237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 11), 'np', False)
    # Obtaining the member 'max' of a type (line 845)
    max_281238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 11), np_281237, 'max')
    # Calling max(args, kwargs) (line 845)
    max_call_result_281245 = invoke(stypy.reporting.localization.Localization(__file__, 845, 11), max_281238, *[abs_call_result_281243], **kwargs_281244)
    
    # Getting the type of 'tol' (line 845)
    tol_281246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 39), 'tol')
    # Applying the binary operator '<' (line 845)
    result_lt_281247 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 11), '<', max_call_result_281245, tol_281246)
    
    # Assigning a type to the variable 'stypy_return_type' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'stypy_return_type', result_lt_281247)
    
    # ################# End of 'check_COLA(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_COLA' in the type store
    # Getting the type of 'stypy_return_type' (line 724)
    stypy_return_type_281248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_281248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_COLA'
    return stypy_return_type_281248

# Assigning a type to the variable 'check_COLA' (line 724)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'check_COLA', check_COLA)

@norecursion
def stft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_281249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 15), 'float')
    str_281250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 27), 'str', 'hann')
    int_281251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 43), 'int')
    # Getting the type of 'None' (line 848)
    None_281252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 57), 'None')
    # Getting the type of 'None' (line 848)
    None_281253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 68), 'None')
    # Getting the type of 'False' (line 849)
    False_281254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 17), 'False')
    # Getting the type of 'True' (line 849)
    True_281255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 40), 'True')
    str_281256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 55), 'str', 'zeros')
    # Getting the type of 'True' (line 849)
    True_281257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 71), 'True')
    int_281258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 14), 'int')
    defaults = [float_281249, str_281250, int_281251, None_281252, None_281253, False_281254, True_281255, str_281256, True_281257, int_281258]
    # Create a new context for function 'stft'
    module_type_store = module_type_store.open_function_context('stft', 848, 0, False)
    
    # Passed parameters checking function
    stft.stypy_localization = localization
    stft.stypy_type_of_self = None
    stft.stypy_type_store = module_type_store
    stft.stypy_function_name = 'stft'
    stft.stypy_param_names_list = ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'boundary', 'padded', 'axis']
    stft.stypy_varargs_param_name = None
    stft.stypy_kwargs_param_name = None
    stft.stypy_call_defaults = defaults
    stft.stypy_call_varargs = varargs
    stft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stft', ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'boundary', 'padded', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stft', localization, ['x', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'boundary', 'padded', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stft(...)' code ##################

    str_281259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 984, (-1)), 'str', '\n    Compute the Short Time Fourier Transform (STFT).\n\n    STFTs can be used as a way of quantifying the change of a\n    nonstationary signal\'s frequency and phase content over time.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to 256.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`. When\n        specified, the COLA constraint must be met (see Notes below).\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to `False`.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned. Defaults to\n        `True`.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``[\'even\', \'odd\', \'constant\', \'zeros\', None]``. Defaults to\n        \'zeros\', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is\n        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `True`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`, as is the\n        default.\n    axis : int, optional\n        Axis along which the STFT is computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of segment times.\n    Zxx : ndarray\n        STFT of `x`. By default, the last axis of `Zxx` corresponds\n        to the segment times.\n\n    See Also\n    --------\n    istft: Inverse Short Time Fourier Transform\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint\n                is met\n    welch: Power spectral density by Welch\'s method.\n    spectrogram: Spectrogram by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT in\n    `istft`, the signal windowing must obey the constraint of "Constant\n    OverLap Add" (COLA), and the input signal must have complete\n    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %\n    (nperseg-noverlap) == 0``). The `padded` argument may be used to\n    accomplish this.\n\n    The COLA constraint ensures that every point in the input data is\n    equally weighted, thereby avoiding aliasing and allowing full\n    reconstruction. Whether a choice of `window`, `nperseg`, and\n    `noverlap` satisfy this constraint can be tested with\n    `check_COLA`.\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Limdt "Signal Estimation from\n           Modified Short Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly\n    modulated around 3kHz, corrupted by white noise of exponentially\n    decreasing magnitude sampled at 10 kHz.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.01 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> mod = 500*np.cos(2*np.pi*0.25*time)\n    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)\n    >>> noise = np.random.normal(scale=np.sqrt(noise_power),\n    ...                          size=time.shape)\n    >>> noise *= np.exp(-time/5)\n    >>> x = carrier + noise\n\n    Compute and plot the STFT\'s magnitude.\n\n    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)\n    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Tuple (line 986):
    
    # Assigning a Subscript to a Name (line 986):
    
    # Obtaining the type of the subscript
    int_281260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 4), 'int')
    
    # Call to _spectral_helper(...): (line 986)
    # Processing the call arguments (line 986)
    # Getting the type of 'x' (line 986)
    x_281262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 40), 'x', False)
    # Getting the type of 'x' (line 986)
    x_281263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 43), 'x', False)
    # Getting the type of 'fs' (line 986)
    fs_281264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 46), 'fs', False)
    # Getting the type of 'window' (line 986)
    window_281265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 50), 'window', False)
    # Getting the type of 'nperseg' (line 986)
    nperseg_281266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 58), 'nperseg', False)
    # Getting the type of 'noverlap' (line 986)
    noverlap_281267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 67), 'noverlap', False)
    # Getting the type of 'nfft' (line 987)
    nfft_281268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 40), 'nfft', False)
    # Getting the type of 'detrend' (line 987)
    detrend_281269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 46), 'detrend', False)
    # Getting the type of 'return_onesided' (line 987)
    return_onesided_281270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 55), 'return_onesided', False)
    # Processing the call keyword arguments (line 986)
    str_281271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 988, 48), 'str', 'spectrum')
    keyword_281272 = str_281271
    # Getting the type of 'axis' (line 988)
    axis_281273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 65), 'axis', False)
    keyword_281274 = axis_281273
    str_281275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 45), 'str', 'stft')
    keyword_281276 = str_281275
    # Getting the type of 'boundary' (line 989)
    boundary_281277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 62), 'boundary', False)
    keyword_281278 = boundary_281277
    # Getting the type of 'padded' (line 990)
    padded_281279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 47), 'padded', False)
    keyword_281280 = padded_281279
    kwargs_281281 = {'scaling': keyword_281272, 'padded': keyword_281280, 'boundary': keyword_281278, 'mode': keyword_281276, 'axis': keyword_281274}
    # Getting the type of '_spectral_helper' (line 986)
    _spectral_helper_281261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 23), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 986)
    _spectral_helper_call_result_281282 = invoke(stypy.reporting.localization.Localization(__file__, 986, 23), _spectral_helper_281261, *[x_281262, x_281263, fs_281264, window_281265, nperseg_281266, noverlap_281267, nfft_281268, detrend_281269, return_onesided_281270], **kwargs_281281)
    
    # Obtaining the member '__getitem__' of a type (line 986)
    getitem___281283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 4), _spectral_helper_call_result_281282, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 986)
    subscript_call_result_281284 = invoke(stypy.reporting.localization.Localization(__file__, 986, 4), getitem___281283, int_281260)
    
    # Assigning a type to the variable 'tuple_var_assignment_280478' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280478', subscript_call_result_281284)
    
    # Assigning a Subscript to a Name (line 986):
    
    # Obtaining the type of the subscript
    int_281285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 4), 'int')
    
    # Call to _spectral_helper(...): (line 986)
    # Processing the call arguments (line 986)
    # Getting the type of 'x' (line 986)
    x_281287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 40), 'x', False)
    # Getting the type of 'x' (line 986)
    x_281288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 43), 'x', False)
    # Getting the type of 'fs' (line 986)
    fs_281289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 46), 'fs', False)
    # Getting the type of 'window' (line 986)
    window_281290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 50), 'window', False)
    # Getting the type of 'nperseg' (line 986)
    nperseg_281291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 58), 'nperseg', False)
    # Getting the type of 'noverlap' (line 986)
    noverlap_281292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 67), 'noverlap', False)
    # Getting the type of 'nfft' (line 987)
    nfft_281293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 40), 'nfft', False)
    # Getting the type of 'detrend' (line 987)
    detrend_281294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 46), 'detrend', False)
    # Getting the type of 'return_onesided' (line 987)
    return_onesided_281295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 55), 'return_onesided', False)
    # Processing the call keyword arguments (line 986)
    str_281296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 988, 48), 'str', 'spectrum')
    keyword_281297 = str_281296
    # Getting the type of 'axis' (line 988)
    axis_281298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 65), 'axis', False)
    keyword_281299 = axis_281298
    str_281300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 45), 'str', 'stft')
    keyword_281301 = str_281300
    # Getting the type of 'boundary' (line 989)
    boundary_281302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 62), 'boundary', False)
    keyword_281303 = boundary_281302
    # Getting the type of 'padded' (line 990)
    padded_281304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 47), 'padded', False)
    keyword_281305 = padded_281304
    kwargs_281306 = {'scaling': keyword_281297, 'padded': keyword_281305, 'boundary': keyword_281303, 'mode': keyword_281301, 'axis': keyword_281299}
    # Getting the type of '_spectral_helper' (line 986)
    _spectral_helper_281286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 23), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 986)
    _spectral_helper_call_result_281307 = invoke(stypy.reporting.localization.Localization(__file__, 986, 23), _spectral_helper_281286, *[x_281287, x_281288, fs_281289, window_281290, nperseg_281291, noverlap_281292, nfft_281293, detrend_281294, return_onesided_281295], **kwargs_281306)
    
    # Obtaining the member '__getitem__' of a type (line 986)
    getitem___281308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 4), _spectral_helper_call_result_281307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 986)
    subscript_call_result_281309 = invoke(stypy.reporting.localization.Localization(__file__, 986, 4), getitem___281308, int_281285)
    
    # Assigning a type to the variable 'tuple_var_assignment_280479' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280479', subscript_call_result_281309)
    
    # Assigning a Subscript to a Name (line 986):
    
    # Obtaining the type of the subscript
    int_281310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 986, 4), 'int')
    
    # Call to _spectral_helper(...): (line 986)
    # Processing the call arguments (line 986)
    # Getting the type of 'x' (line 986)
    x_281312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 40), 'x', False)
    # Getting the type of 'x' (line 986)
    x_281313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 43), 'x', False)
    # Getting the type of 'fs' (line 986)
    fs_281314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 46), 'fs', False)
    # Getting the type of 'window' (line 986)
    window_281315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 50), 'window', False)
    # Getting the type of 'nperseg' (line 986)
    nperseg_281316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 58), 'nperseg', False)
    # Getting the type of 'noverlap' (line 986)
    noverlap_281317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 67), 'noverlap', False)
    # Getting the type of 'nfft' (line 987)
    nfft_281318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 40), 'nfft', False)
    # Getting the type of 'detrend' (line 987)
    detrend_281319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 46), 'detrend', False)
    # Getting the type of 'return_onesided' (line 987)
    return_onesided_281320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 55), 'return_onesided', False)
    # Processing the call keyword arguments (line 986)
    str_281321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 988, 48), 'str', 'spectrum')
    keyword_281322 = str_281321
    # Getting the type of 'axis' (line 988)
    axis_281323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 988, 65), 'axis', False)
    keyword_281324 = axis_281323
    str_281325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 45), 'str', 'stft')
    keyword_281326 = str_281325
    # Getting the type of 'boundary' (line 989)
    boundary_281327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 62), 'boundary', False)
    keyword_281328 = boundary_281327
    # Getting the type of 'padded' (line 990)
    padded_281329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 47), 'padded', False)
    keyword_281330 = padded_281329
    kwargs_281331 = {'scaling': keyword_281322, 'padded': keyword_281330, 'boundary': keyword_281328, 'mode': keyword_281326, 'axis': keyword_281324}
    # Getting the type of '_spectral_helper' (line 986)
    _spectral_helper_281311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 23), '_spectral_helper', False)
    # Calling _spectral_helper(args, kwargs) (line 986)
    _spectral_helper_call_result_281332 = invoke(stypy.reporting.localization.Localization(__file__, 986, 23), _spectral_helper_281311, *[x_281312, x_281313, fs_281314, window_281315, nperseg_281316, noverlap_281317, nfft_281318, detrend_281319, return_onesided_281320], **kwargs_281331)
    
    # Obtaining the member '__getitem__' of a type (line 986)
    getitem___281333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 986, 4), _spectral_helper_call_result_281332, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 986)
    subscript_call_result_281334 = invoke(stypy.reporting.localization.Localization(__file__, 986, 4), getitem___281333, int_281310)
    
    # Assigning a type to the variable 'tuple_var_assignment_280480' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280480', subscript_call_result_281334)
    
    # Assigning a Name to a Name (line 986):
    # Getting the type of 'tuple_var_assignment_280478' (line 986)
    tuple_var_assignment_280478_281335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280478')
    # Assigning a type to the variable 'freqs' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'freqs', tuple_var_assignment_280478_281335)
    
    # Assigning a Name to a Name (line 986):
    # Getting the type of 'tuple_var_assignment_280479' (line 986)
    tuple_var_assignment_280479_281336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280479')
    # Assigning a type to the variable 'time' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 11), 'time', tuple_var_assignment_280479_281336)
    
    # Assigning a Name to a Name (line 986):
    # Getting the type of 'tuple_var_assignment_280480' (line 986)
    tuple_var_assignment_280480_281337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 4), 'tuple_var_assignment_280480')
    # Assigning a type to the variable 'Zxx' (line 986)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 986, 17), 'Zxx', tuple_var_assignment_280480_281337)
    
    # Obtaining an instance of the builtin type 'tuple' (line 992)
    tuple_281338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 992, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 992)
    # Adding element type (line 992)
    # Getting the type of 'freqs' (line 992)
    freqs_281339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 992, 11), tuple_281338, freqs_281339)
    # Adding element type (line 992)
    # Getting the type of 'time' (line 992)
    time_281340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 18), 'time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 992, 11), tuple_281338, time_281340)
    # Adding element type (line 992)
    # Getting the type of 'Zxx' (line 992)
    Zxx_281341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 24), 'Zxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 992, 11), tuple_281338, Zxx_281341)
    
    # Assigning a type to the variable 'stypy_return_type' (line 992)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 992, 4), 'stypy_return_type', tuple_281338)
    
    # ################# End of 'stft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stft' in the type store
    # Getting the type of 'stypy_return_type' (line 848)
    stypy_return_type_281342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_281342)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stft'
    return stypy_return_type_281342

# Assigning a type to the variable 'stft' (line 848)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 0), 'stft', stft)

@norecursion
def istft(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_281343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 18), 'float')
    str_281344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 30), 'str', 'hann')
    # Getting the type of 'None' (line 995)
    None_281345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 46), 'None')
    # Getting the type of 'None' (line 995)
    None_281346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 61), 'None')
    # Getting the type of 'None' (line 995)
    None_281347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 72), 'None')
    # Getting the type of 'True' (line 996)
    True_281348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 25), 'True')
    # Getting the type of 'True' (line 996)
    True_281349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 996, 40), 'True')
    int_281350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 56), 'int')
    int_281351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, 70), 'int')
    defaults = [float_281343, str_281344, None_281345, None_281346, None_281347, True_281348, True_281349, int_281350, int_281351]
    # Create a new context for function 'istft'
    module_type_store = module_type_store.open_function_context('istft', 995, 0, False)
    
    # Passed parameters checking function
    istft.stypy_localization = localization
    istft.stypy_type_of_self = None
    istft.stypy_type_store = module_type_store
    istft.stypy_function_name = 'istft'
    istft.stypy_param_names_list = ['Zxx', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary', 'time_axis', 'freq_axis']
    istft.stypy_varargs_param_name = None
    istft.stypy_kwargs_param_name = None
    istft.stypy_call_defaults = defaults
    istft.stypy_call_varargs = varargs
    istft.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'istft', ['Zxx', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary', 'time_axis', 'freq_axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'istft', localization, ['Zxx', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary', 'time_axis', 'freq_axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'istft(...)' code ##################

    str_281352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1152, (-1)), 'str', '\n    Perform the inverse Short Time Fourier transform (iSTFT).\n\n    Parameters\n    ----------\n    Zxx : array_like\n        STFT of the signal to be reconstructed. If a purely real array\n        is passed, it will be cast to a complex data type.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window. Must match the window used to generate the\n        STFT for faithful inversion.\n    nperseg : int, optional\n        Number of data points corresponding to each STFT segment. This\n        parameter must be specified if the number of data points per\n        segment is odd, or if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the value depends on the shape of\n        `Zxx` and `input_onesided`. If `input_onesided` is True,\n        ``nperseg=2*(Zxx.shape[freq_axis] - 1)``. Otherwise,\n        ``nperseg=Zxx.shape[freq_axis]``. Defaults to `None`.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`, half\n        of the segment length. Defaults to `None`. When specified, the\n        COLA constraint must be met (see Notes below), and should match\n        the parameter used to generate the STFT. Defaults to `None`.\n    nfft : int, optional\n        Number of FFT points corresponding to each STFT segment. This\n        parameter must be specified if the STFT was padded via ``nfft >\n        nperseg``. If `None`, the default values are the same as for\n        `nperseg`, detailed above, with one exception: if\n        `input_onesided` is True and\n        ``nperseg==2*Zxx.shape[freq_axis] - 1``, `nfft` also takes on\n        that value. This case allows the proper inversion of an\n        odd-length unpadded STFT using ``nfft=None``. Defaults to\n        `None`.\n    input_onesided : bool, optional\n        If `True`, interpret the input array as one-sided FFTs, such\n        as is returned by `stft` with ``return_onesided=True`` and\n        `numpy.fft.rfft`. If `False`, interpret the input as a a\n        two-sided FFT. Defaults to `True`.\n    boundary : bool, optional\n        Specifies whether the input signal was extended at its\n        boundaries by supplying a non-`None` ``boundary`` argument to\n        `stft`. Defaults to `True`.\n    time_axis : int, optional\n        Where the time segments of the STFT is located; the default is\n        the last axis (i.e. ``axis=-1``).\n    freq_axis : int, optional\n        Where the frequency axis of the STFT is located; the default is\n        the penultimate axis (i.e. ``axis=-2``).\n\n    Returns\n    -------\n    t : ndarray\n        Array of output data times.\n    x : ndarray\n        iSTFT of `Zxx`.\n\n    See Also\n    --------\n    stft: Short Time Fourier Transform\n    check_COLA: Check whether the Constant OverLap Add (COLA) constraint\n                is met\n\n    Notes\n    -----\n    In order to enable inversion of an STFT via the inverse STFT with\n    `istft`, the signal windowing must obey the constraint of "Constant\n    OverLap Add" (COLA). This ensures that every point in the input data\n    is equally weighted, thereby avoiding aliasing and allowing full\n    reconstruction. Whether a choice of `window`, `nperseg`, and\n    `noverlap` satisfy this constraint can be tested with\n    `check_COLA`, by using ``nperseg = Zxx.shape[freq_axis]``.\n\n    An STFT which has been modified (via masking or otherwise) is not\n    guaranteed to correspond to a exactly realizible signal. This\n    function implements the iSTFT via the least-squares esimation\n    algorithm detailed in [2]_, which produces a signal that minimizes\n    the mean squared error between the STFT of the returned signal and\n    the modified STFT.\n\n    .. versionadded:: 0.19.0\n\n    References\n    ----------\n    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck\n           "Discrete-Time Signal Processing", Prentice Hall, 1999.\n    .. [2] Daniel W. Griffin, Jae S. Limdt "Signal Estimation from\n           Modified Short Fourier Transform", IEEE 1984,\n           10.1109/TASSP.1984.1164317\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    Generate a test signal, a 2 Vrms sine wave at 50Hz corrupted by\n    0.001 V**2/Hz of white noise sampled at 1024 Hz.\n\n    >>> fs = 1024\n    >>> N = 10*fs\n    >>> nperseg = 512\n    >>> amp = 2 * np.sqrt(2)\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / float(fs)\n    >>> carrier = amp * np.sin(2*np.pi*50*time)\n    >>> noise = np.random.normal(scale=np.sqrt(noise_power),\n    ...                          size=time.shape)\n    >>> x = carrier + noise\n\n    Compute the STFT, and plot its magnitude\n\n    >>> f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg)\n    >>> plt.figure()\n    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)\n    >>> plt.ylim([f[1], f[-1]])\n    >>> plt.title(\'STFT Magnitude\')\n    >>> plt.ylabel(\'Frequency [Hz]\')\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.yscale(\'log\')\n    >>> plt.show()\n\n    Zero the components that are 10% or less of the carrier magnitude,\n    then convert back to a time series via inverse STFT\n\n    >>> Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)\n    >>> _, xrec = signal.istft(Zxx, fs)\n\n    Compare the cleaned signal with the original and true carrier signals.\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([2, 2.1])\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n\n    Note that the cleaned signal does not start as abruptly as the original,\n    since some of the coefficients of the transient were also removed:\n\n    >>> plt.figure()\n    >>> plt.plot(time, x, time, xrec, time, carrier)\n    >>> plt.xlim([0, 0.1])\n    >>> plt.xlabel(\'Time [sec]\')\n    >>> plt.ylabel(\'Signal\')\n    >>> plt.legend([\'Carrier + Noise\', \'Filtered via STFT\', \'True Carrier\'])\n    >>> plt.show()\n\n    ')
    
    # Assigning a BinOp to a Name (line 1155):
    
    # Assigning a BinOp to a Name (line 1155):
    
    # Call to asarray(...): (line 1155)
    # Processing the call arguments (line 1155)
    # Getting the type of 'Zxx' (line 1155)
    Zxx_281355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 21), 'Zxx', False)
    # Processing the call keyword arguments (line 1155)
    kwargs_281356 = {}
    # Getting the type of 'np' (line 1155)
    np_281353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1155)
    asarray_281354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1155, 10), np_281353, 'asarray')
    # Calling asarray(args, kwargs) (line 1155)
    asarray_call_result_281357 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 10), asarray_281354, *[Zxx_281355], **kwargs_281356)
    
    complex_281358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1155, 28), 'complex')
    # Applying the binary operator '+' (line 1155)
    result_add_281359 = python_operator(stypy.reporting.localization.Localization(__file__, 1155, 10), '+', asarray_call_result_281357, complex_281358)
    
    # Assigning a type to the variable 'Zxx' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 4), 'Zxx', result_add_281359)
    
    # Assigning a Call to a Name (line 1156):
    
    # Assigning a Call to a Name (line 1156):
    
    # Call to int(...): (line 1156)
    # Processing the call arguments (line 1156)
    # Getting the type of 'freq_axis' (line 1156)
    freq_axis_281361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 20), 'freq_axis', False)
    # Processing the call keyword arguments (line 1156)
    kwargs_281362 = {}
    # Getting the type of 'int' (line 1156)
    int_281360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 16), 'int', False)
    # Calling int(args, kwargs) (line 1156)
    int_call_result_281363 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 16), int_281360, *[freq_axis_281361], **kwargs_281362)
    
    # Assigning a type to the variable 'freq_axis' (line 1156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'freq_axis', int_call_result_281363)
    
    # Assigning a Call to a Name (line 1157):
    
    # Assigning a Call to a Name (line 1157):
    
    # Call to int(...): (line 1157)
    # Processing the call arguments (line 1157)
    # Getting the type of 'time_axis' (line 1157)
    time_axis_281365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 20), 'time_axis', False)
    # Processing the call keyword arguments (line 1157)
    kwargs_281366 = {}
    # Getting the type of 'int' (line 1157)
    int_281364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 16), 'int', False)
    # Calling int(args, kwargs) (line 1157)
    int_call_result_281367 = invoke(stypy.reporting.localization.Localization(__file__, 1157, 16), int_281364, *[time_axis_281365], **kwargs_281366)
    
    # Assigning a type to the variable 'time_axis' (line 1157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 4), 'time_axis', int_call_result_281367)
    
    
    # Getting the type of 'Zxx' (line 1159)
    Zxx_281368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 7), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1159)
    ndim_281369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 7), Zxx_281368, 'ndim')
    int_281370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 18), 'int')
    # Applying the binary operator '<' (line 1159)
    result_lt_281371 = python_operator(stypy.reporting.localization.Localization(__file__, 1159, 7), '<', ndim_281369, int_281370)
    
    # Testing the type of an if condition (line 1159)
    if_condition_281372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1159, 4), result_lt_281371)
    # Assigning a type to the variable 'if_condition_281372' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'if_condition_281372', if_condition_281372)
    # SSA begins for if statement (line 1159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1160)
    # Processing the call arguments (line 1160)
    str_281374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 25), 'str', 'Input stft must be at least 2d!')
    # Processing the call keyword arguments (line 1160)
    kwargs_281375 = {}
    # Getting the type of 'ValueError' (line 1160)
    ValueError_281373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1160)
    ValueError_call_result_281376 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 14), ValueError_281373, *[str_281374], **kwargs_281375)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1160, 8), ValueError_call_result_281376, 'raise parameter', BaseException)
    # SSA join for if statement (line 1159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'freq_axis' (line 1162)
    freq_axis_281377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 7), 'freq_axis')
    # Getting the type of 'time_axis' (line 1162)
    time_axis_281378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1162, 20), 'time_axis')
    # Applying the binary operator '==' (line 1162)
    result_eq_281379 = python_operator(stypy.reporting.localization.Localization(__file__, 1162, 7), '==', freq_axis_281377, time_axis_281378)
    
    # Testing the type of an if condition (line 1162)
    if_condition_281380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1162, 4), result_eq_281379)
    # Assigning a type to the variable 'if_condition_281380' (line 1162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1162, 4), 'if_condition_281380', if_condition_281380)
    # SSA begins for if statement (line 1162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1163)
    # Processing the call arguments (line 1163)
    str_281382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 25), 'str', 'Must specify differing time and frequency axes!')
    # Processing the call keyword arguments (line 1163)
    kwargs_281383 = {}
    # Getting the type of 'ValueError' (line 1163)
    ValueError_281381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1163, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1163)
    ValueError_call_result_281384 = invoke(stypy.reporting.localization.Localization(__file__, 1163, 14), ValueError_281381, *[str_281382], **kwargs_281383)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1163, 8), ValueError_call_result_281384, 'raise parameter', BaseException)
    # SSA join for if statement (line 1162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1165):
    
    # Assigning a Subscript to a Name (line 1165):
    
    # Obtaining the type of the subscript
    # Getting the type of 'time_axis' (line 1165)
    time_axis_281385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 21), 'time_axis')
    # Getting the type of 'Zxx' (line 1165)
    Zxx_281386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 11), 'Zxx')
    # Obtaining the member 'shape' of a type (line 1165)
    shape_281387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 11), Zxx_281386, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1165)
    getitem___281388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1165, 11), shape_281387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1165)
    subscript_call_result_281389 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 11), getitem___281388, time_axis_281385)
    
    # Assigning a type to the variable 'nseg' (line 1165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'nseg', subscript_call_result_281389)
    
    # Getting the type of 'input_onesided' (line 1167)
    input_onesided_281390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 7), 'input_onesided')
    # Testing the type of an if condition (line 1167)
    if_condition_281391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1167, 4), input_onesided_281390)
    # Assigning a type to the variable 'if_condition_281391' (line 1167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 4), 'if_condition_281391', if_condition_281391)
    # SSA begins for if statement (line 1167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1169):
    
    # Assigning a BinOp to a Name (line 1169):
    int_281392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 20), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'freq_axis' (line 1169)
    freq_axis_281393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 33), 'freq_axis')
    # Getting the type of 'Zxx' (line 1169)
    Zxx_281394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 23), 'Zxx')
    # Obtaining the member 'shape' of a type (line 1169)
    shape_281395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 23), Zxx_281394, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1169)
    getitem___281396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 23), shape_281395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1169)
    subscript_call_result_281397 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 23), getitem___281396, freq_axis_281393)
    
    int_281398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 46), 'int')
    # Applying the binary operator '-' (line 1169)
    result_sub_281399 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 23), '-', subscript_call_result_281397, int_281398)
    
    # Applying the binary operator '*' (line 1169)
    result_mul_281400 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 20), '*', int_281392, result_sub_281399)
    
    # Assigning a type to the variable 'n_default' (line 1169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 8), 'n_default', result_mul_281400)
    # SSA branch for the else part of an if statement (line 1167)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 1171):
    
    # Assigning a Subscript to a Name (line 1171):
    
    # Obtaining the type of the subscript
    # Getting the type of 'freq_axis' (line 1171)
    freq_axis_281401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 30), 'freq_axis')
    # Getting the type of 'Zxx' (line 1171)
    Zxx_281402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 20), 'Zxx')
    # Obtaining the member 'shape' of a type (line 1171)
    shape_281403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1171, 20), Zxx_281402, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1171)
    getitem___281404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1171, 20), shape_281403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1171)
    subscript_call_result_281405 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 20), getitem___281404, freq_axis_281401)
    
    # Assigning a type to the variable 'n_default' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 8), 'n_default', subscript_call_result_281405)
    # SSA join for if statement (line 1167)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1174)
    # Getting the type of 'nperseg' (line 1174)
    nperseg_281406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 7), 'nperseg')
    # Getting the type of 'None' (line 1174)
    None_281407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 18), 'None')
    
    (may_be_281408, more_types_in_union_281409) = may_be_none(nperseg_281406, None_281407)

    if may_be_281408:

        if more_types_in_union_281409:
            # Runtime conditional SSA (line 1174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 1175):
        
        # Assigning a Name to a Name (line 1175):
        # Getting the type of 'n_default' (line 1175)
        n_default_281410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 18), 'n_default')
        # Assigning a type to the variable 'nperseg' (line 1175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1175, 8), 'nperseg', n_default_281410)

        if more_types_in_union_281409:
            # Runtime conditional SSA for else branch (line 1174)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_281408) or more_types_in_union_281409):
        
        # Assigning a Call to a Name (line 1177):
        
        # Assigning a Call to a Name (line 1177):
        
        # Call to int(...): (line 1177)
        # Processing the call arguments (line 1177)
        # Getting the type of 'nperseg' (line 1177)
        nperseg_281412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 22), 'nperseg', False)
        # Processing the call keyword arguments (line 1177)
        kwargs_281413 = {}
        # Getting the type of 'int' (line 1177)
        int_281411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 18), 'int', False)
        # Calling int(args, kwargs) (line 1177)
        int_call_result_281414 = invoke(stypy.reporting.localization.Localization(__file__, 1177, 18), int_281411, *[nperseg_281412], **kwargs_281413)
        
        # Assigning a type to the variable 'nperseg' (line 1177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 8), 'nperseg', int_call_result_281414)
        
        
        # Getting the type of 'nperseg' (line 1178)
        nperseg_281415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 11), 'nperseg')
        int_281416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 21), 'int')
        # Applying the binary operator '<' (line 1178)
        result_lt_281417 = python_operator(stypy.reporting.localization.Localization(__file__, 1178, 11), '<', nperseg_281415, int_281416)
        
        # Testing the type of an if condition (line 1178)
        if_condition_281418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1178, 8), result_lt_281417)
        # Assigning a type to the variable 'if_condition_281418' (line 1178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1178, 8), 'if_condition_281418', if_condition_281418)
        # SSA begins for if statement (line 1178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1179)
        # Processing the call arguments (line 1179)
        str_281420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1179, 29), 'str', 'nperseg must be a positive integer')
        # Processing the call keyword arguments (line 1179)
        kwargs_281421 = {}
        # Getting the type of 'ValueError' (line 1179)
        ValueError_281419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1179, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1179)
        ValueError_call_result_281422 = invoke(stypy.reporting.localization.Localization(__file__, 1179, 18), ValueError_281419, *[str_281420], **kwargs_281421)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1179, 12), ValueError_call_result_281422, 'raise parameter', BaseException)
        # SSA join for if statement (line 1178)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_281408 and more_types_in_union_281409):
            # SSA join for if statement (line 1174)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1181)
    # Getting the type of 'nfft' (line 1181)
    nfft_281423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 7), 'nfft')
    # Getting the type of 'None' (line 1181)
    None_281424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 15), 'None')
    
    (may_be_281425, more_types_in_union_281426) = may_be_none(nfft_281423, None_281424)

    if may_be_281425:

        if more_types_in_union_281426:
            # Runtime conditional SSA (line 1181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Evaluating a boolean operation
        # Getting the type of 'input_onesided' (line 1182)
        input_onesided_281427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 12), 'input_onesided')
        
        # Getting the type of 'nperseg' (line 1182)
        nperseg_281428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 33), 'nperseg')
        # Getting the type of 'n_default' (line 1182)
        n_default_281429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 44), 'n_default')
        int_281430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 56), 'int')
        # Applying the binary operator '+' (line 1182)
        result_add_281431 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 44), '+', n_default_281429, int_281430)
        
        # Applying the binary operator '==' (line 1182)
        result_eq_281432 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 33), '==', nperseg_281428, result_add_281431)
        
        # Applying the binary operator 'and' (line 1182)
        result_and_keyword_281433 = python_operator(stypy.reporting.localization.Localization(__file__, 1182, 11), 'and', input_onesided_281427, result_eq_281432)
        
        # Testing the type of an if condition (line 1182)
        if_condition_281434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1182, 8), result_and_keyword_281433)
        # Assigning a type to the variable 'if_condition_281434' (line 1182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1182, 8), 'if_condition_281434', if_condition_281434)
        # SSA begins for if statement (line 1182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 1184):
        
        # Assigning a Name to a Name (line 1184):
        # Getting the type of 'nperseg' (line 1184)
        nperseg_281435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1184, 19), 'nperseg')
        # Assigning a type to the variable 'nfft' (line 1184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1184, 12), 'nfft', nperseg_281435)
        # SSA branch for the else part of an if statement (line 1182)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1186):
        
        # Assigning a Name to a Name (line 1186):
        # Getting the type of 'n_default' (line 1186)
        n_default_281436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1186, 19), 'n_default')
        # Assigning a type to the variable 'nfft' (line 1186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1186, 12), 'nfft', n_default_281436)
        # SSA join for if statement (line 1182)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_281426:
            # Runtime conditional SSA for else branch (line 1181)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_281425) or more_types_in_union_281426):
        
        
        # Getting the type of 'nfft' (line 1187)
        nfft_281437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 9), 'nfft')
        # Getting the type of 'nperseg' (line 1187)
        nperseg_281438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1187, 16), 'nperseg')
        # Applying the binary operator '<' (line 1187)
        result_lt_281439 = python_operator(stypy.reporting.localization.Localization(__file__, 1187, 9), '<', nfft_281437, nperseg_281438)
        
        # Testing the type of an if condition (line 1187)
        if_condition_281440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1187, 9), result_lt_281439)
        # Assigning a type to the variable 'if_condition_281440' (line 1187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1187, 9), 'if_condition_281440', if_condition_281440)
        # SSA begins for if statement (line 1187)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1188)
        # Processing the call arguments (line 1188)
        str_281442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 25), 'str', 'nfft must be greater than or equal to nperseg.')
        # Processing the call keyword arguments (line 1188)
        kwargs_281443 = {}
        # Getting the type of 'ValueError' (line 1188)
        ValueError_281441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1188)
        ValueError_call_result_281444 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 14), ValueError_281441, *[str_281442], **kwargs_281443)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1188, 8), ValueError_call_result_281444, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 1187)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1190):
        
        # Assigning a Call to a Name (line 1190):
        
        # Call to int(...): (line 1190)
        # Processing the call arguments (line 1190)
        # Getting the type of 'nfft' (line 1190)
        nfft_281446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 19), 'nfft', False)
        # Processing the call keyword arguments (line 1190)
        kwargs_281447 = {}
        # Getting the type of 'int' (line 1190)
        int_281445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 15), 'int', False)
        # Calling int(args, kwargs) (line 1190)
        int_call_result_281448 = invoke(stypy.reporting.localization.Localization(__file__, 1190, 15), int_281445, *[nfft_281446], **kwargs_281447)
        
        # Assigning a type to the variable 'nfft' (line 1190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1190, 8), 'nfft', int_call_result_281448)
        # SSA join for if statement (line 1187)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_281425 and more_types_in_union_281426):
            # SSA join for if statement (line 1181)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1192)
    # Getting the type of 'noverlap' (line 1192)
    noverlap_281449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 7), 'noverlap')
    # Getting the type of 'None' (line 1192)
    None_281450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 19), 'None')
    
    (may_be_281451, more_types_in_union_281452) = may_be_none(noverlap_281449, None_281450)

    if may_be_281451:

        if more_types_in_union_281452:
            # Runtime conditional SSA (line 1192)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1193):
        
        # Assigning a BinOp to a Name (line 1193):
        # Getting the type of 'nperseg' (line 1193)
        nperseg_281453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 19), 'nperseg')
        int_281454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1193, 28), 'int')
        # Applying the binary operator '//' (line 1193)
        result_floordiv_281455 = python_operator(stypy.reporting.localization.Localization(__file__, 1193, 19), '//', nperseg_281453, int_281454)
        
        # Assigning a type to the variable 'noverlap' (line 1193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1193, 8), 'noverlap', result_floordiv_281455)

        if more_types_in_union_281452:
            # Runtime conditional SSA for else branch (line 1192)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_281451) or more_types_in_union_281452):
        
        # Assigning a Call to a Name (line 1195):
        
        # Assigning a Call to a Name (line 1195):
        
        # Call to int(...): (line 1195)
        # Processing the call arguments (line 1195)
        # Getting the type of 'noverlap' (line 1195)
        noverlap_281457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 23), 'noverlap', False)
        # Processing the call keyword arguments (line 1195)
        kwargs_281458 = {}
        # Getting the type of 'int' (line 1195)
        int_281456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1195, 19), 'int', False)
        # Calling int(args, kwargs) (line 1195)
        int_call_result_281459 = invoke(stypy.reporting.localization.Localization(__file__, 1195, 19), int_281456, *[noverlap_281457], **kwargs_281458)
        
        # Assigning a type to the variable 'noverlap' (line 1195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1195, 8), 'noverlap', int_call_result_281459)

        if (may_be_281451 and more_types_in_union_281452):
            # SSA join for if statement (line 1192)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'noverlap' (line 1196)
    noverlap_281460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 7), 'noverlap')
    # Getting the type of 'nperseg' (line 1196)
    nperseg_281461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 19), 'nperseg')
    # Applying the binary operator '>=' (line 1196)
    result_ge_281462 = python_operator(stypy.reporting.localization.Localization(__file__, 1196, 7), '>=', noverlap_281460, nperseg_281461)
    
    # Testing the type of an if condition (line 1196)
    if_condition_281463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1196, 4), result_ge_281462)
    # Assigning a type to the variable 'if_condition_281463' (line 1196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 4), 'if_condition_281463', if_condition_281463)
    # SSA begins for if statement (line 1196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1197)
    # Processing the call arguments (line 1197)
    str_281465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1197, 25), 'str', 'noverlap must be less than nperseg.')
    # Processing the call keyword arguments (line 1197)
    kwargs_281466 = {}
    # Getting the type of 'ValueError' (line 1197)
    ValueError_281464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1197)
    ValueError_call_result_281467 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 14), ValueError_281464, *[str_281465], **kwargs_281466)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1197, 8), ValueError_call_result_281467, 'raise parameter', BaseException)
    # SSA join for if statement (line 1196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1198):
    
    # Assigning a BinOp to a Name (line 1198):
    # Getting the type of 'nperseg' (line 1198)
    nperseg_281468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 12), 'nperseg')
    # Getting the type of 'noverlap' (line 1198)
    noverlap_281469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1198, 22), 'noverlap')
    # Applying the binary operator '-' (line 1198)
    result_sub_281470 = python_operator(stypy.reporting.localization.Localization(__file__, 1198, 12), '-', nperseg_281468, noverlap_281469)
    
    # Assigning a type to the variable 'nstep' (line 1198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1198, 4), 'nstep', result_sub_281470)
    
    
    
    # Call to check_COLA(...): (line 1200)
    # Processing the call arguments (line 1200)
    # Getting the type of 'window' (line 1200)
    window_281472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 22), 'window', False)
    # Getting the type of 'nperseg' (line 1200)
    nperseg_281473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 30), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1200)
    noverlap_281474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 39), 'noverlap', False)
    # Processing the call keyword arguments (line 1200)
    kwargs_281475 = {}
    # Getting the type of 'check_COLA' (line 1200)
    check_COLA_281471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 11), 'check_COLA', False)
    # Calling check_COLA(args, kwargs) (line 1200)
    check_COLA_call_result_281476 = invoke(stypy.reporting.localization.Localization(__file__, 1200, 11), check_COLA_281471, *[window_281472, nperseg_281473, noverlap_281474], **kwargs_281475)
    
    # Applying the 'not' unary operator (line 1200)
    result_not__281477 = python_operator(stypy.reporting.localization.Localization(__file__, 1200, 7), 'not', check_COLA_call_result_281476)
    
    # Testing the type of an if condition (line 1200)
    if_condition_281478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1200, 4), result_not__281477)
    # Assigning a type to the variable 'if_condition_281478' (line 1200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1200, 4), 'if_condition_281478', if_condition_281478)
    # SSA begins for if statement (line 1200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1201)
    # Processing the call arguments (line 1201)
    str_281480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1201, 25), 'str', 'Window, STFT shape and noverlap do not satisfy the COLA constraint.')
    # Processing the call keyword arguments (line 1201)
    kwargs_281481 = {}
    # Getting the type of 'ValueError' (line 1201)
    ValueError_281479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1201, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1201)
    ValueError_call_result_281482 = invoke(stypy.reporting.localization.Localization(__file__, 1201, 14), ValueError_281479, *[str_281480], **kwargs_281481)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1201, 8), ValueError_call_result_281482, 'raise parameter', BaseException)
    # SSA join for if statement (line 1200)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'time_axis' (line 1205)
    time_axis_281483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 7), 'time_axis')
    # Getting the type of 'Zxx' (line 1205)
    Zxx_281484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 20), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1205)
    ndim_281485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 20), Zxx_281484, 'ndim')
    int_281486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 29), 'int')
    # Applying the binary operator '-' (line 1205)
    result_sub_281487 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 20), '-', ndim_281485, int_281486)
    
    # Applying the binary operator '!=' (line 1205)
    result_ne_281488 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 7), '!=', time_axis_281483, result_sub_281487)
    
    
    # Getting the type of 'freq_axis' (line 1205)
    freq_axis_281489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 34), 'freq_axis')
    # Getting the type of 'Zxx' (line 1205)
    Zxx_281490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 47), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1205)
    ndim_281491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 47), Zxx_281490, 'ndim')
    int_281492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 56), 'int')
    # Applying the binary operator '-' (line 1205)
    result_sub_281493 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 47), '-', ndim_281491, int_281492)
    
    # Applying the binary operator '!=' (line 1205)
    result_ne_281494 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 34), '!=', freq_axis_281489, result_sub_281493)
    
    # Applying the binary operator 'or' (line 1205)
    result_or_keyword_281495 = python_operator(stypy.reporting.localization.Localization(__file__, 1205, 7), 'or', result_ne_281488, result_ne_281494)
    
    # Testing the type of an if condition (line 1205)
    if_condition_281496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1205, 4), result_or_keyword_281495)
    # Assigning a type to the variable 'if_condition_281496' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 4), 'if_condition_281496', if_condition_281496)
    # SSA begins for if statement (line 1205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'freq_axis' (line 1207)
    freq_axis_281497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 11), 'freq_axis')
    int_281498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1207, 23), 'int')
    # Applying the binary operator '<' (line 1207)
    result_lt_281499 = python_operator(stypy.reporting.localization.Localization(__file__, 1207, 11), '<', freq_axis_281497, int_281498)
    
    # Testing the type of an if condition (line 1207)
    if_condition_281500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1207, 8), result_lt_281499)
    # Assigning a type to the variable 'if_condition_281500' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 8), 'if_condition_281500', if_condition_281500)
    # SSA begins for if statement (line 1207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1208):
    
    # Assigning a BinOp to a Name (line 1208):
    # Getting the type of 'Zxx' (line 1208)
    Zxx_281501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 24), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1208)
    ndim_281502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1208, 24), Zxx_281501, 'ndim')
    # Getting the type of 'freq_axis' (line 1208)
    freq_axis_281503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1208, 35), 'freq_axis')
    # Applying the binary operator '+' (line 1208)
    result_add_281504 = python_operator(stypy.reporting.localization.Localization(__file__, 1208, 24), '+', ndim_281502, freq_axis_281503)
    
    # Assigning a type to the variable 'freq_axis' (line 1208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1208, 12), 'freq_axis', result_add_281504)
    # SSA join for if statement (line 1207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'time_axis' (line 1209)
    time_axis_281505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 11), 'time_axis')
    int_281506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1209, 23), 'int')
    # Applying the binary operator '<' (line 1209)
    result_lt_281507 = python_operator(stypy.reporting.localization.Localization(__file__, 1209, 11), '<', time_axis_281505, int_281506)
    
    # Testing the type of an if condition (line 1209)
    if_condition_281508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1209, 8), result_lt_281507)
    # Assigning a type to the variable 'if_condition_281508' (line 1209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 8), 'if_condition_281508', if_condition_281508)
    # SSA begins for if statement (line 1209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1210):
    
    # Assigning a BinOp to a Name (line 1210):
    # Getting the type of 'Zxx' (line 1210)
    Zxx_281509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 24), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1210)
    ndim_281510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1210, 24), Zxx_281509, 'ndim')
    # Getting the type of 'time_axis' (line 1210)
    time_axis_281511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 35), 'time_axis')
    # Applying the binary operator '+' (line 1210)
    result_add_281512 = python_operator(stypy.reporting.localization.Localization(__file__, 1210, 24), '+', ndim_281510, time_axis_281511)
    
    # Assigning a type to the variable 'time_axis' (line 1210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 12), 'time_axis', result_add_281512)
    # SSA join for if statement (line 1209)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1211):
    
    # Assigning a Call to a Name (line 1211):
    
    # Call to list(...): (line 1211)
    # Processing the call arguments (line 1211)
    
    # Call to range(...): (line 1211)
    # Processing the call arguments (line 1211)
    # Getting the type of 'Zxx' (line 1211)
    Zxx_281515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 28), 'Zxx', False)
    # Obtaining the member 'ndim' of a type (line 1211)
    ndim_281516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1211, 28), Zxx_281515, 'ndim')
    # Processing the call keyword arguments (line 1211)
    kwargs_281517 = {}
    # Getting the type of 'range' (line 1211)
    range_281514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 22), 'range', False)
    # Calling range(args, kwargs) (line 1211)
    range_call_result_281518 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 22), range_281514, *[ndim_281516], **kwargs_281517)
    
    # Processing the call keyword arguments (line 1211)
    kwargs_281519 = {}
    # Getting the type of 'list' (line 1211)
    list_281513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 17), 'list', False)
    # Calling list(args, kwargs) (line 1211)
    list_call_result_281520 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 17), list_281513, *[range_call_result_281518], **kwargs_281519)
    
    # Assigning a type to the variable 'zouter' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 8), 'zouter', list_call_result_281520)
    
    
    # Call to sorted(...): (line 1212)
    # Processing the call arguments (line 1212)
    
    # Obtaining an instance of the builtin type 'list' (line 1212)
    list_281522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1212)
    # Adding element type (line 1212)
    # Getting the type of 'time_axis' (line 1212)
    time_axis_281523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 26), 'time_axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1212, 25), list_281522, time_axis_281523)
    # Adding element type (line 1212)
    # Getting the type of 'freq_axis' (line 1212)
    freq_axis_281524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 37), 'freq_axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1212, 25), list_281522, freq_axis_281524)
    
    # Processing the call keyword arguments (line 1212)
    # Getting the type of 'True' (line 1212)
    True_281525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 57), 'True', False)
    keyword_281526 = True_281525
    kwargs_281527 = {'reverse': keyword_281526}
    # Getting the type of 'sorted' (line 1212)
    sorted_281521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 18), 'sorted', False)
    # Calling sorted(args, kwargs) (line 1212)
    sorted_call_result_281528 = invoke(stypy.reporting.localization.Localization(__file__, 1212, 18), sorted_281521, *[list_281522], **kwargs_281527)
    
    # Testing the type of a for loop iterable (line 1212)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1212, 8), sorted_call_result_281528)
    # Getting the type of the for loop variable (line 1212)
    for_loop_var_281529 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1212, 8), sorted_call_result_281528)
    # Assigning a type to the variable 'ax' (line 1212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 8), 'ax', for_loop_var_281529)
    # SSA begins for a for statement (line 1212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to pop(...): (line 1213)
    # Processing the call arguments (line 1213)
    # Getting the type of 'ax' (line 1213)
    ax_281532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 23), 'ax', False)
    # Processing the call keyword arguments (line 1213)
    kwargs_281533 = {}
    # Getting the type of 'zouter' (line 1213)
    zouter_281530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1213, 12), 'zouter', False)
    # Obtaining the member 'pop' of a type (line 1213)
    pop_281531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1213, 12), zouter_281530, 'pop')
    # Calling pop(args, kwargs) (line 1213)
    pop_call_result_281534 = invoke(stypy.reporting.localization.Localization(__file__, 1213, 12), pop_281531, *[ax_281532], **kwargs_281533)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1214):
    
    # Assigning a Call to a Name (line 1214):
    
    # Call to transpose(...): (line 1214)
    # Processing the call arguments (line 1214)
    # Getting the type of 'Zxx' (line 1214)
    Zxx_281537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 27), 'Zxx', False)
    # Getting the type of 'zouter' (line 1214)
    zouter_281538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 32), 'zouter', False)
    
    # Obtaining an instance of the builtin type 'list' (line 1214)
    list_281539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1214, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1214)
    # Adding element type (line 1214)
    # Getting the type of 'freq_axis' (line 1214)
    freq_axis_281540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 40), 'freq_axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 39), list_281539, freq_axis_281540)
    # Adding element type (line 1214)
    # Getting the type of 'time_axis' (line 1214)
    time_axis_281541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 51), 'time_axis', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1214, 39), list_281539, time_axis_281541)
    
    # Applying the binary operator '+' (line 1214)
    result_add_281542 = python_operator(stypy.reporting.localization.Localization(__file__, 1214, 32), '+', zouter_281538, list_281539)
    
    # Processing the call keyword arguments (line 1214)
    kwargs_281543 = {}
    # Getting the type of 'np' (line 1214)
    np_281535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 14), 'np', False)
    # Obtaining the member 'transpose' of a type (line 1214)
    transpose_281536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1214, 14), np_281535, 'transpose')
    # Calling transpose(args, kwargs) (line 1214)
    transpose_call_result_281544 = invoke(stypy.reporting.localization.Localization(__file__, 1214, 14), transpose_281536, *[Zxx_281537, result_add_281542], **kwargs_281543)
    
    # Assigning a type to the variable 'Zxx' (line 1214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 8), 'Zxx', transpose_call_result_281544)
    # SSA join for if statement (line 1205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 1217)
    # Processing the call arguments (line 1217)
    # Getting the type of 'window' (line 1217)
    window_281546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 18), 'window', False)
    # Getting the type of 'string_types' (line 1217)
    string_types_281547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 26), 'string_types', False)
    # Processing the call keyword arguments (line 1217)
    kwargs_281548 = {}
    # Getting the type of 'isinstance' (line 1217)
    isinstance_281545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1217)
    isinstance_call_result_281549 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 7), isinstance_281545, *[window_281546, string_types_281547], **kwargs_281548)
    
    
    
    # Call to type(...): (line 1217)
    # Processing the call arguments (line 1217)
    # Getting the type of 'window' (line 1217)
    window_281551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 48), 'window', False)
    # Processing the call keyword arguments (line 1217)
    kwargs_281552 = {}
    # Getting the type of 'type' (line 1217)
    type_281550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 43), 'type', False)
    # Calling type(args, kwargs) (line 1217)
    type_call_result_281553 = invoke(stypy.reporting.localization.Localization(__file__, 1217, 43), type_281550, *[window_281551], **kwargs_281552)
    
    # Getting the type of 'tuple' (line 1217)
    tuple_281554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1217, 59), 'tuple')
    # Applying the binary operator 'is' (line 1217)
    result_is__281555 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 43), 'is', type_call_result_281553, tuple_281554)
    
    # Applying the binary operator 'or' (line 1217)
    result_or_keyword_281556 = python_operator(stypy.reporting.localization.Localization(__file__, 1217, 7), 'or', isinstance_call_result_281549, result_is__281555)
    
    # Testing the type of an if condition (line 1217)
    if_condition_281557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1217, 4), result_or_keyword_281556)
    # Assigning a type to the variable 'if_condition_281557' (line 1217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1217, 4), 'if_condition_281557', if_condition_281557)
    # SSA begins for if statement (line 1217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1218):
    
    # Assigning a Call to a Name (line 1218):
    
    # Call to get_window(...): (line 1218)
    # Processing the call arguments (line 1218)
    # Getting the type of 'window' (line 1218)
    window_281559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 25), 'window', False)
    # Getting the type of 'nperseg' (line 1218)
    nperseg_281560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 33), 'nperseg', False)
    # Processing the call keyword arguments (line 1218)
    kwargs_281561 = {}
    # Getting the type of 'get_window' (line 1218)
    get_window_281558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1218, 14), 'get_window', False)
    # Calling get_window(args, kwargs) (line 1218)
    get_window_call_result_281562 = invoke(stypy.reporting.localization.Localization(__file__, 1218, 14), get_window_281558, *[window_281559, nperseg_281560], **kwargs_281561)
    
    # Assigning a type to the variable 'win' (line 1218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1218, 8), 'win', get_window_call_result_281562)
    # SSA branch for the else part of an if statement (line 1217)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1220):
    
    # Assigning a Call to a Name (line 1220):
    
    # Call to asarray(...): (line 1220)
    # Processing the call arguments (line 1220)
    # Getting the type of 'window' (line 1220)
    window_281565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 25), 'window', False)
    # Processing the call keyword arguments (line 1220)
    kwargs_281566 = {}
    # Getting the type of 'np' (line 1220)
    np_281563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1220)
    asarray_281564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1220, 14), np_281563, 'asarray')
    # Calling asarray(args, kwargs) (line 1220)
    asarray_call_result_281567 = invoke(stypy.reporting.localization.Localization(__file__, 1220, 14), asarray_281564, *[window_281565], **kwargs_281566)
    
    # Assigning a type to the variable 'win' (line 1220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1220, 8), 'win', asarray_call_result_281567)
    
    
    
    # Call to len(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'win' (line 1221)
    win_281569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 15), 'win', False)
    # Obtaining the member 'shape' of a type (line 1221)
    shape_281570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1221, 15), win_281569, 'shape')
    # Processing the call keyword arguments (line 1221)
    kwargs_281571 = {}
    # Getting the type of 'len' (line 1221)
    len_281568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 11), 'len', False)
    # Calling len(args, kwargs) (line 1221)
    len_call_result_281572 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 11), len_281568, *[shape_281570], **kwargs_281571)
    
    int_281573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 29), 'int')
    # Applying the binary operator '!=' (line 1221)
    result_ne_281574 = python_operator(stypy.reporting.localization.Localization(__file__, 1221, 11), '!=', len_call_result_281572, int_281573)
    
    # Testing the type of an if condition (line 1221)
    if_condition_281575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1221, 8), result_ne_281574)
    # Assigning a type to the variable 'if_condition_281575' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 8), 'if_condition_281575', if_condition_281575)
    # SSA begins for if statement (line 1221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1222)
    # Processing the call arguments (line 1222)
    str_281577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1222, 29), 'str', 'window must be 1-D')
    # Processing the call keyword arguments (line 1222)
    kwargs_281578 = {}
    # Getting the type of 'ValueError' (line 1222)
    ValueError_281576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1222)
    ValueError_call_result_281579 = invoke(stypy.reporting.localization.Localization(__file__, 1222, 18), ValueError_281576, *[str_281577], **kwargs_281578)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1222, 12), ValueError_call_result_281579, 'raise parameter', BaseException)
    # SSA join for if statement (line 1221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_281580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 21), 'int')
    # Getting the type of 'win' (line 1223)
    win_281581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 11), 'win')
    # Obtaining the member 'shape' of a type (line 1223)
    shape_281582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1223, 11), win_281581, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1223)
    getitem___281583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1223, 11), shape_281582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1223)
    subscript_call_result_281584 = invoke(stypy.reporting.localization.Localization(__file__, 1223, 11), getitem___281583, int_281580)
    
    # Getting the type of 'nperseg' (line 1223)
    nperseg_281585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 27), 'nperseg')
    # Applying the binary operator '!=' (line 1223)
    result_ne_281586 = python_operator(stypy.reporting.localization.Localization(__file__, 1223, 11), '!=', subscript_call_result_281584, nperseg_281585)
    
    # Testing the type of an if condition (line 1223)
    if_condition_281587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1223, 8), result_ne_281586)
    # Assigning a type to the variable 'if_condition_281587' (line 1223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1223, 8), 'if_condition_281587', if_condition_281587)
    # SSA begins for if statement (line 1223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1224)
    # Processing the call arguments (line 1224)
    
    # Call to format(...): (line 1224)
    # Processing the call arguments (line 1224)
    # Getting the type of 'nperseg' (line 1224)
    nperseg_281591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 69), 'nperseg', False)
    # Processing the call keyword arguments (line 1224)
    kwargs_281592 = {}
    str_281589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 29), 'str', 'window must have length of {0}')
    # Obtaining the member 'format' of a type (line 1224)
    format_281590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1224, 29), str_281589, 'format')
    # Calling format(args, kwargs) (line 1224)
    format_call_result_281593 = invoke(stypy.reporting.localization.Localization(__file__, 1224, 29), format_281590, *[nperseg_281591], **kwargs_281592)
    
    # Processing the call keyword arguments (line 1224)
    kwargs_281594 = {}
    # Getting the type of 'ValueError' (line 1224)
    ValueError_281588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1224)
    ValueError_call_result_281595 = invoke(stypy.reporting.localization.Localization(__file__, 1224, 18), ValueError_281588, *[format_call_result_281593], **kwargs_281594)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1224, 12), ValueError_call_result_281595, 'raise parameter', BaseException)
    # SSA join for if statement (line 1223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1217)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'input_onesided' (line 1226)
    input_onesided_281596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 7), 'input_onesided')
    # Testing the type of an if condition (line 1226)
    if_condition_281597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1226, 4), input_onesided_281596)
    # Assigning a type to the variable 'if_condition_281597' (line 1226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1226, 4), 'if_condition_281597', if_condition_281597)
    # SSA begins for if statement (line 1226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1227):
    
    # Assigning a Attribute to a Name (line 1227):
    # Getting the type of 'np' (line 1227)
    np_281598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 16), 'np')
    # Obtaining the member 'fft' of a type (line 1227)
    fft_281599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 16), np_281598, 'fft')
    # Obtaining the member 'irfft' of a type (line 1227)
    irfft_281600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 16), fft_281599, 'irfft')
    # Assigning a type to the variable 'ifunc' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'ifunc', irfft_281600)
    # SSA branch for the else part of an if statement (line 1226)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 1229):
    
    # Assigning a Attribute to a Name (line 1229):
    # Getting the type of 'fftpack' (line 1229)
    fftpack_281601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 16), 'fftpack')
    # Obtaining the member 'ifft' of a type (line 1229)
    ifft_281602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1229, 16), fftpack_281601, 'ifft')
    # Assigning a type to the variable 'ifunc' (line 1229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 8), 'ifunc', ifft_281602)
    # SSA join for if statement (line 1226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 1231):
    
    # Assigning a Subscript to a Name (line 1231):
    
    # Obtaining the type of the subscript
    Ellipsis_281603 = Ellipsis
    # Getting the type of 'nperseg' (line 1231)
    nperseg_281604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 46), 'nperseg')
    slice_281605 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1231, 12), None, nperseg_281604, None)
    slice_281606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1231, 12), None, None, None)
    
    # Call to ifunc(...): (line 1231)
    # Processing the call arguments (line 1231)
    # Getting the type of 'Zxx' (line 1231)
    Zxx_281608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 18), 'Zxx', False)
    # Processing the call keyword arguments (line 1231)
    int_281609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 28), 'int')
    keyword_281610 = int_281609
    # Getting the type of 'nfft' (line 1231)
    nfft_281611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 34), 'nfft', False)
    keyword_281612 = nfft_281611
    kwargs_281613 = {'n': keyword_281612, 'axis': keyword_281610}
    # Getting the type of 'ifunc' (line 1231)
    ifunc_281607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 12), 'ifunc', False)
    # Calling ifunc(args, kwargs) (line 1231)
    ifunc_call_result_281614 = invoke(stypy.reporting.localization.Localization(__file__, 1231, 12), ifunc_281607, *[Zxx_281608], **kwargs_281613)
    
    # Obtaining the member '__getitem__' of a type (line 1231)
    getitem___281615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1231, 12), ifunc_call_result_281614, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1231)
    subscript_call_result_281616 = invoke(stypy.reporting.localization.Localization(__file__, 1231, 12), getitem___281615, (Ellipsis_281603, slice_281605, slice_281606))
    
    # Assigning a type to the variable 'xsubs' (line 1231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 4), 'xsubs', subscript_call_result_281616)
    
    # Assigning a BinOp to a Name (line 1234):
    
    # Assigning a BinOp to a Name (line 1234):
    # Getting the type of 'nperseg' (line 1234)
    nperseg_281617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 19), 'nperseg')
    # Getting the type of 'nseg' (line 1234)
    nseg_281618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 30), 'nseg')
    int_281619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 35), 'int')
    # Applying the binary operator '-' (line 1234)
    result_sub_281620 = python_operator(stypy.reporting.localization.Localization(__file__, 1234, 30), '-', nseg_281618, int_281619)
    
    # Getting the type of 'nstep' (line 1234)
    nstep_281621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 38), 'nstep')
    # Applying the binary operator '*' (line 1234)
    result_mul_281622 = python_operator(stypy.reporting.localization.Localization(__file__, 1234, 29), '*', result_sub_281620, nstep_281621)
    
    # Applying the binary operator '+' (line 1234)
    result_add_281623 = python_operator(stypy.reporting.localization.Localization(__file__, 1234, 19), '+', nperseg_281617, result_mul_281622)
    
    # Assigning a type to the variable 'outputlength' (line 1234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 4), 'outputlength', result_add_281623)
    
    # Assigning a Call to a Name (line 1235):
    
    # Assigning a Call to a Name (line 1235):
    
    # Call to zeros(...): (line 1235)
    # Processing the call arguments (line 1235)
    
    # Call to list(...): (line 1235)
    # Processing the call arguments (line 1235)
    
    # Obtaining the type of the subscript
    int_281627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 33), 'int')
    slice_281628 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1235, 22), None, int_281627, None)
    # Getting the type of 'Zxx' (line 1235)
    Zxx_281629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 22), 'Zxx', False)
    # Obtaining the member 'shape' of a type (line 1235)
    shape_281630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 22), Zxx_281629, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___281631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 22), shape_281630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_281632 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 22), getitem___281631, slice_281628)
    
    # Processing the call keyword arguments (line 1235)
    kwargs_281633 = {}
    # Getting the type of 'list' (line 1235)
    list_281626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 17), 'list', False)
    # Calling list(args, kwargs) (line 1235)
    list_call_result_281634 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 17), list_281626, *[subscript_call_result_281632], **kwargs_281633)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1235)
    list_281635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1235)
    # Adding element type (line 1235)
    # Getting the type of 'outputlength' (line 1235)
    outputlength_281636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 39), 'outputlength', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1235, 38), list_281635, outputlength_281636)
    
    # Applying the binary operator '+' (line 1235)
    result_add_281637 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 17), '+', list_call_result_281634, list_281635)
    
    # Processing the call keyword arguments (line 1235)
    # Getting the type of 'xsubs' (line 1235)
    xsubs_281638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 60), 'xsubs', False)
    # Obtaining the member 'dtype' of a type (line 1235)
    dtype_281639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 60), xsubs_281638, 'dtype')
    keyword_281640 = dtype_281639
    kwargs_281641 = {'dtype': keyword_281640}
    # Getting the type of 'np' (line 1235)
    np_281624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1235)
    zeros_281625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 8), np_281624, 'zeros')
    # Calling zeros(args, kwargs) (line 1235)
    zeros_call_result_281642 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 8), zeros_281625, *[result_add_281637], **kwargs_281641)
    
    # Assigning a type to the variable 'x' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 4), 'x', zeros_call_result_281642)
    
    # Assigning a Call to a Name (line 1236):
    
    # Assigning a Call to a Name (line 1236):
    
    # Call to zeros(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'outputlength' (line 1236)
    outputlength_281645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 20), 'outputlength', False)
    # Processing the call keyword arguments (line 1236)
    # Getting the type of 'xsubs' (line 1236)
    xsubs_281646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 40), 'xsubs', False)
    # Obtaining the member 'dtype' of a type (line 1236)
    dtype_281647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 40), xsubs_281646, 'dtype')
    keyword_281648 = dtype_281647
    kwargs_281649 = {'dtype': keyword_281648}
    # Getting the type of 'np' (line 1236)
    np_281643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 11), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1236)
    zeros_281644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 11), np_281643, 'zeros')
    # Calling zeros(args, kwargs) (line 1236)
    zeros_call_result_281650 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 11), zeros_281644, *[outputlength_281645], **kwargs_281649)
    
    # Assigning a type to the variable 'norm' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'norm', zeros_call_result_281650)
    
    
    
    # Call to result_type(...): (line 1238)
    # Processing the call arguments (line 1238)
    # Getting the type of 'win' (line 1238)
    win_281653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 22), 'win', False)
    # Getting the type of 'xsubs' (line 1238)
    xsubs_281654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 27), 'xsubs', False)
    # Processing the call keyword arguments (line 1238)
    kwargs_281655 = {}
    # Getting the type of 'np' (line 1238)
    np_281651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 7), 'np', False)
    # Obtaining the member 'result_type' of a type (line 1238)
    result_type_281652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 7), np_281651, 'result_type')
    # Calling result_type(args, kwargs) (line 1238)
    result_type_call_result_281656 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 7), result_type_281652, *[win_281653, xsubs_281654], **kwargs_281655)
    
    # Getting the type of 'xsubs' (line 1238)
    xsubs_281657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 37), 'xsubs')
    # Obtaining the member 'dtype' of a type (line 1238)
    dtype_281658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 37), xsubs_281657, 'dtype')
    # Applying the binary operator '!=' (line 1238)
    result_ne_281659 = python_operator(stypy.reporting.localization.Localization(__file__, 1238, 7), '!=', result_type_call_result_281656, dtype_281658)
    
    # Testing the type of an if condition (line 1238)
    if_condition_281660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1238, 4), result_ne_281659)
    # Assigning a type to the variable 'if_condition_281660' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 4), 'if_condition_281660', if_condition_281660)
    # SSA begins for if statement (line 1238)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1239):
    
    # Assigning a Call to a Name (line 1239):
    
    # Call to astype(...): (line 1239)
    # Processing the call arguments (line 1239)
    # Getting the type of 'xsubs' (line 1239)
    xsubs_281663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 25), 'xsubs', False)
    # Obtaining the member 'dtype' of a type (line 1239)
    dtype_281664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 25), xsubs_281663, 'dtype')
    # Processing the call keyword arguments (line 1239)
    kwargs_281665 = {}
    # Getting the type of 'win' (line 1239)
    win_281661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 14), 'win', False)
    # Obtaining the member 'astype' of a type (line 1239)
    astype_281662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1239, 14), win_281661, 'astype')
    # Calling astype(args, kwargs) (line 1239)
    astype_call_result_281666 = invoke(stypy.reporting.localization.Localization(__file__, 1239, 14), astype_281662, *[dtype_281664], **kwargs_281665)
    
    # Assigning a type to the variable 'win' (line 1239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 8), 'win', astype_call_result_281666)
    # SSA join for if statement (line 1238)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'xsubs' (line 1241)
    xsubs_281667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 4), 'xsubs')
    
    # Call to sum(...): (line 1241)
    # Processing the call keyword arguments (line 1241)
    kwargs_281670 = {}
    # Getting the type of 'win' (line 1241)
    win_281668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 13), 'win', False)
    # Obtaining the member 'sum' of a type (line 1241)
    sum_281669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1241, 13), win_281668, 'sum')
    # Calling sum(args, kwargs) (line 1241)
    sum_call_result_281671 = invoke(stypy.reporting.localization.Localization(__file__, 1241, 13), sum_281669, *[], **kwargs_281670)
    
    # Applying the binary operator '*=' (line 1241)
    result_imul_281672 = python_operator(stypy.reporting.localization.Localization(__file__, 1241, 4), '*=', xsubs_281667, sum_call_result_281671)
    # Assigning a type to the variable 'xsubs' (line 1241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1241, 4), 'xsubs', result_imul_281672)
    
    
    
    # Call to range(...): (line 1245)
    # Processing the call arguments (line 1245)
    # Getting the type of 'nseg' (line 1245)
    nseg_281674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 20), 'nseg', False)
    # Processing the call keyword arguments (line 1245)
    kwargs_281675 = {}
    # Getting the type of 'range' (line 1245)
    range_281673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 14), 'range', False)
    # Calling range(args, kwargs) (line 1245)
    range_call_result_281676 = invoke(stypy.reporting.localization.Localization(__file__, 1245, 14), range_281673, *[nseg_281674], **kwargs_281675)
    
    # Testing the type of a for loop iterable (line 1245)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1245, 4), range_call_result_281676)
    # Getting the type of the for loop variable (line 1245)
    for_loop_var_281677 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1245, 4), range_call_result_281676)
    # Assigning a type to the variable 'ii' (line 1245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1245, 4), 'ii', for_loop_var_281677)
    # SSA begins for a for statement (line 1245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'x' (line 1247)
    x_281678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'x')
    
    # Obtaining the type of the subscript
    Ellipsis_281679 = Ellipsis
    # Getting the type of 'ii' (line 1247)
    ii_281680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 15), 'ii')
    # Getting the type of 'nstep' (line 1247)
    nstep_281681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 18), 'nstep')
    # Applying the binary operator '*' (line 1247)
    result_mul_281682 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 15), '*', ii_281680, nstep_281681)
    
    # Getting the type of 'ii' (line 1247)
    ii_281683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 24), 'ii')
    # Getting the type of 'nstep' (line 1247)
    nstep_281684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 27), 'nstep')
    # Applying the binary operator '*' (line 1247)
    result_mul_281685 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 24), '*', ii_281683, nstep_281684)
    
    # Getting the type of 'nperseg' (line 1247)
    nperseg_281686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 33), 'nperseg')
    # Applying the binary operator '+' (line 1247)
    result_add_281687 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 24), '+', result_mul_281685, nperseg_281686)
    
    slice_281688 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1247, 8), result_mul_281682, result_add_281687, None)
    # Getting the type of 'x' (line 1247)
    x_281689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'x')
    # Obtaining the member '__getitem__' of a type (line 1247)
    getitem___281690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 8), x_281689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1247)
    subscript_call_result_281691 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 8), getitem___281690, (Ellipsis_281679, slice_281688))
    
    
    # Obtaining the type of the subscript
    Ellipsis_281692 = Ellipsis
    # Getting the type of 'ii' (line 1247)
    ii_281693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 56), 'ii')
    # Getting the type of 'xsubs' (line 1247)
    xsubs_281694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 45), 'xsubs')
    # Obtaining the member '__getitem__' of a type (line 1247)
    getitem___281695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 45), xsubs_281694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1247)
    subscript_call_result_281696 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 45), getitem___281695, (Ellipsis_281692, ii_281693))
    
    # Getting the type of 'win' (line 1247)
    win_281697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 62), 'win')
    # Applying the binary operator '*' (line 1247)
    result_mul_281698 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 45), '*', subscript_call_result_281696, win_281697)
    
    # Applying the binary operator '+=' (line 1247)
    result_iadd_281699 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 8), '+=', subscript_call_result_281691, result_mul_281698)
    # Getting the type of 'x' (line 1247)
    x_281700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'x')
    Ellipsis_281701 = Ellipsis
    # Getting the type of 'ii' (line 1247)
    ii_281702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 15), 'ii')
    # Getting the type of 'nstep' (line 1247)
    nstep_281703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 18), 'nstep')
    # Applying the binary operator '*' (line 1247)
    result_mul_281704 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 15), '*', ii_281702, nstep_281703)
    
    # Getting the type of 'ii' (line 1247)
    ii_281705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 24), 'ii')
    # Getting the type of 'nstep' (line 1247)
    nstep_281706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 27), 'nstep')
    # Applying the binary operator '*' (line 1247)
    result_mul_281707 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 24), '*', ii_281705, nstep_281706)
    
    # Getting the type of 'nperseg' (line 1247)
    nperseg_281708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 33), 'nperseg')
    # Applying the binary operator '+' (line 1247)
    result_add_281709 = python_operator(stypy.reporting.localization.Localization(__file__, 1247, 24), '+', result_mul_281707, nperseg_281708)
    
    slice_281710 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1247, 8), result_mul_281704, result_add_281709, None)
    # Storing an element on a container (line 1247)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1247, 8), x_281700, ((Ellipsis_281701, slice_281710), result_iadd_281699))
    
    
    # Getting the type of 'norm' (line 1248)
    norm_281711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'norm')
    
    # Obtaining the type of the subscript
    Ellipsis_281712 = Ellipsis
    # Getting the type of 'ii' (line 1248)
    ii_281713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 18), 'ii')
    # Getting the type of 'nstep' (line 1248)
    nstep_281714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 21), 'nstep')
    # Applying the binary operator '*' (line 1248)
    result_mul_281715 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 18), '*', ii_281713, nstep_281714)
    
    # Getting the type of 'ii' (line 1248)
    ii_281716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 27), 'ii')
    # Getting the type of 'nstep' (line 1248)
    nstep_281717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 30), 'nstep')
    # Applying the binary operator '*' (line 1248)
    result_mul_281718 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 27), '*', ii_281716, nstep_281717)
    
    # Getting the type of 'nperseg' (line 1248)
    nperseg_281719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 36), 'nperseg')
    # Applying the binary operator '+' (line 1248)
    result_add_281720 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 27), '+', result_mul_281718, nperseg_281719)
    
    slice_281721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1248, 8), result_mul_281715, result_add_281720, None)
    # Getting the type of 'norm' (line 1248)
    norm_281722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'norm')
    # Obtaining the member '__getitem__' of a type (line 1248)
    getitem___281723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1248, 8), norm_281722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1248)
    subscript_call_result_281724 = invoke(stypy.reporting.localization.Localization(__file__, 1248, 8), getitem___281723, (Ellipsis_281712, slice_281721))
    
    # Getting the type of 'win' (line 1248)
    win_281725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 48), 'win')
    int_281726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1248, 53), 'int')
    # Applying the binary operator '**' (line 1248)
    result_pow_281727 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 48), '**', win_281725, int_281726)
    
    # Applying the binary operator '+=' (line 1248)
    result_iadd_281728 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 8), '+=', subscript_call_result_281724, result_pow_281727)
    # Getting the type of 'norm' (line 1248)
    norm_281729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'norm')
    Ellipsis_281730 = Ellipsis
    # Getting the type of 'ii' (line 1248)
    ii_281731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 18), 'ii')
    # Getting the type of 'nstep' (line 1248)
    nstep_281732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 21), 'nstep')
    # Applying the binary operator '*' (line 1248)
    result_mul_281733 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 18), '*', ii_281731, nstep_281732)
    
    # Getting the type of 'ii' (line 1248)
    ii_281734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 27), 'ii')
    # Getting the type of 'nstep' (line 1248)
    nstep_281735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 30), 'nstep')
    # Applying the binary operator '*' (line 1248)
    result_mul_281736 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 27), '*', ii_281734, nstep_281735)
    
    # Getting the type of 'nperseg' (line 1248)
    nperseg_281737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 36), 'nperseg')
    # Applying the binary operator '+' (line 1248)
    result_add_281738 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 27), '+', result_mul_281736, nperseg_281737)
    
    slice_281739 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1248, 8), result_mul_281733, result_add_281738, None)
    # Storing an element on a container (line 1248)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1248, 8), norm_281729, ((Ellipsis_281730, slice_281739), result_iadd_281728))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'x' (line 1251)
    x_281740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 4), 'x')
    
    # Call to where(...): (line 1251)
    # Processing the call arguments (line 1251)
    
    # Getting the type of 'norm' (line 1251)
    norm_281743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 18), 'norm', False)
    float_281744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1251, 25), 'float')
    # Applying the binary operator '>' (line 1251)
    result_gt_281745 = python_operator(stypy.reporting.localization.Localization(__file__, 1251, 18), '>', norm_281743, float_281744)
    
    # Getting the type of 'norm' (line 1251)
    norm_281746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 32), 'norm', False)
    float_281747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1251, 38), 'float')
    # Processing the call keyword arguments (line 1251)
    kwargs_281748 = {}
    # Getting the type of 'np' (line 1251)
    np_281741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 9), 'np', False)
    # Obtaining the member 'where' of a type (line 1251)
    where_281742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 9), np_281741, 'where')
    # Calling where(args, kwargs) (line 1251)
    where_call_result_281749 = invoke(stypy.reporting.localization.Localization(__file__, 1251, 9), where_281742, *[result_gt_281745, norm_281746, float_281747], **kwargs_281748)
    
    # Applying the binary operator 'div=' (line 1251)
    result_div_281750 = python_operator(stypy.reporting.localization.Localization(__file__, 1251, 4), 'div=', x_281740, where_call_result_281749)
    # Assigning a type to the variable 'x' (line 1251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1251, 4), 'x', result_div_281750)
    
    
    # Getting the type of 'boundary' (line 1254)
    boundary_281751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1254, 7), 'boundary')
    # Testing the type of an if condition (line 1254)
    if_condition_281752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1254, 4), boundary_281751)
    # Assigning a type to the variable 'if_condition_281752' (line 1254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1254, 4), 'if_condition_281752', if_condition_281752)
    # SSA begins for if statement (line 1254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1255):
    
    # Assigning a Subscript to a Name (line 1255):
    
    # Obtaining the type of the subscript
    Ellipsis_281753 = Ellipsis
    # Getting the type of 'nperseg' (line 1255)
    nperseg_281754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 19), 'nperseg')
    int_281755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1255, 28), 'int')
    # Applying the binary operator '//' (line 1255)
    result_floordiv_281756 = python_operator(stypy.reporting.localization.Localization(__file__, 1255, 19), '//', nperseg_281754, int_281755)
    
    
    # Getting the type of 'nperseg' (line 1255)
    nperseg_281757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 32), 'nperseg')
    int_281758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1255, 41), 'int')
    # Applying the binary operator '//' (line 1255)
    result_floordiv_281759 = python_operator(stypy.reporting.localization.Localization(__file__, 1255, 32), '//', nperseg_281757, int_281758)
    
    # Applying the 'usub' unary operator (line 1255)
    result___neg___281760 = python_operator(stypy.reporting.localization.Localization(__file__, 1255, 30), 'usub', result_floordiv_281759)
    
    slice_281761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1255, 12), result_floordiv_281756, result___neg___281760, None)
    # Getting the type of 'x' (line 1255)
    x_281762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 1255)
    getitem___281763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1255, 12), x_281762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1255)
    subscript_call_result_281764 = invoke(stypy.reporting.localization.Localization(__file__, 1255, 12), getitem___281763, (Ellipsis_281753, slice_281761))
    
    # Assigning a type to the variable 'x' (line 1255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1255, 8), 'x', subscript_call_result_281764)
    # SSA join for if statement (line 1254)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'input_onesided' (line 1257)
    input_onesided_281765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 7), 'input_onesided')
    # Testing the type of an if condition (line 1257)
    if_condition_281766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1257, 4), input_onesided_281765)
    # Assigning a type to the variable 'if_condition_281766' (line 1257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1257, 4), 'if_condition_281766', if_condition_281766)
    # SSA begins for if statement (line 1257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1258):
    
    # Assigning a Attribute to a Name (line 1258):
    # Getting the type of 'x' (line 1258)
    x_281767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 12), 'x')
    # Obtaining the member 'real' of a type (line 1258)
    real_281768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 12), x_281767, 'real')
    # Assigning a type to the variable 'x' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'x', real_281768)
    # SSA join for if statement (line 1257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1261)
    x_281769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1261)
    ndim_281770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1261, 7), x_281769, 'ndim')
    int_281771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1261, 16), 'int')
    # Applying the binary operator '>' (line 1261)
    result_gt_281772 = python_operator(stypy.reporting.localization.Localization(__file__, 1261, 7), '>', ndim_281770, int_281771)
    
    # Testing the type of an if condition (line 1261)
    if_condition_281773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1261, 4), result_gt_281772)
    # Assigning a type to the variable 'if_condition_281773' (line 1261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1261, 4), 'if_condition_281773', if_condition_281773)
    # SSA begins for if statement (line 1261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'time_axis' (line 1262)
    time_axis_281774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 11), 'time_axis')
    # Getting the type of 'Zxx' (line 1262)
    Zxx_281775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 24), 'Zxx')
    # Obtaining the member 'ndim' of a type (line 1262)
    ndim_281776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1262, 24), Zxx_281775, 'ndim')
    int_281777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1262, 33), 'int')
    # Applying the binary operator '-' (line 1262)
    result_sub_281778 = python_operator(stypy.reporting.localization.Localization(__file__, 1262, 24), '-', ndim_281776, int_281777)
    
    # Applying the binary operator '!=' (line 1262)
    result_ne_281779 = python_operator(stypy.reporting.localization.Localization(__file__, 1262, 11), '!=', time_axis_281774, result_sub_281778)
    
    # Testing the type of an if condition (line 1262)
    if_condition_281780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1262, 8), result_ne_281779)
    # Assigning a type to the variable 'if_condition_281780' (line 1262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1262, 8), 'if_condition_281780', if_condition_281780)
    # SSA begins for if statement (line 1262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'freq_axis' (line 1263)
    freq_axis_281781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 15), 'freq_axis')
    # Getting the type of 'time_axis' (line 1263)
    time_axis_281782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 27), 'time_axis')
    # Applying the binary operator '<' (line 1263)
    result_lt_281783 = python_operator(stypy.reporting.localization.Localization(__file__, 1263, 15), '<', freq_axis_281781, time_axis_281782)
    
    # Testing the type of an if condition (line 1263)
    if_condition_281784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1263, 12), result_lt_281783)
    # Assigning a type to the variable 'if_condition_281784' (line 1263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1263, 12), 'if_condition_281784', if_condition_281784)
    # SSA begins for if statement (line 1263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'time_axis' (line 1264)
    time_axis_281785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 16), 'time_axis')
    int_281786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1264, 29), 'int')
    # Applying the binary operator '-=' (line 1264)
    result_isub_281787 = python_operator(stypy.reporting.localization.Localization(__file__, 1264, 16), '-=', time_axis_281785, int_281786)
    # Assigning a type to the variable 'time_axis' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 16), 'time_axis', result_isub_281787)
    
    # SSA join for if statement (line 1263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1265):
    
    # Assigning a Call to a Name (line 1265):
    
    # Call to rollaxis(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'x' (line 1265)
    x_281790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), 'x', False)
    int_281791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 31), 'int')
    # Getting the type of 'time_axis' (line 1265)
    time_axis_281792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 35), 'time_axis', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_281793 = {}
    # Getting the type of 'np' (line 1265)
    np_281788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 16), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1265)
    rollaxis_281789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 16), np_281788, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1265)
    rollaxis_call_result_281794 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 16), rollaxis_281789, *[x_281790, int_281791, time_axis_281792], **kwargs_281793)
    
    # Assigning a type to the variable 'x' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 12), 'x', rollaxis_call_result_281794)
    # SSA join for if statement (line 1262)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1267):
    
    # Assigning a BinOp to a Name (line 1267):
    
    # Call to arange(...): (line 1267)
    # Processing the call arguments (line 1267)
    
    # Obtaining the type of the subscript
    int_281797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1267, 29), 'int')
    # Getting the type of 'x' (line 1267)
    x_281798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 21), 'x', False)
    # Obtaining the member 'shape' of a type (line 1267)
    shape_281799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 21), x_281798, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1267)
    getitem___281800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 21), shape_281799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1267)
    subscript_call_result_281801 = invoke(stypy.reporting.localization.Localization(__file__, 1267, 21), getitem___281800, int_281797)
    
    # Processing the call keyword arguments (line 1267)
    kwargs_281802 = {}
    # Getting the type of 'np' (line 1267)
    np_281795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 1267)
    arange_281796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 11), np_281795, 'arange')
    # Calling arange(args, kwargs) (line 1267)
    arange_call_result_281803 = invoke(stypy.reporting.localization.Localization(__file__, 1267, 11), arange_281796, *[subscript_call_result_281801], **kwargs_281802)
    
    
    # Call to float(...): (line 1267)
    # Processing the call arguments (line 1267)
    # Getting the type of 'fs' (line 1267)
    fs_281805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 39), 'fs', False)
    # Processing the call keyword arguments (line 1267)
    kwargs_281806 = {}
    # Getting the type of 'float' (line 1267)
    float_281804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 33), 'float', False)
    # Calling float(args, kwargs) (line 1267)
    float_call_result_281807 = invoke(stypy.reporting.localization.Localization(__file__, 1267, 33), float_281804, *[fs_281805], **kwargs_281806)
    
    # Applying the binary operator 'div' (line 1267)
    result_div_281808 = python_operator(stypy.reporting.localization.Localization(__file__, 1267, 11), 'div', arange_call_result_281803, float_call_result_281807)
    
    # Assigning a type to the variable 'time' (line 1267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1267, 4), 'time', result_div_281808)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1268)
    tuple_281809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1268, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1268)
    # Adding element type (line 1268)
    # Getting the type of 'time' (line 1268)
    time_281810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 11), 'time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1268, 11), tuple_281809, time_281810)
    # Adding element type (line 1268)
    # Getting the type of 'x' (line 1268)
    x_281811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 17), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1268, 11), tuple_281809, x_281811)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1268, 4), 'stypy_return_type', tuple_281809)
    
    # ################# End of 'istft(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'istft' in the type store
    # Getting the type of 'stypy_return_type' (line 995)
    stypy_return_type_281812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_281812)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'istft'
    return stypy_return_type_281812

# Assigning a type to the variable 'istft' (line 995)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 0), 'istft', istft)

@norecursion
def coherence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_281813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1271, 23), 'float')
    str_281814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1271, 35), 'str', 'hann')
    # Getting the type of 'None' (line 1271)
    None_281815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 51), 'None')
    # Getting the type of 'None' (line 1271)
    None_281816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 66), 'None')
    # Getting the type of 'None' (line 1272)
    None_281817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1272, 19), 'None')
    str_281818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1272, 33), 'str', 'constant')
    int_281819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1272, 50), 'int')
    defaults = [float_281813, str_281814, None_281815, None_281816, None_281817, str_281818, int_281819]
    # Create a new context for function 'coherence'
    module_type_store = module_type_store.open_function_context('coherence', 1271, 0, False)
    
    # Passed parameters checking function
    coherence.stypy_localization = localization
    coherence.stypy_type_of_self = None
    coherence.stypy_type_store = module_type_store
    coherence.stypy_function_name = 'coherence'
    coherence.stypy_param_names_list = ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'axis']
    coherence.stypy_varargs_param_name = None
    coherence.stypy_kwargs_param_name = None
    coherence.stypy_call_defaults = defaults
    coherence.stypy_call_varargs = varargs
    coherence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'coherence', ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'coherence', localization, ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'coherence(...)' code ##################

    str_281820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1376, (-1)), 'str', '\n    Estimate the magnitude squared coherence estimate, Cxy, of\n    discrete-time signals X and Y using Welch\'s method.\n\n    ``Cxy = abs(Pxy)**2/(Pxx*Pyy)``, where `Pxx` and `Pyy` are power\n    spectral density estimates of X and Y, and `Pxy` is the cross\n    spectral density estimate of X and Y.\n\n    Parameters\n    ----------\n    x : array_like\n        Time series of measurement values\n    y : array_like\n        Time series of measurement values\n    fs : float, optional\n        Sampling frequency of the `x` and `y` time series. Defaults\n        to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap: int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    axis : int, optional\n        Axis along which the coherence is computed for both inputs; the\n        default is over the last axis (i.e. ``axis=-1``).\n\n    Returns\n    -------\n    f : ndarray\n        Array of sample frequencies.\n    Cxy : ndarray\n        Magnitude squared coherence of x and y.\n\n    See Also\n    --------\n    periodogram: Simple, optionally modified periodogram\n    lombscargle: Lomb-Scargle periodogram for unevenly sampled data\n    welch: Power spectral density by Welch\'s method.\n    csd: Cross spectral density by Welch\'s method.\n\n    Notes\n    --------\n    An appropriate amount of overlap will depend on the choice of window\n    and on your requirements. For the default Hann window an overlap of\n    50% is a reasonable trade off between accurately estimating the\n    signal power, while not over counting any of the data. Narrower\n    windows may require a larger overlap.\n\n    .. versionadded:: 0.16.0\n\n    References\n    ----------\n    .. [1] P. Welch, "The use of the fast Fourier transform for the\n           estimation of power spectra: A method based on time averaging\n           over short, modified periodograms", IEEE Trans. Audio\n           Electroacoust. vol. 15, pp. 70-73, 1967.\n    .. [2] Stoica, Petre, and Randolph Moses, "Spectral Analysis of\n           Signals" Prentice Hall, 2005\n\n    Examples\n    --------\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n\n    Generate two test signals with some common features.\n\n    >>> fs = 10e3\n    >>> N = 1e5\n    >>> amp = 20\n    >>> freq = 1234.0\n    >>> noise_power = 0.001 * fs / 2\n    >>> time = np.arange(N) / fs\n    >>> b, a = signal.butter(2, 0.25, \'low\')\n    >>> x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)\n    >>> y = signal.lfilter(b, a, x)\n    >>> x += amp*np.sin(2*np.pi*freq*time)\n    >>> y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)\n\n    Compute and plot the coherence.\n\n    >>> f, Cxy = signal.coherence(x, y, fs, nperseg=1024)\n    >>> plt.semilogy(f, Cxy)\n    >>> plt.xlabel(\'frequency [Hz]\')\n    >>> plt.ylabel(\'Coherence\')\n    >>> plt.show()\n    ')
    
    # Assigning a Call to a Tuple (line 1378):
    
    # Assigning a Subscript to a Name (line 1378):
    
    # Obtaining the type of the subscript
    int_281821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 4), 'int')
    
    # Call to welch(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'x' (line 1378)
    x_281823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 23), 'x', False)
    # Getting the type of 'fs' (line 1378)
    fs_281824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 26), 'fs', False)
    # Getting the type of 'window' (line 1378)
    window_281825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 30), 'window', False)
    # Getting the type of 'nperseg' (line 1378)
    nperseg_281826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 38), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1378)
    noverlap_281827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 47), 'noverlap', False)
    # Getting the type of 'nfft' (line 1378)
    nfft_281828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 57), 'nfft', False)
    # Getting the type of 'detrend' (line 1378)
    detrend_281829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 63), 'detrend', False)
    # Processing the call keyword arguments (line 1378)
    # Getting the type of 'axis' (line 1379)
    axis_281830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 28), 'axis', False)
    keyword_281831 = axis_281830
    kwargs_281832 = {'axis': keyword_281831}
    # Getting the type of 'welch' (line 1378)
    welch_281822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 17), 'welch', False)
    # Calling welch(args, kwargs) (line 1378)
    welch_call_result_281833 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 17), welch_281822, *[x_281823, fs_281824, window_281825, nperseg_281826, noverlap_281827, nfft_281828, detrend_281829], **kwargs_281832)
    
    # Obtaining the member '__getitem__' of a type (line 1378)
    getitem___281834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1378, 4), welch_call_result_281833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1378)
    subscript_call_result_281835 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 4), getitem___281834, int_281821)
    
    # Assigning a type to the variable 'tuple_var_assignment_280481' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'tuple_var_assignment_280481', subscript_call_result_281835)
    
    # Assigning a Subscript to a Name (line 1378):
    
    # Obtaining the type of the subscript
    int_281836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1378, 4), 'int')
    
    # Call to welch(...): (line 1378)
    # Processing the call arguments (line 1378)
    # Getting the type of 'x' (line 1378)
    x_281838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 23), 'x', False)
    # Getting the type of 'fs' (line 1378)
    fs_281839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 26), 'fs', False)
    # Getting the type of 'window' (line 1378)
    window_281840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 30), 'window', False)
    # Getting the type of 'nperseg' (line 1378)
    nperseg_281841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 38), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1378)
    noverlap_281842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 47), 'noverlap', False)
    # Getting the type of 'nfft' (line 1378)
    nfft_281843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 57), 'nfft', False)
    # Getting the type of 'detrend' (line 1378)
    detrend_281844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 63), 'detrend', False)
    # Processing the call keyword arguments (line 1378)
    # Getting the type of 'axis' (line 1379)
    axis_281845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 28), 'axis', False)
    keyword_281846 = axis_281845
    kwargs_281847 = {'axis': keyword_281846}
    # Getting the type of 'welch' (line 1378)
    welch_281837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 17), 'welch', False)
    # Calling welch(args, kwargs) (line 1378)
    welch_call_result_281848 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 17), welch_281837, *[x_281838, fs_281839, window_281840, nperseg_281841, noverlap_281842, nfft_281843, detrend_281844], **kwargs_281847)
    
    # Obtaining the member '__getitem__' of a type (line 1378)
    getitem___281849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1378, 4), welch_call_result_281848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1378)
    subscript_call_result_281850 = invoke(stypy.reporting.localization.Localization(__file__, 1378, 4), getitem___281849, int_281836)
    
    # Assigning a type to the variable 'tuple_var_assignment_280482' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'tuple_var_assignment_280482', subscript_call_result_281850)
    
    # Assigning a Name to a Name (line 1378):
    # Getting the type of 'tuple_var_assignment_280481' (line 1378)
    tuple_var_assignment_280481_281851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'tuple_var_assignment_280481')
    # Assigning a type to the variable 'freqs' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'freqs', tuple_var_assignment_280481_281851)
    
    # Assigning a Name to a Name (line 1378):
    # Getting the type of 'tuple_var_assignment_280482' (line 1378)
    tuple_var_assignment_280482_281852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'tuple_var_assignment_280482')
    # Assigning a type to the variable 'Pxx' (line 1378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1378, 11), 'Pxx', tuple_var_assignment_280482_281852)
    
    # Assigning a Call to a Tuple (line 1380):
    
    # Assigning a Subscript to a Name (line 1380):
    
    # Obtaining the type of the subscript
    int_281853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1380, 4), 'int')
    
    # Call to welch(...): (line 1380)
    # Processing the call arguments (line 1380)
    # Getting the type of 'y' (line 1380)
    y_281855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 19), 'y', False)
    # Getting the type of 'fs' (line 1380)
    fs_281856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 22), 'fs', False)
    # Getting the type of 'window' (line 1380)
    window_281857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 26), 'window', False)
    # Getting the type of 'nperseg' (line 1380)
    nperseg_281858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 34), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1380)
    noverlap_281859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 43), 'noverlap', False)
    # Getting the type of 'nfft' (line 1380)
    nfft_281860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 53), 'nfft', False)
    # Getting the type of 'detrend' (line 1380)
    detrend_281861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 59), 'detrend', False)
    # Processing the call keyword arguments (line 1380)
    # Getting the type of 'axis' (line 1380)
    axis_281862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 73), 'axis', False)
    keyword_281863 = axis_281862
    kwargs_281864 = {'axis': keyword_281863}
    # Getting the type of 'welch' (line 1380)
    welch_281854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 13), 'welch', False)
    # Calling welch(args, kwargs) (line 1380)
    welch_call_result_281865 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 13), welch_281854, *[y_281855, fs_281856, window_281857, nperseg_281858, noverlap_281859, nfft_281860, detrend_281861], **kwargs_281864)
    
    # Obtaining the member '__getitem__' of a type (line 1380)
    getitem___281866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 4), welch_call_result_281865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1380)
    subscript_call_result_281867 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 4), getitem___281866, int_281853)
    
    # Assigning a type to the variable 'tuple_var_assignment_280483' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), 'tuple_var_assignment_280483', subscript_call_result_281867)
    
    # Assigning a Subscript to a Name (line 1380):
    
    # Obtaining the type of the subscript
    int_281868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1380, 4), 'int')
    
    # Call to welch(...): (line 1380)
    # Processing the call arguments (line 1380)
    # Getting the type of 'y' (line 1380)
    y_281870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 19), 'y', False)
    # Getting the type of 'fs' (line 1380)
    fs_281871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 22), 'fs', False)
    # Getting the type of 'window' (line 1380)
    window_281872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 26), 'window', False)
    # Getting the type of 'nperseg' (line 1380)
    nperseg_281873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 34), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1380)
    noverlap_281874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 43), 'noverlap', False)
    # Getting the type of 'nfft' (line 1380)
    nfft_281875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 53), 'nfft', False)
    # Getting the type of 'detrend' (line 1380)
    detrend_281876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 59), 'detrend', False)
    # Processing the call keyword arguments (line 1380)
    # Getting the type of 'axis' (line 1380)
    axis_281877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 73), 'axis', False)
    keyword_281878 = axis_281877
    kwargs_281879 = {'axis': keyword_281878}
    # Getting the type of 'welch' (line 1380)
    welch_281869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 13), 'welch', False)
    # Calling welch(args, kwargs) (line 1380)
    welch_call_result_281880 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 13), welch_281869, *[y_281870, fs_281871, window_281872, nperseg_281873, noverlap_281874, nfft_281875, detrend_281876], **kwargs_281879)
    
    # Obtaining the member '__getitem__' of a type (line 1380)
    getitem___281881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1380, 4), welch_call_result_281880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1380)
    subscript_call_result_281882 = invoke(stypy.reporting.localization.Localization(__file__, 1380, 4), getitem___281881, int_281868)
    
    # Assigning a type to the variable 'tuple_var_assignment_280484' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), 'tuple_var_assignment_280484', subscript_call_result_281882)
    
    # Assigning a Name to a Name (line 1380):
    # Getting the type of 'tuple_var_assignment_280483' (line 1380)
    tuple_var_assignment_280483_281883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), 'tuple_var_assignment_280483')
    # Assigning a type to the variable '_' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), '_', tuple_var_assignment_280483_281883)
    
    # Assigning a Name to a Name (line 1380):
    # Getting the type of 'tuple_var_assignment_280484' (line 1380)
    tuple_var_assignment_280484_281884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), 'tuple_var_assignment_280484')
    # Assigning a type to the variable 'Pyy' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 7), 'Pyy', tuple_var_assignment_280484_281884)
    
    # Assigning a Call to a Tuple (line 1381):
    
    # Assigning a Subscript to a Name (line 1381):
    
    # Obtaining the type of the subscript
    int_281885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 4), 'int')
    
    # Call to csd(...): (line 1381)
    # Processing the call arguments (line 1381)
    # Getting the type of 'x' (line 1381)
    x_281887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 17), 'x', False)
    # Getting the type of 'y' (line 1381)
    y_281888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 20), 'y', False)
    # Getting the type of 'fs' (line 1381)
    fs_281889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 23), 'fs', False)
    # Getting the type of 'window' (line 1381)
    window_281890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 27), 'window', False)
    # Getting the type of 'nperseg' (line 1381)
    nperseg_281891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 35), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1381)
    noverlap_281892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 1381)
    nfft_281893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 1381)
    detrend_281894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 60), 'detrend', False)
    # Processing the call keyword arguments (line 1381)
    # Getting the type of 'axis' (line 1381)
    axis_281895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 74), 'axis', False)
    keyword_281896 = axis_281895
    kwargs_281897 = {'axis': keyword_281896}
    # Getting the type of 'csd' (line 1381)
    csd_281886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 13), 'csd', False)
    # Calling csd(args, kwargs) (line 1381)
    csd_call_result_281898 = invoke(stypy.reporting.localization.Localization(__file__, 1381, 13), csd_281886, *[x_281887, y_281888, fs_281889, window_281890, nperseg_281891, noverlap_281892, nfft_281893, detrend_281894], **kwargs_281897)
    
    # Obtaining the member '__getitem__' of a type (line 1381)
    getitem___281899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1381, 4), csd_call_result_281898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1381)
    subscript_call_result_281900 = invoke(stypy.reporting.localization.Localization(__file__, 1381, 4), getitem___281899, int_281885)
    
    # Assigning a type to the variable 'tuple_var_assignment_280485' (line 1381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), 'tuple_var_assignment_280485', subscript_call_result_281900)
    
    # Assigning a Subscript to a Name (line 1381):
    
    # Obtaining the type of the subscript
    int_281901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, 4), 'int')
    
    # Call to csd(...): (line 1381)
    # Processing the call arguments (line 1381)
    # Getting the type of 'x' (line 1381)
    x_281903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 17), 'x', False)
    # Getting the type of 'y' (line 1381)
    y_281904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 20), 'y', False)
    # Getting the type of 'fs' (line 1381)
    fs_281905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 23), 'fs', False)
    # Getting the type of 'window' (line 1381)
    window_281906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 27), 'window', False)
    # Getting the type of 'nperseg' (line 1381)
    nperseg_281907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 35), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1381)
    noverlap_281908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 44), 'noverlap', False)
    # Getting the type of 'nfft' (line 1381)
    nfft_281909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 54), 'nfft', False)
    # Getting the type of 'detrend' (line 1381)
    detrend_281910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 60), 'detrend', False)
    # Processing the call keyword arguments (line 1381)
    # Getting the type of 'axis' (line 1381)
    axis_281911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 74), 'axis', False)
    keyword_281912 = axis_281911
    kwargs_281913 = {'axis': keyword_281912}
    # Getting the type of 'csd' (line 1381)
    csd_281902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 13), 'csd', False)
    # Calling csd(args, kwargs) (line 1381)
    csd_call_result_281914 = invoke(stypy.reporting.localization.Localization(__file__, 1381, 13), csd_281902, *[x_281903, y_281904, fs_281905, window_281906, nperseg_281907, noverlap_281908, nfft_281909, detrend_281910], **kwargs_281913)
    
    # Obtaining the member '__getitem__' of a type (line 1381)
    getitem___281915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1381, 4), csd_call_result_281914, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1381)
    subscript_call_result_281916 = invoke(stypy.reporting.localization.Localization(__file__, 1381, 4), getitem___281915, int_281901)
    
    # Assigning a type to the variable 'tuple_var_assignment_280486' (line 1381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), 'tuple_var_assignment_280486', subscript_call_result_281916)
    
    # Assigning a Name to a Name (line 1381):
    # Getting the type of 'tuple_var_assignment_280485' (line 1381)
    tuple_var_assignment_280485_281917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), 'tuple_var_assignment_280485')
    # Assigning a type to the variable '_' (line 1381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), '_', tuple_var_assignment_280485_281917)
    
    # Assigning a Name to a Name (line 1381):
    # Getting the type of 'tuple_var_assignment_280486' (line 1381)
    tuple_var_assignment_280486_281918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 4), 'tuple_var_assignment_280486')
    # Assigning a type to the variable 'Pxy' (line 1381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 7), 'Pxy', tuple_var_assignment_280486_281918)
    
    # Assigning a BinOp to a Name (line 1383):
    
    # Assigning a BinOp to a Name (line 1383):
    
    # Call to abs(...): (line 1383)
    # Processing the call arguments (line 1383)
    # Getting the type of 'Pxy' (line 1383)
    Pxy_281921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 17), 'Pxy', False)
    # Processing the call keyword arguments (line 1383)
    kwargs_281922 = {}
    # Getting the type of 'np' (line 1383)
    np_281919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1383)
    abs_281920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1383, 10), np_281919, 'abs')
    # Calling abs(args, kwargs) (line 1383)
    abs_call_result_281923 = invoke(stypy.reporting.localization.Localization(__file__, 1383, 10), abs_281920, *[Pxy_281921], **kwargs_281922)
    
    int_281924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 23), 'int')
    # Applying the binary operator '**' (line 1383)
    result_pow_281925 = python_operator(stypy.reporting.localization.Localization(__file__, 1383, 10), '**', abs_call_result_281923, int_281924)
    
    # Getting the type of 'Pxx' (line 1383)
    Pxx_281926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 27), 'Pxx')
    # Applying the binary operator 'div' (line 1383)
    result_div_281927 = python_operator(stypy.reporting.localization.Localization(__file__, 1383, 10), 'div', result_pow_281925, Pxx_281926)
    
    # Getting the type of 'Pyy' (line 1383)
    Pyy_281928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 33), 'Pyy')
    # Applying the binary operator 'div' (line 1383)
    result_div_281929 = python_operator(stypy.reporting.localization.Localization(__file__, 1383, 31), 'div', result_div_281927, Pyy_281928)
    
    # Assigning a type to the variable 'Cxy' (line 1383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1383, 4), 'Cxy', result_div_281929)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1385)
    tuple_281930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1385, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1385)
    # Adding element type (line 1385)
    # Getting the type of 'freqs' (line 1385)
    freqs_281931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 11), tuple_281930, freqs_281931)
    # Adding element type (line 1385)
    # Getting the type of 'Cxy' (line 1385)
    Cxy_281932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 18), 'Cxy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1385, 11), tuple_281930, Cxy_281932)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'stypy_return_type', tuple_281930)
    
    # ################# End of 'coherence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'coherence' in the type store
    # Getting the type of 'stypy_return_type' (line 1271)
    stypy_return_type_281933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_281933)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'coherence'
    return stypy_return_type_281933

# Assigning a type to the variable 'coherence' (line 1271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1271, 0), 'coherence', coherence)

@norecursion
def _spectral_helper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_281934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 30), 'float')
    str_281935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1388, 42), 'str', 'hann')
    # Getting the type of 'None' (line 1388)
    None_281936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 58), 'None')
    # Getting the type of 'None' (line 1388)
    None_281937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 73), 'None')
    # Getting the type of 'None' (line 1389)
    None_281938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 26), 'None')
    str_281939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1389, 40), 'str', 'constant')
    # Getting the type of 'True' (line 1389)
    True_281940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 68), 'True')
    str_281941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1390, 29), 'str', 'spectrum')
    int_281942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1390, 46), 'int')
    str_281943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1390, 55), 'str', 'psd')
    # Getting the type of 'None' (line 1390)
    None_281944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1390, 71), 'None')
    # Getting the type of 'False' (line 1391)
    False_281945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 28), 'False')
    defaults = [float_281934, str_281935, None_281936, None_281937, None_281938, str_281939, True_281940, str_281941, int_281942, str_281943, None_281944, False_281945]
    # Create a new context for function '_spectral_helper'
    module_type_store = module_type_store.open_function_context('_spectral_helper', 1388, 0, False)
    
    # Passed parameters checking function
    _spectral_helper.stypy_localization = localization
    _spectral_helper.stypy_type_of_self = None
    _spectral_helper.stypy_type_store = module_type_store
    _spectral_helper.stypy_function_name = '_spectral_helper'
    _spectral_helper.stypy_param_names_list = ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode', 'boundary', 'padded']
    _spectral_helper.stypy_varargs_param_name = None
    _spectral_helper.stypy_kwargs_param_name = None
    _spectral_helper.stypy_call_defaults = defaults
    _spectral_helper.stypy_call_varargs = varargs
    _spectral_helper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_spectral_helper', ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode', 'boundary', 'padded'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_spectral_helper', localization, ['x', 'y', 'fs', 'window', 'nperseg', 'noverlap', 'nfft', 'detrend', 'return_onesided', 'scaling', 'axis', 'mode', 'boundary', 'padded'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_spectral_helper(...)' code ##################

    str_281946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1484, (-1)), 'str', '\n    Calculate various forms of windowed FFTs for PSD, CSD, etc.\n\n    This is a helper function that implements the commonality between\n    the stft, psd, csd, and spectrogram functions. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Parameters\n    ---------\n    x : array_like\n        Array or sequence containing the data to be analyzed.\n    y : array_like\n        Array or sequence containing the data to be analyzed. If this is\n        the same object in memory as `x` (i.e. ``_spectral_helper(x,\n        x, ...)``), the extra computations are spared.\n    fs : float, optional\n        Sampling frequency of the time series. Defaults to 1.0.\n    window : str or tuple or array_like, optional\n        Desired window to use. If `window` is a string or tuple, it is\n        passed to `get_window` to generate the window values, which are\n        DFT-even by default. See `get_window` for a list of windows and\n        required parameters. If `window` is array_like it will be used\n        directly as the window and its length must be nperseg. Defaults\n        to a Hann window.\n    nperseg : int, optional\n        Length of each segment. Defaults to None, but if window is str or\n        tuple, is set to 256, and if window is array_like, is set to the\n        length of the window.\n    noverlap : int, optional\n        Number of points to overlap between segments. If `None`,\n        ``noverlap = nperseg // 2``. Defaults to `None`.\n    nfft : int, optional\n        Length of the FFT used, if a zero padded FFT is desired. If\n        `None`, the FFT length is `nperseg`. Defaults to `None`.\n    detrend : str or function or `False`, optional\n        Specifies how to detrend each segment. If `detrend` is a\n        string, it is passed as the `type` argument to the `detrend`\n        function. If it is a function, it takes a segment and returns a\n        detrended segment. If `detrend` is `False`, no detrending is\n        done. Defaults to \'constant\'.\n    return_onesided : bool, optional\n        If `True`, return a one-sided spectrum for real data. If\n        `False` return a two-sided spectrum. Note that for complex\n        data, a two-sided spectrum is always returned.\n    scaling : { \'density\', \'spectrum\' }, optional\n        Selects between computing the cross spectral density (\'density\')\n        where `Pxy` has units of V**2/Hz and computing the cross\n        spectrum (\'spectrum\') where `Pxy` has units of V**2, if `x`\n        and `y` are measured in V and `fs` is measured in Hz.\n        Defaults to \'density\'\n    axis : int, optional\n        Axis along which the FFTs are computed; the default is over the\n        last axis (i.e. ``axis=-1``).\n    mode: str {\'psd\', \'stft\'}, optional\n        Defines what kind of return values are expected. Defaults to\n        \'psd\'.\n    boundary : str or None, optional\n        Specifies whether the input signal is extended at both ends, and\n        how to generate the new values, in order to center the first\n        windowed segment on the first input point. This has the benefit\n        of enabling reconstruction of the first input point when the\n        employed window function starts at zero. Valid options are\n        ``[\'even\', \'odd\', \'constant\', \'zeros\', None]``. Defaults to\n        `None`.\n    padded : bool, optional\n        Specifies whether the input signal is zero-padded at the end to\n        make the signal fit exactly into an integer number of window\n        segments, so that all of the signal is included in the output.\n        Defaults to `False`. Padding occurs after boundary extension, if\n        `boundary` is not `None`, and `padded` is `True`.\n    Returns\n    -------\n    freqs : ndarray\n        Array of sample frequencies.\n    t : ndarray\n        Array of times corresponding to each data segment\n    result : ndarray\n        Array of output data, contents dependant on *mode* kwarg.\n\n    References\n    ----------\n    .. [1] Stack Overflow, "Rolling window for 1D arrays in Numpy?",\n           http://stackoverflow.com/a/6811241\n    .. [2] Stack Overflow, "Using strides for an efficient moving\n           average filter", http://stackoverflow.com/a/4947453\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    .. versionadded:: 0.16.0\n    ')
    
    
    # Getting the type of 'mode' (line 1485)
    mode_281947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1485, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 1485)
    list_281948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1485, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1485)
    # Adding element type (line 1485)
    str_281949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1485, 20), 'str', 'psd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1485, 19), list_281948, str_281949)
    # Adding element type (line 1485)
    str_281950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1485, 27), 'str', 'stft')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1485, 19), list_281948, str_281950)
    
    # Applying the binary operator 'notin' (line 1485)
    result_contains_281951 = python_operator(stypy.reporting.localization.Localization(__file__, 1485, 7), 'notin', mode_281947, list_281948)
    
    # Testing the type of an if condition (line 1485)
    if_condition_281952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1485, 4), result_contains_281951)
    # Assigning a type to the variable 'if_condition_281952' (line 1485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1485, 4), 'if_condition_281952', if_condition_281952)
    # SSA begins for if statement (line 1485)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1486)
    # Processing the call arguments (line 1486)
    str_281954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1486, 25), 'str', "Unknown value for mode %s, must be one of: {'psd', 'stft'}")
    # Getting the type of 'mode' (line 1487)
    mode_281955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1487, 45), 'mode', False)
    # Applying the binary operator '%' (line 1486)
    result_mod_281956 = python_operator(stypy.reporting.localization.Localization(__file__, 1486, 25), '%', str_281954, mode_281955)
    
    # Processing the call keyword arguments (line 1486)
    kwargs_281957 = {}
    # Getting the type of 'ValueError' (line 1486)
    ValueError_281953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1486, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1486)
    ValueError_call_result_281958 = invoke(stypy.reporting.localization.Localization(__file__, 1486, 14), ValueError_281953, *[result_mod_281956], **kwargs_281957)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1486, 8), ValueError_call_result_281958, 'raise parameter', BaseException)
    # SSA join for if statement (line 1485)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 1489):
    
    # Assigning a Dict to a Name (line 1489):
    
    # Obtaining an instance of the builtin type 'dict' (line 1489)
    dict_281959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 21), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1489)
    # Adding element type (key, value) (line 1489)
    str_281960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 22), 'str', 'even')
    # Getting the type of 'even_ext' (line 1489)
    even_ext_281961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 30), 'even_ext')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 21), dict_281959, (str_281960, even_ext_281961))
    # Adding element type (key, value) (line 1489)
    str_281962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 22), 'str', 'odd')
    # Getting the type of 'odd_ext' (line 1490)
    odd_ext_281963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 29), 'odd_ext')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 21), dict_281959, (str_281962, odd_ext_281963))
    # Adding element type (key, value) (line 1489)
    str_281964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1491, 22), 'str', 'constant')
    # Getting the type of 'const_ext' (line 1491)
    const_ext_281965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 34), 'const_ext')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 21), dict_281959, (str_281964, const_ext_281965))
    # Adding element type (key, value) (line 1489)
    str_281966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1492, 22), 'str', 'zeros')
    # Getting the type of 'zero_ext' (line 1492)
    zero_ext_281967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 31), 'zero_ext')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 21), dict_281959, (str_281966, zero_ext_281967))
    # Adding element type (key, value) (line 1489)
    # Getting the type of 'None' (line 1493)
    None_281968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 22), 'None')
    # Getting the type of 'None' (line 1493)
    None_281969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 28), 'None')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1489, 21), dict_281959, (None_281968, None_281969))
    
    # Assigning a type to the variable 'boundary_funcs' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'boundary_funcs', dict_281959)
    
    
    # Getting the type of 'boundary' (line 1495)
    boundary_281970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 7), 'boundary')
    # Getting the type of 'boundary_funcs' (line 1495)
    boundary_funcs_281971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 23), 'boundary_funcs')
    # Applying the binary operator 'notin' (line 1495)
    result_contains_281972 = python_operator(stypy.reporting.localization.Localization(__file__, 1495, 7), 'notin', boundary_281970, boundary_funcs_281971)
    
    # Testing the type of an if condition (line 1495)
    if_condition_281973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1495, 4), result_contains_281972)
    # Assigning a type to the variable 'if_condition_281973' (line 1495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1495, 4), 'if_condition_281973', if_condition_281973)
    # SSA begins for if statement (line 1495)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1496)
    # Processing the call arguments (line 1496)
    
    # Call to format(...): (line 1496)
    # Processing the call arguments (line 1496)
    # Getting the type of 'boundary' (line 1497)
    boundary_281977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 34), 'boundary', False)
    
    # Call to list(...): (line 1497)
    # Processing the call arguments (line 1497)
    
    # Call to keys(...): (line 1497)
    # Processing the call keyword arguments (line 1497)
    kwargs_281981 = {}
    # Getting the type of 'boundary_funcs' (line 1497)
    boundary_funcs_281979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 49), 'boundary_funcs', False)
    # Obtaining the member 'keys' of a type (line 1497)
    keys_281980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1497, 49), boundary_funcs_281979, 'keys')
    # Calling keys(args, kwargs) (line 1497)
    keys_call_result_281982 = invoke(stypy.reporting.localization.Localization(__file__, 1497, 49), keys_281980, *[], **kwargs_281981)
    
    # Processing the call keyword arguments (line 1497)
    kwargs_281983 = {}
    # Getting the type of 'list' (line 1497)
    list_281978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 44), 'list', False)
    # Calling list(args, kwargs) (line 1497)
    list_call_result_281984 = invoke(stypy.reporting.localization.Localization(__file__, 1497, 44), list_281978, *[keys_call_result_281982], **kwargs_281983)
    
    # Processing the call keyword arguments (line 1496)
    kwargs_281985 = {}
    str_281975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 25), 'str', "Unknown boundary option '{0}', must be one of: {1}")
    # Obtaining the member 'format' of a type (line 1496)
    format_281976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 25), str_281975, 'format')
    # Calling format(args, kwargs) (line 1496)
    format_call_result_281986 = invoke(stypy.reporting.localization.Localization(__file__, 1496, 25), format_281976, *[boundary_281977, list_call_result_281984], **kwargs_281985)
    
    # Processing the call keyword arguments (line 1496)
    kwargs_281987 = {}
    # Getting the type of 'ValueError' (line 1496)
    ValueError_281974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1496)
    ValueError_call_result_281988 = invoke(stypy.reporting.localization.Localization(__file__, 1496, 14), ValueError_281974, *[format_call_result_281986], **kwargs_281987)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1496, 8), ValueError_call_result_281988, 'raise parameter', BaseException)
    # SSA join for if statement (line 1495)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 1500):
    
    # Assigning a Compare to a Name (line 1500):
    
    # Getting the type of 'y' (line 1500)
    y_281989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1500, 16), 'y')
    # Getting the type of 'x' (line 1500)
    x_281990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1500, 21), 'x')
    # Applying the binary operator 'is' (line 1500)
    result_is__281991 = python_operator(stypy.reporting.localization.Localization(__file__, 1500, 16), 'is', y_281989, x_281990)
    
    # Assigning a type to the variable 'same_data' (line 1500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1500, 4), 'same_data', result_is__281991)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'same_data' (line 1502)
    same_data_281992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 11), 'same_data')
    # Applying the 'not' unary operator (line 1502)
    result_not__281993 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), 'not', same_data_281992)
    
    
    # Getting the type of 'mode' (line 1502)
    mode_281994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 25), 'mode')
    str_281995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 33), 'str', 'psd')
    # Applying the binary operator '!=' (line 1502)
    result_ne_281996 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 25), '!=', mode_281994, str_281995)
    
    # Applying the binary operator 'and' (line 1502)
    result_and_keyword_281997 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), 'and', result_not__281993, result_ne_281996)
    
    # Testing the type of an if condition (line 1502)
    if_condition_281998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1502, 4), result_and_keyword_281997)
    # Assigning a type to the variable 'if_condition_281998' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'if_condition_281998', if_condition_281998)
    # SSA begins for if statement (line 1502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1503)
    # Processing the call arguments (line 1503)
    str_282000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1503, 25), 'str', "x and y must be equal if mode is 'stft'")
    # Processing the call keyword arguments (line 1503)
    kwargs_282001 = {}
    # Getting the type of 'ValueError' (line 1503)
    ValueError_281999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1503)
    ValueError_call_result_282002 = invoke(stypy.reporting.localization.Localization(__file__, 1503, 14), ValueError_281999, *[str_282000], **kwargs_282001)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1503, 8), ValueError_call_result_282002, 'raise parameter', BaseException)
    # SSA join for if statement (line 1502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1505):
    
    # Assigning a Call to a Name (line 1505):
    
    # Call to int(...): (line 1505)
    # Processing the call arguments (line 1505)
    # Getting the type of 'axis' (line 1505)
    axis_282004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 15), 'axis', False)
    # Processing the call keyword arguments (line 1505)
    kwargs_282005 = {}
    # Getting the type of 'int' (line 1505)
    int_282003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 11), 'int', False)
    # Calling int(args, kwargs) (line 1505)
    int_call_result_282006 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 11), int_282003, *[axis_282004], **kwargs_282005)
    
    # Assigning a type to the variable 'axis' (line 1505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 4), 'axis', int_call_result_282006)
    
    # Assigning a Call to a Name (line 1508):
    
    # Assigning a Call to a Name (line 1508):
    
    # Call to asarray(...): (line 1508)
    # Processing the call arguments (line 1508)
    # Getting the type of 'x' (line 1508)
    x_282009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 19), 'x', False)
    # Processing the call keyword arguments (line 1508)
    kwargs_282010 = {}
    # Getting the type of 'np' (line 1508)
    np_282007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1508)
    asarray_282008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1508, 8), np_282007, 'asarray')
    # Calling asarray(args, kwargs) (line 1508)
    asarray_call_result_282011 = invoke(stypy.reporting.localization.Localization(__file__, 1508, 8), asarray_282008, *[x_282009], **kwargs_282010)
    
    # Assigning a type to the variable 'x' (line 1508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1508, 4), 'x', asarray_call_result_282011)
    
    
    # Getting the type of 'same_data' (line 1509)
    same_data_282012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1509, 11), 'same_data')
    # Applying the 'not' unary operator (line 1509)
    result_not__282013 = python_operator(stypy.reporting.localization.Localization(__file__, 1509, 7), 'not', same_data_282012)
    
    # Testing the type of an if condition (line 1509)
    if_condition_282014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1509, 4), result_not__282013)
    # Assigning a type to the variable 'if_condition_282014' (line 1509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1509, 4), 'if_condition_282014', if_condition_282014)
    # SSA begins for if statement (line 1509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1510):
    
    # Assigning a Call to a Name (line 1510):
    
    # Call to asarray(...): (line 1510)
    # Processing the call arguments (line 1510)
    # Getting the type of 'y' (line 1510)
    y_282017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 23), 'y', False)
    # Processing the call keyword arguments (line 1510)
    kwargs_282018 = {}
    # Getting the type of 'np' (line 1510)
    np_282015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1510)
    asarray_282016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 12), np_282015, 'asarray')
    # Calling asarray(args, kwargs) (line 1510)
    asarray_call_result_282019 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 12), asarray_282016, *[y_282017], **kwargs_282018)
    
    # Assigning a type to the variable 'y' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 8), 'y', asarray_call_result_282019)
    
    # Assigning a Call to a Name (line 1511):
    
    # Assigning a Call to a Name (line 1511):
    
    # Call to result_type(...): (line 1511)
    # Processing the call arguments (line 1511)
    # Getting the type of 'x' (line 1511)
    x_282022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 34), 'x', False)
    # Getting the type of 'y' (line 1511)
    y_282023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 37), 'y', False)
    # Getting the type of 'np' (line 1511)
    np_282024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 40), 'np', False)
    # Obtaining the member 'complex64' of a type (line 1511)
    complex64_282025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1511, 40), np_282024, 'complex64')
    # Processing the call keyword arguments (line 1511)
    kwargs_282026 = {}
    # Getting the type of 'np' (line 1511)
    np_282020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 19), 'np', False)
    # Obtaining the member 'result_type' of a type (line 1511)
    result_type_282021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1511, 19), np_282020, 'result_type')
    # Calling result_type(args, kwargs) (line 1511)
    result_type_call_result_282027 = invoke(stypy.reporting.localization.Localization(__file__, 1511, 19), result_type_282021, *[x_282022, y_282023, complex64_282025], **kwargs_282026)
    
    # Assigning a type to the variable 'outdtype' (line 1511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1511, 8), 'outdtype', result_type_call_result_282027)
    # SSA branch for the else part of an if statement (line 1509)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1513):
    
    # Assigning a Call to a Name (line 1513):
    
    # Call to result_type(...): (line 1513)
    # Processing the call arguments (line 1513)
    # Getting the type of 'x' (line 1513)
    x_282030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 34), 'x', False)
    # Getting the type of 'np' (line 1513)
    np_282031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 37), 'np', False)
    # Obtaining the member 'complex64' of a type (line 1513)
    complex64_282032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1513, 37), np_282031, 'complex64')
    # Processing the call keyword arguments (line 1513)
    kwargs_282033 = {}
    # Getting the type of 'np' (line 1513)
    np_282028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 19), 'np', False)
    # Obtaining the member 'result_type' of a type (line 1513)
    result_type_282029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1513, 19), np_282028, 'result_type')
    # Calling result_type(args, kwargs) (line 1513)
    result_type_call_result_282034 = invoke(stypy.reporting.localization.Localization(__file__, 1513, 19), result_type_282029, *[x_282030, complex64_282032], **kwargs_282033)
    
    # Assigning a type to the variable 'outdtype' (line 1513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1513, 8), 'outdtype', result_type_call_result_282034)
    # SSA join for if statement (line 1509)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'same_data' (line 1515)
    same_data_282035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 11), 'same_data')
    # Applying the 'not' unary operator (line 1515)
    result_not__282036 = python_operator(stypy.reporting.localization.Localization(__file__, 1515, 7), 'not', same_data_282035)
    
    # Testing the type of an if condition (line 1515)
    if_condition_282037 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1515, 4), result_not__282036)
    # Assigning a type to the variable 'if_condition_282037' (line 1515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1515, 4), 'if_condition_282037', if_condition_282037)
    # SSA begins for if statement (line 1515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1517):
    
    # Assigning a Call to a Name (line 1517):
    
    # Call to list(...): (line 1517)
    # Processing the call arguments (line 1517)
    # Getting the type of 'x' (line 1517)
    x_282039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 22), 'x', False)
    # Obtaining the member 'shape' of a type (line 1517)
    shape_282040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1517, 22), x_282039, 'shape')
    # Processing the call keyword arguments (line 1517)
    kwargs_282041 = {}
    # Getting the type of 'list' (line 1517)
    list_282038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 17), 'list', False)
    # Calling list(args, kwargs) (line 1517)
    list_call_result_282042 = invoke(stypy.reporting.localization.Localization(__file__, 1517, 17), list_282038, *[shape_282040], **kwargs_282041)
    
    # Assigning a type to the variable 'xouter' (line 1517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1517, 8), 'xouter', list_call_result_282042)
    
    # Assigning a Call to a Name (line 1518):
    
    # Assigning a Call to a Name (line 1518):
    
    # Call to list(...): (line 1518)
    # Processing the call arguments (line 1518)
    # Getting the type of 'y' (line 1518)
    y_282044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 22), 'y', False)
    # Obtaining the member 'shape' of a type (line 1518)
    shape_282045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1518, 22), y_282044, 'shape')
    # Processing the call keyword arguments (line 1518)
    kwargs_282046 = {}
    # Getting the type of 'list' (line 1518)
    list_282043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 17), 'list', False)
    # Calling list(args, kwargs) (line 1518)
    list_call_result_282047 = invoke(stypy.reporting.localization.Localization(__file__, 1518, 17), list_282043, *[shape_282045], **kwargs_282046)
    
    # Assigning a type to the variable 'youter' (line 1518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1518, 8), 'youter', list_call_result_282047)
    
    # Call to pop(...): (line 1519)
    # Processing the call arguments (line 1519)
    # Getting the type of 'axis' (line 1519)
    axis_282050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 19), 'axis', False)
    # Processing the call keyword arguments (line 1519)
    kwargs_282051 = {}
    # Getting the type of 'xouter' (line 1519)
    xouter_282048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 8), 'xouter', False)
    # Obtaining the member 'pop' of a type (line 1519)
    pop_282049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1519, 8), xouter_282048, 'pop')
    # Calling pop(args, kwargs) (line 1519)
    pop_call_result_282052 = invoke(stypy.reporting.localization.Localization(__file__, 1519, 8), pop_282049, *[axis_282050], **kwargs_282051)
    
    
    # Call to pop(...): (line 1520)
    # Processing the call arguments (line 1520)
    # Getting the type of 'axis' (line 1520)
    axis_282055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 19), 'axis', False)
    # Processing the call keyword arguments (line 1520)
    kwargs_282056 = {}
    # Getting the type of 'youter' (line 1520)
    youter_282053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 8), 'youter', False)
    # Obtaining the member 'pop' of a type (line 1520)
    pop_282054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1520, 8), youter_282053, 'pop')
    # Calling pop(args, kwargs) (line 1520)
    pop_call_result_282057 = invoke(stypy.reporting.localization.Localization(__file__, 1520, 8), pop_282054, *[axis_282055], **kwargs_282056)
    
    
    
    # SSA begins for try-except statement (line 1521)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 1522):
    
    # Assigning a Attribute to a Name (line 1522):
    
    # Call to broadcast(...): (line 1522)
    # Processing the call arguments (line 1522)
    
    # Call to empty(...): (line 1522)
    # Processing the call arguments (line 1522)
    # Getting the type of 'xouter' (line 1522)
    xouter_282062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 47), 'xouter', False)
    # Processing the call keyword arguments (line 1522)
    kwargs_282063 = {}
    # Getting the type of 'np' (line 1522)
    np_282060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 38), 'np', False)
    # Obtaining the member 'empty' of a type (line 1522)
    empty_282061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 38), np_282060, 'empty')
    # Calling empty(args, kwargs) (line 1522)
    empty_call_result_282064 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 38), empty_282061, *[xouter_282062], **kwargs_282063)
    
    
    # Call to empty(...): (line 1522)
    # Processing the call arguments (line 1522)
    # Getting the type of 'youter' (line 1522)
    youter_282067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 65), 'youter', False)
    # Processing the call keyword arguments (line 1522)
    kwargs_282068 = {}
    # Getting the type of 'np' (line 1522)
    np_282065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 56), 'np', False)
    # Obtaining the member 'empty' of a type (line 1522)
    empty_282066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 56), np_282065, 'empty')
    # Calling empty(args, kwargs) (line 1522)
    empty_call_result_282069 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 56), empty_282066, *[youter_282067], **kwargs_282068)
    
    # Processing the call keyword arguments (line 1522)
    kwargs_282070 = {}
    # Getting the type of 'np' (line 1522)
    np_282058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 25), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 1522)
    broadcast_282059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 25), np_282058, 'broadcast')
    # Calling broadcast(args, kwargs) (line 1522)
    broadcast_call_result_282071 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 25), broadcast_282059, *[empty_call_result_282064, empty_call_result_282069], **kwargs_282070)
    
    # Obtaining the member 'shape' of a type (line 1522)
    shape_282072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 25), broadcast_call_result_282071, 'shape')
    # Assigning a type to the variable 'outershape' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 12), 'outershape', shape_282072)
    # SSA branch for the except part of a try statement (line 1521)
    # SSA branch for the except 'ValueError' branch of a try statement (line 1521)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1524)
    # Processing the call arguments (line 1524)
    str_282074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 29), 'str', 'x and y cannot be broadcast together.')
    # Processing the call keyword arguments (line 1524)
    kwargs_282075 = {}
    # Getting the type of 'ValueError' (line 1524)
    ValueError_282073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1524)
    ValueError_call_result_282076 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 18), ValueError_282073, *[str_282074], **kwargs_282075)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1524, 12), ValueError_call_result_282076, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1521)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'same_data' (line 1526)
    same_data_282077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 7), 'same_data')
    # Testing the type of an if condition (line 1526)
    if_condition_282078 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1526, 4), same_data_282077)
    # Assigning a type to the variable 'if_condition_282078' (line 1526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1526, 4), 'if_condition_282078', if_condition_282078)
    # SSA begins for if statement (line 1526)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'x' (line 1527)
    x_282079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 11), 'x')
    # Obtaining the member 'size' of a type (line 1527)
    size_282080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1527, 11), x_282079, 'size')
    int_282081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1527, 21), 'int')
    # Applying the binary operator '==' (line 1527)
    result_eq_282082 = python_operator(stypy.reporting.localization.Localization(__file__, 1527, 11), '==', size_282080, int_282081)
    
    # Testing the type of an if condition (line 1527)
    if_condition_282083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1527, 8), result_eq_282082)
    # Assigning a type to the variable 'if_condition_282083' (line 1527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1527, 8), 'if_condition_282083', if_condition_282083)
    # SSA begins for if statement (line 1527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1528)
    tuple_282084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1528, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1528)
    # Adding element type (line 1528)
    
    # Call to empty(...): (line 1528)
    # Processing the call arguments (line 1528)
    # Getting the type of 'x' (line 1528)
    x_282087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 28), 'x', False)
    # Obtaining the member 'shape' of a type (line 1528)
    shape_282088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 28), x_282087, 'shape')
    # Processing the call keyword arguments (line 1528)
    kwargs_282089 = {}
    # Getting the type of 'np' (line 1528)
    np_282085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 19), 'np', False)
    # Obtaining the member 'empty' of a type (line 1528)
    empty_282086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 19), np_282085, 'empty')
    # Calling empty(args, kwargs) (line 1528)
    empty_call_result_282090 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 19), empty_282086, *[shape_282088], **kwargs_282089)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1528, 19), tuple_282084, empty_call_result_282090)
    # Adding element type (line 1528)
    
    # Call to empty(...): (line 1528)
    # Processing the call arguments (line 1528)
    # Getting the type of 'x' (line 1528)
    x_282093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 47), 'x', False)
    # Obtaining the member 'shape' of a type (line 1528)
    shape_282094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 47), x_282093, 'shape')
    # Processing the call keyword arguments (line 1528)
    kwargs_282095 = {}
    # Getting the type of 'np' (line 1528)
    np_282091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 38), 'np', False)
    # Obtaining the member 'empty' of a type (line 1528)
    empty_282092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 38), np_282091, 'empty')
    # Calling empty(args, kwargs) (line 1528)
    empty_call_result_282096 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 38), empty_282092, *[shape_282094], **kwargs_282095)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1528, 19), tuple_282084, empty_call_result_282096)
    # Adding element type (line 1528)
    
    # Call to empty(...): (line 1528)
    # Processing the call arguments (line 1528)
    # Getting the type of 'x' (line 1528)
    x_282099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 66), 'x', False)
    # Obtaining the member 'shape' of a type (line 1528)
    shape_282100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 66), x_282099, 'shape')
    # Processing the call keyword arguments (line 1528)
    kwargs_282101 = {}
    # Getting the type of 'np' (line 1528)
    np_282097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 57), 'np', False)
    # Obtaining the member 'empty' of a type (line 1528)
    empty_282098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 57), np_282097, 'empty')
    # Calling empty(args, kwargs) (line 1528)
    empty_call_result_282102 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 57), empty_282098, *[shape_282100], **kwargs_282101)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1528, 19), tuple_282084, empty_call_result_282102)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1528, 12), 'stypy_return_type', tuple_282084)
    # SSA join for if statement (line 1527)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1526)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 1530)
    x_282103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1530, 11), 'x')
    # Obtaining the member 'size' of a type (line 1530)
    size_282104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1530, 11), x_282103, 'size')
    int_282105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1530, 21), 'int')
    # Applying the binary operator '==' (line 1530)
    result_eq_282106 = python_operator(stypy.reporting.localization.Localization(__file__, 1530, 11), '==', size_282104, int_282105)
    
    
    # Getting the type of 'y' (line 1530)
    y_282107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1530, 26), 'y')
    # Obtaining the member 'size' of a type (line 1530)
    size_282108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1530, 26), y_282107, 'size')
    int_282109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1530, 36), 'int')
    # Applying the binary operator '==' (line 1530)
    result_eq_282110 = python_operator(stypy.reporting.localization.Localization(__file__, 1530, 26), '==', size_282108, int_282109)
    
    # Applying the binary operator 'or' (line 1530)
    result_or_keyword_282111 = python_operator(stypy.reporting.localization.Localization(__file__, 1530, 11), 'or', result_eq_282106, result_eq_282110)
    
    # Testing the type of an if condition (line 1530)
    if_condition_282112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1530, 8), result_or_keyword_282111)
    # Assigning a type to the variable 'if_condition_282112' (line 1530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1530, 8), 'if_condition_282112', if_condition_282112)
    # SSA begins for if statement (line 1530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1531):
    
    # Assigning a BinOp to a Name (line 1531):
    # Getting the type of 'outershape' (line 1531)
    outershape_282113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 23), 'outershape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1531)
    tuple_282114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1531, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1531)
    # Adding element type (line 1531)
    
    # Call to min(...): (line 1531)
    # Processing the call arguments (line 1531)
    
    # Obtaining an instance of the builtin type 'list' (line 1531)
    list_282116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1531, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1531)
    # Adding element type (line 1531)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1531)
    axis_282117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 50), 'axis', False)
    # Getting the type of 'x' (line 1531)
    x_282118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 42), 'x', False)
    # Obtaining the member 'shape' of a type (line 1531)
    shape_282119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1531, 42), x_282118, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1531)
    getitem___282120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1531, 42), shape_282119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1531)
    subscript_call_result_282121 = invoke(stypy.reporting.localization.Localization(__file__, 1531, 42), getitem___282120, axis_282117)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1531, 41), list_282116, subscript_call_result_282121)
    # Adding element type (line 1531)
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 1531)
    axis_282122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 65), 'axis', False)
    # Getting the type of 'y' (line 1531)
    y_282123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 57), 'y', False)
    # Obtaining the member 'shape' of a type (line 1531)
    shape_282124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1531, 57), y_282123, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1531)
    getitem___282125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1531, 57), shape_282124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1531)
    subscript_call_result_282126 = invoke(stypy.reporting.localization.Localization(__file__, 1531, 57), getitem___282125, axis_282122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1531, 41), list_282116, subscript_call_result_282126)
    
    # Processing the call keyword arguments (line 1531)
    kwargs_282127 = {}
    # Getting the type of 'min' (line 1531)
    min_282115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 37), 'min', False)
    # Calling min(args, kwargs) (line 1531)
    min_call_result_282128 = invoke(stypy.reporting.localization.Localization(__file__, 1531, 37), min_282115, *[list_282116], **kwargs_282127)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1531, 37), tuple_282114, min_call_result_282128)
    
    # Applying the binary operator '+' (line 1531)
    result_add_282129 = python_operator(stypy.reporting.localization.Localization(__file__, 1531, 23), '+', outershape_282113, tuple_282114)
    
    # Assigning a type to the variable 'outshape' (line 1531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1531, 12), 'outshape', result_add_282129)
    
    # Assigning a Call to a Name (line 1532):
    
    # Assigning a Call to a Name (line 1532):
    
    # Call to rollaxis(...): (line 1532)
    # Processing the call arguments (line 1532)
    
    # Call to empty(...): (line 1532)
    # Processing the call arguments (line 1532)
    # Getting the type of 'outshape' (line 1532)
    outshape_282134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 44), 'outshape', False)
    # Processing the call keyword arguments (line 1532)
    kwargs_282135 = {}
    # Getting the type of 'np' (line 1532)
    np_282132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 35), 'np', False)
    # Obtaining the member 'empty' of a type (line 1532)
    empty_282133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1532, 35), np_282132, 'empty')
    # Calling empty(args, kwargs) (line 1532)
    empty_call_result_282136 = invoke(stypy.reporting.localization.Localization(__file__, 1532, 35), empty_282133, *[outshape_282134], **kwargs_282135)
    
    int_282137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1532, 55), 'int')
    # Getting the type of 'axis' (line 1532)
    axis_282138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 59), 'axis', False)
    # Processing the call keyword arguments (line 1532)
    kwargs_282139 = {}
    # Getting the type of 'np' (line 1532)
    np_282130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 23), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1532)
    rollaxis_282131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1532, 23), np_282130, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1532)
    rollaxis_call_result_282140 = invoke(stypy.reporting.localization.Localization(__file__, 1532, 23), rollaxis_282131, *[empty_call_result_282136, int_282137, axis_282138], **kwargs_282139)
    
    # Assigning a type to the variable 'emptyout' (line 1532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 12), 'emptyout', rollaxis_call_result_282140)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1533)
    tuple_282141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1533, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1533)
    # Adding element type (line 1533)
    # Getting the type of 'emptyout' (line 1533)
    emptyout_282142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 19), 'emptyout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1533, 19), tuple_282141, emptyout_282142)
    # Adding element type (line 1533)
    # Getting the type of 'emptyout' (line 1533)
    emptyout_282143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 29), 'emptyout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1533, 19), tuple_282141, emptyout_282143)
    # Adding element type (line 1533)
    # Getting the type of 'emptyout' (line 1533)
    emptyout_282144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 39), 'emptyout')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1533, 19), tuple_282141, emptyout_282144)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1533, 12), 'stypy_return_type', tuple_282141)
    # SSA join for if statement (line 1530)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1526)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1535)
    x_282145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1535)
    ndim_282146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1535, 7), x_282145, 'ndim')
    int_282147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1535, 16), 'int')
    # Applying the binary operator '>' (line 1535)
    result_gt_282148 = python_operator(stypy.reporting.localization.Localization(__file__, 1535, 7), '>', ndim_282146, int_282147)
    
    # Testing the type of an if condition (line 1535)
    if_condition_282149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1535, 4), result_gt_282148)
    # Assigning a type to the variable 'if_condition_282149' (line 1535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1535, 4), 'if_condition_282149', if_condition_282149)
    # SSA begins for if statement (line 1535)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'axis' (line 1536)
    axis_282150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 11), 'axis')
    int_282151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1536, 19), 'int')
    # Applying the binary operator '!=' (line 1536)
    result_ne_282152 = python_operator(stypy.reporting.localization.Localization(__file__, 1536, 11), '!=', axis_282150, int_282151)
    
    # Testing the type of an if condition (line 1536)
    if_condition_282153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1536, 8), result_ne_282152)
    # Assigning a type to the variable 'if_condition_282153' (line 1536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1536, 8), 'if_condition_282153', if_condition_282153)
    # SSA begins for if statement (line 1536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1537):
    
    # Assigning a Call to a Name (line 1537):
    
    # Call to rollaxis(...): (line 1537)
    # Processing the call arguments (line 1537)
    # Getting the type of 'x' (line 1537)
    x_282156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 28), 'x', False)
    # Getting the type of 'axis' (line 1537)
    axis_282157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 31), 'axis', False)
    
    # Call to len(...): (line 1537)
    # Processing the call arguments (line 1537)
    # Getting the type of 'x' (line 1537)
    x_282159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 41), 'x', False)
    # Obtaining the member 'shape' of a type (line 1537)
    shape_282160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 41), x_282159, 'shape')
    # Processing the call keyword arguments (line 1537)
    kwargs_282161 = {}
    # Getting the type of 'len' (line 1537)
    len_282158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 37), 'len', False)
    # Calling len(args, kwargs) (line 1537)
    len_call_result_282162 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 37), len_282158, *[shape_282160], **kwargs_282161)
    
    # Processing the call keyword arguments (line 1537)
    kwargs_282163 = {}
    # Getting the type of 'np' (line 1537)
    np_282154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 16), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1537)
    rollaxis_282155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 16), np_282154, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1537)
    rollaxis_call_result_282164 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 16), rollaxis_282155, *[x_282156, axis_282157, len_call_result_282162], **kwargs_282163)
    
    # Assigning a type to the variable 'x' (line 1537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1537, 12), 'x', rollaxis_call_result_282164)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'same_data' (line 1538)
    same_data_282165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 19), 'same_data')
    # Applying the 'not' unary operator (line 1538)
    result_not__282166 = python_operator(stypy.reporting.localization.Localization(__file__, 1538, 15), 'not', same_data_282165)
    
    
    # Getting the type of 'y' (line 1538)
    y_282167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1538, 33), 'y')
    # Obtaining the member 'ndim' of a type (line 1538)
    ndim_282168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1538, 33), y_282167, 'ndim')
    int_282169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1538, 42), 'int')
    # Applying the binary operator '>' (line 1538)
    result_gt_282170 = python_operator(stypy.reporting.localization.Localization(__file__, 1538, 33), '>', ndim_282168, int_282169)
    
    # Applying the binary operator 'and' (line 1538)
    result_and_keyword_282171 = python_operator(stypy.reporting.localization.Localization(__file__, 1538, 15), 'and', result_not__282166, result_gt_282170)
    
    # Testing the type of an if condition (line 1538)
    if_condition_282172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1538, 12), result_and_keyword_282171)
    # Assigning a type to the variable 'if_condition_282172' (line 1538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1538, 12), 'if_condition_282172', if_condition_282172)
    # SSA begins for if statement (line 1538)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1539):
    
    # Assigning a Call to a Name (line 1539):
    
    # Call to rollaxis(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'y' (line 1539)
    y_282175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 32), 'y', False)
    # Getting the type of 'axis' (line 1539)
    axis_282176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 35), 'axis', False)
    
    # Call to len(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'y' (line 1539)
    y_282178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 45), 'y', False)
    # Obtaining the member 'shape' of a type (line 1539)
    shape_282179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 45), y_282178, 'shape')
    # Processing the call keyword arguments (line 1539)
    kwargs_282180 = {}
    # Getting the type of 'len' (line 1539)
    len_282177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 41), 'len', False)
    # Calling len(args, kwargs) (line 1539)
    len_call_result_282181 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 41), len_282177, *[shape_282179], **kwargs_282180)
    
    # Processing the call keyword arguments (line 1539)
    kwargs_282182 = {}
    # Getting the type of 'np' (line 1539)
    np_282173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 20), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1539)
    rollaxis_282174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 20), np_282173, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1539)
    rollaxis_call_result_282183 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 20), rollaxis_282174, *[y_282175, axis_282176, len_call_result_282181], **kwargs_282182)
    
    # Assigning a type to the variable 'y' (line 1539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1539, 16), 'y', rollaxis_call_result_282183)
    # SSA join for if statement (line 1538)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1536)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1535)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'same_data' (line 1542)
    same_data_282184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 11), 'same_data')
    # Applying the 'not' unary operator (line 1542)
    result_not__282185 = python_operator(stypy.reporting.localization.Localization(__file__, 1542, 7), 'not', same_data_282184)
    
    # Testing the type of an if condition (line 1542)
    if_condition_282186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1542, 4), result_not__282185)
    # Assigning a type to the variable 'if_condition_282186' (line 1542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1542, 4), 'if_condition_282186', if_condition_282186)
    # SSA begins for if statement (line 1542)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_282187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 19), 'int')
    # Getting the type of 'x' (line 1543)
    x_282188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 11), 'x')
    # Obtaining the member 'shape' of a type (line 1543)
    shape_282189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 11), x_282188, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___282190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 11), shape_282189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1543)
    subscript_call_result_282191 = invoke(stypy.reporting.localization.Localization(__file__, 1543, 11), getitem___282190, int_282187)
    
    
    # Obtaining the type of the subscript
    int_282192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 34), 'int')
    # Getting the type of 'y' (line 1543)
    y_282193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 26), 'y')
    # Obtaining the member 'shape' of a type (line 1543)
    shape_282194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 26), y_282193, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___282195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 26), shape_282194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1543)
    subscript_call_result_282196 = invoke(stypy.reporting.localization.Localization(__file__, 1543, 26), getitem___282195, int_282192)
    
    # Applying the binary operator '!=' (line 1543)
    result_ne_282197 = python_operator(stypy.reporting.localization.Localization(__file__, 1543, 11), '!=', subscript_call_result_282191, subscript_call_result_282196)
    
    # Testing the type of an if condition (line 1543)
    if_condition_282198 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1543, 8), result_ne_282197)
    # Assigning a type to the variable 'if_condition_282198' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 8), 'if_condition_282198', if_condition_282198)
    # SSA begins for if statement (line 1543)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_282199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1544, 23), 'int')
    # Getting the type of 'x' (line 1544)
    x_282200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 15), 'x')
    # Obtaining the member 'shape' of a type (line 1544)
    shape_282201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 15), x_282200, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1544)
    getitem___282202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 15), shape_282201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1544)
    subscript_call_result_282203 = invoke(stypy.reporting.localization.Localization(__file__, 1544, 15), getitem___282202, int_282199)
    
    
    # Obtaining the type of the subscript
    int_282204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1544, 37), 'int')
    # Getting the type of 'y' (line 1544)
    y_282205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 29), 'y')
    # Obtaining the member 'shape' of a type (line 1544)
    shape_282206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 29), y_282205, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1544)
    getitem___282207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 29), shape_282206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1544)
    subscript_call_result_282208 = invoke(stypy.reporting.localization.Localization(__file__, 1544, 29), getitem___282207, int_282204)
    
    # Applying the binary operator '<' (line 1544)
    result_lt_282209 = python_operator(stypy.reporting.localization.Localization(__file__, 1544, 15), '<', subscript_call_result_282203, subscript_call_result_282208)
    
    # Testing the type of an if condition (line 1544)
    if_condition_282210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1544, 12), result_lt_282209)
    # Assigning a type to the variable 'if_condition_282210' (line 1544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1544, 12), 'if_condition_282210', if_condition_282210)
    # SSA begins for if statement (line 1544)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1545):
    
    # Assigning a Call to a Name (line 1545):
    
    # Call to list(...): (line 1545)
    # Processing the call arguments (line 1545)
    # Getting the type of 'x' (line 1545)
    x_282212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 33), 'x', False)
    # Obtaining the member 'shape' of a type (line 1545)
    shape_282213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1545, 33), x_282212, 'shape')
    # Processing the call keyword arguments (line 1545)
    kwargs_282214 = {}
    # Getting the type of 'list' (line 1545)
    list_282211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1545, 28), 'list', False)
    # Calling list(args, kwargs) (line 1545)
    list_call_result_282215 = invoke(stypy.reporting.localization.Localization(__file__, 1545, 28), list_282211, *[shape_282213], **kwargs_282214)
    
    # Assigning a type to the variable 'pad_shape' (line 1545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1545, 16), 'pad_shape', list_call_result_282215)
    
    # Assigning a BinOp to a Subscript (line 1546):
    
    # Assigning a BinOp to a Subscript (line 1546):
    
    # Obtaining the type of the subscript
    int_282216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 40), 'int')
    # Getting the type of 'y' (line 1546)
    y_282217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 32), 'y')
    # Obtaining the member 'shape' of a type (line 1546)
    shape_282218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 32), y_282217, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___282219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 32), shape_282218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1546)
    subscript_call_result_282220 = invoke(stypy.reporting.localization.Localization(__file__, 1546, 32), getitem___282219, int_282216)
    
    
    # Obtaining the type of the subscript
    int_282221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 54), 'int')
    # Getting the type of 'x' (line 1546)
    x_282222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 46), 'x')
    # Obtaining the member 'shape' of a type (line 1546)
    shape_282223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 46), x_282222, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___282224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 46), shape_282223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1546)
    subscript_call_result_282225 = invoke(stypy.reporting.localization.Localization(__file__, 1546, 46), getitem___282224, int_282221)
    
    # Applying the binary operator '-' (line 1546)
    result_sub_282226 = python_operator(stypy.reporting.localization.Localization(__file__, 1546, 32), '-', subscript_call_result_282220, subscript_call_result_282225)
    
    # Getting the type of 'pad_shape' (line 1546)
    pad_shape_282227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 16), 'pad_shape')
    int_282228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 26), 'int')
    # Storing an element on a container (line 1546)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1546, 16), pad_shape_282227, (int_282228, result_sub_282226))
    
    # Assigning a Call to a Name (line 1547):
    
    # Assigning a Call to a Name (line 1547):
    
    # Call to concatenate(...): (line 1547)
    # Processing the call arguments (line 1547)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1547)
    tuple_282231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1547, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1547)
    # Adding element type (line 1547)
    # Getting the type of 'x' (line 1547)
    x_282232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 36), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1547, 36), tuple_282231, x_282232)
    # Adding element type (line 1547)
    
    # Call to zeros(...): (line 1547)
    # Processing the call arguments (line 1547)
    # Getting the type of 'pad_shape' (line 1547)
    pad_shape_282235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 48), 'pad_shape', False)
    # Processing the call keyword arguments (line 1547)
    kwargs_282236 = {}
    # Getting the type of 'np' (line 1547)
    np_282233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 39), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1547)
    zeros_282234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 39), np_282233, 'zeros')
    # Calling zeros(args, kwargs) (line 1547)
    zeros_call_result_282237 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 39), zeros_282234, *[pad_shape_282235], **kwargs_282236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1547, 36), tuple_282231, zeros_call_result_282237)
    
    int_282238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1547, 61), 'int')
    # Processing the call keyword arguments (line 1547)
    kwargs_282239 = {}
    # Getting the type of 'np' (line 1547)
    np_282229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 20), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1547)
    concatenate_282230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 20), np_282229, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1547)
    concatenate_call_result_282240 = invoke(stypy.reporting.localization.Localization(__file__, 1547, 20), concatenate_282230, *[tuple_282231, int_282238], **kwargs_282239)
    
    # Assigning a type to the variable 'x' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 16), 'x', concatenate_call_result_282240)
    # SSA branch for the else part of an if statement (line 1544)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1549):
    
    # Assigning a Call to a Name (line 1549):
    
    # Call to list(...): (line 1549)
    # Processing the call arguments (line 1549)
    # Getting the type of 'y' (line 1549)
    y_282242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 33), 'y', False)
    # Obtaining the member 'shape' of a type (line 1549)
    shape_282243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 33), y_282242, 'shape')
    # Processing the call keyword arguments (line 1549)
    kwargs_282244 = {}
    # Getting the type of 'list' (line 1549)
    list_282241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 28), 'list', False)
    # Calling list(args, kwargs) (line 1549)
    list_call_result_282245 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 28), list_282241, *[shape_282243], **kwargs_282244)
    
    # Assigning a type to the variable 'pad_shape' (line 1549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1549, 16), 'pad_shape', list_call_result_282245)
    
    # Assigning a BinOp to a Subscript (line 1550):
    
    # Assigning a BinOp to a Subscript (line 1550):
    
    # Obtaining the type of the subscript
    int_282246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1550, 40), 'int')
    # Getting the type of 'x' (line 1550)
    x_282247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 32), 'x')
    # Obtaining the member 'shape' of a type (line 1550)
    shape_282248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 32), x_282247, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1550)
    getitem___282249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 32), shape_282248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1550)
    subscript_call_result_282250 = invoke(stypy.reporting.localization.Localization(__file__, 1550, 32), getitem___282249, int_282246)
    
    
    # Obtaining the type of the subscript
    int_282251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1550, 54), 'int')
    # Getting the type of 'y' (line 1550)
    y_282252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 46), 'y')
    # Obtaining the member 'shape' of a type (line 1550)
    shape_282253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 46), y_282252, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1550)
    getitem___282254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 46), shape_282253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1550)
    subscript_call_result_282255 = invoke(stypy.reporting.localization.Localization(__file__, 1550, 46), getitem___282254, int_282251)
    
    # Applying the binary operator '-' (line 1550)
    result_sub_282256 = python_operator(stypy.reporting.localization.Localization(__file__, 1550, 32), '-', subscript_call_result_282250, subscript_call_result_282255)
    
    # Getting the type of 'pad_shape' (line 1550)
    pad_shape_282257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 16), 'pad_shape')
    int_282258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1550, 26), 'int')
    # Storing an element on a container (line 1550)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1550, 16), pad_shape_282257, (int_282258, result_sub_282256))
    
    # Assigning a Call to a Name (line 1551):
    
    # Assigning a Call to a Name (line 1551):
    
    # Call to concatenate(...): (line 1551)
    # Processing the call arguments (line 1551)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1551)
    tuple_282261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1551)
    # Adding element type (line 1551)
    # Getting the type of 'y' (line 1551)
    y_282262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 36), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1551, 36), tuple_282261, y_282262)
    # Adding element type (line 1551)
    
    # Call to zeros(...): (line 1551)
    # Processing the call arguments (line 1551)
    # Getting the type of 'pad_shape' (line 1551)
    pad_shape_282265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 48), 'pad_shape', False)
    # Processing the call keyword arguments (line 1551)
    kwargs_282266 = {}
    # Getting the type of 'np' (line 1551)
    np_282263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 39), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1551)
    zeros_282264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 39), np_282263, 'zeros')
    # Calling zeros(args, kwargs) (line 1551)
    zeros_call_result_282267 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 39), zeros_282264, *[pad_shape_282265], **kwargs_282266)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1551, 36), tuple_282261, zeros_call_result_282267)
    
    int_282268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 61), 'int')
    # Processing the call keyword arguments (line 1551)
    kwargs_282269 = {}
    # Getting the type of 'np' (line 1551)
    np_282259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 20), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1551)
    concatenate_282260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 20), np_282259, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1551)
    concatenate_call_result_282270 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 20), concatenate_282260, *[tuple_282261, int_282268], **kwargs_282269)
    
    # Assigning a type to the variable 'y' (line 1551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 16), 'y', concatenate_call_result_282270)
    # SSA join for if statement (line 1544)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1543)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1542)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1553)
    # Getting the type of 'nperseg' (line 1553)
    nperseg_282271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 4), 'nperseg')
    # Getting the type of 'None' (line 1553)
    None_282272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 22), 'None')
    
    (may_be_282273, more_types_in_union_282274) = may_not_be_none(nperseg_282271, None_282272)

    if may_be_282273:

        if more_types_in_union_282274:
            # Runtime conditional SSA (line 1553)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 1554):
        
        # Assigning a Call to a Name (line 1554):
        
        # Call to int(...): (line 1554)
        # Processing the call arguments (line 1554)
        # Getting the type of 'nperseg' (line 1554)
        nperseg_282276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 22), 'nperseg', False)
        # Processing the call keyword arguments (line 1554)
        kwargs_282277 = {}
        # Getting the type of 'int' (line 1554)
        int_282275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 18), 'int', False)
        # Calling int(args, kwargs) (line 1554)
        int_call_result_282278 = invoke(stypy.reporting.localization.Localization(__file__, 1554, 18), int_282275, *[nperseg_282276], **kwargs_282277)
        
        # Assigning a type to the variable 'nperseg' (line 1554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1554, 8), 'nperseg', int_call_result_282278)
        
        
        # Getting the type of 'nperseg' (line 1555)
        nperseg_282279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 11), 'nperseg')
        int_282280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1555, 21), 'int')
        # Applying the binary operator '<' (line 1555)
        result_lt_282281 = python_operator(stypy.reporting.localization.Localization(__file__, 1555, 11), '<', nperseg_282279, int_282280)
        
        # Testing the type of an if condition (line 1555)
        if_condition_282282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1555, 8), result_lt_282281)
        # Assigning a type to the variable 'if_condition_282282' (line 1555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1555, 8), 'if_condition_282282', if_condition_282282)
        # SSA begins for if statement (line 1555)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1556)
        # Processing the call arguments (line 1556)
        str_282284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1556, 29), 'str', 'nperseg must be a positive integer')
        # Processing the call keyword arguments (line 1556)
        kwargs_282285 = {}
        # Getting the type of 'ValueError' (line 1556)
        ValueError_282283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1556)
        ValueError_call_result_282286 = invoke(stypy.reporting.localization.Localization(__file__, 1556, 18), ValueError_282283, *[str_282284], **kwargs_282285)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1556, 12), ValueError_call_result_282286, 'raise parameter', BaseException)
        # SSA join for if statement (line 1555)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_282274:
            # SSA join for if statement (line 1553)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 1559):
    
    # Assigning a Subscript to a Name (line 1559):
    
    # Obtaining the type of the subscript
    int_282287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1559, 4), 'int')
    
    # Call to _triage_segments(...): (line 1559)
    # Processing the call arguments (line 1559)
    # Getting the type of 'window' (line 1559)
    window_282289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 36), 'window', False)
    # Getting the type of 'nperseg' (line 1559)
    nperseg_282290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 44), 'nperseg', False)
    # Processing the call keyword arguments (line 1559)
    
    # Obtaining the type of the subscript
    int_282291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1559, 73), 'int')
    # Getting the type of 'x' (line 1559)
    x_282292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 65), 'x', False)
    # Obtaining the member 'shape' of a type (line 1559)
    shape_282293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 65), x_282292, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1559)
    getitem___282294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 65), shape_282293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1559)
    subscript_call_result_282295 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 65), getitem___282294, int_282291)
    
    keyword_282296 = subscript_call_result_282295
    kwargs_282297 = {'input_length': keyword_282296}
    # Getting the type of '_triage_segments' (line 1559)
    _triage_segments_282288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 19), '_triage_segments', False)
    # Calling _triage_segments(args, kwargs) (line 1559)
    _triage_segments_call_result_282298 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 19), _triage_segments_282288, *[window_282289, nperseg_282290], **kwargs_282297)
    
    # Obtaining the member '__getitem__' of a type (line 1559)
    getitem___282299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 4), _triage_segments_call_result_282298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1559)
    subscript_call_result_282300 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 4), getitem___282299, int_282287)
    
    # Assigning a type to the variable 'tuple_var_assignment_280487' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'tuple_var_assignment_280487', subscript_call_result_282300)
    
    # Assigning a Subscript to a Name (line 1559):
    
    # Obtaining the type of the subscript
    int_282301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1559, 4), 'int')
    
    # Call to _triage_segments(...): (line 1559)
    # Processing the call arguments (line 1559)
    # Getting the type of 'window' (line 1559)
    window_282303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 36), 'window', False)
    # Getting the type of 'nperseg' (line 1559)
    nperseg_282304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 44), 'nperseg', False)
    # Processing the call keyword arguments (line 1559)
    
    # Obtaining the type of the subscript
    int_282305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1559, 73), 'int')
    # Getting the type of 'x' (line 1559)
    x_282306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 65), 'x', False)
    # Obtaining the member 'shape' of a type (line 1559)
    shape_282307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 65), x_282306, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1559)
    getitem___282308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 65), shape_282307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1559)
    subscript_call_result_282309 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 65), getitem___282308, int_282305)
    
    keyword_282310 = subscript_call_result_282309
    kwargs_282311 = {'input_length': keyword_282310}
    # Getting the type of '_triage_segments' (line 1559)
    _triage_segments_282302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 19), '_triage_segments', False)
    # Calling _triage_segments(args, kwargs) (line 1559)
    _triage_segments_call_result_282312 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 19), _triage_segments_282302, *[window_282303, nperseg_282304], **kwargs_282311)
    
    # Obtaining the member '__getitem__' of a type (line 1559)
    getitem___282313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1559, 4), _triage_segments_call_result_282312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1559)
    subscript_call_result_282314 = invoke(stypy.reporting.localization.Localization(__file__, 1559, 4), getitem___282313, int_282301)
    
    # Assigning a type to the variable 'tuple_var_assignment_280488' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'tuple_var_assignment_280488', subscript_call_result_282314)
    
    # Assigning a Name to a Name (line 1559):
    # Getting the type of 'tuple_var_assignment_280487' (line 1559)
    tuple_var_assignment_280487_282315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'tuple_var_assignment_280487')
    # Assigning a type to the variable 'win' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'win', tuple_var_assignment_280487_282315)
    
    # Assigning a Name to a Name (line 1559):
    # Getting the type of 'tuple_var_assignment_280488' (line 1559)
    tuple_var_assignment_280488_282316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'tuple_var_assignment_280488')
    # Assigning a type to the variable 'nperseg' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 9), 'nperseg', tuple_var_assignment_280488_282316)
    
    # Type idiom detected: calculating its left and rigth part (line 1561)
    # Getting the type of 'nfft' (line 1561)
    nfft_282317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 7), 'nfft')
    # Getting the type of 'None' (line 1561)
    None_282318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 15), 'None')
    
    (may_be_282319, more_types_in_union_282320) = may_be_none(nfft_282317, None_282318)

    if may_be_282319:

        if more_types_in_union_282320:
            # Runtime conditional SSA (line 1561)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 1562):
        
        # Assigning a Name to a Name (line 1562):
        # Getting the type of 'nperseg' (line 1562)
        nperseg_282321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1562, 15), 'nperseg')
        # Assigning a type to the variable 'nfft' (line 1562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1562, 8), 'nfft', nperseg_282321)

        if more_types_in_union_282320:
            # Runtime conditional SSA for else branch (line 1561)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_282319) or more_types_in_union_282320):
        
        
        # Getting the type of 'nfft' (line 1563)
        nfft_282322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 9), 'nfft')
        # Getting the type of 'nperseg' (line 1563)
        nperseg_282323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 16), 'nperseg')
        # Applying the binary operator '<' (line 1563)
        result_lt_282324 = python_operator(stypy.reporting.localization.Localization(__file__, 1563, 9), '<', nfft_282322, nperseg_282323)
        
        # Testing the type of an if condition (line 1563)
        if_condition_282325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1563, 9), result_lt_282324)
        # Assigning a type to the variable 'if_condition_282325' (line 1563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1563, 9), 'if_condition_282325', if_condition_282325)
        # SSA begins for if statement (line 1563)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 1564)
        # Processing the call arguments (line 1564)
        str_282327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1564, 25), 'str', 'nfft must be greater than or equal to nperseg.')
        # Processing the call keyword arguments (line 1564)
        kwargs_282328 = {}
        # Getting the type of 'ValueError' (line 1564)
        ValueError_282326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 1564)
        ValueError_call_result_282329 = invoke(stypy.reporting.localization.Localization(__file__, 1564, 14), ValueError_282326, *[str_282327], **kwargs_282328)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1564, 8), ValueError_call_result_282329, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 1563)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 1566):
        
        # Assigning a Call to a Name (line 1566):
        
        # Call to int(...): (line 1566)
        # Processing the call arguments (line 1566)
        # Getting the type of 'nfft' (line 1566)
        nfft_282331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 19), 'nfft', False)
        # Processing the call keyword arguments (line 1566)
        kwargs_282332 = {}
        # Getting the type of 'int' (line 1566)
        int_282330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 15), 'int', False)
        # Calling int(args, kwargs) (line 1566)
        int_call_result_282333 = invoke(stypy.reporting.localization.Localization(__file__, 1566, 15), int_282330, *[nfft_282331], **kwargs_282332)
        
        # Assigning a type to the variable 'nfft' (line 1566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1566, 8), 'nfft', int_call_result_282333)
        # SSA join for if statement (line 1563)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_282319 and more_types_in_union_282320):
            # SSA join for if statement (line 1561)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1568)
    # Getting the type of 'noverlap' (line 1568)
    noverlap_282334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1568, 7), 'noverlap')
    # Getting the type of 'None' (line 1568)
    None_282335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1568, 19), 'None')
    
    (may_be_282336, more_types_in_union_282337) = may_be_none(noverlap_282334, None_282335)

    if may_be_282336:

        if more_types_in_union_282337:
            # Runtime conditional SSA (line 1568)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1569):
        
        # Assigning a BinOp to a Name (line 1569):
        # Getting the type of 'nperseg' (line 1569)
        nperseg_282338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1569, 19), 'nperseg')
        int_282339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1569, 28), 'int')
        # Applying the binary operator '//' (line 1569)
        result_floordiv_282340 = python_operator(stypy.reporting.localization.Localization(__file__, 1569, 19), '//', nperseg_282338, int_282339)
        
        # Assigning a type to the variable 'noverlap' (line 1569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1569, 8), 'noverlap', result_floordiv_282340)

        if more_types_in_union_282337:
            # Runtime conditional SSA for else branch (line 1568)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_282336) or more_types_in_union_282337):
        
        # Assigning a Call to a Name (line 1571):
        
        # Assigning a Call to a Name (line 1571):
        
        # Call to int(...): (line 1571)
        # Processing the call arguments (line 1571)
        # Getting the type of 'noverlap' (line 1571)
        noverlap_282342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1571, 23), 'noverlap', False)
        # Processing the call keyword arguments (line 1571)
        kwargs_282343 = {}
        # Getting the type of 'int' (line 1571)
        int_282341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1571, 19), 'int', False)
        # Calling int(args, kwargs) (line 1571)
        int_call_result_282344 = invoke(stypy.reporting.localization.Localization(__file__, 1571, 19), int_282341, *[noverlap_282342], **kwargs_282343)
        
        # Assigning a type to the variable 'noverlap' (line 1571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1571, 8), 'noverlap', int_call_result_282344)

        if (may_be_282336 and more_types_in_union_282337):
            # SSA join for if statement (line 1568)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'noverlap' (line 1572)
    noverlap_282345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1572, 7), 'noverlap')
    # Getting the type of 'nperseg' (line 1572)
    nperseg_282346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1572, 19), 'nperseg')
    # Applying the binary operator '>=' (line 1572)
    result_ge_282347 = python_operator(stypy.reporting.localization.Localization(__file__, 1572, 7), '>=', noverlap_282345, nperseg_282346)
    
    # Testing the type of an if condition (line 1572)
    if_condition_282348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1572, 4), result_ge_282347)
    # Assigning a type to the variable 'if_condition_282348' (line 1572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1572, 4), 'if_condition_282348', if_condition_282348)
    # SSA begins for if statement (line 1572)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1573)
    # Processing the call arguments (line 1573)
    str_282350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 25), 'str', 'noverlap must be less than nperseg.')
    # Processing the call keyword arguments (line 1573)
    kwargs_282351 = {}
    # Getting the type of 'ValueError' (line 1573)
    ValueError_282349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1573)
    ValueError_call_result_282352 = invoke(stypy.reporting.localization.Localization(__file__, 1573, 14), ValueError_282349, *[str_282350], **kwargs_282351)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1573, 8), ValueError_call_result_282352, 'raise parameter', BaseException)
    # SSA join for if statement (line 1572)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1574):
    
    # Assigning a BinOp to a Name (line 1574):
    # Getting the type of 'nperseg' (line 1574)
    nperseg_282353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 12), 'nperseg')
    # Getting the type of 'noverlap' (line 1574)
    noverlap_282354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1574, 22), 'noverlap')
    # Applying the binary operator '-' (line 1574)
    result_sub_282355 = python_operator(stypy.reporting.localization.Localization(__file__, 1574, 12), '-', nperseg_282353, noverlap_282354)
    
    # Assigning a type to the variable 'nstep' (line 1574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1574, 4), 'nstep', result_sub_282355)
    
    # Type idiom detected: calculating its left and rigth part (line 1582)
    # Getting the type of 'boundary' (line 1582)
    boundary_282356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 4), 'boundary')
    # Getting the type of 'None' (line 1582)
    None_282357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 23), 'None')
    
    (may_be_282358, more_types_in_union_282359) = may_not_be_none(boundary_282356, None_282357)

    if may_be_282358:

        if more_types_in_union_282359:
            # Runtime conditional SSA (line 1582)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 1583):
        
        # Assigning a Subscript to a Name (line 1583):
        
        # Obtaining the type of the subscript
        # Getting the type of 'boundary' (line 1583)
        boundary_282360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 34), 'boundary')
        # Getting the type of 'boundary_funcs' (line 1583)
        boundary_funcs_282361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 19), 'boundary_funcs')
        # Obtaining the member '__getitem__' of a type (line 1583)
        getitem___282362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1583, 19), boundary_funcs_282361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1583)
        subscript_call_result_282363 = invoke(stypy.reporting.localization.Localization(__file__, 1583, 19), getitem___282362, boundary_282360)
        
        # Assigning a type to the variable 'ext_func' (line 1583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1583, 8), 'ext_func', subscript_call_result_282363)
        
        # Assigning a Call to a Name (line 1584):
        
        # Assigning a Call to a Name (line 1584):
        
        # Call to ext_func(...): (line 1584)
        # Processing the call arguments (line 1584)
        # Getting the type of 'x' (line 1584)
        x_282365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 21), 'x', False)
        # Getting the type of 'nperseg' (line 1584)
        nperseg_282366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 24), 'nperseg', False)
        int_282367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1584, 33), 'int')
        # Applying the binary operator '//' (line 1584)
        result_floordiv_282368 = python_operator(stypy.reporting.localization.Localization(__file__, 1584, 24), '//', nperseg_282366, int_282367)
        
        # Processing the call keyword arguments (line 1584)
        int_282369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1584, 41), 'int')
        keyword_282370 = int_282369
        kwargs_282371 = {'axis': keyword_282370}
        # Getting the type of 'ext_func' (line 1584)
        ext_func_282364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 12), 'ext_func', False)
        # Calling ext_func(args, kwargs) (line 1584)
        ext_func_call_result_282372 = invoke(stypy.reporting.localization.Localization(__file__, 1584, 12), ext_func_282364, *[x_282365, result_floordiv_282368], **kwargs_282371)
        
        # Assigning a type to the variable 'x' (line 1584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 8), 'x', ext_func_call_result_282372)
        
        
        # Getting the type of 'same_data' (line 1585)
        same_data_282373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 15), 'same_data')
        # Applying the 'not' unary operator (line 1585)
        result_not__282374 = python_operator(stypy.reporting.localization.Localization(__file__, 1585, 11), 'not', same_data_282373)
        
        # Testing the type of an if condition (line 1585)
        if_condition_282375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1585, 8), result_not__282374)
        # Assigning a type to the variable 'if_condition_282375' (line 1585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 8), 'if_condition_282375', if_condition_282375)
        # SSA begins for if statement (line 1585)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 1586):
        
        # Assigning a Call to a Name (line 1586):
        
        # Call to ext_func(...): (line 1586)
        # Processing the call arguments (line 1586)
        # Getting the type of 'y' (line 1586)
        y_282377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 25), 'y', False)
        # Getting the type of 'nperseg' (line 1586)
        nperseg_282378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 28), 'nperseg', False)
        int_282379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1586, 37), 'int')
        # Applying the binary operator '//' (line 1586)
        result_floordiv_282380 = python_operator(stypy.reporting.localization.Localization(__file__, 1586, 28), '//', nperseg_282378, int_282379)
        
        # Processing the call keyword arguments (line 1586)
        int_282381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1586, 45), 'int')
        keyword_282382 = int_282381
        kwargs_282383 = {'axis': keyword_282382}
        # Getting the type of 'ext_func' (line 1586)
        ext_func_282376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 16), 'ext_func', False)
        # Calling ext_func(args, kwargs) (line 1586)
        ext_func_call_result_282384 = invoke(stypy.reporting.localization.Localization(__file__, 1586, 16), ext_func_282376, *[y_282377, result_floordiv_282380], **kwargs_282383)
        
        # Assigning a type to the variable 'y' (line 1586)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1586, 12), 'y', ext_func_call_result_282384)
        # SSA join for if statement (line 1585)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_282359:
            # SSA join for if statement (line 1582)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'padded' (line 1588)
    padded_282385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 7), 'padded')
    # Testing the type of an if condition (line 1588)
    if_condition_282386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1588, 4), padded_282385)
    # Assigning a type to the variable 'if_condition_282386' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'if_condition_282386', if_condition_282386)
    # SSA begins for if statement (line 1588)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1591):
    
    # Assigning a BinOp to a Name (line 1591):
    
    
    # Obtaining the type of the subscript
    int_282387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1591, 26), 'int')
    # Getting the type of 'x' (line 1591)
    x_282388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 18), 'x')
    # Obtaining the member 'shape' of a type (line 1591)
    shape_282389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1591, 18), x_282388, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1591)
    getitem___282390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1591, 18), shape_282389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1591)
    subscript_call_result_282391 = invoke(stypy.reporting.localization.Localization(__file__, 1591, 18), getitem___282390, int_282387)
    
    # Getting the type of 'nperseg' (line 1591)
    nperseg_282392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 30), 'nperseg')
    # Applying the binary operator '-' (line 1591)
    result_sub_282393 = python_operator(stypy.reporting.localization.Localization(__file__, 1591, 18), '-', subscript_call_result_282391, nperseg_282392)
    
    # Applying the 'usub' unary operator (line 1591)
    result___neg___282394 = python_operator(stypy.reporting.localization.Localization(__file__, 1591, 16), 'usub', result_sub_282393)
    
    # Getting the type of 'nstep' (line 1591)
    nstep_282395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 41), 'nstep')
    # Applying the binary operator '%' (line 1591)
    result_mod_282396 = python_operator(stypy.reporting.localization.Localization(__file__, 1591, 16), '%', result___neg___282394, nstep_282395)
    
    # Getting the type of 'nperseg' (line 1591)
    nperseg_282397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 50), 'nperseg')
    # Applying the binary operator '%' (line 1591)
    result_mod_282398 = python_operator(stypy.reporting.localization.Localization(__file__, 1591, 15), '%', result_mod_282396, nperseg_282397)
    
    # Assigning a type to the variable 'nadd' (line 1591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1591, 8), 'nadd', result_mod_282398)
    
    # Assigning a BinOp to a Name (line 1592):
    
    # Assigning a BinOp to a Name (line 1592):
    
    # Call to list(...): (line 1592)
    # Processing the call arguments (line 1592)
    
    # Obtaining the type of the subscript
    int_282400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 36), 'int')
    slice_282401 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1592, 27), None, int_282400, None)
    # Getting the type of 'x' (line 1592)
    x_282402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 27), 'x', False)
    # Obtaining the member 'shape' of a type (line 1592)
    shape_282403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1592, 27), x_282402, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1592)
    getitem___282404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1592, 27), shape_282403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1592)
    subscript_call_result_282405 = invoke(stypy.reporting.localization.Localization(__file__, 1592, 27), getitem___282404, slice_282401)
    
    # Processing the call keyword arguments (line 1592)
    kwargs_282406 = {}
    # Getting the type of 'list' (line 1592)
    list_282399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 22), 'list', False)
    # Calling list(args, kwargs) (line 1592)
    list_call_result_282407 = invoke(stypy.reporting.localization.Localization(__file__, 1592, 22), list_282399, *[subscript_call_result_282405], **kwargs_282406)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1592)
    list_282408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1592)
    # Adding element type (line 1592)
    # Getting the type of 'nadd' (line 1592)
    nadd_282409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 44), 'nadd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1592, 43), list_282408, nadd_282409)
    
    # Applying the binary operator '+' (line 1592)
    result_add_282410 = python_operator(stypy.reporting.localization.Localization(__file__, 1592, 22), '+', list_call_result_282407, list_282408)
    
    # Assigning a type to the variable 'zeros_shape' (line 1592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1592, 8), 'zeros_shape', result_add_282410)
    
    # Assigning a Call to a Name (line 1593):
    
    # Assigning a Call to a Name (line 1593):
    
    # Call to concatenate(...): (line 1593)
    # Processing the call arguments (line 1593)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1593)
    tuple_282413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1593, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1593)
    # Adding element type (line 1593)
    # Getting the type of 'x' (line 1593)
    x_282414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1593, 28), tuple_282413, x_282414)
    # Adding element type (line 1593)
    
    # Call to zeros(...): (line 1593)
    # Processing the call arguments (line 1593)
    # Getting the type of 'zeros_shape' (line 1593)
    zeros_shape_282417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 40), 'zeros_shape', False)
    # Processing the call keyword arguments (line 1593)
    kwargs_282418 = {}
    # Getting the type of 'np' (line 1593)
    np_282415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 31), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1593)
    zeros_282416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1593, 31), np_282415, 'zeros')
    # Calling zeros(args, kwargs) (line 1593)
    zeros_call_result_282419 = invoke(stypy.reporting.localization.Localization(__file__, 1593, 31), zeros_282416, *[zeros_shape_282417], **kwargs_282418)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1593, 28), tuple_282413, zeros_call_result_282419)
    
    # Processing the call keyword arguments (line 1593)
    int_282420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1593, 60), 'int')
    keyword_282421 = int_282420
    kwargs_282422 = {'axis': keyword_282421}
    # Getting the type of 'np' (line 1593)
    np_282411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1593, 12), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1593)
    concatenate_282412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1593, 12), np_282411, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1593)
    concatenate_call_result_282423 = invoke(stypy.reporting.localization.Localization(__file__, 1593, 12), concatenate_282412, *[tuple_282413], **kwargs_282422)
    
    # Assigning a type to the variable 'x' (line 1593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1593, 8), 'x', concatenate_call_result_282423)
    
    
    # Getting the type of 'same_data' (line 1594)
    same_data_282424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 15), 'same_data')
    # Applying the 'not' unary operator (line 1594)
    result_not__282425 = python_operator(stypy.reporting.localization.Localization(__file__, 1594, 11), 'not', same_data_282424)
    
    # Testing the type of an if condition (line 1594)
    if_condition_282426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1594, 8), result_not__282425)
    # Assigning a type to the variable 'if_condition_282426' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 8), 'if_condition_282426', if_condition_282426)
    # SSA begins for if statement (line 1594)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1595):
    
    # Assigning a BinOp to a Name (line 1595):
    
    # Call to list(...): (line 1595)
    # Processing the call arguments (line 1595)
    
    # Obtaining the type of the subscript
    int_282428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1595, 40), 'int')
    slice_282429 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1595, 31), None, int_282428, None)
    # Getting the type of 'y' (line 1595)
    y_282430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 31), 'y', False)
    # Obtaining the member 'shape' of a type (line 1595)
    shape_282431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1595, 31), y_282430, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1595)
    getitem___282432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1595, 31), shape_282431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1595)
    subscript_call_result_282433 = invoke(stypy.reporting.localization.Localization(__file__, 1595, 31), getitem___282432, slice_282429)
    
    # Processing the call keyword arguments (line 1595)
    kwargs_282434 = {}
    # Getting the type of 'list' (line 1595)
    list_282427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 26), 'list', False)
    # Calling list(args, kwargs) (line 1595)
    list_call_result_282435 = invoke(stypy.reporting.localization.Localization(__file__, 1595, 26), list_282427, *[subscript_call_result_282433], **kwargs_282434)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1595)
    list_282436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1595, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1595)
    # Adding element type (line 1595)
    # Getting the type of 'nadd' (line 1595)
    nadd_282437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 48), 'nadd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1595, 47), list_282436, nadd_282437)
    
    # Applying the binary operator '+' (line 1595)
    result_add_282438 = python_operator(stypy.reporting.localization.Localization(__file__, 1595, 26), '+', list_call_result_282435, list_282436)
    
    # Assigning a type to the variable 'zeros_shape' (line 1595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1595, 12), 'zeros_shape', result_add_282438)
    
    # Assigning a Call to a Name (line 1596):
    
    # Assigning a Call to a Name (line 1596):
    
    # Call to concatenate(...): (line 1596)
    # Processing the call arguments (line 1596)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1596)
    tuple_282441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1596)
    # Adding element type (line 1596)
    # Getting the type of 'y' (line 1596)
    y_282442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 32), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1596, 32), tuple_282441, y_282442)
    # Adding element type (line 1596)
    
    # Call to zeros(...): (line 1596)
    # Processing the call arguments (line 1596)
    # Getting the type of 'zeros_shape' (line 1596)
    zeros_shape_282445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 44), 'zeros_shape', False)
    # Processing the call keyword arguments (line 1596)
    kwargs_282446 = {}
    # Getting the type of 'np' (line 1596)
    np_282443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 35), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1596)
    zeros_282444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 35), np_282443, 'zeros')
    # Calling zeros(args, kwargs) (line 1596)
    zeros_call_result_282447 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 35), zeros_282444, *[zeros_shape_282445], **kwargs_282446)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1596, 32), tuple_282441, zeros_call_result_282447)
    
    # Processing the call keyword arguments (line 1596)
    int_282448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 64), 'int')
    keyword_282449 = int_282448
    kwargs_282450 = {'axis': keyword_282449}
    # Getting the type of 'np' (line 1596)
    np_282439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 16), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 1596)
    concatenate_282440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 16), np_282439, 'concatenate')
    # Calling concatenate(args, kwargs) (line 1596)
    concatenate_call_result_282451 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 16), concatenate_282440, *[tuple_282441], **kwargs_282450)
    
    # Assigning a type to the variable 'y' (line 1596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 12), 'y', concatenate_call_result_282451)
    # SSA join for if statement (line 1594)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1588)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'detrend' (line 1599)
    detrend_282452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1599, 11), 'detrend')
    # Applying the 'not' unary operator (line 1599)
    result_not__282453 = python_operator(stypy.reporting.localization.Localization(__file__, 1599, 7), 'not', detrend_282452)
    
    # Testing the type of an if condition (line 1599)
    if_condition_282454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1599, 4), result_not__282453)
    # Assigning a type to the variable 'if_condition_282454' (line 1599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1599, 4), 'if_condition_282454', if_condition_282454)
    # SSA begins for if statement (line 1599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def detrend_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'detrend_func'
        module_type_store = module_type_store.open_function_context('detrend_func', 1600, 8, False)
        
        # Passed parameters checking function
        detrend_func.stypy_localization = localization
        detrend_func.stypy_type_of_self = None
        detrend_func.stypy_type_store = module_type_store
        detrend_func.stypy_function_name = 'detrend_func'
        detrend_func.stypy_param_names_list = ['d']
        detrend_func.stypy_varargs_param_name = None
        detrend_func.stypy_kwargs_param_name = None
        detrend_func.stypy_call_defaults = defaults
        detrend_func.stypy_call_varargs = varargs
        detrend_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'detrend_func', ['d'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'detrend_func', localization, ['d'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'detrend_func(...)' code ##################

        # Getting the type of 'd' (line 1601)
        d_282455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 19), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 1601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 12), 'stypy_return_type', d_282455)
        
        # ################# End of 'detrend_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'detrend_func' in the type store
        # Getting the type of 'stypy_return_type' (line 1600)
        stypy_return_type_282456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_282456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'detrend_func'
        return stypy_return_type_282456

    # Assigning a type to the variable 'detrend_func' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 8), 'detrend_func', detrend_func)
    # SSA branch for the else part of an if statement (line 1599)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 1602)
    str_282457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 30), 'str', '__call__')
    # Getting the type of 'detrend' (line 1602)
    detrend_282458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 21), 'detrend')
    
    (may_be_282459, more_types_in_union_282460) = may_not_provide_member(str_282457, detrend_282458)

    if may_be_282459:

        if more_types_in_union_282460:
            # Runtime conditional SSA (line 1602)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'detrend' (line 1602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 9), 'detrend', remove_member_provider_from_union(detrend_282458, '__call__'))

        @norecursion
        def detrend_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'detrend_func'
            module_type_store = module_type_store.open_function_context('detrend_func', 1603, 8, False)
            
            # Passed parameters checking function
            detrend_func.stypy_localization = localization
            detrend_func.stypy_type_of_self = None
            detrend_func.stypy_type_store = module_type_store
            detrend_func.stypy_function_name = 'detrend_func'
            detrend_func.stypy_param_names_list = ['d']
            detrend_func.stypy_varargs_param_name = None
            detrend_func.stypy_kwargs_param_name = None
            detrend_func.stypy_call_defaults = defaults
            detrend_func.stypy_call_varargs = varargs
            detrend_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'detrend_func', ['d'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'detrend_func', localization, ['d'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'detrend_func(...)' code ##################

            
            # Call to detrend(...): (line 1604)
            # Processing the call arguments (line 1604)
            # Getting the type of 'd' (line 1604)
            d_282463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 39), 'd', False)
            # Processing the call keyword arguments (line 1604)
            # Getting the type of 'detrend' (line 1604)
            detrend_282464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 47), 'detrend', False)
            keyword_282465 = detrend_282464
            int_282466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 61), 'int')
            keyword_282467 = int_282466
            kwargs_282468 = {'type': keyword_282465, 'axis': keyword_282467}
            # Getting the type of 'signaltools' (line 1604)
            signaltools_282461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 19), 'signaltools', False)
            # Obtaining the member 'detrend' of a type (line 1604)
            detrend_282462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 19), signaltools_282461, 'detrend')
            # Calling detrend(args, kwargs) (line 1604)
            detrend_call_result_282469 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 19), detrend_282462, *[d_282463], **kwargs_282468)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1604)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1604, 12), 'stypy_return_type', detrend_call_result_282469)
            
            # ################# End of 'detrend_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'detrend_func' in the type store
            # Getting the type of 'stypy_return_type' (line 1603)
            stypy_return_type_282470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_282470)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'detrend_func'
            return stypy_return_type_282470

        # Assigning a type to the variable 'detrend_func' (line 1603)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1603, 8), 'detrend_func', detrend_func)

        if more_types_in_union_282460:
            # Runtime conditional SSA for else branch (line 1602)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_282459) or more_types_in_union_282460):
        # Assigning a type to the variable 'detrend' (line 1602)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 9), 'detrend', remove_not_member_provider_from_union(detrend_282458, '__call__'))
        
        
        # Getting the type of 'axis' (line 1605)
        axis_282471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 9), 'axis')
        int_282472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 17), 'int')
        # Applying the binary operator '!=' (line 1605)
        result_ne_282473 = python_operator(stypy.reporting.localization.Localization(__file__, 1605, 9), '!=', axis_282471, int_282472)
        
        # Testing the type of an if condition (line 1605)
        if_condition_282474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1605, 9), result_ne_282473)
        # Assigning a type to the variable 'if_condition_282474' (line 1605)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1605, 9), 'if_condition_282474', if_condition_282474)
        # SSA begins for if statement (line 1605)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def detrend_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'detrend_func'
            module_type_store = module_type_store.open_function_context('detrend_func', 1608, 8, False)
            
            # Passed parameters checking function
            detrend_func.stypy_localization = localization
            detrend_func.stypy_type_of_self = None
            detrend_func.stypy_type_store = module_type_store
            detrend_func.stypy_function_name = 'detrend_func'
            detrend_func.stypy_param_names_list = ['d']
            detrend_func.stypy_varargs_param_name = None
            detrend_func.stypy_kwargs_param_name = None
            detrend_func.stypy_call_defaults = defaults
            detrend_func.stypy_call_varargs = varargs
            detrend_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'detrend_func', ['d'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'detrend_func', localization, ['d'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'detrend_func(...)' code ##################

            
            # Assigning a Call to a Name (line 1609):
            
            # Assigning a Call to a Name (line 1609):
            
            # Call to rollaxis(...): (line 1609)
            # Processing the call arguments (line 1609)
            # Getting the type of 'd' (line 1609)
            d_282477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 28), 'd', False)
            int_282478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1609, 31), 'int')
            # Getting the type of 'axis' (line 1609)
            axis_282479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 35), 'axis', False)
            # Processing the call keyword arguments (line 1609)
            kwargs_282480 = {}
            # Getting the type of 'np' (line 1609)
            np_282475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 16), 'np', False)
            # Obtaining the member 'rollaxis' of a type (line 1609)
            rollaxis_282476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1609, 16), np_282475, 'rollaxis')
            # Calling rollaxis(args, kwargs) (line 1609)
            rollaxis_call_result_282481 = invoke(stypy.reporting.localization.Localization(__file__, 1609, 16), rollaxis_282476, *[d_282477, int_282478, axis_282479], **kwargs_282480)
            
            # Assigning a type to the variable 'd' (line 1609)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1609, 12), 'd', rollaxis_call_result_282481)
            
            # Assigning a Call to a Name (line 1610):
            
            # Assigning a Call to a Name (line 1610):
            
            # Call to detrend(...): (line 1610)
            # Processing the call arguments (line 1610)
            # Getting the type of 'd' (line 1610)
            d_282483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 24), 'd', False)
            # Processing the call keyword arguments (line 1610)
            kwargs_282484 = {}
            # Getting the type of 'detrend' (line 1610)
            detrend_282482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 16), 'detrend', False)
            # Calling detrend(args, kwargs) (line 1610)
            detrend_call_result_282485 = invoke(stypy.reporting.localization.Localization(__file__, 1610, 16), detrend_282482, *[d_282483], **kwargs_282484)
            
            # Assigning a type to the variable 'd' (line 1610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1610, 12), 'd', detrend_call_result_282485)
            
            # Call to rollaxis(...): (line 1611)
            # Processing the call arguments (line 1611)
            # Getting the type of 'd' (line 1611)
            d_282488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 31), 'd', False)
            # Getting the type of 'axis' (line 1611)
            axis_282489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 34), 'axis', False)
            
            # Call to len(...): (line 1611)
            # Processing the call arguments (line 1611)
            # Getting the type of 'd' (line 1611)
            d_282491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 44), 'd', False)
            # Obtaining the member 'shape' of a type (line 1611)
            shape_282492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1611, 44), d_282491, 'shape')
            # Processing the call keyword arguments (line 1611)
            kwargs_282493 = {}
            # Getting the type of 'len' (line 1611)
            len_282490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 40), 'len', False)
            # Calling len(args, kwargs) (line 1611)
            len_call_result_282494 = invoke(stypy.reporting.localization.Localization(__file__, 1611, 40), len_282490, *[shape_282492], **kwargs_282493)
            
            # Processing the call keyword arguments (line 1611)
            kwargs_282495 = {}
            # Getting the type of 'np' (line 1611)
            np_282486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 19), 'np', False)
            # Obtaining the member 'rollaxis' of a type (line 1611)
            rollaxis_282487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1611, 19), np_282486, 'rollaxis')
            # Calling rollaxis(args, kwargs) (line 1611)
            rollaxis_call_result_282496 = invoke(stypy.reporting.localization.Localization(__file__, 1611, 19), rollaxis_282487, *[d_282488, axis_282489, len_call_result_282494], **kwargs_282495)
            
            # Assigning a type to the variable 'stypy_return_type' (line 1611)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1611, 12), 'stypy_return_type', rollaxis_call_result_282496)
            
            # ################# End of 'detrend_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'detrend_func' in the type store
            # Getting the type of 'stypy_return_type' (line 1608)
            stypy_return_type_282497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_282497)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'detrend_func'
            return stypy_return_type_282497

        # Assigning a type to the variable 'detrend_func' (line 1608)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1608, 8), 'detrend_func', detrend_func)
        # SSA branch for the else part of an if statement (line 1605)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 1613):
        
        # Assigning a Name to a Name (line 1613):
        # Getting the type of 'detrend' (line 1613)
        detrend_282498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1613, 23), 'detrend')
        # Assigning a type to the variable 'detrend_func' (line 1613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1613, 8), 'detrend_func', detrend_282498)
        # SSA join for if statement (line 1605)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_282459 and more_types_in_union_282460):
            # SSA join for if statement (line 1602)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 1599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to result_type(...): (line 1615)
    # Processing the call arguments (line 1615)
    # Getting the type of 'win' (line 1615)
    win_282501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 22), 'win', False)
    # Getting the type of 'np' (line 1615)
    np_282502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 26), 'np', False)
    # Obtaining the member 'complex64' of a type (line 1615)
    complex64_282503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1615, 26), np_282502, 'complex64')
    # Processing the call keyword arguments (line 1615)
    kwargs_282504 = {}
    # Getting the type of 'np' (line 1615)
    np_282499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 7), 'np', False)
    # Obtaining the member 'result_type' of a type (line 1615)
    result_type_282500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1615, 7), np_282499, 'result_type')
    # Calling result_type(args, kwargs) (line 1615)
    result_type_call_result_282505 = invoke(stypy.reporting.localization.Localization(__file__, 1615, 7), result_type_282500, *[win_282501, complex64_282503], **kwargs_282504)
    
    # Getting the type of 'outdtype' (line 1615)
    outdtype_282506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1615, 43), 'outdtype')
    # Applying the binary operator '!=' (line 1615)
    result_ne_282507 = python_operator(stypy.reporting.localization.Localization(__file__, 1615, 7), '!=', result_type_call_result_282505, outdtype_282506)
    
    # Testing the type of an if condition (line 1615)
    if_condition_282508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1615, 4), result_ne_282507)
    # Assigning a type to the variable 'if_condition_282508' (line 1615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1615, 4), 'if_condition_282508', if_condition_282508)
    # SSA begins for if statement (line 1615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1616):
    
    # Assigning a Call to a Name (line 1616):
    
    # Call to astype(...): (line 1616)
    # Processing the call arguments (line 1616)
    # Getting the type of 'outdtype' (line 1616)
    outdtype_282511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1616, 25), 'outdtype', False)
    # Processing the call keyword arguments (line 1616)
    kwargs_282512 = {}
    # Getting the type of 'win' (line 1616)
    win_282509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1616, 14), 'win', False)
    # Obtaining the member 'astype' of a type (line 1616)
    astype_282510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1616, 14), win_282509, 'astype')
    # Calling astype(args, kwargs) (line 1616)
    astype_call_result_282513 = invoke(stypy.reporting.localization.Localization(__file__, 1616, 14), astype_282510, *[outdtype_282511], **kwargs_282512)
    
    # Assigning a type to the variable 'win' (line 1616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1616, 8), 'win', astype_call_result_282513)
    # SSA join for if statement (line 1615)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'scaling' (line 1618)
    scaling_282514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1618, 7), 'scaling')
    str_282515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1618, 18), 'str', 'density')
    # Applying the binary operator '==' (line 1618)
    result_eq_282516 = python_operator(stypy.reporting.localization.Localization(__file__, 1618, 7), '==', scaling_282514, str_282515)
    
    # Testing the type of an if condition (line 1618)
    if_condition_282517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1618, 4), result_eq_282516)
    # Assigning a type to the variable 'if_condition_282517' (line 1618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1618, 4), 'if_condition_282517', if_condition_282517)
    # SSA begins for if statement (line 1618)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1619):
    
    # Assigning a BinOp to a Name (line 1619):
    float_282518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1619, 16), 'float')
    # Getting the type of 'fs' (line 1619)
    fs_282519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 23), 'fs')
    
    # Call to sum(...): (line 1619)
    # Processing the call keyword arguments (line 1619)
    kwargs_282524 = {}
    # Getting the type of 'win' (line 1619)
    win_282520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 29), 'win', False)
    # Getting the type of 'win' (line 1619)
    win_282521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 33), 'win', False)
    # Applying the binary operator '*' (line 1619)
    result_mul_282522 = python_operator(stypy.reporting.localization.Localization(__file__, 1619, 29), '*', win_282520, win_282521)
    
    # Obtaining the member 'sum' of a type (line 1619)
    sum_282523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1619, 29), result_mul_282522, 'sum')
    # Calling sum(args, kwargs) (line 1619)
    sum_call_result_282525 = invoke(stypy.reporting.localization.Localization(__file__, 1619, 29), sum_282523, *[], **kwargs_282524)
    
    # Applying the binary operator '*' (line 1619)
    result_mul_282526 = python_operator(stypy.reporting.localization.Localization(__file__, 1619, 23), '*', fs_282519, sum_call_result_282525)
    
    # Applying the binary operator 'div' (line 1619)
    result_div_282527 = python_operator(stypy.reporting.localization.Localization(__file__, 1619, 16), 'div', float_282518, result_mul_282526)
    
    # Assigning a type to the variable 'scale' (line 1619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1619, 8), 'scale', result_div_282527)
    # SSA branch for the else part of an if statement (line 1618)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'scaling' (line 1620)
    scaling_282528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1620, 9), 'scaling')
    str_282529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1620, 20), 'str', 'spectrum')
    # Applying the binary operator '==' (line 1620)
    result_eq_282530 = python_operator(stypy.reporting.localization.Localization(__file__, 1620, 9), '==', scaling_282528, str_282529)
    
    # Testing the type of an if condition (line 1620)
    if_condition_282531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1620, 9), result_eq_282530)
    # Assigning a type to the variable 'if_condition_282531' (line 1620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1620, 9), 'if_condition_282531', if_condition_282531)
    # SSA begins for if statement (line 1620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1621):
    
    # Assigning a BinOp to a Name (line 1621):
    float_282532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1621, 16), 'float')
    
    # Call to sum(...): (line 1621)
    # Processing the call keyword arguments (line 1621)
    kwargs_282535 = {}
    # Getting the type of 'win' (line 1621)
    win_282533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1621, 22), 'win', False)
    # Obtaining the member 'sum' of a type (line 1621)
    sum_282534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1621, 22), win_282533, 'sum')
    # Calling sum(args, kwargs) (line 1621)
    sum_call_result_282536 = invoke(stypy.reporting.localization.Localization(__file__, 1621, 22), sum_282534, *[], **kwargs_282535)
    
    int_282537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1621, 33), 'int')
    # Applying the binary operator '**' (line 1621)
    result_pow_282538 = python_operator(stypy.reporting.localization.Localization(__file__, 1621, 22), '**', sum_call_result_282536, int_282537)
    
    # Applying the binary operator 'div' (line 1621)
    result_div_282539 = python_operator(stypy.reporting.localization.Localization(__file__, 1621, 16), 'div', float_282532, result_pow_282538)
    
    # Assigning a type to the variable 'scale' (line 1621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1621, 8), 'scale', result_div_282539)
    # SSA branch for the else part of an if statement (line 1620)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 1623)
    # Processing the call arguments (line 1623)
    str_282541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, 25), 'str', 'Unknown scaling: %r')
    # Getting the type of 'scaling' (line 1623)
    scaling_282542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 49), 'scaling', False)
    # Applying the binary operator '%' (line 1623)
    result_mod_282543 = python_operator(stypy.reporting.localization.Localization(__file__, 1623, 25), '%', str_282541, scaling_282542)
    
    # Processing the call keyword arguments (line 1623)
    kwargs_282544 = {}
    # Getting the type of 'ValueError' (line 1623)
    ValueError_282540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1623)
    ValueError_call_result_282545 = invoke(stypy.reporting.localization.Localization(__file__, 1623, 14), ValueError_282540, *[result_mod_282543], **kwargs_282544)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1623, 8), ValueError_call_result_282545, 'raise parameter', BaseException)
    # SSA join for if statement (line 1620)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1618)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 1625)
    mode_282546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1625, 7), 'mode')
    str_282547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1625, 15), 'str', 'stft')
    # Applying the binary operator '==' (line 1625)
    result_eq_282548 = python_operator(stypy.reporting.localization.Localization(__file__, 1625, 7), '==', mode_282546, str_282547)
    
    # Testing the type of an if condition (line 1625)
    if_condition_282549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1625, 4), result_eq_282548)
    # Assigning a type to the variable 'if_condition_282549' (line 1625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1625, 4), 'if_condition_282549', if_condition_282549)
    # SSA begins for if statement (line 1625)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1626):
    
    # Assigning a Call to a Name (line 1626):
    
    # Call to sqrt(...): (line 1626)
    # Processing the call arguments (line 1626)
    # Getting the type of 'scale' (line 1626)
    scale_282552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 24), 'scale', False)
    # Processing the call keyword arguments (line 1626)
    kwargs_282553 = {}
    # Getting the type of 'np' (line 1626)
    np_282550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1626, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1626)
    sqrt_282551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1626, 16), np_282550, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1626)
    sqrt_call_result_282554 = invoke(stypy.reporting.localization.Localization(__file__, 1626, 16), sqrt_282551, *[scale_282552], **kwargs_282553)
    
    # Assigning a type to the variable 'scale' (line 1626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1626, 8), 'scale', sqrt_call_result_282554)
    # SSA join for if statement (line 1625)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_onesided' (line 1628)
    return_onesided_282555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1628, 7), 'return_onesided')
    # Testing the type of an if condition (line 1628)
    if_condition_282556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1628, 4), return_onesided_282555)
    # Assigning a type to the variable 'if_condition_282556' (line 1628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1628, 4), 'if_condition_282556', if_condition_282556)
    # SSA begins for if statement (line 1628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to iscomplexobj(...): (line 1629)
    # Processing the call arguments (line 1629)
    # Getting the type of 'x' (line 1629)
    x_282559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 27), 'x', False)
    # Processing the call keyword arguments (line 1629)
    kwargs_282560 = {}
    # Getting the type of 'np' (line 1629)
    np_282557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1629, 11), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1629)
    iscomplexobj_282558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1629, 11), np_282557, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1629)
    iscomplexobj_call_result_282561 = invoke(stypy.reporting.localization.Localization(__file__, 1629, 11), iscomplexobj_282558, *[x_282559], **kwargs_282560)
    
    # Testing the type of an if condition (line 1629)
    if_condition_282562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1629, 8), iscomplexobj_call_result_282561)
    # Assigning a type to the variable 'if_condition_282562' (line 1629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1629, 8), 'if_condition_282562', if_condition_282562)
    # SSA begins for if statement (line 1629)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1630):
    
    # Assigning a Str to a Name (line 1630):
    str_282563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1630, 20), 'str', 'twosided')
    # Assigning a type to the variable 'sides' (line 1630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1630, 12), 'sides', str_282563)
    
    # Call to warn(...): (line 1631)
    # Processing the call arguments (line 1631)
    str_282566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1631, 26), 'str', 'Input data is complex, switching to return_onesided=False')
    # Processing the call keyword arguments (line 1631)
    kwargs_282567 = {}
    # Getting the type of 'warnings' (line 1631)
    warnings_282564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1631)
    warn_282565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1631, 12), warnings_282564, 'warn')
    # Calling warn(args, kwargs) (line 1631)
    warn_call_result_282568 = invoke(stypy.reporting.localization.Localization(__file__, 1631, 12), warn_282565, *[str_282566], **kwargs_282567)
    
    # SSA branch for the else part of an if statement (line 1629)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 1634):
    
    # Assigning a Str to a Name (line 1634):
    str_282569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1634, 20), 'str', 'onesided')
    # Assigning a type to the variable 'sides' (line 1634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1634, 12), 'sides', str_282569)
    
    
    # Getting the type of 'same_data' (line 1635)
    same_data_282570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1635, 19), 'same_data')
    # Applying the 'not' unary operator (line 1635)
    result_not__282571 = python_operator(stypy.reporting.localization.Localization(__file__, 1635, 15), 'not', same_data_282570)
    
    # Testing the type of an if condition (line 1635)
    if_condition_282572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1635, 12), result_not__282571)
    # Assigning a type to the variable 'if_condition_282572' (line 1635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1635, 12), 'if_condition_282572', if_condition_282572)
    # SSA begins for if statement (line 1635)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to iscomplexobj(...): (line 1636)
    # Processing the call arguments (line 1636)
    # Getting the type of 'y' (line 1636)
    y_282575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 35), 'y', False)
    # Processing the call keyword arguments (line 1636)
    kwargs_282576 = {}
    # Getting the type of 'np' (line 1636)
    np_282573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1636, 19), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1636)
    iscomplexobj_282574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1636, 19), np_282573, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1636)
    iscomplexobj_call_result_282577 = invoke(stypy.reporting.localization.Localization(__file__, 1636, 19), iscomplexobj_282574, *[y_282575], **kwargs_282576)
    
    # Testing the type of an if condition (line 1636)
    if_condition_282578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1636, 16), iscomplexobj_call_result_282577)
    # Assigning a type to the variable 'if_condition_282578' (line 1636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1636, 16), 'if_condition_282578', if_condition_282578)
    # SSA begins for if statement (line 1636)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1637):
    
    # Assigning a Str to a Name (line 1637):
    str_282579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1637, 28), 'str', 'twosided')
    # Assigning a type to the variable 'sides' (line 1637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1637, 20), 'sides', str_282579)
    
    # Call to warn(...): (line 1638)
    # Processing the call arguments (line 1638)
    str_282582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1638, 34), 'str', 'Input data is complex, switching to return_onesided=False')
    # Processing the call keyword arguments (line 1638)
    kwargs_282583 = {}
    # Getting the type of 'warnings' (line 1638)
    warnings_282580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1638, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1638)
    warn_282581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1638, 20), warnings_282580, 'warn')
    # Calling warn(args, kwargs) (line 1638)
    warn_call_result_282584 = invoke(stypy.reporting.localization.Localization(__file__, 1638, 20), warn_282581, *[str_282582], **kwargs_282583)
    
    # SSA join for if statement (line 1636)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1635)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1629)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 1628)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 1641):
    
    # Assigning a Str to a Name (line 1641):
    str_282585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1641, 16), 'str', 'twosided')
    # Assigning a type to the variable 'sides' (line 1641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1641, 8), 'sides', str_282585)
    # SSA join for if statement (line 1628)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'sides' (line 1643)
    sides_282586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1643, 7), 'sides')
    str_282587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1643, 16), 'str', 'twosided')
    # Applying the binary operator '==' (line 1643)
    result_eq_282588 = python_operator(stypy.reporting.localization.Localization(__file__, 1643, 7), '==', sides_282586, str_282587)
    
    # Testing the type of an if condition (line 1643)
    if_condition_282589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1643, 4), result_eq_282588)
    # Assigning a type to the variable 'if_condition_282589' (line 1643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1643, 4), 'if_condition_282589', if_condition_282589)
    # SSA begins for if statement (line 1643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1644):
    
    # Assigning a Call to a Name (line 1644):
    
    # Call to fftfreq(...): (line 1644)
    # Processing the call arguments (line 1644)
    # Getting the type of 'nfft' (line 1644)
    nfft_282592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1644, 32), 'nfft', False)
    int_282593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1644, 38), 'int')
    # Getting the type of 'fs' (line 1644)
    fs_282594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1644, 40), 'fs', False)
    # Applying the binary operator 'div' (line 1644)
    result_div_282595 = python_operator(stypy.reporting.localization.Localization(__file__, 1644, 38), 'div', int_282593, fs_282594)
    
    # Processing the call keyword arguments (line 1644)
    kwargs_282596 = {}
    # Getting the type of 'fftpack' (line 1644)
    fftpack_282590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1644, 16), 'fftpack', False)
    # Obtaining the member 'fftfreq' of a type (line 1644)
    fftfreq_282591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1644, 16), fftpack_282590, 'fftfreq')
    # Calling fftfreq(args, kwargs) (line 1644)
    fftfreq_call_result_282597 = invoke(stypy.reporting.localization.Localization(__file__, 1644, 16), fftfreq_282591, *[nfft_282592, result_div_282595], **kwargs_282596)
    
    # Assigning a type to the variable 'freqs' (line 1644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1644, 8), 'freqs', fftfreq_call_result_282597)
    # SSA branch for the else part of an if statement (line 1643)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'sides' (line 1645)
    sides_282598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1645, 9), 'sides')
    str_282599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1645, 18), 'str', 'onesided')
    # Applying the binary operator '==' (line 1645)
    result_eq_282600 = python_operator(stypy.reporting.localization.Localization(__file__, 1645, 9), '==', sides_282598, str_282599)
    
    # Testing the type of an if condition (line 1645)
    if_condition_282601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1645, 9), result_eq_282600)
    # Assigning a type to the variable 'if_condition_282601' (line 1645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1645, 9), 'if_condition_282601', if_condition_282601)
    # SSA begins for if statement (line 1645)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1646):
    
    # Assigning a Call to a Name (line 1646):
    
    # Call to rfftfreq(...): (line 1646)
    # Processing the call arguments (line 1646)
    # Getting the type of 'nfft' (line 1646)
    nfft_282605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1646, 32), 'nfft', False)
    int_282606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, 38), 'int')
    # Getting the type of 'fs' (line 1646)
    fs_282607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1646, 40), 'fs', False)
    # Applying the binary operator 'div' (line 1646)
    result_div_282608 = python_operator(stypy.reporting.localization.Localization(__file__, 1646, 38), 'div', int_282606, fs_282607)
    
    # Processing the call keyword arguments (line 1646)
    kwargs_282609 = {}
    # Getting the type of 'np' (line 1646)
    np_282602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1646, 16), 'np', False)
    # Obtaining the member 'fft' of a type (line 1646)
    fft_282603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1646, 16), np_282602, 'fft')
    # Obtaining the member 'rfftfreq' of a type (line 1646)
    rfftfreq_282604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1646, 16), fft_282603, 'rfftfreq')
    # Calling rfftfreq(args, kwargs) (line 1646)
    rfftfreq_call_result_282610 = invoke(stypy.reporting.localization.Localization(__file__, 1646, 16), rfftfreq_282604, *[nfft_282605, result_div_282608], **kwargs_282609)
    
    # Assigning a type to the variable 'freqs' (line 1646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1646, 8), 'freqs', rfftfreq_call_result_282610)
    # SSA join for if statement (line 1645)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1643)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1649):
    
    # Assigning a Call to a Name (line 1649):
    
    # Call to _fft_helper(...): (line 1649)
    # Processing the call arguments (line 1649)
    # Getting the type of 'x' (line 1649)
    x_282612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 25), 'x', False)
    # Getting the type of 'win' (line 1649)
    win_282613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 28), 'win', False)
    # Getting the type of 'detrend_func' (line 1649)
    detrend_func_282614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 33), 'detrend_func', False)
    # Getting the type of 'nperseg' (line 1649)
    nperseg_282615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 47), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1649)
    noverlap_282616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 56), 'noverlap', False)
    # Getting the type of 'nfft' (line 1649)
    nfft_282617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 66), 'nfft', False)
    # Getting the type of 'sides' (line 1649)
    sides_282618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 72), 'sides', False)
    # Processing the call keyword arguments (line 1649)
    kwargs_282619 = {}
    # Getting the type of '_fft_helper' (line 1649)
    _fft_helper_282611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1649, 13), '_fft_helper', False)
    # Calling _fft_helper(args, kwargs) (line 1649)
    _fft_helper_call_result_282620 = invoke(stypy.reporting.localization.Localization(__file__, 1649, 13), _fft_helper_282611, *[x_282612, win_282613, detrend_func_282614, nperseg_282615, noverlap_282616, nfft_282617, sides_282618], **kwargs_282619)
    
    # Assigning a type to the variable 'result' (line 1649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1649, 4), 'result', _fft_helper_call_result_282620)
    
    
    # Getting the type of 'same_data' (line 1651)
    same_data_282621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1651, 11), 'same_data')
    # Applying the 'not' unary operator (line 1651)
    result_not__282622 = python_operator(stypy.reporting.localization.Localization(__file__, 1651, 7), 'not', same_data_282621)
    
    # Testing the type of an if condition (line 1651)
    if_condition_282623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1651, 4), result_not__282622)
    # Assigning a type to the variable 'if_condition_282623' (line 1651)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1651, 4), 'if_condition_282623', if_condition_282623)
    # SSA begins for if statement (line 1651)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1653):
    
    # Assigning a Call to a Name (line 1653):
    
    # Call to _fft_helper(...): (line 1653)
    # Processing the call arguments (line 1653)
    # Getting the type of 'y' (line 1653)
    y_282625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 31), 'y', False)
    # Getting the type of 'win' (line 1653)
    win_282626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 34), 'win', False)
    # Getting the type of 'detrend_func' (line 1653)
    detrend_func_282627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 39), 'detrend_func', False)
    # Getting the type of 'nperseg' (line 1653)
    nperseg_282628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 53), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1653)
    noverlap_282629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 62), 'noverlap', False)
    # Getting the type of 'nfft' (line 1653)
    nfft_282630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 72), 'nfft', False)
    # Getting the type of 'sides' (line 1654)
    sides_282631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1654, 31), 'sides', False)
    # Processing the call keyword arguments (line 1653)
    kwargs_282632 = {}
    # Getting the type of '_fft_helper' (line 1653)
    _fft_helper_282624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1653, 19), '_fft_helper', False)
    # Calling _fft_helper(args, kwargs) (line 1653)
    _fft_helper_call_result_282633 = invoke(stypy.reporting.localization.Localization(__file__, 1653, 19), _fft_helper_282624, *[y_282625, win_282626, detrend_func_282627, nperseg_282628, noverlap_282629, nfft_282630, sides_282631], **kwargs_282632)
    
    # Assigning a type to the variable 'result_y' (line 1653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1653, 8), 'result_y', _fft_helper_call_result_282633)
    
    # Assigning a BinOp to a Name (line 1655):
    
    # Assigning a BinOp to a Name (line 1655):
    
    # Call to conjugate(...): (line 1655)
    # Processing the call arguments (line 1655)
    # Getting the type of 'result' (line 1655)
    result_282636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1655, 30), 'result', False)
    # Processing the call keyword arguments (line 1655)
    kwargs_282637 = {}
    # Getting the type of 'np' (line 1655)
    np_282634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1655, 17), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 1655)
    conjugate_282635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1655, 17), np_282634, 'conjugate')
    # Calling conjugate(args, kwargs) (line 1655)
    conjugate_call_result_282638 = invoke(stypy.reporting.localization.Localization(__file__, 1655, 17), conjugate_282635, *[result_282636], **kwargs_282637)
    
    # Getting the type of 'result_y' (line 1655)
    result_y_282639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1655, 40), 'result_y')
    # Applying the binary operator '*' (line 1655)
    result_mul_282640 = python_operator(stypy.reporting.localization.Localization(__file__, 1655, 17), '*', conjugate_call_result_282638, result_y_282639)
    
    # Assigning a type to the variable 'result' (line 1655)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1655, 8), 'result', result_mul_282640)
    # SSA branch for the else part of an if statement (line 1651)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 1656)
    mode_282641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1656, 9), 'mode')
    str_282642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1656, 17), 'str', 'psd')
    # Applying the binary operator '==' (line 1656)
    result_eq_282643 = python_operator(stypy.reporting.localization.Localization(__file__, 1656, 9), '==', mode_282641, str_282642)
    
    # Testing the type of an if condition (line 1656)
    if_condition_282644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1656, 9), result_eq_282643)
    # Assigning a type to the variable 'if_condition_282644' (line 1656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1656, 9), 'if_condition_282644', if_condition_282644)
    # SSA begins for if statement (line 1656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1657):
    
    # Assigning a BinOp to a Name (line 1657):
    
    # Call to conjugate(...): (line 1657)
    # Processing the call arguments (line 1657)
    # Getting the type of 'result' (line 1657)
    result_282647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1657, 30), 'result', False)
    # Processing the call keyword arguments (line 1657)
    kwargs_282648 = {}
    # Getting the type of 'np' (line 1657)
    np_282645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1657, 17), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 1657)
    conjugate_282646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1657, 17), np_282645, 'conjugate')
    # Calling conjugate(args, kwargs) (line 1657)
    conjugate_call_result_282649 = invoke(stypy.reporting.localization.Localization(__file__, 1657, 17), conjugate_282646, *[result_282647], **kwargs_282648)
    
    # Getting the type of 'result' (line 1657)
    result_282650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1657, 40), 'result')
    # Applying the binary operator '*' (line 1657)
    result_mul_282651 = python_operator(stypy.reporting.localization.Localization(__file__, 1657, 17), '*', conjugate_call_result_282649, result_282650)
    
    # Assigning a type to the variable 'result' (line 1657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1657, 8), 'result', result_mul_282651)
    # SSA join for if statement (line 1656)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1651)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'result' (line 1659)
    result_282652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'result')
    # Getting the type of 'scale' (line 1659)
    scale_282653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 14), 'scale')
    # Applying the binary operator '*=' (line 1659)
    result_imul_282654 = python_operator(stypy.reporting.localization.Localization(__file__, 1659, 4), '*=', result_282652, scale_282653)
    # Assigning a type to the variable 'result' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'result', result_imul_282654)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sides' (line 1660)
    sides_282655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 7), 'sides')
    str_282656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 16), 'str', 'onesided')
    # Applying the binary operator '==' (line 1660)
    result_eq_282657 = python_operator(stypy.reporting.localization.Localization(__file__, 1660, 7), '==', sides_282655, str_282656)
    
    
    # Getting the type of 'mode' (line 1660)
    mode_282658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 31), 'mode')
    str_282659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 39), 'str', 'psd')
    # Applying the binary operator '==' (line 1660)
    result_eq_282660 = python_operator(stypy.reporting.localization.Localization(__file__, 1660, 31), '==', mode_282658, str_282659)
    
    # Applying the binary operator 'and' (line 1660)
    result_and_keyword_282661 = python_operator(stypy.reporting.localization.Localization(__file__, 1660, 7), 'and', result_eq_282657, result_eq_282660)
    
    # Testing the type of an if condition (line 1660)
    if_condition_282662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1660, 4), result_and_keyword_282661)
    # Assigning a type to the variable 'if_condition_282662' (line 1660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1660, 4), 'if_condition_282662', if_condition_282662)
    # SSA begins for if statement (line 1660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'nfft' (line 1661)
    nfft_282663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 11), 'nfft')
    int_282664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 18), 'int')
    # Applying the binary operator '%' (line 1661)
    result_mod_282665 = python_operator(stypy.reporting.localization.Localization(__file__, 1661, 11), '%', nfft_282663, int_282664)
    
    # Testing the type of an if condition (line 1661)
    if_condition_282666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1661, 8), result_mod_282665)
    # Assigning a type to the variable 'if_condition_282666' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 8), 'if_condition_282666', if_condition_282666)
    # SSA begins for if statement (line 1661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'result' (line 1662)
    result_282667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 12), 'result')
    
    # Obtaining the type of the subscript
    Ellipsis_282668 = Ellipsis
    int_282669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 24), 'int')
    slice_282670 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1662, 12), int_282669, None, None)
    # Getting the type of 'result' (line 1662)
    result_282671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 12), 'result')
    # Obtaining the member '__getitem__' of a type (line 1662)
    getitem___282672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1662, 12), result_282671, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1662)
    subscript_call_result_282673 = invoke(stypy.reporting.localization.Localization(__file__, 1662, 12), getitem___282672, (Ellipsis_282668, slice_282670))
    
    int_282674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 31), 'int')
    # Applying the binary operator '*=' (line 1662)
    result_imul_282675 = python_operator(stypy.reporting.localization.Localization(__file__, 1662, 12), '*=', subscript_call_result_282673, int_282674)
    # Getting the type of 'result' (line 1662)
    result_282676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 12), 'result')
    Ellipsis_282677 = Ellipsis
    int_282678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 24), 'int')
    slice_282679 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1662, 12), int_282678, None, None)
    # Storing an element on a container (line 1662)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1662, 12), result_282676, ((Ellipsis_282677, slice_282679), result_imul_282675))
    
    # SSA branch for the else part of an if statement (line 1661)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'result' (line 1665)
    result_282680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 12), 'result')
    
    # Obtaining the type of the subscript
    Ellipsis_282681 = Ellipsis
    int_282682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 24), 'int')
    int_282683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 26), 'int')
    slice_282684 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1665, 12), int_282682, int_282683, None)
    # Getting the type of 'result' (line 1665)
    result_282685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 12), 'result')
    # Obtaining the member '__getitem__' of a type (line 1665)
    getitem___282686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1665, 12), result_282685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1665)
    subscript_call_result_282687 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 12), getitem___282686, (Ellipsis_282681, slice_282684))
    
    int_282688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 33), 'int')
    # Applying the binary operator '*=' (line 1665)
    result_imul_282689 = python_operator(stypy.reporting.localization.Localization(__file__, 1665, 12), '*=', subscript_call_result_282687, int_282688)
    # Getting the type of 'result' (line 1665)
    result_282690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 12), 'result')
    Ellipsis_282691 = Ellipsis
    int_282692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 24), 'int')
    int_282693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 26), 'int')
    slice_282694 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1665, 12), int_282692, int_282693, None)
    # Storing an element on a container (line 1665)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1665, 12), result_282690, ((Ellipsis_282691, slice_282694), result_imul_282689))
    
    # SSA join for if statement (line 1661)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1667):
    
    # Assigning a BinOp to a Name (line 1667):
    
    # Call to arange(...): (line 1667)
    # Processing the call arguments (line 1667)
    # Getting the type of 'nperseg' (line 1667)
    nperseg_282697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 21), 'nperseg', False)
    int_282698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1667, 29), 'int')
    # Applying the binary operator 'div' (line 1667)
    result_div_282699 = python_operator(stypy.reporting.localization.Localization(__file__, 1667, 21), 'div', nperseg_282697, int_282698)
    
    
    # Obtaining the type of the subscript
    int_282700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1667, 40), 'int')
    # Getting the type of 'x' (line 1667)
    x_282701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 32), 'x', False)
    # Obtaining the member 'shape' of a type (line 1667)
    shape_282702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1667, 32), x_282701, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1667)
    getitem___282703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1667, 32), shape_282702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1667)
    subscript_call_result_282704 = invoke(stypy.reporting.localization.Localization(__file__, 1667, 32), getitem___282703, int_282700)
    
    # Getting the type of 'nperseg' (line 1667)
    nperseg_282705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 46), 'nperseg', False)
    int_282706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1667, 54), 'int')
    # Applying the binary operator 'div' (line 1667)
    result_div_282707 = python_operator(stypy.reporting.localization.Localization(__file__, 1667, 46), 'div', nperseg_282705, int_282706)
    
    # Applying the binary operator '-' (line 1667)
    result_sub_282708 = python_operator(stypy.reporting.localization.Localization(__file__, 1667, 32), '-', subscript_call_result_282704, result_div_282707)
    
    int_282709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1667, 58), 'int')
    # Applying the binary operator '+' (line 1667)
    result_add_282710 = python_operator(stypy.reporting.localization.Localization(__file__, 1667, 56), '+', result_sub_282708, int_282709)
    
    # Getting the type of 'nperseg' (line 1668)
    nperseg_282711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 21), 'nperseg', False)
    # Getting the type of 'noverlap' (line 1668)
    noverlap_282712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 31), 'noverlap', False)
    # Applying the binary operator '-' (line 1668)
    result_sub_282713 = python_operator(stypy.reporting.localization.Localization(__file__, 1668, 21), '-', nperseg_282711, noverlap_282712)
    
    # Processing the call keyword arguments (line 1667)
    kwargs_282714 = {}
    # Getting the type of 'np' (line 1667)
    np_282695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 1667)
    arange_282696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1667, 11), np_282695, 'arange')
    # Calling arange(args, kwargs) (line 1667)
    arange_call_result_282715 = invoke(stypy.reporting.localization.Localization(__file__, 1667, 11), arange_282696, *[result_div_282699, result_add_282710, result_sub_282713], **kwargs_282714)
    
    
    # Call to float(...): (line 1668)
    # Processing the call arguments (line 1668)
    # Getting the type of 'fs' (line 1668)
    fs_282717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 47), 'fs', False)
    # Processing the call keyword arguments (line 1668)
    kwargs_282718 = {}
    # Getting the type of 'float' (line 1668)
    float_282716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 41), 'float', False)
    # Calling float(args, kwargs) (line 1668)
    float_call_result_282719 = invoke(stypy.reporting.localization.Localization(__file__, 1668, 41), float_282716, *[fs_282717], **kwargs_282718)
    
    # Applying the binary operator 'div' (line 1667)
    result_div_282720 = python_operator(stypy.reporting.localization.Localization(__file__, 1667, 11), 'div', arange_call_result_282715, float_call_result_282719)
    
    # Assigning a type to the variable 'time' (line 1667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1667, 4), 'time', result_div_282720)
    
    # Type idiom detected: calculating its left and rigth part (line 1669)
    # Getting the type of 'boundary' (line 1669)
    boundary_282721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 4), 'boundary')
    # Getting the type of 'None' (line 1669)
    None_282722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 23), 'None')
    
    (may_be_282723, more_types_in_union_282724) = may_not_be_none(boundary_282721, None_282722)

    if may_be_282723:

        if more_types_in_union_282724:
            # Runtime conditional SSA (line 1669)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'time' (line 1670)
        time_282725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1670, 8), 'time')
        # Getting the type of 'nperseg' (line 1670)
        nperseg_282726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1670, 17), 'nperseg')
        int_282727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1670, 25), 'int')
        # Applying the binary operator 'div' (line 1670)
        result_div_282728 = python_operator(stypy.reporting.localization.Localization(__file__, 1670, 17), 'div', nperseg_282726, int_282727)
        
        # Getting the type of 'fs' (line 1670)
        fs_282729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1670, 30), 'fs')
        # Applying the binary operator 'div' (line 1670)
        result_div_282730 = python_operator(stypy.reporting.localization.Localization(__file__, 1670, 16), 'div', result_div_282728, fs_282729)
        
        # Applying the binary operator '-=' (line 1670)
        result_isub_282731 = python_operator(stypy.reporting.localization.Localization(__file__, 1670, 8), '-=', time_282725, result_div_282730)
        # Assigning a type to the variable 'time' (line 1670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1670, 8), 'time', result_isub_282731)
        

        if more_types_in_union_282724:
            # SSA join for if statement (line 1669)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1672):
    
    # Assigning a Call to a Name (line 1672):
    
    # Call to astype(...): (line 1672)
    # Processing the call arguments (line 1672)
    # Getting the type of 'outdtype' (line 1672)
    outdtype_282734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1672, 27), 'outdtype', False)
    # Processing the call keyword arguments (line 1672)
    kwargs_282735 = {}
    # Getting the type of 'result' (line 1672)
    result_282732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1672, 13), 'result', False)
    # Obtaining the member 'astype' of a type (line 1672)
    astype_282733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1672, 13), result_282732, 'astype')
    # Calling astype(args, kwargs) (line 1672)
    astype_call_result_282736 = invoke(stypy.reporting.localization.Localization(__file__, 1672, 13), astype_282733, *[outdtype_282734], **kwargs_282735)
    
    # Assigning a type to the variable 'result' (line 1672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1672, 4), 'result', astype_call_result_282736)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'same_data' (line 1675)
    same_data_282737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1675, 7), 'same_data')
    
    # Getting the type of 'mode' (line 1675)
    mode_282738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1675, 21), 'mode')
    str_282739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1675, 29), 'str', 'stft')
    # Applying the binary operator '!=' (line 1675)
    result_ne_282740 = python_operator(stypy.reporting.localization.Localization(__file__, 1675, 21), '!=', mode_282738, str_282739)
    
    # Applying the binary operator 'and' (line 1675)
    result_and_keyword_282741 = python_operator(stypy.reporting.localization.Localization(__file__, 1675, 7), 'and', same_data_282737, result_ne_282740)
    
    # Testing the type of an if condition (line 1675)
    if_condition_282742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1675, 4), result_and_keyword_282741)
    # Assigning a type to the variable 'if_condition_282742' (line 1675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1675, 4), 'if_condition_282742', if_condition_282742)
    # SSA begins for if statement (line 1675)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1676):
    
    # Assigning a Attribute to a Name (line 1676):
    # Getting the type of 'result' (line 1676)
    result_282743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1676, 17), 'result')
    # Obtaining the member 'real' of a type (line 1676)
    real_282744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1676, 17), result_282743, 'real')
    # Assigning a type to the variable 'result' (line 1676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1676, 8), 'result', real_282744)
    # SSA join for if statement (line 1675)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'axis' (line 1680)
    axis_282745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1680, 7), 'axis')
    int_282746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1680, 14), 'int')
    # Applying the binary operator '<' (line 1680)
    result_lt_282747 = python_operator(stypy.reporting.localization.Localization(__file__, 1680, 7), '<', axis_282745, int_282746)
    
    # Testing the type of an if condition (line 1680)
    if_condition_282748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1680, 4), result_lt_282747)
    # Assigning a type to the variable 'if_condition_282748' (line 1680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1680, 4), 'if_condition_282748', if_condition_282748)
    # SSA begins for if statement (line 1680)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axis' (line 1681)
    axis_282749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1681, 8), 'axis')
    int_282750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1681, 16), 'int')
    # Applying the binary operator '-=' (line 1681)
    result_isub_282751 = python_operator(stypy.reporting.localization.Localization(__file__, 1681, 8), '-=', axis_282749, int_282750)
    # Assigning a type to the variable 'axis' (line 1681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1681, 8), 'axis', result_isub_282751)
    
    # SSA join for if statement (line 1680)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1684):
    
    # Assigning a Call to a Name (line 1684):
    
    # Call to rollaxis(...): (line 1684)
    # Processing the call arguments (line 1684)
    # Getting the type of 'result' (line 1684)
    result_282754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 25), 'result', False)
    int_282755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1684, 33), 'int')
    # Getting the type of 'axis' (line 1684)
    axis_282756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 37), 'axis', False)
    # Processing the call keyword arguments (line 1684)
    kwargs_282757 = {}
    # Getting the type of 'np' (line 1684)
    np_282752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1684, 13), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1684)
    rollaxis_282753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1684, 13), np_282752, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1684)
    rollaxis_call_result_282758 = invoke(stypy.reporting.localization.Localization(__file__, 1684, 13), rollaxis_282753, *[result_282754, int_282755, axis_282756], **kwargs_282757)
    
    # Assigning a type to the variable 'result' (line 1684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1684, 4), 'result', rollaxis_call_result_282758)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1686)
    tuple_282759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1686, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1686)
    # Adding element type (line 1686)
    # Getting the type of 'freqs' (line 1686)
    freqs_282760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 11), 'freqs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1686, 11), tuple_282759, freqs_282760)
    # Adding element type (line 1686)
    # Getting the type of 'time' (line 1686)
    time_282761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 18), 'time')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1686, 11), tuple_282759, time_282761)
    # Adding element type (line 1686)
    # Getting the type of 'result' (line 1686)
    result_282762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 24), 'result')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1686, 11), tuple_282759, result_282762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1686, 4), 'stypy_return_type', tuple_282759)
    
    # ################# End of '_spectral_helper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_spectral_helper' in the type store
    # Getting the type of 'stypy_return_type' (line 1388)
    stypy_return_type_282763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_282763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_spectral_helper'
    return stypy_return_type_282763

# Assigning a type to the variable '_spectral_helper' (line 1388)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 0), '_spectral_helper', _spectral_helper)

@norecursion
def _fft_helper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fft_helper'
    module_type_store = module_type_store.open_function_context('_fft_helper', 1689, 0, False)
    
    # Passed parameters checking function
    _fft_helper.stypy_localization = localization
    _fft_helper.stypy_type_of_self = None
    _fft_helper.stypy_type_store = module_type_store
    _fft_helper.stypy_function_name = '_fft_helper'
    _fft_helper.stypy_param_names_list = ['x', 'win', 'detrend_func', 'nperseg', 'noverlap', 'nfft', 'sides']
    _fft_helper.stypy_varargs_param_name = None
    _fft_helper.stypy_kwargs_param_name = None
    _fft_helper.stypy_call_defaults = defaults
    _fft_helper.stypy_call_varargs = varargs
    _fft_helper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fft_helper', ['x', 'win', 'detrend_func', 'nperseg', 'noverlap', 'nfft', 'sides'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fft_helper', localization, ['x', 'win', 'detrend_func', 'nperseg', 'noverlap', 'nfft', 'sides'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fft_helper(...)' code ##################

    str_282764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1715, (-1)), 'str', '\n    Calculate windowed FFT, for internal use by\n    scipy.signal._spectral_helper\n\n    This is a helper function that does the main FFT calculation for\n    `_spectral helper`. All input valdiation is performed there, and the\n    data axis is assumed to be the last axis of x. It is not designed to\n    be called externally. The windows are not averaged over; the result\n    from each window is returned.\n\n    Returns\n    -------\n    result : ndarray\n        Array of FFT data\n\n    References\n    ----------\n    .. [1] Stack Overflow, "Repeat NumPy array without replicating\n           data?", http://stackoverflow.com/a/5568169\n\n    Notes\n    -----\n    Adapted from matplotlib.mlab\n\n    .. versionadded:: 0.16.0\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'nperseg' (line 1717)
    nperseg_282765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1717, 7), 'nperseg')
    int_282766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1717, 18), 'int')
    # Applying the binary operator '==' (line 1717)
    result_eq_282767 = python_operator(stypy.reporting.localization.Localization(__file__, 1717, 7), '==', nperseg_282765, int_282766)
    
    
    # Getting the type of 'noverlap' (line 1717)
    noverlap_282768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1717, 24), 'noverlap')
    int_282769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1717, 36), 'int')
    # Applying the binary operator '==' (line 1717)
    result_eq_282770 = python_operator(stypy.reporting.localization.Localization(__file__, 1717, 24), '==', noverlap_282768, int_282769)
    
    # Applying the binary operator 'and' (line 1717)
    result_and_keyword_282771 = python_operator(stypy.reporting.localization.Localization(__file__, 1717, 7), 'and', result_eq_282767, result_eq_282770)
    
    # Testing the type of an if condition (line 1717)
    if_condition_282772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1717, 4), result_and_keyword_282771)
    # Assigning a type to the variable 'if_condition_282772' (line 1717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1717, 4), 'if_condition_282772', if_condition_282772)
    # SSA begins for if statement (line 1717)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1718):
    
    # Assigning a Subscript to a Name (line 1718):
    
    # Obtaining the type of the subscript
    Ellipsis_282773 = Ellipsis
    # Getting the type of 'np' (line 1718)
    np_282774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 24), 'np')
    # Obtaining the member 'newaxis' of a type (line 1718)
    newaxis_282775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1718, 24), np_282774, 'newaxis')
    # Getting the type of 'x' (line 1718)
    x_282776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1718, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 1718)
    getitem___282777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1718, 17), x_282776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1718)
    subscript_call_result_282778 = invoke(stypy.reporting.localization.Localization(__file__, 1718, 17), getitem___282777, (Ellipsis_282773, newaxis_282775))
    
    # Assigning a type to the variable 'result' (line 1718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1718, 8), 'result', subscript_call_result_282778)
    # SSA branch for the else part of an if statement (line 1717)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 1720):
    
    # Assigning a BinOp to a Name (line 1720):
    # Getting the type of 'nperseg' (line 1720)
    nperseg_282779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1720, 15), 'nperseg')
    # Getting the type of 'noverlap' (line 1720)
    noverlap_282780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1720, 25), 'noverlap')
    # Applying the binary operator '-' (line 1720)
    result_sub_282781 = python_operator(stypy.reporting.localization.Localization(__file__, 1720, 15), '-', nperseg_282779, noverlap_282780)
    
    # Assigning a type to the variable 'step' (line 1720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1720, 8), 'step', result_sub_282781)
    
    # Assigning a BinOp to a Name (line 1721):
    
    # Assigning a BinOp to a Name (line 1721):
    
    # Obtaining the type of the subscript
    int_282782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 25), 'int')
    slice_282783 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1721, 16), None, int_282782, None)
    # Getting the type of 'x' (line 1721)
    x_282784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 16), 'x')
    # Obtaining the member 'shape' of a type (line 1721)
    shape_282785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 16), x_282784, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1721)
    getitem___282786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 16), shape_282785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1721)
    subscript_call_result_282787 = invoke(stypy.reporting.localization.Localization(__file__, 1721, 16), getitem___282786, slice_282783)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1721)
    tuple_282788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1721)
    # Adding element type (line 1721)
    
    # Obtaining the type of the subscript
    int_282789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1721, 39), 'int')
    # Getting the type of 'x' (line 1721)
    x_282790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 31), 'x')
    # Obtaining the member 'shape' of a type (line 1721)
    shape_282791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 31), x_282790, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1721)
    getitem___282792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1721, 31), shape_282791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1721)
    subscript_call_result_282793 = invoke(stypy.reporting.localization.Localization(__file__, 1721, 31), getitem___282792, int_282789)
    
    # Getting the type of 'noverlap' (line 1721)
    noverlap_282794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 43), 'noverlap')
    # Applying the binary operator '-' (line 1721)
    result_sub_282795 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 31), '-', subscript_call_result_282793, noverlap_282794)
    
    # Getting the type of 'step' (line 1721)
    step_282796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 54), 'step')
    # Applying the binary operator '//' (line 1721)
    result_floordiv_282797 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 30), '//', result_sub_282795, step_282796)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1721, 30), tuple_282788, result_floordiv_282797)
    # Adding element type (line 1721)
    # Getting the type of 'nperseg' (line 1721)
    nperseg_282798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1721, 60), 'nperseg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1721, 30), tuple_282788, nperseg_282798)
    
    # Applying the binary operator '+' (line 1721)
    result_add_282799 = python_operator(stypy.reporting.localization.Localization(__file__, 1721, 16), '+', subscript_call_result_282787, tuple_282788)
    
    # Assigning a type to the variable 'shape' (line 1721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1721, 8), 'shape', result_add_282799)
    
    # Assigning a BinOp to a Name (line 1722):
    
    # Assigning a BinOp to a Name (line 1722):
    
    # Obtaining the type of the subscript
    int_282800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1722, 29), 'int')
    slice_282801 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1722, 18), None, int_282800, None)
    # Getting the type of 'x' (line 1722)
    x_282802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1722, 18), 'x')
    # Obtaining the member 'strides' of a type (line 1722)
    strides_282803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 18), x_282802, 'strides')
    # Obtaining the member '__getitem__' of a type (line 1722)
    getitem___282804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 18), strides_282803, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1722)
    subscript_call_result_282805 = invoke(stypy.reporting.localization.Localization(__file__, 1722, 18), getitem___282804, slice_282801)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1722)
    tuple_282806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1722, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1722)
    # Adding element type (line 1722)
    # Getting the type of 'step' (line 1722)
    step_282807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1722, 34), 'step')
    
    # Obtaining the type of the subscript
    int_282808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1722, 49), 'int')
    # Getting the type of 'x' (line 1722)
    x_282809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1722, 39), 'x')
    # Obtaining the member 'strides' of a type (line 1722)
    strides_282810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 39), x_282809, 'strides')
    # Obtaining the member '__getitem__' of a type (line 1722)
    getitem___282811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 39), strides_282810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1722)
    subscript_call_result_282812 = invoke(stypy.reporting.localization.Localization(__file__, 1722, 39), getitem___282811, int_282808)
    
    # Applying the binary operator '*' (line 1722)
    result_mul_282813 = python_operator(stypy.reporting.localization.Localization(__file__, 1722, 34), '*', step_282807, subscript_call_result_282812)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1722, 34), tuple_282806, result_mul_282813)
    # Adding element type (line 1722)
    
    # Obtaining the type of the subscript
    int_282814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1722, 64), 'int')
    # Getting the type of 'x' (line 1722)
    x_282815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1722, 54), 'x')
    # Obtaining the member 'strides' of a type (line 1722)
    strides_282816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 54), x_282815, 'strides')
    # Obtaining the member '__getitem__' of a type (line 1722)
    getitem___282817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1722, 54), strides_282816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1722)
    subscript_call_result_282818 = invoke(stypy.reporting.localization.Localization(__file__, 1722, 54), getitem___282817, int_282814)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1722, 34), tuple_282806, subscript_call_result_282818)
    
    # Applying the binary operator '+' (line 1722)
    result_add_282819 = python_operator(stypy.reporting.localization.Localization(__file__, 1722, 18), '+', subscript_call_result_282805, tuple_282806)
    
    # Assigning a type to the variable 'strides' (line 1722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1722, 8), 'strides', result_add_282819)
    
    # Assigning a Call to a Name (line 1723):
    
    # Assigning a Call to a Name (line 1723):
    
    # Call to as_strided(...): (line 1723)
    # Processing the call arguments (line 1723)
    # Getting the type of 'x' (line 1723)
    x_282824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1723, 49), 'x', False)
    # Processing the call keyword arguments (line 1723)
    # Getting the type of 'shape' (line 1723)
    shape_282825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1723, 58), 'shape', False)
    keyword_282826 = shape_282825
    # Getting the type of 'strides' (line 1724)
    strides_282827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1724, 57), 'strides', False)
    keyword_282828 = strides_282827
    kwargs_282829 = {'strides': keyword_282828, 'shape': keyword_282826}
    # Getting the type of 'np' (line 1723)
    np_282820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1723, 17), 'np', False)
    # Obtaining the member 'lib' of a type (line 1723)
    lib_282821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1723, 17), np_282820, 'lib')
    # Obtaining the member 'stride_tricks' of a type (line 1723)
    stride_tricks_282822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1723, 17), lib_282821, 'stride_tricks')
    # Obtaining the member 'as_strided' of a type (line 1723)
    as_strided_282823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1723, 17), stride_tricks_282822, 'as_strided')
    # Calling as_strided(args, kwargs) (line 1723)
    as_strided_call_result_282830 = invoke(stypy.reporting.localization.Localization(__file__, 1723, 17), as_strided_282823, *[x_282824], **kwargs_282829)
    
    # Assigning a type to the variable 'result' (line 1723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1723, 8), 'result', as_strided_call_result_282830)
    # SSA join for if statement (line 1717)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1727):
    
    # Assigning a Call to a Name (line 1727):
    
    # Call to detrend_func(...): (line 1727)
    # Processing the call arguments (line 1727)
    # Getting the type of 'result' (line 1727)
    result_282832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 26), 'result', False)
    # Processing the call keyword arguments (line 1727)
    kwargs_282833 = {}
    # Getting the type of 'detrend_func' (line 1727)
    detrend_func_282831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 13), 'detrend_func', False)
    # Calling detrend_func(args, kwargs) (line 1727)
    detrend_func_call_result_282834 = invoke(stypy.reporting.localization.Localization(__file__, 1727, 13), detrend_func_282831, *[result_282832], **kwargs_282833)
    
    # Assigning a type to the variable 'result' (line 1727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1727, 4), 'result', detrend_func_call_result_282834)
    
    # Assigning a BinOp to a Name (line 1730):
    
    # Assigning a BinOp to a Name (line 1730):
    # Getting the type of 'win' (line 1730)
    win_282835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1730, 13), 'win')
    # Getting the type of 'result' (line 1730)
    result_282836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1730, 19), 'result')
    # Applying the binary operator '*' (line 1730)
    result_mul_282837 = python_operator(stypy.reporting.localization.Localization(__file__, 1730, 13), '*', win_282835, result_282836)
    
    # Assigning a type to the variable 'result' (line 1730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1730, 4), 'result', result_mul_282837)
    
    
    # Getting the type of 'sides' (line 1733)
    sides_282838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1733, 7), 'sides')
    str_282839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1733, 16), 'str', 'twosided')
    # Applying the binary operator '==' (line 1733)
    result_eq_282840 = python_operator(stypy.reporting.localization.Localization(__file__, 1733, 7), '==', sides_282838, str_282839)
    
    # Testing the type of an if condition (line 1733)
    if_condition_282841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1733, 4), result_eq_282840)
    # Assigning a type to the variable 'if_condition_282841' (line 1733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1733, 4), 'if_condition_282841', if_condition_282841)
    # SSA begins for if statement (line 1733)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 1734):
    
    # Assigning a Attribute to a Name (line 1734):
    # Getting the type of 'fftpack' (line 1734)
    fftpack_282842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1734, 15), 'fftpack')
    # Obtaining the member 'fft' of a type (line 1734)
    fft_282843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1734, 15), fftpack_282842, 'fft')
    # Assigning a type to the variable 'func' (line 1734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1734, 8), 'func', fft_282843)
    # SSA branch for the else part of an if statement (line 1733)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 1736):
    
    # Assigning a Attribute to a Name (line 1736):
    # Getting the type of 'result' (line 1736)
    result_282844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1736, 17), 'result')
    # Obtaining the member 'real' of a type (line 1736)
    real_282845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1736, 17), result_282844, 'real')
    # Assigning a type to the variable 'result' (line 1736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1736, 8), 'result', real_282845)
    
    # Assigning a Attribute to a Name (line 1737):
    
    # Assigning a Attribute to a Name (line 1737):
    # Getting the type of 'np' (line 1737)
    np_282846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1737, 15), 'np')
    # Obtaining the member 'fft' of a type (line 1737)
    fft_282847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1737, 15), np_282846, 'fft')
    # Obtaining the member 'rfft' of a type (line 1737)
    rfft_282848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1737, 15), fft_282847, 'rfft')
    # Assigning a type to the variable 'func' (line 1737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1737, 8), 'func', rfft_282848)
    # SSA join for if statement (line 1733)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1738):
    
    # Assigning a Call to a Name (line 1738):
    
    # Call to func(...): (line 1738)
    # Processing the call arguments (line 1738)
    # Getting the type of 'result' (line 1738)
    result_282850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 18), 'result', False)
    # Processing the call keyword arguments (line 1738)
    # Getting the type of 'nfft' (line 1738)
    nfft_282851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 28), 'nfft', False)
    keyword_282852 = nfft_282851
    kwargs_282853 = {'n': keyword_282852}
    # Getting the type of 'func' (line 1738)
    func_282849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1738, 13), 'func', False)
    # Calling func(args, kwargs) (line 1738)
    func_call_result_282854 = invoke(stypy.reporting.localization.Localization(__file__, 1738, 13), func_282849, *[result_282850], **kwargs_282853)
    
    # Assigning a type to the variable 'result' (line 1738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1738, 4), 'result', func_call_result_282854)
    # Getting the type of 'result' (line 1740)
    result_282855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1740, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 1740)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1740, 4), 'stypy_return_type', result_282855)
    
    # ################# End of '_fft_helper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fft_helper' in the type store
    # Getting the type of 'stypy_return_type' (line 1689)
    stypy_return_type_282856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1689, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_282856)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fft_helper'
    return stypy_return_type_282856

# Assigning a type to the variable '_fft_helper' (line 1689)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1689, 0), '_fft_helper', _fft_helper)

@norecursion
def _triage_segments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_triage_segments'
    module_type_store = module_type_store.open_function_context('_triage_segments', 1742, 0, False)
    
    # Passed parameters checking function
    _triage_segments.stypy_localization = localization
    _triage_segments.stypy_type_of_self = None
    _triage_segments.stypy_type_store = module_type_store
    _triage_segments.stypy_function_name = '_triage_segments'
    _triage_segments.stypy_param_names_list = ['window', 'nperseg', 'input_length']
    _triage_segments.stypy_varargs_param_name = None
    _triage_segments.stypy_kwargs_param_name = None
    _triage_segments.stypy_call_defaults = defaults
    _triage_segments.stypy_call_varargs = varargs
    _triage_segments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_triage_segments', ['window', 'nperseg', 'input_length'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_triage_segments', localization, ['window', 'nperseg', 'input_length'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_triage_segments(...)' code ##################

    str_282857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1775, (-1)), 'str', '\n    Parses window and nperseg arguments for spectrogram and _spectral_helper.\n    This is a helper function, not meant to be called externally.\n\n    Parameters\n    ---------\n    window : string, tuple, or ndarray\n        If window is specified by a string or tuple and nperseg is not\n        specified, nperseg is set to the default of 256 and returns a window of\n        that length.\n        If instead the window is array_like and nperseg is not specified, then\n        nperseg is set to the length of the window. A ValueError is raised if\n        the user supplies both an array_like window and a value for nperseg but\n        nperseg does not equal the length of the window.\n\n    nperseg : int\n        Length of each segment\n\n    input_length: int\n        Length of input signal, i.e. x.shape[-1]. Used to test for errors.\n\n    Returns\n    -------\n    win : ndarray\n        window. If function was called with string or tuple than this will hold\n        the actual array used as a window.\n\n    nperseg : int\n        Length of each segment. If window is str or tuple, nperseg is set to\n        256. If window is array_like, nperseg is set to the length of the\n        6\n        window.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 1778)
    # Processing the call arguments (line 1778)
    # Getting the type of 'window' (line 1778)
    window_282859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 18), 'window', False)
    # Getting the type of 'string_types' (line 1778)
    string_types_282860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 26), 'string_types', False)
    # Processing the call keyword arguments (line 1778)
    kwargs_282861 = {}
    # Getting the type of 'isinstance' (line 1778)
    isinstance_282858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1778)
    isinstance_call_result_282862 = invoke(stypy.reporting.localization.Localization(__file__, 1778, 7), isinstance_282858, *[window_282859, string_types_282860], **kwargs_282861)
    
    
    # Call to isinstance(...): (line 1778)
    # Processing the call arguments (line 1778)
    # Getting the type of 'window' (line 1778)
    window_282864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 54), 'window', False)
    # Getting the type of 'tuple' (line 1778)
    tuple_282865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 62), 'tuple', False)
    # Processing the call keyword arguments (line 1778)
    kwargs_282866 = {}
    # Getting the type of 'isinstance' (line 1778)
    isinstance_282863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 43), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1778)
    isinstance_call_result_282867 = invoke(stypy.reporting.localization.Localization(__file__, 1778, 43), isinstance_282863, *[window_282864, tuple_282865], **kwargs_282866)
    
    # Applying the binary operator 'or' (line 1778)
    result_or_keyword_282868 = python_operator(stypy.reporting.localization.Localization(__file__, 1778, 7), 'or', isinstance_call_result_282862, isinstance_call_result_282867)
    
    # Testing the type of an if condition (line 1778)
    if_condition_282869 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1778, 4), result_or_keyword_282868)
    # Assigning a type to the variable 'if_condition_282869' (line 1778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1778, 4), 'if_condition_282869', if_condition_282869)
    # SSA begins for if statement (line 1778)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 1780)
    # Getting the type of 'nperseg' (line 1780)
    nperseg_282870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1780, 11), 'nperseg')
    # Getting the type of 'None' (line 1780)
    None_282871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1780, 22), 'None')
    
    (may_be_282872, more_types_in_union_282873) = may_be_none(nperseg_282870, None_282871)

    if may_be_282872:

        if more_types_in_union_282873:
            # Runtime conditional SSA (line 1780)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Num to a Name (line 1781):
        
        # Assigning a Num to a Name (line 1781):
        int_282874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1781, 22), 'int')
        # Assigning a type to the variable 'nperseg' (line 1781)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1781, 12), 'nperseg', int_282874)

        if more_types_in_union_282873:
            # SSA join for if statement (line 1780)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'nperseg' (line 1782)
    nperseg_282875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1782, 11), 'nperseg')
    # Getting the type of 'input_length' (line 1782)
    input_length_282876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1782, 21), 'input_length')
    # Applying the binary operator '>' (line 1782)
    result_gt_282877 = python_operator(stypy.reporting.localization.Localization(__file__, 1782, 11), '>', nperseg_282875, input_length_282876)
    
    # Testing the type of an if condition (line 1782)
    if_condition_282878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1782, 8), result_gt_282877)
    # Assigning a type to the variable 'if_condition_282878' (line 1782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1782, 8), 'if_condition_282878', if_condition_282878)
    # SSA begins for if statement (line 1782)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1783)
    # Processing the call arguments (line 1783)
    
    # Call to format(...): (line 1783)
    # Processing the call arguments (line 1783)
    # Getting the type of 'nperseg' (line 1785)
    nperseg_282883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1785, 38), 'nperseg', False)
    # Getting the type of 'input_length' (line 1785)
    input_length_282884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1785, 47), 'input_length', False)
    # Processing the call keyword arguments (line 1783)
    kwargs_282885 = {}
    str_282881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1783, 26), 'str', 'nperseg = {0:d} is greater than input length  = {1:d}, using nperseg = {1:d}')
    # Obtaining the member 'format' of a type (line 1783)
    format_282882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1783, 26), str_282881, 'format')
    # Calling format(args, kwargs) (line 1783)
    format_call_result_282886 = invoke(stypy.reporting.localization.Localization(__file__, 1783, 26), format_282882, *[nperseg_282883, input_length_282884], **kwargs_282885)
    
    # Processing the call keyword arguments (line 1783)
    kwargs_282887 = {}
    # Getting the type of 'warnings' (line 1783)
    warnings_282879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1783)
    warn_282880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1783, 12), warnings_282879, 'warn')
    # Calling warn(args, kwargs) (line 1783)
    warn_call_result_282888 = invoke(stypy.reporting.localization.Localization(__file__, 1783, 12), warn_282880, *[format_call_result_282886], **kwargs_282887)
    
    
    # Assigning a Name to a Name (line 1786):
    
    # Assigning a Name to a Name (line 1786):
    # Getting the type of 'input_length' (line 1786)
    input_length_282889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1786, 22), 'input_length')
    # Assigning a type to the variable 'nperseg' (line 1786)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1786, 12), 'nperseg', input_length_282889)
    # SSA join for if statement (line 1782)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1787):
    
    # Assigning a Call to a Name (line 1787):
    
    # Call to get_window(...): (line 1787)
    # Processing the call arguments (line 1787)
    # Getting the type of 'window' (line 1787)
    window_282891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1787, 25), 'window', False)
    # Getting the type of 'nperseg' (line 1787)
    nperseg_282892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1787, 33), 'nperseg', False)
    # Processing the call keyword arguments (line 1787)
    kwargs_282893 = {}
    # Getting the type of 'get_window' (line 1787)
    get_window_282890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1787, 14), 'get_window', False)
    # Calling get_window(args, kwargs) (line 1787)
    get_window_call_result_282894 = invoke(stypy.reporting.localization.Localization(__file__, 1787, 14), get_window_282890, *[window_282891, nperseg_282892], **kwargs_282893)
    
    # Assigning a type to the variable 'win' (line 1787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1787, 8), 'win', get_window_call_result_282894)
    # SSA branch for the else part of an if statement (line 1778)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1789):
    
    # Assigning a Call to a Name (line 1789):
    
    # Call to asarray(...): (line 1789)
    # Processing the call arguments (line 1789)
    # Getting the type of 'window' (line 1789)
    window_282897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1789, 25), 'window', False)
    # Processing the call keyword arguments (line 1789)
    kwargs_282898 = {}
    # Getting the type of 'np' (line 1789)
    np_282895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1789, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1789)
    asarray_282896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1789, 14), np_282895, 'asarray')
    # Calling asarray(args, kwargs) (line 1789)
    asarray_call_result_282899 = invoke(stypy.reporting.localization.Localization(__file__, 1789, 14), asarray_282896, *[window_282897], **kwargs_282898)
    
    # Assigning a type to the variable 'win' (line 1789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1789, 8), 'win', asarray_call_result_282899)
    
    
    
    # Call to len(...): (line 1790)
    # Processing the call arguments (line 1790)
    # Getting the type of 'win' (line 1790)
    win_282901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1790, 15), 'win', False)
    # Obtaining the member 'shape' of a type (line 1790)
    shape_282902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1790, 15), win_282901, 'shape')
    # Processing the call keyword arguments (line 1790)
    kwargs_282903 = {}
    # Getting the type of 'len' (line 1790)
    len_282900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1790, 11), 'len', False)
    # Calling len(args, kwargs) (line 1790)
    len_call_result_282904 = invoke(stypy.reporting.localization.Localization(__file__, 1790, 11), len_282900, *[shape_282902], **kwargs_282903)
    
    int_282905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1790, 29), 'int')
    # Applying the binary operator '!=' (line 1790)
    result_ne_282906 = python_operator(stypy.reporting.localization.Localization(__file__, 1790, 11), '!=', len_call_result_282904, int_282905)
    
    # Testing the type of an if condition (line 1790)
    if_condition_282907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1790, 8), result_ne_282906)
    # Assigning a type to the variable 'if_condition_282907' (line 1790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1790, 8), 'if_condition_282907', if_condition_282907)
    # SSA begins for if statement (line 1790)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1791)
    # Processing the call arguments (line 1791)
    str_282909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1791, 29), 'str', 'window must be 1-D')
    # Processing the call keyword arguments (line 1791)
    kwargs_282910 = {}
    # Getting the type of 'ValueError' (line 1791)
    ValueError_282908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1791, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1791)
    ValueError_call_result_282911 = invoke(stypy.reporting.localization.Localization(__file__, 1791, 18), ValueError_282908, *[str_282909], **kwargs_282910)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1791, 12), ValueError_call_result_282911, 'raise parameter', BaseException)
    # SSA join for if statement (line 1790)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'input_length' (line 1792)
    input_length_282912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1792, 11), 'input_length')
    
    # Obtaining the type of the subscript
    int_282913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1792, 36), 'int')
    # Getting the type of 'win' (line 1792)
    win_282914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1792, 26), 'win')
    # Obtaining the member 'shape' of a type (line 1792)
    shape_282915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1792, 26), win_282914, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1792)
    getitem___282916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1792, 26), shape_282915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1792)
    subscript_call_result_282917 = invoke(stypy.reporting.localization.Localization(__file__, 1792, 26), getitem___282916, int_282913)
    
    # Applying the binary operator '<' (line 1792)
    result_lt_282918 = python_operator(stypy.reporting.localization.Localization(__file__, 1792, 11), '<', input_length_282912, subscript_call_result_282917)
    
    # Testing the type of an if condition (line 1792)
    if_condition_282919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1792, 8), result_lt_282918)
    # Assigning a type to the variable 'if_condition_282919' (line 1792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1792, 8), 'if_condition_282919', if_condition_282919)
    # SSA begins for if statement (line 1792)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1793)
    # Processing the call arguments (line 1793)
    str_282921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1793, 29), 'str', 'window is longer than input signal')
    # Processing the call keyword arguments (line 1793)
    kwargs_282922 = {}
    # Getting the type of 'ValueError' (line 1793)
    ValueError_282920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1793, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1793)
    ValueError_call_result_282923 = invoke(stypy.reporting.localization.Localization(__file__, 1793, 18), ValueError_282920, *[str_282921], **kwargs_282922)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1793, 12), ValueError_call_result_282923, 'raise parameter', BaseException)
    # SSA join for if statement (line 1792)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1794)
    # Getting the type of 'nperseg' (line 1794)
    nperseg_282924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1794, 11), 'nperseg')
    # Getting the type of 'None' (line 1794)
    None_282925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1794, 22), 'None')
    
    (may_be_282926, more_types_in_union_282927) = may_be_none(nperseg_282924, None_282925)

    if may_be_282926:

        if more_types_in_union_282927:
            # Runtime conditional SSA (line 1794)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 1795):
        
        # Assigning a Subscript to a Name (line 1795):
        
        # Obtaining the type of the subscript
        int_282928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1795, 32), 'int')
        # Getting the type of 'win' (line 1795)
        win_282929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1795, 22), 'win')
        # Obtaining the member 'shape' of a type (line 1795)
        shape_282930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1795, 22), win_282929, 'shape')
        # Obtaining the member '__getitem__' of a type (line 1795)
        getitem___282931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1795, 22), shape_282930, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1795)
        subscript_call_result_282932 = invoke(stypy.reporting.localization.Localization(__file__, 1795, 22), getitem___282931, int_282928)
        
        # Assigning a type to the variable 'nperseg' (line 1795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1795, 12), 'nperseg', subscript_call_result_282932)

        if more_types_in_union_282927:
            # Runtime conditional SSA for else branch (line 1794)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_282926) or more_types_in_union_282927):
        
        # Type idiom detected: calculating its left and rigth part (line 1796)
        # Getting the type of 'nperseg' (line 1796)
        nperseg_282933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1796, 13), 'nperseg')
        # Getting the type of 'None' (line 1796)
        None_282934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1796, 28), 'None')
        
        (may_be_282935, more_types_in_union_282936) = may_not_be_none(nperseg_282933, None_282934)

        if may_be_282935:

            if more_types_in_union_282936:
                # Runtime conditional SSA (line 1796)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Getting the type of 'nperseg' (line 1797)
            nperseg_282937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1797, 15), 'nperseg')
            
            # Obtaining the type of the subscript
            int_282938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1797, 36), 'int')
            # Getting the type of 'win' (line 1797)
            win_282939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1797, 26), 'win')
            # Obtaining the member 'shape' of a type (line 1797)
            shape_282940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1797, 26), win_282939, 'shape')
            # Obtaining the member '__getitem__' of a type (line 1797)
            getitem___282941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1797, 26), shape_282940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 1797)
            subscript_call_result_282942 = invoke(stypy.reporting.localization.Localization(__file__, 1797, 26), getitem___282941, int_282938)
            
            # Applying the binary operator '!=' (line 1797)
            result_ne_282943 = python_operator(stypy.reporting.localization.Localization(__file__, 1797, 15), '!=', nperseg_282937, subscript_call_result_282942)
            
            # Testing the type of an if condition (line 1797)
            if_condition_282944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1797, 12), result_ne_282943)
            # Assigning a type to the variable 'if_condition_282944' (line 1797)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1797, 12), 'if_condition_282944', if_condition_282944)
            # SSA begins for if statement (line 1797)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 1798)
            # Processing the call arguments (line 1798)
            str_282946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1798, 33), 'str', 'value specified for nperseg is different from length of window')
            # Processing the call keyword arguments (line 1798)
            kwargs_282947 = {}
            # Getting the type of 'ValueError' (line 1798)
            ValueError_282945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1798, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 1798)
            ValueError_call_result_282948 = invoke(stypy.reporting.localization.Localization(__file__, 1798, 22), ValueError_282945, *[str_282946], **kwargs_282947)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1798, 16), ValueError_call_result_282948, 'raise parameter', BaseException)
            # SSA join for if statement (line 1797)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_282936:
                # SSA join for if statement (line 1796)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_282926 and more_types_in_union_282927):
            # SSA join for if statement (line 1794)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 1778)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1800)
    tuple_282949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1800, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1800)
    # Adding element type (line 1800)
    # Getting the type of 'win' (line 1800)
    win_282950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1800, 11), 'win')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1800, 11), tuple_282949, win_282950)
    # Adding element type (line 1800)
    # Getting the type of 'nperseg' (line 1800)
    nperseg_282951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1800, 16), 'nperseg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1800, 11), tuple_282949, nperseg_282951)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1800, 4), 'stypy_return_type', tuple_282949)
    
    # ################# End of '_triage_segments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_triage_segments' in the type store
    # Getting the type of 'stypy_return_type' (line 1742)
    stypy_return_type_282952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1742, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_282952)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_triage_segments'
    return stypy_return_type_282952

# Assigning a type to the variable '_triage_segments' (line 1742)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1742, 0), '_triage_segments', _triage_segments)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
