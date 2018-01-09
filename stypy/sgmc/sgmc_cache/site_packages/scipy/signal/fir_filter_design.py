
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: '''Functions for FIR filter design.'''
3: from __future__ import division, print_function, absolute_import
4: 
5: from math import ceil, log
6: import warnings
7: 
8: import numpy as np
9: from numpy.fft import irfft, fft, ifft
10: from scipy.special import sinc
11: from scipy.linalg import toeplitz, hankel, pinv
12: from scipy._lib.six import string_types
13: 
14: from . import sigtools
15: 
16: __all__ = ['kaiser_beta', 'kaiser_atten', 'kaiserord',
17:            'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase']
18: 
19: 
20: def _get_fs(fs, nyq):
21:     '''
22:     Utility for replacing the argument 'nyq' (with default 1) with 'fs'.
23:     '''
24:     if nyq is None and fs is None:
25:         fs = 2
26:     elif nyq is not None:
27:         if fs is not None:
28:             raise ValueError("Values cannot be given for both 'nyq' and 'fs'.")
29:         fs = 2*nyq
30:     return fs
31: 
32: 
33: # Some notes on function parameters:
34: #
35: # `cutoff` and `width` are given as a numbers between 0 and 1.  These are
36: # relative frequencies, expressed as a fraction of the Nyquist frequency.
37: # For example, if the Nyquist frequency is 2KHz, then width=0.15 is a width
38: # of 300 Hz.
39: #
40: # The `order` of a FIR filter is one less than the number of taps.
41: # This is a potential source of confusion, so in the following code,
42: # we will always use the number of taps as the parameterization of
43: # the 'size' of the filter. The "number of taps" means the number
44: # of coefficients, which is the same as the length of the impulse
45: # response of the filter.
46: 
47: 
48: def kaiser_beta(a):
49:     '''Compute the Kaiser parameter `beta`, given the attenuation `a`.
50: 
51:     Parameters
52:     ----------
53:     a : float
54:         The desired attenuation in the stopband and maximum ripple in
55:         the passband, in dB.  This should be a *positive* number.
56: 
57:     Returns
58:     -------
59:     beta : float
60:         The `beta` parameter to be used in the formula for a Kaiser window.
61: 
62:     References
63:     ----------
64:     Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
65: 
66:     Examples
67:     --------
68:     Suppose we want to design a lowpass filter, with 65 dB attenuation
69:     in the stop band.  The Kaiser window parameter to be used in the
70:     window method is computed by `kaiser_beta(65)`:
71: 
72:     >>> from scipy.signal import kaiser_beta
73:     >>> kaiser_beta(65)
74:     6.20426
75: 
76:     '''
77:     if a > 50:
78:         beta = 0.1102 * (a - 8.7)
79:     elif a > 21:
80:         beta = 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
81:     else:
82:         beta = 0.0
83:     return beta
84: 
85: 
86: def kaiser_atten(numtaps, width):
87:     '''Compute the attenuation of a Kaiser FIR filter.
88: 
89:     Given the number of taps `N` and the transition width `width`, compute the
90:     attenuation `a` in dB, given by Kaiser's formula:
91: 
92:         a = 2.285 * (N - 1) * pi * width + 7.95
93: 
94:     Parameters
95:     ----------
96:     numtaps : int
97:         The number of taps in the FIR filter.
98:     width : float
99:         The desired width of the transition region between passband and
100:         stopband (or, in general, at any discontinuity) for the filter,
101:         expressed as a fraction of the Nyquist frequency.
102: 
103:     Returns
104:     -------
105:     a : float
106:         The attenuation of the ripple, in dB.
107: 
108:     See Also
109:     --------
110:     kaiserord, kaiser_beta
111: 
112:     Examples
113:     --------
114:     Suppose we want to design a FIR filter using the Kaiser window method
115:     that will have 211 taps and a transition width of 9 Hz for a signal that
116:     is sampled at 480 Hz.  Expressed as a fraction of the Nyquist frequency,
117:     the width is 9/(0.5*480) = 0.0375.  The approximate attenuation (in dB)
118:     is computed as follows:
119: 
120:     >>> from scipy.signal import kaiser_atten
121:     >>> kaiser_atten(211, 0.0375)
122:     64.48099630593983
123: 
124:     '''
125:     a = 2.285 * (numtaps - 1) * np.pi * width + 7.95
126:     return a
127: 
128: 
129: def kaiserord(ripple, width):
130:     '''
131:     Determine the filter window parameters for the Kaiser window method.
132: 
133:     The parameters returned by this function are generally used to create
134:     a finite impulse response filter using the window method, with either
135:     `firwin` or `firwin2`.
136: 
137:     Parameters
138:     ----------
139:     ripple : float
140:         Upper bound for the deviation (in dB) of the magnitude of the
141:         filter's frequency response from that of the desired filter (not
142:         including frequencies in any transition intervals).  That is, if w
143:         is the frequency expressed as a fraction of the Nyquist frequency,
144:         A(w) is the actual frequency response of the filter and D(w) is the
145:         desired frequency response, the design requirement is that::
146: 
147:             abs(A(w) - D(w))) < 10**(-ripple/20)
148: 
149:         for 0 <= w <= 1 and w not in a transition interval.
150:     width : float
151:         Width of transition region, normalized so that 1 corresponds to pi
152:         radians / sample.  That is, the frequency is expressed as a fraction
153:         of the Nyquist frequency.
154: 
155:     Returns
156:     -------
157:     numtaps : int
158:         The length of the Kaiser window.
159:     beta : float
160:         The beta parameter for the Kaiser window.
161: 
162:     See Also
163:     --------
164:     kaiser_beta, kaiser_atten
165: 
166:     Notes
167:     -----
168:     There are several ways to obtain the Kaiser window:
169: 
170:     - ``signal.kaiser(numtaps, beta, sym=True)``
171:     - ``signal.get_window(beta, numtaps)``
172:     - ``signal.get_window(('kaiser', beta), numtaps)``
173: 
174:     The empirical equations discovered by Kaiser are used.
175: 
176:     References
177:     ----------
178:     Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.
179: 
180:     Examples
181:     --------
182:     We will use the Kaiser window method to design a lowpass FIR filter
183:     for a signal that is sampled at 1000 Hz.
184: 
185:     We want at least 65 dB rejection in the stop band, and in the pass
186:     band the gain should vary no more than 0.5%.
187: 
188:     We want a cutoff frequency of 175 Hz, with a transition between the
189:     pass band and the stop band of 24 Hz.  That is, in the band [0, 163],
190:     the gain varies no more than 0.5%, and in the band [187, 500], the
191:     signal is attenuated by at least 65 dB.
192: 
193:     >>> from scipy.signal import kaiserord, firwin, freqz
194:     >>> import matplotlib.pyplot as plt
195:     >>> fs = 1000.0
196:     >>> cutoff = 175
197:     >>> width = 24
198: 
199:     The Kaiser method accepts just a single parameter to control the pass
200:     band ripple and the stop band rejection, so we use the more restrictive
201:     of the two.  In this case, the pass band ripple is 0.005, or 46.02 dB,
202:     so we will use 65 dB as the design parameter.
203: 
204:     Use `kaiserord` to determine the length of the filter and the
205:     parameter for the Kaiser window.
206: 
207:     >>> numtaps, beta = kaiserord(65, width/(0.5*fs))
208:     >>> numtaps
209:     167
210:     >>> beta
211:     6.20426
212: 
213:     Use `firwin` to create the FIR filter.
214: 
215:     >>> taps = firwin(numtaps, cutoff, window=('kaiser', beta),
216:     ...               scale=False, nyq=0.5*fs)
217: 
218:     Compute the frequency response of the filter.  ``w`` is the array of
219:     frequencies, and ``h`` is the corresponding complex array of frequency
220:     responses.
221: 
222:     >>> w, h = freqz(taps, worN=8000)
223:     >>> w *= 0.5*fs/np.pi  # Convert w to Hz.
224: 
225:     Compute the deviation of the magnitude of the filter's response from
226:     that of the ideal lowpass filter.  Values in the transition region are
227:     set to ``nan``, so they won't appear in the plot.
228: 
229:     >>> ideal = w < cutoff  # The "ideal" frequency response.
230:     >>> deviation = np.abs(np.abs(h) - ideal)
231:     >>> deviation[(w > cutoff - 0.5*width) & (w < cutoff + 0.5*width)] = np.nan
232: 
233:     Plot the deviation.  A close look at the left end of the stop band shows
234:     that the requirement for 65 dB attenuation is violated in the first lobe
235:     by about 0.125 dB.  This is not unusual for the Kaiser window method.
236: 
237:     >>> plt.plot(w, 20*np.log10(np.abs(deviation)))
238:     >>> plt.xlim(0, 0.5*fs)
239:     >>> plt.ylim(-90, -60)
240:     >>> plt.grid(alpha=0.25)
241:     >>> plt.axhline(-65, color='r', ls='--', alpha=0.3)
242:     >>> plt.xlabel('Frequency (Hz)')
243:     >>> plt.ylabel('Deviation from ideal (dB)')
244:     >>> plt.title('Lowpass Filter Frequency Response')
245:     >>> plt.show()
246: 
247:     '''
248:     A = abs(ripple)  # in case somebody is confused as to what's meant
249:     if A < 8:
250:         # Formula for N is not valid in this range.
251:         raise ValueError("Requested maximum ripple attentuation %f is too "
252:                          "small for the Kaiser formula." % A)
253:     beta = kaiser_beta(A)
254: 
255:     # Kaiser's formula (as given in Oppenheim and Schafer) is for the filter
256:     # order, so we have to add 1 to get the number of taps.
257:     numtaps = (A - 7.95) / 2.285 / (np.pi * width) + 1
258: 
259:     return int(ceil(numtaps)), beta
260: 
261: 
262: def firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True,
263:            scale=True, nyq=None, fs=None):
264:     '''
265:     FIR filter design using the window method.
266: 
267:     This function computes the coefficients of a finite impulse response
268:     filter.  The filter will have linear phase; it will be Type I if
269:     `numtaps` is odd and Type II if `numtaps` is even.
270: 
271:     Type II filters always have zero response at the Nyquist frequency, so a
272:     ValueError exception is raised if firwin is called with `numtaps` even and
273:     having a passband whose right end is at the Nyquist frequency.
274: 
275:     Parameters
276:     ----------
277:     numtaps : int
278:         Length of the filter (number of coefficients, i.e. the filter
279:         order + 1).  `numtaps` must be even if a passband includes the
280:         Nyquist frequency.
281:     cutoff : float or 1D array_like
282:         Cutoff frequency of filter (expressed in the same units as `nyq`)
283:         OR an array of cutoff frequencies (that is, band edges). In the
284:         latter case, the frequencies in `cutoff` should be positive and
285:         monotonically increasing between 0 and `nyq`.  The values 0 and
286:         `nyq` must not be included in `cutoff`.
287:     width : float or None, optional
288:         If `width` is not None, then assume it is the approximate width
289:         of the transition region (expressed in the same units as `nyq`)
290:         for use in Kaiser FIR filter design.  In this case, the `window`
291:         argument is ignored.
292:     window : string or tuple of string and parameter values, optional
293:         Desired window to use. See `scipy.signal.get_window` for a list
294:         of windows and required parameters.
295:     pass_zero : bool, optional
296:         If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.
297:         Otherwise the DC gain is 0.
298:     scale : bool, optional
299:         Set to True to scale the coefficients so that the frequency
300:         response is exactly unity at a certain frequency.
301:         That frequency is either:
302: 
303:         - 0 (DC) if the first passband starts at 0 (i.e. pass_zero
304:           is True)
305:         - `nyq` (the Nyquist frequency) if the first passband ends at
306:           `nyq` (i.e the filter is a single band highpass filter);
307:           center of first passband otherwise
308: 
309:     nyq : float, optional
310:         *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.
311:         Each frequency in `cutoff` must be between 0 and `nyq`. Default
312:         is 1.
313:     fs : float, optional
314:         The sampling frequency of the signal.  Each frequency in `cutoff`
315:         must be between 0 and ``fs/2``.  Default is 2.
316: 
317:     Returns
318:     -------
319:     h : (numtaps,) ndarray
320:         Coefficients of length `numtaps` FIR filter.
321: 
322:     Raises
323:     ------
324:     ValueError
325:         If any value in `cutoff` is less than or equal to 0 or greater
326:         than or equal to ``fs/2``, if the values in `cutoff` are not strictly
327:         monotonically increasing, or if `numtaps` is even but a passband
328:         includes the Nyquist frequency.
329: 
330:     See Also
331:     --------
332:     firwin2
333:     firls
334:     minimum_phase
335:     remez
336: 
337:     Examples
338:     --------
339:     Low-pass from 0 to f:
340: 
341:     >>> from scipy import signal
342:     >>> numtaps = 3
343:     >>> f = 0.1
344:     >>> signal.firwin(numtaps, f)
345:     array([ 0.06799017,  0.86401967,  0.06799017])
346: 
347:     Use a specific window function:
348: 
349:     >>> signal.firwin(numtaps, f, window='nuttall')
350:     array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])
351: 
352:     High-pass ('stop' from 0 to f):
353: 
354:     >>> signal.firwin(numtaps, f, pass_zero=False)
355:     array([-0.00859313,  0.98281375, -0.00859313])
356: 
357:     Band-pass:
358: 
359:     >>> f1, f2 = 0.1, 0.2
360:     >>> signal.firwin(numtaps, [f1, f2], pass_zero=False)
361:     array([ 0.06301614,  0.88770441,  0.06301614])
362: 
363:     Band-stop:
364: 
365:     >>> signal.firwin(numtaps, [f1, f2])
366:     array([-0.00801395,  1.0160279 , -0.00801395])
367: 
368:     Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):
369: 
370:     >>> f3, f4 = 0.3, 0.4
371:     >>> signal.firwin(numtaps, [f1, f2, f3, f4])
372:     array([-0.01376344,  1.02752689, -0.01376344])
373: 
374:     Multi-band (passbands are [f1, f2] and [f3,f4]):
375: 
376:     >>> signal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)
377:     array([ 0.04890915,  0.91284326,  0.04890915])
378: 
379:     '''
380:     # The major enhancements to this function added in November 2010 were
381:     # developed by Tom Krauss (see ticket #902).
382: 
383:     nyq = 0.5 * _get_fs(fs, nyq)
384: 
385:     cutoff = np.atleast_1d(cutoff) / float(nyq)
386: 
387:     # Check for invalid input.
388:     if cutoff.ndim > 1:
389:         raise ValueError("The cutoff argument must be at most "
390:                          "one-dimensional.")
391:     if cutoff.size == 0:
392:         raise ValueError("At least one cutoff frequency must be given.")
393:     if cutoff.min() <= 0 or cutoff.max() >= 1:
394:         raise ValueError("Invalid cutoff frequency: frequencies must be "
395:                          "greater than 0 and less than fs/2.")
396:     if np.any(np.diff(cutoff) <= 0):
397:         raise ValueError("Invalid cutoff frequencies: the frequencies "
398:                          "must be strictly increasing.")
399: 
400:     if width is not None:
401:         # A width was given.  Find the beta parameter of the Kaiser window
402:         # and set `window`.  This overrides the value of `window` passed in.
403:         atten = kaiser_atten(numtaps, float(width) / nyq)
404:         beta = kaiser_beta(atten)
405:         window = ('kaiser', beta)
406: 
407:     pass_nyquist = bool(cutoff.size & 1) ^ pass_zero
408:     if pass_nyquist and numtaps % 2 == 0:
409:         raise ValueError("A filter with an even number of coefficients must "
410:                          "have zero response at the Nyquist frequency.")
411: 
412:     # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
413:     # is even, and each pair in cutoff corresponds to passband.
414:     cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))
415: 
416:     # `bands` is a 2D array; each row gives the left and right edges of
417:     # a passband.
418:     bands = cutoff.reshape(-1, 2)
419: 
420:     # Build up the coefficients.
421:     alpha = 0.5 * (numtaps - 1)
422:     m = np.arange(0, numtaps) - alpha
423:     h = 0
424:     for left, right in bands:
425:         h += right * sinc(right * m)
426:         h -= left * sinc(left * m)
427: 
428:     # Get and apply the window function.
429:     from .signaltools import get_window
430:     win = get_window(window, numtaps, fftbins=False)
431:     h *= win
432: 
433:     # Now handle scaling if desired.
434:     if scale:
435:         # Get the first passband.
436:         left, right = bands[0]
437:         if left == 0:
438:             scale_frequency = 0.0
439:         elif right == 1:
440:             scale_frequency = 1.0
441:         else:
442:             scale_frequency = 0.5 * (left + right)
443:         c = np.cos(np.pi * m * scale_frequency)
444:         s = np.sum(h * c)
445:         h /= s
446: 
447:     return h
448: 
449: 
450: # Original version of firwin2 from scipy ticket #457, submitted by "tash".
451: #
452: # Rewritten by Warren Weckesser, 2010.
453: 
454: def firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', nyq=None,
455:             antisymmetric=False, fs=None):
456:     '''
457:     FIR filter design using the window method.
458: 
459:     From the given frequencies `freq` and corresponding gains `gain`,
460:     this function constructs an FIR filter with linear phase and
461:     (approximately) the given frequency response.
462: 
463:     Parameters
464:     ----------
465:     numtaps : int
466:         The number of taps in the FIR filter.  `numtaps` must be less than
467:         `nfreqs`.
468:     freq : array_like, 1D
469:         The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being
470:         Nyquist.  The Nyquist frequency is half `fs`.
471:         The values in `freq` must be nondecreasing.  A value can be repeated
472:         once to implement a discontinuity.  The first value in `freq` must
473:         be 0, and the last value must be ``fs/2``.
474:     gain : array_like
475:         The filter gains at the frequency sampling points. Certain
476:         constraints to gain values, depending on the filter type, are applied,
477:         see Notes for details.
478:     nfreqs : int, optional
479:         The size of the interpolation mesh used to construct the filter.
480:         For most efficient behavior, this should be a power of 2 plus 1
481:         (e.g, 129, 257, etc).  The default is one more than the smallest
482:         power of 2 that is not less than `numtaps`.  `nfreqs` must be greater
483:         than `numtaps`.
484:     window : string or (string, float) or float, or None, optional
485:         Window function to use. Default is "hamming".  See
486:         `scipy.signal.get_window` for the complete list of possible values.
487:         If None, no window function is applied.
488:     nyq : float, optional
489:         *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.
490:         Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.
491:     antisymmetric : bool, optional
492:         Whether resulting impulse response is symmetric/antisymmetric.
493:         See Notes for more details.
494:     fs : float, optional
495:         The sampling frequency of the signal.  Each frequency in `cutoff`
496:         must be between 0 and ``fs/2``.  Default is 2.
497: 
498:     Returns
499:     -------
500:     taps : ndarray
501:         The filter coefficients of the FIR filter, as a 1-D array of length
502:         `numtaps`.
503: 
504:     See also
505:     --------
506:     firls
507:     firwin
508:     minimum_phase
509:     remez
510: 
511:     Notes
512:     -----
513:     From the given set of frequencies and gains, the desired response is
514:     constructed in the frequency domain.  The inverse FFT is applied to the
515:     desired response to create the associated convolution kernel, and the
516:     first `numtaps` coefficients of this kernel, scaled by `window`, are
517:     returned.
518: 
519:     The FIR filter will have linear phase. The type of filter is determined by
520:     the value of 'numtaps` and `antisymmetric` flag.
521:     There are four possible combinations:
522: 
523:        - odd  `numtaps`, `antisymmetric` is False, type I filter is produced
524:        - even `numtaps`, `antisymmetric` is False, type II filter is produced
525:        - odd  `numtaps`, `antisymmetric` is True, type III filter is produced
526:        - even `numtaps`, `antisymmetric` is True, type IV filter is produced
527: 
528:     Magnitude response of all but type I filters are subjects to following
529:     constraints:
530: 
531:        - type II  -- zero at the Nyquist frequency
532:        - type III -- zero at zero and Nyquist frequencies
533:        - type IV  -- zero at zero frequency
534: 
535:     .. versionadded:: 0.9.0
536: 
537:     References
538:     ----------
539:     .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal
540:        Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).
541:        (See, for example, Section 7.4.)
542: 
543:     .. [2] Smith, Steven W., "The Scientist and Engineer's Guide to Digital
544:        Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm
545: 
546:     Examples
547:     --------
548:     A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and
549:     that decreases linearly on [0.5, 1.0] from 1 to 0:
550: 
551:     >>> from scipy import signal
552:     >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
553:     >>> print(taps[72:78])
554:     [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]
555: 
556:     '''
557:     nyq = 0.5 * _get_fs(fs, nyq)
558: 
559:     if len(freq) != len(gain):
560:         raise ValueError('freq and gain must be of same length.')
561: 
562:     if nfreqs is not None and numtaps >= nfreqs:
563:         raise ValueError(('ntaps must be less than nfreqs, but firwin2 was '
564:                           'called with ntaps=%d and nfreqs=%s') %
565:                          (numtaps, nfreqs))
566: 
567:     if freq[0] != 0 or freq[-1] != nyq:
568:         raise ValueError('freq must start with 0 and end with fs/2.')
569:     d = np.diff(freq)
570:     if (d < 0).any():
571:         raise ValueError('The values in freq must be nondecreasing.')
572:     d2 = d[:-1] + d[1:]
573:     if (d2 == 0).any():
574:         raise ValueError('A value in freq must not occur more than twice.')
575: 
576:     if antisymmetric:
577:         if numtaps % 2 == 0:
578:             ftype = 4
579:         else:
580:             ftype = 3
581:     else:
582:         if numtaps % 2 == 0:
583:             ftype = 2
584:         else:
585:             ftype = 1
586: 
587:     if ftype == 2 and gain[-1] != 0.0:
588:         raise ValueError("A Type II filter must have zero gain at the "
589:                          "Nyquist frequency.")
590:     elif ftype == 3 and (gain[0] != 0.0 or gain[-1] != 0.0):
591:         raise ValueError("A Type III filter must have zero gain at zero "
592:                          "and Nyquist frequencies.")
593:     elif ftype == 4 and gain[0] != 0.0:
594:         raise ValueError("A Type IV filter must have zero gain at zero "
595:                          "frequency.")
596: 
597:     if nfreqs is None:
598:         nfreqs = 1 + 2 ** int(ceil(log(numtaps, 2)))
599: 
600:     # Tweak any repeated values in freq so that interp works.
601:     eps = np.finfo(float).eps
602:     for k in range(len(freq)):
603:         if k < len(freq) - 1 and freq[k] == freq[k + 1]:
604:             freq[k] = freq[k] - eps
605:             freq[k + 1] = freq[k + 1] + eps
606: 
607:     # Linearly interpolate the desired response on a uniform mesh `x`.
608:     x = np.linspace(0.0, nyq, nfreqs)
609:     fx = np.interp(x, freq, gain)
610: 
611:     # Adjust the phases of the coefficients so that the first `ntaps` of the
612:     # inverse FFT are the desired filter coefficients.
613:     shift = np.exp(-(numtaps - 1) / 2. * 1.j * np.pi * x / nyq)
614:     if ftype > 2:
615:         shift *= 1j
616: 
617:     fx2 = fx * shift
618: 
619:     # Use irfft to compute the inverse FFT.
620:     out_full = irfft(fx2)
621: 
622:     if window is not None:
623:         # Create the window to apply to the filter coefficients.
624:         from .signaltools import get_window
625:         wind = get_window(window, numtaps, fftbins=False)
626:     else:
627:         wind = 1
628: 
629:     # Keep only the first `numtaps` coefficients in `out`, and multiply by
630:     # the window.
631:     out = out_full[:numtaps] * wind
632: 
633:     if ftype == 3:
634:         out[out.size // 2] = 0.0
635: 
636:     return out
637: 
638: 
639: def remez(numtaps, bands, desired, weight=None, Hz=None, type='bandpass',
640:           maxiter=25, grid_density=16, fs=None):
641:     '''
642:     Calculate the minimax optimal filter using the Remez exchange algorithm.
643: 
644:     Calculate the filter-coefficients for the finite impulse response
645:     (FIR) filter whose transfer function minimizes the maximum error
646:     between the desired gain and the realized gain in the specified
647:     frequency bands using the Remez exchange algorithm.
648: 
649:     Parameters
650:     ----------
651:     numtaps : int
652:         The desired number of taps in the filter. The number of taps is
653:         the number of terms in the filter, or the filter order plus one.
654:     bands : array_like
655:         A monotonic sequence containing the band edges.
656:         All elements must be non-negative and less than half the sampling
657:         frequency as given by `fs`.
658:     desired : array_like
659:         A sequence half the size of bands containing the desired gain
660:         in each of the specified bands.
661:     weight : array_like, optional
662:         A relative weighting to give to each band region. The length of
663:         `weight` has to be half the length of `bands`.
664:     Hz : scalar, optional
665:         *Deprecated.  Use `fs` instead.*
666:         The sampling frequency in Hz. Default is 1.
667:     type : {'bandpass', 'differentiator', 'hilbert'}, optional
668:         The type of filter:
669: 
670:           * 'bandpass' : flat response in bands. This is the default.
671: 
672:           * 'differentiator' : frequency proportional response in bands.
673: 
674:           * 'hilbert' : filter with odd symmetry, that is, type III
675:                         (for even order) or type IV (for odd order)
676:                         linear phase filters.
677: 
678:     maxiter : int, optional
679:         Maximum number of iterations of the algorithm. Default is 25.
680:     grid_density : int, optional
681:         Grid density. The dense grid used in `remez` is of size
682:         ``(numtaps + 1) * grid_density``. Default is 16.
683:     fs : float, optional
684:         The sampling frequency of the signal.  Default is 1.
685: 
686:     Returns
687:     -------
688:     out : ndarray
689:         A rank-1 array containing the coefficients of the optimal
690:         (in a minimax sense) filter.
691: 
692:     See Also
693:     --------
694:     firls
695:     firwin
696:     firwin2
697:     minimum_phase
698: 
699:     References
700:     ----------
701:     .. [1] J. H. McClellan and T. W. Parks, "A unified approach to the
702:            design of optimum FIR linear phase digital filters",
703:            IEEE Trans. Circuit Theory, vol. CT-20, pp. 697-701, 1973.
704:     .. [2] J. H. McClellan, T. W. Parks and L. R. Rabiner, "A Computer
705:            Program for Designing Optimum FIR Linear Phase Digital
706:            Filters", IEEE Trans. Audio Electroacoust., vol. AU-21,
707:            pp. 506-525, 1973.
708: 
709:     Examples
710:     --------
711:     For a signal sampled at 100 Hz, we want to construct a filter with a
712:     passband at 20-40 Hz, and stop bands at 0-10 Hz and 45-50 Hz. Note that
713:     this means that the behavior in the frequency ranges between those bands
714:     is unspecified and may overshoot.
715: 
716:     >>> from scipy import signal
717:     >>> fs = 100
718:     >>> bpass = signal.remez(72, [0, 10, 20, 40, 45, 50], [0, 1, 0], fs=fs)
719:     >>> freq, response = signal.freqz(bpass)
720: 
721:     >>> import matplotlib.pyplot as plt
722:     >>> plt.semilogy(0.5*fs*freq/np.pi, np.abs(response), 'b-')
723:     >>> plt.grid(alpha=0.25)
724:     >>> plt.xlabel('Frequency (Hz)')
725:     >>> plt.ylabel('Gain')
726:     >>> plt.show()
727: 
728:     '''
729:     if Hz is None and fs is None:
730:         fs = 1.0
731:     elif Hz is not None:
732:         if fs is not None:
733:             raise ValueError("Values cannot be given for both 'Hz' and 'fs'.")
734:         fs = Hz
735: 
736:     # Convert type
737:     try:
738:         tnum = {'bandpass': 1, 'differentiator': 2, 'hilbert': 3}[type]
739:     except KeyError:
740:         raise ValueError("Type must be 'bandpass', 'differentiator', "
741:                          "or 'hilbert'")
742: 
743:     # Convert weight
744:     if weight is None:
745:         weight = [1] * len(desired)
746: 
747:     bands = np.asarray(bands).copy()
748:     return sigtools._remez(numtaps, bands, desired, weight, tnum, fs,
749:                            maxiter, grid_density)
750: 
751: 
752: def firls(numtaps, bands, desired, weight=None, nyq=None, fs=None):
753:     '''
754:     FIR filter design using least-squares error minimization.
755: 
756:     Calculate the filter coefficients for the linear-phase finite
757:     impulse response (FIR) filter which has the best approximation
758:     to the desired frequency response described by `bands` and
759:     `desired` in the least squares sense (i.e., the integral of the
760:     weighted mean-squared error within the specified bands is
761:     minimized).
762: 
763:     Parameters
764:     ----------
765:     numtaps : int
766:         The number of taps in the FIR filter.  `numtaps` must be odd.
767:     bands : array_like
768:         A monotonic nondecreasing sequence containing the band edges in
769:         Hz. All elements must be non-negative and less than or equal to
770:         the Nyquist frequency given by `nyq`.
771:     desired : array_like
772:         A sequence the same size as `bands` containing the desired gain
773:         at the start and end point of each band.
774:     weight : array_like, optional
775:         A relative weighting to give to each band region when solving
776:         the least squares problem. `weight` has to be half the size of
777:         `bands`.
778:     nyq : float, optional
779:         *Deprecated.  Use `fs` instead.*
780:         Nyquist frequency. Each frequency in `bands` must be between 0
781:         and `nyq` (inclusive).  Default is 1.
782:     fs : float, optional
783:         The sampling frequency of the signal. Each frequency in `bands`
784:         must be between 0 and ``fs/2`` (inclusive).  Default is 2.
785: 
786:     Returns
787:     -------
788:     coeffs : ndarray
789:         Coefficients of the optimal (in a least squares sense) FIR filter.
790: 
791:     See also
792:     --------
793:     firwin
794:     firwin2
795:     minimum_phase
796:     remez
797: 
798:     Notes
799:     -----
800:     This implementation follows the algorithm given in [1]_.
801:     As noted there, least squares design has multiple advantages:
802: 
803:         1. Optimal in a least-squares sense.
804:         2. Simple, non-iterative method.
805:         3. The general solution can obtained by solving a linear
806:            system of equations.
807:         4. Allows the use of a frequency dependent weighting function.
808: 
809:     This function constructs a Type I linear phase FIR filter, which
810:     contains an odd number of `coeffs` satisfying for :math:`n < numtaps`:
811: 
812:     .. math:: coeffs(n) = coeffs(numtaps - 1 - n)
813: 
814:     The odd number of coefficients and filter symmetry avoid boundary
815:     conditions that could otherwise occur at the Nyquist and 0 frequencies
816:     (e.g., for Type II, III, or IV variants).
817: 
818:     .. versionadded:: 0.18
819: 
820:     References
821:     ----------
822:     .. [1] Ivan Selesnick, Linear-Phase Fir Filter Design By Least Squares.
823:            OpenStax CNX. Aug 9, 2005.
824:            http://cnx.org/contents/eb1ecb35-03a9-4610-ba87-41cd771c95f2@7
825: 
826:     Examples
827:     --------
828:     We want to construct a band-pass filter. Note that the behavior in the
829:     frequency ranges between our stop bands and pass bands is unspecified,
830:     and thus may overshoot depending on the parameters of our filter:
831: 
832:     >>> from scipy import signal
833:     >>> import matplotlib.pyplot as plt
834:     >>> fig, axs = plt.subplots(2)
835:     >>> fs = 10.0  # Hz
836:     >>> desired = (0, 0, 1, 1, 0, 0)
837:     >>> for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):
838:     ...     fir_firls = signal.firls(73, bands, desired, fs=fs)
839:     ...     fir_remez = signal.remez(73, bands, desired[::2], fs=fs)
840:     ...     fir_firwin2 = signal.firwin2(73, bands, desired, fs=fs)
841:     ...     hs = list()
842:     ...     ax = axs[bi]
843:     ...     for fir in (fir_firls, fir_remez, fir_firwin2):
844:     ...         freq, response = signal.freqz(fir)
845:     ...         hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])
846:     ...     for band, gains in zip(zip(bands[::2], bands[1::2]),
847:     ...                            zip(desired[::2], desired[1::2])):
848:     ...         ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)
849:     ...     if bi == 0:
850:     ...         ax.legend(hs, ('firls', 'remez', 'firwin2'),
851:     ...                   loc='lower center', frameon=False)
852:     ...     else:
853:     ...         ax.set_xlabel('Frequency (Hz)')
854:     ...     ax.grid(True)
855:     ...     ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')
856:     ...
857:     >>> fig.tight_layout()
858:     >>> plt.show()
859: 
860:     '''  # noqa
861:     nyq = 0.5 * _get_fs(fs, nyq)
862: 
863:     numtaps = int(numtaps)
864:     if numtaps % 2 == 0 or numtaps < 1:
865:         raise ValueError("numtaps must be odd and >= 1")
866:     M = (numtaps-1) // 2
867: 
868:     # normalize bands 0->1 and make it 2 columns
869:     nyq = float(nyq)
870:     if nyq <= 0:
871:         raise ValueError('nyq must be positive, got %s <= 0.' % nyq)
872:     bands = np.asarray(bands).flatten() / nyq
873:     if len(bands) % 2 != 0:
874:         raise ValueError("bands must contain frequency pairs.")
875:     bands.shape = (-1, 2)
876: 
877:     # check remaining params
878:     desired = np.asarray(desired).flatten()
879:     if bands.size != desired.size:
880:         raise ValueError("desired must have one entry per frequency, got %s "
881:                          "gains for %s frequencies."
882:                          % (desired.size, bands.size))
883:     desired.shape = (-1, 2)
884:     if (np.diff(bands) <= 0).any() or (np.diff(bands[:, 0]) < 0).any():
885:         raise ValueError("bands must be monotonically nondecreasing and have "
886:                          "width > 0.")
887:     if (bands[:-1, 1] > bands[1:, 0]).any():
888:         raise ValueError("bands must not overlap.")
889:     if (desired < 0).any():
890:         raise ValueError("desired must be non-negative.")
891:     if weight is None:
892:         weight = np.ones(len(desired))
893:     weight = np.asarray(weight).flatten()
894:     if len(weight) != len(desired):
895:         raise ValueError("weight must be the same size as the number of "
896:                          "band pairs (%s)." % (len(bands),))
897:     if (weight < 0).any():
898:         raise ValueError("weight must be non-negative.")
899: 
900:     # Set up the linear matrix equation to be solved, Qa = b
901: 
902:     # We can express Q(k,n) = 0.5 Q1(k,n) + 0.5 Q2(k,n)
903:     # where Q1(k,n)=q(k−n) and Q2(k,n)=q(k+n), i.e. a Toeplitz plus Hankel.
904: 
905:     # We omit the factor of 0.5 above, instead adding it during coefficient
906:     # calculation.
907: 
908:     # We also omit the 1/π from both Q and b equations, as they cancel
909:     # during solving.
910: 
911:     # We have that:
912:     #     q(n) = 1/π ∫W(ω)cos(nω)dω (over 0->π)
913:     # Using our nomalization ω=πf and with a constant weight W over each
914:     # interval f1->f2 we get:
915:     #     q(n) = W∫cos(πnf)df (0->1) = Wf sin(πnf)/πnf
916:     # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
917:     n = np.arange(numtaps)[:, np.newaxis, np.newaxis]
918:     q = np.dot(np.diff(np.sinc(bands * n) * bands, axis=2)[:, :, 0], weight)
919: 
920:     # Now we assemble our sum of Toeplitz and Hankel
921:     Q1 = toeplitz(q[:M+1])
922:     Q2 = hankel(q[:M+1], q[M:])
923:     Q = Q1 + Q2
924: 
925:     # Now for b(n) we have that:
926:     #     b(n) = 1/π ∫ W(ω)D(ω)cos(nω)dω (over 0->π)
927:     # Using our normalization ω=πf and with a constant weight W over each
928:     # interval and a linear term for D(ω) we get (over each f1->f2 interval):
929:     #     b(n) = W ∫ (mf+c)cos(πnf)df
930:     #          = f(mf+c)sin(πnf)/πnf + mf**2 cos(nπf)/(πnf)**2
931:     # integrated over each f1->f2 pair (i.e., value at f2 - value at f1).
932:     n = n[:M + 1]  # only need this many coefficients here
933:     # Choose m and c such that we are at the start and end weights
934:     m = (np.diff(desired, axis=1) / np.diff(bands, axis=1))
935:     c = desired[:, [0]] - bands[:, [0]] * m
936:     b = bands * (m*bands + c) * np.sinc(bands * n)
937:     # Use L'Hospital's rule here for cos(nπf)/(πnf)**2 @ n=0
938:     b[0] -= m * bands * bands / 2.
939:     b[1:] += m * np.cos(n[1:] * np.pi * bands) / (np.pi * n[1:]) ** 2
940:     b = np.dot(np.diff(b, axis=2)[:, :, 0], weight)
941: 
942:     # Now we can solve the equation (use pinv because Q can be rank deficient)
943:     a = np.dot(pinv(Q), b)
944: 
945:     # make coefficients symmetric (linear phase)
946:     coeffs = np.hstack((a[:0:-1], 2 * a[0], a[1:]))
947:     return coeffs
948: 
949: 
950: def _dhtm(mag):
951:     '''Compute the modified 1D discrete Hilbert transform
952: 
953:     Parameters
954:     ----------
955:     mag : ndarray
956:         The magnitude spectrum. Should be 1D with an even length, and
957:         preferably a fast length for FFT/IFFT.
958:     '''
959:     # Adapted based on code by Niranjan Damera-Venkata,
960:     # Brian L. Evans and Shawn R. McCaslin (see refs for `minimum_phase`)
961:     sig = np.zeros(len(mag))
962:     # Leave Nyquist and DC at 0, knowing np.abs(fftfreq(N)[midpt]) == 0.5
963:     midpt = len(mag) // 2
964:     sig[1:midpt] = 1
965:     sig[midpt+1:] = -1
966:     # eventually if we want to support complex filters, we will need a
967:     # np.abs() on the mag inside the log, and should remove the .real
968:     recon = ifft(mag * np.exp(fft(sig * ifft(np.log(mag))))).real
969:     return recon
970: 
971: 
972: def minimum_phase(h, method='homomorphic', n_fft=None):
973:     '''Convert a linear-phase FIR filter to minimum phase
974: 
975:     Parameters
976:     ----------
977:     h : array
978:         Linear-phase FIR filter coefficients.
979:     method : {'hilbert', 'homomorphic'}
980:         The method to use:
981: 
982:             'homomorphic' (default)
983:                 This method [4]_ [5]_ works best with filters with an
984:                 odd number of taps, and the resulting minimum phase filter
985:                 will have a magnitude response that approximates the square
986:                 root of the the original filter's magnitude response.
987: 
988:             'hilbert'
989:                 This method [1]_ is designed to be used with equiripple
990:                 filters (e.g., from `remez`) with unity or zero gain
991:                 regions.
992: 
993:     n_fft : int
994:         The number of points to use for the FFT. Should be at least a
995:         few times larger than the signal length (see Notes).
996: 
997:     Returns
998:     -------
999:     h_minimum : array
1000:         The minimum-phase version of the filter, with length
1001:         ``(length(h) + 1) // 2``.
1002: 
1003:     See Also
1004:     --------
1005:     firwin
1006:     firwin2
1007:     remez
1008: 
1009:     Notes
1010:     -----
1011:     Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection
1012:     of an FFT length to estimate the complex cepstrum of the filter.
1013: 
1014:     In the case of the Hilbert method, the deviation from the ideal
1015:     spectrum ``epsilon`` is related to the number of stopband zeros
1016:     ``n_stop`` and FFT length ``n_fft`` as::
1017: 
1018:         epsilon = 2. * n_stop / n_fft
1019: 
1020:     For example, with 100 stopband zeros and a FFT length of 2048,
1021:     ``epsilon = 0.0976``. If we conservatively assume that the number of
1022:     stopband zeros is one less than the filter length, we can take the FFT
1023:     length to be the next power of 2 that satisfies ``epsilon=0.01`` as::
1024: 
1025:         n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
1026: 
1027:     This gives reasonable results for both the Hilbert and homomorphic
1028:     methods, and gives the value used when ``n_fft=None``.
1029: 
1030:     Alternative implementations exist for creating minimum-phase filters,
1031:     including zero inversion [2]_ and spectral factorization [3]_ [4]_.
1032:     For more information, see:
1033: 
1034:         http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters
1035: 
1036:     Examples
1037:     --------
1038:     Create an optimal linear-phase filter, then convert it to minimum phase:
1039: 
1040:     >>> from scipy.signal import remez, minimum_phase, freqz, group_delay
1041:     >>> import matplotlib.pyplot as plt
1042:     >>> freq = [0, 0.2, 0.3, 1.0]
1043:     >>> desired = [1, 0]
1044:     >>> h_linear = remez(151, freq, desired, Hz=2.)
1045: 
1046:     Convert it to minimum phase:
1047: 
1048:     >>> h_min_hom = minimum_phase(h_linear, method='homomorphic')
1049:     >>> h_min_hil = minimum_phase(h_linear, method='hilbert')
1050: 
1051:     Compare the three filters:
1052: 
1053:     >>> fig, axs = plt.subplots(4, figsize=(4, 8))
1054:     >>> for h, style, color in zip((h_linear, h_min_hom, h_min_hil),
1055:     ...                            ('-', '-', '--'), ('k', 'r', 'c')):
1056:     ...     w, H = freqz(h)
1057:     ...     w, gd = group_delay((h, 1))
1058:     ...     w /= np.pi
1059:     ...     axs[0].plot(h, color=color, linestyle=style)
1060:     ...     axs[1].plot(w, np.abs(H), color=color, linestyle=style)
1061:     ...     axs[2].plot(w, 20 * np.log10(np.abs(H)), color=color, linestyle=style)
1062:     ...     axs[3].plot(w, gd, color=color, linestyle=style)
1063:     >>> for ax in axs:
1064:     ...     ax.grid(True, color='0.5')
1065:     ...     ax.fill_between(freq[1:3], *ax.get_ylim(), color='#ffeeaa', zorder=1)
1066:     >>> axs[0].set(xlim=[0, len(h_linear) - 1], ylabel='Amplitude', xlabel='Samples')
1067:     >>> axs[1].legend(['Linear', 'Min-Hom', 'Min-Hil'], title='Phase')
1068:     >>> for ax, ylim in zip(axs[1:], ([0, 1.1], [-150, 10], [-60, 60])):
1069:     ...     ax.set(xlim=[0, 1], ylim=ylim, xlabel='Frequency')
1070:     >>> axs[1].set(ylabel='Magnitude')
1071:     >>> axs[2].set(ylabel='Magnitude (dB)')
1072:     >>> axs[3].set(ylabel='Group delay')
1073:     >>> plt.tight_layout()
1074: 
1075:     References
1076:     ----------
1077:     .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and
1078:            complex minimum phase digital FIR filters," Acoustics, Speech,
1079:            and Signal Processing, 1999. Proceedings., 1999 IEEE International
1080:            Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.
1081:            doi: 10.1109/ICASSP.1999.756179
1082:     .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR
1083:            filters by direct factorization," Signal Processing,
1084:            vol. 10, no. 4, pp. 369–383, Jun. 1986.
1085:     .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in
1086:            Handbook for Digital Signal Processing, chapter 4,
1087:            New York: Wiley-Interscience, 1993.
1088:     .. [4] J. S. Lim, Advanced Topics in Signal Processing.
1089:            Englewood Cliffs, N.J.: Prentice Hall, 1988.
1090:     .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,
1091:            "Discrete-Time Signal Processing," 2nd edition.
1092:            Upper Saddle River, N.J.: Prentice Hall, 1999.
1093:     '''  # noqa
1094:     h = np.asarray(h)
1095:     if np.iscomplexobj(h):
1096:         raise ValueError('Complex filters not supported')
1097:     if h.ndim != 1 or h.size <= 2:
1098:         raise ValueError('h must be 1D and at least 2 samples long')
1099:     n_half = len(h) // 2
1100:     if not np.allclose(h[-n_half:][::-1], h[:n_half]):
1101:         warnings.warn('h does not appear to by symmetric, conversion may '
1102:                       'fail', RuntimeWarning)
1103:     if not isinstance(method, string_types) or method not in \
1104:             ('homomorphic', 'hilbert',):
1105:         raise ValueError('method must be "homomorphic" or "hilbert", got %r'
1106:                          % (method,))
1107:     if n_fft is None:
1108:         n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
1109:     n_fft = int(n_fft)
1110:     if n_fft < len(h):
1111:         raise ValueError('n_fft must be at least len(h)==%s' % len(h))
1112:     if method == 'hilbert':
1113:         w = np.arange(n_fft) * (2 * np.pi / n_fft * n_half)
1114:         H = np.real(fft(h, n_fft) * np.exp(1j * w))
1115:         dp = max(H) - 1
1116:         ds = 0 - min(H)
1117:         S = 4. / (np.sqrt(1+dp+ds) + np.sqrt(1-dp+ds)) ** 2
1118:         H += ds
1119:         H *= S
1120:         H = np.sqrt(H, out=H)
1121:         H += 1e-10  # ensure that the log does not explode
1122:         h_minimum = _dhtm(H)
1123:     else:  # method == 'homomorphic'
1124:         # zero-pad; calculate the DFT
1125:         h_temp = np.abs(fft(h, n_fft))
1126:         # take 0.25*log(|H|**2) = 0.5*log(|H|)
1127:         h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
1128:         np.log(h_temp, out=h_temp)
1129:         h_temp *= 0.5
1130:         # IDFT
1131:         h_temp = ifft(h_temp).real
1132:         # multiply pointwise by the homomorphic filter
1133:         # lmin[n] = 2u[n] - d[n]
1134:         win = np.zeros(n_fft)
1135:         win[0] = 1
1136:         stop = (len(h) + 1) // 2
1137:         win[1:stop] = 2
1138:         if len(h) % 2:
1139:             win[stop] = 1
1140:         h_temp *= win
1141:         h_temp = ifft(np.exp(fft(h_temp)))
1142:         h_minimum = h_temp.real
1143:     n_out = n_half + len(h) % 2
1144:     return h_minimum[:n_out]
1145: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_265525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 0), 'str', 'Functions for FIR filter design.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from math import ceil, log' statement (line 5)
try:
    from math import ceil, log

except:
    ceil = UndefinedType
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'math', None, module_type_store, ['ceil', 'log'], [ceil, log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265526 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_265526) is not StypyTypeError):

    if (import_265526 != 'pyd_module'):
        __import__(import_265526)
        sys_modules_265527 = sys.modules[import_265526]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_265527.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_265526)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.fft import irfft, fft, ifft' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265528 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.fft')

if (type(import_265528) is not StypyTypeError):

    if (import_265528 != 'pyd_module'):
        __import__(import_265528)
        sys_modules_265529 = sys.modules[import_265528]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.fft', sys_modules_265529.module_type_store, module_type_store, ['irfft', 'fft', 'ifft'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_265529, sys_modules_265529.module_type_store, module_type_store)
    else:
        from numpy.fft import irfft, fft, ifft

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.fft', None, module_type_store, ['irfft', 'fft', 'ifft'], [irfft, fft, ifft])

else:
    # Assigning a type to the variable 'numpy.fft' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.fft', import_265528)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.special import sinc' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265530 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special')

if (type(import_265530) is not StypyTypeError):

    if (import_265530 != 'pyd_module'):
        __import__(import_265530)
        sys_modules_265531 = sys.modules[import_265530]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', sys_modules_265531.module_type_store, module_type_store, ['sinc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_265531, sys_modules_265531.module_type_store, module_type_store)
    else:
        from scipy.special import sinc

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', None, module_type_store, ['sinc'], [sinc])

else:
    # Assigning a type to the variable 'scipy.special' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', import_265530)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg import toeplitz, hankel, pinv' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265532 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_265532) is not StypyTypeError):

    if (import_265532 != 'pyd_module'):
        __import__(import_265532)
        sys_modules_265533 = sys.modules[import_265532]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_265533.module_type_store, module_type_store, ['toeplitz', 'hankel', 'pinv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_265533, sys_modules_265533.module_type_store, module_type_store)
    else:
        from scipy.linalg import toeplitz, hankel, pinv

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', None, module_type_store, ['toeplitz', 'hankel', 'pinv'], [toeplitz, hankel, pinv])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_265532)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib.six import string_types' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265534 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six')

if (type(import_265534) is not StypyTypeError):

    if (import_265534 != 'pyd_module'):
        __import__(import_265534)
        sys_modules_265535 = sys.modules[import_265534]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', sys_modules_265535.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_265535, sys_modules_265535.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', import_265534)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.signal import sigtools' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_265536 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.signal')

if (type(import_265536) is not StypyTypeError):

    if (import_265536 != 'pyd_module'):
        __import__(import_265536)
        sys_modules_265537 = sys.modules[import_265536]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.signal', sys_modules_265537.module_type_store, module_type_store, ['sigtools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_265537, sys_modules_265537.module_type_store, module_type_store)
    else:
        from scipy.signal import sigtools

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.signal', None, module_type_store, ['sigtools'], [sigtools])

else:
    # Assigning a type to the variable 'scipy.signal' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.signal', import_265536)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = ['kaiser_beta', 'kaiser_atten', 'kaiserord', 'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase']
module_type_store.set_exportable_members(['kaiser_beta', 'kaiser_atten', 'kaiserord', 'firwin', 'firwin2', 'remez', 'firls', 'minimum_phase'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_265538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_265539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'kaiser_beta')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265539)
# Adding element type (line 16)
str_265540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'kaiser_atten')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265540)
# Adding element type (line 16)
str_265541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 42), 'str', 'kaiserord')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265541)
# Adding element type (line 16)
str_265542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'firwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265542)
# Adding element type (line 16)
str_265543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'str', 'firwin2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265543)
# Adding element type (line 16)
str_265544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'str', 'remez')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265544)
# Adding element type (line 16)
str_265545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'str', 'firls')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265545)
# Adding element type (line 16)
str_265546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 50), 'str', 'minimum_phase')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_265538, str_265546)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_265538)

@norecursion
def _get_fs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_fs'
    module_type_store = module_type_store.open_function_context('_get_fs', 20, 0, False)
    
    # Passed parameters checking function
    _get_fs.stypy_localization = localization
    _get_fs.stypy_type_of_self = None
    _get_fs.stypy_type_store = module_type_store
    _get_fs.stypy_function_name = '_get_fs'
    _get_fs.stypy_param_names_list = ['fs', 'nyq']
    _get_fs.stypy_varargs_param_name = None
    _get_fs.stypy_kwargs_param_name = None
    _get_fs.stypy_call_defaults = defaults
    _get_fs.stypy_call_varargs = varargs
    _get_fs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_fs', ['fs', 'nyq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_fs', localization, ['fs', 'nyq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_fs(...)' code ##################

    str_265547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', "\n    Utility for replacing the argument 'nyq' (with default 1) with 'fs'.\n    ")
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'nyq' (line 24)
    nyq_265548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'nyq')
    # Getting the type of 'None' (line 24)
    None_265549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'None')
    # Applying the binary operator 'is' (line 24)
    result_is__265550 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), 'is', nyq_265548, None_265549)
    
    
    # Getting the type of 'fs' (line 24)
    fs_265551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'fs')
    # Getting the type of 'None' (line 24)
    None_265552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'None')
    # Applying the binary operator 'is' (line 24)
    result_is__265553 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 23), 'is', fs_265551, None_265552)
    
    # Applying the binary operator 'and' (line 24)
    result_and_keyword_265554 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), 'and', result_is__265550, result_is__265553)
    
    # Testing the type of an if condition (line 24)
    if_condition_265555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_and_keyword_265554)
    # Assigning a type to the variable 'if_condition_265555' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_265555', if_condition_265555)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 25):
    
    # Assigning a Num to a Name (line 25):
    int_265556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Assigning a type to the variable 'fs' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'fs', int_265556)
    # SSA branch for the else part of an if statement (line 24)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 26)
    # Getting the type of 'nyq' (line 26)
    nyq_265557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'nyq')
    # Getting the type of 'None' (line 26)
    None_265558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'None')
    
    (may_be_265559, more_types_in_union_265560) = may_not_be_none(nyq_265557, None_265558)

    if may_be_265559:

        if more_types_in_union_265560:
            # Runtime conditional SSA (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 27)
        # Getting the type of 'fs' (line 27)
        fs_265561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'fs')
        # Getting the type of 'None' (line 27)
        None_265562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'None')
        
        (may_be_265563, more_types_in_union_265564) = may_not_be_none(fs_265561, None_265562)

        if may_be_265563:

            if more_types_in_union_265564:
                # Runtime conditional SSA (line 27)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 28)
            # Processing the call arguments (line 28)
            str_265566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', "Values cannot be given for both 'nyq' and 'fs'.")
            # Processing the call keyword arguments (line 28)
            kwargs_265567 = {}
            # Getting the type of 'ValueError' (line 28)
            ValueError_265565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 28)
            ValueError_call_result_265568 = invoke(stypy.reporting.localization.Localization(__file__, 28, 18), ValueError_265565, *[str_265566], **kwargs_265567)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 28, 12), ValueError_call_result_265568, 'raise parameter', BaseException)

            if more_types_in_union_265564:
                # SSA join for if statement (line 27)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 29):
        
        # Assigning a BinOp to a Name (line 29):
        int_265569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'int')
        # Getting the type of 'nyq' (line 29)
        nyq_265570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'nyq')
        # Applying the binary operator '*' (line 29)
        result_mul_265571 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), '*', int_265569, nyq_265570)
        
        # Assigning a type to the variable 'fs' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'fs', result_mul_265571)

        if more_types_in_union_265560:
            # SSA join for if statement (line 26)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'fs' (line 30)
    fs_265572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'fs')
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', fs_265572)
    
    # ################# End of '_get_fs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_fs' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_265573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_265573)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_fs'
    return stypy_return_type_265573

# Assigning a type to the variable '_get_fs' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_get_fs', _get_fs)

@norecursion
def kaiser_beta(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kaiser_beta'
    module_type_store = module_type_store.open_function_context('kaiser_beta', 48, 0, False)
    
    # Passed parameters checking function
    kaiser_beta.stypy_localization = localization
    kaiser_beta.stypy_type_of_self = None
    kaiser_beta.stypy_type_store = module_type_store
    kaiser_beta.stypy_function_name = 'kaiser_beta'
    kaiser_beta.stypy_param_names_list = ['a']
    kaiser_beta.stypy_varargs_param_name = None
    kaiser_beta.stypy_kwargs_param_name = None
    kaiser_beta.stypy_call_defaults = defaults
    kaiser_beta.stypy_call_varargs = varargs
    kaiser_beta.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kaiser_beta', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kaiser_beta', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kaiser_beta(...)' code ##################

    str_265574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', 'Compute the Kaiser parameter `beta`, given the attenuation `a`.\n\n    Parameters\n    ----------\n    a : float\n        The desired attenuation in the stopband and maximum ripple in\n        the passband, in dB.  This should be a *positive* number.\n\n    Returns\n    -------\n    beta : float\n        The `beta` parameter to be used in the formula for a Kaiser window.\n\n    References\n    ----------\n    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.\n\n    Examples\n    --------\n    Suppose we want to design a lowpass filter, with 65 dB attenuation\n    in the stop band.  The Kaiser window parameter to be used in the\n    window method is computed by `kaiser_beta(65)`:\n\n    >>> from scipy.signal import kaiser_beta\n    >>> kaiser_beta(65)\n    6.20426\n\n    ')
    
    
    # Getting the type of 'a' (line 77)
    a_265575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'a')
    int_265576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 11), 'int')
    # Applying the binary operator '>' (line 77)
    result_gt_265577 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), '>', a_265575, int_265576)
    
    # Testing the type of an if condition (line 77)
    if_condition_265578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_gt_265577)
    # Assigning a type to the variable 'if_condition_265578' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_265578', if_condition_265578)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 78):
    
    # Assigning a BinOp to a Name (line 78):
    float_265579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'float')
    # Getting the type of 'a' (line 78)
    a_265580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'a')
    float_265581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'float')
    # Applying the binary operator '-' (line 78)
    result_sub_265582 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 25), '-', a_265580, float_265581)
    
    # Applying the binary operator '*' (line 78)
    result_mul_265583 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '*', float_265579, result_sub_265582)
    
    # Assigning a type to the variable 'beta' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'beta', result_mul_265583)
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'a' (line 79)
    a_265584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'a')
    int_265585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'int')
    # Applying the binary operator '>' (line 79)
    result_gt_265586 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), '>', a_265584, int_265585)
    
    # Testing the type of an if condition (line 79)
    if_condition_265587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 9), result_gt_265586)
    # Assigning a type to the variable 'if_condition_265587' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'if_condition_265587', if_condition_265587)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 80):
    
    # Assigning a BinOp to a Name (line 80):
    float_265588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 15), 'float')
    # Getting the type of 'a' (line 80)
    a_265589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'a')
    int_265590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
    # Applying the binary operator '-' (line 80)
    result_sub_265591 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 25), '-', a_265589, int_265590)
    
    float_265592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'float')
    # Applying the binary operator '**' (line 80)
    result_pow_265593 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 24), '**', result_sub_265591, float_265592)
    
    # Applying the binary operator '*' (line 80)
    result_mul_265594 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '*', float_265588, result_pow_265593)
    
    float_265595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 42), 'float')
    # Getting the type of 'a' (line 80)
    a_265596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 53), 'a')
    int_265597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'int')
    # Applying the binary operator '-' (line 80)
    result_sub_265598 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 53), '-', a_265596, int_265597)
    
    # Applying the binary operator '*' (line 80)
    result_mul_265599 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 42), '*', float_265595, result_sub_265598)
    
    # Applying the binary operator '+' (line 80)
    result_add_265600 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '+', result_mul_265594, result_mul_265599)
    
    # Assigning a type to the variable 'beta' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'beta', result_add_265600)
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 82):
    
    # Assigning a Num to a Name (line 82):
    float_265601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'float')
    # Assigning a type to the variable 'beta' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'beta', float_265601)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'beta' (line 83)
    beta_265602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'beta')
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', beta_265602)
    
    # ################# End of 'kaiser_beta(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kaiser_beta' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_265603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_265603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kaiser_beta'
    return stypy_return_type_265603

# Assigning a type to the variable 'kaiser_beta' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'kaiser_beta', kaiser_beta)

@norecursion
def kaiser_atten(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kaiser_atten'
    module_type_store = module_type_store.open_function_context('kaiser_atten', 86, 0, False)
    
    # Passed parameters checking function
    kaiser_atten.stypy_localization = localization
    kaiser_atten.stypy_type_of_self = None
    kaiser_atten.stypy_type_store = module_type_store
    kaiser_atten.stypy_function_name = 'kaiser_atten'
    kaiser_atten.stypy_param_names_list = ['numtaps', 'width']
    kaiser_atten.stypy_varargs_param_name = None
    kaiser_atten.stypy_kwargs_param_name = None
    kaiser_atten.stypy_call_defaults = defaults
    kaiser_atten.stypy_call_varargs = varargs
    kaiser_atten.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kaiser_atten', ['numtaps', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kaiser_atten', localization, ['numtaps', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kaiser_atten(...)' code ##################

    str_265604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', "Compute the attenuation of a Kaiser FIR filter.\n\n    Given the number of taps `N` and the transition width `width`, compute the\n    attenuation `a` in dB, given by Kaiser's formula:\n\n        a = 2.285 * (N - 1) * pi * width + 7.95\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter.\n    width : float\n        The desired width of the transition region between passband and\n        stopband (or, in general, at any discontinuity) for the filter,\n        expressed as a fraction of the Nyquist frequency.\n\n    Returns\n    -------\n    a : float\n        The attenuation of the ripple, in dB.\n\n    See Also\n    --------\n    kaiserord, kaiser_beta\n\n    Examples\n    --------\n    Suppose we want to design a FIR filter using the Kaiser window method\n    that will have 211 taps and a transition width of 9 Hz for a signal that\n    is sampled at 480 Hz.  Expressed as a fraction of the Nyquist frequency,\n    the width is 9/(0.5*480) = 0.0375.  The approximate attenuation (in dB)\n    is computed as follows:\n\n    >>> from scipy.signal import kaiser_atten\n    >>> kaiser_atten(211, 0.0375)\n    64.48099630593983\n\n    ")
    
    # Assigning a BinOp to a Name (line 125):
    
    # Assigning a BinOp to a Name (line 125):
    float_265605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'float')
    # Getting the type of 'numtaps' (line 125)
    numtaps_265606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'numtaps')
    int_265607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 27), 'int')
    # Applying the binary operator '-' (line 125)
    result_sub_265608 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 17), '-', numtaps_265606, int_265607)
    
    # Applying the binary operator '*' (line 125)
    result_mul_265609 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 8), '*', float_265605, result_sub_265608)
    
    # Getting the type of 'np' (line 125)
    np_265610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'np')
    # Obtaining the member 'pi' of a type (line 125)
    pi_265611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 32), np_265610, 'pi')
    # Applying the binary operator '*' (line 125)
    result_mul_265612 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 30), '*', result_mul_265609, pi_265611)
    
    # Getting the type of 'width' (line 125)
    width_265613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'width')
    # Applying the binary operator '*' (line 125)
    result_mul_265614 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 38), '*', result_mul_265612, width_265613)
    
    float_265615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 48), 'float')
    # Applying the binary operator '+' (line 125)
    result_add_265616 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 8), '+', result_mul_265614, float_265615)
    
    # Assigning a type to the variable 'a' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'a', result_add_265616)
    # Getting the type of 'a' (line 126)
    a_265617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type', a_265617)
    
    # ################# End of 'kaiser_atten(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kaiser_atten' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_265618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_265618)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kaiser_atten'
    return stypy_return_type_265618

# Assigning a type to the variable 'kaiser_atten' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'kaiser_atten', kaiser_atten)

@norecursion
def kaiserord(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kaiserord'
    module_type_store = module_type_store.open_function_context('kaiserord', 129, 0, False)
    
    # Passed parameters checking function
    kaiserord.stypy_localization = localization
    kaiserord.stypy_type_of_self = None
    kaiserord.stypy_type_store = module_type_store
    kaiserord.stypy_function_name = 'kaiserord'
    kaiserord.stypy_param_names_list = ['ripple', 'width']
    kaiserord.stypy_varargs_param_name = None
    kaiserord.stypy_kwargs_param_name = None
    kaiserord.stypy_call_defaults = defaults
    kaiserord.stypy_call_varargs = varargs
    kaiserord.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kaiserord', ['ripple', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kaiserord', localization, ['ripple', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kaiserord(...)' code ##################

    str_265619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, (-1)), 'str', '\n    Determine the filter window parameters for the Kaiser window method.\n\n    The parameters returned by this function are generally used to create\n    a finite impulse response filter using the window method, with either\n    `firwin` or `firwin2`.\n\n    Parameters\n    ----------\n    ripple : float\n        Upper bound for the deviation (in dB) of the magnitude of the\n        filter\'s frequency response from that of the desired filter (not\n        including frequencies in any transition intervals).  That is, if w\n        is the frequency expressed as a fraction of the Nyquist frequency,\n        A(w) is the actual frequency response of the filter and D(w) is the\n        desired frequency response, the design requirement is that::\n\n            abs(A(w) - D(w))) < 10**(-ripple/20)\n\n        for 0 <= w <= 1 and w not in a transition interval.\n    width : float\n        Width of transition region, normalized so that 1 corresponds to pi\n        radians / sample.  That is, the frequency is expressed as a fraction\n        of the Nyquist frequency.\n\n    Returns\n    -------\n    numtaps : int\n        The length of the Kaiser window.\n    beta : float\n        The beta parameter for the Kaiser window.\n\n    See Also\n    --------\n    kaiser_beta, kaiser_atten\n\n    Notes\n    -----\n    There are several ways to obtain the Kaiser window:\n\n    - ``signal.kaiser(numtaps, beta, sym=True)``\n    - ``signal.get_window(beta, numtaps)``\n    - ``signal.get_window((\'kaiser\', beta), numtaps)``\n\n    The empirical equations discovered by Kaiser are used.\n\n    References\n    ----------\n    Oppenheim, Schafer, "Discrete-Time Signal Processing", p.475-476.\n\n    Examples\n    --------\n    We will use the Kaiser window method to design a lowpass FIR filter\n    for a signal that is sampled at 1000 Hz.\n\n    We want at least 65 dB rejection in the stop band, and in the pass\n    band the gain should vary no more than 0.5%.\n\n    We want a cutoff frequency of 175 Hz, with a transition between the\n    pass band and the stop band of 24 Hz.  That is, in the band [0, 163],\n    the gain varies no more than 0.5%, and in the band [187, 500], the\n    signal is attenuated by at least 65 dB.\n\n    >>> from scipy.signal import kaiserord, firwin, freqz\n    >>> import matplotlib.pyplot as plt\n    >>> fs = 1000.0\n    >>> cutoff = 175\n    >>> width = 24\n\n    The Kaiser method accepts just a single parameter to control the pass\n    band ripple and the stop band rejection, so we use the more restrictive\n    of the two.  In this case, the pass band ripple is 0.005, or 46.02 dB,\n    so we will use 65 dB as the design parameter.\n\n    Use `kaiserord` to determine the length of the filter and the\n    parameter for the Kaiser window.\n\n    >>> numtaps, beta = kaiserord(65, width/(0.5*fs))\n    >>> numtaps\n    167\n    >>> beta\n    6.20426\n\n    Use `firwin` to create the FIR filter.\n\n    >>> taps = firwin(numtaps, cutoff, window=(\'kaiser\', beta),\n    ...               scale=False, nyq=0.5*fs)\n\n    Compute the frequency response of the filter.  ``w`` is the array of\n    frequencies, and ``h`` is the corresponding complex array of frequency\n    responses.\n\n    >>> w, h = freqz(taps, worN=8000)\n    >>> w *= 0.5*fs/np.pi  # Convert w to Hz.\n\n    Compute the deviation of the magnitude of the filter\'s response from\n    that of the ideal lowpass filter.  Values in the transition region are\n    set to ``nan``, so they won\'t appear in the plot.\n\n    >>> ideal = w < cutoff  # The "ideal" frequency response.\n    >>> deviation = np.abs(np.abs(h) - ideal)\n    >>> deviation[(w > cutoff - 0.5*width) & (w < cutoff + 0.5*width)] = np.nan\n\n    Plot the deviation.  A close look at the left end of the stop band shows\n    that the requirement for 65 dB attenuation is violated in the first lobe\n    by about 0.125 dB.  This is not unusual for the Kaiser window method.\n\n    >>> plt.plot(w, 20*np.log10(np.abs(deviation)))\n    >>> plt.xlim(0, 0.5*fs)\n    >>> plt.ylim(-90, -60)\n    >>> plt.grid(alpha=0.25)\n    >>> plt.axhline(-65, color=\'r\', ls=\'--\', alpha=0.3)\n    >>> plt.xlabel(\'Frequency (Hz)\')\n    >>> plt.ylabel(\'Deviation from ideal (dB)\')\n    >>> plt.title(\'Lowpass Filter Frequency Response\')\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 248):
    
    # Assigning a Call to a Name (line 248):
    
    # Call to abs(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'ripple' (line 248)
    ripple_265621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'ripple', False)
    # Processing the call keyword arguments (line 248)
    kwargs_265622 = {}
    # Getting the type of 'abs' (line 248)
    abs_265620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'abs', False)
    # Calling abs(args, kwargs) (line 248)
    abs_call_result_265623 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), abs_265620, *[ripple_265621], **kwargs_265622)
    
    # Assigning a type to the variable 'A' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'A', abs_call_result_265623)
    
    
    # Getting the type of 'A' (line 249)
    A_265624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 7), 'A')
    int_265625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 11), 'int')
    # Applying the binary operator '<' (line 249)
    result_lt_265626 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 7), '<', A_265624, int_265625)
    
    # Testing the type of an if condition (line 249)
    if_condition_265627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), result_lt_265626)
    # Assigning a type to the variable 'if_condition_265627' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_265627', if_condition_265627)
    # SSA begins for if statement (line 249)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 251)
    # Processing the call arguments (line 251)
    str_265629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'str', 'Requested maximum ripple attentuation %f is too small for the Kaiser formula.')
    # Getting the type of 'A' (line 252)
    A_265630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 59), 'A', False)
    # Applying the binary operator '%' (line 251)
    result_mod_265631 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 25), '%', str_265629, A_265630)
    
    # Processing the call keyword arguments (line 251)
    kwargs_265632 = {}
    # Getting the type of 'ValueError' (line 251)
    ValueError_265628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 251)
    ValueError_call_result_265633 = invoke(stypy.reporting.localization.Localization(__file__, 251, 14), ValueError_265628, *[result_mod_265631], **kwargs_265632)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 251, 8), ValueError_call_result_265633, 'raise parameter', BaseException)
    # SSA join for if statement (line 249)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to kaiser_beta(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'A' (line 253)
    A_265635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'A', False)
    # Processing the call keyword arguments (line 253)
    kwargs_265636 = {}
    # Getting the type of 'kaiser_beta' (line 253)
    kaiser_beta_265634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'kaiser_beta', False)
    # Calling kaiser_beta(args, kwargs) (line 253)
    kaiser_beta_call_result_265637 = invoke(stypy.reporting.localization.Localization(__file__, 253, 11), kaiser_beta_265634, *[A_265635], **kwargs_265636)
    
    # Assigning a type to the variable 'beta' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'beta', kaiser_beta_call_result_265637)
    
    # Assigning a BinOp to a Name (line 257):
    
    # Assigning a BinOp to a Name (line 257):
    # Getting the type of 'A' (line 257)
    A_265638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'A')
    float_265639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 19), 'float')
    # Applying the binary operator '-' (line 257)
    result_sub_265640 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), '-', A_265638, float_265639)
    
    float_265641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 27), 'float')
    # Applying the binary operator 'div' (line 257)
    result_div_265642 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 14), 'div', result_sub_265640, float_265641)
    
    # Getting the type of 'np' (line 257)
    np_265643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 36), 'np')
    # Obtaining the member 'pi' of a type (line 257)
    pi_265644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 36), np_265643, 'pi')
    # Getting the type of 'width' (line 257)
    width_265645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 44), 'width')
    # Applying the binary operator '*' (line 257)
    result_mul_265646 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 36), '*', pi_265644, width_265645)
    
    # Applying the binary operator 'div' (line 257)
    result_div_265647 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 33), 'div', result_div_265642, result_mul_265646)
    
    int_265648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 53), 'int')
    # Applying the binary operator '+' (line 257)
    result_add_265649 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 14), '+', result_div_265647, int_265648)
    
    # Assigning a type to the variable 'numtaps' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'numtaps', result_add_265649)
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_265650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    
    # Call to int(...): (line 259)
    # Processing the call arguments (line 259)
    
    # Call to ceil(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'numtaps' (line 259)
    numtaps_265653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'numtaps', False)
    # Processing the call keyword arguments (line 259)
    kwargs_265654 = {}
    # Getting the type of 'ceil' (line 259)
    ceil_265652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'ceil', False)
    # Calling ceil(args, kwargs) (line 259)
    ceil_call_result_265655 = invoke(stypy.reporting.localization.Localization(__file__, 259, 15), ceil_265652, *[numtaps_265653], **kwargs_265654)
    
    # Processing the call keyword arguments (line 259)
    kwargs_265656 = {}
    # Getting the type of 'int' (line 259)
    int_265651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'int', False)
    # Calling int(args, kwargs) (line 259)
    int_call_result_265657 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), int_265651, *[ceil_call_result_265655], **kwargs_265656)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), tuple_265650, int_call_result_265657)
    # Adding element type (line 259)
    # Getting the type of 'beta' (line 259)
    beta_265658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'beta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 11), tuple_265650, beta_265658)
    
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', tuple_265650)
    
    # ################# End of 'kaiserord(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kaiserord' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_265659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_265659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kaiserord'
    return stypy_return_type_265659

# Assigning a type to the variable 'kaiserord' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'kaiserord', kaiserord)

@norecursion
def firwin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 262)
    None_265660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 34), 'None')
    str_265661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 47), 'str', 'hamming')
    # Getting the type of 'True' (line 262)
    True_265662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 68), 'True')
    # Getting the type of 'True' (line 263)
    True_265663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'True')
    # Getting the type of 'None' (line 263)
    None_265664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'None')
    # Getting the type of 'None' (line 263)
    None_265665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 36), 'None')
    defaults = [None_265660, str_265661, True_265662, True_265663, None_265664, None_265665]
    # Create a new context for function 'firwin'
    module_type_store = module_type_store.open_function_context('firwin', 262, 0, False)
    
    # Passed parameters checking function
    firwin.stypy_localization = localization
    firwin.stypy_type_of_self = None
    firwin.stypy_type_store = module_type_store
    firwin.stypy_function_name = 'firwin'
    firwin.stypy_param_names_list = ['numtaps', 'cutoff', 'width', 'window', 'pass_zero', 'scale', 'nyq', 'fs']
    firwin.stypy_varargs_param_name = None
    firwin.stypy_kwargs_param_name = None
    firwin.stypy_call_defaults = defaults
    firwin.stypy_call_varargs = varargs
    firwin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'firwin', ['numtaps', 'cutoff', 'width', 'window', 'pass_zero', 'scale', 'nyq', 'fs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'firwin', localization, ['numtaps', 'cutoff', 'width', 'window', 'pass_zero', 'scale', 'nyq', 'fs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'firwin(...)' code ##################

    str_265666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, (-1)), 'str', '\n    FIR filter design using the window method.\n\n    This function computes the coefficients of a finite impulse response\n    filter.  The filter will have linear phase; it will be Type I if\n    `numtaps` is odd and Type II if `numtaps` is even.\n\n    Type II filters always have zero response at the Nyquist frequency, so a\n    ValueError exception is raised if firwin is called with `numtaps` even and\n    having a passband whose right end is at the Nyquist frequency.\n\n    Parameters\n    ----------\n    numtaps : int\n        Length of the filter (number of coefficients, i.e. the filter\n        order + 1).  `numtaps` must be even if a passband includes the\n        Nyquist frequency.\n    cutoff : float or 1D array_like\n        Cutoff frequency of filter (expressed in the same units as `nyq`)\n        OR an array of cutoff frequencies (that is, band edges). In the\n        latter case, the frequencies in `cutoff` should be positive and\n        monotonically increasing between 0 and `nyq`.  The values 0 and\n        `nyq` must not be included in `cutoff`.\n    width : float or None, optional\n        If `width` is not None, then assume it is the approximate width\n        of the transition region (expressed in the same units as `nyq`)\n        for use in Kaiser FIR filter design.  In this case, the `window`\n        argument is ignored.\n    window : string or tuple of string and parameter values, optional\n        Desired window to use. See `scipy.signal.get_window` for a list\n        of windows and required parameters.\n    pass_zero : bool, optional\n        If True, the gain at the frequency 0 (i.e. the "DC gain") is 1.\n        Otherwise the DC gain is 0.\n    scale : bool, optional\n        Set to True to scale the coefficients so that the frequency\n        response is exactly unity at a certain frequency.\n        That frequency is either:\n\n        - 0 (DC) if the first passband starts at 0 (i.e. pass_zero\n          is True)\n        - `nyq` (the Nyquist frequency) if the first passband ends at\n          `nyq` (i.e the filter is a single band highpass filter);\n          center of first passband otherwise\n\n    nyq : float, optional\n        *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.\n        Each frequency in `cutoff` must be between 0 and `nyq`. Default\n        is 1.\n    fs : float, optional\n        The sampling frequency of the signal.  Each frequency in `cutoff`\n        must be between 0 and ``fs/2``.  Default is 2.\n\n    Returns\n    -------\n    h : (numtaps,) ndarray\n        Coefficients of length `numtaps` FIR filter.\n\n    Raises\n    ------\n    ValueError\n        If any value in `cutoff` is less than or equal to 0 or greater\n        than or equal to ``fs/2``, if the values in `cutoff` are not strictly\n        monotonically increasing, or if `numtaps` is even but a passband\n        includes the Nyquist frequency.\n\n    See Also\n    --------\n    firwin2\n    firls\n    minimum_phase\n    remez\n\n    Examples\n    --------\n    Low-pass from 0 to f:\n\n    >>> from scipy import signal\n    >>> numtaps = 3\n    >>> f = 0.1\n    >>> signal.firwin(numtaps, f)\n    array([ 0.06799017,  0.86401967,  0.06799017])\n\n    Use a specific window function:\n\n    >>> signal.firwin(numtaps, f, window=\'nuttall\')\n    array([  3.56607041e-04,   9.99286786e-01,   3.56607041e-04])\n\n    High-pass (\'stop\' from 0 to f):\n\n    >>> signal.firwin(numtaps, f, pass_zero=False)\n    array([-0.00859313,  0.98281375, -0.00859313])\n\n    Band-pass:\n\n    >>> f1, f2 = 0.1, 0.2\n    >>> signal.firwin(numtaps, [f1, f2], pass_zero=False)\n    array([ 0.06301614,  0.88770441,  0.06301614])\n\n    Band-stop:\n\n    >>> signal.firwin(numtaps, [f1, f2])\n    array([-0.00801395,  1.0160279 , -0.00801395])\n\n    Multi-band (passbands are [0, f1], [f2, f3] and [f4, 1]):\n\n    >>> f3, f4 = 0.3, 0.4\n    >>> signal.firwin(numtaps, [f1, f2, f3, f4])\n    array([-0.01376344,  1.02752689, -0.01376344])\n\n    Multi-band (passbands are [f1, f2] and [f3,f4]):\n\n    >>> signal.firwin(numtaps, [f1, f2, f3, f4], pass_zero=False)\n    array([ 0.04890915,  0.91284326,  0.04890915])\n\n    ')
    
    # Assigning a BinOp to a Name (line 383):
    
    # Assigning a BinOp to a Name (line 383):
    float_265667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 10), 'float')
    
    # Call to _get_fs(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'fs' (line 383)
    fs_265669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 24), 'fs', False)
    # Getting the type of 'nyq' (line 383)
    nyq_265670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 28), 'nyq', False)
    # Processing the call keyword arguments (line 383)
    kwargs_265671 = {}
    # Getting the type of '_get_fs' (line 383)
    _get_fs_265668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), '_get_fs', False)
    # Calling _get_fs(args, kwargs) (line 383)
    _get_fs_call_result_265672 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), _get_fs_265668, *[fs_265669, nyq_265670], **kwargs_265671)
    
    # Applying the binary operator '*' (line 383)
    result_mul_265673 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 10), '*', float_265667, _get_fs_call_result_265672)
    
    # Assigning a type to the variable 'nyq' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'nyq', result_mul_265673)
    
    # Assigning a BinOp to a Name (line 385):
    
    # Assigning a BinOp to a Name (line 385):
    
    # Call to atleast_1d(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'cutoff' (line 385)
    cutoff_265676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 27), 'cutoff', False)
    # Processing the call keyword arguments (line 385)
    kwargs_265677 = {}
    # Getting the type of 'np' (line 385)
    np_265674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 385)
    atleast_1d_265675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 13), np_265674, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 385)
    atleast_1d_call_result_265678 = invoke(stypy.reporting.localization.Localization(__file__, 385, 13), atleast_1d_265675, *[cutoff_265676], **kwargs_265677)
    
    
    # Call to float(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'nyq' (line 385)
    nyq_265680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 43), 'nyq', False)
    # Processing the call keyword arguments (line 385)
    kwargs_265681 = {}
    # Getting the type of 'float' (line 385)
    float_265679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 37), 'float', False)
    # Calling float(args, kwargs) (line 385)
    float_call_result_265682 = invoke(stypy.reporting.localization.Localization(__file__, 385, 37), float_265679, *[nyq_265680], **kwargs_265681)
    
    # Applying the binary operator 'div' (line 385)
    result_div_265683 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 13), 'div', atleast_1d_call_result_265678, float_call_result_265682)
    
    # Assigning a type to the variable 'cutoff' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'cutoff', result_div_265683)
    
    
    # Getting the type of 'cutoff' (line 388)
    cutoff_265684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 7), 'cutoff')
    # Obtaining the member 'ndim' of a type (line 388)
    ndim_265685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 7), cutoff_265684, 'ndim')
    int_265686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 21), 'int')
    # Applying the binary operator '>' (line 388)
    result_gt_265687 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 7), '>', ndim_265685, int_265686)
    
    # Testing the type of an if condition (line 388)
    if_condition_265688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 4), result_gt_265687)
    # Assigning a type to the variable 'if_condition_265688' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'if_condition_265688', if_condition_265688)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 389)
    # Processing the call arguments (line 389)
    str_265690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 25), 'str', 'The cutoff argument must be at most one-dimensional.')
    # Processing the call keyword arguments (line 389)
    kwargs_265691 = {}
    # Getting the type of 'ValueError' (line 389)
    ValueError_265689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 389)
    ValueError_call_result_265692 = invoke(stypy.reporting.localization.Localization(__file__, 389, 14), ValueError_265689, *[str_265690], **kwargs_265691)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 389, 8), ValueError_call_result_265692, 'raise parameter', BaseException)
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cutoff' (line 391)
    cutoff_265693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'cutoff')
    # Obtaining the member 'size' of a type (line 391)
    size_265694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 7), cutoff_265693, 'size')
    int_265695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 22), 'int')
    # Applying the binary operator '==' (line 391)
    result_eq_265696 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), '==', size_265694, int_265695)
    
    # Testing the type of an if condition (line 391)
    if_condition_265697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), result_eq_265696)
    # Assigning a type to the variable 'if_condition_265697' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_265697', if_condition_265697)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 392)
    # Processing the call arguments (line 392)
    str_265699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 25), 'str', 'At least one cutoff frequency must be given.')
    # Processing the call keyword arguments (line 392)
    kwargs_265700 = {}
    # Getting the type of 'ValueError' (line 392)
    ValueError_265698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 392)
    ValueError_call_result_265701 = invoke(stypy.reporting.localization.Localization(__file__, 392, 14), ValueError_265698, *[str_265699], **kwargs_265700)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 392, 8), ValueError_call_result_265701, 'raise parameter', BaseException)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to min(...): (line 393)
    # Processing the call keyword arguments (line 393)
    kwargs_265704 = {}
    # Getting the type of 'cutoff' (line 393)
    cutoff_265702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 7), 'cutoff', False)
    # Obtaining the member 'min' of a type (line 393)
    min_265703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 7), cutoff_265702, 'min')
    # Calling min(args, kwargs) (line 393)
    min_call_result_265705 = invoke(stypy.reporting.localization.Localization(__file__, 393, 7), min_265703, *[], **kwargs_265704)
    
    int_265706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 23), 'int')
    # Applying the binary operator '<=' (line 393)
    result_le_265707 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 7), '<=', min_call_result_265705, int_265706)
    
    
    
    # Call to max(...): (line 393)
    # Processing the call keyword arguments (line 393)
    kwargs_265710 = {}
    # Getting the type of 'cutoff' (line 393)
    cutoff_265708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 28), 'cutoff', False)
    # Obtaining the member 'max' of a type (line 393)
    max_265709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 28), cutoff_265708, 'max')
    # Calling max(args, kwargs) (line 393)
    max_call_result_265711 = invoke(stypy.reporting.localization.Localization(__file__, 393, 28), max_265709, *[], **kwargs_265710)
    
    int_265712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 44), 'int')
    # Applying the binary operator '>=' (line 393)
    result_ge_265713 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 28), '>=', max_call_result_265711, int_265712)
    
    # Applying the binary operator 'or' (line 393)
    result_or_keyword_265714 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 7), 'or', result_le_265707, result_ge_265713)
    
    # Testing the type of an if condition (line 393)
    if_condition_265715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), result_or_keyword_265714)
    # Assigning a type to the variable 'if_condition_265715' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_265715', if_condition_265715)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 394)
    # Processing the call arguments (line 394)
    str_265717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 25), 'str', 'Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2.')
    # Processing the call keyword arguments (line 394)
    kwargs_265718 = {}
    # Getting the type of 'ValueError' (line 394)
    ValueError_265716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 394)
    ValueError_call_result_265719 = invoke(stypy.reporting.localization.Localization(__file__, 394, 14), ValueError_265716, *[str_265717], **kwargs_265718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 394, 8), ValueError_call_result_265719, 'raise parameter', BaseException)
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 396)
    # Processing the call arguments (line 396)
    
    
    # Call to diff(...): (line 396)
    # Processing the call arguments (line 396)
    # Getting the type of 'cutoff' (line 396)
    cutoff_265724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'cutoff', False)
    # Processing the call keyword arguments (line 396)
    kwargs_265725 = {}
    # Getting the type of 'np' (line 396)
    np_265722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 14), 'np', False)
    # Obtaining the member 'diff' of a type (line 396)
    diff_265723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 14), np_265722, 'diff')
    # Calling diff(args, kwargs) (line 396)
    diff_call_result_265726 = invoke(stypy.reporting.localization.Localization(__file__, 396, 14), diff_265723, *[cutoff_265724], **kwargs_265725)
    
    int_265727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 33), 'int')
    # Applying the binary operator '<=' (line 396)
    result_le_265728 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 14), '<=', diff_call_result_265726, int_265727)
    
    # Processing the call keyword arguments (line 396)
    kwargs_265729 = {}
    # Getting the type of 'np' (line 396)
    np_265720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 396)
    any_265721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 7), np_265720, 'any')
    # Calling any(args, kwargs) (line 396)
    any_call_result_265730 = invoke(stypy.reporting.localization.Localization(__file__, 396, 7), any_265721, *[result_le_265728], **kwargs_265729)
    
    # Testing the type of an if condition (line 396)
    if_condition_265731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 4), any_call_result_265730)
    # Assigning a type to the variable 'if_condition_265731' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'if_condition_265731', if_condition_265731)
    # SSA begins for if statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 397)
    # Processing the call arguments (line 397)
    str_265733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 25), 'str', 'Invalid cutoff frequencies: the frequencies must be strictly increasing.')
    # Processing the call keyword arguments (line 397)
    kwargs_265734 = {}
    # Getting the type of 'ValueError' (line 397)
    ValueError_265732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 397)
    ValueError_call_result_265735 = invoke(stypy.reporting.localization.Localization(__file__, 397, 14), ValueError_265732, *[str_265733], **kwargs_265734)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 397, 8), ValueError_call_result_265735, 'raise parameter', BaseException)
    # SSA join for if statement (line 396)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 400)
    # Getting the type of 'width' (line 400)
    width_265736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'width')
    # Getting the type of 'None' (line 400)
    None_265737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'None')
    
    (may_be_265738, more_types_in_union_265739) = may_not_be_none(width_265736, None_265737)

    if may_be_265738:

        if more_types_in_union_265739:
            # Runtime conditional SSA (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 403):
        
        # Assigning a Call to a Name (line 403):
        
        # Call to kaiser_atten(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'numtaps' (line 403)
        numtaps_265741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 29), 'numtaps', False)
        
        # Call to float(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'width' (line 403)
        width_265743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 44), 'width', False)
        # Processing the call keyword arguments (line 403)
        kwargs_265744 = {}
        # Getting the type of 'float' (line 403)
        float_265742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 38), 'float', False)
        # Calling float(args, kwargs) (line 403)
        float_call_result_265745 = invoke(stypy.reporting.localization.Localization(__file__, 403, 38), float_265742, *[width_265743], **kwargs_265744)
        
        # Getting the type of 'nyq' (line 403)
        nyq_265746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 53), 'nyq', False)
        # Applying the binary operator 'div' (line 403)
        result_div_265747 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 38), 'div', float_call_result_265745, nyq_265746)
        
        # Processing the call keyword arguments (line 403)
        kwargs_265748 = {}
        # Getting the type of 'kaiser_atten' (line 403)
        kaiser_atten_265740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 16), 'kaiser_atten', False)
        # Calling kaiser_atten(args, kwargs) (line 403)
        kaiser_atten_call_result_265749 = invoke(stypy.reporting.localization.Localization(__file__, 403, 16), kaiser_atten_265740, *[numtaps_265741, result_div_265747], **kwargs_265748)
        
        # Assigning a type to the variable 'atten' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'atten', kaiser_atten_call_result_265749)
        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to kaiser_beta(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'atten' (line 404)
        atten_265751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 27), 'atten', False)
        # Processing the call keyword arguments (line 404)
        kwargs_265752 = {}
        # Getting the type of 'kaiser_beta' (line 404)
        kaiser_beta_265750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'kaiser_beta', False)
        # Calling kaiser_beta(args, kwargs) (line 404)
        kaiser_beta_call_result_265753 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), kaiser_beta_265750, *[atten_265751], **kwargs_265752)
        
        # Assigning a type to the variable 'beta' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'beta', kaiser_beta_call_result_265753)
        
        # Assigning a Tuple to a Name (line 405):
        
        # Assigning a Tuple to a Name (line 405):
        
        # Obtaining an instance of the builtin type 'tuple' (line 405)
        tuple_265754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 405)
        # Adding element type (line 405)
        str_265755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 18), tuple_265754, str_265755)
        # Adding element type (line 405)
        # Getting the type of 'beta' (line 405)
        beta_265756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 28), 'beta')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 18), tuple_265754, beta_265756)
        
        # Assigning a type to the variable 'window' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'window', tuple_265754)

        if more_types_in_union_265739:
            # SSA join for if statement (line 400)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 407):
    
    # Assigning a BinOp to a Name (line 407):
    
    # Call to bool(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'cutoff' (line 407)
    cutoff_265758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 24), 'cutoff', False)
    # Obtaining the member 'size' of a type (line 407)
    size_265759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 24), cutoff_265758, 'size')
    int_265760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 38), 'int')
    # Applying the binary operator '&' (line 407)
    result_and__265761 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 24), '&', size_265759, int_265760)
    
    # Processing the call keyword arguments (line 407)
    kwargs_265762 = {}
    # Getting the type of 'bool' (line 407)
    bool_265757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 19), 'bool', False)
    # Calling bool(args, kwargs) (line 407)
    bool_call_result_265763 = invoke(stypy.reporting.localization.Localization(__file__, 407, 19), bool_265757, *[result_and__265761], **kwargs_265762)
    
    # Getting the type of 'pass_zero' (line 407)
    pass_zero_265764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 43), 'pass_zero')
    # Applying the binary operator '^' (line 407)
    result_xor_265765 = python_operator(stypy.reporting.localization.Localization(__file__, 407, 19), '^', bool_call_result_265763, pass_zero_265764)
    
    # Assigning a type to the variable 'pass_nyquist' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'pass_nyquist', result_xor_265765)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'pass_nyquist' (line 408)
    pass_nyquist_265766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 7), 'pass_nyquist')
    
    # Getting the type of 'numtaps' (line 408)
    numtaps_265767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'numtaps')
    int_265768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 34), 'int')
    # Applying the binary operator '%' (line 408)
    result_mod_265769 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 24), '%', numtaps_265767, int_265768)
    
    int_265770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 39), 'int')
    # Applying the binary operator '==' (line 408)
    result_eq_265771 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 24), '==', result_mod_265769, int_265770)
    
    # Applying the binary operator 'and' (line 408)
    result_and_keyword_265772 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 7), 'and', pass_nyquist_265766, result_eq_265771)
    
    # Testing the type of an if condition (line 408)
    if_condition_265773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 4), result_and_keyword_265772)
    # Assigning a type to the variable 'if_condition_265773' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'if_condition_265773', if_condition_265773)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 409)
    # Processing the call arguments (line 409)
    str_265775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 25), 'str', 'A filter with an even number of coefficients must have zero response at the Nyquist frequency.')
    # Processing the call keyword arguments (line 409)
    kwargs_265776 = {}
    # Getting the type of 'ValueError' (line 409)
    ValueError_265774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 409)
    ValueError_call_result_265777 = invoke(stypy.reporting.localization.Localization(__file__, 409, 14), ValueError_265774, *[str_265775], **kwargs_265776)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 409, 8), ValueError_call_result_265777, 'raise parameter', BaseException)
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 414):
    
    # Assigning a Call to a Name (line 414):
    
    # Call to hstack(...): (line 414)
    # Processing the call arguments (line 414)
    
    # Obtaining an instance of the builtin type 'tuple' (line 414)
    tuple_265780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 414)
    # Adding element type (line 414)
    
    # Obtaining an instance of the builtin type 'list' (line 414)
    list_265781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 414)
    # Adding element type (line 414)
    float_265782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 24), list_265781, float_265782)
    
    # Getting the type of 'pass_zero' (line 414)
    pass_zero_265783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 32), 'pass_zero', False)
    # Applying the binary operator '*' (line 414)
    result_mul_265784 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 24), '*', list_265781, pass_zero_265783)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 24), tuple_265780, result_mul_265784)
    # Adding element type (line 414)
    # Getting the type of 'cutoff' (line 414)
    cutoff_265785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 43), 'cutoff', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 24), tuple_265780, cutoff_265785)
    # Adding element type (line 414)
    
    # Obtaining an instance of the builtin type 'list' (line 414)
    list_265786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 414)
    # Adding element type (line 414)
    float_265787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 52), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 51), list_265786, float_265787)
    
    # Getting the type of 'pass_nyquist' (line 414)
    pass_nyquist_265788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 59), 'pass_nyquist', False)
    # Applying the binary operator '*' (line 414)
    result_mul_265789 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 51), '*', list_265786, pass_nyquist_265788)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 24), tuple_265780, result_mul_265789)
    
    # Processing the call keyword arguments (line 414)
    kwargs_265790 = {}
    # Getting the type of 'np' (line 414)
    np_265778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 13), 'np', False)
    # Obtaining the member 'hstack' of a type (line 414)
    hstack_265779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 13), np_265778, 'hstack')
    # Calling hstack(args, kwargs) (line 414)
    hstack_call_result_265791 = invoke(stypy.reporting.localization.Localization(__file__, 414, 13), hstack_265779, *[tuple_265780], **kwargs_265790)
    
    # Assigning a type to the variable 'cutoff' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'cutoff', hstack_call_result_265791)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to reshape(...): (line 418)
    # Processing the call arguments (line 418)
    int_265794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 27), 'int')
    int_265795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 31), 'int')
    # Processing the call keyword arguments (line 418)
    kwargs_265796 = {}
    # Getting the type of 'cutoff' (line 418)
    cutoff_265792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'cutoff', False)
    # Obtaining the member 'reshape' of a type (line 418)
    reshape_265793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 12), cutoff_265792, 'reshape')
    # Calling reshape(args, kwargs) (line 418)
    reshape_call_result_265797 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), reshape_265793, *[int_265794, int_265795], **kwargs_265796)
    
    # Assigning a type to the variable 'bands' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'bands', reshape_call_result_265797)
    
    # Assigning a BinOp to a Name (line 421):
    
    # Assigning a BinOp to a Name (line 421):
    float_265798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 12), 'float')
    # Getting the type of 'numtaps' (line 421)
    numtaps_265799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 'numtaps')
    int_265800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 29), 'int')
    # Applying the binary operator '-' (line 421)
    result_sub_265801 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 19), '-', numtaps_265799, int_265800)
    
    # Applying the binary operator '*' (line 421)
    result_mul_265802 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 12), '*', float_265798, result_sub_265801)
    
    # Assigning a type to the variable 'alpha' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'alpha', result_mul_265802)
    
    # Assigning a BinOp to a Name (line 422):
    
    # Assigning a BinOp to a Name (line 422):
    
    # Call to arange(...): (line 422)
    # Processing the call arguments (line 422)
    int_265805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 18), 'int')
    # Getting the type of 'numtaps' (line 422)
    numtaps_265806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 21), 'numtaps', False)
    # Processing the call keyword arguments (line 422)
    kwargs_265807 = {}
    # Getting the type of 'np' (line 422)
    np_265803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 422)
    arange_265804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), np_265803, 'arange')
    # Calling arange(args, kwargs) (line 422)
    arange_call_result_265808 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), arange_265804, *[int_265805, numtaps_265806], **kwargs_265807)
    
    # Getting the type of 'alpha' (line 422)
    alpha_265809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'alpha')
    # Applying the binary operator '-' (line 422)
    result_sub_265810 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 8), '-', arange_call_result_265808, alpha_265809)
    
    # Assigning a type to the variable 'm' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'm', result_sub_265810)
    
    # Assigning a Num to a Name (line 423):
    
    # Assigning a Num to a Name (line 423):
    int_265811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
    # Assigning a type to the variable 'h' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'h', int_265811)
    
    # Getting the type of 'bands' (line 424)
    bands_265812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'bands')
    # Testing the type of a for loop iterable (line 424)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 424, 4), bands_265812)
    # Getting the type of the for loop variable (line 424)
    for_loop_var_265813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 424, 4), bands_265812)
    # Assigning a type to the variable 'left' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'left', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 4), for_loop_var_265813))
    # Assigning a type to the variable 'right' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'right', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 4), for_loop_var_265813))
    # SSA begins for a for statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'h' (line 425)
    h_265814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'h')
    # Getting the type of 'right' (line 425)
    right_265815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 13), 'right')
    
    # Call to sinc(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'right' (line 425)
    right_265817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'right', False)
    # Getting the type of 'm' (line 425)
    m_265818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 34), 'm', False)
    # Applying the binary operator '*' (line 425)
    result_mul_265819 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 26), '*', right_265817, m_265818)
    
    # Processing the call keyword arguments (line 425)
    kwargs_265820 = {}
    # Getting the type of 'sinc' (line 425)
    sinc_265816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 21), 'sinc', False)
    # Calling sinc(args, kwargs) (line 425)
    sinc_call_result_265821 = invoke(stypy.reporting.localization.Localization(__file__, 425, 21), sinc_265816, *[result_mul_265819], **kwargs_265820)
    
    # Applying the binary operator '*' (line 425)
    result_mul_265822 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 13), '*', right_265815, sinc_call_result_265821)
    
    # Applying the binary operator '+=' (line 425)
    result_iadd_265823 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 8), '+=', h_265814, result_mul_265822)
    # Assigning a type to the variable 'h' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'h', result_iadd_265823)
    
    
    # Getting the type of 'h' (line 426)
    h_265824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'h')
    # Getting the type of 'left' (line 426)
    left_265825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'left')
    
    # Call to sinc(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'left' (line 426)
    left_265827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 'left', False)
    # Getting the type of 'm' (line 426)
    m_265828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 32), 'm', False)
    # Applying the binary operator '*' (line 426)
    result_mul_265829 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 25), '*', left_265827, m_265828)
    
    # Processing the call keyword arguments (line 426)
    kwargs_265830 = {}
    # Getting the type of 'sinc' (line 426)
    sinc_265826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 20), 'sinc', False)
    # Calling sinc(args, kwargs) (line 426)
    sinc_call_result_265831 = invoke(stypy.reporting.localization.Localization(__file__, 426, 20), sinc_265826, *[result_mul_265829], **kwargs_265830)
    
    # Applying the binary operator '*' (line 426)
    result_mul_265832 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 13), '*', left_265825, sinc_call_result_265831)
    
    # Applying the binary operator '-=' (line 426)
    result_isub_265833 = python_operator(stypy.reporting.localization.Localization(__file__, 426, 8), '-=', h_265824, result_mul_265832)
    # Assigning a type to the variable 'h' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'h', result_isub_265833)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 429, 4))
    
    # 'from scipy.signal.signaltools import get_window' statement (line 429)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
    import_265834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 429, 4), 'scipy.signal.signaltools')

    if (type(import_265834) is not StypyTypeError):

        if (import_265834 != 'pyd_module'):
            __import__(import_265834)
            sys_modules_265835 = sys.modules[import_265834]
            import_from_module(stypy.reporting.localization.Localization(__file__, 429, 4), 'scipy.signal.signaltools', sys_modules_265835.module_type_store, module_type_store, ['get_window'])
            nest_module(stypy.reporting.localization.Localization(__file__, 429, 4), __file__, sys_modules_265835, sys_modules_265835.module_type_store, module_type_store)
        else:
            from scipy.signal.signaltools import get_window

            import_from_module(stypy.reporting.localization.Localization(__file__, 429, 4), 'scipy.signal.signaltools', None, module_type_store, ['get_window'], [get_window])

    else:
        # Assigning a type to the variable 'scipy.signal.signaltools' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'scipy.signal.signaltools', import_265834)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')
    
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to get_window(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'window' (line 430)
    window_265837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 21), 'window', False)
    # Getting the type of 'numtaps' (line 430)
    numtaps_265838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 29), 'numtaps', False)
    # Processing the call keyword arguments (line 430)
    # Getting the type of 'False' (line 430)
    False_265839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 46), 'False', False)
    keyword_265840 = False_265839
    kwargs_265841 = {'fftbins': keyword_265840}
    # Getting the type of 'get_window' (line 430)
    get_window_265836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 10), 'get_window', False)
    # Calling get_window(args, kwargs) (line 430)
    get_window_call_result_265842 = invoke(stypy.reporting.localization.Localization(__file__, 430, 10), get_window_265836, *[window_265837, numtaps_265838], **kwargs_265841)
    
    # Assigning a type to the variable 'win' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'win', get_window_call_result_265842)
    
    # Getting the type of 'h' (line 431)
    h_265843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'h')
    # Getting the type of 'win' (line 431)
    win_265844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 9), 'win')
    # Applying the binary operator '*=' (line 431)
    result_imul_265845 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 4), '*=', h_265843, win_265844)
    # Assigning a type to the variable 'h' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'h', result_imul_265845)
    
    
    # Getting the type of 'scale' (line 434)
    scale_265846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 7), 'scale')
    # Testing the type of an if condition (line 434)
    if_condition_265847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 434, 4), scale_265846)
    # Assigning a type to the variable 'if_condition_265847' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'if_condition_265847', if_condition_265847)
    # SSA begins for if statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 436):
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_265848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
    
    # Obtaining the type of the subscript
    int_265849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 28), 'int')
    # Getting the type of 'bands' (line 436)
    bands_265850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'bands')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___265851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 22), bands_265850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_265852 = invoke(stypy.reporting.localization.Localization(__file__, 436, 22), getitem___265851, int_265849)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___265853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), subscript_call_result_265852, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_265854 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), getitem___265853, int_265848)
    
    # Assigning a type to the variable 'tuple_var_assignment_265523' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_265523', subscript_call_result_265854)
    
    # Assigning a Subscript to a Name (line 436):
    
    # Obtaining the type of the subscript
    int_265855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 8), 'int')
    
    # Obtaining the type of the subscript
    int_265856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 28), 'int')
    # Getting the type of 'bands' (line 436)
    bands_265857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'bands')
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___265858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 22), bands_265857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_265859 = invoke(stypy.reporting.localization.Localization(__file__, 436, 22), getitem___265858, int_265856)
    
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___265860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), subscript_call_result_265859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 436)
    subscript_call_result_265861 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), getitem___265860, int_265855)
    
    # Assigning a type to the variable 'tuple_var_assignment_265524' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_265524', subscript_call_result_265861)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_265523' (line 436)
    tuple_var_assignment_265523_265862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_265523')
    # Assigning a type to the variable 'left' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'left', tuple_var_assignment_265523_265862)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'tuple_var_assignment_265524' (line 436)
    tuple_var_assignment_265524_265863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'tuple_var_assignment_265524')
    # Assigning a type to the variable 'right' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 14), 'right', tuple_var_assignment_265524_265863)
    
    
    # Getting the type of 'left' (line 437)
    left_265864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 11), 'left')
    int_265865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'int')
    # Applying the binary operator '==' (line 437)
    result_eq_265866 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 11), '==', left_265864, int_265865)
    
    # Testing the type of an if condition (line 437)
    if_condition_265867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 8), result_eq_265866)
    # Assigning a type to the variable 'if_condition_265867' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'if_condition_265867', if_condition_265867)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 438):
    
    # Assigning a Num to a Name (line 438):
    float_265868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 30), 'float')
    # Assigning a type to the variable 'scale_frequency' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'scale_frequency', float_265868)
    # SSA branch for the else part of an if statement (line 437)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'right' (line 439)
    right_265869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 13), 'right')
    int_265870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 22), 'int')
    # Applying the binary operator '==' (line 439)
    result_eq_265871 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 13), '==', right_265869, int_265870)
    
    # Testing the type of an if condition (line 439)
    if_condition_265872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 13), result_eq_265871)
    # Assigning a type to the variable 'if_condition_265872' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 13), 'if_condition_265872', if_condition_265872)
    # SSA begins for if statement (line 439)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 440):
    
    # Assigning a Num to a Name (line 440):
    float_265873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 30), 'float')
    # Assigning a type to the variable 'scale_frequency' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'scale_frequency', float_265873)
    # SSA branch for the else part of an if statement (line 439)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 442):
    
    # Assigning a BinOp to a Name (line 442):
    float_265874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 30), 'float')
    # Getting the type of 'left' (line 442)
    left_265875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 37), 'left')
    # Getting the type of 'right' (line 442)
    right_265876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 44), 'right')
    # Applying the binary operator '+' (line 442)
    result_add_265877 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 37), '+', left_265875, right_265876)
    
    # Applying the binary operator '*' (line 442)
    result_mul_265878 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 30), '*', float_265874, result_add_265877)
    
    # Assigning a type to the variable 'scale_frequency' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'scale_frequency', result_mul_265878)
    # SSA join for if statement (line 439)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 443):
    
    # Assigning a Call to a Name (line 443):
    
    # Call to cos(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'np' (line 443)
    np_265881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 19), 'np', False)
    # Obtaining the member 'pi' of a type (line 443)
    pi_265882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 19), np_265881, 'pi')
    # Getting the type of 'm' (line 443)
    m_265883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 27), 'm', False)
    # Applying the binary operator '*' (line 443)
    result_mul_265884 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 19), '*', pi_265882, m_265883)
    
    # Getting the type of 'scale_frequency' (line 443)
    scale_frequency_265885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 31), 'scale_frequency', False)
    # Applying the binary operator '*' (line 443)
    result_mul_265886 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 29), '*', result_mul_265884, scale_frequency_265885)
    
    # Processing the call keyword arguments (line 443)
    kwargs_265887 = {}
    # Getting the type of 'np' (line 443)
    np_265879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'np', False)
    # Obtaining the member 'cos' of a type (line 443)
    cos_265880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 12), np_265879, 'cos')
    # Calling cos(args, kwargs) (line 443)
    cos_call_result_265888 = invoke(stypy.reporting.localization.Localization(__file__, 443, 12), cos_265880, *[result_mul_265886], **kwargs_265887)
    
    # Assigning a type to the variable 'c' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'c', cos_call_result_265888)
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to sum(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'h' (line 444)
    h_265891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'h', False)
    # Getting the type of 'c' (line 444)
    c_265892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 23), 'c', False)
    # Applying the binary operator '*' (line 444)
    result_mul_265893 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 19), '*', h_265891, c_265892)
    
    # Processing the call keyword arguments (line 444)
    kwargs_265894 = {}
    # Getting the type of 'np' (line 444)
    np_265889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'np', False)
    # Obtaining the member 'sum' of a type (line 444)
    sum_265890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 12), np_265889, 'sum')
    # Calling sum(args, kwargs) (line 444)
    sum_call_result_265895 = invoke(stypy.reporting.localization.Localization(__file__, 444, 12), sum_265890, *[result_mul_265893], **kwargs_265894)
    
    # Assigning a type to the variable 's' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 's', sum_call_result_265895)
    
    # Getting the type of 'h' (line 445)
    h_265896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'h')
    # Getting the type of 's' (line 445)
    s_265897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 13), 's')
    # Applying the binary operator 'div=' (line 445)
    result_div_265898 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 8), 'div=', h_265896, s_265897)
    # Assigning a type to the variable 'h' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'h', result_div_265898)
    
    # SSA join for if statement (line 434)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'h' (line 447)
    h_265899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'h')
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type', h_265899)
    
    # ################# End of 'firwin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'firwin' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_265900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_265900)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'firwin'
    return stypy_return_type_265900

# Assigning a type to the variable 'firwin' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'firwin', firwin)

@norecursion
def firwin2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 454)
    None_265901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 40), 'None')
    str_265902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 53), 'str', 'hamming')
    # Getting the type of 'None' (line 454)
    None_265903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 68), 'None')
    # Getting the type of 'False' (line 455)
    False_265904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'False')
    # Getting the type of 'None' (line 455)
    None_265905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 36), 'None')
    defaults = [None_265901, str_265902, None_265903, False_265904, None_265905]
    # Create a new context for function 'firwin2'
    module_type_store = module_type_store.open_function_context('firwin2', 454, 0, False)
    
    # Passed parameters checking function
    firwin2.stypy_localization = localization
    firwin2.stypy_type_of_self = None
    firwin2.stypy_type_store = module_type_store
    firwin2.stypy_function_name = 'firwin2'
    firwin2.stypy_param_names_list = ['numtaps', 'freq', 'gain', 'nfreqs', 'window', 'nyq', 'antisymmetric', 'fs']
    firwin2.stypy_varargs_param_name = None
    firwin2.stypy_kwargs_param_name = None
    firwin2.stypy_call_defaults = defaults
    firwin2.stypy_call_varargs = varargs
    firwin2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'firwin2', ['numtaps', 'freq', 'gain', 'nfreqs', 'window', 'nyq', 'antisymmetric', 'fs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'firwin2', localization, ['numtaps', 'freq', 'gain', 'nfreqs', 'window', 'nyq', 'antisymmetric', 'fs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'firwin2(...)' code ##################

    str_265906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'str', '\n    FIR filter design using the window method.\n\n    From the given frequencies `freq` and corresponding gains `gain`,\n    this function constructs an FIR filter with linear phase and\n    (approximately) the given frequency response.\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter.  `numtaps` must be less than\n        `nfreqs`.\n    freq : array_like, 1D\n        The frequency sampling points. Typically 0.0 to 1.0 with 1.0 being\n        Nyquist.  The Nyquist frequency is half `fs`.\n        The values in `freq` must be nondecreasing.  A value can be repeated\n        once to implement a discontinuity.  The first value in `freq` must\n        be 0, and the last value must be ``fs/2``.\n    gain : array_like\n        The filter gains at the frequency sampling points. Certain\n        constraints to gain values, depending on the filter type, are applied,\n        see Notes for details.\n    nfreqs : int, optional\n        The size of the interpolation mesh used to construct the filter.\n        For most efficient behavior, this should be a power of 2 plus 1\n        (e.g, 129, 257, etc).  The default is one more than the smallest\n        power of 2 that is not less than `numtaps`.  `nfreqs` must be greater\n        than `numtaps`.\n    window : string or (string, float) or float, or None, optional\n        Window function to use. Default is "hamming".  See\n        `scipy.signal.get_window` for the complete list of possible values.\n        If None, no window function is applied.\n    nyq : float, optional\n        *Deprecated.  Use `fs` instead.*  This is the Nyquist frequency.\n        Each frequency in `freq` must be between 0 and `nyq`.  Default is 1.\n    antisymmetric : bool, optional\n        Whether resulting impulse response is symmetric/antisymmetric.\n        See Notes for more details.\n    fs : float, optional\n        The sampling frequency of the signal.  Each frequency in `cutoff`\n        must be between 0 and ``fs/2``.  Default is 2.\n\n    Returns\n    -------\n    taps : ndarray\n        The filter coefficients of the FIR filter, as a 1-D array of length\n        `numtaps`.\n\n    See also\n    --------\n    firls\n    firwin\n    minimum_phase\n    remez\n\n    Notes\n    -----\n    From the given set of frequencies and gains, the desired response is\n    constructed in the frequency domain.  The inverse FFT is applied to the\n    desired response to create the associated convolution kernel, and the\n    first `numtaps` coefficients of this kernel, scaled by `window`, are\n    returned.\n\n    The FIR filter will have linear phase. The type of filter is determined by\n    the value of \'numtaps` and `antisymmetric` flag.\n    There are four possible combinations:\n\n       - odd  `numtaps`, `antisymmetric` is False, type I filter is produced\n       - even `numtaps`, `antisymmetric` is False, type II filter is produced\n       - odd  `numtaps`, `antisymmetric` is True, type III filter is produced\n       - even `numtaps`, `antisymmetric` is True, type IV filter is produced\n\n    Magnitude response of all but type I filters are subjects to following\n    constraints:\n\n       - type II  -- zero at the Nyquist frequency\n       - type III -- zero at zero and Nyquist frequencies\n       - type IV  -- zero at zero frequency\n\n    .. versionadded:: 0.9.0\n\n    References\n    ----------\n    .. [1] Oppenheim, A. V. and Schafer, R. W., "Discrete-Time Signal\n       Processing", Prentice-Hall, Englewood Cliffs, New Jersey (1989).\n       (See, for example, Section 7.4.)\n\n    .. [2] Smith, Steven W., "The Scientist and Engineer\'s Guide to Digital\n       Signal Processing", Ch. 17. http://www.dspguide.com/ch17/1.htm\n\n    Examples\n    --------\n    A lowpass FIR filter with a response that is 1 on [0.0, 0.5], and\n    that decreases linearly on [0.5, 1.0] from 1 to 0:\n\n    >>> from scipy import signal\n    >>> taps = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])\n    >>> print(taps[72:78])\n    [-0.02286961 -0.06362756  0.57310236  0.57310236 -0.06362756 -0.02286961]\n\n    ')
    
    # Assigning a BinOp to a Name (line 557):
    
    # Assigning a BinOp to a Name (line 557):
    float_265907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 10), 'float')
    
    # Call to _get_fs(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'fs' (line 557)
    fs_265909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 24), 'fs', False)
    # Getting the type of 'nyq' (line 557)
    nyq_265910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 28), 'nyq', False)
    # Processing the call keyword arguments (line 557)
    kwargs_265911 = {}
    # Getting the type of '_get_fs' (line 557)
    _get_fs_265908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), '_get_fs', False)
    # Calling _get_fs(args, kwargs) (line 557)
    _get_fs_call_result_265912 = invoke(stypy.reporting.localization.Localization(__file__, 557, 16), _get_fs_265908, *[fs_265909, nyq_265910], **kwargs_265911)
    
    # Applying the binary operator '*' (line 557)
    result_mul_265913 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 10), '*', float_265907, _get_fs_call_result_265912)
    
    # Assigning a type to the variable 'nyq' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'nyq', result_mul_265913)
    
    
    
    # Call to len(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'freq' (line 559)
    freq_265915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), 'freq', False)
    # Processing the call keyword arguments (line 559)
    kwargs_265916 = {}
    # Getting the type of 'len' (line 559)
    len_265914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 7), 'len', False)
    # Calling len(args, kwargs) (line 559)
    len_call_result_265917 = invoke(stypy.reporting.localization.Localization(__file__, 559, 7), len_265914, *[freq_265915], **kwargs_265916)
    
    
    # Call to len(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'gain' (line 559)
    gain_265919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 24), 'gain', False)
    # Processing the call keyword arguments (line 559)
    kwargs_265920 = {}
    # Getting the type of 'len' (line 559)
    len_265918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'len', False)
    # Calling len(args, kwargs) (line 559)
    len_call_result_265921 = invoke(stypy.reporting.localization.Localization(__file__, 559, 20), len_265918, *[gain_265919], **kwargs_265920)
    
    # Applying the binary operator '!=' (line 559)
    result_ne_265922 = python_operator(stypy.reporting.localization.Localization(__file__, 559, 7), '!=', len_call_result_265917, len_call_result_265921)
    
    # Testing the type of an if condition (line 559)
    if_condition_265923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 4), result_ne_265922)
    # Assigning a type to the variable 'if_condition_265923' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'if_condition_265923', if_condition_265923)
    # SSA begins for if statement (line 559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 560)
    # Processing the call arguments (line 560)
    str_265925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 25), 'str', 'freq and gain must be of same length.')
    # Processing the call keyword arguments (line 560)
    kwargs_265926 = {}
    # Getting the type of 'ValueError' (line 560)
    ValueError_265924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 560)
    ValueError_call_result_265927 = invoke(stypy.reporting.localization.Localization(__file__, 560, 14), ValueError_265924, *[str_265925], **kwargs_265926)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 560, 8), ValueError_call_result_265927, 'raise parameter', BaseException)
    # SSA join for if statement (line 559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'nfreqs' (line 562)
    nfreqs_265928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'nfreqs')
    # Getting the type of 'None' (line 562)
    None_265929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 21), 'None')
    # Applying the binary operator 'isnot' (line 562)
    result_is_not_265930 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 7), 'isnot', nfreqs_265928, None_265929)
    
    
    # Getting the type of 'numtaps' (line 562)
    numtaps_265931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 30), 'numtaps')
    # Getting the type of 'nfreqs' (line 562)
    nfreqs_265932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 41), 'nfreqs')
    # Applying the binary operator '>=' (line 562)
    result_ge_265933 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 30), '>=', numtaps_265931, nfreqs_265932)
    
    # Applying the binary operator 'and' (line 562)
    result_and_keyword_265934 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 7), 'and', result_is_not_265930, result_ge_265933)
    
    # Testing the type of an if condition (line 562)
    if_condition_265935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 4), result_and_keyword_265934)
    # Assigning a type to the variable 'if_condition_265935' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'if_condition_265935', if_condition_265935)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 563)
    # Processing the call arguments (line 563)
    str_265937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 26), 'str', 'ntaps must be less than nfreqs, but firwin2 was called with ntaps=%d and nfreqs=%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 565)
    tuple_265938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 565)
    # Adding element type (line 565)
    # Getting the type of 'numtaps' (line 565)
    numtaps_265939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 26), 'numtaps', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 26), tuple_265938, numtaps_265939)
    # Adding element type (line 565)
    # Getting the type of 'nfreqs' (line 565)
    nfreqs_265940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 35), 'nfreqs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 565, 26), tuple_265938, nfreqs_265940)
    
    # Applying the binary operator '%' (line 563)
    result_mod_265941 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 25), '%', str_265937, tuple_265938)
    
    # Processing the call keyword arguments (line 563)
    kwargs_265942 = {}
    # Getting the type of 'ValueError' (line 563)
    ValueError_265936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 563)
    ValueError_call_result_265943 = invoke(stypy.reporting.localization.Localization(__file__, 563, 14), ValueError_265936, *[result_mod_265941], **kwargs_265942)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 563, 8), ValueError_call_result_265943, 'raise parameter', BaseException)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_265944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 12), 'int')
    # Getting the type of 'freq' (line 567)
    freq_265945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 7), 'freq')
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___265946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 7), freq_265945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_265947 = invoke(stypy.reporting.localization.Localization(__file__, 567, 7), getitem___265946, int_265944)
    
    int_265948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 18), 'int')
    # Applying the binary operator '!=' (line 567)
    result_ne_265949 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 7), '!=', subscript_call_result_265947, int_265948)
    
    
    
    # Obtaining the type of the subscript
    int_265950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 28), 'int')
    # Getting the type of 'freq' (line 567)
    freq_265951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'freq')
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___265952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), freq_265951, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_265953 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), getitem___265952, int_265950)
    
    # Getting the type of 'nyq' (line 567)
    nyq_265954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 35), 'nyq')
    # Applying the binary operator '!=' (line 567)
    result_ne_265955 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 23), '!=', subscript_call_result_265953, nyq_265954)
    
    # Applying the binary operator 'or' (line 567)
    result_or_keyword_265956 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 7), 'or', result_ne_265949, result_ne_265955)
    
    # Testing the type of an if condition (line 567)
    if_condition_265957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 4), result_or_keyword_265956)
    # Assigning a type to the variable 'if_condition_265957' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'if_condition_265957', if_condition_265957)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 568)
    # Processing the call arguments (line 568)
    str_265959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 25), 'str', 'freq must start with 0 and end with fs/2.')
    # Processing the call keyword arguments (line 568)
    kwargs_265960 = {}
    # Getting the type of 'ValueError' (line 568)
    ValueError_265958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 568)
    ValueError_call_result_265961 = invoke(stypy.reporting.localization.Localization(__file__, 568, 14), ValueError_265958, *[str_265959], **kwargs_265960)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 568, 8), ValueError_call_result_265961, 'raise parameter', BaseException)
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 569):
    
    # Assigning a Call to a Name (line 569):
    
    # Call to diff(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'freq' (line 569)
    freq_265964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), 'freq', False)
    # Processing the call keyword arguments (line 569)
    kwargs_265965 = {}
    # Getting the type of 'np' (line 569)
    np_265962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 569)
    diff_265963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), np_265962, 'diff')
    # Calling diff(args, kwargs) (line 569)
    diff_call_result_265966 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), diff_265963, *[freq_265964], **kwargs_265965)
    
    # Assigning a type to the variable 'd' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'd', diff_call_result_265966)
    
    
    # Call to any(...): (line 570)
    # Processing the call keyword arguments (line 570)
    kwargs_265971 = {}
    
    # Getting the type of 'd' (line 570)
    d_265967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'd', False)
    int_265968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 12), 'int')
    # Applying the binary operator '<' (line 570)
    result_lt_265969 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 8), '<', d_265967, int_265968)
    
    # Obtaining the member 'any' of a type (line 570)
    any_265970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), result_lt_265969, 'any')
    # Calling any(args, kwargs) (line 570)
    any_call_result_265972 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), any_265970, *[], **kwargs_265971)
    
    # Testing the type of an if condition (line 570)
    if_condition_265973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 570, 4), any_call_result_265972)
    # Assigning a type to the variable 'if_condition_265973' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'if_condition_265973', if_condition_265973)
    # SSA begins for if statement (line 570)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 571)
    # Processing the call arguments (line 571)
    str_265975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 25), 'str', 'The values in freq must be nondecreasing.')
    # Processing the call keyword arguments (line 571)
    kwargs_265976 = {}
    # Getting the type of 'ValueError' (line 571)
    ValueError_265974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 571)
    ValueError_call_result_265977 = invoke(stypy.reporting.localization.Localization(__file__, 571, 14), ValueError_265974, *[str_265975], **kwargs_265976)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 571, 8), ValueError_call_result_265977, 'raise parameter', BaseException)
    # SSA join for if statement (line 570)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 572):
    
    # Assigning a BinOp to a Name (line 572):
    
    # Obtaining the type of the subscript
    int_265978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 12), 'int')
    slice_265979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 9), None, int_265978, None)
    # Getting the type of 'd' (line 572)
    d_265980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 9), 'd')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___265981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 9), d_265980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_265982 = invoke(stypy.reporting.localization.Localization(__file__, 572, 9), getitem___265981, slice_265979)
    
    
    # Obtaining the type of the subscript
    int_265983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 20), 'int')
    slice_265984 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 572, 18), int_265983, None, None)
    # Getting the type of 'd' (line 572)
    d_265985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 18), 'd')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___265986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 18), d_265985, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_265987 = invoke(stypy.reporting.localization.Localization(__file__, 572, 18), getitem___265986, slice_265984)
    
    # Applying the binary operator '+' (line 572)
    result_add_265988 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 9), '+', subscript_call_result_265982, subscript_call_result_265987)
    
    # Assigning a type to the variable 'd2' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 4), 'd2', result_add_265988)
    
    
    # Call to any(...): (line 573)
    # Processing the call keyword arguments (line 573)
    kwargs_265993 = {}
    
    # Getting the type of 'd2' (line 573)
    d2_265989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'd2', False)
    int_265990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 14), 'int')
    # Applying the binary operator '==' (line 573)
    result_eq_265991 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 8), '==', d2_265989, int_265990)
    
    # Obtaining the member 'any' of a type (line 573)
    any_265992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), result_eq_265991, 'any')
    # Calling any(args, kwargs) (line 573)
    any_call_result_265994 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), any_265992, *[], **kwargs_265993)
    
    # Testing the type of an if condition (line 573)
    if_condition_265995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 4), any_call_result_265994)
    # Assigning a type to the variable 'if_condition_265995' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'if_condition_265995', if_condition_265995)
    # SSA begins for if statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 574)
    # Processing the call arguments (line 574)
    str_265997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 25), 'str', 'A value in freq must not occur more than twice.')
    # Processing the call keyword arguments (line 574)
    kwargs_265998 = {}
    # Getting the type of 'ValueError' (line 574)
    ValueError_265996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 574)
    ValueError_call_result_265999 = invoke(stypy.reporting.localization.Localization(__file__, 574, 14), ValueError_265996, *[str_265997], **kwargs_265998)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 574, 8), ValueError_call_result_265999, 'raise parameter', BaseException)
    # SSA join for if statement (line 573)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'antisymmetric' (line 576)
    antisymmetric_266000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 7), 'antisymmetric')
    # Testing the type of an if condition (line 576)
    if_condition_266001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 4), antisymmetric_266000)
    # Assigning a type to the variable 'if_condition_266001' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'if_condition_266001', if_condition_266001)
    # SSA begins for if statement (line 576)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'numtaps' (line 577)
    numtaps_266002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 11), 'numtaps')
    int_266003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 21), 'int')
    # Applying the binary operator '%' (line 577)
    result_mod_266004 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), '%', numtaps_266002, int_266003)
    
    int_266005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 26), 'int')
    # Applying the binary operator '==' (line 577)
    result_eq_266006 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 11), '==', result_mod_266004, int_266005)
    
    # Testing the type of an if condition (line 577)
    if_condition_266007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 577, 8), result_eq_266006)
    # Assigning a type to the variable 'if_condition_266007' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'if_condition_266007', if_condition_266007)
    # SSA begins for if statement (line 577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 578):
    
    # Assigning a Num to a Name (line 578):
    int_266008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 20), 'int')
    # Assigning a type to the variable 'ftype' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'ftype', int_266008)
    # SSA branch for the else part of an if statement (line 577)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 580):
    
    # Assigning a Num to a Name (line 580):
    int_266009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 20), 'int')
    # Assigning a type to the variable 'ftype' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'ftype', int_266009)
    # SSA join for if statement (line 577)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 576)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'numtaps' (line 582)
    numtaps_266010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'numtaps')
    int_266011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 21), 'int')
    # Applying the binary operator '%' (line 582)
    result_mod_266012 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 11), '%', numtaps_266010, int_266011)
    
    int_266013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 26), 'int')
    # Applying the binary operator '==' (line 582)
    result_eq_266014 = python_operator(stypy.reporting.localization.Localization(__file__, 582, 11), '==', result_mod_266012, int_266013)
    
    # Testing the type of an if condition (line 582)
    if_condition_266015 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 8), result_eq_266014)
    # Assigning a type to the variable 'if_condition_266015' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'if_condition_266015', if_condition_266015)
    # SSA begins for if statement (line 582)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 583):
    
    # Assigning a Num to a Name (line 583):
    int_266016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 20), 'int')
    # Assigning a type to the variable 'ftype' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'ftype', int_266016)
    # SSA branch for the else part of an if statement (line 582)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 585):
    
    # Assigning a Num to a Name (line 585):
    int_266017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 20), 'int')
    # Assigning a type to the variable 'ftype' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'ftype', int_266017)
    # SSA join for if statement (line 582)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 576)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ftype' (line 587)
    ftype_266018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 7), 'ftype')
    int_266019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 16), 'int')
    # Applying the binary operator '==' (line 587)
    result_eq_266020 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 7), '==', ftype_266018, int_266019)
    
    
    
    # Obtaining the type of the subscript
    int_266021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 27), 'int')
    # Getting the type of 'gain' (line 587)
    gain_266022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'gain')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___266023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 22), gain_266022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_266024 = invoke(stypy.reporting.localization.Localization(__file__, 587, 22), getitem___266023, int_266021)
    
    float_266025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 34), 'float')
    # Applying the binary operator '!=' (line 587)
    result_ne_266026 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 22), '!=', subscript_call_result_266024, float_266025)
    
    # Applying the binary operator 'and' (line 587)
    result_and_keyword_266027 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 7), 'and', result_eq_266020, result_ne_266026)
    
    # Testing the type of an if condition (line 587)
    if_condition_266028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 4), result_and_keyword_266027)
    # Assigning a type to the variable 'if_condition_266028' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'if_condition_266028', if_condition_266028)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 588)
    # Processing the call arguments (line 588)
    str_266030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 25), 'str', 'A Type II filter must have zero gain at the Nyquist frequency.')
    # Processing the call keyword arguments (line 588)
    kwargs_266031 = {}
    # Getting the type of 'ValueError' (line 588)
    ValueError_266029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 588)
    ValueError_call_result_266032 = invoke(stypy.reporting.localization.Localization(__file__, 588, 14), ValueError_266029, *[str_266030], **kwargs_266031)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 588, 8), ValueError_call_result_266032, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 587)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ftype' (line 590)
    ftype_266033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'ftype')
    int_266034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 18), 'int')
    # Applying the binary operator '==' (line 590)
    result_eq_266035 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 9), '==', ftype_266033, int_266034)
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_266036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 30), 'int')
    # Getting the type of 'gain' (line 590)
    gain_266037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 25), 'gain')
    # Obtaining the member '__getitem__' of a type (line 590)
    getitem___266038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 25), gain_266037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 590)
    subscript_call_result_266039 = invoke(stypy.reporting.localization.Localization(__file__, 590, 25), getitem___266038, int_266036)
    
    float_266040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 36), 'float')
    # Applying the binary operator '!=' (line 590)
    result_ne_266041 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 25), '!=', subscript_call_result_266039, float_266040)
    
    
    
    # Obtaining the type of the subscript
    int_266042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 48), 'int')
    # Getting the type of 'gain' (line 590)
    gain_266043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 43), 'gain')
    # Obtaining the member '__getitem__' of a type (line 590)
    getitem___266044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 43), gain_266043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 590)
    subscript_call_result_266045 = invoke(stypy.reporting.localization.Localization(__file__, 590, 43), getitem___266044, int_266042)
    
    float_266046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 55), 'float')
    # Applying the binary operator '!=' (line 590)
    result_ne_266047 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 43), '!=', subscript_call_result_266045, float_266046)
    
    # Applying the binary operator 'or' (line 590)
    result_or_keyword_266048 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 25), 'or', result_ne_266041, result_ne_266047)
    
    # Applying the binary operator 'and' (line 590)
    result_and_keyword_266049 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 9), 'and', result_eq_266035, result_or_keyword_266048)
    
    # Testing the type of an if condition (line 590)
    if_condition_266050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 9), result_and_keyword_266049)
    # Assigning a type to the variable 'if_condition_266050' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 9), 'if_condition_266050', if_condition_266050)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 591)
    # Processing the call arguments (line 591)
    str_266052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 25), 'str', 'A Type III filter must have zero gain at zero and Nyquist frequencies.')
    # Processing the call keyword arguments (line 591)
    kwargs_266053 = {}
    # Getting the type of 'ValueError' (line 591)
    ValueError_266051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 591)
    ValueError_call_result_266054 = invoke(stypy.reporting.localization.Localization(__file__, 591, 14), ValueError_266051, *[str_266052], **kwargs_266053)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 591, 8), ValueError_call_result_266054, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 590)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ftype' (line 593)
    ftype_266055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 9), 'ftype')
    int_266056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 18), 'int')
    # Applying the binary operator '==' (line 593)
    result_eq_266057 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 9), '==', ftype_266055, int_266056)
    
    
    
    # Obtaining the type of the subscript
    int_266058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 29), 'int')
    # Getting the type of 'gain' (line 593)
    gain_266059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 24), 'gain')
    # Obtaining the member '__getitem__' of a type (line 593)
    getitem___266060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 24), gain_266059, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 593)
    subscript_call_result_266061 = invoke(stypy.reporting.localization.Localization(__file__, 593, 24), getitem___266060, int_266058)
    
    float_266062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 35), 'float')
    # Applying the binary operator '!=' (line 593)
    result_ne_266063 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 24), '!=', subscript_call_result_266061, float_266062)
    
    # Applying the binary operator 'and' (line 593)
    result_and_keyword_266064 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 9), 'and', result_eq_266057, result_ne_266063)
    
    # Testing the type of an if condition (line 593)
    if_condition_266065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 9), result_and_keyword_266064)
    # Assigning a type to the variable 'if_condition_266065' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 9), 'if_condition_266065', if_condition_266065)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 594)
    # Processing the call arguments (line 594)
    str_266067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 25), 'str', 'A Type IV filter must have zero gain at zero frequency.')
    # Processing the call keyword arguments (line 594)
    kwargs_266068 = {}
    # Getting the type of 'ValueError' (line 594)
    ValueError_266066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 594)
    ValueError_call_result_266069 = invoke(stypy.reporting.localization.Localization(__file__, 594, 14), ValueError_266066, *[str_266067], **kwargs_266068)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 594, 8), ValueError_call_result_266069, 'raise parameter', BaseException)
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 597)
    # Getting the type of 'nfreqs' (line 597)
    nfreqs_266070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 7), 'nfreqs')
    # Getting the type of 'None' (line 597)
    None_266071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 17), 'None')
    
    (may_be_266072, more_types_in_union_266073) = may_be_none(nfreqs_266070, None_266071)

    if may_be_266072:

        if more_types_in_union_266073:
            # Runtime conditional SSA (line 597)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 598):
        
        # Assigning a BinOp to a Name (line 598):
        int_266074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 17), 'int')
        int_266075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 21), 'int')
        
        # Call to int(...): (line 598)
        # Processing the call arguments (line 598)
        
        # Call to ceil(...): (line 598)
        # Processing the call arguments (line 598)
        
        # Call to log(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'numtaps' (line 598)
        numtaps_266079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 39), 'numtaps', False)
        int_266080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 48), 'int')
        # Processing the call keyword arguments (line 598)
        kwargs_266081 = {}
        # Getting the type of 'log' (line 598)
        log_266078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 35), 'log', False)
        # Calling log(args, kwargs) (line 598)
        log_call_result_266082 = invoke(stypy.reporting.localization.Localization(__file__, 598, 35), log_266078, *[numtaps_266079, int_266080], **kwargs_266081)
        
        # Processing the call keyword arguments (line 598)
        kwargs_266083 = {}
        # Getting the type of 'ceil' (line 598)
        ceil_266077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 30), 'ceil', False)
        # Calling ceil(args, kwargs) (line 598)
        ceil_call_result_266084 = invoke(stypy.reporting.localization.Localization(__file__, 598, 30), ceil_266077, *[log_call_result_266082], **kwargs_266083)
        
        # Processing the call keyword arguments (line 598)
        kwargs_266085 = {}
        # Getting the type of 'int' (line 598)
        int_266076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 26), 'int', False)
        # Calling int(args, kwargs) (line 598)
        int_call_result_266086 = invoke(stypy.reporting.localization.Localization(__file__, 598, 26), int_266076, *[ceil_call_result_266084], **kwargs_266085)
        
        # Applying the binary operator '**' (line 598)
        result_pow_266087 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 21), '**', int_266075, int_call_result_266086)
        
        # Applying the binary operator '+' (line 598)
        result_add_266088 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 17), '+', int_266074, result_pow_266087)
        
        # Assigning a type to the variable 'nfreqs' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'nfreqs', result_add_266088)

        if more_types_in_union_266073:
            # SSA join for if statement (line 597)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 601):
    
    # Assigning a Attribute to a Name (line 601):
    
    # Call to finfo(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'float' (line 601)
    float_266091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 19), 'float', False)
    # Processing the call keyword arguments (line 601)
    kwargs_266092 = {}
    # Getting the type of 'np' (line 601)
    np_266089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 10), 'np', False)
    # Obtaining the member 'finfo' of a type (line 601)
    finfo_266090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 10), np_266089, 'finfo')
    # Calling finfo(args, kwargs) (line 601)
    finfo_call_result_266093 = invoke(stypy.reporting.localization.Localization(__file__, 601, 10), finfo_266090, *[float_266091], **kwargs_266092)
    
    # Obtaining the member 'eps' of a type (line 601)
    eps_266094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 10), finfo_call_result_266093, 'eps')
    # Assigning a type to the variable 'eps' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'eps', eps_266094)
    
    
    # Call to range(...): (line 602)
    # Processing the call arguments (line 602)
    
    # Call to len(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'freq' (line 602)
    freq_266097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 23), 'freq', False)
    # Processing the call keyword arguments (line 602)
    kwargs_266098 = {}
    # Getting the type of 'len' (line 602)
    len_266096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 19), 'len', False)
    # Calling len(args, kwargs) (line 602)
    len_call_result_266099 = invoke(stypy.reporting.localization.Localization(__file__, 602, 19), len_266096, *[freq_266097], **kwargs_266098)
    
    # Processing the call keyword arguments (line 602)
    kwargs_266100 = {}
    # Getting the type of 'range' (line 602)
    range_266095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 13), 'range', False)
    # Calling range(args, kwargs) (line 602)
    range_call_result_266101 = invoke(stypy.reporting.localization.Localization(__file__, 602, 13), range_266095, *[len_call_result_266099], **kwargs_266100)
    
    # Testing the type of a for loop iterable (line 602)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 602, 4), range_call_result_266101)
    # Getting the type of the for loop variable (line 602)
    for_loop_var_266102 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 602, 4), range_call_result_266101)
    # Assigning a type to the variable 'k' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'k', for_loop_var_266102)
    # SSA begins for a for statement (line 602)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'k' (line 603)
    k_266103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'k')
    
    # Call to len(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'freq' (line 603)
    freq_266105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 19), 'freq', False)
    # Processing the call keyword arguments (line 603)
    kwargs_266106 = {}
    # Getting the type of 'len' (line 603)
    len_266104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 15), 'len', False)
    # Calling len(args, kwargs) (line 603)
    len_call_result_266107 = invoke(stypy.reporting.localization.Localization(__file__, 603, 15), len_266104, *[freq_266105], **kwargs_266106)
    
    int_266108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 27), 'int')
    # Applying the binary operator '-' (line 603)
    result_sub_266109 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 15), '-', len_call_result_266107, int_266108)
    
    # Applying the binary operator '<' (line 603)
    result_lt_266110 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 11), '<', k_266103, result_sub_266109)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 603)
    k_266111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 38), 'k')
    # Getting the type of 'freq' (line 603)
    freq_266112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 33), 'freq')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___266113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 33), freq_266112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_266114 = invoke(stypy.reporting.localization.Localization(__file__, 603, 33), getitem___266113, k_266111)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 603)
    k_266115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 49), 'k')
    int_266116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 53), 'int')
    # Applying the binary operator '+' (line 603)
    result_add_266117 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 49), '+', k_266115, int_266116)
    
    # Getting the type of 'freq' (line 603)
    freq_266118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 44), 'freq')
    # Obtaining the member '__getitem__' of a type (line 603)
    getitem___266119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 44), freq_266118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 603)
    subscript_call_result_266120 = invoke(stypy.reporting.localization.Localization(__file__, 603, 44), getitem___266119, result_add_266117)
    
    # Applying the binary operator '==' (line 603)
    result_eq_266121 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 33), '==', subscript_call_result_266114, subscript_call_result_266120)
    
    # Applying the binary operator 'and' (line 603)
    result_and_keyword_266122 = python_operator(stypy.reporting.localization.Localization(__file__, 603, 11), 'and', result_lt_266110, result_eq_266121)
    
    # Testing the type of an if condition (line 603)
    if_condition_266123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 603, 8), result_and_keyword_266122)
    # Assigning a type to the variable 'if_condition_266123' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'if_condition_266123', if_condition_266123)
    # SSA begins for if statement (line 603)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 604):
    
    # Assigning a BinOp to a Subscript (line 604):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 604)
    k_266124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 27), 'k')
    # Getting the type of 'freq' (line 604)
    freq_266125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 22), 'freq')
    # Obtaining the member '__getitem__' of a type (line 604)
    getitem___266126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 22), freq_266125, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 604)
    subscript_call_result_266127 = invoke(stypy.reporting.localization.Localization(__file__, 604, 22), getitem___266126, k_266124)
    
    # Getting the type of 'eps' (line 604)
    eps_266128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 32), 'eps')
    # Applying the binary operator '-' (line 604)
    result_sub_266129 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 22), '-', subscript_call_result_266127, eps_266128)
    
    # Getting the type of 'freq' (line 604)
    freq_266130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'freq')
    # Getting the type of 'k' (line 604)
    k_266131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 17), 'k')
    # Storing an element on a container (line 604)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), freq_266130, (k_266131, result_sub_266129))
    
    # Assigning a BinOp to a Subscript (line 605):
    
    # Assigning a BinOp to a Subscript (line 605):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 605)
    k_266132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 31), 'k')
    int_266133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 35), 'int')
    # Applying the binary operator '+' (line 605)
    result_add_266134 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 31), '+', k_266132, int_266133)
    
    # Getting the type of 'freq' (line 605)
    freq_266135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 26), 'freq')
    # Obtaining the member '__getitem__' of a type (line 605)
    getitem___266136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 26), freq_266135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 605)
    subscript_call_result_266137 = invoke(stypy.reporting.localization.Localization(__file__, 605, 26), getitem___266136, result_add_266134)
    
    # Getting the type of 'eps' (line 605)
    eps_266138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 40), 'eps')
    # Applying the binary operator '+' (line 605)
    result_add_266139 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 26), '+', subscript_call_result_266137, eps_266138)
    
    # Getting the type of 'freq' (line 605)
    freq_266140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'freq')
    # Getting the type of 'k' (line 605)
    k_266141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 17), 'k')
    int_266142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 21), 'int')
    # Applying the binary operator '+' (line 605)
    result_add_266143 = python_operator(stypy.reporting.localization.Localization(__file__, 605, 17), '+', k_266141, int_266142)
    
    # Storing an element on a container (line 605)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 605, 12), freq_266140, (result_add_266143, result_add_266139))
    # SSA join for if statement (line 603)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 608):
    
    # Assigning a Call to a Name (line 608):
    
    # Call to linspace(...): (line 608)
    # Processing the call arguments (line 608)
    float_266146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 20), 'float')
    # Getting the type of 'nyq' (line 608)
    nyq_266147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 25), 'nyq', False)
    # Getting the type of 'nfreqs' (line 608)
    nfreqs_266148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 30), 'nfreqs', False)
    # Processing the call keyword arguments (line 608)
    kwargs_266149 = {}
    # Getting the type of 'np' (line 608)
    np_266144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 608)
    linspace_266145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 8), np_266144, 'linspace')
    # Calling linspace(args, kwargs) (line 608)
    linspace_call_result_266150 = invoke(stypy.reporting.localization.Localization(__file__, 608, 8), linspace_266145, *[float_266146, nyq_266147, nfreqs_266148], **kwargs_266149)
    
    # Assigning a type to the variable 'x' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'x', linspace_call_result_266150)
    
    # Assigning a Call to a Name (line 609):
    
    # Assigning a Call to a Name (line 609):
    
    # Call to interp(...): (line 609)
    # Processing the call arguments (line 609)
    # Getting the type of 'x' (line 609)
    x_266153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'x', False)
    # Getting the type of 'freq' (line 609)
    freq_266154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 22), 'freq', False)
    # Getting the type of 'gain' (line 609)
    gain_266155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 28), 'gain', False)
    # Processing the call keyword arguments (line 609)
    kwargs_266156 = {}
    # Getting the type of 'np' (line 609)
    np_266151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 9), 'np', False)
    # Obtaining the member 'interp' of a type (line 609)
    interp_266152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 9), np_266151, 'interp')
    # Calling interp(args, kwargs) (line 609)
    interp_call_result_266157 = invoke(stypy.reporting.localization.Localization(__file__, 609, 9), interp_266152, *[x_266153, freq_266154, gain_266155], **kwargs_266156)
    
    # Assigning a type to the variable 'fx' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 4), 'fx', interp_call_result_266157)
    
    # Assigning a Call to a Name (line 613):
    
    # Assigning a Call to a Name (line 613):
    
    # Call to exp(...): (line 613)
    # Processing the call arguments (line 613)
    
    # Getting the type of 'numtaps' (line 613)
    numtaps_266160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 21), 'numtaps', False)
    int_266161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 31), 'int')
    # Applying the binary operator '-' (line 613)
    result_sub_266162 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 21), '-', numtaps_266160, int_266161)
    
    # Applying the 'usub' unary operator (line 613)
    result___neg___266163 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 19), 'usub', result_sub_266162)
    
    float_266164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 36), 'float')
    # Applying the binary operator 'div' (line 613)
    result_div_266165 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 19), 'div', result___neg___266163, float_266164)
    
    complex_266166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 41), 'complex')
    # Applying the binary operator '*' (line 613)
    result_mul_266167 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 39), '*', result_div_266165, complex_266166)
    
    # Getting the type of 'np' (line 613)
    np_266168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 47), 'np', False)
    # Obtaining the member 'pi' of a type (line 613)
    pi_266169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 47), np_266168, 'pi')
    # Applying the binary operator '*' (line 613)
    result_mul_266170 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 45), '*', result_mul_266167, pi_266169)
    
    # Getting the type of 'x' (line 613)
    x_266171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 55), 'x', False)
    # Applying the binary operator '*' (line 613)
    result_mul_266172 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 53), '*', result_mul_266170, x_266171)
    
    # Getting the type of 'nyq' (line 613)
    nyq_266173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 59), 'nyq', False)
    # Applying the binary operator 'div' (line 613)
    result_div_266174 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 57), 'div', result_mul_266172, nyq_266173)
    
    # Processing the call keyword arguments (line 613)
    kwargs_266175 = {}
    # Getting the type of 'np' (line 613)
    np_266158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 12), 'np', False)
    # Obtaining the member 'exp' of a type (line 613)
    exp_266159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 12), np_266158, 'exp')
    # Calling exp(args, kwargs) (line 613)
    exp_call_result_266176 = invoke(stypy.reporting.localization.Localization(__file__, 613, 12), exp_266159, *[result_div_266174], **kwargs_266175)
    
    # Assigning a type to the variable 'shift' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'shift', exp_call_result_266176)
    
    
    # Getting the type of 'ftype' (line 614)
    ftype_266177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 7), 'ftype')
    int_266178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 15), 'int')
    # Applying the binary operator '>' (line 614)
    result_gt_266179 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 7), '>', ftype_266177, int_266178)
    
    # Testing the type of an if condition (line 614)
    if_condition_266180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 4), result_gt_266179)
    # Assigning a type to the variable 'if_condition_266180' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'if_condition_266180', if_condition_266180)
    # SSA begins for if statement (line 614)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'shift' (line 615)
    shift_266181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'shift')
    complex_266182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 17), 'complex')
    # Applying the binary operator '*=' (line 615)
    result_imul_266183 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 8), '*=', shift_266181, complex_266182)
    # Assigning a type to the variable 'shift' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'shift', result_imul_266183)
    
    # SSA join for if statement (line 614)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 617):
    
    # Assigning a BinOp to a Name (line 617):
    # Getting the type of 'fx' (line 617)
    fx_266184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 10), 'fx')
    # Getting the type of 'shift' (line 617)
    shift_266185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 15), 'shift')
    # Applying the binary operator '*' (line 617)
    result_mul_266186 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 10), '*', fx_266184, shift_266185)
    
    # Assigning a type to the variable 'fx2' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'fx2', result_mul_266186)
    
    # Assigning a Call to a Name (line 620):
    
    # Assigning a Call to a Name (line 620):
    
    # Call to irfft(...): (line 620)
    # Processing the call arguments (line 620)
    # Getting the type of 'fx2' (line 620)
    fx2_266188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 21), 'fx2', False)
    # Processing the call keyword arguments (line 620)
    kwargs_266189 = {}
    # Getting the type of 'irfft' (line 620)
    irfft_266187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'irfft', False)
    # Calling irfft(args, kwargs) (line 620)
    irfft_call_result_266190 = invoke(stypy.reporting.localization.Localization(__file__, 620, 15), irfft_266187, *[fx2_266188], **kwargs_266189)
    
    # Assigning a type to the variable 'out_full' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'out_full', irfft_call_result_266190)
    
    # Type idiom detected: calculating its left and rigth part (line 622)
    # Getting the type of 'window' (line 622)
    window_266191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'window')
    # Getting the type of 'None' (line 622)
    None_266192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 21), 'None')
    
    (may_be_266193, more_types_in_union_266194) = may_not_be_none(window_266191, None_266192)

    if may_be_266193:

        if more_types_in_union_266194:
            # Runtime conditional SSA (line 622)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 624, 8))
        
        # 'from scipy.signal.signaltools import get_window' statement (line 624)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
        import_266195 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 624, 8), 'scipy.signal.signaltools')

        if (type(import_266195) is not StypyTypeError):

            if (import_266195 != 'pyd_module'):
                __import__(import_266195)
                sys_modules_266196 = sys.modules[import_266195]
                import_from_module(stypy.reporting.localization.Localization(__file__, 624, 8), 'scipy.signal.signaltools', sys_modules_266196.module_type_store, module_type_store, ['get_window'])
                nest_module(stypy.reporting.localization.Localization(__file__, 624, 8), __file__, sys_modules_266196, sys_modules_266196.module_type_store, module_type_store)
            else:
                from scipy.signal.signaltools import get_window

                import_from_module(stypy.reporting.localization.Localization(__file__, 624, 8), 'scipy.signal.signaltools', None, module_type_store, ['get_window'], [get_window])

        else:
            # Assigning a type to the variable 'scipy.signal.signaltools' (line 624)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'scipy.signal.signaltools', import_266195)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')
        
        
        # Assigning a Call to a Name (line 625):
        
        # Assigning a Call to a Name (line 625):
        
        # Call to get_window(...): (line 625)
        # Processing the call arguments (line 625)
        # Getting the type of 'window' (line 625)
        window_266198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 26), 'window', False)
        # Getting the type of 'numtaps' (line 625)
        numtaps_266199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 34), 'numtaps', False)
        # Processing the call keyword arguments (line 625)
        # Getting the type of 'False' (line 625)
        False_266200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 51), 'False', False)
        keyword_266201 = False_266200
        kwargs_266202 = {'fftbins': keyword_266201}
        # Getting the type of 'get_window' (line 625)
        get_window_266197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'get_window', False)
        # Calling get_window(args, kwargs) (line 625)
        get_window_call_result_266203 = invoke(stypy.reporting.localization.Localization(__file__, 625, 15), get_window_266197, *[window_266198, numtaps_266199], **kwargs_266202)
        
        # Assigning a type to the variable 'wind' (line 625)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'wind', get_window_call_result_266203)

        if more_types_in_union_266194:
            # Runtime conditional SSA for else branch (line 622)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_266193) or more_types_in_union_266194):
        
        # Assigning a Num to a Name (line 627):
        
        # Assigning a Num to a Name (line 627):
        int_266204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 15), 'int')
        # Assigning a type to the variable 'wind' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'wind', int_266204)

        if (may_be_266193 and more_types_in_union_266194):
            # SSA join for if statement (line 622)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 631):
    
    # Assigning a BinOp to a Name (line 631):
    
    # Obtaining the type of the subscript
    # Getting the type of 'numtaps' (line 631)
    numtaps_266205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 20), 'numtaps')
    slice_266206 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 631, 10), None, numtaps_266205, None)
    # Getting the type of 'out_full' (line 631)
    out_full_266207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 10), 'out_full')
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___266208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 10), out_full_266207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_266209 = invoke(stypy.reporting.localization.Localization(__file__, 631, 10), getitem___266208, slice_266206)
    
    # Getting the type of 'wind' (line 631)
    wind_266210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 31), 'wind')
    # Applying the binary operator '*' (line 631)
    result_mul_266211 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 10), '*', subscript_call_result_266209, wind_266210)
    
    # Assigning a type to the variable 'out' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'out', result_mul_266211)
    
    
    # Getting the type of 'ftype' (line 633)
    ftype_266212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 7), 'ftype')
    int_266213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 16), 'int')
    # Applying the binary operator '==' (line 633)
    result_eq_266214 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 7), '==', ftype_266212, int_266213)
    
    # Testing the type of an if condition (line 633)
    if_condition_266215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 633, 4), result_eq_266214)
    # Assigning a type to the variable 'if_condition_266215' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'if_condition_266215', if_condition_266215)
    # SSA begins for if statement (line 633)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 634):
    
    # Assigning a Num to a Subscript (line 634):
    float_266216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 29), 'float')
    # Getting the type of 'out' (line 634)
    out_266217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'out')
    # Getting the type of 'out' (line 634)
    out_266218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 12), 'out')
    # Obtaining the member 'size' of a type (line 634)
    size_266219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 12), out_266218, 'size')
    int_266220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 24), 'int')
    # Applying the binary operator '//' (line 634)
    result_floordiv_266221 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 12), '//', size_266219, int_266220)
    
    # Storing an element on a container (line 634)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 8), out_266217, (result_floordiv_266221, float_266216))
    # SSA join for if statement (line 633)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 636)
    out_266222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'stypy_return_type', out_266222)
    
    # ################# End of 'firwin2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'firwin2' in the type store
    # Getting the type of 'stypy_return_type' (line 454)
    stypy_return_type_266223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_266223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'firwin2'
    return stypy_return_type_266223

# Assigning a type to the variable 'firwin2' (line 454)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'firwin2', firwin2)

@norecursion
def remez(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 639)
    None_266224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 42), 'None')
    # Getting the type of 'None' (line 639)
    None_266225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 51), 'None')
    str_266226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 639, 62), 'str', 'bandpass')
    int_266227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 18), 'int')
    int_266228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 35), 'int')
    # Getting the type of 'None' (line 640)
    None_266229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 42), 'None')
    defaults = [None_266224, None_266225, str_266226, int_266227, int_266228, None_266229]
    # Create a new context for function 'remez'
    module_type_store = module_type_store.open_function_context('remez', 639, 0, False)
    
    # Passed parameters checking function
    remez.stypy_localization = localization
    remez.stypy_type_of_self = None
    remez.stypy_type_store = module_type_store
    remez.stypy_function_name = 'remez'
    remez.stypy_param_names_list = ['numtaps', 'bands', 'desired', 'weight', 'Hz', 'type', 'maxiter', 'grid_density', 'fs']
    remez.stypy_varargs_param_name = None
    remez.stypy_kwargs_param_name = None
    remez.stypy_call_defaults = defaults
    remez.stypy_call_varargs = varargs
    remez.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remez', ['numtaps', 'bands', 'desired', 'weight', 'Hz', 'type', 'maxiter', 'grid_density', 'fs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remez', localization, ['numtaps', 'bands', 'desired', 'weight', 'Hz', 'type', 'maxiter', 'grid_density', 'fs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remez(...)' code ##################

    str_266230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, (-1)), 'str', '\n    Calculate the minimax optimal filter using the Remez exchange algorithm.\n\n    Calculate the filter-coefficients for the finite impulse response\n    (FIR) filter whose transfer function minimizes the maximum error\n    between the desired gain and the realized gain in the specified\n    frequency bands using the Remez exchange algorithm.\n\n    Parameters\n    ----------\n    numtaps : int\n        The desired number of taps in the filter. The number of taps is\n        the number of terms in the filter, or the filter order plus one.\n    bands : array_like\n        A monotonic sequence containing the band edges.\n        All elements must be non-negative and less than half the sampling\n        frequency as given by `fs`.\n    desired : array_like\n        A sequence half the size of bands containing the desired gain\n        in each of the specified bands.\n    weight : array_like, optional\n        A relative weighting to give to each band region. The length of\n        `weight` has to be half the length of `bands`.\n    Hz : scalar, optional\n        *Deprecated.  Use `fs` instead.*\n        The sampling frequency in Hz. Default is 1.\n    type : {\'bandpass\', \'differentiator\', \'hilbert\'}, optional\n        The type of filter:\n\n          * \'bandpass\' : flat response in bands. This is the default.\n\n          * \'differentiator\' : frequency proportional response in bands.\n\n          * \'hilbert\' : filter with odd symmetry, that is, type III\n                        (for even order) or type IV (for odd order)\n                        linear phase filters.\n\n    maxiter : int, optional\n        Maximum number of iterations of the algorithm. Default is 25.\n    grid_density : int, optional\n        Grid density. The dense grid used in `remez` is of size\n        ``(numtaps + 1) * grid_density``. Default is 16.\n    fs : float, optional\n        The sampling frequency of the signal.  Default is 1.\n\n    Returns\n    -------\n    out : ndarray\n        A rank-1 array containing the coefficients of the optimal\n        (in a minimax sense) filter.\n\n    See Also\n    --------\n    firls\n    firwin\n    firwin2\n    minimum_phase\n\n    References\n    ----------\n    .. [1] J. H. McClellan and T. W. Parks, "A unified approach to the\n           design of optimum FIR linear phase digital filters",\n           IEEE Trans. Circuit Theory, vol. CT-20, pp. 697-701, 1973.\n    .. [2] J. H. McClellan, T. W. Parks and L. R. Rabiner, "A Computer\n           Program for Designing Optimum FIR Linear Phase Digital\n           Filters", IEEE Trans. Audio Electroacoust., vol. AU-21,\n           pp. 506-525, 1973.\n\n    Examples\n    --------\n    For a signal sampled at 100 Hz, we want to construct a filter with a\n    passband at 20-40 Hz, and stop bands at 0-10 Hz and 45-50 Hz. Note that\n    this means that the behavior in the frequency ranges between those bands\n    is unspecified and may overshoot.\n\n    >>> from scipy import signal\n    >>> fs = 100\n    >>> bpass = signal.remez(72, [0, 10, 20, 40, 45, 50], [0, 1, 0], fs=fs)\n    >>> freq, response = signal.freqz(bpass)\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.semilogy(0.5*fs*freq/np.pi, np.abs(response), \'b-\')\n    >>> plt.grid(alpha=0.25)\n    >>> plt.xlabel(\'Frequency (Hz)\')\n    >>> plt.ylabel(\'Gain\')\n    >>> plt.show()\n\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'Hz' (line 729)
    Hz_266231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'Hz')
    # Getting the type of 'None' (line 729)
    None_266232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 13), 'None')
    # Applying the binary operator 'is' (line 729)
    result_is__266233 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), 'is', Hz_266231, None_266232)
    
    
    # Getting the type of 'fs' (line 729)
    fs_266234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 22), 'fs')
    # Getting the type of 'None' (line 729)
    None_266235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 28), 'None')
    # Applying the binary operator 'is' (line 729)
    result_is__266236 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 22), 'is', fs_266234, None_266235)
    
    # Applying the binary operator 'and' (line 729)
    result_and_keyword_266237 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), 'and', result_is__266233, result_is__266236)
    
    # Testing the type of an if condition (line 729)
    if_condition_266238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 4), result_and_keyword_266237)
    # Assigning a type to the variable 'if_condition_266238' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'if_condition_266238', if_condition_266238)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 730):
    
    # Assigning a Num to a Name (line 730):
    float_266239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 13), 'float')
    # Assigning a type to the variable 'fs' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'fs', float_266239)
    # SSA branch for the else part of an if statement (line 729)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 731)
    # Getting the type of 'Hz' (line 731)
    Hz_266240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 9), 'Hz')
    # Getting the type of 'None' (line 731)
    None_266241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 19), 'None')
    
    (may_be_266242, more_types_in_union_266243) = may_not_be_none(Hz_266240, None_266241)

    if may_be_266242:

        if more_types_in_union_266243:
            # Runtime conditional SSA (line 731)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 732)
        # Getting the type of 'fs' (line 732)
        fs_266244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'fs')
        # Getting the type of 'None' (line 732)
        None_266245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 21), 'None')
        
        (may_be_266246, more_types_in_union_266247) = may_not_be_none(fs_266244, None_266245)

        if may_be_266246:

            if more_types_in_union_266247:
                # Runtime conditional SSA (line 732)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 733)
            # Processing the call arguments (line 733)
            str_266249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 29), 'str', "Values cannot be given for both 'Hz' and 'fs'.")
            # Processing the call keyword arguments (line 733)
            kwargs_266250 = {}
            # Getting the type of 'ValueError' (line 733)
            ValueError_266248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 733)
            ValueError_call_result_266251 = invoke(stypy.reporting.localization.Localization(__file__, 733, 18), ValueError_266248, *[str_266249], **kwargs_266250)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 733, 12), ValueError_call_result_266251, 'raise parameter', BaseException)

            if more_types_in_union_266247:
                # SSA join for if statement (line 732)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 734):
        
        # Assigning a Name to a Name (line 734):
        # Getting the type of 'Hz' (line 734)
        Hz_266252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 13), 'Hz')
        # Assigning a type to the variable 'fs' (line 734)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 8), 'fs', Hz_266252)

        if more_types_in_union_266243:
            # SSA join for if statement (line 731)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 737)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 738):
    
    # Assigning a Subscript to a Name (line 738):
    
    # Obtaining the type of the subscript
    # Getting the type of 'type' (line 738)
    type_266253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 66), 'type')
    
    # Obtaining an instance of the builtin type 'dict' (line 738)
    dict_266254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 738)
    # Adding element type (key, value) (line 738)
    str_266255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 16), 'str', 'bandpass')
    int_266256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 28), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 15), dict_266254, (str_266255, int_266256))
    # Adding element type (key, value) (line 738)
    str_266257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 31), 'str', 'differentiator')
    int_266258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 49), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 15), dict_266254, (str_266257, int_266258))
    # Adding element type (key, value) (line 738)
    str_266259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 52), 'str', 'hilbert')
    int_266260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 63), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 15), dict_266254, (str_266259, int_266260))
    
    # Obtaining the member '__getitem__' of a type (line 738)
    getitem___266261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 15), dict_266254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 738)
    subscript_call_result_266262 = invoke(stypy.reporting.localization.Localization(__file__, 738, 15), getitem___266261, type_266253)
    
    # Assigning a type to the variable 'tnum' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'tnum', subscript_call_result_266262)
    # SSA branch for the except part of a try statement (line 737)
    # SSA branch for the except 'KeyError' branch of a try statement (line 737)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 740)
    # Processing the call arguments (line 740)
    str_266264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 25), 'str', "Type must be 'bandpass', 'differentiator', or 'hilbert'")
    # Processing the call keyword arguments (line 740)
    kwargs_266265 = {}
    # Getting the type of 'ValueError' (line 740)
    ValueError_266263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 740)
    ValueError_call_result_266266 = invoke(stypy.reporting.localization.Localization(__file__, 740, 14), ValueError_266263, *[str_266264], **kwargs_266265)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 740, 8), ValueError_call_result_266266, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 737)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 744)
    # Getting the type of 'weight' (line 744)
    weight_266267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 7), 'weight')
    # Getting the type of 'None' (line 744)
    None_266268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 17), 'None')
    
    (may_be_266269, more_types_in_union_266270) = may_be_none(weight_266267, None_266268)

    if may_be_266269:

        if more_types_in_union_266270:
            # Runtime conditional SSA (line 744)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 745):
        
        # Assigning a BinOp to a Name (line 745):
        
        # Obtaining an instance of the builtin type 'list' (line 745)
        list_266271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 745)
        # Adding element type (line 745)
        int_266272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 17), list_266271, int_266272)
        
        
        # Call to len(...): (line 745)
        # Processing the call arguments (line 745)
        # Getting the type of 'desired' (line 745)
        desired_266274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 27), 'desired', False)
        # Processing the call keyword arguments (line 745)
        kwargs_266275 = {}
        # Getting the type of 'len' (line 745)
        len_266273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 23), 'len', False)
        # Calling len(args, kwargs) (line 745)
        len_call_result_266276 = invoke(stypy.reporting.localization.Localization(__file__, 745, 23), len_266273, *[desired_266274], **kwargs_266275)
        
        # Applying the binary operator '*' (line 745)
        result_mul_266277 = python_operator(stypy.reporting.localization.Localization(__file__, 745, 17), '*', list_266271, len_call_result_266276)
        
        # Assigning a type to the variable 'weight' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'weight', result_mul_266277)

        if more_types_in_union_266270:
            # SSA join for if statement (line 744)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 747):
    
    # Assigning a Call to a Name (line 747):
    
    # Call to copy(...): (line 747)
    # Processing the call keyword arguments (line 747)
    kwargs_266284 = {}
    
    # Call to asarray(...): (line 747)
    # Processing the call arguments (line 747)
    # Getting the type of 'bands' (line 747)
    bands_266280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 23), 'bands', False)
    # Processing the call keyword arguments (line 747)
    kwargs_266281 = {}
    # Getting the type of 'np' (line 747)
    np_266278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 747)
    asarray_266279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 12), np_266278, 'asarray')
    # Calling asarray(args, kwargs) (line 747)
    asarray_call_result_266282 = invoke(stypy.reporting.localization.Localization(__file__, 747, 12), asarray_266279, *[bands_266280], **kwargs_266281)
    
    # Obtaining the member 'copy' of a type (line 747)
    copy_266283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 12), asarray_call_result_266282, 'copy')
    # Calling copy(args, kwargs) (line 747)
    copy_call_result_266285 = invoke(stypy.reporting.localization.Localization(__file__, 747, 12), copy_266283, *[], **kwargs_266284)
    
    # Assigning a type to the variable 'bands' (line 747)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 4), 'bands', copy_call_result_266285)
    
    # Call to _remez(...): (line 748)
    # Processing the call arguments (line 748)
    # Getting the type of 'numtaps' (line 748)
    numtaps_266288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 27), 'numtaps', False)
    # Getting the type of 'bands' (line 748)
    bands_266289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 36), 'bands', False)
    # Getting the type of 'desired' (line 748)
    desired_266290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 43), 'desired', False)
    # Getting the type of 'weight' (line 748)
    weight_266291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 52), 'weight', False)
    # Getting the type of 'tnum' (line 748)
    tnum_266292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 60), 'tnum', False)
    # Getting the type of 'fs' (line 748)
    fs_266293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 66), 'fs', False)
    # Getting the type of 'maxiter' (line 749)
    maxiter_266294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 27), 'maxiter', False)
    # Getting the type of 'grid_density' (line 749)
    grid_density_266295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 36), 'grid_density', False)
    # Processing the call keyword arguments (line 748)
    kwargs_266296 = {}
    # Getting the type of 'sigtools' (line 748)
    sigtools_266286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 11), 'sigtools', False)
    # Obtaining the member '_remez' of a type (line 748)
    _remez_266287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 748, 11), sigtools_266286, '_remez')
    # Calling _remez(args, kwargs) (line 748)
    _remez_call_result_266297 = invoke(stypy.reporting.localization.Localization(__file__, 748, 11), _remez_266287, *[numtaps_266288, bands_266289, desired_266290, weight_266291, tnum_266292, fs_266293, maxiter_266294, grid_density_266295], **kwargs_266296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 748)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'stypy_return_type', _remez_call_result_266297)
    
    # ################# End of 'remez(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remez' in the type store
    # Getting the type of 'stypy_return_type' (line 639)
    stypy_return_type_266298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_266298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remez'
    return stypy_return_type_266298

# Assigning a type to the variable 'remez' (line 639)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 639, 0), 'remez', remez)

@norecursion
def firls(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 752)
    None_266299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 42), 'None')
    # Getting the type of 'None' (line 752)
    None_266300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 52), 'None')
    # Getting the type of 'None' (line 752)
    None_266301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 61), 'None')
    defaults = [None_266299, None_266300, None_266301]
    # Create a new context for function 'firls'
    module_type_store = module_type_store.open_function_context('firls', 752, 0, False)
    
    # Passed parameters checking function
    firls.stypy_localization = localization
    firls.stypy_type_of_self = None
    firls.stypy_type_store = module_type_store
    firls.stypy_function_name = 'firls'
    firls.stypy_param_names_list = ['numtaps', 'bands', 'desired', 'weight', 'nyq', 'fs']
    firls.stypy_varargs_param_name = None
    firls.stypy_kwargs_param_name = None
    firls.stypy_call_defaults = defaults
    firls.stypy_call_varargs = varargs
    firls.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'firls', ['numtaps', 'bands', 'desired', 'weight', 'nyq', 'fs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'firls', localization, ['numtaps', 'bands', 'desired', 'weight', 'nyq', 'fs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'firls(...)' code ##################

    str_266302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, (-1)), 'str', "\n    FIR filter design using least-squares error minimization.\n\n    Calculate the filter coefficients for the linear-phase finite\n    impulse response (FIR) filter which has the best approximation\n    to the desired frequency response described by `bands` and\n    `desired` in the least squares sense (i.e., the integral of the\n    weighted mean-squared error within the specified bands is\n    minimized).\n\n    Parameters\n    ----------\n    numtaps : int\n        The number of taps in the FIR filter.  `numtaps` must be odd.\n    bands : array_like\n        A monotonic nondecreasing sequence containing the band edges in\n        Hz. All elements must be non-negative and less than or equal to\n        the Nyquist frequency given by `nyq`.\n    desired : array_like\n        A sequence the same size as `bands` containing the desired gain\n        at the start and end point of each band.\n    weight : array_like, optional\n        A relative weighting to give to each band region when solving\n        the least squares problem. `weight` has to be half the size of\n        `bands`.\n    nyq : float, optional\n        *Deprecated.  Use `fs` instead.*\n        Nyquist frequency. Each frequency in `bands` must be between 0\n        and `nyq` (inclusive).  Default is 1.\n    fs : float, optional\n        The sampling frequency of the signal. Each frequency in `bands`\n        must be between 0 and ``fs/2`` (inclusive).  Default is 2.\n\n    Returns\n    -------\n    coeffs : ndarray\n        Coefficients of the optimal (in a least squares sense) FIR filter.\n\n    See also\n    --------\n    firwin\n    firwin2\n    minimum_phase\n    remez\n\n    Notes\n    -----\n    This implementation follows the algorithm given in [1]_.\n    As noted there, least squares design has multiple advantages:\n\n        1. Optimal in a least-squares sense.\n        2. Simple, non-iterative method.\n        3. The general solution can obtained by solving a linear\n           system of equations.\n        4. Allows the use of a frequency dependent weighting function.\n\n    This function constructs a Type I linear phase FIR filter, which\n    contains an odd number of `coeffs` satisfying for :math:`n < numtaps`:\n\n    .. math:: coeffs(n) = coeffs(numtaps - 1 - n)\n\n    The odd number of coefficients and filter symmetry avoid boundary\n    conditions that could otherwise occur at the Nyquist and 0 frequencies\n    (e.g., for Type II, III, or IV variants).\n\n    .. versionadded:: 0.18\n\n    References\n    ----------\n    .. [1] Ivan Selesnick, Linear-Phase Fir Filter Design By Least Squares.\n           OpenStax CNX. Aug 9, 2005.\n           http://cnx.org/contents/eb1ecb35-03a9-4610-ba87-41cd771c95f2@7\n\n    Examples\n    --------\n    We want to construct a band-pass filter. Note that the behavior in the\n    frequency ranges between our stop bands and pass bands is unspecified,\n    and thus may overshoot depending on the parameters of our filter:\n\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> fig, axs = plt.subplots(2)\n    >>> fs = 10.0  # Hz\n    >>> desired = (0, 0, 1, 1, 0, 0)\n    >>> for bi, bands in enumerate(((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 4.5, 5))):\n    ...     fir_firls = signal.firls(73, bands, desired, fs=fs)\n    ...     fir_remez = signal.remez(73, bands, desired[::2], fs=fs)\n    ...     fir_firwin2 = signal.firwin2(73, bands, desired, fs=fs)\n    ...     hs = list()\n    ...     ax = axs[bi]\n    ...     for fir in (fir_firls, fir_remez, fir_firwin2):\n    ...         freq, response = signal.freqz(fir)\n    ...         hs.append(ax.semilogy(0.5*fs*freq/np.pi, np.abs(response))[0])\n    ...     for band, gains in zip(zip(bands[::2], bands[1::2]),\n    ...                            zip(desired[::2], desired[1::2])):\n    ...         ax.semilogy(band, np.maximum(gains, 1e-7), 'k--', linewidth=2)\n    ...     if bi == 0:\n    ...         ax.legend(hs, ('firls', 'remez', 'firwin2'),\n    ...                   loc='lower center', frameon=False)\n    ...     else:\n    ...         ax.set_xlabel('Frequency (Hz)')\n    ...     ax.grid(True)\n    ...     ax.set(title='Band-pass %d-%d Hz' % bands[2:4], ylabel='Magnitude')\n    ...\n    >>> fig.tight_layout()\n    >>> plt.show()\n\n    ")
    
    # Assigning a BinOp to a Name (line 861):
    
    # Assigning a BinOp to a Name (line 861):
    float_266303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 10), 'float')
    
    # Call to _get_fs(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'fs' (line 861)
    fs_266305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 24), 'fs', False)
    # Getting the type of 'nyq' (line 861)
    nyq_266306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 28), 'nyq', False)
    # Processing the call keyword arguments (line 861)
    kwargs_266307 = {}
    # Getting the type of '_get_fs' (line 861)
    _get_fs_266304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 16), '_get_fs', False)
    # Calling _get_fs(args, kwargs) (line 861)
    _get_fs_call_result_266308 = invoke(stypy.reporting.localization.Localization(__file__, 861, 16), _get_fs_266304, *[fs_266305, nyq_266306], **kwargs_266307)
    
    # Applying the binary operator '*' (line 861)
    result_mul_266309 = python_operator(stypy.reporting.localization.Localization(__file__, 861, 10), '*', float_266303, _get_fs_call_result_266308)
    
    # Assigning a type to the variable 'nyq' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'nyq', result_mul_266309)
    
    # Assigning a Call to a Name (line 863):
    
    # Assigning a Call to a Name (line 863):
    
    # Call to int(...): (line 863)
    # Processing the call arguments (line 863)
    # Getting the type of 'numtaps' (line 863)
    numtaps_266311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 18), 'numtaps', False)
    # Processing the call keyword arguments (line 863)
    kwargs_266312 = {}
    # Getting the type of 'int' (line 863)
    int_266310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 14), 'int', False)
    # Calling int(args, kwargs) (line 863)
    int_call_result_266313 = invoke(stypy.reporting.localization.Localization(__file__, 863, 14), int_266310, *[numtaps_266311], **kwargs_266312)
    
    # Assigning a type to the variable 'numtaps' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'numtaps', int_call_result_266313)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'numtaps' (line 864)
    numtaps_266314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 7), 'numtaps')
    int_266315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 17), 'int')
    # Applying the binary operator '%' (line 864)
    result_mod_266316 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 7), '%', numtaps_266314, int_266315)
    
    int_266317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 22), 'int')
    # Applying the binary operator '==' (line 864)
    result_eq_266318 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 7), '==', result_mod_266316, int_266317)
    
    
    # Getting the type of 'numtaps' (line 864)
    numtaps_266319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 27), 'numtaps')
    int_266320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 37), 'int')
    # Applying the binary operator '<' (line 864)
    result_lt_266321 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 27), '<', numtaps_266319, int_266320)
    
    # Applying the binary operator 'or' (line 864)
    result_or_keyword_266322 = python_operator(stypy.reporting.localization.Localization(__file__, 864, 7), 'or', result_eq_266318, result_lt_266321)
    
    # Testing the type of an if condition (line 864)
    if_condition_266323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 864, 4), result_or_keyword_266322)
    # Assigning a type to the variable 'if_condition_266323' (line 864)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'if_condition_266323', if_condition_266323)
    # SSA begins for if statement (line 864)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 865)
    # Processing the call arguments (line 865)
    str_266325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 865, 25), 'str', 'numtaps must be odd and >= 1')
    # Processing the call keyword arguments (line 865)
    kwargs_266326 = {}
    # Getting the type of 'ValueError' (line 865)
    ValueError_266324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 865)
    ValueError_call_result_266327 = invoke(stypy.reporting.localization.Localization(__file__, 865, 14), ValueError_266324, *[str_266325], **kwargs_266326)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 865, 8), ValueError_call_result_266327, 'raise parameter', BaseException)
    # SSA join for if statement (line 864)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 866):
    
    # Assigning a BinOp to a Name (line 866):
    # Getting the type of 'numtaps' (line 866)
    numtaps_266328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 9), 'numtaps')
    int_266329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 17), 'int')
    # Applying the binary operator '-' (line 866)
    result_sub_266330 = python_operator(stypy.reporting.localization.Localization(__file__, 866, 9), '-', numtaps_266328, int_266329)
    
    int_266331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 23), 'int')
    # Applying the binary operator '//' (line 866)
    result_floordiv_266332 = python_operator(stypy.reporting.localization.Localization(__file__, 866, 8), '//', result_sub_266330, int_266331)
    
    # Assigning a type to the variable 'M' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), 'M', result_floordiv_266332)
    
    # Assigning a Call to a Name (line 869):
    
    # Assigning a Call to a Name (line 869):
    
    # Call to float(...): (line 869)
    # Processing the call arguments (line 869)
    # Getting the type of 'nyq' (line 869)
    nyq_266334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 16), 'nyq', False)
    # Processing the call keyword arguments (line 869)
    kwargs_266335 = {}
    # Getting the type of 'float' (line 869)
    float_266333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 10), 'float', False)
    # Calling float(args, kwargs) (line 869)
    float_call_result_266336 = invoke(stypy.reporting.localization.Localization(__file__, 869, 10), float_266333, *[nyq_266334], **kwargs_266335)
    
    # Assigning a type to the variable 'nyq' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'nyq', float_call_result_266336)
    
    
    # Getting the type of 'nyq' (line 870)
    nyq_266337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 7), 'nyq')
    int_266338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 14), 'int')
    # Applying the binary operator '<=' (line 870)
    result_le_266339 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 7), '<=', nyq_266337, int_266338)
    
    # Testing the type of an if condition (line 870)
    if_condition_266340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 870, 4), result_le_266339)
    # Assigning a type to the variable 'if_condition_266340' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 4), 'if_condition_266340', if_condition_266340)
    # SSA begins for if statement (line 870)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 871)
    # Processing the call arguments (line 871)
    str_266342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 25), 'str', 'nyq must be positive, got %s <= 0.')
    # Getting the type of 'nyq' (line 871)
    nyq_266343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 64), 'nyq', False)
    # Applying the binary operator '%' (line 871)
    result_mod_266344 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 25), '%', str_266342, nyq_266343)
    
    # Processing the call keyword arguments (line 871)
    kwargs_266345 = {}
    # Getting the type of 'ValueError' (line 871)
    ValueError_266341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 871)
    ValueError_call_result_266346 = invoke(stypy.reporting.localization.Localization(__file__, 871, 14), ValueError_266341, *[result_mod_266344], **kwargs_266345)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 871, 8), ValueError_call_result_266346, 'raise parameter', BaseException)
    # SSA join for if statement (line 870)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 872):
    
    # Assigning a BinOp to a Name (line 872):
    
    # Call to flatten(...): (line 872)
    # Processing the call keyword arguments (line 872)
    kwargs_266353 = {}
    
    # Call to asarray(...): (line 872)
    # Processing the call arguments (line 872)
    # Getting the type of 'bands' (line 872)
    bands_266349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 23), 'bands', False)
    # Processing the call keyword arguments (line 872)
    kwargs_266350 = {}
    # Getting the type of 'np' (line 872)
    np_266347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 872)
    asarray_266348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 12), np_266347, 'asarray')
    # Calling asarray(args, kwargs) (line 872)
    asarray_call_result_266351 = invoke(stypy.reporting.localization.Localization(__file__, 872, 12), asarray_266348, *[bands_266349], **kwargs_266350)
    
    # Obtaining the member 'flatten' of a type (line 872)
    flatten_266352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 12), asarray_call_result_266351, 'flatten')
    # Calling flatten(args, kwargs) (line 872)
    flatten_call_result_266354 = invoke(stypy.reporting.localization.Localization(__file__, 872, 12), flatten_266352, *[], **kwargs_266353)
    
    # Getting the type of 'nyq' (line 872)
    nyq_266355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 42), 'nyq')
    # Applying the binary operator 'div' (line 872)
    result_div_266356 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 12), 'div', flatten_call_result_266354, nyq_266355)
    
    # Assigning a type to the variable 'bands' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 4), 'bands', result_div_266356)
    
    
    
    # Call to len(...): (line 873)
    # Processing the call arguments (line 873)
    # Getting the type of 'bands' (line 873)
    bands_266358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 11), 'bands', False)
    # Processing the call keyword arguments (line 873)
    kwargs_266359 = {}
    # Getting the type of 'len' (line 873)
    len_266357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 7), 'len', False)
    # Calling len(args, kwargs) (line 873)
    len_call_result_266360 = invoke(stypy.reporting.localization.Localization(__file__, 873, 7), len_266357, *[bands_266358], **kwargs_266359)
    
    int_266361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 20), 'int')
    # Applying the binary operator '%' (line 873)
    result_mod_266362 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 7), '%', len_call_result_266360, int_266361)
    
    int_266363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 25), 'int')
    # Applying the binary operator '!=' (line 873)
    result_ne_266364 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 7), '!=', result_mod_266362, int_266363)
    
    # Testing the type of an if condition (line 873)
    if_condition_266365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 873, 4), result_ne_266364)
    # Assigning a type to the variable 'if_condition_266365' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 4), 'if_condition_266365', if_condition_266365)
    # SSA begins for if statement (line 873)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 874)
    # Processing the call arguments (line 874)
    str_266367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 874, 25), 'str', 'bands must contain frequency pairs.')
    # Processing the call keyword arguments (line 874)
    kwargs_266368 = {}
    # Getting the type of 'ValueError' (line 874)
    ValueError_266366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 874)
    ValueError_call_result_266369 = invoke(stypy.reporting.localization.Localization(__file__, 874, 14), ValueError_266366, *[str_266367], **kwargs_266368)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 874, 8), ValueError_call_result_266369, 'raise parameter', BaseException)
    # SSA join for if statement (line 873)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Attribute (line 875):
    
    # Assigning a Tuple to a Attribute (line 875):
    
    # Obtaining an instance of the builtin type 'tuple' (line 875)
    tuple_266370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 875)
    # Adding element type (line 875)
    int_266371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 19), tuple_266370, int_266371)
    # Adding element type (line 875)
    int_266372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 19), tuple_266370, int_266372)
    
    # Getting the type of 'bands' (line 875)
    bands_266373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'bands')
    # Setting the type of the member 'shape' of a type (line 875)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 4), bands_266373, 'shape', tuple_266370)
    
    # Assigning a Call to a Name (line 878):
    
    # Assigning a Call to a Name (line 878):
    
    # Call to flatten(...): (line 878)
    # Processing the call keyword arguments (line 878)
    kwargs_266380 = {}
    
    # Call to asarray(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'desired' (line 878)
    desired_266376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 25), 'desired', False)
    # Processing the call keyword arguments (line 878)
    kwargs_266377 = {}
    # Getting the type of 'np' (line 878)
    np_266374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 878)
    asarray_266375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 14), np_266374, 'asarray')
    # Calling asarray(args, kwargs) (line 878)
    asarray_call_result_266378 = invoke(stypy.reporting.localization.Localization(__file__, 878, 14), asarray_266375, *[desired_266376], **kwargs_266377)
    
    # Obtaining the member 'flatten' of a type (line 878)
    flatten_266379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 878, 14), asarray_call_result_266378, 'flatten')
    # Calling flatten(args, kwargs) (line 878)
    flatten_call_result_266381 = invoke(stypy.reporting.localization.Localization(__file__, 878, 14), flatten_266379, *[], **kwargs_266380)
    
    # Assigning a type to the variable 'desired' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'desired', flatten_call_result_266381)
    
    
    # Getting the type of 'bands' (line 879)
    bands_266382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 7), 'bands')
    # Obtaining the member 'size' of a type (line 879)
    size_266383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 7), bands_266382, 'size')
    # Getting the type of 'desired' (line 879)
    desired_266384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 21), 'desired')
    # Obtaining the member 'size' of a type (line 879)
    size_266385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 21), desired_266384, 'size')
    # Applying the binary operator '!=' (line 879)
    result_ne_266386 = python_operator(stypy.reporting.localization.Localization(__file__, 879, 7), '!=', size_266383, size_266385)
    
    # Testing the type of an if condition (line 879)
    if_condition_266387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 879, 4), result_ne_266386)
    # Assigning a type to the variable 'if_condition_266387' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 4), 'if_condition_266387', if_condition_266387)
    # SSA begins for if statement (line 879)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 880)
    # Processing the call arguments (line 880)
    str_266389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 25), 'str', 'desired must have one entry per frequency, got %s gains for %s frequencies.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 882)
    tuple_266390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 882)
    # Adding element type (line 882)
    # Getting the type of 'desired' (line 882)
    desired_266391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 28), 'desired', False)
    # Obtaining the member 'size' of a type (line 882)
    size_266392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 28), desired_266391, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 882, 28), tuple_266390, size_266392)
    # Adding element type (line 882)
    # Getting the type of 'bands' (line 882)
    bands_266393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 42), 'bands', False)
    # Obtaining the member 'size' of a type (line 882)
    size_266394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 42), bands_266393, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 882, 28), tuple_266390, size_266394)
    
    # Applying the binary operator '%' (line 880)
    result_mod_266395 = python_operator(stypy.reporting.localization.Localization(__file__, 880, 25), '%', str_266389, tuple_266390)
    
    # Processing the call keyword arguments (line 880)
    kwargs_266396 = {}
    # Getting the type of 'ValueError' (line 880)
    ValueError_266388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 880)
    ValueError_call_result_266397 = invoke(stypy.reporting.localization.Localization(__file__, 880, 14), ValueError_266388, *[result_mod_266395], **kwargs_266396)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 880, 8), ValueError_call_result_266397, 'raise parameter', BaseException)
    # SSA join for if statement (line 879)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Attribute (line 883):
    
    # Assigning a Tuple to a Attribute (line 883):
    
    # Obtaining an instance of the builtin type 'tuple' (line 883)
    tuple_266398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 883)
    # Adding element type (line 883)
    int_266399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 21), tuple_266398, int_266399)
    # Adding element type (line 883)
    int_266400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 883, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 21), tuple_266398, int_266400)
    
    # Getting the type of 'desired' (line 883)
    desired_266401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'desired')
    # Setting the type of the member 'shape' of a type (line 883)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 883, 4), desired_266401, 'shape', tuple_266398)
    
    
    # Evaluating a boolean operation
    
    # Call to any(...): (line 884)
    # Processing the call keyword arguments (line 884)
    kwargs_266410 = {}
    
    
    # Call to diff(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'bands' (line 884)
    bands_266404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 16), 'bands', False)
    # Processing the call keyword arguments (line 884)
    kwargs_266405 = {}
    # Getting the type of 'np' (line 884)
    np_266402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 884)
    diff_266403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 8), np_266402, 'diff')
    # Calling diff(args, kwargs) (line 884)
    diff_call_result_266406 = invoke(stypy.reporting.localization.Localization(__file__, 884, 8), diff_266403, *[bands_266404], **kwargs_266405)
    
    int_266407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 26), 'int')
    # Applying the binary operator '<=' (line 884)
    result_le_266408 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 8), '<=', diff_call_result_266406, int_266407)
    
    # Obtaining the member 'any' of a type (line 884)
    any_266409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 8), result_le_266408, 'any')
    # Calling any(args, kwargs) (line 884)
    any_call_result_266411 = invoke(stypy.reporting.localization.Localization(__file__, 884, 8), any_266409, *[], **kwargs_266410)
    
    
    # Call to any(...): (line 884)
    # Processing the call keyword arguments (line 884)
    kwargs_266424 = {}
    
    
    # Call to diff(...): (line 884)
    # Processing the call arguments (line 884)
    
    # Obtaining the type of the subscript
    slice_266414 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 884, 47), None, None, None)
    int_266415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 56), 'int')
    # Getting the type of 'bands' (line 884)
    bands_266416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 47), 'bands', False)
    # Obtaining the member '__getitem__' of a type (line 884)
    getitem___266417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 47), bands_266416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 884)
    subscript_call_result_266418 = invoke(stypy.reporting.localization.Localization(__file__, 884, 47), getitem___266417, (slice_266414, int_266415))
    
    # Processing the call keyword arguments (line 884)
    kwargs_266419 = {}
    # Getting the type of 'np' (line 884)
    np_266412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 39), 'np', False)
    # Obtaining the member 'diff' of a type (line 884)
    diff_266413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 39), np_266412, 'diff')
    # Calling diff(args, kwargs) (line 884)
    diff_call_result_266420 = invoke(stypy.reporting.localization.Localization(__file__, 884, 39), diff_266413, *[subscript_call_result_266418], **kwargs_266419)
    
    int_266421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 62), 'int')
    # Applying the binary operator '<' (line 884)
    result_lt_266422 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 39), '<', diff_call_result_266420, int_266421)
    
    # Obtaining the member 'any' of a type (line 884)
    any_266423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 39), result_lt_266422, 'any')
    # Calling any(args, kwargs) (line 884)
    any_call_result_266425 = invoke(stypy.reporting.localization.Localization(__file__, 884, 39), any_266423, *[], **kwargs_266424)
    
    # Applying the binary operator 'or' (line 884)
    result_or_keyword_266426 = python_operator(stypy.reporting.localization.Localization(__file__, 884, 7), 'or', any_call_result_266411, any_call_result_266425)
    
    # Testing the type of an if condition (line 884)
    if_condition_266427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 884, 4), result_or_keyword_266426)
    # Assigning a type to the variable 'if_condition_266427' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 4), 'if_condition_266427', if_condition_266427)
    # SSA begins for if statement (line 884)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 885)
    # Processing the call arguments (line 885)
    str_266429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 25), 'str', 'bands must be monotonically nondecreasing and have width > 0.')
    # Processing the call keyword arguments (line 885)
    kwargs_266430 = {}
    # Getting the type of 'ValueError' (line 885)
    ValueError_266428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 885)
    ValueError_call_result_266431 = invoke(stypy.reporting.localization.Localization(__file__, 885, 14), ValueError_266428, *[str_266429], **kwargs_266430)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 885, 8), ValueError_call_result_266431, 'raise parameter', BaseException)
    # SSA join for if statement (line 884)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 887)
    # Processing the call keyword arguments (line 887)
    kwargs_266446 = {}
    
    
    # Obtaining the type of the subscript
    int_266432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 15), 'int')
    slice_266433 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 887, 8), None, int_266432, None)
    int_266434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 19), 'int')
    # Getting the type of 'bands' (line 887)
    bands_266435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 8), 'bands', False)
    # Obtaining the member '__getitem__' of a type (line 887)
    getitem___266436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), bands_266435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 887)
    subscript_call_result_266437 = invoke(stypy.reporting.localization.Localization(__file__, 887, 8), getitem___266436, (slice_266433, int_266434))
    
    
    # Obtaining the type of the subscript
    int_266438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 30), 'int')
    slice_266439 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 887, 24), int_266438, None, None)
    int_266440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 887, 34), 'int')
    # Getting the type of 'bands' (line 887)
    bands_266441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 24), 'bands', False)
    # Obtaining the member '__getitem__' of a type (line 887)
    getitem___266442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 24), bands_266441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 887)
    subscript_call_result_266443 = invoke(stypy.reporting.localization.Localization(__file__, 887, 24), getitem___266442, (slice_266439, int_266440))
    
    # Applying the binary operator '>' (line 887)
    result_gt_266444 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 8), '>', subscript_call_result_266437, subscript_call_result_266443)
    
    # Obtaining the member 'any' of a type (line 887)
    any_266445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 8), result_gt_266444, 'any')
    # Calling any(args, kwargs) (line 887)
    any_call_result_266447 = invoke(stypy.reporting.localization.Localization(__file__, 887, 8), any_266445, *[], **kwargs_266446)
    
    # Testing the type of an if condition (line 887)
    if_condition_266448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 887, 4), any_call_result_266447)
    # Assigning a type to the variable 'if_condition_266448' (line 887)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 4), 'if_condition_266448', if_condition_266448)
    # SSA begins for if statement (line 887)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 888)
    # Processing the call arguments (line 888)
    str_266450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 25), 'str', 'bands must not overlap.')
    # Processing the call keyword arguments (line 888)
    kwargs_266451 = {}
    # Getting the type of 'ValueError' (line 888)
    ValueError_266449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 888)
    ValueError_call_result_266452 = invoke(stypy.reporting.localization.Localization(__file__, 888, 14), ValueError_266449, *[str_266450], **kwargs_266451)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 888, 8), ValueError_call_result_266452, 'raise parameter', BaseException)
    # SSA join for if statement (line 887)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 889)
    # Processing the call keyword arguments (line 889)
    kwargs_266457 = {}
    
    # Getting the type of 'desired' (line 889)
    desired_266453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'desired', False)
    int_266454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 18), 'int')
    # Applying the binary operator '<' (line 889)
    result_lt_266455 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 8), '<', desired_266453, int_266454)
    
    # Obtaining the member 'any' of a type (line 889)
    any_266456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 8), result_lt_266455, 'any')
    # Calling any(args, kwargs) (line 889)
    any_call_result_266458 = invoke(stypy.reporting.localization.Localization(__file__, 889, 8), any_266456, *[], **kwargs_266457)
    
    # Testing the type of an if condition (line 889)
    if_condition_266459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 889, 4), any_call_result_266458)
    # Assigning a type to the variable 'if_condition_266459' (line 889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 4), 'if_condition_266459', if_condition_266459)
    # SSA begins for if statement (line 889)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 890)
    # Processing the call arguments (line 890)
    str_266461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 890, 25), 'str', 'desired must be non-negative.')
    # Processing the call keyword arguments (line 890)
    kwargs_266462 = {}
    # Getting the type of 'ValueError' (line 890)
    ValueError_266460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 890)
    ValueError_call_result_266463 = invoke(stypy.reporting.localization.Localization(__file__, 890, 14), ValueError_266460, *[str_266461], **kwargs_266462)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 890, 8), ValueError_call_result_266463, 'raise parameter', BaseException)
    # SSA join for if statement (line 889)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 891)
    # Getting the type of 'weight' (line 891)
    weight_266464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 7), 'weight')
    # Getting the type of 'None' (line 891)
    None_266465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 17), 'None')
    
    (may_be_266466, more_types_in_union_266467) = may_be_none(weight_266464, None_266465)

    if may_be_266466:

        if more_types_in_union_266467:
            # Runtime conditional SSA (line 891)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 892):
        
        # Assigning a Call to a Name (line 892):
        
        # Call to ones(...): (line 892)
        # Processing the call arguments (line 892)
        
        # Call to len(...): (line 892)
        # Processing the call arguments (line 892)
        # Getting the type of 'desired' (line 892)
        desired_266471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 29), 'desired', False)
        # Processing the call keyword arguments (line 892)
        kwargs_266472 = {}
        # Getting the type of 'len' (line 892)
        len_266470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 25), 'len', False)
        # Calling len(args, kwargs) (line 892)
        len_call_result_266473 = invoke(stypy.reporting.localization.Localization(__file__, 892, 25), len_266470, *[desired_266471], **kwargs_266472)
        
        # Processing the call keyword arguments (line 892)
        kwargs_266474 = {}
        # Getting the type of 'np' (line 892)
        np_266468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 17), 'np', False)
        # Obtaining the member 'ones' of a type (line 892)
        ones_266469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 17), np_266468, 'ones')
        # Calling ones(args, kwargs) (line 892)
        ones_call_result_266475 = invoke(stypy.reporting.localization.Localization(__file__, 892, 17), ones_266469, *[len_call_result_266473], **kwargs_266474)
        
        # Assigning a type to the variable 'weight' (line 892)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 8), 'weight', ones_call_result_266475)

        if more_types_in_union_266467:
            # SSA join for if statement (line 891)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 893):
    
    # Assigning a Call to a Name (line 893):
    
    # Call to flatten(...): (line 893)
    # Processing the call keyword arguments (line 893)
    kwargs_266482 = {}
    
    # Call to asarray(...): (line 893)
    # Processing the call arguments (line 893)
    # Getting the type of 'weight' (line 893)
    weight_266478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 24), 'weight', False)
    # Processing the call keyword arguments (line 893)
    kwargs_266479 = {}
    # Getting the type of 'np' (line 893)
    np_266476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 893)
    asarray_266477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 13), np_266476, 'asarray')
    # Calling asarray(args, kwargs) (line 893)
    asarray_call_result_266480 = invoke(stypy.reporting.localization.Localization(__file__, 893, 13), asarray_266477, *[weight_266478], **kwargs_266479)
    
    # Obtaining the member 'flatten' of a type (line 893)
    flatten_266481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 13), asarray_call_result_266480, 'flatten')
    # Calling flatten(args, kwargs) (line 893)
    flatten_call_result_266483 = invoke(stypy.reporting.localization.Localization(__file__, 893, 13), flatten_266481, *[], **kwargs_266482)
    
    # Assigning a type to the variable 'weight' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 4), 'weight', flatten_call_result_266483)
    
    
    
    # Call to len(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'weight' (line 894)
    weight_266485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 11), 'weight', False)
    # Processing the call keyword arguments (line 894)
    kwargs_266486 = {}
    # Getting the type of 'len' (line 894)
    len_266484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 7), 'len', False)
    # Calling len(args, kwargs) (line 894)
    len_call_result_266487 = invoke(stypy.reporting.localization.Localization(__file__, 894, 7), len_266484, *[weight_266485], **kwargs_266486)
    
    
    # Call to len(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'desired' (line 894)
    desired_266489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 26), 'desired', False)
    # Processing the call keyword arguments (line 894)
    kwargs_266490 = {}
    # Getting the type of 'len' (line 894)
    len_266488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 22), 'len', False)
    # Calling len(args, kwargs) (line 894)
    len_call_result_266491 = invoke(stypy.reporting.localization.Localization(__file__, 894, 22), len_266488, *[desired_266489], **kwargs_266490)
    
    # Applying the binary operator '!=' (line 894)
    result_ne_266492 = python_operator(stypy.reporting.localization.Localization(__file__, 894, 7), '!=', len_call_result_266487, len_call_result_266491)
    
    # Testing the type of an if condition (line 894)
    if_condition_266493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 894, 4), result_ne_266492)
    # Assigning a type to the variable 'if_condition_266493' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 4), 'if_condition_266493', if_condition_266493)
    # SSA begins for if statement (line 894)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 895)
    # Processing the call arguments (line 895)
    str_266495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 895, 25), 'str', 'weight must be the same size as the number of band pairs (%s).')
    
    # Obtaining an instance of the builtin type 'tuple' (line 896)
    tuple_266496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 896, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 896)
    # Adding element type (line 896)
    
    # Call to len(...): (line 896)
    # Processing the call arguments (line 896)
    # Getting the type of 'bands' (line 896)
    bands_266498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 51), 'bands', False)
    # Processing the call keyword arguments (line 896)
    kwargs_266499 = {}
    # Getting the type of 'len' (line 896)
    len_266497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 47), 'len', False)
    # Calling len(args, kwargs) (line 896)
    len_call_result_266500 = invoke(stypy.reporting.localization.Localization(__file__, 896, 47), len_266497, *[bands_266498], **kwargs_266499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 896, 47), tuple_266496, len_call_result_266500)
    
    # Applying the binary operator '%' (line 895)
    result_mod_266501 = python_operator(stypy.reporting.localization.Localization(__file__, 895, 25), '%', str_266495, tuple_266496)
    
    # Processing the call keyword arguments (line 895)
    kwargs_266502 = {}
    # Getting the type of 'ValueError' (line 895)
    ValueError_266494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 895, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 895)
    ValueError_call_result_266503 = invoke(stypy.reporting.localization.Localization(__file__, 895, 14), ValueError_266494, *[result_mod_266501], **kwargs_266502)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 895, 8), ValueError_call_result_266503, 'raise parameter', BaseException)
    # SSA join for if statement (line 894)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 897)
    # Processing the call keyword arguments (line 897)
    kwargs_266508 = {}
    
    # Getting the type of 'weight' (line 897)
    weight_266504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'weight', False)
    int_266505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 17), 'int')
    # Applying the binary operator '<' (line 897)
    result_lt_266506 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 8), '<', weight_266504, int_266505)
    
    # Obtaining the member 'any' of a type (line 897)
    any_266507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 8), result_lt_266506, 'any')
    # Calling any(args, kwargs) (line 897)
    any_call_result_266509 = invoke(stypy.reporting.localization.Localization(__file__, 897, 8), any_266507, *[], **kwargs_266508)
    
    # Testing the type of an if condition (line 897)
    if_condition_266510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 897, 4), any_call_result_266509)
    # Assigning a type to the variable 'if_condition_266510' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 4), 'if_condition_266510', if_condition_266510)
    # SSA begins for if statement (line 897)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 898)
    # Processing the call arguments (line 898)
    str_266512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 25), 'str', 'weight must be non-negative.')
    # Processing the call keyword arguments (line 898)
    kwargs_266513 = {}
    # Getting the type of 'ValueError' (line 898)
    ValueError_266511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 898)
    ValueError_call_result_266514 = invoke(stypy.reporting.localization.Localization(__file__, 898, 14), ValueError_266511, *[str_266512], **kwargs_266513)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 898, 8), ValueError_call_result_266514, 'raise parameter', BaseException)
    # SSA join for if statement (line 897)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 917):
    
    # Assigning a Subscript to a Name (line 917):
    
    # Obtaining the type of the subscript
    slice_266515 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 917, 8), None, None, None)
    # Getting the type of 'np' (line 917)
    np_266516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 30), 'np')
    # Obtaining the member 'newaxis' of a type (line 917)
    newaxis_266517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 30), np_266516, 'newaxis')
    # Getting the type of 'np' (line 917)
    np_266518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 42), 'np')
    # Obtaining the member 'newaxis' of a type (line 917)
    newaxis_266519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 42), np_266518, 'newaxis')
    
    # Call to arange(...): (line 917)
    # Processing the call arguments (line 917)
    # Getting the type of 'numtaps' (line 917)
    numtaps_266522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 18), 'numtaps', False)
    # Processing the call keyword arguments (line 917)
    kwargs_266523 = {}
    # Getting the type of 'np' (line 917)
    np_266520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 917)
    arange_266521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 8), np_266520, 'arange')
    # Calling arange(args, kwargs) (line 917)
    arange_call_result_266524 = invoke(stypy.reporting.localization.Localization(__file__, 917, 8), arange_266521, *[numtaps_266522], **kwargs_266523)
    
    # Obtaining the member '__getitem__' of a type (line 917)
    getitem___266525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 917, 8), arange_call_result_266524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 917)
    subscript_call_result_266526 = invoke(stypy.reporting.localization.Localization(__file__, 917, 8), getitem___266525, (slice_266515, newaxis_266517, newaxis_266519))
    
    # Assigning a type to the variable 'n' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 4), 'n', subscript_call_result_266526)
    
    # Assigning a Call to a Name (line 918):
    
    # Assigning a Call to a Name (line 918):
    
    # Call to dot(...): (line 918)
    # Processing the call arguments (line 918)
    
    # Obtaining the type of the subscript
    slice_266529 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 918, 15), None, None, None)
    slice_266530 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 918, 15), None, None, None)
    int_266531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 65), 'int')
    
    # Call to diff(...): (line 918)
    # Processing the call arguments (line 918)
    
    # Call to sinc(...): (line 918)
    # Processing the call arguments (line 918)
    # Getting the type of 'bands' (line 918)
    bands_266536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 31), 'bands', False)
    # Getting the type of 'n' (line 918)
    n_266537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 39), 'n', False)
    # Applying the binary operator '*' (line 918)
    result_mul_266538 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 31), '*', bands_266536, n_266537)
    
    # Processing the call keyword arguments (line 918)
    kwargs_266539 = {}
    # Getting the type of 'np' (line 918)
    np_266534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 23), 'np', False)
    # Obtaining the member 'sinc' of a type (line 918)
    sinc_266535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 23), np_266534, 'sinc')
    # Calling sinc(args, kwargs) (line 918)
    sinc_call_result_266540 = invoke(stypy.reporting.localization.Localization(__file__, 918, 23), sinc_266535, *[result_mul_266538], **kwargs_266539)
    
    # Getting the type of 'bands' (line 918)
    bands_266541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 44), 'bands', False)
    # Applying the binary operator '*' (line 918)
    result_mul_266542 = python_operator(stypy.reporting.localization.Localization(__file__, 918, 23), '*', sinc_call_result_266540, bands_266541)
    
    # Processing the call keyword arguments (line 918)
    int_266543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 56), 'int')
    keyword_266544 = int_266543
    kwargs_266545 = {'axis': keyword_266544}
    # Getting the type of 'np' (line 918)
    np_266532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 15), 'np', False)
    # Obtaining the member 'diff' of a type (line 918)
    diff_266533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 15), np_266532, 'diff')
    # Calling diff(args, kwargs) (line 918)
    diff_call_result_266546 = invoke(stypy.reporting.localization.Localization(__file__, 918, 15), diff_266533, *[result_mul_266542], **kwargs_266545)
    
    # Obtaining the member '__getitem__' of a type (line 918)
    getitem___266547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 15), diff_call_result_266546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 918)
    subscript_call_result_266548 = invoke(stypy.reporting.localization.Localization(__file__, 918, 15), getitem___266547, (slice_266529, slice_266530, int_266531))
    
    # Getting the type of 'weight' (line 918)
    weight_266549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 69), 'weight', False)
    # Processing the call keyword arguments (line 918)
    kwargs_266550 = {}
    # Getting the type of 'np' (line 918)
    np_266527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 918)
    dot_266528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 8), np_266527, 'dot')
    # Calling dot(args, kwargs) (line 918)
    dot_call_result_266551 = invoke(stypy.reporting.localization.Localization(__file__, 918, 8), dot_266528, *[subscript_call_result_266548, weight_266549], **kwargs_266550)
    
    # Assigning a type to the variable 'q' (line 918)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 4), 'q', dot_call_result_266551)
    
    # Assigning a Call to a Name (line 921):
    
    # Assigning a Call to a Name (line 921):
    
    # Call to toeplitz(...): (line 921)
    # Processing the call arguments (line 921)
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 921)
    M_266553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 21), 'M', False)
    int_266554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 23), 'int')
    # Applying the binary operator '+' (line 921)
    result_add_266555 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 21), '+', M_266553, int_266554)
    
    slice_266556 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 921, 18), None, result_add_266555, None)
    # Getting the type of 'q' (line 921)
    q_266557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 18), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 921)
    getitem___266558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 18), q_266557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 921)
    subscript_call_result_266559 = invoke(stypy.reporting.localization.Localization(__file__, 921, 18), getitem___266558, slice_266556)
    
    # Processing the call keyword arguments (line 921)
    kwargs_266560 = {}
    # Getting the type of 'toeplitz' (line 921)
    toeplitz_266552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 9), 'toeplitz', False)
    # Calling toeplitz(args, kwargs) (line 921)
    toeplitz_call_result_266561 = invoke(stypy.reporting.localization.Localization(__file__, 921, 9), toeplitz_266552, *[subscript_call_result_266559], **kwargs_266560)
    
    # Assigning a type to the variable 'Q1' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'Q1', toeplitz_call_result_266561)
    
    # Assigning a Call to a Name (line 922):
    
    # Assigning a Call to a Name (line 922):
    
    # Call to hankel(...): (line 922)
    # Processing the call arguments (line 922)
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 922)
    M_266563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 19), 'M', False)
    int_266564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 21), 'int')
    # Applying the binary operator '+' (line 922)
    result_add_266565 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 19), '+', M_266563, int_266564)
    
    slice_266566 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 922, 16), None, result_add_266565, None)
    # Getting the type of 'q' (line 922)
    q_266567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 16), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 922)
    getitem___266568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 16), q_266567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 922)
    subscript_call_result_266569 = invoke(stypy.reporting.localization.Localization(__file__, 922, 16), getitem___266568, slice_266566)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 922)
    M_266570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 27), 'M', False)
    slice_266571 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 922, 25), M_266570, None, None)
    # Getting the type of 'q' (line 922)
    q_266572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 25), 'q', False)
    # Obtaining the member '__getitem__' of a type (line 922)
    getitem___266573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 25), q_266572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 922)
    subscript_call_result_266574 = invoke(stypy.reporting.localization.Localization(__file__, 922, 25), getitem___266573, slice_266571)
    
    # Processing the call keyword arguments (line 922)
    kwargs_266575 = {}
    # Getting the type of 'hankel' (line 922)
    hankel_266562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 9), 'hankel', False)
    # Calling hankel(args, kwargs) (line 922)
    hankel_call_result_266576 = invoke(stypy.reporting.localization.Localization(__file__, 922, 9), hankel_266562, *[subscript_call_result_266569, subscript_call_result_266574], **kwargs_266575)
    
    # Assigning a type to the variable 'Q2' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'Q2', hankel_call_result_266576)
    
    # Assigning a BinOp to a Name (line 923):
    
    # Assigning a BinOp to a Name (line 923):
    # Getting the type of 'Q1' (line 923)
    Q1_266577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'Q1')
    # Getting the type of 'Q2' (line 923)
    Q2_266578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 13), 'Q2')
    # Applying the binary operator '+' (line 923)
    result_add_266579 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 8), '+', Q1_266577, Q2_266578)
    
    # Assigning a type to the variable 'Q' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'Q', result_add_266579)
    
    # Assigning a Subscript to a Name (line 932):
    
    # Assigning a Subscript to a Name (line 932):
    
    # Obtaining the type of the subscript
    # Getting the type of 'M' (line 932)
    M_266580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 11), 'M')
    int_266581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 15), 'int')
    # Applying the binary operator '+' (line 932)
    result_add_266582 = python_operator(stypy.reporting.localization.Localization(__file__, 932, 11), '+', M_266580, int_266581)
    
    slice_266583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 932, 8), None, result_add_266582, None)
    # Getting the type of 'n' (line 932)
    n_266584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'n')
    # Obtaining the member '__getitem__' of a type (line 932)
    getitem___266585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 8), n_266584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 932)
    subscript_call_result_266586 = invoke(stypy.reporting.localization.Localization(__file__, 932, 8), getitem___266585, slice_266583)
    
    # Assigning a type to the variable 'n' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 4), 'n', subscript_call_result_266586)
    
    # Assigning a BinOp to a Name (line 934):
    
    # Assigning a BinOp to a Name (line 934):
    
    # Call to diff(...): (line 934)
    # Processing the call arguments (line 934)
    # Getting the type of 'desired' (line 934)
    desired_266589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 17), 'desired', False)
    # Processing the call keyword arguments (line 934)
    int_266590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 31), 'int')
    keyword_266591 = int_266590
    kwargs_266592 = {'axis': keyword_266591}
    # Getting the type of 'np' (line 934)
    np_266587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 9), 'np', False)
    # Obtaining the member 'diff' of a type (line 934)
    diff_266588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 9), np_266587, 'diff')
    # Calling diff(args, kwargs) (line 934)
    diff_call_result_266593 = invoke(stypy.reporting.localization.Localization(__file__, 934, 9), diff_266588, *[desired_266589], **kwargs_266592)
    
    
    # Call to diff(...): (line 934)
    # Processing the call arguments (line 934)
    # Getting the type of 'bands' (line 934)
    bands_266596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 44), 'bands', False)
    # Processing the call keyword arguments (line 934)
    int_266597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 56), 'int')
    keyword_266598 = int_266597
    kwargs_266599 = {'axis': keyword_266598}
    # Getting the type of 'np' (line 934)
    np_266594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 36), 'np', False)
    # Obtaining the member 'diff' of a type (line 934)
    diff_266595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 36), np_266594, 'diff')
    # Calling diff(args, kwargs) (line 934)
    diff_call_result_266600 = invoke(stypy.reporting.localization.Localization(__file__, 934, 36), diff_266595, *[bands_266596], **kwargs_266599)
    
    # Applying the binary operator 'div' (line 934)
    result_div_266601 = python_operator(stypy.reporting.localization.Localization(__file__, 934, 9), 'div', diff_call_result_266593, diff_call_result_266600)
    
    # Assigning a type to the variable 'm' (line 934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 4), 'm', result_div_266601)
    
    # Assigning a BinOp to a Name (line 935):
    
    # Assigning a BinOp to a Name (line 935):
    
    # Obtaining the type of the subscript
    slice_266602 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 935, 8), None, None, None)
    
    # Obtaining an instance of the builtin type 'list' (line 935)
    list_266603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 935)
    # Adding element type (line 935)
    int_266604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 935, 19), list_266603, int_266604)
    
    # Getting the type of 'desired' (line 935)
    desired_266605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'desired')
    # Obtaining the member '__getitem__' of a type (line 935)
    getitem___266606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 8), desired_266605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 935)
    subscript_call_result_266607 = invoke(stypy.reporting.localization.Localization(__file__, 935, 8), getitem___266606, (slice_266602, list_266603))
    
    
    # Obtaining the type of the subscript
    slice_266608 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 935, 26), None, None, None)
    
    # Obtaining an instance of the builtin type 'list' (line 935)
    list_266609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 935)
    # Adding element type (line 935)
    int_266610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 935, 35), list_266609, int_266610)
    
    # Getting the type of 'bands' (line 935)
    bands_266611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 26), 'bands')
    # Obtaining the member '__getitem__' of a type (line 935)
    getitem___266612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 26), bands_266611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 935)
    subscript_call_result_266613 = invoke(stypy.reporting.localization.Localization(__file__, 935, 26), getitem___266612, (slice_266608, list_266609))
    
    # Getting the type of 'm' (line 935)
    m_266614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 42), 'm')
    # Applying the binary operator '*' (line 935)
    result_mul_266615 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 26), '*', subscript_call_result_266613, m_266614)
    
    # Applying the binary operator '-' (line 935)
    result_sub_266616 = python_operator(stypy.reporting.localization.Localization(__file__, 935, 8), '-', subscript_call_result_266607, result_mul_266615)
    
    # Assigning a type to the variable 'c' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 4), 'c', result_sub_266616)
    
    # Assigning a BinOp to a Name (line 936):
    
    # Assigning a BinOp to a Name (line 936):
    # Getting the type of 'bands' (line 936)
    bands_266617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'bands')
    # Getting the type of 'm' (line 936)
    m_266618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 17), 'm')
    # Getting the type of 'bands' (line 936)
    bands_266619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 19), 'bands')
    # Applying the binary operator '*' (line 936)
    result_mul_266620 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 17), '*', m_266618, bands_266619)
    
    # Getting the type of 'c' (line 936)
    c_266621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 27), 'c')
    # Applying the binary operator '+' (line 936)
    result_add_266622 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 17), '+', result_mul_266620, c_266621)
    
    # Applying the binary operator '*' (line 936)
    result_mul_266623 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 8), '*', bands_266617, result_add_266622)
    
    
    # Call to sinc(...): (line 936)
    # Processing the call arguments (line 936)
    # Getting the type of 'bands' (line 936)
    bands_266626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 40), 'bands', False)
    # Getting the type of 'n' (line 936)
    n_266627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 48), 'n', False)
    # Applying the binary operator '*' (line 936)
    result_mul_266628 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 40), '*', bands_266626, n_266627)
    
    # Processing the call keyword arguments (line 936)
    kwargs_266629 = {}
    # Getting the type of 'np' (line 936)
    np_266624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 32), 'np', False)
    # Obtaining the member 'sinc' of a type (line 936)
    sinc_266625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 32), np_266624, 'sinc')
    # Calling sinc(args, kwargs) (line 936)
    sinc_call_result_266630 = invoke(stypy.reporting.localization.Localization(__file__, 936, 32), sinc_266625, *[result_mul_266628], **kwargs_266629)
    
    # Applying the binary operator '*' (line 936)
    result_mul_266631 = python_operator(stypy.reporting.localization.Localization(__file__, 936, 30), '*', result_mul_266623, sinc_call_result_266630)
    
    # Assigning a type to the variable 'b' (line 936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 4), 'b', result_mul_266631)
    
    # Getting the type of 'b' (line 938)
    b_266632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'b')
    
    # Obtaining the type of the subscript
    int_266633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 6), 'int')
    # Getting the type of 'b' (line 938)
    b_266634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'b')
    # Obtaining the member '__getitem__' of a type (line 938)
    getitem___266635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 4), b_266634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 938)
    subscript_call_result_266636 = invoke(stypy.reporting.localization.Localization(__file__, 938, 4), getitem___266635, int_266633)
    
    # Getting the type of 'm' (line 938)
    m_266637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 12), 'm')
    # Getting the type of 'bands' (line 938)
    bands_266638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 16), 'bands')
    # Applying the binary operator '*' (line 938)
    result_mul_266639 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 12), '*', m_266637, bands_266638)
    
    # Getting the type of 'bands' (line 938)
    bands_266640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 24), 'bands')
    # Applying the binary operator '*' (line 938)
    result_mul_266641 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 22), '*', result_mul_266639, bands_266640)
    
    float_266642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 32), 'float')
    # Applying the binary operator 'div' (line 938)
    result_div_266643 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 30), 'div', result_mul_266641, float_266642)
    
    # Applying the binary operator '-=' (line 938)
    result_isub_266644 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 4), '-=', subscript_call_result_266636, result_div_266643)
    # Getting the type of 'b' (line 938)
    b_266645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 4), 'b')
    int_266646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 6), 'int')
    # Storing an element on a container (line 938)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 4), b_266645, (int_266646, result_isub_266644))
    
    
    # Getting the type of 'b' (line 939)
    b_266647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'b')
    
    # Obtaining the type of the subscript
    int_266648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 6), 'int')
    slice_266649 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 939, 4), int_266648, None, None)
    # Getting the type of 'b' (line 939)
    b_266650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'b')
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___266651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 4), b_266650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_266652 = invoke(stypy.reporting.localization.Localization(__file__, 939, 4), getitem___266651, slice_266649)
    
    # Getting the type of 'm' (line 939)
    m_266653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 13), 'm')
    
    # Call to cos(...): (line 939)
    # Processing the call arguments (line 939)
    
    # Obtaining the type of the subscript
    int_266656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 26), 'int')
    slice_266657 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 939, 24), int_266656, None, None)
    # Getting the type of 'n' (line 939)
    n_266658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 24), 'n', False)
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___266659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 24), n_266658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_266660 = invoke(stypy.reporting.localization.Localization(__file__, 939, 24), getitem___266659, slice_266657)
    
    # Getting the type of 'np' (line 939)
    np_266661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 32), 'np', False)
    # Obtaining the member 'pi' of a type (line 939)
    pi_266662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 32), np_266661, 'pi')
    # Applying the binary operator '*' (line 939)
    result_mul_266663 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 24), '*', subscript_call_result_266660, pi_266662)
    
    # Getting the type of 'bands' (line 939)
    bands_266664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 40), 'bands', False)
    # Applying the binary operator '*' (line 939)
    result_mul_266665 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 38), '*', result_mul_266663, bands_266664)
    
    # Processing the call keyword arguments (line 939)
    kwargs_266666 = {}
    # Getting the type of 'np' (line 939)
    np_266654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 17), 'np', False)
    # Obtaining the member 'cos' of a type (line 939)
    cos_266655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 17), np_266654, 'cos')
    # Calling cos(args, kwargs) (line 939)
    cos_call_result_266667 = invoke(stypy.reporting.localization.Localization(__file__, 939, 17), cos_266655, *[result_mul_266665], **kwargs_266666)
    
    # Applying the binary operator '*' (line 939)
    result_mul_266668 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 13), '*', m_266653, cos_call_result_266667)
    
    # Getting the type of 'np' (line 939)
    np_266669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 50), 'np')
    # Obtaining the member 'pi' of a type (line 939)
    pi_266670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 50), np_266669, 'pi')
    
    # Obtaining the type of the subscript
    int_266671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 60), 'int')
    slice_266672 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 939, 58), int_266671, None, None)
    # Getting the type of 'n' (line 939)
    n_266673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 58), 'n')
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___266674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 58), n_266673, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_266675 = invoke(stypy.reporting.localization.Localization(__file__, 939, 58), getitem___266674, slice_266672)
    
    # Applying the binary operator '*' (line 939)
    result_mul_266676 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 50), '*', pi_266670, subscript_call_result_266675)
    
    int_266677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 68), 'int')
    # Applying the binary operator '**' (line 939)
    result_pow_266678 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 49), '**', result_mul_266676, int_266677)
    
    # Applying the binary operator 'div' (line 939)
    result_div_266679 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 47), 'div', result_mul_266668, result_pow_266678)
    
    # Applying the binary operator '+=' (line 939)
    result_iadd_266680 = python_operator(stypy.reporting.localization.Localization(__file__, 939, 4), '+=', subscript_call_result_266652, result_div_266679)
    # Getting the type of 'b' (line 939)
    b_266681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'b')
    int_266682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 6), 'int')
    slice_266683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 939, 4), int_266682, None, None)
    # Storing an element on a container (line 939)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 4), b_266681, (slice_266683, result_iadd_266680))
    
    
    # Assigning a Call to a Name (line 940):
    
    # Assigning a Call to a Name (line 940):
    
    # Call to dot(...): (line 940)
    # Processing the call arguments (line 940)
    
    # Obtaining the type of the subscript
    slice_266686 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 940, 15), None, None, None)
    slice_266687 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 940, 15), None, None, None)
    int_266688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 40), 'int')
    
    # Call to diff(...): (line 940)
    # Processing the call arguments (line 940)
    # Getting the type of 'b' (line 940)
    b_266691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 23), 'b', False)
    # Processing the call keyword arguments (line 940)
    int_266692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 31), 'int')
    keyword_266693 = int_266692
    kwargs_266694 = {'axis': keyword_266693}
    # Getting the type of 'np' (line 940)
    np_266689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 15), 'np', False)
    # Obtaining the member 'diff' of a type (line 940)
    diff_266690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 15), np_266689, 'diff')
    # Calling diff(args, kwargs) (line 940)
    diff_call_result_266695 = invoke(stypy.reporting.localization.Localization(__file__, 940, 15), diff_266690, *[b_266691], **kwargs_266694)
    
    # Obtaining the member '__getitem__' of a type (line 940)
    getitem___266696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 15), diff_call_result_266695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 940)
    subscript_call_result_266697 = invoke(stypy.reporting.localization.Localization(__file__, 940, 15), getitem___266696, (slice_266686, slice_266687, int_266688))
    
    # Getting the type of 'weight' (line 940)
    weight_266698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 44), 'weight', False)
    # Processing the call keyword arguments (line 940)
    kwargs_266699 = {}
    # Getting the type of 'np' (line 940)
    np_266684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 940)
    dot_266685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 940, 8), np_266684, 'dot')
    # Calling dot(args, kwargs) (line 940)
    dot_call_result_266700 = invoke(stypy.reporting.localization.Localization(__file__, 940, 8), dot_266685, *[subscript_call_result_266697, weight_266698], **kwargs_266699)
    
    # Assigning a type to the variable 'b' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'b', dot_call_result_266700)
    
    # Assigning a Call to a Name (line 943):
    
    # Assigning a Call to a Name (line 943):
    
    # Call to dot(...): (line 943)
    # Processing the call arguments (line 943)
    
    # Call to pinv(...): (line 943)
    # Processing the call arguments (line 943)
    # Getting the type of 'Q' (line 943)
    Q_266704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 20), 'Q', False)
    # Processing the call keyword arguments (line 943)
    kwargs_266705 = {}
    # Getting the type of 'pinv' (line 943)
    pinv_266703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 15), 'pinv', False)
    # Calling pinv(args, kwargs) (line 943)
    pinv_call_result_266706 = invoke(stypy.reporting.localization.Localization(__file__, 943, 15), pinv_266703, *[Q_266704], **kwargs_266705)
    
    # Getting the type of 'b' (line 943)
    b_266707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 24), 'b', False)
    # Processing the call keyword arguments (line 943)
    kwargs_266708 = {}
    # Getting the type of 'np' (line 943)
    np_266701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 943)
    dot_266702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 8), np_266701, 'dot')
    # Calling dot(args, kwargs) (line 943)
    dot_call_result_266709 = invoke(stypy.reporting.localization.Localization(__file__, 943, 8), dot_266702, *[pinv_call_result_266706, b_266707], **kwargs_266708)
    
    # Assigning a type to the variable 'a' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 4), 'a', dot_call_result_266709)
    
    # Assigning a Call to a Name (line 946):
    
    # Assigning a Call to a Name (line 946):
    
    # Call to hstack(...): (line 946)
    # Processing the call arguments (line 946)
    
    # Obtaining an instance of the builtin type 'tuple' (line 946)
    tuple_266712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 946)
    # Adding element type (line 946)
    
    # Obtaining the type of the subscript
    int_266713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 27), 'int')
    int_266714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 29), 'int')
    slice_266715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 946, 24), None, int_266713, int_266714)
    # Getting the type of 'a' (line 946)
    a_266716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 24), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 946)
    getitem___266717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 24), a_266716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 946)
    subscript_call_result_266718 = invoke(stypy.reporting.localization.Localization(__file__, 946, 24), getitem___266717, slice_266715)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 946, 24), tuple_266712, subscript_call_result_266718)
    # Adding element type (line 946)
    int_266719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 34), 'int')
    
    # Obtaining the type of the subscript
    int_266720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 40), 'int')
    # Getting the type of 'a' (line 946)
    a_266721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 38), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 946)
    getitem___266722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 38), a_266721, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 946)
    subscript_call_result_266723 = invoke(stypy.reporting.localization.Localization(__file__, 946, 38), getitem___266722, int_266720)
    
    # Applying the binary operator '*' (line 946)
    result_mul_266724 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 34), '*', int_266719, subscript_call_result_266723)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 946, 24), tuple_266712, result_mul_266724)
    # Adding element type (line 946)
    
    # Obtaining the type of the subscript
    int_266725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 46), 'int')
    slice_266726 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 946, 44), int_266725, None, None)
    # Getting the type of 'a' (line 946)
    a_266727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 44), 'a', False)
    # Obtaining the member '__getitem__' of a type (line 946)
    getitem___266728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 44), a_266727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 946)
    subscript_call_result_266729 = invoke(stypy.reporting.localization.Localization(__file__, 946, 44), getitem___266728, slice_266726)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 946, 24), tuple_266712, subscript_call_result_266729)
    
    # Processing the call keyword arguments (line 946)
    kwargs_266730 = {}
    # Getting the type of 'np' (line 946)
    np_266710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 13), 'np', False)
    # Obtaining the member 'hstack' of a type (line 946)
    hstack_266711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 946, 13), np_266710, 'hstack')
    # Calling hstack(args, kwargs) (line 946)
    hstack_call_result_266731 = invoke(stypy.reporting.localization.Localization(__file__, 946, 13), hstack_266711, *[tuple_266712], **kwargs_266730)
    
    # Assigning a type to the variable 'coeffs' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 4), 'coeffs', hstack_call_result_266731)
    # Getting the type of 'coeffs' (line 947)
    coeffs_266732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 11), 'coeffs')
    # Assigning a type to the variable 'stypy_return_type' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'stypy_return_type', coeffs_266732)
    
    # ################# End of 'firls(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'firls' in the type store
    # Getting the type of 'stypy_return_type' (line 752)
    stypy_return_type_266733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_266733)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'firls'
    return stypy_return_type_266733

# Assigning a type to the variable 'firls' (line 752)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 0), 'firls', firls)

@norecursion
def _dhtm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_dhtm'
    module_type_store = module_type_store.open_function_context('_dhtm', 950, 0, False)
    
    # Passed parameters checking function
    _dhtm.stypy_localization = localization
    _dhtm.stypy_type_of_self = None
    _dhtm.stypy_type_store = module_type_store
    _dhtm.stypy_function_name = '_dhtm'
    _dhtm.stypy_param_names_list = ['mag']
    _dhtm.stypy_varargs_param_name = None
    _dhtm.stypy_kwargs_param_name = None
    _dhtm.stypy_call_defaults = defaults
    _dhtm.stypy_call_varargs = varargs
    _dhtm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dhtm', ['mag'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dhtm', localization, ['mag'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dhtm(...)' code ##################

    str_266734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 958, (-1)), 'str', 'Compute the modified 1D discrete Hilbert transform\n\n    Parameters\n    ----------\n    mag : ndarray\n        The magnitude spectrum. Should be 1D with an even length, and\n        preferably a fast length for FFT/IFFT.\n    ')
    
    # Assigning a Call to a Name (line 961):
    
    # Assigning a Call to a Name (line 961):
    
    # Call to zeros(...): (line 961)
    # Processing the call arguments (line 961)
    
    # Call to len(...): (line 961)
    # Processing the call arguments (line 961)
    # Getting the type of 'mag' (line 961)
    mag_266738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 23), 'mag', False)
    # Processing the call keyword arguments (line 961)
    kwargs_266739 = {}
    # Getting the type of 'len' (line 961)
    len_266737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 19), 'len', False)
    # Calling len(args, kwargs) (line 961)
    len_call_result_266740 = invoke(stypy.reporting.localization.Localization(__file__, 961, 19), len_266737, *[mag_266738], **kwargs_266739)
    
    # Processing the call keyword arguments (line 961)
    kwargs_266741 = {}
    # Getting the type of 'np' (line 961)
    np_266735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 961)
    zeros_266736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 10), np_266735, 'zeros')
    # Calling zeros(args, kwargs) (line 961)
    zeros_call_result_266742 = invoke(stypy.reporting.localization.Localization(__file__, 961, 10), zeros_266736, *[len_call_result_266740], **kwargs_266741)
    
    # Assigning a type to the variable 'sig' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'sig', zeros_call_result_266742)
    
    # Assigning a BinOp to a Name (line 963):
    
    # Assigning a BinOp to a Name (line 963):
    
    # Call to len(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'mag' (line 963)
    mag_266744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 16), 'mag', False)
    # Processing the call keyword arguments (line 963)
    kwargs_266745 = {}
    # Getting the type of 'len' (line 963)
    len_266743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 12), 'len', False)
    # Calling len(args, kwargs) (line 963)
    len_call_result_266746 = invoke(stypy.reporting.localization.Localization(__file__, 963, 12), len_266743, *[mag_266744], **kwargs_266745)
    
    int_266747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 24), 'int')
    # Applying the binary operator '//' (line 963)
    result_floordiv_266748 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 12), '//', len_call_result_266746, int_266747)
    
    # Assigning a type to the variable 'midpt' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'midpt', result_floordiv_266748)
    
    # Assigning a Num to a Subscript (line 964):
    
    # Assigning a Num to a Subscript (line 964):
    int_266749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 19), 'int')
    # Getting the type of 'sig' (line 964)
    sig_266750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'sig')
    int_266751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 8), 'int')
    # Getting the type of 'midpt' (line 964)
    midpt_266752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 10), 'midpt')
    slice_266753 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 964, 4), int_266751, midpt_266752, None)
    # Storing an element on a container (line 964)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 4), sig_266750, (slice_266753, int_266749))
    
    # Assigning a Num to a Subscript (line 965):
    
    # Assigning a Num to a Subscript (line 965):
    int_266754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 20), 'int')
    # Getting the type of 'sig' (line 965)
    sig_266755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'sig')
    # Getting the type of 'midpt' (line 965)
    midpt_266756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'midpt')
    int_266757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 14), 'int')
    # Applying the binary operator '+' (line 965)
    result_add_266758 = python_operator(stypy.reporting.localization.Localization(__file__, 965, 8), '+', midpt_266756, int_266757)
    
    slice_266759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 965, 4), result_add_266758, None, None)
    # Storing an element on a container (line 965)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 965, 4), sig_266755, (slice_266759, int_266754))
    
    # Assigning a Attribute to a Name (line 968):
    
    # Assigning a Attribute to a Name (line 968):
    
    # Call to ifft(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'mag' (line 968)
    mag_266761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 17), 'mag', False)
    
    # Call to exp(...): (line 968)
    # Processing the call arguments (line 968)
    
    # Call to fft(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'sig' (line 968)
    sig_266765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 34), 'sig', False)
    
    # Call to ifft(...): (line 968)
    # Processing the call arguments (line 968)
    
    # Call to log(...): (line 968)
    # Processing the call arguments (line 968)
    # Getting the type of 'mag' (line 968)
    mag_266769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 52), 'mag', False)
    # Processing the call keyword arguments (line 968)
    kwargs_266770 = {}
    # Getting the type of 'np' (line 968)
    np_266767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 45), 'np', False)
    # Obtaining the member 'log' of a type (line 968)
    log_266768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 45), np_266767, 'log')
    # Calling log(args, kwargs) (line 968)
    log_call_result_266771 = invoke(stypy.reporting.localization.Localization(__file__, 968, 45), log_266768, *[mag_266769], **kwargs_266770)
    
    # Processing the call keyword arguments (line 968)
    kwargs_266772 = {}
    # Getting the type of 'ifft' (line 968)
    ifft_266766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 40), 'ifft', False)
    # Calling ifft(args, kwargs) (line 968)
    ifft_call_result_266773 = invoke(stypy.reporting.localization.Localization(__file__, 968, 40), ifft_266766, *[log_call_result_266771], **kwargs_266772)
    
    # Applying the binary operator '*' (line 968)
    result_mul_266774 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 34), '*', sig_266765, ifft_call_result_266773)
    
    # Processing the call keyword arguments (line 968)
    kwargs_266775 = {}
    # Getting the type of 'fft' (line 968)
    fft_266764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 30), 'fft', False)
    # Calling fft(args, kwargs) (line 968)
    fft_call_result_266776 = invoke(stypy.reporting.localization.Localization(__file__, 968, 30), fft_266764, *[result_mul_266774], **kwargs_266775)
    
    # Processing the call keyword arguments (line 968)
    kwargs_266777 = {}
    # Getting the type of 'np' (line 968)
    np_266762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 23), 'np', False)
    # Obtaining the member 'exp' of a type (line 968)
    exp_266763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 23), np_266762, 'exp')
    # Calling exp(args, kwargs) (line 968)
    exp_call_result_266778 = invoke(stypy.reporting.localization.Localization(__file__, 968, 23), exp_266763, *[fft_call_result_266776], **kwargs_266777)
    
    # Applying the binary operator '*' (line 968)
    result_mul_266779 = python_operator(stypy.reporting.localization.Localization(__file__, 968, 17), '*', mag_266761, exp_call_result_266778)
    
    # Processing the call keyword arguments (line 968)
    kwargs_266780 = {}
    # Getting the type of 'ifft' (line 968)
    ifft_266760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 968, 12), 'ifft', False)
    # Calling ifft(args, kwargs) (line 968)
    ifft_call_result_266781 = invoke(stypy.reporting.localization.Localization(__file__, 968, 12), ifft_266760, *[result_mul_266779], **kwargs_266780)
    
    # Obtaining the member 'real' of a type (line 968)
    real_266782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 968, 12), ifft_call_result_266781, 'real')
    # Assigning a type to the variable 'recon' (line 968)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 968, 4), 'recon', real_266782)
    # Getting the type of 'recon' (line 969)
    recon_266783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 11), 'recon')
    # Assigning a type to the variable 'stypy_return_type' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 4), 'stypy_return_type', recon_266783)
    
    # ################# End of '_dhtm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dhtm' in the type store
    # Getting the type of 'stypy_return_type' (line 950)
    stypy_return_type_266784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_266784)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dhtm'
    return stypy_return_type_266784

# Assigning a type to the variable '_dhtm' (line 950)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 0), '_dhtm', _dhtm)

@norecursion
def minimum_phase(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_266785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 28), 'str', 'homomorphic')
    # Getting the type of 'None' (line 972)
    None_266786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 49), 'None')
    defaults = [str_266785, None_266786]
    # Create a new context for function 'minimum_phase'
    module_type_store = module_type_store.open_function_context('minimum_phase', 972, 0, False)
    
    # Passed parameters checking function
    minimum_phase.stypy_localization = localization
    minimum_phase.stypy_type_of_self = None
    minimum_phase.stypy_type_store = module_type_store
    minimum_phase.stypy_function_name = 'minimum_phase'
    minimum_phase.stypy_param_names_list = ['h', 'method', 'n_fft']
    minimum_phase.stypy_varargs_param_name = None
    minimum_phase.stypy_kwargs_param_name = None
    minimum_phase.stypy_call_defaults = defaults
    minimum_phase.stypy_call_varargs = varargs
    minimum_phase.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'minimum_phase', ['h', 'method', 'n_fft'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'minimum_phase', localization, ['h', 'method', 'n_fft'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'minimum_phase(...)' code ##################

    str_266787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1093, (-1)), 'str', 'Convert a linear-phase FIR filter to minimum phase\n\n    Parameters\n    ----------\n    h : array\n        Linear-phase FIR filter coefficients.\n    method : {\'hilbert\', \'homomorphic\'}\n        The method to use:\n\n            \'homomorphic\' (default)\n                This method [4]_ [5]_ works best with filters with an\n                odd number of taps, and the resulting minimum phase filter\n                will have a magnitude response that approximates the square\n                root of the the original filter\'s magnitude response.\n\n            \'hilbert\'\n                This method [1]_ is designed to be used with equiripple\n                filters (e.g., from `remez`) with unity or zero gain\n                regions.\n\n    n_fft : int\n        The number of points to use for the FFT. Should be at least a\n        few times larger than the signal length (see Notes).\n\n    Returns\n    -------\n    h_minimum : array\n        The minimum-phase version of the filter, with length\n        ``(length(h) + 1) // 2``.\n\n    See Also\n    --------\n    firwin\n    firwin2\n    remez\n\n    Notes\n    -----\n    Both the Hilbert [1]_ or homomorphic [4]_ [5]_ methods require selection\n    of an FFT length to estimate the complex cepstrum of the filter.\n\n    In the case of the Hilbert method, the deviation from the ideal\n    spectrum ``epsilon`` is related to the number of stopband zeros\n    ``n_stop`` and FFT length ``n_fft`` as::\n\n        epsilon = 2. * n_stop / n_fft\n\n    For example, with 100 stopband zeros and a FFT length of 2048,\n    ``epsilon = 0.0976``. If we conservatively assume that the number of\n    stopband zeros is one less than the filter length, we can take the FFT\n    length to be the next power of 2 that satisfies ``epsilon=0.01`` as::\n\n        n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))\n\n    This gives reasonable results for both the Hilbert and homomorphic\n    methods, and gives the value used when ``n_fft=None``.\n\n    Alternative implementations exist for creating minimum-phase filters,\n    including zero inversion [2]_ and spectral factorization [3]_ [4]_.\n    For more information, see:\n\n        http://dspguru.com/dsp/howtos/how-to-design-minimum-phase-fir-filters\n\n    Examples\n    --------\n    Create an optimal linear-phase filter, then convert it to minimum phase:\n\n    >>> from scipy.signal import remez, minimum_phase, freqz, group_delay\n    >>> import matplotlib.pyplot as plt\n    >>> freq = [0, 0.2, 0.3, 1.0]\n    >>> desired = [1, 0]\n    >>> h_linear = remez(151, freq, desired, Hz=2.)\n\n    Convert it to minimum phase:\n\n    >>> h_min_hom = minimum_phase(h_linear, method=\'homomorphic\')\n    >>> h_min_hil = minimum_phase(h_linear, method=\'hilbert\')\n\n    Compare the three filters:\n\n    >>> fig, axs = plt.subplots(4, figsize=(4, 8))\n    >>> for h, style, color in zip((h_linear, h_min_hom, h_min_hil),\n    ...                            (\'-\', \'-\', \'--\'), (\'k\', \'r\', \'c\')):\n    ...     w, H = freqz(h)\n    ...     w, gd = group_delay((h, 1))\n    ...     w /= np.pi\n    ...     axs[0].plot(h, color=color, linestyle=style)\n    ...     axs[1].plot(w, np.abs(H), color=color, linestyle=style)\n    ...     axs[2].plot(w, 20 * np.log10(np.abs(H)), color=color, linestyle=style)\n    ...     axs[3].plot(w, gd, color=color, linestyle=style)\n    >>> for ax in axs:\n    ...     ax.grid(True, color=\'0.5\')\n    ...     ax.fill_between(freq[1:3], *ax.get_ylim(), color=\'#ffeeaa\', zorder=1)\n    >>> axs[0].set(xlim=[0, len(h_linear) - 1], ylabel=\'Amplitude\', xlabel=\'Samples\')\n    >>> axs[1].legend([\'Linear\', \'Min-Hom\', \'Min-Hil\'], title=\'Phase\')\n    >>> for ax, ylim in zip(axs[1:], ([0, 1.1], [-150, 10], [-60, 60])):\n    ...     ax.set(xlim=[0, 1], ylim=ylim, xlabel=\'Frequency\')\n    >>> axs[1].set(ylabel=\'Magnitude\')\n    >>> axs[2].set(ylabel=\'Magnitude (dB)\')\n    >>> axs[3].set(ylabel=\'Group delay\')\n    >>> plt.tight_layout()\n\n    References\n    ----------\n    .. [1] N. Damera-Venkata and B. L. Evans, "Optimal design of real and\n           complex minimum phase digital FIR filters," Acoustics, Speech,\n           and Signal Processing, 1999. Proceedings., 1999 IEEE International\n           Conference on, Phoenix, AZ, 1999, pp. 1145-1148 vol.3.\n           doi: 10.1109/ICASSP.1999.756179\n    .. [2] X. Chen and T. W. Parks, "Design of optimal minimum phase FIR\n           filters by direct factorization," Signal Processing,\n           vol. 10, no. 4, pp. 369\xe2\x80\x93383, Jun. 1986.\n    .. [3] T. Saramaki, "Finite Impulse Response Filter Design," in\n           Handbook for Digital Signal Processing, chapter 4,\n           New York: Wiley-Interscience, 1993.\n    .. [4] J. S. Lim, Advanced Topics in Signal Processing.\n           Englewood Cliffs, N.J.: Prentice Hall, 1988.\n    .. [5] A. V. Oppenheim, R. W. Schafer, and J. R. Buck,\n           "Discrete-Time Signal Processing," 2nd edition.\n           Upper Saddle River, N.J.: Prentice Hall, 1999.\n    ')
    
    # Assigning a Call to a Name (line 1094):
    
    # Assigning a Call to a Name (line 1094):
    
    # Call to asarray(...): (line 1094)
    # Processing the call arguments (line 1094)
    # Getting the type of 'h' (line 1094)
    h_266790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 19), 'h', False)
    # Processing the call keyword arguments (line 1094)
    kwargs_266791 = {}
    # Getting the type of 'np' (line 1094)
    np_266788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1094)
    asarray_266789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 8), np_266788, 'asarray')
    # Calling asarray(args, kwargs) (line 1094)
    asarray_call_result_266792 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 8), asarray_266789, *[h_266790], **kwargs_266791)
    
    # Assigning a type to the variable 'h' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'h', asarray_call_result_266792)
    
    
    # Call to iscomplexobj(...): (line 1095)
    # Processing the call arguments (line 1095)
    # Getting the type of 'h' (line 1095)
    h_266795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 23), 'h', False)
    # Processing the call keyword arguments (line 1095)
    kwargs_266796 = {}
    # Getting the type of 'np' (line 1095)
    np_266793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1095, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 1095)
    iscomplexobj_266794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1095, 7), np_266793, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 1095)
    iscomplexobj_call_result_266797 = invoke(stypy.reporting.localization.Localization(__file__, 1095, 7), iscomplexobj_266794, *[h_266795], **kwargs_266796)
    
    # Testing the type of an if condition (line 1095)
    if_condition_266798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1095, 4), iscomplexobj_call_result_266797)
    # Assigning a type to the variable 'if_condition_266798' (line 1095)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1095, 4), 'if_condition_266798', if_condition_266798)
    # SSA begins for if statement (line 1095)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1096)
    # Processing the call arguments (line 1096)
    str_266800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1096, 25), 'str', 'Complex filters not supported')
    # Processing the call keyword arguments (line 1096)
    kwargs_266801 = {}
    # Getting the type of 'ValueError' (line 1096)
    ValueError_266799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1096)
    ValueError_call_result_266802 = invoke(stypy.reporting.localization.Localization(__file__, 1096, 14), ValueError_266799, *[str_266800], **kwargs_266801)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1096, 8), ValueError_call_result_266802, 'raise parameter', BaseException)
    # SSA join for if statement (line 1095)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'h' (line 1097)
    h_266803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 7), 'h')
    # Obtaining the member 'ndim' of a type (line 1097)
    ndim_266804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 7), h_266803, 'ndim')
    int_266805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 17), 'int')
    # Applying the binary operator '!=' (line 1097)
    result_ne_266806 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 7), '!=', ndim_266804, int_266805)
    
    
    # Getting the type of 'h' (line 1097)
    h_266807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1097, 22), 'h')
    # Obtaining the member 'size' of a type (line 1097)
    size_266808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1097, 22), h_266807, 'size')
    int_266809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, 32), 'int')
    # Applying the binary operator '<=' (line 1097)
    result_le_266810 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 22), '<=', size_266808, int_266809)
    
    # Applying the binary operator 'or' (line 1097)
    result_or_keyword_266811 = python_operator(stypy.reporting.localization.Localization(__file__, 1097, 7), 'or', result_ne_266806, result_le_266810)
    
    # Testing the type of an if condition (line 1097)
    if_condition_266812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1097, 4), result_or_keyword_266811)
    # Assigning a type to the variable 'if_condition_266812' (line 1097)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1097, 4), 'if_condition_266812', if_condition_266812)
    # SSA begins for if statement (line 1097)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1098)
    # Processing the call arguments (line 1098)
    str_266814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1098, 25), 'str', 'h must be 1D and at least 2 samples long')
    # Processing the call keyword arguments (line 1098)
    kwargs_266815 = {}
    # Getting the type of 'ValueError' (line 1098)
    ValueError_266813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1098, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1098)
    ValueError_call_result_266816 = invoke(stypy.reporting.localization.Localization(__file__, 1098, 14), ValueError_266813, *[str_266814], **kwargs_266815)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1098, 8), ValueError_call_result_266816, 'raise parameter', BaseException)
    # SSA join for if statement (line 1097)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1099):
    
    # Assigning a BinOp to a Name (line 1099):
    
    # Call to len(...): (line 1099)
    # Processing the call arguments (line 1099)
    # Getting the type of 'h' (line 1099)
    h_266818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 17), 'h', False)
    # Processing the call keyword arguments (line 1099)
    kwargs_266819 = {}
    # Getting the type of 'len' (line 1099)
    len_266817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 13), 'len', False)
    # Calling len(args, kwargs) (line 1099)
    len_call_result_266820 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 13), len_266817, *[h_266818], **kwargs_266819)
    
    int_266821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 23), 'int')
    # Applying the binary operator '//' (line 1099)
    result_floordiv_266822 = python_operator(stypy.reporting.localization.Localization(__file__, 1099, 13), '//', len_call_result_266820, int_266821)
    
    # Assigning a type to the variable 'n_half' (line 1099)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1099, 4), 'n_half', result_floordiv_266822)
    
    
    
    # Call to allclose(...): (line 1100)
    # Processing the call arguments (line 1100)
    
    # Obtaining the type of the subscript
    int_266825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1100, 37), 'int')
    slice_266826 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1100, 23), None, None, int_266825)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'n_half' (line 1100)
    n_half_266827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 26), 'n_half', False)
    # Applying the 'usub' unary operator (line 1100)
    result___neg___266828 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 25), 'usub', n_half_266827)
    
    slice_266829 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1100, 23), result___neg___266828, None, None)
    # Getting the type of 'h' (line 1100)
    h_266830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 23), 'h', False)
    # Obtaining the member '__getitem__' of a type (line 1100)
    getitem___266831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 23), h_266830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1100)
    subscript_call_result_266832 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 23), getitem___266831, slice_266829)
    
    # Obtaining the member '__getitem__' of a type (line 1100)
    getitem___266833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 23), subscript_call_result_266832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1100)
    subscript_call_result_266834 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 23), getitem___266833, slice_266826)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'n_half' (line 1100)
    n_half_266835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 45), 'n_half', False)
    slice_266836 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1100, 42), None, n_half_266835, None)
    # Getting the type of 'h' (line 1100)
    h_266837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 42), 'h', False)
    # Obtaining the member '__getitem__' of a type (line 1100)
    getitem___266838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 42), h_266837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1100)
    subscript_call_result_266839 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 42), getitem___266838, slice_266836)
    
    # Processing the call keyword arguments (line 1100)
    kwargs_266840 = {}
    # Getting the type of 'np' (line 1100)
    np_266823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1100, 11), 'np', False)
    # Obtaining the member 'allclose' of a type (line 1100)
    allclose_266824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1100, 11), np_266823, 'allclose')
    # Calling allclose(args, kwargs) (line 1100)
    allclose_call_result_266841 = invoke(stypy.reporting.localization.Localization(__file__, 1100, 11), allclose_266824, *[subscript_call_result_266834, subscript_call_result_266839], **kwargs_266840)
    
    # Applying the 'not' unary operator (line 1100)
    result_not__266842 = python_operator(stypy.reporting.localization.Localization(__file__, 1100, 7), 'not', allclose_call_result_266841)
    
    # Testing the type of an if condition (line 1100)
    if_condition_266843 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1100, 4), result_not__266842)
    # Assigning a type to the variable 'if_condition_266843' (line 1100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1100, 4), 'if_condition_266843', if_condition_266843)
    # SSA begins for if statement (line 1100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 1101)
    # Processing the call arguments (line 1101)
    str_266846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1101, 22), 'str', 'h does not appear to by symmetric, conversion may fail')
    # Getting the type of 'RuntimeWarning' (line 1102)
    RuntimeWarning_266847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1102, 30), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 1101)
    kwargs_266848 = {}
    # Getting the type of 'warnings' (line 1101)
    warnings_266844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1101)
    warn_266845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1101, 8), warnings_266844, 'warn')
    # Calling warn(args, kwargs) (line 1101)
    warn_call_result_266849 = invoke(stypy.reporting.localization.Localization(__file__, 1101, 8), warn_266845, *[str_266846, RuntimeWarning_266847], **kwargs_266848)
    
    # SSA join for if statement (line 1100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 1103)
    # Processing the call arguments (line 1103)
    # Getting the type of 'method' (line 1103)
    method_266851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 22), 'method', False)
    # Getting the type of 'string_types' (line 1103)
    string_types_266852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 30), 'string_types', False)
    # Processing the call keyword arguments (line 1103)
    kwargs_266853 = {}
    # Getting the type of 'isinstance' (line 1103)
    isinstance_266850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1103)
    isinstance_call_result_266854 = invoke(stypy.reporting.localization.Localization(__file__, 1103, 11), isinstance_266850, *[method_266851, string_types_266852], **kwargs_266853)
    
    # Applying the 'not' unary operator (line 1103)
    result_not__266855 = python_operator(stypy.reporting.localization.Localization(__file__, 1103, 7), 'not', isinstance_call_result_266854)
    
    
    # Getting the type of 'method' (line 1103)
    method_266856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1103, 47), 'method')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1104)
    tuple_266857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1104)
    # Adding element type (line 1104)
    str_266858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 13), 'str', 'homomorphic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 13), tuple_266857, str_266858)
    # Adding element type (line 1104)
    str_266859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, 28), 'str', 'hilbert')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1104, 13), tuple_266857, str_266859)
    
    # Applying the binary operator 'notin' (line 1103)
    result_contains_266860 = python_operator(stypy.reporting.localization.Localization(__file__, 1103, 47), 'notin', method_266856, tuple_266857)
    
    # Applying the binary operator 'or' (line 1103)
    result_or_keyword_266861 = python_operator(stypy.reporting.localization.Localization(__file__, 1103, 7), 'or', result_not__266855, result_contains_266860)
    
    # Testing the type of an if condition (line 1103)
    if_condition_266862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1103, 4), result_or_keyword_266861)
    # Assigning a type to the variable 'if_condition_266862' (line 1103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1103, 4), 'if_condition_266862', if_condition_266862)
    # SSA begins for if statement (line 1103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1105)
    # Processing the call arguments (line 1105)
    str_266864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1105, 25), 'str', 'method must be "homomorphic" or "hilbert", got %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1106)
    tuple_266865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1106)
    # Adding element type (line 1106)
    # Getting the type of 'method' (line 1106)
    method_266866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 28), 'method', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 28), tuple_266865, method_266866)
    
    # Applying the binary operator '%' (line 1105)
    result_mod_266867 = python_operator(stypy.reporting.localization.Localization(__file__, 1105, 25), '%', str_266864, tuple_266865)
    
    # Processing the call keyword arguments (line 1105)
    kwargs_266868 = {}
    # Getting the type of 'ValueError' (line 1105)
    ValueError_266863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1105, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1105)
    ValueError_call_result_266869 = invoke(stypy.reporting.localization.Localization(__file__, 1105, 14), ValueError_266863, *[result_mod_266867], **kwargs_266868)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1105, 8), ValueError_call_result_266869, 'raise parameter', BaseException)
    # SSA join for if statement (line 1103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 1107)
    # Getting the type of 'n_fft' (line 1107)
    n_fft_266870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 7), 'n_fft')
    # Getting the type of 'None' (line 1107)
    None_266871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1107, 16), 'None')
    
    (may_be_266872, more_types_in_union_266873) = may_be_none(n_fft_266870, None_266871)

    if may_be_266872:

        if more_types_in_union_266873:
            # Runtime conditional SSA (line 1107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1108):
        
        # Assigning a BinOp to a Name (line 1108):
        int_266874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 16), 'int')
        
        # Call to int(...): (line 1108)
        # Processing the call arguments (line 1108)
        
        # Call to ceil(...): (line 1108)
        # Processing the call arguments (line 1108)
        
        # Call to log2(...): (line 1108)
        # Processing the call arguments (line 1108)
        int_266880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 41), 'int')
        
        # Call to len(...): (line 1108)
        # Processing the call arguments (line 1108)
        # Getting the type of 'h' (line 1108)
        h_266882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 50), 'h', False)
        # Processing the call keyword arguments (line 1108)
        kwargs_266883 = {}
        # Getting the type of 'len' (line 1108)
        len_266881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 46), 'len', False)
        # Calling len(args, kwargs) (line 1108)
        len_call_result_266884 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 46), len_266881, *[h_266882], **kwargs_266883)
        
        int_266885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 55), 'int')
        # Applying the binary operator '-' (line 1108)
        result_sub_266886 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 46), '-', len_call_result_266884, int_266885)
        
        # Applying the binary operator '*' (line 1108)
        result_mul_266887 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 41), '*', int_266880, result_sub_266886)
        
        float_266888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 60), 'float')
        # Applying the binary operator 'div' (line 1108)
        result_div_266889 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 58), 'div', result_mul_266887, float_266888)
        
        # Processing the call keyword arguments (line 1108)
        kwargs_266890 = {}
        # Getting the type of 'np' (line 1108)
        np_266878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 33), 'np', False)
        # Obtaining the member 'log2' of a type (line 1108)
        log2_266879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 33), np_266878, 'log2')
        # Calling log2(args, kwargs) (line 1108)
        log2_call_result_266891 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 33), log2_266879, *[result_div_266889], **kwargs_266890)
        
        # Processing the call keyword arguments (line 1108)
        kwargs_266892 = {}
        # Getting the type of 'np' (line 1108)
        np_266876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 25), 'np', False)
        # Obtaining the member 'ceil' of a type (line 1108)
        ceil_266877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 25), np_266876, 'ceil')
        # Calling ceil(args, kwargs) (line 1108)
        ceil_call_result_266893 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 25), ceil_266877, *[log2_call_result_266891], **kwargs_266892)
        
        # Processing the call keyword arguments (line 1108)
        kwargs_266894 = {}
        # Getting the type of 'int' (line 1108)
        int_266875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 21), 'int', False)
        # Calling int(args, kwargs) (line 1108)
        int_call_result_266895 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 21), int_266875, *[ceil_call_result_266893], **kwargs_266894)
        
        # Applying the binary operator '**' (line 1108)
        result_pow_266896 = python_operator(stypy.reporting.localization.Localization(__file__, 1108, 16), '**', int_266874, int_call_result_266895)
        
        # Assigning a type to the variable 'n_fft' (line 1108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'n_fft', result_pow_266896)

        if more_types_in_union_266873:
            # SSA join for if statement (line 1107)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 1109):
    
    # Assigning a Call to a Name (line 1109):
    
    # Call to int(...): (line 1109)
    # Processing the call arguments (line 1109)
    # Getting the type of 'n_fft' (line 1109)
    n_fft_266898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 16), 'n_fft', False)
    # Processing the call keyword arguments (line 1109)
    kwargs_266899 = {}
    # Getting the type of 'int' (line 1109)
    int_266897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1109, 12), 'int', False)
    # Calling int(args, kwargs) (line 1109)
    int_call_result_266900 = invoke(stypy.reporting.localization.Localization(__file__, 1109, 12), int_266897, *[n_fft_266898], **kwargs_266899)
    
    # Assigning a type to the variable 'n_fft' (line 1109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1109, 4), 'n_fft', int_call_result_266900)
    
    
    # Getting the type of 'n_fft' (line 1110)
    n_fft_266901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 7), 'n_fft')
    
    # Call to len(...): (line 1110)
    # Processing the call arguments (line 1110)
    # Getting the type of 'h' (line 1110)
    h_266903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 19), 'h', False)
    # Processing the call keyword arguments (line 1110)
    kwargs_266904 = {}
    # Getting the type of 'len' (line 1110)
    len_266902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 15), 'len', False)
    # Calling len(args, kwargs) (line 1110)
    len_call_result_266905 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 15), len_266902, *[h_266903], **kwargs_266904)
    
    # Applying the binary operator '<' (line 1110)
    result_lt_266906 = python_operator(stypy.reporting.localization.Localization(__file__, 1110, 7), '<', n_fft_266901, len_call_result_266905)
    
    # Testing the type of an if condition (line 1110)
    if_condition_266907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1110, 4), result_lt_266906)
    # Assigning a type to the variable 'if_condition_266907' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 4), 'if_condition_266907', if_condition_266907)
    # SSA begins for if statement (line 1110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1111)
    # Processing the call arguments (line 1111)
    str_266909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, 25), 'str', 'n_fft must be at least len(h)==%s')
    
    # Call to len(...): (line 1111)
    # Processing the call arguments (line 1111)
    # Getting the type of 'h' (line 1111)
    h_266911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 67), 'h', False)
    # Processing the call keyword arguments (line 1111)
    kwargs_266912 = {}
    # Getting the type of 'len' (line 1111)
    len_266910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 63), 'len', False)
    # Calling len(args, kwargs) (line 1111)
    len_call_result_266913 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 63), len_266910, *[h_266911], **kwargs_266912)
    
    # Applying the binary operator '%' (line 1111)
    result_mod_266914 = python_operator(stypy.reporting.localization.Localization(__file__, 1111, 25), '%', str_266909, len_call_result_266913)
    
    # Processing the call keyword arguments (line 1111)
    kwargs_266915 = {}
    # Getting the type of 'ValueError' (line 1111)
    ValueError_266908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1111)
    ValueError_call_result_266916 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 14), ValueError_266908, *[result_mod_266914], **kwargs_266915)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1111, 8), ValueError_call_result_266916, 'raise parameter', BaseException)
    # SSA join for if statement (line 1110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'method' (line 1112)
    method_266917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 7), 'method')
    str_266918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, 17), 'str', 'hilbert')
    # Applying the binary operator '==' (line 1112)
    result_eq_266919 = python_operator(stypy.reporting.localization.Localization(__file__, 1112, 7), '==', method_266917, str_266918)
    
    # Testing the type of an if condition (line 1112)
    if_condition_266920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1112, 4), result_eq_266919)
    # Assigning a type to the variable 'if_condition_266920' (line 1112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 4), 'if_condition_266920', if_condition_266920)
    # SSA begins for if statement (line 1112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1113):
    
    # Assigning a BinOp to a Name (line 1113):
    
    # Call to arange(...): (line 1113)
    # Processing the call arguments (line 1113)
    # Getting the type of 'n_fft' (line 1113)
    n_fft_266923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 22), 'n_fft', False)
    # Processing the call keyword arguments (line 1113)
    kwargs_266924 = {}
    # Getting the type of 'np' (line 1113)
    np_266921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 1113)
    arange_266922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1113, 12), np_266921, 'arange')
    # Calling arange(args, kwargs) (line 1113)
    arange_call_result_266925 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 12), arange_266922, *[n_fft_266923], **kwargs_266924)
    
    int_266926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 32), 'int')
    # Getting the type of 'np' (line 1113)
    np_266927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 36), 'np')
    # Obtaining the member 'pi' of a type (line 1113)
    pi_266928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1113, 36), np_266927, 'pi')
    # Applying the binary operator '*' (line 1113)
    result_mul_266929 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 32), '*', int_266926, pi_266928)
    
    # Getting the type of 'n_fft' (line 1113)
    n_fft_266930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 44), 'n_fft')
    # Applying the binary operator 'div' (line 1113)
    result_div_266931 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 42), 'div', result_mul_266929, n_fft_266930)
    
    # Getting the type of 'n_half' (line 1113)
    n_half_266932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 52), 'n_half')
    # Applying the binary operator '*' (line 1113)
    result_mul_266933 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 50), '*', result_div_266931, n_half_266932)
    
    # Applying the binary operator '*' (line 1113)
    result_mul_266934 = python_operator(stypy.reporting.localization.Localization(__file__, 1113, 12), '*', arange_call_result_266925, result_mul_266933)
    
    # Assigning a type to the variable 'w' (line 1113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 8), 'w', result_mul_266934)
    
    # Assigning a Call to a Name (line 1114):
    
    # Assigning a Call to a Name (line 1114):
    
    # Call to real(...): (line 1114)
    # Processing the call arguments (line 1114)
    
    # Call to fft(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'h' (line 1114)
    h_266938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 24), 'h', False)
    # Getting the type of 'n_fft' (line 1114)
    n_fft_266939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 27), 'n_fft', False)
    # Processing the call keyword arguments (line 1114)
    kwargs_266940 = {}
    # Getting the type of 'fft' (line 1114)
    fft_266937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 20), 'fft', False)
    # Calling fft(args, kwargs) (line 1114)
    fft_call_result_266941 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 20), fft_266937, *[h_266938, n_fft_266939], **kwargs_266940)
    
    
    # Call to exp(...): (line 1114)
    # Processing the call arguments (line 1114)
    complex_266944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 43), 'complex')
    # Getting the type of 'w' (line 1114)
    w_266945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 48), 'w', False)
    # Applying the binary operator '*' (line 1114)
    result_mul_266946 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 43), '*', complex_266944, w_266945)
    
    # Processing the call keyword arguments (line 1114)
    kwargs_266947 = {}
    # Getting the type of 'np' (line 1114)
    np_266942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 36), 'np', False)
    # Obtaining the member 'exp' of a type (line 1114)
    exp_266943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 36), np_266942, 'exp')
    # Calling exp(args, kwargs) (line 1114)
    exp_call_result_266948 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 36), exp_266943, *[result_mul_266946], **kwargs_266947)
    
    # Applying the binary operator '*' (line 1114)
    result_mul_266949 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 20), '*', fft_call_result_266941, exp_call_result_266948)
    
    # Processing the call keyword arguments (line 1114)
    kwargs_266950 = {}
    # Getting the type of 'np' (line 1114)
    np_266935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 12), 'np', False)
    # Obtaining the member 'real' of a type (line 1114)
    real_266936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 12), np_266935, 'real')
    # Calling real(args, kwargs) (line 1114)
    real_call_result_266951 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 12), real_266936, *[result_mul_266949], **kwargs_266950)
    
    # Assigning a type to the variable 'H' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 8), 'H', real_call_result_266951)
    
    # Assigning a BinOp to a Name (line 1115):
    
    # Assigning a BinOp to a Name (line 1115):
    
    # Call to max(...): (line 1115)
    # Processing the call arguments (line 1115)
    # Getting the type of 'H' (line 1115)
    H_266953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 17), 'H', False)
    # Processing the call keyword arguments (line 1115)
    kwargs_266954 = {}
    # Getting the type of 'max' (line 1115)
    max_266952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 13), 'max', False)
    # Calling max(args, kwargs) (line 1115)
    max_call_result_266955 = invoke(stypy.reporting.localization.Localization(__file__, 1115, 13), max_266952, *[H_266953], **kwargs_266954)
    
    int_266956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 22), 'int')
    # Applying the binary operator '-' (line 1115)
    result_sub_266957 = python_operator(stypy.reporting.localization.Localization(__file__, 1115, 13), '-', max_call_result_266955, int_266956)
    
    # Assigning a type to the variable 'dp' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 8), 'dp', result_sub_266957)
    
    # Assigning a BinOp to a Name (line 1116):
    
    # Assigning a BinOp to a Name (line 1116):
    int_266958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1116, 13), 'int')
    
    # Call to min(...): (line 1116)
    # Processing the call arguments (line 1116)
    # Getting the type of 'H' (line 1116)
    H_266960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 21), 'H', False)
    # Processing the call keyword arguments (line 1116)
    kwargs_266961 = {}
    # Getting the type of 'min' (line 1116)
    min_266959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 17), 'min', False)
    # Calling min(args, kwargs) (line 1116)
    min_call_result_266962 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 17), min_266959, *[H_266960], **kwargs_266961)
    
    # Applying the binary operator '-' (line 1116)
    result_sub_266963 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 13), '-', int_266958, min_call_result_266962)
    
    # Assigning a type to the variable 'ds' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 8), 'ds', result_sub_266963)
    
    # Assigning a BinOp to a Name (line 1117):
    
    # Assigning a BinOp to a Name (line 1117):
    float_266964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 12), 'float')
    
    # Call to sqrt(...): (line 1117)
    # Processing the call arguments (line 1117)
    int_266967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 26), 'int')
    # Getting the type of 'dp' (line 1117)
    dp_266968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 28), 'dp', False)
    # Applying the binary operator '+' (line 1117)
    result_add_266969 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 26), '+', int_266967, dp_266968)
    
    # Getting the type of 'ds' (line 1117)
    ds_266970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 31), 'ds', False)
    # Applying the binary operator '+' (line 1117)
    result_add_266971 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 30), '+', result_add_266969, ds_266970)
    
    # Processing the call keyword arguments (line 1117)
    kwargs_266972 = {}
    # Getting the type of 'np' (line 1117)
    np_266965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 18), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1117)
    sqrt_266966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1117, 18), np_266965, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1117)
    sqrt_call_result_266973 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 18), sqrt_266966, *[result_add_266971], **kwargs_266972)
    
    
    # Call to sqrt(...): (line 1117)
    # Processing the call arguments (line 1117)
    int_266976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 45), 'int')
    # Getting the type of 'dp' (line 1117)
    dp_266977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 47), 'dp', False)
    # Applying the binary operator '-' (line 1117)
    result_sub_266978 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 45), '-', int_266976, dp_266977)
    
    # Getting the type of 'ds' (line 1117)
    ds_266979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 50), 'ds', False)
    # Applying the binary operator '+' (line 1117)
    result_add_266980 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 49), '+', result_sub_266978, ds_266979)
    
    # Processing the call keyword arguments (line 1117)
    kwargs_266981 = {}
    # Getting the type of 'np' (line 1117)
    np_266974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1117, 37), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1117)
    sqrt_266975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1117, 37), np_266974, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1117)
    sqrt_call_result_266982 = invoke(stypy.reporting.localization.Localization(__file__, 1117, 37), sqrt_266975, *[result_add_266980], **kwargs_266981)
    
    # Applying the binary operator '+' (line 1117)
    result_add_266983 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 18), '+', sqrt_call_result_266973, sqrt_call_result_266982)
    
    int_266984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, 58), 'int')
    # Applying the binary operator '**' (line 1117)
    result_pow_266985 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 17), '**', result_add_266983, int_266984)
    
    # Applying the binary operator 'div' (line 1117)
    result_div_266986 = python_operator(stypy.reporting.localization.Localization(__file__, 1117, 12), 'div', float_266964, result_pow_266985)
    
    # Assigning a type to the variable 'S' (line 1117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1117, 8), 'S', result_div_266986)
    
    # Getting the type of 'H' (line 1118)
    H_266987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 8), 'H')
    # Getting the type of 'ds' (line 1118)
    ds_266988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 13), 'ds')
    # Applying the binary operator '+=' (line 1118)
    result_iadd_266989 = python_operator(stypy.reporting.localization.Localization(__file__, 1118, 8), '+=', H_266987, ds_266988)
    # Assigning a type to the variable 'H' (line 1118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 8), 'H', result_iadd_266989)
    
    
    # Getting the type of 'H' (line 1119)
    H_266990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 8), 'H')
    # Getting the type of 'S' (line 1119)
    S_266991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 13), 'S')
    # Applying the binary operator '*=' (line 1119)
    result_imul_266992 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 8), '*=', H_266990, S_266991)
    # Assigning a type to the variable 'H' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 8), 'H', result_imul_266992)
    
    
    # Assigning a Call to a Name (line 1120):
    
    # Assigning a Call to a Name (line 1120):
    
    # Call to sqrt(...): (line 1120)
    # Processing the call arguments (line 1120)
    # Getting the type of 'H' (line 1120)
    H_266995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 20), 'H', False)
    # Processing the call keyword arguments (line 1120)
    # Getting the type of 'H' (line 1120)
    H_266996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 27), 'H', False)
    keyword_266997 = H_266996
    kwargs_266998 = {'out': keyword_266997}
    # Getting the type of 'np' (line 1120)
    np_266993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1120)
    sqrt_266994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1120, 12), np_266993, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1120)
    sqrt_call_result_266999 = invoke(stypy.reporting.localization.Localization(__file__, 1120, 12), sqrt_266994, *[H_266995], **kwargs_266998)
    
    # Assigning a type to the variable 'H' (line 1120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1120, 8), 'H', sqrt_call_result_266999)
    
    # Getting the type of 'H' (line 1121)
    H_267000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 8), 'H')
    float_267001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 13), 'float')
    # Applying the binary operator '+=' (line 1121)
    result_iadd_267002 = python_operator(stypy.reporting.localization.Localization(__file__, 1121, 8), '+=', H_267000, float_267001)
    # Assigning a type to the variable 'H' (line 1121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 8), 'H', result_iadd_267002)
    
    
    # Assigning a Call to a Name (line 1122):
    
    # Assigning a Call to a Name (line 1122):
    
    # Call to _dhtm(...): (line 1122)
    # Processing the call arguments (line 1122)
    # Getting the type of 'H' (line 1122)
    H_267004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 26), 'H', False)
    # Processing the call keyword arguments (line 1122)
    kwargs_267005 = {}
    # Getting the type of '_dhtm' (line 1122)
    _dhtm_267003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 20), '_dhtm', False)
    # Calling _dhtm(args, kwargs) (line 1122)
    _dhtm_call_result_267006 = invoke(stypy.reporting.localization.Localization(__file__, 1122, 20), _dhtm_267003, *[H_267004], **kwargs_267005)
    
    # Assigning a type to the variable 'h_minimum' (line 1122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 8), 'h_minimum', _dhtm_call_result_267006)
    # SSA branch for the else part of an if statement (line 1112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1125):
    
    # Assigning a Call to a Name (line 1125):
    
    # Call to abs(...): (line 1125)
    # Processing the call arguments (line 1125)
    
    # Call to fft(...): (line 1125)
    # Processing the call arguments (line 1125)
    # Getting the type of 'h' (line 1125)
    h_267010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 28), 'h', False)
    # Getting the type of 'n_fft' (line 1125)
    n_fft_267011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 31), 'n_fft', False)
    # Processing the call keyword arguments (line 1125)
    kwargs_267012 = {}
    # Getting the type of 'fft' (line 1125)
    fft_267009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 24), 'fft', False)
    # Calling fft(args, kwargs) (line 1125)
    fft_call_result_267013 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 24), fft_267009, *[h_267010, n_fft_267011], **kwargs_267012)
    
    # Processing the call keyword arguments (line 1125)
    kwargs_267014 = {}
    # Getting the type of 'np' (line 1125)
    np_267007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 17), 'np', False)
    # Obtaining the member 'abs' of a type (line 1125)
    abs_267008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 17), np_267007, 'abs')
    # Calling abs(args, kwargs) (line 1125)
    abs_call_result_267015 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 17), abs_267008, *[fft_call_result_267013], **kwargs_267014)
    
    # Assigning a type to the variable 'h_temp' (line 1125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1125, 8), 'h_temp', abs_call_result_267015)
    
    # Getting the type of 'h_temp' (line 1127)
    h_temp_267016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 8), 'h_temp')
    float_267017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 18), 'float')
    
    # Call to min(...): (line 1127)
    # Processing the call keyword arguments (line 1127)
    kwargs_267025 = {}
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'h_temp' (line 1127)
    h_temp_267018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 32), 'h_temp', False)
    int_267019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 41), 'int')
    # Applying the binary operator '>' (line 1127)
    result_gt_267020 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 32), '>', h_temp_267018, int_267019)
    
    # Getting the type of 'h_temp' (line 1127)
    h_temp_267021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 25), 'h_temp', False)
    # Obtaining the member '__getitem__' of a type (line 1127)
    getitem___267022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1127, 25), h_temp_267021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1127)
    subscript_call_result_267023 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 25), getitem___267022, result_gt_267020)
    
    # Obtaining the member 'min' of a type (line 1127)
    min_267024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1127, 25), subscript_call_result_267023, 'min')
    # Calling min(args, kwargs) (line 1127)
    min_call_result_267026 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 25), min_267024, *[], **kwargs_267025)
    
    # Applying the binary operator '*' (line 1127)
    result_mul_267027 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 18), '*', float_267017, min_call_result_267026)
    
    # Applying the binary operator '+=' (line 1127)
    result_iadd_267028 = python_operator(stypy.reporting.localization.Localization(__file__, 1127, 8), '+=', h_temp_267016, result_mul_267027)
    # Assigning a type to the variable 'h_temp' (line 1127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 8), 'h_temp', result_iadd_267028)
    
    
    # Call to log(...): (line 1128)
    # Processing the call arguments (line 1128)
    # Getting the type of 'h_temp' (line 1128)
    h_temp_267031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 15), 'h_temp', False)
    # Processing the call keyword arguments (line 1128)
    # Getting the type of 'h_temp' (line 1128)
    h_temp_267032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 27), 'h_temp', False)
    keyword_267033 = h_temp_267032
    kwargs_267034 = {'out': keyword_267033}
    # Getting the type of 'np' (line 1128)
    np_267029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 8), 'np', False)
    # Obtaining the member 'log' of a type (line 1128)
    log_267030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1128, 8), np_267029, 'log')
    # Calling log(args, kwargs) (line 1128)
    log_call_result_267035 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), log_267030, *[h_temp_267031], **kwargs_267034)
    
    
    # Getting the type of 'h_temp' (line 1129)
    h_temp_267036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 8), 'h_temp')
    float_267037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 18), 'float')
    # Applying the binary operator '*=' (line 1129)
    result_imul_267038 = python_operator(stypy.reporting.localization.Localization(__file__, 1129, 8), '*=', h_temp_267036, float_267037)
    # Assigning a type to the variable 'h_temp' (line 1129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 8), 'h_temp', result_imul_267038)
    
    
    # Assigning a Attribute to a Name (line 1131):
    
    # Assigning a Attribute to a Name (line 1131):
    
    # Call to ifft(...): (line 1131)
    # Processing the call arguments (line 1131)
    # Getting the type of 'h_temp' (line 1131)
    h_temp_267040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 22), 'h_temp', False)
    # Processing the call keyword arguments (line 1131)
    kwargs_267041 = {}
    # Getting the type of 'ifft' (line 1131)
    ifft_267039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1131, 17), 'ifft', False)
    # Calling ifft(args, kwargs) (line 1131)
    ifft_call_result_267042 = invoke(stypy.reporting.localization.Localization(__file__, 1131, 17), ifft_267039, *[h_temp_267040], **kwargs_267041)
    
    # Obtaining the member 'real' of a type (line 1131)
    real_267043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1131, 17), ifft_call_result_267042, 'real')
    # Assigning a type to the variable 'h_temp' (line 1131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1131, 8), 'h_temp', real_267043)
    
    # Assigning a Call to a Name (line 1134):
    
    # Assigning a Call to a Name (line 1134):
    
    # Call to zeros(...): (line 1134)
    # Processing the call arguments (line 1134)
    # Getting the type of 'n_fft' (line 1134)
    n_fft_267046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 23), 'n_fft', False)
    # Processing the call keyword arguments (line 1134)
    kwargs_267047 = {}
    # Getting the type of 'np' (line 1134)
    np_267044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 14), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1134)
    zeros_267045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1134, 14), np_267044, 'zeros')
    # Calling zeros(args, kwargs) (line 1134)
    zeros_call_result_267048 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 14), zeros_267045, *[n_fft_267046], **kwargs_267047)
    
    # Assigning a type to the variable 'win' (line 1134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 8), 'win', zeros_call_result_267048)
    
    # Assigning a Num to a Subscript (line 1135):
    
    # Assigning a Num to a Subscript (line 1135):
    int_267049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 17), 'int')
    # Getting the type of 'win' (line 1135)
    win_267050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 8), 'win')
    int_267051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1135, 12), 'int')
    # Storing an element on a container (line 1135)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1135, 8), win_267050, (int_267051, int_267049))
    
    # Assigning a BinOp to a Name (line 1136):
    
    # Assigning a BinOp to a Name (line 1136):
    
    # Call to len(...): (line 1136)
    # Processing the call arguments (line 1136)
    # Getting the type of 'h' (line 1136)
    h_267053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 20), 'h', False)
    # Processing the call keyword arguments (line 1136)
    kwargs_267054 = {}
    # Getting the type of 'len' (line 1136)
    len_267052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 16), 'len', False)
    # Calling len(args, kwargs) (line 1136)
    len_call_result_267055 = invoke(stypy.reporting.localization.Localization(__file__, 1136, 16), len_267052, *[h_267053], **kwargs_267054)
    
    int_267056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 25), 'int')
    # Applying the binary operator '+' (line 1136)
    result_add_267057 = python_operator(stypy.reporting.localization.Localization(__file__, 1136, 16), '+', len_call_result_267055, int_267056)
    
    int_267058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 31), 'int')
    # Applying the binary operator '//' (line 1136)
    result_floordiv_267059 = python_operator(stypy.reporting.localization.Localization(__file__, 1136, 15), '//', result_add_267057, int_267058)
    
    # Assigning a type to the variable 'stop' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 8), 'stop', result_floordiv_267059)
    
    # Assigning a Num to a Subscript (line 1137):
    
    # Assigning a Num to a Subscript (line 1137):
    int_267060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 22), 'int')
    # Getting the type of 'win' (line 1137)
    win_267061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 8), 'win')
    int_267062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1137, 12), 'int')
    # Getting the type of 'stop' (line 1137)
    stop_267063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1137, 14), 'stop')
    slice_267064 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1137, 8), int_267062, stop_267063, None)
    # Storing an element on a container (line 1137)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1137, 8), win_267061, (slice_267064, int_267060))
    
    
    # Call to len(...): (line 1138)
    # Processing the call arguments (line 1138)
    # Getting the type of 'h' (line 1138)
    h_267066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 15), 'h', False)
    # Processing the call keyword arguments (line 1138)
    kwargs_267067 = {}
    # Getting the type of 'len' (line 1138)
    len_267065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1138, 11), 'len', False)
    # Calling len(args, kwargs) (line 1138)
    len_call_result_267068 = invoke(stypy.reporting.localization.Localization(__file__, 1138, 11), len_267065, *[h_267066], **kwargs_267067)
    
    int_267069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1138, 20), 'int')
    # Applying the binary operator '%' (line 1138)
    result_mod_267070 = python_operator(stypy.reporting.localization.Localization(__file__, 1138, 11), '%', len_call_result_267068, int_267069)
    
    # Testing the type of an if condition (line 1138)
    if_condition_267071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1138, 8), result_mod_267070)
    # Assigning a type to the variable 'if_condition_267071' (line 1138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1138, 8), 'if_condition_267071', if_condition_267071)
    # SSA begins for if statement (line 1138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 1139):
    
    # Assigning a Num to a Subscript (line 1139):
    int_267072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 24), 'int')
    # Getting the type of 'win' (line 1139)
    win_267073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 12), 'win')
    # Getting the type of 'stop' (line 1139)
    stop_267074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 16), 'stop')
    # Storing an element on a container (line 1139)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 12), win_267073, (stop_267074, int_267072))
    # SSA join for if statement (line 1138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'h_temp' (line 1140)
    h_temp_267075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'h_temp')
    # Getting the type of 'win' (line 1140)
    win_267076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 18), 'win')
    # Applying the binary operator '*=' (line 1140)
    result_imul_267077 = python_operator(stypy.reporting.localization.Localization(__file__, 1140, 8), '*=', h_temp_267075, win_267076)
    # Assigning a type to the variable 'h_temp' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 8), 'h_temp', result_imul_267077)
    
    
    # Assigning a Call to a Name (line 1141):
    
    # Assigning a Call to a Name (line 1141):
    
    # Call to ifft(...): (line 1141)
    # Processing the call arguments (line 1141)
    
    # Call to exp(...): (line 1141)
    # Processing the call arguments (line 1141)
    
    # Call to fft(...): (line 1141)
    # Processing the call arguments (line 1141)
    # Getting the type of 'h_temp' (line 1141)
    h_temp_267082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 33), 'h_temp', False)
    # Processing the call keyword arguments (line 1141)
    kwargs_267083 = {}
    # Getting the type of 'fft' (line 1141)
    fft_267081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 29), 'fft', False)
    # Calling fft(args, kwargs) (line 1141)
    fft_call_result_267084 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 29), fft_267081, *[h_temp_267082], **kwargs_267083)
    
    # Processing the call keyword arguments (line 1141)
    kwargs_267085 = {}
    # Getting the type of 'np' (line 1141)
    np_267079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 22), 'np', False)
    # Obtaining the member 'exp' of a type (line 1141)
    exp_267080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1141, 22), np_267079, 'exp')
    # Calling exp(args, kwargs) (line 1141)
    exp_call_result_267086 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 22), exp_267080, *[fft_call_result_267084], **kwargs_267085)
    
    # Processing the call keyword arguments (line 1141)
    kwargs_267087 = {}
    # Getting the type of 'ifft' (line 1141)
    ifft_267078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 17), 'ifft', False)
    # Calling ifft(args, kwargs) (line 1141)
    ifft_call_result_267088 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 17), ifft_267078, *[exp_call_result_267086], **kwargs_267087)
    
    # Assigning a type to the variable 'h_temp' (line 1141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1141, 8), 'h_temp', ifft_call_result_267088)
    
    # Assigning a Attribute to a Name (line 1142):
    
    # Assigning a Attribute to a Name (line 1142):
    # Getting the type of 'h_temp' (line 1142)
    h_temp_267089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 20), 'h_temp')
    # Obtaining the member 'real' of a type (line 1142)
    real_267090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 20), h_temp_267089, 'real')
    # Assigning a type to the variable 'h_minimum' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 8), 'h_minimum', real_267090)
    # SSA join for if statement (line 1112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1143):
    
    # Assigning a BinOp to a Name (line 1143):
    # Getting the type of 'n_half' (line 1143)
    n_half_267091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 12), 'n_half')
    
    # Call to len(...): (line 1143)
    # Processing the call arguments (line 1143)
    # Getting the type of 'h' (line 1143)
    h_267093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 25), 'h', False)
    # Processing the call keyword arguments (line 1143)
    kwargs_267094 = {}
    # Getting the type of 'len' (line 1143)
    len_267092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 21), 'len', False)
    # Calling len(args, kwargs) (line 1143)
    len_call_result_267095 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 21), len_267092, *[h_267093], **kwargs_267094)
    
    int_267096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 30), 'int')
    # Applying the binary operator '%' (line 1143)
    result_mod_267097 = python_operator(stypy.reporting.localization.Localization(__file__, 1143, 21), '%', len_call_result_267095, int_267096)
    
    # Applying the binary operator '+' (line 1143)
    result_add_267098 = python_operator(stypy.reporting.localization.Localization(__file__, 1143, 12), '+', n_half_267091, result_mod_267097)
    
    # Assigning a type to the variable 'n_out' (line 1143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1143, 4), 'n_out', result_add_267098)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n_out' (line 1144)
    n_out_267099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 22), 'n_out')
    slice_267100 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1144, 11), None, n_out_267099, None)
    # Getting the type of 'h_minimum' (line 1144)
    h_minimum_267101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 11), 'h_minimum')
    # Obtaining the member '__getitem__' of a type (line 1144)
    getitem___267102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1144, 11), h_minimum_267101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1144)
    subscript_call_result_267103 = invoke(stypy.reporting.localization.Localization(__file__, 1144, 11), getitem___267102, slice_267100)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'stypy_return_type', subscript_call_result_267103)
    
    # ################# End of 'minimum_phase(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'minimum_phase' in the type store
    # Getting the type of 'stypy_return_type' (line 972)
    stypy_return_type_267104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_267104)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'minimum_phase'
    return stypy_return_type_267104

# Assigning a type to the variable 'minimum_phase' (line 972)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 0), 'minimum_phase', minimum_phase)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
