
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Travis Oliphant
2: # 2003
3: #
4: # Feb. 2010: Updated by Warren Weckesser:
5: #   Rewrote much of chirp()
6: #   Added sweep_poly()
7: from __future__ import division, print_function, absolute_import
8: 
9: import numpy as np
10: from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, \
11:     exp, cos, sin, polyval, polyint
12: 
13: from scipy._lib.six import string_types
14: 
15: 
16: __all__ = ['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
17:            'unit_impulse']
18: 
19: 
20: def sawtooth(t, width=1):
21:     '''
22:     Return a periodic sawtooth or triangle waveform.
23: 
24:     The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the
25:     interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval
26:     ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].
27: 
28:     Note that this is not band-limited.  It produces an infinite number
29:     of harmonics, which are aliased back and forth across the frequency
30:     spectrum.
31: 
32:     Parameters
33:     ----------
34:     t : array_like
35:         Time.
36:     width : array_like, optional
37:         Width of the rising ramp as a proportion of the total cycle.
38:         Default is 1, producing a rising ramp, while 0 produces a falling
39:         ramp.  `width` = 0.5 produces a triangle wave.
40:         If an array, causes wave shape to change over time, and must be the
41:         same length as t.
42: 
43:     Returns
44:     -------
45:     y : ndarray
46:         Output array containing the sawtooth waveform.
47: 
48:     Examples
49:     --------
50:     A 5 Hz waveform sampled at 500 Hz for 1 second:
51: 
52:     >>> from scipy import signal
53:     >>> import matplotlib.pyplot as plt
54:     >>> t = np.linspace(0, 1, 500)
55:     >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))
56: 
57:     '''
58:     t, w = asarray(t), asarray(width)
59:     w = asarray(w + (t - t))
60:     t = asarray(t + (w - w))
61:     if t.dtype.char in ['fFdD']:
62:         ytype = t.dtype.char
63:     else:
64:         ytype = 'd'
65:     y = zeros(t.shape, ytype)
66: 
67:     # width must be between 0 and 1 inclusive
68:     mask1 = (w > 1) | (w < 0)
69:     place(y, mask1, nan)
70: 
71:     # take t modulo 2*pi
72:     tmod = mod(t, 2 * pi)
73: 
74:     # on the interval 0 to width*2*pi function is
75:     #  tmod / (pi*w) - 1
76:     mask2 = (1 - mask1) & (tmod < w * 2 * pi)
77:     tsub = extract(mask2, tmod)
78:     wsub = extract(mask2, w)
79:     place(y, mask2, tsub / (pi * wsub) - 1)
80: 
81:     # on the interval width*2*pi to 2*pi function is
82:     #  (pi*(w+1)-tmod) / (pi*(1-w))
83: 
84:     mask3 = (1 - mask1) & (1 - mask2)
85:     tsub = extract(mask3, tmod)
86:     wsub = extract(mask3, w)
87:     place(y, mask3, (pi * (wsub + 1) - tsub) / (pi * (1 - wsub)))
88:     return y
89: 
90: 
91: def square(t, duty=0.5):
92:     '''
93:     Return a periodic square-wave waveform.
94: 
95:     The square wave has a period ``2*pi``, has value +1 from 0 to
96:     ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in
97:     the interval [0,1].
98: 
99:     Note that this is not band-limited.  It produces an infinite number
100:     of harmonics, which are aliased back and forth across the frequency
101:     spectrum.
102: 
103:     Parameters
104:     ----------
105:     t : array_like
106:         The input time array.
107:     duty : array_like, optional
108:         Duty cycle.  Default is 0.5 (50% duty cycle).
109:         If an array, causes wave shape to change over time, and must be the
110:         same length as t.
111: 
112:     Returns
113:     -------
114:     y : ndarray
115:         Output array containing the square waveform.
116: 
117:     Examples
118:     --------
119:     A 5 Hz waveform sampled at 500 Hz for 1 second:
120: 
121:     >>> from scipy import signal
122:     >>> import matplotlib.pyplot as plt
123:     >>> t = np.linspace(0, 1, 500, endpoint=False)
124:     >>> plt.plot(t, signal.square(2 * np.pi * 5 * t))
125:     >>> plt.ylim(-2, 2)
126: 
127:     A pulse-width modulated sine wave:
128: 
129:     >>> plt.figure()
130:     >>> sig = np.sin(2 * np.pi * t)
131:     >>> pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
132:     >>> plt.subplot(2, 1, 1)
133:     >>> plt.plot(t, sig)
134:     >>> plt.subplot(2, 1, 2)
135:     >>> plt.plot(t, pwm)
136:     >>> plt.ylim(-1.5, 1.5)
137: 
138:     '''
139:     t, w = asarray(t), asarray(duty)
140:     w = asarray(w + (t - t))
141:     t = asarray(t + (w - w))
142:     if t.dtype.char in ['fFdD']:
143:         ytype = t.dtype.char
144:     else:
145:         ytype = 'd'
146: 
147:     y = zeros(t.shape, ytype)
148: 
149:     # width must be between 0 and 1 inclusive
150:     mask1 = (w > 1) | (w < 0)
151:     place(y, mask1, nan)
152: 
153:     # on the interval 0 to duty*2*pi function is 1
154:     tmod = mod(t, 2 * pi)
155:     mask2 = (1 - mask1) & (tmod < w * 2 * pi)
156:     place(y, mask2, 1)
157: 
158:     # on the interval duty*2*pi to 2*pi function is
159:     #  (pi*(w+1)-tmod) / (pi*(1-w))
160:     mask3 = (1 - mask1) & (1 - mask2)
161:     place(y, mask3, -1)
162:     return y
163: 
164: 
165: def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
166:                retenv=False):
167:     '''
168:     Return a Gaussian modulated sinusoid:
169: 
170:         ``exp(-a t^2) exp(1j*2*pi*fc*t).``
171: 
172:     If `retquad` is True, then return the real and imaginary parts
173:     (in-phase and quadrature).
174:     If `retenv` is True, then return the envelope (unmodulated signal).
175:     Otherwise, return the real part of the modulated sinusoid.
176: 
177:     Parameters
178:     ----------
179:     t : ndarray or the string 'cutoff'
180:         Input array.
181:     fc : int, optional
182:         Center frequency (e.g. Hz).  Default is 1000.
183:     bw : float, optional
184:         Fractional bandwidth in frequency domain of pulse (e.g. Hz).
185:         Default is 0.5.
186:     bwr : float, optional
187:         Reference level at which fractional bandwidth is calculated (dB).
188:         Default is -6.
189:     tpr : float, optional
190:         If `t` is 'cutoff', then the function returns the cutoff
191:         time for when the pulse amplitude falls below `tpr` (in dB).
192:         Default is -60.
193:     retquad : bool, optional
194:         If True, return the quadrature (imaginary) as well as the real part
195:         of the signal.  Default is False.
196:     retenv : bool, optional
197:         If True, return the envelope of the signal.  Default is False.
198: 
199:     Returns
200:     -------
201:     yI : ndarray
202:         Real part of signal.  Always returned.
203:     yQ : ndarray
204:         Imaginary part of signal.  Only returned if `retquad` is True.
205:     yenv : ndarray
206:         Envelope of signal.  Only returned if `retenv` is True.
207: 
208:     See Also
209:     --------
210:     scipy.signal.morlet
211: 
212:     Examples
213:     --------
214:     Plot real component, imaginary component, and envelope for a 5 Hz pulse,
215:     sampled at 100 Hz for 2 seconds:
216: 
217:     >>> from scipy import signal
218:     >>> import matplotlib.pyplot as plt
219:     >>> t = np.linspace(-1, 1, 2 * 100, endpoint=False)
220:     >>> i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
221:     >>> plt.plot(t, i, t, q, t, e, '--')
222: 
223:     '''
224:     if fc < 0:
225:         raise ValueError("Center frequency (fc=%.2f) must be >=0." % fc)
226:     if bw <= 0:
227:         raise ValueError("Fractional bandwidth (bw=%.2f) must be > 0." % bw)
228:     if bwr >= 0:
229:         raise ValueError("Reference level for bandwidth (bwr=%.2f) must "
230:                          "be < 0 dB" % bwr)
231: 
232:     # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)
233: 
234:     ref = pow(10.0, bwr / 20.0)
235:     # fdel = fc*bw/2:  g(fdel) = ref --- solve this for a
236:     #
237:     # pi^2/a * fc^2 * bw^2 /4=-log(ref)
238:     a = -(pi * fc * bw) ** 2 / (4.0 * log(ref))
239: 
240:     if isinstance(t, string_types):
241:         if t == 'cutoff':  # compute cut_off point
242:             #  Solve exp(-a tc**2) = tref  for tc
243:             #   tc = sqrt(-log(tref) / a) where tref = 10^(tpr/20)
244:             if tpr >= 0:
245:                 raise ValueError("Reference level for time cutoff must be < 0 dB")
246:             tref = pow(10.0, tpr / 20.0)
247:             return sqrt(-log(tref) / a)
248:         else:
249:             raise ValueError("If `t` is a string, it must be 'cutoff'")
250: 
251:     yenv = exp(-a * t * t)
252:     yI = yenv * cos(2 * pi * fc * t)
253:     yQ = yenv * sin(2 * pi * fc * t)
254:     if not retquad and not retenv:
255:         return yI
256:     if not retquad and retenv:
257:         return yI, yenv
258:     if retquad and not retenv:
259:         return yI, yQ
260:     if retquad and retenv:
261:         return yI, yQ, yenv
262: 
263: 
264: def chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
265:     '''Frequency-swept cosine generator.
266: 
267:     In the following, 'Hz' should be interpreted as 'cycles per unit';
268:     there is no requirement here that the unit is one second.  The
269:     important distinction is that the units of rotation are cycles, not
270:     radians. Likewise, `t` could be a measurement of space instead of time.
271: 
272:     Parameters
273:     ----------
274:     t : array_like
275:         Times at which to evaluate the waveform.
276:     f0 : float
277:         Frequency (e.g. Hz) at time t=0.
278:     t1 : float
279:         Time at which `f1` is specified.
280:     f1 : float
281:         Frequency (e.g. Hz) of the waveform at time `t1`.
282:     method : {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}, optional
283:         Kind of frequency sweep.  If not given, `linear` is assumed.  See
284:         Notes below for more details.
285:     phi : float, optional
286:         Phase offset, in degrees. Default is 0.
287:     vertex_zero : bool, optional
288:         This parameter is only used when `method` is 'quadratic'.
289:         It determines whether the vertex of the parabola that is the graph
290:         of the frequency is at t=0 or t=t1.
291: 
292:     Returns
293:     -------
294:     y : ndarray
295:         A numpy array containing the signal evaluated at `t` with the
296:         requested time-varying frequency.  More precisely, the function
297:         returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral
298:         (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.
299: 
300:     See Also
301:     --------
302:     sweep_poly
303: 
304:     Notes
305:     -----
306:     There are four options for the `method`.  The following formulas give
307:     the instantaneous frequency (in Hz) of the signal generated by
308:     `chirp()`.  For convenience, the shorter names shown below may also be
309:     used.
310: 
311:     linear, lin, li:
312: 
313:         ``f(t) = f0 + (f1 - f0) * t / t1``
314: 
315:     quadratic, quad, q:
316: 
317:         The graph of the frequency f(t) is a parabola through (0, f0) and
318:         (t1, f1).  By default, the vertex of the parabola is at (0, f0).
319:         If `vertex_zero` is False, then the vertex is at (t1, f1).  The
320:         formula is:
321: 
322:         if vertex_zero is True:
323: 
324:             ``f(t) = f0 + (f1 - f0) * t**2 / t1**2``
325: 
326:         else:
327: 
328:             ``f(t) = f1 - (f1 - f0) * (t1 - t)**2 / t1**2``
329: 
330:         To use a more general quadratic function, or an arbitrary
331:         polynomial, use the function `scipy.signal.waveforms.sweep_poly`.
332: 
333:     logarithmic, log, lo:
334: 
335:         ``f(t) = f0 * (f1/f0)**(t/t1)``
336: 
337:         f0 and f1 must be nonzero and have the same sign.
338: 
339:         This signal is also known as a geometric or exponential chirp.
340: 
341:     hyperbolic, hyp:
342: 
343:         ``f(t) = f0*f1*t1 / ((f0 - f1)*t + f1*t1)``
344: 
345:         f0 and f1 must be nonzero.
346: 
347:     Examples
348:     --------
349:     The following will be used in the examples:
350: 
351:     >>> from scipy.signal import chirp, spectrogram
352:     >>> import matplotlib.pyplot as plt
353: 
354:     For the first example, we'll plot the waveform for a linear chirp
355:     from 6 Hz to 1 Hz over 10 seconds:
356: 
357:     >>> t = np.linspace(0, 10, 5001)
358:     >>> w = chirp(t, f0=6, f1=1, t1=10, method='linear')
359:     >>> plt.plot(t, w)
360:     >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")
361:     >>> plt.xlabel('t (sec)')
362:     >>> plt.show()
363: 
364:     For the remaining examples, we'll use higher frequency ranges,
365:     and demonstrate the result using `scipy.signal.spectrogram`.
366:     We'll use a 10 second interval sampled at 8000 Hz.
367: 
368:     >>> fs = 8000
369:     >>> T = 10
370:     >>> t = np.linspace(0, T, T*fs, endpoint=False)
371: 
372:     Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
373:     (vertex of the parabolic curve of the frequency is at t=0):
374: 
375:     >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic')
376:     >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
377:     ...                           nfft=2048)
378:     >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
379:     >>> plt.title('Quadratic Chirp, f(0)=1500, f(10)=250')
380:     >>> plt.xlabel('t (sec)')
381:     >>> plt.ylabel('Frequency (Hz)')
382:     >>> plt.grid()
383:     >>> plt.show()
384: 
385:     Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds
386:     (vertex of the parabolic curve of the frequency is at t=10):
387: 
388:     >>> w = chirp(t, f0=1500, f1=250, t1=10, method='quadratic',
389:     ...           vertex_zero=False)
390:     >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
391:     ...                           nfft=2048)
392:     >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
393:     >>> plt.title('Quadratic Chirp, f(0)=2500, f(10)=250\\n' +
394:     ...           '(vertex_zero=False)')
395:     >>> plt.xlabel('t (sec)')
396:     >>> plt.ylabel('Frequency (Hz)')
397:     >>> plt.grid()
398:     >>> plt.show()
399: 
400:     Logarithmic chirp from 1500 Hz to 250 Hz over 10 seconds:
401: 
402:     >>> w = chirp(t, f0=1500, f1=250, t1=10, method='logarithmic')
403:     >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
404:     ...                           nfft=2048)
405:     >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
406:     >>> plt.title('Logarithmic Chirp, f(0)=1500, f(10)=250')
407:     >>> plt.xlabel('t (sec)')
408:     >>> plt.ylabel('Frequency (Hz)')
409:     >>> plt.grid()
410:     >>> plt.show()
411: 
412:     Hyperbolic chirp from 1500 Hz to 250 Hz over 10 seconds:
413: 
414:     >>> w = chirp(t, f0=1500, f1=250, t1=10, method='hyperbolic')
415:     >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,
416:     ...                           nfft=2048)
417:     >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap='gray_r')
418:     >>> plt.title('Hyperbolic Chirp, f(0)=1500, f(10)=250')
419:     >>> plt.xlabel('t (sec)')
420:     >>> plt.ylabel('Frequency (Hz)')
421:     >>> plt.grid()
422:     >>> plt.show()
423: 
424:     '''
425:     # 'phase' is computed in _chirp_phase, to make testing easier.
426:     phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
427:     # Convert  phi to radians.
428:     phi *= pi / 180
429:     return cos(phase + phi)
430: 
431: 
432: def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
433:     '''
434:     Calculate the phase used by chirp_phase to generate its output.
435: 
436:     See `chirp` for a description of the arguments.
437: 
438:     '''
439:     t = asarray(t)
440:     f0 = float(f0)
441:     t1 = float(t1)
442:     f1 = float(f1)
443:     if method in ['linear', 'lin', 'li']:
444:         beta = (f1 - f0) / t1
445:         phase = 2 * pi * (f0 * t + 0.5 * beta * t * t)
446: 
447:     elif method in ['quadratic', 'quad', 'q']:
448:         beta = (f1 - f0) / (t1 ** 2)
449:         if vertex_zero:
450:             phase = 2 * pi * (f0 * t + beta * t ** 3 / 3)
451:         else:
452:             phase = 2 * pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)
453: 
454:     elif method in ['logarithmic', 'log', 'lo']:
455:         if f0 * f1 <= 0.0:
456:             raise ValueError("For a logarithmic chirp, f0 and f1 must be "
457:                              "nonzero and have the same sign.")
458:         if f0 == f1:
459:             phase = 2 * pi * f0 * t
460:         else:
461:             beta = t1 / log(f1 / f0)
462:             phase = 2 * pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0)
463: 
464:     elif method in ['hyperbolic', 'hyp']:
465:         if f0 == 0 or f1 == 0:
466:             raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
467:                              "nonzero.")
468:         if f0 == f1:
469:             # Degenerate case: constant frequency.
470:             phase = 2 * pi * f0 * t
471:         else:
472:             # Singular point: the instantaneous frequency blows up
473:             # when t == sing.
474:             sing = -f1 * t1 / (f0 - f1)
475:             phase = 2 * pi * (-sing * f0) * log(np.abs(1 - t/sing))
476: 
477:     else:
478:         raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
479:                          " or 'hyperbolic', but a value of %r was given."
480:                          % method)
481: 
482:     return phase
483: 
484: 
485: def sweep_poly(t, poly, phi=0):
486:     '''
487:     Frequency-swept cosine generator, with a time-dependent frequency.
488: 
489:     This function generates a sinusoidal function whose instantaneous
490:     frequency varies with time.  The frequency at time `t` is given by
491:     the polynomial `poly`.
492: 
493:     Parameters
494:     ----------
495:     t : ndarray
496:         Times at which to evaluate the waveform.
497:     poly : 1-D array_like or instance of numpy.poly1d
498:         The desired frequency expressed as a polynomial.  If `poly` is
499:         a list or ndarray of length n, then the elements of `poly` are
500:         the coefficients of the polynomial, and the instantaneous
501:         frequency is
502: 
503:           ``f(t) = poly[0]*t**(n-1) + poly[1]*t**(n-2) + ... + poly[n-1]``
504: 
505:         If `poly` is an instance of numpy.poly1d, then the
506:         instantaneous frequency is
507: 
508:           ``f(t) = poly(t)``
509: 
510:     phi : float, optional
511:         Phase offset, in degrees, Default: 0.
512: 
513:     Returns
514:     -------
515:     sweep_poly : ndarray
516:         A numpy array containing the signal evaluated at `t` with the
517:         requested time-varying frequency.  More precisely, the function
518:         returns ``cos(phase + (pi/180)*phi)``, where `phase` is the integral
519:         (from 0 to t) of ``2 * pi * f(t)``; ``f(t)`` is defined above.
520: 
521:     See Also
522:     --------
523:     chirp
524: 
525:     Notes
526:     -----
527:     .. versionadded:: 0.8.0
528: 
529:     If `poly` is a list or ndarray of length `n`, then the elements of
530:     `poly` are the coefficients of the polynomial, and the instantaneous
531:     frequency is:
532: 
533:         ``f(t) = poly[0]*t**(n-1) + poly[1]*t**(n-2) + ... + poly[n-1]``
534: 
535:     If `poly` is an instance of `numpy.poly1d`, then the instantaneous
536:     frequency is:
537: 
538:           ``f(t) = poly(t)``
539: 
540:     Finally, the output `s` is:
541: 
542:         ``cos(phase + (pi/180)*phi)``
543: 
544:     where `phase` is the integral from 0 to `t` of ``2 * pi * f(t)``,
545:     ``f(t)`` as defined above.
546: 
547:     Examples
548:     --------
549:     Compute the waveform with instantaneous frequency::
550: 
551:         f(t) = 0.025*t**3 - 0.36*t**2 + 1.25*t + 2
552: 
553:     over the interval 0 <= t <= 10.
554: 
555:     >>> from scipy.signal import sweep_poly
556:     >>> p = np.poly1d([0.025, -0.36, 1.25, 2.0])
557:     >>> t = np.linspace(0, 10, 5001)
558:     >>> w = sweep_poly(t, p)
559: 
560:     Plot it:
561: 
562:     >>> import matplotlib.pyplot as plt
563:     >>> plt.subplot(2, 1, 1)
564:     >>> plt.plot(t, w)
565:     >>> plt.title("Sweep Poly\\nwith frequency " +
566:     ...           "$f(t) = 0.025t^3 - 0.36t^2 + 1.25t + 2$")
567:     >>> plt.subplot(2, 1, 2)
568:     >>> plt.plot(t, p(t), 'r', label='f(t)')
569:     >>> plt.legend()
570:     >>> plt.xlabel('t')
571:     >>> plt.tight_layout()
572:     >>> plt.show()
573: 
574:     '''
575:     # 'phase' is computed in _sweep_poly_phase, to make testing easier.
576:     phase = _sweep_poly_phase(t, poly)
577:     # Convert to radians.
578:     phi *= pi / 180
579:     return cos(phase + phi)
580: 
581: 
582: def _sweep_poly_phase(t, poly):
583:     '''
584:     Calculate the phase used by sweep_poly to generate its output.
585: 
586:     See `sweep_poly` for a description of the arguments.
587: 
588:     '''
589:     # polyint handles lists, ndarrays and instances of poly1d automatically.
590:     intpoly = polyint(poly)
591:     phase = 2 * pi * polyval(intpoly, t)
592:     return phase
593: 
594: 
595: def unit_impulse(shape, idx=None, dtype=float):
596:     '''
597:     Unit impulse signal (discrete delta function) or unit basis vector.
598: 
599:     Parameters
600:     ----------
601:     shape : int or tuple of int
602:         Number of samples in the output (1-D), or a tuple that represents the
603:         shape of the output (N-D).
604:     idx : None or int or tuple of int or 'mid', optional
605:         Index at which the value is 1.  If None, defaults to the 0th element.
606:         If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in
607:         all dimensions.  If an int, the impulse will be at `idx` in all
608:         dimensions.
609:     dtype : data-type, optional
610:         The desired data-type for the array, e.g., `numpy.int8`.  Default is
611:         `numpy.float64`.
612: 
613:     Returns
614:     -------
615:     y : ndarray
616:         Output array containing an impulse signal.
617: 
618:     Notes
619:     -----
620:     The 1D case is also known as the Kronecker delta.
621: 
622:     .. versionadded:: 0.19.0
623: 
624:     Examples
625:     --------
626:     An impulse at the 0th element (:math:`\\delta[n]`):
627: 
628:     >>> from scipy import signal
629:     >>> signal.unit_impulse(8)
630:     array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
631: 
632:     Impulse offset by 2 samples (:math:`\\delta[n-2]`):
633: 
634:     >>> signal.unit_impulse(7, 2)
635:     array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])
636: 
637:     2-dimensional impulse, centered:
638: 
639:     >>> signal.unit_impulse((3, 3), 'mid')
640:     array([[ 0.,  0.,  0.],
641:            [ 0.,  1.,  0.],
642:            [ 0.,  0.,  0.]])
643: 
644:     Impulse at (2, 2), using broadcasting:
645: 
646:     >>> signal.unit_impulse((4, 4), 2)
647:     array([[ 0.,  0.,  0.,  0.],
648:            [ 0.,  0.,  0.,  0.],
649:            [ 0.,  0.,  1.,  0.],
650:            [ 0.,  0.,  0.,  0.]])
651: 
652:     Plot the impulse response of a 4th-order Butterworth lowpass filter:
653: 
654:     >>> imp = signal.unit_impulse(100, 'mid')
655:     >>> b, a = signal.butter(4, 0.2)
656:     >>> response = signal.lfilter(b, a, imp)
657: 
658:     >>> import matplotlib.pyplot as plt
659:     >>> plt.plot(np.arange(-50, 50), imp)
660:     >>> plt.plot(np.arange(-50, 50), response)
661:     >>> plt.margins(0.1, 0.1)
662:     >>> plt.xlabel('Time [samples]')
663:     >>> plt.ylabel('Amplitude')
664:     >>> plt.grid(True)
665:     >>> plt.show()
666: 
667:     '''
668:     out = zeros(shape, dtype)
669: 
670:     shape = np.atleast_1d(shape)
671: 
672:     if idx is None:
673:         idx = (0,) * len(shape)
674:     elif idx == 'mid':
675:         idx = tuple(shape // 2)
676:     elif not hasattr(idx, "__iter__"):
677:         idx = (idx,) * len(shape)
678: 
679:     out[idx] = 1
680:     return out
681: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_282957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_282957) is not StypyTypeError):

    if (import_282957 != 'pyd_module'):
        __import__(import_282957)
        sys_modules_282958 = sys.modules[import_282957]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_282958.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_282957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, exp, cos, sin, polyval, polyint' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_282959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_282959) is not StypyTypeError):

    if (import_282959 != 'pyd_module'):
        __import__(import_282959)
        sys_modules_282960 = sys.modules[import_282959]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_282960.module_type_store, module_type_store, ['asarray', 'zeros', 'place', 'nan', 'mod', 'pi', 'extract', 'log', 'sqrt', 'exp', 'cos', 'sin', 'polyval', 'polyint'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_282960, sys_modules_282960.module_type_store, module_type_store)
    else:
        from numpy import asarray, zeros, place, nan, mod, pi, extract, log, sqrt, exp, cos, sin, polyval, polyint

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['asarray', 'zeros', 'place', 'nan', 'mod', 'pi', 'extract', 'log', 'sqrt', 'exp', 'cos', 'sin', 'polyval', 'polyint'], [asarray, zeros, place, nan, mod, pi, extract, log, sqrt, exp, cos, sin, polyval, polyint])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_282959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib.six import string_types' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_282961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six')

if (type(import_282961) is not StypyTypeError):

    if (import_282961 != 'pyd_module'):
        __import__(import_282961)
        sys_modules_282962 = sys.modules[import_282961]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', sys_modules_282962.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_282962, sys_modules_282962.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib.six', import_282961)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = ['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly', 'unit_impulse']
module_type_store.set_exportable_members(['sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly', 'unit_impulse'])

# Obtaining an instance of the builtin type 'list' (line 16)
list_282963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_282964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'sawtooth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282964)
# Adding element type (line 16)
str_282965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'str', 'square')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282965)
# Adding element type (line 16)
str_282966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'str', 'gausspulse')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282966)
# Adding element type (line 16)
str_282967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 47), 'str', 'chirp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282967)
# Adding element type (line 16)
str_282968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 56), 'str', 'sweep_poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282968)
# Adding element type (line 16)
str_282969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'unit_impulse')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), list_282963, str_282969)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_282963)

@norecursion
def sawtooth(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_282970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'int')
    defaults = [int_282970]
    # Create a new context for function 'sawtooth'
    module_type_store = module_type_store.open_function_context('sawtooth', 20, 0, False)
    
    # Passed parameters checking function
    sawtooth.stypy_localization = localization
    sawtooth.stypy_type_of_self = None
    sawtooth.stypy_type_store = module_type_store
    sawtooth.stypy_function_name = 'sawtooth'
    sawtooth.stypy_param_names_list = ['t', 'width']
    sawtooth.stypy_varargs_param_name = None
    sawtooth.stypy_kwargs_param_name = None
    sawtooth.stypy_call_defaults = defaults
    sawtooth.stypy_call_varargs = varargs
    sawtooth.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sawtooth', ['t', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sawtooth', localization, ['t', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sawtooth(...)' code ##################

    str_282971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', '\n    Return a periodic sawtooth or triangle waveform.\n\n    The sawtooth waveform has a period ``2*pi``, rises from -1 to 1 on the\n    interval 0 to ``width*2*pi``, then drops from 1 to -1 on the interval\n    ``width*2*pi`` to ``2*pi``. `width` must be in the interval [0, 1].\n\n    Note that this is not band-limited.  It produces an infinite number\n    of harmonics, which are aliased back and forth across the frequency\n    spectrum.\n\n    Parameters\n    ----------\n    t : array_like\n        Time.\n    width : array_like, optional\n        Width of the rising ramp as a proportion of the total cycle.\n        Default is 1, producing a rising ramp, while 0 produces a falling\n        ramp.  `width` = 0.5 produces a triangle wave.\n        If an array, causes wave shape to change over time, and must be the\n        same length as t.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing the sawtooth waveform.\n\n    Examples\n    --------\n    A 5 Hz waveform sampled at 500 Hz for 1 second:\n\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.linspace(0, 1, 500)\n    >>> plt.plot(t, signal.sawtooth(2 * np.pi * 5 * t))\n\n    ')
    
    # Assigning a Tuple to a Tuple (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to asarray(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 't' (line 58)
    t_282973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 't', False)
    # Processing the call keyword arguments (line 58)
    kwargs_282974 = {}
    # Getting the type of 'asarray' (line 58)
    asarray_282972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'asarray', False)
    # Calling asarray(args, kwargs) (line 58)
    asarray_call_result_282975 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), asarray_282972, *[t_282973], **kwargs_282974)
    
    # Assigning a type to the variable 'tuple_assignment_282953' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_assignment_282953', asarray_call_result_282975)
    
    # Assigning a Call to a Name (line 58):
    
    # Call to asarray(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'width' (line 58)
    width_282977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'width', False)
    # Processing the call keyword arguments (line 58)
    kwargs_282978 = {}
    # Getting the type of 'asarray' (line 58)
    asarray_282976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'asarray', False)
    # Calling asarray(args, kwargs) (line 58)
    asarray_call_result_282979 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), asarray_282976, *[width_282977], **kwargs_282978)
    
    # Assigning a type to the variable 'tuple_assignment_282954' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_assignment_282954', asarray_call_result_282979)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'tuple_assignment_282953' (line 58)
    tuple_assignment_282953_282980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_assignment_282953')
    # Assigning a type to the variable 't' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 't', tuple_assignment_282953_282980)
    
    # Assigning a Name to a Name (line 58):
    # Getting the type of 'tuple_assignment_282954' (line 58)
    tuple_assignment_282954_282981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'tuple_assignment_282954')
    # Assigning a type to the variable 'w' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'w', tuple_assignment_282954_282981)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to asarray(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'w' (line 59)
    w_282983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'w', False)
    # Getting the type of 't' (line 59)
    t_282984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 't', False)
    # Getting the type of 't' (line 59)
    t_282985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 't', False)
    # Applying the binary operator '-' (line 59)
    result_sub_282986 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '-', t_282984, t_282985)
    
    # Applying the binary operator '+' (line 59)
    result_add_282987 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 16), '+', w_282983, result_sub_282986)
    
    # Processing the call keyword arguments (line 59)
    kwargs_282988 = {}
    # Getting the type of 'asarray' (line 59)
    asarray_282982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 59)
    asarray_call_result_282989 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), asarray_282982, *[result_add_282987], **kwargs_282988)
    
    # Assigning a type to the variable 'w' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'w', asarray_call_result_282989)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to asarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 't' (line 60)
    t_282991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 't', False)
    # Getting the type of 'w' (line 60)
    w_282992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'w', False)
    # Getting the type of 'w' (line 60)
    w_282993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'w', False)
    # Applying the binary operator '-' (line 60)
    result_sub_282994 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 21), '-', w_282992, w_282993)
    
    # Applying the binary operator '+' (line 60)
    result_add_282995 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '+', t_282991, result_sub_282994)
    
    # Processing the call keyword arguments (line 60)
    kwargs_282996 = {}
    # Getting the type of 'asarray' (line 60)
    asarray_282990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 60)
    asarray_call_result_282997 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), asarray_282990, *[result_add_282995], **kwargs_282996)
    
    # Assigning a type to the variable 't' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 't', asarray_call_result_282997)
    
    
    # Getting the type of 't' (line 61)
    t_282998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 7), 't')
    # Obtaining the member 'dtype' of a type (line 61)
    dtype_282999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 7), t_282998, 'dtype')
    # Obtaining the member 'char' of a type (line 61)
    char_283000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 7), dtype_282999, 'char')
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_283001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    str_283002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'str', 'fFdD')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), list_283001, str_283002)
    
    # Applying the binary operator 'in' (line 61)
    result_contains_283003 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 7), 'in', char_283000, list_283001)
    
    # Testing the type of an if condition (line 61)
    if_condition_283004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 4), result_contains_283003)
    # Assigning a type to the variable 'if_condition_283004' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'if_condition_283004', if_condition_283004)
    # SSA begins for if statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 62):
    
    # Assigning a Attribute to a Name (line 62):
    # Getting the type of 't' (line 62)
    t_283005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 't')
    # Obtaining the member 'dtype' of a type (line 62)
    dtype_283006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), t_283005, 'dtype')
    # Obtaining the member 'char' of a type (line 62)
    char_283007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), dtype_283006, 'char')
    # Assigning a type to the variable 'ytype' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'ytype', char_283007)
    # SSA branch for the else part of an if statement (line 61)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 64):
    
    # Assigning a Str to a Name (line 64):
    str_283008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'str', 'd')
    # Assigning a type to the variable 'ytype' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'ytype', str_283008)
    # SSA join for if statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to zeros(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 't' (line 65)
    t_283010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 't', False)
    # Obtaining the member 'shape' of a type (line 65)
    shape_283011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 14), t_283010, 'shape')
    # Getting the type of 'ytype' (line 65)
    ytype_283012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'ytype', False)
    # Processing the call keyword arguments (line 65)
    kwargs_283013 = {}
    # Getting the type of 'zeros' (line 65)
    zeros_283009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 65)
    zeros_call_result_283014 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), zeros_283009, *[shape_283011, ytype_283012], **kwargs_283013)
    
    # Assigning a type to the variable 'y' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'y', zeros_call_result_283014)
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    
    # Getting the type of 'w' (line 68)
    w_283015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'w')
    int_283016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'int')
    # Applying the binary operator '>' (line 68)
    result_gt_283017 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 13), '>', w_283015, int_283016)
    
    
    # Getting the type of 'w' (line 68)
    w_283018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'w')
    int_283019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'int')
    # Applying the binary operator '<' (line 68)
    result_lt_283020 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 23), '<', w_283018, int_283019)
    
    # Applying the binary operator '|' (line 68)
    result_or__283021 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 12), '|', result_gt_283017, result_lt_283020)
    
    # Assigning a type to the variable 'mask1' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'mask1', result_or__283021)
    
    # Call to place(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'y' (line 69)
    y_283023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 10), 'y', False)
    # Getting the type of 'mask1' (line 69)
    mask1_283024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'mask1', False)
    # Getting the type of 'nan' (line 69)
    nan_283025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'nan', False)
    # Processing the call keyword arguments (line 69)
    kwargs_283026 = {}
    # Getting the type of 'place' (line 69)
    place_283022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'place', False)
    # Calling place(args, kwargs) (line 69)
    place_call_result_283027 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), place_283022, *[y_283023, mask1_283024, nan_283025], **kwargs_283026)
    
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to mod(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 't' (line 72)
    t_283029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 't', False)
    int_283030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'int')
    # Getting the type of 'pi' (line 72)
    pi_283031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'pi', False)
    # Applying the binary operator '*' (line 72)
    result_mul_283032 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 18), '*', int_283030, pi_283031)
    
    # Processing the call keyword arguments (line 72)
    kwargs_283033 = {}
    # Getting the type of 'mod' (line 72)
    mod_283028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'mod', False)
    # Calling mod(args, kwargs) (line 72)
    mod_call_result_283034 = invoke(stypy.reporting.localization.Localization(__file__, 72, 11), mod_283028, *[t_283029, result_mul_283032], **kwargs_283033)
    
    # Assigning a type to the variable 'tmod' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tmod', mod_call_result_283034)
    
    # Assigning a BinOp to a Name (line 76):
    
    # Assigning a BinOp to a Name (line 76):
    int_283035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'int')
    # Getting the type of 'mask1' (line 76)
    mask1_283036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'mask1')
    # Applying the binary operator '-' (line 76)
    result_sub_283037 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 13), '-', int_283035, mask1_283036)
    
    
    # Getting the type of 'tmod' (line 76)
    tmod_283038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'tmod')
    # Getting the type of 'w' (line 76)
    w_283039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'w')
    int_283040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 38), 'int')
    # Applying the binary operator '*' (line 76)
    result_mul_283041 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 34), '*', w_283039, int_283040)
    
    # Getting the type of 'pi' (line 76)
    pi_283042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'pi')
    # Applying the binary operator '*' (line 76)
    result_mul_283043 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 40), '*', result_mul_283041, pi_283042)
    
    # Applying the binary operator '<' (line 76)
    result_lt_283044 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), '<', tmod_283038, result_mul_283043)
    
    # Applying the binary operator '&' (line 76)
    result_and__283045 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 12), '&', result_sub_283037, result_lt_283044)
    
    # Assigning a type to the variable 'mask2' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'mask2', result_and__283045)
    
    # Assigning a Call to a Name (line 77):
    
    # Assigning a Call to a Name (line 77):
    
    # Call to extract(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'mask2' (line 77)
    mask2_283047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'mask2', False)
    # Getting the type of 'tmod' (line 77)
    tmod_283048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'tmod', False)
    # Processing the call keyword arguments (line 77)
    kwargs_283049 = {}
    # Getting the type of 'extract' (line 77)
    extract_283046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'extract', False)
    # Calling extract(args, kwargs) (line 77)
    extract_call_result_283050 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), extract_283046, *[mask2_283047, tmod_283048], **kwargs_283049)
    
    # Assigning a type to the variable 'tsub' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'tsub', extract_call_result_283050)
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to extract(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'mask2' (line 78)
    mask2_283052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'mask2', False)
    # Getting the type of 'w' (line 78)
    w_283053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'w', False)
    # Processing the call keyword arguments (line 78)
    kwargs_283054 = {}
    # Getting the type of 'extract' (line 78)
    extract_283051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'extract', False)
    # Calling extract(args, kwargs) (line 78)
    extract_call_result_283055 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), extract_283051, *[mask2_283052, w_283053], **kwargs_283054)
    
    # Assigning a type to the variable 'wsub' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'wsub', extract_call_result_283055)
    
    # Call to place(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'y' (line 79)
    y_283057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 10), 'y', False)
    # Getting the type of 'mask2' (line 79)
    mask2_283058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'mask2', False)
    # Getting the type of 'tsub' (line 79)
    tsub_283059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'tsub', False)
    # Getting the type of 'pi' (line 79)
    pi_283060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'pi', False)
    # Getting the type of 'wsub' (line 79)
    wsub_283061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'wsub', False)
    # Applying the binary operator '*' (line 79)
    result_mul_283062 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '*', pi_283060, wsub_283061)
    
    # Applying the binary operator 'div' (line 79)
    result_div_283063 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 20), 'div', tsub_283059, result_mul_283062)
    
    int_283064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'int')
    # Applying the binary operator '-' (line 79)
    result_sub_283065 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 20), '-', result_div_283063, int_283064)
    
    # Processing the call keyword arguments (line 79)
    kwargs_283066 = {}
    # Getting the type of 'place' (line 79)
    place_283056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'place', False)
    # Calling place(args, kwargs) (line 79)
    place_call_result_283067 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), place_283056, *[y_283057, mask2_283058, result_sub_283065], **kwargs_283066)
    
    
    # Assigning a BinOp to a Name (line 84):
    
    # Assigning a BinOp to a Name (line 84):
    int_283068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 13), 'int')
    # Getting the type of 'mask1' (line 84)
    mask1_283069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'mask1')
    # Applying the binary operator '-' (line 84)
    result_sub_283070 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 13), '-', int_283068, mask1_283069)
    
    int_283071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'int')
    # Getting the type of 'mask2' (line 84)
    mask2_283072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'mask2')
    # Applying the binary operator '-' (line 84)
    result_sub_283073 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 27), '-', int_283071, mask2_283072)
    
    # Applying the binary operator '&' (line 84)
    result_and__283074 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '&', result_sub_283070, result_sub_283073)
    
    # Assigning a type to the variable 'mask3' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'mask3', result_and__283074)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to extract(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'mask3' (line 85)
    mask3_283076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'mask3', False)
    # Getting the type of 'tmod' (line 85)
    tmod_283077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'tmod', False)
    # Processing the call keyword arguments (line 85)
    kwargs_283078 = {}
    # Getting the type of 'extract' (line 85)
    extract_283075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'extract', False)
    # Calling extract(args, kwargs) (line 85)
    extract_call_result_283079 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), extract_283075, *[mask3_283076, tmod_283077], **kwargs_283078)
    
    # Assigning a type to the variable 'tsub' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'tsub', extract_call_result_283079)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to extract(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'mask3' (line 86)
    mask3_283081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'mask3', False)
    # Getting the type of 'w' (line 86)
    w_283082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'w', False)
    # Processing the call keyword arguments (line 86)
    kwargs_283083 = {}
    # Getting the type of 'extract' (line 86)
    extract_283080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'extract', False)
    # Calling extract(args, kwargs) (line 86)
    extract_call_result_283084 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), extract_283080, *[mask3_283081, w_283082], **kwargs_283083)
    
    # Assigning a type to the variable 'wsub' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'wsub', extract_call_result_283084)
    
    # Call to place(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'y' (line 87)
    y_283086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 10), 'y', False)
    # Getting the type of 'mask3' (line 87)
    mask3_283087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'mask3', False)
    # Getting the type of 'pi' (line 87)
    pi_283088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'pi', False)
    # Getting the type of 'wsub' (line 87)
    wsub_283089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'wsub', False)
    int_283090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'int')
    # Applying the binary operator '+' (line 87)
    result_add_283091 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 27), '+', wsub_283089, int_283090)
    
    # Applying the binary operator '*' (line 87)
    result_mul_283092 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 21), '*', pi_283088, result_add_283091)
    
    # Getting the type of 'tsub' (line 87)
    tsub_283093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 39), 'tsub', False)
    # Applying the binary operator '-' (line 87)
    result_sub_283094 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 21), '-', result_mul_283092, tsub_283093)
    
    # Getting the type of 'pi' (line 87)
    pi_283095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 48), 'pi', False)
    int_283096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 54), 'int')
    # Getting the type of 'wsub' (line 87)
    wsub_283097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 58), 'wsub', False)
    # Applying the binary operator '-' (line 87)
    result_sub_283098 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 54), '-', int_283096, wsub_283097)
    
    # Applying the binary operator '*' (line 87)
    result_mul_283099 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 48), '*', pi_283095, result_sub_283098)
    
    # Applying the binary operator 'div' (line 87)
    result_div_283100 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 20), 'div', result_sub_283094, result_mul_283099)
    
    # Processing the call keyword arguments (line 87)
    kwargs_283101 = {}
    # Getting the type of 'place' (line 87)
    place_283085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'place', False)
    # Calling place(args, kwargs) (line 87)
    place_call_result_283102 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), place_283085, *[y_283086, mask3_283087, result_div_283100], **kwargs_283101)
    
    # Getting the type of 'y' (line 88)
    y_283103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', y_283103)
    
    # ################# End of 'sawtooth(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sawtooth' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_283104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283104)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sawtooth'
    return stypy_return_type_283104

# Assigning a type to the variable 'sawtooth' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'sawtooth', sawtooth)

@norecursion
def square(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_283105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'float')
    defaults = [float_283105]
    # Create a new context for function 'square'
    module_type_store = module_type_store.open_function_context('square', 91, 0, False)
    
    # Passed parameters checking function
    square.stypy_localization = localization
    square.stypy_type_of_self = None
    square.stypy_type_store = module_type_store
    square.stypy_function_name = 'square'
    square.stypy_param_names_list = ['t', 'duty']
    square.stypy_varargs_param_name = None
    square.stypy_kwargs_param_name = None
    square.stypy_call_defaults = defaults
    square.stypy_call_varargs = varargs
    square.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'square', ['t', 'duty'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'square', localization, ['t', 'duty'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'square(...)' code ##################

    str_283106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Return a periodic square-wave waveform.\n\n    The square wave has a period ``2*pi``, has value +1 from 0 to\n    ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``. `duty` must be in\n    the interval [0,1].\n\n    Note that this is not band-limited.  It produces an infinite number\n    of harmonics, which are aliased back and forth across the frequency\n    spectrum.\n\n    Parameters\n    ----------\n    t : array_like\n        The input time array.\n    duty : array_like, optional\n        Duty cycle.  Default is 0.5 (50% duty cycle).\n        If an array, causes wave shape to change over time, and must be the\n        same length as t.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing the square waveform.\n\n    Examples\n    --------\n    A 5 Hz waveform sampled at 500 Hz for 1 second:\n\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.linspace(0, 1, 500, endpoint=False)\n    >>> plt.plot(t, signal.square(2 * np.pi * 5 * t))\n    >>> plt.ylim(-2, 2)\n\n    A pulse-width modulated sine wave:\n\n    >>> plt.figure()\n    >>> sig = np.sin(2 * np.pi * t)\n    >>> pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)\n    >>> plt.subplot(2, 1, 1)\n    >>> plt.plot(t, sig)\n    >>> plt.subplot(2, 1, 2)\n    >>> plt.plot(t, pwm)\n    >>> plt.ylim(-1.5, 1.5)\n\n    ')
    
    # Assigning a Tuple to a Tuple (line 139):
    
    # Assigning a Call to a Name (line 139):
    
    # Call to asarray(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 't' (line 139)
    t_283108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 't', False)
    # Processing the call keyword arguments (line 139)
    kwargs_283109 = {}
    # Getting the type of 'asarray' (line 139)
    asarray_283107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'asarray', False)
    # Calling asarray(args, kwargs) (line 139)
    asarray_call_result_283110 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), asarray_283107, *[t_283108], **kwargs_283109)
    
    # Assigning a type to the variable 'tuple_assignment_282955' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tuple_assignment_282955', asarray_call_result_283110)
    
    # Assigning a Call to a Name (line 139):
    
    # Call to asarray(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'duty' (line 139)
    duty_283112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'duty', False)
    # Processing the call keyword arguments (line 139)
    kwargs_283113 = {}
    # Getting the type of 'asarray' (line 139)
    asarray_283111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'asarray', False)
    # Calling asarray(args, kwargs) (line 139)
    asarray_call_result_283114 = invoke(stypy.reporting.localization.Localization(__file__, 139, 23), asarray_283111, *[duty_283112], **kwargs_283113)
    
    # Assigning a type to the variable 'tuple_assignment_282956' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tuple_assignment_282956', asarray_call_result_283114)
    
    # Assigning a Name to a Name (line 139):
    # Getting the type of 'tuple_assignment_282955' (line 139)
    tuple_assignment_282955_283115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tuple_assignment_282955')
    # Assigning a type to the variable 't' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 't', tuple_assignment_282955_283115)
    
    # Assigning a Name to a Name (line 139):
    # Getting the type of 'tuple_assignment_282956' (line 139)
    tuple_assignment_282956_283116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'tuple_assignment_282956')
    # Assigning a type to the variable 'w' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), 'w', tuple_assignment_282956_283116)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to asarray(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'w' (line 140)
    w_283118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'w', False)
    # Getting the type of 't' (line 140)
    t_283119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 't', False)
    # Getting the type of 't' (line 140)
    t_283120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 't', False)
    # Applying the binary operator '-' (line 140)
    result_sub_283121 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 21), '-', t_283119, t_283120)
    
    # Applying the binary operator '+' (line 140)
    result_add_283122 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 16), '+', w_283118, result_sub_283121)
    
    # Processing the call keyword arguments (line 140)
    kwargs_283123 = {}
    # Getting the type of 'asarray' (line 140)
    asarray_283117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 140)
    asarray_call_result_283124 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), asarray_283117, *[result_add_283122], **kwargs_283123)
    
    # Assigning a type to the variable 'w' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'w', asarray_call_result_283124)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to asarray(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 't' (line 141)
    t_283126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 't', False)
    # Getting the type of 'w' (line 141)
    w_283127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'w', False)
    # Getting the type of 'w' (line 141)
    w_283128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'w', False)
    # Applying the binary operator '-' (line 141)
    result_sub_283129 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 21), '-', w_283127, w_283128)
    
    # Applying the binary operator '+' (line 141)
    result_add_283130 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '+', t_283126, result_sub_283129)
    
    # Processing the call keyword arguments (line 141)
    kwargs_283131 = {}
    # Getting the type of 'asarray' (line 141)
    asarray_283125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 141)
    asarray_call_result_283132 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), asarray_283125, *[result_add_283130], **kwargs_283131)
    
    # Assigning a type to the variable 't' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 't', asarray_call_result_283132)
    
    
    # Getting the type of 't' (line 142)
    t_283133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 't')
    # Obtaining the member 'dtype' of a type (line 142)
    dtype_283134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 7), t_283133, 'dtype')
    # Obtaining the member 'char' of a type (line 142)
    char_283135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 7), dtype_283134, 'char')
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_283136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    # Adding element type (line 142)
    str_283137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'str', 'fFdD')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 23), list_283136, str_283137)
    
    # Applying the binary operator 'in' (line 142)
    result_contains_283138 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), 'in', char_283135, list_283136)
    
    # Testing the type of an if condition (line 142)
    if_condition_283139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), result_contains_283138)
    # Assigning a type to the variable 'if_condition_283139' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_283139', if_condition_283139)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 143):
    
    # Assigning a Attribute to a Name (line 143):
    # Getting the type of 't' (line 143)
    t_283140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 't')
    # Obtaining the member 'dtype' of a type (line 143)
    dtype_283141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), t_283140, 'dtype')
    # Obtaining the member 'char' of a type (line 143)
    char_283142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), dtype_283141, 'char')
    # Assigning a type to the variable 'ytype' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'ytype', char_283142)
    # SSA branch for the else part of an if statement (line 142)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 145):
    
    # Assigning a Str to a Name (line 145):
    str_283143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'str', 'd')
    # Assigning a type to the variable 'ytype' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'ytype', str_283143)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 147):
    
    # Call to zeros(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 't' (line 147)
    t_283145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 't', False)
    # Obtaining the member 'shape' of a type (line 147)
    shape_283146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 14), t_283145, 'shape')
    # Getting the type of 'ytype' (line 147)
    ytype_283147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'ytype', False)
    # Processing the call keyword arguments (line 147)
    kwargs_283148 = {}
    # Getting the type of 'zeros' (line 147)
    zeros_283144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'zeros', False)
    # Calling zeros(args, kwargs) (line 147)
    zeros_call_result_283149 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), zeros_283144, *[shape_283146, ytype_283147], **kwargs_283148)
    
    # Assigning a type to the variable 'y' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'y', zeros_call_result_283149)
    
    # Assigning a BinOp to a Name (line 150):
    
    # Assigning a BinOp to a Name (line 150):
    
    # Getting the type of 'w' (line 150)
    w_283150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'w')
    int_283151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'int')
    # Applying the binary operator '>' (line 150)
    result_gt_283152 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 13), '>', w_283150, int_283151)
    
    
    # Getting the type of 'w' (line 150)
    w_283153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 23), 'w')
    int_283154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 27), 'int')
    # Applying the binary operator '<' (line 150)
    result_lt_283155 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 23), '<', w_283153, int_283154)
    
    # Applying the binary operator '|' (line 150)
    result_or__283156 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '|', result_gt_283152, result_lt_283155)
    
    # Assigning a type to the variable 'mask1' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'mask1', result_or__283156)
    
    # Call to place(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'y' (line 151)
    y_283158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'y', False)
    # Getting the type of 'mask1' (line 151)
    mask1_283159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'mask1', False)
    # Getting the type of 'nan' (line 151)
    nan_283160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'nan', False)
    # Processing the call keyword arguments (line 151)
    kwargs_283161 = {}
    # Getting the type of 'place' (line 151)
    place_283157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'place', False)
    # Calling place(args, kwargs) (line 151)
    place_call_result_283162 = invoke(stypy.reporting.localization.Localization(__file__, 151, 4), place_283157, *[y_283158, mask1_283159, nan_283160], **kwargs_283161)
    
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to mod(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 't' (line 154)
    t_283164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 't', False)
    int_283165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 18), 'int')
    # Getting the type of 'pi' (line 154)
    pi_283166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'pi', False)
    # Applying the binary operator '*' (line 154)
    result_mul_283167 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 18), '*', int_283165, pi_283166)
    
    # Processing the call keyword arguments (line 154)
    kwargs_283168 = {}
    # Getting the type of 'mod' (line 154)
    mod_283163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'mod', False)
    # Calling mod(args, kwargs) (line 154)
    mod_call_result_283169 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), mod_283163, *[t_283164, result_mul_283167], **kwargs_283168)
    
    # Assigning a type to the variable 'tmod' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'tmod', mod_call_result_283169)
    
    # Assigning a BinOp to a Name (line 155):
    
    # Assigning a BinOp to a Name (line 155):
    int_283170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 13), 'int')
    # Getting the type of 'mask1' (line 155)
    mask1_283171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'mask1')
    # Applying the binary operator '-' (line 155)
    result_sub_283172 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 13), '-', int_283170, mask1_283171)
    
    
    # Getting the type of 'tmod' (line 155)
    tmod_283173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'tmod')
    # Getting the type of 'w' (line 155)
    w_283174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'w')
    int_283175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 38), 'int')
    # Applying the binary operator '*' (line 155)
    result_mul_283176 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 34), '*', w_283174, int_283175)
    
    # Getting the type of 'pi' (line 155)
    pi_283177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'pi')
    # Applying the binary operator '*' (line 155)
    result_mul_283178 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 40), '*', result_mul_283176, pi_283177)
    
    # Applying the binary operator '<' (line 155)
    result_lt_283179 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 27), '<', tmod_283173, result_mul_283178)
    
    # Applying the binary operator '&' (line 155)
    result_and__283180 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 12), '&', result_sub_283172, result_lt_283179)
    
    # Assigning a type to the variable 'mask2' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'mask2', result_and__283180)
    
    # Call to place(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'y' (line 156)
    y_283182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 10), 'y', False)
    # Getting the type of 'mask2' (line 156)
    mask2_283183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 13), 'mask2', False)
    int_283184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 20), 'int')
    # Processing the call keyword arguments (line 156)
    kwargs_283185 = {}
    # Getting the type of 'place' (line 156)
    place_283181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'place', False)
    # Calling place(args, kwargs) (line 156)
    place_call_result_283186 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), place_283181, *[y_283182, mask2_283183, int_283184], **kwargs_283185)
    
    
    # Assigning a BinOp to a Name (line 160):
    
    # Assigning a BinOp to a Name (line 160):
    int_283187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'int')
    # Getting the type of 'mask1' (line 160)
    mask1_283188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 17), 'mask1')
    # Applying the binary operator '-' (line 160)
    result_sub_283189 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 13), '-', int_283187, mask1_283188)
    
    int_283190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 27), 'int')
    # Getting the type of 'mask2' (line 160)
    mask2_283191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'mask2')
    # Applying the binary operator '-' (line 160)
    result_sub_283192 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 27), '-', int_283190, mask2_283191)
    
    # Applying the binary operator '&' (line 160)
    result_and__283193 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 12), '&', result_sub_283189, result_sub_283192)
    
    # Assigning a type to the variable 'mask3' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'mask3', result_and__283193)
    
    # Call to place(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'y' (line 161)
    y_283195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 10), 'y', False)
    # Getting the type of 'mask3' (line 161)
    mask3_283196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'mask3', False)
    int_283197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_283198 = {}
    # Getting the type of 'place' (line 161)
    place_283194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'place', False)
    # Calling place(args, kwargs) (line 161)
    place_call_result_283199 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), place_283194, *[y_283195, mask3_283196, int_283197], **kwargs_283198)
    
    # Getting the type of 'y' (line 162)
    y_283200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type', y_283200)
    
    # ################# End of 'square(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'square' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_283201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283201)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'square'
    return stypy_return_type_283201

# Assigning a type to the variable 'square' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'square', square)

@norecursion
def gausspulse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_283202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 21), 'int')
    float_283203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 30), 'float')
    int_283204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'int')
    int_283205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 47), 'int')
    # Getting the type of 'False' (line 165)
    False_283206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'False')
    # Getting the type of 'False' (line 166)
    False_283207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'False')
    defaults = [int_283202, float_283203, int_283204, int_283205, False_283206, False_283207]
    # Create a new context for function 'gausspulse'
    module_type_store = module_type_store.open_function_context('gausspulse', 165, 0, False)
    
    # Passed parameters checking function
    gausspulse.stypy_localization = localization
    gausspulse.stypy_type_of_self = None
    gausspulse.stypy_type_store = module_type_store
    gausspulse.stypy_function_name = 'gausspulse'
    gausspulse.stypy_param_names_list = ['t', 'fc', 'bw', 'bwr', 'tpr', 'retquad', 'retenv']
    gausspulse.stypy_varargs_param_name = None
    gausspulse.stypy_kwargs_param_name = None
    gausspulse.stypy_call_defaults = defaults
    gausspulse.stypy_call_varargs = varargs
    gausspulse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gausspulse', ['t', 'fc', 'bw', 'bwr', 'tpr', 'retquad', 'retenv'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gausspulse', localization, ['t', 'fc', 'bw', 'bwr', 'tpr', 'retquad', 'retenv'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gausspulse(...)' code ##################

    str_283208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', "\n    Return a Gaussian modulated sinusoid:\n\n        ``exp(-a t^2) exp(1j*2*pi*fc*t).``\n\n    If `retquad` is True, then return the real and imaginary parts\n    (in-phase and quadrature).\n    If `retenv` is True, then return the envelope (unmodulated signal).\n    Otherwise, return the real part of the modulated sinusoid.\n\n    Parameters\n    ----------\n    t : ndarray or the string 'cutoff'\n        Input array.\n    fc : int, optional\n        Center frequency (e.g. Hz).  Default is 1000.\n    bw : float, optional\n        Fractional bandwidth in frequency domain of pulse (e.g. Hz).\n        Default is 0.5.\n    bwr : float, optional\n        Reference level at which fractional bandwidth is calculated (dB).\n        Default is -6.\n    tpr : float, optional\n        If `t` is 'cutoff', then the function returns the cutoff\n        time for when the pulse amplitude falls below `tpr` (in dB).\n        Default is -60.\n    retquad : bool, optional\n        If True, return the quadrature (imaginary) as well as the real part\n        of the signal.  Default is False.\n    retenv : bool, optional\n        If True, return the envelope of the signal.  Default is False.\n\n    Returns\n    -------\n    yI : ndarray\n        Real part of signal.  Always returned.\n    yQ : ndarray\n        Imaginary part of signal.  Only returned if `retquad` is True.\n    yenv : ndarray\n        Envelope of signal.  Only returned if `retenv` is True.\n\n    See Also\n    --------\n    scipy.signal.morlet\n\n    Examples\n    --------\n    Plot real component, imaginary component, and envelope for a 5 Hz pulse,\n    sampled at 100 Hz for 2 seconds:\n\n    >>> from scipy import signal\n    >>> import matplotlib.pyplot as plt\n    >>> t = np.linspace(-1, 1, 2 * 100, endpoint=False)\n    >>> i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)\n    >>> plt.plot(t, i, t, q, t, e, '--')\n\n    ")
    
    
    # Getting the type of 'fc' (line 224)
    fc_283209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 7), 'fc')
    int_283210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 12), 'int')
    # Applying the binary operator '<' (line 224)
    result_lt_283211 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 7), '<', fc_283209, int_283210)
    
    # Testing the type of an if condition (line 224)
    if_condition_283212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 4), result_lt_283211)
    # Assigning a type to the variable 'if_condition_283212' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'if_condition_283212', if_condition_283212)
    # SSA begins for if statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 225)
    # Processing the call arguments (line 225)
    str_283214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'str', 'Center frequency (fc=%.2f) must be >=0.')
    # Getting the type of 'fc' (line 225)
    fc_283215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 69), 'fc', False)
    # Applying the binary operator '%' (line 225)
    result_mod_283216 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 25), '%', str_283214, fc_283215)
    
    # Processing the call keyword arguments (line 225)
    kwargs_283217 = {}
    # Getting the type of 'ValueError' (line 225)
    ValueError_283213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 225)
    ValueError_call_result_283218 = invoke(stypy.reporting.localization.Localization(__file__, 225, 14), ValueError_283213, *[result_mod_283216], **kwargs_283217)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 225, 8), ValueError_call_result_283218, 'raise parameter', BaseException)
    # SSA join for if statement (line 224)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'bw' (line 226)
    bw_283219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'bw')
    int_283220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 13), 'int')
    # Applying the binary operator '<=' (line 226)
    result_le_283221 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 7), '<=', bw_283219, int_283220)
    
    # Testing the type of an if condition (line 226)
    if_condition_283222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), result_le_283221)
    # Assigning a type to the variable 'if_condition_283222' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_283222', if_condition_283222)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 227)
    # Processing the call arguments (line 227)
    str_283224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'str', 'Fractional bandwidth (bw=%.2f) must be > 0.')
    # Getting the type of 'bw' (line 227)
    bw_283225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 73), 'bw', False)
    # Applying the binary operator '%' (line 227)
    result_mod_283226 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 25), '%', str_283224, bw_283225)
    
    # Processing the call keyword arguments (line 227)
    kwargs_283227 = {}
    # Getting the type of 'ValueError' (line 227)
    ValueError_283223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 227)
    ValueError_call_result_283228 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), ValueError_283223, *[result_mod_283226], **kwargs_283227)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 227, 8), ValueError_call_result_283228, 'raise parameter', BaseException)
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'bwr' (line 228)
    bwr_283229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 7), 'bwr')
    int_283230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 14), 'int')
    # Applying the binary operator '>=' (line 228)
    result_ge_283231 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 7), '>=', bwr_283229, int_283230)
    
    # Testing the type of an if condition (line 228)
    if_condition_283232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 4), result_ge_283231)
    # Assigning a type to the variable 'if_condition_283232' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'if_condition_283232', if_condition_283232)
    # SSA begins for if statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 229)
    # Processing the call arguments (line 229)
    str_283234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 25), 'str', 'Reference level for bandwidth (bwr=%.2f) must be < 0 dB')
    # Getting the type of 'bwr' (line 230)
    bwr_283235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'bwr', False)
    # Applying the binary operator '%' (line 229)
    result_mod_283236 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 25), '%', str_283234, bwr_283235)
    
    # Processing the call keyword arguments (line 229)
    kwargs_283237 = {}
    # Getting the type of 'ValueError' (line 229)
    ValueError_283233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 229)
    ValueError_call_result_283238 = invoke(stypy.reporting.localization.Localization(__file__, 229, 14), ValueError_283233, *[result_mod_283236], **kwargs_283237)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 229, 8), ValueError_call_result_283238, 'raise parameter', BaseException)
    # SSA join for if statement (line 228)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to pow(...): (line 234)
    # Processing the call arguments (line 234)
    float_283240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 14), 'float')
    # Getting the type of 'bwr' (line 234)
    bwr_283241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'bwr', False)
    float_283242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 26), 'float')
    # Applying the binary operator 'div' (line 234)
    result_div_283243 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 20), 'div', bwr_283241, float_283242)
    
    # Processing the call keyword arguments (line 234)
    kwargs_283244 = {}
    # Getting the type of 'pow' (line 234)
    pow_283239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 234)
    pow_call_result_283245 = invoke(stypy.reporting.localization.Localization(__file__, 234, 10), pow_283239, *[float_283240, result_div_283243], **kwargs_283244)
    
    # Assigning a type to the variable 'ref' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'ref', pow_call_result_283245)
    
    # Assigning a BinOp to a Name (line 238):
    
    # Assigning a BinOp to a Name (line 238):
    
    # Getting the type of 'pi' (line 238)
    pi_283246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 10), 'pi')
    # Getting the type of 'fc' (line 238)
    fc_283247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'fc')
    # Applying the binary operator '*' (line 238)
    result_mul_283248 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 10), '*', pi_283246, fc_283247)
    
    # Getting the type of 'bw' (line 238)
    bw_283249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'bw')
    # Applying the binary operator '*' (line 238)
    result_mul_283250 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 18), '*', result_mul_283248, bw_283249)
    
    int_283251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 27), 'int')
    # Applying the binary operator '**' (line 238)
    result_pow_283252 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 9), '**', result_mul_283250, int_283251)
    
    # Applying the 'usub' unary operator (line 238)
    result___neg___283253 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 8), 'usub', result_pow_283252)
    
    float_283254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 32), 'float')
    
    # Call to log(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'ref' (line 238)
    ref_283256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 42), 'ref', False)
    # Processing the call keyword arguments (line 238)
    kwargs_283257 = {}
    # Getting the type of 'log' (line 238)
    log_283255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), 'log', False)
    # Calling log(args, kwargs) (line 238)
    log_call_result_283258 = invoke(stypy.reporting.localization.Localization(__file__, 238, 38), log_283255, *[ref_283256], **kwargs_283257)
    
    # Applying the binary operator '*' (line 238)
    result_mul_283259 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 32), '*', float_283254, log_call_result_283258)
    
    # Applying the binary operator 'div' (line 238)
    result_div_283260 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 8), 'div', result___neg___283253, result_mul_283259)
    
    # Assigning a type to the variable 'a' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'a', result_div_283260)
    
    
    # Call to isinstance(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 't' (line 240)
    t_283262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 't', False)
    # Getting the type of 'string_types' (line 240)
    string_types_283263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'string_types', False)
    # Processing the call keyword arguments (line 240)
    kwargs_283264 = {}
    # Getting the type of 'isinstance' (line 240)
    isinstance_283261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 240)
    isinstance_call_result_283265 = invoke(stypy.reporting.localization.Localization(__file__, 240, 7), isinstance_283261, *[t_283262, string_types_283263], **kwargs_283264)
    
    # Testing the type of an if condition (line 240)
    if_condition_283266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), isinstance_call_result_283265)
    # Assigning a type to the variable 'if_condition_283266' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_283266', if_condition_283266)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 't' (line 241)
    t_283267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 11), 't')
    str_283268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 16), 'str', 'cutoff')
    # Applying the binary operator '==' (line 241)
    result_eq_283269 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 11), '==', t_283267, str_283268)
    
    # Testing the type of an if condition (line 241)
    if_condition_283270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 8), result_eq_283269)
    # Assigning a type to the variable 'if_condition_283270' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'if_condition_283270', if_condition_283270)
    # SSA begins for if statement (line 241)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'tpr' (line 244)
    tpr_283271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'tpr')
    int_283272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'int')
    # Applying the binary operator '>=' (line 244)
    result_ge_283273 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 15), '>=', tpr_283271, int_283272)
    
    # Testing the type of an if condition (line 244)
    if_condition_283274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 12), result_ge_283273)
    # Assigning a type to the variable 'if_condition_283274' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'if_condition_283274', if_condition_283274)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 245)
    # Processing the call arguments (line 245)
    str_283276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 33), 'str', 'Reference level for time cutoff must be < 0 dB')
    # Processing the call keyword arguments (line 245)
    kwargs_283277 = {}
    # Getting the type of 'ValueError' (line 245)
    ValueError_283275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 245)
    ValueError_call_result_283278 = invoke(stypy.reporting.localization.Localization(__file__, 245, 22), ValueError_283275, *[str_283276], **kwargs_283277)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 245, 16), ValueError_call_result_283278, 'raise parameter', BaseException)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to pow(...): (line 246)
    # Processing the call arguments (line 246)
    float_283280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'float')
    # Getting the type of 'tpr' (line 246)
    tpr_283281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'tpr', False)
    float_283282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 35), 'float')
    # Applying the binary operator 'div' (line 246)
    result_div_283283 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 29), 'div', tpr_283281, float_283282)
    
    # Processing the call keyword arguments (line 246)
    kwargs_283284 = {}
    # Getting the type of 'pow' (line 246)
    pow_283279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 'pow', False)
    # Calling pow(args, kwargs) (line 246)
    pow_call_result_283285 = invoke(stypy.reporting.localization.Localization(__file__, 246, 19), pow_283279, *[float_283280, result_div_283283], **kwargs_283284)
    
    # Assigning a type to the variable 'tref' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'tref', pow_call_result_283285)
    
    # Call to sqrt(...): (line 247)
    # Processing the call arguments (line 247)
    
    
    # Call to log(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'tref' (line 247)
    tref_283288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'tref', False)
    # Processing the call keyword arguments (line 247)
    kwargs_283289 = {}
    # Getting the type of 'log' (line 247)
    log_283287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'log', False)
    # Calling log(args, kwargs) (line 247)
    log_call_result_283290 = invoke(stypy.reporting.localization.Localization(__file__, 247, 25), log_283287, *[tref_283288], **kwargs_283289)
    
    # Applying the 'usub' unary operator (line 247)
    result___neg___283291 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 24), 'usub', log_call_result_283290)
    
    # Getting the type of 'a' (line 247)
    a_283292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'a', False)
    # Applying the binary operator 'div' (line 247)
    result_div_283293 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 24), 'div', result___neg___283291, a_283292)
    
    # Processing the call keyword arguments (line 247)
    kwargs_283294 = {}
    # Getting the type of 'sqrt' (line 247)
    sqrt_283286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 19), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 247)
    sqrt_call_result_283295 = invoke(stypy.reporting.localization.Localization(__file__, 247, 19), sqrt_283286, *[result_div_283293], **kwargs_283294)
    
    # Assigning a type to the variable 'stypy_return_type' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'stypy_return_type', sqrt_call_result_283295)
    # SSA branch for the else part of an if statement (line 241)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 249)
    # Processing the call arguments (line 249)
    str_283297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'str', "If `t` is a string, it must be 'cutoff'")
    # Processing the call keyword arguments (line 249)
    kwargs_283298 = {}
    # Getting the type of 'ValueError' (line 249)
    ValueError_283296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 249)
    ValueError_call_result_283299 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), ValueError_283296, *[str_283297], **kwargs_283298)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 249, 12), ValueError_call_result_283299, 'raise parameter', BaseException)
    # SSA join for if statement (line 241)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 251):
    
    # Assigning a Call to a Name (line 251):
    
    # Call to exp(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Getting the type of 'a' (line 251)
    a_283301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'a', False)
    # Applying the 'usub' unary operator (line 251)
    result___neg___283302 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), 'usub', a_283301)
    
    # Getting the type of 't' (line 251)
    t_283303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 't', False)
    # Applying the binary operator '*' (line 251)
    result_mul_283304 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), '*', result___neg___283302, t_283303)
    
    # Getting the type of 't' (line 251)
    t_283305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 't', False)
    # Applying the binary operator '*' (line 251)
    result_mul_283306 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 22), '*', result_mul_283304, t_283305)
    
    # Processing the call keyword arguments (line 251)
    kwargs_283307 = {}
    # Getting the type of 'exp' (line 251)
    exp_283300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'exp', False)
    # Calling exp(args, kwargs) (line 251)
    exp_call_result_283308 = invoke(stypy.reporting.localization.Localization(__file__, 251, 11), exp_283300, *[result_mul_283306], **kwargs_283307)
    
    # Assigning a type to the variable 'yenv' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'yenv', exp_call_result_283308)
    
    # Assigning a BinOp to a Name (line 252):
    
    # Assigning a BinOp to a Name (line 252):
    # Getting the type of 'yenv' (line 252)
    yenv_283309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 9), 'yenv')
    
    # Call to cos(...): (line 252)
    # Processing the call arguments (line 252)
    int_283311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 20), 'int')
    # Getting the type of 'pi' (line 252)
    pi_283312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'pi', False)
    # Applying the binary operator '*' (line 252)
    result_mul_283313 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 20), '*', int_283311, pi_283312)
    
    # Getting the type of 'fc' (line 252)
    fc_283314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'fc', False)
    # Applying the binary operator '*' (line 252)
    result_mul_283315 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 27), '*', result_mul_283313, fc_283314)
    
    # Getting the type of 't' (line 252)
    t_283316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 't', False)
    # Applying the binary operator '*' (line 252)
    result_mul_283317 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 32), '*', result_mul_283315, t_283316)
    
    # Processing the call keyword arguments (line 252)
    kwargs_283318 = {}
    # Getting the type of 'cos' (line 252)
    cos_283310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'cos', False)
    # Calling cos(args, kwargs) (line 252)
    cos_call_result_283319 = invoke(stypy.reporting.localization.Localization(__file__, 252, 16), cos_283310, *[result_mul_283317], **kwargs_283318)
    
    # Applying the binary operator '*' (line 252)
    result_mul_283320 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 9), '*', yenv_283309, cos_call_result_283319)
    
    # Assigning a type to the variable 'yI' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'yI', result_mul_283320)
    
    # Assigning a BinOp to a Name (line 253):
    
    # Assigning a BinOp to a Name (line 253):
    # Getting the type of 'yenv' (line 253)
    yenv_283321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 9), 'yenv')
    
    # Call to sin(...): (line 253)
    # Processing the call arguments (line 253)
    int_283323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'int')
    # Getting the type of 'pi' (line 253)
    pi_283324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'pi', False)
    # Applying the binary operator '*' (line 253)
    result_mul_283325 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 20), '*', int_283323, pi_283324)
    
    # Getting the type of 'fc' (line 253)
    fc_283326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 29), 'fc', False)
    # Applying the binary operator '*' (line 253)
    result_mul_283327 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 27), '*', result_mul_283325, fc_283326)
    
    # Getting the type of 't' (line 253)
    t_283328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 't', False)
    # Applying the binary operator '*' (line 253)
    result_mul_283329 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 32), '*', result_mul_283327, t_283328)
    
    # Processing the call keyword arguments (line 253)
    kwargs_283330 = {}
    # Getting the type of 'sin' (line 253)
    sin_283322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'sin', False)
    # Calling sin(args, kwargs) (line 253)
    sin_call_result_283331 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), sin_283322, *[result_mul_283329], **kwargs_283330)
    
    # Applying the binary operator '*' (line 253)
    result_mul_283332 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 9), '*', yenv_283321, sin_call_result_283331)
    
    # Assigning a type to the variable 'yQ' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'yQ', result_mul_283332)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'retquad' (line 254)
    retquad_283333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'retquad')
    # Applying the 'not' unary operator (line 254)
    result_not__283334 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 7), 'not', retquad_283333)
    
    
    # Getting the type of 'retenv' (line 254)
    retenv_283335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'retenv')
    # Applying the 'not' unary operator (line 254)
    result_not__283336 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 23), 'not', retenv_283335)
    
    # Applying the binary operator 'and' (line 254)
    result_and_keyword_283337 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 7), 'and', result_not__283334, result_not__283336)
    
    # Testing the type of an if condition (line 254)
    if_condition_283338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 4), result_and_keyword_283337)
    # Assigning a type to the variable 'if_condition_283338' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'if_condition_283338', if_condition_283338)
    # SSA begins for if statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'yI' (line 255)
    yI_283339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'yI')
    # Assigning a type to the variable 'stypy_return_type' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', yI_283339)
    # SSA join for if statement (line 254)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'retquad' (line 256)
    retquad_283340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'retquad')
    # Applying the 'not' unary operator (line 256)
    result_not__283341 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 7), 'not', retquad_283340)
    
    # Getting the type of 'retenv' (line 256)
    retenv_283342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'retenv')
    # Applying the binary operator 'and' (line 256)
    result_and_keyword_283343 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 7), 'and', result_not__283341, retenv_283342)
    
    # Testing the type of an if condition (line 256)
    if_condition_283344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 4), result_and_keyword_283343)
    # Assigning a type to the variable 'if_condition_283344' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'if_condition_283344', if_condition_283344)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 257)
    tuple_283345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 257)
    # Adding element type (line 257)
    # Getting the type of 'yI' (line 257)
    yI_283346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'yI')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), tuple_283345, yI_283346)
    # Adding element type (line 257)
    # Getting the type of 'yenv' (line 257)
    yenv_283347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 19), 'yenv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), tuple_283345, yenv_283347)
    
    # Assigning a type to the variable 'stypy_return_type' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'stypy_return_type', tuple_283345)
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'retquad' (line 258)
    retquad_283348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 7), 'retquad')
    
    # Getting the type of 'retenv' (line 258)
    retenv_283349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'retenv')
    # Applying the 'not' unary operator (line 258)
    result_not__283350 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 19), 'not', retenv_283349)
    
    # Applying the binary operator 'and' (line 258)
    result_and_keyword_283351 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 7), 'and', retquad_283348, result_not__283350)
    
    # Testing the type of an if condition (line 258)
    if_condition_283352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 4), result_and_keyword_283351)
    # Assigning a type to the variable 'if_condition_283352' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'if_condition_283352', if_condition_283352)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 259)
    tuple_283353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 259)
    # Adding element type (line 259)
    # Getting the type of 'yI' (line 259)
    yI_283354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'yI')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 15), tuple_283353, yI_283354)
    # Adding element type (line 259)
    # Getting the type of 'yQ' (line 259)
    yQ_283355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 19), 'yQ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 15), tuple_283353, yQ_283355)
    
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'stypy_return_type', tuple_283353)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'retquad' (line 260)
    retquad_283356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'retquad')
    # Getting the type of 'retenv' (line 260)
    retenv_283357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'retenv')
    # Applying the binary operator 'and' (line 260)
    result_and_keyword_283358 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), 'and', retquad_283356, retenv_283357)
    
    # Testing the type of an if condition (line 260)
    if_condition_283359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 4), result_and_keyword_283358)
    # Assigning a type to the variable 'if_condition_283359' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'if_condition_283359', if_condition_283359)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 261)
    tuple_283360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 261)
    # Adding element type (line 261)
    # Getting the type of 'yI' (line 261)
    yI_283361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'yI')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 15), tuple_283360, yI_283361)
    # Adding element type (line 261)
    # Getting the type of 'yQ' (line 261)
    yQ_283362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'yQ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 15), tuple_283360, yQ_283362)
    # Adding element type (line 261)
    # Getting the type of 'yenv' (line 261)
    yenv_283363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'yenv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 15), tuple_283360, yenv_283363)
    
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'stypy_return_type', tuple_283360)
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'gausspulse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gausspulse' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_283364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283364)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gausspulse'
    return stypy_return_type_283364

# Assigning a type to the variable 'gausspulse' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'gausspulse', gausspulse)

@norecursion
def chirp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_283365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'str', 'linear')
    int_283366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 46), 'int')
    # Getting the type of 'True' (line 264)
    True_283367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 61), 'True')
    defaults = [str_283365, int_283366, True_283367]
    # Create a new context for function 'chirp'
    module_type_store = module_type_store.open_function_context('chirp', 264, 0, False)
    
    # Passed parameters checking function
    chirp.stypy_localization = localization
    chirp.stypy_type_of_self = None
    chirp.stypy_type_store = module_type_store
    chirp.stypy_function_name = 'chirp'
    chirp.stypy_param_names_list = ['t', 'f0', 't1', 'f1', 'method', 'phi', 'vertex_zero']
    chirp.stypy_varargs_param_name = None
    chirp.stypy_kwargs_param_name = None
    chirp.stypy_call_defaults = defaults
    chirp.stypy_call_varargs = varargs
    chirp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chirp', ['t', 'f0', 't1', 'f1', 'method', 'phi', 'vertex_zero'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chirp', localization, ['t', 'f0', 't1', 'f1', 'method', 'phi', 'vertex_zero'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chirp(...)' code ##################

    str_283368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, (-1)), 'str', 'Frequency-swept cosine generator.\n\n    In the following, \'Hz\' should be interpreted as \'cycles per unit\';\n    there is no requirement here that the unit is one second.  The\n    important distinction is that the units of rotation are cycles, not\n    radians. Likewise, `t` could be a measurement of space instead of time.\n\n    Parameters\n    ----------\n    t : array_like\n        Times at which to evaluate the waveform.\n    f0 : float\n        Frequency (e.g. Hz) at time t=0.\n    t1 : float\n        Time at which `f1` is specified.\n    f1 : float\n        Frequency (e.g. Hz) of the waveform at time `t1`.\n    method : {\'linear\', \'quadratic\', \'logarithmic\', \'hyperbolic\'}, optional\n        Kind of frequency sweep.  If not given, `linear` is assumed.  See\n        Notes below for more details.\n    phi : float, optional\n        Phase offset, in degrees. Default is 0.\n    vertex_zero : bool, optional\n        This parameter is only used when `method` is \'quadratic\'.\n        It determines whether the vertex of the parabola that is the graph\n        of the frequency is at t=0 or t=t1.\n\n    Returns\n    -------\n    y : ndarray\n        A numpy array containing the signal evaluated at `t` with the\n        requested time-varying frequency.  More precisely, the function\n        returns ``cos(phase + (pi/180)*phi)`` where `phase` is the integral\n        (from 0 to `t`) of ``2*pi*f(t)``. ``f(t)`` is defined below.\n\n    See Also\n    --------\n    sweep_poly\n\n    Notes\n    -----\n    There are four options for the `method`.  The following formulas give\n    the instantaneous frequency (in Hz) of the signal generated by\n    `chirp()`.  For convenience, the shorter names shown below may also be\n    used.\n\n    linear, lin, li:\n\n        ``f(t) = f0 + (f1 - f0) * t / t1``\n\n    quadratic, quad, q:\n\n        The graph of the frequency f(t) is a parabola through (0, f0) and\n        (t1, f1).  By default, the vertex of the parabola is at (0, f0).\n        If `vertex_zero` is False, then the vertex is at (t1, f1).  The\n        formula is:\n\n        if vertex_zero is True:\n\n            ``f(t) = f0 + (f1 - f0) * t**2 / t1**2``\n\n        else:\n\n            ``f(t) = f1 - (f1 - f0) * (t1 - t)**2 / t1**2``\n\n        To use a more general quadratic function, or an arbitrary\n        polynomial, use the function `scipy.signal.waveforms.sweep_poly`.\n\n    logarithmic, log, lo:\n\n        ``f(t) = f0 * (f1/f0)**(t/t1)``\n\n        f0 and f1 must be nonzero and have the same sign.\n\n        This signal is also known as a geometric or exponential chirp.\n\n    hyperbolic, hyp:\n\n        ``f(t) = f0*f1*t1 / ((f0 - f1)*t + f1*t1)``\n\n        f0 and f1 must be nonzero.\n\n    Examples\n    --------\n    The following will be used in the examples:\n\n    >>> from scipy.signal import chirp, spectrogram\n    >>> import matplotlib.pyplot as plt\n\n    For the first example, we\'ll plot the waveform for a linear chirp\n    from 6 Hz to 1 Hz over 10 seconds:\n\n    >>> t = np.linspace(0, 10, 5001)\n    >>> w = chirp(t, f0=6, f1=1, t1=10, method=\'linear\')\n    >>> plt.plot(t, w)\n    >>> plt.title("Linear Chirp, f(0)=6, f(10)=1")\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.show()\n\n    For the remaining examples, we\'ll use higher frequency ranges,\n    and demonstrate the result using `scipy.signal.spectrogram`.\n    We\'ll use a 10 second interval sampled at 8000 Hz.\n\n    >>> fs = 8000\n    >>> T = 10\n    >>> t = np.linspace(0, T, T*fs, endpoint=False)\n\n    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds\n    (vertex of the parabolic curve of the frequency is at t=0):\n\n    >>> w = chirp(t, f0=1500, f1=250, t1=10, method=\'quadratic\')\n    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,\n    ...                           nfft=2048)\n    >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap=\'gray_r\')\n    >>> plt.title(\'Quadratic Chirp, f(0)=1500, f(10)=250\')\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.ylabel(\'Frequency (Hz)\')\n    >>> plt.grid()\n    >>> plt.show()\n\n    Quadratic chirp from 1500 Hz to 250 Hz over 10 seconds\n    (vertex of the parabolic curve of the frequency is at t=10):\n\n    >>> w = chirp(t, f0=1500, f1=250, t1=10, method=\'quadratic\',\n    ...           vertex_zero=False)\n    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,\n    ...                           nfft=2048)\n    >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap=\'gray_r\')\n    >>> plt.title(\'Quadratic Chirp, f(0)=2500, f(10)=250\\n\' +\n    ...           \'(vertex_zero=False)\')\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.ylabel(\'Frequency (Hz)\')\n    >>> plt.grid()\n    >>> plt.show()\n\n    Logarithmic chirp from 1500 Hz to 250 Hz over 10 seconds:\n\n    >>> w = chirp(t, f0=1500, f1=250, t1=10, method=\'logarithmic\')\n    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,\n    ...                           nfft=2048)\n    >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap=\'gray_r\')\n    >>> plt.title(\'Logarithmic Chirp, f(0)=1500, f(10)=250\')\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.ylabel(\'Frequency (Hz)\')\n    >>> plt.grid()\n    >>> plt.show()\n\n    Hyperbolic chirp from 1500 Hz to 250 Hz over 10 seconds:\n\n    >>> w = chirp(t, f0=1500, f1=250, t1=10, method=\'hyperbolic\')\n    >>> ff, tt, Sxx = spectrogram(w, fs=fs, noverlap=256, nperseg=512,\n    ...                           nfft=2048)\n    >>> plt.pcolormesh(tt, ff[:513], Sxx[:513], cmap=\'gray_r\')\n    >>> plt.title(\'Hyperbolic Chirp, f(0)=1500, f(10)=250\')\n    >>> plt.xlabel(\'t (sec)\')\n    >>> plt.ylabel(\'Frequency (Hz)\')\n    >>> plt.grid()\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 426):
    
    # Assigning a Call to a Name (line 426):
    
    # Call to _chirp_phase(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 't' (line 426)
    t_283370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 25), 't', False)
    # Getting the type of 'f0' (line 426)
    f0_283371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'f0', False)
    # Getting the type of 't1' (line 426)
    t1_283372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 32), 't1', False)
    # Getting the type of 'f1' (line 426)
    f1_283373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 36), 'f1', False)
    # Getting the type of 'method' (line 426)
    method_283374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 40), 'method', False)
    # Getting the type of 'vertex_zero' (line 426)
    vertex_zero_283375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 48), 'vertex_zero', False)
    # Processing the call keyword arguments (line 426)
    kwargs_283376 = {}
    # Getting the type of '_chirp_phase' (line 426)
    _chirp_phase_283369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), '_chirp_phase', False)
    # Calling _chirp_phase(args, kwargs) (line 426)
    _chirp_phase_call_result_283377 = invoke(stypy.reporting.localization.Localization(__file__, 426, 12), _chirp_phase_283369, *[t_283370, f0_283371, t1_283372, f1_283373, method_283374, vertex_zero_283375], **kwargs_283376)
    
    # Assigning a type to the variable 'phase' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'phase', _chirp_phase_call_result_283377)
    
    # Getting the type of 'phi' (line 428)
    phi_283378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'phi')
    # Getting the type of 'pi' (line 428)
    pi_283379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 11), 'pi')
    int_283380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 16), 'int')
    # Applying the binary operator 'div' (line 428)
    result_div_283381 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 11), 'div', pi_283379, int_283380)
    
    # Applying the binary operator '*=' (line 428)
    result_imul_283382 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 4), '*=', phi_283378, result_div_283381)
    # Assigning a type to the variable 'phi' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'phi', result_imul_283382)
    
    
    # Call to cos(...): (line 429)
    # Processing the call arguments (line 429)
    # Getting the type of 'phase' (line 429)
    phase_283384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 15), 'phase', False)
    # Getting the type of 'phi' (line 429)
    phi_283385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 23), 'phi', False)
    # Applying the binary operator '+' (line 429)
    result_add_283386 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 15), '+', phase_283384, phi_283385)
    
    # Processing the call keyword arguments (line 429)
    kwargs_283387 = {}
    # Getting the type of 'cos' (line 429)
    cos_283383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 11), 'cos', False)
    # Calling cos(args, kwargs) (line 429)
    cos_call_result_283388 = invoke(stypy.reporting.localization.Localization(__file__, 429, 11), cos_283383, *[result_add_283386], **kwargs_283387)
    
    # Assigning a type to the variable 'stypy_return_type' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'stypy_return_type', cos_call_result_283388)
    
    # ################# End of 'chirp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chirp' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_283389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283389)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chirp'
    return stypy_return_type_283389

# Assigning a type to the variable 'chirp' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'chirp', chirp)

@norecursion
def _chirp_phase(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_283390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 39), 'str', 'linear')
    # Getting the type of 'True' (line 432)
    True_283391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 61), 'True')
    defaults = [str_283390, True_283391]
    # Create a new context for function '_chirp_phase'
    module_type_store = module_type_store.open_function_context('_chirp_phase', 432, 0, False)
    
    # Passed parameters checking function
    _chirp_phase.stypy_localization = localization
    _chirp_phase.stypy_type_of_self = None
    _chirp_phase.stypy_type_store = module_type_store
    _chirp_phase.stypy_function_name = '_chirp_phase'
    _chirp_phase.stypy_param_names_list = ['t', 'f0', 't1', 'f1', 'method', 'vertex_zero']
    _chirp_phase.stypy_varargs_param_name = None
    _chirp_phase.stypy_kwargs_param_name = None
    _chirp_phase.stypy_call_defaults = defaults
    _chirp_phase.stypy_call_varargs = varargs
    _chirp_phase.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_chirp_phase', ['t', 'f0', 't1', 'f1', 'method', 'vertex_zero'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_chirp_phase', localization, ['t', 'f0', 't1', 'f1', 'method', 'vertex_zero'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_chirp_phase(...)' code ##################

    str_283392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, (-1)), 'str', '\n    Calculate the phase used by chirp_phase to generate its output.\n\n    See `chirp` for a description of the arguments.\n\n    ')
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to asarray(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 't' (line 439)
    t_283394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 't', False)
    # Processing the call keyword arguments (line 439)
    kwargs_283395 = {}
    # Getting the type of 'asarray' (line 439)
    asarray_283393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'asarray', False)
    # Calling asarray(args, kwargs) (line 439)
    asarray_call_result_283396 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), asarray_283393, *[t_283394], **kwargs_283395)
    
    # Assigning a type to the variable 't' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 't', asarray_call_result_283396)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to float(...): (line 440)
    # Processing the call arguments (line 440)
    # Getting the type of 'f0' (line 440)
    f0_283398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'f0', False)
    # Processing the call keyword arguments (line 440)
    kwargs_283399 = {}
    # Getting the type of 'float' (line 440)
    float_283397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 9), 'float', False)
    # Calling float(args, kwargs) (line 440)
    float_call_result_283400 = invoke(stypy.reporting.localization.Localization(__file__, 440, 9), float_283397, *[f0_283398], **kwargs_283399)
    
    # Assigning a type to the variable 'f0' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'f0', float_call_result_283400)
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to float(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 't1' (line 441)
    t1_283402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), 't1', False)
    # Processing the call keyword arguments (line 441)
    kwargs_283403 = {}
    # Getting the type of 'float' (line 441)
    float_283401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 9), 'float', False)
    # Calling float(args, kwargs) (line 441)
    float_call_result_283404 = invoke(stypy.reporting.localization.Localization(__file__, 441, 9), float_283401, *[t1_283402], **kwargs_283403)
    
    # Assigning a type to the variable 't1' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 't1', float_call_result_283404)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to float(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'f1' (line 442)
    f1_283406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'f1', False)
    # Processing the call keyword arguments (line 442)
    kwargs_283407 = {}
    # Getting the type of 'float' (line 442)
    float_283405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 9), 'float', False)
    # Calling float(args, kwargs) (line 442)
    float_call_result_283408 = invoke(stypy.reporting.localization.Localization(__file__, 442, 9), float_283405, *[f1_283406], **kwargs_283407)
    
    # Assigning a type to the variable 'f1' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'f1', float_call_result_283408)
    
    
    # Getting the type of 'method' (line 443)
    method_283409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 443)
    list_283410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 443)
    # Adding element type (line 443)
    str_283411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 18), 'str', 'linear')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_283410, str_283411)
    # Adding element type (line 443)
    str_283412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 28), 'str', 'lin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_283410, str_283412)
    # Adding element type (line 443)
    str_283413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 35), 'str', 'li')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 17), list_283410, str_283413)
    
    # Applying the binary operator 'in' (line 443)
    result_contains_283414 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 7), 'in', method_283409, list_283410)
    
    # Testing the type of an if condition (line 443)
    if_condition_283415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), result_contains_283414)
    # Assigning a type to the variable 'if_condition_283415' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_283415', if_condition_283415)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 444):
    
    # Assigning a BinOp to a Name (line 444):
    # Getting the type of 'f1' (line 444)
    f1_283416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'f1')
    # Getting the type of 'f0' (line 444)
    f0_283417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 21), 'f0')
    # Applying the binary operator '-' (line 444)
    result_sub_283418 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 16), '-', f1_283416, f0_283417)
    
    # Getting the type of 't1' (line 444)
    t1_283419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 27), 't1')
    # Applying the binary operator 'div' (line 444)
    result_div_283420 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 15), 'div', result_sub_283418, t1_283419)
    
    # Assigning a type to the variable 'beta' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'beta', result_div_283420)
    
    # Assigning a BinOp to a Name (line 445):
    
    # Assigning a BinOp to a Name (line 445):
    int_283421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'int')
    # Getting the type of 'pi' (line 445)
    pi_283422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 20), 'pi')
    # Applying the binary operator '*' (line 445)
    result_mul_283423 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 16), '*', int_283421, pi_283422)
    
    # Getting the type of 'f0' (line 445)
    f0_283424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'f0')
    # Getting the type of 't' (line 445)
    t_283425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 31), 't')
    # Applying the binary operator '*' (line 445)
    result_mul_283426 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 26), '*', f0_283424, t_283425)
    
    float_283427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 35), 'float')
    # Getting the type of 'beta' (line 445)
    beta_283428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 41), 'beta')
    # Applying the binary operator '*' (line 445)
    result_mul_283429 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 35), '*', float_283427, beta_283428)
    
    # Getting the type of 't' (line 445)
    t_283430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 48), 't')
    # Applying the binary operator '*' (line 445)
    result_mul_283431 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 46), '*', result_mul_283429, t_283430)
    
    # Getting the type of 't' (line 445)
    t_283432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 52), 't')
    # Applying the binary operator '*' (line 445)
    result_mul_283433 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 50), '*', result_mul_283431, t_283432)
    
    # Applying the binary operator '+' (line 445)
    result_add_283434 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 26), '+', result_mul_283426, result_mul_283433)
    
    # Applying the binary operator '*' (line 445)
    result_mul_283435 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 23), '*', result_mul_283423, result_add_283434)
    
    # Assigning a type to the variable 'phase' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'phase', result_mul_283435)
    # SSA branch for the else part of an if statement (line 443)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 447)
    method_283436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 9), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 447)
    list_283437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 447)
    # Adding element type (line 447)
    str_283438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 20), 'str', 'quadratic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 19), list_283437, str_283438)
    # Adding element type (line 447)
    str_283439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 33), 'str', 'quad')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 19), list_283437, str_283439)
    # Adding element type (line 447)
    str_283440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 41), 'str', 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 19), list_283437, str_283440)
    
    # Applying the binary operator 'in' (line 447)
    result_contains_283441 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 9), 'in', method_283436, list_283437)
    
    # Testing the type of an if condition (line 447)
    if_condition_283442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 9), result_contains_283441)
    # Assigning a type to the variable 'if_condition_283442' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 9), 'if_condition_283442', if_condition_283442)
    # SSA begins for if statement (line 447)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 448):
    
    # Assigning a BinOp to a Name (line 448):
    # Getting the type of 'f1' (line 448)
    f1_283443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 16), 'f1')
    # Getting the type of 'f0' (line 448)
    f0_283444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 21), 'f0')
    # Applying the binary operator '-' (line 448)
    result_sub_283445 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 16), '-', f1_283443, f0_283444)
    
    # Getting the type of 't1' (line 448)
    t1_283446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), 't1')
    int_283447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 34), 'int')
    # Applying the binary operator '**' (line 448)
    result_pow_283448 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 28), '**', t1_283446, int_283447)
    
    # Applying the binary operator 'div' (line 448)
    result_div_283449 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 15), 'div', result_sub_283445, result_pow_283448)
    
    # Assigning a type to the variable 'beta' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'beta', result_div_283449)
    
    # Getting the type of 'vertex_zero' (line 449)
    vertex_zero_283450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 11), 'vertex_zero')
    # Testing the type of an if condition (line 449)
    if_condition_283451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 8), vertex_zero_283450)
    # Assigning a type to the variable 'if_condition_283451' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'if_condition_283451', if_condition_283451)
    # SSA begins for if statement (line 449)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 450):
    
    # Assigning a BinOp to a Name (line 450):
    int_283452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 20), 'int')
    # Getting the type of 'pi' (line 450)
    pi_283453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 24), 'pi')
    # Applying the binary operator '*' (line 450)
    result_mul_283454 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 20), '*', int_283452, pi_283453)
    
    # Getting the type of 'f0' (line 450)
    f0_283455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'f0')
    # Getting the type of 't' (line 450)
    t_283456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 35), 't')
    # Applying the binary operator '*' (line 450)
    result_mul_283457 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 30), '*', f0_283455, t_283456)
    
    # Getting the type of 'beta' (line 450)
    beta_283458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 39), 'beta')
    # Getting the type of 't' (line 450)
    t_283459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 46), 't')
    int_283460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 51), 'int')
    # Applying the binary operator '**' (line 450)
    result_pow_283461 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 46), '**', t_283459, int_283460)
    
    # Applying the binary operator '*' (line 450)
    result_mul_283462 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 39), '*', beta_283458, result_pow_283461)
    
    int_283463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 55), 'int')
    # Applying the binary operator 'div' (line 450)
    result_div_283464 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 53), 'div', result_mul_283462, int_283463)
    
    # Applying the binary operator '+' (line 450)
    result_add_283465 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 30), '+', result_mul_283457, result_div_283464)
    
    # Applying the binary operator '*' (line 450)
    result_mul_283466 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 27), '*', result_mul_283454, result_add_283465)
    
    # Assigning a type to the variable 'phase' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'phase', result_mul_283466)
    # SSA branch for the else part of an if statement (line 449)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 452):
    
    # Assigning a BinOp to a Name (line 452):
    int_283467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 20), 'int')
    # Getting the type of 'pi' (line 452)
    pi_283468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 24), 'pi')
    # Applying the binary operator '*' (line 452)
    result_mul_283469 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 20), '*', int_283467, pi_283468)
    
    # Getting the type of 'f1' (line 452)
    f1_283470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 30), 'f1')
    # Getting the type of 't' (line 452)
    t_283471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 't')
    # Applying the binary operator '*' (line 452)
    result_mul_283472 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 30), '*', f1_283470, t_283471)
    
    # Getting the type of 'beta' (line 452)
    beta_283473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 39), 'beta')
    # Getting the type of 't1' (line 452)
    t1_283474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 48), 't1')
    # Getting the type of 't' (line 452)
    t_283475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 53), 't')
    # Applying the binary operator '-' (line 452)
    result_sub_283476 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 48), '-', t1_283474, t_283475)
    
    int_283477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 59), 'int')
    # Applying the binary operator '**' (line 452)
    result_pow_283478 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 47), '**', result_sub_283476, int_283477)
    
    # Getting the type of 't1' (line 452)
    t1_283479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 63), 't1')
    int_283480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 69), 'int')
    # Applying the binary operator '**' (line 452)
    result_pow_283481 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 63), '**', t1_283479, int_283480)
    
    # Applying the binary operator '-' (line 452)
    result_sub_283482 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 47), '-', result_pow_283478, result_pow_283481)
    
    # Applying the binary operator '*' (line 452)
    result_mul_283483 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 39), '*', beta_283473, result_sub_283482)
    
    int_283484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 74), 'int')
    # Applying the binary operator 'div' (line 452)
    result_div_283485 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 72), 'div', result_mul_283483, int_283484)
    
    # Applying the binary operator '+' (line 452)
    result_add_283486 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 30), '+', result_mul_283472, result_div_283485)
    
    # Applying the binary operator '*' (line 452)
    result_mul_283487 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 27), '*', result_mul_283469, result_add_283486)
    
    # Assigning a type to the variable 'phase' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'phase', result_mul_283487)
    # SSA join for if statement (line 449)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 447)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 454)
    method_283488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 9), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 454)
    list_283489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 454)
    # Adding element type (line 454)
    str_283490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 20), 'str', 'logarithmic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 19), list_283489, str_283490)
    # Adding element type (line 454)
    str_283491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 35), 'str', 'log')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 19), list_283489, str_283491)
    # Adding element type (line 454)
    str_283492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 42), 'str', 'lo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 19), list_283489, str_283492)
    
    # Applying the binary operator 'in' (line 454)
    result_contains_283493 = python_operator(stypy.reporting.localization.Localization(__file__, 454, 9), 'in', method_283488, list_283489)
    
    # Testing the type of an if condition (line 454)
    if_condition_283494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 9), result_contains_283493)
    # Assigning a type to the variable 'if_condition_283494' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 9), 'if_condition_283494', if_condition_283494)
    # SSA begins for if statement (line 454)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'f0' (line 455)
    f0_283495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'f0')
    # Getting the type of 'f1' (line 455)
    f1_283496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'f1')
    # Applying the binary operator '*' (line 455)
    result_mul_283497 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), '*', f0_283495, f1_283496)
    
    float_283498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 22), 'float')
    # Applying the binary operator '<=' (line 455)
    result_le_283499 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), '<=', result_mul_283497, float_283498)
    
    # Testing the type of an if condition (line 455)
    if_condition_283500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), result_le_283499)
    # Assigning a type to the variable 'if_condition_283500' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_283500', if_condition_283500)
    # SSA begins for if statement (line 455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 456)
    # Processing the call arguments (line 456)
    str_283502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 29), 'str', 'For a logarithmic chirp, f0 and f1 must be nonzero and have the same sign.')
    # Processing the call keyword arguments (line 456)
    kwargs_283503 = {}
    # Getting the type of 'ValueError' (line 456)
    ValueError_283501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 456)
    ValueError_call_result_283504 = invoke(stypy.reporting.localization.Localization(__file__, 456, 18), ValueError_283501, *[str_283502], **kwargs_283503)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 456, 12), ValueError_call_result_283504, 'raise parameter', BaseException)
    # SSA join for if statement (line 455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'f0' (line 458)
    f0_283505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 11), 'f0')
    # Getting the type of 'f1' (line 458)
    f1_283506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 17), 'f1')
    # Applying the binary operator '==' (line 458)
    result_eq_283507 = python_operator(stypy.reporting.localization.Localization(__file__, 458, 11), '==', f0_283505, f1_283506)
    
    # Testing the type of an if condition (line 458)
    if_condition_283508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 8), result_eq_283507)
    # Assigning a type to the variable 'if_condition_283508' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'if_condition_283508', if_condition_283508)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 459):
    
    # Assigning a BinOp to a Name (line 459):
    int_283509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 20), 'int')
    # Getting the type of 'pi' (line 459)
    pi_283510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 24), 'pi')
    # Applying the binary operator '*' (line 459)
    result_mul_283511 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 20), '*', int_283509, pi_283510)
    
    # Getting the type of 'f0' (line 459)
    f0_283512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 29), 'f0')
    # Applying the binary operator '*' (line 459)
    result_mul_283513 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 27), '*', result_mul_283511, f0_283512)
    
    # Getting the type of 't' (line 459)
    t_283514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), 't')
    # Applying the binary operator '*' (line 459)
    result_mul_283515 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 32), '*', result_mul_283513, t_283514)
    
    # Assigning a type to the variable 'phase' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'phase', result_mul_283515)
    # SSA branch for the else part of an if statement (line 458)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 461):
    
    # Assigning a BinOp to a Name (line 461):
    # Getting the type of 't1' (line 461)
    t1_283516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 't1')
    
    # Call to log(...): (line 461)
    # Processing the call arguments (line 461)
    # Getting the type of 'f1' (line 461)
    f1_283518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'f1', False)
    # Getting the type of 'f0' (line 461)
    f0_283519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 33), 'f0', False)
    # Applying the binary operator 'div' (line 461)
    result_div_283520 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 28), 'div', f1_283518, f0_283519)
    
    # Processing the call keyword arguments (line 461)
    kwargs_283521 = {}
    # Getting the type of 'log' (line 461)
    log_283517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 24), 'log', False)
    # Calling log(args, kwargs) (line 461)
    log_call_result_283522 = invoke(stypy.reporting.localization.Localization(__file__, 461, 24), log_283517, *[result_div_283520], **kwargs_283521)
    
    # Applying the binary operator 'div' (line 461)
    result_div_283523 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 19), 'div', t1_283516, log_call_result_283522)
    
    # Assigning a type to the variable 'beta' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'beta', result_div_283523)
    
    # Assigning a BinOp to a Name (line 462):
    
    # Assigning a BinOp to a Name (line 462):
    int_283524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 20), 'int')
    # Getting the type of 'pi' (line 462)
    pi_283525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 24), 'pi')
    # Applying the binary operator '*' (line 462)
    result_mul_283526 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 20), '*', int_283524, pi_283525)
    
    # Getting the type of 'beta' (line 462)
    beta_283527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 29), 'beta')
    # Applying the binary operator '*' (line 462)
    result_mul_283528 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 27), '*', result_mul_283526, beta_283527)
    
    # Getting the type of 'f0' (line 462)
    f0_283529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 36), 'f0')
    # Applying the binary operator '*' (line 462)
    result_mul_283530 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 34), '*', result_mul_283528, f0_283529)
    
    
    # Call to pow(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'f1' (line 462)
    f1_283532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 46), 'f1', False)
    # Getting the type of 'f0' (line 462)
    f0_283533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 51), 'f0', False)
    # Applying the binary operator 'div' (line 462)
    result_div_283534 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 46), 'div', f1_283532, f0_283533)
    
    # Getting the type of 't' (line 462)
    t_283535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 55), 't', False)
    # Getting the type of 't1' (line 462)
    t1_283536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 't1', False)
    # Applying the binary operator 'div' (line 462)
    result_div_283537 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 55), 'div', t_283535, t1_283536)
    
    # Processing the call keyword arguments (line 462)
    kwargs_283538 = {}
    # Getting the type of 'pow' (line 462)
    pow_283531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'pow', False)
    # Calling pow(args, kwargs) (line 462)
    pow_call_result_283539 = invoke(stypy.reporting.localization.Localization(__file__, 462, 42), pow_283531, *[result_div_283534, result_div_283537], **kwargs_283538)
    
    float_283540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 65), 'float')
    # Applying the binary operator '-' (line 462)
    result_sub_283541 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 42), '-', pow_call_result_283539, float_283540)
    
    # Applying the binary operator '*' (line 462)
    result_mul_283542 = python_operator(stypy.reporting.localization.Localization(__file__, 462, 39), '*', result_mul_283530, result_sub_283541)
    
    # Assigning a type to the variable 'phase' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'phase', result_mul_283542)
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 454)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 464)
    method_283543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 9), 'method')
    
    # Obtaining an instance of the builtin type 'list' (line 464)
    list_283544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 464)
    # Adding element type (line 464)
    str_283545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 20), 'str', 'hyperbolic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_283544, str_283545)
    # Adding element type (line 464)
    str_283546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 34), 'str', 'hyp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 19), list_283544, str_283546)
    
    # Applying the binary operator 'in' (line 464)
    result_contains_283547 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 9), 'in', method_283543, list_283544)
    
    # Testing the type of an if condition (line 464)
    if_condition_283548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 9), result_contains_283547)
    # Assigning a type to the variable 'if_condition_283548' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 9), 'if_condition_283548', if_condition_283548)
    # SSA begins for if statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'f0' (line 465)
    f0_283549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'f0')
    int_283550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 17), 'int')
    # Applying the binary operator '==' (line 465)
    result_eq_283551 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 11), '==', f0_283549, int_283550)
    
    
    # Getting the type of 'f1' (line 465)
    f1_283552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 22), 'f1')
    int_283553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 28), 'int')
    # Applying the binary operator '==' (line 465)
    result_eq_283554 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 22), '==', f1_283552, int_283553)
    
    # Applying the binary operator 'or' (line 465)
    result_or_keyword_283555 = python_operator(stypy.reporting.localization.Localization(__file__, 465, 11), 'or', result_eq_283551, result_eq_283554)
    
    # Testing the type of an if condition (line 465)
    if_condition_283556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 8), result_or_keyword_283555)
    # Assigning a type to the variable 'if_condition_283556' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'if_condition_283556', if_condition_283556)
    # SSA begins for if statement (line 465)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 466)
    # Processing the call arguments (line 466)
    str_283558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 29), 'str', 'For a hyperbolic chirp, f0 and f1 must be nonzero.')
    # Processing the call keyword arguments (line 466)
    kwargs_283559 = {}
    # Getting the type of 'ValueError' (line 466)
    ValueError_283557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 466)
    ValueError_call_result_283560 = invoke(stypy.reporting.localization.Localization(__file__, 466, 18), ValueError_283557, *[str_283558], **kwargs_283559)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 466, 12), ValueError_call_result_283560, 'raise parameter', BaseException)
    # SSA join for if statement (line 465)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'f0' (line 468)
    f0_283561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'f0')
    # Getting the type of 'f1' (line 468)
    f1_283562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 17), 'f1')
    # Applying the binary operator '==' (line 468)
    result_eq_283563 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), '==', f0_283561, f1_283562)
    
    # Testing the type of an if condition (line 468)
    if_condition_283564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), result_eq_283563)
    # Assigning a type to the variable 'if_condition_283564' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_283564', if_condition_283564)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 470):
    
    # Assigning a BinOp to a Name (line 470):
    int_283565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 20), 'int')
    # Getting the type of 'pi' (line 470)
    pi_283566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 24), 'pi')
    # Applying the binary operator '*' (line 470)
    result_mul_283567 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 20), '*', int_283565, pi_283566)
    
    # Getting the type of 'f0' (line 470)
    f0_283568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 29), 'f0')
    # Applying the binary operator '*' (line 470)
    result_mul_283569 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 27), '*', result_mul_283567, f0_283568)
    
    # Getting the type of 't' (line 470)
    t_283570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 34), 't')
    # Applying the binary operator '*' (line 470)
    result_mul_283571 = python_operator(stypy.reporting.localization.Localization(__file__, 470, 32), '*', result_mul_283569, t_283570)
    
    # Assigning a type to the variable 'phase' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'phase', result_mul_283571)
    # SSA branch for the else part of an if statement (line 468)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 474):
    
    # Assigning a BinOp to a Name (line 474):
    
    # Getting the type of 'f1' (line 474)
    f1_283572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 20), 'f1')
    # Applying the 'usub' unary operator (line 474)
    result___neg___283573 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 19), 'usub', f1_283572)
    
    # Getting the type of 't1' (line 474)
    t1_283574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 25), 't1')
    # Applying the binary operator '*' (line 474)
    result_mul_283575 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 19), '*', result___neg___283573, t1_283574)
    
    # Getting the type of 'f0' (line 474)
    f0_283576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 31), 'f0')
    # Getting the type of 'f1' (line 474)
    f1_283577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 36), 'f1')
    # Applying the binary operator '-' (line 474)
    result_sub_283578 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 31), '-', f0_283576, f1_283577)
    
    # Applying the binary operator 'div' (line 474)
    result_div_283579 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 28), 'div', result_mul_283575, result_sub_283578)
    
    # Assigning a type to the variable 'sing' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'sing', result_div_283579)
    
    # Assigning a BinOp to a Name (line 475):
    
    # Assigning a BinOp to a Name (line 475):
    int_283580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 20), 'int')
    # Getting the type of 'pi' (line 475)
    pi_283581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 24), 'pi')
    # Applying the binary operator '*' (line 475)
    result_mul_283582 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 20), '*', int_283580, pi_283581)
    
    
    # Getting the type of 'sing' (line 475)
    sing_283583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'sing')
    # Applying the 'usub' unary operator (line 475)
    result___neg___283584 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 30), 'usub', sing_283583)
    
    # Getting the type of 'f0' (line 475)
    f0_283585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 38), 'f0')
    # Applying the binary operator '*' (line 475)
    result_mul_283586 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 30), '*', result___neg___283584, f0_283585)
    
    # Applying the binary operator '*' (line 475)
    result_mul_283587 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 27), '*', result_mul_283582, result_mul_283586)
    
    
    # Call to log(...): (line 475)
    # Processing the call arguments (line 475)
    
    # Call to abs(...): (line 475)
    # Processing the call arguments (line 475)
    int_283591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 55), 'int')
    # Getting the type of 't' (line 475)
    t_283592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 59), 't', False)
    # Getting the type of 'sing' (line 475)
    sing_283593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 61), 'sing', False)
    # Applying the binary operator 'div' (line 475)
    result_div_283594 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 59), 'div', t_283592, sing_283593)
    
    # Applying the binary operator '-' (line 475)
    result_sub_283595 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 55), '-', int_283591, result_div_283594)
    
    # Processing the call keyword arguments (line 475)
    kwargs_283596 = {}
    # Getting the type of 'np' (line 475)
    np_283589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 48), 'np', False)
    # Obtaining the member 'abs' of a type (line 475)
    abs_283590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 48), np_283589, 'abs')
    # Calling abs(args, kwargs) (line 475)
    abs_call_result_283597 = invoke(stypy.reporting.localization.Localization(__file__, 475, 48), abs_283590, *[result_sub_283595], **kwargs_283596)
    
    # Processing the call keyword arguments (line 475)
    kwargs_283598 = {}
    # Getting the type of 'log' (line 475)
    log_283588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 44), 'log', False)
    # Calling log(args, kwargs) (line 475)
    log_call_result_283599 = invoke(stypy.reporting.localization.Localization(__file__, 475, 44), log_283588, *[abs_call_result_283597], **kwargs_283598)
    
    # Applying the binary operator '*' (line 475)
    result_mul_283600 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 42), '*', result_mul_283587, log_call_result_283599)
    
    # Assigning a type to the variable 'phase' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'phase', result_mul_283600)
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 464)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 478)
    # Processing the call arguments (line 478)
    str_283602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 25), 'str', "method must be 'linear', 'quadratic', 'logarithmic', or 'hyperbolic', but a value of %r was given.")
    # Getting the type of 'method' (line 480)
    method_283603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 27), 'method', False)
    # Applying the binary operator '%' (line 478)
    result_mod_283604 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 25), '%', str_283602, method_283603)
    
    # Processing the call keyword arguments (line 478)
    kwargs_283605 = {}
    # Getting the type of 'ValueError' (line 478)
    ValueError_283601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 478)
    ValueError_call_result_283606 = invoke(stypy.reporting.localization.Localization(__file__, 478, 14), ValueError_283601, *[result_mod_283604], **kwargs_283605)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 478, 8), ValueError_call_result_283606, 'raise parameter', BaseException)
    # SSA join for if statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 454)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 447)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'phase' (line 482)
    phase_283607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'phase')
    # Assigning a type to the variable 'stypy_return_type' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type', phase_283607)
    
    # ################# End of '_chirp_phase(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_chirp_phase' in the type store
    # Getting the type of 'stypy_return_type' (line 432)
    stypy_return_type_283608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283608)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_chirp_phase'
    return stypy_return_type_283608

# Assigning a type to the variable '_chirp_phase' (line 432)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 0), '_chirp_phase', _chirp_phase)

@norecursion
def sweep_poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_283609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 28), 'int')
    defaults = [int_283609]
    # Create a new context for function 'sweep_poly'
    module_type_store = module_type_store.open_function_context('sweep_poly', 485, 0, False)
    
    # Passed parameters checking function
    sweep_poly.stypy_localization = localization
    sweep_poly.stypy_type_of_self = None
    sweep_poly.stypy_type_store = module_type_store
    sweep_poly.stypy_function_name = 'sweep_poly'
    sweep_poly.stypy_param_names_list = ['t', 'poly', 'phi']
    sweep_poly.stypy_varargs_param_name = None
    sweep_poly.stypy_kwargs_param_name = None
    sweep_poly.stypy_call_defaults = defaults
    sweep_poly.stypy_call_varargs = varargs
    sweep_poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sweep_poly', ['t', 'poly', 'phi'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sweep_poly', localization, ['t', 'poly', 'phi'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sweep_poly(...)' code ##################

    str_283610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, (-1)), 'str', '\n    Frequency-swept cosine generator, with a time-dependent frequency.\n\n    This function generates a sinusoidal function whose instantaneous\n    frequency varies with time.  The frequency at time `t` is given by\n    the polynomial `poly`.\n\n    Parameters\n    ----------\n    t : ndarray\n        Times at which to evaluate the waveform.\n    poly : 1-D array_like or instance of numpy.poly1d\n        The desired frequency expressed as a polynomial.  If `poly` is\n        a list or ndarray of length n, then the elements of `poly` are\n        the coefficients of the polynomial, and the instantaneous\n        frequency is\n\n          ``f(t) = poly[0]*t**(n-1) + poly[1]*t**(n-2) + ... + poly[n-1]``\n\n        If `poly` is an instance of numpy.poly1d, then the\n        instantaneous frequency is\n\n          ``f(t) = poly(t)``\n\n    phi : float, optional\n        Phase offset, in degrees, Default: 0.\n\n    Returns\n    -------\n    sweep_poly : ndarray\n        A numpy array containing the signal evaluated at `t` with the\n        requested time-varying frequency.  More precisely, the function\n        returns ``cos(phase + (pi/180)*phi)``, where `phase` is the integral\n        (from 0 to t) of ``2 * pi * f(t)``; ``f(t)`` is defined above.\n\n    See Also\n    --------\n    chirp\n\n    Notes\n    -----\n    .. versionadded:: 0.8.0\n\n    If `poly` is a list or ndarray of length `n`, then the elements of\n    `poly` are the coefficients of the polynomial, and the instantaneous\n    frequency is:\n\n        ``f(t) = poly[0]*t**(n-1) + poly[1]*t**(n-2) + ... + poly[n-1]``\n\n    If `poly` is an instance of `numpy.poly1d`, then the instantaneous\n    frequency is:\n\n          ``f(t) = poly(t)``\n\n    Finally, the output `s` is:\n\n        ``cos(phase + (pi/180)*phi)``\n\n    where `phase` is the integral from 0 to `t` of ``2 * pi * f(t)``,\n    ``f(t)`` as defined above.\n\n    Examples\n    --------\n    Compute the waveform with instantaneous frequency::\n\n        f(t) = 0.025*t**3 - 0.36*t**2 + 1.25*t + 2\n\n    over the interval 0 <= t <= 10.\n\n    >>> from scipy.signal import sweep_poly\n    >>> p = np.poly1d([0.025, -0.36, 1.25, 2.0])\n    >>> t = np.linspace(0, 10, 5001)\n    >>> w = sweep_poly(t, p)\n\n    Plot it:\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.subplot(2, 1, 1)\n    >>> plt.plot(t, w)\n    >>> plt.title("Sweep Poly\\nwith frequency " +\n    ...           "$f(t) = 0.025t^3 - 0.36t^2 + 1.25t + 2$")\n    >>> plt.subplot(2, 1, 2)\n    >>> plt.plot(t, p(t), \'r\', label=\'f(t)\')\n    >>> plt.legend()\n    >>> plt.xlabel(\'t\')\n    >>> plt.tight_layout()\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to _sweep_poly_phase(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 't' (line 576)
    t_283612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 30), 't', False)
    # Getting the type of 'poly' (line 576)
    poly_283613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 33), 'poly', False)
    # Processing the call keyword arguments (line 576)
    kwargs_283614 = {}
    # Getting the type of '_sweep_poly_phase' (line 576)
    _sweep_poly_phase_283611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), '_sweep_poly_phase', False)
    # Calling _sweep_poly_phase(args, kwargs) (line 576)
    _sweep_poly_phase_call_result_283615 = invoke(stypy.reporting.localization.Localization(__file__, 576, 12), _sweep_poly_phase_283611, *[t_283612, poly_283613], **kwargs_283614)
    
    # Assigning a type to the variable 'phase' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'phase', _sweep_poly_phase_call_result_283615)
    
    # Getting the type of 'phi' (line 578)
    phi_283616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'phi')
    # Getting the type of 'pi' (line 578)
    pi_283617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'pi')
    int_283618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 16), 'int')
    # Applying the binary operator 'div' (line 578)
    result_div_283619 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 11), 'div', pi_283617, int_283618)
    
    # Applying the binary operator '*=' (line 578)
    result_imul_283620 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 4), '*=', phi_283616, result_div_283619)
    # Assigning a type to the variable 'phi' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'phi', result_imul_283620)
    
    
    # Call to cos(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'phase' (line 579)
    phase_283622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 15), 'phase', False)
    # Getting the type of 'phi' (line 579)
    phi_283623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 23), 'phi', False)
    # Applying the binary operator '+' (line 579)
    result_add_283624 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 15), '+', phase_283622, phi_283623)
    
    # Processing the call keyword arguments (line 579)
    kwargs_283625 = {}
    # Getting the type of 'cos' (line 579)
    cos_283621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 11), 'cos', False)
    # Calling cos(args, kwargs) (line 579)
    cos_call_result_283626 = invoke(stypy.reporting.localization.Localization(__file__, 579, 11), cos_283621, *[result_add_283624], **kwargs_283625)
    
    # Assigning a type to the variable 'stypy_return_type' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type', cos_call_result_283626)
    
    # ################# End of 'sweep_poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sweep_poly' in the type store
    # Getting the type of 'stypy_return_type' (line 485)
    stypy_return_type_283627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283627)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sweep_poly'
    return stypy_return_type_283627

# Assigning a type to the variable 'sweep_poly' (line 485)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 0), 'sweep_poly', sweep_poly)

@norecursion
def _sweep_poly_phase(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_sweep_poly_phase'
    module_type_store = module_type_store.open_function_context('_sweep_poly_phase', 582, 0, False)
    
    # Passed parameters checking function
    _sweep_poly_phase.stypy_localization = localization
    _sweep_poly_phase.stypy_type_of_self = None
    _sweep_poly_phase.stypy_type_store = module_type_store
    _sweep_poly_phase.stypy_function_name = '_sweep_poly_phase'
    _sweep_poly_phase.stypy_param_names_list = ['t', 'poly']
    _sweep_poly_phase.stypy_varargs_param_name = None
    _sweep_poly_phase.stypy_kwargs_param_name = None
    _sweep_poly_phase.stypy_call_defaults = defaults
    _sweep_poly_phase.stypy_call_varargs = varargs
    _sweep_poly_phase.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_sweep_poly_phase', ['t', 'poly'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_sweep_poly_phase', localization, ['t', 'poly'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_sweep_poly_phase(...)' code ##################

    str_283628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, (-1)), 'str', '\n    Calculate the phase used by sweep_poly to generate its output.\n\n    See `sweep_poly` for a description of the arguments.\n\n    ')
    
    # Assigning a Call to a Name (line 590):
    
    # Assigning a Call to a Name (line 590):
    
    # Call to polyint(...): (line 590)
    # Processing the call arguments (line 590)
    # Getting the type of 'poly' (line 590)
    poly_283630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 22), 'poly', False)
    # Processing the call keyword arguments (line 590)
    kwargs_283631 = {}
    # Getting the type of 'polyint' (line 590)
    polyint_283629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 14), 'polyint', False)
    # Calling polyint(args, kwargs) (line 590)
    polyint_call_result_283632 = invoke(stypy.reporting.localization.Localization(__file__, 590, 14), polyint_283629, *[poly_283630], **kwargs_283631)
    
    # Assigning a type to the variable 'intpoly' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'intpoly', polyint_call_result_283632)
    
    # Assigning a BinOp to a Name (line 591):
    
    # Assigning a BinOp to a Name (line 591):
    int_283633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 12), 'int')
    # Getting the type of 'pi' (line 591)
    pi_283634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 16), 'pi')
    # Applying the binary operator '*' (line 591)
    result_mul_283635 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 12), '*', int_283633, pi_283634)
    
    
    # Call to polyval(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'intpoly' (line 591)
    intpoly_283637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 29), 'intpoly', False)
    # Getting the type of 't' (line 591)
    t_283638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 38), 't', False)
    # Processing the call keyword arguments (line 591)
    kwargs_283639 = {}
    # Getting the type of 'polyval' (line 591)
    polyval_283636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 21), 'polyval', False)
    # Calling polyval(args, kwargs) (line 591)
    polyval_call_result_283640 = invoke(stypy.reporting.localization.Localization(__file__, 591, 21), polyval_283636, *[intpoly_283637, t_283638], **kwargs_283639)
    
    # Applying the binary operator '*' (line 591)
    result_mul_283641 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 19), '*', result_mul_283635, polyval_call_result_283640)
    
    # Assigning a type to the variable 'phase' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'phase', result_mul_283641)
    # Getting the type of 'phase' (line 592)
    phase_283642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'phase')
    # Assigning a type to the variable 'stypy_return_type' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'stypy_return_type', phase_283642)
    
    # ################# End of '_sweep_poly_phase(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_sweep_poly_phase' in the type store
    # Getting the type of 'stypy_return_type' (line 582)
    stypy_return_type_283643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283643)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_sweep_poly_phase'
    return stypy_return_type_283643

# Assigning a type to the variable '_sweep_poly_phase' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), '_sweep_poly_phase', _sweep_poly_phase)

@norecursion
def unit_impulse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 595)
    None_283644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 28), 'None')
    # Getting the type of 'float' (line 595)
    float_283645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 40), 'float')
    defaults = [None_283644, float_283645]
    # Create a new context for function 'unit_impulse'
    module_type_store = module_type_store.open_function_context('unit_impulse', 595, 0, False)
    
    # Passed parameters checking function
    unit_impulse.stypy_localization = localization
    unit_impulse.stypy_type_of_self = None
    unit_impulse.stypy_type_store = module_type_store
    unit_impulse.stypy_function_name = 'unit_impulse'
    unit_impulse.stypy_param_names_list = ['shape', 'idx', 'dtype']
    unit_impulse.stypy_varargs_param_name = None
    unit_impulse.stypy_kwargs_param_name = None
    unit_impulse.stypy_call_defaults = defaults
    unit_impulse.stypy_call_varargs = varargs
    unit_impulse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unit_impulse', ['shape', 'idx', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unit_impulse', localization, ['shape', 'idx', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unit_impulse(...)' code ##################

    str_283646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, (-1)), 'str', "\n    Unit impulse signal (discrete delta function) or unit basis vector.\n\n    Parameters\n    ----------\n    shape : int or tuple of int\n        Number of samples in the output (1-D), or a tuple that represents the\n        shape of the output (N-D).\n    idx : None or int or tuple of int or 'mid', optional\n        Index at which the value is 1.  If None, defaults to the 0th element.\n        If ``idx='mid'``, the impulse will be centered at ``shape // 2`` in\n        all dimensions.  If an int, the impulse will be at `idx` in all\n        dimensions.\n    dtype : data-type, optional\n        The desired data-type for the array, e.g., `numpy.int8`.  Default is\n        `numpy.float64`.\n\n    Returns\n    -------\n    y : ndarray\n        Output array containing an impulse signal.\n\n    Notes\n    -----\n    The 1D case is also known as the Kronecker delta.\n\n    .. versionadded:: 0.19.0\n\n    Examples\n    --------\n    An impulse at the 0th element (:math:`\\delta[n]`):\n\n    >>> from scipy import signal\n    >>> signal.unit_impulse(8)\n    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])\n\n    Impulse offset by 2 samples (:math:`\\delta[n-2]`):\n\n    >>> signal.unit_impulse(7, 2)\n    array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.])\n\n    2-dimensional impulse, centered:\n\n    >>> signal.unit_impulse((3, 3), 'mid')\n    array([[ 0.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  0.]])\n\n    Impulse at (2, 2), using broadcasting:\n\n    >>> signal.unit_impulse((4, 4), 2)\n    array([[ 0.,  0.,  0.,  0.],\n           [ 0.,  0.,  0.,  0.],\n           [ 0.,  0.,  1.,  0.],\n           [ 0.,  0.,  0.,  0.]])\n\n    Plot the impulse response of a 4th-order Butterworth lowpass filter:\n\n    >>> imp = signal.unit_impulse(100, 'mid')\n    >>> b, a = signal.butter(4, 0.2)\n    >>> response = signal.lfilter(b, a, imp)\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(np.arange(-50, 50), imp)\n    >>> plt.plot(np.arange(-50, 50), response)\n    >>> plt.margins(0.1, 0.1)\n    >>> plt.xlabel('Time [samples]')\n    >>> plt.ylabel('Amplitude')\n    >>> plt.grid(True)\n    >>> plt.show()\n\n    ")
    
    # Assigning a Call to a Name (line 668):
    
    # Assigning a Call to a Name (line 668):
    
    # Call to zeros(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'shape' (line 668)
    shape_283648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 16), 'shape', False)
    # Getting the type of 'dtype' (line 668)
    dtype_283649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 23), 'dtype', False)
    # Processing the call keyword arguments (line 668)
    kwargs_283650 = {}
    # Getting the type of 'zeros' (line 668)
    zeros_283647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 10), 'zeros', False)
    # Calling zeros(args, kwargs) (line 668)
    zeros_call_result_283651 = invoke(stypy.reporting.localization.Localization(__file__, 668, 10), zeros_283647, *[shape_283648, dtype_283649], **kwargs_283650)
    
    # Assigning a type to the variable 'out' (line 668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'out', zeros_call_result_283651)
    
    # Assigning a Call to a Name (line 670):
    
    # Assigning a Call to a Name (line 670):
    
    # Call to atleast_1d(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'shape' (line 670)
    shape_283654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 26), 'shape', False)
    # Processing the call keyword arguments (line 670)
    kwargs_283655 = {}
    # Getting the type of 'np' (line 670)
    np_283652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 670)
    atleast_1d_283653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 12), np_283652, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 670)
    atleast_1d_call_result_283656 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), atleast_1d_283653, *[shape_283654], **kwargs_283655)
    
    # Assigning a type to the variable 'shape' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'shape', atleast_1d_call_result_283656)
    
    # Type idiom detected: calculating its left and rigth part (line 672)
    # Getting the type of 'idx' (line 672)
    idx_283657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 7), 'idx')
    # Getting the type of 'None' (line 672)
    None_283658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 14), 'None')
    
    (may_be_283659, more_types_in_union_283660) = may_be_none(idx_283657, None_283658)

    if may_be_283659:

        if more_types_in_union_283660:
            # Runtime conditional SSA (line 672)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 673):
        
        # Assigning a BinOp to a Name (line 673):
        
        # Obtaining an instance of the builtin type 'tuple' (line 673)
        tuple_283661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 673)
        # Adding element type (line 673)
        int_283662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 673, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 673, 15), tuple_283661, int_283662)
        
        
        # Call to len(...): (line 673)
        # Processing the call arguments (line 673)
        # Getting the type of 'shape' (line 673)
        shape_283664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 25), 'shape', False)
        # Processing the call keyword arguments (line 673)
        kwargs_283665 = {}
        # Getting the type of 'len' (line 673)
        len_283663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 21), 'len', False)
        # Calling len(args, kwargs) (line 673)
        len_call_result_283666 = invoke(stypy.reporting.localization.Localization(__file__, 673, 21), len_283663, *[shape_283664], **kwargs_283665)
        
        # Applying the binary operator '*' (line 673)
        result_mul_283667 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 14), '*', tuple_283661, len_call_result_283666)
        
        # Assigning a type to the variable 'idx' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 8), 'idx', result_mul_283667)

        if more_types_in_union_283660:
            # Runtime conditional SSA for else branch (line 672)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_283659) or more_types_in_union_283660):
        
        
        # Getting the type of 'idx' (line 674)
        idx_283668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 9), 'idx')
        str_283669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 16), 'str', 'mid')
        # Applying the binary operator '==' (line 674)
        result_eq_283670 = python_operator(stypy.reporting.localization.Localization(__file__, 674, 9), '==', idx_283668, str_283669)
        
        # Testing the type of an if condition (line 674)
        if_condition_283671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 674, 9), result_eq_283670)
        # Assigning a type to the variable 'if_condition_283671' (line 674)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 9), 'if_condition_283671', if_condition_283671)
        # SSA begins for if statement (line 674)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 675):
        
        # Assigning a Call to a Name (line 675):
        
        # Call to tuple(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'shape' (line 675)
        shape_283673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 20), 'shape', False)
        int_283674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 29), 'int')
        # Applying the binary operator '//' (line 675)
        result_floordiv_283675 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 20), '//', shape_283673, int_283674)
        
        # Processing the call keyword arguments (line 675)
        kwargs_283676 = {}
        # Getting the type of 'tuple' (line 675)
        tuple_283672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 14), 'tuple', False)
        # Calling tuple(args, kwargs) (line 675)
        tuple_call_result_283677 = invoke(stypy.reporting.localization.Localization(__file__, 675, 14), tuple_283672, *[result_floordiv_283675], **kwargs_283676)
        
        # Assigning a type to the variable 'idx' (line 675)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 8), 'idx', tuple_call_result_283677)
        # SSA branch for the else part of an if statement (line 674)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 676)
        str_283678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 26), 'str', '__iter__')
        # Getting the type of 'idx' (line 676)
        idx_283679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 21), 'idx')
        
        (may_be_283680, more_types_in_union_283681) = may_not_provide_member(str_283678, idx_283679)

        if may_be_283680:

            if more_types_in_union_283681:
                # Runtime conditional SSA (line 676)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'idx' (line 676)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 9), 'idx', remove_member_provider_from_union(idx_283679, '__iter__'))
            
            # Assigning a BinOp to a Name (line 677):
            
            # Assigning a BinOp to a Name (line 677):
            
            # Obtaining an instance of the builtin type 'tuple' (line 677)
            tuple_283682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 15), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 677)
            # Adding element type (line 677)
            # Getting the type of 'idx' (line 677)
            idx_283683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 15), 'idx')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 15), tuple_283682, idx_283683)
            
            
            # Call to len(...): (line 677)
            # Processing the call arguments (line 677)
            # Getting the type of 'shape' (line 677)
            shape_283685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 27), 'shape', False)
            # Processing the call keyword arguments (line 677)
            kwargs_283686 = {}
            # Getting the type of 'len' (line 677)
            len_283684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 23), 'len', False)
            # Calling len(args, kwargs) (line 677)
            len_call_result_283687 = invoke(stypy.reporting.localization.Localization(__file__, 677, 23), len_283684, *[shape_283685], **kwargs_283686)
            
            # Applying the binary operator '*' (line 677)
            result_mul_283688 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 14), '*', tuple_283682, len_call_result_283687)
            
            # Assigning a type to the variable 'idx' (line 677)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 8), 'idx', result_mul_283688)

            if more_types_in_union_283681:
                # SSA join for if statement (line 676)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 674)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_283659 and more_types_in_union_283660):
            # SSA join for if statement (line 672)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Num to a Subscript (line 679):
    
    # Assigning a Num to a Subscript (line 679):
    int_283689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 15), 'int')
    # Getting the type of 'out' (line 679)
    out_283690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'out')
    # Getting the type of 'idx' (line 679)
    idx_283691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), 'idx')
    # Storing an element on a container (line 679)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 679, 4), out_283690, (idx_283691, int_283689))
    # Getting the type of 'out' (line 680)
    out_283692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'stypy_return_type', out_283692)
    
    # ################# End of 'unit_impulse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unit_impulse' in the type store
    # Getting the type of 'stypy_return_type' (line 595)
    stypy_return_type_283693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_283693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unit_impulse'
    return stypy_return_type_283693

# Assigning a type to the variable 'unit_impulse' (line 595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'unit_impulse', unit_impulse)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
