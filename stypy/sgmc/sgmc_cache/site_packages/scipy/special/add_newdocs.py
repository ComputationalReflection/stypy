
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Docstrings for generated ufuncs
2: #
3: # The syntax is designed to look like the function add_newdoc is being
4: # called from numpy.lib, but in this file add_newdoc puts the
5: # docstrings in a dictionary. This dictionary is used in
6: # generate_ufuncs.py to generate the docstrings for the ufuncs in
7: # scipy.special at the C level when the ufuncs are created at compile
8: # time.
9: #
10: # Note : After editing this file and committing changes, please run
11: # generate_funcs.py and commit the changes as a separate commit with a comment
12: # such as : GEN: special: run generate_ufuncs.py
13: 
14: 
15: from __future__ import division, print_function, absolute_import
16: 
17: docdict = {}
18: 
19: 
20: def get(name):
21:     return docdict.get(name)
22: 
23: 
24: def add_newdoc(place, name, doc):
25:     docdict['.'.join((place, name))] = doc
26: 
27: 
28: add_newdoc("scipy.special", "_sf_error_test_function",
29:     '''
30:     Private function; do not use.
31:     ''')
32: 
33: add_newdoc("scipy.special", "sph_harm",
34:     r'''
35:     sph_harm(m, n, theta, phi)
36: 
37:     Compute spherical harmonics.
38: 
39:     The spherical harmonics are defined as
40: 
41:     .. math::
42: 
43:         Y^m_n(\theta,\phi) = \sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}}
44:           e^{i m \theta} P^m_n(\cos(\phi))
45: 
46:     where :math:`P_n^m` are the associated Legendre functions; see `lpmv`.
47: 
48:     Parameters
49:     ----------
50:     m : array_like
51:         Order of the harmonic (int); must have ``|m| <= n``.
52:     n : array_like
53:        Degree of the harmonic (int); must have ``n >= 0``. This is
54:        often denoted by ``l`` (lower case L) in descriptions of
55:        spherical harmonics.
56:     theta : array_like
57:        Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.
58:     phi : array_like
59:        Polar (colatitudinal) coordinate; must be in ``[0, pi]``.
60: 
61:     Returns
62:     -------
63:     y_mn : complex float
64:        The harmonic :math:`Y^m_n` sampled at ``theta`` and ``phi``.
65: 
66:     Notes
67:     -----
68:     There are different conventions for the meanings of the input
69:     arguments ``theta`` and ``phi``. In SciPy ``theta`` is the
70:     azimuthal angle and ``phi`` is the polar angle. It is common to
71:     see the opposite convention, that is, ``theta`` as the polar angle
72:     and ``phi`` as the azimuthal angle.
73: 
74:     Note that SciPy's spherical harmonics include the Condon-Shortley
75:     phase [2]_ because it is part of `lpmv`.
76: 
77:     With SciPy's conventions, the first several spherical harmonics
78:     are
79: 
80:     .. math::
81: 
82:         Y_0^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{1}{\pi}} \\
83:         Y_1^{-1}(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{2\pi}}
84:                                     e^{-i\theta} \sin(\phi) \\
85:         Y_1^0(\theta, \phi) &= \frac{1}{2} \sqrt{\frac{3}{\pi}}
86:                                  \cos(\phi) \\
87:         Y_1^1(\theta, \phi) &= -\frac{1}{2} \sqrt{\frac{3}{2\pi}}
88:                                  e^{i\theta} \sin(\phi).
89: 
90:     References
91:     ----------
92:     .. [1] Digital Library of Mathematical Functions, 14.30.
93:            http://dlmf.nist.gov/14.30
94:     .. [2] https://en.wikipedia.org/wiki/Spherical_harmonics#Condon.E2.80.93Shortley_phase
95:     ''')
96: 
97: add_newdoc("scipy.special", "_ellip_harm",
98:     '''
99:     Internal function, use `ellip_harm` instead.
100:     ''')
101: 
102: add_newdoc("scipy.special", "_ellip_norm",
103:     '''
104:     Internal function, use `ellip_norm` instead.
105:     ''')
106: 
107: add_newdoc("scipy.special", "_lambertw",
108:     '''
109:     Internal function, use `lambertw` instead.
110:     ''')
111: 
112: add_newdoc("scipy.special", "wrightomega",
113:     r'''
114:     wrightomega(z, out=None)
115: 
116:     Wright Omega function.
117: 
118:     Defined as the solution to
119: 
120:     .. math::
121: 
122:         \omega + \log(\omega) = z
123: 
124:     where :math:`\log` is the principal branch of the complex logarithm.
125: 
126:     Parameters
127:     ----------
128:     z : array_like
129:         Points at which to evaluate the Wright Omega function
130: 
131:     Returns
132:     -------
133:     omega : ndarray
134:         Values of the Wright Omega function
135: 
136:     Notes
137:     -----
138:     .. versionadded:: 0.19.0
139: 
140:     The function can also be defined as
141: 
142:     .. math::
143: 
144:         \omega(z) = W_{K(z)}(e^z)
145: 
146:     where :math:`K(z) = \lceil (\Im(z) - \pi)/(2\pi) \rceil` is the
147:     unwinding number and :math:`W` is the Lambert W function.
148: 
149:     The implementation here is taken from [1]_.
150: 
151:     See Also
152:     --------
153:     lambertw : The Lambert W function
154: 
155:     References
156:     ----------
157:     .. [1] Lawrence, Corless, and Jeffrey, "Algorithm 917: Complex
158:            Double-Precision Evaluation of the Wright :math:`\omega`
159:            Function." ACM Transactions on Mathematical Software,
160:            2012. :doi:`10.1145/2168773.2168779`.
161: 
162:     ''')
163: 
164: 
165: add_newdoc("scipy.special", "agm",
166:     '''
167:     agm(a, b)
168: 
169:     Compute the arithmetic-geometric mean of `a` and `b`.
170: 
171:     Start with a_0 = a and b_0 = b and iteratively compute::
172: 
173:         a_{n+1} = (a_n + b_n)/2
174:         b_{n+1} = sqrt(a_n*b_n)
175: 
176:     a_n and b_n converge to the same limit as n increases; their common
177:     limit is agm(a, b).
178: 
179:     Parameters
180:     ----------
181:     a, b : array_like
182:         Real values only.  If the values are both negative, the result
183:         is negative.  If one value is negative and the other is positive,
184:         `nan` is returned.
185: 
186:     Returns
187:     -------
188:     float
189:         The arithmetic-geometric mean of `a` and `b`.
190: 
191:     Examples
192:     --------
193:     >>> from scipy.special import agm
194:     >>> a, b = 24.0, 6.0
195:     >>> agm(a, b)
196:     13.458171481725614
197: 
198:     Compare that result to the iteration:
199: 
200:     >>> while a != b:
201:     ...     a, b = (a + b)/2, np.sqrt(a*b)
202:     ...     print("a = %19.16f  b=%19.16f" % (a, b))
203:     ...
204:     a = 15.0000000000000000  b=12.0000000000000000
205:     a = 13.5000000000000000  b=13.4164078649987388
206:     a = 13.4582039324993694  b=13.4581390309909850
207:     a = 13.4581714817451772  b=13.4581714817060547
208:     a = 13.4581714817256159  b=13.4581714817256159
209: 
210:     When array-like arguments are given, broadcasting applies:
211: 
212:     >>> a = np.array([[1.5], [3], [6]])  # a has shape (3, 1).
213:     >>> b = np.array([6, 12, 24, 48])    # b has shape (4,).
214:     >>> agm(a, b)
215:     array([[  3.36454287,   5.42363427,   9.05798751,  15.53650756],
216:            [  4.37037309,   6.72908574,  10.84726853,  18.11597502],
217:            [  6.        ,   8.74074619,  13.45817148,  21.69453707]])
218:     ''')
219: 
220: add_newdoc("scipy.special", "airy",
221:     r'''
222:     airy(z)
223: 
224:     Airy functions and their derivatives.
225: 
226:     Parameters
227:     ----------
228:     z : array_like
229:         Real or complex argument.
230: 
231:     Returns
232:     -------
233:     Ai, Aip, Bi, Bip : ndarrays
234:         Airy functions Ai and Bi, and their derivatives Aip and Bip.
235: 
236:     Notes
237:     -----
238:     The Airy functions Ai and Bi are two independent solutions of
239: 
240:     .. math:: y''(x) = x y(x).
241: 
242:     For real `z` in [-10, 10], the computation is carried out by calling
243:     the Cephes [1]_ `airy` routine, which uses power series summation
244:     for small `z` and rational minimax approximations for large `z`.
245: 
246:     Outside this range, the AMOS [2]_ `zairy` and `zbiry` routines are
247:     employed.  They are computed using power series for :math:`|z| < 1` and
248:     the following relations to modified Bessel functions for larger `z`
249:     (where :math:`t \equiv 2 z^{3/2}/3`):
250: 
251:     .. math::
252: 
253:         Ai(z) = \frac{1}{\pi \sqrt{3}} K_{1/3}(t)
254: 
255:         Ai'(z) = -\frac{z}{\pi \sqrt{3}} K_{2/3}(t)
256: 
257:         Bi(z) = \sqrt{\frac{z}{3}} \left(I_{-1/3}(t) + I_{1/3}(t) \right)
258: 
259:         Bi'(z) = \frac{z}{\sqrt{3}} \left(I_{-2/3}(t) + I_{2/3}(t)\right)
260: 
261:     See also
262:     --------
263:     airye : exponentially scaled Airy functions.
264: 
265:     References
266:     ----------
267:     .. [1] Cephes Mathematical Functions Library,
268:            http://www.netlib.org/cephes/index.html
269:     .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
270:            of a Complex Argument and Nonnegative Order",
271:            http://netlib.org/amos/
272: 
273:     Examples
274:     --------
275:     Compute the Airy functions on the interval [-15, 5].
276: 
277:     >>> from scipy import special
278:     >>> x = np.linspace(-15, 5, 201)
279:     >>> ai, aip, bi, bip = special.airy(x)
280: 
281:     Plot Ai(x) and Bi(x).
282: 
283:     >>> import matplotlib.pyplot as plt
284:     >>> plt.plot(x, ai, 'r', label='Ai(x)')
285:     >>> plt.plot(x, bi, 'b--', label='Bi(x)')
286:     >>> plt.ylim(-0.5, 1.0)
287:     >>> plt.grid()
288:     >>> plt.legend(loc='upper left')
289:     >>> plt.show()
290: 
291:     ''')
292: 
293: add_newdoc("scipy.special", "airye",
294:     '''
295:     airye(z)
296: 
297:     Exponentially scaled Airy functions and their derivatives.
298: 
299:     Scaling::
300: 
301:         eAi  = Ai  * exp(2.0/3.0*z*sqrt(z))
302:         eAip = Aip * exp(2.0/3.0*z*sqrt(z))
303:         eBi  = Bi  * exp(-abs(2.0/3.0*(z*sqrt(z)).real))
304:         eBip = Bip * exp(-abs(2.0/3.0*(z*sqrt(z)).real))
305: 
306:     Parameters
307:     ----------
308:     z : array_like
309:         Real or complex argument.
310: 
311:     Returns
312:     -------
313:     eAi, eAip, eBi, eBip : array_like
314:         Airy functions Ai and Bi, and their derivatives Aip and Bip
315: 
316:     Notes
317:     -----
318:     Wrapper for the AMOS [1]_ routines `zairy` and `zbiry`.
319: 
320:     See also
321:     --------
322:     airy
323: 
324:     References
325:     ----------
326:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
327:            of a Complex Argument and Nonnegative Order",
328:            http://netlib.org/amos/
329:     ''')
330: 
331: add_newdoc("scipy.special", "bdtr",
332:     r'''
333:     bdtr(k, n, p)
334: 
335:     Binomial distribution cumulative distribution function.
336: 
337:     Sum of the terms 0 through `k` of the Binomial probability density.
338: 
339:     .. math::
340:         \mathrm{bdtr}(k, n, p) = \sum_{j=0}^k {{n}\choose{j}} p^j (1-p)^{n-j}
341: 
342:     Parameters
343:     ----------
344:     k : array_like
345:         Number of successes (int).
346:     n : array_like
347:         Number of events (int).
348:     p : array_like
349:         Probability of success in a single event (float).
350: 
351:     Returns
352:     -------
353:     y : ndarray
354:         Probability of `k` or fewer successes in `n` independent events with
355:         success probabilities of `p`.
356: 
357:     Notes
358:     -----
359:     The terms are not summed directly; instead the regularized incomplete beta
360:     function is employed, according to the formula,
361: 
362:     .. math::
363:         \mathrm{bdtr}(k, n, p) = I_{1 - p}(n - k, k + 1).
364: 
365:     Wrapper for the Cephes [1]_ routine `bdtr`.
366: 
367:     References
368:     ----------
369:     .. [1] Cephes Mathematical Functions Library,
370:            http://www.netlib.org/cephes/index.html
371: 
372:     ''')
373: 
374: add_newdoc("scipy.special", "bdtrc",
375:     r'''
376:     bdtrc(k, n, p)
377: 
378:     Binomial distribution survival function.
379: 
380:     Sum of the terms `k + 1` through `n` of the binomial probability density,
381: 
382:     .. math::
383:         \mathrm{bdtrc}(k, n, p) = \sum_{j=k+1}^n {{n}\choose{j}} p^j (1-p)^{n-j}
384: 
385:     Parameters
386:     ----------
387:     k : array_like
388:         Number of successes (int).
389:     n : array_like
390:         Number of events (int)
391:     p : array_like
392:         Probability of success in a single event.
393: 
394:     Returns
395:     -------
396:     y : ndarray
397:         Probability of `k + 1` or more successes in `n` independent events
398:         with success probabilities of `p`.
399: 
400:     See also
401:     --------
402:     bdtr
403:     betainc
404: 
405:     Notes
406:     -----
407:     The terms are not summed directly; instead the regularized incomplete beta
408:     function is employed, according to the formula,
409: 
410:     .. math::
411:         \mathrm{bdtrc}(k, n, p) = I_{p}(k + 1, n - k).
412: 
413:     Wrapper for the Cephes [1]_ routine `bdtrc`.
414: 
415:     References
416:     ----------
417:     .. [1] Cephes Mathematical Functions Library,
418:            http://www.netlib.org/cephes/index.html
419: 
420:     ''')
421: 
422: add_newdoc("scipy.special", "bdtri",
423:     '''
424:     bdtri(k, n, y)
425: 
426:     Inverse function to `bdtr` with respect to `p`.
427: 
428:     Finds the event probability `p` such that the sum of the terms 0 through
429:     `k` of the binomial probability density is equal to the given cumulative
430:     probability `y`.
431: 
432:     Parameters
433:     ----------
434:     k : array_like
435:         Number of successes (float).
436:     n : array_like
437:         Number of events (float)
438:     y : array_like
439:         Cumulative probability (probability of `k` or fewer successes in `n`
440:         events).
441: 
442:     Returns
443:     -------
444:     p : ndarray
445:         The event probability such that `bdtr(k, n, p) = y`.
446: 
447:     See also
448:     --------
449:     bdtr
450:     betaincinv
451: 
452:     Notes
453:     -----
454:     The computation is carried out using the inverse beta integral function
455:     and the relation,::
456: 
457:         1 - p = betaincinv(n - k, k + 1, y).
458: 
459:     Wrapper for the Cephes [1]_ routine `bdtri`.
460: 
461:     References
462:     ----------
463:     .. [1] Cephes Mathematical Functions Library,
464:            http://www.netlib.org/cephes/index.html
465:     ''')
466: 
467: add_newdoc("scipy.special", "bdtrik",
468:     '''
469:     bdtrik(y, n, p)
470: 
471:     Inverse function to `bdtr` with respect to `k`.
472: 
473:     Finds the number of successes `k` such that the sum of the terms 0 through
474:     `k` of the Binomial probability density for `n` events with probability
475:     `p` is equal to the given cumulative probability `y`.
476: 
477:     Parameters
478:     ----------
479:     y : array_like
480:         Cumulative probability (probability of `k` or fewer successes in `n`
481:         events).
482:     n : array_like
483:         Number of events (float).
484:     p : array_like
485:         Success probability (float).
486: 
487:     Returns
488:     -------
489:     k : ndarray
490:         The number of successes `k` such that `bdtr(k, n, p) = y`.
491: 
492:     See also
493:     --------
494:     bdtr
495: 
496:     Notes
497:     -----
498:     Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
499:     cumulative incomplete beta distribution.
500: 
501:     Computation of `k` involves a search for a value that produces the desired
502:     value of `y`.  The search relies on the monotonicity of `y` with `k`.
503: 
504:     Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.
505: 
506:     References
507:     ----------
508:     .. [1] Milton Abramowitz and Irene A. Stegun, eds.
509:            Handbook of Mathematical Functions with Formulas,
510:            Graphs, and Mathematical Tables. New York: Dover, 1972.
511:     .. [2] Barry Brown, James Lovato, and Kathy Russell,
512:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
513:            Functions, Inverses, and Other Parameters.
514: 
515:     ''')
516: 
517: add_newdoc("scipy.special", "bdtrin",
518:     '''
519:     bdtrin(k, y, p)
520: 
521:     Inverse function to `bdtr` with respect to `n`.
522: 
523:     Finds the number of events `n` such that the sum of the terms 0 through
524:     `k` of the Binomial probability density for events with probability `p` is
525:     equal to the given cumulative probability `y`.
526: 
527:     Parameters
528:     ----------
529:     k : array_like
530:         Number of successes (float).
531:     y : array_like
532:         Cumulative probability (probability of `k` or fewer successes in `n`
533:         events).
534:     p : array_like
535:         Success probability (float).
536: 
537:     Returns
538:     -------
539:     n : ndarray
540:         The number of events `n` such that `bdtr(k, n, p) = y`.
541: 
542:     See also
543:     --------
544:     bdtr
545: 
546:     Notes
547:     -----
548:     Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the
549:     cumulative incomplete beta distribution.
550: 
551:     Computation of `n` involves a search for a value that produces the desired
552:     value of `y`.  The search relies on the monotonicity of `y` with `n`.
553: 
554:     Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.
555: 
556:     References
557:     ----------
558:     .. [1] Milton Abramowitz and Irene A. Stegun, eds.
559:            Handbook of Mathematical Functions with Formulas,
560:            Graphs, and Mathematical Tables. New York: Dover, 1972.
561:     .. [2] Barry Brown, James Lovato, and Kathy Russell,
562:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
563:            Functions, Inverses, and Other Parameters.
564:     ''')
565: 
566: add_newdoc("scipy.special", "binom",
567:     '''
568:     binom(n, k)
569: 
570:     Binomial coefficient
571: 
572:     See Also
573:     --------
574:     comb : The number of combinations of N things taken k at a time.
575: 
576:     ''')
577: 
578: add_newdoc("scipy.special", "btdtria",
579:     r'''
580:     btdtria(p, b, x)
581: 
582:     Inverse of `btdtr` with respect to `a`.
583: 
584:     This is the inverse of the beta cumulative distribution function, `btdtr`,
585:     considered as a function of `a`, returning the value of `a` for which
586:     `btdtr(a, b, x) = p`, or
587: 
588:     .. math::
589:         p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt
590: 
591:     Parameters
592:     ----------
593:     p : array_like
594:         Cumulative probability, in [0, 1].
595:     b : array_like
596:         Shape parameter (`b` > 0).
597:     x : array_like
598:         The quantile, in [0, 1].
599: 
600:     Returns
601:     -------
602:     a : ndarray
603:         The value of the shape parameter `a` such that `btdtr(a, b, x) = p`.
604: 
605:     See Also
606:     --------
607:     btdtr : Cumulative density function of the beta distribution.
608:     btdtri : Inverse with respect to `x`.
609:     btdtrib : Inverse with respect to `b`.
610: 
611:     Notes
612:     -----
613:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.
614: 
615:     The cumulative distribution function `p` is computed using a routine by
616:     DiDinato and Morris [2]_.  Computation of `a` involves a search for a value
617:     that produces the desired value of `p`.  The search relies on the
618:     monotonicity of `p` with `a`.
619: 
620:     References
621:     ----------
622:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
623:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
624:            Functions, Inverses, and Other Parameters.
625:     .. [2] DiDinato, A. R. and Morris, A. H.,
626:            Algorithm 708: Significant Digit Computation of the Incomplete Beta
627:            Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.
628: 
629:     ''')
630: 
631: add_newdoc("scipy.special", "btdtrib",
632:     r'''
633:     btdtria(a, p, x)
634: 
635:     Inverse of `btdtr` with respect to `b`.
636: 
637:     This is the inverse of the beta cumulative distribution function, `btdtr`,
638:     considered as a function of `b`, returning the value of `b` for which
639:     `btdtr(a, b, x) = p`, or
640: 
641:     .. math::
642:         p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt
643: 
644:     Parameters
645:     ----------
646:     a : array_like
647:         Shape parameter (`a` > 0).
648:     p : array_like
649:         Cumulative probability, in [0, 1].
650:     x : array_like
651:         The quantile, in [0, 1].
652: 
653:     Returns
654:     -------
655:     b : ndarray
656:         The value of the shape parameter `b` such that `btdtr(a, b, x) = p`.
657: 
658:     See Also
659:     --------
660:     btdtr : Cumulative density function of the beta distribution.
661:     btdtri : Inverse with respect to `x`.
662:     btdtria : Inverse with respect to `a`.
663: 
664:     Notes
665:     -----
666:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.
667: 
668:     The cumulative distribution function `p` is computed using a routine by
669:     DiDinato and Morris [2]_.  Computation of `b` involves a search for a value
670:     that produces the desired value of `p`.  The search relies on the
671:     monotonicity of `p` with `b`.
672: 
673:     References
674:     ----------
675:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
676:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
677:            Functions, Inverses, and Other Parameters.
678:     .. [2] DiDinato, A. R. and Morris, A. H.,
679:            Algorithm 708: Significant Digit Computation of the Incomplete Beta
680:            Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.
681: 
682: 
683:     ''')
684: 
685: add_newdoc("scipy.special", "bei",
686:     '''
687:     bei(x)
688: 
689:     Kelvin function bei
690:     ''')
691: 
692: add_newdoc("scipy.special", "beip",
693:     '''
694:     beip(x)
695: 
696:     Derivative of the Kelvin function `bei`
697:     ''')
698: 
699: add_newdoc("scipy.special", "ber",
700:     '''
701:     ber(x)
702: 
703:     Kelvin function ber.
704:     ''')
705: 
706: add_newdoc("scipy.special", "berp",
707:     '''
708:     berp(x)
709: 
710:     Derivative of the Kelvin function `ber`
711:     ''')
712: 
713: add_newdoc("scipy.special", "besselpoly",
714:     r'''
715:     besselpoly(a, lmb, nu)
716: 
717:     Weighted integral of a Bessel function.
718: 
719:     .. math::
720: 
721:        \int_0^1 x^\lambda J_\nu(2 a x) \, dx
722: 
723:     where :math:`J_\nu` is a Bessel function and :math:`\lambda=lmb`,
724:     :math:`\nu=nu`.
725: 
726:     ''')
727: 
728: add_newdoc("scipy.special", "beta",
729:     '''
730:     beta(a, b)
731: 
732:     Beta function.
733: 
734:     ::
735: 
736:         beta(a, b) =  gamma(a) * gamma(b) / gamma(a+b)
737:     ''')
738: 
739: add_newdoc("scipy.special", "betainc",
740:     '''
741:     betainc(a, b, x)
742: 
743:     Incomplete beta integral.
744: 
745:     Compute the incomplete beta integral of the arguments, evaluated
746:     from zero to `x`::
747: 
748:         gamma(a+b) / (gamma(a)*gamma(b)) * integral(t**(a-1) (1-t)**(b-1), t=0..x).
749: 
750:     Notes
751:     -----
752:     The incomplete beta is also sometimes defined without the terms
753:     in gamma, in which case the above definition is the so-called regularized
754:     incomplete beta. Under this definition, you can get the incomplete beta by
755:     multiplying the result of the scipy function by beta(a, b).
756: 
757:     ''')
758: 
759: add_newdoc("scipy.special", "betaincinv",
760:     '''
761:     betaincinv(a, b, y)
762: 
763:     Inverse function to beta integral.
764: 
765:     Compute `x` such that betainc(a, b, x) = y.
766:     ''')
767: 
768: add_newdoc("scipy.special", "betaln",
769:     '''
770:     betaln(a, b)
771: 
772:     Natural logarithm of absolute value of beta function.
773: 
774:     Computes ``ln(abs(beta(a, b)))``.
775:     ''')
776: 
777: add_newdoc("scipy.special", "boxcox",
778:     '''
779:     boxcox(x, lmbda)
780: 
781:     Compute the Box-Cox transformation.
782: 
783:     The Box-Cox transformation is::
784: 
785:         y = (x**lmbda - 1) / lmbda  if lmbda != 0
786:             log(x)                  if lmbda == 0
787: 
788:     Returns `nan` if ``x < 0``.
789:     Returns `-inf` if ``x == 0`` and ``lmbda < 0``.
790: 
791:     Parameters
792:     ----------
793:     x : array_like
794:         Data to be transformed.
795:     lmbda : array_like
796:         Power parameter of the Box-Cox transform.
797: 
798:     Returns
799:     -------
800:     y : array
801:         Transformed data.
802: 
803:     Notes
804:     -----
805: 
806:     .. versionadded:: 0.14.0
807: 
808:     Examples
809:     --------
810:     >>> from scipy.special import boxcox
811:     >>> boxcox([1, 4, 10], 2.5)
812:     array([   0.        ,   12.4       ,  126.09110641])
813:     >>> boxcox(2, [0, 1, 2])
814:     array([ 0.69314718,  1.        ,  1.5       ])
815:     ''')
816: 
817: add_newdoc("scipy.special", "boxcox1p",
818:     '''
819:     boxcox1p(x, lmbda)
820: 
821:     Compute the Box-Cox transformation of 1 + `x`.
822: 
823:     The Box-Cox transformation computed by `boxcox1p` is::
824: 
825:         y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
826:             log(1+x)                    if lmbda == 0
827: 
828:     Returns `nan` if ``x < -1``.
829:     Returns `-inf` if ``x == -1`` and ``lmbda < 0``.
830: 
831:     Parameters
832:     ----------
833:     x : array_like
834:         Data to be transformed.
835:     lmbda : array_like
836:         Power parameter of the Box-Cox transform.
837: 
838:     Returns
839:     -------
840:     y : array
841:         Transformed data.
842: 
843:     Notes
844:     -----
845: 
846:     .. versionadded:: 0.14.0
847: 
848:     Examples
849:     --------
850:     >>> from scipy.special import boxcox1p
851:     >>> boxcox1p(1e-4, [0, 0.5, 1])
852:     array([  9.99950003e-05,   9.99975001e-05,   1.00000000e-04])
853:     >>> boxcox1p([0.01, 0.1], 0.25)
854:     array([ 0.00996272,  0.09645476])
855:     ''')
856: 
857: add_newdoc("scipy.special", "inv_boxcox",
858:     '''
859:     inv_boxcox(y, lmbda)
860: 
861:     Compute the inverse of the Box-Cox transformation.
862: 
863:     Find ``x`` such that::
864: 
865:         y = (x**lmbda - 1) / lmbda  if lmbda != 0
866:             log(x)                  if lmbda == 0
867: 
868:     Parameters
869:     ----------
870:     y : array_like
871:         Data to be transformed.
872:     lmbda : array_like
873:         Power parameter of the Box-Cox transform.
874: 
875:     Returns
876:     -------
877:     x : array
878:         Transformed data.
879: 
880:     Notes
881:     -----
882: 
883:     .. versionadded:: 0.16.0
884: 
885:     Examples
886:     --------
887:     >>> from scipy.special import boxcox, inv_boxcox
888:     >>> y = boxcox([1, 4, 10], 2.5)
889:     >>> inv_boxcox(y, 2.5)
890:     array([1., 4., 10.])
891:     ''')
892: 
893: add_newdoc("scipy.special", "inv_boxcox1p",
894:     '''
895:     inv_boxcox1p(y, lmbda)
896: 
897:     Compute the inverse of the Box-Cox transformation.
898: 
899:     Find ``x`` such that::
900: 
901:         y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
902:             log(1+x)                    if lmbda == 0
903: 
904:     Parameters
905:     ----------
906:     y : array_like
907:         Data to be transformed.
908:     lmbda : array_like
909:         Power parameter of the Box-Cox transform.
910: 
911:     Returns
912:     -------
913:     x : array
914:         Transformed data.
915: 
916:     Notes
917:     -----
918: 
919:     .. versionadded:: 0.16.0
920: 
921:     Examples
922:     --------
923:     >>> from scipy.special import boxcox1p, inv_boxcox1p
924:     >>> y = boxcox1p([1, 4, 10], 2.5)
925:     >>> inv_boxcox1p(y, 2.5)
926:     array([1., 4., 10.])
927:     ''')
928: 
929: add_newdoc("scipy.special", "btdtr",
930:     r'''
931:     btdtr(a, b, x)
932: 
933:     Cumulative density function of the beta distribution.
934: 
935:     Returns the integral from zero to `x` of the beta probability density
936:     function,
937: 
938:     .. math::
939:         I = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt
940: 
941:     where :math:`\Gamma` is the gamma function.
942: 
943:     Parameters
944:     ----------
945:     a : array_like
946:         Shape parameter (a > 0).
947:     b : array_like
948:         Shape parameter (b > 0).
949:     x : array_like
950:         Upper limit of integration, in [0, 1].
951: 
952:     Returns
953:     -------
954:     I : ndarray
955:         Cumulative density function of the beta distribution with parameters
956:         `a` and `b` at `x`.
957: 
958:     See Also
959:     --------
960:     betainc
961: 
962:     Notes
963:     -----
964:     This function is identical to the incomplete beta integral function
965:     `betainc`.
966: 
967:     Wrapper for the Cephes [1]_ routine `btdtr`.
968: 
969:     References
970:     ----------
971:     .. [1] Cephes Mathematical Functions Library,
972:            http://www.netlib.org/cephes/index.html
973: 
974:     ''')
975: 
976: add_newdoc("scipy.special", "btdtri",
977:     r'''
978:     btdtri(a, b, p)
979: 
980:     The `p`-th quantile of the beta distribution.
981: 
982:     This function is the inverse of the beta cumulative distribution function,
983:     `btdtr`, returning the value of `x` for which `btdtr(a, b, x) = p`, or
984: 
985:     .. math::
986:         p = \int_0^x \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} t^{a-1} (1-t)^{b-1}\,dt
987: 
988:     Parameters
989:     ----------
990:     a : array_like
991:         Shape parameter (`a` > 0).
992:     b : array_like
993:         Shape parameter (`b` > 0).
994:     p : array_like
995:         Cumulative probability, in [0, 1].
996: 
997:     Returns
998:     -------
999:     x : ndarray
1000:         The quantile corresponding to `p`.
1001: 
1002:     See Also
1003:     --------
1004:     betaincinv
1005:     btdtr
1006: 
1007:     Notes
1008:     -----
1009:     The value of `x` is found by interval halving or Newton iterations.
1010: 
1011:     Wrapper for the Cephes [1]_ routine `incbi`, which solves the equivalent
1012:     problem of finding the inverse of the incomplete beta integral.
1013: 
1014:     References
1015:     ----------
1016:     .. [1] Cephes Mathematical Functions Library,
1017:            http://www.netlib.org/cephes/index.html
1018: 
1019:     ''')
1020: 
1021: add_newdoc("scipy.special", "cbrt",
1022:     '''
1023:     cbrt(x)
1024: 
1025:     Element-wise cube root of `x`.
1026: 
1027:     Parameters
1028:     ----------
1029:     x : array_like
1030:         `x` must contain real numbers.
1031: 
1032:     Returns
1033:     -------
1034:     float
1035:         The cube root of each value in `x`.
1036: 
1037:     Examples
1038:     --------
1039:     >>> from scipy.special import cbrt
1040: 
1041:     >>> cbrt(8)
1042:     2.0
1043:     >>> cbrt([-8, -3, 0.125, 1.331])
1044:     array([-2.        , -1.44224957,  0.5       ,  1.1       ])
1045: 
1046:     ''')
1047: 
1048: add_newdoc("scipy.special", "chdtr",
1049:     '''
1050:     chdtr(v, x)
1051: 
1052:     Chi square cumulative distribution function
1053: 
1054:     Returns the area under the left hand tail (from 0 to `x`) of the Chi
1055:     square probability density function with `v` degrees of freedom::
1056: 
1057:         1/(2**(v/2) * gamma(v/2)) * integral(t**(v/2-1) * exp(-t/2), t=0..x)
1058:     ''')
1059: 
1060: add_newdoc("scipy.special", "chdtrc",
1061:     '''
1062:     chdtrc(v, x)
1063: 
1064:     Chi square survival function
1065: 
1066:     Returns the area under the right hand tail (from `x` to
1067:     infinity) of the Chi square probability density function with `v`
1068:     degrees of freedom::
1069: 
1070:         1/(2**(v/2) * gamma(v/2)) * integral(t**(v/2-1) * exp(-t/2), t=x..inf)
1071:     ''')
1072: 
1073: add_newdoc("scipy.special", "chdtri",
1074:     '''
1075:     chdtri(v, p)
1076: 
1077:     Inverse to `chdtrc`
1078: 
1079:     Returns the argument x such that ``chdtrc(v, x) == p``.
1080:     ''')
1081: 
1082: add_newdoc("scipy.special", "chdtriv",
1083:     '''
1084:     chdtriv(p, x)
1085: 
1086:     Inverse to `chdtr` vs `v`
1087: 
1088:     Returns the argument v such that ``chdtr(v, x) == p``.
1089:     ''')
1090: 
1091: add_newdoc("scipy.special", "chndtr",
1092:     '''
1093:     chndtr(x, df, nc)
1094: 
1095:     Non-central chi square cumulative distribution function
1096: 
1097:     ''')
1098: 
1099: add_newdoc("scipy.special", "chndtrix",
1100:     '''
1101:     chndtrix(p, df, nc)
1102: 
1103:     Inverse to `chndtr` vs `x`
1104:     ''')
1105: 
1106: add_newdoc("scipy.special", "chndtridf",
1107:     '''
1108:     chndtridf(x, p, nc)
1109: 
1110:     Inverse to `chndtr` vs `df`
1111:     ''')
1112: 
1113: add_newdoc("scipy.special", "chndtrinc",
1114:     '''
1115:     chndtrinc(x, df, p)
1116: 
1117:     Inverse to `chndtr` vs `nc`
1118:     ''')
1119: 
1120: add_newdoc("scipy.special", "cosdg",
1121:     '''
1122:     cosdg(x)
1123: 
1124:     Cosine of the angle `x` given in degrees.
1125:     ''')
1126: 
1127: add_newdoc("scipy.special", "cosm1",
1128:     '''
1129:     cosm1(x)
1130: 
1131:     cos(x) - 1 for use when `x` is near zero.
1132:     ''')
1133: 
1134: add_newdoc("scipy.special", "cotdg",
1135:     '''
1136:     cotdg(x)
1137: 
1138:     Cotangent of the angle `x` given in degrees.
1139:     ''')
1140: 
1141: add_newdoc("scipy.special", "dawsn",
1142:     '''
1143:     dawsn(x)
1144: 
1145:     Dawson's integral.
1146: 
1147:     Computes::
1148: 
1149:         exp(-x**2) * integral(exp(t**2), t=0..x).
1150: 
1151:     See Also
1152:     --------
1153:     wofz, erf, erfc, erfcx, erfi
1154: 
1155:     References
1156:     ----------
1157:     .. [1] Steven G. Johnson, Faddeeva W function implementation.
1158:        http://ab-initio.mit.edu/Faddeeva
1159: 
1160:     Examples
1161:     --------
1162:     >>> from scipy import special
1163:     >>> import matplotlib.pyplot as plt
1164:     >>> x = np.linspace(-15, 15, num=1000)
1165:     >>> plt.plot(x, special.dawsn(x))
1166:     >>> plt.xlabel('$x$')
1167:     >>> plt.ylabel('$dawsn(x)$')
1168:     >>> plt.show()
1169: 
1170:     ''')
1171: 
1172: add_newdoc("scipy.special", "ellipe",
1173:     r'''
1174:     ellipe(m)
1175: 
1176:     Complete elliptic integral of the second kind
1177: 
1178:     This function is defined as
1179: 
1180:     .. math:: E(m) = \int_0^{\pi/2} [1 - m \sin(t)^2]^{1/2} dt
1181: 
1182:     Parameters
1183:     ----------
1184:     m : array_like
1185:         Defines the parameter of the elliptic integral.
1186: 
1187:     Returns
1188:     -------
1189:     E : ndarray
1190:         Value of the elliptic integral.
1191: 
1192:     Notes
1193:     -----
1194:     Wrapper for the Cephes [1]_ routine `ellpe`.
1195: 
1196:     For `m > 0` the computation uses the approximation,
1197: 
1198:     .. math:: E(m) \approx P(1-m) - (1-m) \log(1-m) Q(1-m),
1199: 
1200:     where :math:`P` and :math:`Q` are tenth-order polynomials.  For
1201:     `m < 0`, the relation
1202: 
1203:     .. math:: E(m) = E(m/(m - 1)) \sqrt(1-m)
1204: 
1205:     is used.
1206: 
1207:     The parameterization in terms of :math:`m` follows that of section
1208:     17.2 in [2]_. Other parameterizations in terms of the
1209:     complementary parameter :math:`1 - m`, modular angle
1210:     :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
1211:     used, so be careful that you choose the correct parameter.
1212: 
1213:     See Also
1214:     --------
1215:     ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
1216:     ellipk : Complete elliptic integral of the first kind
1217:     ellipkinc : Incomplete elliptic integral of the first kind
1218:     ellipeinc : Incomplete elliptic integral of the second kind
1219: 
1220:     References
1221:     ----------
1222:     .. [1] Cephes Mathematical Functions Library,
1223:            http://www.netlib.org/cephes/index.html
1224:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
1225:            Handbook of Mathematical Functions with Formulas,
1226:            Graphs, and Mathematical Tables. New York: Dover, 1972.
1227:     ''')
1228: 
1229: add_newdoc("scipy.special", "ellipeinc",
1230:     r'''
1231:     ellipeinc(phi, m)
1232: 
1233:     Incomplete elliptic integral of the second kind
1234: 
1235:     This function is defined as
1236: 
1237:     .. math:: E(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{1/2} dt
1238: 
1239:     Parameters
1240:     ----------
1241:     phi : array_like
1242:         amplitude of the elliptic integral.
1243: 
1244:     m : array_like
1245:         parameter of the elliptic integral.
1246: 
1247:     Returns
1248:     -------
1249:     E : ndarray
1250:         Value of the elliptic integral.
1251: 
1252:     Notes
1253:     -----
1254:     Wrapper for the Cephes [1]_ routine `ellie`.
1255: 
1256:     Computation uses arithmetic-geometric means algorithm.
1257: 
1258:     The parameterization in terms of :math:`m` follows that of section
1259:     17.2 in [2]_. Other parameterizations in terms of the
1260:     complementary parameter :math:`1 - m`, modular angle
1261:     :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
1262:     used, so be careful that you choose the correct parameter.
1263: 
1264:     See Also
1265:     --------
1266:     ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
1267:     ellipk : Complete elliptic integral of the first kind
1268:     ellipkinc : Incomplete elliptic integral of the first kind
1269:     ellipe : Complete elliptic integral of the second kind
1270: 
1271:     References
1272:     ----------
1273:     .. [1] Cephes Mathematical Functions Library,
1274:            http://www.netlib.org/cephes/index.html
1275:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
1276:            Handbook of Mathematical Functions with Formulas,
1277:            Graphs, and Mathematical Tables. New York: Dover, 1972.
1278:     ''')
1279: 
1280: add_newdoc("scipy.special", "ellipj",
1281:     '''
1282:     ellipj(u, m)
1283: 
1284:     Jacobian elliptic functions
1285: 
1286:     Calculates the Jacobian elliptic functions of parameter `m` between
1287:     0 and 1, and real argument `u`.
1288: 
1289:     Parameters
1290:     ----------
1291:     m : array_like
1292:         Parameter.
1293:     u : array_like
1294:         Argument.
1295: 
1296:     Returns
1297:     -------
1298:     sn, cn, dn, ph : ndarrays
1299:         The returned functions::
1300: 
1301:             sn(u|m), cn(u|m), dn(u|m)
1302: 
1303:         The value `ph` is such that if `u = ellipk(ph, m)`,
1304:         then `sn(u|m) = sin(ph)` and `cn(u|m) = cos(ph)`.
1305: 
1306:     Notes
1307:     -----
1308:     Wrapper for the Cephes [1]_ routine `ellpj`.
1309: 
1310:     These functions are periodic, with quarter-period on the real axis
1311:     equal to the complete elliptic integral `ellipk(m)`.
1312: 
1313:     Relation to incomplete elliptic integral: If `u = ellipk(phi,m)`, then
1314:     `sn(u|m) = sin(phi)`, and `cn(u|m) = cos(phi)`.  The `phi` is called
1315:     the amplitude of `u`.
1316: 
1317:     Computation is by means of the arithmetic-geometric mean algorithm,
1318:     except when `m` is within 1e-9 of 0 or 1.  In the latter case with `m`
1319:     close to 1, the approximation applies only for `phi < pi/2`.
1320: 
1321:     See also
1322:     --------
1323:     ellipk : Complete elliptic integral of the first kind.
1324: 
1325:     References
1326:     ----------
1327:     .. [1] Cephes Mathematical Functions Library,
1328:            http://www.netlib.org/cephes/index.html
1329:     ''')
1330: 
1331: add_newdoc("scipy.special", "ellipkm1",
1332:     '''
1333:     ellipkm1(p)
1334: 
1335:     Complete elliptic integral of the first kind around `m` = 1
1336: 
1337:     This function is defined as
1338: 
1339:     .. math:: K(p) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt
1340: 
1341:     where `m = 1 - p`.
1342: 
1343:     Parameters
1344:     ----------
1345:     p : array_like
1346:         Defines the parameter of the elliptic integral as `m = 1 - p`.
1347: 
1348:     Returns
1349:     -------
1350:     K : ndarray
1351:         Value of the elliptic integral.
1352: 
1353:     Notes
1354:     -----
1355:     Wrapper for the Cephes [1]_ routine `ellpk`.
1356: 
1357:     For `p <= 1`, computation uses the approximation,
1358: 
1359:     .. math:: K(p) \\approx P(p) - \\log(p) Q(p),
1360: 
1361:     where :math:`P` and :math:`Q` are tenth-order polynomials.  The
1362:     argument `p` is used internally rather than `m` so that the logarithmic
1363:     singularity at `m = 1` will be shifted to the origin; this preserves
1364:     maximum accuracy.  For `p > 1`, the identity
1365: 
1366:     .. math:: K(p) = K(1/p)/\\sqrt(p)
1367: 
1368:     is used.
1369: 
1370:     See Also
1371:     --------
1372:     ellipk : Complete elliptic integral of the first kind
1373:     ellipkinc : Incomplete elliptic integral of the first kind
1374:     ellipe : Complete elliptic integral of the second kind
1375:     ellipeinc : Incomplete elliptic integral of the second kind
1376: 
1377:     References
1378:     ----------
1379:     .. [1] Cephes Mathematical Functions Library,
1380:            http://www.netlib.org/cephes/index.html
1381:     ''')
1382: 
1383: add_newdoc("scipy.special", "ellipkinc",
1384:     r'''
1385:     ellipkinc(phi, m)
1386: 
1387:     Incomplete elliptic integral of the first kind
1388: 
1389:     This function is defined as
1390: 
1391:     .. math:: K(\phi, m) = \int_0^{\phi} [1 - m \sin(t)^2]^{-1/2} dt
1392: 
1393:     This function is also called `F(phi, m)`.
1394: 
1395:     Parameters
1396:     ----------
1397:     phi : array_like
1398:         amplitude of the elliptic integral
1399: 
1400:     m : array_like
1401:         parameter of the elliptic integral
1402: 
1403:     Returns
1404:     -------
1405:     K : ndarray
1406:         Value of the elliptic integral
1407: 
1408:     Notes
1409:     -----
1410:     Wrapper for the Cephes [1]_ routine `ellik`.  The computation is
1411:     carried out using the arithmetic-geometric mean algorithm.
1412: 
1413:     The parameterization in terms of :math:`m` follows that of section
1414:     17.2 in [2]_. Other parameterizations in terms of the
1415:     complementary parameter :math:`1 - m`, modular angle
1416:     :math:`\sin^2(\alpha) = m`, or modulus :math:`k^2 = m` are also
1417:     used, so be careful that you choose the correct parameter.
1418: 
1419:     See Also
1420:     --------
1421:     ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1
1422:     ellipk : Complete elliptic integral of the first kind
1423:     ellipe : Complete elliptic integral of the second kind
1424:     ellipeinc : Incomplete elliptic integral of the second kind
1425: 
1426:     References
1427:     ----------
1428:     .. [1] Cephes Mathematical Functions Library,
1429:            http://www.netlib.org/cephes/index.html
1430:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
1431:            Handbook of Mathematical Functions with Formulas,
1432:            Graphs, and Mathematical Tables. New York: Dover, 1972.
1433:     ''')
1434: 
1435: add_newdoc("scipy.special", "entr",
1436:     r'''
1437:     entr(x)
1438: 
1439:     Elementwise function for computing entropy.
1440: 
1441:     .. math:: \text{entr}(x) = \begin{cases} - x \log(x) & x > 0  \\ 0 & x = 0 \\ -\infty & \text{otherwise} \end{cases}
1442: 
1443:     Parameters
1444:     ----------
1445:     x : ndarray
1446:         Input array.
1447: 
1448:     Returns
1449:     -------
1450:     res : ndarray
1451:         The value of the elementwise entropy function at the given points `x`.
1452: 
1453:     See Also
1454:     --------
1455:     kl_div, rel_entr
1456: 
1457:     Notes
1458:     -----
1459:     This function is concave.
1460: 
1461:     .. versionadded:: 0.15.0
1462: 
1463:     ''')
1464: 
1465: add_newdoc("scipy.special", "erf",
1466:     '''
1467:     erf(z)
1468: 
1469:     Returns the error function of complex argument.
1470: 
1471:     It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.
1472: 
1473:     Parameters
1474:     ----------
1475:     x : ndarray
1476:         Input array.
1477: 
1478:     Returns
1479:     -------
1480:     res : ndarray
1481:         The values of the error function at the given points `x`.
1482: 
1483:     See Also
1484:     --------
1485:     erfc, erfinv, erfcinv, wofz, erfcx, erfi
1486: 
1487:     Notes
1488:     -----
1489:     The cumulative of the unit normal distribution is given by
1490:     ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.
1491: 
1492:     References
1493:     ----------
1494:     .. [1] http://en.wikipedia.org/wiki/Error_function
1495:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
1496:         Handbook of Mathematical Functions with Formulas,
1497:         Graphs, and Mathematical Tables. New York: Dover,
1498:         1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm
1499:     .. [3] Steven G. Johnson, Faddeeva W function implementation.
1500:        http://ab-initio.mit.edu/Faddeeva
1501: 
1502:     Examples
1503:     --------
1504:     >>> from scipy import special
1505:     >>> import matplotlib.pyplot as plt
1506:     >>> x = np.linspace(-3, 3)
1507:     >>> plt.plot(x, special.erf(x))
1508:     >>> plt.xlabel('$x$')
1509:     >>> plt.ylabel('$erf(x)$')
1510:     >>> plt.show()
1511: 
1512:     ''')
1513: 
1514: add_newdoc("scipy.special", "erfc",
1515:     '''
1516:     erfc(x)
1517: 
1518:     Complementary error function, ``1 - erf(x)``.
1519: 
1520:     See Also
1521:     --------
1522:     erf, erfi, erfcx, dawsn, wofz
1523: 
1524:     References
1525:     ----------
1526:     .. [1] Steven G. Johnson, Faddeeva W function implementation.
1527:        http://ab-initio.mit.edu/Faddeeva
1528: 
1529:     Examples
1530:     --------
1531:     >>> from scipy import special
1532:     >>> import matplotlib.pyplot as plt
1533:     >>> x = np.linspace(-3, 3)
1534:     >>> plt.plot(x, special.erfc(x))
1535:     >>> plt.xlabel('$x$')
1536:     >>> plt.ylabel('$erfc(x)$')
1537:     >>> plt.show()
1538: 
1539:     ''')
1540: 
1541: add_newdoc("scipy.special", "erfi",
1542:     '''
1543:     erfi(z)
1544: 
1545:     Imaginary error function, ``-i erf(i z)``.
1546: 
1547:     See Also
1548:     --------
1549:     erf, erfc, erfcx, dawsn, wofz
1550: 
1551:     Notes
1552:     -----
1553: 
1554:     .. versionadded:: 0.12.0
1555: 
1556:     References
1557:     ----------
1558:     .. [1] Steven G. Johnson, Faddeeva W function implementation.
1559:        http://ab-initio.mit.edu/Faddeeva
1560: 
1561:     Examples
1562:     --------
1563:     >>> from scipy import special
1564:     >>> import matplotlib.pyplot as plt
1565:     >>> x = np.linspace(-3, 3)
1566:     >>> plt.plot(x, special.erfi(x))
1567:     >>> plt.xlabel('$x$')
1568:     >>> plt.ylabel('$erfi(x)$')
1569:     >>> plt.show()
1570: 
1571:     ''')
1572: 
1573: add_newdoc("scipy.special", "erfcx",
1574:     '''
1575:     erfcx(x)
1576: 
1577:     Scaled complementary error function, ``exp(x**2) * erfc(x)``.
1578: 
1579:     See Also
1580:     --------
1581:     erf, erfc, erfi, dawsn, wofz
1582: 
1583:     Notes
1584:     -----
1585: 
1586:     .. versionadded:: 0.12.0
1587: 
1588:     References
1589:     ----------
1590:     .. [1] Steven G. Johnson, Faddeeva W function implementation.
1591:        http://ab-initio.mit.edu/Faddeeva
1592: 
1593:     Examples
1594:     --------
1595:     >>> from scipy import special
1596:     >>> import matplotlib.pyplot as plt
1597:     >>> x = np.linspace(-3, 3)
1598:     >>> plt.plot(x, special.erfcx(x))
1599:     >>> plt.xlabel('$x$')
1600:     >>> plt.ylabel('$erfcx(x)$')
1601:     >>> plt.show()
1602: 
1603:     ''')
1604: 
1605: add_newdoc("scipy.special", "eval_jacobi",
1606:     r'''
1607:     eval_jacobi(n, alpha, beta, x, out=None)
1608: 
1609:     Evaluate Jacobi polynomial at a point.
1610: 
1611:     The Jacobi polynomials can be defined via the Gauss hypergeometric
1612:     function :math:`{}_2F_1` as
1613: 
1614:     .. math::
1615: 
1616:         P_n^{(\alpha, \beta)}(x) = \frac{(\alpha + 1)_n}{\Gamma(n + 1)}
1617:           {}_2F_1(-n, 1 + \alpha + \beta + n; \alpha + 1; (1 - z)/2)
1618: 
1619:     where :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
1620:     :math:`n` is an integer the result is a polynomial of degree
1621:     :math:`n`.
1622: 
1623:     Parameters
1624:     ----------
1625:     n : array_like
1626:         Degree of the polynomial. If not an integer the result is
1627:         determined via the relation to the Gauss hypergeometric
1628:         function.
1629:     alpha : array_like
1630:         Parameter
1631:     beta : array_like
1632:         Parameter
1633:     x : array_like
1634:         Points at which to evaluate the polynomial
1635: 
1636:     Returns
1637:     -------
1638:     P : ndarray
1639:         Values of the Jacobi polynomial
1640: 
1641:     See Also
1642:     --------
1643:     roots_jacobi : roots and quadrature weights of Jacobi polynomials
1644:     jacobi : Jacobi polynomial object
1645:     hyp2f1 : Gauss hypergeometric function
1646:     ''')
1647: 
1648: add_newdoc("scipy.special", "eval_sh_jacobi",
1649:     r'''
1650:     eval_sh_jacobi(n, p, q, x, out=None)
1651: 
1652:     Evaluate shifted Jacobi polynomial at a point.
1653: 
1654:     Defined by
1655: 
1656:     .. math::
1657: 
1658:         G_n^{(p, q)}(x)
1659:           = \binom{2n + p - 1}{n}^{-1} P_n^{(p - q, q - 1)}(2x - 1),
1660: 
1661:     where :math:`P_n^{(\cdot, \cdot)}` is the n-th Jacobi polynomial.
1662: 
1663:     Parameters
1664:     ----------
1665:     n : int
1666:         Degree of the polynomial. If not an integer, the result is
1667:         determined via the relation to `binom` and `eval_jacobi`.
1668:     p : float
1669:         Parameter
1670:     q : float
1671:         Parameter
1672: 
1673:     Returns
1674:     -------
1675:     G : ndarray
1676:         Values of the shifted Jacobi polynomial.
1677: 
1678:     See Also
1679:     --------
1680:     roots_sh_jacobi : roots and quadrature weights of shifted Jacobi
1681:                       polynomials
1682:     sh_jacobi : shifted Jacobi polynomial object
1683:     eval_jacobi : evaluate Jacobi polynomials
1684:     ''')
1685: 
1686: add_newdoc("scipy.special", "eval_gegenbauer",
1687:     r'''
1688:     eval_gegenbauer(n, alpha, x, out=None)
1689: 
1690:     Evaluate Gegenbauer polynomial at a point.
1691: 
1692:     The Gegenbauer polynomials can be defined via the Gauss
1693:     hypergeometric function :math:`{}_2F_1` as
1694: 
1695:     .. math::
1696: 
1697:         C_n^{(\alpha)} = \frac{(2\alpha)_n}{\Gamma(n + 1)}
1698:           {}_2F_1(-n, 2\alpha + n; \alpha + 1/2; (1 - z)/2).
1699: 
1700:     When :math:`n` is an integer the result is a polynomial of degree
1701:     :math:`n`.
1702: 
1703:     Parameters
1704:     ----------
1705:     n : array_like
1706:         Degree of the polynomial. If not an integer, the result is
1707:         determined via the relation to the Gauss hypergeometric
1708:         function.
1709:     alpha : array_like
1710:         Parameter
1711:     x : array_like
1712:         Points at which to evaluate the Gegenbauer polynomial
1713: 
1714:     Returns
1715:     -------
1716:     C : ndarray
1717:         Values of the Gegenbauer polynomial
1718: 
1719:     See Also
1720:     --------
1721:     roots_gegenbauer : roots and quadrature weights of Gegenbauer
1722:                        polynomials
1723:     gegenbauer : Gegenbauer polynomial object
1724:     hyp2f1 : Gauss hypergeometric function
1725:     ''')
1726: 
1727: add_newdoc("scipy.special", "eval_chebyt",
1728:     r'''
1729:     eval_chebyt(n, x, out=None)
1730: 
1731:     Evaluate Chebyshev polynomial of the first kind at a point.
1732: 
1733:     The Chebyshev polynomials of the first kind can be defined via the
1734:     Gauss hypergeometric function :math:`{}_2F_1` as
1735: 
1736:     .. math::
1737: 
1738:         T_n(x) = {}_2F_1(n, -n; 1/2; (1 - x)/2).
1739: 
1740:     When :math:`n` is an integer the result is a polynomial of degree
1741:     :math:`n`.
1742: 
1743:     Parameters
1744:     ----------
1745:     n : array_like
1746:         Degree of the polynomial. If not an integer, the result is
1747:         determined via the relation to the Gauss hypergeometric
1748:         function.
1749:     x : array_like
1750:         Points at which to evaluate the Chebyshev polynomial
1751: 
1752:     Returns
1753:     -------
1754:     T : ndarray
1755:         Values of the Chebyshev polynomial
1756: 
1757:     See Also
1758:     --------
1759:     roots_chebyt : roots and quadrature weights of Chebyshev
1760:                    polynomials of the first kind
1761:     chebyu : Chebychev polynomial object
1762:     eval_chebyu : evaluate Chebyshev polynomials of the second kind
1763:     hyp2f1 : Gauss hypergeometric function
1764:     numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
1765: 
1766:     Notes
1767:     -----
1768:     This routine is numerically stable for `x` in ``[-1, 1]`` at least
1769:     up to order ``10000``.
1770:     ''')
1771: 
1772: add_newdoc("scipy.special", "eval_chebyu",
1773:     r'''
1774:     eval_chebyu(n, x, out=None)
1775: 
1776:     Evaluate Chebyshev polynomial of the second kind at a point.
1777: 
1778:     The Chebyshev polynomials of the second kind can be defined via
1779:     the Gauss hypergeometric function :math:`{}_2F_1` as
1780: 
1781:     .. math::
1782: 
1783:         U_n(x) = (n + 1) {}_2F_1(-n, n + 2; 3/2; (1 - x)/2).
1784: 
1785:     When :math:`n` is an integer the result is a polynomial of degree
1786:     :math:`n`.
1787: 
1788:     Parameters
1789:     ----------
1790:     n : array_like
1791:         Degree of the polynomial. If not an integer, the result is
1792:         determined via the relation to the Gauss hypergeometric
1793:         function.
1794:     x : array_like
1795:         Points at which to evaluate the Chebyshev polynomial
1796: 
1797:     Returns
1798:     -------
1799:     U : ndarray
1800:         Values of the Chebyshev polynomial
1801: 
1802:     See Also
1803:     --------
1804:     roots_chebyu : roots and quadrature weights of Chebyshev
1805:                    polynomials of the second kind
1806:     chebyu : Chebyshev polynomial object
1807:     eval_chebyt : evaluate Chebyshev polynomials of the first kind
1808:     hyp2f1 : Gauss hypergeometric function
1809:     ''')
1810: 
1811: add_newdoc("scipy.special", "eval_chebys",
1812:     r'''
1813:     eval_chebys(n, x, out=None)
1814: 
1815:     Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a
1816:     point.
1817: 
1818:     These polynomials are defined as
1819: 
1820:     .. math::
1821: 
1822:         S_n(x) = U_n(x/2)
1823: 
1824:     where :math:`U_n` is a Chebyshev polynomial of the second kind.
1825: 
1826:     Parameters
1827:     ----------
1828:     n : array_like
1829:         Degree of the polynomial. If not an integer, the result is
1830:         determined via the relation to `eval_chebyu`.
1831:     x : array_like
1832:         Points at which to evaluate the Chebyshev polynomial
1833: 
1834:     Returns
1835:     -------
1836:     S : ndarray
1837:         Values of the Chebyshev polynomial
1838: 
1839:     See Also
1840:     --------
1841:     roots_chebys : roots and quadrature weights of Chebyshev
1842:                    polynomials of the second kind on [-2, 2]
1843:     chebys : Chebyshev polynomial object
1844:     eval_chebyu : evaluate Chebyshev polynomials of the second kind
1845:     ''')
1846: 
1847: add_newdoc("scipy.special", "eval_chebyc",
1848:     r'''
1849:     eval_chebyc(n, x, out=None)
1850: 
1851:     Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a
1852:     point.
1853: 
1854:     These polynomials are defined as
1855: 
1856:     .. math::
1857: 
1858:         S_n(x) = T_n(x/2)
1859: 
1860:     where :math:`T_n` is a Chebyshev polynomial of the first kind.
1861: 
1862:     Parameters
1863:     ----------
1864:     n : array_like
1865:         Degree of the polynomial. If not an integer, the result is
1866:         determined via the relation to `eval_chebyt`.
1867:     x : array_like
1868:         Points at which to evaluate the Chebyshev polynomial
1869: 
1870:     Returns
1871:     -------
1872:     C : ndarray
1873:         Values of the Chebyshev polynomial
1874: 
1875:     See Also
1876:     --------
1877:     roots_chebyc : roots and quadrature weights of Chebyshev
1878:                    polynomials of the first kind on [-2, 2]
1879:     chebyc : Chebyshev polynomial object
1880:     numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
1881:     eval_chebyt : evaluate Chebycshev polynomials of the first kind
1882:     ''')
1883: 
1884: add_newdoc("scipy.special", "eval_sh_chebyt",
1885:     r'''
1886:     eval_sh_chebyt(n, x, out=None)
1887: 
1888:     Evaluate shifted Chebyshev polynomial of the first kind at a
1889:     point.
1890: 
1891:     These polynomials are defined as
1892: 
1893:     .. math::
1894: 
1895:         T_n^*(x) = T_n(2x - 1)
1896: 
1897:     where :math:`T_n` is a Chebyshev polynomial of the first kind.
1898: 
1899:     Parameters
1900:     ----------
1901:     n : array_like
1902:         Degree of the polynomial. If not an integer, the result is
1903:         determined via the relation to `eval_chebyt`.
1904:     x : array_like
1905:         Points at which to evaluate the shifted Chebyshev polynomial
1906: 
1907:     Returns
1908:     -------
1909:     T : ndarray
1910:         Values of the shifted Chebyshev polynomial
1911: 
1912:     See Also
1913:     --------
1914:     roots_sh_chebyt : roots and quadrature weights of shifted
1915:                       Chebyshev polynomials of the first kind
1916:     sh_chebyt : shifted Chebyshev polynomial object
1917:     eval_chebyt : evaluate Chebyshev polynomials of the first kind
1918:     numpy.polynomial.chebyshev.Chebyshev : Chebyshev series
1919:     ''')
1920: 
1921: add_newdoc("scipy.special", "eval_sh_chebyu",
1922:     r'''
1923:     eval_sh_chebyu(n, x, out=None)
1924: 
1925:     Evaluate shifted Chebyshev polynomial of the second kind at a
1926:     point.
1927: 
1928:     These polynomials are defined as
1929: 
1930:     .. math::
1931: 
1932:         U_n^*(x) = U_n(2x - 1)
1933: 
1934:     where :math:`U_n` is a Chebyshev polynomial of the first kind.
1935: 
1936:     Parameters
1937:     ----------
1938:     n : array_like
1939:         Degree of the polynomial. If not an integer, the result is
1940:         determined via the relation to `eval_chebyu`.
1941:     x : array_like
1942:         Points at which to evaluate the shifted Chebyshev polynomial
1943: 
1944:     Returns
1945:     -------
1946:     U : ndarray
1947:         Values of the shifted Chebyshev polynomial
1948: 
1949:     See Also
1950:     --------
1951:     roots_sh_chebyu : roots and quadrature weights of shifted
1952:                       Chebychev polynomials of the second kind
1953:     sh_chebyu : shifted Chebyshev polynomial object
1954:     eval_chebyu : evaluate Chebyshev polynomials of the second kind
1955:     ''')
1956: 
1957: add_newdoc("scipy.special", "eval_legendre",
1958:     r'''
1959:     eval_legendre(n, x, out=None)
1960: 
1961:     Evaluate Legendre polynomial at a point.
1962: 
1963:     The Legendre polynomials can be defined via the Gauss
1964:     hypergeometric function :math:`{}_2F_1` as
1965: 
1966:     .. math::
1967: 
1968:         P_n(x) = {}_2F_1(-n, n + 1; 1; (1 - x)/2).
1969: 
1970:     When :math:`n` is an integer the result is a polynomial of degree
1971:     :math:`n`.
1972: 
1973:     Parameters
1974:     ----------
1975:     n : array_like
1976:         Degree of the polynomial. If not an integer, the result is
1977:         determined via the relation to the Gauss hypergeometric
1978:         function.
1979:     x : array_like
1980:         Points at which to evaluate the Legendre polynomial
1981: 
1982:     Returns
1983:     -------
1984:     P : ndarray
1985:         Values of the Legendre polynomial
1986: 
1987:     See Also
1988:     --------
1989:     roots_legendre : roots and quadrature weights of Legendre
1990:                      polynomials
1991:     legendre : Legendre polynomial object
1992:     hyp2f1 : Gauss hypergeometric function
1993:     numpy.polynomial.legendre.Legendre : Legendre series
1994:     ''')
1995: 
1996: add_newdoc("scipy.special", "eval_sh_legendre",
1997:     r'''
1998:     eval_sh_legendre(n, x, out=None)
1999: 
2000:     Evaluate shifted Legendre polynomial at a point.
2001: 
2002:     These polynomials are defined as
2003: 
2004:     .. math::
2005: 
2006:         P_n^*(x) = P_n(2x - 1)
2007: 
2008:     where :math:`P_n` is a Legendre polynomial.
2009: 
2010:     Parameters
2011:     ----------
2012:     n : array_like
2013:         Degree of the polynomial. If not an integer, the value is
2014:         determined via the relation to `eval_legendre`.
2015:     x : array_like
2016:         Points at which to evaluate the shifted Legendre polynomial
2017: 
2018:     Returns
2019:     -------
2020:     P : ndarray
2021:         Values of the shifted Legendre polynomial
2022: 
2023:     See Also
2024:     --------
2025:     roots_sh_legendre : roots and quadrature weights of shifted
2026:                         Legendre polynomials
2027:     sh_legendre : shifted Legendre polynomial object
2028:     eval_legendre : evaluate Legendre polynomials
2029:     numpy.polynomial.legendre.Legendre : Legendre series
2030:     ''')
2031: 
2032: add_newdoc("scipy.special", "eval_genlaguerre",
2033:     r'''
2034:     eval_genlaguerre(n, alpha, x, out=None)
2035: 
2036:     Evaluate generalized Laguerre polynomial at a point.
2037: 
2038:     The generalized Laguerre polynomials can be defined via the
2039:     confluent hypergeometric function :math:`{}_1F_1` as
2040: 
2041:     .. math::
2042: 
2043:         L_n^{(\alpha)}(x) = \binom{n + \alpha}{n}
2044:           {}_1F_1(-n, \alpha + 1, x).
2045: 
2046:     When :math:`n` is an integer the result is a polynomial of degree
2047:     :math:`n`. The Laguerre polynomials are the special case where
2048:     :math:`\alpha = 0`.
2049: 
2050:     Parameters
2051:     ----------
2052:     n : array_like
2053:         Degree of the polynomial. If not an integer the result is
2054:         determined via the relation to the confluent hypergeometric
2055:         function.
2056:     alpha : array_like
2057:         Parameter; must have ``alpha > -1``
2058:     x : array_like
2059:         Points at which to evaluate the generalized Laguerre
2060:         polynomial
2061: 
2062:     Returns
2063:     -------
2064:     L : ndarray
2065:         Values of the generalized Laguerre polynomial
2066: 
2067:     See Also
2068:     --------
2069:     roots_genlaguerre : roots and quadrature weights of generalized
2070:                         Laguerre polynomials
2071:     genlaguerre : generalized Laguerre polynomial object
2072:     hyp1f1 : confluent hypergeometric function
2073:     eval_laguerre : evaluate Laguerre polynomials
2074:     ''')
2075: 
2076: add_newdoc("scipy.special", "eval_laguerre",
2077:      r'''
2078:      eval_laguerre(n, x, out=None)
2079: 
2080:      Evaluate Laguerre polynomial at a point.
2081: 
2082:      The Laguerre polynomials can be defined via the confluent
2083:      hypergeometric function :math:`{}_1F_1` as
2084: 
2085:      .. math::
2086: 
2087:          L_n(x) = {}_1F_1(-n, 1, x).
2088: 
2089:      When :math:`n` is an integer the result is a polynomial of degree
2090:      :math:`n`.
2091: 
2092:      Parameters
2093:      ----------
2094:      n : array_like
2095:          Degree of the polynomial. If not an integer the result is
2096:          determined via the relation to the confluent hypergeometric
2097:          function.
2098:      x : array_like
2099:          Points at which to evaluate the Laguerre polynomial
2100: 
2101:      Returns
2102:      -------
2103:      L : ndarray
2104:          Values of the Laguerre polynomial
2105: 
2106:      See Also
2107:      --------
2108:      roots_laguerre : roots and quadrature weights of Laguerre
2109:                       polynomials
2110:      laguerre : Laguerre polynomial object
2111:      numpy.polynomial.laguerre.Laguerre : Laguerre series
2112:      eval_genlaguerre : evaluate generalized Laguerre polynomials
2113:      ''')
2114: 
2115: add_newdoc("scipy.special", "eval_hermite",
2116:     r'''
2117:     eval_hermite(n, x, out=None)
2118: 
2119:     Evaluate physicist's Hermite polynomial at a point.
2120: 
2121:     Defined by
2122: 
2123:     .. math::
2124: 
2125:         H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2};
2126: 
2127:     :math:`H_n` is a polynomial of degree :math:`n`.
2128: 
2129:     Parameters
2130:     ----------
2131:     n : array_like
2132:         Degree of the polynomial
2133:     x : array_like
2134:         Points at which to evaluate the Hermite polynomial
2135: 
2136:     Returns
2137:     -------
2138:     H : ndarray
2139:         Values of the Hermite polynomial
2140: 
2141:     See Also
2142:     --------
2143:     roots_hermite : roots and quadrature weights of physicist's
2144:                     Hermite polynomials
2145:     hermite : physicist's Hermite polynomial object
2146:     numpy.polynomial.hermite.Hermite : Physicist's Hermite series
2147:     eval_hermitenorm : evaluate Probabilist's Hermite polynomials
2148:     ''')
2149: 
2150: add_newdoc("scipy.special", "eval_hermitenorm",
2151:     r'''
2152:     eval_hermitenorm(n, x, out=None)
2153: 
2154:     Evaluate probabilist's (normalized) Hermite polynomial at a
2155:     point.
2156: 
2157:     Defined by
2158: 
2159:     .. math::
2160: 
2161:         He_n(x) = (-1)^n e^{x^2/2} \frac{d^n}{dx^n} e^{-x^2/2};
2162: 
2163:     :math:`He_n` is a polynomial of degree :math:`n`.
2164: 
2165:     Parameters
2166:     ----------
2167:     n : array_like
2168:         Degree of the polynomial
2169:     x : array_like
2170:         Points at which to evaluate the Hermite polynomial
2171: 
2172:     Returns
2173:     -------
2174:     He : ndarray
2175:         Values of the Hermite polynomial
2176: 
2177:     See Also
2178:     --------
2179:     roots_hermitenorm : roots and quadrature weights of probabilist's
2180:                         Hermite polynomials
2181:     hermitenorm : probabilist's Hermite polynomial object
2182:     numpy.polynomial.hermite_e.HermiteE : Probabilist's Hermite series
2183:     eval_hermite : evaluate physicist's Hermite polynomials
2184:     ''')
2185: 
2186: add_newdoc("scipy.special", "exp1",
2187:     '''
2188:     exp1(z)
2189: 
2190:     Exponential integral E_1 of complex argument z
2191: 
2192:     ::
2193: 
2194:         integral(exp(-z*t)/t, t=1..inf).
2195:     ''')
2196: 
2197: add_newdoc("scipy.special", "exp10",
2198:     '''
2199:     exp10(x)
2200: 
2201:     Compute ``10**x`` element-wise.
2202: 
2203:     Parameters
2204:     ----------
2205:     x : array_like
2206:         `x` must contain real numbers.
2207: 
2208:     Returns
2209:     -------
2210:     float
2211:         ``10**x``, computed element-wise.
2212: 
2213:     Examples
2214:     --------
2215:     >>> from scipy.special import exp10
2216: 
2217:     >>> exp10(3)
2218:     1000.0
2219:     >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
2220:     >>> exp10(x)
2221:     array([[  0.1       ,   0.31622777,   1.        ],
2222:            [  3.16227766,  10.        ,  31.6227766 ]])
2223: 
2224:     ''')
2225: 
2226: add_newdoc("scipy.special", "exp2",
2227:     '''
2228:     exp2(x)
2229: 
2230:     Compute ``2**x`` element-wise.
2231: 
2232:     Parameters
2233:     ----------
2234:     x : array_like
2235:         `x` must contain real numbers.
2236: 
2237:     Returns
2238:     -------
2239:     float
2240:         ``2**x``, computed element-wise.
2241: 
2242:     Examples
2243:     --------
2244:     >>> from scipy.special import exp2
2245: 
2246:     >>> exp2(3)
2247:     8.0
2248:     >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])
2249:     >>> exp2(x)
2250:     array([[ 0.5       ,  0.70710678,  1.        ],
2251:            [ 1.41421356,  2.        ,  2.82842712]])
2252:     ''')
2253: 
2254: add_newdoc("scipy.special", "expi",
2255:     '''
2256:     expi(x)
2257: 
2258:     Exponential integral Ei
2259: 
2260:     Defined as::
2261: 
2262:         integral(exp(t)/t, t=-inf..x)
2263: 
2264:     See `expn` for a different exponential integral.
2265:     ''')
2266: 
2267: add_newdoc('scipy.special', 'expit',
2268:     '''
2269:     expit(x)
2270: 
2271:     Expit ufunc for ndarrays.
2272: 
2273:     The expit function, also known as the logistic function, is defined as
2274:     expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.
2275: 
2276:     Parameters
2277:     ----------
2278:     x : ndarray
2279:         The ndarray to apply expit to element-wise.
2280: 
2281:     Returns
2282:     -------
2283:     out : ndarray
2284:         An ndarray of the same shape as x. Its entries
2285:         are expit of the corresponding entry of x.
2286: 
2287:     See Also
2288:     --------
2289:     logit
2290: 
2291:     Notes
2292:     -----
2293:     As a ufunc expit takes a number of optional
2294:     keyword arguments. For more information
2295:     see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
2296: 
2297:     .. versionadded:: 0.10.0
2298: 
2299:     Examples
2300:     --------
2301:     >>> from scipy.special import expit, logit
2302: 
2303:     >>> expit([-np.inf, -1.5, 0, 1.5, np.inf])
2304:     array([ 0.        ,  0.18242552,  0.5       ,  0.81757448,  1.        ])
2305: 
2306:     `logit` is the inverse of `expit`:
2307: 
2308:     >>> logit(expit([-2.5, 0, 3.1, 5.0]))
2309:     array([-2.5,  0. ,  3.1,  5. ])
2310: 
2311:     Plot expit(x) for x in [-6, 6]:
2312: 
2313:     >>> import matplotlib.pyplot as plt
2314:     >>> x = np.linspace(-6, 6, 121)
2315:     >>> y = expit(x)
2316:     >>> plt.plot(x, y)
2317:     >>> plt.grid()
2318:     >>> plt.xlim(-6, 6)
2319:     >>> plt.xlabel('x')
2320:     >>> plt.title('expit(x)')
2321:     >>> plt.show()
2322: 
2323:     ''')
2324: 
2325: add_newdoc("scipy.special", "expm1",
2326:     '''
2327:     expm1(x)
2328: 
2329:     Compute ``exp(x) - 1``.
2330: 
2331:     When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
2332:     of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
2333:     ``expm1(x)`` is implemented to avoid the loss of precision that occurs when
2334:     `x` is near zero.
2335: 
2336:     Parameters
2337:     ----------
2338:     x : array_like
2339:         `x` must contain real numbers.
2340: 
2341:     Returns
2342:     -------
2343:     float
2344:         ``exp(x) - 1`` computed element-wise.
2345: 
2346:     Examples
2347:     --------
2348:     >>> from scipy.special import expm1
2349: 
2350:     >>> expm1(1.0)
2351:     1.7182818284590451
2352:     >>> expm1([-0.2, -0.1, 0, 0.1, 0.2])
2353:     array([-0.18126925, -0.09516258,  0.        ,  0.10517092,  0.22140276])
2354: 
2355:     The exact value of ``exp(7.5e-13) - 1`` is::
2356: 
2357:         7.5000000000028125000000007031250000001318...*10**-13.
2358: 
2359:     Here is what ``expm1(7.5e-13)`` gives:
2360: 
2361:     >>> expm1(7.5e-13)
2362:     7.5000000000028135e-13
2363: 
2364:     Compare that to ``exp(7.5e-13) - 1``, where the subtraction results in
2365:     a "catastrophic" loss of precision:
2366: 
2367:     >>> np.exp(7.5e-13) - 1
2368:     7.5006667543675576e-13
2369: 
2370:     ''')
2371: 
2372: add_newdoc("scipy.special", "expn",
2373:     '''
2374:     expn(n, x)
2375: 
2376:     Exponential integral E_n
2377: 
2378:     Returns the exponential integral for integer `n` and non-negative `x` and
2379:     `n`::
2380: 
2381:         integral(exp(-x*t) / t**n, t=1..inf).
2382:     ''')
2383: 
2384: add_newdoc("scipy.special", "exprel",
2385:     r'''
2386:     exprel(x)
2387: 
2388:     Relative error exponential, ``(exp(x) - 1)/x``.
2389: 
2390:     When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation
2391:     of ``exp(x) - 1`` can suffer from catastrophic loss of precision.
2392:     ``exprel(x)`` is implemented to avoid the loss of precision that occurs when
2393:     `x` is near zero.
2394: 
2395:     Parameters
2396:     ----------
2397:     x : ndarray
2398:         Input array.  `x` must contain real numbers.
2399: 
2400:     Returns
2401:     -------
2402:     float
2403:         ``(exp(x) - 1)/x``, computed element-wise.
2404: 
2405:     See Also
2406:     --------
2407:     expm1
2408: 
2409:     Notes
2410:     -----
2411:     .. versionadded:: 0.17.0
2412: 
2413:     Examples
2414:     --------
2415:     >>> from scipy.special import exprel
2416: 
2417:     >>> exprel(0.01)
2418:     1.0050167084168056
2419:     >>> exprel([-0.25, -0.1, 0, 0.1, 0.25])
2420:     array([ 0.88479687,  0.95162582,  1.        ,  1.05170918,  1.13610167])
2421: 
2422:     Compare ``exprel(5e-9)`` to the naive calculation.  The exact value
2423:     is ``1.00000000250000000416...``.
2424: 
2425:     >>> exprel(5e-9)
2426:     1.0000000025
2427: 
2428:     >>> (np.exp(5e-9) - 1)/5e-9
2429:     0.99999999392252903
2430:     ''')
2431: 
2432: add_newdoc("scipy.special", "fdtr",
2433:     r'''
2434:     fdtr(dfn, dfd, x)
2435: 
2436:     F cumulative distribution function.
2437: 
2438:     Returns the value of the cumulative density function of the
2439:     F-distribution, also known as Snedecor's F-distribution or the
2440:     Fisher-Snedecor distribution.
2441: 
2442:     The F-distribution with parameters :math:`d_n` and :math:`d_d` is the
2443:     distribution of the random variable,
2444: 
2445:     .. math::
2446:         X = \frac{U_n/d_n}{U_d/d_d},
2447: 
2448:     where :math:`U_n` and :math:`U_d` are random variables distributed
2449:     :math:`\chi^2`, with :math:`d_n` and :math:`d_d` degrees of freedom,
2450:     respectively.
2451: 
2452:     Parameters
2453:     ----------
2454:     dfn : array_like
2455:         First parameter (positive float).
2456:     dfd : array_like
2457:         Second parameter (positive float).
2458:     x : array_like
2459:         Argument (nonnegative float).
2460: 
2461:     Returns
2462:     -------
2463:     y : ndarray
2464:         The CDF of the F-distribution with parameters `dfn` and `dfd` at `x`.
2465: 
2466:     Notes
2467:     -----
2468:     The regularized incomplete beta function is used, according to the
2469:     formula,
2470: 
2471:     .. math::
2472:         F(d_n, d_d; x) = I_{xd_n/(d_d + xd_n)}(d_n/2, d_d/2).
2473: 
2474:     Wrapper for the Cephes [1]_ routine `fdtr`.
2475: 
2476:     References
2477:     ----------
2478:     .. [1] Cephes Mathematical Functions Library,
2479:            http://www.netlib.org/cephes/index.html
2480: 
2481:     ''')
2482: 
2483: add_newdoc("scipy.special", "fdtrc",
2484:     r'''
2485:     fdtrc(dfn, dfd, x)
2486: 
2487:     F survival function.
2488: 
2489:     Returns the complemented F-distribution function (the integral of the
2490:     density from `x` to infinity).
2491: 
2492:     Parameters
2493:     ----------
2494:     dfn : array_like
2495:         First parameter (positive float).
2496:     dfd : array_like
2497:         Second parameter (positive float).
2498:     x : array_like
2499:         Argument (nonnegative float).
2500: 
2501:     Returns
2502:     -------
2503:     y : ndarray
2504:         The complemented F-distribution function with parameters `dfn` and
2505:         `dfd` at `x`.
2506: 
2507:     See also
2508:     --------
2509:     fdtr
2510: 
2511:     Notes
2512:     -----
2513:     The regularized incomplete beta function is used, according to the
2514:     formula,
2515: 
2516:     .. math::
2517:         F(d_n, d_d; x) = I_{d_d/(d_d + xd_n)}(d_d/2, d_n/2).
2518: 
2519:     Wrapper for the Cephes [1]_ routine `fdtrc`.
2520: 
2521:     References
2522:     ----------
2523:     .. [1] Cephes Mathematical Functions Library,
2524:            http://www.netlib.org/cephes/index.html
2525:     ''')
2526: 
2527: add_newdoc("scipy.special", "fdtri",
2528:     r'''
2529:     fdtri(dfn, dfd, p)
2530: 
2531:     The `p`-th quantile of the F-distribution.
2532: 
2533:     This function is the inverse of the F-distribution CDF, `fdtr`, returning
2534:     the `x` such that `fdtr(dfn, dfd, x) = p`.
2535: 
2536:     Parameters
2537:     ----------
2538:     dfn : array_like
2539:         First parameter (positive float).
2540:     dfd : array_like
2541:         Second parameter (positive float).
2542:     p : array_like
2543:         Cumulative probability, in [0, 1].
2544: 
2545:     Returns
2546:     -------
2547:     x : ndarray
2548:         The quantile corresponding to `p`.
2549: 
2550:     Notes
2551:     -----
2552:     The computation is carried out using the relation to the inverse
2553:     regularized beta function, :math:`I^{-1}_x(a, b)`.  Let
2554:     :math:`z = I^{-1}_p(d_d/2, d_n/2).`  Then,
2555: 
2556:     .. math::
2557:         x = \frac{d_d (1 - z)}{d_n z}.
2558: 
2559:     If `p` is such that :math:`x < 0.5`, the following relation is used
2560:     instead for improved stability: let
2561:     :math:`z' = I^{-1}_{1 - p}(d_n/2, d_d/2).` Then,
2562: 
2563:     .. math::
2564:         x = \frac{d_d z'}{d_n (1 - z')}.
2565: 
2566:     Wrapper for the Cephes [1]_ routine `fdtri`.
2567: 
2568:     References
2569:     ----------
2570:     .. [1] Cephes Mathematical Functions Library,
2571:            http://www.netlib.org/cephes/index.html
2572: 
2573:     ''')
2574: 
2575: add_newdoc("scipy.special", "fdtridfd",
2576:     '''
2577:     fdtridfd(dfn, p, x)
2578: 
2579:     Inverse to `fdtr` vs dfd
2580: 
2581:     Finds the F density argument dfd such that ``fdtr(dfn, dfd, x) == p``.
2582:     ''')
2583: 
2584: add_newdoc("scipy.special", "fdtridfn",
2585:     '''
2586:     fdtridfn(p, dfd, x)
2587: 
2588:     Inverse to `fdtr` vs dfn
2589: 
2590:     finds the F density argument dfn such that ``fdtr(dfn, dfd, x) == p``.
2591:     ''')
2592: 
2593: add_newdoc("scipy.special", "fresnel",
2594:     '''
2595:     fresnel(z)
2596: 
2597:     Fresnel sin and cos integrals
2598: 
2599:     Defined as::
2600: 
2601:         ssa = integral(sin(pi/2 * t**2), t=0..z)
2602:         csa = integral(cos(pi/2 * t**2), t=0..z)
2603: 
2604:     Parameters
2605:     ----------
2606:     z : float or complex array_like
2607:         Argument
2608: 
2609:     Returns
2610:     -------
2611:     ssa, csa
2612:         Fresnel sin and cos integral values
2613: 
2614:     ''')
2615: 
2616: add_newdoc("scipy.special", "gamma",
2617:     r'''
2618:     gamma(z)
2619: 
2620:     Gamma function.
2621: 
2622:     .. math::
2623: 
2624:           \Gamma(z) = \int_0^\infty x^{z-1} e^{-x} dx = (z - 1)!
2625: 
2626:     The gamma function is often referred to as the generalized
2627:     factorial since ``z*gamma(z) = gamma(z+1)`` and ``gamma(n+1) =
2628:     n!`` for natural number *n*.
2629: 
2630:     Parameters
2631:     ----------
2632:     z : float or complex array_like
2633: 
2634:     Returns
2635:     -------
2636:     float or complex
2637:         The value(s) of gamma(z)
2638: 
2639:     Examples
2640:     --------
2641:     >>> from scipy.special import gamma, factorial
2642: 
2643:     >>> gamma([0, 0.5, 1, 5])
2644:     array([         inf,   1.77245385,   1.        ,  24.        ])
2645: 
2646:     >>> z = 2.5 + 1j
2647:     >>> gamma(z)
2648:     (0.77476210455108352+0.70763120437959293j)
2649:     >>> gamma(z+1), z*gamma(z)  # Recurrence property
2650:     ((1.2292740569981171+2.5438401155000685j),
2651:      (1.2292740569981158+2.5438401155000658j))
2652: 
2653:     >>> gamma(0.5)**2  # gamma(0.5) = sqrt(pi)
2654:     3.1415926535897927
2655: 
2656:     Plot gamma(x) for real x
2657: 
2658:     >>> x = np.linspace(-3.5, 5.5, 2251)
2659:     >>> y = gamma(x)
2660: 
2661:     >>> import matplotlib.pyplot as plt
2662:     >>> plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')
2663:     >>> k = np.arange(1, 7)
2664:     >>> plt.plot(k, factorial(k-1), 'k*', alpha=0.6,
2665:     ...          label='(x-1)!, x = 1, 2, ...')
2666:     >>> plt.xlim(-3.5, 5.5)
2667:     >>> plt.ylim(-10, 25)
2668:     >>> plt.grid()
2669:     >>> plt.xlabel('x')
2670:     >>> plt.legend(loc='lower right')
2671:     >>> plt.show()
2672: 
2673:     ''')
2674: 
2675: add_newdoc("scipy.special", "gammainc",
2676:     r'''
2677:     gammainc(a, x)
2678: 
2679:     Regularized lower incomplete gamma function.
2680: 
2681:     Defined as
2682: 
2683:     .. math::
2684: 
2685:         \frac{1}{\Gamma(a)} \int_0^x t^{a - 1}e^{-t} dt
2686: 
2687:     for :math:`a > 0` and :math:`x \geq 0`. The function satisfies the
2688:     relation ``gammainc(a, x) + gammaincc(a, x) = 1`` where
2689:     `gammaincc` is the regularized upper incomplete gamma function.
2690: 
2691:     Notes
2692:     -----
2693:     The implementation largely follows that of [1]_.
2694: 
2695:     See also
2696:     --------
2697:     gammaincc : regularized upper incomplete gamma function
2698:     gammaincinv : inverse to ``gammainc`` versus ``x``
2699:     gammainccinv : inverse to ``gammaincc`` versus ``x``
2700: 
2701:     References
2702:     ----------
2703:     .. [1] Maddock et. al., "Incomplete Gamma Functions",
2704:        http://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
2705:     ''')
2706: 
2707: add_newdoc("scipy.special", "gammaincc",
2708:     r'''
2709:     gammaincc(a, x)
2710: 
2711:     Regularized upper incomplete gamma function.
2712: 
2713:     Defined as
2714: 
2715:     .. math::
2716: 
2717:         \frac{1}{\Gamma(a)} \int_x^\infty t^{a - 1}e^{-t} dt
2718: 
2719:     for :math:`a > 0` and :math:`x \geq 0`. The function satisfies the
2720:     relation ``gammainc(a, x) + gammaincc(a, x) = 1`` where `gammainc`
2721:     is the regularized lower incomplete gamma function.
2722: 
2723:     Notes
2724:     -----
2725:     The implementation largely follows that of [1]_.
2726: 
2727:     See also
2728:     --------
2729:     gammainc : regularized lower incomplete gamma function
2730:     gammaincinv : inverse to ``gammainc`` versus ``x``
2731:     gammainccinv : inverse to ``gammaincc`` versus ``x``
2732: 
2733:     References
2734:     ----------
2735:     .. [1] Maddock et. al., "Incomplete Gamma Functions",
2736:        http://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
2737:     ''')
2738: 
2739: add_newdoc("scipy.special", "gammainccinv",
2740:     '''
2741:     gammainccinv(a, y)
2742: 
2743:     Inverse to `gammaincc`
2744: 
2745:     Returns `x` such that ``gammaincc(a, x) == y``.
2746:     ''')
2747: 
2748: add_newdoc("scipy.special", "gammaincinv",
2749:     '''
2750:     gammaincinv(a, y)
2751: 
2752:     Inverse to `gammainc`
2753: 
2754:     Returns `x` such that ``gammainc(a, x) = y``.
2755:     ''')
2756: 
2757: add_newdoc("scipy.special", "gammaln",
2758:     '''
2759:     Logarithm of the absolute value of the Gamma function.
2760: 
2761:     Parameters
2762:     ----------
2763:     x : array-like
2764:         Values on the real line at which to compute ``gammaln``
2765: 
2766:     Returns
2767:     -------
2768:     gammaln : ndarray
2769:         Values of ``gammaln`` at x.
2770: 
2771:     See Also
2772:     --------
2773:     gammasgn : sign of the gamma function
2774:     loggamma : principal branch of the logarithm of the gamma function
2775: 
2776:     Notes
2777:     -----
2778:     When used in conjunction with `gammasgn`, this function is useful
2779:     for working in logspace on the real axis without having to deal with
2780:     complex numbers, via the relation ``exp(gammaln(x)) = gammasgn(x)*gamma(x)``.
2781: 
2782:     For complex-valued log-gamma, use `loggamma` instead of `gammaln`.
2783:     ''')
2784: 
2785: add_newdoc("scipy.special", "gammasgn",
2786:     '''
2787:     gammasgn(x)
2788: 
2789:     Sign of the gamma function.
2790: 
2791:     See Also
2792:     --------
2793:     gammaln
2794:     loggamma
2795:     ''')
2796: 
2797: add_newdoc("scipy.special", "gdtr",
2798:     r'''
2799:     gdtr(a, b, x)
2800: 
2801:     Gamma distribution cumulative density function.
2802: 
2803:     Returns the integral from zero to `x` of the gamma probability density
2804:     function,
2805: 
2806:     .. math::
2807: 
2808:         F = \int_0^x \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,
2809: 
2810:     where :math:`\Gamma` is the gamma function.
2811: 
2812:     Parameters
2813:     ----------
2814:     a : array_like
2815:         The rate parameter of the gamma distribution, sometimes denoted
2816:         :math:`\beta` (float).  It is also the reciprocal of the scale
2817:         parameter :math:`\theta`.
2818:     b : array_like
2819:         The shape parameter of the gamma distribution, sometimes denoted
2820:         :math:`\alpha` (float).
2821:     x : array_like
2822:         The quantile (upper limit of integration; float).
2823: 
2824:     See also
2825:     --------
2826:     gdtrc : 1 - CDF of the gamma distribution.
2827: 
2828:     Returns
2829:     -------
2830:     F : ndarray
2831:         The CDF of the gamma distribution with parameters `a` and `b`
2832:         evaluated at `x`.
2833: 
2834:     Notes
2835:     -----
2836:     The evaluation is carried out using the relation to the incomplete gamma
2837:     integral (regularized gamma function).
2838: 
2839:     Wrapper for the Cephes [1]_ routine `gdtr`.
2840: 
2841:     References
2842:     ----------
2843:     .. [1] Cephes Mathematical Functions Library,
2844:            http://www.netlib.org/cephes/index.html
2845: 
2846:     ''')
2847: 
2848: add_newdoc("scipy.special", "gdtrc",
2849:     r'''
2850:     gdtrc(a, b, x)
2851: 
2852:     Gamma distribution survival function.
2853: 
2854:     Integral from `x` to infinity of the gamma probability density function,
2855: 
2856:     .. math::
2857: 
2858:         F = \int_x^\infty \frac{a^b}{\Gamma(b)} t^{b-1} e^{-at}\,dt,
2859: 
2860:     where :math:`\Gamma` is the gamma function.
2861: 
2862:     Parameters
2863:     ----------
2864:     a : array_like
2865:         The rate parameter of the gamma distribution, sometimes denoted
2866:         :math:`\beta` (float).  It is also the reciprocal of the scale
2867:         parameter :math:`\theta`.
2868:     b : array_like
2869:         The shape parameter of the gamma distribution, sometimes denoted
2870:         :math:`\alpha` (float).
2871:     x : array_like
2872:         The quantile (lower limit of integration; float).
2873: 
2874:     Returns
2875:     -------
2876:     F : ndarray
2877:         The survival function of the gamma distribution with parameters `a`
2878:         and `b` evaluated at `x`.
2879: 
2880:     See Also
2881:     --------
2882:     gdtr, gdtri
2883: 
2884:     Notes
2885:     -----
2886:     The evaluation is carried out using the relation to the incomplete gamma
2887:     integral (regularized gamma function).
2888: 
2889:     Wrapper for the Cephes [1]_ routine `gdtrc`.
2890: 
2891:     References
2892:     ----------
2893:     .. [1] Cephes Mathematical Functions Library,
2894:            http://www.netlib.org/cephes/index.html
2895: 
2896:     ''')
2897: 
2898: add_newdoc("scipy.special", "gdtria",
2899:     '''
2900:     gdtria(p, b, x, out=None)
2901: 
2902:     Inverse of `gdtr` vs a.
2903: 
2904:     Returns the inverse with respect to the parameter `a` of ``p =
2905:     gdtr(a, b, x)``, the cumulative distribution function of the gamma
2906:     distribution.
2907: 
2908:     Parameters
2909:     ----------
2910:     p : array_like
2911:         Probability values.
2912:     b : array_like
2913:         `b` parameter values of `gdtr(a, b, x)`.  `b` is the "shape" parameter
2914:         of the gamma distribution.
2915:     x : array_like
2916:         Nonnegative real values, from the domain of the gamma distribution.
2917:     out : ndarray, optional
2918:         If a fourth argument is given, it must be a numpy.ndarray whose size
2919:         matches the broadcast result of `a`, `b` and `x`.  `out` is then the
2920:         array returned by the function.
2921: 
2922:     Returns
2923:     -------
2924:     a : ndarray
2925:         Values of the `a` parameter such that `p = gdtr(a, b, x)`.  `1/a`
2926:         is the "scale" parameter of the gamma distribution.
2927: 
2928:     See Also
2929:     --------
2930:     gdtr : CDF of the gamma distribution.
2931:     gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.
2932:     gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.
2933: 
2934:     Notes
2935:     -----
2936:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.
2937: 
2938:     The cumulative distribution function `p` is computed using a routine by
2939:     DiDinato and Morris [2]_.  Computation of `a` involves a search for a value
2940:     that produces the desired value of `p`.  The search relies on the
2941:     monotonicity of `p` with `a`.
2942: 
2943:     References
2944:     ----------
2945:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
2946:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
2947:            Functions, Inverses, and Other Parameters.
2948:     .. [2] DiDinato, A. R. and Morris, A. H.,
2949:            Computation of the incomplete gamma function ratios and their
2950:            inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.
2951: 
2952:     Examples
2953:     --------
2954:     First evaluate `gdtr`.
2955: 
2956:     >>> from scipy.special import gdtr, gdtria
2957:     >>> p = gdtr(1.2, 3.4, 5.6)
2958:     >>> print(p)
2959:     0.94378087442
2960: 
2961:     Verify the inverse.
2962: 
2963:     >>> gdtria(p, 3.4, 5.6)
2964:     1.2
2965:     ''')
2966: 
2967: add_newdoc("scipy.special", "gdtrib",
2968:     '''
2969:     gdtrib(a, p, x, out=None)
2970: 
2971:     Inverse of `gdtr` vs b.
2972: 
2973:     Returns the inverse with respect to the parameter `b` of ``p =
2974:     gdtr(a, b, x)``, the cumulative distribution function of the gamma
2975:     distribution.
2976: 
2977:     Parameters
2978:     ----------
2979:     a : array_like
2980:         `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"
2981:         parameter of the gamma distribution.
2982:     p : array_like
2983:         Probability values.
2984:     x : array_like
2985:         Nonnegative real values, from the domain of the gamma distribution.
2986:     out : ndarray, optional
2987:         If a fourth argument is given, it must be a numpy.ndarray whose size
2988:         matches the broadcast result of `a`, `b` and `x`.  `out` is then the
2989:         array returned by the function.
2990: 
2991:     Returns
2992:     -------
2993:     b : ndarray
2994:         Values of the `b` parameter such that `p = gdtr(a, b, x)`.  `b` is
2995:         the "shape" parameter of the gamma distribution.
2996: 
2997:     See Also
2998:     --------
2999:     gdtr : CDF of the gamma distribution.
3000:     gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
3001:     gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.
3002: 
3003:     Notes
3004:     -----
3005:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.
3006: 
3007:     The cumulative distribution function `p` is computed using a routine by
3008:     DiDinato and Morris [2]_.  Computation of `b` involves a search for a value
3009:     that produces the desired value of `p`.  The search relies on the
3010:     monotonicity of `p` with `b`.
3011: 
3012:     References
3013:     ----------
3014:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
3015:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
3016:            Functions, Inverses, and Other Parameters.
3017:     .. [2] DiDinato, A. R. and Morris, A. H.,
3018:            Computation of the incomplete gamma function ratios and their
3019:            inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.
3020: 
3021:     Examples
3022:     --------
3023:     First evaluate `gdtr`.
3024: 
3025:     >>> from scipy.special import gdtr, gdtrib
3026:     >>> p = gdtr(1.2, 3.4, 5.6)
3027:     >>> print(p)
3028:     0.94378087442
3029: 
3030:     Verify the inverse.
3031: 
3032:     >>> gdtrib(1.2, p, 5.6)
3033:     3.3999999999723882
3034:     ''')
3035: 
3036: add_newdoc("scipy.special", "gdtrix",
3037:     '''
3038:     gdtrix(a, b, p, out=None)
3039: 
3040:     Inverse of `gdtr` vs x.
3041: 
3042:     Returns the inverse with respect to the parameter `x` of ``p =
3043:     gdtr(a, b, x)``, the cumulative distribution function of the gamma
3044:     distribution. This is also known as the p'th quantile of the
3045:     distribution.
3046: 
3047:     Parameters
3048:     ----------
3049:     a : array_like
3050:         `a` parameter values of `gdtr(a, b, x)`.  `1/a` is the "scale"
3051:         parameter of the gamma distribution.
3052:     b : array_like
3053:         `b` parameter values of `gdtr(a, b, x)`.  `b` is the "shape" parameter
3054:         of the gamma distribution.
3055:     p : array_like
3056:         Probability values.
3057:     out : ndarray, optional
3058:         If a fourth argument is given, it must be a numpy.ndarray whose size
3059:         matches the broadcast result of `a`, `b` and `x`.  `out` is then the
3060:         array returned by the function.
3061: 
3062:     Returns
3063:     -------
3064:     x : ndarray
3065:         Values of the `x` parameter such that `p = gdtr(a, b, x)`.
3066: 
3067:     See Also
3068:     --------
3069:     gdtr : CDF of the gamma distribution.
3070:     gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.
3071:     gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.
3072: 
3073:     Notes
3074:     -----
3075:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.
3076: 
3077:     The cumulative distribution function `p` is computed using a routine by
3078:     DiDinato and Morris [2]_.  Computation of `x` involves a search for a value
3079:     that produces the desired value of `p`.  The search relies on the
3080:     monotonicity of `p` with `x`.
3081: 
3082:     References
3083:     ----------
3084:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
3085:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
3086:            Functions, Inverses, and Other Parameters.
3087:     .. [2] DiDinato, A. R. and Morris, A. H.,
3088:            Computation of the incomplete gamma function ratios and their
3089:            inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.
3090: 
3091:     Examples
3092:     --------
3093:     First evaluate `gdtr`.
3094: 
3095:     >>> from scipy.special import gdtr, gdtrix
3096:     >>> p = gdtr(1.2, 3.4, 5.6)
3097:     >>> print(p)
3098:     0.94378087442
3099: 
3100:     Verify the inverse.
3101: 
3102:     >>> gdtrix(1.2, 3.4, p)
3103:     5.5999999999999996
3104:     ''')
3105: 
3106: add_newdoc("scipy.special", "hankel1",
3107:     r'''
3108:     hankel1(v, z)
3109: 
3110:     Hankel function of the first kind
3111: 
3112:     Parameters
3113:     ----------
3114:     v : array_like
3115:         Order (float).
3116:     z : array_like
3117:         Argument (float or complex).
3118: 
3119:     Returns
3120:     -------
3121:     out : Values of the Hankel function of the first kind.
3122: 
3123:     Notes
3124:     -----
3125:     A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
3126:     computation using the relation,
3127: 
3128:     .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))
3129: 
3130:     where :math:`K_v` is the modified Bessel function of the second kind.
3131:     For negative orders, the relation
3132: 
3133:     .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)
3134: 
3135:     is used.
3136: 
3137:     See also
3138:     --------
3139:     hankel1e : this function with leading exponential behavior stripped off.
3140: 
3141:     References
3142:     ----------
3143:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3144:            of a Complex Argument and Nonnegative Order",
3145:            http://netlib.org/amos/
3146:     ''')
3147: 
3148: add_newdoc("scipy.special", "hankel1e",
3149:     r'''
3150:     hankel1e(v, z)
3151: 
3152:     Exponentially scaled Hankel function of the first kind
3153: 
3154:     Defined as::
3155: 
3156:         hankel1e(v, z) = hankel1(v, z) * exp(-1j * z)
3157: 
3158:     Parameters
3159:     ----------
3160:     v : array_like
3161:         Order (float).
3162:     z : array_like
3163:         Argument (float or complex).
3164: 
3165:     Returns
3166:     -------
3167:     out : Values of the exponentially scaled Hankel function.
3168: 
3169:     Notes
3170:     -----
3171:     A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
3172:     computation using the relation,
3173: 
3174:     .. math:: H^{(1)}_v(z) = \frac{2}{\imath\pi} \exp(-\imath \pi v/2) K_v(z \exp(-\imath\pi/2))
3175: 
3176:     where :math:`K_v` is the modified Bessel function of the second kind.
3177:     For negative orders, the relation
3178: 
3179:     .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \exp(\imath\pi v)
3180: 
3181:     is used.
3182: 
3183:     References
3184:     ----------
3185:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3186:            of a Complex Argument and Nonnegative Order",
3187:            http://netlib.org/amos/
3188:     ''')
3189: 
3190: add_newdoc("scipy.special", "hankel2",
3191:     r'''
3192:     hankel2(v, z)
3193: 
3194:     Hankel function of the second kind
3195: 
3196:     Parameters
3197:     ----------
3198:     v : array_like
3199:         Order (float).
3200:     z : array_like
3201:         Argument (float or complex).
3202: 
3203:     Returns
3204:     -------
3205:     out : Values of the Hankel function of the second kind.
3206: 
3207:     Notes
3208:     -----
3209:     A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
3210:     computation using the relation,
3211: 
3212:     .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\imath \pi v/2) K_v(z \exp(\imath\pi/2))
3213: 
3214:     where :math:`K_v` is the modified Bessel function of the second kind.
3215:     For negative orders, the relation
3216: 
3217:     .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)
3218: 
3219:     is used.
3220: 
3221:     See also
3222:     --------
3223:     hankel2e : this function with leading exponential behavior stripped off.
3224: 
3225:     References
3226:     ----------
3227:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3228:            of a Complex Argument and Nonnegative Order",
3229:            http://netlib.org/amos/
3230:     ''')
3231: 
3232: add_newdoc("scipy.special", "hankel2e",
3233:     r'''
3234:     hankel2e(v, z)
3235: 
3236:     Exponentially scaled Hankel function of the second kind
3237: 
3238:     Defined as::
3239: 
3240:         hankel2e(v, z) = hankel2(v, z) * exp(1j * z)
3241: 
3242:     Parameters
3243:     ----------
3244:     v : array_like
3245:         Order (float).
3246:     z : array_like
3247:         Argument (float or complex).
3248: 
3249:     Returns
3250:     -------
3251:     out : Values of the exponentially scaled Hankel function of the second kind.
3252: 
3253:     Notes
3254:     -----
3255:     A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the
3256:     computation using the relation,
3257: 
3258:     .. math:: H^{(2)}_v(z) = -\frac{2}{\imath\pi} \exp(\frac{\imath \pi v}{2}) K_v(z exp(\frac{\imath\pi}{2}))
3259: 
3260:     where :math:`K_v` is the modified Bessel function of the second kind.
3261:     For negative orders, the relation
3262: 
3263:     .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \exp(-\imath\pi v)
3264: 
3265:     is used.
3266: 
3267:     References
3268:     ----------
3269:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3270:            of a Complex Argument and Nonnegative Order",
3271:            http://netlib.org/amos/
3272: 
3273:     ''')
3274: 
3275: add_newdoc("scipy.special", "huber",
3276:     r'''
3277:     huber(delta, r)
3278: 
3279:     Huber loss function.
3280: 
3281:     .. math:: \text{huber}(\delta, r) = \begin{cases} \infty & \delta < 0  \\ \frac{1}{2}r^2 & 0 \le \delta, | r | \le \delta \\ \delta ( |r| - \frac{1}{2}\delta ) & \text{otherwise} \end{cases}
3282: 
3283:     Parameters
3284:     ----------
3285:     delta : ndarray
3286:         Input array, indicating the quadratic vs. linear loss changepoint.
3287:     r : ndarray
3288:         Input array, possibly representing residuals.
3289: 
3290:     Returns
3291:     -------
3292:     res : ndarray
3293:         The computed Huber loss function values.
3294: 
3295:     Notes
3296:     -----
3297:     This function is convex in r.
3298: 
3299:     .. versionadded:: 0.15.0
3300: 
3301:     ''')
3302: 
3303: add_newdoc("scipy.special", "hyp0f1",
3304:     r'''
3305:     hyp0f1(v, x)
3306: 
3307:     Confluent hypergeometric limit function 0F1.
3308: 
3309:     Parameters
3310:     ----------
3311:     v, z : array_like
3312:         Input values.
3313: 
3314:     Returns
3315:     -------
3316:     hyp0f1 : ndarray
3317:         The confluent hypergeometric limit function.
3318: 
3319:     Notes
3320:     -----
3321:     This function is defined as:
3322: 
3323:     .. math:: _0F_1(v, z) = \sum_{k=0}^{\infty}\frac{z^k}{(v)_k k!}.
3324: 
3325:     It's also the limit as :math:`q \to \infty` of :math:`_1F_1(q; v; z/q)`,
3326:     and satisfies the differential equation :math:`f''(z) + vf'(z) = f(z)`.
3327:     ''')
3328: 
3329: add_newdoc("scipy.special", "hyp1f1",
3330:     '''
3331:     hyp1f1(a, b, x)
3332: 
3333:     Confluent hypergeometric function 1F1(a, b; x)
3334:     ''')
3335: 
3336: add_newdoc("scipy.special", "hyp1f2",
3337:     '''
3338:     hyp1f2(a, b, c, x)
3339: 
3340:     Hypergeometric function 1F2 and error estimate
3341: 
3342:     Returns
3343:     -------
3344:     y
3345:         Value of the function
3346:     err
3347:         Error estimate
3348:     ''')
3349: 
3350: add_newdoc("scipy.special", "hyp2f0",
3351:     '''
3352:     hyp2f0(a, b, x, type)
3353: 
3354:     Hypergeometric function 2F0 in y and an error estimate
3355: 
3356:     The parameter `type` determines a convergence factor and can be
3357:     either 1 or 2.
3358: 
3359:     Returns
3360:     -------
3361:     y
3362:         Value of the function
3363:     err
3364:         Error estimate
3365:     ''')
3366: 
3367: add_newdoc("scipy.special", "hyp2f1",
3368:     r'''
3369:     hyp2f1(a, b, c, z)
3370: 
3371:     Gauss hypergeometric function 2F1(a, b; c; z)
3372: 
3373:     Parameters
3374:     ----------
3375:     a, b, c : array_like
3376:         Arguments, should be real-valued.
3377:     z : array_like
3378:         Argument, real or complex.
3379: 
3380:     Returns
3381:     -------
3382:     hyp2f1 : scalar or ndarray
3383:         The values of the gaussian hypergeometric function.
3384: 
3385:     See also
3386:     --------
3387:     hyp0f1 : confluent hypergeometric limit function.
3388:     hyp1f1 : Kummer's (confluent hypergeometric) function.
3389: 
3390:     Notes
3391:     -----
3392:     This function is defined for :math:`|z| < 1` as
3393: 
3394:     .. math::
3395: 
3396:        \mathrm{hyp2f1}(a, b, c, z) = \sum_{n=0}^\infty
3397:        \frac{(a)_n (b)_n}{(c)_n}\frac{z^n}{n!},
3398: 
3399:     and defined on the rest of the complex z-plane by analytic continuation.
3400:     Here :math:`(\cdot)_n` is the Pochhammer symbol; see `poch`. When
3401:     :math:`n` is an integer the result is a polynomial of degree :math:`n`.
3402: 
3403:     The implementation for complex values of ``z`` is described in [1]_.
3404: 
3405:     References
3406:     ----------
3407:     .. [1] J.M. Jin and Z. S. Jjie, "Computation of special functions", Wiley, 1996.
3408:     .. [2] Cephes Mathematical Functions Library,
3409:            http://www.netlib.org/cephes/index.html
3410:     .. [3] NIST Digital Library of Mathematical Functions
3411:            http://dlmf.nist.gov/
3412: 
3413:     ''')
3414: 
3415: add_newdoc("scipy.special", "hyp3f0",
3416:     '''
3417:     hyp3f0(a, b, c, x)
3418: 
3419:     Hypergeometric function 3F0 in y and an error estimate
3420: 
3421:     Returns
3422:     -------
3423:     y
3424:         Value of the function
3425:     err
3426:         Error estimate
3427:     ''')
3428: 
3429: add_newdoc("scipy.special", "hyperu",
3430:     '''
3431:     hyperu(a, b, x)
3432: 
3433:     Confluent hypergeometric function U(a, b, x) of the second kind
3434:     ''')
3435: 
3436: add_newdoc("scipy.special", "i0",
3437:     r'''
3438:     i0(x)
3439: 
3440:     Modified Bessel function of order 0.
3441: 
3442:     Defined as,
3443: 
3444:     .. math::
3445:         I_0(x) = \sum_{k=0}^\infty \frac{(x^2/4)^k}{(k!)^2} = J_0(\imath x),
3446: 
3447:     where :math:`J_0` is the Bessel function of the first kind of order 0.
3448: 
3449:     Parameters
3450:     ----------
3451:     x : array_like
3452:         Argument (float)
3453: 
3454:     Returns
3455:     -------
3456:     I : ndarray
3457:         Value of the modified Bessel function of order 0 at `x`.
3458: 
3459:     Notes
3460:     -----
3461:     The range is partitioned into the two intervals [0, 8] and (8, infinity).
3462:     Chebyshev polynomial expansions are employed in each interval.
3463: 
3464:     This function is a wrapper for the Cephes [1]_ routine `i0`.
3465: 
3466:     See also
3467:     --------
3468:     iv
3469:     i0e
3470: 
3471:     References
3472:     ----------
3473:     .. [1] Cephes Mathematical Functions Library,
3474:            http://www.netlib.org/cephes/index.html
3475:     ''')
3476: 
3477: add_newdoc("scipy.special", "i0e",
3478:     '''
3479:     i0e(x)
3480: 
3481:     Exponentially scaled modified Bessel function of order 0.
3482: 
3483:     Defined as::
3484: 
3485:         i0e(x) = exp(-abs(x)) * i0(x).
3486: 
3487:     Parameters
3488:     ----------
3489:     x : array_like
3490:         Argument (float)
3491: 
3492:     Returns
3493:     -------
3494:     I : ndarray
3495:         Value of the exponentially scaled modified Bessel function of order 0
3496:         at `x`.
3497: 
3498:     Notes
3499:     -----
3500:     The range is partitioned into the two intervals [0, 8] and (8, infinity).
3501:     Chebyshev polynomial expansions are employed in each interval.  The
3502:     polynomial expansions used are the same as those in `i0`, but
3503:     they are not multiplied by the dominant exponential factor.
3504: 
3505:     This function is a wrapper for the Cephes [1]_ routine `i0e`.
3506: 
3507:     See also
3508:     --------
3509:     iv
3510:     i0
3511: 
3512:     References
3513:     ----------
3514:     .. [1] Cephes Mathematical Functions Library,
3515:            http://www.netlib.org/cephes/index.html
3516:     ''')
3517: 
3518: add_newdoc("scipy.special", "i1",
3519:     r'''
3520:     i1(x)
3521: 
3522:     Modified Bessel function of order 1.
3523: 
3524:     Defined as,
3525: 
3526:     .. math::
3527:         I_1(x) = \frac{1}{2}x \sum_{k=0}^\infty \frac{(x^2/4)^k}{k! (k + 1)!}
3528:                = -\imath J_1(\imath x),
3529: 
3530:     where :math:`J_1` is the Bessel function of the first kind of order 1.
3531: 
3532:     Parameters
3533:     ----------
3534:     x : array_like
3535:         Argument (float)
3536: 
3537:     Returns
3538:     -------
3539:     I : ndarray
3540:         Value of the modified Bessel function of order 1 at `x`.
3541: 
3542:     Notes
3543:     -----
3544:     The range is partitioned into the two intervals [0, 8] and (8, infinity).
3545:     Chebyshev polynomial expansions are employed in each interval.
3546: 
3547:     This function is a wrapper for the Cephes [1]_ routine `i1`.
3548: 
3549:     See also
3550:     --------
3551:     iv
3552:     i1e
3553: 
3554:     References
3555:     ----------
3556:     .. [1] Cephes Mathematical Functions Library,
3557:            http://www.netlib.org/cephes/index.html
3558:     ''')
3559: 
3560: add_newdoc("scipy.special", "i1e",
3561:     '''
3562:     i1e(x)
3563: 
3564:     Exponentially scaled modified Bessel function of order 1.
3565: 
3566:     Defined as::
3567: 
3568:         i1e(x) = exp(-abs(x)) * i1(x)
3569: 
3570:     Parameters
3571:     ----------
3572:     x : array_like
3573:         Argument (float)
3574: 
3575:     Returns
3576:     -------
3577:     I : ndarray
3578:         Value of the exponentially scaled modified Bessel function of order 1
3579:         at `x`.
3580: 
3581:     Notes
3582:     -----
3583:     The range is partitioned into the two intervals [0, 8] and (8, infinity).
3584:     Chebyshev polynomial expansions are employed in each interval. The
3585:     polynomial expansions used are the same as those in `i1`, but
3586:     they are not multiplied by the dominant exponential factor.
3587: 
3588:     This function is a wrapper for the Cephes [1]_ routine `i1e`.
3589: 
3590:     See also
3591:     --------
3592:     iv
3593:     i1
3594: 
3595:     References
3596:     ----------
3597:     .. [1] Cephes Mathematical Functions Library,
3598:            http://www.netlib.org/cephes/index.html
3599:     ''')
3600: 
3601: add_newdoc("scipy.special", "_igam_fac",
3602:     '''
3603:     Internal function, do not use.
3604:     ''')
3605: 
3606: add_newdoc("scipy.special", "it2i0k0",
3607:     '''
3608:     it2i0k0(x)
3609: 
3610:     Integrals related to modified Bessel functions of order 0
3611: 
3612:     Returns
3613:     -------
3614:     ii0
3615:         ``integral((i0(t)-1)/t, t=0..x)``
3616:     ik0
3617:         ``int(k0(t)/t, t=x..inf)``
3618:     ''')
3619: 
3620: add_newdoc("scipy.special", "it2j0y0",
3621:     '''
3622:     it2j0y0(x)
3623: 
3624:     Integrals related to Bessel functions of order 0
3625: 
3626:     Returns
3627:     -------
3628:     ij0
3629:         ``integral((1-j0(t))/t, t=0..x)``
3630:     iy0
3631:         ``integral(y0(t)/t, t=x..inf)``
3632:     ''')
3633: 
3634: add_newdoc("scipy.special", "it2struve0",
3635:     r'''
3636:     it2struve0(x)
3637: 
3638:     Integral related to the Struve function of order 0.
3639: 
3640:     Returns the integral,
3641: 
3642:     .. math::
3643:         \int_x^\infty \frac{H_0(t)}{t}\,dt
3644: 
3645:     where :math:`H_0` is the Struve function of order 0.
3646: 
3647:     Parameters
3648:     ----------
3649:     x : array_like
3650:         Lower limit of integration.
3651: 
3652:     Returns
3653:     -------
3654:     I : ndarray
3655:         The value of the integral.
3656: 
3657:     See also
3658:     --------
3659:     struve
3660: 
3661:     Notes
3662:     -----
3663:     Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
3664:     Jin [1]_.
3665: 
3666:     References
3667:     ----------
3668:     .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
3669:            Functions", John Wiley and Sons, 1996.
3670:            https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
3671:     ''')
3672: 
3673: add_newdoc("scipy.special", "itairy",
3674:     '''
3675:     itairy(x)
3676: 
3677:     Integrals of Airy functions
3678: 
3679:     Calculates the integrals of Airy functions from 0 to `x`.
3680: 
3681:     Parameters
3682:     ----------
3683: 
3684:     x: array_like
3685:         Upper limit of integration (float).
3686: 
3687:     Returns
3688:     -------
3689:     Apt
3690:         Integral of Ai(t) from 0 to x.
3691:     Bpt
3692:         Integral of Bi(t) from 0 to x.
3693:     Ant
3694:         Integral of Ai(-t) from 0 to x.
3695:     Bnt
3696:         Integral of Bi(-t) from 0 to x.
3697: 
3698:     Notes
3699:     -----
3700: 
3701:     Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
3702:     Jin [1]_.
3703: 
3704:     References
3705:     ----------
3706: 
3707:     .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
3708:            Functions", John Wiley and Sons, 1996.
3709:            https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
3710:     ''')
3711: 
3712: add_newdoc("scipy.special", "iti0k0",
3713:     '''
3714:     iti0k0(x)
3715: 
3716:     Integrals of modified Bessel functions of order 0
3717: 
3718:     Returns simple integrals from 0 to `x` of the zeroth order modified
3719:     Bessel functions `i0` and `k0`.
3720: 
3721:     Returns
3722:     -------
3723:     ii0, ik0
3724:     ''')
3725: 
3726: add_newdoc("scipy.special", "itj0y0",
3727:     '''
3728:     itj0y0(x)
3729: 
3730:     Integrals of Bessel functions of order 0
3731: 
3732:     Returns simple integrals from 0 to `x` of the zeroth order Bessel
3733:     functions `j0` and `y0`.
3734: 
3735:     Returns
3736:     -------
3737:     ij0, iy0
3738:     ''')
3739: 
3740: add_newdoc("scipy.special", "itmodstruve0",
3741:     r'''
3742:     itmodstruve0(x)
3743: 
3744:     Integral of the modified Struve function of order 0.
3745: 
3746:     .. math::
3747:         I = \int_0^x L_0(t)\,dt
3748: 
3749:     Parameters
3750:     ----------
3751:     x : array_like
3752:         Upper limit of integration (float).
3753: 
3754:     Returns
3755:     -------
3756:     I : ndarray
3757:         The integral of :math:`L_0` from 0 to `x`.
3758: 
3759:     Notes
3760:     -----
3761:     Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
3762:     Jin [1]_.
3763: 
3764:     References
3765:     ----------
3766:     .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
3767:            Functions", John Wiley and Sons, 1996.
3768:            https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
3769: 
3770:     ''')
3771: 
3772: add_newdoc("scipy.special", "itstruve0",
3773:     r'''
3774:     itstruve0(x)
3775: 
3776:     Integral of the Struve function of order 0.
3777: 
3778:     .. math::
3779:         I = \int_0^x H_0(t)\,dt
3780: 
3781:     Parameters
3782:     ----------
3783:     x : array_like
3784:         Upper limit of integration (float).
3785: 
3786:     Returns
3787:     -------
3788:     I : ndarray
3789:         The integral of :math:`H_0` from 0 to `x`.
3790: 
3791:     See also
3792:     --------
3793:     struve
3794: 
3795:     Notes
3796:     -----
3797:     Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
3798:     Jin [1]_.
3799: 
3800:     References
3801:     ----------
3802:     .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
3803:            Functions", John Wiley and Sons, 1996.
3804:            https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
3805: 
3806:     ''')
3807: 
3808: add_newdoc("scipy.special", "iv",
3809:     r'''
3810:     iv(v, z)
3811: 
3812:     Modified Bessel function of the first kind of real order.
3813: 
3814:     Parameters
3815:     ----------
3816:     v : array_like
3817:         Order. If `z` is of real type and negative, `v` must be integer
3818:         valued.
3819:     z : array_like of float or complex
3820:         Argument.
3821: 
3822:     Returns
3823:     -------
3824:     out : ndarray
3825:         Values of the modified Bessel function.
3826: 
3827:     Notes
3828:     -----
3829:     For real `z` and :math:`v \in [-50, 50]`, the evaluation is carried out
3830:     using Temme's method [1]_.  For larger orders, uniform asymptotic
3831:     expansions are applied.
3832: 
3833:     For complex `z` and positive `v`, the AMOS [2]_ `zbesi` routine is
3834:     called. It uses a power series for small `z`, the asymptotic expansion
3835:     for large `abs(z)`, the Miller algorithm normalized by the Wronskian
3836:     and a Neumann series for intermediate magnitudes, and the uniform
3837:     asymptotic expansions for :math:`I_v(z)` and :math:`J_v(z)` for large
3838:     orders.  Backward recurrence is used to generate sequences or reduce
3839:     orders when necessary.
3840: 
3841:     The calculations above are done in the right half plane and continued
3842:     into the left half plane by the formula,
3843: 
3844:     .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)
3845: 
3846:     (valid when the real part of `z` is positive).  For negative `v`, the
3847:     formula
3848: 
3849:     .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)
3850: 
3851:     is used, where :math:`K_v(z)` is the modified Bessel function of the
3852:     second kind, evaluated using the AMOS routine `zbesk`.
3853: 
3854:     See also
3855:     --------
3856:     kve : This function with leading exponential behavior stripped off.
3857: 
3858:     References
3859:     ----------
3860:     .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)
3861:     .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3862:            of a Complex Argument and Nonnegative Order",
3863:            http://netlib.org/amos/
3864:     ''')
3865: 
3866: add_newdoc("scipy.special", "ive",
3867:     r'''
3868:     ive(v, z)
3869: 
3870:     Exponentially scaled modified Bessel function of the first kind
3871: 
3872:     Defined as::
3873: 
3874:         ive(v, z) = iv(v, z) * exp(-abs(z.real))
3875: 
3876:     Parameters
3877:     ----------
3878:     v : array_like of float
3879:         Order.
3880:     z : array_like of float or complex
3881:         Argument.
3882: 
3883:     Returns
3884:     -------
3885:     out : ndarray
3886:         Values of the exponentially scaled modified Bessel function.
3887: 
3888:     Notes
3889:     -----
3890:     For positive `v`, the AMOS [1]_ `zbesi` routine is called. It uses a
3891:     power series for small `z`, the asymptotic expansion for large
3892:     `abs(z)`, the Miller algorithm normalized by the Wronskian and a
3893:     Neumann series for intermediate magnitudes, and the uniform asymptotic
3894:     expansions for :math:`I_v(z)` and :math:`J_v(z)` for large orders.
3895:     Backward recurrence is used to generate sequences or reduce orders when
3896:     necessary.
3897: 
3898:     The calculations above are done in the right half plane and continued
3899:     into the left half plane by the formula,
3900: 
3901:     .. math:: I_v(z \exp(\pm\imath\pi)) = \exp(\pm\pi v) I_v(z)
3902: 
3903:     (valid when the real part of `z` is positive).  For negative `v`, the
3904:     formula
3905: 
3906:     .. math:: I_{-v}(z) = I_v(z) + \frac{2}{\pi} \sin(\pi v) K_v(z)
3907: 
3908:     is used, where :math:`K_v(z)` is the modified Bessel function of the
3909:     second kind, evaluated using the AMOS routine `zbesk`.
3910: 
3911:     References
3912:     ----------
3913:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
3914:            of a Complex Argument and Nonnegative Order",
3915:            http://netlib.org/amos/
3916:     ''')
3917: 
3918: add_newdoc("scipy.special", "j0",
3919:     r'''
3920:     j0(x)
3921: 
3922:     Bessel function of the first kind of order 0.
3923: 
3924:     Parameters
3925:     ----------
3926:     x : array_like
3927:         Argument (float).
3928: 
3929:     Returns
3930:     -------
3931:     J : ndarray
3932:         Value of the Bessel function of the first kind of order 0 at `x`.
3933: 
3934:     Notes
3935:     -----
3936:     The domain is divided into the intervals [0, 5] and (5, infinity). In the
3937:     first interval the following rational approximation is used:
3938: 
3939:     .. math::
3940: 
3941:         J_0(x) \approx (w - r_1^2)(w - r_2^2) \frac{P_3(w)}{Q_8(w)},
3942: 
3943:     where :math:`w = x^2` and :math:`r_1`, :math:`r_2` are the zeros of
3944:     :math:`J_0`, and :math:`P_3` and :math:`Q_8` are polynomials of degrees 3
3945:     and 8, respectively.
3946: 
3947:     In the second interval, the Hankel asymptotic expansion is employed with
3948:     two rational functions of degree 6/6 and 7/7.
3949: 
3950:     This function is a wrapper for the Cephes [1]_ routine `j0`.
3951:     It should not to be confused with the spherical Bessel functions (see
3952:     `spherical_jn`).
3953: 
3954:     See also
3955:     --------
3956:     jv : Bessel function of real order and complex argument.
3957:     spherical_jn : spherical Bessel functions.
3958: 
3959:     References
3960:     ----------
3961:     .. [1] Cephes Mathematical Functions Library,
3962:            http://www.netlib.org/cephes/index.html
3963:     ''')
3964: 
3965: add_newdoc("scipy.special", "j1",
3966:     '''
3967:     j1(x)
3968: 
3969:     Bessel function of the first kind of order 1.
3970: 
3971:     Parameters
3972:     ----------
3973:     x : array_like
3974:         Argument (float).
3975: 
3976:     Returns
3977:     -------
3978:     J : ndarray
3979:         Value of the Bessel function of the first kind of order 1 at `x`.
3980: 
3981:     Notes
3982:     -----
3983:     The domain is divided into the intervals [0, 8] and (8, infinity). In the
3984:     first interval a 24 term Chebyshev expansion is used. In the second, the
3985:     asymptotic trigonometric representation is employed using two rational
3986:     functions of degree 5/5.
3987: 
3988:     This function is a wrapper for the Cephes [1]_ routine `j1`.
3989:     It should not to be confused with the spherical Bessel functions (see
3990:     `spherical_jn`).
3991: 
3992:     See also
3993:     --------
3994:     jv
3995:     spherical_jn : spherical Bessel functions.
3996: 
3997:     References
3998:     ----------
3999:     .. [1] Cephes Mathematical Functions Library,
4000:            http://www.netlib.org/cephes/index.html
4001: 
4002:     ''')
4003: 
4004: add_newdoc("scipy.special", "jn",
4005:     '''
4006:     jn(n, x)
4007: 
4008:     Bessel function of the first kind of integer order and real argument.
4009: 
4010:     Notes
4011:     -----
4012:     `jn` is an alias of `jv`.
4013:     Not to be confused with the spherical Bessel functions (see `spherical_jn`).
4014: 
4015:     See also
4016:     --------
4017:     jv
4018:     spherical_jn : spherical Bessel functions.
4019: 
4020:     ''')
4021: 
4022: add_newdoc("scipy.special", "jv",
4023:     r'''
4024:     jv(v, z)
4025: 
4026:     Bessel function of the first kind of real order and complex argument.
4027: 
4028:     Parameters
4029:     ----------
4030:     v : array_like
4031:         Order (float).
4032:     z : array_like
4033:         Argument (float or complex).
4034: 
4035:     Returns
4036:     -------
4037:     J : ndarray
4038:         Value of the Bessel function, :math:`J_v(z)`.
4039: 
4040:     Notes
4041:     -----
4042:     For positive `v` values, the computation is carried out using the AMOS
4043:     [1]_ `zbesj` routine, which exploits the connection to the modified
4044:     Bessel function :math:`I_v`,
4045: 
4046:     .. math::
4047:         J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)
4048: 
4049:         J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)
4050: 
4051:     For negative `v` values the formula,
4052: 
4053:     .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)
4054: 
4055:     is used, where :math:`Y_v(z)` is the Bessel function of the second
4056:     kind, computed using the AMOS routine `zbesy`.  Note that the second
4057:     term is exactly zero for integer `v`; to improve accuracy the second
4058:     term is explicitly omitted for `v` values such that `v = floor(v)`.
4059: 
4060:     Not to be confused with the spherical Bessel functions (see `spherical_jn`).
4061: 
4062:     See also
4063:     --------
4064:     jve : :math:`J_v` with leading exponential behavior stripped off.
4065:     spherical_jn : spherical Bessel functions.
4066: 
4067:     References
4068:     ----------
4069:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
4070:            of a Complex Argument and Nonnegative Order",
4071:            http://netlib.org/amos/
4072:     ''')
4073: 
4074: add_newdoc("scipy.special", "jve",
4075:     r'''
4076:     jve(v, z)
4077: 
4078:     Exponentially scaled Bessel function of order `v`.
4079: 
4080:     Defined as::
4081: 
4082:         jve(v, z) = jv(v, z) * exp(-abs(z.imag))
4083: 
4084:     Parameters
4085:     ----------
4086:     v : array_like
4087:         Order (float).
4088:     z : array_like
4089:         Argument (float or complex).
4090: 
4091:     Returns
4092:     -------
4093:     J : ndarray
4094:         Value of the exponentially scaled Bessel function.
4095: 
4096:     Notes
4097:     -----
4098:     For positive `v` values, the computation is carried out using the AMOS
4099:     [1]_ `zbesj` routine, which exploits the connection to the modified
4100:     Bessel function :math:`I_v`,
4101: 
4102:     .. math::
4103:         J_v(z) = \exp(v\pi\imath/2) I_v(-\imath z)\qquad (\Im z > 0)
4104: 
4105:         J_v(z) = \exp(-v\pi\imath/2) I_v(\imath z)\qquad (\Im z < 0)
4106: 
4107:     For negative `v` values the formula,
4108: 
4109:     .. math:: J_{-v}(z) = J_v(z) \cos(\pi v) - Y_v(z) \sin(\pi v)
4110: 
4111:     is used, where :math:`Y_v(z)` is the Bessel function of the second
4112:     kind, computed using the AMOS routine `zbesy`.  Note that the second
4113:     term is exactly zero for integer `v`; to improve accuracy the second
4114:     term is explicitly omitted for `v` values such that `v = floor(v)`.
4115: 
4116:     References
4117:     ----------
4118:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
4119:            of a Complex Argument and Nonnegative Order",
4120:            http://netlib.org/amos/
4121:     ''')
4122: 
4123: add_newdoc("scipy.special", "k0",
4124:     r'''
4125:     k0(x)
4126: 
4127:     Modified Bessel function of the second kind of order 0, :math:`K_0`.
4128: 
4129:     This function is also sometimes referred to as the modified Bessel
4130:     function of the third kind of order 0.
4131: 
4132:     Parameters
4133:     ----------
4134:     x : array_like
4135:         Argument (float).
4136: 
4137:     Returns
4138:     -------
4139:     K : ndarray
4140:         Value of the modified Bessel function :math:`K_0` at `x`.
4141: 
4142:     Notes
4143:     -----
4144:     The range is partitioned into the two intervals [0, 2] and (2, infinity).
4145:     Chebyshev polynomial expansions are employed in each interval.
4146: 
4147:     This function is a wrapper for the Cephes [1]_ routine `k0`.
4148: 
4149:     See also
4150:     --------
4151:     kv
4152:     k0e
4153: 
4154:     References
4155:     ----------
4156:     .. [1] Cephes Mathematical Functions Library,
4157:            http://www.netlib.org/cephes/index.html
4158:     ''')
4159: 
4160: add_newdoc("scipy.special", "k0e",
4161:     '''
4162:     k0e(x)
4163: 
4164:     Exponentially scaled modified Bessel function K of order 0
4165: 
4166:     Defined as::
4167: 
4168:         k0e(x) = exp(x) * k0(x).
4169: 
4170:     Parameters
4171:     ----------
4172:     x : array_like
4173:         Argument (float)
4174: 
4175:     Returns
4176:     -------
4177:     K : ndarray
4178:         Value of the exponentially scaled modified Bessel function K of order
4179:         0 at `x`.
4180: 
4181:     Notes
4182:     -----
4183:     The range is partitioned into the two intervals [0, 2] and (2, infinity).
4184:     Chebyshev polynomial expansions are employed in each interval.
4185: 
4186:     This function is a wrapper for the Cephes [1]_ routine `k0e`.
4187: 
4188:     See also
4189:     --------
4190:     kv
4191:     k0
4192: 
4193:     References
4194:     ----------
4195:     .. [1] Cephes Mathematical Functions Library,
4196:            http://www.netlib.org/cephes/index.html
4197:     ''')
4198: 
4199: add_newdoc("scipy.special", "k1",
4200:     '''
4201:     k1(x)
4202: 
4203:     Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.
4204: 
4205:     Parameters
4206:     ----------
4207:     x : array_like
4208:         Argument (float)
4209: 
4210:     Returns
4211:     -------
4212:     K : ndarray
4213:         Value of the modified Bessel function K of order 1 at `x`.
4214: 
4215:     Notes
4216:     -----
4217:     The range is partitioned into the two intervals [0, 2] and (2, infinity).
4218:     Chebyshev polynomial expansions are employed in each interval.
4219: 
4220:     This function is a wrapper for the Cephes [1]_ routine `k1`.
4221: 
4222:     See also
4223:     --------
4224:     kv
4225:     k1e
4226: 
4227:     References
4228:     ----------
4229:     .. [1] Cephes Mathematical Functions Library,
4230:            http://www.netlib.org/cephes/index.html
4231:     ''')
4232: 
4233: add_newdoc("scipy.special", "k1e",
4234:     '''
4235:     k1e(x)
4236: 
4237:     Exponentially scaled modified Bessel function K of order 1
4238: 
4239:     Defined as::
4240: 
4241:         k1e(x) = exp(x) * k1(x)
4242: 
4243:     Parameters
4244:     ----------
4245:     x : array_like
4246:         Argument (float)
4247: 
4248:     Returns
4249:     -------
4250:     K : ndarray
4251:         Value of the exponentially scaled modified Bessel function K of order
4252:         1 at `x`.
4253: 
4254:     Notes
4255:     -----
4256:     The range is partitioned into the two intervals [0, 2] and (2, infinity).
4257:     Chebyshev polynomial expansions are employed in each interval.
4258: 
4259:     This function is a wrapper for the Cephes [1]_ routine `k1e`.
4260: 
4261:     See also
4262:     --------
4263:     kv
4264:     k1
4265: 
4266:     References
4267:     ----------
4268:     .. [1] Cephes Mathematical Functions Library,
4269:            http://www.netlib.org/cephes/index.html
4270:     ''')
4271: 
4272: add_newdoc("scipy.special", "kei",
4273:     '''
4274:     kei(x)
4275: 
4276:     Kelvin function ker
4277:     ''')
4278: 
4279: add_newdoc("scipy.special", "keip",
4280:     '''
4281:     keip(x)
4282: 
4283:     Derivative of the Kelvin function kei
4284:     ''')
4285: 
4286: add_newdoc("scipy.special", "kelvin",
4287:     '''
4288:     kelvin(x)
4289: 
4290:     Kelvin functions as complex numbers
4291: 
4292:     Returns
4293:     -------
4294:     Be, Ke, Bep, Kep
4295:         The tuple (Be, Ke, Bep, Kep) contains complex numbers
4296:         representing the real and imaginary Kelvin functions and their
4297:         derivatives evaluated at `x`.  For example, kelvin(x)[0].real =
4298:         ber x and kelvin(x)[0].imag = bei x with similar relationships
4299:         for ker and kei.
4300:     ''')
4301: 
4302: add_newdoc("scipy.special", "ker",
4303:     '''
4304:     ker(x)
4305: 
4306:     Kelvin function ker
4307:     ''')
4308: 
4309: add_newdoc("scipy.special", "kerp",
4310:     '''
4311:     kerp(x)
4312: 
4313:     Derivative of the Kelvin function ker
4314:     ''')
4315: 
4316: add_newdoc("scipy.special", "kl_div",
4317:     r'''
4318:     kl_div(x, y)
4319: 
4320:     Elementwise function for computing Kullback-Leibler divergence.
4321: 
4322:     .. math:: \mathrm{kl\_div}(x, y) = \begin{cases} x \log(x / y) - x + y & x > 0, y > 0 \\ y & x = 0, y \ge 0 \\ \infty & \text{otherwise} \end{cases}
4323: 
4324:     Parameters
4325:     ----------
4326:     x : ndarray
4327:         First input array.
4328:     y : ndarray
4329:         Second input array.
4330: 
4331:     Returns
4332:     -------
4333:     res : ndarray
4334:         Output array.
4335: 
4336:     See Also
4337:     --------
4338:     entr, rel_entr
4339: 
4340:     Notes
4341:     -----
4342:     This function is non-negative and is jointly convex in `x` and `y`.
4343: 
4344:     .. versionadded:: 0.15.0
4345: 
4346:     ''')
4347: 
4348: add_newdoc("scipy.special", "kn",
4349:     r'''
4350:     kn(n, x)
4351: 
4352:     Modified Bessel function of the second kind of integer order `n`
4353: 
4354:     Returns the modified Bessel function of the second kind for integer order
4355:     `n` at real `z`.
4356: 
4357:     These are also sometimes called functions of the third kind, Basset
4358:     functions, or Macdonald functions.
4359: 
4360:     Parameters
4361:     ----------
4362:     n : array_like of int
4363:         Order of Bessel functions (floats will truncate with a warning)
4364:     z : array_like of float
4365:         Argument at which to evaluate the Bessel functions
4366: 
4367:     Returns
4368:     -------
4369:     out : ndarray
4370:         The results
4371: 
4372:     Notes
4373:     -----
4374:     Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
4375:     algorithm used, see [2]_ and the references therein.
4376: 
4377:     See Also
4378:     --------
4379:     kv : Same function, but accepts real order and complex argument
4380:     kvp : Derivative of this function
4381: 
4382:     References
4383:     ----------
4384:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
4385:            of a Complex Argument and Nonnegative Order",
4386:            http://netlib.org/amos/
4387:     .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
4388:            functions of a complex argument and nonnegative order", ACM
4389:            TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
4390: 
4391:     Examples
4392:     --------
4393:     Plot the function of several orders for real input:
4394: 
4395:     >>> from scipy.special import kn
4396:     >>> import matplotlib.pyplot as plt
4397:     >>> x = np.linspace(0, 5, 1000)
4398:     >>> for N in range(6):
4399:     ...     plt.plot(x, kn(N, x), label='$K_{}(x)$'.format(N))
4400:     >>> plt.ylim(0, 10)
4401:     >>> plt.legend()
4402:     >>> plt.title(r'Modified Bessel function of the second kind $K_n(x)$')
4403:     >>> plt.show()
4404: 
4405:     Calculate for a single value at multiple orders:
4406: 
4407:     >>> kn([4, 5, 6], 1)
4408:     array([   44.23241585,   360.9605896 ,  3653.83831186])
4409:     ''')
4410: 
4411: add_newdoc("scipy.special", "kolmogi",
4412:     '''
4413:     kolmogi(p)
4414: 
4415:     Inverse function to kolmogorov
4416: 
4417:     Returns y such that ``kolmogorov(y) == p``.
4418:     ''')
4419: 
4420: add_newdoc("scipy.special", "kolmogorov",
4421:     '''
4422:     kolmogorov(y)
4423: 
4424:     Complementary cumulative distribution function of Kolmogorov distribution
4425: 
4426:     Returns the complementary cumulative distribution function of
4427:     Kolmogorov's limiting distribution (Kn* for large n) of a
4428:     two-sided test for equality between an empirical and a theoretical
4429:     distribution. It is equal to the (limit as n->infinity of the)
4430:     probability that sqrt(n) * max absolute deviation > y.
4431:     ''')
4432: 
4433: add_newdoc("scipy.special", "kv",
4434:     r'''
4435:     kv(v, z)
4436: 
4437:     Modified Bessel function of the second kind of real order `v`
4438: 
4439:     Returns the modified Bessel function of the second kind for real order
4440:     `v` at complex `z`.
4441: 
4442:     These are also sometimes called functions of the third kind, Basset
4443:     functions, or Macdonald functions.  They are defined as those solutions
4444:     of the modified Bessel equation for which,
4445: 
4446:     .. math::
4447:         K_v(x) \sim \sqrt{\pi/(2x)} \exp(-x)
4448: 
4449:     as :math:`x \to \infty` [3]_.
4450: 
4451:     Parameters
4452:     ----------
4453:     v : array_like of float
4454:         Order of Bessel functions
4455:     z : array_like of complex
4456:         Argument at which to evaluate the Bessel functions
4457: 
4458:     Returns
4459:     -------
4460:     out : ndarray
4461:         The results. Note that input must be of complex type to get complex
4462:         output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.
4463: 
4464:     Notes
4465:     -----
4466:     Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
4467:     algorithm used, see [2]_ and the references therein.
4468: 
4469:     See Also
4470:     --------
4471:     kve : This function with leading exponential behavior stripped off.
4472:     kvp : Derivative of this function
4473: 
4474:     References
4475:     ----------
4476:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
4477:            of a Complex Argument and Nonnegative Order",
4478:            http://netlib.org/amos/
4479:     .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
4480:            functions of a complex argument and nonnegative order", ACM
4481:            TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
4482:     .. [3] NIST Digital Library of Mathematical Functions,
4483:            Eq. 10.25.E3. http://dlmf.nist.gov/10.25.E3
4484: 
4485:     Examples
4486:     --------
4487:     Plot the function of several orders for real input:
4488: 
4489:     >>> from scipy.special import kv
4490:     >>> import matplotlib.pyplot as plt
4491:     >>> x = np.linspace(0, 5, 1000)
4492:     >>> for N in np.linspace(0, 6, 5):
4493:     ...     plt.plot(x, kv(N, x), label='$K_{{{}}}(x)$'.format(N))
4494:     >>> plt.ylim(0, 10)
4495:     >>> plt.legend()
4496:     >>> plt.title(r'Modified Bessel function of the second kind $K_\nu(x)$')
4497:     >>> plt.show()
4498: 
4499:     Calculate for a single value at multiple orders:
4500: 
4501:     >>> kv([4, 4.5, 5], 1+2j)
4502:     array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])
4503: 
4504:     ''')
4505: 
4506: add_newdoc("scipy.special", "kve",
4507:     r'''
4508:     kve(v, z)
4509: 
4510:     Exponentially scaled modified Bessel function of the second kind.
4511: 
4512:     Returns the exponentially scaled, modified Bessel function of the
4513:     second kind (sometimes called the third kind) for real order `v` at
4514:     complex `z`::
4515: 
4516:         kve(v, z) = kv(v, z) * exp(z)
4517: 
4518:     Parameters
4519:     ----------
4520:     v : array_like of float
4521:         Order of Bessel functions
4522:     z : array_like of complex
4523:         Argument at which to evaluate the Bessel functions
4524: 
4525:     Returns
4526:     -------
4527:     out : ndarray
4528:         The exponentially scaled modified Bessel function of the second kind.
4529: 
4530:     Notes
4531:     -----
4532:     Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the
4533:     algorithm used, see [2]_ and the references therein.
4534: 
4535:     References
4536:     ----------
4537:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
4538:            of a Complex Argument and Nonnegative Order",
4539:            http://netlib.org/amos/
4540:     .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel
4541:            functions of a complex argument and nonnegative order", ACM
4542:            TOMS Vol. 12 Issue 3, Sept. 1986, p. 265
4543:     ''')
4544: 
4545: add_newdoc("scipy.special", "_lanczos_sum_expg_scaled",
4546:     '''
4547:     Internal function, do not use.
4548:     ''')
4549: 
4550: add_newdoc("scipy.special", "_lgam1p",
4551:     '''
4552:     Internal function, do not use.
4553:     ''')
4554: 
4555: add_newdoc("scipy.special", "log1p",
4556:     '''
4557:     log1p(x)
4558: 
4559:     Calculates log(1+x) for use when `x` is near zero
4560:     ''')
4561: 
4562: add_newdoc("scipy.special", "_log1pmx",
4563:     '''
4564:     Internal function, do not use.
4565:     ''')
4566: 
4567: add_newdoc('scipy.special', 'logit',
4568:     '''
4569:     logit(x)
4570: 
4571:     Logit ufunc for ndarrays.
4572: 
4573:     The logit function is defined as logit(p) = log(p/(1-p)).
4574:     Note that logit(0) = -inf, logit(1) = inf, and logit(p)
4575:     for p<0 or p>1 yields nan.
4576: 
4577:     Parameters
4578:     ----------
4579:     x : ndarray
4580:         The ndarray to apply logit to element-wise.
4581: 
4582:     Returns
4583:     -------
4584:     out : ndarray
4585:         An ndarray of the same shape as x. Its entries
4586:         are logit of the corresponding entry of x.
4587: 
4588:     See Also
4589:     --------
4590:     expit
4591: 
4592:     Notes
4593:     -----
4594:     As a ufunc logit takes a number of optional
4595:     keyword arguments. For more information
4596:     see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_
4597: 
4598:     .. versionadded:: 0.10.0
4599: 
4600:     Examples
4601:     --------
4602:     >>> from scipy.special import logit, expit
4603: 
4604:     >>> logit([0, 0.25, 0.5, 0.75, 1])
4605:     array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])
4606: 
4607:     `expit` is the inverse of `logit`:
4608: 
4609:     >>> expit(logit([0.1, 0.75, 0.999]))
4610:     array([ 0.1  ,  0.75 ,  0.999])
4611: 
4612:     Plot logit(x) for x in [0, 1]:
4613: 
4614:     >>> import matplotlib.pyplot as plt
4615:     >>> x = np.linspace(0, 1, 501)
4616:     >>> y = logit(x)
4617:     >>> plt.plot(x, y)
4618:     >>> plt.grid()
4619:     >>> plt.ylim(-6, 6)
4620:     >>> plt.xlabel('x')
4621:     >>> plt.title('logit(x)')
4622:     >>> plt.show()
4623: 
4624:     ''')
4625: 
4626: add_newdoc("scipy.special", "lpmv",
4627:     r'''
4628:     lpmv(m, v, x)
4629: 
4630:     Associated Legendre function of integer order and real degree.
4631: 
4632:     Defined as
4633: 
4634:     .. math::
4635: 
4636:         P_v^m = (-1)^m (1 - x^2)^{m/2} \frac{d^m}{dx^m} P_v(x)
4637: 
4638:     where
4639: 
4640:     .. math::
4641: 
4642:         P_v = \sum_{k = 0}^\infty \frac{(-v)_k (v + 1)_k}{(k!)^2}
4643:                 \left(\frac{1 - x}{2}\right)^k
4644: 
4645:     is the Legendre function of the first kind. Here :math:`(\cdot)_k`
4646:     is the Pochhammer symbol; see `poch`.
4647: 
4648:     Parameters
4649:     ----------
4650:     m : array_like
4651:         Order (int or float). If passed a float not equal to an
4652:         integer the function returns NaN.
4653:     v : array_like
4654:         Degree (float).
4655:     x : array_like
4656:         Argument (float). Must have ``|x| <= 1``.
4657: 
4658:     Returns
4659:     -------
4660:     pmv : ndarray
4661:         Value of the associated Legendre function.
4662: 
4663:     See Also
4664:     --------
4665:     lpmn : Compute the associated Legendre function for all orders
4666:            ``0, ..., m`` and degrees ``0, ..., n``.
4667:     clpmn : Compute the associated Legendre function at complex
4668:             arguments.
4669: 
4670:     Notes
4671:     -----
4672:     Note that this implementation includes the Condon-Shortley phase.
4673: 
4674:     References
4675:     ----------
4676:     .. [1] Zhang, Jin, "Computation of Special Functions", John Wiley
4677:            and Sons, Inc, 1996.
4678: 
4679:     ''')
4680: 
4681: add_newdoc("scipy.special", "mathieu_a",
4682:     '''
4683:     mathieu_a(m, q)
4684: 
4685:     Characteristic value of even Mathieu functions
4686: 
4687:     Returns the characteristic value for the even solution,
4688:     ``ce_m(z, q)``, of Mathieu's equation.
4689:     ''')
4690: 
4691: add_newdoc("scipy.special", "mathieu_b",
4692:     '''
4693:     mathieu_b(m, q)
4694: 
4695:     Characteristic value of odd Mathieu functions
4696: 
4697:     Returns the characteristic value for the odd solution,
4698:     ``se_m(z, q)``, of Mathieu's equation.
4699:     ''')
4700: 
4701: add_newdoc("scipy.special", "mathieu_cem",
4702:     '''
4703:     mathieu_cem(m, q, x)
4704: 
4705:     Even Mathieu function and its derivative
4706: 
4707:     Returns the even Mathieu function, ``ce_m(x, q)``, of order `m` and
4708:     parameter `q` evaluated at `x` (given in degrees).  Also returns the
4709:     derivative with respect to `x` of ce_m(x, q)
4710: 
4711:     Parameters
4712:     ----------
4713:     m
4714:         Order of the function
4715:     q
4716:         Parameter of the function
4717:     x
4718:         Argument of the function, *given in degrees, not radians*
4719: 
4720:     Returns
4721:     -------
4722:     y
4723:         Value of the function
4724:     yp
4725:         Value of the derivative vs x
4726:     ''')
4727: 
4728: add_newdoc("scipy.special", "mathieu_modcem1",
4729:     '''
4730:     mathieu_modcem1(m, q, x)
4731: 
4732:     Even modified Mathieu function of the first kind and its derivative
4733: 
4734:     Evaluates the even modified Mathieu function of the first kind,
4735:     ``Mc1m(x, q)``, and its derivative at `x` for order `m` and parameter
4736:     `q`.
4737: 
4738:     Returns
4739:     -------
4740:     y
4741:         Value of the function
4742:     yp
4743:         Value of the derivative vs x
4744:     ''')
4745: 
4746: add_newdoc("scipy.special", "mathieu_modcem2",
4747:     '''
4748:     mathieu_modcem2(m, q, x)
4749: 
4750:     Even modified Mathieu function of the second kind and its derivative
4751: 
4752:     Evaluates the even modified Mathieu function of the second kind,
4753:     Mc2m(x, q), and its derivative at `x` (given in degrees) for order `m`
4754:     and parameter `q`.
4755: 
4756:     Returns
4757:     -------
4758:     y
4759:         Value of the function
4760:     yp
4761:         Value of the derivative vs x
4762:     ''')
4763: 
4764: add_newdoc("scipy.special", "mathieu_modsem1",
4765:     '''
4766:     mathieu_modsem1(m, q, x)
4767: 
4768:     Odd modified Mathieu function of the first kind and its derivative
4769: 
4770:     Evaluates the odd modified Mathieu function of the first kind,
4771:     Ms1m(x, q), and its derivative at `x` (given in degrees) for order `m`
4772:     and parameter `q`.
4773: 
4774:     Returns
4775:     -------
4776:     y
4777:         Value of the function
4778:     yp
4779:         Value of the derivative vs x
4780:     ''')
4781: 
4782: add_newdoc("scipy.special", "mathieu_modsem2",
4783:     '''
4784:     mathieu_modsem2(m, q, x)
4785: 
4786:     Odd modified Mathieu function of the second kind and its derivative
4787: 
4788:     Evaluates the odd modified Mathieu function of the second kind,
4789:     Ms2m(x, q), and its derivative at `x` (given in degrees) for order `m`
4790:     and parameter q.
4791: 
4792:     Returns
4793:     -------
4794:     y
4795:         Value of the function
4796:     yp
4797:         Value of the derivative vs x
4798:     ''')
4799: 
4800: add_newdoc("scipy.special", "mathieu_sem",
4801:     '''
4802:     mathieu_sem(m, q, x)
4803: 
4804:     Odd Mathieu function and its derivative
4805: 
4806:     Returns the odd Mathieu function, se_m(x, q), of order `m` and
4807:     parameter `q` evaluated at `x` (given in degrees).  Also returns the
4808:     derivative with respect to `x` of se_m(x, q).
4809: 
4810:     Parameters
4811:     ----------
4812:     m
4813:         Order of the function
4814:     q
4815:         Parameter of the function
4816:     x
4817:         Argument of the function, *given in degrees, not radians*.
4818: 
4819:     Returns
4820:     -------
4821:     y
4822:         Value of the function
4823:     yp
4824:         Value of the derivative vs x
4825:     ''')
4826: 
4827: add_newdoc("scipy.special", "modfresnelm",
4828:     '''
4829:     modfresnelm(x)
4830: 
4831:     Modified Fresnel negative integrals
4832: 
4833:     Returns
4834:     -------
4835:     fm
4836:         Integral ``F_-(x)``: ``integral(exp(-1j*t*t), t=x..inf)``
4837:     km
4838:         Integral ``K_-(x)``: ``1/sqrt(pi)*exp(1j*(x*x+pi/4))*fp``
4839:     ''')
4840: 
4841: add_newdoc("scipy.special", "modfresnelp",
4842:     '''
4843:     modfresnelp(x)
4844: 
4845:     Modified Fresnel positive integrals
4846: 
4847:     Returns
4848:     -------
4849:     fp
4850:         Integral ``F_+(x)``: ``integral(exp(1j*t*t), t=x..inf)``
4851:     kp
4852:         Integral ``K_+(x)``: ``1/sqrt(pi)*exp(-1j*(x*x+pi/4))*fp``
4853:     ''')
4854: 
4855: add_newdoc("scipy.special", "modstruve",
4856:     r'''
4857:     modstruve(v, x)
4858: 
4859:     Modified Struve function.
4860: 
4861:     Return the value of the modified Struve function of order `v` at `x`.  The
4862:     modified Struve function is defined as,
4863: 
4864:     .. math::
4865:         L_v(x) = -\imath \exp(-\pi\imath v/2) H_v(x),
4866: 
4867:     where :math:`H_v` is the Struve function.
4868: 
4869:     Parameters
4870:     ----------
4871:     v : array_like
4872:         Order of the modified Struve function (float).
4873:     x : array_like
4874:         Argument of the Struve function (float; must be positive unless `v` is
4875:         an integer).
4876: 
4877:     Returns
4878:     -------
4879:     L : ndarray
4880:         Value of the modified Struve function of order `v` at `x`.
4881: 
4882:     Notes
4883:     -----
4884:     Three methods discussed in [1]_ are used to evaluate the function:
4885: 
4886:     - power series
4887:     - expansion in Bessel functions (if :math:`|z| < |v| + 20`)
4888:     - asymptotic large-z expansion (if :math:`z \geq 0.7v + 12`)
4889: 
4890:     Rounding errors are estimated based on the largest terms in the sums, and
4891:     the result associated with the smallest error is returned.
4892: 
4893:     See also
4894:     --------
4895:     struve
4896: 
4897:     References
4898:     ----------
4899:     .. [1] NIST Digital Library of Mathematical Functions
4900:            http://dlmf.nist.gov/11
4901:     ''')
4902: 
4903: add_newdoc("scipy.special", "nbdtr",
4904:     r'''
4905:     nbdtr(k, n, p)
4906: 
4907:     Negative binomial cumulative distribution function.
4908: 
4909:     Returns the sum of the terms 0 through `k` of the negative binomial
4910:     distribution probability mass function,
4911: 
4912:     .. math::
4913: 
4914:         F = \sum_{j=0}^k {{n + j - 1}\choose{j}} p^n (1 - p)^j.
4915: 
4916:     In a sequence of Bernoulli trials with individual success probabilities
4917:     `p`, this is the probability that `k` or fewer failures precede the nth
4918:     success.
4919: 
4920:     Parameters
4921:     ----------
4922:     k : array_like
4923:         The maximum number of allowed failures (nonnegative int).
4924:     n : array_like
4925:         The target number of successes (positive int).
4926:     p : array_like
4927:         Probability of success in a single event (float).
4928: 
4929:     Returns
4930:     -------
4931:     F : ndarray
4932:         The probability of `k` or fewer failures before `n` successes in a
4933:         sequence of events with individual success probability `p`.
4934: 
4935:     See also
4936:     --------
4937:     nbdtrc
4938: 
4939:     Notes
4940:     -----
4941:     If floating point values are passed for `k` or `n`, they will be truncated
4942:     to integers.
4943: 
4944:     The terms are not summed directly; instead the regularized incomplete beta
4945:     function is employed, according to the formula,
4946: 
4947:     .. math::
4948:         \mathrm{nbdtr}(k, n, p) = I_{p}(n, k + 1).
4949: 
4950:     Wrapper for the Cephes [1]_ routine `nbdtr`.
4951: 
4952:     References
4953:     ----------
4954:     .. [1] Cephes Mathematical Functions Library,
4955:            http://www.netlib.org/cephes/index.html
4956: 
4957:     ''')
4958: 
4959: add_newdoc("scipy.special", "nbdtrc",
4960:     r'''
4961:     nbdtrc(k, n, p)
4962: 
4963:     Negative binomial survival function.
4964: 
4965:     Returns the sum of the terms `k + 1` to infinity of the negative binomial
4966:     distribution probability mass function,
4967: 
4968:     .. math::
4969: 
4970:         F = \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j.
4971: 
4972:     In a sequence of Bernoulli trials with individual success probabilities
4973:     `p`, this is the probability that more than `k` failures precede the nth
4974:     success.
4975: 
4976:     Parameters
4977:     ----------
4978:     k : array_like
4979:         The maximum number of allowed failures (nonnegative int).
4980:     n : array_like
4981:         The target number of successes (positive int).
4982:     p : array_like
4983:         Probability of success in a single event (float).
4984: 
4985:     Returns
4986:     -------
4987:     F : ndarray
4988:         The probability of `k + 1` or more failures before `n` successes in a
4989:         sequence of events with individual success probability `p`.
4990: 
4991:     Notes
4992:     -----
4993:     If floating point values are passed for `k` or `n`, they will be truncated
4994:     to integers.
4995: 
4996:     The terms are not summed directly; instead the regularized incomplete beta
4997:     function is employed, according to the formula,
4998: 
4999:     .. math::
5000:         \mathrm{nbdtrc}(k, n, p) = I_{1 - p}(k + 1, n).
5001: 
5002:     Wrapper for the Cephes [1]_ routine `nbdtrc`.
5003: 
5004:     References
5005:     ----------
5006:     .. [1] Cephes Mathematical Functions Library,
5007:            http://www.netlib.org/cephes/index.html
5008:     ''')
5009: 
5010: add_newdoc("scipy.special", "nbdtri",
5011:     '''
5012:     nbdtri(k, n, y)
5013: 
5014:     Inverse of `nbdtr` vs `p`.
5015: 
5016:     Returns the inverse with respect to the parameter `p` of
5017:     `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
5018:     function.
5019: 
5020:     Parameters
5021:     ----------
5022:     k : array_like
5023:         The maximum number of allowed failures (nonnegative int).
5024:     n : array_like
5025:         The target number of successes (positive int).
5026:     y : array_like
5027:         The probability of `k` or fewer failures before `n` successes (float).
5028: 
5029:     Returns
5030:     -------
5031:     p : ndarray
5032:         Probability of success in a single event (float) such that
5033:         `nbdtr(k, n, p) = y`.
5034: 
5035:     See also
5036:     --------
5037:     nbdtr : Cumulative distribution function of the negative binomial.
5038:     nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.
5039:     nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
5040: 
5041:     Notes
5042:     -----
5043:     Wrapper for the Cephes [1]_ routine `nbdtri`.
5044: 
5045:     References
5046:     ----------
5047:     .. [1] Cephes Mathematical Functions Library,
5048:            http://www.netlib.org/cephes/index.html
5049: 
5050:     ''')
5051: 
5052: add_newdoc("scipy.special", "nbdtrik",
5053:     r'''
5054:     nbdtrik(y, n, p)
5055: 
5056:     Inverse of `nbdtr` vs `k`.
5057: 
5058:     Returns the inverse with respect to the parameter `k` of
5059:     `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
5060:     function.
5061: 
5062:     Parameters
5063:     ----------
5064:     y : array_like
5065:         The probability of `k` or fewer failures before `n` successes (float).
5066:     n : array_like
5067:         The target number of successes (positive int).
5068:     p : array_like
5069:         Probability of success in a single event (float).
5070: 
5071:     Returns
5072:     -------
5073:     k : ndarray
5074:         The maximum number of allowed failures such that `nbdtr(k, n, p) = y`.
5075: 
5076:     See also
5077:     --------
5078:     nbdtr : Cumulative distribution function of the negative binomial.
5079:     nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
5080:     nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.
5081: 
5082:     Notes
5083:     -----
5084:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.
5085: 
5086:     Formula 26.5.26 of [2]_,
5087: 
5088:     .. math::
5089:         \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),
5090: 
5091:     is used to reduce calculation of the cumulative distribution function to
5092:     that of a regularized incomplete beta :math:`I`.
5093: 
5094:     Computation of `k` involves a search for a value that produces the desired
5095:     value of `y`.  The search relies on the monotonicity of `y` with `k`.
5096: 
5097:     References
5098:     ----------
5099:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
5100:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
5101:            Functions, Inverses, and Other Parameters.
5102:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
5103:            Handbook of Mathematical Functions with Formulas,
5104:            Graphs, and Mathematical Tables. New York: Dover, 1972.
5105: 
5106:     ''')
5107: 
5108: add_newdoc("scipy.special", "nbdtrin",
5109:     r'''
5110:     nbdtrin(k, y, p)
5111: 
5112:     Inverse of `nbdtr` vs `n`.
5113: 
5114:     Returns the inverse with respect to the parameter `n` of
5115:     `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution
5116:     function.
5117: 
5118:     Parameters
5119:     ----------
5120:     k : array_like
5121:         The maximum number of allowed failures (nonnegative int).
5122:     y : array_like
5123:         The probability of `k` or fewer failures before `n` successes (float).
5124:     p : array_like
5125:         Probability of success in a single event (float).
5126: 
5127:     Returns
5128:     -------
5129:     n : ndarray
5130:         The number of successes `n` such that `nbdtr(k, n, p) = y`.
5131: 
5132:     See also
5133:     --------
5134:     nbdtr : Cumulative distribution function of the negative binomial.
5135:     nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.
5136:     nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.
5137: 
5138:     Notes
5139:     -----
5140:     Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.
5141: 
5142:     Formula 26.5.26 of [2]_,
5143: 
5144:     .. math::
5145:         \sum_{j=k + 1}^\infty {{n + j - 1}\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),
5146: 
5147:     is used to reduce calculation of the cumulative distribution function to
5148:     that of a regularized incomplete beta :math:`I`.
5149: 
5150:     Computation of `n` involves a search for a value that produces the desired
5151:     value of `y`.  The search relies on the monotonicity of `y` with `n`.
5152: 
5153:     References
5154:     ----------
5155:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
5156:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
5157:            Functions, Inverses, and Other Parameters.
5158:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
5159:            Handbook of Mathematical Functions with Formulas,
5160:            Graphs, and Mathematical Tables. New York: Dover, 1972.
5161: 
5162:     ''')
5163: 
5164: add_newdoc("scipy.special", "ncfdtr",
5165:     r'''
5166:     ncfdtr(dfn, dfd, nc, f)
5167: 
5168:     Cumulative distribution function of the non-central F distribution.
5169: 
5170:     The non-central F describes the distribution of,
5171: 
5172:     .. math::
5173:         Z = \frac{X/d_n}{Y/d_d}
5174: 
5175:     where :math:`X` and :math:`Y` are independently distributed, with
5176:     :math:`X` distributed non-central :math:`\chi^2` with noncentrality
5177:     parameter `nc` and :math:`d_n` degrees of freedom, and :math:`Y`
5178:     distributed :math:`\chi^2` with :math:`d_d` degrees of freedom.
5179: 
5180:     Parameters
5181:     ----------
5182:     dfn : array_like
5183:         Degrees of freedom of the numerator sum of squares.  Range (0, inf).
5184:     dfd : array_like
5185:         Degrees of freedom of the denominator sum of squares.  Range (0, inf).
5186:     nc : array_like
5187:         Noncentrality parameter.  Should be in range (0, 1e4).
5188:     f : array_like
5189:         Quantiles, i.e. the upper limit of integration.
5190: 
5191:     Returns
5192:     -------
5193:     cdf : float or ndarray
5194:         The calculated CDF.  If all inputs are scalar, the return will be a
5195:         float.  Otherwise it will be an array.
5196: 
5197:     See Also
5198:     --------
5199:     ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
5200:     ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
5201:     ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
5202:     ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.
5203: 
5204:     Notes
5205:     -----
5206:     Wrapper for the CDFLIB [1]_ Fortran routine `cdffnc`.
5207: 
5208:     The cumulative distribution function is computed using Formula 26.6.20 of
5209:     [2]_:
5210: 
5211:     .. math::
5212:         F(d_n, d_d, n_c, f) = \sum_{j=0}^\infty e^{-n_c/2} \frac{(n_c/2)^j}{j!} I_{x}(\frac{d_n}{2} + j, \frac{d_d}{2}),
5213: 
5214:     where :math:`I` is the regularized incomplete beta function, and
5215:     :math:`x = f d_n/(f d_n + d_d)`.
5216: 
5217:     The computation time required for this routine is proportional to the
5218:     noncentrality parameter `nc`.  Very large values of this parameter can
5219:     consume immense computer resources.  This is why the search range is
5220:     bounded by 10,000.
5221: 
5222:     References
5223:     ----------
5224:     .. [1] Barry Brown, James Lovato, and Kathy Russell,
5225:            CDFLIB: Library of Fortran Routines for Cumulative Distribution
5226:            Functions, Inverses, and Other Parameters.
5227:     .. [2] Milton Abramowitz and Irene A. Stegun, eds.
5228:            Handbook of Mathematical Functions with Formulas,
5229:            Graphs, and Mathematical Tables. New York: Dover, 1972.
5230: 
5231:     Examples
5232:     --------
5233:     >>> from scipy import special
5234:     >>> from scipy import stats
5235:     >>> import matplotlib.pyplot as plt
5236: 
5237:     Plot the CDF of the non-central F distribution, for nc=0.  Compare with the
5238:     F-distribution from scipy.stats:
5239: 
5240:     >>> x = np.linspace(-1, 8, num=500)
5241:     >>> dfn = 3
5242:     >>> dfd = 2
5243:     >>> ncf_stats = stats.f.cdf(x, dfn, dfd)
5244:     >>> ncf_special = special.ncfdtr(dfn, dfd, 0, x)
5245: 
5246:     >>> fig = plt.figure()
5247:     >>> ax = fig.add_subplot(111)
5248:     >>> ax.plot(x, ncf_stats, 'b-', lw=3)
5249:     >>> ax.plot(x, ncf_special, 'r-')
5250:     >>> plt.show()
5251: 
5252:     ''')
5253: 
5254: add_newdoc("scipy.special", "ncfdtri",
5255:     '''
5256:     ncfdtri(dfn, dfd, nc, p)
5257: 
5258:     Inverse with respect to `f` of the CDF of the non-central F distribution.
5259: 
5260:     See `ncfdtr` for more details.
5261: 
5262:     Parameters
5263:     ----------
5264:     dfn : array_like
5265:         Degrees of freedom of the numerator sum of squares.  Range (0, inf).
5266:     dfd : array_like
5267:         Degrees of freedom of the denominator sum of squares.  Range (0, inf).
5268:     nc : array_like
5269:         Noncentrality parameter.  Should be in range (0, 1e4).
5270:     p : array_like
5271:         Value of the cumulative distribution function.  Must be in the
5272:         range [0, 1].
5273: 
5274:     Returns
5275:     -------
5276:     f : float
5277:         Quantiles, i.e. the upper limit of integration.
5278: 
5279:     See Also
5280:     --------
5281:     ncfdtr : CDF of the non-central F distribution.
5282:     ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
5283:     ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
5284:     ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.
5285: 
5286:     Examples
5287:     --------
5288:     >>> from scipy.special import ncfdtr, ncfdtri
5289: 
5290:     Compute the CDF for several values of `f`:
5291: 
5292:     >>> f = [0.5, 1, 1.5]
5293:     >>> p = ncfdtr(2, 3, 1.5, f)
5294:     >>> p
5295:     array([ 0.20782291,  0.36107392,  0.47345752])
5296: 
5297:     Compute the inverse.  We recover the values of `f`, as expected:
5298: 
5299:     >>> ncfdtri(2, 3, 1.5, p)
5300:     array([ 0.5,  1. ,  1.5])
5301: 
5302:     ''')
5303: 
5304: add_newdoc("scipy.special", "ncfdtridfd",
5305:     '''
5306:     ncfdtridfd(dfn, p, nc, f)
5307: 
5308:     Calculate degrees of freedom (denominator) for the noncentral F-distribution.
5309: 
5310:     This is the inverse with respect to `dfd` of `ncfdtr`.
5311:     See `ncfdtr` for more details.
5312: 
5313:     Parameters
5314:     ----------
5315:     dfn : array_like
5316:         Degrees of freedom of the numerator sum of squares.  Range (0, inf).
5317:     p : array_like
5318:         Value of the cumulative distribution function.  Must be in the
5319:         range [0, 1].
5320:     nc : array_like
5321:         Noncentrality parameter.  Should be in range (0, 1e4).
5322:     f : array_like
5323:         Quantiles, i.e. the upper limit of integration.
5324: 
5325:     Returns
5326:     -------
5327:     dfd : float
5328:         Degrees of freedom of the denominator sum of squares.
5329: 
5330:     See Also
5331:     --------
5332:     ncfdtr : CDF of the non-central F distribution.
5333:     ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
5334:     ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
5335:     ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.
5336: 
5337:     Notes
5338:     -----
5339:     The value of the cumulative noncentral F distribution is not necessarily
5340:     monotone in either degrees of freedom.  There thus may be two values that
5341:     provide a given CDF value.  This routine assumes monotonicity and will
5342:     find an arbitrary one of the two values.
5343: 
5344:     Examples
5345:     --------
5346:     >>> from scipy.special import ncfdtr, ncfdtridfd
5347: 
5348:     Compute the CDF for several values of `dfd`:
5349: 
5350:     >>> dfd = [1, 2, 3]
5351:     >>> p = ncfdtr(2, dfd, 0.25, 15)
5352:     >>> p
5353:     array([ 0.8097138 ,  0.93020416,  0.96787852])
5354: 
5355:     Compute the inverse.  We recover the values of `dfd`, as expected:
5356: 
5357:     >>> ncfdtridfd(2, p, 0.25, 15)
5358:     array([ 1.,  2.,  3.])
5359: 
5360:     ''')
5361: 
5362: add_newdoc("scipy.special", "ncfdtridfn",
5363:     '''
5364:     ncfdtridfn(p, dfd, nc, f)
5365: 
5366:     Calculate degrees of freedom (numerator) for the noncentral F-distribution.
5367: 
5368:     This is the inverse with respect to `dfn` of `ncfdtr`.
5369:     See `ncfdtr` for more details.
5370: 
5371:     Parameters
5372:     ----------
5373:     p : array_like
5374:         Value of the cumulative distribution function.  Must be in the
5375:         range [0, 1].
5376:     dfd : array_like
5377:         Degrees of freedom of the denominator sum of squares.  Range (0, inf).
5378:     nc : array_like
5379:         Noncentrality parameter.  Should be in range (0, 1e4).
5380:     f : float
5381:         Quantiles, i.e. the upper limit of integration.
5382: 
5383:     Returns
5384:     -------
5385:     dfn : float
5386:         Degrees of freedom of the numerator sum of squares.
5387: 
5388:     See Also
5389:     --------
5390:     ncfdtr : CDF of the non-central F distribution.
5391:     ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
5392:     ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
5393:     ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.
5394: 
5395:     Notes
5396:     -----
5397:     The value of the cumulative noncentral F distribution is not necessarily
5398:     monotone in either degrees of freedom.  There thus may be two values that
5399:     provide a given CDF value.  This routine assumes monotonicity and will
5400:     find an arbitrary one of the two values.
5401: 
5402:     Examples
5403:     --------
5404:     >>> from scipy.special import ncfdtr, ncfdtridfn
5405: 
5406:     Compute the CDF for several values of `dfn`:
5407: 
5408:     >>> dfn = [1, 2, 3]
5409:     >>> p = ncfdtr(dfn, 2, 0.25, 15)
5410:     >>> p
5411:     array([ 0.92562363,  0.93020416,  0.93188394])
5412: 
5413:     Compute the inverse.  We recover the values of `dfn`, as expected:
5414: 
5415:     >>> ncfdtridfn(p, 2, 0.25, 15)
5416:     array([ 1.,  2.,  3.])
5417: 
5418:     ''')
5419: 
5420: add_newdoc("scipy.special", "ncfdtrinc",
5421:     '''
5422:     ncfdtrinc(dfn, dfd, p, f)
5423: 
5424:     Calculate non-centrality parameter for non-central F distribution.
5425: 
5426:     This is the inverse with respect to `nc` of `ncfdtr`.
5427:     See `ncfdtr` for more details.
5428: 
5429:     Parameters
5430:     ----------
5431:     dfn : array_like
5432:         Degrees of freedom of the numerator sum of squares.  Range (0, inf).
5433:     dfd : array_like
5434:         Degrees of freedom of the denominator sum of squares.  Range (0, inf).
5435:     p : array_like
5436:         Value of the cumulative distribution function.  Must be in the
5437:         range [0, 1].
5438:     f : array_like
5439:         Quantiles, i.e. the upper limit of integration.
5440: 
5441:     Returns
5442:     -------
5443:     nc : float
5444:         Noncentrality parameter.
5445: 
5446:     See Also
5447:     --------
5448:     ncfdtr : CDF of the non-central F distribution.
5449:     ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.
5450:     ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.
5451:     ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.
5452: 
5453:     Examples
5454:     --------
5455:     >>> from scipy.special import ncfdtr, ncfdtrinc
5456: 
5457:     Compute the CDF for several values of `nc`:
5458: 
5459:     >>> nc = [0.5, 1.5, 2.0]
5460:     >>> p = ncfdtr(2, 3, nc, 15)
5461:     >>> p
5462:     array([ 0.96309246,  0.94327955,  0.93304098])
5463: 
5464:     Compute the inverse.  We recover the values of `nc`, as expected:
5465: 
5466:     >>> ncfdtrinc(2, 3, p, 15)
5467:     array([ 0.5,  1.5,  2. ])
5468: 
5469:     ''')
5470: 
5471: add_newdoc("scipy.special", "nctdtr",
5472:     '''
5473:     nctdtr(df, nc, t)
5474: 
5475:     Cumulative distribution function of the non-central `t` distribution.
5476: 
5477:     Parameters
5478:     ----------
5479:     df : array_like
5480:         Degrees of freedom of the distribution.  Should be in range (0, inf).
5481:     nc : array_like
5482:         Noncentrality parameter.  Should be in range (-1e6, 1e6).
5483:     t : array_like
5484:         Quantiles, i.e. the upper limit of integration.
5485: 
5486:     Returns
5487:     -------
5488:     cdf : float or ndarray
5489:         The calculated CDF.  If all inputs are scalar, the return will be a
5490:         float.  Otherwise it will be an array.
5491: 
5492:     See Also
5493:     --------
5494:     nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.
5495:     nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.
5496:     nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.
5497: 
5498:     Examples
5499:     --------
5500:     >>> from scipy import special
5501:     >>> from scipy import stats
5502:     >>> import matplotlib.pyplot as plt
5503: 
5504:     Plot the CDF of the non-central t distribution, for nc=0.  Compare with the
5505:     t-distribution from scipy.stats:
5506: 
5507:     >>> x = np.linspace(-5, 5, num=500)
5508:     >>> df = 3
5509:     >>> nct_stats = stats.t.cdf(x, df)
5510:     >>> nct_special = special.nctdtr(df, 0, x)
5511: 
5512:     >>> fig = plt.figure()
5513:     >>> ax = fig.add_subplot(111)
5514:     >>> ax.plot(x, nct_stats, 'b-', lw=3)
5515:     >>> ax.plot(x, nct_special, 'r-')
5516:     >>> plt.show()
5517: 
5518:     ''')
5519: 
5520: add_newdoc("scipy.special", "nctdtridf",
5521:     '''
5522:     nctdtridf(p, nc, t)
5523: 
5524:     Calculate degrees of freedom for non-central t distribution.
5525: 
5526:     See `nctdtr` for more details.
5527: 
5528:     Parameters
5529:     ----------
5530:     p : array_like
5531:         CDF values, in range (0, 1].
5532:     nc : array_like
5533:         Noncentrality parameter.  Should be in range (-1e6, 1e6).
5534:     t : array_like
5535:         Quantiles, i.e. the upper limit of integration.
5536: 
5537:     ''')
5538: 
5539: add_newdoc("scipy.special", "nctdtrinc",
5540:     '''
5541:     nctdtrinc(df, p, t)
5542: 
5543:     Calculate non-centrality parameter for non-central t distribution.
5544: 
5545:     See `nctdtr` for more details.
5546: 
5547:     Parameters
5548:     ----------
5549:     df : array_like
5550:         Degrees of freedom of the distribution.  Should be in range (0, inf).
5551:     p : array_like
5552:         CDF values, in range (0, 1].
5553:     t : array_like
5554:         Quantiles, i.e. the upper limit of integration.
5555: 
5556:     ''')
5557: 
5558: add_newdoc("scipy.special", "nctdtrit",
5559:     '''
5560:     nctdtrit(df, nc, p)
5561: 
5562:     Inverse cumulative distribution function of the non-central t distribution.
5563: 
5564:     See `nctdtr` for more details.
5565: 
5566:     Parameters
5567:     ----------
5568:     df : array_like
5569:         Degrees of freedom of the distribution.  Should be in range (0, inf).
5570:     nc : array_like
5571:         Noncentrality parameter.  Should be in range (-1e6, 1e6).
5572:     p : array_like
5573:         CDF values, in range (0, 1].
5574: 
5575:     ''')
5576: 
5577: add_newdoc("scipy.special", "ndtr",
5578:     r'''
5579:     ndtr(x)
5580: 
5581:     Gaussian cumulative distribution function.
5582: 
5583:     Returns the area under the standard Gaussian probability
5584:     density function, integrated from minus infinity to `x`
5585: 
5586:     .. math::
5587: 
5588:        \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x \exp(-t^2/2) dt
5589: 
5590:     Parameters
5591:     ----------
5592:     x : array_like, real or complex
5593:         Argument
5594: 
5595:     Returns
5596:     -------
5597:     ndarray
5598:         The value of the normal CDF evaluated at `x`
5599: 
5600:     See Also
5601:     --------
5602:     erf
5603:     erfc
5604:     scipy.stats.norm
5605:     log_ndtr
5606: 
5607:     ''')
5608: 
5609: 
5610: add_newdoc("scipy.special", "nrdtrimn",
5611:     '''
5612:     nrdtrimn(p, x, std)
5613: 
5614:     Calculate mean of normal distribution given other params.
5615: 
5616:     Parameters
5617:     ----------
5618:     p : array_like
5619:         CDF values, in range (0, 1].
5620:     x : array_like
5621:         Quantiles, i.e. the upper limit of integration.
5622:     std : array_like
5623:         Standard deviation.
5624: 
5625:     Returns
5626:     -------
5627:     mn : float or ndarray
5628:         The mean of the normal distribution.
5629: 
5630:     See Also
5631:     --------
5632:     nrdtrimn, ndtr
5633: 
5634:     ''')
5635: 
5636: add_newdoc("scipy.special", "nrdtrisd",
5637:     '''
5638:     nrdtrisd(p, x, mn)
5639: 
5640:     Calculate standard deviation of normal distribution given other params.
5641: 
5642:     Parameters
5643:     ----------
5644:     p : array_like
5645:         CDF values, in range (0, 1].
5646:     x : array_like
5647:         Quantiles, i.e. the upper limit of integration.
5648:     mn : float or ndarray
5649:         The mean of the normal distribution.
5650: 
5651:     Returns
5652:     -------
5653:     std : array_like
5654:         Standard deviation.
5655: 
5656:     See Also
5657:     --------
5658:     nrdtristd, ndtr
5659: 
5660:     ''')
5661: 
5662: add_newdoc("scipy.special", "log_ndtr",
5663:     '''
5664:     log_ndtr(x)
5665: 
5666:     Logarithm of Gaussian cumulative distribution function.
5667: 
5668:     Returns the log of the area under the standard Gaussian probability
5669:     density function, integrated from minus infinity to `x`::
5670: 
5671:         log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))
5672: 
5673:     Parameters
5674:     ----------
5675:     x : array_like, real or complex
5676:         Argument
5677: 
5678:     Returns
5679:     -------
5680:     ndarray
5681:         The value of the log of the normal CDF evaluated at `x`
5682: 
5683:     See Also
5684:     --------
5685:     erf
5686:     erfc
5687:     scipy.stats.norm
5688:     ndtr
5689: 
5690:     ''')
5691: 
5692: add_newdoc("scipy.special", "ndtri",
5693:     '''
5694:     ndtri(y)
5695: 
5696:     Inverse of `ndtr` vs x
5697: 
5698:     Returns the argument x for which the area under the Gaussian
5699:     probability density function (integrated from minus infinity to `x`)
5700:     is equal to y.
5701:     ''')
5702: 
5703: add_newdoc("scipy.special", "obl_ang1",
5704:     '''
5705:     obl_ang1(m, n, c, x)
5706: 
5707:     Oblate spheroidal angular function of the first kind and its derivative
5708: 
5709:     Computes the oblate spheroidal angular function of the first kind
5710:     and its derivative (with respect to `x`) for mode parameters m>=0
5711:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
5712: 
5713:     Returns
5714:     -------
5715:     s
5716:         Value of the function
5717:     sp
5718:         Value of the derivative vs x
5719:     ''')
5720: 
5721: add_newdoc("scipy.special", "obl_ang1_cv",
5722:     '''
5723:     obl_ang1_cv(m, n, c, cv, x)
5724: 
5725:     Oblate spheroidal angular function obl_ang1 for precomputed characteristic value
5726: 
5727:     Computes the oblate spheroidal angular function of the first kind
5728:     and its derivative (with respect to `x`) for mode parameters m>=0
5729:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
5730:     pre-computed characteristic value.
5731: 
5732:     Returns
5733:     -------
5734:     s
5735:         Value of the function
5736:     sp
5737:         Value of the derivative vs x
5738:     ''')
5739: 
5740: add_newdoc("scipy.special", "obl_cv",
5741:     '''
5742:     obl_cv(m, n, c)
5743: 
5744:     Characteristic value of oblate spheroidal function
5745: 
5746:     Computes the characteristic value of oblate spheroidal wave
5747:     functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.
5748:     ''')
5749: 
5750: add_newdoc("scipy.special", "obl_rad1",
5751:     '''
5752:     obl_rad1(m, n, c, x)
5753: 
5754:     Oblate spheroidal radial function of the first kind and its derivative
5755: 
5756:     Computes the oblate spheroidal radial function of the first kind
5757:     and its derivative (with respect to `x`) for mode parameters m>=0
5758:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
5759: 
5760:     Returns
5761:     -------
5762:     s
5763:         Value of the function
5764:     sp
5765:         Value of the derivative vs x
5766:     ''')
5767: 
5768: add_newdoc("scipy.special", "obl_rad1_cv",
5769:     '''
5770:     obl_rad1_cv(m, n, c, cv, x)
5771: 
5772:     Oblate spheroidal radial function obl_rad1 for precomputed characteristic value
5773: 
5774:     Computes the oblate spheroidal radial function of the first kind
5775:     and its derivative (with respect to `x`) for mode parameters m>=0
5776:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
5777:     pre-computed characteristic value.
5778: 
5779:     Returns
5780:     -------
5781:     s
5782:         Value of the function
5783:     sp
5784:         Value of the derivative vs x
5785:     ''')
5786: 
5787: add_newdoc("scipy.special", "obl_rad2",
5788:     '''
5789:     obl_rad2(m, n, c, x)
5790: 
5791:     Oblate spheroidal radial function of the second kind and its derivative.
5792: 
5793:     Computes the oblate spheroidal radial function of the second kind
5794:     and its derivative (with respect to `x`) for mode parameters m>=0
5795:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
5796: 
5797:     Returns
5798:     -------
5799:     s
5800:         Value of the function
5801:     sp
5802:         Value of the derivative vs x
5803:     ''')
5804: 
5805: add_newdoc("scipy.special", "obl_rad2_cv",
5806:     '''
5807:     obl_rad2_cv(m, n, c, cv, x)
5808: 
5809:     Oblate spheroidal radial function obl_rad2 for precomputed characteristic value
5810: 
5811:     Computes the oblate spheroidal radial function of the second kind
5812:     and its derivative (with respect to `x`) for mode parameters m>=0
5813:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
5814:     pre-computed characteristic value.
5815: 
5816:     Returns
5817:     -------
5818:     s
5819:         Value of the function
5820:     sp
5821:         Value of the derivative vs x
5822:     ''')
5823: 
5824: add_newdoc("scipy.special", "pbdv",
5825:     '''
5826:     pbdv(v, x)
5827: 
5828:     Parabolic cylinder function D
5829: 
5830:     Returns (d, dp) the parabolic cylinder function Dv(x) in d and the
5831:     derivative, Dv'(x) in dp.
5832: 
5833:     Returns
5834:     -------
5835:     d
5836:         Value of the function
5837:     dp
5838:         Value of the derivative vs x
5839:     ''')
5840: 
5841: add_newdoc("scipy.special", "pbvv",
5842:     '''
5843:     pbvv(v, x)
5844: 
5845:     Parabolic cylinder function V
5846: 
5847:     Returns the parabolic cylinder function Vv(x) in v and the
5848:     derivative, Vv'(x) in vp.
5849: 
5850:     Returns
5851:     -------
5852:     v
5853:         Value of the function
5854:     vp
5855:         Value of the derivative vs x
5856:     ''')
5857: 
5858: add_newdoc("scipy.special", "pbwa",
5859:     r'''
5860:     pbwa(a, x)
5861: 
5862:     Parabolic cylinder function W.
5863: 
5864:     The function is a particular solution to the differential equation
5865: 
5866:     .. math::
5867: 
5868:         y'' + \left(\frac{1}{4}x^2 - a\right)y = 0,
5869: 
5870:     for a full definition see section 12.14 in [1]_.
5871: 
5872:     Parameters
5873:     ----------
5874:     a : array_like
5875:         Real parameter
5876:     x : array_like
5877:         Real argument
5878: 
5879:     Returns
5880:     -------
5881:     w : scalar or ndarray
5882:         Value of the function
5883:     wp : scalar or ndarray
5884:         Value of the derivative in x
5885: 
5886:     Notes
5887:     -----
5888:     The function is a wrapper for a Fortran routine by Zhang and Jin
5889:     [2]_. The implementation is accurate only for ``|a|, |x| < 5`` and
5890:     returns NaN outside that range.
5891: 
5892:     References
5893:     ----------
5894:     .. [1] Digital Library of Mathematical Functions, 14.30.
5895:            http://dlmf.nist.gov/14.30
5896:     .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
5897:            Functions", John Wiley and Sons, 1996.
5898:            https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html
5899:     ''')
5900: 
5901: add_newdoc("scipy.special", "pdtr",
5902:     '''
5903:     pdtr(k, m)
5904: 
5905:     Poisson cumulative distribution function
5906: 
5907:     Returns the sum of the first `k` terms of the Poisson distribution:
5908:     sum(exp(-m) * m**j / j!, j=0..k) = gammaincc( k+1, m).  Arguments
5909:     must both be positive and `k` an integer.
5910:     ''')
5911: 
5912: add_newdoc("scipy.special", "pdtrc",
5913:     '''
5914:     pdtrc(k, m)
5915: 
5916:     Poisson survival function
5917: 
5918:     Returns the sum of the terms from k+1 to infinity of the Poisson
5919:     distribution: sum(exp(-m) * m**j / j!, j=k+1..inf) = gammainc(
5920:     k+1, m).  Arguments must both be positive and `k` an integer.
5921:     ''')
5922: 
5923: add_newdoc("scipy.special", "pdtri",
5924:     '''
5925:     pdtri(k, y)
5926: 
5927:     Inverse to `pdtr` vs m
5928: 
5929:     Returns the Poisson variable `m` such that the sum from 0 to `k` of
5930:     the Poisson density is equal to the given probability `y`:
5931:     calculated by gammaincinv(k+1, y). `k` must be a nonnegative
5932:     integer and `y` between 0 and 1.
5933:     ''')
5934: 
5935: add_newdoc("scipy.special", "pdtrik",
5936:     '''
5937:     pdtrik(p, m)
5938: 
5939:     Inverse to `pdtr` vs k
5940: 
5941:     Returns the quantile k such that ``pdtr(k, m) = p``
5942:     ''')
5943: 
5944: add_newdoc("scipy.special", "poch",
5945:     r'''
5946:     poch(z, m)
5947: 
5948:     Rising factorial (z)_m
5949: 
5950:     The Pochhammer symbol (rising factorial), is defined as
5951: 
5952:     .. math::
5953: 
5954:         (z)_m = \frac{\Gamma(z + m)}{\Gamma(z)}
5955: 
5956:     For positive integer `m` it reads
5957: 
5958:     .. math::
5959: 
5960:         (z)_m = z (z + 1) ... (z + m - 1)
5961: 
5962:     Parameters
5963:     ----------
5964:     z : array_like
5965:         (int or float)
5966:     m : array_like
5967:         (int or float)
5968: 
5969:     Returns
5970:     -------
5971:     poch : ndarray
5972:         The value of the function.
5973:     ''')
5974: 
5975: add_newdoc("scipy.special", "pro_ang1",
5976:     '''
5977:     pro_ang1(m, n, c, x)
5978: 
5979:     Prolate spheroidal angular function of the first kind and its derivative
5980: 
5981:     Computes the prolate spheroidal angular function of the first kind
5982:     and its derivative (with respect to `x`) for mode parameters m>=0
5983:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
5984: 
5985:     Returns
5986:     -------
5987:     s
5988:         Value of the function
5989:     sp
5990:         Value of the derivative vs x
5991:     ''')
5992: 
5993: add_newdoc("scipy.special", "pro_ang1_cv",
5994:     '''
5995:     pro_ang1_cv(m, n, c, cv, x)
5996: 
5997:     Prolate spheroidal angular function pro_ang1 for precomputed characteristic value
5998: 
5999:     Computes the prolate spheroidal angular function of the first kind
6000:     and its derivative (with respect to `x`) for mode parameters m>=0
6001:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
6002:     pre-computed characteristic value.
6003: 
6004:     Returns
6005:     -------
6006:     s
6007:         Value of the function
6008:     sp
6009:         Value of the derivative vs x
6010:     ''')
6011: 
6012: add_newdoc("scipy.special", "pro_cv",
6013:     '''
6014:     pro_cv(m, n, c)
6015: 
6016:     Characteristic value of prolate spheroidal function
6017: 
6018:     Computes the characteristic value of prolate spheroidal wave
6019:     functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.
6020:     ''')
6021: 
6022: add_newdoc("scipy.special", "pro_rad1",
6023:     '''
6024:     pro_rad1(m, n, c, x)
6025: 
6026:     Prolate spheroidal radial function of the first kind and its derivative
6027: 
6028:     Computes the prolate spheroidal radial function of the first kind
6029:     and its derivative (with respect to `x`) for mode parameters m>=0
6030:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
6031: 
6032:     Returns
6033:     -------
6034:     s
6035:         Value of the function
6036:     sp
6037:         Value of the derivative vs x
6038:     ''')
6039: 
6040: add_newdoc("scipy.special", "pro_rad1_cv",
6041:     '''
6042:     pro_rad1_cv(m, n, c, cv, x)
6043: 
6044:     Prolate spheroidal radial function pro_rad1 for precomputed characteristic value
6045: 
6046:     Computes the prolate spheroidal radial function of the first kind
6047:     and its derivative (with respect to `x`) for mode parameters m>=0
6048:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
6049:     pre-computed characteristic value.
6050: 
6051:     Returns
6052:     -------
6053:     s
6054:         Value of the function
6055:     sp
6056:         Value of the derivative vs x
6057:     ''')
6058: 
6059: add_newdoc("scipy.special", "pro_rad2",
6060:     '''
6061:     pro_rad2(m, n, c, x)
6062: 
6063:     Prolate spheroidal radial function of the second kind and its derivative
6064: 
6065:     Computes the prolate spheroidal radial function of the second kind
6066:     and its derivative (with respect to `x`) for mode parameters m>=0
6067:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.
6068: 
6069:     Returns
6070:     -------
6071:     s
6072:         Value of the function
6073:     sp
6074:         Value of the derivative vs x
6075:     ''')
6076: 
6077: add_newdoc("scipy.special", "pro_rad2_cv",
6078:     '''
6079:     pro_rad2_cv(m, n, c, cv, x)
6080: 
6081:     Prolate spheroidal radial function pro_rad2 for precomputed characteristic value
6082: 
6083:     Computes the prolate spheroidal radial function of the second kind
6084:     and its derivative (with respect to `x`) for mode parameters m>=0
6085:     and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires
6086:     pre-computed characteristic value.
6087: 
6088:     Returns
6089:     -------
6090:     s
6091:         Value of the function
6092:     sp
6093:         Value of the derivative vs x
6094:     ''')
6095: 
6096: add_newdoc("scipy.special", "pseudo_huber",
6097:     r'''
6098:     pseudo_huber(delta, r)
6099: 
6100:     Pseudo-Huber loss function.
6101: 
6102:     .. math:: \mathrm{pseudo\_huber}(\delta, r) = \delta^2 \left( \sqrt{ 1 + \left( \frac{r}{\delta} \right)^2 } - 1 \right)
6103: 
6104:     Parameters
6105:     ----------
6106:     delta : ndarray
6107:         Input array, indicating the soft quadratic vs. linear loss changepoint.
6108:     r : ndarray
6109:         Input array, possibly representing residuals.
6110: 
6111:     Returns
6112:     -------
6113:     res : ndarray
6114:         The computed Pseudo-Huber loss function values.
6115: 
6116:     Notes
6117:     -----
6118:     This function is convex in :math:`r`.
6119: 
6120:     .. versionadded:: 0.15.0
6121: 
6122:     ''')
6123: 
6124: add_newdoc("scipy.special", "psi",
6125:     '''
6126:     psi(z, out=None)
6127: 
6128:     The digamma function.
6129: 
6130:     The logarithmic derivative of the gamma function evaluated at ``z``.
6131: 
6132:     Parameters
6133:     ----------
6134:     z : array_like
6135:         Real or complex argument.
6136:     out : ndarray, optional
6137:         Array for the computed values of ``psi``.
6138: 
6139:     Returns
6140:     -------
6141:     digamma : ndarray
6142:         Computed values of ``psi``.
6143: 
6144:     Notes
6145:     -----
6146:     For large values not close to the negative real axis ``psi`` is
6147:     computed using the asymptotic series (5.11.2) from [1]_. For small
6148:     arguments not close to the negative real axis the recurrence
6149:     relation (5.5.2) from [1]_ is used until the argument is large
6150:     enough to use the asymptotic series. For values close to the
6151:     negative real axis the reflection formula (5.5.4) from [1]_ is
6152:     used first.  Note that ``psi`` has a family of zeros on the
6153:     negative real axis which occur between the poles at nonpositive
6154:     integers. Around the zeros the reflection formula suffers from
6155:     cancellation and the implementation loses precision. The sole
6156:     positive zero and the first negative zero, however, are handled
6157:     separately by precomputing series expansions using [2]_, so the
6158:     function should maintain full accuracy around the origin.
6159: 
6160:     References
6161:     ----------
6162:     .. [1] NIST Digital Library of Mathematical Functions
6163:            http://dlmf.nist.gov/5
6164:     .. [2] Fredrik Johansson and others.
6165:            "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
6166:            (Version 0.19) http://mpmath.org/
6167: 
6168:     ''')
6169: 
6170: add_newdoc("scipy.special", "radian",
6171:     '''
6172:     radian(d, m, s)
6173: 
6174:     Convert from degrees to radians
6175: 
6176:     Returns the angle given in (d)egrees, (m)inutes, and (s)econds in
6177:     radians.
6178:     ''')
6179: 
6180: add_newdoc("scipy.special", "rel_entr",
6181:     r'''
6182:     rel_entr(x, y)
6183: 
6184:     Elementwise function for computing relative entropy.
6185: 
6186:     .. math:: \mathrm{rel\_entr}(x, y) = \begin{cases} x \log(x / y) & x > 0, y > 0 \\ 0 & x = 0, y \ge 0 \\ \infty & \text{otherwise} \end{cases}
6187: 
6188:     Parameters
6189:     ----------
6190:     x : ndarray
6191:         First input array.
6192:     y : ndarray
6193:         Second input array.
6194: 
6195:     Returns
6196:     -------
6197:     res : ndarray
6198:         Output array.
6199: 
6200:     See Also
6201:     --------
6202:     entr, kl_div
6203: 
6204:     Notes
6205:     -----
6206:     This function is jointly convex in x and y.
6207: 
6208:     .. versionadded:: 0.15.0
6209: 
6210:     ''')
6211: 
6212: add_newdoc("scipy.special", "rgamma",
6213:     '''
6214:     rgamma(z)
6215: 
6216:     Gamma function inverted
6217: 
6218:     Returns ``1/gamma(x)``
6219:     ''')
6220: 
6221: add_newdoc("scipy.special", "round",
6222:     '''
6223:     round(x)
6224: 
6225:     Round to nearest integer
6226: 
6227:     Returns the nearest integer to `x` as a double precision floating
6228:     point result.  If `x` ends in 0.5 exactly, the nearest even integer
6229:     is chosen.
6230:     ''')
6231: 
6232: add_newdoc("scipy.special", "shichi",
6233:     r'''
6234:     shichi(x, out=None)
6235: 
6236:     Hyperbolic sine and cosine integrals.
6237: 
6238:     The hyperbolic sine integral is
6239: 
6240:     .. math::
6241: 
6242:       \int_0^x \frac{\sinh{t}}{t}dt
6243: 
6244:     and the hyperbolic cosine integral is
6245: 
6246:     .. math::
6247: 
6248:       \gamma + \log(x) + \int_0^x \frac{\cosh{t} - 1}{t} dt
6249: 
6250:     where :math:`\gamma` is Euler's constant and :math:`\log` is the
6251:     principle branch of the logarithm.
6252: 
6253:     Parameters
6254:     ----------
6255:     x : array_like
6256:         Real or complex points at which to compute the hyperbolic sine
6257:         and cosine integrals.
6258: 
6259:     Returns
6260:     -------
6261:     si : ndarray
6262:         Hyperbolic sine integral at ``x``
6263:     ci : ndarray
6264:         Hyperbolic cosine integral at ``x``
6265: 
6266:     Notes
6267:     -----
6268:     For real arguments with ``x < 0``, ``chi`` is the real part of the
6269:     hyperbolic cosine integral. For such points ``chi(x)`` and ``chi(x
6270:     + 0j)`` differ by a factor of ``1j*pi``.
6271: 
6272:     For real arguments the function is computed by calling Cephes'
6273:     [1]_ *shichi* routine. For complex arguments the algorithm is based
6274:     on Mpmath's [2]_ *shi* and *chi* routines.
6275: 
6276:     References
6277:     ----------
6278:     .. [1] Cephes Mathematical Functions Library,
6279:            http://www.netlib.org/cephes/index.html
6280:     .. [2] Fredrik Johansson and others.
6281:            "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
6282:            (Version 0.19) http://mpmath.org/
6283:     ''')
6284: 
6285: add_newdoc("scipy.special", "sici",
6286:     r'''
6287:     sici(x, out=None)
6288: 
6289:     Sine and cosine integrals.
6290: 
6291:     The sine integral is
6292: 
6293:     .. math::
6294: 
6295:       \int_0^x \frac{\sin{t}}{t}dt
6296: 
6297:     and the cosine integral is
6298: 
6299:     .. math::
6300: 
6301:       \gamma + \log(x) + \int_0^x \frac{\cos{t} - 1}{t}dt
6302: 
6303:     where :math:`\gamma` is Euler's constant and :math:`\log` is the
6304:     principle branch of the logarithm.
6305: 
6306:     Parameters
6307:     ----------
6308:     x : array_like
6309:         Real or complex points at which to compute the sine and cosine
6310:         integrals.
6311: 
6312:     Returns
6313:     -------
6314:     si : ndarray
6315:         Sine integral at ``x``
6316:     ci : ndarray
6317:         Cosine integral at ``x``
6318: 
6319:     Notes
6320:     -----
6321:     For real arguments with ``x < 0``, ``ci`` is the real part of the
6322:     cosine integral. For such points ``ci(x)`` and ``ci(x + 0j)``
6323:     differ by a factor of ``1j*pi``.
6324: 
6325:     For real arguments the function is computed by calling Cephes'
6326:     [1]_ *sici* routine. For complex arguments the algorithm is based
6327:     on Mpmath's [2]_ *si* and *ci* routines.
6328: 
6329:     References
6330:     ----------
6331:     .. [1] Cephes Mathematical Functions Library,
6332:            http://www.netlib.org/cephes/index.html
6333:     .. [2] Fredrik Johansson and others.
6334:            "mpmath: a Python library for arbitrary-precision floating-point arithmetic"
6335:            (Version 0.19) http://mpmath.org/
6336:     ''')
6337: 
6338: add_newdoc("scipy.special", "sindg",
6339:     '''
6340:     sindg(x)
6341: 
6342:     Sine of angle given in degrees
6343:     ''')
6344: 
6345: add_newdoc("scipy.special", "smirnov",
6346:     '''
6347:     smirnov(n, e)
6348: 
6349:     Kolmogorov-Smirnov complementary cumulative distribution function
6350: 
6351:     Returns the exact Kolmogorov-Smirnov complementary cumulative
6352:     distribution function (Dn+ or Dn-) for a one-sided test of
6353:     equality between an empirical and a theoretical distribution. It
6354:     is equal to the probability that the maximum difference between a
6355:     theoretical distribution and an empirical one based on `n` samples
6356:     is greater than e.
6357:     ''')
6358: 
6359: add_newdoc("scipy.special", "smirnovi",
6360:     '''
6361:     smirnovi(n, y)
6362: 
6363:     Inverse to `smirnov`
6364: 
6365:     Returns ``e`` such that ``smirnov(n, e) = y``.
6366:     ''')
6367: 
6368: add_newdoc("scipy.special", "spence",
6369:     r'''
6370:     spence(z, out=None)
6371: 
6372:     Spence's function, also known as the dilogarithm.
6373: 
6374:     It is defined to be
6375: 
6376:     .. math::
6377:       \int_0^z \frac{\log(t)}{1 - t}dt
6378: 
6379:     for complex :math:`z`, where the contour of integration is taken
6380:     to avoid the branch cut of the logarithm. Spence's function is
6381:     analytic everywhere except the negative real axis where it has a
6382:     branch cut.
6383: 
6384:     Parameters
6385:     ----------
6386:     z : array_like
6387:         Points at which to evaluate Spence's function
6388: 
6389:     Returns
6390:     -------
6391:     s : ndarray
6392:         Computed values of Spence's function
6393: 
6394:     Notes
6395:     -----
6396:     There is a different convention which defines Spence's function by
6397:     the integral
6398: 
6399:     .. math::
6400:       -\int_0^z \frac{\log(1 - t)}{t}dt;
6401: 
6402:     this is our ``spence(1 - z)``.
6403:     ''')
6404: 
6405: add_newdoc("scipy.special", "stdtr",
6406:     '''
6407:     stdtr(df, t)
6408: 
6409:     Student t distribution cumulative density function
6410: 
6411:     Returns the integral from minus infinity to t of the Student t
6412:     distribution with df > 0 degrees of freedom::
6413: 
6414:        gamma((df+1)/2)/(sqrt(df*pi)*gamma(df/2)) *
6415:        integral((1+x**2/df)**(-df/2-1/2), x=-inf..t)
6416: 
6417:     ''')
6418: 
6419: add_newdoc("scipy.special", "stdtridf",
6420:     '''
6421:     stdtridf(p, t)
6422: 
6423:     Inverse of `stdtr` vs df
6424: 
6425:     Returns the argument df such that stdtr(df, t) is equal to `p`.
6426:     ''')
6427: 
6428: add_newdoc("scipy.special", "stdtrit",
6429:     '''
6430:     stdtrit(df, p)
6431: 
6432:     Inverse of `stdtr` vs `t`
6433: 
6434:     Returns the argument `t` such that stdtr(df, t) is equal to `p`.
6435:     ''')
6436: 
6437: add_newdoc("scipy.special", "struve",
6438:     r'''
6439:     struve(v, x)
6440: 
6441:     Struve function.
6442: 
6443:     Return the value of the Struve function of order `v` at `x`.  The Struve
6444:     function is defined as,
6445: 
6446:     .. math::
6447:         H_v(x) = (z/2)^{v + 1} \sum_{n=0}^\infty \frac{(-1)^n (z/2)^{2n}}{\Gamma(n + \frac{3}{2}) \Gamma(n + v + \frac{3}{2})},
6448: 
6449:     where :math:`\Gamma` is the gamma function.
6450: 
6451:     Parameters
6452:     ----------
6453:     v : array_like
6454:         Order of the Struve function (float).
6455:     x : array_like
6456:         Argument of the Struve function (float; must be positive unless `v` is
6457:         an integer).
6458: 
6459:     Returns
6460:     -------
6461:     H : ndarray
6462:         Value of the Struve function of order `v` at `x`.
6463: 
6464:     Notes
6465:     -----
6466:     Three methods discussed in [1]_ are used to evaluate the Struve function:
6467: 
6468:     - power series
6469:     - expansion in Bessel functions (if :math:`|z| < |v| + 20`)
6470:     - asymptotic large-z expansion (if :math:`z \geq 0.7v + 12`)
6471: 
6472:     Rounding errors are estimated based on the largest terms in the sums, and
6473:     the result associated with the smallest error is returned.
6474: 
6475:     See also
6476:     --------
6477:     modstruve
6478: 
6479:     References
6480:     ----------
6481:     .. [1] NIST Digital Library of Mathematical Functions
6482:            http://dlmf.nist.gov/11
6483: 
6484:     ''')
6485: 
6486: add_newdoc("scipy.special", "tandg",
6487:     '''
6488:     tandg(x)
6489: 
6490:     Tangent of angle x given in degrees.
6491:     ''')
6492: 
6493: add_newdoc("scipy.special", "tklmbda",
6494:     '''
6495:     tklmbda(x, lmbda)
6496: 
6497:     Tukey-Lambda cumulative distribution function
6498: 
6499:     ''')
6500: 
6501: add_newdoc("scipy.special", "wofz",
6502:     '''
6503:     wofz(z)
6504: 
6505:     Faddeeva function
6506: 
6507:     Returns the value of the Faddeeva function for complex argument::
6508: 
6509:         exp(-z**2) * erfc(-i*z)
6510: 
6511:     See Also
6512:     --------
6513:     dawsn, erf, erfc, erfcx, erfi
6514: 
6515:     References
6516:     ----------
6517:     .. [1] Steven G. Johnson, Faddeeva W function implementation.
6518:        http://ab-initio.mit.edu/Faddeeva
6519: 
6520:     Examples
6521:     --------
6522:     >>> from scipy import special
6523:     >>> import matplotlib.pyplot as plt
6524: 
6525:     >>> x = np.linspace(-3, 3)
6526:     >>> z = special.wofz(x)
6527: 
6528:     >>> plt.plot(x, z.real, label='wofz(x).real')
6529:     >>> plt.plot(x, z.imag, label='wofz(x).imag')
6530:     >>> plt.xlabel('$x$')
6531:     >>> plt.legend(framealpha=1, shadow=True)
6532:     >>> plt.grid(alpha=0.25)
6533:     >>> plt.show()
6534: 
6535:     ''')
6536: 
6537: add_newdoc("scipy.special", "xlogy",
6538:     '''
6539:     xlogy(x, y)
6540: 
6541:     Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.
6542: 
6543:     Parameters
6544:     ----------
6545:     x : array_like
6546:         Multiplier
6547:     y : array_like
6548:         Argument
6549: 
6550:     Returns
6551:     -------
6552:     z : array_like
6553:         Computed x*log(y)
6554: 
6555:     Notes
6556:     -----
6557: 
6558:     .. versionadded:: 0.13.0
6559: 
6560:     ''')
6561: 
6562: add_newdoc("scipy.special", "xlog1py",
6563:     '''
6564:     xlog1py(x, y)
6565: 
6566:     Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.
6567: 
6568:     Parameters
6569:     ----------
6570:     x : array_like
6571:         Multiplier
6572:     y : array_like
6573:         Argument
6574: 
6575:     Returns
6576:     -------
6577:     z : array_like
6578:         Computed x*log1p(y)
6579: 
6580:     Notes
6581:     -----
6582: 
6583:     .. versionadded:: 0.13.0
6584: 
6585:     ''')
6586: 
6587: add_newdoc("scipy.special", "y0",
6588:     r'''
6589:     y0(x)
6590: 
6591:     Bessel function of the second kind of order 0.
6592: 
6593:     Parameters
6594:     ----------
6595:     x : array_like
6596:         Argument (float).
6597: 
6598:     Returns
6599:     -------
6600:     Y : ndarray
6601:         Value of the Bessel function of the second kind of order 0 at `x`.
6602: 
6603:     Notes
6604:     -----
6605: 
6606:     The domain is divided into the intervals [0, 5] and (5, infinity). In the
6607:     first interval a rational approximation :math:`R(x)` is employed to
6608:     compute,
6609: 
6610:     .. math::
6611: 
6612:         Y_0(x) = R(x) + \frac{2 \log(x) J_0(x)}{\pi},
6613: 
6614:     where :math:`J_0` is the Bessel function of the first kind of order 0.
6615: 
6616:     In the second interval, the Hankel asymptotic expansion is employed with
6617:     two rational functions of degree 6/6 and 7/7.
6618: 
6619:     This function is a wrapper for the Cephes [1]_ routine `y0`.
6620: 
6621:     See also
6622:     --------
6623:     j0
6624:     yv
6625: 
6626:     References
6627:     ----------
6628:     .. [1] Cephes Mathematical Functions Library,
6629:            http://www.netlib.org/cephes/index.html
6630:     ''')
6631: 
6632: add_newdoc("scipy.special", "y1",
6633:     '''
6634:     y1(x)
6635: 
6636:     Bessel function of the second kind of order 1.
6637: 
6638:     Parameters
6639:     ----------
6640:     x : array_like
6641:         Argument (float).
6642: 
6643:     Returns
6644:     -------
6645:     Y : ndarray
6646:         Value of the Bessel function of the second kind of order 1 at `x`.
6647: 
6648:     Notes
6649:     -----
6650: 
6651:     The domain is divided into the intervals [0, 8] and (8, infinity). In the
6652:     first interval a 25 term Chebyshev expansion is used, and computing
6653:     :math:`J_1` (the Bessel function of the first kind) is required. In the
6654:     second, the asymptotic trigonometric representation is employed using two
6655:     rational functions of degree 5/5.
6656: 
6657:     This function is a wrapper for the Cephes [1]_ routine `y1`.
6658: 
6659:     See also
6660:     --------
6661:     j1
6662:     yn
6663:     yv
6664: 
6665:     References
6666:     ----------
6667:     .. [1] Cephes Mathematical Functions Library,
6668:            http://www.netlib.org/cephes/index.html
6669:     ''')
6670: 
6671: add_newdoc("scipy.special", "yn",
6672:     r'''
6673:     yn(n, x)
6674: 
6675:     Bessel function of the second kind of integer order and real argument.
6676: 
6677:     Parameters
6678:     ----------
6679:     n : array_like
6680:         Order (integer).
6681:     z : array_like
6682:         Argument (float).
6683: 
6684:     Returns
6685:     -------
6686:     Y : ndarray
6687:         Value of the Bessel function, :math:`Y_n(x)`.
6688: 
6689:     Notes
6690:     -----
6691:     Wrapper for the Cephes [1]_ routine `yn`.
6692: 
6693:     The function is evaluated by forward recurrence on `n`, starting with
6694:     values computed by the Cephes routines `y0` and `y1`. If `n = 0` or 1,
6695:     the routine for `y0` or `y1` is called directly.
6696: 
6697:     See also
6698:     --------
6699:     yv : For real order and real or complex argument.
6700: 
6701:     References
6702:     ----------
6703:     .. [1] Cephes Mathematical Functions Library,
6704:            http://www.netlib.org/cephes/index.html
6705:     ''')
6706: 
6707: add_newdoc("scipy.special", "yv",
6708:     r'''
6709:     yv(v, z)
6710: 
6711:     Bessel function of the second kind of real order and complex argument.
6712: 
6713:     Parameters
6714:     ----------
6715:     v : array_like
6716:         Order (float).
6717:     z : array_like
6718:         Argument (float or complex).
6719: 
6720:     Returns
6721:     -------
6722:     Y : ndarray
6723:         Value of the Bessel function of the second kind, :math:`Y_v(x)`.
6724: 
6725:     Notes
6726:     -----
6727:     For positive `v` values, the computation is carried out using the
6728:     AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
6729:     Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,
6730: 
6731:     .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).
6732: 
6733:     For negative `v` values the formula,
6734: 
6735:     .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)
6736: 
6737:     is used, where :math:`J_v(z)` is the Bessel function of the first kind,
6738:     computed using the AMOS routine `zbesj`.  Note that the second term is
6739:     exactly zero for integer `v`; to improve accuracy the second term is
6740:     explicitly omitted for `v` values such that `v = floor(v)`.
6741: 
6742:     See also
6743:     --------
6744:     yve : :math:`Y_v` with leading exponential behavior stripped off.
6745: 
6746:     References
6747:     ----------
6748:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
6749:            of a Complex Argument and Nonnegative Order",
6750:            http://netlib.org/amos/
6751: 
6752:     ''')
6753: 
6754: add_newdoc("scipy.special", "yve",
6755:     r'''
6756:     yve(v, z)
6757: 
6758:     Exponentially scaled Bessel function of the second kind of real order.
6759: 
6760:     Returns the exponentially scaled Bessel function of the second
6761:     kind of real order `v` at complex `z`::
6762: 
6763:         yve(v, z) = yv(v, z) * exp(-abs(z.imag))
6764: 
6765:     Parameters
6766:     ----------
6767:     v : array_like
6768:         Order (float).
6769:     z : array_like
6770:         Argument (float or complex).
6771: 
6772:     Returns
6773:     -------
6774:     Y : ndarray
6775:         Value of the exponentially scaled Bessel function.
6776: 
6777:     Notes
6778:     -----
6779:     For positive `v` values, the computation is carried out using the
6780:     AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel
6781:     Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,
6782: 
6783:     .. math:: Y_v(z) = \frac{1}{2\imath} (H_v^{(1)} - H_v^{(2)}).
6784: 
6785:     For negative `v` values the formula,
6786: 
6787:     .. math:: Y_{-v}(z) = Y_v(z) \cos(\pi v) + J_v(z) \sin(\pi v)
6788: 
6789:     is used, where :math:`J_v(z)` is the Bessel function of the first kind,
6790:     computed using the AMOS routine `zbesj`.  Note that the second term is
6791:     exactly zero for integer `v`; to improve accuracy the second term is
6792:     explicitly omitted for `v` values such that `v = floor(v)`.
6793: 
6794:     References
6795:     ----------
6796:     .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions
6797:            of a Complex Argument and Nonnegative Order",
6798:            http://netlib.org/amos/
6799:     ''')
6800: 
6801: add_newdoc("scipy.special", "_zeta",
6802:     '''
6803:     _zeta(x, q)
6804: 
6805:     Internal function, Hurwitz zeta.
6806: 
6807:     ''')
6808: 
6809: add_newdoc("scipy.special", "zetac",
6810:     '''
6811:     zetac(x)
6812: 
6813:     Riemann zeta function minus 1.
6814: 
6815:     This function is defined as
6816: 
6817:     .. math:: \\zeta(x) = \\sum_{k=2}^{\\infty} 1 / k^x,
6818: 
6819:     where ``x > 1``.
6820: 
6821:     See Also
6822:     --------
6823:     zeta
6824: 
6825:     ''')
6826: 
6827: add_newdoc("scipy.special", "_struve_asymp_large_z",
6828:     '''
6829:     _struve_asymp_large_z(v, z, is_h)
6830: 
6831:     Internal function for testing `struve` & `modstruve`
6832: 
6833:     Evaluates using asymptotic expansion
6834: 
6835:     Returns
6836:     -------
6837:     v, err
6838:     ''')
6839: 
6840: add_newdoc("scipy.special", "_struve_power_series",
6841:     '''
6842:     _struve_power_series(v, z, is_h)
6843: 
6844:     Internal function for testing `struve` & `modstruve`
6845: 
6846:     Evaluates using power series
6847: 
6848:     Returns
6849:     -------
6850:     v, err
6851:     ''')
6852: 
6853: add_newdoc("scipy.special", "_struve_bessel_series",
6854:     '''
6855:     _struve_bessel_series(v, z, is_h)
6856: 
6857:     Internal function for testing `struve` & `modstruve`
6858: 
6859:     Evaluates using Bessel function series
6860: 
6861:     Returns
6862:     -------
6863:     v, err
6864:     ''')
6865: 
6866: add_newdoc("scipy.special", "_spherical_jn",
6867:     '''
6868:     Internal function, use `spherical_jn` instead.
6869:     ''')
6870: 
6871: add_newdoc("scipy.special", "_spherical_jn_d",
6872:     '''
6873:     Internal function, use `spherical_jn` instead.
6874:     ''')
6875: 
6876: add_newdoc("scipy.special", "_spherical_yn",
6877:     '''
6878:     Internal function, use `spherical_yn` instead.
6879:     ''')
6880: 
6881: add_newdoc("scipy.special", "_spherical_yn_d",
6882:     '''
6883:     Internal function, use `spherical_yn` instead.
6884:     ''')
6885: 
6886: add_newdoc("scipy.special", "_spherical_in",
6887:     '''
6888:     Internal function, use `spherical_in` instead.
6889:     ''')
6890: 
6891: add_newdoc("scipy.special", "_spherical_in_d",
6892:     '''
6893:     Internal function, use `spherical_in` instead.
6894:     ''')
6895: 
6896: add_newdoc("scipy.special", "_spherical_kn",
6897:     '''
6898:     Internal function, use `spherical_kn` instead.
6899:     ''')
6900: 
6901: add_newdoc("scipy.special", "_spherical_kn_d",
6902:     '''
6903:     Internal function, use `spherical_kn` instead.
6904:     ''')
6905: 
6906: add_newdoc("scipy.special", "loggamma",
6907:     r'''
6908:     loggamma(z, out=None)
6909: 
6910:     Principal branch of the logarithm of the Gamma function.
6911: 
6912:     Defined to be :math:`\log(\Gamma(x))` for :math:`x > 0` and
6913:     extended to the complex plane by analytic continuation. The
6914:     function has a single branch cut on the negative real axis.
6915: 
6916:     .. versionadded:: 0.18.0
6917: 
6918:     Parameters
6919:     ----------
6920:     z : array-like
6921:         Values in the complex plain at which to compute ``loggamma``
6922:     out : ndarray, optional
6923:         Output array for computed values of ``loggamma``
6924: 
6925:     Returns
6926:     -------
6927:     loggamma : ndarray
6928:         Values of ``loggamma`` at z.
6929: 
6930:     Notes
6931:     -----
6932:     It is not generally true that :math:`\log\Gamma(z) =
6933:     \log(\Gamma(z))`, though the real parts of the functions do
6934:     agree. The benefit of not defining ``loggamma`` as
6935:     :math:`\log(\Gamma(z))` is that the latter function has a
6936:     complicated branch cut structure whereas ``loggamma`` is analytic
6937:     except for on the negative real axis.
6938: 
6939:     The identities
6940: 
6941:     .. math::
6942:       \exp(\log\Gamma(z)) &= \Gamma(z) \\
6943:       \log\Gamma(z + 1) &= \log(z) + \log\Gamma(z)
6944: 
6945:     make ``loggama`` useful for working in complex logspace. However,
6946:     ``loggamma`` necessarily returns complex outputs for real inputs,
6947:     so if you want to work only with real numbers use `gammaln`. On
6948:     the real line the two functions are related by ``exp(loggamma(x))
6949:     = gammasgn(x)*exp(gammaln(x))``, though in practice rounding
6950:     errors will introduce small spurious imaginary components in
6951:     ``exp(loggamma(x))``.
6952: 
6953:     The implementation here is based on [hare1997]_.
6954: 
6955:     See also
6956:     --------
6957:     gammaln : logarithm of the absolute value of the Gamma function
6958:     gammasgn : sign of the gamma function
6959: 
6960:     References
6961:     ----------
6962:     .. [hare1997] D.E.G. Hare,
6963:       *Computing the Principal Branch of log-Gamma*,
6964:       Journal of Algorithms, Volume 25, Issue 2, November 1997, pages 221-236.
6965:     ''')
6966: 
6967: add_newdoc("scipy.special", "_sinpi",
6968:     '''
6969:     Internal function, do not use.
6970:     ''')
6971: 
6972: add_newdoc("scipy.special", "_cospi",
6973:     '''
6974:     Internal function, do not use.
6975:     ''')
6976: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_493254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)

# Assigning a type to the variable 'docdict' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'docdict', dict_493254)

@norecursion
def get(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get'
    module_type_store = module_type_store.open_function_context('get', 20, 0, False)
    
    # Passed parameters checking function
    get.stypy_localization = localization
    get.stypy_type_of_self = None
    get.stypy_type_store = module_type_store
    get.stypy_function_name = 'get'
    get.stypy_param_names_list = ['name']
    get.stypy_varargs_param_name = None
    get.stypy_kwargs_param_name = None
    get.stypy_call_defaults = defaults
    get.stypy_call_varargs = varargs
    get.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get(...)' code ##################

    
    # Call to get(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'name' (line 21)
    name_493257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'name', False)
    # Processing the call keyword arguments (line 21)
    kwargs_493258 = {}
    # Getting the type of 'docdict' (line 21)
    docdict_493255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'docdict', False)
    # Obtaining the member 'get' of a type (line 21)
    get_493256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 11), docdict_493255, 'get')
    # Calling get(args, kwargs) (line 21)
    get_call_result_493259 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), get_493256, *[name_493257], **kwargs_493258)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', get_call_result_493259)
    
    # ################# End of 'get(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_493260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_493260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get'
    return stypy_return_type_493260

# Assigning a type to the variable 'get' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'get', get)

@norecursion
def add_newdoc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'add_newdoc'
    module_type_store = module_type_store.open_function_context('add_newdoc', 24, 0, False)
    
    # Passed parameters checking function
    add_newdoc.stypy_localization = localization
    add_newdoc.stypy_type_of_self = None
    add_newdoc.stypy_type_store = module_type_store
    add_newdoc.stypy_function_name = 'add_newdoc'
    add_newdoc.stypy_param_names_list = ['place', 'name', 'doc']
    add_newdoc.stypy_varargs_param_name = None
    add_newdoc.stypy_kwargs_param_name = None
    add_newdoc.stypy_call_defaults = defaults
    add_newdoc.stypy_call_varargs = varargs
    add_newdoc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_newdoc', ['place', 'name', 'doc'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_newdoc', localization, ['place', 'name', 'doc'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_newdoc(...)' code ##################

    
    # Assigning a Name to a Subscript (line 25):
    # Getting the type of 'doc' (line 25)
    doc_493261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'doc')
    # Getting the type of 'docdict' (line 25)
    docdict_493262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'docdict')
    
    # Call to join(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_493265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    # Getting the type of 'place' (line 25)
    place_493266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'place', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 22), tuple_493265, place_493266)
    # Adding element type (line 25)
    # Getting the type of 'name' (line 25)
    name_493267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 22), tuple_493265, name_493267)
    
    # Processing the call keyword arguments (line 25)
    kwargs_493268 = {}
    str_493263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'str', '.')
    # Obtaining the member 'join' of a type (line 25)
    join_493264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), str_493263, 'join')
    # Calling join(args, kwargs) (line 25)
    join_call_result_493269 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), join_493264, *[tuple_493265], **kwargs_493268)
    
    # Storing an element on a container (line 25)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 4), docdict_493262, (join_call_result_493269, doc_493261))
    
    # ################# End of 'add_newdoc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_newdoc' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_493270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_493270)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_newdoc'
    return stypy_return_type_493270

# Assigning a type to the variable 'add_newdoc' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'add_newdoc', add_newdoc)

# Call to add_newdoc(...): (line 28)
# Processing the call arguments (line 28)
str_493272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', 'scipy.special')
str_493273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 28), 'str', '_sf_error_test_function')
str_493274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n    Private function; do not use.\n    ')
# Processing the call keyword arguments (line 28)
kwargs_493275 = {}
# Getting the type of 'add_newdoc' (line 28)
add_newdoc_493271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 28)
add_newdoc_call_result_493276 = invoke(stypy.reporting.localization.Localization(__file__, 28, 0), add_newdoc_493271, *[str_493272, str_493273, str_493274], **kwargs_493275)


# Call to add_newdoc(...): (line 33)
# Processing the call arguments (line 33)
str_493278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'str', 'scipy.special')
str_493279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'str', 'sph_harm')
str_493280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', "\n    sph_harm(m, n, theta, phi)\n\n    Compute spherical harmonics.\n\n    The spherical harmonics are defined as\n\n    .. math::\n\n        Y^m_n(\\theta,\\phi) = \\sqrt{\\frac{2n+1}{4\\pi} \\frac{(n-m)!}{(n+m)!}}\n          e^{i m \\theta} P^m_n(\\cos(\\phi))\n\n    where :math:`P_n^m` are the associated Legendre functions; see `lpmv`.\n\n    Parameters\n    ----------\n    m : array_like\n        Order of the harmonic (int); must have ``|m| <= n``.\n    n : array_like\n       Degree of the harmonic (int); must have ``n >= 0``. This is\n       often denoted by ``l`` (lower case L) in descriptions of\n       spherical harmonics.\n    theta : array_like\n       Azimuthal (longitudinal) coordinate; must be in ``[0, 2*pi]``.\n    phi : array_like\n       Polar (colatitudinal) coordinate; must be in ``[0, pi]``.\n\n    Returns\n    -------\n    y_mn : complex float\n       The harmonic :math:`Y^m_n` sampled at ``theta`` and ``phi``.\n\n    Notes\n    -----\n    There are different conventions for the meanings of the input\n    arguments ``theta`` and ``phi``. In SciPy ``theta`` is the\n    azimuthal angle and ``phi`` is the polar angle. It is common to\n    see the opposite convention, that is, ``theta`` as the polar angle\n    and ``phi`` as the azimuthal angle.\n\n    Note that SciPy's spherical harmonics include the Condon-Shortley\n    phase [2]_ because it is part of `lpmv`.\n\n    With SciPy's conventions, the first several spherical harmonics\n    are\n\n    .. math::\n\n        Y_0^0(\\theta, \\phi) &= \\frac{1}{2} \\sqrt{\\frac{1}{\\pi}} \\\\\n        Y_1^{-1}(\\theta, \\phi) &= \\frac{1}{2} \\sqrt{\\frac{3}{2\\pi}}\n                                    e^{-i\\theta} \\sin(\\phi) \\\\\n        Y_1^0(\\theta, \\phi) &= \\frac{1}{2} \\sqrt{\\frac{3}{\\pi}}\n                                 \\cos(\\phi) \\\\\n        Y_1^1(\\theta, \\phi) &= -\\frac{1}{2} \\sqrt{\\frac{3}{2\\pi}}\n                                 e^{i\\theta} \\sin(\\phi).\n\n    References\n    ----------\n    .. [1] Digital Library of Mathematical Functions, 14.30.\n           http://dlmf.nist.gov/14.30\n    .. [2] https://en.wikipedia.org/wiki/Spherical_harmonics#Condon.E2.80.93Shortley_phase\n    ")
# Processing the call keyword arguments (line 33)
kwargs_493281 = {}
# Getting the type of 'add_newdoc' (line 33)
add_newdoc_493277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 33)
add_newdoc_call_result_493282 = invoke(stypy.reporting.localization.Localization(__file__, 33, 0), add_newdoc_493277, *[str_493278, str_493279, str_493280], **kwargs_493281)


# Call to add_newdoc(...): (line 97)
# Processing the call arguments (line 97)
str_493284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'str', 'scipy.special')
str_493285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 28), 'str', '_ellip_harm')
str_493286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', '\n    Internal function, use `ellip_harm` instead.\n    ')
# Processing the call keyword arguments (line 97)
kwargs_493287 = {}
# Getting the type of 'add_newdoc' (line 97)
add_newdoc_493283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 97)
add_newdoc_call_result_493288 = invoke(stypy.reporting.localization.Localization(__file__, 97, 0), add_newdoc_493283, *[str_493284, str_493285, str_493286], **kwargs_493287)


# Call to add_newdoc(...): (line 102)
# Processing the call arguments (line 102)
str_493290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 11), 'str', 'scipy.special')
str_493291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 28), 'str', '_ellip_norm')
str_493292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n    Internal function, use `ellip_norm` instead.\n    ')
# Processing the call keyword arguments (line 102)
kwargs_493293 = {}
# Getting the type of 'add_newdoc' (line 102)
add_newdoc_493289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 102)
add_newdoc_call_result_493294 = invoke(stypy.reporting.localization.Localization(__file__, 102, 0), add_newdoc_493289, *[str_493290, str_493291, str_493292], **kwargs_493293)


# Call to add_newdoc(...): (line 107)
# Processing the call arguments (line 107)
str_493296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 11), 'str', 'scipy.special')
str_493297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'str', '_lambertw')
str_493298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '\n    Internal function, use `lambertw` instead.\n    ')
# Processing the call keyword arguments (line 107)
kwargs_493299 = {}
# Getting the type of 'add_newdoc' (line 107)
add_newdoc_493295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 107)
add_newdoc_call_result_493300 = invoke(stypy.reporting.localization.Localization(__file__, 107, 0), add_newdoc_493295, *[str_493296, str_493297, str_493298], **kwargs_493299)


# Call to add_newdoc(...): (line 112)
# Processing the call arguments (line 112)
str_493302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 11), 'str', 'scipy.special')
str_493303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 28), 'str', 'wrightomega')
str_493304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, (-1)), 'str', '\n    wrightomega(z, out=None)\n\n    Wright Omega function.\n\n    Defined as the solution to\n\n    .. math::\n\n        \\omega + \\log(\\omega) = z\n\n    where :math:`\\log` is the principal branch of the complex logarithm.\n\n    Parameters\n    ----------\n    z : array_like\n        Points at which to evaluate the Wright Omega function\n\n    Returns\n    -------\n    omega : ndarray\n        Values of the Wright Omega function\n\n    Notes\n    -----\n    .. versionadded:: 0.19.0\n\n    The function can also be defined as\n\n    .. math::\n\n        \\omega(z) = W_{K(z)}(e^z)\n\n    where :math:`K(z) = \\lceil (\\Im(z) - \\pi)/(2\\pi) \\rceil` is the\n    unwinding number and :math:`W` is the Lambert W function.\n\n    The implementation here is taken from [1]_.\n\n    See Also\n    --------\n    lambertw : The Lambert W function\n\n    References\n    ----------\n    .. [1] Lawrence, Corless, and Jeffrey, "Algorithm 917: Complex\n           Double-Precision Evaluation of the Wright :math:`\\omega`\n           Function." ACM Transactions on Mathematical Software,\n           2012. :doi:`10.1145/2168773.2168779`.\n\n    ')
# Processing the call keyword arguments (line 112)
kwargs_493305 = {}
# Getting the type of 'add_newdoc' (line 112)
add_newdoc_493301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 112)
add_newdoc_call_result_493306 = invoke(stypy.reporting.localization.Localization(__file__, 112, 0), add_newdoc_493301, *[str_493302, str_493303, str_493304], **kwargs_493305)


# Call to add_newdoc(...): (line 165)
# Processing the call arguments (line 165)
str_493308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 11), 'str', 'scipy.special')
str_493309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 28), 'str', 'agm')
str_493310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, (-1)), 'str', '\n    agm(a, b)\n\n    Compute the arithmetic-geometric mean of `a` and `b`.\n\n    Start with a_0 = a and b_0 = b and iteratively compute::\n\n        a_{n+1} = (a_n + b_n)/2\n        b_{n+1} = sqrt(a_n*b_n)\n\n    a_n and b_n converge to the same limit as n increases; their common\n    limit is agm(a, b).\n\n    Parameters\n    ----------\n    a, b : array_like\n        Real values only.  If the values are both negative, the result\n        is negative.  If one value is negative and the other is positive,\n        `nan` is returned.\n\n    Returns\n    -------\n    float\n        The arithmetic-geometric mean of `a` and `b`.\n\n    Examples\n    --------\n    >>> from scipy.special import agm\n    >>> a, b = 24.0, 6.0\n    >>> agm(a, b)\n    13.458171481725614\n\n    Compare that result to the iteration:\n\n    >>> while a != b:\n    ...     a, b = (a + b)/2, np.sqrt(a*b)\n    ...     print("a = %19.16f  b=%19.16f" % (a, b))\n    ...\n    a = 15.0000000000000000  b=12.0000000000000000\n    a = 13.5000000000000000  b=13.4164078649987388\n    a = 13.4582039324993694  b=13.4581390309909850\n    a = 13.4581714817451772  b=13.4581714817060547\n    a = 13.4581714817256159  b=13.4581714817256159\n\n    When array-like arguments are given, broadcasting applies:\n\n    >>> a = np.array([[1.5], [3], [6]])  # a has shape (3, 1).\n    >>> b = np.array([6, 12, 24, 48])    # b has shape (4,).\n    >>> agm(a, b)\n    array([[  3.36454287,   5.42363427,   9.05798751,  15.53650756],\n           [  4.37037309,   6.72908574,  10.84726853,  18.11597502],\n           [  6.        ,   8.74074619,  13.45817148,  21.69453707]])\n    ')
# Processing the call keyword arguments (line 165)
kwargs_493311 = {}
# Getting the type of 'add_newdoc' (line 165)
add_newdoc_493307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 165)
add_newdoc_call_result_493312 = invoke(stypy.reporting.localization.Localization(__file__, 165, 0), add_newdoc_493307, *[str_493308, str_493309, str_493310], **kwargs_493311)


# Call to add_newdoc(...): (line 220)
# Processing the call arguments (line 220)
str_493314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 11), 'str', 'scipy.special')
str_493315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 28), 'str', 'airy')
str_493316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, (-1)), 'str', '\n    airy(z)\n\n    Airy functions and their derivatives.\n\n    Parameters\n    ----------\n    z : array_like\n        Real or complex argument.\n\n    Returns\n    -------\n    Ai, Aip, Bi, Bip : ndarrays\n        Airy functions Ai and Bi, and their derivatives Aip and Bip.\n\n    Notes\n    -----\n    The Airy functions Ai and Bi are two independent solutions of\n\n    .. math:: y\'\'(x) = x y(x).\n\n    For real `z` in [-10, 10], the computation is carried out by calling\n    the Cephes [1]_ `airy` routine, which uses power series summation\n    for small `z` and rational minimax approximations for large `z`.\n\n    Outside this range, the AMOS [2]_ `zairy` and `zbiry` routines are\n    employed.  They are computed using power series for :math:`|z| < 1` and\n    the following relations to modified Bessel functions for larger `z`\n    (where :math:`t \\equiv 2 z^{3/2}/3`):\n\n    .. math::\n\n        Ai(z) = \\frac{1}{\\pi \\sqrt{3}} K_{1/3}(t)\n\n        Ai\'(z) = -\\frac{z}{\\pi \\sqrt{3}} K_{2/3}(t)\n\n        Bi(z) = \\sqrt{\\frac{z}{3}} \\left(I_{-1/3}(t) + I_{1/3}(t) \\right)\n\n        Bi\'(z) = \\frac{z}{\\sqrt{3}} \\left(I_{-2/3}(t) + I_{2/3}(t)\\right)\n\n    See also\n    --------\n    airye : exponentially scaled Airy functions.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n\n    Examples\n    --------\n    Compute the Airy functions on the interval [-15, 5].\n\n    >>> from scipy import special\n    >>> x = np.linspace(-15, 5, 201)\n    >>> ai, aip, bi, bip = special.airy(x)\n\n    Plot Ai(x) and Bi(x).\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x, ai, \'r\', label=\'Ai(x)\')\n    >>> plt.plot(x, bi, \'b--\', label=\'Bi(x)\')\n    >>> plt.ylim(-0.5, 1.0)\n    >>> plt.grid()\n    >>> plt.legend(loc=\'upper left\')\n    >>> plt.show()\n\n    ')
# Processing the call keyword arguments (line 220)
kwargs_493317 = {}
# Getting the type of 'add_newdoc' (line 220)
add_newdoc_493313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 220)
add_newdoc_call_result_493318 = invoke(stypy.reporting.localization.Localization(__file__, 220, 0), add_newdoc_493313, *[str_493314, str_493315, str_493316], **kwargs_493317)


# Call to add_newdoc(...): (line 293)
# Processing the call arguments (line 293)
str_493320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 11), 'str', 'scipy.special')
str_493321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 28), 'str', 'airye')
str_493322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', '\n    airye(z)\n\n    Exponentially scaled Airy functions and their derivatives.\n\n    Scaling::\n\n        eAi  = Ai  * exp(2.0/3.0*z*sqrt(z))\n        eAip = Aip * exp(2.0/3.0*z*sqrt(z))\n        eBi  = Bi  * exp(-abs(2.0/3.0*(z*sqrt(z)).real))\n        eBip = Bip * exp(-abs(2.0/3.0*(z*sqrt(z)).real))\n\n    Parameters\n    ----------\n    z : array_like\n        Real or complex argument.\n\n    Returns\n    -------\n    eAi, eAip, eBi, eBip : array_like\n        Airy functions Ai and Bi, and their derivatives Aip and Bip\n\n    Notes\n    -----\n    Wrapper for the AMOS [1]_ routines `zairy` and `zbiry`.\n\n    See also\n    --------\n    airy\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 293)
kwargs_493323 = {}
# Getting the type of 'add_newdoc' (line 293)
add_newdoc_493319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 293)
add_newdoc_call_result_493324 = invoke(stypy.reporting.localization.Localization(__file__, 293, 0), add_newdoc_493319, *[str_493320, str_493321, str_493322], **kwargs_493323)


# Call to add_newdoc(...): (line 331)
# Processing the call arguments (line 331)
str_493326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 11), 'str', 'scipy.special')
str_493327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 28), 'str', 'bdtr')
str_493328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, (-1)), 'str', '\n    bdtr(k, n, p)\n\n    Binomial distribution cumulative distribution function.\n\n    Sum of the terms 0 through `k` of the Binomial probability density.\n\n    .. math::\n        \\mathrm{bdtr}(k, n, p) = \\sum_{j=0}^k {{n}\\choose{j}} p^j (1-p)^{n-j}\n\n    Parameters\n    ----------\n    k : array_like\n        Number of successes (int).\n    n : array_like\n        Number of events (int).\n    p : array_like\n        Probability of success in a single event (float).\n\n    Returns\n    -------\n    y : ndarray\n        Probability of `k` or fewer successes in `n` independent events with\n        success probabilities of `p`.\n\n    Notes\n    -----\n    The terms are not summed directly; instead the regularized incomplete beta\n    function is employed, according to the formula,\n\n    .. math::\n        \\mathrm{bdtr}(k, n, p) = I_{1 - p}(n - k, k + 1).\n\n    Wrapper for the Cephes [1]_ routine `bdtr`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 331)
kwargs_493329 = {}
# Getting the type of 'add_newdoc' (line 331)
add_newdoc_493325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 331)
add_newdoc_call_result_493330 = invoke(stypy.reporting.localization.Localization(__file__, 331, 0), add_newdoc_493325, *[str_493326, str_493327, str_493328], **kwargs_493329)


# Call to add_newdoc(...): (line 374)
# Processing the call arguments (line 374)
str_493332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 11), 'str', 'scipy.special')
str_493333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 28), 'str', 'bdtrc')
str_493334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, (-1)), 'str', '\n    bdtrc(k, n, p)\n\n    Binomial distribution survival function.\n\n    Sum of the terms `k + 1` through `n` of the binomial probability density,\n\n    .. math::\n        \\mathrm{bdtrc}(k, n, p) = \\sum_{j=k+1}^n {{n}\\choose{j}} p^j (1-p)^{n-j}\n\n    Parameters\n    ----------\n    k : array_like\n        Number of successes (int).\n    n : array_like\n        Number of events (int)\n    p : array_like\n        Probability of success in a single event.\n\n    Returns\n    -------\n    y : ndarray\n        Probability of `k + 1` or more successes in `n` independent events\n        with success probabilities of `p`.\n\n    See also\n    --------\n    bdtr\n    betainc\n\n    Notes\n    -----\n    The terms are not summed directly; instead the regularized incomplete beta\n    function is employed, according to the formula,\n\n    .. math::\n        \\mathrm{bdtrc}(k, n, p) = I_{p}(k + 1, n - k).\n\n    Wrapper for the Cephes [1]_ routine `bdtrc`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 374)
kwargs_493335 = {}
# Getting the type of 'add_newdoc' (line 374)
add_newdoc_493331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 374)
add_newdoc_call_result_493336 = invoke(stypy.reporting.localization.Localization(__file__, 374, 0), add_newdoc_493331, *[str_493332, str_493333, str_493334], **kwargs_493335)


# Call to add_newdoc(...): (line 422)
# Processing the call arguments (line 422)
str_493338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 11), 'str', 'scipy.special')
str_493339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 28), 'str', 'bdtri')
str_493340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, (-1)), 'str', '\n    bdtri(k, n, y)\n\n    Inverse function to `bdtr` with respect to `p`.\n\n    Finds the event probability `p` such that the sum of the terms 0 through\n    `k` of the binomial probability density is equal to the given cumulative\n    probability `y`.\n\n    Parameters\n    ----------\n    k : array_like\n        Number of successes (float).\n    n : array_like\n        Number of events (float)\n    y : array_like\n        Cumulative probability (probability of `k` or fewer successes in `n`\n        events).\n\n    Returns\n    -------\n    p : ndarray\n        The event probability such that `bdtr(k, n, p) = y`.\n\n    See also\n    --------\n    bdtr\n    betaincinv\n\n    Notes\n    -----\n    The computation is carried out using the inverse beta integral function\n    and the relation,::\n\n        1 - p = betaincinv(n - k, k + 1, y).\n\n    Wrapper for the Cephes [1]_ routine `bdtri`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 422)
kwargs_493341 = {}
# Getting the type of 'add_newdoc' (line 422)
add_newdoc_493337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 422)
add_newdoc_call_result_493342 = invoke(stypy.reporting.localization.Localization(__file__, 422, 0), add_newdoc_493337, *[str_493338, str_493339, str_493340], **kwargs_493341)


# Call to add_newdoc(...): (line 467)
# Processing the call arguments (line 467)
str_493344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 11), 'str', 'scipy.special')
str_493345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 28), 'str', 'bdtrik')
str_493346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, (-1)), 'str', '\n    bdtrik(y, n, p)\n\n    Inverse function to `bdtr` with respect to `k`.\n\n    Finds the number of successes `k` such that the sum of the terms 0 through\n    `k` of the Binomial probability density for `n` events with probability\n    `p` is equal to the given cumulative probability `y`.\n\n    Parameters\n    ----------\n    y : array_like\n        Cumulative probability (probability of `k` or fewer successes in `n`\n        events).\n    n : array_like\n        Number of events (float).\n    p : array_like\n        Success probability (float).\n\n    Returns\n    -------\n    k : ndarray\n        The number of successes `k` such that `bdtr(k, n, p) = y`.\n\n    See also\n    --------\n    bdtr\n\n    Notes\n    -----\n    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the\n    cumulative incomplete beta distribution.\n\n    Computation of `k` involves a search for a value that produces the desired\n    value of `y`.  The search relies on the monotonicity of `y` with `k`.\n\n    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.\n\n    References\n    ----------\n    .. [1] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n    .. [2] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n\n    ')
# Processing the call keyword arguments (line 467)
kwargs_493347 = {}
# Getting the type of 'add_newdoc' (line 467)
add_newdoc_493343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 467)
add_newdoc_call_result_493348 = invoke(stypy.reporting.localization.Localization(__file__, 467, 0), add_newdoc_493343, *[str_493344, str_493345, str_493346], **kwargs_493347)


# Call to add_newdoc(...): (line 517)
# Processing the call arguments (line 517)
str_493350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 11), 'str', 'scipy.special')
str_493351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 28), 'str', 'bdtrin')
str_493352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, (-1)), 'str', '\n    bdtrin(k, y, p)\n\n    Inverse function to `bdtr` with respect to `n`.\n\n    Finds the number of events `n` such that the sum of the terms 0 through\n    `k` of the Binomial probability density for events with probability `p` is\n    equal to the given cumulative probability `y`.\n\n    Parameters\n    ----------\n    k : array_like\n        Number of successes (float).\n    y : array_like\n        Cumulative probability (probability of `k` or fewer successes in `n`\n        events).\n    p : array_like\n        Success probability (float).\n\n    Returns\n    -------\n    n : ndarray\n        The number of events `n` such that `bdtr(k, n, p) = y`.\n\n    See also\n    --------\n    bdtr\n\n    Notes\n    -----\n    Formula 26.5.24 of [1]_ is used to reduce the binomial distribution to the\n    cumulative incomplete beta distribution.\n\n    Computation of `n` involves a search for a value that produces the desired\n    value of `y`.  The search relies on the monotonicity of `y` with `n`.\n\n    Wrapper for the CDFLIB [2]_ Fortran routine `cdfbin`.\n\n    References\n    ----------\n    .. [1] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n    .. [2] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    ')
# Processing the call keyword arguments (line 517)
kwargs_493353 = {}
# Getting the type of 'add_newdoc' (line 517)
add_newdoc_493349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 517)
add_newdoc_call_result_493354 = invoke(stypy.reporting.localization.Localization(__file__, 517, 0), add_newdoc_493349, *[str_493350, str_493351, str_493352], **kwargs_493353)


# Call to add_newdoc(...): (line 566)
# Processing the call arguments (line 566)
str_493356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 11), 'str', 'scipy.special')
str_493357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 28), 'str', 'binom')
str_493358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, (-1)), 'str', '\n    binom(n, k)\n\n    Binomial coefficient\n\n    See Also\n    --------\n    comb : The number of combinations of N things taken k at a time.\n\n    ')
# Processing the call keyword arguments (line 566)
kwargs_493359 = {}
# Getting the type of 'add_newdoc' (line 566)
add_newdoc_493355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 566)
add_newdoc_call_result_493360 = invoke(stypy.reporting.localization.Localization(__file__, 566, 0), add_newdoc_493355, *[str_493356, str_493357, str_493358], **kwargs_493359)


# Call to add_newdoc(...): (line 578)
# Processing the call arguments (line 578)
str_493362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 11), 'str', 'scipy.special')
str_493363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 28), 'str', 'btdtria')
str_493364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, (-1)), 'str', '\n    btdtria(p, b, x)\n\n    Inverse of `btdtr` with respect to `a`.\n\n    This is the inverse of the beta cumulative distribution function, `btdtr`,\n    considered as a function of `a`, returning the value of `a` for which\n    `btdtr(a, b, x) = p`, or\n\n    .. math::\n        p = \\int_0^x \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} t^{a-1} (1-t)^{b-1}\\,dt\n\n    Parameters\n    ----------\n    p : array_like\n        Cumulative probability, in [0, 1].\n    b : array_like\n        Shape parameter (`b` > 0).\n    x : array_like\n        The quantile, in [0, 1].\n\n    Returns\n    -------\n    a : ndarray\n        The value of the shape parameter `a` such that `btdtr(a, b, x) = p`.\n\n    See Also\n    --------\n    btdtr : Cumulative density function of the beta distribution.\n    btdtri : Inverse with respect to `x`.\n    btdtrib : Inverse with respect to `b`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.\n\n    The cumulative distribution function `p` is computed using a routine by\n    DiDinato and Morris [2]_.  Computation of `a` involves a search for a value\n    that produces the desired value of `p`.  The search relies on the\n    monotonicity of `p` with `a`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] DiDinato, A. R. and Morris, A. H.,\n           Algorithm 708: Significant Digit Computation of the Incomplete Beta\n           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.\n\n    ')
# Processing the call keyword arguments (line 578)
kwargs_493365 = {}
# Getting the type of 'add_newdoc' (line 578)
add_newdoc_493361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 578)
add_newdoc_call_result_493366 = invoke(stypy.reporting.localization.Localization(__file__, 578, 0), add_newdoc_493361, *[str_493362, str_493363, str_493364], **kwargs_493365)


# Call to add_newdoc(...): (line 631)
# Processing the call arguments (line 631)
str_493368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 11), 'str', 'scipy.special')
str_493369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 28), 'str', 'btdtrib')
str_493370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, (-1)), 'str', '\n    btdtria(a, p, x)\n\n    Inverse of `btdtr` with respect to `b`.\n\n    This is the inverse of the beta cumulative distribution function, `btdtr`,\n    considered as a function of `b`, returning the value of `b` for which\n    `btdtr(a, b, x) = p`, or\n\n    .. math::\n        p = \\int_0^x \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} t^{a-1} (1-t)^{b-1}\\,dt\n\n    Parameters\n    ----------\n    a : array_like\n        Shape parameter (`a` > 0).\n    p : array_like\n        Cumulative probability, in [0, 1].\n    x : array_like\n        The quantile, in [0, 1].\n\n    Returns\n    -------\n    b : ndarray\n        The value of the shape parameter `b` such that `btdtr(a, b, x) = p`.\n\n    See Also\n    --------\n    btdtr : Cumulative density function of the beta distribution.\n    btdtri : Inverse with respect to `x`.\n    btdtria : Inverse with respect to `a`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfbet`.\n\n    The cumulative distribution function `p` is computed using a routine by\n    DiDinato and Morris [2]_.  Computation of `b` involves a search for a value\n    that produces the desired value of `p`.  The search relies on the\n    monotonicity of `p` with `b`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] DiDinato, A. R. and Morris, A. H.,\n           Algorithm 708: Significant Digit Computation of the Incomplete Beta\n           Function Ratios. ACM Trans. Math. Softw. 18 (1993), 360-373.\n\n\n    ')
# Processing the call keyword arguments (line 631)
kwargs_493371 = {}
# Getting the type of 'add_newdoc' (line 631)
add_newdoc_493367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 631)
add_newdoc_call_result_493372 = invoke(stypy.reporting.localization.Localization(__file__, 631, 0), add_newdoc_493367, *[str_493368, str_493369, str_493370], **kwargs_493371)


# Call to add_newdoc(...): (line 685)
# Processing the call arguments (line 685)
str_493374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 11), 'str', 'scipy.special')
str_493375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 28), 'str', 'bei')
str_493376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, (-1)), 'str', '\n    bei(x)\n\n    Kelvin function bei\n    ')
# Processing the call keyword arguments (line 685)
kwargs_493377 = {}
# Getting the type of 'add_newdoc' (line 685)
add_newdoc_493373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 685)
add_newdoc_call_result_493378 = invoke(stypy.reporting.localization.Localization(__file__, 685, 0), add_newdoc_493373, *[str_493374, str_493375, str_493376], **kwargs_493377)


# Call to add_newdoc(...): (line 692)
# Processing the call arguments (line 692)
str_493380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 11), 'str', 'scipy.special')
str_493381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 28), 'str', 'beip')
str_493382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, (-1)), 'str', '\n    beip(x)\n\n    Derivative of the Kelvin function `bei`\n    ')
# Processing the call keyword arguments (line 692)
kwargs_493383 = {}
# Getting the type of 'add_newdoc' (line 692)
add_newdoc_493379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 692)
add_newdoc_call_result_493384 = invoke(stypy.reporting.localization.Localization(__file__, 692, 0), add_newdoc_493379, *[str_493380, str_493381, str_493382], **kwargs_493383)


# Call to add_newdoc(...): (line 699)
# Processing the call arguments (line 699)
str_493386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 11), 'str', 'scipy.special')
str_493387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 28), 'str', 'ber')
str_493388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, (-1)), 'str', '\n    ber(x)\n\n    Kelvin function ber.\n    ')
# Processing the call keyword arguments (line 699)
kwargs_493389 = {}
# Getting the type of 'add_newdoc' (line 699)
add_newdoc_493385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 699)
add_newdoc_call_result_493390 = invoke(stypy.reporting.localization.Localization(__file__, 699, 0), add_newdoc_493385, *[str_493386, str_493387, str_493388], **kwargs_493389)


# Call to add_newdoc(...): (line 706)
# Processing the call arguments (line 706)
str_493392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 11), 'str', 'scipy.special')
str_493393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 28), 'str', 'berp')
str_493394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, (-1)), 'str', '\n    berp(x)\n\n    Derivative of the Kelvin function `ber`\n    ')
# Processing the call keyword arguments (line 706)
kwargs_493395 = {}
# Getting the type of 'add_newdoc' (line 706)
add_newdoc_493391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 706)
add_newdoc_call_result_493396 = invoke(stypy.reporting.localization.Localization(__file__, 706, 0), add_newdoc_493391, *[str_493392, str_493393, str_493394], **kwargs_493395)


# Call to add_newdoc(...): (line 713)
# Processing the call arguments (line 713)
str_493398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 11), 'str', 'scipy.special')
str_493399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 28), 'str', 'besselpoly')
str_493400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, (-1)), 'str', '\n    besselpoly(a, lmb, nu)\n\n    Weighted integral of a Bessel function.\n\n    .. math::\n\n       \\int_0^1 x^\\lambda J_\\nu(2 a x) \\, dx\n\n    where :math:`J_\\nu` is a Bessel function and :math:`\\lambda=lmb`,\n    :math:`\\nu=nu`.\n\n    ')
# Processing the call keyword arguments (line 713)
kwargs_493401 = {}
# Getting the type of 'add_newdoc' (line 713)
add_newdoc_493397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 713)
add_newdoc_call_result_493402 = invoke(stypy.reporting.localization.Localization(__file__, 713, 0), add_newdoc_493397, *[str_493398, str_493399, str_493400], **kwargs_493401)


# Call to add_newdoc(...): (line 728)
# Processing the call arguments (line 728)
str_493404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 11), 'str', 'scipy.special')
str_493405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 28), 'str', 'beta')
str_493406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, (-1)), 'str', '\n    beta(a, b)\n\n    Beta function.\n\n    ::\n\n        beta(a, b) =  gamma(a) * gamma(b) / gamma(a+b)\n    ')
# Processing the call keyword arguments (line 728)
kwargs_493407 = {}
# Getting the type of 'add_newdoc' (line 728)
add_newdoc_493403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 728)
add_newdoc_call_result_493408 = invoke(stypy.reporting.localization.Localization(__file__, 728, 0), add_newdoc_493403, *[str_493404, str_493405, str_493406], **kwargs_493407)


# Call to add_newdoc(...): (line 739)
# Processing the call arguments (line 739)
str_493410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 11), 'str', 'scipy.special')
str_493411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 28), 'str', 'betainc')
str_493412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, (-1)), 'str', '\n    betainc(a, b, x)\n\n    Incomplete beta integral.\n\n    Compute the incomplete beta integral of the arguments, evaluated\n    from zero to `x`::\n\n        gamma(a+b) / (gamma(a)*gamma(b)) * integral(t**(a-1) (1-t)**(b-1), t=0..x).\n\n    Notes\n    -----\n    The incomplete beta is also sometimes defined without the terms\n    in gamma, in which case the above definition is the so-called regularized\n    incomplete beta. Under this definition, you can get the incomplete beta by\n    multiplying the result of the scipy function by beta(a, b).\n\n    ')
# Processing the call keyword arguments (line 739)
kwargs_493413 = {}
# Getting the type of 'add_newdoc' (line 739)
add_newdoc_493409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 739)
add_newdoc_call_result_493414 = invoke(stypy.reporting.localization.Localization(__file__, 739, 0), add_newdoc_493409, *[str_493410, str_493411, str_493412], **kwargs_493413)


# Call to add_newdoc(...): (line 759)
# Processing the call arguments (line 759)
str_493416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 11), 'str', 'scipy.special')
str_493417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 759, 28), 'str', 'betaincinv')
str_493418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 766, (-1)), 'str', '\n    betaincinv(a, b, y)\n\n    Inverse function to beta integral.\n\n    Compute `x` such that betainc(a, b, x) = y.\n    ')
# Processing the call keyword arguments (line 759)
kwargs_493419 = {}
# Getting the type of 'add_newdoc' (line 759)
add_newdoc_493415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 759)
add_newdoc_call_result_493420 = invoke(stypy.reporting.localization.Localization(__file__, 759, 0), add_newdoc_493415, *[str_493416, str_493417, str_493418], **kwargs_493419)


# Call to add_newdoc(...): (line 768)
# Processing the call arguments (line 768)
str_493422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 11), 'str', 'scipy.special')
str_493423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 28), 'str', 'betaln')
str_493424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 775, (-1)), 'str', '\n    betaln(a, b)\n\n    Natural logarithm of absolute value of beta function.\n\n    Computes ``ln(abs(beta(a, b)))``.\n    ')
# Processing the call keyword arguments (line 768)
kwargs_493425 = {}
# Getting the type of 'add_newdoc' (line 768)
add_newdoc_493421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 768)
add_newdoc_call_result_493426 = invoke(stypy.reporting.localization.Localization(__file__, 768, 0), add_newdoc_493421, *[str_493422, str_493423, str_493424], **kwargs_493425)


# Call to add_newdoc(...): (line 777)
# Processing the call arguments (line 777)
str_493428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 11), 'str', 'scipy.special')
str_493429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 777, 28), 'str', 'boxcox')
str_493430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, (-1)), 'str', '\n    boxcox(x, lmbda)\n\n    Compute the Box-Cox transformation.\n\n    The Box-Cox transformation is::\n\n        y = (x**lmbda - 1) / lmbda  if lmbda != 0\n            log(x)                  if lmbda == 0\n\n    Returns `nan` if ``x < 0``.\n    Returns `-inf` if ``x == 0`` and ``lmbda < 0``.\n\n    Parameters\n    ----------\n    x : array_like\n        Data to be transformed.\n    lmbda : array_like\n        Power parameter of the Box-Cox transform.\n\n    Returns\n    -------\n    y : array\n        Transformed data.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> from scipy.special import boxcox\n    >>> boxcox([1, 4, 10], 2.5)\n    array([   0.        ,   12.4       ,  126.09110641])\n    >>> boxcox(2, [0, 1, 2])\n    array([ 0.69314718,  1.        ,  1.5       ])\n    ')
# Processing the call keyword arguments (line 777)
kwargs_493431 = {}
# Getting the type of 'add_newdoc' (line 777)
add_newdoc_493427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 777)
add_newdoc_call_result_493432 = invoke(stypy.reporting.localization.Localization(__file__, 777, 0), add_newdoc_493427, *[str_493428, str_493429, str_493430], **kwargs_493431)


# Call to add_newdoc(...): (line 817)
# Processing the call arguments (line 817)
str_493434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 11), 'str', 'scipy.special')
str_493435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 28), 'str', 'boxcox1p')
str_493436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 855, (-1)), 'str', '\n    boxcox1p(x, lmbda)\n\n    Compute the Box-Cox transformation of 1 + `x`.\n\n    The Box-Cox transformation computed by `boxcox1p` is::\n\n        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0\n            log(1+x)                    if lmbda == 0\n\n    Returns `nan` if ``x < -1``.\n    Returns `-inf` if ``x == -1`` and ``lmbda < 0``.\n\n    Parameters\n    ----------\n    x : array_like\n        Data to be transformed.\n    lmbda : array_like\n        Power parameter of the Box-Cox transform.\n\n    Returns\n    -------\n    y : array\n        Transformed data.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> from scipy.special import boxcox1p\n    >>> boxcox1p(1e-4, [0, 0.5, 1])\n    array([  9.99950003e-05,   9.99975001e-05,   1.00000000e-04])\n    >>> boxcox1p([0.01, 0.1], 0.25)\n    array([ 0.00996272,  0.09645476])\n    ')
# Processing the call keyword arguments (line 817)
kwargs_493437 = {}
# Getting the type of 'add_newdoc' (line 817)
add_newdoc_493433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 817)
add_newdoc_call_result_493438 = invoke(stypy.reporting.localization.Localization(__file__, 817, 0), add_newdoc_493433, *[str_493434, str_493435, str_493436], **kwargs_493437)


# Call to add_newdoc(...): (line 857)
# Processing the call arguments (line 857)
str_493440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 11), 'str', 'scipy.special')
str_493441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 857, 28), 'str', 'inv_boxcox')
str_493442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 891, (-1)), 'str', '\n    inv_boxcox(y, lmbda)\n\n    Compute the inverse of the Box-Cox transformation.\n\n    Find ``x`` such that::\n\n        y = (x**lmbda - 1) / lmbda  if lmbda != 0\n            log(x)                  if lmbda == 0\n\n    Parameters\n    ----------\n    y : array_like\n        Data to be transformed.\n    lmbda : array_like\n        Power parameter of the Box-Cox transform.\n\n    Returns\n    -------\n    x : array\n        Transformed data.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.16.0\n\n    Examples\n    --------\n    >>> from scipy.special import boxcox, inv_boxcox\n    >>> y = boxcox([1, 4, 10], 2.5)\n    >>> inv_boxcox(y, 2.5)\n    array([1., 4., 10.])\n    ')
# Processing the call keyword arguments (line 857)
kwargs_493443 = {}
# Getting the type of 'add_newdoc' (line 857)
add_newdoc_493439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 857)
add_newdoc_call_result_493444 = invoke(stypy.reporting.localization.Localization(__file__, 857, 0), add_newdoc_493439, *[str_493440, str_493441, str_493442], **kwargs_493443)


# Call to add_newdoc(...): (line 893)
# Processing the call arguments (line 893)
str_493446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 11), 'str', 'scipy.special')
str_493447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 28), 'str', 'inv_boxcox1p')
str_493448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, (-1)), 'str', '\n    inv_boxcox1p(y, lmbda)\n\n    Compute the inverse of the Box-Cox transformation.\n\n    Find ``x`` such that::\n\n        y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0\n            log(1+x)                    if lmbda == 0\n\n    Parameters\n    ----------\n    y : array_like\n        Data to be transformed.\n    lmbda : array_like\n        Power parameter of the Box-Cox transform.\n\n    Returns\n    -------\n    x : array\n        Transformed data.\n\n    Notes\n    -----\n\n    .. versionadded:: 0.16.0\n\n    Examples\n    --------\n    >>> from scipy.special import boxcox1p, inv_boxcox1p\n    >>> y = boxcox1p([1, 4, 10], 2.5)\n    >>> inv_boxcox1p(y, 2.5)\n    array([1., 4., 10.])\n    ')
# Processing the call keyword arguments (line 893)
kwargs_493449 = {}
# Getting the type of 'add_newdoc' (line 893)
add_newdoc_493445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 893)
add_newdoc_call_result_493450 = invoke(stypy.reporting.localization.Localization(__file__, 893, 0), add_newdoc_493445, *[str_493446, str_493447, str_493448], **kwargs_493449)


# Call to add_newdoc(...): (line 929)
# Processing the call arguments (line 929)
str_493452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 11), 'str', 'scipy.special')
str_493453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 28), 'str', 'btdtr')
str_493454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, (-1)), 'str', '\n    btdtr(a, b, x)\n\n    Cumulative density function of the beta distribution.\n\n    Returns the integral from zero to `x` of the beta probability density\n    function,\n\n    .. math::\n        I = \\int_0^x \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} t^{a-1} (1-t)^{b-1}\\,dt\n\n    where :math:`\\Gamma` is the gamma function.\n\n    Parameters\n    ----------\n    a : array_like\n        Shape parameter (a > 0).\n    b : array_like\n        Shape parameter (b > 0).\n    x : array_like\n        Upper limit of integration, in [0, 1].\n\n    Returns\n    -------\n    I : ndarray\n        Cumulative density function of the beta distribution with parameters\n        `a` and `b` at `x`.\n\n    See Also\n    --------\n    betainc\n\n    Notes\n    -----\n    This function is identical to the incomplete beta integral function\n    `betainc`.\n\n    Wrapper for the Cephes [1]_ routine `btdtr`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 929)
kwargs_493455 = {}
# Getting the type of 'add_newdoc' (line 929)
add_newdoc_493451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 929)
add_newdoc_call_result_493456 = invoke(stypy.reporting.localization.Localization(__file__, 929, 0), add_newdoc_493451, *[str_493452, str_493453, str_493454], **kwargs_493455)


# Call to add_newdoc(...): (line 976)
# Processing the call arguments (line 976)
str_493458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 11), 'str', 'scipy.special')
str_493459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 28), 'str', 'btdtri')
str_493460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, (-1)), 'str', '\n    btdtri(a, b, p)\n\n    The `p`-th quantile of the beta distribution.\n\n    This function is the inverse of the beta cumulative distribution function,\n    `btdtr`, returning the value of `x` for which `btdtr(a, b, x) = p`, or\n\n    .. math::\n        p = \\int_0^x \\frac{\\Gamma(a + b)}{\\Gamma(a)\\Gamma(b)} t^{a-1} (1-t)^{b-1}\\,dt\n\n    Parameters\n    ----------\n    a : array_like\n        Shape parameter (`a` > 0).\n    b : array_like\n        Shape parameter (`b` > 0).\n    p : array_like\n        Cumulative probability, in [0, 1].\n\n    Returns\n    -------\n    x : ndarray\n        The quantile corresponding to `p`.\n\n    See Also\n    --------\n    betaincinv\n    btdtr\n\n    Notes\n    -----\n    The value of `x` is found by interval halving or Newton iterations.\n\n    Wrapper for the Cephes [1]_ routine `incbi`, which solves the equivalent\n    problem of finding the inverse of the incomplete beta integral.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 976)
kwargs_493461 = {}
# Getting the type of 'add_newdoc' (line 976)
add_newdoc_493457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 976)
add_newdoc_call_result_493462 = invoke(stypy.reporting.localization.Localization(__file__, 976, 0), add_newdoc_493457, *[str_493458, str_493459, str_493460], **kwargs_493461)


# Call to add_newdoc(...): (line 1021)
# Processing the call arguments (line 1021)
str_493464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 11), 'str', 'scipy.special')
str_493465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 28), 'str', 'cbrt')
str_493466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, (-1)), 'str', '\n    cbrt(x)\n\n    Element-wise cube root of `x`.\n\n    Parameters\n    ----------\n    x : array_like\n        `x` must contain real numbers.\n\n    Returns\n    -------\n    float\n        The cube root of each value in `x`.\n\n    Examples\n    --------\n    >>> from scipy.special import cbrt\n\n    >>> cbrt(8)\n    2.0\n    >>> cbrt([-8, -3, 0.125, 1.331])\n    array([-2.        , -1.44224957,  0.5       ,  1.1       ])\n\n    ')
# Processing the call keyword arguments (line 1021)
kwargs_493467 = {}
# Getting the type of 'add_newdoc' (line 1021)
add_newdoc_493463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1021)
add_newdoc_call_result_493468 = invoke(stypy.reporting.localization.Localization(__file__, 1021, 0), add_newdoc_493463, *[str_493464, str_493465, str_493466], **kwargs_493467)


# Call to add_newdoc(...): (line 1048)
# Processing the call arguments (line 1048)
str_493470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 11), 'str', 'scipy.special')
str_493471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 28), 'str', 'chdtr')
str_493472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, (-1)), 'str', '\n    chdtr(v, x)\n\n    Chi square cumulative distribution function\n\n    Returns the area under the left hand tail (from 0 to `x`) of the Chi\n    square probability density function with `v` degrees of freedom::\n\n        1/(2**(v/2) * gamma(v/2)) * integral(t**(v/2-1) * exp(-t/2), t=0..x)\n    ')
# Processing the call keyword arguments (line 1048)
kwargs_493473 = {}
# Getting the type of 'add_newdoc' (line 1048)
add_newdoc_493469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1048)
add_newdoc_call_result_493474 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 0), add_newdoc_493469, *[str_493470, str_493471, str_493472], **kwargs_493473)


# Call to add_newdoc(...): (line 1060)
# Processing the call arguments (line 1060)
str_493476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 11), 'str', 'scipy.special')
str_493477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 28), 'str', 'chdtrc')
str_493478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, (-1)), 'str', '\n    chdtrc(v, x)\n\n    Chi square survival function\n\n    Returns the area under the right hand tail (from `x` to\n    infinity) of the Chi square probability density function with `v`\n    degrees of freedom::\n\n        1/(2**(v/2) * gamma(v/2)) * integral(t**(v/2-1) * exp(-t/2), t=x..inf)\n    ')
# Processing the call keyword arguments (line 1060)
kwargs_493479 = {}
# Getting the type of 'add_newdoc' (line 1060)
add_newdoc_493475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1060)
add_newdoc_call_result_493480 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 0), add_newdoc_493475, *[str_493476, str_493477, str_493478], **kwargs_493479)


# Call to add_newdoc(...): (line 1073)
# Processing the call arguments (line 1073)
str_493482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 11), 'str', 'scipy.special')
str_493483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 28), 'str', 'chdtri')
str_493484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1080, (-1)), 'str', '\n    chdtri(v, p)\n\n    Inverse to `chdtrc`\n\n    Returns the argument x such that ``chdtrc(v, x) == p``.\n    ')
# Processing the call keyword arguments (line 1073)
kwargs_493485 = {}
# Getting the type of 'add_newdoc' (line 1073)
add_newdoc_493481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1073)
add_newdoc_call_result_493486 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 0), add_newdoc_493481, *[str_493482, str_493483, str_493484], **kwargs_493485)


# Call to add_newdoc(...): (line 1082)
# Processing the call arguments (line 1082)
str_493488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 11), 'str', 'scipy.special')
str_493489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1082, 28), 'str', 'chdtriv')
str_493490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1089, (-1)), 'str', '\n    chdtriv(p, x)\n\n    Inverse to `chdtr` vs `v`\n\n    Returns the argument v such that ``chdtr(v, x) == p``.\n    ')
# Processing the call keyword arguments (line 1082)
kwargs_493491 = {}
# Getting the type of 'add_newdoc' (line 1082)
add_newdoc_493487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1082, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1082)
add_newdoc_call_result_493492 = invoke(stypy.reporting.localization.Localization(__file__, 1082, 0), add_newdoc_493487, *[str_493488, str_493489, str_493490], **kwargs_493491)


# Call to add_newdoc(...): (line 1091)
# Processing the call arguments (line 1091)
str_493494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 11), 'str', 'scipy.special')
str_493495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 28), 'str', 'chndtr')
str_493496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1097, (-1)), 'str', '\n    chndtr(x, df, nc)\n\n    Non-central chi square cumulative distribution function\n\n    ')
# Processing the call keyword arguments (line 1091)
kwargs_493497 = {}
# Getting the type of 'add_newdoc' (line 1091)
add_newdoc_493493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1091)
add_newdoc_call_result_493498 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 0), add_newdoc_493493, *[str_493494, str_493495, str_493496], **kwargs_493497)


# Call to add_newdoc(...): (line 1099)
# Processing the call arguments (line 1099)
str_493500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 11), 'str', 'scipy.special')
str_493501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1099, 28), 'str', 'chndtrix')
str_493502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, (-1)), 'str', '\n    chndtrix(p, df, nc)\n\n    Inverse to `chndtr` vs `x`\n    ')
# Processing the call keyword arguments (line 1099)
kwargs_493503 = {}
# Getting the type of 'add_newdoc' (line 1099)
add_newdoc_493499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1099, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1099)
add_newdoc_call_result_493504 = invoke(stypy.reporting.localization.Localization(__file__, 1099, 0), add_newdoc_493499, *[str_493500, str_493501, str_493502], **kwargs_493503)


# Call to add_newdoc(...): (line 1106)
# Processing the call arguments (line 1106)
str_493506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 11), 'str', 'scipy.special')
str_493507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 28), 'str', 'chndtridf')
str_493508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1111, (-1)), 'str', '\n    chndtridf(x, p, nc)\n\n    Inverse to `chndtr` vs `df`\n    ')
# Processing the call keyword arguments (line 1106)
kwargs_493509 = {}
# Getting the type of 'add_newdoc' (line 1106)
add_newdoc_493505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1106)
add_newdoc_call_result_493510 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 0), add_newdoc_493505, *[str_493506, str_493507, str_493508], **kwargs_493509)


# Call to add_newdoc(...): (line 1113)
# Processing the call arguments (line 1113)
str_493512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 11), 'str', 'scipy.special')
str_493513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1113, 28), 'str', 'chndtrinc')
str_493514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1118, (-1)), 'str', '\n    chndtrinc(x, df, p)\n\n    Inverse to `chndtr` vs `nc`\n    ')
# Processing the call keyword arguments (line 1113)
kwargs_493515 = {}
# Getting the type of 'add_newdoc' (line 1113)
add_newdoc_493511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1113)
add_newdoc_call_result_493516 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 0), add_newdoc_493511, *[str_493512, str_493513, str_493514], **kwargs_493515)


# Call to add_newdoc(...): (line 1120)
# Processing the call arguments (line 1120)
str_493518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 11), 'str', 'scipy.special')
str_493519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1120, 28), 'str', 'cosdg')
str_493520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, (-1)), 'str', '\n    cosdg(x)\n\n    Cosine of the angle `x` given in degrees.\n    ')
# Processing the call keyword arguments (line 1120)
kwargs_493521 = {}
# Getting the type of 'add_newdoc' (line 1120)
add_newdoc_493517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1120)
add_newdoc_call_result_493522 = invoke(stypy.reporting.localization.Localization(__file__, 1120, 0), add_newdoc_493517, *[str_493518, str_493519, str_493520], **kwargs_493521)


# Call to add_newdoc(...): (line 1127)
# Processing the call arguments (line 1127)
str_493524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 11), 'str', 'scipy.special')
str_493525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 28), 'str', 'cosm1')
str_493526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, (-1)), 'str', '\n    cosm1(x)\n\n    cos(x) - 1 for use when `x` is near zero.\n    ')
# Processing the call keyword arguments (line 1127)
kwargs_493527 = {}
# Getting the type of 'add_newdoc' (line 1127)
add_newdoc_493523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1127)
add_newdoc_call_result_493528 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 0), add_newdoc_493523, *[str_493524, str_493525, str_493526], **kwargs_493527)


# Call to add_newdoc(...): (line 1134)
# Processing the call arguments (line 1134)
str_493530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 11), 'str', 'scipy.special')
str_493531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 28), 'str', 'cotdg')
str_493532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, (-1)), 'str', '\n    cotdg(x)\n\n    Cotangent of the angle `x` given in degrees.\n    ')
# Processing the call keyword arguments (line 1134)
kwargs_493533 = {}
# Getting the type of 'add_newdoc' (line 1134)
add_newdoc_493529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1134)
add_newdoc_call_result_493534 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 0), add_newdoc_493529, *[str_493530, str_493531, str_493532], **kwargs_493533)


# Call to add_newdoc(...): (line 1141)
# Processing the call arguments (line 1141)
str_493536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 11), 'str', 'scipy.special')
str_493537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1141, 28), 'str', 'dawsn')
str_493538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, (-1)), 'str', "\n    dawsn(x)\n\n    Dawson's integral.\n\n    Computes::\n\n        exp(-x**2) * integral(exp(t**2), t=0..x).\n\n    See Also\n    --------\n    wofz, erf, erfc, erfcx, erfi\n\n    References\n    ----------\n    .. [1] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-15, 15, num=1000)\n    >>> plt.plot(x, special.dawsn(x))\n    >>> plt.xlabel('$x$')\n    >>> plt.ylabel('$dawsn(x)$')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 1141)
kwargs_493539 = {}
# Getting the type of 'add_newdoc' (line 1141)
add_newdoc_493535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1141, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1141)
add_newdoc_call_result_493540 = invoke(stypy.reporting.localization.Localization(__file__, 1141, 0), add_newdoc_493535, *[str_493536, str_493537, str_493538], **kwargs_493539)


# Call to add_newdoc(...): (line 1172)
# Processing the call arguments (line 1172)
str_493542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 11), 'str', 'scipy.special')
str_493543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 28), 'str', 'ellipe')
str_493544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, (-1)), 'str', '\n    ellipe(m)\n\n    Complete elliptic integral of the second kind\n\n    This function is defined as\n\n    .. math:: E(m) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{1/2} dt\n\n    Parameters\n    ----------\n    m : array_like\n        Defines the parameter of the elliptic integral.\n\n    Returns\n    -------\n    E : ndarray\n        Value of the elliptic integral.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `ellpe`.\n\n    For `m > 0` the computation uses the approximation,\n\n    .. math:: E(m) \\approx P(1-m) - (1-m) \\log(1-m) Q(1-m),\n\n    where :math:`P` and :math:`Q` are tenth-order polynomials.  For\n    `m < 0`, the relation\n\n    .. math:: E(m) = E(m/(m - 1)) \\sqrt(1-m)\n\n    is used.\n\n    The parameterization in terms of :math:`m` follows that of section\n    17.2 in [2]_. Other parameterizations in terms of the\n    complementary parameter :math:`1 - m`, modular angle\n    :math:`\\sin^2(\\alpha) = m`, or modulus :math:`k^2 = m` are also\n    used, so be careful that you choose the correct parameter.\n\n    See Also\n    --------\n    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1\n    ellipk : Complete elliptic integral of the first kind\n    ellipkinc : Incomplete elliptic integral of the first kind\n    ellipeinc : Incomplete elliptic integral of the second kind\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n    ')
# Processing the call keyword arguments (line 1172)
kwargs_493545 = {}
# Getting the type of 'add_newdoc' (line 1172)
add_newdoc_493541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1172)
add_newdoc_call_result_493546 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 0), add_newdoc_493541, *[str_493542, str_493543, str_493544], **kwargs_493545)


# Call to add_newdoc(...): (line 1229)
# Processing the call arguments (line 1229)
str_493548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 11), 'str', 'scipy.special')
str_493549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 28), 'str', 'ellipeinc')
str_493550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1278, (-1)), 'str', '\n    ellipeinc(phi, m)\n\n    Incomplete elliptic integral of the second kind\n\n    This function is defined as\n\n    .. math:: E(\\phi, m) = \\int_0^{\\phi} [1 - m \\sin(t)^2]^{1/2} dt\n\n    Parameters\n    ----------\n    phi : array_like\n        amplitude of the elliptic integral.\n\n    m : array_like\n        parameter of the elliptic integral.\n\n    Returns\n    -------\n    E : ndarray\n        Value of the elliptic integral.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `ellie`.\n\n    Computation uses arithmetic-geometric means algorithm.\n\n    The parameterization in terms of :math:`m` follows that of section\n    17.2 in [2]_. Other parameterizations in terms of the\n    complementary parameter :math:`1 - m`, modular angle\n    :math:`\\sin^2(\\alpha) = m`, or modulus :math:`k^2 = m` are also\n    used, so be careful that you choose the correct parameter.\n\n    See Also\n    --------\n    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1\n    ellipk : Complete elliptic integral of the first kind\n    ellipkinc : Incomplete elliptic integral of the first kind\n    ellipe : Complete elliptic integral of the second kind\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n    ')
# Processing the call keyword arguments (line 1229)
kwargs_493551 = {}
# Getting the type of 'add_newdoc' (line 1229)
add_newdoc_493547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1229)
add_newdoc_call_result_493552 = invoke(stypy.reporting.localization.Localization(__file__, 1229, 0), add_newdoc_493547, *[str_493548, str_493549, str_493550], **kwargs_493551)


# Call to add_newdoc(...): (line 1280)
# Processing the call arguments (line 1280)
str_493554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1280, 11), 'str', 'scipy.special')
str_493555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1280, 28), 'str', 'ellipj')
str_493556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, (-1)), 'str', '\n    ellipj(u, m)\n\n    Jacobian elliptic functions\n\n    Calculates the Jacobian elliptic functions of parameter `m` between\n    0 and 1, and real argument `u`.\n\n    Parameters\n    ----------\n    m : array_like\n        Parameter.\n    u : array_like\n        Argument.\n\n    Returns\n    -------\n    sn, cn, dn, ph : ndarrays\n        The returned functions::\n\n            sn(u|m), cn(u|m), dn(u|m)\n\n        The value `ph` is such that if `u = ellipk(ph, m)`,\n        then `sn(u|m) = sin(ph)` and `cn(u|m) = cos(ph)`.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `ellpj`.\n\n    These functions are periodic, with quarter-period on the real axis\n    equal to the complete elliptic integral `ellipk(m)`.\n\n    Relation to incomplete elliptic integral: If `u = ellipk(phi,m)`, then\n    `sn(u|m) = sin(phi)`, and `cn(u|m) = cos(phi)`.  The `phi` is called\n    the amplitude of `u`.\n\n    Computation is by means of the arithmetic-geometric mean algorithm,\n    except when `m` is within 1e-9 of 0 or 1.  In the latter case with `m`\n    close to 1, the approximation applies only for `phi < pi/2`.\n\n    See also\n    --------\n    ellipk : Complete elliptic integral of the first kind.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 1280)
kwargs_493557 = {}
# Getting the type of 'add_newdoc' (line 1280)
add_newdoc_493553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1280, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1280)
add_newdoc_call_result_493558 = invoke(stypy.reporting.localization.Localization(__file__, 1280, 0), add_newdoc_493553, *[str_493554, str_493555, str_493556], **kwargs_493557)


# Call to add_newdoc(...): (line 1331)
# Processing the call arguments (line 1331)
str_493560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1331, 11), 'str', 'scipy.special')
str_493561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1331, 28), 'str', 'ellipkm1')
str_493562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1381, (-1)), 'str', '\n    ellipkm1(p)\n\n    Complete elliptic integral of the first kind around `m` = 1\n\n    This function is defined as\n\n    .. math:: K(p) = \\int_0^{\\pi/2} [1 - m \\sin(t)^2]^{-1/2} dt\n\n    where `m = 1 - p`.\n\n    Parameters\n    ----------\n    p : array_like\n        Defines the parameter of the elliptic integral as `m = 1 - p`.\n\n    Returns\n    -------\n    K : ndarray\n        Value of the elliptic integral.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `ellpk`.\n\n    For `p <= 1`, computation uses the approximation,\n\n    .. math:: K(p) \\approx P(p) - \\log(p) Q(p),\n\n    where :math:`P` and :math:`Q` are tenth-order polynomials.  The\n    argument `p` is used internally rather than `m` so that the logarithmic\n    singularity at `m = 1` will be shifted to the origin; this preserves\n    maximum accuracy.  For `p > 1`, the identity\n\n    .. math:: K(p) = K(1/p)/\\sqrt(p)\n\n    is used.\n\n    See Also\n    --------\n    ellipk : Complete elliptic integral of the first kind\n    ellipkinc : Incomplete elliptic integral of the first kind\n    ellipe : Complete elliptic integral of the second kind\n    ellipeinc : Incomplete elliptic integral of the second kind\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 1331)
kwargs_493563 = {}
# Getting the type of 'add_newdoc' (line 1331)
add_newdoc_493559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1331)
add_newdoc_call_result_493564 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 0), add_newdoc_493559, *[str_493560, str_493561, str_493562], **kwargs_493563)


# Call to add_newdoc(...): (line 1383)
# Processing the call arguments (line 1383)
str_493566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 11), 'str', 'scipy.special')
str_493567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1383, 28), 'str', 'ellipkinc')
str_493568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1433, (-1)), 'str', '\n    ellipkinc(phi, m)\n\n    Incomplete elliptic integral of the first kind\n\n    This function is defined as\n\n    .. math:: K(\\phi, m) = \\int_0^{\\phi} [1 - m \\sin(t)^2]^{-1/2} dt\n\n    This function is also called `F(phi, m)`.\n\n    Parameters\n    ----------\n    phi : array_like\n        amplitude of the elliptic integral\n\n    m : array_like\n        parameter of the elliptic integral\n\n    Returns\n    -------\n    K : ndarray\n        Value of the elliptic integral\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `ellik`.  The computation is\n    carried out using the arithmetic-geometric mean algorithm.\n\n    The parameterization in terms of :math:`m` follows that of section\n    17.2 in [2]_. Other parameterizations in terms of the\n    complementary parameter :math:`1 - m`, modular angle\n    :math:`\\sin^2(\\alpha) = m`, or modulus :math:`k^2 = m` are also\n    used, so be careful that you choose the correct parameter.\n\n    See Also\n    --------\n    ellipkm1 : Complete elliptic integral of the first kind, near `m` = 1\n    ellipk : Complete elliptic integral of the first kind\n    ellipe : Complete elliptic integral of the second kind\n    ellipeinc : Incomplete elliptic integral of the second kind\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n    ')
# Processing the call keyword arguments (line 1383)
kwargs_493569 = {}
# Getting the type of 'add_newdoc' (line 1383)
add_newdoc_493565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1383)
add_newdoc_call_result_493570 = invoke(stypy.reporting.localization.Localization(__file__, 1383, 0), add_newdoc_493565, *[str_493566, str_493567, str_493568], **kwargs_493569)


# Call to add_newdoc(...): (line 1435)
# Processing the call arguments (line 1435)
str_493572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1435, 11), 'str', 'scipy.special')
str_493573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1435, 28), 'str', 'entr')
str_493574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1463, (-1)), 'str', '\n    entr(x)\n\n    Elementwise function for computing entropy.\n\n    .. math:: \\text{entr}(x) = \\begin{cases} - x \\log(x) & x > 0  \\\\ 0 & x = 0 \\\\ -\\infty & \\text{otherwise} \\end{cases}\n\n    Parameters\n    ----------\n    x : ndarray\n        Input array.\n\n    Returns\n    -------\n    res : ndarray\n        The value of the elementwise entropy function at the given points `x`.\n\n    See Also\n    --------\n    kl_div, rel_entr\n\n    Notes\n    -----\n    This function is concave.\n\n    .. versionadded:: 0.15.0\n\n    ')
# Processing the call keyword arguments (line 1435)
kwargs_493575 = {}
# Getting the type of 'add_newdoc' (line 1435)
add_newdoc_493571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1435, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1435)
add_newdoc_call_result_493576 = invoke(stypy.reporting.localization.Localization(__file__, 1435, 0), add_newdoc_493571, *[str_493572, str_493573, str_493574], **kwargs_493575)


# Call to add_newdoc(...): (line 1465)
# Processing the call arguments (line 1465)
str_493578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1465, 11), 'str', 'scipy.special')
str_493579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1465, 28), 'str', 'erf')
str_493580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1512, (-1)), 'str', "\n    erf(z)\n\n    Returns the error function of complex argument.\n\n    It is defined as ``2/sqrt(pi)*integral(exp(-t**2), t=0..z)``.\n\n    Parameters\n    ----------\n    x : ndarray\n        Input array.\n\n    Returns\n    -------\n    res : ndarray\n        The values of the error function at the given points `x`.\n\n    See Also\n    --------\n    erfc, erfinv, erfcinv, wofz, erfcx, erfi\n\n    Notes\n    -----\n    The cumulative of the unit normal distribution is given by\n    ``Phi(z) = 1/2[1 + erf(z/sqrt(2))]``.\n\n    References\n    ----------\n    .. [1] http://en.wikipedia.org/wiki/Error_function\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n        Handbook of Mathematical Functions with Formulas,\n        Graphs, and Mathematical Tables. New York: Dover,\n        1972. http://www.math.sfu.ca/~cbm/aands/page_297.htm\n    .. [3] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-3, 3)\n    >>> plt.plot(x, special.erf(x))\n    >>> plt.xlabel('$x$')\n    >>> plt.ylabel('$erf(x)$')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 1465)
kwargs_493581 = {}
# Getting the type of 'add_newdoc' (line 1465)
add_newdoc_493577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1465)
add_newdoc_call_result_493582 = invoke(stypy.reporting.localization.Localization(__file__, 1465, 0), add_newdoc_493577, *[str_493578, str_493579, str_493580], **kwargs_493581)


# Call to add_newdoc(...): (line 1514)
# Processing the call arguments (line 1514)
str_493584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1514, 11), 'str', 'scipy.special')
str_493585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1514, 28), 'str', 'erfc')
str_493586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1539, (-1)), 'str', "\n    erfc(x)\n\n    Complementary error function, ``1 - erf(x)``.\n\n    See Also\n    --------\n    erf, erfi, erfcx, dawsn, wofz\n\n    References\n    ----------\n    .. [1] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-3, 3)\n    >>> plt.plot(x, special.erfc(x))\n    >>> plt.xlabel('$x$')\n    >>> plt.ylabel('$erfc(x)$')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 1514)
kwargs_493587 = {}
# Getting the type of 'add_newdoc' (line 1514)
add_newdoc_493583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1514)
add_newdoc_call_result_493588 = invoke(stypy.reporting.localization.Localization(__file__, 1514, 0), add_newdoc_493583, *[str_493584, str_493585, str_493586], **kwargs_493587)


# Call to add_newdoc(...): (line 1541)
# Processing the call arguments (line 1541)
str_493590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1541, 11), 'str', 'scipy.special')
str_493591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1541, 28), 'str', 'erfi')
str_493592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1571, (-1)), 'str', "\n    erfi(z)\n\n    Imaginary error function, ``-i erf(i z)``.\n\n    See Also\n    --------\n    erf, erfc, erfcx, dawsn, wofz\n\n    Notes\n    -----\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-3, 3)\n    >>> plt.plot(x, special.erfi(x))\n    >>> plt.xlabel('$x$')\n    >>> plt.ylabel('$erfi(x)$')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 1541)
kwargs_493593 = {}
# Getting the type of 'add_newdoc' (line 1541)
add_newdoc_493589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1541, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1541)
add_newdoc_call_result_493594 = invoke(stypy.reporting.localization.Localization(__file__, 1541, 0), add_newdoc_493589, *[str_493590, str_493591, str_493592], **kwargs_493593)


# Call to add_newdoc(...): (line 1573)
# Processing the call arguments (line 1573)
str_493596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 11), 'str', 'scipy.special')
str_493597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, 28), 'str', 'erfcx')
str_493598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, (-1)), 'str', "\n    erfcx(x)\n\n    Scaled complementary error function, ``exp(x**2) * erfc(x)``.\n\n    See Also\n    --------\n    erf, erfc, erfi, dawsn, wofz\n\n    Notes\n    -----\n\n    .. versionadded:: 0.12.0\n\n    References\n    ----------\n    .. [1] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-3, 3)\n    >>> plt.plot(x, special.erfcx(x))\n    >>> plt.xlabel('$x$')\n    >>> plt.ylabel('$erfcx(x)$')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 1573)
kwargs_493599 = {}
# Getting the type of 'add_newdoc' (line 1573)
add_newdoc_493595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1573, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1573)
add_newdoc_call_result_493600 = invoke(stypy.reporting.localization.Localization(__file__, 1573, 0), add_newdoc_493595, *[str_493596, str_493597, str_493598], **kwargs_493599)


# Call to add_newdoc(...): (line 1605)
# Processing the call arguments (line 1605)
str_493602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 11), 'str', 'scipy.special')
str_493603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 28), 'str', 'eval_jacobi')
str_493604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1646, (-1)), 'str', '\n    eval_jacobi(n, alpha, beta, x, out=None)\n\n    Evaluate Jacobi polynomial at a point.\n\n    The Jacobi polynomials can be defined via the Gauss hypergeometric\n    function :math:`{}_2F_1` as\n\n    .. math::\n\n        P_n^{(\\alpha, \\beta)}(x) = \\frac{(\\alpha + 1)_n}{\\Gamma(n + 1)}\n          {}_2F_1(-n, 1 + \\alpha + \\beta + n; \\alpha + 1; (1 - z)/2)\n\n    where :math:`(\\cdot)_n` is the Pochhammer symbol; see `poch`. When\n    :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer the result is\n        determined via the relation to the Gauss hypergeometric\n        function.\n    alpha : array_like\n        Parameter\n    beta : array_like\n        Parameter\n    x : array_like\n        Points at which to evaluate the polynomial\n\n    Returns\n    -------\n    P : ndarray\n        Values of the Jacobi polynomial\n\n    See Also\n    --------\n    roots_jacobi : roots and quadrature weights of Jacobi polynomials\n    jacobi : Jacobi polynomial object\n    hyp2f1 : Gauss hypergeometric function\n    ')
# Processing the call keyword arguments (line 1605)
kwargs_493605 = {}
# Getting the type of 'add_newdoc' (line 1605)
add_newdoc_493601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1605)
add_newdoc_call_result_493606 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 0), add_newdoc_493601, *[str_493602, str_493603, str_493604], **kwargs_493605)


# Call to add_newdoc(...): (line 1648)
# Processing the call arguments (line 1648)
str_493608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 11), 'str', 'scipy.special')
str_493609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1648, 28), 'str', 'eval_sh_jacobi')
str_493610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1684, (-1)), 'str', '\n    eval_sh_jacobi(n, p, q, x, out=None)\n\n    Evaluate shifted Jacobi polynomial at a point.\n\n    Defined by\n\n    .. math::\n\n        G_n^{(p, q)}(x)\n          = \\binom{2n + p - 1}{n}^{-1} P_n^{(p - q, q - 1)}(2x - 1),\n\n    where :math:`P_n^{(\\cdot, \\cdot)}` is the n-th Jacobi polynomial.\n\n    Parameters\n    ----------\n    n : int\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to `binom` and `eval_jacobi`.\n    p : float\n        Parameter\n    q : float\n        Parameter\n\n    Returns\n    -------\n    G : ndarray\n        Values of the shifted Jacobi polynomial.\n\n    See Also\n    --------\n    roots_sh_jacobi : roots and quadrature weights of shifted Jacobi\n                      polynomials\n    sh_jacobi : shifted Jacobi polynomial object\n    eval_jacobi : evaluate Jacobi polynomials\n    ')
# Processing the call keyword arguments (line 1648)
kwargs_493611 = {}
# Getting the type of 'add_newdoc' (line 1648)
add_newdoc_493607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1648, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1648)
add_newdoc_call_result_493612 = invoke(stypy.reporting.localization.Localization(__file__, 1648, 0), add_newdoc_493607, *[str_493608, str_493609, str_493610], **kwargs_493611)


# Call to add_newdoc(...): (line 1686)
# Processing the call arguments (line 1686)
str_493614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1686, 11), 'str', 'scipy.special')
str_493615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1686, 28), 'str', 'eval_gegenbauer')
str_493616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1725, (-1)), 'str', '\n    eval_gegenbauer(n, alpha, x, out=None)\n\n    Evaluate Gegenbauer polynomial at a point.\n\n    The Gegenbauer polynomials can be defined via the Gauss\n    hypergeometric function :math:`{}_2F_1` as\n\n    .. math::\n\n        C_n^{(\\alpha)} = \\frac{(2\\alpha)_n}{\\Gamma(n + 1)}\n          {}_2F_1(-n, 2\\alpha + n; \\alpha + 1/2; (1 - z)/2).\n\n    When :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to the Gauss hypergeometric\n        function.\n    alpha : array_like\n        Parameter\n    x : array_like\n        Points at which to evaluate the Gegenbauer polynomial\n\n    Returns\n    -------\n    C : ndarray\n        Values of the Gegenbauer polynomial\n\n    See Also\n    --------\n    roots_gegenbauer : roots and quadrature weights of Gegenbauer\n                       polynomials\n    gegenbauer : Gegenbauer polynomial object\n    hyp2f1 : Gauss hypergeometric function\n    ')
# Processing the call keyword arguments (line 1686)
kwargs_493617 = {}
# Getting the type of 'add_newdoc' (line 1686)
add_newdoc_493613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1686, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1686)
add_newdoc_call_result_493618 = invoke(stypy.reporting.localization.Localization(__file__, 1686, 0), add_newdoc_493613, *[str_493614, str_493615, str_493616], **kwargs_493617)


# Call to add_newdoc(...): (line 1727)
# Processing the call arguments (line 1727)
str_493620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1727, 11), 'str', 'scipy.special')
str_493621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1727, 28), 'str', 'eval_chebyt')
str_493622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1770, (-1)), 'str', '\n    eval_chebyt(n, x, out=None)\n\n    Evaluate Chebyshev polynomial of the first kind at a point.\n\n    The Chebyshev polynomials of the first kind can be defined via the\n    Gauss hypergeometric function :math:`{}_2F_1` as\n\n    .. math::\n\n        T_n(x) = {}_2F_1(n, -n; 1/2; (1 - x)/2).\n\n    When :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to the Gauss hypergeometric\n        function.\n    x : array_like\n        Points at which to evaluate the Chebyshev polynomial\n\n    Returns\n    -------\n    T : ndarray\n        Values of the Chebyshev polynomial\n\n    See Also\n    --------\n    roots_chebyt : roots and quadrature weights of Chebyshev\n                   polynomials of the first kind\n    chebyu : Chebychev polynomial object\n    eval_chebyu : evaluate Chebyshev polynomials of the second kind\n    hyp2f1 : Gauss hypergeometric function\n    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series\n\n    Notes\n    -----\n    This routine is numerically stable for `x` in ``[-1, 1]`` at least\n    up to order ``10000``.\n    ')
# Processing the call keyword arguments (line 1727)
kwargs_493623 = {}
# Getting the type of 'add_newdoc' (line 1727)
add_newdoc_493619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1727, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1727)
add_newdoc_call_result_493624 = invoke(stypy.reporting.localization.Localization(__file__, 1727, 0), add_newdoc_493619, *[str_493620, str_493621, str_493622], **kwargs_493623)


# Call to add_newdoc(...): (line 1772)
# Processing the call arguments (line 1772)
str_493626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1772, 11), 'str', 'scipy.special')
str_493627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1772, 28), 'str', 'eval_chebyu')
str_493628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1809, (-1)), 'str', '\n    eval_chebyu(n, x, out=None)\n\n    Evaluate Chebyshev polynomial of the second kind at a point.\n\n    The Chebyshev polynomials of the second kind can be defined via\n    the Gauss hypergeometric function :math:`{}_2F_1` as\n\n    .. math::\n\n        U_n(x) = (n + 1) {}_2F_1(-n, n + 2; 3/2; (1 - x)/2).\n\n    When :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to the Gauss hypergeometric\n        function.\n    x : array_like\n        Points at which to evaluate the Chebyshev polynomial\n\n    Returns\n    -------\n    U : ndarray\n        Values of the Chebyshev polynomial\n\n    See Also\n    --------\n    roots_chebyu : roots and quadrature weights of Chebyshev\n                   polynomials of the second kind\n    chebyu : Chebyshev polynomial object\n    eval_chebyt : evaluate Chebyshev polynomials of the first kind\n    hyp2f1 : Gauss hypergeometric function\n    ')
# Processing the call keyword arguments (line 1772)
kwargs_493629 = {}
# Getting the type of 'add_newdoc' (line 1772)
add_newdoc_493625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1772, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1772)
add_newdoc_call_result_493630 = invoke(stypy.reporting.localization.Localization(__file__, 1772, 0), add_newdoc_493625, *[str_493626, str_493627, str_493628], **kwargs_493629)


# Call to add_newdoc(...): (line 1811)
# Processing the call arguments (line 1811)
str_493632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1811, 11), 'str', 'scipy.special')
str_493633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1811, 28), 'str', 'eval_chebys')
str_493634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1845, (-1)), 'str', '\n    eval_chebys(n, x, out=None)\n\n    Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a\n    point.\n\n    These polynomials are defined as\n\n    .. math::\n\n        S_n(x) = U_n(x/2)\n\n    where :math:`U_n` is a Chebyshev polynomial of the second kind.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to `eval_chebyu`.\n    x : array_like\n        Points at which to evaluate the Chebyshev polynomial\n\n    Returns\n    -------\n    S : ndarray\n        Values of the Chebyshev polynomial\n\n    See Also\n    --------\n    roots_chebys : roots and quadrature weights of Chebyshev\n                   polynomials of the second kind on [-2, 2]\n    chebys : Chebyshev polynomial object\n    eval_chebyu : evaluate Chebyshev polynomials of the second kind\n    ')
# Processing the call keyword arguments (line 1811)
kwargs_493635 = {}
# Getting the type of 'add_newdoc' (line 1811)
add_newdoc_493631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1811, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1811)
add_newdoc_call_result_493636 = invoke(stypy.reporting.localization.Localization(__file__, 1811, 0), add_newdoc_493631, *[str_493632, str_493633, str_493634], **kwargs_493635)


# Call to add_newdoc(...): (line 1847)
# Processing the call arguments (line 1847)
str_493638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1847, 11), 'str', 'scipy.special')
str_493639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1847, 28), 'str', 'eval_chebyc')
str_493640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1882, (-1)), 'str', '\n    eval_chebyc(n, x, out=None)\n\n    Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a\n    point.\n\n    These polynomials are defined as\n\n    .. math::\n\n        S_n(x) = T_n(x/2)\n\n    where :math:`T_n` is a Chebyshev polynomial of the first kind.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to `eval_chebyt`.\n    x : array_like\n        Points at which to evaluate the Chebyshev polynomial\n\n    Returns\n    -------\n    C : ndarray\n        Values of the Chebyshev polynomial\n\n    See Also\n    --------\n    roots_chebyc : roots and quadrature weights of Chebyshev\n                   polynomials of the first kind on [-2, 2]\n    chebyc : Chebyshev polynomial object\n    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series\n    eval_chebyt : evaluate Chebycshev polynomials of the first kind\n    ')
# Processing the call keyword arguments (line 1847)
kwargs_493641 = {}
# Getting the type of 'add_newdoc' (line 1847)
add_newdoc_493637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1847, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1847)
add_newdoc_call_result_493642 = invoke(stypy.reporting.localization.Localization(__file__, 1847, 0), add_newdoc_493637, *[str_493638, str_493639, str_493640], **kwargs_493641)


# Call to add_newdoc(...): (line 1884)
# Processing the call arguments (line 1884)
str_493644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1884, 11), 'str', 'scipy.special')
str_493645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1884, 28), 'str', 'eval_sh_chebyt')
str_493646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1919, (-1)), 'str', '\n    eval_sh_chebyt(n, x, out=None)\n\n    Evaluate shifted Chebyshev polynomial of the first kind at a\n    point.\n\n    These polynomials are defined as\n\n    .. math::\n\n        T_n^*(x) = T_n(2x - 1)\n\n    where :math:`T_n` is a Chebyshev polynomial of the first kind.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to `eval_chebyt`.\n    x : array_like\n        Points at which to evaluate the shifted Chebyshev polynomial\n\n    Returns\n    -------\n    T : ndarray\n        Values of the shifted Chebyshev polynomial\n\n    See Also\n    --------\n    roots_sh_chebyt : roots and quadrature weights of shifted\n                      Chebyshev polynomials of the first kind\n    sh_chebyt : shifted Chebyshev polynomial object\n    eval_chebyt : evaluate Chebyshev polynomials of the first kind\n    numpy.polynomial.chebyshev.Chebyshev : Chebyshev series\n    ')
# Processing the call keyword arguments (line 1884)
kwargs_493647 = {}
# Getting the type of 'add_newdoc' (line 1884)
add_newdoc_493643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1884, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1884)
add_newdoc_call_result_493648 = invoke(stypy.reporting.localization.Localization(__file__, 1884, 0), add_newdoc_493643, *[str_493644, str_493645, str_493646], **kwargs_493647)


# Call to add_newdoc(...): (line 1921)
# Processing the call arguments (line 1921)
str_493650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1921, 11), 'str', 'scipy.special')
str_493651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1921, 28), 'str', 'eval_sh_chebyu')
str_493652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1955, (-1)), 'str', '\n    eval_sh_chebyu(n, x, out=None)\n\n    Evaluate shifted Chebyshev polynomial of the second kind at a\n    point.\n\n    These polynomials are defined as\n\n    .. math::\n\n        U_n^*(x) = U_n(2x - 1)\n\n    where :math:`U_n` is a Chebyshev polynomial of the first kind.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to `eval_chebyu`.\n    x : array_like\n        Points at which to evaluate the shifted Chebyshev polynomial\n\n    Returns\n    -------\n    U : ndarray\n        Values of the shifted Chebyshev polynomial\n\n    See Also\n    --------\n    roots_sh_chebyu : roots and quadrature weights of shifted\n                      Chebychev polynomials of the second kind\n    sh_chebyu : shifted Chebyshev polynomial object\n    eval_chebyu : evaluate Chebyshev polynomials of the second kind\n    ')
# Processing the call keyword arguments (line 1921)
kwargs_493653 = {}
# Getting the type of 'add_newdoc' (line 1921)
add_newdoc_493649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1921, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1921)
add_newdoc_call_result_493654 = invoke(stypy.reporting.localization.Localization(__file__, 1921, 0), add_newdoc_493649, *[str_493650, str_493651, str_493652], **kwargs_493653)


# Call to add_newdoc(...): (line 1957)
# Processing the call arguments (line 1957)
str_493656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1957, 11), 'str', 'scipy.special')
str_493657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1957, 28), 'str', 'eval_legendre')
str_493658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1994, (-1)), 'str', '\n    eval_legendre(n, x, out=None)\n\n    Evaluate Legendre polynomial at a point.\n\n    The Legendre polynomials can be defined via the Gauss\n    hypergeometric function :math:`{}_2F_1` as\n\n    .. math::\n\n        P_n(x) = {}_2F_1(-n, n + 1; 1; (1 - x)/2).\n\n    When :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the result is\n        determined via the relation to the Gauss hypergeometric\n        function.\n    x : array_like\n        Points at which to evaluate the Legendre polynomial\n\n    Returns\n    -------\n    P : ndarray\n        Values of the Legendre polynomial\n\n    See Also\n    --------\n    roots_legendre : roots and quadrature weights of Legendre\n                     polynomials\n    legendre : Legendre polynomial object\n    hyp2f1 : Gauss hypergeometric function\n    numpy.polynomial.legendre.Legendre : Legendre series\n    ')
# Processing the call keyword arguments (line 1957)
kwargs_493659 = {}
# Getting the type of 'add_newdoc' (line 1957)
add_newdoc_493655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1957, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1957)
add_newdoc_call_result_493660 = invoke(stypy.reporting.localization.Localization(__file__, 1957, 0), add_newdoc_493655, *[str_493656, str_493657, str_493658], **kwargs_493659)


# Call to add_newdoc(...): (line 1996)
# Processing the call arguments (line 1996)
str_493662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1996, 11), 'str', 'scipy.special')
str_493663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1996, 28), 'str', 'eval_sh_legendre')
str_493664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2030, (-1)), 'str', '\n    eval_sh_legendre(n, x, out=None)\n\n    Evaluate shifted Legendre polynomial at a point.\n\n    These polynomials are defined as\n\n    .. math::\n\n        P_n^*(x) = P_n(2x - 1)\n\n    where :math:`P_n` is a Legendre polynomial.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer, the value is\n        determined via the relation to `eval_legendre`.\n    x : array_like\n        Points at which to evaluate the shifted Legendre polynomial\n\n    Returns\n    -------\n    P : ndarray\n        Values of the shifted Legendre polynomial\n\n    See Also\n    --------\n    roots_sh_legendre : roots and quadrature weights of shifted\n                        Legendre polynomials\n    sh_legendre : shifted Legendre polynomial object\n    eval_legendre : evaluate Legendre polynomials\n    numpy.polynomial.legendre.Legendre : Legendre series\n    ')
# Processing the call keyword arguments (line 1996)
kwargs_493665 = {}
# Getting the type of 'add_newdoc' (line 1996)
add_newdoc_493661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1996, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 1996)
add_newdoc_call_result_493666 = invoke(stypy.reporting.localization.Localization(__file__, 1996, 0), add_newdoc_493661, *[str_493662, str_493663, str_493664], **kwargs_493665)


# Call to add_newdoc(...): (line 2032)
# Processing the call arguments (line 2032)
str_493668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2032, 11), 'str', 'scipy.special')
str_493669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2032, 28), 'str', 'eval_genlaguerre')
str_493670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2074, (-1)), 'str', '\n    eval_genlaguerre(n, alpha, x, out=None)\n\n    Evaluate generalized Laguerre polynomial at a point.\n\n    The generalized Laguerre polynomials can be defined via the\n    confluent hypergeometric function :math:`{}_1F_1` as\n\n    .. math::\n\n        L_n^{(\\alpha)}(x) = \\binom{n + \\alpha}{n}\n          {}_1F_1(-n, \\alpha + 1, x).\n\n    When :math:`n` is an integer the result is a polynomial of degree\n    :math:`n`. The Laguerre polynomials are the special case where\n    :math:`\\alpha = 0`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial. If not an integer the result is\n        determined via the relation to the confluent hypergeometric\n        function.\n    alpha : array_like\n        Parameter; must have ``alpha > -1``\n    x : array_like\n        Points at which to evaluate the generalized Laguerre\n        polynomial\n\n    Returns\n    -------\n    L : ndarray\n        Values of the generalized Laguerre polynomial\n\n    See Also\n    --------\n    roots_genlaguerre : roots and quadrature weights of generalized\n                        Laguerre polynomials\n    genlaguerre : generalized Laguerre polynomial object\n    hyp1f1 : confluent hypergeometric function\n    eval_laguerre : evaluate Laguerre polynomials\n    ')
# Processing the call keyword arguments (line 2032)
kwargs_493671 = {}
# Getting the type of 'add_newdoc' (line 2032)
add_newdoc_493667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2032, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2032)
add_newdoc_call_result_493672 = invoke(stypy.reporting.localization.Localization(__file__, 2032, 0), add_newdoc_493667, *[str_493668, str_493669, str_493670], **kwargs_493671)


# Call to add_newdoc(...): (line 2076)
# Processing the call arguments (line 2076)
str_493674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2076, 11), 'str', 'scipy.special')
str_493675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2076, 28), 'str', 'eval_laguerre')
str_493676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2113, (-1)), 'str', '\n     eval_laguerre(n, x, out=None)\n\n     Evaluate Laguerre polynomial at a point.\n\n     The Laguerre polynomials can be defined via the confluent\n     hypergeometric function :math:`{}_1F_1` as\n\n     .. math::\n\n         L_n(x) = {}_1F_1(-n, 1, x).\n\n     When :math:`n` is an integer the result is a polynomial of degree\n     :math:`n`.\n\n     Parameters\n     ----------\n     n : array_like\n         Degree of the polynomial. If not an integer the result is\n         determined via the relation to the confluent hypergeometric\n         function.\n     x : array_like\n         Points at which to evaluate the Laguerre polynomial\n\n     Returns\n     -------\n     L : ndarray\n         Values of the Laguerre polynomial\n\n     See Also\n     --------\n     roots_laguerre : roots and quadrature weights of Laguerre\n                      polynomials\n     laguerre : Laguerre polynomial object\n     numpy.polynomial.laguerre.Laguerre : Laguerre series\n     eval_genlaguerre : evaluate generalized Laguerre polynomials\n     ')
# Processing the call keyword arguments (line 2076)
kwargs_493677 = {}
# Getting the type of 'add_newdoc' (line 2076)
add_newdoc_493673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2076, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2076)
add_newdoc_call_result_493678 = invoke(stypy.reporting.localization.Localization(__file__, 2076, 0), add_newdoc_493673, *[str_493674, str_493675, str_493676], **kwargs_493677)


# Call to add_newdoc(...): (line 2115)
# Processing the call arguments (line 2115)
str_493680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2115, 11), 'str', 'scipy.special')
str_493681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2115, 28), 'str', 'eval_hermite')
str_493682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2148, (-1)), 'str', "\n    eval_hermite(n, x, out=None)\n\n    Evaluate physicist's Hermite polynomial at a point.\n\n    Defined by\n\n    .. math::\n\n        H_n(x) = (-1)^n e^{x^2} \\frac{d^n}{dx^n} e^{-x^2};\n\n    :math:`H_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial\n    x : array_like\n        Points at which to evaluate the Hermite polynomial\n\n    Returns\n    -------\n    H : ndarray\n        Values of the Hermite polynomial\n\n    See Also\n    --------\n    roots_hermite : roots and quadrature weights of physicist's\n                    Hermite polynomials\n    hermite : physicist's Hermite polynomial object\n    numpy.polynomial.hermite.Hermite : Physicist's Hermite series\n    eval_hermitenorm : evaluate Probabilist's Hermite polynomials\n    ")
# Processing the call keyword arguments (line 2115)
kwargs_493683 = {}
# Getting the type of 'add_newdoc' (line 2115)
add_newdoc_493679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2115, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2115)
add_newdoc_call_result_493684 = invoke(stypy.reporting.localization.Localization(__file__, 2115, 0), add_newdoc_493679, *[str_493680, str_493681, str_493682], **kwargs_493683)


# Call to add_newdoc(...): (line 2150)
# Processing the call arguments (line 2150)
str_493686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2150, 11), 'str', 'scipy.special')
str_493687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2150, 28), 'str', 'eval_hermitenorm')
str_493688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2184, (-1)), 'str', "\n    eval_hermitenorm(n, x, out=None)\n\n    Evaluate probabilist's (normalized) Hermite polynomial at a\n    point.\n\n    Defined by\n\n    .. math::\n\n        He_n(x) = (-1)^n e^{x^2/2} \\frac{d^n}{dx^n} e^{-x^2/2};\n\n    :math:`He_n` is a polynomial of degree :math:`n`.\n\n    Parameters\n    ----------\n    n : array_like\n        Degree of the polynomial\n    x : array_like\n        Points at which to evaluate the Hermite polynomial\n\n    Returns\n    -------\n    He : ndarray\n        Values of the Hermite polynomial\n\n    See Also\n    --------\n    roots_hermitenorm : roots and quadrature weights of probabilist's\n                        Hermite polynomials\n    hermitenorm : probabilist's Hermite polynomial object\n    numpy.polynomial.hermite_e.HermiteE : Probabilist's Hermite series\n    eval_hermite : evaluate physicist's Hermite polynomials\n    ")
# Processing the call keyword arguments (line 2150)
kwargs_493689 = {}
# Getting the type of 'add_newdoc' (line 2150)
add_newdoc_493685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2150, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2150)
add_newdoc_call_result_493690 = invoke(stypy.reporting.localization.Localization(__file__, 2150, 0), add_newdoc_493685, *[str_493686, str_493687, str_493688], **kwargs_493689)


# Call to add_newdoc(...): (line 2186)
# Processing the call arguments (line 2186)
str_493692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2186, 11), 'str', 'scipy.special')
str_493693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2186, 28), 'str', 'exp1')
str_493694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2195, (-1)), 'str', '\n    exp1(z)\n\n    Exponential integral E_1 of complex argument z\n\n    ::\n\n        integral(exp(-z*t)/t, t=1..inf).\n    ')
# Processing the call keyword arguments (line 2186)
kwargs_493695 = {}
# Getting the type of 'add_newdoc' (line 2186)
add_newdoc_493691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2186, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2186)
add_newdoc_call_result_493696 = invoke(stypy.reporting.localization.Localization(__file__, 2186, 0), add_newdoc_493691, *[str_493692, str_493693, str_493694], **kwargs_493695)


# Call to add_newdoc(...): (line 2197)
# Processing the call arguments (line 2197)
str_493698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2197, 11), 'str', 'scipy.special')
str_493699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2197, 28), 'str', 'exp10')
str_493700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2224, (-1)), 'str', '\n    exp10(x)\n\n    Compute ``10**x`` element-wise.\n\n    Parameters\n    ----------\n    x : array_like\n        `x` must contain real numbers.\n\n    Returns\n    -------\n    float\n        ``10**x``, computed element-wise.\n\n    Examples\n    --------\n    >>> from scipy.special import exp10\n\n    >>> exp10(3)\n    1000.0\n    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])\n    >>> exp10(x)\n    array([[  0.1       ,   0.31622777,   1.        ],\n           [  3.16227766,  10.        ,  31.6227766 ]])\n\n    ')
# Processing the call keyword arguments (line 2197)
kwargs_493701 = {}
# Getting the type of 'add_newdoc' (line 2197)
add_newdoc_493697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2197, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2197)
add_newdoc_call_result_493702 = invoke(stypy.reporting.localization.Localization(__file__, 2197, 0), add_newdoc_493697, *[str_493698, str_493699, str_493700], **kwargs_493701)


# Call to add_newdoc(...): (line 2226)
# Processing the call arguments (line 2226)
str_493704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2226, 11), 'str', 'scipy.special')
str_493705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2226, 28), 'str', 'exp2')
str_493706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2252, (-1)), 'str', '\n    exp2(x)\n\n    Compute ``2**x`` element-wise.\n\n    Parameters\n    ----------\n    x : array_like\n        `x` must contain real numbers.\n\n    Returns\n    -------\n    float\n        ``2**x``, computed element-wise.\n\n    Examples\n    --------\n    >>> from scipy.special import exp2\n\n    >>> exp2(3)\n    8.0\n    >>> x = np.array([[-1, -0.5, 0], [0.5, 1, 1.5]])\n    >>> exp2(x)\n    array([[ 0.5       ,  0.70710678,  1.        ],\n           [ 1.41421356,  2.        ,  2.82842712]])\n    ')
# Processing the call keyword arguments (line 2226)
kwargs_493707 = {}
# Getting the type of 'add_newdoc' (line 2226)
add_newdoc_493703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2226, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2226)
add_newdoc_call_result_493708 = invoke(stypy.reporting.localization.Localization(__file__, 2226, 0), add_newdoc_493703, *[str_493704, str_493705, str_493706], **kwargs_493707)


# Call to add_newdoc(...): (line 2254)
# Processing the call arguments (line 2254)
str_493710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2254, 11), 'str', 'scipy.special')
str_493711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2254, 28), 'str', 'expi')
str_493712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2265, (-1)), 'str', '\n    expi(x)\n\n    Exponential integral Ei\n\n    Defined as::\n\n        integral(exp(t)/t, t=-inf..x)\n\n    See `expn` for a different exponential integral.\n    ')
# Processing the call keyword arguments (line 2254)
kwargs_493713 = {}
# Getting the type of 'add_newdoc' (line 2254)
add_newdoc_493709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2254, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2254)
add_newdoc_call_result_493714 = invoke(stypy.reporting.localization.Localization(__file__, 2254, 0), add_newdoc_493709, *[str_493710, str_493711, str_493712], **kwargs_493713)


# Call to add_newdoc(...): (line 2267)
# Processing the call arguments (line 2267)
str_493716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2267, 11), 'str', 'scipy.special')
str_493717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2267, 28), 'str', 'expit')
str_493718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2323, (-1)), 'str', "\n    expit(x)\n\n    Expit ufunc for ndarrays.\n\n    The expit function, also known as the logistic function, is defined as\n    expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.\n\n    Parameters\n    ----------\n    x : ndarray\n        The ndarray to apply expit to element-wise.\n\n    Returns\n    -------\n    out : ndarray\n        An ndarray of the same shape as x. Its entries\n        are expit of the corresponding entry of x.\n\n    See Also\n    --------\n    logit\n\n    Notes\n    -----\n    As a ufunc expit takes a number of optional\n    keyword arguments. For more information\n    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_\n\n    .. versionadded:: 0.10.0\n\n    Examples\n    --------\n    >>> from scipy.special import expit, logit\n\n    >>> expit([-np.inf, -1.5, 0, 1.5, np.inf])\n    array([ 0.        ,  0.18242552,  0.5       ,  0.81757448,  1.        ])\n\n    `logit` is the inverse of `expit`:\n\n    >>> logit(expit([-2.5, 0, 3.1, 5.0]))\n    array([-2.5,  0. ,  3.1,  5. ])\n\n    Plot expit(x) for x in [-6, 6]:\n\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(-6, 6, 121)\n    >>> y = expit(x)\n    >>> plt.plot(x, y)\n    >>> plt.grid()\n    >>> plt.xlim(-6, 6)\n    >>> plt.xlabel('x')\n    >>> plt.title('expit(x)')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 2267)
kwargs_493719 = {}
# Getting the type of 'add_newdoc' (line 2267)
add_newdoc_493715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2267, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2267)
add_newdoc_call_result_493720 = invoke(stypy.reporting.localization.Localization(__file__, 2267, 0), add_newdoc_493715, *[str_493716, str_493717, str_493718], **kwargs_493719)


# Call to add_newdoc(...): (line 2325)
# Processing the call arguments (line 2325)
str_493722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2325, 11), 'str', 'scipy.special')
str_493723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2325, 28), 'str', 'expm1')
str_493724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2370, (-1)), 'str', '\n    expm1(x)\n\n    Compute ``exp(x) - 1``.\n\n    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation\n    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.\n    ``expm1(x)`` is implemented to avoid the loss of precision that occurs when\n    `x` is near zero.\n\n    Parameters\n    ----------\n    x : array_like\n        `x` must contain real numbers.\n\n    Returns\n    -------\n    float\n        ``exp(x) - 1`` computed element-wise.\n\n    Examples\n    --------\n    >>> from scipy.special import expm1\n\n    >>> expm1(1.0)\n    1.7182818284590451\n    >>> expm1([-0.2, -0.1, 0, 0.1, 0.2])\n    array([-0.18126925, -0.09516258,  0.        ,  0.10517092,  0.22140276])\n\n    The exact value of ``exp(7.5e-13) - 1`` is::\n\n        7.5000000000028125000000007031250000001318...*10**-13.\n\n    Here is what ``expm1(7.5e-13)`` gives:\n\n    >>> expm1(7.5e-13)\n    7.5000000000028135e-13\n\n    Compare that to ``exp(7.5e-13) - 1``, where the subtraction results in\n    a "catastrophic" loss of precision:\n\n    >>> np.exp(7.5e-13) - 1\n    7.5006667543675576e-13\n\n    ')
# Processing the call keyword arguments (line 2325)
kwargs_493725 = {}
# Getting the type of 'add_newdoc' (line 2325)
add_newdoc_493721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2325, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2325)
add_newdoc_call_result_493726 = invoke(stypy.reporting.localization.Localization(__file__, 2325, 0), add_newdoc_493721, *[str_493722, str_493723, str_493724], **kwargs_493725)


# Call to add_newdoc(...): (line 2372)
# Processing the call arguments (line 2372)
str_493728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2372, 11), 'str', 'scipy.special')
str_493729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2372, 28), 'str', 'expn')
str_493730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2382, (-1)), 'str', '\n    expn(n, x)\n\n    Exponential integral E_n\n\n    Returns the exponential integral for integer `n` and non-negative `x` and\n    `n`::\n\n        integral(exp(-x*t) / t**n, t=1..inf).\n    ')
# Processing the call keyword arguments (line 2372)
kwargs_493731 = {}
# Getting the type of 'add_newdoc' (line 2372)
add_newdoc_493727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2372, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2372)
add_newdoc_call_result_493732 = invoke(stypy.reporting.localization.Localization(__file__, 2372, 0), add_newdoc_493727, *[str_493728, str_493729, str_493730], **kwargs_493731)


# Call to add_newdoc(...): (line 2384)
# Processing the call arguments (line 2384)
str_493734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2384, 11), 'str', 'scipy.special')
str_493735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2384, 28), 'str', 'exprel')
str_493736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2430, (-1)), 'str', '\n    exprel(x)\n\n    Relative error exponential, ``(exp(x) - 1)/x``.\n\n    When `x` is near zero, ``exp(x)`` is near 1, so the numerical calculation\n    of ``exp(x) - 1`` can suffer from catastrophic loss of precision.\n    ``exprel(x)`` is implemented to avoid the loss of precision that occurs when\n    `x` is near zero.\n\n    Parameters\n    ----------\n    x : ndarray\n        Input array.  `x` must contain real numbers.\n\n    Returns\n    -------\n    float\n        ``(exp(x) - 1)/x``, computed element-wise.\n\n    See Also\n    --------\n    expm1\n\n    Notes\n    -----\n    .. versionadded:: 0.17.0\n\n    Examples\n    --------\n    >>> from scipy.special import exprel\n\n    >>> exprel(0.01)\n    1.0050167084168056\n    >>> exprel([-0.25, -0.1, 0, 0.1, 0.25])\n    array([ 0.88479687,  0.95162582,  1.        ,  1.05170918,  1.13610167])\n\n    Compare ``exprel(5e-9)`` to the naive calculation.  The exact value\n    is ``1.00000000250000000416...``.\n\n    >>> exprel(5e-9)\n    1.0000000025\n\n    >>> (np.exp(5e-9) - 1)/5e-9\n    0.99999999392252903\n    ')
# Processing the call keyword arguments (line 2384)
kwargs_493737 = {}
# Getting the type of 'add_newdoc' (line 2384)
add_newdoc_493733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2384, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2384)
add_newdoc_call_result_493738 = invoke(stypy.reporting.localization.Localization(__file__, 2384, 0), add_newdoc_493733, *[str_493734, str_493735, str_493736], **kwargs_493737)


# Call to add_newdoc(...): (line 2432)
# Processing the call arguments (line 2432)
str_493740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2432, 11), 'str', 'scipy.special')
str_493741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2432, 28), 'str', 'fdtr')
str_493742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2481, (-1)), 'str', "\n    fdtr(dfn, dfd, x)\n\n    F cumulative distribution function.\n\n    Returns the value of the cumulative density function of the\n    F-distribution, also known as Snedecor's F-distribution or the\n    Fisher-Snedecor distribution.\n\n    The F-distribution with parameters :math:`d_n` and :math:`d_d` is the\n    distribution of the random variable,\n\n    .. math::\n        X = \\frac{U_n/d_n}{U_d/d_d},\n\n    where :math:`U_n` and :math:`U_d` are random variables distributed\n    :math:`\\chi^2`, with :math:`d_n` and :math:`d_d` degrees of freedom,\n    respectively.\n\n    Parameters\n    ----------\n    dfn : array_like\n        First parameter (positive float).\n    dfd : array_like\n        Second parameter (positive float).\n    x : array_like\n        Argument (nonnegative float).\n\n    Returns\n    -------\n    y : ndarray\n        The CDF of the F-distribution with parameters `dfn` and `dfd` at `x`.\n\n    Notes\n    -----\n    The regularized incomplete beta function is used, according to the\n    formula,\n\n    .. math::\n        F(d_n, d_d; x) = I_{xd_n/(d_d + xd_n)}(d_n/2, d_d/2).\n\n    Wrapper for the Cephes [1]_ routine `fdtr`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ")
# Processing the call keyword arguments (line 2432)
kwargs_493743 = {}
# Getting the type of 'add_newdoc' (line 2432)
add_newdoc_493739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2432, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2432)
add_newdoc_call_result_493744 = invoke(stypy.reporting.localization.Localization(__file__, 2432, 0), add_newdoc_493739, *[str_493740, str_493741, str_493742], **kwargs_493743)


# Call to add_newdoc(...): (line 2483)
# Processing the call arguments (line 2483)
str_493746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2483, 11), 'str', 'scipy.special')
str_493747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2483, 28), 'str', 'fdtrc')
str_493748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2525, (-1)), 'str', '\n    fdtrc(dfn, dfd, x)\n\n    F survival function.\n\n    Returns the complemented F-distribution function (the integral of the\n    density from `x` to infinity).\n\n    Parameters\n    ----------\n    dfn : array_like\n        First parameter (positive float).\n    dfd : array_like\n        Second parameter (positive float).\n    x : array_like\n        Argument (nonnegative float).\n\n    Returns\n    -------\n    y : ndarray\n        The complemented F-distribution function with parameters `dfn` and\n        `dfd` at `x`.\n\n    See also\n    --------\n    fdtr\n\n    Notes\n    -----\n    The regularized incomplete beta function is used, according to the\n    formula,\n\n    .. math::\n        F(d_n, d_d; x) = I_{d_d/(d_d + xd_n)}(d_d/2, d_n/2).\n\n    Wrapper for the Cephes [1]_ routine `fdtrc`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 2483)
kwargs_493749 = {}
# Getting the type of 'add_newdoc' (line 2483)
add_newdoc_493745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2483, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2483)
add_newdoc_call_result_493750 = invoke(stypy.reporting.localization.Localization(__file__, 2483, 0), add_newdoc_493745, *[str_493746, str_493747, str_493748], **kwargs_493749)


# Call to add_newdoc(...): (line 2527)
# Processing the call arguments (line 2527)
str_493752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2527, 11), 'str', 'scipy.special')
str_493753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2527, 28), 'str', 'fdtri')
str_493754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2573, (-1)), 'str', "\n    fdtri(dfn, dfd, p)\n\n    The `p`-th quantile of the F-distribution.\n\n    This function is the inverse of the F-distribution CDF, `fdtr`, returning\n    the `x` such that `fdtr(dfn, dfd, x) = p`.\n\n    Parameters\n    ----------\n    dfn : array_like\n        First parameter (positive float).\n    dfd : array_like\n        Second parameter (positive float).\n    p : array_like\n        Cumulative probability, in [0, 1].\n\n    Returns\n    -------\n    x : ndarray\n        The quantile corresponding to `p`.\n\n    Notes\n    -----\n    The computation is carried out using the relation to the inverse\n    regularized beta function, :math:`I^{-1}_x(a, b)`.  Let\n    :math:`z = I^{-1}_p(d_d/2, d_n/2).`  Then,\n\n    .. math::\n        x = \\frac{d_d (1 - z)}{d_n z}.\n\n    If `p` is such that :math:`x < 0.5`, the following relation is used\n    instead for improved stability: let\n    :math:`z' = I^{-1}_{1 - p}(d_n/2, d_d/2).` Then,\n\n    .. math::\n        x = \\frac{d_d z'}{d_n (1 - z')}.\n\n    Wrapper for the Cephes [1]_ routine `fdtri`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ")
# Processing the call keyword arguments (line 2527)
kwargs_493755 = {}
# Getting the type of 'add_newdoc' (line 2527)
add_newdoc_493751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2527, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2527)
add_newdoc_call_result_493756 = invoke(stypy.reporting.localization.Localization(__file__, 2527, 0), add_newdoc_493751, *[str_493752, str_493753, str_493754], **kwargs_493755)


# Call to add_newdoc(...): (line 2575)
# Processing the call arguments (line 2575)
str_493758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2575, 11), 'str', 'scipy.special')
str_493759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2575, 28), 'str', 'fdtridfd')
str_493760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2582, (-1)), 'str', '\n    fdtridfd(dfn, p, x)\n\n    Inverse to `fdtr` vs dfd\n\n    Finds the F density argument dfd such that ``fdtr(dfn, dfd, x) == p``.\n    ')
# Processing the call keyword arguments (line 2575)
kwargs_493761 = {}
# Getting the type of 'add_newdoc' (line 2575)
add_newdoc_493757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2575, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2575)
add_newdoc_call_result_493762 = invoke(stypy.reporting.localization.Localization(__file__, 2575, 0), add_newdoc_493757, *[str_493758, str_493759, str_493760], **kwargs_493761)


# Call to add_newdoc(...): (line 2584)
# Processing the call arguments (line 2584)
str_493764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2584, 11), 'str', 'scipy.special')
str_493765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2584, 28), 'str', 'fdtridfn')
str_493766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2591, (-1)), 'str', '\n    fdtridfn(p, dfd, x)\n\n    Inverse to `fdtr` vs dfn\n\n    finds the F density argument dfn such that ``fdtr(dfn, dfd, x) == p``.\n    ')
# Processing the call keyword arguments (line 2584)
kwargs_493767 = {}
# Getting the type of 'add_newdoc' (line 2584)
add_newdoc_493763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2584, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2584)
add_newdoc_call_result_493768 = invoke(stypy.reporting.localization.Localization(__file__, 2584, 0), add_newdoc_493763, *[str_493764, str_493765, str_493766], **kwargs_493767)


# Call to add_newdoc(...): (line 2593)
# Processing the call arguments (line 2593)
str_493770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2593, 11), 'str', 'scipy.special')
str_493771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2593, 28), 'str', 'fresnel')
str_493772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2614, (-1)), 'str', '\n    fresnel(z)\n\n    Fresnel sin and cos integrals\n\n    Defined as::\n\n        ssa = integral(sin(pi/2 * t**2), t=0..z)\n        csa = integral(cos(pi/2 * t**2), t=0..z)\n\n    Parameters\n    ----------\n    z : float or complex array_like\n        Argument\n\n    Returns\n    -------\n    ssa, csa\n        Fresnel sin and cos integral values\n\n    ')
# Processing the call keyword arguments (line 2593)
kwargs_493773 = {}
# Getting the type of 'add_newdoc' (line 2593)
add_newdoc_493769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2593, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2593)
add_newdoc_call_result_493774 = invoke(stypy.reporting.localization.Localization(__file__, 2593, 0), add_newdoc_493769, *[str_493770, str_493771, str_493772], **kwargs_493773)


# Call to add_newdoc(...): (line 2616)
# Processing the call arguments (line 2616)
str_493776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2616, 11), 'str', 'scipy.special')
str_493777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2616, 28), 'str', 'gamma')
str_493778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2673, (-1)), 'str', "\n    gamma(z)\n\n    Gamma function.\n\n    .. math::\n\n          \\Gamma(z) = \\int_0^\\infty x^{z-1} e^{-x} dx = (z - 1)!\n\n    The gamma function is often referred to as the generalized\n    factorial since ``z*gamma(z) = gamma(z+1)`` and ``gamma(n+1) =\n    n!`` for natural number *n*.\n\n    Parameters\n    ----------\n    z : float or complex array_like\n\n    Returns\n    -------\n    float or complex\n        The value(s) of gamma(z)\n\n    Examples\n    --------\n    >>> from scipy.special import gamma, factorial\n\n    >>> gamma([0, 0.5, 1, 5])\n    array([         inf,   1.77245385,   1.        ,  24.        ])\n\n    >>> z = 2.5 + 1j\n    >>> gamma(z)\n    (0.77476210455108352+0.70763120437959293j)\n    >>> gamma(z+1), z*gamma(z)  # Recurrence property\n    ((1.2292740569981171+2.5438401155000685j),\n     (1.2292740569981158+2.5438401155000658j))\n\n    >>> gamma(0.5)**2  # gamma(0.5) = sqrt(pi)\n    3.1415926535897927\n\n    Plot gamma(x) for real x\n\n    >>> x = np.linspace(-3.5, 5.5, 2251)\n    >>> y = gamma(x)\n\n    >>> import matplotlib.pyplot as plt\n    >>> plt.plot(x, y, 'b', alpha=0.6, label='gamma(x)')\n    >>> k = np.arange(1, 7)\n    >>> plt.plot(k, factorial(k-1), 'k*', alpha=0.6,\n    ...          label='(x-1)!, x = 1, 2, ...')\n    >>> plt.xlim(-3.5, 5.5)\n    >>> plt.ylim(-10, 25)\n    >>> plt.grid()\n    >>> plt.xlabel('x')\n    >>> plt.legend(loc='lower right')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 2616)
kwargs_493779 = {}
# Getting the type of 'add_newdoc' (line 2616)
add_newdoc_493775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2616, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2616)
add_newdoc_call_result_493780 = invoke(stypy.reporting.localization.Localization(__file__, 2616, 0), add_newdoc_493775, *[str_493776, str_493777, str_493778], **kwargs_493779)


# Call to add_newdoc(...): (line 2675)
# Processing the call arguments (line 2675)
str_493782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2675, 11), 'str', 'scipy.special')
str_493783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2675, 28), 'str', 'gammainc')
str_493784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2705, (-1)), 'str', '\n    gammainc(a, x)\n\n    Regularized lower incomplete gamma function.\n\n    Defined as\n\n    .. math::\n\n        \\frac{1}{\\Gamma(a)} \\int_0^x t^{a - 1}e^{-t} dt\n\n    for :math:`a > 0` and :math:`x \\geq 0`. The function satisfies the\n    relation ``gammainc(a, x) + gammaincc(a, x) = 1`` where\n    `gammaincc` is the regularized upper incomplete gamma function.\n\n    Notes\n    -----\n    The implementation largely follows that of [1]_.\n\n    See also\n    --------\n    gammaincc : regularized upper incomplete gamma function\n    gammaincinv : inverse to ``gammainc`` versus ``x``\n    gammainccinv : inverse to ``gammaincc`` versus ``x``\n\n    References\n    ----------\n    .. [1] Maddock et. al., "Incomplete Gamma Functions",\n       http://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html\n    ')
# Processing the call keyword arguments (line 2675)
kwargs_493785 = {}
# Getting the type of 'add_newdoc' (line 2675)
add_newdoc_493781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2675, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2675)
add_newdoc_call_result_493786 = invoke(stypy.reporting.localization.Localization(__file__, 2675, 0), add_newdoc_493781, *[str_493782, str_493783, str_493784], **kwargs_493785)


# Call to add_newdoc(...): (line 2707)
# Processing the call arguments (line 2707)
str_493788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2707, 11), 'str', 'scipy.special')
str_493789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2707, 28), 'str', 'gammaincc')
str_493790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2737, (-1)), 'str', '\n    gammaincc(a, x)\n\n    Regularized upper incomplete gamma function.\n\n    Defined as\n\n    .. math::\n\n        \\frac{1}{\\Gamma(a)} \\int_x^\\infty t^{a - 1}e^{-t} dt\n\n    for :math:`a > 0` and :math:`x \\geq 0`. The function satisfies the\n    relation ``gammainc(a, x) + gammaincc(a, x) = 1`` where `gammainc`\n    is the regularized lower incomplete gamma function.\n\n    Notes\n    -----\n    The implementation largely follows that of [1]_.\n\n    See also\n    --------\n    gammainc : regularized lower incomplete gamma function\n    gammaincinv : inverse to ``gammainc`` versus ``x``\n    gammainccinv : inverse to ``gammaincc`` versus ``x``\n\n    References\n    ----------\n    .. [1] Maddock et. al., "Incomplete Gamma Functions",\n       http://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html\n    ')
# Processing the call keyword arguments (line 2707)
kwargs_493791 = {}
# Getting the type of 'add_newdoc' (line 2707)
add_newdoc_493787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2707, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2707)
add_newdoc_call_result_493792 = invoke(stypy.reporting.localization.Localization(__file__, 2707, 0), add_newdoc_493787, *[str_493788, str_493789, str_493790], **kwargs_493791)


# Call to add_newdoc(...): (line 2739)
# Processing the call arguments (line 2739)
str_493794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2739, 11), 'str', 'scipy.special')
str_493795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2739, 28), 'str', 'gammainccinv')
str_493796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2746, (-1)), 'str', '\n    gammainccinv(a, y)\n\n    Inverse to `gammaincc`\n\n    Returns `x` such that ``gammaincc(a, x) == y``.\n    ')
# Processing the call keyword arguments (line 2739)
kwargs_493797 = {}
# Getting the type of 'add_newdoc' (line 2739)
add_newdoc_493793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2739, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2739)
add_newdoc_call_result_493798 = invoke(stypy.reporting.localization.Localization(__file__, 2739, 0), add_newdoc_493793, *[str_493794, str_493795, str_493796], **kwargs_493797)


# Call to add_newdoc(...): (line 2748)
# Processing the call arguments (line 2748)
str_493800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2748, 11), 'str', 'scipy.special')
str_493801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2748, 28), 'str', 'gammaincinv')
str_493802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2755, (-1)), 'str', '\n    gammaincinv(a, y)\n\n    Inverse to `gammainc`\n\n    Returns `x` such that ``gammainc(a, x) = y``.\n    ')
# Processing the call keyword arguments (line 2748)
kwargs_493803 = {}
# Getting the type of 'add_newdoc' (line 2748)
add_newdoc_493799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2748, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2748)
add_newdoc_call_result_493804 = invoke(stypy.reporting.localization.Localization(__file__, 2748, 0), add_newdoc_493799, *[str_493800, str_493801, str_493802], **kwargs_493803)


# Call to add_newdoc(...): (line 2757)
# Processing the call arguments (line 2757)
str_493806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2757, 11), 'str', 'scipy.special')
str_493807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2757, 28), 'str', 'gammaln')
str_493808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2783, (-1)), 'str', '\n    Logarithm of the absolute value of the Gamma function.\n\n    Parameters\n    ----------\n    x : array-like\n        Values on the real line at which to compute ``gammaln``\n\n    Returns\n    -------\n    gammaln : ndarray\n        Values of ``gammaln`` at x.\n\n    See Also\n    --------\n    gammasgn : sign of the gamma function\n    loggamma : principal branch of the logarithm of the gamma function\n\n    Notes\n    -----\n    When used in conjunction with `gammasgn`, this function is useful\n    for working in logspace on the real axis without having to deal with\n    complex numbers, via the relation ``exp(gammaln(x)) = gammasgn(x)*gamma(x)``.\n\n    For complex-valued log-gamma, use `loggamma` instead of `gammaln`.\n    ')
# Processing the call keyword arguments (line 2757)
kwargs_493809 = {}
# Getting the type of 'add_newdoc' (line 2757)
add_newdoc_493805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2757, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2757)
add_newdoc_call_result_493810 = invoke(stypy.reporting.localization.Localization(__file__, 2757, 0), add_newdoc_493805, *[str_493806, str_493807, str_493808], **kwargs_493809)


# Call to add_newdoc(...): (line 2785)
# Processing the call arguments (line 2785)
str_493812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2785, 11), 'str', 'scipy.special')
str_493813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2785, 28), 'str', 'gammasgn')
str_493814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2795, (-1)), 'str', '\n    gammasgn(x)\n\n    Sign of the gamma function.\n\n    See Also\n    --------\n    gammaln\n    loggamma\n    ')
# Processing the call keyword arguments (line 2785)
kwargs_493815 = {}
# Getting the type of 'add_newdoc' (line 2785)
add_newdoc_493811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2785, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2785)
add_newdoc_call_result_493816 = invoke(stypy.reporting.localization.Localization(__file__, 2785, 0), add_newdoc_493811, *[str_493812, str_493813, str_493814], **kwargs_493815)


# Call to add_newdoc(...): (line 2797)
# Processing the call arguments (line 2797)
str_493818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2797, 11), 'str', 'scipy.special')
str_493819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2797, 28), 'str', 'gdtr')
str_493820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2846, (-1)), 'str', '\n    gdtr(a, b, x)\n\n    Gamma distribution cumulative density function.\n\n    Returns the integral from zero to `x` of the gamma probability density\n    function,\n\n    .. math::\n\n        F = \\int_0^x \\frac{a^b}{\\Gamma(b)} t^{b-1} e^{-at}\\,dt,\n\n    where :math:`\\Gamma` is the gamma function.\n\n    Parameters\n    ----------\n    a : array_like\n        The rate parameter of the gamma distribution, sometimes denoted\n        :math:`\\beta` (float).  It is also the reciprocal of the scale\n        parameter :math:`\\theta`.\n    b : array_like\n        The shape parameter of the gamma distribution, sometimes denoted\n        :math:`\\alpha` (float).\n    x : array_like\n        The quantile (upper limit of integration; float).\n\n    See also\n    --------\n    gdtrc : 1 - CDF of the gamma distribution.\n\n    Returns\n    -------\n    F : ndarray\n        The CDF of the gamma distribution with parameters `a` and `b`\n        evaluated at `x`.\n\n    Notes\n    -----\n    The evaluation is carried out using the relation to the incomplete gamma\n    integral (regularized gamma function).\n\n    Wrapper for the Cephes [1]_ routine `gdtr`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 2797)
kwargs_493821 = {}
# Getting the type of 'add_newdoc' (line 2797)
add_newdoc_493817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2797, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2797)
add_newdoc_call_result_493822 = invoke(stypy.reporting.localization.Localization(__file__, 2797, 0), add_newdoc_493817, *[str_493818, str_493819, str_493820], **kwargs_493821)


# Call to add_newdoc(...): (line 2848)
# Processing the call arguments (line 2848)
str_493824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2848, 11), 'str', 'scipy.special')
str_493825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2848, 28), 'str', 'gdtrc')
str_493826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2896, (-1)), 'str', '\n    gdtrc(a, b, x)\n\n    Gamma distribution survival function.\n\n    Integral from `x` to infinity of the gamma probability density function,\n\n    .. math::\n\n        F = \\int_x^\\infty \\frac{a^b}{\\Gamma(b)} t^{b-1} e^{-at}\\,dt,\n\n    where :math:`\\Gamma` is the gamma function.\n\n    Parameters\n    ----------\n    a : array_like\n        The rate parameter of the gamma distribution, sometimes denoted\n        :math:`\\beta` (float).  It is also the reciprocal of the scale\n        parameter :math:`\\theta`.\n    b : array_like\n        The shape parameter of the gamma distribution, sometimes denoted\n        :math:`\\alpha` (float).\n    x : array_like\n        The quantile (lower limit of integration; float).\n\n    Returns\n    -------\n    F : ndarray\n        The survival function of the gamma distribution with parameters `a`\n        and `b` evaluated at `x`.\n\n    See Also\n    --------\n    gdtr, gdtri\n\n    Notes\n    -----\n    The evaluation is carried out using the relation to the incomplete gamma\n    integral (regularized gamma function).\n\n    Wrapper for the Cephes [1]_ routine `gdtrc`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 2848)
kwargs_493827 = {}
# Getting the type of 'add_newdoc' (line 2848)
add_newdoc_493823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2848, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2848)
add_newdoc_call_result_493828 = invoke(stypy.reporting.localization.Localization(__file__, 2848, 0), add_newdoc_493823, *[str_493824, str_493825, str_493826], **kwargs_493827)


# Call to add_newdoc(...): (line 2898)
# Processing the call arguments (line 2898)
str_493830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2898, 11), 'str', 'scipy.special')
str_493831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2898, 28), 'str', 'gdtria')
str_493832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2965, (-1)), 'str', '\n    gdtria(p, b, x, out=None)\n\n    Inverse of `gdtr` vs a.\n\n    Returns the inverse with respect to the parameter `a` of ``p =\n    gdtr(a, b, x)``, the cumulative distribution function of the gamma\n    distribution.\n\n    Parameters\n    ----------\n    p : array_like\n        Probability values.\n    b : array_like\n        `b` parameter values of `gdtr(a, b, x)`.  `b` is the "shape" parameter\n        of the gamma distribution.\n    x : array_like\n        Nonnegative real values, from the domain of the gamma distribution.\n    out : ndarray, optional\n        If a fourth argument is given, it must be a numpy.ndarray whose size\n        matches the broadcast result of `a`, `b` and `x`.  `out` is then the\n        array returned by the function.\n\n    Returns\n    -------\n    a : ndarray\n        Values of the `a` parameter such that `p = gdtr(a, b, x)`.  `1/a`\n        is the "scale" parameter of the gamma distribution.\n\n    See Also\n    --------\n    gdtr : CDF of the gamma distribution.\n    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.\n    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.\n\n    The cumulative distribution function `p` is computed using a routine by\n    DiDinato and Morris [2]_.  Computation of `a` involves a search for a value\n    that produces the desired value of `p`.  The search relies on the\n    monotonicity of `p` with `a`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] DiDinato, A. R. and Morris, A. H.,\n           Computation of the incomplete gamma function ratios and their\n           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.\n\n    Examples\n    --------\n    First evaluate `gdtr`.\n\n    >>> from scipy.special import gdtr, gdtria\n    >>> p = gdtr(1.2, 3.4, 5.6)\n    >>> print(p)\n    0.94378087442\n\n    Verify the inverse.\n\n    >>> gdtria(p, 3.4, 5.6)\n    1.2\n    ')
# Processing the call keyword arguments (line 2898)
kwargs_493833 = {}
# Getting the type of 'add_newdoc' (line 2898)
add_newdoc_493829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2898, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2898)
add_newdoc_call_result_493834 = invoke(stypy.reporting.localization.Localization(__file__, 2898, 0), add_newdoc_493829, *[str_493830, str_493831, str_493832], **kwargs_493833)


# Call to add_newdoc(...): (line 2967)
# Processing the call arguments (line 2967)
str_493836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2967, 11), 'str', 'scipy.special')
str_493837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2967, 28), 'str', 'gdtrib')
str_493838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3034, (-1)), 'str', '\n    gdtrib(a, p, x, out=None)\n\n    Inverse of `gdtr` vs b.\n\n    Returns the inverse with respect to the parameter `b` of ``p =\n    gdtr(a, b, x)``, the cumulative distribution function of the gamma\n    distribution.\n\n    Parameters\n    ----------\n    a : array_like\n        `a` parameter values of `gdtr(a, b, x)`. `1/a` is the "scale"\n        parameter of the gamma distribution.\n    p : array_like\n        Probability values.\n    x : array_like\n        Nonnegative real values, from the domain of the gamma distribution.\n    out : ndarray, optional\n        If a fourth argument is given, it must be a numpy.ndarray whose size\n        matches the broadcast result of `a`, `b` and `x`.  `out` is then the\n        array returned by the function.\n\n    Returns\n    -------\n    b : ndarray\n        Values of the `b` parameter such that `p = gdtr(a, b, x)`.  `b` is\n        the "shape" parameter of the gamma distribution.\n\n    See Also\n    --------\n    gdtr : CDF of the gamma distribution.\n    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.\n    gdtrix : Inverse with respect to `x` of `gdtr(a, b, x)`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.\n\n    The cumulative distribution function `p` is computed using a routine by\n    DiDinato and Morris [2]_.  Computation of `b` involves a search for a value\n    that produces the desired value of `p`.  The search relies on the\n    monotonicity of `p` with `b`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] DiDinato, A. R. and Morris, A. H.,\n           Computation of the incomplete gamma function ratios and their\n           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.\n\n    Examples\n    --------\n    First evaluate `gdtr`.\n\n    >>> from scipy.special import gdtr, gdtrib\n    >>> p = gdtr(1.2, 3.4, 5.6)\n    >>> print(p)\n    0.94378087442\n\n    Verify the inverse.\n\n    >>> gdtrib(1.2, p, 5.6)\n    3.3999999999723882\n    ')
# Processing the call keyword arguments (line 2967)
kwargs_493839 = {}
# Getting the type of 'add_newdoc' (line 2967)
add_newdoc_493835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2967, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 2967)
add_newdoc_call_result_493840 = invoke(stypy.reporting.localization.Localization(__file__, 2967, 0), add_newdoc_493835, *[str_493836, str_493837, str_493838], **kwargs_493839)


# Call to add_newdoc(...): (line 3036)
# Processing the call arguments (line 3036)
str_493842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3036, 11), 'str', 'scipy.special')
str_493843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3036, 28), 'str', 'gdtrix')
str_493844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3104, (-1)), 'str', '\n    gdtrix(a, b, p, out=None)\n\n    Inverse of `gdtr` vs x.\n\n    Returns the inverse with respect to the parameter `x` of ``p =\n    gdtr(a, b, x)``, the cumulative distribution function of the gamma\n    distribution. This is also known as the p\'th quantile of the\n    distribution.\n\n    Parameters\n    ----------\n    a : array_like\n        `a` parameter values of `gdtr(a, b, x)`.  `1/a` is the "scale"\n        parameter of the gamma distribution.\n    b : array_like\n        `b` parameter values of `gdtr(a, b, x)`.  `b` is the "shape" parameter\n        of the gamma distribution.\n    p : array_like\n        Probability values.\n    out : ndarray, optional\n        If a fourth argument is given, it must be a numpy.ndarray whose size\n        matches the broadcast result of `a`, `b` and `x`.  `out` is then the\n        array returned by the function.\n\n    Returns\n    -------\n    x : ndarray\n        Values of the `x` parameter such that `p = gdtr(a, b, x)`.\n\n    See Also\n    --------\n    gdtr : CDF of the gamma distribution.\n    gdtria : Inverse with respect to `a` of `gdtr(a, b, x)`.\n    gdtrib : Inverse with respect to `b` of `gdtr(a, b, x)`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfgam`.\n\n    The cumulative distribution function `p` is computed using a routine by\n    DiDinato and Morris [2]_.  Computation of `x` involves a search for a value\n    that produces the desired value of `p`.  The search relies on the\n    monotonicity of `p` with `x`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] DiDinato, A. R. and Morris, A. H.,\n           Computation of the incomplete gamma function ratios and their\n           inverse.  ACM Trans. Math. Softw. 12 (1986), 377-393.\n\n    Examples\n    --------\n    First evaluate `gdtr`.\n\n    >>> from scipy.special import gdtr, gdtrix\n    >>> p = gdtr(1.2, 3.4, 5.6)\n    >>> print(p)\n    0.94378087442\n\n    Verify the inverse.\n\n    >>> gdtrix(1.2, 3.4, p)\n    5.5999999999999996\n    ')
# Processing the call keyword arguments (line 3036)
kwargs_493845 = {}
# Getting the type of 'add_newdoc' (line 3036)
add_newdoc_493841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3036, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3036)
add_newdoc_call_result_493846 = invoke(stypy.reporting.localization.Localization(__file__, 3036, 0), add_newdoc_493841, *[str_493842, str_493843, str_493844], **kwargs_493845)


# Call to add_newdoc(...): (line 3106)
# Processing the call arguments (line 3106)
str_493848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3106, 11), 'str', 'scipy.special')
str_493849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3106, 28), 'str', 'hankel1')
str_493850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3146, (-1)), 'str', '\n    hankel1(v, z)\n\n    Hankel function of the first kind\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    out : Values of the Hankel function of the first kind.\n\n    Notes\n    -----\n    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the\n    computation using the relation,\n\n    .. math:: H^{(1)}_v(z) = \\frac{2}{\\imath\\pi} \\exp(-\\imath \\pi v/2) K_v(z \\exp(-\\imath\\pi/2))\n\n    where :math:`K_v` is the modified Bessel function of the second kind.\n    For negative orders, the relation\n\n    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \\exp(\\imath\\pi v)\n\n    is used.\n\n    See also\n    --------\n    hankel1e : this function with leading exponential behavior stripped off.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 3106)
kwargs_493851 = {}
# Getting the type of 'add_newdoc' (line 3106)
add_newdoc_493847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3106, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3106)
add_newdoc_call_result_493852 = invoke(stypy.reporting.localization.Localization(__file__, 3106, 0), add_newdoc_493847, *[str_493848, str_493849, str_493850], **kwargs_493851)


# Call to add_newdoc(...): (line 3148)
# Processing the call arguments (line 3148)
str_493854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3148, 11), 'str', 'scipy.special')
str_493855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3148, 28), 'str', 'hankel1e')
str_493856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3188, (-1)), 'str', '\n    hankel1e(v, z)\n\n    Exponentially scaled Hankel function of the first kind\n\n    Defined as::\n\n        hankel1e(v, z) = hankel1(v, z) * exp(-1j * z)\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    out : Values of the exponentially scaled Hankel function.\n\n    Notes\n    -----\n    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the\n    computation using the relation,\n\n    .. math:: H^{(1)}_v(z) = \\frac{2}{\\imath\\pi} \\exp(-\\imath \\pi v/2) K_v(z \\exp(-\\imath\\pi/2))\n\n    where :math:`K_v` is the modified Bessel function of the second kind.\n    For negative orders, the relation\n\n    .. math:: H^{(1)}_{-v}(z) = H^{(1)}_v(z) \\exp(\\imath\\pi v)\n\n    is used.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 3148)
kwargs_493857 = {}
# Getting the type of 'add_newdoc' (line 3148)
add_newdoc_493853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3148, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3148)
add_newdoc_call_result_493858 = invoke(stypy.reporting.localization.Localization(__file__, 3148, 0), add_newdoc_493853, *[str_493854, str_493855, str_493856], **kwargs_493857)


# Call to add_newdoc(...): (line 3190)
# Processing the call arguments (line 3190)
str_493860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3190, 11), 'str', 'scipy.special')
str_493861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3190, 28), 'str', 'hankel2')
str_493862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3230, (-1)), 'str', '\n    hankel2(v, z)\n\n    Hankel function of the second kind\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    out : Values of the Hankel function of the second kind.\n\n    Notes\n    -----\n    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the\n    computation using the relation,\n\n    .. math:: H^{(2)}_v(z) = -\\frac{2}{\\imath\\pi} \\exp(\\imath \\pi v/2) K_v(z \\exp(\\imath\\pi/2))\n\n    where :math:`K_v` is the modified Bessel function of the second kind.\n    For negative orders, the relation\n\n    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \\exp(-\\imath\\pi v)\n\n    is used.\n\n    See also\n    --------\n    hankel2e : this function with leading exponential behavior stripped off.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 3190)
kwargs_493863 = {}
# Getting the type of 'add_newdoc' (line 3190)
add_newdoc_493859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3190, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3190)
add_newdoc_call_result_493864 = invoke(stypy.reporting.localization.Localization(__file__, 3190, 0), add_newdoc_493859, *[str_493860, str_493861, str_493862], **kwargs_493863)


# Call to add_newdoc(...): (line 3232)
# Processing the call arguments (line 3232)
str_493866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3232, 11), 'str', 'scipy.special')
str_493867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3232, 28), 'str', 'hankel2e')
str_493868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3273, (-1)), 'str', '\n    hankel2e(v, z)\n\n    Exponentially scaled Hankel function of the second kind\n\n    Defined as::\n\n        hankel2e(v, z) = hankel2(v, z) * exp(1j * z)\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    out : Values of the exponentially scaled Hankel function of the second kind.\n\n    Notes\n    -----\n    A wrapper for the AMOS [1]_ routine `zbesh`, which carries out the\n    computation using the relation,\n\n    .. math:: H^{(2)}_v(z) = -\\frac{2}{\\imath\\pi} \\exp(\\frac{\\imath \\pi v}{2}) K_v(z exp(\\frac{\\imath\\pi}{2}))\n\n    where :math:`K_v` is the modified Bessel function of the second kind.\n    For negative orders, the relation\n\n    .. math:: H^{(2)}_{-v}(z) = H^{(2)}_v(z) \\exp(-\\imath\\pi v)\n\n    is used.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n\n    ')
# Processing the call keyword arguments (line 3232)
kwargs_493869 = {}
# Getting the type of 'add_newdoc' (line 3232)
add_newdoc_493865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3232, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3232)
add_newdoc_call_result_493870 = invoke(stypy.reporting.localization.Localization(__file__, 3232, 0), add_newdoc_493865, *[str_493866, str_493867, str_493868], **kwargs_493869)


# Call to add_newdoc(...): (line 3275)
# Processing the call arguments (line 3275)
str_493872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3275, 11), 'str', 'scipy.special')
str_493873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3275, 28), 'str', 'huber')
str_493874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3301, (-1)), 'str', '\n    huber(delta, r)\n\n    Huber loss function.\n\n    .. math:: \\text{huber}(\\delta, r) = \\begin{cases} \\infty & \\delta < 0  \\\\ \\frac{1}{2}r^2 & 0 \\le \\delta, | r | \\le \\delta \\\\ \\delta ( |r| - \\frac{1}{2}\\delta ) & \\text{otherwise} \\end{cases}\n\n    Parameters\n    ----------\n    delta : ndarray\n        Input array, indicating the quadratic vs. linear loss changepoint.\n    r : ndarray\n        Input array, possibly representing residuals.\n\n    Returns\n    -------\n    res : ndarray\n        The computed Huber loss function values.\n\n    Notes\n    -----\n    This function is convex in r.\n\n    .. versionadded:: 0.15.0\n\n    ')
# Processing the call keyword arguments (line 3275)
kwargs_493875 = {}
# Getting the type of 'add_newdoc' (line 3275)
add_newdoc_493871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3275, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3275)
add_newdoc_call_result_493876 = invoke(stypy.reporting.localization.Localization(__file__, 3275, 0), add_newdoc_493871, *[str_493872, str_493873, str_493874], **kwargs_493875)


# Call to add_newdoc(...): (line 3303)
# Processing the call arguments (line 3303)
str_493878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3303, 11), 'str', 'scipy.special')
str_493879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3303, 28), 'str', 'hyp0f1')
str_493880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3327, (-1)), 'str', "\n    hyp0f1(v, x)\n\n    Confluent hypergeometric limit function 0F1.\n\n    Parameters\n    ----------\n    v, z : array_like\n        Input values.\n\n    Returns\n    -------\n    hyp0f1 : ndarray\n        The confluent hypergeometric limit function.\n\n    Notes\n    -----\n    This function is defined as:\n\n    .. math:: _0F_1(v, z) = \\sum_{k=0}^{\\infty}\\frac{z^k}{(v)_k k!}.\n\n    It's also the limit as :math:`q \\to \\infty` of :math:`_1F_1(q; v; z/q)`,\n    and satisfies the differential equation :math:`f''(z) + vf'(z) = f(z)`.\n    ")
# Processing the call keyword arguments (line 3303)
kwargs_493881 = {}
# Getting the type of 'add_newdoc' (line 3303)
add_newdoc_493877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3303, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3303)
add_newdoc_call_result_493882 = invoke(stypy.reporting.localization.Localization(__file__, 3303, 0), add_newdoc_493877, *[str_493878, str_493879, str_493880], **kwargs_493881)


# Call to add_newdoc(...): (line 3329)
# Processing the call arguments (line 3329)
str_493884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 11), 'str', 'scipy.special')
str_493885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3329, 28), 'str', 'hyp1f1')
str_493886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3334, (-1)), 'str', '\n    hyp1f1(a, b, x)\n\n    Confluent hypergeometric function 1F1(a, b; x)\n    ')
# Processing the call keyword arguments (line 3329)
kwargs_493887 = {}
# Getting the type of 'add_newdoc' (line 3329)
add_newdoc_493883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3329, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3329)
add_newdoc_call_result_493888 = invoke(stypy.reporting.localization.Localization(__file__, 3329, 0), add_newdoc_493883, *[str_493884, str_493885, str_493886], **kwargs_493887)


# Call to add_newdoc(...): (line 3336)
# Processing the call arguments (line 3336)
str_493890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3336, 11), 'str', 'scipy.special')
str_493891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3336, 28), 'str', 'hyp1f2')
str_493892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3348, (-1)), 'str', '\n    hyp1f2(a, b, c, x)\n\n    Hypergeometric function 1F2 and error estimate\n\n    Returns\n    -------\n    y\n        Value of the function\n    err\n        Error estimate\n    ')
# Processing the call keyword arguments (line 3336)
kwargs_493893 = {}
# Getting the type of 'add_newdoc' (line 3336)
add_newdoc_493889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3336, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3336)
add_newdoc_call_result_493894 = invoke(stypy.reporting.localization.Localization(__file__, 3336, 0), add_newdoc_493889, *[str_493890, str_493891, str_493892], **kwargs_493893)


# Call to add_newdoc(...): (line 3350)
# Processing the call arguments (line 3350)
str_493896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3350, 11), 'str', 'scipy.special')
str_493897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3350, 28), 'str', 'hyp2f0')
str_493898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3365, (-1)), 'str', '\n    hyp2f0(a, b, x, type)\n\n    Hypergeometric function 2F0 in y and an error estimate\n\n    The parameter `type` determines a convergence factor and can be\n    either 1 or 2.\n\n    Returns\n    -------\n    y\n        Value of the function\n    err\n        Error estimate\n    ')
# Processing the call keyword arguments (line 3350)
kwargs_493899 = {}
# Getting the type of 'add_newdoc' (line 3350)
add_newdoc_493895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3350, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3350)
add_newdoc_call_result_493900 = invoke(stypy.reporting.localization.Localization(__file__, 3350, 0), add_newdoc_493895, *[str_493896, str_493897, str_493898], **kwargs_493899)


# Call to add_newdoc(...): (line 3367)
# Processing the call arguments (line 3367)
str_493902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3367, 11), 'str', 'scipy.special')
str_493903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3367, 28), 'str', 'hyp2f1')
str_493904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3413, (-1)), 'str', '\n    hyp2f1(a, b, c, z)\n\n    Gauss hypergeometric function 2F1(a, b; c; z)\n\n    Parameters\n    ----------\n    a, b, c : array_like\n        Arguments, should be real-valued.\n    z : array_like\n        Argument, real or complex.\n\n    Returns\n    -------\n    hyp2f1 : scalar or ndarray\n        The values of the gaussian hypergeometric function.\n\n    See also\n    --------\n    hyp0f1 : confluent hypergeometric limit function.\n    hyp1f1 : Kummer\'s (confluent hypergeometric) function.\n\n    Notes\n    -----\n    This function is defined for :math:`|z| < 1` as\n\n    .. math::\n\n       \\mathrm{hyp2f1}(a, b, c, z) = \\sum_{n=0}^\\infty\n       \\frac{(a)_n (b)_n}{(c)_n}\\frac{z^n}{n!},\n\n    and defined on the rest of the complex z-plane by analytic continuation.\n    Here :math:`(\\cdot)_n` is the Pochhammer symbol; see `poch`. When\n    :math:`n` is an integer the result is a polynomial of degree :math:`n`.\n\n    The implementation for complex values of ``z`` is described in [1]_.\n\n    References\n    ----------\n    .. [1] J.M. Jin and Z. S. Jjie, "Computation of special functions", Wiley, 1996.\n    .. [2] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [3] NIST Digital Library of Mathematical Functions\n           http://dlmf.nist.gov/\n\n    ')
# Processing the call keyword arguments (line 3367)
kwargs_493905 = {}
# Getting the type of 'add_newdoc' (line 3367)
add_newdoc_493901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3367, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3367)
add_newdoc_call_result_493906 = invoke(stypy.reporting.localization.Localization(__file__, 3367, 0), add_newdoc_493901, *[str_493902, str_493903, str_493904], **kwargs_493905)


# Call to add_newdoc(...): (line 3415)
# Processing the call arguments (line 3415)
str_493908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3415, 11), 'str', 'scipy.special')
str_493909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3415, 28), 'str', 'hyp3f0')
str_493910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3427, (-1)), 'str', '\n    hyp3f0(a, b, c, x)\n\n    Hypergeometric function 3F0 in y and an error estimate\n\n    Returns\n    -------\n    y\n        Value of the function\n    err\n        Error estimate\n    ')
# Processing the call keyword arguments (line 3415)
kwargs_493911 = {}
# Getting the type of 'add_newdoc' (line 3415)
add_newdoc_493907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3415, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3415)
add_newdoc_call_result_493912 = invoke(stypy.reporting.localization.Localization(__file__, 3415, 0), add_newdoc_493907, *[str_493908, str_493909, str_493910], **kwargs_493911)


# Call to add_newdoc(...): (line 3429)
# Processing the call arguments (line 3429)
str_493914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3429, 11), 'str', 'scipy.special')
str_493915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3429, 28), 'str', 'hyperu')
str_493916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3434, (-1)), 'str', '\n    hyperu(a, b, x)\n\n    Confluent hypergeometric function U(a, b, x) of the second kind\n    ')
# Processing the call keyword arguments (line 3429)
kwargs_493917 = {}
# Getting the type of 'add_newdoc' (line 3429)
add_newdoc_493913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3429, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3429)
add_newdoc_call_result_493918 = invoke(stypy.reporting.localization.Localization(__file__, 3429, 0), add_newdoc_493913, *[str_493914, str_493915, str_493916], **kwargs_493917)


# Call to add_newdoc(...): (line 3436)
# Processing the call arguments (line 3436)
str_493920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3436, 11), 'str', 'scipy.special')
str_493921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3436, 28), 'str', 'i0')
str_493922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3475, (-1)), 'str', '\n    i0(x)\n\n    Modified Bessel function of order 0.\n\n    Defined as,\n\n    .. math::\n        I_0(x) = \\sum_{k=0}^\\infty \\frac{(x^2/4)^k}{(k!)^2} = J_0(\\imath x),\n\n    where :math:`J_0` is the Bessel function of the first kind of order 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    I : ndarray\n        Value of the modified Bessel function of order 0 at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 8] and (8, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `i0`.\n\n    See also\n    --------\n    iv\n    i0e\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 3436)
kwargs_493923 = {}
# Getting the type of 'add_newdoc' (line 3436)
add_newdoc_493919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3436, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3436)
add_newdoc_call_result_493924 = invoke(stypy.reporting.localization.Localization(__file__, 3436, 0), add_newdoc_493919, *[str_493920, str_493921, str_493922], **kwargs_493923)


# Call to add_newdoc(...): (line 3477)
# Processing the call arguments (line 3477)
str_493926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3477, 11), 'str', 'scipy.special')
str_493927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3477, 28), 'str', 'i0e')
str_493928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3516, (-1)), 'str', '\n    i0e(x)\n\n    Exponentially scaled modified Bessel function of order 0.\n\n    Defined as::\n\n        i0e(x) = exp(-abs(x)) * i0(x).\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    I : ndarray\n        Value of the exponentially scaled modified Bessel function of order 0\n        at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 8] and (8, infinity).\n    Chebyshev polynomial expansions are employed in each interval.  The\n    polynomial expansions used are the same as those in `i0`, but\n    they are not multiplied by the dominant exponential factor.\n\n    This function is a wrapper for the Cephes [1]_ routine `i0e`.\n\n    See also\n    --------\n    iv\n    i0\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 3477)
kwargs_493929 = {}
# Getting the type of 'add_newdoc' (line 3477)
add_newdoc_493925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3477, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3477)
add_newdoc_call_result_493930 = invoke(stypy.reporting.localization.Localization(__file__, 3477, 0), add_newdoc_493925, *[str_493926, str_493927, str_493928], **kwargs_493929)


# Call to add_newdoc(...): (line 3518)
# Processing the call arguments (line 3518)
str_493932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3518, 11), 'str', 'scipy.special')
str_493933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3518, 28), 'str', 'i1')
str_493934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3558, (-1)), 'str', '\n    i1(x)\n\n    Modified Bessel function of order 1.\n\n    Defined as,\n\n    .. math::\n        I_1(x) = \\frac{1}{2}x \\sum_{k=0}^\\infty \\frac{(x^2/4)^k}{k! (k + 1)!}\n               = -\\imath J_1(\\imath x),\n\n    where :math:`J_1` is the Bessel function of the first kind of order 1.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    I : ndarray\n        Value of the modified Bessel function of order 1 at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 8] and (8, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `i1`.\n\n    See also\n    --------\n    iv\n    i1e\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 3518)
kwargs_493935 = {}
# Getting the type of 'add_newdoc' (line 3518)
add_newdoc_493931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3518, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3518)
add_newdoc_call_result_493936 = invoke(stypy.reporting.localization.Localization(__file__, 3518, 0), add_newdoc_493931, *[str_493932, str_493933, str_493934], **kwargs_493935)


# Call to add_newdoc(...): (line 3560)
# Processing the call arguments (line 3560)
str_493938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3560, 11), 'str', 'scipy.special')
str_493939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3560, 28), 'str', 'i1e')
str_493940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3599, (-1)), 'str', '\n    i1e(x)\n\n    Exponentially scaled modified Bessel function of order 1.\n\n    Defined as::\n\n        i1e(x) = exp(-abs(x)) * i1(x)\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    I : ndarray\n        Value of the exponentially scaled modified Bessel function of order 1\n        at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 8] and (8, infinity).\n    Chebyshev polynomial expansions are employed in each interval. The\n    polynomial expansions used are the same as those in `i1`, but\n    they are not multiplied by the dominant exponential factor.\n\n    This function is a wrapper for the Cephes [1]_ routine `i1e`.\n\n    See also\n    --------\n    iv\n    i1\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 3560)
kwargs_493941 = {}
# Getting the type of 'add_newdoc' (line 3560)
add_newdoc_493937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3560, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3560)
add_newdoc_call_result_493942 = invoke(stypy.reporting.localization.Localization(__file__, 3560, 0), add_newdoc_493937, *[str_493938, str_493939, str_493940], **kwargs_493941)


# Call to add_newdoc(...): (line 3601)
# Processing the call arguments (line 3601)
str_493944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3601, 11), 'str', 'scipy.special')
str_493945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3601, 28), 'str', '_igam_fac')
str_493946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3604, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 3601)
kwargs_493947 = {}
# Getting the type of 'add_newdoc' (line 3601)
add_newdoc_493943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3601, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3601)
add_newdoc_call_result_493948 = invoke(stypy.reporting.localization.Localization(__file__, 3601, 0), add_newdoc_493943, *[str_493944, str_493945, str_493946], **kwargs_493947)


# Call to add_newdoc(...): (line 3606)
# Processing the call arguments (line 3606)
str_493950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3606, 11), 'str', 'scipy.special')
str_493951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3606, 28), 'str', 'it2i0k0')
str_493952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3618, (-1)), 'str', '\n    it2i0k0(x)\n\n    Integrals related to modified Bessel functions of order 0\n\n    Returns\n    -------\n    ii0\n        ``integral((i0(t)-1)/t, t=0..x)``\n    ik0\n        ``int(k0(t)/t, t=x..inf)``\n    ')
# Processing the call keyword arguments (line 3606)
kwargs_493953 = {}
# Getting the type of 'add_newdoc' (line 3606)
add_newdoc_493949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3606, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3606)
add_newdoc_call_result_493954 = invoke(stypy.reporting.localization.Localization(__file__, 3606, 0), add_newdoc_493949, *[str_493950, str_493951, str_493952], **kwargs_493953)


# Call to add_newdoc(...): (line 3620)
# Processing the call arguments (line 3620)
str_493956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3620, 11), 'str', 'scipy.special')
str_493957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3620, 28), 'str', 'it2j0y0')
str_493958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3632, (-1)), 'str', '\n    it2j0y0(x)\n\n    Integrals related to Bessel functions of order 0\n\n    Returns\n    -------\n    ij0\n        ``integral((1-j0(t))/t, t=0..x)``\n    iy0\n        ``integral(y0(t)/t, t=x..inf)``\n    ')
# Processing the call keyword arguments (line 3620)
kwargs_493959 = {}
# Getting the type of 'add_newdoc' (line 3620)
add_newdoc_493955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3620, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3620)
add_newdoc_call_result_493960 = invoke(stypy.reporting.localization.Localization(__file__, 3620, 0), add_newdoc_493955, *[str_493956, str_493957, str_493958], **kwargs_493959)


# Call to add_newdoc(...): (line 3634)
# Processing the call arguments (line 3634)
str_493962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3634, 11), 'str', 'scipy.special')
str_493963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3634, 28), 'str', 'it2struve0')
str_493964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3671, (-1)), 'str', '\n    it2struve0(x)\n\n    Integral related to the Struve function of order 0.\n\n    Returns the integral,\n\n    .. math::\n        \\int_x^\\infty \\frac{H_0(t)}{t}\\,dt\n\n    where :math:`H_0` is the Struve function of order 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Lower limit of integration.\n\n    Returns\n    -------\n    I : ndarray\n        The value of the integral.\n\n    See also\n    --------\n    struve\n\n    Notes\n    -----\n    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming\n    Jin [1]_.\n\n    References\n    ----------\n    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special\n           Functions", John Wiley and Sons, 1996.\n           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html\n    ')
# Processing the call keyword arguments (line 3634)
kwargs_493965 = {}
# Getting the type of 'add_newdoc' (line 3634)
add_newdoc_493961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3634, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3634)
add_newdoc_call_result_493966 = invoke(stypy.reporting.localization.Localization(__file__, 3634, 0), add_newdoc_493961, *[str_493962, str_493963, str_493964], **kwargs_493965)


# Call to add_newdoc(...): (line 3673)
# Processing the call arguments (line 3673)
str_493968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3673, 11), 'str', 'scipy.special')
str_493969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3673, 28), 'str', 'itairy')
str_493970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3710, (-1)), 'str', '\n    itairy(x)\n\n    Integrals of Airy functions\n\n    Calculates the integrals of Airy functions from 0 to `x`.\n\n    Parameters\n    ----------\n\n    x: array_like\n        Upper limit of integration (float).\n\n    Returns\n    -------\n    Apt\n        Integral of Ai(t) from 0 to x.\n    Bpt\n        Integral of Bi(t) from 0 to x.\n    Ant\n        Integral of Ai(-t) from 0 to x.\n    Bnt\n        Integral of Bi(-t) from 0 to x.\n\n    Notes\n    -----\n\n    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming\n    Jin [1]_.\n\n    References\n    ----------\n\n    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special\n           Functions", John Wiley and Sons, 1996.\n           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html\n    ')
# Processing the call keyword arguments (line 3673)
kwargs_493971 = {}
# Getting the type of 'add_newdoc' (line 3673)
add_newdoc_493967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3673, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3673)
add_newdoc_call_result_493972 = invoke(stypy.reporting.localization.Localization(__file__, 3673, 0), add_newdoc_493967, *[str_493968, str_493969, str_493970], **kwargs_493971)


# Call to add_newdoc(...): (line 3712)
# Processing the call arguments (line 3712)
str_493974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3712, 11), 'str', 'scipy.special')
str_493975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3712, 28), 'str', 'iti0k0')
str_493976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3724, (-1)), 'str', '\n    iti0k0(x)\n\n    Integrals of modified Bessel functions of order 0\n\n    Returns simple integrals from 0 to `x` of the zeroth order modified\n    Bessel functions `i0` and `k0`.\n\n    Returns\n    -------\n    ii0, ik0\n    ')
# Processing the call keyword arguments (line 3712)
kwargs_493977 = {}
# Getting the type of 'add_newdoc' (line 3712)
add_newdoc_493973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3712, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3712)
add_newdoc_call_result_493978 = invoke(stypy.reporting.localization.Localization(__file__, 3712, 0), add_newdoc_493973, *[str_493974, str_493975, str_493976], **kwargs_493977)


# Call to add_newdoc(...): (line 3726)
# Processing the call arguments (line 3726)
str_493980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3726, 11), 'str', 'scipy.special')
str_493981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3726, 28), 'str', 'itj0y0')
str_493982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3738, (-1)), 'str', '\n    itj0y0(x)\n\n    Integrals of Bessel functions of order 0\n\n    Returns simple integrals from 0 to `x` of the zeroth order Bessel\n    functions `j0` and `y0`.\n\n    Returns\n    -------\n    ij0, iy0\n    ')
# Processing the call keyword arguments (line 3726)
kwargs_493983 = {}
# Getting the type of 'add_newdoc' (line 3726)
add_newdoc_493979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3726, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3726)
add_newdoc_call_result_493984 = invoke(stypy.reporting.localization.Localization(__file__, 3726, 0), add_newdoc_493979, *[str_493980, str_493981, str_493982], **kwargs_493983)


# Call to add_newdoc(...): (line 3740)
# Processing the call arguments (line 3740)
str_493986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3740, 11), 'str', 'scipy.special')
str_493987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3740, 28), 'str', 'itmodstruve0')
str_493988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3770, (-1)), 'str', '\n    itmodstruve0(x)\n\n    Integral of the modified Struve function of order 0.\n\n    .. math::\n        I = \\int_0^x L_0(t)\\,dt\n\n    Parameters\n    ----------\n    x : array_like\n        Upper limit of integration (float).\n\n    Returns\n    -------\n    I : ndarray\n        The integral of :math:`L_0` from 0 to `x`.\n\n    Notes\n    -----\n    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming\n    Jin [1]_.\n\n    References\n    ----------\n    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special\n           Functions", John Wiley and Sons, 1996.\n           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html\n\n    ')
# Processing the call keyword arguments (line 3740)
kwargs_493989 = {}
# Getting the type of 'add_newdoc' (line 3740)
add_newdoc_493985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3740, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3740)
add_newdoc_call_result_493990 = invoke(stypy.reporting.localization.Localization(__file__, 3740, 0), add_newdoc_493985, *[str_493986, str_493987, str_493988], **kwargs_493989)


# Call to add_newdoc(...): (line 3772)
# Processing the call arguments (line 3772)
str_493992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3772, 11), 'str', 'scipy.special')
str_493993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3772, 28), 'str', 'itstruve0')
str_493994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3806, (-1)), 'str', '\n    itstruve0(x)\n\n    Integral of the Struve function of order 0.\n\n    .. math::\n        I = \\int_0^x H_0(t)\\,dt\n\n    Parameters\n    ----------\n    x : array_like\n        Upper limit of integration (float).\n\n    Returns\n    -------\n    I : ndarray\n        The integral of :math:`H_0` from 0 to `x`.\n\n    See also\n    --------\n    struve\n\n    Notes\n    -----\n    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming\n    Jin [1]_.\n\n    References\n    ----------\n    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special\n           Functions", John Wiley and Sons, 1996.\n           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html\n\n    ')
# Processing the call keyword arguments (line 3772)
kwargs_493995 = {}
# Getting the type of 'add_newdoc' (line 3772)
add_newdoc_493991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3772, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3772)
add_newdoc_call_result_493996 = invoke(stypy.reporting.localization.Localization(__file__, 3772, 0), add_newdoc_493991, *[str_493992, str_493993, str_493994], **kwargs_493995)


# Call to add_newdoc(...): (line 3808)
# Processing the call arguments (line 3808)
str_493998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3808, 11), 'str', 'scipy.special')
str_493999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3808, 28), 'str', 'iv')
str_494000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3864, (-1)), 'str', '\n    iv(v, z)\n\n    Modified Bessel function of the first kind of real order.\n\n    Parameters\n    ----------\n    v : array_like\n        Order. If `z` is of real type and negative, `v` must be integer\n        valued.\n    z : array_like of float or complex\n        Argument.\n\n    Returns\n    -------\n    out : ndarray\n        Values of the modified Bessel function.\n\n    Notes\n    -----\n    For real `z` and :math:`v \\in [-50, 50]`, the evaluation is carried out\n    using Temme\'s method [1]_.  For larger orders, uniform asymptotic\n    expansions are applied.\n\n    For complex `z` and positive `v`, the AMOS [2]_ `zbesi` routine is\n    called. It uses a power series for small `z`, the asymptotic expansion\n    for large `abs(z)`, the Miller algorithm normalized by the Wronskian\n    and a Neumann series for intermediate magnitudes, and the uniform\n    asymptotic expansions for :math:`I_v(z)` and :math:`J_v(z)` for large\n    orders.  Backward recurrence is used to generate sequences or reduce\n    orders when necessary.\n\n    The calculations above are done in the right half plane and continued\n    into the left half plane by the formula,\n\n    .. math:: I_v(z \\exp(\\pm\\imath\\pi)) = \\exp(\\pm\\pi v) I_v(z)\n\n    (valid when the real part of `z` is positive).  For negative `v`, the\n    formula\n\n    .. math:: I_{-v}(z) = I_v(z) + \\frac{2}{\\pi} \\sin(\\pi v) K_v(z)\n\n    is used, where :math:`K_v(z)` is the modified Bessel function of the\n    second kind, evaluated using the AMOS routine `zbesk`.\n\n    See also\n    --------\n    kve : This function with leading exponential behavior stripped off.\n\n    References\n    ----------\n    .. [1] Temme, Journal of Computational Physics, vol 21, 343 (1976)\n    .. [2] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 3808)
kwargs_494001 = {}
# Getting the type of 'add_newdoc' (line 3808)
add_newdoc_493997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3808, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3808)
add_newdoc_call_result_494002 = invoke(stypy.reporting.localization.Localization(__file__, 3808, 0), add_newdoc_493997, *[str_493998, str_493999, str_494000], **kwargs_494001)


# Call to add_newdoc(...): (line 3866)
# Processing the call arguments (line 3866)
str_494004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3866, 11), 'str', 'scipy.special')
str_494005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3866, 28), 'str', 'ive')
str_494006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3916, (-1)), 'str', '\n    ive(v, z)\n\n    Exponentially scaled modified Bessel function of the first kind\n\n    Defined as::\n\n        ive(v, z) = iv(v, z) * exp(-abs(z.real))\n\n    Parameters\n    ----------\n    v : array_like of float\n        Order.\n    z : array_like of float or complex\n        Argument.\n\n    Returns\n    -------\n    out : ndarray\n        Values of the exponentially scaled modified Bessel function.\n\n    Notes\n    -----\n    For positive `v`, the AMOS [1]_ `zbesi` routine is called. It uses a\n    power series for small `z`, the asymptotic expansion for large\n    `abs(z)`, the Miller algorithm normalized by the Wronskian and a\n    Neumann series for intermediate magnitudes, and the uniform asymptotic\n    expansions for :math:`I_v(z)` and :math:`J_v(z)` for large orders.\n    Backward recurrence is used to generate sequences or reduce orders when\n    necessary.\n\n    The calculations above are done in the right half plane and continued\n    into the left half plane by the formula,\n\n    .. math:: I_v(z \\exp(\\pm\\imath\\pi)) = \\exp(\\pm\\pi v) I_v(z)\n\n    (valid when the real part of `z` is positive).  For negative `v`, the\n    formula\n\n    .. math:: I_{-v}(z) = I_v(z) + \\frac{2}{\\pi} \\sin(\\pi v) K_v(z)\n\n    is used, where :math:`K_v(z)` is the modified Bessel function of the\n    second kind, evaluated using the AMOS routine `zbesk`.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 3866)
kwargs_494007 = {}
# Getting the type of 'add_newdoc' (line 3866)
add_newdoc_494003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3866, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3866)
add_newdoc_call_result_494008 = invoke(stypy.reporting.localization.Localization(__file__, 3866, 0), add_newdoc_494003, *[str_494004, str_494005, str_494006], **kwargs_494007)


# Call to add_newdoc(...): (line 3918)
# Processing the call arguments (line 3918)
str_494010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3918, 11), 'str', 'scipy.special')
str_494011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3918, 28), 'str', 'j0')
str_494012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3963, (-1)), 'str', '\n    j0(x)\n\n    Bessel function of the first kind of order 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float).\n\n    Returns\n    -------\n    J : ndarray\n        Value of the Bessel function of the first kind of order 0 at `x`.\n\n    Notes\n    -----\n    The domain is divided into the intervals [0, 5] and (5, infinity). In the\n    first interval the following rational approximation is used:\n\n    .. math::\n\n        J_0(x) \\approx (w - r_1^2)(w - r_2^2) \\frac{P_3(w)}{Q_8(w)},\n\n    where :math:`w = x^2` and :math:`r_1`, :math:`r_2` are the zeros of\n    :math:`J_0`, and :math:`P_3` and :math:`Q_8` are polynomials of degrees 3\n    and 8, respectively.\n\n    In the second interval, the Hankel asymptotic expansion is employed with\n    two rational functions of degree 6/6 and 7/7.\n\n    This function is a wrapper for the Cephes [1]_ routine `j0`.\n    It should not to be confused with the spherical Bessel functions (see\n    `spherical_jn`).\n\n    See also\n    --------\n    jv : Bessel function of real order and complex argument.\n    spherical_jn : spherical Bessel functions.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 3918)
kwargs_494013 = {}
# Getting the type of 'add_newdoc' (line 3918)
add_newdoc_494009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3918, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3918)
add_newdoc_call_result_494014 = invoke(stypy.reporting.localization.Localization(__file__, 3918, 0), add_newdoc_494009, *[str_494010, str_494011, str_494012], **kwargs_494013)


# Call to add_newdoc(...): (line 3965)
# Processing the call arguments (line 3965)
str_494016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3965, 11), 'str', 'scipy.special')
str_494017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3965, 28), 'str', 'j1')
str_494018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4002, (-1)), 'str', '\n    j1(x)\n\n    Bessel function of the first kind of order 1.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float).\n\n    Returns\n    -------\n    J : ndarray\n        Value of the Bessel function of the first kind of order 1 at `x`.\n\n    Notes\n    -----\n    The domain is divided into the intervals [0, 8] and (8, infinity). In the\n    first interval a 24 term Chebyshev expansion is used. In the second, the\n    asymptotic trigonometric representation is employed using two rational\n    functions of degree 5/5.\n\n    This function is a wrapper for the Cephes [1]_ routine `j1`.\n    It should not to be confused with the spherical Bessel functions (see\n    `spherical_jn`).\n\n    See also\n    --------\n    jv\n    spherical_jn : spherical Bessel functions.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 3965)
kwargs_494019 = {}
# Getting the type of 'add_newdoc' (line 3965)
add_newdoc_494015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3965, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 3965)
add_newdoc_call_result_494020 = invoke(stypy.reporting.localization.Localization(__file__, 3965, 0), add_newdoc_494015, *[str_494016, str_494017, str_494018], **kwargs_494019)


# Call to add_newdoc(...): (line 4004)
# Processing the call arguments (line 4004)
str_494022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4004, 11), 'str', 'scipy.special')
str_494023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4004, 28), 'str', 'jn')
str_494024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4020, (-1)), 'str', '\n    jn(n, x)\n\n    Bessel function of the first kind of integer order and real argument.\n\n    Notes\n    -----\n    `jn` is an alias of `jv`.\n    Not to be confused with the spherical Bessel functions (see `spherical_jn`).\n\n    See also\n    --------\n    jv\n    spherical_jn : spherical Bessel functions.\n\n    ')
# Processing the call keyword arguments (line 4004)
kwargs_494025 = {}
# Getting the type of 'add_newdoc' (line 4004)
add_newdoc_494021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4004, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4004)
add_newdoc_call_result_494026 = invoke(stypy.reporting.localization.Localization(__file__, 4004, 0), add_newdoc_494021, *[str_494022, str_494023, str_494024], **kwargs_494025)


# Call to add_newdoc(...): (line 4022)
# Processing the call arguments (line 4022)
str_494028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4022, 11), 'str', 'scipy.special')
str_494029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4022, 28), 'str', 'jv')
str_494030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4072, (-1)), 'str', '\n    jv(v, z)\n\n    Bessel function of the first kind of real order and complex argument.\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    J : ndarray\n        Value of the Bessel function, :math:`J_v(z)`.\n\n    Notes\n    -----\n    For positive `v` values, the computation is carried out using the AMOS\n    [1]_ `zbesj` routine, which exploits the connection to the modified\n    Bessel function :math:`I_v`,\n\n    .. math::\n        J_v(z) = \\exp(v\\pi\\imath/2) I_v(-\\imath z)\\qquad (\\Im z > 0)\n\n        J_v(z) = \\exp(-v\\pi\\imath/2) I_v(\\imath z)\\qquad (\\Im z < 0)\n\n    For negative `v` values the formula,\n\n    .. math:: J_{-v}(z) = J_v(z) \\cos(\\pi v) - Y_v(z) \\sin(\\pi v)\n\n    is used, where :math:`Y_v(z)` is the Bessel function of the second\n    kind, computed using the AMOS routine `zbesy`.  Note that the second\n    term is exactly zero for integer `v`; to improve accuracy the second\n    term is explicitly omitted for `v` values such that `v = floor(v)`.\n\n    Not to be confused with the spherical Bessel functions (see `spherical_jn`).\n\n    See also\n    --------\n    jve : :math:`J_v` with leading exponential behavior stripped off.\n    spherical_jn : spherical Bessel functions.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 4022)
kwargs_494031 = {}
# Getting the type of 'add_newdoc' (line 4022)
add_newdoc_494027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4022, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4022)
add_newdoc_call_result_494032 = invoke(stypy.reporting.localization.Localization(__file__, 4022, 0), add_newdoc_494027, *[str_494028, str_494029, str_494030], **kwargs_494031)


# Call to add_newdoc(...): (line 4074)
# Processing the call arguments (line 4074)
str_494034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4074, 11), 'str', 'scipy.special')
str_494035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4074, 28), 'str', 'jve')
str_494036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4121, (-1)), 'str', '\n    jve(v, z)\n\n    Exponentially scaled Bessel function of order `v`.\n\n    Defined as::\n\n        jve(v, z) = jv(v, z) * exp(-abs(z.imag))\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    J : ndarray\n        Value of the exponentially scaled Bessel function.\n\n    Notes\n    -----\n    For positive `v` values, the computation is carried out using the AMOS\n    [1]_ `zbesj` routine, which exploits the connection to the modified\n    Bessel function :math:`I_v`,\n\n    .. math::\n        J_v(z) = \\exp(v\\pi\\imath/2) I_v(-\\imath z)\\qquad (\\Im z > 0)\n\n        J_v(z) = \\exp(-v\\pi\\imath/2) I_v(\\imath z)\\qquad (\\Im z < 0)\n\n    For negative `v` values the formula,\n\n    .. math:: J_{-v}(z) = J_v(z) \\cos(\\pi v) - Y_v(z) \\sin(\\pi v)\n\n    is used, where :math:`Y_v(z)` is the Bessel function of the second\n    kind, computed using the AMOS routine `zbesy`.  Note that the second\n    term is exactly zero for integer `v`; to improve accuracy the second\n    term is explicitly omitted for `v` values such that `v = floor(v)`.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 4074)
kwargs_494037 = {}
# Getting the type of 'add_newdoc' (line 4074)
add_newdoc_494033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4074, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4074)
add_newdoc_call_result_494038 = invoke(stypy.reporting.localization.Localization(__file__, 4074, 0), add_newdoc_494033, *[str_494034, str_494035, str_494036], **kwargs_494037)


# Call to add_newdoc(...): (line 4123)
# Processing the call arguments (line 4123)
str_494040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4123, 11), 'str', 'scipy.special')
str_494041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4123, 28), 'str', 'k0')
str_494042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4158, (-1)), 'str', '\n    k0(x)\n\n    Modified Bessel function of the second kind of order 0, :math:`K_0`.\n\n    This function is also sometimes referred to as the modified Bessel\n    function of the third kind of order 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float).\n\n    Returns\n    -------\n    K : ndarray\n        Value of the modified Bessel function :math:`K_0` at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 2] and (2, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `k0`.\n\n    See also\n    --------\n    kv\n    k0e\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 4123)
kwargs_494043 = {}
# Getting the type of 'add_newdoc' (line 4123)
add_newdoc_494039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4123, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4123)
add_newdoc_call_result_494044 = invoke(stypy.reporting.localization.Localization(__file__, 4123, 0), add_newdoc_494039, *[str_494040, str_494041, str_494042], **kwargs_494043)


# Call to add_newdoc(...): (line 4160)
# Processing the call arguments (line 4160)
str_494046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4160, 11), 'str', 'scipy.special')
str_494047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4160, 28), 'str', 'k0e')
str_494048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4197, (-1)), 'str', '\n    k0e(x)\n\n    Exponentially scaled modified Bessel function K of order 0\n\n    Defined as::\n\n        k0e(x) = exp(x) * k0(x).\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    K : ndarray\n        Value of the exponentially scaled modified Bessel function K of order\n        0 at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 2] and (2, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `k0e`.\n\n    See also\n    --------\n    kv\n    k0\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 4160)
kwargs_494049 = {}
# Getting the type of 'add_newdoc' (line 4160)
add_newdoc_494045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4160, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4160)
add_newdoc_call_result_494050 = invoke(stypy.reporting.localization.Localization(__file__, 4160, 0), add_newdoc_494045, *[str_494046, str_494047, str_494048], **kwargs_494049)


# Call to add_newdoc(...): (line 4199)
# Processing the call arguments (line 4199)
str_494052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4199, 11), 'str', 'scipy.special')
str_494053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4199, 28), 'str', 'k1')
str_494054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4231, (-1)), 'str', '\n    k1(x)\n\n    Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    K : ndarray\n        Value of the modified Bessel function K of order 1 at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 2] and (2, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `k1`.\n\n    See also\n    --------\n    kv\n    k1e\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 4199)
kwargs_494055 = {}
# Getting the type of 'add_newdoc' (line 4199)
add_newdoc_494051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4199, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4199)
add_newdoc_call_result_494056 = invoke(stypy.reporting.localization.Localization(__file__, 4199, 0), add_newdoc_494051, *[str_494052, str_494053, str_494054], **kwargs_494055)


# Call to add_newdoc(...): (line 4233)
# Processing the call arguments (line 4233)
str_494058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4233, 11), 'str', 'scipy.special')
str_494059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4233, 28), 'str', 'k1e')
str_494060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4270, (-1)), 'str', '\n    k1e(x)\n\n    Exponentially scaled modified Bessel function K of order 1\n\n    Defined as::\n\n        k1e(x) = exp(x) * k1(x)\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float)\n\n    Returns\n    -------\n    K : ndarray\n        Value of the exponentially scaled modified Bessel function K of order\n        1 at `x`.\n\n    Notes\n    -----\n    The range is partitioned into the two intervals [0, 2] and (2, infinity).\n    Chebyshev polynomial expansions are employed in each interval.\n\n    This function is a wrapper for the Cephes [1]_ routine `k1e`.\n\n    See also\n    --------\n    kv\n    k1\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 4233)
kwargs_494061 = {}
# Getting the type of 'add_newdoc' (line 4233)
add_newdoc_494057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4233, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4233)
add_newdoc_call_result_494062 = invoke(stypy.reporting.localization.Localization(__file__, 4233, 0), add_newdoc_494057, *[str_494058, str_494059, str_494060], **kwargs_494061)


# Call to add_newdoc(...): (line 4272)
# Processing the call arguments (line 4272)
str_494064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4272, 11), 'str', 'scipy.special')
str_494065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4272, 28), 'str', 'kei')
str_494066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4277, (-1)), 'str', '\n    kei(x)\n\n    Kelvin function ker\n    ')
# Processing the call keyword arguments (line 4272)
kwargs_494067 = {}
# Getting the type of 'add_newdoc' (line 4272)
add_newdoc_494063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4272, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4272)
add_newdoc_call_result_494068 = invoke(stypy.reporting.localization.Localization(__file__, 4272, 0), add_newdoc_494063, *[str_494064, str_494065, str_494066], **kwargs_494067)


# Call to add_newdoc(...): (line 4279)
# Processing the call arguments (line 4279)
str_494070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4279, 11), 'str', 'scipy.special')
str_494071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4279, 28), 'str', 'keip')
str_494072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4284, (-1)), 'str', '\n    keip(x)\n\n    Derivative of the Kelvin function kei\n    ')
# Processing the call keyword arguments (line 4279)
kwargs_494073 = {}
# Getting the type of 'add_newdoc' (line 4279)
add_newdoc_494069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4279, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4279)
add_newdoc_call_result_494074 = invoke(stypy.reporting.localization.Localization(__file__, 4279, 0), add_newdoc_494069, *[str_494070, str_494071, str_494072], **kwargs_494073)


# Call to add_newdoc(...): (line 4286)
# Processing the call arguments (line 4286)
str_494076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4286, 11), 'str', 'scipy.special')
str_494077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4286, 28), 'str', 'kelvin')
str_494078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4300, (-1)), 'str', '\n    kelvin(x)\n\n    Kelvin functions as complex numbers\n\n    Returns\n    -------\n    Be, Ke, Bep, Kep\n        The tuple (Be, Ke, Bep, Kep) contains complex numbers\n        representing the real and imaginary Kelvin functions and their\n        derivatives evaluated at `x`.  For example, kelvin(x)[0].real =\n        ber x and kelvin(x)[0].imag = bei x with similar relationships\n        for ker and kei.\n    ')
# Processing the call keyword arguments (line 4286)
kwargs_494079 = {}
# Getting the type of 'add_newdoc' (line 4286)
add_newdoc_494075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4286, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4286)
add_newdoc_call_result_494080 = invoke(stypy.reporting.localization.Localization(__file__, 4286, 0), add_newdoc_494075, *[str_494076, str_494077, str_494078], **kwargs_494079)


# Call to add_newdoc(...): (line 4302)
# Processing the call arguments (line 4302)
str_494082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4302, 11), 'str', 'scipy.special')
str_494083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4302, 28), 'str', 'ker')
str_494084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4307, (-1)), 'str', '\n    ker(x)\n\n    Kelvin function ker\n    ')
# Processing the call keyword arguments (line 4302)
kwargs_494085 = {}
# Getting the type of 'add_newdoc' (line 4302)
add_newdoc_494081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4302, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4302)
add_newdoc_call_result_494086 = invoke(stypy.reporting.localization.Localization(__file__, 4302, 0), add_newdoc_494081, *[str_494082, str_494083, str_494084], **kwargs_494085)


# Call to add_newdoc(...): (line 4309)
# Processing the call arguments (line 4309)
str_494088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4309, 11), 'str', 'scipy.special')
str_494089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4309, 28), 'str', 'kerp')
str_494090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4314, (-1)), 'str', '\n    kerp(x)\n\n    Derivative of the Kelvin function ker\n    ')
# Processing the call keyword arguments (line 4309)
kwargs_494091 = {}
# Getting the type of 'add_newdoc' (line 4309)
add_newdoc_494087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4309, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4309)
add_newdoc_call_result_494092 = invoke(stypy.reporting.localization.Localization(__file__, 4309, 0), add_newdoc_494087, *[str_494088, str_494089, str_494090], **kwargs_494091)


# Call to add_newdoc(...): (line 4316)
# Processing the call arguments (line 4316)
str_494094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4316, 11), 'str', 'scipy.special')
str_494095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4316, 28), 'str', 'kl_div')
str_494096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4346, (-1)), 'str', '\n    kl_div(x, y)\n\n    Elementwise function for computing Kullback-Leibler divergence.\n\n    .. math:: \\mathrm{kl\\_div}(x, y) = \\begin{cases} x \\log(x / y) - x + y & x > 0, y > 0 \\\\ y & x = 0, y \\ge 0 \\\\ \\infty & \\text{otherwise} \\end{cases}\n\n    Parameters\n    ----------\n    x : ndarray\n        First input array.\n    y : ndarray\n        Second input array.\n\n    Returns\n    -------\n    res : ndarray\n        Output array.\n\n    See Also\n    --------\n    entr, rel_entr\n\n    Notes\n    -----\n    This function is non-negative and is jointly convex in `x` and `y`.\n\n    .. versionadded:: 0.15.0\n\n    ')
# Processing the call keyword arguments (line 4316)
kwargs_494097 = {}
# Getting the type of 'add_newdoc' (line 4316)
add_newdoc_494093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4316, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4316)
add_newdoc_call_result_494098 = invoke(stypy.reporting.localization.Localization(__file__, 4316, 0), add_newdoc_494093, *[str_494094, str_494095, str_494096], **kwargs_494097)


# Call to add_newdoc(...): (line 4348)
# Processing the call arguments (line 4348)
str_494100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4348, 11), 'str', 'scipy.special')
str_494101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4348, 28), 'str', 'kn')
str_494102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4409, (-1)), 'str', '\n    kn(n, x)\n\n    Modified Bessel function of the second kind of integer order `n`\n\n    Returns the modified Bessel function of the second kind for integer order\n    `n` at real `z`.\n\n    These are also sometimes called functions of the third kind, Basset\n    functions, or Macdonald functions.\n\n    Parameters\n    ----------\n    n : array_like of int\n        Order of Bessel functions (floats will truncate with a warning)\n    z : array_like of float\n        Argument at which to evaluate the Bessel functions\n\n    Returns\n    -------\n    out : ndarray\n        The results\n\n    Notes\n    -----\n    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the\n    algorithm used, see [2]_ and the references therein.\n\n    See Also\n    --------\n    kv : Same function, but accepts real order and complex argument\n    kvp : Derivative of this function\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel\n           functions of a complex argument and nonnegative order", ACM\n           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265\n\n    Examples\n    --------\n    Plot the function of several orders for real input:\n\n    >>> from scipy.special import kn\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(0, 5, 1000)\n    >>> for N in range(6):\n    ...     plt.plot(x, kn(N, x), label=\'$K_{}(x)$\'.format(N))\n    >>> plt.ylim(0, 10)\n    >>> plt.legend()\n    >>> plt.title(r\'Modified Bessel function of the second kind $K_n(x)$\')\n    >>> plt.show()\n\n    Calculate for a single value at multiple orders:\n\n    >>> kn([4, 5, 6], 1)\n    array([   44.23241585,   360.9605896 ,  3653.83831186])\n    ')
# Processing the call keyword arguments (line 4348)
kwargs_494103 = {}
# Getting the type of 'add_newdoc' (line 4348)
add_newdoc_494099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4348, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4348)
add_newdoc_call_result_494104 = invoke(stypy.reporting.localization.Localization(__file__, 4348, 0), add_newdoc_494099, *[str_494100, str_494101, str_494102], **kwargs_494103)


# Call to add_newdoc(...): (line 4411)
# Processing the call arguments (line 4411)
str_494106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4411, 11), 'str', 'scipy.special')
str_494107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4411, 28), 'str', 'kolmogi')
str_494108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4418, (-1)), 'str', '\n    kolmogi(p)\n\n    Inverse function to kolmogorov\n\n    Returns y such that ``kolmogorov(y) == p``.\n    ')
# Processing the call keyword arguments (line 4411)
kwargs_494109 = {}
# Getting the type of 'add_newdoc' (line 4411)
add_newdoc_494105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4411, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4411)
add_newdoc_call_result_494110 = invoke(stypy.reporting.localization.Localization(__file__, 4411, 0), add_newdoc_494105, *[str_494106, str_494107, str_494108], **kwargs_494109)


# Call to add_newdoc(...): (line 4420)
# Processing the call arguments (line 4420)
str_494112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4420, 11), 'str', 'scipy.special')
str_494113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4420, 28), 'str', 'kolmogorov')
str_494114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4431, (-1)), 'str', "\n    kolmogorov(y)\n\n    Complementary cumulative distribution function of Kolmogorov distribution\n\n    Returns the complementary cumulative distribution function of\n    Kolmogorov's limiting distribution (Kn* for large n) of a\n    two-sided test for equality between an empirical and a theoretical\n    distribution. It is equal to the (limit as n->infinity of the)\n    probability that sqrt(n) * max absolute deviation > y.\n    ")
# Processing the call keyword arguments (line 4420)
kwargs_494115 = {}
# Getting the type of 'add_newdoc' (line 4420)
add_newdoc_494111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4420, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4420)
add_newdoc_call_result_494116 = invoke(stypy.reporting.localization.Localization(__file__, 4420, 0), add_newdoc_494111, *[str_494112, str_494113, str_494114], **kwargs_494115)


# Call to add_newdoc(...): (line 4433)
# Processing the call arguments (line 4433)
str_494118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4433, 11), 'str', 'scipy.special')
str_494119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4433, 28), 'str', 'kv')
str_494120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4504, (-1)), 'str', '\n    kv(v, z)\n\n    Modified Bessel function of the second kind of real order `v`\n\n    Returns the modified Bessel function of the second kind for real order\n    `v` at complex `z`.\n\n    These are also sometimes called functions of the third kind, Basset\n    functions, or Macdonald functions.  They are defined as those solutions\n    of the modified Bessel equation for which,\n\n    .. math::\n        K_v(x) \\sim \\sqrt{\\pi/(2x)} \\exp(-x)\n\n    as :math:`x \\to \\infty` [3]_.\n\n    Parameters\n    ----------\n    v : array_like of float\n        Order of Bessel functions\n    z : array_like of complex\n        Argument at which to evaluate the Bessel functions\n\n    Returns\n    -------\n    out : ndarray\n        The results. Note that input must be of complex type to get complex\n        output, e.g. ``kv(3, -2+0j)`` instead of ``kv(3, -2)``.\n\n    Notes\n    -----\n    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the\n    algorithm used, see [2]_ and the references therein.\n\n    See Also\n    --------\n    kve : This function with leading exponential behavior stripped off.\n    kvp : Derivative of this function\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel\n           functions of a complex argument and nonnegative order", ACM\n           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265\n    .. [3] NIST Digital Library of Mathematical Functions,\n           Eq. 10.25.E3. http://dlmf.nist.gov/10.25.E3\n\n    Examples\n    --------\n    Plot the function of several orders for real input:\n\n    >>> from scipy.special import kv\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(0, 5, 1000)\n    >>> for N in np.linspace(0, 6, 5):\n    ...     plt.plot(x, kv(N, x), label=\'$K_{{{}}}(x)$\'.format(N))\n    >>> plt.ylim(0, 10)\n    >>> plt.legend()\n    >>> plt.title(r\'Modified Bessel function of the second kind $K_\\nu(x)$\')\n    >>> plt.show()\n\n    Calculate for a single value at multiple orders:\n\n    >>> kv([4, 4.5, 5], 1+2j)\n    array([ 0.1992+2.3892j,  2.3493+3.6j   ,  7.2827+3.8104j])\n\n    ')
# Processing the call keyword arguments (line 4433)
kwargs_494121 = {}
# Getting the type of 'add_newdoc' (line 4433)
add_newdoc_494117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4433, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4433)
add_newdoc_call_result_494122 = invoke(stypy.reporting.localization.Localization(__file__, 4433, 0), add_newdoc_494117, *[str_494118, str_494119, str_494120], **kwargs_494121)


# Call to add_newdoc(...): (line 4506)
# Processing the call arguments (line 4506)
str_494124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4506, 11), 'str', 'scipy.special')
str_494125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4506, 28), 'str', 'kve')
str_494126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4543, (-1)), 'str', '\n    kve(v, z)\n\n    Exponentially scaled modified Bessel function of the second kind.\n\n    Returns the exponentially scaled, modified Bessel function of the\n    second kind (sometimes called the third kind) for real order `v` at\n    complex `z`::\n\n        kve(v, z) = kv(v, z) * exp(z)\n\n    Parameters\n    ----------\n    v : array_like of float\n        Order of Bessel functions\n    z : array_like of complex\n        Argument at which to evaluate the Bessel functions\n\n    Returns\n    -------\n    out : ndarray\n        The exponentially scaled modified Bessel function of the second kind.\n\n    Notes\n    -----\n    Wrapper for AMOS [1]_ routine `zbesk`.  For a discussion of the\n    algorithm used, see [2]_ and the references therein.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    .. [2] Donald E. Amos, "Algorithm 644: A portable package for Bessel\n           functions of a complex argument and nonnegative order", ACM\n           TOMS Vol. 12 Issue 3, Sept. 1986, p. 265\n    ')
# Processing the call keyword arguments (line 4506)
kwargs_494127 = {}
# Getting the type of 'add_newdoc' (line 4506)
add_newdoc_494123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4506, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4506)
add_newdoc_call_result_494128 = invoke(stypy.reporting.localization.Localization(__file__, 4506, 0), add_newdoc_494123, *[str_494124, str_494125, str_494126], **kwargs_494127)


# Call to add_newdoc(...): (line 4545)
# Processing the call arguments (line 4545)
str_494130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4545, 11), 'str', 'scipy.special')
str_494131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4545, 28), 'str', '_lanczos_sum_expg_scaled')
str_494132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4548, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 4545)
kwargs_494133 = {}
# Getting the type of 'add_newdoc' (line 4545)
add_newdoc_494129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4545, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4545)
add_newdoc_call_result_494134 = invoke(stypy.reporting.localization.Localization(__file__, 4545, 0), add_newdoc_494129, *[str_494130, str_494131, str_494132], **kwargs_494133)


# Call to add_newdoc(...): (line 4550)
# Processing the call arguments (line 4550)
str_494136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4550, 11), 'str', 'scipy.special')
str_494137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4550, 28), 'str', '_lgam1p')
str_494138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4553, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 4550)
kwargs_494139 = {}
# Getting the type of 'add_newdoc' (line 4550)
add_newdoc_494135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4550, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4550)
add_newdoc_call_result_494140 = invoke(stypy.reporting.localization.Localization(__file__, 4550, 0), add_newdoc_494135, *[str_494136, str_494137, str_494138], **kwargs_494139)


# Call to add_newdoc(...): (line 4555)
# Processing the call arguments (line 4555)
str_494142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4555, 11), 'str', 'scipy.special')
str_494143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4555, 28), 'str', 'log1p')
str_494144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4560, (-1)), 'str', '\n    log1p(x)\n\n    Calculates log(1+x) for use when `x` is near zero\n    ')
# Processing the call keyword arguments (line 4555)
kwargs_494145 = {}
# Getting the type of 'add_newdoc' (line 4555)
add_newdoc_494141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4555, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4555)
add_newdoc_call_result_494146 = invoke(stypy.reporting.localization.Localization(__file__, 4555, 0), add_newdoc_494141, *[str_494142, str_494143, str_494144], **kwargs_494145)


# Call to add_newdoc(...): (line 4562)
# Processing the call arguments (line 4562)
str_494148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4562, 11), 'str', 'scipy.special')
str_494149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4562, 28), 'str', '_log1pmx')
str_494150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4565, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 4562)
kwargs_494151 = {}
# Getting the type of 'add_newdoc' (line 4562)
add_newdoc_494147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4562, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4562)
add_newdoc_call_result_494152 = invoke(stypy.reporting.localization.Localization(__file__, 4562, 0), add_newdoc_494147, *[str_494148, str_494149, str_494150], **kwargs_494151)


# Call to add_newdoc(...): (line 4567)
# Processing the call arguments (line 4567)
str_494154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4567, 11), 'str', 'scipy.special')
str_494155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4567, 28), 'str', 'logit')
str_494156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4624, (-1)), 'str', "\n    logit(x)\n\n    Logit ufunc for ndarrays.\n\n    The logit function is defined as logit(p) = log(p/(1-p)).\n    Note that logit(0) = -inf, logit(1) = inf, and logit(p)\n    for p<0 or p>1 yields nan.\n\n    Parameters\n    ----------\n    x : ndarray\n        The ndarray to apply logit to element-wise.\n\n    Returns\n    -------\n    out : ndarray\n        An ndarray of the same shape as x. Its entries\n        are logit of the corresponding entry of x.\n\n    See Also\n    --------\n    expit\n\n    Notes\n    -----\n    As a ufunc logit takes a number of optional\n    keyword arguments. For more information\n    see `ufuncs <https://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_\n\n    .. versionadded:: 0.10.0\n\n    Examples\n    --------\n    >>> from scipy.special import logit, expit\n\n    >>> logit([0, 0.25, 0.5, 0.75, 1])\n    array([       -inf, -1.09861229,  0.        ,  1.09861229,         inf])\n\n    `expit` is the inverse of `logit`:\n\n    >>> expit(logit([0.1, 0.75, 0.999]))\n    array([ 0.1  ,  0.75 ,  0.999])\n\n    Plot logit(x) for x in [0, 1]:\n\n    >>> import matplotlib.pyplot as plt\n    >>> x = np.linspace(0, 1, 501)\n    >>> y = logit(x)\n    >>> plt.plot(x, y)\n    >>> plt.grid()\n    >>> plt.ylim(-6, 6)\n    >>> plt.xlabel('x')\n    >>> plt.title('logit(x)')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 4567)
kwargs_494157 = {}
# Getting the type of 'add_newdoc' (line 4567)
add_newdoc_494153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4567, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4567)
add_newdoc_call_result_494158 = invoke(stypy.reporting.localization.Localization(__file__, 4567, 0), add_newdoc_494153, *[str_494154, str_494155, str_494156], **kwargs_494157)


# Call to add_newdoc(...): (line 4626)
# Processing the call arguments (line 4626)
str_494160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4626, 11), 'str', 'scipy.special')
str_494161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4626, 28), 'str', 'lpmv')
str_494162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4679, (-1)), 'str', '\n    lpmv(m, v, x)\n\n    Associated Legendre function of integer order and real degree.\n\n    Defined as\n\n    .. math::\n\n        P_v^m = (-1)^m (1 - x^2)^{m/2} \\frac{d^m}{dx^m} P_v(x)\n\n    where\n\n    .. math::\n\n        P_v = \\sum_{k = 0}^\\infty \\frac{(-v)_k (v + 1)_k}{(k!)^2}\n                \\left(\\frac{1 - x}{2}\\right)^k\n\n    is the Legendre function of the first kind. Here :math:`(\\cdot)_k`\n    is the Pochhammer symbol; see `poch`.\n\n    Parameters\n    ----------\n    m : array_like\n        Order (int or float). If passed a float not equal to an\n        integer the function returns NaN.\n    v : array_like\n        Degree (float).\n    x : array_like\n        Argument (float). Must have ``|x| <= 1``.\n\n    Returns\n    -------\n    pmv : ndarray\n        Value of the associated Legendre function.\n\n    See Also\n    --------\n    lpmn : Compute the associated Legendre function for all orders\n           ``0, ..., m`` and degrees ``0, ..., n``.\n    clpmn : Compute the associated Legendre function at complex\n            arguments.\n\n    Notes\n    -----\n    Note that this implementation includes the Condon-Shortley phase.\n\n    References\n    ----------\n    .. [1] Zhang, Jin, "Computation of Special Functions", John Wiley\n           and Sons, Inc, 1996.\n\n    ')
# Processing the call keyword arguments (line 4626)
kwargs_494163 = {}
# Getting the type of 'add_newdoc' (line 4626)
add_newdoc_494159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4626, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4626)
add_newdoc_call_result_494164 = invoke(stypy.reporting.localization.Localization(__file__, 4626, 0), add_newdoc_494159, *[str_494160, str_494161, str_494162], **kwargs_494163)


# Call to add_newdoc(...): (line 4681)
# Processing the call arguments (line 4681)
str_494166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4681, 11), 'str', 'scipy.special')
str_494167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4681, 28), 'str', 'mathieu_a')
str_494168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4689, (-1)), 'str', "\n    mathieu_a(m, q)\n\n    Characteristic value of even Mathieu functions\n\n    Returns the characteristic value for the even solution,\n    ``ce_m(z, q)``, of Mathieu's equation.\n    ")
# Processing the call keyword arguments (line 4681)
kwargs_494169 = {}
# Getting the type of 'add_newdoc' (line 4681)
add_newdoc_494165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4681, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4681)
add_newdoc_call_result_494170 = invoke(stypy.reporting.localization.Localization(__file__, 4681, 0), add_newdoc_494165, *[str_494166, str_494167, str_494168], **kwargs_494169)


# Call to add_newdoc(...): (line 4691)
# Processing the call arguments (line 4691)
str_494172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4691, 11), 'str', 'scipy.special')
str_494173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4691, 28), 'str', 'mathieu_b')
str_494174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4699, (-1)), 'str', "\n    mathieu_b(m, q)\n\n    Characteristic value of odd Mathieu functions\n\n    Returns the characteristic value for the odd solution,\n    ``se_m(z, q)``, of Mathieu's equation.\n    ")
# Processing the call keyword arguments (line 4691)
kwargs_494175 = {}
# Getting the type of 'add_newdoc' (line 4691)
add_newdoc_494171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4691, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4691)
add_newdoc_call_result_494176 = invoke(stypy.reporting.localization.Localization(__file__, 4691, 0), add_newdoc_494171, *[str_494172, str_494173, str_494174], **kwargs_494175)


# Call to add_newdoc(...): (line 4701)
# Processing the call arguments (line 4701)
str_494178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4701, 11), 'str', 'scipy.special')
str_494179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4701, 28), 'str', 'mathieu_cem')
str_494180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4726, (-1)), 'str', '\n    mathieu_cem(m, q, x)\n\n    Even Mathieu function and its derivative\n\n    Returns the even Mathieu function, ``ce_m(x, q)``, of order `m` and\n    parameter `q` evaluated at `x` (given in degrees).  Also returns the\n    derivative with respect to `x` of ce_m(x, q)\n\n    Parameters\n    ----------\n    m\n        Order of the function\n    q\n        Parameter of the function\n    x\n        Argument of the function, *given in degrees, not radians*\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4701)
kwargs_494181 = {}
# Getting the type of 'add_newdoc' (line 4701)
add_newdoc_494177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4701, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4701)
add_newdoc_call_result_494182 = invoke(stypy.reporting.localization.Localization(__file__, 4701, 0), add_newdoc_494177, *[str_494178, str_494179, str_494180], **kwargs_494181)


# Call to add_newdoc(...): (line 4728)
# Processing the call arguments (line 4728)
str_494184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4728, 11), 'str', 'scipy.special')
str_494185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4728, 28), 'str', 'mathieu_modcem1')
str_494186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4744, (-1)), 'str', '\n    mathieu_modcem1(m, q, x)\n\n    Even modified Mathieu function of the first kind and its derivative\n\n    Evaluates the even modified Mathieu function of the first kind,\n    ``Mc1m(x, q)``, and its derivative at `x` for order `m` and parameter\n    `q`.\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4728)
kwargs_494187 = {}
# Getting the type of 'add_newdoc' (line 4728)
add_newdoc_494183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4728, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4728)
add_newdoc_call_result_494188 = invoke(stypy.reporting.localization.Localization(__file__, 4728, 0), add_newdoc_494183, *[str_494184, str_494185, str_494186], **kwargs_494187)


# Call to add_newdoc(...): (line 4746)
# Processing the call arguments (line 4746)
str_494190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4746, 11), 'str', 'scipy.special')
str_494191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4746, 28), 'str', 'mathieu_modcem2')
str_494192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4762, (-1)), 'str', '\n    mathieu_modcem2(m, q, x)\n\n    Even modified Mathieu function of the second kind and its derivative\n\n    Evaluates the even modified Mathieu function of the second kind,\n    Mc2m(x, q), and its derivative at `x` (given in degrees) for order `m`\n    and parameter `q`.\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4746)
kwargs_494193 = {}
# Getting the type of 'add_newdoc' (line 4746)
add_newdoc_494189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4746, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4746)
add_newdoc_call_result_494194 = invoke(stypy.reporting.localization.Localization(__file__, 4746, 0), add_newdoc_494189, *[str_494190, str_494191, str_494192], **kwargs_494193)


# Call to add_newdoc(...): (line 4764)
# Processing the call arguments (line 4764)
str_494196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4764, 11), 'str', 'scipy.special')
str_494197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4764, 28), 'str', 'mathieu_modsem1')
str_494198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4780, (-1)), 'str', '\n    mathieu_modsem1(m, q, x)\n\n    Odd modified Mathieu function of the first kind and its derivative\n\n    Evaluates the odd modified Mathieu function of the first kind,\n    Ms1m(x, q), and its derivative at `x` (given in degrees) for order `m`\n    and parameter `q`.\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4764)
kwargs_494199 = {}
# Getting the type of 'add_newdoc' (line 4764)
add_newdoc_494195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4764, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4764)
add_newdoc_call_result_494200 = invoke(stypy.reporting.localization.Localization(__file__, 4764, 0), add_newdoc_494195, *[str_494196, str_494197, str_494198], **kwargs_494199)


# Call to add_newdoc(...): (line 4782)
# Processing the call arguments (line 4782)
str_494202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4782, 11), 'str', 'scipy.special')
str_494203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4782, 28), 'str', 'mathieu_modsem2')
str_494204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4798, (-1)), 'str', '\n    mathieu_modsem2(m, q, x)\n\n    Odd modified Mathieu function of the second kind and its derivative\n\n    Evaluates the odd modified Mathieu function of the second kind,\n    Ms2m(x, q), and its derivative at `x` (given in degrees) for order `m`\n    and parameter q.\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4782)
kwargs_494205 = {}
# Getting the type of 'add_newdoc' (line 4782)
add_newdoc_494201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4782, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4782)
add_newdoc_call_result_494206 = invoke(stypy.reporting.localization.Localization(__file__, 4782, 0), add_newdoc_494201, *[str_494202, str_494203, str_494204], **kwargs_494205)


# Call to add_newdoc(...): (line 4800)
# Processing the call arguments (line 4800)
str_494208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4800, 11), 'str', 'scipy.special')
str_494209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4800, 28), 'str', 'mathieu_sem')
str_494210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4825, (-1)), 'str', '\n    mathieu_sem(m, q, x)\n\n    Odd Mathieu function and its derivative\n\n    Returns the odd Mathieu function, se_m(x, q), of order `m` and\n    parameter `q` evaluated at `x` (given in degrees).  Also returns the\n    derivative with respect to `x` of se_m(x, q).\n\n    Parameters\n    ----------\n    m\n        Order of the function\n    q\n        Parameter of the function\n    x\n        Argument of the function, *given in degrees, not radians*.\n\n    Returns\n    -------\n    y\n        Value of the function\n    yp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 4800)
kwargs_494211 = {}
# Getting the type of 'add_newdoc' (line 4800)
add_newdoc_494207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4800, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4800)
add_newdoc_call_result_494212 = invoke(stypy.reporting.localization.Localization(__file__, 4800, 0), add_newdoc_494207, *[str_494208, str_494209, str_494210], **kwargs_494211)


# Call to add_newdoc(...): (line 4827)
# Processing the call arguments (line 4827)
str_494214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4827, 11), 'str', 'scipy.special')
str_494215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4827, 28), 'str', 'modfresnelm')
str_494216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4839, (-1)), 'str', '\n    modfresnelm(x)\n\n    Modified Fresnel negative integrals\n\n    Returns\n    -------\n    fm\n        Integral ``F_-(x)``: ``integral(exp(-1j*t*t), t=x..inf)``\n    km\n        Integral ``K_-(x)``: ``1/sqrt(pi)*exp(1j*(x*x+pi/4))*fp``\n    ')
# Processing the call keyword arguments (line 4827)
kwargs_494217 = {}
# Getting the type of 'add_newdoc' (line 4827)
add_newdoc_494213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4827, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4827)
add_newdoc_call_result_494218 = invoke(stypy.reporting.localization.Localization(__file__, 4827, 0), add_newdoc_494213, *[str_494214, str_494215, str_494216], **kwargs_494217)


# Call to add_newdoc(...): (line 4841)
# Processing the call arguments (line 4841)
str_494220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4841, 11), 'str', 'scipy.special')
str_494221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4841, 28), 'str', 'modfresnelp')
str_494222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4853, (-1)), 'str', '\n    modfresnelp(x)\n\n    Modified Fresnel positive integrals\n\n    Returns\n    -------\n    fp\n        Integral ``F_+(x)``: ``integral(exp(1j*t*t), t=x..inf)``\n    kp\n        Integral ``K_+(x)``: ``1/sqrt(pi)*exp(-1j*(x*x+pi/4))*fp``\n    ')
# Processing the call keyword arguments (line 4841)
kwargs_494223 = {}
# Getting the type of 'add_newdoc' (line 4841)
add_newdoc_494219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4841, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4841)
add_newdoc_call_result_494224 = invoke(stypy.reporting.localization.Localization(__file__, 4841, 0), add_newdoc_494219, *[str_494220, str_494221, str_494222], **kwargs_494223)


# Call to add_newdoc(...): (line 4855)
# Processing the call arguments (line 4855)
str_494226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4855, 11), 'str', 'scipy.special')
str_494227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4855, 28), 'str', 'modstruve')
str_494228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4901, (-1)), 'str', '\n    modstruve(v, x)\n\n    Modified Struve function.\n\n    Return the value of the modified Struve function of order `v` at `x`.  The\n    modified Struve function is defined as,\n\n    .. math::\n        L_v(x) = -\\imath \\exp(-\\pi\\imath v/2) H_v(x),\n\n    where :math:`H_v` is the Struve function.\n\n    Parameters\n    ----------\n    v : array_like\n        Order of the modified Struve function (float).\n    x : array_like\n        Argument of the Struve function (float; must be positive unless `v` is\n        an integer).\n\n    Returns\n    -------\n    L : ndarray\n        Value of the modified Struve function of order `v` at `x`.\n\n    Notes\n    -----\n    Three methods discussed in [1]_ are used to evaluate the function:\n\n    - power series\n    - expansion in Bessel functions (if :math:`|z| < |v| + 20`)\n    - asymptotic large-z expansion (if :math:`z \\geq 0.7v + 12`)\n\n    Rounding errors are estimated based on the largest terms in the sums, and\n    the result associated with the smallest error is returned.\n\n    See also\n    --------\n    struve\n\n    References\n    ----------\n    .. [1] NIST Digital Library of Mathematical Functions\n           http://dlmf.nist.gov/11\n    ')
# Processing the call keyword arguments (line 4855)
kwargs_494229 = {}
# Getting the type of 'add_newdoc' (line 4855)
add_newdoc_494225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4855, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4855)
add_newdoc_call_result_494230 = invoke(stypy.reporting.localization.Localization(__file__, 4855, 0), add_newdoc_494225, *[str_494226, str_494227, str_494228], **kwargs_494229)


# Call to add_newdoc(...): (line 4903)
# Processing the call arguments (line 4903)
str_494232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4903, 11), 'str', 'scipy.special')
str_494233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4903, 28), 'str', 'nbdtr')
str_494234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4957, (-1)), 'str', '\n    nbdtr(k, n, p)\n\n    Negative binomial cumulative distribution function.\n\n    Returns the sum of the terms 0 through `k` of the negative binomial\n    distribution probability mass function,\n\n    .. math::\n\n        F = \\sum_{j=0}^k {{n + j - 1}\\choose{j}} p^n (1 - p)^j.\n\n    In a sequence of Bernoulli trials with individual success probabilities\n    `p`, this is the probability that `k` or fewer failures precede the nth\n    success.\n\n    Parameters\n    ----------\n    k : array_like\n        The maximum number of allowed failures (nonnegative int).\n    n : array_like\n        The target number of successes (positive int).\n    p : array_like\n        Probability of success in a single event (float).\n\n    Returns\n    -------\n    F : ndarray\n        The probability of `k` or fewer failures before `n` successes in a\n        sequence of events with individual success probability `p`.\n\n    See also\n    --------\n    nbdtrc\n\n    Notes\n    -----\n    If floating point values are passed for `k` or `n`, they will be truncated\n    to integers.\n\n    The terms are not summed directly; instead the regularized incomplete beta\n    function is employed, according to the formula,\n\n    .. math::\n        \\mathrm{nbdtr}(k, n, p) = I_{p}(n, k + 1).\n\n    Wrapper for the Cephes [1]_ routine `nbdtr`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 4903)
kwargs_494235 = {}
# Getting the type of 'add_newdoc' (line 4903)
add_newdoc_494231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4903, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4903)
add_newdoc_call_result_494236 = invoke(stypy.reporting.localization.Localization(__file__, 4903, 0), add_newdoc_494231, *[str_494232, str_494233, str_494234], **kwargs_494235)


# Call to add_newdoc(...): (line 4959)
# Processing the call arguments (line 4959)
str_494238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4959, 11), 'str', 'scipy.special')
str_494239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4959, 28), 'str', 'nbdtrc')
str_494240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5008, (-1)), 'str', '\n    nbdtrc(k, n, p)\n\n    Negative binomial survival function.\n\n    Returns the sum of the terms `k + 1` to infinity of the negative binomial\n    distribution probability mass function,\n\n    .. math::\n\n        F = \\sum_{j=k + 1}^\\infty {{n + j - 1}\\choose{j}} p^n (1 - p)^j.\n\n    In a sequence of Bernoulli trials with individual success probabilities\n    `p`, this is the probability that more than `k` failures precede the nth\n    success.\n\n    Parameters\n    ----------\n    k : array_like\n        The maximum number of allowed failures (nonnegative int).\n    n : array_like\n        The target number of successes (positive int).\n    p : array_like\n        Probability of success in a single event (float).\n\n    Returns\n    -------\n    F : ndarray\n        The probability of `k + 1` or more failures before `n` successes in a\n        sequence of events with individual success probability `p`.\n\n    Notes\n    -----\n    If floating point values are passed for `k` or `n`, they will be truncated\n    to integers.\n\n    The terms are not summed directly; instead the regularized incomplete beta\n    function is employed, according to the formula,\n\n    .. math::\n        \\mathrm{nbdtrc}(k, n, p) = I_{1 - p}(k + 1, n).\n\n    Wrapper for the Cephes [1]_ routine `nbdtrc`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 4959)
kwargs_494241 = {}
# Getting the type of 'add_newdoc' (line 4959)
add_newdoc_494237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4959, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 4959)
add_newdoc_call_result_494242 = invoke(stypy.reporting.localization.Localization(__file__, 4959, 0), add_newdoc_494237, *[str_494238, str_494239, str_494240], **kwargs_494241)


# Call to add_newdoc(...): (line 5010)
# Processing the call arguments (line 5010)
str_494244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5010, 11), 'str', 'scipy.special')
str_494245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5010, 28), 'str', 'nbdtri')
str_494246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5050, (-1)), 'str', '\n    nbdtri(k, n, y)\n\n    Inverse of `nbdtr` vs `p`.\n\n    Returns the inverse with respect to the parameter `p` of\n    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution\n    function.\n\n    Parameters\n    ----------\n    k : array_like\n        The maximum number of allowed failures (nonnegative int).\n    n : array_like\n        The target number of successes (positive int).\n    y : array_like\n        The probability of `k` or fewer failures before `n` successes (float).\n\n    Returns\n    -------\n    p : ndarray\n        Probability of success in a single event (float) such that\n        `nbdtr(k, n, p) = y`.\n\n    See also\n    --------\n    nbdtr : Cumulative distribution function of the negative binomial.\n    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.\n    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `nbdtri`.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n\n    ')
# Processing the call keyword arguments (line 5010)
kwargs_494247 = {}
# Getting the type of 'add_newdoc' (line 5010)
add_newdoc_494243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5010, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5010)
add_newdoc_call_result_494248 = invoke(stypy.reporting.localization.Localization(__file__, 5010, 0), add_newdoc_494243, *[str_494244, str_494245, str_494246], **kwargs_494247)


# Call to add_newdoc(...): (line 5052)
# Processing the call arguments (line 5052)
str_494250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5052, 11), 'str', 'scipy.special')
str_494251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5052, 28), 'str', 'nbdtrik')
str_494252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5106, (-1)), 'str', '\n    nbdtrik(y, n, p)\n\n    Inverse of `nbdtr` vs `k`.\n\n    Returns the inverse with respect to the parameter `k` of\n    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution\n    function.\n\n    Parameters\n    ----------\n    y : array_like\n        The probability of `k` or fewer failures before `n` successes (float).\n    n : array_like\n        The target number of successes (positive int).\n    p : array_like\n        Probability of success in a single event (float).\n\n    Returns\n    -------\n    k : ndarray\n        The maximum number of allowed failures such that `nbdtr(k, n, p) = y`.\n\n    See also\n    --------\n    nbdtr : Cumulative distribution function of the negative binomial.\n    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.\n    nbdtrin : Inverse with respect to `n` of `nbdtr(k, n, p)`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.\n\n    Formula 26.5.26 of [2]_,\n\n    .. math::\n        \\sum_{j=k + 1}^\\infty {{n + j - 1}\\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),\n\n    is used to reduce calculation of the cumulative distribution function to\n    that of a regularized incomplete beta :math:`I`.\n\n    Computation of `k` involves a search for a value that produces the desired\n    value of `y`.  The search relies on the monotonicity of `y` with `k`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    ')
# Processing the call keyword arguments (line 5052)
kwargs_494253 = {}
# Getting the type of 'add_newdoc' (line 5052)
add_newdoc_494249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5052, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5052)
add_newdoc_call_result_494254 = invoke(stypy.reporting.localization.Localization(__file__, 5052, 0), add_newdoc_494249, *[str_494250, str_494251, str_494252], **kwargs_494253)


# Call to add_newdoc(...): (line 5108)
# Processing the call arguments (line 5108)
str_494256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5108, 11), 'str', 'scipy.special')
str_494257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5108, 28), 'str', 'nbdtrin')
str_494258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5162, (-1)), 'str', '\n    nbdtrin(k, y, p)\n\n    Inverse of `nbdtr` vs `n`.\n\n    Returns the inverse with respect to the parameter `n` of\n    `y = nbdtr(k, n, p)`, the negative binomial cumulative distribution\n    function.\n\n    Parameters\n    ----------\n    k : array_like\n        The maximum number of allowed failures (nonnegative int).\n    y : array_like\n        The probability of `k` or fewer failures before `n` successes (float).\n    p : array_like\n        Probability of success in a single event (float).\n\n    Returns\n    -------\n    n : ndarray\n        The number of successes `n` such that `nbdtr(k, n, p) = y`.\n\n    See also\n    --------\n    nbdtr : Cumulative distribution function of the negative binomial.\n    nbdtri : Inverse with respect to `p` of `nbdtr(k, n, p)`.\n    nbdtrik : Inverse with respect to `k` of `nbdtr(k, n, p)`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdfnbn`.\n\n    Formula 26.5.26 of [2]_,\n\n    .. math::\n        \\sum_{j=k + 1}^\\infty {{n + j - 1}\\choose{j}} p^n (1 - p)^j = I_{1 - p}(k + 1, n),\n\n    is used to reduce calculation of the cumulative distribution function to\n    that of a regularized incomplete beta :math:`I`.\n\n    Computation of `n` involves a search for a value that produces the desired\n    value of `y`.  The search relies on the monotonicity of `y` with `n`.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    ')
# Processing the call keyword arguments (line 5108)
kwargs_494259 = {}
# Getting the type of 'add_newdoc' (line 5108)
add_newdoc_494255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5108, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5108)
add_newdoc_call_result_494260 = invoke(stypy.reporting.localization.Localization(__file__, 5108, 0), add_newdoc_494255, *[str_494256, str_494257, str_494258], **kwargs_494259)


# Call to add_newdoc(...): (line 5164)
# Processing the call arguments (line 5164)
str_494262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5164, 11), 'str', 'scipy.special')
str_494263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5164, 28), 'str', 'ncfdtr')
str_494264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5252, (-1)), 'str', "\n    ncfdtr(dfn, dfd, nc, f)\n\n    Cumulative distribution function of the non-central F distribution.\n\n    The non-central F describes the distribution of,\n\n    .. math::\n        Z = \\frac{X/d_n}{Y/d_d}\n\n    where :math:`X` and :math:`Y` are independently distributed, with\n    :math:`X` distributed non-central :math:`\\chi^2` with noncentrality\n    parameter `nc` and :math:`d_n` degrees of freedom, and :math:`Y`\n    distributed :math:`\\chi^2` with :math:`d_d` degrees of freedom.\n\n    Parameters\n    ----------\n    dfn : array_like\n        Degrees of freedom of the numerator sum of squares.  Range (0, inf).\n    dfd : array_like\n        Degrees of freedom of the denominator sum of squares.  Range (0, inf).\n    nc : array_like\n        Noncentrality parameter.  Should be in range (0, 1e4).\n    f : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    Returns\n    -------\n    cdf : float or ndarray\n        The calculated CDF.  If all inputs are scalar, the return will be a\n        float.  Otherwise it will be an array.\n\n    See Also\n    --------\n    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.\n    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.\n    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.\n    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.\n\n    Notes\n    -----\n    Wrapper for the CDFLIB [1]_ Fortran routine `cdffnc`.\n\n    The cumulative distribution function is computed using Formula 26.6.20 of\n    [2]_:\n\n    .. math::\n        F(d_n, d_d, n_c, f) = \\sum_{j=0}^\\infty e^{-n_c/2} \\frac{(n_c/2)^j}{j!} I_{x}(\\frac{d_n}{2} + j, \\frac{d_d}{2}),\n\n    where :math:`I` is the regularized incomplete beta function, and\n    :math:`x = f d_n/(f d_n + d_d)`.\n\n    The computation time required for this routine is proportional to the\n    noncentrality parameter `nc`.  Very large values of this parameter can\n    consume immense computer resources.  This is why the search range is\n    bounded by 10,000.\n\n    References\n    ----------\n    .. [1] Barry Brown, James Lovato, and Kathy Russell,\n           CDFLIB: Library of Fortran Routines for Cumulative Distribution\n           Functions, Inverses, and Other Parameters.\n    .. [2] Milton Abramowitz and Irene A. Stegun, eds.\n           Handbook of Mathematical Functions with Formulas,\n           Graphs, and Mathematical Tables. New York: Dover, 1972.\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    Plot the CDF of the non-central F distribution, for nc=0.  Compare with the\n    F-distribution from scipy.stats:\n\n    >>> x = np.linspace(-1, 8, num=500)\n    >>> dfn = 3\n    >>> dfd = 2\n    >>> ncf_stats = stats.f.cdf(x, dfn, dfd)\n    >>> ncf_special = special.ncfdtr(dfn, dfd, 0, x)\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, ncf_stats, 'b-', lw=3)\n    >>> ax.plot(x, ncf_special, 'r-')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 5164)
kwargs_494265 = {}
# Getting the type of 'add_newdoc' (line 5164)
add_newdoc_494261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5164, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5164)
add_newdoc_call_result_494266 = invoke(stypy.reporting.localization.Localization(__file__, 5164, 0), add_newdoc_494261, *[str_494262, str_494263, str_494264], **kwargs_494265)


# Call to add_newdoc(...): (line 5254)
# Processing the call arguments (line 5254)
str_494268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5254, 11), 'str', 'scipy.special')
str_494269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5254, 28), 'str', 'ncfdtri')
str_494270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5302, (-1)), 'str', '\n    ncfdtri(dfn, dfd, nc, p)\n\n    Inverse with respect to `f` of the CDF of the non-central F distribution.\n\n    See `ncfdtr` for more details.\n\n    Parameters\n    ----------\n    dfn : array_like\n        Degrees of freedom of the numerator sum of squares.  Range (0, inf).\n    dfd : array_like\n        Degrees of freedom of the denominator sum of squares.  Range (0, inf).\n    nc : array_like\n        Noncentrality parameter.  Should be in range (0, 1e4).\n    p : array_like\n        Value of the cumulative distribution function.  Must be in the\n        range [0, 1].\n\n    Returns\n    -------\n    f : float\n        Quantiles, i.e. the upper limit of integration.\n\n    See Also\n    --------\n    ncfdtr : CDF of the non-central F distribution.\n    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.\n    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.\n    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.\n\n    Examples\n    --------\n    >>> from scipy.special import ncfdtr, ncfdtri\n\n    Compute the CDF for several values of `f`:\n\n    >>> f = [0.5, 1, 1.5]\n    >>> p = ncfdtr(2, 3, 1.5, f)\n    >>> p\n    array([ 0.20782291,  0.36107392,  0.47345752])\n\n    Compute the inverse.  We recover the values of `f`, as expected:\n\n    >>> ncfdtri(2, 3, 1.5, p)\n    array([ 0.5,  1. ,  1.5])\n\n    ')
# Processing the call keyword arguments (line 5254)
kwargs_494271 = {}
# Getting the type of 'add_newdoc' (line 5254)
add_newdoc_494267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5254, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5254)
add_newdoc_call_result_494272 = invoke(stypy.reporting.localization.Localization(__file__, 5254, 0), add_newdoc_494267, *[str_494268, str_494269, str_494270], **kwargs_494271)


# Call to add_newdoc(...): (line 5304)
# Processing the call arguments (line 5304)
str_494274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5304, 11), 'str', 'scipy.special')
str_494275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5304, 28), 'str', 'ncfdtridfd')
str_494276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5360, (-1)), 'str', '\n    ncfdtridfd(dfn, p, nc, f)\n\n    Calculate degrees of freedom (denominator) for the noncentral F-distribution.\n\n    This is the inverse with respect to `dfd` of `ncfdtr`.\n    See `ncfdtr` for more details.\n\n    Parameters\n    ----------\n    dfn : array_like\n        Degrees of freedom of the numerator sum of squares.  Range (0, inf).\n    p : array_like\n        Value of the cumulative distribution function.  Must be in the\n        range [0, 1].\n    nc : array_like\n        Noncentrality parameter.  Should be in range (0, 1e4).\n    f : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    Returns\n    -------\n    dfd : float\n        Degrees of freedom of the denominator sum of squares.\n\n    See Also\n    --------\n    ncfdtr : CDF of the non-central F distribution.\n    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.\n    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.\n    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.\n\n    Notes\n    -----\n    The value of the cumulative noncentral F distribution is not necessarily\n    monotone in either degrees of freedom.  There thus may be two values that\n    provide a given CDF value.  This routine assumes monotonicity and will\n    find an arbitrary one of the two values.\n\n    Examples\n    --------\n    >>> from scipy.special import ncfdtr, ncfdtridfd\n\n    Compute the CDF for several values of `dfd`:\n\n    >>> dfd = [1, 2, 3]\n    >>> p = ncfdtr(2, dfd, 0.25, 15)\n    >>> p\n    array([ 0.8097138 ,  0.93020416,  0.96787852])\n\n    Compute the inverse.  We recover the values of `dfd`, as expected:\n\n    >>> ncfdtridfd(2, p, 0.25, 15)\n    array([ 1.,  2.,  3.])\n\n    ')
# Processing the call keyword arguments (line 5304)
kwargs_494277 = {}
# Getting the type of 'add_newdoc' (line 5304)
add_newdoc_494273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5304, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5304)
add_newdoc_call_result_494278 = invoke(stypy.reporting.localization.Localization(__file__, 5304, 0), add_newdoc_494273, *[str_494274, str_494275, str_494276], **kwargs_494277)


# Call to add_newdoc(...): (line 5362)
# Processing the call arguments (line 5362)
str_494280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5362, 11), 'str', 'scipy.special')
str_494281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5362, 28), 'str', 'ncfdtridfn')
str_494282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5418, (-1)), 'str', '\n    ncfdtridfn(p, dfd, nc, f)\n\n    Calculate degrees of freedom (numerator) for the noncentral F-distribution.\n\n    This is the inverse with respect to `dfn` of `ncfdtr`.\n    See `ncfdtr` for more details.\n\n    Parameters\n    ----------\n    p : array_like\n        Value of the cumulative distribution function.  Must be in the\n        range [0, 1].\n    dfd : array_like\n        Degrees of freedom of the denominator sum of squares.  Range (0, inf).\n    nc : array_like\n        Noncentrality parameter.  Should be in range (0, 1e4).\n    f : float\n        Quantiles, i.e. the upper limit of integration.\n\n    Returns\n    -------\n    dfn : float\n        Degrees of freedom of the numerator sum of squares.\n\n    See Also\n    --------\n    ncfdtr : CDF of the non-central F distribution.\n    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.\n    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.\n    ncfdtrinc : Inverse of `ncfdtr` with respect to `nc`.\n\n    Notes\n    -----\n    The value of the cumulative noncentral F distribution is not necessarily\n    monotone in either degrees of freedom.  There thus may be two values that\n    provide a given CDF value.  This routine assumes monotonicity and will\n    find an arbitrary one of the two values.\n\n    Examples\n    --------\n    >>> from scipy.special import ncfdtr, ncfdtridfn\n\n    Compute the CDF for several values of `dfn`:\n\n    >>> dfn = [1, 2, 3]\n    >>> p = ncfdtr(dfn, 2, 0.25, 15)\n    >>> p\n    array([ 0.92562363,  0.93020416,  0.93188394])\n\n    Compute the inverse.  We recover the values of `dfn`, as expected:\n\n    >>> ncfdtridfn(p, 2, 0.25, 15)\n    array([ 1.,  2.,  3.])\n\n    ')
# Processing the call keyword arguments (line 5362)
kwargs_494283 = {}
# Getting the type of 'add_newdoc' (line 5362)
add_newdoc_494279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5362, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5362)
add_newdoc_call_result_494284 = invoke(stypy.reporting.localization.Localization(__file__, 5362, 0), add_newdoc_494279, *[str_494280, str_494281, str_494282], **kwargs_494283)


# Call to add_newdoc(...): (line 5420)
# Processing the call arguments (line 5420)
str_494286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5420, 11), 'str', 'scipy.special')
str_494287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5420, 28), 'str', 'ncfdtrinc')
str_494288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5469, (-1)), 'str', '\n    ncfdtrinc(dfn, dfd, p, f)\n\n    Calculate non-centrality parameter for non-central F distribution.\n\n    This is the inverse with respect to `nc` of `ncfdtr`.\n    See `ncfdtr` for more details.\n\n    Parameters\n    ----------\n    dfn : array_like\n        Degrees of freedom of the numerator sum of squares.  Range (0, inf).\n    dfd : array_like\n        Degrees of freedom of the denominator sum of squares.  Range (0, inf).\n    p : array_like\n        Value of the cumulative distribution function.  Must be in the\n        range [0, 1].\n    f : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    Returns\n    -------\n    nc : float\n        Noncentrality parameter.\n\n    See Also\n    --------\n    ncfdtr : CDF of the non-central F distribution.\n    ncfdtri : Quantile function; inverse of `ncfdtr` with respect to `f`.\n    ncfdtridfd : Inverse of `ncfdtr` with respect to `dfd`.\n    ncfdtridfn : Inverse of `ncfdtr` with respect to `dfn`.\n\n    Examples\n    --------\n    >>> from scipy.special import ncfdtr, ncfdtrinc\n\n    Compute the CDF for several values of `nc`:\n\n    >>> nc = [0.5, 1.5, 2.0]\n    >>> p = ncfdtr(2, 3, nc, 15)\n    >>> p\n    array([ 0.96309246,  0.94327955,  0.93304098])\n\n    Compute the inverse.  We recover the values of `nc`, as expected:\n\n    >>> ncfdtrinc(2, 3, p, 15)\n    array([ 0.5,  1.5,  2. ])\n\n    ')
# Processing the call keyword arguments (line 5420)
kwargs_494289 = {}
# Getting the type of 'add_newdoc' (line 5420)
add_newdoc_494285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5420, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5420)
add_newdoc_call_result_494290 = invoke(stypy.reporting.localization.Localization(__file__, 5420, 0), add_newdoc_494285, *[str_494286, str_494287, str_494288], **kwargs_494289)


# Call to add_newdoc(...): (line 5471)
# Processing the call arguments (line 5471)
str_494292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5471, 11), 'str', 'scipy.special')
str_494293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5471, 28), 'str', 'nctdtr')
str_494294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5518, (-1)), 'str', "\n    nctdtr(df, nc, t)\n\n    Cumulative distribution function of the non-central `t` distribution.\n\n    Parameters\n    ----------\n    df : array_like\n        Degrees of freedom of the distribution.  Should be in range (0, inf).\n    nc : array_like\n        Noncentrality parameter.  Should be in range (-1e6, 1e6).\n    t : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    Returns\n    -------\n    cdf : float or ndarray\n        The calculated CDF.  If all inputs are scalar, the return will be a\n        float.  Otherwise it will be an array.\n\n    See Also\n    --------\n    nctdtrit : Inverse CDF (iCDF) of the non-central t distribution.\n    nctdtridf : Calculate degrees of freedom, given CDF and iCDF values.\n    nctdtrinc : Calculate non-centrality parameter, given CDF iCDF values.\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    Plot the CDF of the non-central t distribution, for nc=0.  Compare with the\n    t-distribution from scipy.stats:\n\n    >>> x = np.linspace(-5, 5, num=500)\n    >>> df = 3\n    >>> nct_stats = stats.t.cdf(x, df)\n    >>> nct_special = special.nctdtr(df, 0, x)\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, nct_stats, 'b-', lw=3)\n    >>> ax.plot(x, nct_special, 'r-')\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 5471)
kwargs_494295 = {}
# Getting the type of 'add_newdoc' (line 5471)
add_newdoc_494291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5471, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5471)
add_newdoc_call_result_494296 = invoke(stypy.reporting.localization.Localization(__file__, 5471, 0), add_newdoc_494291, *[str_494292, str_494293, str_494294], **kwargs_494295)


# Call to add_newdoc(...): (line 5520)
# Processing the call arguments (line 5520)
str_494298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5520, 11), 'str', 'scipy.special')
str_494299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5520, 28), 'str', 'nctdtridf')
str_494300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5537, (-1)), 'str', '\n    nctdtridf(p, nc, t)\n\n    Calculate degrees of freedom for non-central t distribution.\n\n    See `nctdtr` for more details.\n\n    Parameters\n    ----------\n    p : array_like\n        CDF values, in range (0, 1].\n    nc : array_like\n        Noncentrality parameter.  Should be in range (-1e6, 1e6).\n    t : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    ')
# Processing the call keyword arguments (line 5520)
kwargs_494301 = {}
# Getting the type of 'add_newdoc' (line 5520)
add_newdoc_494297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5520, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5520)
add_newdoc_call_result_494302 = invoke(stypy.reporting.localization.Localization(__file__, 5520, 0), add_newdoc_494297, *[str_494298, str_494299, str_494300], **kwargs_494301)


# Call to add_newdoc(...): (line 5539)
# Processing the call arguments (line 5539)
str_494304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5539, 11), 'str', 'scipy.special')
str_494305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5539, 28), 'str', 'nctdtrinc')
str_494306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5556, (-1)), 'str', '\n    nctdtrinc(df, p, t)\n\n    Calculate non-centrality parameter for non-central t distribution.\n\n    See `nctdtr` for more details.\n\n    Parameters\n    ----------\n    df : array_like\n        Degrees of freedom of the distribution.  Should be in range (0, inf).\n    p : array_like\n        CDF values, in range (0, 1].\n    t : array_like\n        Quantiles, i.e. the upper limit of integration.\n\n    ')
# Processing the call keyword arguments (line 5539)
kwargs_494307 = {}
# Getting the type of 'add_newdoc' (line 5539)
add_newdoc_494303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5539, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5539)
add_newdoc_call_result_494308 = invoke(stypy.reporting.localization.Localization(__file__, 5539, 0), add_newdoc_494303, *[str_494304, str_494305, str_494306], **kwargs_494307)


# Call to add_newdoc(...): (line 5558)
# Processing the call arguments (line 5558)
str_494310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5558, 11), 'str', 'scipy.special')
str_494311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5558, 28), 'str', 'nctdtrit')
str_494312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5575, (-1)), 'str', '\n    nctdtrit(df, nc, p)\n\n    Inverse cumulative distribution function of the non-central t distribution.\n\n    See `nctdtr` for more details.\n\n    Parameters\n    ----------\n    df : array_like\n        Degrees of freedom of the distribution.  Should be in range (0, inf).\n    nc : array_like\n        Noncentrality parameter.  Should be in range (-1e6, 1e6).\n    p : array_like\n        CDF values, in range (0, 1].\n\n    ')
# Processing the call keyword arguments (line 5558)
kwargs_494313 = {}
# Getting the type of 'add_newdoc' (line 5558)
add_newdoc_494309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5558, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5558)
add_newdoc_call_result_494314 = invoke(stypy.reporting.localization.Localization(__file__, 5558, 0), add_newdoc_494309, *[str_494310, str_494311, str_494312], **kwargs_494313)


# Call to add_newdoc(...): (line 5577)
# Processing the call arguments (line 5577)
str_494316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5577, 11), 'str', 'scipy.special')
str_494317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5577, 28), 'str', 'ndtr')
str_494318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5607, (-1)), 'str', '\n    ndtr(x)\n\n    Gaussian cumulative distribution function.\n\n    Returns the area under the standard Gaussian probability\n    density function, integrated from minus infinity to `x`\n\n    .. math::\n\n       \\frac{1}{\\sqrt{2\\pi}} \\int_{-\\infty}^x \\exp(-t^2/2) dt\n\n    Parameters\n    ----------\n    x : array_like, real or complex\n        Argument\n\n    Returns\n    -------\n    ndarray\n        The value of the normal CDF evaluated at `x`\n\n    See Also\n    --------\n    erf\n    erfc\n    scipy.stats.norm\n    log_ndtr\n\n    ')
# Processing the call keyword arguments (line 5577)
kwargs_494319 = {}
# Getting the type of 'add_newdoc' (line 5577)
add_newdoc_494315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5577, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5577)
add_newdoc_call_result_494320 = invoke(stypy.reporting.localization.Localization(__file__, 5577, 0), add_newdoc_494315, *[str_494316, str_494317, str_494318], **kwargs_494319)


# Call to add_newdoc(...): (line 5610)
# Processing the call arguments (line 5610)
str_494322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5610, 11), 'str', 'scipy.special')
str_494323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5610, 28), 'str', 'nrdtrimn')
str_494324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5634, (-1)), 'str', '\n    nrdtrimn(p, x, std)\n\n    Calculate mean of normal distribution given other params.\n\n    Parameters\n    ----------\n    p : array_like\n        CDF values, in range (0, 1].\n    x : array_like\n        Quantiles, i.e. the upper limit of integration.\n    std : array_like\n        Standard deviation.\n\n    Returns\n    -------\n    mn : float or ndarray\n        The mean of the normal distribution.\n\n    See Also\n    --------\n    nrdtrimn, ndtr\n\n    ')
# Processing the call keyword arguments (line 5610)
kwargs_494325 = {}
# Getting the type of 'add_newdoc' (line 5610)
add_newdoc_494321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5610, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5610)
add_newdoc_call_result_494326 = invoke(stypy.reporting.localization.Localization(__file__, 5610, 0), add_newdoc_494321, *[str_494322, str_494323, str_494324], **kwargs_494325)


# Call to add_newdoc(...): (line 5636)
# Processing the call arguments (line 5636)
str_494328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5636, 11), 'str', 'scipy.special')
str_494329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5636, 28), 'str', 'nrdtrisd')
str_494330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5660, (-1)), 'str', '\n    nrdtrisd(p, x, mn)\n\n    Calculate standard deviation of normal distribution given other params.\n\n    Parameters\n    ----------\n    p : array_like\n        CDF values, in range (0, 1].\n    x : array_like\n        Quantiles, i.e. the upper limit of integration.\n    mn : float or ndarray\n        The mean of the normal distribution.\n\n    Returns\n    -------\n    std : array_like\n        Standard deviation.\n\n    See Also\n    --------\n    nrdtristd, ndtr\n\n    ')
# Processing the call keyword arguments (line 5636)
kwargs_494331 = {}
# Getting the type of 'add_newdoc' (line 5636)
add_newdoc_494327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5636, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5636)
add_newdoc_call_result_494332 = invoke(stypy.reporting.localization.Localization(__file__, 5636, 0), add_newdoc_494327, *[str_494328, str_494329, str_494330], **kwargs_494331)


# Call to add_newdoc(...): (line 5662)
# Processing the call arguments (line 5662)
str_494334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5662, 11), 'str', 'scipy.special')
str_494335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5662, 28), 'str', 'log_ndtr')
str_494336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5690, (-1)), 'str', '\n    log_ndtr(x)\n\n    Logarithm of Gaussian cumulative distribution function.\n\n    Returns the log of the area under the standard Gaussian probability\n    density function, integrated from minus infinity to `x`::\n\n        log(1/sqrt(2*pi) * integral(exp(-t**2 / 2), t=-inf..x))\n\n    Parameters\n    ----------\n    x : array_like, real or complex\n        Argument\n\n    Returns\n    -------\n    ndarray\n        The value of the log of the normal CDF evaluated at `x`\n\n    See Also\n    --------\n    erf\n    erfc\n    scipy.stats.norm\n    ndtr\n\n    ')
# Processing the call keyword arguments (line 5662)
kwargs_494337 = {}
# Getting the type of 'add_newdoc' (line 5662)
add_newdoc_494333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5662, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5662)
add_newdoc_call_result_494338 = invoke(stypy.reporting.localization.Localization(__file__, 5662, 0), add_newdoc_494333, *[str_494334, str_494335, str_494336], **kwargs_494337)


# Call to add_newdoc(...): (line 5692)
# Processing the call arguments (line 5692)
str_494340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5692, 11), 'str', 'scipy.special')
str_494341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5692, 28), 'str', 'ndtri')
str_494342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5701, (-1)), 'str', '\n    ndtri(y)\n\n    Inverse of `ndtr` vs x\n\n    Returns the argument x for which the area under the Gaussian\n    probability density function (integrated from minus infinity to `x`)\n    is equal to y.\n    ')
# Processing the call keyword arguments (line 5692)
kwargs_494343 = {}
# Getting the type of 'add_newdoc' (line 5692)
add_newdoc_494339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5692, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5692)
add_newdoc_call_result_494344 = invoke(stypy.reporting.localization.Localization(__file__, 5692, 0), add_newdoc_494339, *[str_494340, str_494341, str_494342], **kwargs_494343)


# Call to add_newdoc(...): (line 5703)
# Processing the call arguments (line 5703)
str_494346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5703, 11), 'str', 'scipy.special')
str_494347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5703, 28), 'str', 'obl_ang1')
str_494348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5719, (-1)), 'str', '\n    obl_ang1(m, n, c, x)\n\n    Oblate spheroidal angular function of the first kind and its derivative\n\n    Computes the oblate spheroidal angular function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5703)
kwargs_494349 = {}
# Getting the type of 'add_newdoc' (line 5703)
add_newdoc_494345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5703, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5703)
add_newdoc_call_result_494350 = invoke(stypy.reporting.localization.Localization(__file__, 5703, 0), add_newdoc_494345, *[str_494346, str_494347, str_494348], **kwargs_494349)


# Call to add_newdoc(...): (line 5721)
# Processing the call arguments (line 5721)
str_494352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5721, 11), 'str', 'scipy.special')
str_494353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5721, 28), 'str', 'obl_ang1_cv')
str_494354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5738, (-1)), 'str', '\n    obl_ang1_cv(m, n, c, cv, x)\n\n    Oblate spheroidal angular function obl_ang1 for precomputed characteristic value\n\n    Computes the oblate spheroidal angular function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5721)
kwargs_494355 = {}
# Getting the type of 'add_newdoc' (line 5721)
add_newdoc_494351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5721, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5721)
add_newdoc_call_result_494356 = invoke(stypy.reporting.localization.Localization(__file__, 5721, 0), add_newdoc_494351, *[str_494352, str_494353, str_494354], **kwargs_494355)


# Call to add_newdoc(...): (line 5740)
# Processing the call arguments (line 5740)
str_494358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5740, 11), 'str', 'scipy.special')
str_494359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5740, 28), 'str', 'obl_cv')
str_494360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5748, (-1)), 'str', '\n    obl_cv(m, n, c)\n\n    Characteristic value of oblate spheroidal function\n\n    Computes the characteristic value of oblate spheroidal wave\n    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.\n    ')
# Processing the call keyword arguments (line 5740)
kwargs_494361 = {}
# Getting the type of 'add_newdoc' (line 5740)
add_newdoc_494357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5740, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5740)
add_newdoc_call_result_494362 = invoke(stypy.reporting.localization.Localization(__file__, 5740, 0), add_newdoc_494357, *[str_494358, str_494359, str_494360], **kwargs_494361)


# Call to add_newdoc(...): (line 5750)
# Processing the call arguments (line 5750)
str_494364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5750, 11), 'str', 'scipy.special')
str_494365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5750, 28), 'str', 'obl_rad1')
str_494366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5766, (-1)), 'str', '\n    obl_rad1(m, n, c, x)\n\n    Oblate spheroidal radial function of the first kind and its derivative\n\n    Computes the oblate spheroidal radial function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5750)
kwargs_494367 = {}
# Getting the type of 'add_newdoc' (line 5750)
add_newdoc_494363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5750, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5750)
add_newdoc_call_result_494368 = invoke(stypy.reporting.localization.Localization(__file__, 5750, 0), add_newdoc_494363, *[str_494364, str_494365, str_494366], **kwargs_494367)


# Call to add_newdoc(...): (line 5768)
# Processing the call arguments (line 5768)
str_494370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5768, 11), 'str', 'scipy.special')
str_494371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5768, 28), 'str', 'obl_rad1_cv')
str_494372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5785, (-1)), 'str', '\n    obl_rad1_cv(m, n, c, cv, x)\n\n    Oblate spheroidal radial function obl_rad1 for precomputed characteristic value\n\n    Computes the oblate spheroidal radial function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5768)
kwargs_494373 = {}
# Getting the type of 'add_newdoc' (line 5768)
add_newdoc_494369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5768, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5768)
add_newdoc_call_result_494374 = invoke(stypy.reporting.localization.Localization(__file__, 5768, 0), add_newdoc_494369, *[str_494370, str_494371, str_494372], **kwargs_494373)


# Call to add_newdoc(...): (line 5787)
# Processing the call arguments (line 5787)
str_494376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5787, 11), 'str', 'scipy.special')
str_494377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5787, 28), 'str', 'obl_rad2')
str_494378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5803, (-1)), 'str', '\n    obl_rad2(m, n, c, x)\n\n    Oblate spheroidal radial function of the second kind and its derivative.\n\n    Computes the oblate spheroidal radial function of the second kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5787)
kwargs_494379 = {}
# Getting the type of 'add_newdoc' (line 5787)
add_newdoc_494375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5787, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5787)
add_newdoc_call_result_494380 = invoke(stypy.reporting.localization.Localization(__file__, 5787, 0), add_newdoc_494375, *[str_494376, str_494377, str_494378], **kwargs_494379)


# Call to add_newdoc(...): (line 5805)
# Processing the call arguments (line 5805)
str_494382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5805, 11), 'str', 'scipy.special')
str_494383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5805, 28), 'str', 'obl_rad2_cv')
str_494384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5822, (-1)), 'str', '\n    obl_rad2_cv(m, n, c, cv, x)\n\n    Oblate spheroidal radial function obl_rad2 for precomputed characteristic value\n\n    Computes the oblate spheroidal radial function of the second kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5805)
kwargs_494385 = {}
# Getting the type of 'add_newdoc' (line 5805)
add_newdoc_494381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5805, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5805)
add_newdoc_call_result_494386 = invoke(stypy.reporting.localization.Localization(__file__, 5805, 0), add_newdoc_494381, *[str_494382, str_494383, str_494384], **kwargs_494385)


# Call to add_newdoc(...): (line 5824)
# Processing the call arguments (line 5824)
str_494388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5824, 11), 'str', 'scipy.special')
str_494389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5824, 28), 'str', 'pbdv')
str_494390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5839, (-1)), 'str', "\n    pbdv(v, x)\n\n    Parabolic cylinder function D\n\n    Returns (d, dp) the parabolic cylinder function Dv(x) in d and the\n    derivative, Dv'(x) in dp.\n\n    Returns\n    -------\n    d\n        Value of the function\n    dp\n        Value of the derivative vs x\n    ")
# Processing the call keyword arguments (line 5824)
kwargs_494391 = {}
# Getting the type of 'add_newdoc' (line 5824)
add_newdoc_494387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5824, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5824)
add_newdoc_call_result_494392 = invoke(stypy.reporting.localization.Localization(__file__, 5824, 0), add_newdoc_494387, *[str_494388, str_494389, str_494390], **kwargs_494391)


# Call to add_newdoc(...): (line 5841)
# Processing the call arguments (line 5841)
str_494394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5841, 11), 'str', 'scipy.special')
str_494395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5841, 28), 'str', 'pbvv')
str_494396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5856, (-1)), 'str', "\n    pbvv(v, x)\n\n    Parabolic cylinder function V\n\n    Returns the parabolic cylinder function Vv(x) in v and the\n    derivative, Vv'(x) in vp.\n\n    Returns\n    -------\n    v\n        Value of the function\n    vp\n        Value of the derivative vs x\n    ")
# Processing the call keyword arguments (line 5841)
kwargs_494397 = {}
# Getting the type of 'add_newdoc' (line 5841)
add_newdoc_494393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5841, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5841)
add_newdoc_call_result_494398 = invoke(stypy.reporting.localization.Localization(__file__, 5841, 0), add_newdoc_494393, *[str_494394, str_494395, str_494396], **kwargs_494397)


# Call to add_newdoc(...): (line 5858)
# Processing the call arguments (line 5858)
str_494400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5858, 11), 'str', 'scipy.special')
str_494401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5858, 28), 'str', 'pbwa')
str_494402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5899, (-1)), 'str', '\n    pbwa(a, x)\n\n    Parabolic cylinder function W.\n\n    The function is a particular solution to the differential equation\n\n    .. math::\n\n        y\'\' + \\left(\\frac{1}{4}x^2 - a\\right)y = 0,\n\n    for a full definition see section 12.14 in [1]_.\n\n    Parameters\n    ----------\n    a : array_like\n        Real parameter\n    x : array_like\n        Real argument\n\n    Returns\n    -------\n    w : scalar or ndarray\n        Value of the function\n    wp : scalar or ndarray\n        Value of the derivative in x\n\n    Notes\n    -----\n    The function is a wrapper for a Fortran routine by Zhang and Jin\n    [2]_. The implementation is accurate only for ``|a|, |x| < 5`` and\n    returns NaN outside that range.\n\n    References\n    ----------\n    .. [1] Digital Library of Mathematical Functions, 14.30.\n           http://dlmf.nist.gov/14.30\n    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special\n           Functions", John Wiley and Sons, 1996.\n           https://people.sc.fsu.edu/~jburkardt/f_src/special_functions/special_functions.html\n    ')
# Processing the call keyword arguments (line 5858)
kwargs_494403 = {}
# Getting the type of 'add_newdoc' (line 5858)
add_newdoc_494399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5858, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5858)
add_newdoc_call_result_494404 = invoke(stypy.reporting.localization.Localization(__file__, 5858, 0), add_newdoc_494399, *[str_494400, str_494401, str_494402], **kwargs_494403)


# Call to add_newdoc(...): (line 5901)
# Processing the call arguments (line 5901)
str_494406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5901, 11), 'str', 'scipy.special')
str_494407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5901, 28), 'str', 'pdtr')
str_494408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5910, (-1)), 'str', '\n    pdtr(k, m)\n\n    Poisson cumulative distribution function\n\n    Returns the sum of the first `k` terms of the Poisson distribution:\n    sum(exp(-m) * m**j / j!, j=0..k) = gammaincc( k+1, m).  Arguments\n    must both be positive and `k` an integer.\n    ')
# Processing the call keyword arguments (line 5901)
kwargs_494409 = {}
# Getting the type of 'add_newdoc' (line 5901)
add_newdoc_494405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5901, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5901)
add_newdoc_call_result_494410 = invoke(stypy.reporting.localization.Localization(__file__, 5901, 0), add_newdoc_494405, *[str_494406, str_494407, str_494408], **kwargs_494409)


# Call to add_newdoc(...): (line 5912)
# Processing the call arguments (line 5912)
str_494412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5912, 11), 'str', 'scipy.special')
str_494413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5912, 28), 'str', 'pdtrc')
str_494414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5921, (-1)), 'str', '\n    pdtrc(k, m)\n\n    Poisson survival function\n\n    Returns the sum of the terms from k+1 to infinity of the Poisson\n    distribution: sum(exp(-m) * m**j / j!, j=k+1..inf) = gammainc(\n    k+1, m).  Arguments must both be positive and `k` an integer.\n    ')
# Processing the call keyword arguments (line 5912)
kwargs_494415 = {}
# Getting the type of 'add_newdoc' (line 5912)
add_newdoc_494411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5912, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5912)
add_newdoc_call_result_494416 = invoke(stypy.reporting.localization.Localization(__file__, 5912, 0), add_newdoc_494411, *[str_494412, str_494413, str_494414], **kwargs_494415)


# Call to add_newdoc(...): (line 5923)
# Processing the call arguments (line 5923)
str_494418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5923, 11), 'str', 'scipy.special')
str_494419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5923, 28), 'str', 'pdtri')
str_494420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5933, (-1)), 'str', '\n    pdtri(k, y)\n\n    Inverse to `pdtr` vs m\n\n    Returns the Poisson variable `m` such that the sum from 0 to `k` of\n    the Poisson density is equal to the given probability `y`:\n    calculated by gammaincinv(k+1, y). `k` must be a nonnegative\n    integer and `y` between 0 and 1.\n    ')
# Processing the call keyword arguments (line 5923)
kwargs_494421 = {}
# Getting the type of 'add_newdoc' (line 5923)
add_newdoc_494417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5923, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5923)
add_newdoc_call_result_494422 = invoke(stypy.reporting.localization.Localization(__file__, 5923, 0), add_newdoc_494417, *[str_494418, str_494419, str_494420], **kwargs_494421)


# Call to add_newdoc(...): (line 5935)
# Processing the call arguments (line 5935)
str_494424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5935, 11), 'str', 'scipy.special')
str_494425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5935, 28), 'str', 'pdtrik')
str_494426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5942, (-1)), 'str', '\n    pdtrik(p, m)\n\n    Inverse to `pdtr` vs k\n\n    Returns the quantile k such that ``pdtr(k, m) = p``\n    ')
# Processing the call keyword arguments (line 5935)
kwargs_494427 = {}
# Getting the type of 'add_newdoc' (line 5935)
add_newdoc_494423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5935, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5935)
add_newdoc_call_result_494428 = invoke(stypy.reporting.localization.Localization(__file__, 5935, 0), add_newdoc_494423, *[str_494424, str_494425, str_494426], **kwargs_494427)


# Call to add_newdoc(...): (line 5944)
# Processing the call arguments (line 5944)
str_494430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5944, 11), 'str', 'scipy.special')
str_494431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5944, 28), 'str', 'poch')
str_494432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5973, (-1)), 'str', '\n    poch(z, m)\n\n    Rising factorial (z)_m\n\n    The Pochhammer symbol (rising factorial), is defined as\n\n    .. math::\n\n        (z)_m = \\frac{\\Gamma(z + m)}{\\Gamma(z)}\n\n    For positive integer `m` it reads\n\n    .. math::\n\n        (z)_m = z (z + 1) ... (z + m - 1)\n\n    Parameters\n    ----------\n    z : array_like\n        (int or float)\n    m : array_like\n        (int or float)\n\n    Returns\n    -------\n    poch : ndarray\n        The value of the function.\n    ')
# Processing the call keyword arguments (line 5944)
kwargs_494433 = {}
# Getting the type of 'add_newdoc' (line 5944)
add_newdoc_494429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5944, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5944)
add_newdoc_call_result_494434 = invoke(stypy.reporting.localization.Localization(__file__, 5944, 0), add_newdoc_494429, *[str_494430, str_494431, str_494432], **kwargs_494433)


# Call to add_newdoc(...): (line 5975)
# Processing the call arguments (line 5975)
str_494436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5975, 11), 'str', 'scipy.special')
str_494437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5975, 28), 'str', 'pro_ang1')
str_494438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5991, (-1)), 'str', '\n    pro_ang1(m, n, c, x)\n\n    Prolate spheroidal angular function of the first kind and its derivative\n\n    Computes the prolate spheroidal angular function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5975)
kwargs_494439 = {}
# Getting the type of 'add_newdoc' (line 5975)
add_newdoc_494435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5975, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5975)
add_newdoc_call_result_494440 = invoke(stypy.reporting.localization.Localization(__file__, 5975, 0), add_newdoc_494435, *[str_494436, str_494437, str_494438], **kwargs_494439)


# Call to add_newdoc(...): (line 5993)
# Processing the call arguments (line 5993)
str_494442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5993, 11), 'str', 'scipy.special')
str_494443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5993, 28), 'str', 'pro_ang1_cv')
str_494444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6010, (-1)), 'str', '\n    pro_ang1_cv(m, n, c, cv, x)\n\n    Prolate spheroidal angular function pro_ang1 for precomputed characteristic value\n\n    Computes the prolate spheroidal angular function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 5993)
kwargs_494445 = {}
# Getting the type of 'add_newdoc' (line 5993)
add_newdoc_494441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5993, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 5993)
add_newdoc_call_result_494446 = invoke(stypy.reporting.localization.Localization(__file__, 5993, 0), add_newdoc_494441, *[str_494442, str_494443, str_494444], **kwargs_494445)


# Call to add_newdoc(...): (line 6012)
# Processing the call arguments (line 6012)
str_494448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6012, 11), 'str', 'scipy.special')
str_494449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6012, 28), 'str', 'pro_cv')
str_494450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6020, (-1)), 'str', '\n    pro_cv(m, n, c)\n\n    Characteristic value of prolate spheroidal function\n\n    Computes the characteristic value of prolate spheroidal wave\n    functions of order `m`, `n` (n>=m) and spheroidal parameter `c`.\n    ')
# Processing the call keyword arguments (line 6012)
kwargs_494451 = {}
# Getting the type of 'add_newdoc' (line 6012)
add_newdoc_494447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6012, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6012)
add_newdoc_call_result_494452 = invoke(stypy.reporting.localization.Localization(__file__, 6012, 0), add_newdoc_494447, *[str_494448, str_494449, str_494450], **kwargs_494451)


# Call to add_newdoc(...): (line 6022)
# Processing the call arguments (line 6022)
str_494454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6022, 11), 'str', 'scipy.special')
str_494455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6022, 28), 'str', 'pro_rad1')
str_494456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6038, (-1)), 'str', '\n    pro_rad1(m, n, c, x)\n\n    Prolate spheroidal radial function of the first kind and its derivative\n\n    Computes the prolate spheroidal radial function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 6022)
kwargs_494457 = {}
# Getting the type of 'add_newdoc' (line 6022)
add_newdoc_494453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6022, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6022)
add_newdoc_call_result_494458 = invoke(stypy.reporting.localization.Localization(__file__, 6022, 0), add_newdoc_494453, *[str_494454, str_494455, str_494456], **kwargs_494457)


# Call to add_newdoc(...): (line 6040)
# Processing the call arguments (line 6040)
str_494460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6040, 11), 'str', 'scipy.special')
str_494461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6040, 28), 'str', 'pro_rad1_cv')
str_494462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6057, (-1)), 'str', '\n    pro_rad1_cv(m, n, c, cv, x)\n\n    Prolate spheroidal radial function pro_rad1 for precomputed characteristic value\n\n    Computes the prolate spheroidal radial function of the first kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 6040)
kwargs_494463 = {}
# Getting the type of 'add_newdoc' (line 6040)
add_newdoc_494459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6040, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6040)
add_newdoc_call_result_494464 = invoke(stypy.reporting.localization.Localization(__file__, 6040, 0), add_newdoc_494459, *[str_494460, str_494461, str_494462], **kwargs_494463)


# Call to add_newdoc(...): (line 6059)
# Processing the call arguments (line 6059)
str_494466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6059, 11), 'str', 'scipy.special')
str_494467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6059, 28), 'str', 'pro_rad2')
str_494468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6075, (-1)), 'str', '\n    pro_rad2(m, n, c, x)\n\n    Prolate spheroidal radial function of the second kind and its derivative\n\n    Computes the prolate spheroidal radial function of the second kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 6059)
kwargs_494469 = {}
# Getting the type of 'add_newdoc' (line 6059)
add_newdoc_494465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6059, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6059)
add_newdoc_call_result_494470 = invoke(stypy.reporting.localization.Localization(__file__, 6059, 0), add_newdoc_494465, *[str_494466, str_494467, str_494468], **kwargs_494469)


# Call to add_newdoc(...): (line 6077)
# Processing the call arguments (line 6077)
str_494472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6077, 11), 'str', 'scipy.special')
str_494473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6077, 28), 'str', 'pro_rad2_cv')
str_494474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6094, (-1)), 'str', '\n    pro_rad2_cv(m, n, c, cv, x)\n\n    Prolate spheroidal radial function pro_rad2 for precomputed characteristic value\n\n    Computes the prolate spheroidal radial function of the second kind\n    and its derivative (with respect to `x`) for mode parameters m>=0\n    and n>=m, spheroidal parameter `c` and ``|x| < 1.0``. Requires\n    pre-computed characteristic value.\n\n    Returns\n    -------\n    s\n        Value of the function\n    sp\n        Value of the derivative vs x\n    ')
# Processing the call keyword arguments (line 6077)
kwargs_494475 = {}
# Getting the type of 'add_newdoc' (line 6077)
add_newdoc_494471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6077, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6077)
add_newdoc_call_result_494476 = invoke(stypy.reporting.localization.Localization(__file__, 6077, 0), add_newdoc_494471, *[str_494472, str_494473, str_494474], **kwargs_494475)


# Call to add_newdoc(...): (line 6096)
# Processing the call arguments (line 6096)
str_494478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6096, 11), 'str', 'scipy.special')
str_494479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6096, 28), 'str', 'pseudo_huber')
str_494480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6122, (-1)), 'str', '\n    pseudo_huber(delta, r)\n\n    Pseudo-Huber loss function.\n\n    .. math:: \\mathrm{pseudo\\_huber}(\\delta, r) = \\delta^2 \\left( \\sqrt{ 1 + \\left( \\frac{r}{\\delta} \\right)^2 } - 1 \\right)\n\n    Parameters\n    ----------\n    delta : ndarray\n        Input array, indicating the soft quadratic vs. linear loss changepoint.\n    r : ndarray\n        Input array, possibly representing residuals.\n\n    Returns\n    -------\n    res : ndarray\n        The computed Pseudo-Huber loss function values.\n\n    Notes\n    -----\n    This function is convex in :math:`r`.\n\n    .. versionadded:: 0.15.0\n\n    ')
# Processing the call keyword arguments (line 6096)
kwargs_494481 = {}
# Getting the type of 'add_newdoc' (line 6096)
add_newdoc_494477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6096, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6096)
add_newdoc_call_result_494482 = invoke(stypy.reporting.localization.Localization(__file__, 6096, 0), add_newdoc_494477, *[str_494478, str_494479, str_494480], **kwargs_494481)


# Call to add_newdoc(...): (line 6124)
# Processing the call arguments (line 6124)
str_494484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6124, 11), 'str', 'scipy.special')
str_494485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6124, 28), 'str', 'psi')
str_494486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6168, (-1)), 'str', '\n    psi(z, out=None)\n\n    The digamma function.\n\n    The logarithmic derivative of the gamma function evaluated at ``z``.\n\n    Parameters\n    ----------\n    z : array_like\n        Real or complex argument.\n    out : ndarray, optional\n        Array for the computed values of ``psi``.\n\n    Returns\n    -------\n    digamma : ndarray\n        Computed values of ``psi``.\n\n    Notes\n    -----\n    For large values not close to the negative real axis ``psi`` is\n    computed using the asymptotic series (5.11.2) from [1]_. For small\n    arguments not close to the negative real axis the recurrence\n    relation (5.5.2) from [1]_ is used until the argument is large\n    enough to use the asymptotic series. For values close to the\n    negative real axis the reflection formula (5.5.4) from [1]_ is\n    used first.  Note that ``psi`` has a family of zeros on the\n    negative real axis which occur between the poles at nonpositive\n    integers. Around the zeros the reflection formula suffers from\n    cancellation and the implementation loses precision. The sole\n    positive zero and the first negative zero, however, are handled\n    separately by precomputing series expansions using [2]_, so the\n    function should maintain full accuracy around the origin.\n\n    References\n    ----------\n    .. [1] NIST Digital Library of Mathematical Functions\n           http://dlmf.nist.gov/5\n    .. [2] Fredrik Johansson and others.\n           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"\n           (Version 0.19) http://mpmath.org/\n\n    ')
# Processing the call keyword arguments (line 6124)
kwargs_494487 = {}
# Getting the type of 'add_newdoc' (line 6124)
add_newdoc_494483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6124, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6124)
add_newdoc_call_result_494488 = invoke(stypy.reporting.localization.Localization(__file__, 6124, 0), add_newdoc_494483, *[str_494484, str_494485, str_494486], **kwargs_494487)


# Call to add_newdoc(...): (line 6170)
# Processing the call arguments (line 6170)
str_494490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6170, 11), 'str', 'scipy.special')
str_494491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6170, 28), 'str', 'radian')
str_494492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6178, (-1)), 'str', '\n    radian(d, m, s)\n\n    Convert from degrees to radians\n\n    Returns the angle given in (d)egrees, (m)inutes, and (s)econds in\n    radians.\n    ')
# Processing the call keyword arguments (line 6170)
kwargs_494493 = {}
# Getting the type of 'add_newdoc' (line 6170)
add_newdoc_494489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6170, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6170)
add_newdoc_call_result_494494 = invoke(stypy.reporting.localization.Localization(__file__, 6170, 0), add_newdoc_494489, *[str_494490, str_494491, str_494492], **kwargs_494493)


# Call to add_newdoc(...): (line 6180)
# Processing the call arguments (line 6180)
str_494496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6180, 11), 'str', 'scipy.special')
str_494497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6180, 28), 'str', 'rel_entr')
str_494498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6210, (-1)), 'str', '\n    rel_entr(x, y)\n\n    Elementwise function for computing relative entropy.\n\n    .. math:: \\mathrm{rel\\_entr}(x, y) = \\begin{cases} x \\log(x / y) & x > 0, y > 0 \\\\ 0 & x = 0, y \\ge 0 \\\\ \\infty & \\text{otherwise} \\end{cases}\n\n    Parameters\n    ----------\n    x : ndarray\n        First input array.\n    y : ndarray\n        Second input array.\n\n    Returns\n    -------\n    res : ndarray\n        Output array.\n\n    See Also\n    --------\n    entr, kl_div\n\n    Notes\n    -----\n    This function is jointly convex in x and y.\n\n    .. versionadded:: 0.15.0\n\n    ')
# Processing the call keyword arguments (line 6180)
kwargs_494499 = {}
# Getting the type of 'add_newdoc' (line 6180)
add_newdoc_494495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6180, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6180)
add_newdoc_call_result_494500 = invoke(stypy.reporting.localization.Localization(__file__, 6180, 0), add_newdoc_494495, *[str_494496, str_494497, str_494498], **kwargs_494499)


# Call to add_newdoc(...): (line 6212)
# Processing the call arguments (line 6212)
str_494502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6212, 11), 'str', 'scipy.special')
str_494503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6212, 28), 'str', 'rgamma')
str_494504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6219, (-1)), 'str', '\n    rgamma(z)\n\n    Gamma function inverted\n\n    Returns ``1/gamma(x)``\n    ')
# Processing the call keyword arguments (line 6212)
kwargs_494505 = {}
# Getting the type of 'add_newdoc' (line 6212)
add_newdoc_494501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6212, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6212)
add_newdoc_call_result_494506 = invoke(stypy.reporting.localization.Localization(__file__, 6212, 0), add_newdoc_494501, *[str_494502, str_494503, str_494504], **kwargs_494505)


# Call to add_newdoc(...): (line 6221)
# Processing the call arguments (line 6221)
str_494508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6221, 11), 'str', 'scipy.special')
str_494509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6221, 28), 'str', 'round')
str_494510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6230, (-1)), 'str', '\n    round(x)\n\n    Round to nearest integer\n\n    Returns the nearest integer to `x` as a double precision floating\n    point result.  If `x` ends in 0.5 exactly, the nearest even integer\n    is chosen.\n    ')
# Processing the call keyword arguments (line 6221)
kwargs_494511 = {}
# Getting the type of 'add_newdoc' (line 6221)
add_newdoc_494507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6221, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6221)
add_newdoc_call_result_494512 = invoke(stypy.reporting.localization.Localization(__file__, 6221, 0), add_newdoc_494507, *[str_494508, str_494509, str_494510], **kwargs_494511)


# Call to add_newdoc(...): (line 6232)
# Processing the call arguments (line 6232)
str_494514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6232, 11), 'str', 'scipy.special')
str_494515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6232, 28), 'str', 'shichi')
str_494516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6283, (-1)), 'str', '\n    shichi(x, out=None)\n\n    Hyperbolic sine and cosine integrals.\n\n    The hyperbolic sine integral is\n\n    .. math::\n\n      \\int_0^x \\frac{\\sinh{t}}{t}dt\n\n    and the hyperbolic cosine integral is\n\n    .. math::\n\n      \\gamma + \\log(x) + \\int_0^x \\frac{\\cosh{t} - 1}{t} dt\n\n    where :math:`\\gamma` is Euler\'s constant and :math:`\\log` is the\n    principle branch of the logarithm.\n\n    Parameters\n    ----------\n    x : array_like\n        Real or complex points at which to compute the hyperbolic sine\n        and cosine integrals.\n\n    Returns\n    -------\n    si : ndarray\n        Hyperbolic sine integral at ``x``\n    ci : ndarray\n        Hyperbolic cosine integral at ``x``\n\n    Notes\n    -----\n    For real arguments with ``x < 0``, ``chi`` is the real part of the\n    hyperbolic cosine integral. For such points ``chi(x)`` and ``chi(x\n    + 0j)`` differ by a factor of ``1j*pi``.\n\n    For real arguments the function is computed by calling Cephes\'\n    [1]_ *shichi* routine. For complex arguments the algorithm is based\n    on Mpmath\'s [2]_ *shi* and *chi* routines.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Fredrik Johansson and others.\n           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"\n           (Version 0.19) http://mpmath.org/\n    ')
# Processing the call keyword arguments (line 6232)
kwargs_494517 = {}
# Getting the type of 'add_newdoc' (line 6232)
add_newdoc_494513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6232, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6232)
add_newdoc_call_result_494518 = invoke(stypy.reporting.localization.Localization(__file__, 6232, 0), add_newdoc_494513, *[str_494514, str_494515, str_494516], **kwargs_494517)


# Call to add_newdoc(...): (line 6285)
# Processing the call arguments (line 6285)
str_494520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6285, 11), 'str', 'scipy.special')
str_494521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6285, 28), 'str', 'sici')
str_494522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6336, (-1)), 'str', '\n    sici(x, out=None)\n\n    Sine and cosine integrals.\n\n    The sine integral is\n\n    .. math::\n\n      \\int_0^x \\frac{\\sin{t}}{t}dt\n\n    and the cosine integral is\n\n    .. math::\n\n      \\gamma + \\log(x) + \\int_0^x \\frac{\\cos{t} - 1}{t}dt\n\n    where :math:`\\gamma` is Euler\'s constant and :math:`\\log` is the\n    principle branch of the logarithm.\n\n    Parameters\n    ----------\n    x : array_like\n        Real or complex points at which to compute the sine and cosine\n        integrals.\n\n    Returns\n    -------\n    si : ndarray\n        Sine integral at ``x``\n    ci : ndarray\n        Cosine integral at ``x``\n\n    Notes\n    -----\n    For real arguments with ``x < 0``, ``ci`` is the real part of the\n    cosine integral. For such points ``ci(x)`` and ``ci(x + 0j)``\n    differ by a factor of ``1j*pi``.\n\n    For real arguments the function is computed by calling Cephes\'\n    [1]_ *sici* routine. For complex arguments the algorithm is based\n    on Mpmath\'s [2]_ *si* and *ci* routines.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    .. [2] Fredrik Johansson and others.\n           "mpmath: a Python library for arbitrary-precision floating-point arithmetic"\n           (Version 0.19) http://mpmath.org/\n    ')
# Processing the call keyword arguments (line 6285)
kwargs_494523 = {}
# Getting the type of 'add_newdoc' (line 6285)
add_newdoc_494519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6285, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6285)
add_newdoc_call_result_494524 = invoke(stypy.reporting.localization.Localization(__file__, 6285, 0), add_newdoc_494519, *[str_494520, str_494521, str_494522], **kwargs_494523)


# Call to add_newdoc(...): (line 6338)
# Processing the call arguments (line 6338)
str_494526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6338, 11), 'str', 'scipy.special')
str_494527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6338, 28), 'str', 'sindg')
str_494528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6343, (-1)), 'str', '\n    sindg(x)\n\n    Sine of angle given in degrees\n    ')
# Processing the call keyword arguments (line 6338)
kwargs_494529 = {}
# Getting the type of 'add_newdoc' (line 6338)
add_newdoc_494525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6338, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6338)
add_newdoc_call_result_494530 = invoke(stypy.reporting.localization.Localization(__file__, 6338, 0), add_newdoc_494525, *[str_494526, str_494527, str_494528], **kwargs_494529)


# Call to add_newdoc(...): (line 6345)
# Processing the call arguments (line 6345)
str_494532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6345, 11), 'str', 'scipy.special')
str_494533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6345, 28), 'str', 'smirnov')
str_494534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6357, (-1)), 'str', '\n    smirnov(n, e)\n\n    Kolmogorov-Smirnov complementary cumulative distribution function\n\n    Returns the exact Kolmogorov-Smirnov complementary cumulative\n    distribution function (Dn+ or Dn-) for a one-sided test of\n    equality between an empirical and a theoretical distribution. It\n    is equal to the probability that the maximum difference between a\n    theoretical distribution and an empirical one based on `n` samples\n    is greater than e.\n    ')
# Processing the call keyword arguments (line 6345)
kwargs_494535 = {}
# Getting the type of 'add_newdoc' (line 6345)
add_newdoc_494531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6345, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6345)
add_newdoc_call_result_494536 = invoke(stypy.reporting.localization.Localization(__file__, 6345, 0), add_newdoc_494531, *[str_494532, str_494533, str_494534], **kwargs_494535)


# Call to add_newdoc(...): (line 6359)
# Processing the call arguments (line 6359)
str_494538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6359, 11), 'str', 'scipy.special')
str_494539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6359, 28), 'str', 'smirnovi')
str_494540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6366, (-1)), 'str', '\n    smirnovi(n, y)\n\n    Inverse to `smirnov`\n\n    Returns ``e`` such that ``smirnov(n, e) = y``.\n    ')
# Processing the call keyword arguments (line 6359)
kwargs_494541 = {}
# Getting the type of 'add_newdoc' (line 6359)
add_newdoc_494537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6359, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6359)
add_newdoc_call_result_494542 = invoke(stypy.reporting.localization.Localization(__file__, 6359, 0), add_newdoc_494537, *[str_494538, str_494539, str_494540], **kwargs_494541)


# Call to add_newdoc(...): (line 6368)
# Processing the call arguments (line 6368)
str_494544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6368, 11), 'str', 'scipy.special')
str_494545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6368, 28), 'str', 'spence')
str_494546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6403, (-1)), 'str', "\n    spence(z, out=None)\n\n    Spence's function, also known as the dilogarithm.\n\n    It is defined to be\n\n    .. math::\n      \\int_0^z \\frac{\\log(t)}{1 - t}dt\n\n    for complex :math:`z`, where the contour of integration is taken\n    to avoid the branch cut of the logarithm. Spence's function is\n    analytic everywhere except the negative real axis where it has a\n    branch cut.\n\n    Parameters\n    ----------\n    z : array_like\n        Points at which to evaluate Spence's function\n\n    Returns\n    -------\n    s : ndarray\n        Computed values of Spence's function\n\n    Notes\n    -----\n    There is a different convention which defines Spence's function by\n    the integral\n\n    .. math::\n      -\\int_0^z \\frac{\\log(1 - t)}{t}dt;\n\n    this is our ``spence(1 - z)``.\n    ")
# Processing the call keyword arguments (line 6368)
kwargs_494547 = {}
# Getting the type of 'add_newdoc' (line 6368)
add_newdoc_494543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6368, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6368)
add_newdoc_call_result_494548 = invoke(stypy.reporting.localization.Localization(__file__, 6368, 0), add_newdoc_494543, *[str_494544, str_494545, str_494546], **kwargs_494547)


# Call to add_newdoc(...): (line 6405)
# Processing the call arguments (line 6405)
str_494550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6405, 11), 'str', 'scipy.special')
str_494551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6405, 28), 'str', 'stdtr')
str_494552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6417, (-1)), 'str', '\n    stdtr(df, t)\n\n    Student t distribution cumulative density function\n\n    Returns the integral from minus infinity to t of the Student t\n    distribution with df > 0 degrees of freedom::\n\n       gamma((df+1)/2)/(sqrt(df*pi)*gamma(df/2)) *\n       integral((1+x**2/df)**(-df/2-1/2), x=-inf..t)\n\n    ')
# Processing the call keyword arguments (line 6405)
kwargs_494553 = {}
# Getting the type of 'add_newdoc' (line 6405)
add_newdoc_494549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6405, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6405)
add_newdoc_call_result_494554 = invoke(stypy.reporting.localization.Localization(__file__, 6405, 0), add_newdoc_494549, *[str_494550, str_494551, str_494552], **kwargs_494553)


# Call to add_newdoc(...): (line 6419)
# Processing the call arguments (line 6419)
str_494556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6419, 11), 'str', 'scipy.special')
str_494557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6419, 28), 'str', 'stdtridf')
str_494558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6426, (-1)), 'str', '\n    stdtridf(p, t)\n\n    Inverse of `stdtr` vs df\n\n    Returns the argument df such that stdtr(df, t) is equal to `p`.\n    ')
# Processing the call keyword arguments (line 6419)
kwargs_494559 = {}
# Getting the type of 'add_newdoc' (line 6419)
add_newdoc_494555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6419, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6419)
add_newdoc_call_result_494560 = invoke(stypy.reporting.localization.Localization(__file__, 6419, 0), add_newdoc_494555, *[str_494556, str_494557, str_494558], **kwargs_494559)


# Call to add_newdoc(...): (line 6428)
# Processing the call arguments (line 6428)
str_494562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6428, 11), 'str', 'scipy.special')
str_494563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6428, 28), 'str', 'stdtrit')
str_494564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6435, (-1)), 'str', '\n    stdtrit(df, p)\n\n    Inverse of `stdtr` vs `t`\n\n    Returns the argument `t` such that stdtr(df, t) is equal to `p`.\n    ')
# Processing the call keyword arguments (line 6428)
kwargs_494565 = {}
# Getting the type of 'add_newdoc' (line 6428)
add_newdoc_494561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6428, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6428)
add_newdoc_call_result_494566 = invoke(stypy.reporting.localization.Localization(__file__, 6428, 0), add_newdoc_494561, *[str_494562, str_494563, str_494564], **kwargs_494565)


# Call to add_newdoc(...): (line 6437)
# Processing the call arguments (line 6437)
str_494568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6437, 11), 'str', 'scipy.special')
str_494569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6437, 28), 'str', 'struve')
str_494570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6484, (-1)), 'str', '\n    struve(v, x)\n\n    Struve function.\n\n    Return the value of the Struve function of order `v` at `x`.  The Struve\n    function is defined as,\n\n    .. math::\n        H_v(x) = (z/2)^{v + 1} \\sum_{n=0}^\\infty \\frac{(-1)^n (z/2)^{2n}}{\\Gamma(n + \\frac{3}{2}) \\Gamma(n + v + \\frac{3}{2})},\n\n    where :math:`\\Gamma` is the gamma function.\n\n    Parameters\n    ----------\n    v : array_like\n        Order of the Struve function (float).\n    x : array_like\n        Argument of the Struve function (float; must be positive unless `v` is\n        an integer).\n\n    Returns\n    -------\n    H : ndarray\n        Value of the Struve function of order `v` at `x`.\n\n    Notes\n    -----\n    Three methods discussed in [1]_ are used to evaluate the Struve function:\n\n    - power series\n    - expansion in Bessel functions (if :math:`|z| < |v| + 20`)\n    - asymptotic large-z expansion (if :math:`z \\geq 0.7v + 12`)\n\n    Rounding errors are estimated based on the largest terms in the sums, and\n    the result associated with the smallest error is returned.\n\n    See also\n    --------\n    modstruve\n\n    References\n    ----------\n    .. [1] NIST Digital Library of Mathematical Functions\n           http://dlmf.nist.gov/11\n\n    ')
# Processing the call keyword arguments (line 6437)
kwargs_494571 = {}
# Getting the type of 'add_newdoc' (line 6437)
add_newdoc_494567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6437, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6437)
add_newdoc_call_result_494572 = invoke(stypy.reporting.localization.Localization(__file__, 6437, 0), add_newdoc_494567, *[str_494568, str_494569, str_494570], **kwargs_494571)


# Call to add_newdoc(...): (line 6486)
# Processing the call arguments (line 6486)
str_494574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6486, 11), 'str', 'scipy.special')
str_494575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6486, 28), 'str', 'tandg')
str_494576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6491, (-1)), 'str', '\n    tandg(x)\n\n    Tangent of angle x given in degrees.\n    ')
# Processing the call keyword arguments (line 6486)
kwargs_494577 = {}
# Getting the type of 'add_newdoc' (line 6486)
add_newdoc_494573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6486, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6486)
add_newdoc_call_result_494578 = invoke(stypy.reporting.localization.Localization(__file__, 6486, 0), add_newdoc_494573, *[str_494574, str_494575, str_494576], **kwargs_494577)


# Call to add_newdoc(...): (line 6493)
# Processing the call arguments (line 6493)
str_494580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6493, 11), 'str', 'scipy.special')
str_494581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6493, 28), 'str', 'tklmbda')
str_494582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6499, (-1)), 'str', '\n    tklmbda(x, lmbda)\n\n    Tukey-Lambda cumulative distribution function\n\n    ')
# Processing the call keyword arguments (line 6493)
kwargs_494583 = {}
# Getting the type of 'add_newdoc' (line 6493)
add_newdoc_494579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6493, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6493)
add_newdoc_call_result_494584 = invoke(stypy.reporting.localization.Localization(__file__, 6493, 0), add_newdoc_494579, *[str_494580, str_494581, str_494582], **kwargs_494583)


# Call to add_newdoc(...): (line 6501)
# Processing the call arguments (line 6501)
str_494586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6501, 11), 'str', 'scipy.special')
str_494587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6501, 28), 'str', 'wofz')
str_494588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6535, (-1)), 'str', "\n    wofz(z)\n\n    Faddeeva function\n\n    Returns the value of the Faddeeva function for complex argument::\n\n        exp(-z**2) * erfc(-i*z)\n\n    See Also\n    --------\n    dawsn, erf, erfc, erfcx, erfi\n\n    References\n    ----------\n    .. [1] Steven G. Johnson, Faddeeva W function implementation.\n       http://ab-initio.mit.edu/Faddeeva\n\n    Examples\n    --------\n    >>> from scipy import special\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-3, 3)\n    >>> z = special.wofz(x)\n\n    >>> plt.plot(x, z.real, label='wofz(x).real')\n    >>> plt.plot(x, z.imag, label='wofz(x).imag')\n    >>> plt.xlabel('$x$')\n    >>> plt.legend(framealpha=1, shadow=True)\n    >>> plt.grid(alpha=0.25)\n    >>> plt.show()\n\n    ")
# Processing the call keyword arguments (line 6501)
kwargs_494589 = {}
# Getting the type of 'add_newdoc' (line 6501)
add_newdoc_494585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6501, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6501)
add_newdoc_call_result_494590 = invoke(stypy.reporting.localization.Localization(__file__, 6501, 0), add_newdoc_494585, *[str_494586, str_494587, str_494588], **kwargs_494589)


# Call to add_newdoc(...): (line 6537)
# Processing the call arguments (line 6537)
str_494592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6537, 11), 'str', 'scipy.special')
str_494593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6537, 28), 'str', 'xlogy')
str_494594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6560, (-1)), 'str', '\n    xlogy(x, y)\n\n    Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.\n\n    Parameters\n    ----------\n    x : array_like\n        Multiplier\n    y : array_like\n        Argument\n\n    Returns\n    -------\n    z : array_like\n        Computed x*log(y)\n\n    Notes\n    -----\n\n    .. versionadded:: 0.13.0\n\n    ')
# Processing the call keyword arguments (line 6537)
kwargs_494595 = {}
# Getting the type of 'add_newdoc' (line 6537)
add_newdoc_494591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6537, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6537)
add_newdoc_call_result_494596 = invoke(stypy.reporting.localization.Localization(__file__, 6537, 0), add_newdoc_494591, *[str_494592, str_494593, str_494594], **kwargs_494595)


# Call to add_newdoc(...): (line 6562)
# Processing the call arguments (line 6562)
str_494598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6562, 11), 'str', 'scipy.special')
str_494599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6562, 28), 'str', 'xlog1py')
str_494600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6585, (-1)), 'str', '\n    xlog1py(x, y)\n\n    Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.\n\n    Parameters\n    ----------\n    x : array_like\n        Multiplier\n    y : array_like\n        Argument\n\n    Returns\n    -------\n    z : array_like\n        Computed x*log1p(y)\n\n    Notes\n    -----\n\n    .. versionadded:: 0.13.0\n\n    ')
# Processing the call keyword arguments (line 6562)
kwargs_494601 = {}
# Getting the type of 'add_newdoc' (line 6562)
add_newdoc_494597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6562, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6562)
add_newdoc_call_result_494602 = invoke(stypy.reporting.localization.Localization(__file__, 6562, 0), add_newdoc_494597, *[str_494598, str_494599, str_494600], **kwargs_494601)


# Call to add_newdoc(...): (line 6587)
# Processing the call arguments (line 6587)
str_494604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6587, 11), 'str', 'scipy.special')
str_494605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6587, 28), 'str', 'y0')
str_494606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6630, (-1)), 'str', '\n    y0(x)\n\n    Bessel function of the second kind of order 0.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float).\n\n    Returns\n    -------\n    Y : ndarray\n        Value of the Bessel function of the second kind of order 0 at `x`.\n\n    Notes\n    -----\n\n    The domain is divided into the intervals [0, 5] and (5, infinity). In the\n    first interval a rational approximation :math:`R(x)` is employed to\n    compute,\n\n    .. math::\n\n        Y_0(x) = R(x) + \\frac{2 \\log(x) J_0(x)}{\\pi},\n\n    where :math:`J_0` is the Bessel function of the first kind of order 0.\n\n    In the second interval, the Hankel asymptotic expansion is employed with\n    two rational functions of degree 6/6 and 7/7.\n\n    This function is a wrapper for the Cephes [1]_ routine `y0`.\n\n    See also\n    --------\n    j0\n    yv\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 6587)
kwargs_494607 = {}
# Getting the type of 'add_newdoc' (line 6587)
add_newdoc_494603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6587, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6587)
add_newdoc_call_result_494608 = invoke(stypy.reporting.localization.Localization(__file__, 6587, 0), add_newdoc_494603, *[str_494604, str_494605, str_494606], **kwargs_494607)


# Call to add_newdoc(...): (line 6632)
# Processing the call arguments (line 6632)
str_494610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6632, 11), 'str', 'scipy.special')
str_494611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6632, 28), 'str', 'y1')
str_494612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6669, (-1)), 'str', '\n    y1(x)\n\n    Bessel function of the second kind of order 1.\n\n    Parameters\n    ----------\n    x : array_like\n        Argument (float).\n\n    Returns\n    -------\n    Y : ndarray\n        Value of the Bessel function of the second kind of order 1 at `x`.\n\n    Notes\n    -----\n\n    The domain is divided into the intervals [0, 8] and (8, infinity). In the\n    first interval a 25 term Chebyshev expansion is used, and computing\n    :math:`J_1` (the Bessel function of the first kind) is required. In the\n    second, the asymptotic trigonometric representation is employed using two\n    rational functions of degree 5/5.\n\n    This function is a wrapper for the Cephes [1]_ routine `y1`.\n\n    See also\n    --------\n    j1\n    yn\n    yv\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 6632)
kwargs_494613 = {}
# Getting the type of 'add_newdoc' (line 6632)
add_newdoc_494609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6632, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6632)
add_newdoc_call_result_494614 = invoke(stypy.reporting.localization.Localization(__file__, 6632, 0), add_newdoc_494609, *[str_494610, str_494611, str_494612], **kwargs_494613)


# Call to add_newdoc(...): (line 6671)
# Processing the call arguments (line 6671)
str_494616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6671, 11), 'str', 'scipy.special')
str_494617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6671, 28), 'str', 'yn')
str_494618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6705, (-1)), 'str', '\n    yn(n, x)\n\n    Bessel function of the second kind of integer order and real argument.\n\n    Parameters\n    ----------\n    n : array_like\n        Order (integer).\n    z : array_like\n        Argument (float).\n\n    Returns\n    -------\n    Y : ndarray\n        Value of the Bessel function, :math:`Y_n(x)`.\n\n    Notes\n    -----\n    Wrapper for the Cephes [1]_ routine `yn`.\n\n    The function is evaluated by forward recurrence on `n`, starting with\n    values computed by the Cephes routines `y0` and `y1`. If `n = 0` or 1,\n    the routine for `y0` or `y1` is called directly.\n\n    See also\n    --------\n    yv : For real order and real or complex argument.\n\n    References\n    ----------\n    .. [1] Cephes Mathematical Functions Library,\n           http://www.netlib.org/cephes/index.html\n    ')
# Processing the call keyword arguments (line 6671)
kwargs_494619 = {}
# Getting the type of 'add_newdoc' (line 6671)
add_newdoc_494615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6671, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6671)
add_newdoc_call_result_494620 = invoke(stypy.reporting.localization.Localization(__file__, 6671, 0), add_newdoc_494615, *[str_494616, str_494617, str_494618], **kwargs_494619)


# Call to add_newdoc(...): (line 6707)
# Processing the call arguments (line 6707)
str_494622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6707, 11), 'str', 'scipy.special')
str_494623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6707, 28), 'str', 'yv')
str_494624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6752, (-1)), 'str', '\n    yv(v, z)\n\n    Bessel function of the second kind of real order and complex argument.\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    Y : ndarray\n        Value of the Bessel function of the second kind, :math:`Y_v(x)`.\n\n    Notes\n    -----\n    For positive `v` values, the computation is carried out using the\n    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel\n    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,\n\n    .. math:: Y_v(z) = \\frac{1}{2\\imath} (H_v^{(1)} - H_v^{(2)}).\n\n    For negative `v` values the formula,\n\n    .. math:: Y_{-v}(z) = Y_v(z) \\cos(\\pi v) + J_v(z) \\sin(\\pi v)\n\n    is used, where :math:`J_v(z)` is the Bessel function of the first kind,\n    computed using the AMOS routine `zbesj`.  Note that the second term is\n    exactly zero for integer `v`; to improve accuracy the second term is\n    explicitly omitted for `v` values such that `v = floor(v)`.\n\n    See also\n    --------\n    yve : :math:`Y_v` with leading exponential behavior stripped off.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n\n    ')
# Processing the call keyword arguments (line 6707)
kwargs_494625 = {}
# Getting the type of 'add_newdoc' (line 6707)
add_newdoc_494621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6707, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6707)
add_newdoc_call_result_494626 = invoke(stypy.reporting.localization.Localization(__file__, 6707, 0), add_newdoc_494621, *[str_494622, str_494623, str_494624], **kwargs_494625)


# Call to add_newdoc(...): (line 6754)
# Processing the call arguments (line 6754)
str_494628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6754, 11), 'str', 'scipy.special')
str_494629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6754, 28), 'str', 'yve')
str_494630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6799, (-1)), 'str', '\n    yve(v, z)\n\n    Exponentially scaled Bessel function of the second kind of real order.\n\n    Returns the exponentially scaled Bessel function of the second\n    kind of real order `v` at complex `z`::\n\n        yve(v, z) = yv(v, z) * exp(-abs(z.imag))\n\n    Parameters\n    ----------\n    v : array_like\n        Order (float).\n    z : array_like\n        Argument (float or complex).\n\n    Returns\n    -------\n    Y : ndarray\n        Value of the exponentially scaled Bessel function.\n\n    Notes\n    -----\n    For positive `v` values, the computation is carried out using the\n    AMOS [1]_ `zbesy` routine, which exploits the connection to the Hankel\n    Bessel functions :math:`H_v^{(1)}` and :math:`H_v^{(2)}`,\n\n    .. math:: Y_v(z) = \\frac{1}{2\\imath} (H_v^{(1)} - H_v^{(2)}).\n\n    For negative `v` values the formula,\n\n    .. math:: Y_{-v}(z) = Y_v(z) \\cos(\\pi v) + J_v(z) \\sin(\\pi v)\n\n    is used, where :math:`J_v(z)` is the Bessel function of the first kind,\n    computed using the AMOS routine `zbesj`.  Note that the second term is\n    exactly zero for integer `v`; to improve accuracy the second term is\n    explicitly omitted for `v` values such that `v = floor(v)`.\n\n    References\n    ----------\n    .. [1] Donald E. Amos, "AMOS, A Portable Package for Bessel Functions\n           of a Complex Argument and Nonnegative Order",\n           http://netlib.org/amos/\n    ')
# Processing the call keyword arguments (line 6754)
kwargs_494631 = {}
# Getting the type of 'add_newdoc' (line 6754)
add_newdoc_494627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6754, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6754)
add_newdoc_call_result_494632 = invoke(stypy.reporting.localization.Localization(__file__, 6754, 0), add_newdoc_494627, *[str_494628, str_494629, str_494630], **kwargs_494631)


# Call to add_newdoc(...): (line 6801)
# Processing the call arguments (line 6801)
str_494634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6801, 11), 'str', 'scipy.special')
str_494635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6801, 28), 'str', '_zeta')
str_494636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6807, (-1)), 'str', '\n    _zeta(x, q)\n\n    Internal function, Hurwitz zeta.\n\n    ')
# Processing the call keyword arguments (line 6801)
kwargs_494637 = {}
# Getting the type of 'add_newdoc' (line 6801)
add_newdoc_494633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6801, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6801)
add_newdoc_call_result_494638 = invoke(stypy.reporting.localization.Localization(__file__, 6801, 0), add_newdoc_494633, *[str_494634, str_494635, str_494636], **kwargs_494637)


# Call to add_newdoc(...): (line 6809)
# Processing the call arguments (line 6809)
str_494640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6809, 11), 'str', 'scipy.special')
str_494641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6809, 28), 'str', 'zetac')
str_494642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6825, (-1)), 'str', '\n    zetac(x)\n\n    Riemann zeta function minus 1.\n\n    This function is defined as\n\n    .. math:: \\zeta(x) = \\sum_{k=2}^{\\infty} 1 / k^x,\n\n    where ``x > 1``.\n\n    See Also\n    --------\n    zeta\n\n    ')
# Processing the call keyword arguments (line 6809)
kwargs_494643 = {}
# Getting the type of 'add_newdoc' (line 6809)
add_newdoc_494639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6809, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6809)
add_newdoc_call_result_494644 = invoke(stypy.reporting.localization.Localization(__file__, 6809, 0), add_newdoc_494639, *[str_494640, str_494641, str_494642], **kwargs_494643)


# Call to add_newdoc(...): (line 6827)
# Processing the call arguments (line 6827)
str_494646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6827, 11), 'str', 'scipy.special')
str_494647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6827, 28), 'str', '_struve_asymp_large_z')
str_494648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6838, (-1)), 'str', '\n    _struve_asymp_large_z(v, z, is_h)\n\n    Internal function for testing `struve` & `modstruve`\n\n    Evaluates using asymptotic expansion\n\n    Returns\n    -------\n    v, err\n    ')
# Processing the call keyword arguments (line 6827)
kwargs_494649 = {}
# Getting the type of 'add_newdoc' (line 6827)
add_newdoc_494645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6827, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6827)
add_newdoc_call_result_494650 = invoke(stypy.reporting.localization.Localization(__file__, 6827, 0), add_newdoc_494645, *[str_494646, str_494647, str_494648], **kwargs_494649)


# Call to add_newdoc(...): (line 6840)
# Processing the call arguments (line 6840)
str_494652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6840, 11), 'str', 'scipy.special')
str_494653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6840, 28), 'str', '_struve_power_series')
str_494654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6851, (-1)), 'str', '\n    _struve_power_series(v, z, is_h)\n\n    Internal function for testing `struve` & `modstruve`\n\n    Evaluates using power series\n\n    Returns\n    -------\n    v, err\n    ')
# Processing the call keyword arguments (line 6840)
kwargs_494655 = {}
# Getting the type of 'add_newdoc' (line 6840)
add_newdoc_494651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6840, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6840)
add_newdoc_call_result_494656 = invoke(stypy.reporting.localization.Localization(__file__, 6840, 0), add_newdoc_494651, *[str_494652, str_494653, str_494654], **kwargs_494655)


# Call to add_newdoc(...): (line 6853)
# Processing the call arguments (line 6853)
str_494658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 11), 'str', 'scipy.special')
str_494659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6853, 28), 'str', '_struve_bessel_series')
str_494660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6864, (-1)), 'str', '\n    _struve_bessel_series(v, z, is_h)\n\n    Internal function for testing `struve` & `modstruve`\n\n    Evaluates using Bessel function series\n\n    Returns\n    -------\n    v, err\n    ')
# Processing the call keyword arguments (line 6853)
kwargs_494661 = {}
# Getting the type of 'add_newdoc' (line 6853)
add_newdoc_494657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6853, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6853)
add_newdoc_call_result_494662 = invoke(stypy.reporting.localization.Localization(__file__, 6853, 0), add_newdoc_494657, *[str_494658, str_494659, str_494660], **kwargs_494661)


# Call to add_newdoc(...): (line 6866)
# Processing the call arguments (line 6866)
str_494664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6866, 11), 'str', 'scipy.special')
str_494665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6866, 28), 'str', '_spherical_jn')
str_494666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6869, (-1)), 'str', '\n    Internal function, use `spherical_jn` instead.\n    ')
# Processing the call keyword arguments (line 6866)
kwargs_494667 = {}
# Getting the type of 'add_newdoc' (line 6866)
add_newdoc_494663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6866, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6866)
add_newdoc_call_result_494668 = invoke(stypy.reporting.localization.Localization(__file__, 6866, 0), add_newdoc_494663, *[str_494664, str_494665, str_494666], **kwargs_494667)


# Call to add_newdoc(...): (line 6871)
# Processing the call arguments (line 6871)
str_494670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6871, 11), 'str', 'scipy.special')
str_494671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6871, 28), 'str', '_spherical_jn_d')
str_494672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6874, (-1)), 'str', '\n    Internal function, use `spherical_jn` instead.\n    ')
# Processing the call keyword arguments (line 6871)
kwargs_494673 = {}
# Getting the type of 'add_newdoc' (line 6871)
add_newdoc_494669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6871, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6871)
add_newdoc_call_result_494674 = invoke(stypy.reporting.localization.Localization(__file__, 6871, 0), add_newdoc_494669, *[str_494670, str_494671, str_494672], **kwargs_494673)


# Call to add_newdoc(...): (line 6876)
# Processing the call arguments (line 6876)
str_494676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6876, 11), 'str', 'scipy.special')
str_494677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6876, 28), 'str', '_spherical_yn')
str_494678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6879, (-1)), 'str', '\n    Internal function, use `spherical_yn` instead.\n    ')
# Processing the call keyword arguments (line 6876)
kwargs_494679 = {}
# Getting the type of 'add_newdoc' (line 6876)
add_newdoc_494675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6876, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6876)
add_newdoc_call_result_494680 = invoke(stypy.reporting.localization.Localization(__file__, 6876, 0), add_newdoc_494675, *[str_494676, str_494677, str_494678], **kwargs_494679)


# Call to add_newdoc(...): (line 6881)
# Processing the call arguments (line 6881)
str_494682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6881, 11), 'str', 'scipy.special')
str_494683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6881, 28), 'str', '_spherical_yn_d')
str_494684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6884, (-1)), 'str', '\n    Internal function, use `spherical_yn` instead.\n    ')
# Processing the call keyword arguments (line 6881)
kwargs_494685 = {}
# Getting the type of 'add_newdoc' (line 6881)
add_newdoc_494681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6881, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6881)
add_newdoc_call_result_494686 = invoke(stypy.reporting.localization.Localization(__file__, 6881, 0), add_newdoc_494681, *[str_494682, str_494683, str_494684], **kwargs_494685)


# Call to add_newdoc(...): (line 6886)
# Processing the call arguments (line 6886)
str_494688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6886, 11), 'str', 'scipy.special')
str_494689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6886, 28), 'str', '_spherical_in')
str_494690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6889, (-1)), 'str', '\n    Internal function, use `spherical_in` instead.\n    ')
# Processing the call keyword arguments (line 6886)
kwargs_494691 = {}
# Getting the type of 'add_newdoc' (line 6886)
add_newdoc_494687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6886, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6886)
add_newdoc_call_result_494692 = invoke(stypy.reporting.localization.Localization(__file__, 6886, 0), add_newdoc_494687, *[str_494688, str_494689, str_494690], **kwargs_494691)


# Call to add_newdoc(...): (line 6891)
# Processing the call arguments (line 6891)
str_494694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6891, 11), 'str', 'scipy.special')
str_494695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6891, 28), 'str', '_spherical_in_d')
str_494696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6894, (-1)), 'str', '\n    Internal function, use `spherical_in` instead.\n    ')
# Processing the call keyword arguments (line 6891)
kwargs_494697 = {}
# Getting the type of 'add_newdoc' (line 6891)
add_newdoc_494693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6891, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6891)
add_newdoc_call_result_494698 = invoke(stypy.reporting.localization.Localization(__file__, 6891, 0), add_newdoc_494693, *[str_494694, str_494695, str_494696], **kwargs_494697)


# Call to add_newdoc(...): (line 6896)
# Processing the call arguments (line 6896)
str_494700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6896, 11), 'str', 'scipy.special')
str_494701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6896, 28), 'str', '_spherical_kn')
str_494702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6899, (-1)), 'str', '\n    Internal function, use `spherical_kn` instead.\n    ')
# Processing the call keyword arguments (line 6896)
kwargs_494703 = {}
# Getting the type of 'add_newdoc' (line 6896)
add_newdoc_494699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6896, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6896)
add_newdoc_call_result_494704 = invoke(stypy.reporting.localization.Localization(__file__, 6896, 0), add_newdoc_494699, *[str_494700, str_494701, str_494702], **kwargs_494703)


# Call to add_newdoc(...): (line 6901)
# Processing the call arguments (line 6901)
str_494706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 11), 'str', 'scipy.special')
str_494707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6901, 28), 'str', '_spherical_kn_d')
str_494708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6904, (-1)), 'str', '\n    Internal function, use `spherical_kn` instead.\n    ')
# Processing the call keyword arguments (line 6901)
kwargs_494709 = {}
# Getting the type of 'add_newdoc' (line 6901)
add_newdoc_494705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6901, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6901)
add_newdoc_call_result_494710 = invoke(stypy.reporting.localization.Localization(__file__, 6901, 0), add_newdoc_494705, *[str_494706, str_494707, str_494708], **kwargs_494709)


# Call to add_newdoc(...): (line 6906)
# Processing the call arguments (line 6906)
str_494712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6906, 11), 'str', 'scipy.special')
str_494713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6906, 28), 'str', 'loggamma')
str_494714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6965, (-1)), 'str', '\n    loggamma(z, out=None)\n\n    Principal branch of the logarithm of the Gamma function.\n\n    Defined to be :math:`\\log(\\Gamma(x))` for :math:`x > 0` and\n    extended to the complex plane by analytic continuation. The\n    function has a single branch cut on the negative real axis.\n\n    .. versionadded:: 0.18.0\n\n    Parameters\n    ----------\n    z : array-like\n        Values in the complex plain at which to compute ``loggamma``\n    out : ndarray, optional\n        Output array for computed values of ``loggamma``\n\n    Returns\n    -------\n    loggamma : ndarray\n        Values of ``loggamma`` at z.\n\n    Notes\n    -----\n    It is not generally true that :math:`\\log\\Gamma(z) =\n    \\log(\\Gamma(z))`, though the real parts of the functions do\n    agree. The benefit of not defining ``loggamma`` as\n    :math:`\\log(\\Gamma(z))` is that the latter function has a\n    complicated branch cut structure whereas ``loggamma`` is analytic\n    except for on the negative real axis.\n\n    The identities\n\n    .. math::\n      \\exp(\\log\\Gamma(z)) &= \\Gamma(z) \\\\\n      \\log\\Gamma(z + 1) &= \\log(z) + \\log\\Gamma(z)\n\n    make ``loggama`` useful for working in complex logspace. However,\n    ``loggamma`` necessarily returns complex outputs for real inputs,\n    so if you want to work only with real numbers use `gammaln`. On\n    the real line the two functions are related by ``exp(loggamma(x))\n    = gammasgn(x)*exp(gammaln(x))``, though in practice rounding\n    errors will introduce small spurious imaginary components in\n    ``exp(loggamma(x))``.\n\n    The implementation here is based on [hare1997]_.\n\n    See also\n    --------\n    gammaln : logarithm of the absolute value of the Gamma function\n    gammasgn : sign of the gamma function\n\n    References\n    ----------\n    .. [hare1997] D.E.G. Hare,\n      *Computing the Principal Branch of log-Gamma*,\n      Journal of Algorithms, Volume 25, Issue 2, November 1997, pages 221-236.\n    ')
# Processing the call keyword arguments (line 6906)
kwargs_494715 = {}
# Getting the type of 'add_newdoc' (line 6906)
add_newdoc_494711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6906, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6906)
add_newdoc_call_result_494716 = invoke(stypy.reporting.localization.Localization(__file__, 6906, 0), add_newdoc_494711, *[str_494712, str_494713, str_494714], **kwargs_494715)


# Call to add_newdoc(...): (line 6967)
# Processing the call arguments (line 6967)
str_494718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6967, 11), 'str', 'scipy.special')
str_494719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6967, 28), 'str', '_sinpi')
str_494720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6970, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 6967)
kwargs_494721 = {}
# Getting the type of 'add_newdoc' (line 6967)
add_newdoc_494717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6967, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6967)
add_newdoc_call_result_494722 = invoke(stypy.reporting.localization.Localization(__file__, 6967, 0), add_newdoc_494717, *[str_494718, str_494719, str_494720], **kwargs_494721)


# Call to add_newdoc(...): (line 6972)
# Processing the call arguments (line 6972)
str_494724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6972, 11), 'str', 'scipy.special')
str_494725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6972, 28), 'str', '_cospi')
str_494726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6975, (-1)), 'str', '\n    Internal function, do not use.\n    ')
# Processing the call keyword arguments (line 6972)
kwargs_494727 = {}
# Getting the type of 'add_newdoc' (line 6972)
add_newdoc_494723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6972, 0), 'add_newdoc', False)
# Calling add_newdoc(args, kwargs) (line 6972)
add_newdoc_call_result_494728 = invoke(stypy.reporting.localization.Localization(__file__, 6972, 0), add_newdoc_494723, *[str_494724, str_494725, str_494726], **kwargs_494727)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
