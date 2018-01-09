
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Objects for dealing with Hermite series.
3: 
4: This module provides a number of objects (mostly functions) useful for
5: dealing with Hermite series, including a `Hermite` class that
6: encapsulates the usual arithmetic operations.  (General information
7: on how this module represents and works with such polynomials is in the
8: docstring for its "parent" sub-package, `numpy.polynomial`).
9: 
10: Constants
11: ---------
12: - `hermdomain` -- Hermite series default domain, [-1,1].
13: - `hermzero` -- Hermite series that evaluates identically to 0.
14: - `hermone` -- Hermite series that evaluates identically to 1.
15: - `hermx` -- Hermite series for the identity map, ``f(x) = x``.
16: 
17: Arithmetic
18: ----------
19: - `hermmulx` -- multiply a Hermite series in ``P_i(x)`` by ``x``.
20: - `hermadd` -- add two Hermite series.
21: - `hermsub` -- subtract one Hermite series from another.
22: - `hermmul` -- multiply two Hermite series.
23: - `hermdiv` -- divide one Hermite series by another.
24: - `hermval` -- evaluate a Hermite series at given points.
25: - `hermval2d` -- evaluate a 2D Hermite series at given points.
26: - `hermval3d` -- evaluate a 3D Hermite series at given points.
27: - `hermgrid2d` -- evaluate a 2D Hermite series on a Cartesian product.
28: - `hermgrid3d` -- evaluate a 3D Hermite series on a Cartesian product.
29: 
30: Calculus
31: --------
32: - `hermder` -- differentiate a Hermite series.
33: - `hermint` -- integrate a Hermite series.
34: 
35: Misc Functions
36: --------------
37: - `hermfromroots` -- create a Hermite series with specified roots.
38: - `hermroots` -- find the roots of a Hermite series.
39: - `hermvander` -- Vandermonde-like matrix for Hermite polynomials.
40: - `hermvander2d` -- Vandermonde-like matrix for 2D power series.
41: - `hermvander3d` -- Vandermonde-like matrix for 3D power series.
42: - `hermgauss` -- Gauss-Hermite quadrature, points and weights.
43: - `hermweight` -- Hermite weight function.
44: - `hermcompanion` -- symmetrized companion matrix in Hermite form.
45: - `hermfit` -- least-squares fit returning a Hermite series.
46: - `hermtrim` -- trim leading coefficients from a Hermite series.
47: - `hermline` -- Hermite series of given straight line.
48: - `herm2poly` -- convert a Hermite series to a polynomial.
49: - `poly2herm` -- convert a polynomial to a Hermite series.
50: 
51: Classes
52: -------
53: - `Hermite` -- A Hermite series class.
54: 
55: See also
56: --------
57: `numpy.polynomial`
58: 
59: '''
60: from __future__ import division, absolute_import, print_function
61: 
62: import warnings
63: import numpy as np
64: import numpy.linalg as la
65: 
66: from . import polyutils as pu
67: from ._polybase import ABCPolyBase
68: 
69: __all__ = [
70:     'hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline', 'hermadd',
71:     'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow', 'hermval',
72:     'hermder', 'hermint', 'herm2poly', 'poly2herm', 'hermfromroots',
73:     'hermvander', 'hermfit', 'hermtrim', 'hermroots', 'Hermite',
74:     'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d', 'hermvander2d',
75:     'hermvander3d', 'hermcompanion', 'hermgauss', 'hermweight']
76: 
77: hermtrim = pu.trimcoef
78: 
79: 
80: def poly2herm(pol):
81:     '''
82:     poly2herm(pol)
83: 
84:     Convert a polynomial to a Hermite series.
85: 
86:     Convert an array representing the coefficients of a polynomial (relative
87:     to the "standard" basis) ordered from lowest degree to highest, to an
88:     array of the coefficients of the equivalent Hermite series, ordered
89:     from lowest to highest degree.
90: 
91:     Parameters
92:     ----------
93:     pol : array_like
94:         1-D array containing the polynomial coefficients
95: 
96:     Returns
97:     -------
98:     c : ndarray
99:         1-D array containing the coefficients of the equivalent Hermite
100:         series.
101: 
102:     See Also
103:     --------
104:     herm2poly
105: 
106:     Notes
107:     -----
108:     The easy way to do conversions between polynomial basis sets
109:     is to use the convert method of a class instance.
110: 
111:     Examples
112:     --------
113:     >>> from numpy.polynomial.hermite import poly2herm
114:     >>> poly2herm(np.arange(4))
115:     array([ 1.   ,  2.75 ,  0.5  ,  0.375])
116: 
117:     '''
118:     [pol] = pu.as_series([pol])
119:     deg = len(pol) - 1
120:     res = 0
121:     for i in range(deg, -1, -1):
122:         res = hermadd(hermmulx(res), pol[i])
123:     return res
124: 
125: 
126: def herm2poly(c):
127:     '''
128:     Convert a Hermite series to a polynomial.
129: 
130:     Convert an array representing the coefficients of a Hermite series,
131:     ordered from lowest degree to highest, to an array of the coefficients
132:     of the equivalent polynomial (relative to the "standard" basis) ordered
133:     from lowest to highest degree.
134: 
135:     Parameters
136:     ----------
137:     c : array_like
138:         1-D array containing the Hermite series coefficients, ordered
139:         from lowest order term to highest.
140: 
141:     Returns
142:     -------
143:     pol : ndarray
144:         1-D array containing the coefficients of the equivalent polynomial
145:         (relative to the "standard" basis) ordered from lowest order term
146:         to highest.
147: 
148:     See Also
149:     --------
150:     poly2herm
151: 
152:     Notes
153:     -----
154:     The easy way to do conversions between polynomial basis sets
155:     is to use the convert method of a class instance.
156: 
157:     Examples
158:     --------
159:     >>> from numpy.polynomial.hermite import herm2poly
160:     >>> herm2poly([ 1.   ,  2.75 ,  0.5  ,  0.375])
161:     array([ 0.,  1.,  2.,  3.])
162: 
163:     '''
164:     from .polynomial import polyadd, polysub, polymulx
165: 
166:     [c] = pu.as_series([c])
167:     n = len(c)
168:     if n == 1:
169:         return c
170:     if n == 2:
171:         c[1] *= 2
172:         return c
173:     else:
174:         c0 = c[-2]
175:         c1 = c[-1]
176:         # i is the current degree of c1
177:         for i in range(n - 1, 1, -1):
178:             tmp = c0
179:             c0 = polysub(c[i - 2], c1*(2*(i - 1)))
180:             c1 = polyadd(tmp, polymulx(c1)*2)
181:         return polyadd(c0, polymulx(c1)*2)
182: 
183: #
184: # These are constant arrays are of integer type so as to be compatible
185: # with the widest range of other types, such as Decimal.
186: #
187: 
188: # Hermite
189: hermdomain = np.array([-1, 1])
190: 
191: # Hermite coefficients representing zero.
192: hermzero = np.array([0])
193: 
194: # Hermite coefficients representing one.
195: hermone = np.array([1])
196: 
197: # Hermite coefficients representing the identity x.
198: hermx = np.array([0, 1/2])
199: 
200: 
201: def hermline(off, scl):
202:     '''
203:     Hermite series whose graph is a straight line.
204: 
205: 
206: 
207:     Parameters
208:     ----------
209:     off, scl : scalars
210:         The specified line is given by ``off + scl*x``.
211: 
212:     Returns
213:     -------
214:     y : ndarray
215:         This module's representation of the Hermite series for
216:         ``off + scl*x``.
217: 
218:     See Also
219:     --------
220:     polyline, chebline
221: 
222:     Examples
223:     --------
224:     >>> from numpy.polynomial.hermite import hermline, hermval
225:     >>> hermval(0,hermline(3, 2))
226:     3.0
227:     >>> hermval(1,hermline(3, 2))
228:     5.0
229: 
230:     '''
231:     if scl != 0:
232:         return np.array([off, scl/2])
233:     else:
234:         return np.array([off])
235: 
236: 
237: def hermfromroots(roots):
238:     '''
239:     Generate a Hermite series with given roots.
240: 
241:     The function returns the coefficients of the polynomial
242: 
243:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
244: 
245:     in Hermite form, where the `r_n` are the roots specified in `roots`.
246:     If a zero has multiplicity n, then it must appear in `roots` n times.
247:     For instance, if 2 is a root of multiplicity three and 3 is a root of
248:     multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
249:     roots can appear in any order.
250: 
251:     If the returned coefficients are `c`, then
252: 
253:     .. math:: p(x) = c_0 + c_1 * H_1(x) + ... +  c_n * H_n(x)
254: 
255:     The coefficient of the last term is not generally 1 for monic
256:     polynomials in Hermite form.
257: 
258:     Parameters
259:     ----------
260:     roots : array_like
261:         Sequence containing the roots.
262: 
263:     Returns
264:     -------
265:     out : ndarray
266:         1-D array of coefficients.  If all roots are real then `out` is a
267:         real array, if some of the roots are complex, then `out` is complex
268:         even if all the coefficients in the result are real (see Examples
269:         below).
270: 
271:     See Also
272:     --------
273:     polyfromroots, legfromroots, lagfromroots, chebfromroots,
274:     hermefromroots.
275: 
276:     Examples
277:     --------
278:     >>> from numpy.polynomial.hermite import hermfromroots, hermval
279:     >>> coef = hermfromroots((-1, 0, 1))
280:     >>> hermval((-1, 0, 1), coef)
281:     array([ 0.,  0.,  0.])
282:     >>> coef = hermfromroots((-1j, 1j))
283:     >>> hermval((-1j, 1j), coef)
284:     array([ 0.+0.j,  0.+0.j])
285: 
286:     '''
287:     if len(roots) == 0:
288:         return np.ones(1)
289:     else:
290:         [roots] = pu.as_series([roots], trim=False)
291:         roots.sort()
292:         p = [hermline(-r, 1) for r in roots]
293:         n = len(p)
294:         while n > 1:
295:             m, r = divmod(n, 2)
296:             tmp = [hermmul(p[i], p[i+m]) for i in range(m)]
297:             if r:
298:                 tmp[0] = hermmul(tmp[0], p[-1])
299:             p = tmp
300:             n = m
301:         return p[0]
302: 
303: 
304: def hermadd(c1, c2):
305:     '''
306:     Add one Hermite series to another.
307: 
308:     Returns the sum of two Hermite series `c1` + `c2`.  The arguments
309:     are sequences of coefficients ordered from lowest order term to
310:     highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
311: 
312:     Parameters
313:     ----------
314:     c1, c2 : array_like
315:         1-D arrays of Hermite series coefficients ordered from low to
316:         high.
317: 
318:     Returns
319:     -------
320:     out : ndarray
321:         Array representing the Hermite series of their sum.
322: 
323:     See Also
324:     --------
325:     hermsub, hermmul, hermdiv, hermpow
326: 
327:     Notes
328:     -----
329:     Unlike multiplication, division, etc., the sum of two Hermite series
330:     is a Hermite series (without having to "reproject" the result onto
331:     the basis set) so addition, just like that of "standard" polynomials,
332:     is simply "component-wise."
333: 
334:     Examples
335:     --------
336:     >>> from numpy.polynomial.hermite import hermadd
337:     >>> hermadd([1, 2, 3], [1, 2, 3, 4])
338:     array([ 2.,  4.,  6.,  4.])
339: 
340:     '''
341:     # c1, c2 are trimmed copies
342:     [c1, c2] = pu.as_series([c1, c2])
343:     if len(c1) > len(c2):
344:         c1[:c2.size] += c2
345:         ret = c1
346:     else:
347:         c2[:c1.size] += c1
348:         ret = c2
349:     return pu.trimseq(ret)
350: 
351: 
352: def hermsub(c1, c2):
353:     '''
354:     Subtract one Hermite series from another.
355: 
356:     Returns the difference of two Hermite series `c1` - `c2`.  The
357:     sequences of coefficients are from lowest order term to highest, i.e.,
358:     [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
359: 
360:     Parameters
361:     ----------
362:     c1, c2 : array_like
363:         1-D arrays of Hermite series coefficients ordered from low to
364:         high.
365: 
366:     Returns
367:     -------
368:     out : ndarray
369:         Of Hermite series coefficients representing their difference.
370: 
371:     See Also
372:     --------
373:     hermadd, hermmul, hermdiv, hermpow
374: 
375:     Notes
376:     -----
377:     Unlike multiplication, division, etc., the difference of two Hermite
378:     series is a Hermite series (without having to "reproject" the result
379:     onto the basis set) so subtraction, just like that of "standard"
380:     polynomials, is simply "component-wise."
381: 
382:     Examples
383:     --------
384:     >>> from numpy.polynomial.hermite import hermsub
385:     >>> hermsub([1, 2, 3, 4], [1, 2, 3])
386:     array([ 0.,  0.,  0.,  4.])
387: 
388:     '''
389:     # c1, c2 are trimmed copies
390:     [c1, c2] = pu.as_series([c1, c2])
391:     if len(c1) > len(c2):
392:         c1[:c2.size] -= c2
393:         ret = c1
394:     else:
395:         c2 = -c2
396:         c2[:c1.size] += c1
397:         ret = c2
398:     return pu.trimseq(ret)
399: 
400: 
401: def hermmulx(c):
402:     '''Multiply a Hermite series by x.
403: 
404:     Multiply the Hermite series `c` by x, where x is the independent
405:     variable.
406: 
407: 
408:     Parameters
409:     ----------
410:     c : array_like
411:         1-D array of Hermite series coefficients ordered from low to
412:         high.
413: 
414:     Returns
415:     -------
416:     out : ndarray
417:         Array representing the result of the multiplication.
418: 
419:     Notes
420:     -----
421:     The multiplication uses the recursion relationship for Hermite
422:     polynomials in the form
423: 
424:     .. math::
425: 
426:     xP_i(x) = (P_{i + 1}(x)/2 + i*P_{i - 1}(x))
427: 
428:     Examples
429:     --------
430:     >>> from numpy.polynomial.hermite import hermmulx
431:     >>> hermmulx([1, 2, 3])
432:     array([ 2. ,  6.5,  1. ,  1.5])
433: 
434:     '''
435:     # c is a trimmed copy
436:     [c] = pu.as_series([c])
437:     # The zero series needs special treatment
438:     if len(c) == 1 and c[0] == 0:
439:         return c
440: 
441:     prd = np.empty(len(c) + 1, dtype=c.dtype)
442:     prd[0] = c[0]*0
443:     prd[1] = c[0]/2
444:     for i in range(1, len(c)):
445:         prd[i + 1] = c[i]/2
446:         prd[i - 1] += c[i]*i
447:     return prd
448: 
449: 
450: def hermmul(c1, c2):
451:     '''
452:     Multiply one Hermite series by another.
453: 
454:     Returns the product of two Hermite series `c1` * `c2`.  The arguments
455:     are sequences of coefficients, from lowest order "term" to highest,
456:     e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
457: 
458:     Parameters
459:     ----------
460:     c1, c2 : array_like
461:         1-D arrays of Hermite series coefficients ordered from low to
462:         high.
463: 
464:     Returns
465:     -------
466:     out : ndarray
467:         Of Hermite series coefficients representing their product.
468: 
469:     See Also
470:     --------
471:     hermadd, hermsub, hermdiv, hermpow
472: 
473:     Notes
474:     -----
475:     In general, the (polynomial) product of two C-series results in terms
476:     that are not in the Hermite polynomial basis set.  Thus, to express
477:     the product as a Hermite series, it is necessary to "reproject" the
478:     product onto said basis set, which may produce "unintuitive" (but
479:     correct) results; see Examples section below.
480: 
481:     Examples
482:     --------
483:     >>> from numpy.polynomial.hermite import hermmul
484:     >>> hermmul([1, 2, 3], [0, 1, 2])
485:     array([ 52.,  29.,  52.,   7.,   6.])
486: 
487:     '''
488:     # s1, s2 are trimmed copies
489:     [c1, c2] = pu.as_series([c1, c2])
490: 
491:     if len(c1) > len(c2):
492:         c = c2
493:         xs = c1
494:     else:
495:         c = c1
496:         xs = c2
497: 
498:     if len(c) == 1:
499:         c0 = c[0]*xs
500:         c1 = 0
501:     elif len(c) == 2:
502:         c0 = c[0]*xs
503:         c1 = c[1]*xs
504:     else:
505:         nd = len(c)
506:         c0 = c[-2]*xs
507:         c1 = c[-1]*xs
508:         for i in range(3, len(c) + 1):
509:             tmp = c0
510:             nd = nd - 1
511:             c0 = hermsub(c[-i]*xs, c1*(2*(nd - 1)))
512:             c1 = hermadd(tmp, hermmulx(c1)*2)
513:     return hermadd(c0, hermmulx(c1)*2)
514: 
515: 
516: def hermdiv(c1, c2):
517:     '''
518:     Divide one Hermite series by another.
519: 
520:     Returns the quotient-with-remainder of two Hermite series
521:     `c1` / `c2`.  The arguments are sequences of coefficients from lowest
522:     order "term" to highest, e.g., [1,2,3] represents the series
523:     ``P_0 + 2*P_1 + 3*P_2``.
524: 
525:     Parameters
526:     ----------
527:     c1, c2 : array_like
528:         1-D arrays of Hermite series coefficients ordered from low to
529:         high.
530: 
531:     Returns
532:     -------
533:     [quo, rem] : ndarrays
534:         Of Hermite series coefficients representing the quotient and
535:         remainder.
536: 
537:     See Also
538:     --------
539:     hermadd, hermsub, hermmul, hermpow
540: 
541:     Notes
542:     -----
543:     In general, the (polynomial) division of one Hermite series by another
544:     results in quotient and remainder terms that are not in the Hermite
545:     polynomial basis set.  Thus, to express these results as a Hermite
546:     series, it is necessary to "reproject" the results onto the Hermite
547:     basis set, which may produce "unintuitive" (but correct) results; see
548:     Examples section below.
549: 
550:     Examples
551:     --------
552:     >>> from numpy.polynomial.hermite import hermdiv
553:     >>> hermdiv([ 52.,  29.,  52.,   7.,   6.], [0, 1, 2])
554:     (array([ 1.,  2.,  3.]), array([ 0.]))
555:     >>> hermdiv([ 54.,  31.,  52.,   7.,   6.], [0, 1, 2])
556:     (array([ 1.,  2.,  3.]), array([ 2.,  2.]))
557:     >>> hermdiv([ 53.,  30.,  52.,   7.,   6.], [0, 1, 2])
558:     (array([ 1.,  2.,  3.]), array([ 1.,  1.]))
559: 
560:     '''
561:     # c1, c2 are trimmed copies
562:     [c1, c2] = pu.as_series([c1, c2])
563:     if c2[-1] == 0:
564:         raise ZeroDivisionError()
565: 
566:     lc1 = len(c1)
567:     lc2 = len(c2)
568:     if lc1 < lc2:
569:         return c1[:1]*0, c1
570:     elif lc2 == 1:
571:         return c1/c2[-1], c1[:1]*0
572:     else:
573:         quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
574:         rem = c1
575:         for i in range(lc1 - lc2, - 1, -1):
576:             p = hermmul([0]*i + [1], c2)
577:             q = rem[-1]/p[-1]
578:             rem = rem[:-1] - q*p[:-1]
579:             quo[i] = q
580:         return quo, pu.trimseq(rem)
581: 
582: 
583: def hermpow(c, pow, maxpower=16):
584:     '''Raise a Hermite series to a power.
585: 
586:     Returns the Hermite series `c` raised to the power `pow`. The
587:     argument `c` is a sequence of coefficients ordered from low to high.
588:     i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
589: 
590:     Parameters
591:     ----------
592:     c : array_like
593:         1-D array of Hermite series coefficients ordered from low to
594:         high.
595:     pow : integer
596:         Power to which the series will be raised
597:     maxpower : integer, optional
598:         Maximum power allowed. This is mainly to limit growth of the series
599:         to unmanageable size. Default is 16
600: 
601:     Returns
602:     -------
603:     coef : ndarray
604:         Hermite series of power.
605: 
606:     See Also
607:     --------
608:     hermadd, hermsub, hermmul, hermdiv
609: 
610:     Examples
611:     --------
612:     >>> from numpy.polynomial.hermite import hermpow
613:     >>> hermpow([1, 2, 3], 2)
614:     array([ 81.,  52.,  82.,  12.,   9.])
615: 
616:     '''
617:     # c is a trimmed copy
618:     [c] = pu.as_series([c])
619:     power = int(pow)
620:     if power != pow or power < 0:
621:         raise ValueError("Power must be a non-negative integer.")
622:     elif maxpower is not None and power > maxpower:
623:         raise ValueError("Power is too large")
624:     elif power == 0:
625:         return np.array([1], dtype=c.dtype)
626:     elif power == 1:
627:         return c
628:     else:
629:         # This can be made more efficient by using powers of two
630:         # in the usual way.
631:         prd = c
632:         for i in range(2, power + 1):
633:             prd = hermmul(prd, c)
634:         return prd
635: 
636: 
637: def hermder(c, m=1, scl=1, axis=0):
638:     '''
639:     Differentiate a Hermite series.
640: 
641:     Returns the Hermite series coefficients `c` differentiated `m` times
642:     along `axis`.  At each iteration the result is multiplied by `scl` (the
643:     scaling factor is for use in a linear change of variable). The argument
644:     `c` is an array of coefficients from low to high degree along each
645:     axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``
646:     while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +
647:     2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is
648:     ``y``.
649: 
650:     Parameters
651:     ----------
652:     c : array_like
653:         Array of Hermite series coefficients. If `c` is multidimensional the
654:         different axis correspond to different variables with the degree in
655:         each axis given by the corresponding index.
656:     m : int, optional
657:         Number of derivatives taken, must be non-negative. (Default: 1)
658:     scl : scalar, optional
659:         Each differentiation is multiplied by `scl`.  The end result is
660:         multiplication by ``scl**m``.  This is for use in a linear change of
661:         variable. (Default: 1)
662:     axis : int, optional
663:         Axis over which the derivative is taken. (Default: 0).
664: 
665:         .. versionadded:: 1.7.0
666: 
667:     Returns
668:     -------
669:     der : ndarray
670:         Hermite series of the derivative.
671: 
672:     See Also
673:     --------
674:     hermint
675: 
676:     Notes
677:     -----
678:     In general, the result of differentiating a Hermite series does not
679:     resemble the same operation on a power series. Thus the result of this
680:     function may be "unintuitive," albeit correct; see Examples section
681:     below.
682: 
683:     Examples
684:     --------
685:     >>> from numpy.polynomial.hermite import hermder
686:     >>> hermder([ 1. ,  0.5,  0.5,  0.5])
687:     array([ 1.,  2.,  3.])
688:     >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)
689:     array([ 1.,  2.,  3.])
690: 
691:     '''
692:     c = np.array(c, ndmin=1, copy=1)
693:     if c.dtype.char in '?bBhHiIlLqQpP':
694:         c = c.astype(np.double)
695:     cnt, iaxis = [int(t) for t in [m, axis]]
696: 
697:     if cnt != m:
698:         raise ValueError("The order of derivation must be integer")
699:     if cnt < 0:
700:         raise ValueError("The order of derivation must be non-negative")
701:     if iaxis != axis:
702:         raise ValueError("The axis must be integer")
703:     if not -c.ndim <= iaxis < c.ndim:
704:         raise ValueError("The axis is out of range")
705:     if iaxis < 0:
706:         iaxis += c.ndim
707: 
708:     if cnt == 0:
709:         return c
710: 
711:     c = np.rollaxis(c, iaxis)
712:     n = len(c)
713:     if cnt >= n:
714:         c = c[:1]*0
715:     else:
716:         for i in range(cnt):
717:             n = n - 1
718:             c *= scl
719:             der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
720:             for j in range(n, 0, -1):
721:                 der[j - 1] = (2*j)*c[j]
722:             c = der
723:     c = np.rollaxis(c, 0, iaxis + 1)
724:     return c
725: 
726: 
727: def hermint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
728:     '''
729:     Integrate a Hermite series.
730: 
731:     Returns the Hermite series coefficients `c` integrated `m` times from
732:     `lbnd` along `axis`. At each iteration the resulting series is
733:     **multiplied** by `scl` and an integration constant, `k`, is added.
734:     The scaling factor is for use in a linear change of variable.  ("Buyer
735:     beware": note that, depending on what one is doing, one may want `scl`
736:     to be the reciprocal of what one might expect; for more information,
737:     see the Notes section below.)  The argument `c` is an array of
738:     coefficients from low to high degree along each axis, e.g., [1,2,3]
739:     represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
740:     represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
741:     2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
742: 
743:     Parameters
744:     ----------
745:     c : array_like
746:         Array of Hermite series coefficients. If c is multidimensional the
747:         different axis correspond to different variables with the degree in
748:         each axis given by the corresponding index.
749:     m : int, optional
750:         Order of integration, must be positive. (Default: 1)
751:     k : {[], list, scalar}, optional
752:         Integration constant(s).  The value of the first integral at
753:         ``lbnd`` is the first value in the list, the value of the second
754:         integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
755:         default), all constants are set to zero.  If ``m == 1``, a single
756:         scalar can be given instead of a list.
757:     lbnd : scalar, optional
758:         The lower bound of the integral. (Default: 0)
759:     scl : scalar, optional
760:         Following each integration the result is *multiplied* by `scl`
761:         before the integration constant is added. (Default: 1)
762:     axis : int, optional
763:         Axis over which the integral is taken. (Default: 0).
764: 
765:         .. versionadded:: 1.7.0
766: 
767:     Returns
768:     -------
769:     S : ndarray
770:         Hermite series coefficients of the integral.
771: 
772:     Raises
773:     ------
774:     ValueError
775:         If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
776:         ``np.isscalar(scl) == False``.
777: 
778:     See Also
779:     --------
780:     hermder
781: 
782:     Notes
783:     -----
784:     Note that the result of each integration is *multiplied* by `scl`.
785:     Why is this important to note?  Say one is making a linear change of
786:     variable :math:`u = ax + b` in an integral relative to `x`.  Then
787:     .. math::`dx = du/a`, so one will need to set `scl` equal to
788:     :math:`1/a` - perhaps not what one would have first thought.
789: 
790:     Also note that, in general, the result of integrating a C-series needs
791:     to be "reprojected" onto the C-series basis set.  Thus, typically,
792:     the result of this function is "unintuitive," albeit correct; see
793:     Examples section below.
794: 
795:     Examples
796:     --------
797:     >>> from numpy.polynomial.hermite import hermint
798:     >>> hermint([1,2,3]) # integrate once, value 0 at 0.
799:     array([ 1. ,  0.5,  0.5,  0.5])
800:     >>> hermint([1,2,3], m=2) # integrate twice, value & deriv 0 at 0
801:     array([-0.5       ,  0.5       ,  0.125     ,  0.08333333,  0.0625    ])
802:     >>> hermint([1,2,3], k=1) # integrate once, value 1 at 0.
803:     array([ 2. ,  0.5,  0.5,  0.5])
804:     >>> hermint([1,2,3], lbnd=-1) # integrate once, value 0 at -1
805:     array([-2. ,  0.5,  0.5,  0.5])
806:     >>> hermint([1,2,3], m=2, k=[1,2], lbnd=-1)
807:     array([ 1.66666667, -0.5       ,  0.125     ,  0.08333333,  0.0625    ])
808: 
809:     '''
810:     c = np.array(c, ndmin=1, copy=1)
811:     if c.dtype.char in '?bBhHiIlLqQpP':
812:         c = c.astype(np.double)
813:     if not np.iterable(k):
814:         k = [k]
815:     cnt, iaxis = [int(t) for t in [m, axis]]
816: 
817:     if cnt != m:
818:         raise ValueError("The order of integration must be integer")
819:     if cnt < 0:
820:         raise ValueError("The order of integration must be non-negative")
821:     if len(k) > cnt:
822:         raise ValueError("Too many integration constants")
823:     if iaxis != axis:
824:         raise ValueError("The axis must be integer")
825:     if not -c.ndim <= iaxis < c.ndim:
826:         raise ValueError("The axis is out of range")
827:     if iaxis < 0:
828:         iaxis += c.ndim
829: 
830:     if cnt == 0:
831:         return c
832: 
833:     c = np.rollaxis(c, iaxis)
834:     k = list(k) + [0]*(cnt - len(k))
835:     for i in range(cnt):
836:         n = len(c)
837:         c *= scl
838:         if n == 1 and np.all(c[0] == 0):
839:             c[0] += k[i]
840:         else:
841:             tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
842:             tmp[0] = c[0]*0
843:             tmp[1] = c[0]/2
844:             for j in range(1, n):
845:                 tmp[j + 1] = c[j]/(2*(j + 1))
846:             tmp[0] += k[i] - hermval(lbnd, tmp)
847:             c = tmp
848:     c = np.rollaxis(c, 0, iaxis + 1)
849:     return c
850: 
851: 
852: def hermval(x, c, tensor=True):
853:     '''
854:     Evaluate an Hermite series at points x.
855: 
856:     If `c` is of length `n + 1`, this function returns the value:
857: 
858:     .. math:: p(x) = c_0 * H_0(x) + c_1 * H_1(x) + ... + c_n * H_n(x)
859: 
860:     The parameter `x` is converted to an array only if it is a tuple or a
861:     list, otherwise it is treated as a scalar. In either case, either `x`
862:     or its elements must support multiplication and addition both with
863:     themselves and with the elements of `c`.
864: 
865:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
866:     `c` is multidimensional, then the shape of the result depends on the
867:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
868:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
869:     scalars have shape (,).
870: 
871:     Trailing zeros in the coefficients will be used in the evaluation, so
872:     they should be avoided if efficiency is a concern.
873: 
874:     Parameters
875:     ----------
876:     x : array_like, compatible object
877:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
878:         it is left unchanged and treated as a scalar. In either case, `x`
879:         or its elements must support addition and multiplication with
880:         with themselves and with the elements of `c`.
881:     c : array_like
882:         Array of coefficients ordered so that the coefficients for terms of
883:         degree n are contained in c[n]. If `c` is multidimensional the
884:         remaining indices enumerate multiple polynomials. In the two
885:         dimensional case the coefficients may be thought of as stored in
886:         the columns of `c`.
887:     tensor : boolean, optional
888:         If True, the shape of the coefficient array is extended with ones
889:         on the right, one for each dimension of `x`. Scalars have dimension 0
890:         for this action. The result is that every column of coefficients in
891:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
892:         over the columns of `c` for the evaluation.  This keyword is useful
893:         when `c` is multidimensional. The default value is True.
894: 
895:         .. versionadded:: 1.7.0
896: 
897:     Returns
898:     -------
899:     values : ndarray, algebra_like
900:         The shape of the return value is described above.
901: 
902:     See Also
903:     --------
904:     hermval2d, hermgrid2d, hermval3d, hermgrid3d
905: 
906:     Notes
907:     -----
908:     The evaluation uses Clenshaw recursion, aka synthetic division.
909: 
910:     Examples
911:     --------
912:     >>> from numpy.polynomial.hermite import hermval
913:     >>> coef = [1,2,3]
914:     >>> hermval(1, coef)
915:     11.0
916:     >>> hermval([[1,2],[3,4]], coef)
917:     array([[  11.,   51.],
918:            [ 115.,  203.]])
919: 
920:     '''
921:     c = np.array(c, ndmin=1, copy=0)
922:     if c.dtype.char in '?bBhHiIlLqQpP':
923:         c = c.astype(np.double)
924:     if isinstance(x, (tuple, list)):
925:         x = np.asarray(x)
926:     if isinstance(x, np.ndarray) and tensor:
927:         c = c.reshape(c.shape + (1,)*x.ndim)
928: 
929:     x2 = x*2
930:     if len(c) == 1:
931:         c0 = c[0]
932:         c1 = 0
933:     elif len(c) == 2:
934:         c0 = c[0]
935:         c1 = c[1]
936:     else:
937:         nd = len(c)
938:         c0 = c[-2]
939:         c1 = c[-1]
940:         for i in range(3, len(c) + 1):
941:             tmp = c0
942:             nd = nd - 1
943:             c0 = c[-i] - c1*(2*(nd - 1))
944:             c1 = tmp + c1*x2
945:     return c0 + c1*x2
946: 
947: 
948: def hermval2d(x, y, c):
949:     '''
950:     Evaluate a 2-D Hermite series at points (x, y).
951: 
952:     This function returns the values:
953: 
954:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * H_i(x) * H_j(y)
955: 
956:     The parameters `x` and `y` are converted to arrays only if they are
957:     tuples or a lists, otherwise they are treated as a scalars and they
958:     must have the same shape after conversion. In either case, either `x`
959:     and `y` or their elements must support multiplication and addition both
960:     with themselves and with the elements of `c`.
961: 
962:     If `c` is a 1-D array a one is implicitly appended to its shape to make
963:     it 2-D. The shape of the result will be c.shape[2:] + x.shape.
964: 
965:     Parameters
966:     ----------
967:     x, y : array_like, compatible objects
968:         The two dimensional series is evaluated at the points `(x, y)`,
969:         where `x` and `y` must have the same shape. If `x` or `y` is a list
970:         or tuple, it is first converted to an ndarray, otherwise it is left
971:         unchanged and if it isn't an ndarray it is treated as a scalar.
972:     c : array_like
973:         Array of coefficients ordered so that the coefficient of the term
974:         of multi-degree i,j is contained in ``c[i,j]``. If `c` has
975:         dimension greater than two the remaining indices enumerate multiple
976:         sets of coefficients.
977: 
978:     Returns
979:     -------
980:     values : ndarray, compatible object
981:         The values of the two dimensional polynomial at points formed with
982:         pairs of corresponding values from `x` and `y`.
983: 
984:     See Also
985:     --------
986:     hermval, hermgrid2d, hermval3d, hermgrid3d
987: 
988:     Notes
989:     -----
990: 
991:     .. versionadded::1.7.0
992: 
993:     '''
994:     try:
995:         x, y = np.array((x, y), copy=0)
996:     except:
997:         raise ValueError('x, y are incompatible')
998: 
999:     c = hermval(x, c)
1000:     c = hermval(y, c, tensor=False)
1001:     return c
1002: 
1003: 
1004: def hermgrid2d(x, y, c):
1005:     '''
1006:     Evaluate a 2-D Hermite series on the Cartesian product of x and y.
1007: 
1008:     This function returns the values:
1009: 
1010:     .. math:: p(a,b) = \sum_{i,j} c_{i,j} * H_i(a) * H_j(b)
1011: 
1012:     where the points `(a, b)` consist of all pairs formed by taking
1013:     `a` from `x` and `b` from `y`. The resulting points form a grid with
1014:     `x` in the first dimension and `y` in the second.
1015: 
1016:     The parameters `x` and `y` are converted to arrays only if they are
1017:     tuples or a lists, otherwise they are treated as a scalars. In either
1018:     case, either `x` and `y` or their elements must support multiplication
1019:     and addition both with themselves and with the elements of `c`.
1020: 
1021:     If `c` has fewer than two dimensions, ones are implicitly appended to
1022:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
1023:     x.shape.
1024: 
1025:     Parameters
1026:     ----------
1027:     x, y : array_like, compatible objects
1028:         The two dimensional series is evaluated at the points in the
1029:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
1030:         tuple, it is first converted to an ndarray, otherwise it is left
1031:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
1032:     c : array_like
1033:         Array of coefficients ordered so that the coefficients for terms of
1034:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1035:         greater than two the remaining indices enumerate multiple sets of
1036:         coefficients.
1037: 
1038:     Returns
1039:     -------
1040:     values : ndarray, compatible object
1041:         The values of the two dimensional polynomial at points in the Cartesian
1042:         product of `x` and `y`.
1043: 
1044:     See Also
1045:     --------
1046:     hermval, hermval2d, hermval3d, hermgrid3d
1047: 
1048:     Notes
1049:     -----
1050: 
1051:     .. versionadded::1.7.0
1052: 
1053:     '''
1054:     c = hermval(x, c)
1055:     c = hermval(y, c)
1056:     return c
1057: 
1058: 
1059: def hermval3d(x, y, z, c):
1060:     '''
1061:     Evaluate a 3-D Hermite series at points (x, y, z).
1062: 
1063:     This function returns the values:
1064: 
1065:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * H_i(x) * H_j(y) * H_k(z)
1066: 
1067:     The parameters `x`, `y`, and `z` are converted to arrays only if
1068:     they are tuples or a lists, otherwise they are treated as a scalars and
1069:     they must have the same shape after conversion. In either case, either
1070:     `x`, `y`, and `z` or their elements must support multiplication and
1071:     addition both with themselves and with the elements of `c`.
1072: 
1073:     If `c` has fewer than 3 dimensions, ones are implicitly appended to its
1074:     shape to make it 3-D. The shape of the result will be c.shape[3:] +
1075:     x.shape.
1076: 
1077:     Parameters
1078:     ----------
1079:     x, y, z : array_like, compatible object
1080:         The three dimensional series is evaluated at the points
1081:         `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
1082:         any of `x`, `y`, or `z` is a list or tuple, it is first converted
1083:         to an ndarray, otherwise it is left unchanged and if it isn't an
1084:         ndarray it is  treated as a scalar.
1085:     c : array_like
1086:         Array of coefficients ordered so that the coefficient of the term of
1087:         multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
1088:         greater than 3 the remaining indices enumerate multiple sets of
1089:         coefficients.
1090: 
1091:     Returns
1092:     -------
1093:     values : ndarray, compatible object
1094:         The values of the multidimensional polynomial on points formed with
1095:         triples of corresponding values from `x`, `y`, and `z`.
1096: 
1097:     See Also
1098:     --------
1099:     hermval, hermval2d, hermgrid2d, hermgrid3d
1100: 
1101:     Notes
1102:     -----
1103: 
1104:     .. versionadded::1.7.0
1105: 
1106:     '''
1107:     try:
1108:         x, y, z = np.array((x, y, z), copy=0)
1109:     except:
1110:         raise ValueError('x, y, z are incompatible')
1111: 
1112:     c = hermval(x, c)
1113:     c = hermval(y, c, tensor=False)
1114:     c = hermval(z, c, tensor=False)
1115:     return c
1116: 
1117: 
1118: def hermgrid3d(x, y, z, c):
1119:     '''
1120:     Evaluate a 3-D Hermite series on the Cartesian product of x, y, and z.
1121: 
1122:     This function returns the values:
1123: 
1124:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * H_i(a) * H_j(b) * H_k(c)
1125: 
1126:     where the points `(a, b, c)` consist of all triples formed by taking
1127:     `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
1128:     a grid with `x` in the first dimension, `y` in the second, and `z` in
1129:     the third.
1130: 
1131:     The parameters `x`, `y`, and `z` are converted to arrays only if they
1132:     are tuples or a lists, otherwise they are treated as a scalars. In
1133:     either case, either `x`, `y`, and `z` or their elements must support
1134:     multiplication and addition both with themselves and with the elements
1135:     of `c`.
1136: 
1137:     If `c` has fewer than three dimensions, ones are implicitly appended to
1138:     its shape to make it 3-D. The shape of the result will be c.shape[3:] +
1139:     x.shape + y.shape + z.shape.
1140: 
1141:     Parameters
1142:     ----------
1143:     x, y, z : array_like, compatible objects
1144:         The three dimensional series is evaluated at the points in the
1145:         Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
1146:         list or tuple, it is first converted to an ndarray, otherwise it is
1147:         left unchanged and, if it isn't an ndarray, it is treated as a
1148:         scalar.
1149:     c : array_like
1150:         Array of coefficients ordered so that the coefficients for terms of
1151:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1152:         greater than two the remaining indices enumerate multiple sets of
1153:         coefficients.
1154: 
1155:     Returns
1156:     -------
1157:     values : ndarray, compatible object
1158:         The values of the two dimensional polynomial at points in the Cartesian
1159:         product of `x` and `y`.
1160: 
1161:     See Also
1162:     --------
1163:     hermval, hermval2d, hermgrid2d, hermval3d
1164: 
1165:     Notes
1166:     -----
1167: 
1168:     .. versionadded::1.7.0
1169: 
1170:     '''
1171:     c = hermval(x, c)
1172:     c = hermval(y, c)
1173:     c = hermval(z, c)
1174:     return c
1175: 
1176: 
1177: def hermvander(x, deg):
1178:     '''Pseudo-Vandermonde matrix of given degree.
1179: 
1180:     Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
1181:     `x`. The pseudo-Vandermonde matrix is defined by
1182: 
1183:     .. math:: V[..., i] = H_i(x),
1184: 
1185:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1186:     `x` and the last index is the degree of the Hermite polynomial.
1187: 
1188:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1189:     array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and
1190:     ``hermval(x, c)`` are the same up to roundoff. This equivalence is
1191:     useful both for least squares fitting and for the evaluation of a large
1192:     number of Hermite series of the same degree and sample points.
1193: 
1194:     Parameters
1195:     ----------
1196:     x : array_like
1197:         Array of points. The dtype is converted to float64 or complex128
1198:         depending on whether any of the elements are complex. If `x` is
1199:         scalar it is converted to a 1-D array.
1200:     deg : int
1201:         Degree of the resulting matrix.
1202: 
1203:     Returns
1204:     -------
1205:     vander : ndarray
1206:         The pseudo-Vandermonde matrix. The shape of the returned matrix is
1207:         ``x.shape + (deg + 1,)``, where The last index is the degree of the
1208:         corresponding Hermite polynomial.  The dtype will be the same as
1209:         the converted `x`.
1210: 
1211:     Examples
1212:     --------
1213:     >>> from numpy.polynomial.hermite import hermvander
1214:     >>> x = np.array([-1, 0, 1])
1215:     >>> hermvander(x, 3)
1216:     array([[ 1., -2.,  2.,  4.],
1217:            [ 1.,  0., -2., -0.],
1218:            [ 1.,  2.,  2., -4.]])
1219: 
1220:     '''
1221:     ideg = int(deg)
1222:     if ideg != deg:
1223:         raise ValueError("deg must be integer")
1224:     if ideg < 0:
1225:         raise ValueError("deg must be non-negative")
1226: 
1227:     x = np.array(x, copy=0, ndmin=1) + 0.0
1228:     dims = (ideg + 1,) + x.shape
1229:     dtyp = x.dtype
1230:     v = np.empty(dims, dtype=dtyp)
1231:     v[0] = x*0 + 1
1232:     if ideg > 0:
1233:         x2 = x*2
1234:         v[1] = x2
1235:         for i in range(2, ideg + 1):
1236:             v[i] = (v[i-1]*x2 - v[i-2]*(2*(i - 1)))
1237:     return np.rollaxis(v, 0, v.ndim)
1238: 
1239: 
1240: def hermvander2d(x, y, deg):
1241:     '''Pseudo-Vandermonde matrix of given degrees.
1242: 
1243:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1244:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1245: 
1246:     .. math:: V[..., deg[1]*i + j] = H_i(x) * H_j(y),
1247: 
1248:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1249:     `V` index the points `(x, y)` and the last index encodes the degrees of
1250:     the Hermite polynomials.
1251: 
1252:     If ``V = hermvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1253:     correspond to the elements of a 2-D coefficient array `c` of shape
1254:     (xdeg + 1, ydeg + 1) in the order
1255: 
1256:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1257: 
1258:     and ``np.dot(V, c.flat)`` and ``hermval2d(x, y, c)`` will be the same
1259:     up to roundoff. This equivalence is useful both for least squares
1260:     fitting and for the evaluation of a large number of 2-D Hermite
1261:     series of the same degrees and sample points.
1262: 
1263:     Parameters
1264:     ----------
1265:     x, y : array_like
1266:         Arrays of point coordinates, all of the same shape. The dtypes
1267:         will be converted to either float64 or complex128 depending on
1268:         whether any of the elements are complex. Scalars are converted to 1-D
1269:         arrays.
1270:     deg : list of ints
1271:         List of maximum degrees of the form [x_deg, y_deg].
1272: 
1273:     Returns
1274:     -------
1275:     vander2d : ndarray
1276:         The shape of the returned matrix is ``x.shape + (order,)``, where
1277:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1278:         as the converted `x` and `y`.
1279: 
1280:     See Also
1281:     --------
1282:     hermvander, hermvander3d. hermval2d, hermval3d
1283: 
1284:     Notes
1285:     -----
1286: 
1287:     .. versionadded::1.7.0
1288: 
1289:     '''
1290:     ideg = [int(d) for d in deg]
1291:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1292:     if is_valid != [1, 1]:
1293:         raise ValueError("degrees must be non-negative integers")
1294:     degx, degy = ideg
1295:     x, y = np.array((x, y), copy=0) + 0.0
1296: 
1297:     vx = hermvander(x, degx)
1298:     vy = hermvander(y, degy)
1299:     v = vx[..., None]*vy[..., None,:]
1300:     return v.reshape(v.shape[:-2] + (-1,))
1301: 
1302: 
1303: def hermvander3d(x, y, z, deg):
1304:     '''Pseudo-Vandermonde matrix of given degrees.
1305: 
1306:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1307:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1308:     then The pseudo-Vandermonde matrix is defined by
1309: 
1310:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = H_i(x)*H_j(y)*H_k(z),
1311: 
1312:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1313:     indices of `V` index the points `(x, y, z)` and the last index encodes
1314:     the degrees of the Hermite polynomials.
1315: 
1316:     If ``V = hermvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1317:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1318:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1319: 
1320:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1321: 
1322:     and  ``np.dot(V, c.flat)`` and ``hermval3d(x, y, z, c)`` will be the
1323:     same up to roundoff. This equivalence is useful both for least squares
1324:     fitting and for the evaluation of a large number of 3-D Hermite
1325:     series of the same degrees and sample points.
1326: 
1327:     Parameters
1328:     ----------
1329:     x, y, z : array_like
1330:         Arrays of point coordinates, all of the same shape. The dtypes will
1331:         be converted to either float64 or complex128 depending on whether
1332:         any of the elements are complex. Scalars are converted to 1-D
1333:         arrays.
1334:     deg : list of ints
1335:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1336: 
1337:     Returns
1338:     -------
1339:     vander3d : ndarray
1340:         The shape of the returned matrix is ``x.shape + (order,)``, where
1341:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1342:         be the same as the converted `x`, `y`, and `z`.
1343: 
1344:     See Also
1345:     --------
1346:     hermvander, hermvander3d. hermval2d, hermval3d
1347: 
1348:     Notes
1349:     -----
1350: 
1351:     .. versionadded::1.7.0
1352: 
1353:     '''
1354:     ideg = [int(d) for d in deg]
1355:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1356:     if is_valid != [1, 1, 1]:
1357:         raise ValueError("degrees must be non-negative integers")
1358:     degx, degy, degz = ideg
1359:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1360: 
1361:     vx = hermvander(x, degx)
1362:     vy = hermvander(y, degy)
1363:     vz = hermvander(z, degz)
1364:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1365:     return v.reshape(v.shape[:-3] + (-1,))
1366: 
1367: 
1368: def hermfit(x, y, deg, rcond=None, full=False, w=None):
1369:     '''
1370:     Least squares fit of Hermite series to data.
1371: 
1372:     Return the coefficients of a Hermite series of degree `deg` that is the
1373:     least squares fit to the data values `y` given at points `x`. If `y` is
1374:     1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
1375:     fits are done, one for each column of `y`, and the resulting
1376:     coefficients are stored in the corresponding columns of a 2-D return.
1377:     The fitted polynomial(s) are in the form
1378: 
1379:     .. math::  p(x) = c_0 + c_1 * H_1(x) + ... + c_n * H_n(x),
1380: 
1381:     where `n` is `deg`.
1382: 
1383:     Parameters
1384:     ----------
1385:     x : array_like, shape (M,)
1386:         x-coordinates of the M sample points ``(x[i], y[i])``.
1387:     y : array_like, shape (M,) or (M, K)
1388:         y-coordinates of the sample points. Several data sets of sample
1389:         points sharing the same x-coordinates can be fitted at once by
1390:         passing in a 2D-array that contains one dataset per column.
1391:     deg : int or 1-D array_like
1392:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1393:         all terms up to and including the `deg`'th term are included in the
1394:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1395:         degrees of the terms to include may be used instead.
1396:     rcond : float, optional
1397:         Relative condition number of the fit. Singular values smaller than
1398:         this relative to the largest singular value will be ignored. The
1399:         default value is len(x)*eps, where eps is the relative precision of
1400:         the float type, about 2e-16 in most cases.
1401:     full : bool, optional
1402:         Switch determining nature of return value. When it is False (the
1403:         default) just the coefficients are returned, when True diagnostic
1404:         information from the singular value decomposition is also returned.
1405:     w : array_like, shape (`M`,), optional
1406:         Weights. If not None, the contribution of each point
1407:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1408:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1409:         all have the same variance.  The default value is None.
1410: 
1411:     Returns
1412:     -------
1413:     coef : ndarray, shape (M,) or (M, K)
1414:         Hermite coefficients ordered from low to high. If `y` was 2-D,
1415:         the coefficients for the data in column k  of `y` are in column
1416:         `k`.
1417: 
1418:     [residuals, rank, singular_values, rcond] : list
1419:         These values are only returned if `full` = True
1420: 
1421:         resid -- sum of squared residuals of the least squares fit
1422:         rank -- the numerical rank of the scaled Vandermonde matrix
1423:         sv -- singular values of the scaled Vandermonde matrix
1424:         rcond -- value of `rcond`.
1425: 
1426:         For more details, see `linalg.lstsq`.
1427: 
1428:     Warns
1429:     -----
1430:     RankWarning
1431:         The rank of the coefficient matrix in the least-squares fit is
1432:         deficient. The warning is only raised if `full` = False.  The
1433:         warnings can be turned off by
1434: 
1435:         >>> import warnings
1436:         >>> warnings.simplefilter('ignore', RankWarning)
1437: 
1438:     See Also
1439:     --------
1440:     chebfit, legfit, lagfit, polyfit, hermefit
1441:     hermval : Evaluates a Hermite series.
1442:     hermvander : Vandermonde matrix of Hermite series.
1443:     hermweight : Hermite weight function
1444:     linalg.lstsq : Computes a least-squares fit from the matrix.
1445:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1446: 
1447:     Notes
1448:     -----
1449:     The solution is the coefficients of the Hermite series `p` that
1450:     minimizes the sum of the weighted squared errors
1451: 
1452:     .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1453: 
1454:     where the :math:`w_j` are the weights. This problem is solved by
1455:     setting up the (typically) overdetermined matrix equation
1456: 
1457:     .. math:: V(x) * c = w * y,
1458: 
1459:     where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
1460:     coefficients to be solved for, `w` are the weights, `y` are the
1461:     observed values.  This equation is then solved using the singular value
1462:     decomposition of `V`.
1463: 
1464:     If some of the singular values of `V` are so small that they are
1465:     neglected, then a `RankWarning` will be issued. This means that the
1466:     coefficient values may be poorly determined. Using a lower order fit
1467:     will usually get rid of the warning.  The `rcond` parameter can also be
1468:     set to a value smaller than its default, but the resulting fit may be
1469:     spurious and have large contributions from roundoff error.
1470: 
1471:     Fits using Hermite series are probably most useful when the data can be
1472:     approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Hermite
1473:     weight. In that case the weight ``sqrt(w(x[i])`` should be used
1474:     together with data values ``y[i]/sqrt(w(x[i])``. The weight function is
1475:     available as `hermweight`.
1476: 
1477:     References
1478:     ----------
1479:     .. [1] Wikipedia, "Curve fitting",
1480:            http://en.wikipedia.org/wiki/Curve_fitting
1481: 
1482:     Examples
1483:     --------
1484:     >>> from numpy.polynomial.hermite import hermfit, hermval
1485:     >>> x = np.linspace(-10, 10)
1486:     >>> err = np.random.randn(len(x))/10
1487:     >>> y = hermval(x, [1, 2, 3]) + err
1488:     >>> hermfit(x, y, 2)
1489:     array([ 0.97902637,  1.99849131,  3.00006   ])
1490: 
1491:     '''
1492:     x = np.asarray(x) + 0.0
1493:     y = np.asarray(y) + 0.0
1494:     deg = np.asarray(deg)
1495: 
1496:     # check arguments.
1497:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1498:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1499:     if deg.min() < 0:
1500:         raise ValueError("expected deg >= 0")
1501:     if x.ndim != 1:
1502:         raise TypeError("expected 1D vector for x")
1503:     if x.size == 0:
1504:         raise TypeError("expected non-empty vector for x")
1505:     if y.ndim < 1 or y.ndim > 2:
1506:         raise TypeError("expected 1D or 2D array for y")
1507:     if len(x) != len(y):
1508:         raise TypeError("expected x and y to have same length")
1509: 
1510:     if deg.ndim == 0:
1511:         lmax = deg
1512:         order = lmax + 1
1513:         van = hermvander(x, lmax)
1514:     else:
1515:         deg = np.sort(deg)
1516:         lmax = deg[-1]
1517:         order = len(deg)
1518:         van = hermvander(x, lmax)[:, deg]
1519: 
1520:     # set up the least squares matrices in transposed form
1521:     lhs = van.T
1522:     rhs = y.T
1523:     if w is not None:
1524:         w = np.asarray(w) + 0.0
1525:         if w.ndim != 1:
1526:             raise TypeError("expected 1D vector for w")
1527:         if len(x) != len(w):
1528:             raise TypeError("expected x and w to have same length")
1529:         # apply weights. Don't use inplace operations as they
1530:         # can cause problems with NA.
1531:         lhs = lhs * w
1532:         rhs = rhs * w
1533: 
1534:     # set rcond
1535:     if rcond is None:
1536:         rcond = len(x)*np.finfo(x.dtype).eps
1537: 
1538:     # Determine the norms of the design matrix columns.
1539:     if issubclass(lhs.dtype.type, np.complexfloating):
1540:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1541:     else:
1542:         scl = np.sqrt(np.square(lhs).sum(1))
1543:     scl[scl == 0] = 1
1544: 
1545:     # Solve the least squares problem.
1546:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1547:     c = (c.T/scl).T
1548: 
1549:     # Expand c to include non-fitted coefficients which are set to zero
1550:     if deg.ndim > 0:
1551:         if c.ndim == 2:
1552:             cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
1553:         else:
1554:             cc = np.zeros(lmax+1, dtype=c.dtype)
1555:         cc[deg] = c
1556:         c = cc
1557: 
1558:     # warn on rank reduction
1559:     if rank != order and not full:
1560:         msg = "The fit may be poorly conditioned"
1561:         warnings.warn(msg, pu.RankWarning)
1562: 
1563:     if full:
1564:         return c, [resids, rank, s, rcond]
1565:     else:
1566:         return c
1567: 
1568: 
1569: def hermcompanion(c):
1570:     '''Return the scaled companion matrix of c.
1571: 
1572:     The basis polynomials are scaled so that the companion matrix is
1573:     symmetric when `c` is an Hermite basis polynomial. This provides
1574:     better eigenvalue estimates than the unscaled case and for basis
1575:     polynomials the eigenvalues are guaranteed to be real if
1576:     `numpy.linalg.eigvalsh` is used to obtain them.
1577: 
1578:     Parameters
1579:     ----------
1580:     c : array_like
1581:         1-D array of Hermite series coefficients ordered from low to high
1582:         degree.
1583: 
1584:     Returns
1585:     -------
1586:     mat : ndarray
1587:         Scaled companion matrix of dimensions (deg, deg).
1588: 
1589:     Notes
1590:     -----
1591: 
1592:     .. versionadded::1.7.0
1593: 
1594:     '''
1595:     # c is a trimmed copy
1596:     [c] = pu.as_series([c])
1597:     if len(c) < 2:
1598:         raise ValueError('Series must have maximum degree of at least 1.')
1599:     if len(c) == 2:
1600:         return np.array([[-.5*c[0]/c[1]]])
1601: 
1602:     n = len(c) - 1
1603:     mat = np.zeros((n, n), dtype=c.dtype)
1604:     scl = np.hstack((1., 1./np.sqrt(2.*np.arange(n - 1, 0, -1))))
1605:     scl = np.multiply.accumulate(scl)[::-1]
1606:     top = mat.reshape(-1)[1::n+1]
1607:     bot = mat.reshape(-1)[n::n+1]
1608:     top[...] = np.sqrt(.5*np.arange(1, n))
1609:     bot[...] = top
1610:     mat[:, -1] -= scl*c[:-1]/(2.0*c[-1])
1611:     return mat
1612: 
1613: 
1614: def hermroots(c):
1615:     '''
1616:     Compute the roots of a Hermite series.
1617: 
1618:     Return the roots (a.k.a. "zeros") of the polynomial
1619: 
1620:     .. math:: p(x) = \\sum_i c[i] * H_i(x).
1621: 
1622:     Parameters
1623:     ----------
1624:     c : 1-D array_like
1625:         1-D array of coefficients.
1626: 
1627:     Returns
1628:     -------
1629:     out : ndarray
1630:         Array of the roots of the series. If all the roots are real,
1631:         then `out` is also real, otherwise it is complex.
1632: 
1633:     See Also
1634:     --------
1635:     polyroots, legroots, lagroots, chebroots, hermeroots
1636: 
1637:     Notes
1638:     -----
1639:     The root estimates are obtained as the eigenvalues of the companion
1640:     matrix, Roots far from the origin of the complex plane may have large
1641:     errors due to the numerical instability of the series for such
1642:     values. Roots with multiplicity greater than 1 will also show larger
1643:     errors as the value of the series near such points is relatively
1644:     insensitive to errors in the roots. Isolated roots near the origin can
1645:     be improved by a few iterations of Newton's method.
1646: 
1647:     The Hermite series basis polynomials aren't powers of `x` so the
1648:     results of this function may seem unintuitive.
1649: 
1650:     Examples
1651:     --------
1652:     >>> from numpy.polynomial.hermite import hermroots, hermfromroots
1653:     >>> coef = hermfromroots([-1, 0, 1])
1654:     >>> coef
1655:     array([ 0.   ,  0.25 ,  0.   ,  0.125])
1656:     >>> hermroots(coef)
1657:     array([ -1.00000000e+00,  -1.38777878e-17,   1.00000000e+00])
1658: 
1659:     '''
1660:     # c is a trimmed copy
1661:     [c] = pu.as_series([c])
1662:     if len(c) <= 1:
1663:         return np.array([], dtype=c.dtype)
1664:     if len(c) == 2:
1665:         return np.array([-.5*c[0]/c[1]])
1666: 
1667:     m = hermcompanion(c)
1668:     r = la.eigvals(m)
1669:     r.sort()
1670:     return r
1671: 
1672: 
1673: def _normed_hermite_n(x, n):
1674:     '''
1675:     Evaluate a normalized Hermite polynomial.
1676: 
1677:     Compute the value of the normalized Hermite polynomial of degree ``n``
1678:     at the points ``x``.
1679: 
1680: 
1681:     Parameters
1682:     ----------
1683:     x : ndarray of double.
1684:         Points at which to evaluate the function
1685:     n : int
1686:         Degree of the normalized Hermite function to be evaluated.
1687: 
1688:     Returns
1689:     -------
1690:     values : ndarray
1691:         The shape of the return value is described above.
1692: 
1693:     Notes
1694:     -----
1695:     .. versionadded:: 1.10.0
1696: 
1697:     This function is needed for finding the Gauss points and integration
1698:     weights for high degrees. The values of the standard Hermite functions
1699:     overflow when n >= 207.
1700: 
1701:     '''
1702:     if n == 0:
1703:         return np.ones(x.shape)/np.sqrt(np.sqrt(np.pi))
1704: 
1705:     c0 = 0.
1706:     c1 = 1./np.sqrt(np.sqrt(np.pi))
1707:     nd = float(n)
1708:     for i in range(n - 1):
1709:         tmp = c0
1710:         c0 = -c1*np.sqrt((nd - 1.)/nd)
1711:         c1 = tmp + c1*x*np.sqrt(2./nd)
1712:         nd = nd - 1.0
1713:     return c0 + c1*x*np.sqrt(2)
1714: 
1715: 
1716: def hermgauss(deg):
1717:     '''
1718:     Gauss-Hermite quadrature.
1719: 
1720:     Computes the sample points and weights for Gauss-Hermite quadrature.
1721:     These sample points and weights will correctly integrate polynomials of
1722:     degree :math:`2*deg - 1` or less over the interval :math:`[-\inf, \inf]`
1723:     with the weight function :math:`f(x) = \exp(-x^2)`.
1724: 
1725:     Parameters
1726:     ----------
1727:     deg : int
1728:         Number of sample points and weights. It must be >= 1.
1729: 
1730:     Returns
1731:     -------
1732:     x : ndarray
1733:         1-D ndarray containing the sample points.
1734:     y : ndarray
1735:         1-D ndarray containing the weights.
1736: 
1737:     Notes
1738:     -----
1739: 
1740:     .. versionadded::1.7.0
1741: 
1742:     The results have only been tested up to degree 100, higher degrees may
1743:     be problematic. The weights are determined by using the fact that
1744: 
1745:     .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))
1746: 
1747:     where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
1748:     is the k'th root of :math:`H_n`, and then scaling the results to get
1749:     the right value when integrating 1.
1750: 
1751:     '''
1752:     ideg = int(deg)
1753:     if ideg != deg or ideg < 1:
1754:         raise ValueError("deg must be a non-negative integer")
1755: 
1756:     # first approximation of roots. We use the fact that the companion
1757:     # matrix is symmetric in this case in order to obtain better zeros.
1758:     c = np.array([0]*deg + [1], dtype=np.float64)
1759:     m = hermcompanion(c)
1760:     x = la.eigvalsh(m)
1761: 
1762:     # improve roots by one application of Newton
1763:     dy = _normed_hermite_n(x, ideg)
1764:     df = _normed_hermite_n(x, ideg - 1) * np.sqrt(2*ideg)
1765:     x -= dy/df
1766: 
1767:     # compute the weights. We scale the factor to avoid possible numerical
1768:     # overflow.
1769:     fm = _normed_hermite_n(x, ideg - 1)
1770:     fm /= np.abs(fm).max()
1771:     w = 1/(fm * fm)
1772: 
1773:     # for Hermite we can also symmetrize
1774:     w = (w + w[::-1])/2
1775:     x = (x - x[::-1])/2
1776: 
1777:     # scale w to get the right value
1778:     w *= np.sqrt(np.pi) / w.sum()
1779: 
1780:     return x, w
1781: 
1782: 
1783: def hermweight(x):
1784:     '''
1785:     Weight function of the Hermite polynomials.
1786: 
1787:     The weight function is :math:`\exp(-x^2)` and the interval of
1788:     integration is :math:`[-\inf, \inf]`. the Hermite polynomials are
1789:     orthogonal, but not normalized, with respect to this weight function.
1790: 
1791:     Parameters
1792:     ----------
1793:     x : array_like
1794:        Values at which the weight function will be computed.
1795: 
1796:     Returns
1797:     -------
1798:     w : ndarray
1799:        The weight function at `x`.
1800: 
1801:     Notes
1802:     -----
1803: 
1804:     .. versionadded::1.7.0
1805: 
1806:     '''
1807:     w = np.exp(-x**2)
1808:     return w
1809: 
1810: 
1811: #
1812: # Hermite series class
1813: #
1814: 
1815: class Hermite(ABCPolyBase):
1816:     '''An Hermite series class.
1817: 
1818:     The Hermite class provides the standard Python numerical methods
1819:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
1820:     attributes and methods listed in the `ABCPolyBase` documentation.
1821: 
1822:     Parameters
1823:     ----------
1824:     coef : array_like
1825:         Hermite coefficients in order of increasing degree, i.e,
1826:         ``(1, 2, 3)`` gives ``1*H_0(x) + 2*H_1(X) + 3*H_2(x)``.
1827:     domain : (2,) array_like, optional
1828:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
1829:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
1830:         The default value is [-1, 1].
1831:     window : (2,) array_like, optional
1832:         Window, see `domain` for its use. The default value is [-1, 1].
1833: 
1834:         .. versionadded:: 1.6.0
1835: 
1836:     '''
1837:     # Virtual Functions
1838:     _add = staticmethod(hermadd)
1839:     _sub = staticmethod(hermsub)
1840:     _mul = staticmethod(hermmul)
1841:     _div = staticmethod(hermdiv)
1842:     _pow = staticmethod(hermpow)
1843:     _val = staticmethod(hermval)
1844:     _int = staticmethod(hermint)
1845:     _der = staticmethod(hermder)
1846:     _fit = staticmethod(hermfit)
1847:     _line = staticmethod(hermline)
1848:     _roots = staticmethod(hermroots)
1849:     _fromroots = staticmethod(hermfromroots)
1850: 
1851:     # Virtual properties
1852:     nickname = 'herm'
1853:     domain = np.array(hermdomain)
1854:     window = np.array(hermdomain)
1855: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_165060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\nObjects for dealing with Hermite series.\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with Hermite series, including a `Hermite` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with such polynomials is in the\ndocstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n- `hermdomain` -- Hermite series default domain, [-1,1].\n- `hermzero` -- Hermite series that evaluates identically to 0.\n- `hermone` -- Hermite series that evaluates identically to 1.\n- `hermx` -- Hermite series for the identity map, ``f(x) = x``.\n\nArithmetic\n----------\n- `hermmulx` -- multiply a Hermite series in ``P_i(x)`` by ``x``.\n- `hermadd` -- add two Hermite series.\n- `hermsub` -- subtract one Hermite series from another.\n- `hermmul` -- multiply two Hermite series.\n- `hermdiv` -- divide one Hermite series by another.\n- `hermval` -- evaluate a Hermite series at given points.\n- `hermval2d` -- evaluate a 2D Hermite series at given points.\n- `hermval3d` -- evaluate a 3D Hermite series at given points.\n- `hermgrid2d` -- evaluate a 2D Hermite series on a Cartesian product.\n- `hermgrid3d` -- evaluate a 3D Hermite series on a Cartesian product.\n\nCalculus\n--------\n- `hermder` -- differentiate a Hermite series.\n- `hermint` -- integrate a Hermite series.\n\nMisc Functions\n--------------\n- `hermfromroots` -- create a Hermite series with specified roots.\n- `hermroots` -- find the roots of a Hermite series.\n- `hermvander` -- Vandermonde-like matrix for Hermite polynomials.\n- `hermvander2d` -- Vandermonde-like matrix for 2D power series.\n- `hermvander3d` -- Vandermonde-like matrix for 3D power series.\n- `hermgauss` -- Gauss-Hermite quadrature, points and weights.\n- `hermweight` -- Hermite weight function.\n- `hermcompanion` -- symmetrized companion matrix in Hermite form.\n- `hermfit` -- least-squares fit returning a Hermite series.\n- `hermtrim` -- trim leading coefficients from a Hermite series.\n- `hermline` -- Hermite series of given straight line.\n- `herm2poly` -- convert a Hermite series to a polynomial.\n- `poly2herm` -- convert a polynomial to a Hermite series.\n\nClasses\n-------\n- `Hermite` -- A Hermite series class.\n\nSee also\n--------\n`numpy.polynomial`\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'import warnings' statement (line 62)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 0))

# 'import numpy' statement (line 63)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_165061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy')

if (type(import_165061) is not StypyTypeError):

    if (import_165061 != 'pyd_module'):
        __import__(import_165061)
        sys_modules_165062 = sys.modules[import_165061]
        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', sys_modules_165062.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy', import_165061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 0))

# 'import numpy.linalg' statement (line 64)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_165063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg')

if (type(import_165063) is not StypyTypeError):

    if (import_165063 != 'pyd_module'):
        __import__(import_165063)
        sys_modules_165064 = sys.modules[import_165063]
        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', sys_modules_165064.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg', import_165063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'from numpy.polynomial import pu' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_165065 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial')

if (type(import_165065) is not StypyTypeError):

    if (import_165065 != 'pyd_module'):
        __import__(import_165065)
        sys_modules_165066 = sys.modules[import_165065]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', sys_modules_165066.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 0), __file__, sys_modules_165066, sys_modules_165066.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', import_165065)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 67, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 67)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_165067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase')

if (type(import_165067) is not StypyTypeError):

    if (import_165067 != 'pyd_module'):
        __import__(import_165067)
        sys_modules_165068 = sys.modules[import_165067]
        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', sys_modules_165068.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 67, 0), __file__, sys_modules_165068, sys_modules_165068.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', import_165067)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 69):

# Assigning a List to a Name (line 69):
__all__ = ['hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline', 'hermadd', 'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow', 'hermval', 'hermder', 'hermint', 'herm2poly', 'poly2herm', 'hermfromroots', 'hermvander', 'hermfit', 'hermtrim', 'hermroots', 'Hermite', 'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d', 'hermvander2d', 'hermvander3d', 'hermcompanion', 'hermgauss', 'hermweight']
module_type_store.set_exportable_members(['hermzero', 'hermone', 'hermx', 'hermdomain', 'hermline', 'hermadd', 'hermsub', 'hermmulx', 'hermmul', 'hermdiv', 'hermpow', 'hermval', 'hermder', 'hermint', 'herm2poly', 'poly2herm', 'hermfromroots', 'hermvander', 'hermfit', 'hermtrim', 'hermroots', 'Hermite', 'hermval2d', 'hermval3d', 'hermgrid2d', 'hermgrid3d', 'hermvander2d', 'hermvander3d', 'hermcompanion', 'hermgauss', 'hermweight'])

# Obtaining an instance of the builtin type 'list' (line 69)
list_165069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
str_165070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'hermzero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165070)
# Adding element type (line 69)
str_165071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 16), 'str', 'hermone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165071)
# Adding element type (line 69)
str_165072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 27), 'str', 'hermx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165072)
# Adding element type (line 69)
str_165073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'str', 'hermdomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165073)
# Adding element type (line 69)
str_165074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 50), 'str', 'hermline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165074)
# Adding element type (line 69)
str_165075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 62), 'str', 'hermadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165075)
# Adding element type (line 69)
str_165076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'hermsub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165076)
# Adding element type (line 69)
str_165077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', 'hermmulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165077)
# Adding element type (line 69)
str_165078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'str', 'hermmul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165078)
# Adding element type (line 69)
str_165079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'str', 'hermdiv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165079)
# Adding element type (line 69)
str_165080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 49), 'str', 'hermpow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165080)
# Adding element type (line 69)
str_165081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 60), 'str', 'hermval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165081)
# Adding element type (line 69)
str_165082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'hermder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165082)
# Adding element type (line 69)
str_165083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'str', 'hermint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165083)
# Adding element type (line 69)
str_165084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 26), 'str', 'herm2poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165084)
# Adding element type (line 69)
str_165085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'str', 'poly2herm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165085)
# Adding element type (line 69)
str_165086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'str', 'hermfromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165086)
# Adding element type (line 69)
str_165087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'hermvander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165087)
# Adding element type (line 69)
str_165088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'str', 'hermfit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165088)
# Adding element type (line 69)
str_165089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'str', 'hermtrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165089)
# Adding element type (line 69)
str_165090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 41), 'str', 'hermroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165090)
# Adding element type (line 69)
str_165091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 54), 'str', 'Hermite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165091)
# Adding element type (line 69)
str_165092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'hermval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165092)
# Adding element type (line 69)
str_165093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 17), 'str', 'hermval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165093)
# Adding element type (line 69)
str_165094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', 'hermgrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165094)
# Adding element type (line 69)
str_165095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', 'hermgrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165095)
# Adding element type (line 69)
str_165096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 58), 'str', 'hermvander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165096)
# Adding element type (line 69)
str_165097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'hermvander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165097)
# Adding element type (line 69)
str_165098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 20), 'str', 'hermcompanion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165098)
# Adding element type (line 69)
str_165099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'str', 'hermgauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165099)
# Adding element type (line 69)
str_165100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'str', 'hermweight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_165069, str_165100)

# Assigning a type to the variable '__all__' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '__all__', list_165069)

# Assigning a Attribute to a Name (line 77):

# Assigning a Attribute to a Name (line 77):
# Getting the type of 'pu' (line 77)
pu_165101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'pu')
# Obtaining the member 'trimcoef' of a type (line 77)
trimcoef_165102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), pu_165101, 'trimcoef')
# Assigning a type to the variable 'hermtrim' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'hermtrim', trimcoef_165102)

@norecursion
def poly2herm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly2herm'
    module_type_store = module_type_store.open_function_context('poly2herm', 80, 0, False)
    
    # Passed parameters checking function
    poly2herm.stypy_localization = localization
    poly2herm.stypy_type_of_self = None
    poly2herm.stypy_type_store = module_type_store
    poly2herm.stypy_function_name = 'poly2herm'
    poly2herm.stypy_param_names_list = ['pol']
    poly2herm.stypy_varargs_param_name = None
    poly2herm.stypy_kwargs_param_name = None
    poly2herm.stypy_call_defaults = defaults
    poly2herm.stypy_call_varargs = varargs
    poly2herm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly2herm', ['pol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly2herm', localization, ['pol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly2herm(...)' code ##################

    str_165103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    poly2herm(pol)\n\n    Convert a polynomial to a Hermite series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Hermite series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Hermite\n        series.\n\n    See Also\n    --------\n    herm2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import poly2herm\n    >>> poly2herm(np.arange(4))\n    array([ 1.   ,  2.75 ,  0.5  ,  0.375])\n\n    ')
    
    # Assigning a Call to a List (line 118):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_165106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    # Adding element type (line 118)
    # Getting the type of 'pol' (line 118)
    pol_165107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'pol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 25), list_165106, pol_165107)
    
    # Processing the call keyword arguments (line 118)
    kwargs_165108 = {}
    # Getting the type of 'pu' (line 118)
    pu_165104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 118)
    as_series_165105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), pu_165104, 'as_series')
    # Calling as_series(args, kwargs) (line 118)
    as_series_call_result_165109 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), as_series_165105, *[list_165106], **kwargs_165108)
    
    # Assigning a type to the variable 'call_assignment_165005' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_165005', as_series_call_result_165109)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165113 = {}
    # Getting the type of 'call_assignment_165005' (line 118)
    call_assignment_165005_165110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_165005', False)
    # Obtaining the member '__getitem__' of a type (line 118)
    getitem___165111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), call_assignment_165005_165110, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165114 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165111, *[int_165112], **kwargs_165113)
    
    # Assigning a type to the variable 'call_assignment_165006' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_165006', getitem___call_result_165114)
    
    # Assigning a Name to a Name (line 118):
    # Getting the type of 'call_assignment_165006' (line 118)
    call_assignment_165006_165115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'call_assignment_165006')
    # Assigning a type to the variable 'pol' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 5), 'pol', call_assignment_165006_165115)
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    
    # Call to len(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'pol' (line 119)
    pol_165117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'pol', False)
    # Processing the call keyword arguments (line 119)
    kwargs_165118 = {}
    # Getting the type of 'len' (line 119)
    len_165116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 10), 'len', False)
    # Calling len(args, kwargs) (line 119)
    len_call_result_165119 = invoke(stypy.reporting.localization.Localization(__file__, 119, 10), len_165116, *[pol_165117], **kwargs_165118)
    
    int_165120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
    # Applying the binary operator '-' (line 119)
    result_sub_165121 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 10), '-', len_call_result_165119, int_165120)
    
    # Assigning a type to the variable 'deg' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'deg', result_sub_165121)
    
    # Assigning a Num to a Name (line 120):
    
    # Assigning a Num to a Name (line 120):
    int_165122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'int')
    # Assigning a type to the variable 'res' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'res', int_165122)
    
    
    # Call to range(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'deg' (line 121)
    deg_165124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'deg', False)
    int_165125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 24), 'int')
    int_165126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'int')
    # Processing the call keyword arguments (line 121)
    kwargs_165127 = {}
    # Getting the type of 'range' (line 121)
    range_165123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 13), 'range', False)
    # Calling range(args, kwargs) (line 121)
    range_call_result_165128 = invoke(stypy.reporting.localization.Localization(__file__, 121, 13), range_165123, *[deg_165124, int_165125, int_165126], **kwargs_165127)
    
    # Testing the type of a for loop iterable (line 121)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 121, 4), range_call_result_165128)
    # Getting the type of the for loop variable (line 121)
    for_loop_var_165129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 121, 4), range_call_result_165128)
    # Assigning a type to the variable 'i' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'i', for_loop_var_165129)
    # SSA begins for a for statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to hermadd(...): (line 122)
    # Processing the call arguments (line 122)
    
    # Call to hermmulx(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'res' (line 122)
    res_165132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'res', False)
    # Processing the call keyword arguments (line 122)
    kwargs_165133 = {}
    # Getting the type of 'hermmulx' (line 122)
    hermmulx_165131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'hermmulx', False)
    # Calling hermmulx(args, kwargs) (line 122)
    hermmulx_call_result_165134 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), hermmulx_165131, *[res_165132], **kwargs_165133)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 122)
    i_165135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 41), 'i', False)
    # Getting the type of 'pol' (line 122)
    pol_165136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'pol', False)
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___165137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 37), pol_165136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_165138 = invoke(stypy.reporting.localization.Localization(__file__, 122, 37), getitem___165137, i_165135)
    
    # Processing the call keyword arguments (line 122)
    kwargs_165139 = {}
    # Getting the type of 'hermadd' (line 122)
    hermadd_165130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'hermadd', False)
    # Calling hermadd(args, kwargs) (line 122)
    hermadd_call_result_165140 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), hermadd_165130, *[hermmulx_call_result_165134, subscript_call_result_165138], **kwargs_165139)
    
    # Assigning a type to the variable 'res' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'res', hermadd_call_result_165140)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 123)
    res_165141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', res_165141)
    
    # ################# End of 'poly2herm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly2herm' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_165142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165142)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly2herm'
    return stypy_return_type_165142

# Assigning a type to the variable 'poly2herm' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'poly2herm', poly2herm)

@norecursion
def herm2poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'herm2poly'
    module_type_store = module_type_store.open_function_context('herm2poly', 126, 0, False)
    
    # Passed parameters checking function
    herm2poly.stypy_localization = localization
    herm2poly.stypy_type_of_self = None
    herm2poly.stypy_type_store = module_type_store
    herm2poly.stypy_function_name = 'herm2poly'
    herm2poly.stypy_param_names_list = ['c']
    herm2poly.stypy_varargs_param_name = None
    herm2poly.stypy_kwargs_param_name = None
    herm2poly.stypy_call_defaults = defaults
    herm2poly.stypy_call_varargs = varargs
    herm2poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'herm2poly', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'herm2poly', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'herm2poly(...)' code ##################

    str_165143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Convert a Hermite series to a polynomial.\n\n    Convert an array representing the coefficients of a Hermite series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Hermite series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2herm\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import herm2poly\n    >>> herm2poly([ 1.   ,  2.75 ,  0.5  ,  0.375])\n    array([ 0.,  1.,  2.,  3.])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 164, 4))
    
    # 'from numpy.polynomial.polynomial import polyadd, polysub, polymulx' statement (line 164)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_165144 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial')

    if (type(import_165144) is not StypyTypeError):

        if (import_165144 != 'pyd_module'):
            __import__(import_165144)
            sys_modules_165145 = sys.modules[import_165144]
            import_from_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', sys_modules_165145.module_type_store, module_type_store, ['polyadd', 'polysub', 'polymulx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 164, 4), __file__, sys_modules_165145, sys_modules_165145.module_type_store, module_type_store)
        else:
            from numpy.polynomial.polynomial import polyadd, polysub, polymulx

            import_from_module(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', None, module_type_store, ['polyadd', 'polysub', 'polymulx'], [polyadd, polysub, polymulx])

    else:
        # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'numpy.polynomial.polynomial', import_165144)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a List (line 166):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 166)
    # Processing the call arguments (line 166)
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_165148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    # Getting the type of 'c' (line 166)
    c_165149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 23), list_165148, c_165149)
    
    # Processing the call keyword arguments (line 166)
    kwargs_165150 = {}
    # Getting the type of 'pu' (line 166)
    pu_165146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 166)
    as_series_165147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 10), pu_165146, 'as_series')
    # Calling as_series(args, kwargs) (line 166)
    as_series_call_result_165151 = invoke(stypy.reporting.localization.Localization(__file__, 166, 10), as_series_165147, *[list_165148], **kwargs_165150)
    
    # Assigning a type to the variable 'call_assignment_165007' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_165007', as_series_call_result_165151)
    
    # Assigning a Call to a Name (line 166):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165155 = {}
    # Getting the type of 'call_assignment_165007' (line 166)
    call_assignment_165007_165152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_165007', False)
    # Obtaining the member '__getitem__' of a type (line 166)
    getitem___165153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 4), call_assignment_165007_165152, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165156 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165153, *[int_165154], **kwargs_165155)
    
    # Assigning a type to the variable 'call_assignment_165008' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_165008', getitem___call_result_165156)
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'call_assignment_165008' (line 166)
    call_assignment_165008_165157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'call_assignment_165008')
    # Assigning a type to the variable 'c' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 5), 'c', call_assignment_165008_165157)
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'c' (line 167)
    c_165159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'c', False)
    # Processing the call keyword arguments (line 167)
    kwargs_165160 = {}
    # Getting the type of 'len' (line 167)
    len_165158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_165161 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), len_165158, *[c_165159], **kwargs_165160)
    
    # Assigning a type to the variable 'n' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'n', len_call_result_165161)
    
    
    # Getting the type of 'n' (line 168)
    n_165162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 7), 'n')
    int_165163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 12), 'int')
    # Applying the binary operator '==' (line 168)
    result_eq_165164 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 7), '==', n_165162, int_165163)
    
    # Testing the type of an if condition (line 168)
    if_condition_165165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 4), result_eq_165164)
    # Assigning a type to the variable 'if_condition_165165' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'if_condition_165165', if_condition_165165)
    # SSA begins for if statement (line 168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 169)
    c_165166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', c_165166)
    # SSA join for if statement (line 168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 170)
    n_165167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'n')
    int_165168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 12), 'int')
    # Applying the binary operator '==' (line 170)
    result_eq_165169 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 7), '==', n_165167, int_165168)
    
    # Testing the type of an if condition (line 170)
    if_condition_165170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), result_eq_165169)
    # Assigning a type to the variable 'if_condition_165170' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_165170', if_condition_165170)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 171)
    c_165171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'c')
    
    # Obtaining the type of the subscript
    int_165172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 10), 'int')
    # Getting the type of 'c' (line 171)
    c_165173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'c')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___165174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), c_165173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_165175 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), getitem___165174, int_165172)
    
    int_165176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 16), 'int')
    # Applying the binary operator '*=' (line 171)
    result_imul_165177 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 8), '*=', subscript_call_result_165175, int_165176)
    # Getting the type of 'c' (line 171)
    c_165178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'c')
    int_165179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 10), 'int')
    # Storing an element on a container (line 171)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 8), c_165178, (int_165179, result_imul_165177))
    
    # Getting the type of 'c' (line 172)
    c_165180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', c_165180)
    # SSA branch for the else part of an if statement (line 170)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 174):
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_165181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'int')
    # Getting the type of 'c' (line 174)
    c_165182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___165183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), c_165182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_165184 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), getitem___165183, int_165181)
    
    # Assigning a type to the variable 'c0' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'c0', subscript_call_result_165184)
    
    # Assigning a Subscript to a Name (line 175):
    
    # Assigning a Subscript to a Name (line 175):
    
    # Obtaining the type of the subscript
    int_165185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 15), 'int')
    # Getting the type of 'c' (line 175)
    c_165186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___165187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), c_165186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_165188 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), getitem___165187, int_165185)
    
    # Assigning a type to the variable 'c1' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'c1', subscript_call_result_165188)
    
    
    # Call to range(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'n' (line 177)
    n_165190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'n', False)
    int_165191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'int')
    # Applying the binary operator '-' (line 177)
    result_sub_165192 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 23), '-', n_165190, int_165191)
    
    int_165193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'int')
    int_165194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 33), 'int')
    # Processing the call keyword arguments (line 177)
    kwargs_165195 = {}
    # Getting the type of 'range' (line 177)
    range_165189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'range', False)
    # Calling range(args, kwargs) (line 177)
    range_call_result_165196 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), range_165189, *[result_sub_165192, int_165193, int_165194], **kwargs_165195)
    
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_165196)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_165197 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_165196)
    # Assigning a type to the variable 'i' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'i', for_loop_var_165197)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 178):
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'c0' (line 178)
    c0_165198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'tmp', c0_165198)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to polysub(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 179)
    i_165200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'i', False)
    int_165201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'int')
    # Applying the binary operator '-' (line 179)
    result_sub_165202 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 27), '-', i_165200, int_165201)
    
    # Getting the type of 'c' (line 179)
    c_165203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___165204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 25), c_165203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_165205 = invoke(stypy.reporting.localization.Localization(__file__, 179, 25), getitem___165204, result_sub_165202)
    
    # Getting the type of 'c1' (line 179)
    c1_165206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 35), 'c1', False)
    int_165207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'int')
    # Getting the type of 'i' (line 179)
    i_165208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 42), 'i', False)
    int_165209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 46), 'int')
    # Applying the binary operator '-' (line 179)
    result_sub_165210 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 42), '-', i_165208, int_165209)
    
    # Applying the binary operator '*' (line 179)
    result_mul_165211 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 39), '*', int_165207, result_sub_165210)
    
    # Applying the binary operator '*' (line 179)
    result_mul_165212 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 35), '*', c1_165206, result_mul_165211)
    
    # Processing the call keyword arguments (line 179)
    kwargs_165213 = {}
    # Getting the type of 'polysub' (line 179)
    polysub_165199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'polysub', False)
    # Calling polysub(args, kwargs) (line 179)
    polysub_call_result_165214 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), polysub_165199, *[subscript_call_result_165205, result_mul_165212], **kwargs_165213)
    
    # Assigning a type to the variable 'c0' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'c0', polysub_call_result_165214)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to polyadd(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'tmp' (line 180)
    tmp_165216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'tmp', False)
    
    # Call to polymulx(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'c1' (line 180)
    c1_165218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'c1', False)
    # Processing the call keyword arguments (line 180)
    kwargs_165219 = {}
    # Getting the type of 'polymulx' (line 180)
    polymulx_165217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 180)
    polymulx_call_result_165220 = invoke(stypy.reporting.localization.Localization(__file__, 180, 30), polymulx_165217, *[c1_165218], **kwargs_165219)
    
    int_165221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 43), 'int')
    # Applying the binary operator '*' (line 180)
    result_mul_165222 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 30), '*', polymulx_call_result_165220, int_165221)
    
    # Processing the call keyword arguments (line 180)
    kwargs_165223 = {}
    # Getting the type of 'polyadd' (line 180)
    polyadd_165215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 180)
    polyadd_call_result_165224 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), polyadd_165215, *[tmp_165216, result_mul_165222], **kwargs_165223)
    
    # Assigning a type to the variable 'c1' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'c1', polyadd_call_result_165224)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to polyadd(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c0' (line 181)
    c0_165226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'c0', False)
    
    # Call to polymulx(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c1' (line 181)
    c1_165228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c1', False)
    # Processing the call keyword arguments (line 181)
    kwargs_165229 = {}
    # Getting the type of 'polymulx' (line 181)
    polymulx_165227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 27), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 181)
    polymulx_call_result_165230 = invoke(stypy.reporting.localization.Localization(__file__, 181, 27), polymulx_165227, *[c1_165228], **kwargs_165229)
    
    int_165231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'int')
    # Applying the binary operator '*' (line 181)
    result_mul_165232 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 27), '*', polymulx_call_result_165230, int_165231)
    
    # Processing the call keyword arguments (line 181)
    kwargs_165233 = {}
    # Getting the type of 'polyadd' (line 181)
    polyadd_165225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 181)
    polyadd_call_result_165234 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), polyadd_165225, *[c0_165226, result_mul_165232], **kwargs_165233)
    
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', polyadd_call_result_165234)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'herm2poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'herm2poly' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_165235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165235)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'herm2poly'
    return stypy_return_type_165235

# Assigning a type to the variable 'herm2poly' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'herm2poly', herm2poly)

# Assigning a Call to a Name (line 189):

# Assigning a Call to a Name (line 189):

# Call to array(...): (line 189)
# Processing the call arguments (line 189)

# Obtaining an instance of the builtin type 'list' (line 189)
list_165238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 189)
# Adding element type (line 189)
int_165239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 22), list_165238, int_165239)
# Adding element type (line 189)
int_165240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 22), list_165238, int_165240)

# Processing the call keyword arguments (line 189)
kwargs_165241 = {}
# Getting the type of 'np' (line 189)
np_165236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), 'np', False)
# Obtaining the member 'array' of a type (line 189)
array_165237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 13), np_165236, 'array')
# Calling array(args, kwargs) (line 189)
array_call_result_165242 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), array_165237, *[list_165238], **kwargs_165241)

# Assigning a type to the variable 'hermdomain' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'hermdomain', array_call_result_165242)

# Assigning a Call to a Name (line 192):

# Assigning a Call to a Name (line 192):

# Call to array(...): (line 192)
# Processing the call arguments (line 192)

# Obtaining an instance of the builtin type 'list' (line 192)
list_165245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 192)
# Adding element type (line 192)
int_165246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 20), list_165245, int_165246)

# Processing the call keyword arguments (line 192)
kwargs_165247 = {}
# Getting the type of 'np' (line 192)
np_165243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'np', False)
# Obtaining the member 'array' of a type (line 192)
array_165244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 11), np_165243, 'array')
# Calling array(args, kwargs) (line 192)
array_call_result_165248 = invoke(stypy.reporting.localization.Localization(__file__, 192, 11), array_165244, *[list_165245], **kwargs_165247)

# Assigning a type to the variable 'hermzero' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'hermzero', array_call_result_165248)

# Assigning a Call to a Name (line 195):

# Assigning a Call to a Name (line 195):

# Call to array(...): (line 195)
# Processing the call arguments (line 195)

# Obtaining an instance of the builtin type 'list' (line 195)
list_165251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
int_165252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_165251, int_165252)

# Processing the call keyword arguments (line 195)
kwargs_165253 = {}
# Getting the type of 'np' (line 195)
np_165249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 10), 'np', False)
# Obtaining the member 'array' of a type (line 195)
array_165250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 10), np_165249, 'array')
# Calling array(args, kwargs) (line 195)
array_call_result_165254 = invoke(stypy.reporting.localization.Localization(__file__, 195, 10), array_165250, *[list_165251], **kwargs_165253)

# Assigning a type to the variable 'hermone' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'hermone', array_call_result_165254)

# Assigning a Call to a Name (line 198):

# Assigning a Call to a Name (line 198):

# Call to array(...): (line 198)
# Processing the call arguments (line 198)

# Obtaining an instance of the builtin type 'list' (line 198)
list_165257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 198)
# Adding element type (line 198)
int_165258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 17), list_165257, int_165258)
# Adding element type (line 198)
int_165259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 21), 'int')
int_165260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
# Applying the binary operator 'div' (line 198)
result_div_165261 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 21), 'div', int_165259, int_165260)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 17), list_165257, result_div_165261)

# Processing the call keyword arguments (line 198)
kwargs_165262 = {}
# Getting the type of 'np' (line 198)
np_165255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'np', False)
# Obtaining the member 'array' of a type (line 198)
array_165256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), np_165255, 'array')
# Calling array(args, kwargs) (line 198)
array_call_result_165263 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), array_165256, *[list_165257], **kwargs_165262)

# Assigning a type to the variable 'hermx' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'hermx', array_call_result_165263)

@norecursion
def hermline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermline'
    module_type_store = module_type_store.open_function_context('hermline', 201, 0, False)
    
    # Passed parameters checking function
    hermline.stypy_localization = localization
    hermline.stypy_type_of_self = None
    hermline.stypy_type_store = module_type_store
    hermline.stypy_function_name = 'hermline'
    hermline.stypy_param_names_list = ['off', 'scl']
    hermline.stypy_varargs_param_name = None
    hermline.stypy_kwargs_param_name = None
    hermline.stypy_call_defaults = defaults
    hermline.stypy_call_varargs = varargs
    hermline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermline(...)' code ##################

    str_165264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', "\n    Hermite series whose graph is a straight line.\n\n\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Hermite series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    polyline, chebline\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermline, hermval\n    >>> hermval(0,hermline(3, 2))\n    3.0\n    >>> hermval(1,hermline(3, 2))\n    5.0\n\n    ")
    
    
    # Getting the type of 'scl' (line 231)
    scl_165265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), 'scl')
    int_165266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 14), 'int')
    # Applying the binary operator '!=' (line 231)
    result_ne_165267 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 7), '!=', scl_165265, int_165266)
    
    # Testing the type of an if condition (line 231)
    if_condition_165268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), result_ne_165267)
    # Assigning a type to the variable 'if_condition_165268' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_165268', if_condition_165268)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 232)
    # Processing the call arguments (line 232)
    
    # Obtaining an instance of the builtin type 'list' (line 232)
    list_165271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 232)
    # Adding element type (line 232)
    # Getting the type of 'off' (line 232)
    off_165272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 24), list_165271, off_165272)
    # Adding element type (line 232)
    # Getting the type of 'scl' (line 232)
    scl_165273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'scl', False)
    int_165274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 34), 'int')
    # Applying the binary operator 'div' (line 232)
    result_div_165275 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 30), 'div', scl_165273, int_165274)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 24), list_165271, result_div_165275)
    
    # Processing the call keyword arguments (line 232)
    kwargs_165276 = {}
    # Getting the type of 'np' (line 232)
    np_165269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 232)
    array_165270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), np_165269, 'array')
    # Calling array(args, kwargs) (line 232)
    array_call_result_165277 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), array_165270, *[list_165271], **kwargs_165276)
    
    # Assigning a type to the variable 'stypy_return_type' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', array_call_result_165277)
    # SSA branch for the else part of an if statement (line 231)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 234)
    # Processing the call arguments (line 234)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_165280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'off' (line 234)
    off_165281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 24), list_165280, off_165281)
    
    # Processing the call keyword arguments (line 234)
    kwargs_165282 = {}
    # Getting the type of 'np' (line 234)
    np_165278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 234)
    array_165279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), np_165278, 'array')
    # Calling array(args, kwargs) (line 234)
    array_call_result_165283 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), array_165279, *[list_165280], **kwargs_165282)
    
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', array_call_result_165283)
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermline' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_165284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165284)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermline'
    return stypy_return_type_165284

# Assigning a type to the variable 'hermline' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'hermline', hermline)

@norecursion
def hermfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermfromroots'
    module_type_store = module_type_store.open_function_context('hermfromroots', 237, 0, False)
    
    # Passed parameters checking function
    hermfromroots.stypy_localization = localization
    hermfromroots.stypy_type_of_self = None
    hermfromroots.stypy_type_store = module_type_store
    hermfromroots.stypy_function_name = 'hermfromroots'
    hermfromroots.stypy_param_names_list = ['roots']
    hermfromroots.stypy_varargs_param_name = None
    hermfromroots.stypy_kwargs_param_name = None
    hermfromroots.stypy_call_defaults = defaults
    hermfromroots.stypy_call_varargs = varargs
    hermfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermfromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermfromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermfromroots(...)' code ##################

    str_165285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'str', '\n    Generate a Hermite series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in Hermite form, where the `r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * H_1(x) + ... +  c_n * H_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in Hermite form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    polyfromroots, legfromroots, lagfromroots, chebfromroots,\n    hermefromroots.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermfromroots, hermval\n    >>> coef = hermfromroots((-1, 0, 1))\n    >>> hermval((-1, 0, 1), coef)\n    array([ 0.,  0.,  0.])\n    >>> coef = hermfromroots((-1j, 1j))\n    >>> hermval((-1j, 1j), coef)\n    array([ 0.+0.j,  0.+0.j])\n\n    ')
    
    
    
    # Call to len(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'roots' (line 287)
    roots_165287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'roots', False)
    # Processing the call keyword arguments (line 287)
    kwargs_165288 = {}
    # Getting the type of 'len' (line 287)
    len_165286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 'len', False)
    # Calling len(args, kwargs) (line 287)
    len_call_result_165289 = invoke(stypy.reporting.localization.Localization(__file__, 287, 7), len_165286, *[roots_165287], **kwargs_165288)
    
    int_165290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'int')
    # Applying the binary operator '==' (line 287)
    result_eq_165291 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 7), '==', len_call_result_165289, int_165290)
    
    # Testing the type of an if condition (line 287)
    if_condition_165292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 4), result_eq_165291)
    # Assigning a type to the variable 'if_condition_165292' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'if_condition_165292', if_condition_165292)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 288)
    # Processing the call arguments (line 288)
    int_165295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 23), 'int')
    # Processing the call keyword arguments (line 288)
    kwargs_165296 = {}
    # Getting the type of 'np' (line 288)
    np_165293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 288)
    ones_165294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), np_165293, 'ones')
    # Calling ones(args, kwargs) (line 288)
    ones_call_result_165297 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), ones_165294, *[int_165295], **kwargs_165296)
    
    # Assigning a type to the variable 'stypy_return_type' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'stypy_return_type', ones_call_result_165297)
    # SSA branch for the else part of an if statement (line 287)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 290):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Obtaining an instance of the builtin type 'list' (line 290)
    list_165300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 290)
    # Adding element type (line 290)
    # Getting the type of 'roots' (line 290)
    roots_165301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 31), list_165300, roots_165301)
    
    # Processing the call keyword arguments (line 290)
    # Getting the type of 'False' (line 290)
    False_165302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 45), 'False', False)
    keyword_165303 = False_165302
    kwargs_165304 = {'trim': keyword_165303}
    # Getting the type of 'pu' (line 290)
    pu_165298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 290)
    as_series_165299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 18), pu_165298, 'as_series')
    # Calling as_series(args, kwargs) (line 290)
    as_series_call_result_165305 = invoke(stypy.reporting.localization.Localization(__file__, 290, 18), as_series_165299, *[list_165300], **kwargs_165304)
    
    # Assigning a type to the variable 'call_assignment_165009' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_165009', as_series_call_result_165305)
    
    # Assigning a Call to a Name (line 290):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
    # Processing the call keyword arguments
    kwargs_165309 = {}
    # Getting the type of 'call_assignment_165009' (line 290)
    call_assignment_165009_165306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_165009', False)
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___165307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), call_assignment_165009_165306, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165310 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165307, *[int_165308], **kwargs_165309)
    
    # Assigning a type to the variable 'call_assignment_165010' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_165010', getitem___call_result_165310)
    
    # Assigning a Name to a Name (line 290):
    # Getting the type of 'call_assignment_165010' (line 290)
    call_assignment_165010_165311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'call_assignment_165010')
    # Assigning a type to the variable 'roots' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 9), 'roots', call_assignment_165010_165311)
    
    # Call to sort(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_165314 = {}
    # Getting the type of 'roots' (line 291)
    roots_165312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 291)
    sort_165313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), roots_165312, 'sort')
    # Calling sort(args, kwargs) (line 291)
    sort_call_result_165315 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), sort_165313, *[], **kwargs_165314)
    
    
    # Assigning a ListComp to a Name (line 292):
    
    # Assigning a ListComp to a Name (line 292):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 292)
    roots_165322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 38), 'roots')
    comprehension_165323 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 13), roots_165322)
    # Assigning a type to the variable 'r' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'r', comprehension_165323)
    
    # Call to hermline(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Getting the type of 'r' (line 292)
    r_165317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'r', False)
    # Applying the 'usub' unary operator (line 292)
    result___neg___165318 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 22), 'usub', r_165317)
    
    int_165319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 26), 'int')
    # Processing the call keyword arguments (line 292)
    kwargs_165320 = {}
    # Getting the type of 'hermline' (line 292)
    hermline_165316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 13), 'hermline', False)
    # Calling hermline(args, kwargs) (line 292)
    hermline_call_result_165321 = invoke(stypy.reporting.localization.Localization(__file__, 292, 13), hermline_165316, *[result___neg___165318, int_165319], **kwargs_165320)
    
    list_165324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 13), list_165324, hermline_call_result_165321)
    # Assigning a type to the variable 'p' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'p', list_165324)
    
    # Assigning a Call to a Name (line 293):
    
    # Assigning a Call to a Name (line 293):
    
    # Call to len(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'p' (line 293)
    p_165326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'p', False)
    # Processing the call keyword arguments (line 293)
    kwargs_165327 = {}
    # Getting the type of 'len' (line 293)
    len_165325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'len', False)
    # Calling len(args, kwargs) (line 293)
    len_call_result_165328 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), len_165325, *[p_165326], **kwargs_165327)
    
    # Assigning a type to the variable 'n' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'n', len_call_result_165328)
    
    
    # Getting the type of 'n' (line 294)
    n_165329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 14), 'n')
    int_165330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 18), 'int')
    # Applying the binary operator '>' (line 294)
    result_gt_165331 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 14), '>', n_165329, int_165330)
    
    # Testing the type of an if condition (line 294)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), result_gt_165331)
    # SSA begins for while statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 295):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 295)
    # Processing the call arguments (line 295)
    # Getting the type of 'n' (line 295)
    n_165333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 26), 'n', False)
    int_165334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 29), 'int')
    # Processing the call keyword arguments (line 295)
    kwargs_165335 = {}
    # Getting the type of 'divmod' (line 295)
    divmod_165332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 295)
    divmod_call_result_165336 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), divmod_165332, *[n_165333, int_165334], **kwargs_165335)
    
    # Assigning a type to the variable 'call_assignment_165011' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165011', divmod_call_result_165336)
    
    # Assigning a Call to a Name (line 295):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'int')
    # Processing the call keyword arguments
    kwargs_165340 = {}
    # Getting the type of 'call_assignment_165011' (line 295)
    call_assignment_165011_165337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165011', False)
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___165338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), call_assignment_165011_165337, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165341 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165338, *[int_165339], **kwargs_165340)
    
    # Assigning a type to the variable 'call_assignment_165012' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165012', getitem___call_result_165341)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'call_assignment_165012' (line 295)
    call_assignment_165012_165342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165012')
    # Assigning a type to the variable 'm' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'm', call_assignment_165012_165342)
    
    # Assigning a Call to a Name (line 295):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'int')
    # Processing the call keyword arguments
    kwargs_165346 = {}
    # Getting the type of 'call_assignment_165011' (line 295)
    call_assignment_165011_165343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165011', False)
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___165344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), call_assignment_165011_165343, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165347 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165344, *[int_165345], **kwargs_165346)
    
    # Assigning a type to the variable 'call_assignment_165013' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165013', getitem___call_result_165347)
    
    # Assigning a Name to a Name (line 295):
    # Getting the type of 'call_assignment_165013' (line 295)
    call_assignment_165013_165348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'call_assignment_165013')
    # Assigning a type to the variable 'r' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'r', call_assignment_165013_165348)
    
    # Assigning a ListComp to a Name (line 296):
    
    # Assigning a ListComp to a Name (line 296):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'm' (line 296)
    m_165363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 56), 'm', False)
    # Processing the call keyword arguments (line 296)
    kwargs_165364 = {}
    # Getting the type of 'range' (line 296)
    range_165362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 50), 'range', False)
    # Calling range(args, kwargs) (line 296)
    range_call_result_165365 = invoke(stypy.reporting.localization.Localization(__file__, 296, 50), range_165362, *[m_165363], **kwargs_165364)
    
    comprehension_165366 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 19), range_call_result_165365)
    # Assigning a type to the variable 'i' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'i', comprehension_165366)
    
    # Call to hermmul(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 296)
    i_165350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'i', False)
    # Getting the type of 'p' (line 296)
    p_165351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___165352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), p_165351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_165353 = invoke(stypy.reporting.localization.Localization(__file__, 296, 27), getitem___165352, i_165350)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 296)
    i_165354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 35), 'i', False)
    # Getting the type of 'm' (line 296)
    m_165355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 37), 'm', False)
    # Applying the binary operator '+' (line 296)
    result_add_165356 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 35), '+', i_165354, m_165355)
    
    # Getting the type of 'p' (line 296)
    p_165357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___165358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 33), p_165357, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_165359 = invoke(stypy.reporting.localization.Localization(__file__, 296, 33), getitem___165358, result_add_165356)
    
    # Processing the call keyword arguments (line 296)
    kwargs_165360 = {}
    # Getting the type of 'hermmul' (line 296)
    hermmul_165349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'hermmul', False)
    # Calling hermmul(args, kwargs) (line 296)
    hermmul_call_result_165361 = invoke(stypy.reporting.localization.Localization(__file__, 296, 19), hermmul_165349, *[subscript_call_result_165353, subscript_call_result_165359], **kwargs_165360)
    
    list_165367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 19), list_165367, hermmul_call_result_165361)
    # Assigning a type to the variable 'tmp' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'tmp', list_165367)
    
    # Getting the type of 'r' (line 297)
    r_165368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'r')
    # Testing the type of an if condition (line 297)
    if_condition_165369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), r_165368)
    # Assigning a type to the variable 'if_condition_165369' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'if_condition_165369', if_condition_165369)
    # SSA begins for if statement (line 297)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 298):
    
    # Assigning a Call to a Subscript (line 298):
    
    # Call to hermmul(...): (line 298)
    # Processing the call arguments (line 298)
    
    # Obtaining the type of the subscript
    int_165371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 37), 'int')
    # Getting the type of 'tmp' (line 298)
    tmp_165372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 33), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___165373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 33), tmp_165372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_165374 = invoke(stypy.reporting.localization.Localization(__file__, 298, 33), getitem___165373, int_165371)
    
    
    # Obtaining the type of the subscript
    int_165375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 43), 'int')
    # Getting the type of 'p' (line 298)
    p_165376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 41), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___165377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 41), p_165376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_165378 = invoke(stypy.reporting.localization.Localization(__file__, 298, 41), getitem___165377, int_165375)
    
    # Processing the call keyword arguments (line 298)
    kwargs_165379 = {}
    # Getting the type of 'hermmul' (line 298)
    hermmul_165370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 25), 'hermmul', False)
    # Calling hermmul(args, kwargs) (line 298)
    hermmul_call_result_165380 = invoke(stypy.reporting.localization.Localization(__file__, 298, 25), hermmul_165370, *[subscript_call_result_165374, subscript_call_result_165378], **kwargs_165379)
    
    # Getting the type of 'tmp' (line 298)
    tmp_165381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'tmp')
    int_165382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 20), 'int')
    # Storing an element on a container (line 298)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 16), tmp_165381, (int_165382, hermmul_call_result_165380))
    # SSA join for if statement (line 297)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 299):
    
    # Assigning a Name to a Name (line 299):
    # Getting the type of 'tmp' (line 299)
    tmp_165383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'p', tmp_165383)
    
    # Assigning a Name to a Name (line 300):
    
    # Assigning a Name to a Name (line 300):
    # Getting the type of 'm' (line 300)
    m_165384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'm')
    # Assigning a type to the variable 'n' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'n', m_165384)
    # SSA join for while statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_165385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 17), 'int')
    # Getting the type of 'p' (line 301)
    p_165386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___165387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 15), p_165386, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_165388 = invoke(stypy.reporting.localization.Localization(__file__, 301, 15), getitem___165387, int_165385)
    
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'stypy_return_type', subscript_call_result_165388)
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 237)
    stypy_return_type_165389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165389)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermfromroots'
    return stypy_return_type_165389

# Assigning a type to the variable 'hermfromroots' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'hermfromroots', hermfromroots)

@norecursion
def hermadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermadd'
    module_type_store = module_type_store.open_function_context('hermadd', 304, 0, False)
    
    # Passed parameters checking function
    hermadd.stypy_localization = localization
    hermadd.stypy_type_of_self = None
    hermadd.stypy_type_store = module_type_store
    hermadd.stypy_function_name = 'hermadd'
    hermadd.stypy_param_names_list = ['c1', 'c2']
    hermadd.stypy_varargs_param_name = None
    hermadd.stypy_kwargs_param_name = None
    hermadd.stypy_call_defaults = defaults
    hermadd.stypy_call_varargs = varargs
    hermadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermadd(...)' code ##################

    str_165390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, (-1)), 'str', '\n    Add one Hermite series to another.\n\n    Returns the sum of two Hermite series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Hermite series of their sum.\n\n    See Also\n    --------\n    hermsub, hermmul, hermdiv, hermpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Hermite series\n    is a Hermite series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermadd\n    >>> hermadd([1, 2, 3], [1, 2, 3, 4])\n    array([ 2.,  4.,  6.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 342):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 342)
    # Processing the call arguments (line 342)
    
    # Obtaining an instance of the builtin type 'list' (line 342)
    list_165393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 342)
    # Adding element type (line 342)
    # Getting the type of 'c1' (line 342)
    c1_165394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 28), list_165393, c1_165394)
    # Adding element type (line 342)
    # Getting the type of 'c2' (line 342)
    c2_165395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 28), list_165393, c2_165395)
    
    # Processing the call keyword arguments (line 342)
    kwargs_165396 = {}
    # Getting the type of 'pu' (line 342)
    pu_165391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 342)
    as_series_165392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), pu_165391, 'as_series')
    # Calling as_series(args, kwargs) (line 342)
    as_series_call_result_165397 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), as_series_165392, *[list_165393], **kwargs_165396)
    
    # Assigning a type to the variable 'call_assignment_165014' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165014', as_series_call_result_165397)
    
    # Assigning a Call to a Name (line 342):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165401 = {}
    # Getting the type of 'call_assignment_165014' (line 342)
    call_assignment_165014_165398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165014', False)
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___165399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), call_assignment_165014_165398, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165402 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165399, *[int_165400], **kwargs_165401)
    
    # Assigning a type to the variable 'call_assignment_165015' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165015', getitem___call_result_165402)
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'call_assignment_165015' (line 342)
    call_assignment_165015_165403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165015')
    # Assigning a type to the variable 'c1' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 5), 'c1', call_assignment_165015_165403)
    
    # Assigning a Call to a Name (line 342):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165407 = {}
    # Getting the type of 'call_assignment_165014' (line 342)
    call_assignment_165014_165404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165014', False)
    # Obtaining the member '__getitem__' of a type (line 342)
    getitem___165405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 4), call_assignment_165014_165404, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165408 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165405, *[int_165406], **kwargs_165407)
    
    # Assigning a type to the variable 'call_assignment_165016' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165016', getitem___call_result_165408)
    
    # Assigning a Name to a Name (line 342):
    # Getting the type of 'call_assignment_165016' (line 342)
    call_assignment_165016_165409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'call_assignment_165016')
    # Assigning a type to the variable 'c2' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 9), 'c2', call_assignment_165016_165409)
    
    
    
    # Call to len(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'c1' (line 343)
    c1_165411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 'c1', False)
    # Processing the call keyword arguments (line 343)
    kwargs_165412 = {}
    # Getting the type of 'len' (line 343)
    len_165410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 7), 'len', False)
    # Calling len(args, kwargs) (line 343)
    len_call_result_165413 = invoke(stypy.reporting.localization.Localization(__file__, 343, 7), len_165410, *[c1_165411], **kwargs_165412)
    
    
    # Call to len(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'c2' (line 343)
    c2_165415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'c2', False)
    # Processing the call keyword arguments (line 343)
    kwargs_165416 = {}
    # Getting the type of 'len' (line 343)
    len_165414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 17), 'len', False)
    # Calling len(args, kwargs) (line 343)
    len_call_result_165417 = invoke(stypy.reporting.localization.Localization(__file__, 343, 17), len_165414, *[c2_165415], **kwargs_165416)
    
    # Applying the binary operator '>' (line 343)
    result_gt_165418 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 7), '>', len_call_result_165413, len_call_result_165417)
    
    # Testing the type of an if condition (line 343)
    if_condition_165419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 4), result_gt_165418)
    # Assigning a type to the variable 'if_condition_165419' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'if_condition_165419', if_condition_165419)
    # SSA begins for if statement (line 343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 344)
    c1_165420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 344)
    c2_165421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'c2')
    # Obtaining the member 'size' of a type (line 344)
    size_165422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), c2_165421, 'size')
    slice_165423 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 8), None, size_165422, None)
    # Getting the type of 'c1' (line 344)
    c1_165424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 344)
    getitem___165425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), c1_165424, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 344)
    subscript_call_result_165426 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), getitem___165425, slice_165423)
    
    # Getting the type of 'c2' (line 344)
    c2_165427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 24), 'c2')
    # Applying the binary operator '+=' (line 344)
    result_iadd_165428 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 8), '+=', subscript_call_result_165426, c2_165427)
    # Getting the type of 'c1' (line 344)
    c1_165429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'c1')
    # Getting the type of 'c2' (line 344)
    c2_165430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'c2')
    # Obtaining the member 'size' of a type (line 344)
    size_165431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), c2_165430, 'size')
    slice_165432 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 344, 8), None, size_165431, None)
    # Storing an element on a container (line 344)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 8), c1_165429, (slice_165432, result_iadd_165428))
    
    
    # Assigning a Name to a Name (line 345):
    
    # Assigning a Name to a Name (line 345):
    # Getting the type of 'c1' (line 345)
    c1_165433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'ret', c1_165433)
    # SSA branch for the else part of an if statement (line 343)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 347)
    c2_165434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 347)
    c1_165435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'c1')
    # Obtaining the member 'size' of a type (line 347)
    size_165436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), c1_165435, 'size')
    slice_165437 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 347, 8), None, size_165436, None)
    # Getting the type of 'c2' (line 347)
    c2_165438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___165439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), c2_165438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_165440 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), getitem___165439, slice_165437)
    
    # Getting the type of 'c1' (line 347)
    c1_165441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'c1')
    # Applying the binary operator '+=' (line 347)
    result_iadd_165442 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 8), '+=', subscript_call_result_165440, c1_165441)
    # Getting the type of 'c2' (line 347)
    c2_165443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'c2')
    # Getting the type of 'c1' (line 347)
    c1_165444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'c1')
    # Obtaining the member 'size' of a type (line 347)
    size_165445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), c1_165444, 'size')
    slice_165446 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 347, 8), None, size_165445, None)
    # Storing an element on a container (line 347)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 8), c2_165443, (slice_165446, result_iadd_165442))
    
    
    # Assigning a Name to a Name (line 348):
    
    # Assigning a Name to a Name (line 348):
    # Getting the type of 'c2' (line 348)
    c2_165447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'ret', c2_165447)
    # SSA join for if statement (line 343)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'ret' (line 349)
    ret_165450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 22), 'ret', False)
    # Processing the call keyword arguments (line 349)
    kwargs_165451 = {}
    # Getting the type of 'pu' (line 349)
    pu_165448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 349)
    trimseq_165449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 11), pu_165448, 'trimseq')
    # Calling trimseq(args, kwargs) (line 349)
    trimseq_call_result_165452 = invoke(stypy.reporting.localization.Localization(__file__, 349, 11), trimseq_165449, *[ret_165450], **kwargs_165451)
    
    # Assigning a type to the variable 'stypy_return_type' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type', trimseq_call_result_165452)
    
    # ################# End of 'hermadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermadd' in the type store
    # Getting the type of 'stypy_return_type' (line 304)
    stypy_return_type_165453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165453)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermadd'
    return stypy_return_type_165453

# Assigning a type to the variable 'hermadd' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'hermadd', hermadd)

@norecursion
def hermsub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermsub'
    module_type_store = module_type_store.open_function_context('hermsub', 352, 0, False)
    
    # Passed parameters checking function
    hermsub.stypy_localization = localization
    hermsub.stypy_type_of_self = None
    hermsub.stypy_type_store = module_type_store
    hermsub.stypy_function_name = 'hermsub'
    hermsub.stypy_param_names_list = ['c1', 'c2']
    hermsub.stypy_varargs_param_name = None
    hermsub.stypy_kwargs_param_name = None
    hermsub.stypy_call_defaults = defaults
    hermsub.stypy_call_varargs = varargs
    hermsub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermsub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermsub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermsub(...)' code ##################

    str_165454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, (-1)), 'str', '\n    Subtract one Hermite series from another.\n\n    Returns the difference of two Hermite series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Hermite series coefficients representing their difference.\n\n    See Also\n    --------\n    hermadd, hermmul, hermdiv, hermpow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Hermite\n    series is a Hermite series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermsub\n    >>> hermsub([1, 2, 3, 4], [1, 2, 3])\n    array([ 0.,  0.,  0.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 390):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 390)
    # Processing the call arguments (line 390)
    
    # Obtaining an instance of the builtin type 'list' (line 390)
    list_165457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 390)
    # Adding element type (line 390)
    # Getting the type of 'c1' (line 390)
    c1_165458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 28), list_165457, c1_165458)
    # Adding element type (line 390)
    # Getting the type of 'c2' (line 390)
    c2_165459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 28), list_165457, c2_165459)
    
    # Processing the call keyword arguments (line 390)
    kwargs_165460 = {}
    # Getting the type of 'pu' (line 390)
    pu_165455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 390)
    as_series_165456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 15), pu_165455, 'as_series')
    # Calling as_series(args, kwargs) (line 390)
    as_series_call_result_165461 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), as_series_165456, *[list_165457], **kwargs_165460)
    
    # Assigning a type to the variable 'call_assignment_165017' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165017', as_series_call_result_165461)
    
    # Assigning a Call to a Name (line 390):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165465 = {}
    # Getting the type of 'call_assignment_165017' (line 390)
    call_assignment_165017_165462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165017', False)
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___165463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 4), call_assignment_165017_165462, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165466 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165463, *[int_165464], **kwargs_165465)
    
    # Assigning a type to the variable 'call_assignment_165018' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165018', getitem___call_result_165466)
    
    # Assigning a Name to a Name (line 390):
    # Getting the type of 'call_assignment_165018' (line 390)
    call_assignment_165018_165467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165018')
    # Assigning a type to the variable 'c1' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 5), 'c1', call_assignment_165018_165467)
    
    # Assigning a Call to a Name (line 390):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165471 = {}
    # Getting the type of 'call_assignment_165017' (line 390)
    call_assignment_165017_165468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165017', False)
    # Obtaining the member '__getitem__' of a type (line 390)
    getitem___165469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 4), call_assignment_165017_165468, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165472 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165469, *[int_165470], **kwargs_165471)
    
    # Assigning a type to the variable 'call_assignment_165019' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165019', getitem___call_result_165472)
    
    # Assigning a Name to a Name (line 390):
    # Getting the type of 'call_assignment_165019' (line 390)
    call_assignment_165019_165473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'call_assignment_165019')
    # Assigning a type to the variable 'c2' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 9), 'c2', call_assignment_165019_165473)
    
    
    
    # Call to len(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'c1' (line 391)
    c1_165475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'c1', False)
    # Processing the call keyword arguments (line 391)
    kwargs_165476 = {}
    # Getting the type of 'len' (line 391)
    len_165474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'len', False)
    # Calling len(args, kwargs) (line 391)
    len_call_result_165477 = invoke(stypy.reporting.localization.Localization(__file__, 391, 7), len_165474, *[c1_165475], **kwargs_165476)
    
    
    # Call to len(...): (line 391)
    # Processing the call arguments (line 391)
    # Getting the type of 'c2' (line 391)
    c2_165479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 21), 'c2', False)
    # Processing the call keyword arguments (line 391)
    kwargs_165480 = {}
    # Getting the type of 'len' (line 391)
    len_165478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 17), 'len', False)
    # Calling len(args, kwargs) (line 391)
    len_call_result_165481 = invoke(stypy.reporting.localization.Localization(__file__, 391, 17), len_165478, *[c2_165479], **kwargs_165480)
    
    # Applying the binary operator '>' (line 391)
    result_gt_165482 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), '>', len_call_result_165477, len_call_result_165481)
    
    # Testing the type of an if condition (line 391)
    if_condition_165483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), result_gt_165482)
    # Assigning a type to the variable 'if_condition_165483' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_165483', if_condition_165483)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 392)
    c1_165484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 392)
    c2_165485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'c2')
    # Obtaining the member 'size' of a type (line 392)
    size_165486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), c2_165485, 'size')
    slice_165487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 392, 8), None, size_165486, None)
    # Getting the type of 'c1' (line 392)
    c1_165488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___165489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), c1_165488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_165490 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), getitem___165489, slice_165487)
    
    # Getting the type of 'c2' (line 392)
    c2_165491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'c2')
    # Applying the binary operator '-=' (line 392)
    result_isub_165492 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 8), '-=', subscript_call_result_165490, c2_165491)
    # Getting the type of 'c1' (line 392)
    c1_165493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'c1')
    # Getting the type of 'c2' (line 392)
    c2_165494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'c2')
    # Obtaining the member 'size' of a type (line 392)
    size_165495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 12), c2_165494, 'size')
    slice_165496 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 392, 8), None, size_165495, None)
    # Storing an element on a container (line 392)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 8), c1_165493, (slice_165496, result_isub_165492))
    
    
    # Assigning a Name to a Name (line 393):
    
    # Assigning a Name to a Name (line 393):
    # Getting the type of 'c1' (line 393)
    c1_165497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'ret', c1_165497)
    # SSA branch for the else part of an if statement (line 391)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 395):
    
    # Assigning a UnaryOp to a Name (line 395):
    
    # Getting the type of 'c2' (line 395)
    c2_165498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'c2')
    # Applying the 'usub' unary operator (line 395)
    result___neg___165499 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 13), 'usub', c2_165498)
    
    # Assigning a type to the variable 'c2' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'c2', result___neg___165499)
    
    # Getting the type of 'c2' (line 396)
    c2_165500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 396)
    c1_165501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'c1')
    # Obtaining the member 'size' of a type (line 396)
    size_165502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), c1_165501, 'size')
    slice_165503 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 8), None, size_165502, None)
    # Getting the type of 'c2' (line 396)
    c2_165504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___165505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), c2_165504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_165506 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), getitem___165505, slice_165503)
    
    # Getting the type of 'c1' (line 396)
    c1_165507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 24), 'c1')
    # Applying the binary operator '+=' (line 396)
    result_iadd_165508 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 8), '+=', subscript_call_result_165506, c1_165507)
    # Getting the type of 'c2' (line 396)
    c2_165509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'c2')
    # Getting the type of 'c1' (line 396)
    c1_165510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'c1')
    # Obtaining the member 'size' of a type (line 396)
    size_165511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 12), c1_165510, 'size')
    slice_165512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 8), None, size_165511, None)
    # Storing an element on a container (line 396)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 8), c2_165509, (slice_165512, result_iadd_165508))
    
    
    # Assigning a Name to a Name (line 397):
    
    # Assigning a Name to a Name (line 397):
    # Getting the type of 'c2' (line 397)
    c2_165513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'ret', c2_165513)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'ret' (line 398)
    ret_165516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 22), 'ret', False)
    # Processing the call keyword arguments (line 398)
    kwargs_165517 = {}
    # Getting the type of 'pu' (line 398)
    pu_165514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 398)
    trimseq_165515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 11), pu_165514, 'trimseq')
    # Calling trimseq(args, kwargs) (line 398)
    trimseq_call_result_165518 = invoke(stypy.reporting.localization.Localization(__file__, 398, 11), trimseq_165515, *[ret_165516], **kwargs_165517)
    
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'stypy_return_type', trimseq_call_result_165518)
    
    # ################# End of 'hermsub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermsub' in the type store
    # Getting the type of 'stypy_return_type' (line 352)
    stypy_return_type_165519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165519)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermsub'
    return stypy_return_type_165519

# Assigning a type to the variable 'hermsub' (line 352)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 0), 'hermsub', hermsub)

@norecursion
def hermmulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermmulx'
    module_type_store = module_type_store.open_function_context('hermmulx', 401, 0, False)
    
    # Passed parameters checking function
    hermmulx.stypy_localization = localization
    hermmulx.stypy_type_of_self = None
    hermmulx.stypy_type_store = module_type_store
    hermmulx.stypy_function_name = 'hermmulx'
    hermmulx.stypy_param_names_list = ['c']
    hermmulx.stypy_varargs_param_name = None
    hermmulx.stypy_kwargs_param_name = None
    hermmulx.stypy_call_defaults = defaults
    hermmulx.stypy_call_varargs = varargs
    hermmulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermmulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermmulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermmulx(...)' code ##################

    str_165520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, (-1)), 'str', 'Multiply a Hermite series by x.\n\n    Multiply the Hermite series `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n    The multiplication uses the recursion relationship for Hermite\n    polynomials in the form\n\n    .. math::\n\n    xP_i(x) = (P_{i + 1}(x)/2 + i*P_{i - 1}(x))\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermmulx\n    >>> hermmulx([1, 2, 3])\n    array([ 2. ,  6.5,  1. ,  1.5])\n\n    ')
    
    # Assigning a Call to a List (line 436):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 436)
    # Processing the call arguments (line 436)
    
    # Obtaining an instance of the builtin type 'list' (line 436)
    list_165523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 436)
    # Adding element type (line 436)
    # Getting the type of 'c' (line 436)
    c_165524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 23), list_165523, c_165524)
    
    # Processing the call keyword arguments (line 436)
    kwargs_165525 = {}
    # Getting the type of 'pu' (line 436)
    pu_165521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 436)
    as_series_165522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 10), pu_165521, 'as_series')
    # Calling as_series(args, kwargs) (line 436)
    as_series_call_result_165526 = invoke(stypy.reporting.localization.Localization(__file__, 436, 10), as_series_165522, *[list_165523], **kwargs_165525)
    
    # Assigning a type to the variable 'call_assignment_165020' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'call_assignment_165020', as_series_call_result_165526)
    
    # Assigning a Call to a Name (line 436):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165530 = {}
    # Getting the type of 'call_assignment_165020' (line 436)
    call_assignment_165020_165527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'call_assignment_165020', False)
    # Obtaining the member '__getitem__' of a type (line 436)
    getitem___165528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), call_assignment_165020_165527, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165531 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165528, *[int_165529], **kwargs_165530)
    
    # Assigning a type to the variable 'call_assignment_165021' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'call_assignment_165021', getitem___call_result_165531)
    
    # Assigning a Name to a Name (line 436):
    # Getting the type of 'call_assignment_165021' (line 436)
    call_assignment_165021_165532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'call_assignment_165021')
    # Assigning a type to the variable 'c' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 5), 'c', call_assignment_165021_165532)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'c' (line 438)
    c_165534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'c', False)
    # Processing the call keyword arguments (line 438)
    kwargs_165535 = {}
    # Getting the type of 'len' (line 438)
    len_165533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 7), 'len', False)
    # Calling len(args, kwargs) (line 438)
    len_call_result_165536 = invoke(stypy.reporting.localization.Localization(__file__, 438, 7), len_165533, *[c_165534], **kwargs_165535)
    
    int_165537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 17), 'int')
    # Applying the binary operator '==' (line 438)
    result_eq_165538 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 7), '==', len_call_result_165536, int_165537)
    
    
    
    # Obtaining the type of the subscript
    int_165539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 25), 'int')
    # Getting the type of 'c' (line 438)
    c_165540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 438)
    getitem___165541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 23), c_165540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 438)
    subscript_call_result_165542 = invoke(stypy.reporting.localization.Localization(__file__, 438, 23), getitem___165541, int_165539)
    
    int_165543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 31), 'int')
    # Applying the binary operator '==' (line 438)
    result_eq_165544 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 23), '==', subscript_call_result_165542, int_165543)
    
    # Applying the binary operator 'and' (line 438)
    result_and_keyword_165545 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 7), 'and', result_eq_165538, result_eq_165544)
    
    # Testing the type of an if condition (line 438)
    if_condition_165546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 4), result_and_keyword_165545)
    # Assigning a type to the variable 'if_condition_165546' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'if_condition_165546', if_condition_165546)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 439)
    c_165547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'stypy_return_type', c_165547)
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to empty(...): (line 441)
    # Processing the call arguments (line 441)
    
    # Call to len(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'c' (line 441)
    c_165551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'c', False)
    # Processing the call keyword arguments (line 441)
    kwargs_165552 = {}
    # Getting the type of 'len' (line 441)
    len_165550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 19), 'len', False)
    # Calling len(args, kwargs) (line 441)
    len_call_result_165553 = invoke(stypy.reporting.localization.Localization(__file__, 441, 19), len_165550, *[c_165551], **kwargs_165552)
    
    int_165554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 28), 'int')
    # Applying the binary operator '+' (line 441)
    result_add_165555 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 19), '+', len_call_result_165553, int_165554)
    
    # Processing the call keyword arguments (line 441)
    # Getting the type of 'c' (line 441)
    c_165556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 441)
    dtype_165557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 37), c_165556, 'dtype')
    keyword_165558 = dtype_165557
    kwargs_165559 = {'dtype': keyword_165558}
    # Getting the type of 'np' (line 441)
    np_165548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 441)
    empty_165549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 10), np_165548, 'empty')
    # Calling empty(args, kwargs) (line 441)
    empty_call_result_165560 = invoke(stypy.reporting.localization.Localization(__file__, 441, 10), empty_165549, *[result_add_165555], **kwargs_165559)
    
    # Assigning a type to the variable 'prd' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'prd', empty_call_result_165560)
    
    # Assigning a BinOp to a Subscript (line 442):
    
    # Assigning a BinOp to a Subscript (line 442):
    
    # Obtaining the type of the subscript
    int_165561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 15), 'int')
    # Getting the type of 'c' (line 442)
    c_165562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___165563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 13), c_165562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_165564 = invoke(stypy.reporting.localization.Localization(__file__, 442, 13), getitem___165563, int_165561)
    
    int_165565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 18), 'int')
    # Applying the binary operator '*' (line 442)
    result_mul_165566 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 13), '*', subscript_call_result_165564, int_165565)
    
    # Getting the type of 'prd' (line 442)
    prd_165567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'prd')
    int_165568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 8), 'int')
    # Storing an element on a container (line 442)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 4), prd_165567, (int_165568, result_mul_165566))
    
    # Assigning a BinOp to a Subscript (line 443):
    
    # Assigning a BinOp to a Subscript (line 443):
    
    # Obtaining the type of the subscript
    int_165569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 15), 'int')
    # Getting the type of 'c' (line 443)
    c_165570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___165571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 13), c_165570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_165572 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), getitem___165571, int_165569)
    
    int_165573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 18), 'int')
    # Applying the binary operator 'div' (line 443)
    result_div_165574 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 13), 'div', subscript_call_result_165572, int_165573)
    
    # Getting the type of 'prd' (line 443)
    prd_165575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'prd')
    int_165576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
    # Storing an element on a container (line 443)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 4), prd_165575, (int_165576, result_div_165574))
    
    
    # Call to range(...): (line 444)
    # Processing the call arguments (line 444)
    int_165578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 19), 'int')
    
    # Call to len(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'c' (line 444)
    c_165580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'c', False)
    # Processing the call keyword arguments (line 444)
    kwargs_165581 = {}
    # Getting the type of 'len' (line 444)
    len_165579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 22), 'len', False)
    # Calling len(args, kwargs) (line 444)
    len_call_result_165582 = invoke(stypy.reporting.localization.Localization(__file__, 444, 22), len_165579, *[c_165580], **kwargs_165581)
    
    # Processing the call keyword arguments (line 444)
    kwargs_165583 = {}
    # Getting the type of 'range' (line 444)
    range_165577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'range', False)
    # Calling range(args, kwargs) (line 444)
    range_call_result_165584 = invoke(stypy.reporting.localization.Localization(__file__, 444, 13), range_165577, *[int_165578, len_call_result_165582], **kwargs_165583)
    
    # Testing the type of a for loop iterable (line 444)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 4), range_call_result_165584)
    # Getting the type of the for loop variable (line 444)
    for_loop_var_165585 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 4), range_call_result_165584)
    # Assigning a type to the variable 'i' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'i', for_loop_var_165585)
    # SSA begins for a for statement (line 444)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 445):
    
    # Assigning a BinOp to a Subscript (line 445):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 445)
    i_165586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 23), 'i')
    # Getting the type of 'c' (line 445)
    c_165587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 445)
    getitem___165588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), c_165587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 445)
    subscript_call_result_165589 = invoke(stypy.reporting.localization.Localization(__file__, 445, 21), getitem___165588, i_165586)
    
    int_165590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 26), 'int')
    # Applying the binary operator 'div' (line 445)
    result_div_165591 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 21), 'div', subscript_call_result_165589, int_165590)
    
    # Getting the type of 'prd' (line 445)
    prd_165592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'prd')
    # Getting the type of 'i' (line 445)
    i_165593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'i')
    int_165594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 16), 'int')
    # Applying the binary operator '+' (line 445)
    result_add_165595 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 12), '+', i_165593, int_165594)
    
    # Storing an element on a container (line 445)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 8), prd_165592, (result_add_165595, result_div_165591))
    
    # Getting the type of 'prd' (line 446)
    prd_165596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'prd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 446)
    i_165597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'i')
    int_165598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 16), 'int')
    # Applying the binary operator '-' (line 446)
    result_sub_165599 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 12), '-', i_165597, int_165598)
    
    # Getting the type of 'prd' (line 446)
    prd_165600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___165601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), prd_165600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_165602 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), getitem___165601, result_sub_165599)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 446)
    i_165603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'i')
    # Getting the type of 'c' (line 446)
    c_165604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___165605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 22), c_165604, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_165606 = invoke(stypy.reporting.localization.Localization(__file__, 446, 22), getitem___165605, i_165603)
    
    # Getting the type of 'i' (line 446)
    i_165607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 27), 'i')
    # Applying the binary operator '*' (line 446)
    result_mul_165608 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 22), '*', subscript_call_result_165606, i_165607)
    
    # Applying the binary operator '+=' (line 446)
    result_iadd_165609 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 8), '+=', subscript_call_result_165602, result_mul_165608)
    # Getting the type of 'prd' (line 446)
    prd_165610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'prd')
    # Getting the type of 'i' (line 446)
    i_165611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'i')
    int_165612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 16), 'int')
    # Applying the binary operator '-' (line 446)
    result_sub_165613 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 12), '-', i_165611, int_165612)
    
    # Storing an element on a container (line 446)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 8), prd_165610, (result_sub_165613, result_iadd_165609))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 447)
    prd_165614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type', prd_165614)
    
    # ################# End of 'hermmulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermmulx' in the type store
    # Getting the type of 'stypy_return_type' (line 401)
    stypy_return_type_165615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermmulx'
    return stypy_return_type_165615

# Assigning a type to the variable 'hermmulx' (line 401)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 0), 'hermmulx', hermmulx)

@norecursion
def hermmul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermmul'
    module_type_store = module_type_store.open_function_context('hermmul', 450, 0, False)
    
    # Passed parameters checking function
    hermmul.stypy_localization = localization
    hermmul.stypy_type_of_self = None
    hermmul.stypy_type_store = module_type_store
    hermmul.stypy_function_name = 'hermmul'
    hermmul.stypy_param_names_list = ['c1', 'c2']
    hermmul.stypy_varargs_param_name = None
    hermmul.stypy_kwargs_param_name = None
    hermmul.stypy_call_defaults = defaults
    hermmul.stypy_call_varargs = varargs
    hermmul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermmul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermmul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermmul(...)' code ##################

    str_165616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, (-1)), 'str', '\n    Multiply one Hermite series by another.\n\n    Returns the product of two Hermite series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Hermite series coefficients representing their product.\n\n    See Also\n    --------\n    hermadd, hermsub, hermdiv, hermpow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Hermite polynomial basis set.  Thus, to express\n    the product as a Hermite series, it is necessary to "reproject" the\n    product onto said basis set, which may produce "unintuitive" (but\n    correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermmul\n    >>> hermmul([1, 2, 3], [0, 1, 2])\n    array([ 52.,  29.,  52.,   7.,   6.])\n\n    ')
    
    # Assigning a Call to a List (line 489):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 489)
    # Processing the call arguments (line 489)
    
    # Obtaining an instance of the builtin type 'list' (line 489)
    list_165619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 489)
    # Adding element type (line 489)
    # Getting the type of 'c1' (line 489)
    c1_165620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 28), list_165619, c1_165620)
    # Adding element type (line 489)
    # Getting the type of 'c2' (line 489)
    c2_165621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 489, 28), list_165619, c2_165621)
    
    # Processing the call keyword arguments (line 489)
    kwargs_165622 = {}
    # Getting the type of 'pu' (line 489)
    pu_165617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 489)
    as_series_165618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), pu_165617, 'as_series')
    # Calling as_series(args, kwargs) (line 489)
    as_series_call_result_165623 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), as_series_165618, *[list_165619], **kwargs_165622)
    
    # Assigning a type to the variable 'call_assignment_165022' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165022', as_series_call_result_165623)
    
    # Assigning a Call to a Name (line 489):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165627 = {}
    # Getting the type of 'call_assignment_165022' (line 489)
    call_assignment_165022_165624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165022', False)
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___165625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 4), call_assignment_165022_165624, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165628 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165625, *[int_165626], **kwargs_165627)
    
    # Assigning a type to the variable 'call_assignment_165023' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165023', getitem___call_result_165628)
    
    # Assigning a Name to a Name (line 489):
    # Getting the type of 'call_assignment_165023' (line 489)
    call_assignment_165023_165629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165023')
    # Assigning a type to the variable 'c1' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 5), 'c1', call_assignment_165023_165629)
    
    # Assigning a Call to a Name (line 489):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165633 = {}
    # Getting the type of 'call_assignment_165022' (line 489)
    call_assignment_165022_165630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165022', False)
    # Obtaining the member '__getitem__' of a type (line 489)
    getitem___165631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 4), call_assignment_165022_165630, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165634 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165631, *[int_165632], **kwargs_165633)
    
    # Assigning a type to the variable 'call_assignment_165024' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165024', getitem___call_result_165634)
    
    # Assigning a Name to a Name (line 489):
    # Getting the type of 'call_assignment_165024' (line 489)
    call_assignment_165024_165635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'call_assignment_165024')
    # Assigning a type to the variable 'c2' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 9), 'c2', call_assignment_165024_165635)
    
    
    
    # Call to len(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'c1' (line 491)
    c1_165637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 11), 'c1', False)
    # Processing the call keyword arguments (line 491)
    kwargs_165638 = {}
    # Getting the type of 'len' (line 491)
    len_165636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 7), 'len', False)
    # Calling len(args, kwargs) (line 491)
    len_call_result_165639 = invoke(stypy.reporting.localization.Localization(__file__, 491, 7), len_165636, *[c1_165637], **kwargs_165638)
    
    
    # Call to len(...): (line 491)
    # Processing the call arguments (line 491)
    # Getting the type of 'c2' (line 491)
    c2_165641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 21), 'c2', False)
    # Processing the call keyword arguments (line 491)
    kwargs_165642 = {}
    # Getting the type of 'len' (line 491)
    len_165640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 17), 'len', False)
    # Calling len(args, kwargs) (line 491)
    len_call_result_165643 = invoke(stypy.reporting.localization.Localization(__file__, 491, 17), len_165640, *[c2_165641], **kwargs_165642)
    
    # Applying the binary operator '>' (line 491)
    result_gt_165644 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 7), '>', len_call_result_165639, len_call_result_165643)
    
    # Testing the type of an if condition (line 491)
    if_condition_165645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 491, 4), result_gt_165644)
    # Assigning a type to the variable 'if_condition_165645' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'if_condition_165645', if_condition_165645)
    # SSA begins for if statement (line 491)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 492):
    
    # Assigning a Name to a Name (line 492):
    # Getting the type of 'c2' (line 492)
    c2_165646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 12), 'c2')
    # Assigning a type to the variable 'c' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'c', c2_165646)
    
    # Assigning a Name to a Name (line 493):
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'c1' (line 493)
    c1_165647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 13), 'c1')
    # Assigning a type to the variable 'xs' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'xs', c1_165647)
    # SSA branch for the else part of an if statement (line 491)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 495):
    
    # Assigning a Name to a Name (line 495):
    # Getting the type of 'c1' (line 495)
    c1_165648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'c1')
    # Assigning a type to the variable 'c' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'c', c1_165648)
    
    # Assigning a Name to a Name (line 496):
    
    # Assigning a Name to a Name (line 496):
    # Getting the type of 'c2' (line 496)
    c2_165649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 13), 'c2')
    # Assigning a type to the variable 'xs' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'xs', c2_165649)
    # SSA join for if statement (line 491)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'c' (line 498)
    c_165651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 11), 'c', False)
    # Processing the call keyword arguments (line 498)
    kwargs_165652 = {}
    # Getting the type of 'len' (line 498)
    len_165650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 7), 'len', False)
    # Calling len(args, kwargs) (line 498)
    len_call_result_165653 = invoke(stypy.reporting.localization.Localization(__file__, 498, 7), len_165650, *[c_165651], **kwargs_165652)
    
    int_165654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 17), 'int')
    # Applying the binary operator '==' (line 498)
    result_eq_165655 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 7), '==', len_call_result_165653, int_165654)
    
    # Testing the type of an if condition (line 498)
    if_condition_165656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 498, 4), result_eq_165655)
    # Assigning a type to the variable 'if_condition_165656' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'if_condition_165656', if_condition_165656)
    # SSA begins for if statement (line 498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 499):
    
    # Assigning a BinOp to a Name (line 499):
    
    # Obtaining the type of the subscript
    int_165657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 15), 'int')
    # Getting the type of 'c' (line 499)
    c_165658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 499)
    getitem___165659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 13), c_165658, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 499)
    subscript_call_result_165660 = invoke(stypy.reporting.localization.Localization(__file__, 499, 13), getitem___165659, int_165657)
    
    # Getting the type of 'xs' (line 499)
    xs_165661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 18), 'xs')
    # Applying the binary operator '*' (line 499)
    result_mul_165662 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 13), '*', subscript_call_result_165660, xs_165661)
    
    # Assigning a type to the variable 'c0' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'c0', result_mul_165662)
    
    # Assigning a Num to a Name (line 500):
    
    # Assigning a Num to a Name (line 500):
    int_165663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 13), 'int')
    # Assigning a type to the variable 'c1' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'c1', int_165663)
    # SSA branch for the else part of an if statement (line 498)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 501)
    # Processing the call arguments (line 501)
    # Getting the type of 'c' (line 501)
    c_165665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 13), 'c', False)
    # Processing the call keyword arguments (line 501)
    kwargs_165666 = {}
    # Getting the type of 'len' (line 501)
    len_165664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 9), 'len', False)
    # Calling len(args, kwargs) (line 501)
    len_call_result_165667 = invoke(stypy.reporting.localization.Localization(__file__, 501, 9), len_165664, *[c_165665], **kwargs_165666)
    
    int_165668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 19), 'int')
    # Applying the binary operator '==' (line 501)
    result_eq_165669 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 9), '==', len_call_result_165667, int_165668)
    
    # Testing the type of an if condition (line 501)
    if_condition_165670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 9), result_eq_165669)
    # Assigning a type to the variable 'if_condition_165670' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 9), 'if_condition_165670', if_condition_165670)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 502):
    
    # Assigning a BinOp to a Name (line 502):
    
    # Obtaining the type of the subscript
    int_165671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 15), 'int')
    # Getting the type of 'c' (line 502)
    c_165672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___165673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 13), c_165672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_165674 = invoke(stypy.reporting.localization.Localization(__file__, 502, 13), getitem___165673, int_165671)
    
    # Getting the type of 'xs' (line 502)
    xs_165675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 18), 'xs')
    # Applying the binary operator '*' (line 502)
    result_mul_165676 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 13), '*', subscript_call_result_165674, xs_165675)
    
    # Assigning a type to the variable 'c0' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'c0', result_mul_165676)
    
    # Assigning a BinOp to a Name (line 503):
    
    # Assigning a BinOp to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_165677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 15), 'int')
    # Getting the type of 'c' (line 503)
    c_165678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___165679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 13), c_165678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_165680 = invoke(stypy.reporting.localization.Localization(__file__, 503, 13), getitem___165679, int_165677)
    
    # Getting the type of 'xs' (line 503)
    xs_165681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 18), 'xs')
    # Applying the binary operator '*' (line 503)
    result_mul_165682 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 13), '*', subscript_call_result_165680, xs_165681)
    
    # Assigning a type to the variable 'c1' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'c1', result_mul_165682)
    # SSA branch for the else part of an if statement (line 501)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 505):
    
    # Assigning a Call to a Name (line 505):
    
    # Call to len(...): (line 505)
    # Processing the call arguments (line 505)
    # Getting the type of 'c' (line 505)
    c_165684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 17), 'c', False)
    # Processing the call keyword arguments (line 505)
    kwargs_165685 = {}
    # Getting the type of 'len' (line 505)
    len_165683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 13), 'len', False)
    # Calling len(args, kwargs) (line 505)
    len_call_result_165686 = invoke(stypy.reporting.localization.Localization(__file__, 505, 13), len_165683, *[c_165684], **kwargs_165685)
    
    # Assigning a type to the variable 'nd' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'nd', len_call_result_165686)
    
    # Assigning a BinOp to a Name (line 506):
    
    # Assigning a BinOp to a Name (line 506):
    
    # Obtaining the type of the subscript
    int_165687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 15), 'int')
    # Getting the type of 'c' (line 506)
    c_165688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___165689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 13), c_165688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_165690 = invoke(stypy.reporting.localization.Localization(__file__, 506, 13), getitem___165689, int_165687)
    
    # Getting the type of 'xs' (line 506)
    xs_165691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 19), 'xs')
    # Applying the binary operator '*' (line 506)
    result_mul_165692 = python_operator(stypy.reporting.localization.Localization(__file__, 506, 13), '*', subscript_call_result_165690, xs_165691)
    
    # Assigning a type to the variable 'c0' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'c0', result_mul_165692)
    
    # Assigning a BinOp to a Name (line 507):
    
    # Assigning a BinOp to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_165693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 15), 'int')
    # Getting the type of 'c' (line 507)
    c_165694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___165695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 13), c_165694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_165696 = invoke(stypy.reporting.localization.Localization(__file__, 507, 13), getitem___165695, int_165693)
    
    # Getting the type of 'xs' (line 507)
    xs_165697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 19), 'xs')
    # Applying the binary operator '*' (line 507)
    result_mul_165698 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 13), '*', subscript_call_result_165696, xs_165697)
    
    # Assigning a type to the variable 'c1' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'c1', result_mul_165698)
    
    
    # Call to range(...): (line 508)
    # Processing the call arguments (line 508)
    int_165700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 23), 'int')
    
    # Call to len(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'c' (line 508)
    c_165702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 30), 'c', False)
    # Processing the call keyword arguments (line 508)
    kwargs_165703 = {}
    # Getting the type of 'len' (line 508)
    len_165701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 26), 'len', False)
    # Calling len(args, kwargs) (line 508)
    len_call_result_165704 = invoke(stypy.reporting.localization.Localization(__file__, 508, 26), len_165701, *[c_165702], **kwargs_165703)
    
    int_165705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 35), 'int')
    # Applying the binary operator '+' (line 508)
    result_add_165706 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 26), '+', len_call_result_165704, int_165705)
    
    # Processing the call keyword arguments (line 508)
    kwargs_165707 = {}
    # Getting the type of 'range' (line 508)
    range_165699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 17), 'range', False)
    # Calling range(args, kwargs) (line 508)
    range_call_result_165708 = invoke(stypy.reporting.localization.Localization(__file__, 508, 17), range_165699, *[int_165700, result_add_165706], **kwargs_165707)
    
    # Testing the type of a for loop iterable (line 508)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 508, 8), range_call_result_165708)
    # Getting the type of the for loop variable (line 508)
    for_loop_var_165709 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 508, 8), range_call_result_165708)
    # Assigning a type to the variable 'i' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'i', for_loop_var_165709)
    # SSA begins for a for statement (line 508)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 509):
    
    # Assigning a Name to a Name (line 509):
    # Getting the type of 'c0' (line 509)
    c0_165710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'tmp', c0_165710)
    
    # Assigning a BinOp to a Name (line 510):
    
    # Assigning a BinOp to a Name (line 510):
    # Getting the type of 'nd' (line 510)
    nd_165711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'nd')
    int_165712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 22), 'int')
    # Applying the binary operator '-' (line 510)
    result_sub_165713 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 17), '-', nd_165711, int_165712)
    
    # Assigning a type to the variable 'nd' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'nd', result_sub_165713)
    
    # Assigning a Call to a Name (line 511):
    
    # Assigning a Call to a Name (line 511):
    
    # Call to hermsub(...): (line 511)
    # Processing the call arguments (line 511)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 511)
    i_165715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 28), 'i', False)
    # Applying the 'usub' unary operator (line 511)
    result___neg___165716 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 27), 'usub', i_165715)
    
    # Getting the type of 'c' (line 511)
    c_165717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___165718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 25), c_165717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_165719 = invoke(stypy.reporting.localization.Localization(__file__, 511, 25), getitem___165718, result___neg___165716)
    
    # Getting the type of 'xs' (line 511)
    xs_165720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 31), 'xs', False)
    # Applying the binary operator '*' (line 511)
    result_mul_165721 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 25), '*', subscript_call_result_165719, xs_165720)
    
    # Getting the type of 'c1' (line 511)
    c1_165722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 35), 'c1', False)
    int_165723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 39), 'int')
    # Getting the type of 'nd' (line 511)
    nd_165724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 42), 'nd', False)
    int_165725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 47), 'int')
    # Applying the binary operator '-' (line 511)
    result_sub_165726 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 42), '-', nd_165724, int_165725)
    
    # Applying the binary operator '*' (line 511)
    result_mul_165727 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 39), '*', int_165723, result_sub_165726)
    
    # Applying the binary operator '*' (line 511)
    result_mul_165728 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 35), '*', c1_165722, result_mul_165727)
    
    # Processing the call keyword arguments (line 511)
    kwargs_165729 = {}
    # Getting the type of 'hermsub' (line 511)
    hermsub_165714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'hermsub', False)
    # Calling hermsub(args, kwargs) (line 511)
    hermsub_call_result_165730 = invoke(stypy.reporting.localization.Localization(__file__, 511, 17), hermsub_165714, *[result_mul_165721, result_mul_165728], **kwargs_165729)
    
    # Assigning a type to the variable 'c0' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'c0', hermsub_call_result_165730)
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to hermadd(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'tmp' (line 512)
    tmp_165732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 25), 'tmp', False)
    
    # Call to hermmulx(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'c1' (line 512)
    c1_165734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 39), 'c1', False)
    # Processing the call keyword arguments (line 512)
    kwargs_165735 = {}
    # Getting the type of 'hermmulx' (line 512)
    hermmulx_165733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 30), 'hermmulx', False)
    # Calling hermmulx(args, kwargs) (line 512)
    hermmulx_call_result_165736 = invoke(stypy.reporting.localization.Localization(__file__, 512, 30), hermmulx_165733, *[c1_165734], **kwargs_165735)
    
    int_165737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 43), 'int')
    # Applying the binary operator '*' (line 512)
    result_mul_165738 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 30), '*', hermmulx_call_result_165736, int_165737)
    
    # Processing the call keyword arguments (line 512)
    kwargs_165739 = {}
    # Getting the type of 'hermadd' (line 512)
    hermadd_165731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 17), 'hermadd', False)
    # Calling hermadd(args, kwargs) (line 512)
    hermadd_call_result_165740 = invoke(stypy.reporting.localization.Localization(__file__, 512, 17), hermadd_165731, *[tmp_165732, result_mul_165738], **kwargs_165739)
    
    # Assigning a type to the variable 'c1' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'c1', hermadd_call_result_165740)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 498)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to hermadd(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'c0' (line 513)
    c0_165742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 19), 'c0', False)
    
    # Call to hermmulx(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'c1' (line 513)
    c1_165744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 32), 'c1', False)
    # Processing the call keyword arguments (line 513)
    kwargs_165745 = {}
    # Getting the type of 'hermmulx' (line 513)
    hermmulx_165743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'hermmulx', False)
    # Calling hermmulx(args, kwargs) (line 513)
    hermmulx_call_result_165746 = invoke(stypy.reporting.localization.Localization(__file__, 513, 23), hermmulx_165743, *[c1_165744], **kwargs_165745)
    
    int_165747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 36), 'int')
    # Applying the binary operator '*' (line 513)
    result_mul_165748 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 23), '*', hermmulx_call_result_165746, int_165747)
    
    # Processing the call keyword arguments (line 513)
    kwargs_165749 = {}
    # Getting the type of 'hermadd' (line 513)
    hermadd_165741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 11), 'hermadd', False)
    # Calling hermadd(args, kwargs) (line 513)
    hermadd_call_result_165750 = invoke(stypy.reporting.localization.Localization(__file__, 513, 11), hermadd_165741, *[c0_165742, result_mul_165748], **kwargs_165749)
    
    # Assigning a type to the variable 'stypy_return_type' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'stypy_return_type', hermadd_call_result_165750)
    
    # ################# End of 'hermmul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermmul' in the type store
    # Getting the type of 'stypy_return_type' (line 450)
    stypy_return_type_165751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermmul'
    return stypy_return_type_165751

# Assigning a type to the variable 'hermmul' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'hermmul', hermmul)

@norecursion
def hermdiv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermdiv'
    module_type_store = module_type_store.open_function_context('hermdiv', 516, 0, False)
    
    # Passed parameters checking function
    hermdiv.stypy_localization = localization
    hermdiv.stypy_type_of_self = None
    hermdiv.stypy_type_store = module_type_store
    hermdiv.stypy_function_name = 'hermdiv'
    hermdiv.stypy_param_names_list = ['c1', 'c2']
    hermdiv.stypy_varargs_param_name = None
    hermdiv.stypy_kwargs_param_name = None
    hermdiv.stypy_call_defaults = defaults
    hermdiv.stypy_call_varargs = varargs
    hermdiv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermdiv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermdiv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermdiv(...)' code ##################

    str_165752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, (-1)), 'str', '\n    Divide one Hermite series by another.\n\n    Returns the quotient-with-remainder of two Hermite series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of Hermite series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    hermadd, hermsub, hermmul, hermpow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one Hermite series by another\n    results in quotient and remainder terms that are not in the Hermite\n    polynomial basis set.  Thus, to express these results as a Hermite\n    series, it is necessary to "reproject" the results onto the Hermite\n    basis set, which may produce "unintuitive" (but correct) results; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermdiv\n    >>> hermdiv([ 52.,  29.,  52.,   7.,   6.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 0.]))\n    >>> hermdiv([ 54.,  31.,  52.,   7.,   6.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 2.,  2.]))\n    >>> hermdiv([ 53.,  30.,  52.,   7.,   6.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 1.,  1.]))\n\n    ')
    
    # Assigning a Call to a List (line 562):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining an instance of the builtin type 'list' (line 562)
    list_165755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 562)
    # Adding element type (line 562)
    # Getting the type of 'c1' (line 562)
    c1_165756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 28), list_165755, c1_165756)
    # Adding element type (line 562)
    # Getting the type of 'c2' (line 562)
    c2_165757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 562, 28), list_165755, c2_165757)
    
    # Processing the call keyword arguments (line 562)
    kwargs_165758 = {}
    # Getting the type of 'pu' (line 562)
    pu_165753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 562)
    as_series_165754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 15), pu_165753, 'as_series')
    # Calling as_series(args, kwargs) (line 562)
    as_series_call_result_165759 = invoke(stypy.reporting.localization.Localization(__file__, 562, 15), as_series_165754, *[list_165755], **kwargs_165758)
    
    # Assigning a type to the variable 'call_assignment_165025' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165025', as_series_call_result_165759)
    
    # Assigning a Call to a Name (line 562):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165763 = {}
    # Getting the type of 'call_assignment_165025' (line 562)
    call_assignment_165025_165760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165025', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___165761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 4), call_assignment_165025_165760, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165764 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165761, *[int_165762], **kwargs_165763)
    
    # Assigning a type to the variable 'call_assignment_165026' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165026', getitem___call_result_165764)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'call_assignment_165026' (line 562)
    call_assignment_165026_165765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165026')
    # Assigning a type to the variable 'c1' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 5), 'c1', call_assignment_165026_165765)
    
    # Assigning a Call to a Name (line 562):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165769 = {}
    # Getting the type of 'call_assignment_165025' (line 562)
    call_assignment_165025_165766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165025', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___165767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 4), call_assignment_165025_165766, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165770 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165767, *[int_165768], **kwargs_165769)
    
    # Assigning a type to the variable 'call_assignment_165027' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165027', getitem___call_result_165770)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'call_assignment_165027' (line 562)
    call_assignment_165027_165771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'call_assignment_165027')
    # Assigning a type to the variable 'c2' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 9), 'c2', call_assignment_165027_165771)
    
    
    
    # Obtaining the type of the subscript
    int_165772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 10), 'int')
    # Getting the type of 'c2' (line 563)
    c2_165773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___165774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 7), c2_165773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_165775 = invoke(stypy.reporting.localization.Localization(__file__, 563, 7), getitem___165774, int_165772)
    
    int_165776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 17), 'int')
    # Applying the binary operator '==' (line 563)
    result_eq_165777 = python_operator(stypy.reporting.localization.Localization(__file__, 563, 7), '==', subscript_call_result_165775, int_165776)
    
    # Testing the type of an if condition (line 563)
    if_condition_165778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 563, 4), result_eq_165777)
    # Assigning a type to the variable 'if_condition_165778' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'if_condition_165778', if_condition_165778)
    # SSA begins for if statement (line 563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 564)
    # Processing the call keyword arguments (line 564)
    kwargs_165780 = {}
    # Getting the type of 'ZeroDivisionError' (line 564)
    ZeroDivisionError_165779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 564)
    ZeroDivisionError_call_result_165781 = invoke(stypy.reporting.localization.Localization(__file__, 564, 14), ZeroDivisionError_165779, *[], **kwargs_165780)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 564, 8), ZeroDivisionError_call_result_165781, 'raise parameter', BaseException)
    # SSA join for if statement (line 563)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to len(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'c1' (line 566)
    c1_165783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 14), 'c1', False)
    # Processing the call keyword arguments (line 566)
    kwargs_165784 = {}
    # Getting the type of 'len' (line 566)
    len_165782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 10), 'len', False)
    # Calling len(args, kwargs) (line 566)
    len_call_result_165785 = invoke(stypy.reporting.localization.Localization(__file__, 566, 10), len_165782, *[c1_165783], **kwargs_165784)
    
    # Assigning a type to the variable 'lc1' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'lc1', len_call_result_165785)
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to len(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'c2' (line 567)
    c2_165787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 14), 'c2', False)
    # Processing the call keyword arguments (line 567)
    kwargs_165788 = {}
    # Getting the type of 'len' (line 567)
    len_165786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 10), 'len', False)
    # Calling len(args, kwargs) (line 567)
    len_call_result_165789 = invoke(stypy.reporting.localization.Localization(__file__, 567, 10), len_165786, *[c2_165787], **kwargs_165788)
    
    # Assigning a type to the variable 'lc2' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'lc2', len_call_result_165789)
    
    
    # Getting the type of 'lc1' (line 568)
    lc1_165790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 7), 'lc1')
    # Getting the type of 'lc2' (line 568)
    lc2_165791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 13), 'lc2')
    # Applying the binary operator '<' (line 568)
    result_lt_165792 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 7), '<', lc1_165790, lc2_165791)
    
    # Testing the type of an if condition (line 568)
    if_condition_165793 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 4), result_lt_165792)
    # Assigning a type to the variable 'if_condition_165793' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'if_condition_165793', if_condition_165793)
    # SSA begins for if statement (line 568)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 569)
    tuple_165794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 569)
    # Adding element type (line 569)
    
    # Obtaining the type of the subscript
    int_165795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 19), 'int')
    slice_165796 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 569, 15), None, int_165795, None)
    # Getting the type of 'c1' (line 569)
    c1_165797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___165798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 15), c1_165797, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_165799 = invoke(stypy.reporting.localization.Localization(__file__, 569, 15), getitem___165798, slice_165796)
    
    int_165800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 22), 'int')
    # Applying the binary operator '*' (line 569)
    result_mul_165801 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 15), '*', subscript_call_result_165799, int_165800)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 15), tuple_165794, result_mul_165801)
    # Adding element type (line 569)
    # Getting the type of 'c1' (line 569)
    c1_165802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 569, 15), tuple_165794, c1_165802)
    
    # Assigning a type to the variable 'stypy_return_type' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'stypy_return_type', tuple_165794)
    # SSA branch for the else part of an if statement (line 568)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lc2' (line 570)
    lc2_165803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 9), 'lc2')
    int_165804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 16), 'int')
    # Applying the binary operator '==' (line 570)
    result_eq_165805 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 9), '==', lc2_165803, int_165804)
    
    # Testing the type of an if condition (line 570)
    if_condition_165806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 570, 9), result_eq_165805)
    # Assigning a type to the variable 'if_condition_165806' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 9), 'if_condition_165806', if_condition_165806)
    # SSA begins for if statement (line 570)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 571)
    tuple_165807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 571)
    # Adding element type (line 571)
    # Getting the type of 'c1' (line 571)
    c1_165808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_165809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 21), 'int')
    # Getting the type of 'c2' (line 571)
    c2_165810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 571)
    getitem___165811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 18), c2_165810, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 571)
    subscript_call_result_165812 = invoke(stypy.reporting.localization.Localization(__file__, 571, 18), getitem___165811, int_165809)
    
    # Applying the binary operator 'div' (line 571)
    result_div_165813 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 15), 'div', c1_165808, subscript_call_result_165812)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 15), tuple_165807, result_div_165813)
    # Adding element type (line 571)
    
    # Obtaining the type of the subscript
    int_165814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 30), 'int')
    slice_165815 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 571, 26), None, int_165814, None)
    # Getting the type of 'c1' (line 571)
    c1_165816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 571)
    getitem___165817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 26), c1_165816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 571)
    subscript_call_result_165818 = invoke(stypy.reporting.localization.Localization(__file__, 571, 26), getitem___165817, slice_165815)
    
    int_165819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 33), 'int')
    # Applying the binary operator '*' (line 571)
    result_mul_165820 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 26), '*', subscript_call_result_165818, int_165819)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 571, 15), tuple_165807, result_mul_165820)
    
    # Assigning a type to the variable 'stypy_return_type' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'stypy_return_type', tuple_165807)
    # SSA branch for the else part of an if statement (line 570)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to empty(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'lc1' (line 573)
    lc1_165823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 573)
    lc2_165824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 29), 'lc2', False)
    # Applying the binary operator '-' (line 573)
    result_sub_165825 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 23), '-', lc1_165823, lc2_165824)
    
    int_165826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 35), 'int')
    # Applying the binary operator '+' (line 573)
    result_add_165827 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 33), '+', result_sub_165825, int_165826)
    
    # Processing the call keyword arguments (line 573)
    # Getting the type of 'c1' (line 573)
    c1_165828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 44), 'c1', False)
    # Obtaining the member 'dtype' of a type (line 573)
    dtype_165829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 44), c1_165828, 'dtype')
    keyword_165830 = dtype_165829
    kwargs_165831 = {'dtype': keyword_165830}
    # Getting the type of 'np' (line 573)
    np_165821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 573)
    empty_165822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 14), np_165821, 'empty')
    # Calling empty(args, kwargs) (line 573)
    empty_call_result_165832 = invoke(stypy.reporting.localization.Localization(__file__, 573, 14), empty_165822, *[result_add_165827], **kwargs_165831)
    
    # Assigning a type to the variable 'quo' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'quo', empty_call_result_165832)
    
    # Assigning a Name to a Name (line 574):
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'c1' (line 574)
    c1_165833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 14), 'c1')
    # Assigning a type to the variable 'rem' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'rem', c1_165833)
    
    
    # Call to range(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'lc1' (line 575)
    lc1_165835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 575)
    lc2_165836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'lc2', False)
    # Applying the binary operator '-' (line 575)
    result_sub_165837 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 23), '-', lc1_165835, lc2_165836)
    
    int_165838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 34), 'int')
    int_165839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 39), 'int')
    # Processing the call keyword arguments (line 575)
    kwargs_165840 = {}
    # Getting the type of 'range' (line 575)
    range_165834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 17), 'range', False)
    # Calling range(args, kwargs) (line 575)
    range_call_result_165841 = invoke(stypy.reporting.localization.Localization(__file__, 575, 17), range_165834, *[result_sub_165837, int_165838, int_165839], **kwargs_165840)
    
    # Testing the type of a for loop iterable (line 575)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 575, 8), range_call_result_165841)
    # Getting the type of the for loop variable (line 575)
    for_loop_var_165842 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 575, 8), range_call_result_165841)
    # Assigning a type to the variable 'i' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'i', for_loop_var_165842)
    # SSA begins for a for statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to hermmul(...): (line 576)
    # Processing the call arguments (line 576)
    
    # Obtaining an instance of the builtin type 'list' (line 576)
    list_165844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 576)
    # Adding element type (line 576)
    int_165845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 24), list_165844, int_165845)
    
    # Getting the type of 'i' (line 576)
    i_165846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 28), 'i', False)
    # Applying the binary operator '*' (line 576)
    result_mul_165847 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 24), '*', list_165844, i_165846)
    
    
    # Obtaining an instance of the builtin type 'list' (line 576)
    list_165848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 576)
    # Adding element type (line 576)
    int_165849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 32), list_165848, int_165849)
    
    # Applying the binary operator '+' (line 576)
    result_add_165850 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 24), '+', result_mul_165847, list_165848)
    
    # Getting the type of 'c2' (line 576)
    c2_165851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 37), 'c2', False)
    # Processing the call keyword arguments (line 576)
    kwargs_165852 = {}
    # Getting the type of 'hermmul' (line 576)
    hermmul_165843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'hermmul', False)
    # Calling hermmul(args, kwargs) (line 576)
    hermmul_call_result_165853 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), hermmul_165843, *[result_add_165850, c2_165851], **kwargs_165852)
    
    # Assigning a type to the variable 'p' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'p', hermmul_call_result_165853)
    
    # Assigning a BinOp to a Name (line 577):
    
    # Assigning a BinOp to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_165854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 20), 'int')
    # Getting the type of 'rem' (line 577)
    rem_165855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'rem')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___165856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), rem_165855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_165857 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___165856, int_165854)
    
    
    # Obtaining the type of the subscript
    int_165858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 26), 'int')
    # Getting the type of 'p' (line 577)
    p_165859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 24), 'p')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___165860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 24), p_165859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_165861 = invoke(stypy.reporting.localization.Localization(__file__, 577, 24), getitem___165860, int_165858)
    
    # Applying the binary operator 'div' (line 577)
    result_div_165862 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 16), 'div', subscript_call_result_165857, subscript_call_result_165861)
    
    # Assigning a type to the variable 'q' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'q', result_div_165862)
    
    # Assigning a BinOp to a Name (line 578):
    
    # Assigning a BinOp to a Name (line 578):
    
    # Obtaining the type of the subscript
    int_165863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 23), 'int')
    slice_165864 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 578, 18), None, int_165863, None)
    # Getting the type of 'rem' (line 578)
    rem_165865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 18), 'rem')
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___165866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 18), rem_165865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_165867 = invoke(stypy.reporting.localization.Localization(__file__, 578, 18), getitem___165866, slice_165864)
    
    # Getting the type of 'q' (line 578)
    q_165868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 29), 'q')
    
    # Obtaining the type of the subscript
    int_165869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 34), 'int')
    slice_165870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 578, 31), None, int_165869, None)
    # Getting the type of 'p' (line 578)
    p_165871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 31), 'p')
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___165872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 31), p_165871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_165873 = invoke(stypy.reporting.localization.Localization(__file__, 578, 31), getitem___165872, slice_165870)
    
    # Applying the binary operator '*' (line 578)
    result_mul_165874 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 29), '*', q_165868, subscript_call_result_165873)
    
    # Applying the binary operator '-' (line 578)
    result_sub_165875 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 18), '-', subscript_call_result_165867, result_mul_165874)
    
    # Assigning a type to the variable 'rem' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'rem', result_sub_165875)
    
    # Assigning a Name to a Subscript (line 579):
    
    # Assigning a Name to a Subscript (line 579):
    # Getting the type of 'q' (line 579)
    q_165876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 21), 'q')
    # Getting the type of 'quo' (line 579)
    quo_165877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'quo')
    # Getting the type of 'i' (line 579)
    i_165878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'i')
    # Storing an element on a container (line 579)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 12), quo_165877, (i_165878, q_165876))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 580)
    tuple_165879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 580)
    # Adding element type (line 580)
    # Getting the type of 'quo' (line 580)
    quo_165880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 15), tuple_165879, quo_165880)
    # Adding element type (line 580)
    
    # Call to trimseq(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'rem' (line 580)
    rem_165883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 31), 'rem', False)
    # Processing the call keyword arguments (line 580)
    kwargs_165884 = {}
    # Getting the type of 'pu' (line 580)
    pu_165881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 20), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 580)
    trimseq_165882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 20), pu_165881, 'trimseq')
    # Calling trimseq(args, kwargs) (line 580)
    trimseq_call_result_165885 = invoke(stypy.reporting.localization.Localization(__file__, 580, 20), trimseq_165882, *[rem_165883], **kwargs_165884)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 15), tuple_165879, trimseq_call_result_165885)
    
    # Assigning a type to the variable 'stypy_return_type' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'stypy_return_type', tuple_165879)
    # SSA join for if statement (line 570)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 568)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermdiv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermdiv' in the type store
    # Getting the type of 'stypy_return_type' (line 516)
    stypy_return_type_165886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165886)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermdiv'
    return stypy_return_type_165886

# Assigning a type to the variable 'hermdiv' (line 516)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 0), 'hermdiv', hermdiv)

@norecursion
def hermpow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_165887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 29), 'int')
    defaults = [int_165887]
    # Create a new context for function 'hermpow'
    module_type_store = module_type_store.open_function_context('hermpow', 583, 0, False)
    
    # Passed parameters checking function
    hermpow.stypy_localization = localization
    hermpow.stypy_type_of_self = None
    hermpow.stypy_type_store = module_type_store
    hermpow.stypy_function_name = 'hermpow'
    hermpow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    hermpow.stypy_varargs_param_name = None
    hermpow.stypy_kwargs_param_name = None
    hermpow.stypy_call_defaults = defaults
    hermpow.stypy_call_varargs = varargs
    hermpow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermpow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermpow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermpow(...)' code ##################

    str_165888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, (-1)), 'str', 'Raise a Hermite series to a power.\n\n    Returns the Hermite series `c` raised to the power `pow`. The\n    argument `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Hermite series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Hermite series of power.\n\n    See Also\n    --------\n    hermadd, hermsub, hermmul, hermdiv\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermpow\n    >>> hermpow([1, 2, 3], 2)\n    array([ 81.,  52.,  82.,  12.,   9.])\n\n    ')
    
    # Assigning a Call to a List (line 618):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 618)
    # Processing the call arguments (line 618)
    
    # Obtaining an instance of the builtin type 'list' (line 618)
    list_165891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 618)
    # Adding element type (line 618)
    # Getting the type of 'c' (line 618)
    c_165892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 23), list_165891, c_165892)
    
    # Processing the call keyword arguments (line 618)
    kwargs_165893 = {}
    # Getting the type of 'pu' (line 618)
    pu_165889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 618)
    as_series_165890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 10), pu_165889, 'as_series')
    # Calling as_series(args, kwargs) (line 618)
    as_series_call_result_165894 = invoke(stypy.reporting.localization.Localization(__file__, 618, 10), as_series_165890, *[list_165891], **kwargs_165893)
    
    # Assigning a type to the variable 'call_assignment_165028' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_165028', as_series_call_result_165894)
    
    # Assigning a Call to a Name (line 618):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_165897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    # Processing the call keyword arguments
    kwargs_165898 = {}
    # Getting the type of 'call_assignment_165028' (line 618)
    call_assignment_165028_165895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_165028', False)
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___165896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), call_assignment_165028_165895, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_165899 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___165896, *[int_165897], **kwargs_165898)
    
    # Assigning a type to the variable 'call_assignment_165029' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_165029', getitem___call_result_165899)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'call_assignment_165029' (line 618)
    call_assignment_165029_165900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'call_assignment_165029')
    # Assigning a type to the variable 'c' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 5), 'c', call_assignment_165029_165900)
    
    # Assigning a Call to a Name (line 619):
    
    # Assigning a Call to a Name (line 619):
    
    # Call to int(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'pow' (line 619)
    pow_165902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'pow', False)
    # Processing the call keyword arguments (line 619)
    kwargs_165903 = {}
    # Getting the type of 'int' (line 619)
    int_165901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'int', False)
    # Calling int(args, kwargs) (line 619)
    int_call_result_165904 = invoke(stypy.reporting.localization.Localization(__file__, 619, 12), int_165901, *[pow_165902], **kwargs_165903)
    
    # Assigning a type to the variable 'power' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'power', int_call_result_165904)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 620)
    power_165905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 7), 'power')
    # Getting the type of 'pow' (line 620)
    pow_165906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 16), 'pow')
    # Applying the binary operator '!=' (line 620)
    result_ne_165907 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 7), '!=', power_165905, pow_165906)
    
    
    # Getting the type of 'power' (line 620)
    power_165908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 23), 'power')
    int_165909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 31), 'int')
    # Applying the binary operator '<' (line 620)
    result_lt_165910 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 23), '<', power_165908, int_165909)
    
    # Applying the binary operator 'or' (line 620)
    result_or_keyword_165911 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 7), 'or', result_ne_165907, result_lt_165910)
    
    # Testing the type of an if condition (line 620)
    if_condition_165912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 4), result_or_keyword_165911)
    # Assigning a type to the variable 'if_condition_165912' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'if_condition_165912', if_condition_165912)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 621)
    # Processing the call arguments (line 621)
    str_165914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 621)
    kwargs_165915 = {}
    # Getting the type of 'ValueError' (line 621)
    ValueError_165913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 621)
    ValueError_call_result_165916 = invoke(stypy.reporting.localization.Localization(__file__, 621, 14), ValueError_165913, *[str_165914], **kwargs_165915)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 621, 8), ValueError_call_result_165916, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 620)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 622)
    maxpower_165917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 9), 'maxpower')
    # Getting the type of 'None' (line 622)
    None_165918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 25), 'None')
    # Applying the binary operator 'isnot' (line 622)
    result_is_not_165919 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 9), 'isnot', maxpower_165917, None_165918)
    
    
    # Getting the type of 'power' (line 622)
    power_165920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 34), 'power')
    # Getting the type of 'maxpower' (line 622)
    maxpower_165921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 42), 'maxpower')
    # Applying the binary operator '>' (line 622)
    result_gt_165922 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 34), '>', power_165920, maxpower_165921)
    
    # Applying the binary operator 'and' (line 622)
    result_and_keyword_165923 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 9), 'and', result_is_not_165919, result_gt_165922)
    
    # Testing the type of an if condition (line 622)
    if_condition_165924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 9), result_and_keyword_165923)
    # Assigning a type to the variable 'if_condition_165924' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 9), 'if_condition_165924', if_condition_165924)
    # SSA begins for if statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 623)
    # Processing the call arguments (line 623)
    str_165926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 623)
    kwargs_165927 = {}
    # Getting the type of 'ValueError' (line 623)
    ValueError_165925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 623)
    ValueError_call_result_165928 = invoke(stypy.reporting.localization.Localization(__file__, 623, 14), ValueError_165925, *[str_165926], **kwargs_165927)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 623, 8), ValueError_call_result_165928, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 622)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 624)
    power_165929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'power')
    int_165930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 18), 'int')
    # Applying the binary operator '==' (line 624)
    result_eq_165931 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 9), '==', power_165929, int_165930)
    
    # Testing the type of an if condition (line 624)
    if_condition_165932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 9), result_eq_165931)
    # Assigning a type to the variable 'if_condition_165932' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'if_condition_165932', if_condition_165932)
    # SSA begins for if statement (line 624)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 625)
    # Processing the call arguments (line 625)
    
    # Obtaining an instance of the builtin type 'list' (line 625)
    list_165935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 625)
    # Adding element type (line 625)
    int_165936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 625, 24), list_165935, int_165936)
    
    # Processing the call keyword arguments (line 625)
    # Getting the type of 'c' (line 625)
    c_165937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 625)
    dtype_165938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 35), c_165937, 'dtype')
    keyword_165939 = dtype_165938
    kwargs_165940 = {'dtype': keyword_165939}
    # Getting the type of 'np' (line 625)
    np_165933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 625)
    array_165934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 15), np_165933, 'array')
    # Calling array(args, kwargs) (line 625)
    array_call_result_165941 = invoke(stypy.reporting.localization.Localization(__file__, 625, 15), array_165934, *[list_165935], **kwargs_165940)
    
    # Assigning a type to the variable 'stypy_return_type' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'stypy_return_type', array_call_result_165941)
    # SSA branch for the else part of an if statement (line 624)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 626)
    power_165942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 9), 'power')
    int_165943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 18), 'int')
    # Applying the binary operator '==' (line 626)
    result_eq_165944 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 9), '==', power_165942, int_165943)
    
    # Testing the type of an if condition (line 626)
    if_condition_165945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 9), result_eq_165944)
    # Assigning a type to the variable 'if_condition_165945' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 9), 'if_condition_165945', if_condition_165945)
    # SSA begins for if statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 627)
    c_165946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'stypy_return_type', c_165946)
    # SSA branch for the else part of an if statement (line 626)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 631):
    
    # Assigning a Name to a Name (line 631):
    # Getting the type of 'c' (line 631)
    c_165947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 14), 'c')
    # Assigning a type to the variable 'prd' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'prd', c_165947)
    
    
    # Call to range(...): (line 632)
    # Processing the call arguments (line 632)
    int_165949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 23), 'int')
    # Getting the type of 'power' (line 632)
    power_165950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 26), 'power', False)
    int_165951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 34), 'int')
    # Applying the binary operator '+' (line 632)
    result_add_165952 = python_operator(stypy.reporting.localization.Localization(__file__, 632, 26), '+', power_165950, int_165951)
    
    # Processing the call keyword arguments (line 632)
    kwargs_165953 = {}
    # Getting the type of 'range' (line 632)
    range_165948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 17), 'range', False)
    # Calling range(args, kwargs) (line 632)
    range_call_result_165954 = invoke(stypy.reporting.localization.Localization(__file__, 632, 17), range_165948, *[int_165949, result_add_165952], **kwargs_165953)
    
    # Testing the type of a for loop iterable (line 632)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 632, 8), range_call_result_165954)
    # Getting the type of the for loop variable (line 632)
    for_loop_var_165955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 632, 8), range_call_result_165954)
    # Assigning a type to the variable 'i' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'i', for_loop_var_165955)
    # SSA begins for a for statement (line 632)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 633):
    
    # Assigning a Call to a Name (line 633):
    
    # Call to hermmul(...): (line 633)
    # Processing the call arguments (line 633)
    # Getting the type of 'prd' (line 633)
    prd_165957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 26), 'prd', False)
    # Getting the type of 'c' (line 633)
    c_165958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 31), 'c', False)
    # Processing the call keyword arguments (line 633)
    kwargs_165959 = {}
    # Getting the type of 'hermmul' (line 633)
    hermmul_165956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 18), 'hermmul', False)
    # Calling hermmul(args, kwargs) (line 633)
    hermmul_call_result_165960 = invoke(stypy.reporting.localization.Localization(__file__, 633, 18), hermmul_165956, *[prd_165957, c_165958], **kwargs_165959)
    
    # Assigning a type to the variable 'prd' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'prd', hermmul_call_result_165960)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 634)
    prd_165961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 15), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'stypy_return_type', prd_165961)
    # SSA join for if statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 624)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 622)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermpow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermpow' in the type store
    # Getting the type of 'stypy_return_type' (line 583)
    stypy_return_type_165962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165962)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermpow'
    return stypy_return_type_165962

# Assigning a type to the variable 'hermpow' (line 583)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 0), 'hermpow', hermpow)

@norecursion
def hermder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_165963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 17), 'int')
    int_165964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 24), 'int')
    int_165965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 32), 'int')
    defaults = [int_165963, int_165964, int_165965]
    # Create a new context for function 'hermder'
    module_type_store = module_type_store.open_function_context('hermder', 637, 0, False)
    
    # Passed parameters checking function
    hermder.stypy_localization = localization
    hermder.stypy_type_of_self = None
    hermder.stypy_type_store = module_type_store
    hermder.stypy_function_name = 'hermder'
    hermder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    hermder.stypy_varargs_param_name = None
    hermder.stypy_kwargs_param_name = None
    hermder.stypy_call_defaults = defaults
    hermder.stypy_call_varargs = varargs
    hermder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermder(...)' code ##################

    str_165966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, (-1)), 'str', '\n    Differentiate a Hermite series.\n\n    Returns the Hermite series coefficients `c` differentiated `m` times\n    along `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*H_0 + 2*H_1 + 3*H_2``\n    while [[1,2],[1,2]] represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) +\n    2*H_0(x)*H_1(y) + 2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is\n    ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Hermite series coefficients. If `c` is multidimensional the\n        different axis correspond to different variables with the degree in\n        each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Hermite series of the derivative.\n\n    See Also\n    --------\n    hermint\n\n    Notes\n    -----\n    In general, the result of differentiating a Hermite series does not\n    resemble the same operation on a power series. Thus the result of this\n    function may be "unintuitive," albeit correct; see Examples section\n    below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermder\n    >>> hermder([ 1. ,  0.5,  0.5,  0.5])\n    array([ 1.,  2.,  3.])\n    >>> hermder([-0.5,  1./2.,  1./8.,  1./12.,  1./16.], m=2)\n    array([ 1.,  2.,  3.])\n\n    ')
    
    # Assigning a Call to a Name (line 692):
    
    # Assigning a Call to a Name (line 692):
    
    # Call to array(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'c' (line 692)
    c_165969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 17), 'c', False)
    # Processing the call keyword arguments (line 692)
    int_165970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 26), 'int')
    keyword_165971 = int_165970
    int_165972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 34), 'int')
    keyword_165973 = int_165972
    kwargs_165974 = {'copy': keyword_165973, 'ndmin': keyword_165971}
    # Getting the type of 'np' (line 692)
    np_165967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 692)
    array_165968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 8), np_165967, 'array')
    # Calling array(args, kwargs) (line 692)
    array_call_result_165975 = invoke(stypy.reporting.localization.Localization(__file__, 692, 8), array_165968, *[c_165969], **kwargs_165974)
    
    # Assigning a type to the variable 'c' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'c', array_call_result_165975)
    
    
    # Getting the type of 'c' (line 693)
    c_165976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 693)
    dtype_165977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 7), c_165976, 'dtype')
    # Obtaining the member 'char' of a type (line 693)
    char_165978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 7), dtype_165977, 'char')
    str_165979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 693)
    result_contains_165980 = python_operator(stypy.reporting.localization.Localization(__file__, 693, 7), 'in', char_165978, str_165979)
    
    # Testing the type of an if condition (line 693)
    if_condition_165981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 693, 4), result_contains_165980)
    # Assigning a type to the variable 'if_condition_165981' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'if_condition_165981', if_condition_165981)
    # SSA begins for if statement (line 693)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 694):
    
    # Assigning a Call to a Name (line 694):
    
    # Call to astype(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'np' (line 694)
    np_165984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 694)
    double_165985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 21), np_165984, 'double')
    # Processing the call keyword arguments (line 694)
    kwargs_165986 = {}
    # Getting the type of 'c' (line 694)
    c_165982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 694)
    astype_165983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 12), c_165982, 'astype')
    # Calling astype(args, kwargs) (line 694)
    astype_call_result_165987 = invoke(stypy.reporting.localization.Localization(__file__, 694, 12), astype_165983, *[double_165985], **kwargs_165986)
    
    # Assigning a type to the variable 'c' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'c', astype_call_result_165987)
    # SSA join for if statement (line 693)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 695):
    
    # Assigning a Subscript to a Name (line 695):
    
    # Obtaining the type of the subscript
    int_165988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 695)
    list_165993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 695)
    # Adding element type (line 695)
    # Getting the type of 'm' (line 695)
    m_165994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 34), list_165993, m_165994)
    # Adding element type (line 695)
    # Getting the type of 'axis' (line 695)
    axis_165995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 34), list_165993, axis_165995)
    
    comprehension_165996 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 18), list_165993)
    # Assigning a type to the variable 't' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 't', comprehension_165996)
    
    # Call to int(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 't' (line 695)
    t_165990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 22), 't', False)
    # Processing the call keyword arguments (line 695)
    kwargs_165991 = {}
    # Getting the type of 'int' (line 695)
    int_165989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'int', False)
    # Calling int(args, kwargs) (line 695)
    int_call_result_165992 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), int_165989, *[t_165990], **kwargs_165991)
    
    list_165997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 18), list_165997, int_call_result_165992)
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___165998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 4), list_165997, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_165999 = invoke(stypy.reporting.localization.Localization(__file__, 695, 4), getitem___165998, int_165988)
    
    # Assigning a type to the variable 'tuple_var_assignment_165030' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_165030', subscript_call_result_165999)
    
    # Assigning a Subscript to a Name (line 695):
    
    # Obtaining the type of the subscript
    int_166000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 695)
    list_166005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 695)
    # Adding element type (line 695)
    # Getting the type of 'm' (line 695)
    m_166006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 34), list_166005, m_166006)
    # Adding element type (line 695)
    # Getting the type of 'axis' (line 695)
    axis_166007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 34), list_166005, axis_166007)
    
    comprehension_166008 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 18), list_166005)
    # Assigning a type to the variable 't' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 't', comprehension_166008)
    
    # Call to int(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 't' (line 695)
    t_166002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 22), 't', False)
    # Processing the call keyword arguments (line 695)
    kwargs_166003 = {}
    # Getting the type of 'int' (line 695)
    int_166001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 18), 'int', False)
    # Calling int(args, kwargs) (line 695)
    int_call_result_166004 = invoke(stypy.reporting.localization.Localization(__file__, 695, 18), int_166001, *[t_166002], **kwargs_166003)
    
    list_166009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 18), list_166009, int_call_result_166004)
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___166010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 4), list_166009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_166011 = invoke(stypy.reporting.localization.Localization(__file__, 695, 4), getitem___166010, int_166000)
    
    # Assigning a type to the variable 'tuple_var_assignment_165031' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_165031', subscript_call_result_166011)
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'tuple_var_assignment_165030' (line 695)
    tuple_var_assignment_165030_166012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_165030')
    # Assigning a type to the variable 'cnt' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'cnt', tuple_var_assignment_165030_166012)
    
    # Assigning a Name to a Name (line 695):
    # Getting the type of 'tuple_var_assignment_165031' (line 695)
    tuple_var_assignment_165031_166013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'tuple_var_assignment_165031')
    # Assigning a type to the variable 'iaxis' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 9), 'iaxis', tuple_var_assignment_165031_166013)
    
    
    # Getting the type of 'cnt' (line 697)
    cnt_166014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 7), 'cnt')
    # Getting the type of 'm' (line 697)
    m_166015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 14), 'm')
    # Applying the binary operator '!=' (line 697)
    result_ne_166016 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 7), '!=', cnt_166014, m_166015)
    
    # Testing the type of an if condition (line 697)
    if_condition_166017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 4), result_ne_166016)
    # Assigning a type to the variable 'if_condition_166017' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'if_condition_166017', if_condition_166017)
    # SSA begins for if statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 698)
    # Processing the call arguments (line 698)
    str_166019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 698)
    kwargs_166020 = {}
    # Getting the type of 'ValueError' (line 698)
    ValueError_166018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 698)
    ValueError_call_result_166021 = invoke(stypy.reporting.localization.Localization(__file__, 698, 14), ValueError_166018, *[str_166019], **kwargs_166020)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 698, 8), ValueError_call_result_166021, 'raise parameter', BaseException)
    # SSA join for if statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 699)
    cnt_166022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 7), 'cnt')
    int_166023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 13), 'int')
    # Applying the binary operator '<' (line 699)
    result_lt_166024 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 7), '<', cnt_166022, int_166023)
    
    # Testing the type of an if condition (line 699)
    if_condition_166025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 699, 4), result_lt_166024)
    # Assigning a type to the variable 'if_condition_166025' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'if_condition_166025', if_condition_166025)
    # SSA begins for if statement (line 699)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 700)
    # Processing the call arguments (line 700)
    str_166027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 700)
    kwargs_166028 = {}
    # Getting the type of 'ValueError' (line 700)
    ValueError_166026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 700)
    ValueError_call_result_166029 = invoke(stypy.reporting.localization.Localization(__file__, 700, 14), ValueError_166026, *[str_166027], **kwargs_166028)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 700, 8), ValueError_call_result_166029, 'raise parameter', BaseException)
    # SSA join for if statement (line 699)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 701)
    iaxis_166030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 7), 'iaxis')
    # Getting the type of 'axis' (line 701)
    axis_166031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'axis')
    # Applying the binary operator '!=' (line 701)
    result_ne_166032 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 7), '!=', iaxis_166030, axis_166031)
    
    # Testing the type of an if condition (line 701)
    if_condition_166033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 701, 4), result_ne_166032)
    # Assigning a type to the variable 'if_condition_166033' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'if_condition_166033', if_condition_166033)
    # SSA begins for if statement (line 701)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 702)
    # Processing the call arguments (line 702)
    str_166035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 702)
    kwargs_166036 = {}
    # Getting the type of 'ValueError' (line 702)
    ValueError_166034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 702)
    ValueError_call_result_166037 = invoke(stypy.reporting.localization.Localization(__file__, 702, 14), ValueError_166034, *[str_166035], **kwargs_166036)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 702, 8), ValueError_call_result_166037, 'raise parameter', BaseException)
    # SSA join for if statement (line 701)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 703)
    c_166038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 703)
    ndim_166039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 12), c_166038, 'ndim')
    # Applying the 'usub' unary operator (line 703)
    result___neg___166040 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), 'usub', ndim_166039)
    
    # Getting the type of 'iaxis' (line 703)
    iaxis_166041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 22), 'iaxis')
    # Applying the binary operator '<=' (line 703)
    result_le_166042 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), '<=', result___neg___166040, iaxis_166041)
    # Getting the type of 'c' (line 703)
    c_166043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 703)
    ndim_166044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 30), c_166043, 'ndim')
    # Applying the binary operator '<' (line 703)
    result_lt_166045 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), '<', iaxis_166041, ndim_166044)
    # Applying the binary operator '&' (line 703)
    result_and__166046 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 11), '&', result_le_166042, result_lt_166045)
    
    # Applying the 'not' unary operator (line 703)
    result_not__166047 = python_operator(stypy.reporting.localization.Localization(__file__, 703, 7), 'not', result_and__166046)
    
    # Testing the type of an if condition (line 703)
    if_condition_166048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 703, 4), result_not__166047)
    # Assigning a type to the variable 'if_condition_166048' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'if_condition_166048', if_condition_166048)
    # SSA begins for if statement (line 703)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 704)
    # Processing the call arguments (line 704)
    str_166050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 704)
    kwargs_166051 = {}
    # Getting the type of 'ValueError' (line 704)
    ValueError_166049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 704)
    ValueError_call_result_166052 = invoke(stypy.reporting.localization.Localization(__file__, 704, 14), ValueError_166049, *[str_166050], **kwargs_166051)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 704, 8), ValueError_call_result_166052, 'raise parameter', BaseException)
    # SSA join for if statement (line 703)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 705)
    iaxis_166053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 7), 'iaxis')
    int_166054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 15), 'int')
    # Applying the binary operator '<' (line 705)
    result_lt_166055 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 7), '<', iaxis_166053, int_166054)
    
    # Testing the type of an if condition (line 705)
    if_condition_166056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 705, 4), result_lt_166055)
    # Assigning a type to the variable 'if_condition_166056' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'if_condition_166056', if_condition_166056)
    # SSA begins for if statement (line 705)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 706)
    iaxis_166057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'iaxis')
    # Getting the type of 'c' (line 706)
    c_166058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 706)
    ndim_166059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 17), c_166058, 'ndim')
    # Applying the binary operator '+=' (line 706)
    result_iadd_166060 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 8), '+=', iaxis_166057, ndim_166059)
    # Assigning a type to the variable 'iaxis' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 8), 'iaxis', result_iadd_166060)
    
    # SSA join for if statement (line 705)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 708)
    cnt_166061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 7), 'cnt')
    int_166062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 14), 'int')
    # Applying the binary operator '==' (line 708)
    result_eq_166063 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 7), '==', cnt_166061, int_166062)
    
    # Testing the type of an if condition (line 708)
    if_condition_166064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 708, 4), result_eq_166063)
    # Assigning a type to the variable 'if_condition_166064' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'if_condition_166064', if_condition_166064)
    # SSA begins for if statement (line 708)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 709)
    c_166065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 8), 'stypy_return_type', c_166065)
    # SSA join for if statement (line 708)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 711):
    
    # Assigning a Call to a Name (line 711):
    
    # Call to rollaxis(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'c' (line 711)
    c_166068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 20), 'c', False)
    # Getting the type of 'iaxis' (line 711)
    iaxis_166069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 711)
    kwargs_166070 = {}
    # Getting the type of 'np' (line 711)
    np_166066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 711)
    rollaxis_166067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), np_166066, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 711)
    rollaxis_call_result_166071 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), rollaxis_166067, *[c_166068, iaxis_166069], **kwargs_166070)
    
    # Assigning a type to the variable 'c' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'c', rollaxis_call_result_166071)
    
    # Assigning a Call to a Name (line 712):
    
    # Assigning a Call to a Name (line 712):
    
    # Call to len(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'c' (line 712)
    c_166073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 12), 'c', False)
    # Processing the call keyword arguments (line 712)
    kwargs_166074 = {}
    # Getting the type of 'len' (line 712)
    len_166072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'len', False)
    # Calling len(args, kwargs) (line 712)
    len_call_result_166075 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), len_166072, *[c_166073], **kwargs_166074)
    
    # Assigning a type to the variable 'n' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'n', len_call_result_166075)
    
    
    # Getting the type of 'cnt' (line 713)
    cnt_166076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 7), 'cnt')
    # Getting the type of 'n' (line 713)
    n_166077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 14), 'n')
    # Applying the binary operator '>=' (line 713)
    result_ge_166078 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 7), '>=', cnt_166076, n_166077)
    
    # Testing the type of an if condition (line 713)
    if_condition_166079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 713, 4), result_ge_166078)
    # Assigning a type to the variable 'if_condition_166079' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'if_condition_166079', if_condition_166079)
    # SSA begins for if statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 714):
    
    # Assigning a BinOp to a Name (line 714):
    
    # Obtaining the type of the subscript
    int_166080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 15), 'int')
    slice_166081 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 714, 12), None, int_166080, None)
    # Getting the type of 'c' (line 714)
    c_166082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 714)
    getitem___166083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 12), c_166082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 714)
    subscript_call_result_166084 = invoke(stypy.reporting.localization.Localization(__file__, 714, 12), getitem___166083, slice_166081)
    
    int_166085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 18), 'int')
    # Applying the binary operator '*' (line 714)
    result_mul_166086 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 12), '*', subscript_call_result_166084, int_166085)
    
    # Assigning a type to the variable 'c' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 8), 'c', result_mul_166086)
    # SSA branch for the else part of an if statement (line 713)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 716)
    # Processing the call arguments (line 716)
    # Getting the type of 'cnt' (line 716)
    cnt_166088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 23), 'cnt', False)
    # Processing the call keyword arguments (line 716)
    kwargs_166089 = {}
    # Getting the type of 'range' (line 716)
    range_166087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 17), 'range', False)
    # Calling range(args, kwargs) (line 716)
    range_call_result_166090 = invoke(stypy.reporting.localization.Localization(__file__, 716, 17), range_166087, *[cnt_166088], **kwargs_166089)
    
    # Testing the type of a for loop iterable (line 716)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 716, 8), range_call_result_166090)
    # Getting the type of the for loop variable (line 716)
    for_loop_var_166091 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 716, 8), range_call_result_166090)
    # Assigning a type to the variable 'i' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'i', for_loop_var_166091)
    # SSA begins for a for statement (line 716)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 717):
    
    # Assigning a BinOp to a Name (line 717):
    # Getting the type of 'n' (line 717)
    n_166092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 16), 'n')
    int_166093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 20), 'int')
    # Applying the binary operator '-' (line 717)
    result_sub_166094 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 16), '-', n_166092, int_166093)
    
    # Assigning a type to the variable 'n' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'n', result_sub_166094)
    
    # Getting the type of 'c' (line 718)
    c_166095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'c')
    # Getting the type of 'scl' (line 718)
    scl_166096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 17), 'scl')
    # Applying the binary operator '*=' (line 718)
    result_imul_166097 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 12), '*=', c_166095, scl_166096)
    # Assigning a type to the variable 'c' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'c', result_imul_166097)
    
    
    # Assigning a Call to a Name (line 719):
    
    # Assigning a Call to a Name (line 719):
    
    # Call to empty(...): (line 719)
    # Processing the call arguments (line 719)
    
    # Obtaining an instance of the builtin type 'tuple' (line 719)
    tuple_166100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 719)
    # Adding element type (line 719)
    # Getting the type of 'n' (line 719)
    n_166101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 28), tuple_166100, n_166101)
    
    
    # Obtaining the type of the subscript
    int_166102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 42), 'int')
    slice_166103 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 719, 34), int_166102, None, None)
    # Getting the type of 'c' (line 719)
    c_166104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 719)
    shape_166105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 34), c_166104, 'shape')
    # Obtaining the member '__getitem__' of a type (line 719)
    getitem___166106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 34), shape_166105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 719)
    subscript_call_result_166107 = invoke(stypy.reporting.localization.Localization(__file__, 719, 34), getitem___166106, slice_166103)
    
    # Applying the binary operator '+' (line 719)
    result_add_166108 = python_operator(stypy.reporting.localization.Localization(__file__, 719, 27), '+', tuple_166100, subscript_call_result_166107)
    
    # Processing the call keyword arguments (line 719)
    # Getting the type of 'c' (line 719)
    c_166109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 719)
    dtype_166110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 53), c_166109, 'dtype')
    keyword_166111 = dtype_166110
    kwargs_166112 = {'dtype': keyword_166111}
    # Getting the type of 'np' (line 719)
    np_166098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 719)
    empty_166099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 719, 18), np_166098, 'empty')
    # Calling empty(args, kwargs) (line 719)
    empty_call_result_166113 = invoke(stypy.reporting.localization.Localization(__file__, 719, 18), empty_166099, *[result_add_166108], **kwargs_166112)
    
    # Assigning a type to the variable 'der' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'der', empty_call_result_166113)
    
    
    # Call to range(...): (line 720)
    # Processing the call arguments (line 720)
    # Getting the type of 'n' (line 720)
    n_166115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 27), 'n', False)
    int_166116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 30), 'int')
    int_166117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 33), 'int')
    # Processing the call keyword arguments (line 720)
    kwargs_166118 = {}
    # Getting the type of 'range' (line 720)
    range_166114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 21), 'range', False)
    # Calling range(args, kwargs) (line 720)
    range_call_result_166119 = invoke(stypy.reporting.localization.Localization(__file__, 720, 21), range_166114, *[n_166115, int_166116, int_166117], **kwargs_166118)
    
    # Testing the type of a for loop iterable (line 720)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 720, 12), range_call_result_166119)
    # Getting the type of the for loop variable (line 720)
    for_loop_var_166120 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 720, 12), range_call_result_166119)
    # Assigning a type to the variable 'j' (line 720)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 720, 12), 'j', for_loop_var_166120)
    # SSA begins for a for statement (line 720)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 721):
    
    # Assigning a BinOp to a Subscript (line 721):
    int_166121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 30), 'int')
    # Getting the type of 'j' (line 721)
    j_166122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 32), 'j')
    # Applying the binary operator '*' (line 721)
    result_mul_166123 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 30), '*', int_166121, j_166122)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 721)
    j_166124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 37), 'j')
    # Getting the type of 'c' (line 721)
    c_166125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 35), 'c')
    # Obtaining the member '__getitem__' of a type (line 721)
    getitem___166126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 721, 35), c_166125, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 721)
    subscript_call_result_166127 = invoke(stypy.reporting.localization.Localization(__file__, 721, 35), getitem___166126, j_166124)
    
    # Applying the binary operator '*' (line 721)
    result_mul_166128 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 29), '*', result_mul_166123, subscript_call_result_166127)
    
    # Getting the type of 'der' (line 721)
    der_166129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 16), 'der')
    # Getting the type of 'j' (line 721)
    j_166130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 20), 'j')
    int_166131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 721, 24), 'int')
    # Applying the binary operator '-' (line 721)
    result_sub_166132 = python_operator(stypy.reporting.localization.Localization(__file__, 721, 20), '-', j_166130, int_166131)
    
    # Storing an element on a container (line 721)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 721, 16), der_166129, (result_sub_166132, result_mul_166128))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 722):
    
    # Assigning a Name to a Name (line 722):
    # Getting the type of 'der' (line 722)
    der_166133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 16), 'der')
    # Assigning a type to the variable 'c' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 12), 'c', der_166133)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 713)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 723):
    
    # Assigning a Call to a Name (line 723):
    
    # Call to rollaxis(...): (line 723)
    # Processing the call arguments (line 723)
    # Getting the type of 'c' (line 723)
    c_166136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 20), 'c', False)
    int_166137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 23), 'int')
    # Getting the type of 'iaxis' (line 723)
    iaxis_166138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 26), 'iaxis', False)
    int_166139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, 34), 'int')
    # Applying the binary operator '+' (line 723)
    result_add_166140 = python_operator(stypy.reporting.localization.Localization(__file__, 723, 26), '+', iaxis_166138, int_166139)
    
    # Processing the call keyword arguments (line 723)
    kwargs_166141 = {}
    # Getting the type of 'np' (line 723)
    np_166134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 723)
    rollaxis_166135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 723, 8), np_166134, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 723)
    rollaxis_call_result_166142 = invoke(stypy.reporting.localization.Localization(__file__, 723, 8), rollaxis_166135, *[c_166136, int_166137, result_add_166140], **kwargs_166141)
    
    # Assigning a type to the variable 'c' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'c', rollaxis_call_result_166142)
    # Getting the type of 'c' (line 724)
    c_166143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'stypy_return_type', c_166143)
    
    # ################# End of 'hermder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermder' in the type store
    # Getting the type of 'stypy_return_type' (line 637)
    stypy_return_type_166144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166144)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermder'
    return stypy_return_type_166144

# Assigning a type to the variable 'hermder' (line 637)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 0), 'hermder', hermder)

@norecursion
def hermint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_166145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 17), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 727)
    list_166146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 727)
    
    int_166147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 31), 'int')
    int_166148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 38), 'int')
    int_166149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 46), 'int')
    defaults = [int_166145, list_166146, int_166147, int_166148, int_166149]
    # Create a new context for function 'hermint'
    module_type_store = module_type_store.open_function_context('hermint', 727, 0, False)
    
    # Passed parameters checking function
    hermint.stypy_localization = localization
    hermint.stypy_type_of_self = None
    hermint.stypy_type_store = module_type_store
    hermint.stypy_function_name = 'hermint'
    hermint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    hermint.stypy_varargs_param_name = None
    hermint.stypy_kwargs_param_name = None
    hermint.stypy_call_defaults = defaults
    hermint.stypy_call_varargs = varargs
    hermint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermint(...)' code ##################

    str_166150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, (-1)), 'str', '\n    Integrate a Hermite series.\n\n    Returns the Hermite series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]\n    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +\n    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Hermite series coefficients. If c is multidimensional the\n        different axis correspond to different variables with the degree in\n        each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at\n        ``lbnd`` is the first value in the list, the value of the second\n        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the\n        default), all constants are set to zero.  If ``m == 1``, a single\n        scalar can be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Hermite series coefficients of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or\n        ``np.isscalar(scl) == False``.\n\n    See Also\n    --------\n    hermder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermint\n    >>> hermint([1,2,3]) # integrate once, value 0 at 0.\n    array([ 1. ,  0.5,  0.5,  0.5])\n    >>> hermint([1,2,3], m=2) # integrate twice, value & deriv 0 at 0\n    array([-0.5       ,  0.5       ,  0.125     ,  0.08333333,  0.0625    ])\n    >>> hermint([1,2,3], k=1) # integrate once, value 1 at 0.\n    array([ 2. ,  0.5,  0.5,  0.5])\n    >>> hermint([1,2,3], lbnd=-1) # integrate once, value 0 at -1\n    array([-2. ,  0.5,  0.5,  0.5])\n    >>> hermint([1,2,3], m=2, k=[1,2], lbnd=-1)\n    array([ 1.66666667, -0.5       ,  0.125     ,  0.08333333,  0.0625    ])\n\n    ')
    
    # Assigning a Call to a Name (line 810):
    
    # Assigning a Call to a Name (line 810):
    
    # Call to array(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'c' (line 810)
    c_166153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 17), 'c', False)
    # Processing the call keyword arguments (line 810)
    int_166154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 26), 'int')
    keyword_166155 = int_166154
    int_166156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 34), 'int')
    keyword_166157 = int_166156
    kwargs_166158 = {'copy': keyword_166157, 'ndmin': keyword_166155}
    # Getting the type of 'np' (line 810)
    np_166151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 810)
    array_166152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 8), np_166151, 'array')
    # Calling array(args, kwargs) (line 810)
    array_call_result_166159 = invoke(stypy.reporting.localization.Localization(__file__, 810, 8), array_166152, *[c_166153], **kwargs_166158)
    
    # Assigning a type to the variable 'c' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'c', array_call_result_166159)
    
    
    # Getting the type of 'c' (line 811)
    c_166160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 811)
    dtype_166161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 7), c_166160, 'dtype')
    # Obtaining the member 'char' of a type (line 811)
    char_166162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 7), dtype_166161, 'char')
    str_166163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 811, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 811)
    result_contains_166164 = python_operator(stypy.reporting.localization.Localization(__file__, 811, 7), 'in', char_166162, str_166163)
    
    # Testing the type of an if condition (line 811)
    if_condition_166165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 811, 4), result_contains_166164)
    # Assigning a type to the variable 'if_condition_166165' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'if_condition_166165', if_condition_166165)
    # SSA begins for if statement (line 811)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 812):
    
    # Assigning a Call to a Name (line 812):
    
    # Call to astype(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'np' (line 812)
    np_166168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 812)
    double_166169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 21), np_166168, 'double')
    # Processing the call keyword arguments (line 812)
    kwargs_166170 = {}
    # Getting the type of 'c' (line 812)
    c_166166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 812)
    astype_166167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 12), c_166166, 'astype')
    # Calling astype(args, kwargs) (line 812)
    astype_call_result_166171 = invoke(stypy.reporting.localization.Localization(__file__, 812, 12), astype_166167, *[double_166169], **kwargs_166170)
    
    # Assigning a type to the variable 'c' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'c', astype_call_result_166171)
    # SSA join for if statement (line 811)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to iterable(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'k' (line 813)
    k_166174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 23), 'k', False)
    # Processing the call keyword arguments (line 813)
    kwargs_166175 = {}
    # Getting the type of 'np' (line 813)
    np_166172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 813)
    iterable_166173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 11), np_166172, 'iterable')
    # Calling iterable(args, kwargs) (line 813)
    iterable_call_result_166176 = invoke(stypy.reporting.localization.Localization(__file__, 813, 11), iterable_166173, *[k_166174], **kwargs_166175)
    
    # Applying the 'not' unary operator (line 813)
    result_not__166177 = python_operator(stypy.reporting.localization.Localization(__file__, 813, 7), 'not', iterable_call_result_166176)
    
    # Testing the type of an if condition (line 813)
    if_condition_166178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 813, 4), result_not__166177)
    # Assigning a type to the variable 'if_condition_166178' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'if_condition_166178', if_condition_166178)
    # SSA begins for if statement (line 813)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 814):
    
    # Assigning a List to a Name (line 814):
    
    # Obtaining an instance of the builtin type 'list' (line 814)
    list_166179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 814)
    # Adding element type (line 814)
    # Getting the type of 'k' (line 814)
    k_166180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 12), list_166179, k_166180)
    
    # Assigning a type to the variable 'k' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'k', list_166179)
    # SSA join for if statement (line 813)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 815):
    
    # Assigning a Subscript to a Name (line 815):
    
    # Obtaining the type of the subscript
    int_166181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 815)
    list_166186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 815)
    # Adding element type (line 815)
    # Getting the type of 'm' (line 815)
    m_166187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_166186, m_166187)
    # Adding element type (line 815)
    # Getting the type of 'axis' (line 815)
    axis_166188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_166186, axis_166188)
    
    comprehension_166189 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_166186)
    # Assigning a type to the variable 't' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 't', comprehension_166189)
    
    # Call to int(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 't' (line 815)
    t_166183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 't', False)
    # Processing the call keyword arguments (line 815)
    kwargs_166184 = {}
    # Getting the type of 'int' (line 815)
    int_166182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'int', False)
    # Calling int(args, kwargs) (line 815)
    int_call_result_166185 = invoke(stypy.reporting.localization.Localization(__file__, 815, 18), int_166182, *[t_166183], **kwargs_166184)
    
    list_166190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_166190, int_call_result_166185)
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___166191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 4), list_166190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_166192 = invoke(stypy.reporting.localization.Localization(__file__, 815, 4), getitem___166191, int_166181)
    
    # Assigning a type to the variable 'tuple_var_assignment_165032' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_165032', subscript_call_result_166192)
    
    # Assigning a Subscript to a Name (line 815):
    
    # Obtaining the type of the subscript
    int_166193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 815)
    list_166198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 815)
    # Adding element type (line 815)
    # Getting the type of 'm' (line 815)
    m_166199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_166198, m_166199)
    # Adding element type (line 815)
    # Getting the type of 'axis' (line 815)
    axis_166200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 34), list_166198, axis_166200)
    
    comprehension_166201 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_166198)
    # Assigning a type to the variable 't' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 't', comprehension_166201)
    
    # Call to int(...): (line 815)
    # Processing the call arguments (line 815)
    # Getting the type of 't' (line 815)
    t_166195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 't', False)
    # Processing the call keyword arguments (line 815)
    kwargs_166196 = {}
    # Getting the type of 'int' (line 815)
    int_166194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'int', False)
    # Calling int(args, kwargs) (line 815)
    int_call_result_166197 = invoke(stypy.reporting.localization.Localization(__file__, 815, 18), int_166194, *[t_166195], **kwargs_166196)
    
    list_166202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 815, 18), list_166202, int_call_result_166197)
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___166203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 4), list_166202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_166204 = invoke(stypy.reporting.localization.Localization(__file__, 815, 4), getitem___166203, int_166193)
    
    # Assigning a type to the variable 'tuple_var_assignment_165033' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_165033', subscript_call_result_166204)
    
    # Assigning a Name to a Name (line 815):
    # Getting the type of 'tuple_var_assignment_165032' (line 815)
    tuple_var_assignment_165032_166205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_165032')
    # Assigning a type to the variable 'cnt' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'cnt', tuple_var_assignment_165032_166205)
    
    # Assigning a Name to a Name (line 815):
    # Getting the type of 'tuple_var_assignment_165033' (line 815)
    tuple_var_assignment_165033_166206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 4), 'tuple_var_assignment_165033')
    # Assigning a type to the variable 'iaxis' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 9), 'iaxis', tuple_var_assignment_165033_166206)
    
    
    # Getting the type of 'cnt' (line 817)
    cnt_166207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 7), 'cnt')
    # Getting the type of 'm' (line 817)
    m_166208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 14), 'm')
    # Applying the binary operator '!=' (line 817)
    result_ne_166209 = python_operator(stypy.reporting.localization.Localization(__file__, 817, 7), '!=', cnt_166207, m_166208)
    
    # Testing the type of an if condition (line 817)
    if_condition_166210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 817, 4), result_ne_166209)
    # Assigning a type to the variable 'if_condition_166210' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'if_condition_166210', if_condition_166210)
    # SSA begins for if statement (line 817)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 818)
    # Processing the call arguments (line 818)
    str_166212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 818)
    kwargs_166213 = {}
    # Getting the type of 'ValueError' (line 818)
    ValueError_166211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 818)
    ValueError_call_result_166214 = invoke(stypy.reporting.localization.Localization(__file__, 818, 14), ValueError_166211, *[str_166212], **kwargs_166213)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 818, 8), ValueError_call_result_166214, 'raise parameter', BaseException)
    # SSA join for if statement (line 817)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 819)
    cnt_166215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 7), 'cnt')
    int_166216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 13), 'int')
    # Applying the binary operator '<' (line 819)
    result_lt_166217 = python_operator(stypy.reporting.localization.Localization(__file__, 819, 7), '<', cnt_166215, int_166216)
    
    # Testing the type of an if condition (line 819)
    if_condition_166218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 819, 4), result_lt_166217)
    # Assigning a type to the variable 'if_condition_166218' (line 819)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 4), 'if_condition_166218', if_condition_166218)
    # SSA begins for if statement (line 819)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 820)
    # Processing the call arguments (line 820)
    str_166220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 820)
    kwargs_166221 = {}
    # Getting the type of 'ValueError' (line 820)
    ValueError_166219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 820)
    ValueError_call_result_166222 = invoke(stypy.reporting.localization.Localization(__file__, 820, 14), ValueError_166219, *[str_166220], **kwargs_166221)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 820, 8), ValueError_call_result_166222, 'raise parameter', BaseException)
    # SSA join for if statement (line 819)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 821)
    # Processing the call arguments (line 821)
    # Getting the type of 'k' (line 821)
    k_166224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 11), 'k', False)
    # Processing the call keyword arguments (line 821)
    kwargs_166225 = {}
    # Getting the type of 'len' (line 821)
    len_166223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 7), 'len', False)
    # Calling len(args, kwargs) (line 821)
    len_call_result_166226 = invoke(stypy.reporting.localization.Localization(__file__, 821, 7), len_166223, *[k_166224], **kwargs_166225)
    
    # Getting the type of 'cnt' (line 821)
    cnt_166227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'cnt')
    # Applying the binary operator '>' (line 821)
    result_gt_166228 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 7), '>', len_call_result_166226, cnt_166227)
    
    # Testing the type of an if condition (line 821)
    if_condition_166229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 4), result_gt_166228)
    # Assigning a type to the variable 'if_condition_166229' (line 821)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 4), 'if_condition_166229', if_condition_166229)
    # SSA begins for if statement (line 821)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 822)
    # Processing the call arguments (line 822)
    str_166231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 822, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 822)
    kwargs_166232 = {}
    # Getting the type of 'ValueError' (line 822)
    ValueError_166230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 822)
    ValueError_call_result_166233 = invoke(stypy.reporting.localization.Localization(__file__, 822, 14), ValueError_166230, *[str_166231], **kwargs_166232)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 822, 8), ValueError_call_result_166233, 'raise parameter', BaseException)
    # SSA join for if statement (line 821)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 823)
    iaxis_166234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 7), 'iaxis')
    # Getting the type of 'axis' (line 823)
    axis_166235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 16), 'axis')
    # Applying the binary operator '!=' (line 823)
    result_ne_166236 = python_operator(stypy.reporting.localization.Localization(__file__, 823, 7), '!=', iaxis_166234, axis_166235)
    
    # Testing the type of an if condition (line 823)
    if_condition_166237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 823, 4), result_ne_166236)
    # Assigning a type to the variable 'if_condition_166237' (line 823)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 4), 'if_condition_166237', if_condition_166237)
    # SSA begins for if statement (line 823)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 824)
    # Processing the call arguments (line 824)
    str_166239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 824)
    kwargs_166240 = {}
    # Getting the type of 'ValueError' (line 824)
    ValueError_166238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 824)
    ValueError_call_result_166241 = invoke(stypy.reporting.localization.Localization(__file__, 824, 14), ValueError_166238, *[str_166239], **kwargs_166240)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 824, 8), ValueError_call_result_166241, 'raise parameter', BaseException)
    # SSA join for if statement (line 823)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 825)
    c_166242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 825)
    ndim_166243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 12), c_166242, 'ndim')
    # Applying the 'usub' unary operator (line 825)
    result___neg___166244 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), 'usub', ndim_166243)
    
    # Getting the type of 'iaxis' (line 825)
    iaxis_166245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'iaxis')
    # Applying the binary operator '<=' (line 825)
    result_le_166246 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '<=', result___neg___166244, iaxis_166245)
    # Getting the type of 'c' (line 825)
    c_166247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 825)
    ndim_166248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 30), c_166247, 'ndim')
    # Applying the binary operator '<' (line 825)
    result_lt_166249 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '<', iaxis_166245, ndim_166248)
    # Applying the binary operator '&' (line 825)
    result_and__166250 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 11), '&', result_le_166246, result_lt_166249)
    
    # Applying the 'not' unary operator (line 825)
    result_not__166251 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 7), 'not', result_and__166250)
    
    # Testing the type of an if condition (line 825)
    if_condition_166252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 825, 4), result_not__166251)
    # Assigning a type to the variable 'if_condition_166252' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'if_condition_166252', if_condition_166252)
    # SSA begins for if statement (line 825)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 826)
    # Processing the call arguments (line 826)
    str_166254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 826)
    kwargs_166255 = {}
    # Getting the type of 'ValueError' (line 826)
    ValueError_166253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 826)
    ValueError_call_result_166256 = invoke(stypy.reporting.localization.Localization(__file__, 826, 14), ValueError_166253, *[str_166254], **kwargs_166255)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 826, 8), ValueError_call_result_166256, 'raise parameter', BaseException)
    # SSA join for if statement (line 825)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 827)
    iaxis_166257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 7), 'iaxis')
    int_166258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 827, 15), 'int')
    # Applying the binary operator '<' (line 827)
    result_lt_166259 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 7), '<', iaxis_166257, int_166258)
    
    # Testing the type of an if condition (line 827)
    if_condition_166260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 827, 4), result_lt_166259)
    # Assigning a type to the variable 'if_condition_166260' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 4), 'if_condition_166260', if_condition_166260)
    # SSA begins for if statement (line 827)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 828)
    iaxis_166261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'iaxis')
    # Getting the type of 'c' (line 828)
    c_166262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 828)
    ndim_166263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 17), c_166262, 'ndim')
    # Applying the binary operator '+=' (line 828)
    result_iadd_166264 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 8), '+=', iaxis_166261, ndim_166263)
    # Assigning a type to the variable 'iaxis' (line 828)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'iaxis', result_iadd_166264)
    
    # SSA join for if statement (line 827)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 830)
    cnt_166265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 7), 'cnt')
    int_166266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 14), 'int')
    # Applying the binary operator '==' (line 830)
    result_eq_166267 = python_operator(stypy.reporting.localization.Localization(__file__, 830, 7), '==', cnt_166265, int_166266)
    
    # Testing the type of an if condition (line 830)
    if_condition_166268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 830, 4), result_eq_166267)
    # Assigning a type to the variable 'if_condition_166268' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'if_condition_166268', if_condition_166268)
    # SSA begins for if statement (line 830)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 831)
    c_166269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 831)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 8), 'stypy_return_type', c_166269)
    # SSA join for if statement (line 830)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 833):
    
    # Assigning a Call to a Name (line 833):
    
    # Call to rollaxis(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'c' (line 833)
    c_166272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 20), 'c', False)
    # Getting the type of 'iaxis' (line 833)
    iaxis_166273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 833)
    kwargs_166274 = {}
    # Getting the type of 'np' (line 833)
    np_166270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 833)
    rollaxis_166271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 8), np_166270, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 833)
    rollaxis_call_result_166275 = invoke(stypy.reporting.localization.Localization(__file__, 833, 8), rollaxis_166271, *[c_166272, iaxis_166273], **kwargs_166274)
    
    # Assigning a type to the variable 'c' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'c', rollaxis_call_result_166275)
    
    # Assigning a BinOp to a Name (line 834):
    
    # Assigning a BinOp to a Name (line 834):
    
    # Call to list(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'k' (line 834)
    k_166277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 13), 'k', False)
    # Processing the call keyword arguments (line 834)
    kwargs_166278 = {}
    # Getting the type of 'list' (line 834)
    list_166276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'list', False)
    # Calling list(args, kwargs) (line 834)
    list_call_result_166279 = invoke(stypy.reporting.localization.Localization(__file__, 834, 8), list_166276, *[k_166277], **kwargs_166278)
    
    
    # Obtaining an instance of the builtin type 'list' (line 834)
    list_166280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 834)
    # Adding element type (line 834)
    int_166281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 834, 18), list_166280, int_166281)
    
    # Getting the type of 'cnt' (line 834)
    cnt_166282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 23), 'cnt')
    
    # Call to len(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'k' (line 834)
    k_166284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 33), 'k', False)
    # Processing the call keyword arguments (line 834)
    kwargs_166285 = {}
    # Getting the type of 'len' (line 834)
    len_166283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 29), 'len', False)
    # Calling len(args, kwargs) (line 834)
    len_call_result_166286 = invoke(stypy.reporting.localization.Localization(__file__, 834, 29), len_166283, *[k_166284], **kwargs_166285)
    
    # Applying the binary operator '-' (line 834)
    result_sub_166287 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 23), '-', cnt_166282, len_call_result_166286)
    
    # Applying the binary operator '*' (line 834)
    result_mul_166288 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 18), '*', list_166280, result_sub_166287)
    
    # Applying the binary operator '+' (line 834)
    result_add_166289 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 8), '+', list_call_result_166279, result_mul_166288)
    
    # Assigning a type to the variable 'k' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'k', result_add_166289)
    
    
    # Call to range(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'cnt' (line 835)
    cnt_166291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 19), 'cnt', False)
    # Processing the call keyword arguments (line 835)
    kwargs_166292 = {}
    # Getting the type of 'range' (line 835)
    range_166290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 13), 'range', False)
    # Calling range(args, kwargs) (line 835)
    range_call_result_166293 = invoke(stypy.reporting.localization.Localization(__file__, 835, 13), range_166290, *[cnt_166291], **kwargs_166292)
    
    # Testing the type of a for loop iterable (line 835)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 835, 4), range_call_result_166293)
    # Getting the type of the for loop variable (line 835)
    for_loop_var_166294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 835, 4), range_call_result_166293)
    # Assigning a type to the variable 'i' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 4), 'i', for_loop_var_166294)
    # SSA begins for a for statement (line 835)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 836):
    
    # Assigning a Call to a Name (line 836):
    
    # Call to len(...): (line 836)
    # Processing the call arguments (line 836)
    # Getting the type of 'c' (line 836)
    c_166296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 16), 'c', False)
    # Processing the call keyword arguments (line 836)
    kwargs_166297 = {}
    # Getting the type of 'len' (line 836)
    len_166295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 12), 'len', False)
    # Calling len(args, kwargs) (line 836)
    len_call_result_166298 = invoke(stypy.reporting.localization.Localization(__file__, 836, 12), len_166295, *[c_166296], **kwargs_166297)
    
    # Assigning a type to the variable 'n' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'n', len_call_result_166298)
    
    # Getting the type of 'c' (line 837)
    c_166299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'c')
    # Getting the type of 'scl' (line 837)
    scl_166300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 13), 'scl')
    # Applying the binary operator '*=' (line 837)
    result_imul_166301 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 8), '*=', c_166299, scl_166300)
    # Assigning a type to the variable 'c' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'c', result_imul_166301)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 838)
    n_166302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'n')
    int_166303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 16), 'int')
    # Applying the binary operator '==' (line 838)
    result_eq_166304 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 11), '==', n_166302, int_166303)
    
    
    # Call to all(...): (line 838)
    # Processing the call arguments (line 838)
    
    
    # Obtaining the type of the subscript
    int_166307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 31), 'int')
    # Getting the type of 'c' (line 838)
    c_166308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___166309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 29), c_166308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_166310 = invoke(stypy.reporting.localization.Localization(__file__, 838, 29), getitem___166309, int_166307)
    
    int_166311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 37), 'int')
    # Applying the binary operator '==' (line 838)
    result_eq_166312 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 29), '==', subscript_call_result_166310, int_166311)
    
    # Processing the call keyword arguments (line 838)
    kwargs_166313 = {}
    # Getting the type of 'np' (line 838)
    np_166305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 838)
    all_166306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 22), np_166305, 'all')
    # Calling all(args, kwargs) (line 838)
    all_call_result_166314 = invoke(stypy.reporting.localization.Localization(__file__, 838, 22), all_166306, *[result_eq_166312], **kwargs_166313)
    
    # Applying the binary operator 'and' (line 838)
    result_and_keyword_166315 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 11), 'and', result_eq_166304, all_call_result_166314)
    
    # Testing the type of an if condition (line 838)
    if_condition_166316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 8), result_and_keyword_166315)
    # Assigning a type to the variable 'if_condition_166316' (line 838)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'if_condition_166316', if_condition_166316)
    # SSA begins for if statement (line 838)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 839)
    c_166317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    
    # Obtaining the type of the subscript
    int_166318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 14), 'int')
    # Getting the type of 'c' (line 839)
    c_166319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___166320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 12), c_166319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_166321 = invoke(stypy.reporting.localization.Localization(__file__, 839, 12), getitem___166320, int_166318)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 839)
    i_166322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 22), 'i')
    # Getting the type of 'k' (line 839)
    k_166323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 839)
    getitem___166324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 20), k_166323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
    subscript_call_result_166325 = invoke(stypy.reporting.localization.Localization(__file__, 839, 20), getitem___166324, i_166322)
    
    # Applying the binary operator '+=' (line 839)
    result_iadd_166326 = python_operator(stypy.reporting.localization.Localization(__file__, 839, 12), '+=', subscript_call_result_166321, subscript_call_result_166325)
    # Getting the type of 'c' (line 839)
    c_166327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'c')
    int_166328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 839, 14), 'int')
    # Storing an element on a container (line 839)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 839, 12), c_166327, (int_166328, result_iadd_166326))
    
    # SSA branch for the else part of an if statement (line 838)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 841):
    
    # Assigning a Call to a Name (line 841):
    
    # Call to empty(...): (line 841)
    # Processing the call arguments (line 841)
    
    # Obtaining an instance of the builtin type 'tuple' (line 841)
    tuple_166331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 841)
    # Adding element type (line 841)
    # Getting the type of 'n' (line 841)
    n_166332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 28), 'n', False)
    int_166333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 32), 'int')
    # Applying the binary operator '+' (line 841)
    result_add_166334 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 28), '+', n_166332, int_166333)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 28), tuple_166331, result_add_166334)
    
    
    # Obtaining the type of the subscript
    int_166335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 46), 'int')
    slice_166336 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 841, 38), int_166335, None, None)
    # Getting the type of 'c' (line 841)
    c_166337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 841)
    shape_166338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 38), c_166337, 'shape')
    # Obtaining the member '__getitem__' of a type (line 841)
    getitem___166339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 38), shape_166338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 841)
    subscript_call_result_166340 = invoke(stypy.reporting.localization.Localization(__file__, 841, 38), getitem___166339, slice_166336)
    
    # Applying the binary operator '+' (line 841)
    result_add_166341 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 27), '+', tuple_166331, subscript_call_result_166340)
    
    # Processing the call keyword arguments (line 841)
    # Getting the type of 'c' (line 841)
    c_166342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 57), 'c', False)
    # Obtaining the member 'dtype' of a type (line 841)
    dtype_166343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 57), c_166342, 'dtype')
    keyword_166344 = dtype_166343
    kwargs_166345 = {'dtype': keyword_166344}
    # Getting the type of 'np' (line 841)
    np_166329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 841)
    empty_166330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 18), np_166329, 'empty')
    # Calling empty(args, kwargs) (line 841)
    empty_call_result_166346 = invoke(stypy.reporting.localization.Localization(__file__, 841, 18), empty_166330, *[result_add_166341], **kwargs_166345)
    
    # Assigning a type to the variable 'tmp' (line 841)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'tmp', empty_call_result_166346)
    
    # Assigning a BinOp to a Subscript (line 842):
    
    # Assigning a BinOp to a Subscript (line 842):
    
    # Obtaining the type of the subscript
    int_166347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 23), 'int')
    # Getting the type of 'c' (line 842)
    c_166348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___166349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 21), c_166348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_166350 = invoke(stypy.reporting.localization.Localization(__file__, 842, 21), getitem___166349, int_166347)
    
    int_166351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 26), 'int')
    # Applying the binary operator '*' (line 842)
    result_mul_166352 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 21), '*', subscript_call_result_166350, int_166351)
    
    # Getting the type of 'tmp' (line 842)
    tmp_166353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'tmp')
    int_166354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 16), 'int')
    # Storing an element on a container (line 842)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 12), tmp_166353, (int_166354, result_mul_166352))
    
    # Assigning a BinOp to a Subscript (line 843):
    
    # Assigning a BinOp to a Subscript (line 843):
    
    # Obtaining the type of the subscript
    int_166355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 23), 'int')
    # Getting the type of 'c' (line 843)
    c_166356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 843)
    getitem___166357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 21), c_166356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 843)
    subscript_call_result_166358 = invoke(stypy.reporting.localization.Localization(__file__, 843, 21), getitem___166357, int_166355)
    
    int_166359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 26), 'int')
    # Applying the binary operator 'div' (line 843)
    result_div_166360 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 21), 'div', subscript_call_result_166358, int_166359)
    
    # Getting the type of 'tmp' (line 843)
    tmp_166361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'tmp')
    int_166362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 16), 'int')
    # Storing an element on a container (line 843)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 843, 12), tmp_166361, (int_166362, result_div_166360))
    
    
    # Call to range(...): (line 844)
    # Processing the call arguments (line 844)
    int_166364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 27), 'int')
    # Getting the type of 'n' (line 844)
    n_166365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 30), 'n', False)
    # Processing the call keyword arguments (line 844)
    kwargs_166366 = {}
    # Getting the type of 'range' (line 844)
    range_166363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'range', False)
    # Calling range(args, kwargs) (line 844)
    range_call_result_166367 = invoke(stypy.reporting.localization.Localization(__file__, 844, 21), range_166363, *[int_166364, n_166365], **kwargs_166366)
    
    # Testing the type of a for loop iterable (line 844)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 844, 12), range_call_result_166367)
    # Getting the type of the for loop variable (line 844)
    for_loop_var_166368 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 844, 12), range_call_result_166367)
    # Assigning a type to the variable 'j' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 12), 'j', for_loop_var_166368)
    # SSA begins for a for statement (line 844)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 845):
    
    # Assigning a BinOp to a Subscript (line 845):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 845)
    j_166369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 31), 'j')
    # Getting the type of 'c' (line 845)
    c_166370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___166371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 29), c_166370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_166372 = invoke(stypy.reporting.localization.Localization(__file__, 845, 29), getitem___166371, j_166369)
    
    int_166373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 35), 'int')
    # Getting the type of 'j' (line 845)
    j_166374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 38), 'j')
    int_166375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 42), 'int')
    # Applying the binary operator '+' (line 845)
    result_add_166376 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 38), '+', j_166374, int_166375)
    
    # Applying the binary operator '*' (line 845)
    result_mul_166377 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 35), '*', int_166373, result_add_166376)
    
    # Applying the binary operator 'div' (line 845)
    result_div_166378 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 29), 'div', subscript_call_result_166372, result_mul_166377)
    
    # Getting the type of 'tmp' (line 845)
    tmp_166379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 16), 'tmp')
    # Getting the type of 'j' (line 845)
    j_166380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 20), 'j')
    int_166381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 24), 'int')
    # Applying the binary operator '+' (line 845)
    result_add_166382 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 20), '+', j_166380, int_166381)
    
    # Storing an element on a container (line 845)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 16), tmp_166379, (result_add_166382, result_div_166378))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 846)
    tmp_166383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_166384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 16), 'int')
    # Getting the type of 'tmp' (line 846)
    tmp_166385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___166386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 12), tmp_166385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_166387 = invoke(stypy.reporting.localization.Localization(__file__, 846, 12), getitem___166386, int_166384)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 846)
    i_166388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 24), 'i')
    # Getting the type of 'k' (line 846)
    k_166389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___166390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 22), k_166389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_166391 = invoke(stypy.reporting.localization.Localization(__file__, 846, 22), getitem___166390, i_166388)
    
    
    # Call to hermval(...): (line 846)
    # Processing the call arguments (line 846)
    # Getting the type of 'lbnd' (line 846)
    lbnd_166393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 37), 'lbnd', False)
    # Getting the type of 'tmp' (line 846)
    tmp_166394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 43), 'tmp', False)
    # Processing the call keyword arguments (line 846)
    kwargs_166395 = {}
    # Getting the type of 'hermval' (line 846)
    hermval_166392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 29), 'hermval', False)
    # Calling hermval(args, kwargs) (line 846)
    hermval_call_result_166396 = invoke(stypy.reporting.localization.Localization(__file__, 846, 29), hermval_166392, *[lbnd_166393, tmp_166394], **kwargs_166395)
    
    # Applying the binary operator '-' (line 846)
    result_sub_166397 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 22), '-', subscript_call_result_166391, hermval_call_result_166396)
    
    # Applying the binary operator '+=' (line 846)
    result_iadd_166398 = python_operator(stypy.reporting.localization.Localization(__file__, 846, 12), '+=', subscript_call_result_166387, result_sub_166397)
    # Getting the type of 'tmp' (line 846)
    tmp_166399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'tmp')
    int_166400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 16), 'int')
    # Storing an element on a container (line 846)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 846, 12), tmp_166399, (int_166400, result_iadd_166398))
    
    
    # Assigning a Name to a Name (line 847):
    
    # Assigning a Name to a Name (line 847):
    # Getting the type of 'tmp' (line 847)
    tmp_166401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 12), 'c', tmp_166401)
    # SSA join for if statement (line 838)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 848):
    
    # Assigning a Call to a Name (line 848):
    
    # Call to rollaxis(...): (line 848)
    # Processing the call arguments (line 848)
    # Getting the type of 'c' (line 848)
    c_166404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 20), 'c', False)
    int_166405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 23), 'int')
    # Getting the type of 'iaxis' (line 848)
    iaxis_166406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 26), 'iaxis', False)
    int_166407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 34), 'int')
    # Applying the binary operator '+' (line 848)
    result_add_166408 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 26), '+', iaxis_166406, int_166407)
    
    # Processing the call keyword arguments (line 848)
    kwargs_166409 = {}
    # Getting the type of 'np' (line 848)
    np_166402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 848)
    rollaxis_166403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 8), np_166402, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 848)
    rollaxis_call_result_166410 = invoke(stypy.reporting.localization.Localization(__file__, 848, 8), rollaxis_166403, *[c_166404, int_166405, result_add_166408], **kwargs_166409)
    
    # Assigning a type to the variable 'c' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'c', rollaxis_call_result_166410)
    # Getting the type of 'c' (line 849)
    c_166411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 849)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 4), 'stypy_return_type', c_166411)
    
    # ################# End of 'hermint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermint' in the type store
    # Getting the type of 'stypy_return_type' (line 727)
    stypy_return_type_166412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166412)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermint'
    return stypy_return_type_166412

# Assigning a type to the variable 'hermint' (line 727)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 0), 'hermint', hermint)

@norecursion
def hermval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 852)
    True_166413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 25), 'True')
    defaults = [True_166413]
    # Create a new context for function 'hermval'
    module_type_store = module_type_store.open_function_context('hermval', 852, 0, False)
    
    # Passed parameters checking function
    hermval.stypy_localization = localization
    hermval.stypy_type_of_self = None
    hermval.stypy_type_store = module_type_store
    hermval.stypy_function_name = 'hermval'
    hermval.stypy_param_names_list = ['x', 'c', 'tensor']
    hermval.stypy_varargs_param_name = None
    hermval.stypy_kwargs_param_name = None
    hermval.stypy_call_defaults = defaults
    hermval.stypy_call_varargs = varargs
    hermval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermval(...)' code ##################

    str_166414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, (-1)), 'str', '\n    Evaluate an Hermite series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * H_0(x) + c_1 * H_1(x) + ... + c_n * H_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    hermval2d, hermgrid2d, hermval3d, hermgrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermval\n    >>> coef = [1,2,3]\n    >>> hermval(1, coef)\n    11.0\n    >>> hermval([[1,2],[3,4]], coef)\n    array([[  11.,   51.],\n           [ 115.,  203.]])\n\n    ')
    
    # Assigning a Call to a Name (line 921):
    
    # Assigning a Call to a Name (line 921):
    
    # Call to array(...): (line 921)
    # Processing the call arguments (line 921)
    # Getting the type of 'c' (line 921)
    c_166417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 17), 'c', False)
    # Processing the call keyword arguments (line 921)
    int_166418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 26), 'int')
    keyword_166419 = int_166418
    int_166420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 34), 'int')
    keyword_166421 = int_166420
    kwargs_166422 = {'copy': keyword_166421, 'ndmin': keyword_166419}
    # Getting the type of 'np' (line 921)
    np_166415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 921)
    array_166416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 8), np_166415, 'array')
    # Calling array(args, kwargs) (line 921)
    array_call_result_166423 = invoke(stypy.reporting.localization.Localization(__file__, 921, 8), array_166416, *[c_166417], **kwargs_166422)
    
    # Assigning a type to the variable 'c' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'c', array_call_result_166423)
    
    
    # Getting the type of 'c' (line 922)
    c_166424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 922)
    dtype_166425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 7), c_166424, 'dtype')
    # Obtaining the member 'char' of a type (line 922)
    char_166426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 7), dtype_166425, 'char')
    str_166427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 922, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 922)
    result_contains_166428 = python_operator(stypy.reporting.localization.Localization(__file__, 922, 7), 'in', char_166426, str_166427)
    
    # Testing the type of an if condition (line 922)
    if_condition_166429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 922, 4), result_contains_166428)
    # Assigning a type to the variable 'if_condition_166429' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'if_condition_166429', if_condition_166429)
    # SSA begins for if statement (line 922)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 923):
    
    # Assigning a Call to a Name (line 923):
    
    # Call to astype(...): (line 923)
    # Processing the call arguments (line 923)
    # Getting the type of 'np' (line 923)
    np_166432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 923)
    double_166433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 21), np_166432, 'double')
    # Processing the call keyword arguments (line 923)
    kwargs_166434 = {}
    # Getting the type of 'c' (line 923)
    c_166430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 923)
    astype_166431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 12), c_166430, 'astype')
    # Calling astype(args, kwargs) (line 923)
    astype_call_result_166435 = invoke(stypy.reporting.localization.Localization(__file__, 923, 12), astype_166431, *[double_166433], **kwargs_166434)
    
    # Assigning a type to the variable 'c' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'c', astype_call_result_166435)
    # SSA join for if statement (line 922)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 924)
    # Processing the call arguments (line 924)
    # Getting the type of 'x' (line 924)
    x_166437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 924)
    tuple_166438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 924)
    # Adding element type (line 924)
    # Getting the type of 'tuple' (line 924)
    tuple_166439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 22), tuple_166438, tuple_166439)
    # Adding element type (line 924)
    # Getting the type of 'list' (line 924)
    list_166440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 924, 22), tuple_166438, list_166440)
    
    # Processing the call keyword arguments (line 924)
    kwargs_166441 = {}
    # Getting the type of 'isinstance' (line 924)
    isinstance_166436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 924)
    isinstance_call_result_166442 = invoke(stypy.reporting.localization.Localization(__file__, 924, 7), isinstance_166436, *[x_166437, tuple_166438], **kwargs_166441)
    
    # Testing the type of an if condition (line 924)
    if_condition_166443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 924, 4), isinstance_call_result_166442)
    # Assigning a type to the variable 'if_condition_166443' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'if_condition_166443', if_condition_166443)
    # SSA begins for if statement (line 924)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 925):
    
    # Assigning a Call to a Name (line 925):
    
    # Call to asarray(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'x' (line 925)
    x_166446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 23), 'x', False)
    # Processing the call keyword arguments (line 925)
    kwargs_166447 = {}
    # Getting the type of 'np' (line 925)
    np_166444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 925)
    asarray_166445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 12), np_166444, 'asarray')
    # Calling asarray(args, kwargs) (line 925)
    asarray_call_result_166448 = invoke(stypy.reporting.localization.Localization(__file__, 925, 12), asarray_166445, *[x_166446], **kwargs_166447)
    
    # Assigning a type to the variable 'x' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 8), 'x', asarray_call_result_166448)
    # SSA join for if statement (line 924)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'x' (line 926)
    x_166450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 18), 'x', False)
    # Getting the type of 'np' (line 926)
    np_166451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 926)
    ndarray_166452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 21), np_166451, 'ndarray')
    # Processing the call keyword arguments (line 926)
    kwargs_166453 = {}
    # Getting the type of 'isinstance' (line 926)
    isinstance_166449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 926)
    isinstance_call_result_166454 = invoke(stypy.reporting.localization.Localization(__file__, 926, 7), isinstance_166449, *[x_166450, ndarray_166452], **kwargs_166453)
    
    # Getting the type of 'tensor' (line 926)
    tensor_166455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 37), 'tensor')
    # Applying the binary operator 'and' (line 926)
    result_and_keyword_166456 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 7), 'and', isinstance_call_result_166454, tensor_166455)
    
    # Testing the type of an if condition (line 926)
    if_condition_166457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 926, 4), result_and_keyword_166456)
    # Assigning a type to the variable 'if_condition_166457' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'if_condition_166457', if_condition_166457)
    # SSA begins for if statement (line 926)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 927):
    
    # Assigning a Call to a Name (line 927):
    
    # Call to reshape(...): (line 927)
    # Processing the call arguments (line 927)
    # Getting the type of 'c' (line 927)
    c_166460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 927)
    shape_166461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 22), c_166460, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 927)
    tuple_166462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 927)
    # Adding element type (line 927)
    int_166463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 927, 33), tuple_166462, int_166463)
    
    # Getting the type of 'x' (line 927)
    x_166464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 927)
    ndim_166465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 37), x_166464, 'ndim')
    # Applying the binary operator '*' (line 927)
    result_mul_166466 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 32), '*', tuple_166462, ndim_166465)
    
    # Applying the binary operator '+' (line 927)
    result_add_166467 = python_operator(stypy.reporting.localization.Localization(__file__, 927, 22), '+', shape_166461, result_mul_166466)
    
    # Processing the call keyword arguments (line 927)
    kwargs_166468 = {}
    # Getting the type of 'c' (line 927)
    c_166458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 927)
    reshape_166459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 12), c_166458, 'reshape')
    # Calling reshape(args, kwargs) (line 927)
    reshape_call_result_166469 = invoke(stypy.reporting.localization.Localization(__file__, 927, 12), reshape_166459, *[result_add_166467], **kwargs_166468)
    
    # Assigning a type to the variable 'c' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 8), 'c', reshape_call_result_166469)
    # SSA join for if statement (line 926)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 929):
    
    # Assigning a BinOp to a Name (line 929):
    # Getting the type of 'x' (line 929)
    x_166470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 9), 'x')
    int_166471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 11), 'int')
    # Applying the binary operator '*' (line 929)
    result_mul_166472 = python_operator(stypy.reporting.localization.Localization(__file__, 929, 9), '*', x_166470, int_166471)
    
    # Assigning a type to the variable 'x2' (line 929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 4), 'x2', result_mul_166472)
    
    
    
    # Call to len(...): (line 930)
    # Processing the call arguments (line 930)
    # Getting the type of 'c' (line 930)
    c_166474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 11), 'c', False)
    # Processing the call keyword arguments (line 930)
    kwargs_166475 = {}
    # Getting the type of 'len' (line 930)
    len_166473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 7), 'len', False)
    # Calling len(args, kwargs) (line 930)
    len_call_result_166476 = invoke(stypy.reporting.localization.Localization(__file__, 930, 7), len_166473, *[c_166474], **kwargs_166475)
    
    int_166477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 17), 'int')
    # Applying the binary operator '==' (line 930)
    result_eq_166478 = python_operator(stypy.reporting.localization.Localization(__file__, 930, 7), '==', len_call_result_166476, int_166477)
    
    # Testing the type of an if condition (line 930)
    if_condition_166479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 930, 4), result_eq_166478)
    # Assigning a type to the variable 'if_condition_166479' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 4), 'if_condition_166479', if_condition_166479)
    # SSA begins for if statement (line 930)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 931):
    
    # Assigning a Subscript to a Name (line 931):
    
    # Obtaining the type of the subscript
    int_166480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 15), 'int')
    # Getting the type of 'c' (line 931)
    c_166481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 931)
    getitem___166482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 13), c_166481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 931)
    subscript_call_result_166483 = invoke(stypy.reporting.localization.Localization(__file__, 931, 13), getitem___166482, int_166480)
    
    # Assigning a type to the variable 'c0' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 8), 'c0', subscript_call_result_166483)
    
    # Assigning a Num to a Name (line 932):
    
    # Assigning a Num to a Name (line 932):
    int_166484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 13), 'int')
    # Assigning a type to the variable 'c1' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'c1', int_166484)
    # SSA branch for the else part of an if statement (line 930)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'c' (line 933)
    c_166486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 13), 'c', False)
    # Processing the call keyword arguments (line 933)
    kwargs_166487 = {}
    # Getting the type of 'len' (line 933)
    len_166485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 9), 'len', False)
    # Calling len(args, kwargs) (line 933)
    len_call_result_166488 = invoke(stypy.reporting.localization.Localization(__file__, 933, 9), len_166485, *[c_166486], **kwargs_166487)
    
    int_166489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 19), 'int')
    # Applying the binary operator '==' (line 933)
    result_eq_166490 = python_operator(stypy.reporting.localization.Localization(__file__, 933, 9), '==', len_call_result_166488, int_166489)
    
    # Testing the type of an if condition (line 933)
    if_condition_166491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 933, 9), result_eq_166490)
    # Assigning a type to the variable 'if_condition_166491' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 9), 'if_condition_166491', if_condition_166491)
    # SSA begins for if statement (line 933)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 934):
    
    # Assigning a Subscript to a Name (line 934):
    
    # Obtaining the type of the subscript
    int_166492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 15), 'int')
    # Getting the type of 'c' (line 934)
    c_166493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 934)
    getitem___166494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 13), c_166493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 934)
    subscript_call_result_166495 = invoke(stypy.reporting.localization.Localization(__file__, 934, 13), getitem___166494, int_166492)
    
    # Assigning a type to the variable 'c0' (line 934)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 934, 8), 'c0', subscript_call_result_166495)
    
    # Assigning a Subscript to a Name (line 935):
    
    # Assigning a Subscript to a Name (line 935):
    
    # Obtaining the type of the subscript
    int_166496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 15), 'int')
    # Getting the type of 'c' (line 935)
    c_166497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 935)
    getitem___166498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 13), c_166497, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 935)
    subscript_call_result_166499 = invoke(stypy.reporting.localization.Localization(__file__, 935, 13), getitem___166498, int_166496)
    
    # Assigning a type to the variable 'c1' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'c1', subscript_call_result_166499)
    # SSA branch for the else part of an if statement (line 933)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 937):
    
    # Assigning a Call to a Name (line 937):
    
    # Call to len(...): (line 937)
    # Processing the call arguments (line 937)
    # Getting the type of 'c' (line 937)
    c_166501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 17), 'c', False)
    # Processing the call keyword arguments (line 937)
    kwargs_166502 = {}
    # Getting the type of 'len' (line 937)
    len_166500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 13), 'len', False)
    # Calling len(args, kwargs) (line 937)
    len_call_result_166503 = invoke(stypy.reporting.localization.Localization(__file__, 937, 13), len_166500, *[c_166501], **kwargs_166502)
    
    # Assigning a type to the variable 'nd' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 8), 'nd', len_call_result_166503)
    
    # Assigning a Subscript to a Name (line 938):
    
    # Assigning a Subscript to a Name (line 938):
    
    # Obtaining the type of the subscript
    int_166504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 15), 'int')
    # Getting the type of 'c' (line 938)
    c_166505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 938)
    getitem___166506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 938, 13), c_166505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 938)
    subscript_call_result_166507 = invoke(stypy.reporting.localization.Localization(__file__, 938, 13), getitem___166506, int_166504)
    
    # Assigning a type to the variable 'c0' (line 938)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 8), 'c0', subscript_call_result_166507)
    
    # Assigning a Subscript to a Name (line 939):
    
    # Assigning a Subscript to a Name (line 939):
    
    # Obtaining the type of the subscript
    int_166508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 15), 'int')
    # Getting the type of 'c' (line 939)
    c_166509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 939)
    getitem___166510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 939, 13), c_166509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 939)
    subscript_call_result_166511 = invoke(stypy.reporting.localization.Localization(__file__, 939, 13), getitem___166510, int_166508)
    
    # Assigning a type to the variable 'c1' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 8), 'c1', subscript_call_result_166511)
    
    
    # Call to range(...): (line 940)
    # Processing the call arguments (line 940)
    int_166513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 23), 'int')
    
    # Call to len(...): (line 940)
    # Processing the call arguments (line 940)
    # Getting the type of 'c' (line 940)
    c_166515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 30), 'c', False)
    # Processing the call keyword arguments (line 940)
    kwargs_166516 = {}
    # Getting the type of 'len' (line 940)
    len_166514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 26), 'len', False)
    # Calling len(args, kwargs) (line 940)
    len_call_result_166517 = invoke(stypy.reporting.localization.Localization(__file__, 940, 26), len_166514, *[c_166515], **kwargs_166516)
    
    int_166518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 35), 'int')
    # Applying the binary operator '+' (line 940)
    result_add_166519 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 26), '+', len_call_result_166517, int_166518)
    
    # Processing the call keyword arguments (line 940)
    kwargs_166520 = {}
    # Getting the type of 'range' (line 940)
    range_166512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 17), 'range', False)
    # Calling range(args, kwargs) (line 940)
    range_call_result_166521 = invoke(stypy.reporting.localization.Localization(__file__, 940, 17), range_166512, *[int_166513, result_add_166519], **kwargs_166520)
    
    # Testing the type of a for loop iterable (line 940)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 940, 8), range_call_result_166521)
    # Getting the type of the for loop variable (line 940)
    for_loop_var_166522 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 940, 8), range_call_result_166521)
    # Assigning a type to the variable 'i' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'i', for_loop_var_166522)
    # SSA begins for a for statement (line 940)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 941):
    
    # Assigning a Name to a Name (line 941):
    # Getting the type of 'c0' (line 941)
    c0_166523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 12), 'tmp', c0_166523)
    
    # Assigning a BinOp to a Name (line 942):
    
    # Assigning a BinOp to a Name (line 942):
    # Getting the type of 'nd' (line 942)
    nd_166524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'nd')
    int_166525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 22), 'int')
    # Applying the binary operator '-' (line 942)
    result_sub_166526 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 17), '-', nd_166524, int_166525)
    
    # Assigning a type to the variable 'nd' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'nd', result_sub_166526)
    
    # Assigning a BinOp to a Name (line 943):
    
    # Assigning a BinOp to a Name (line 943):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 943)
    i_166527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 20), 'i')
    # Applying the 'usub' unary operator (line 943)
    result___neg___166528 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 19), 'usub', i_166527)
    
    # Getting the type of 'c' (line 943)
    c_166529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 17), 'c')
    # Obtaining the member '__getitem__' of a type (line 943)
    getitem___166530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 943, 17), c_166529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 943)
    subscript_call_result_166531 = invoke(stypy.reporting.localization.Localization(__file__, 943, 17), getitem___166530, result___neg___166528)
    
    # Getting the type of 'c1' (line 943)
    c1_166532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 25), 'c1')
    int_166533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 29), 'int')
    # Getting the type of 'nd' (line 943)
    nd_166534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 32), 'nd')
    int_166535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 37), 'int')
    # Applying the binary operator '-' (line 943)
    result_sub_166536 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 32), '-', nd_166534, int_166535)
    
    # Applying the binary operator '*' (line 943)
    result_mul_166537 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 29), '*', int_166533, result_sub_166536)
    
    # Applying the binary operator '*' (line 943)
    result_mul_166538 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 25), '*', c1_166532, result_mul_166537)
    
    # Applying the binary operator '-' (line 943)
    result_sub_166539 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 17), '-', subscript_call_result_166531, result_mul_166538)
    
    # Assigning a type to the variable 'c0' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 12), 'c0', result_sub_166539)
    
    # Assigning a BinOp to a Name (line 944):
    
    # Assigning a BinOp to a Name (line 944):
    # Getting the type of 'tmp' (line 944)
    tmp_166540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 17), 'tmp')
    # Getting the type of 'c1' (line 944)
    c1_166541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 23), 'c1')
    # Getting the type of 'x2' (line 944)
    x2_166542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 26), 'x2')
    # Applying the binary operator '*' (line 944)
    result_mul_166543 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 23), '*', c1_166541, x2_166542)
    
    # Applying the binary operator '+' (line 944)
    result_add_166544 = python_operator(stypy.reporting.localization.Localization(__file__, 944, 17), '+', tmp_166540, result_mul_166543)
    
    # Assigning a type to the variable 'c1' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 12), 'c1', result_add_166544)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 933)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 930)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 945)
    c0_166545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 11), 'c0')
    # Getting the type of 'c1' (line 945)
    c1_166546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 16), 'c1')
    # Getting the type of 'x2' (line 945)
    x2_166547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 19), 'x2')
    # Applying the binary operator '*' (line 945)
    result_mul_166548 = python_operator(stypy.reporting.localization.Localization(__file__, 945, 16), '*', c1_166546, x2_166547)
    
    # Applying the binary operator '+' (line 945)
    result_add_166549 = python_operator(stypy.reporting.localization.Localization(__file__, 945, 11), '+', c0_166545, result_mul_166548)
    
    # Assigning a type to the variable 'stypy_return_type' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 4), 'stypy_return_type', result_add_166549)
    
    # ################# End of 'hermval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermval' in the type store
    # Getting the type of 'stypy_return_type' (line 852)
    stypy_return_type_166550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermval'
    return stypy_return_type_166550

# Assigning a type to the variable 'hermval' (line 852)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 0), 'hermval', hermval)

@norecursion
def hermval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermval2d'
    module_type_store = module_type_store.open_function_context('hermval2d', 948, 0, False)
    
    # Passed parameters checking function
    hermval2d.stypy_localization = localization
    hermval2d.stypy_type_of_self = None
    hermval2d.stypy_type_store = module_type_store
    hermval2d.stypy_function_name = 'hermval2d'
    hermval2d.stypy_param_names_list = ['x', 'y', 'c']
    hermval2d.stypy_varargs_param_name = None
    hermval2d.stypy_kwargs_param_name = None
    hermval2d.stypy_call_defaults = defaults
    hermval2d.stypy_call_varargs = varargs
    hermval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermval2d(...)' code ##################

    str_166551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, (-1)), 'str', "\n    Evaluate a 2-D Hermite series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * H_i(x) * H_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points formed with\n        pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    hermval, hermgrid2d, hermval3d, hermgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 994)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 995):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 995)
    # Processing the call arguments (line 995)
    
    # Obtaining an instance of the builtin type 'tuple' (line 995)
    tuple_166554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 995)
    # Adding element type (line 995)
    # Getting the type of 'x' (line 995)
    x_166555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 25), tuple_166554, x_166555)
    # Adding element type (line 995)
    # Getting the type of 'y' (line 995)
    y_166556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 995, 25), tuple_166554, y_166556)
    
    # Processing the call keyword arguments (line 995)
    int_166557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 37), 'int')
    keyword_166558 = int_166557
    kwargs_166559 = {'copy': keyword_166558}
    # Getting the type of 'np' (line 995)
    np_166552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 995)
    array_166553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 15), np_166552, 'array')
    # Calling array(args, kwargs) (line 995)
    array_call_result_166560 = invoke(stypy.reporting.localization.Localization(__file__, 995, 15), array_166553, *[tuple_166554], **kwargs_166559)
    
    # Assigning a type to the variable 'call_assignment_165034' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165034', array_call_result_166560)
    
    # Assigning a Call to a Name (line 995):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_166563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 8), 'int')
    # Processing the call keyword arguments
    kwargs_166564 = {}
    # Getting the type of 'call_assignment_165034' (line 995)
    call_assignment_165034_166561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165034', False)
    # Obtaining the member '__getitem__' of a type (line 995)
    getitem___166562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 8), call_assignment_165034_166561, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_166565 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166562, *[int_166563], **kwargs_166564)
    
    # Assigning a type to the variable 'call_assignment_165035' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165035', getitem___call_result_166565)
    
    # Assigning a Name to a Name (line 995):
    # Getting the type of 'call_assignment_165035' (line 995)
    call_assignment_165035_166566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165035')
    # Assigning a type to the variable 'x' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'x', call_assignment_165035_166566)
    
    # Assigning a Call to a Name (line 995):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_166569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 8), 'int')
    # Processing the call keyword arguments
    kwargs_166570 = {}
    # Getting the type of 'call_assignment_165034' (line 995)
    call_assignment_165034_166567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165034', False)
    # Obtaining the member '__getitem__' of a type (line 995)
    getitem___166568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 995, 8), call_assignment_165034_166567, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_166571 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166568, *[int_166569], **kwargs_166570)
    
    # Assigning a type to the variable 'call_assignment_165036' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165036', getitem___call_result_166571)
    
    # Assigning a Name to a Name (line 995):
    # Getting the type of 'call_assignment_165036' (line 995)
    call_assignment_165036_166572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 8), 'call_assignment_165036')
    # Assigning a type to the variable 'y' (line 995)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 995, 11), 'y', call_assignment_165036_166572)
    # SSA branch for the except part of a try statement (line 994)
    # SSA branch for the except '<any exception>' branch of a try statement (line 994)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 997)
    # Processing the call arguments (line 997)
    str_166574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 997, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 997)
    kwargs_166575 = {}
    # Getting the type of 'ValueError' (line 997)
    ValueError_166573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 997)
    ValueError_call_result_166576 = invoke(stypy.reporting.localization.Localization(__file__, 997, 14), ValueError_166573, *[str_166574], **kwargs_166575)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 997, 8), ValueError_call_result_166576, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 994)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 999):
    
    # Assigning a Call to a Name (line 999):
    
    # Call to hermval(...): (line 999)
    # Processing the call arguments (line 999)
    # Getting the type of 'x' (line 999)
    x_166578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 16), 'x', False)
    # Getting the type of 'c' (line 999)
    c_166579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 19), 'c', False)
    # Processing the call keyword arguments (line 999)
    kwargs_166580 = {}
    # Getting the type of 'hermval' (line 999)
    hermval_166577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 999)
    hermval_call_result_166581 = invoke(stypy.reporting.localization.Localization(__file__, 999, 8), hermval_166577, *[x_166578, c_166579], **kwargs_166580)
    
    # Assigning a type to the variable 'c' (line 999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'c', hermval_call_result_166581)
    
    # Assigning a Call to a Name (line 1000):
    
    # Assigning a Call to a Name (line 1000):
    
    # Call to hermval(...): (line 1000)
    # Processing the call arguments (line 1000)
    # Getting the type of 'y' (line 1000)
    y_166583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 16), 'y', False)
    # Getting the type of 'c' (line 1000)
    c_166584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 19), 'c', False)
    # Processing the call keyword arguments (line 1000)
    # Getting the type of 'False' (line 1000)
    False_166585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 29), 'False', False)
    keyword_166586 = False_166585
    kwargs_166587 = {'tensor': keyword_166586}
    # Getting the type of 'hermval' (line 1000)
    hermval_166582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1000)
    hermval_call_result_166588 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 8), hermval_166582, *[y_166583, c_166584], **kwargs_166587)
    
    # Assigning a type to the variable 'c' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'c', hermval_call_result_166588)
    # Getting the type of 'c' (line 1001)
    c_166589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'stypy_return_type', c_166589)
    
    # ################# End of 'hermval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 948)
    stypy_return_type_166590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermval2d'
    return stypy_return_type_166590

# Assigning a type to the variable 'hermval2d' (line 948)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 0), 'hermval2d', hermval2d)

@norecursion
def hermgrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermgrid2d'
    module_type_store = module_type_store.open_function_context('hermgrid2d', 1004, 0, False)
    
    # Passed parameters checking function
    hermgrid2d.stypy_localization = localization
    hermgrid2d.stypy_type_of_self = None
    hermgrid2d.stypy_type_store = module_type_store
    hermgrid2d.stypy_function_name = 'hermgrid2d'
    hermgrid2d.stypy_param_names_list = ['x', 'y', 'c']
    hermgrid2d.stypy_varargs_param_name = None
    hermgrid2d.stypy_kwargs_param_name = None
    hermgrid2d.stypy_call_defaults = defaults
    hermgrid2d.stypy_call_varargs = varargs
    hermgrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermgrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermgrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermgrid2d(...)' code ##################

    str_166591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1053, (-1)), 'str', "\n    Evaluate a 2-D Hermite series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    hermval, hermval2d, hermval3d, hermgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1054):
    
    # Assigning a Call to a Name (line 1054):
    
    # Call to hermval(...): (line 1054)
    # Processing the call arguments (line 1054)
    # Getting the type of 'x' (line 1054)
    x_166593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 16), 'x', False)
    # Getting the type of 'c' (line 1054)
    c_166594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 19), 'c', False)
    # Processing the call keyword arguments (line 1054)
    kwargs_166595 = {}
    # Getting the type of 'hermval' (line 1054)
    hermval_166592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1054)
    hermval_call_result_166596 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 8), hermval_166592, *[x_166593, c_166594], **kwargs_166595)
    
    # Assigning a type to the variable 'c' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 4), 'c', hermval_call_result_166596)
    
    # Assigning a Call to a Name (line 1055):
    
    # Assigning a Call to a Name (line 1055):
    
    # Call to hermval(...): (line 1055)
    # Processing the call arguments (line 1055)
    # Getting the type of 'y' (line 1055)
    y_166598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 16), 'y', False)
    # Getting the type of 'c' (line 1055)
    c_166599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 19), 'c', False)
    # Processing the call keyword arguments (line 1055)
    kwargs_166600 = {}
    # Getting the type of 'hermval' (line 1055)
    hermval_166597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1055)
    hermval_call_result_166601 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 8), hermval_166597, *[y_166598, c_166599], **kwargs_166600)
    
    # Assigning a type to the variable 'c' (line 1055)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 4), 'c', hermval_call_result_166601)
    # Getting the type of 'c' (line 1056)
    c_166602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1056)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 4), 'stypy_return_type', c_166602)
    
    # ################# End of 'hermgrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermgrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1004)
    stypy_return_type_166603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermgrid2d'
    return stypy_return_type_166603

# Assigning a type to the variable 'hermgrid2d' (line 1004)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 0), 'hermgrid2d', hermgrid2d)

@norecursion
def hermval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermval3d'
    module_type_store = module_type_store.open_function_context('hermval3d', 1059, 0, False)
    
    # Passed parameters checking function
    hermval3d.stypy_localization = localization
    hermval3d.stypy_type_of_self = None
    hermval3d.stypy_type_store = module_type_store
    hermval3d.stypy_function_name = 'hermval3d'
    hermval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    hermval3d.stypy_varargs_param_name = None
    hermval3d.stypy_kwargs_param_name = None
    hermval3d.stypy_call_defaults = defaults
    hermval3d.stypy_call_varargs = varargs
    hermval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermval3d(...)' code ##################

    str_166604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, (-1)), 'str', "\n    Evaluate a 3-D Hermite series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * H_i(x) * H_j(y) * H_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    hermval, hermval2d, hermgrid2d, hermgrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1108):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1108)
    # Processing the call arguments (line 1108)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1108)
    tuple_166607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1108)
    # Adding element type (line 1108)
    # Getting the type of 'x' (line 1108)
    x_166608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_166607, x_166608)
    # Adding element type (line 1108)
    # Getting the type of 'y' (line 1108)
    y_166609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_166607, y_166609)
    # Adding element type (line 1108)
    # Getting the type of 'z' (line 1108)
    z_166610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1108, 28), tuple_166607, z_166610)
    
    # Processing the call keyword arguments (line 1108)
    int_166611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 43), 'int')
    keyword_166612 = int_166611
    kwargs_166613 = {'copy': keyword_166612}
    # Getting the type of 'np' (line 1108)
    np_166605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1108)
    array_166606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 18), np_166605, 'array')
    # Calling array(args, kwargs) (line 1108)
    array_call_result_166614 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 18), array_166606, *[tuple_166607], **kwargs_166613)
    
    # Assigning a type to the variable 'call_assignment_165037' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165037', array_call_result_166614)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_166617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_166618 = {}
    # Getting the type of 'call_assignment_165037' (line 1108)
    call_assignment_165037_166615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165037', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___166616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_165037_166615, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_166619 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166616, *[int_166617], **kwargs_166618)
    
    # Assigning a type to the variable 'call_assignment_165038' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165038', getitem___call_result_166619)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_165038' (line 1108)
    call_assignment_165038_166620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165038')
    # Assigning a type to the variable 'x' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'x', call_assignment_165038_166620)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_166623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_166624 = {}
    # Getting the type of 'call_assignment_165037' (line 1108)
    call_assignment_165037_166621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165037', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___166622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_165037_166621, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_166625 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166622, *[int_166623], **kwargs_166624)
    
    # Assigning a type to the variable 'call_assignment_165039' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165039', getitem___call_result_166625)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_165039' (line 1108)
    call_assignment_165039_166626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165039')
    # Assigning a type to the variable 'y' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 11), 'y', call_assignment_165039_166626)
    
    # Assigning a Call to a Name (line 1108):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_166629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 8), 'int')
    # Processing the call keyword arguments
    kwargs_166630 = {}
    # Getting the type of 'call_assignment_165037' (line 1108)
    call_assignment_165037_166627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165037', False)
    # Obtaining the member '__getitem__' of a type (line 1108)
    getitem___166628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1108, 8), call_assignment_165037_166627, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_166631 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166628, *[int_166629], **kwargs_166630)
    
    # Assigning a type to the variable 'call_assignment_165040' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165040', getitem___call_result_166631)
    
    # Assigning a Name to a Name (line 1108):
    # Getting the type of 'call_assignment_165040' (line 1108)
    call_assignment_165040_166632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 8), 'call_assignment_165040')
    # Assigning a type to the variable 'z' (line 1108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1108, 14), 'z', call_assignment_165040_166632)
    # SSA branch for the except part of a try statement (line 1107)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1107)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1110)
    # Processing the call arguments (line 1110)
    str_166634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1110, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 1110)
    kwargs_166635 = {}
    # Getting the type of 'ValueError' (line 1110)
    ValueError_166633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1110)
    ValueError_call_result_166636 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 14), ValueError_166633, *[str_166634], **kwargs_166635)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1110, 8), ValueError_call_result_166636, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1112):
    
    # Assigning a Call to a Name (line 1112):
    
    # Call to hermval(...): (line 1112)
    # Processing the call arguments (line 1112)
    # Getting the type of 'x' (line 1112)
    x_166638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 16), 'x', False)
    # Getting the type of 'c' (line 1112)
    c_166639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 19), 'c', False)
    # Processing the call keyword arguments (line 1112)
    kwargs_166640 = {}
    # Getting the type of 'hermval' (line 1112)
    hermval_166637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1112)
    hermval_call_result_166641 = invoke(stypy.reporting.localization.Localization(__file__, 1112, 8), hermval_166637, *[x_166638, c_166639], **kwargs_166640)
    
    # Assigning a type to the variable 'c' (line 1112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 4), 'c', hermval_call_result_166641)
    
    # Assigning a Call to a Name (line 1113):
    
    # Assigning a Call to a Name (line 1113):
    
    # Call to hermval(...): (line 1113)
    # Processing the call arguments (line 1113)
    # Getting the type of 'y' (line 1113)
    y_166643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 16), 'y', False)
    # Getting the type of 'c' (line 1113)
    c_166644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 19), 'c', False)
    # Processing the call keyword arguments (line 1113)
    # Getting the type of 'False' (line 1113)
    False_166645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 29), 'False', False)
    keyword_166646 = False_166645
    kwargs_166647 = {'tensor': keyword_166646}
    # Getting the type of 'hermval' (line 1113)
    hermval_166642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1113)
    hermval_call_result_166648 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 8), hermval_166642, *[y_166643, c_166644], **kwargs_166647)
    
    # Assigning a type to the variable 'c' (line 1113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 4), 'c', hermval_call_result_166648)
    
    # Assigning a Call to a Name (line 1114):
    
    # Assigning a Call to a Name (line 1114):
    
    # Call to hermval(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'z' (line 1114)
    z_166650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 16), 'z', False)
    # Getting the type of 'c' (line 1114)
    c_166651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 19), 'c', False)
    # Processing the call keyword arguments (line 1114)
    # Getting the type of 'False' (line 1114)
    False_166652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 29), 'False', False)
    keyword_166653 = False_166652
    kwargs_166654 = {'tensor': keyword_166653}
    # Getting the type of 'hermval' (line 1114)
    hermval_166649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1114)
    hermval_call_result_166655 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 8), hermval_166649, *[z_166650, c_166651], **kwargs_166654)
    
    # Assigning a type to the variable 'c' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'c', hermval_call_result_166655)
    # Getting the type of 'c' (line 1115)
    c_166656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1115, 4), 'stypy_return_type', c_166656)
    
    # ################# End of 'hermval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1059)
    stypy_return_type_166657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166657)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermval3d'
    return stypy_return_type_166657

# Assigning a type to the variable 'hermval3d' (line 1059)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 0), 'hermval3d', hermval3d)

@norecursion
def hermgrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermgrid3d'
    module_type_store = module_type_store.open_function_context('hermgrid3d', 1118, 0, False)
    
    # Passed parameters checking function
    hermgrid3d.stypy_localization = localization
    hermgrid3d.stypy_type_of_self = None
    hermgrid3d.stypy_type_store = module_type_store
    hermgrid3d.stypy_function_name = 'hermgrid3d'
    hermgrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    hermgrid3d.stypy_varargs_param_name = None
    hermgrid3d.stypy_kwargs_param_name = None
    hermgrid3d.stypy_call_defaults = defaults
    hermgrid3d.stypy_call_varargs = varargs
    hermgrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermgrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermgrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermgrid3d(...)' code ##################

    str_166658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1170, (-1)), 'str', "\n    Evaluate a 3-D Hermite series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * H_i(a) * H_j(b) * H_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    hermval, hermval2d, hermgrid2d, hermval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1171):
    
    # Assigning a Call to a Name (line 1171):
    
    # Call to hermval(...): (line 1171)
    # Processing the call arguments (line 1171)
    # Getting the type of 'x' (line 1171)
    x_166660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 16), 'x', False)
    # Getting the type of 'c' (line 1171)
    c_166661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 19), 'c', False)
    # Processing the call keyword arguments (line 1171)
    kwargs_166662 = {}
    # Getting the type of 'hermval' (line 1171)
    hermval_166659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1171)
    hermval_call_result_166663 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 8), hermval_166659, *[x_166660, c_166661], **kwargs_166662)
    
    # Assigning a type to the variable 'c' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 4), 'c', hermval_call_result_166663)
    
    # Assigning a Call to a Name (line 1172):
    
    # Assigning a Call to a Name (line 1172):
    
    # Call to hermval(...): (line 1172)
    # Processing the call arguments (line 1172)
    # Getting the type of 'y' (line 1172)
    y_166665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 16), 'y', False)
    # Getting the type of 'c' (line 1172)
    c_166666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 19), 'c', False)
    # Processing the call keyword arguments (line 1172)
    kwargs_166667 = {}
    # Getting the type of 'hermval' (line 1172)
    hermval_166664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1172)
    hermval_call_result_166668 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 8), hermval_166664, *[y_166665, c_166666], **kwargs_166667)
    
    # Assigning a type to the variable 'c' (line 1172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 4), 'c', hermval_call_result_166668)
    
    # Assigning a Call to a Name (line 1173):
    
    # Assigning a Call to a Name (line 1173):
    
    # Call to hermval(...): (line 1173)
    # Processing the call arguments (line 1173)
    # Getting the type of 'z' (line 1173)
    z_166670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 16), 'z', False)
    # Getting the type of 'c' (line 1173)
    c_166671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 19), 'c', False)
    # Processing the call keyword arguments (line 1173)
    kwargs_166672 = {}
    # Getting the type of 'hermval' (line 1173)
    hermval_166669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1173, 8), 'hermval', False)
    # Calling hermval(args, kwargs) (line 1173)
    hermval_call_result_166673 = invoke(stypy.reporting.localization.Localization(__file__, 1173, 8), hermval_166669, *[z_166670, c_166671], **kwargs_166672)
    
    # Assigning a type to the variable 'c' (line 1173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1173, 4), 'c', hermval_call_result_166673)
    # Getting the type of 'c' (line 1174)
    c_166674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 4), 'stypy_return_type', c_166674)
    
    # ################# End of 'hermgrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermgrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1118)
    stypy_return_type_166675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166675)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermgrid3d'
    return stypy_return_type_166675

# Assigning a type to the variable 'hermgrid3d' (line 1118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1118, 0), 'hermgrid3d', hermgrid3d)

@norecursion
def hermvander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermvander'
    module_type_store = module_type_store.open_function_context('hermvander', 1177, 0, False)
    
    # Passed parameters checking function
    hermvander.stypy_localization = localization
    hermvander.stypy_type_of_self = None
    hermvander.stypy_type_store = module_type_store
    hermvander.stypy_function_name = 'hermvander'
    hermvander.stypy_param_names_list = ['x', 'deg']
    hermvander.stypy_varargs_param_name = None
    hermvander.stypy_kwargs_param_name = None
    hermvander.stypy_call_defaults = defaults
    hermvander.stypy_call_varargs = varargs
    hermvander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermvander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermvander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermvander(...)' code ##################

    str_166676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1220, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = H_i(x),\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the Hermite polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    array ``V = hermvander(x, n)``, then ``np.dot(V, c)`` and\n    ``hermval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of Hermite series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo-Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding Hermite polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermvander\n    >>> x = np.array([-1, 0, 1])\n    >>> hermvander(x, 3)\n    array([[ 1., -2.,  2.,  4.],\n           [ 1.,  0., -2., -0.],\n           [ 1.,  2.,  2., -4.]])\n\n    ')
    
    # Assigning a Call to a Name (line 1221):
    
    # Assigning a Call to a Name (line 1221):
    
    # Call to int(...): (line 1221)
    # Processing the call arguments (line 1221)
    # Getting the type of 'deg' (line 1221)
    deg_166678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 15), 'deg', False)
    # Processing the call keyword arguments (line 1221)
    kwargs_166679 = {}
    # Getting the type of 'int' (line 1221)
    int_166677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 11), 'int', False)
    # Calling int(args, kwargs) (line 1221)
    int_call_result_166680 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 11), int_166677, *[deg_166678], **kwargs_166679)
    
    # Assigning a type to the variable 'ideg' (line 1221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1221, 4), 'ideg', int_call_result_166680)
    
    
    # Getting the type of 'ideg' (line 1222)
    ideg_166681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 7), 'ideg')
    # Getting the type of 'deg' (line 1222)
    deg_166682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 15), 'deg')
    # Applying the binary operator '!=' (line 1222)
    result_ne_166683 = python_operator(stypy.reporting.localization.Localization(__file__, 1222, 7), '!=', ideg_166681, deg_166682)
    
    # Testing the type of an if condition (line 1222)
    if_condition_166684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1222, 4), result_ne_166683)
    # Assigning a type to the variable 'if_condition_166684' (line 1222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 4), 'if_condition_166684', if_condition_166684)
    # SSA begins for if statement (line 1222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1223)
    # Processing the call arguments (line 1223)
    str_166686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1223)
    kwargs_166687 = {}
    # Getting the type of 'ValueError' (line 1223)
    ValueError_166685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1223)
    ValueError_call_result_166688 = invoke(stypy.reporting.localization.Localization(__file__, 1223, 14), ValueError_166685, *[str_166686], **kwargs_166687)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1223, 8), ValueError_call_result_166688, 'raise parameter', BaseException)
    # SSA join for if statement (line 1222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1224)
    ideg_166689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 7), 'ideg')
    int_166690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 14), 'int')
    # Applying the binary operator '<' (line 1224)
    result_lt_166691 = python_operator(stypy.reporting.localization.Localization(__file__, 1224, 7), '<', ideg_166689, int_166690)
    
    # Testing the type of an if condition (line 1224)
    if_condition_166692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1224, 4), result_lt_166691)
    # Assigning a type to the variable 'if_condition_166692' (line 1224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1224, 4), 'if_condition_166692', if_condition_166692)
    # SSA begins for if statement (line 1224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1225)
    # Processing the call arguments (line 1225)
    str_166694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1225)
    kwargs_166695 = {}
    # Getting the type of 'ValueError' (line 1225)
    ValueError_166693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1225)
    ValueError_call_result_166696 = invoke(stypy.reporting.localization.Localization(__file__, 1225, 14), ValueError_166693, *[str_166694], **kwargs_166695)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1225, 8), ValueError_call_result_166696, 'raise parameter', BaseException)
    # SSA join for if statement (line 1224)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1227):
    
    # Assigning a BinOp to a Name (line 1227):
    
    # Call to array(...): (line 1227)
    # Processing the call arguments (line 1227)
    # Getting the type of 'x' (line 1227)
    x_166699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 17), 'x', False)
    # Processing the call keyword arguments (line 1227)
    int_166700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 25), 'int')
    keyword_166701 = int_166700
    int_166702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 34), 'int')
    keyword_166703 = int_166702
    kwargs_166704 = {'copy': keyword_166701, 'ndmin': keyword_166703}
    # Getting the type of 'np' (line 1227)
    np_166697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1227)
    array_166698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 8), np_166697, 'array')
    # Calling array(args, kwargs) (line 1227)
    array_call_result_166705 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 8), array_166698, *[x_166699], **kwargs_166704)
    
    float_166706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1227, 39), 'float')
    # Applying the binary operator '+' (line 1227)
    result_add_166707 = python_operator(stypy.reporting.localization.Localization(__file__, 1227, 8), '+', array_call_result_166705, float_166706)
    
    # Assigning a type to the variable 'x' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'x', result_add_166707)
    
    # Assigning a BinOp to a Name (line 1228):
    
    # Assigning a BinOp to a Name (line 1228):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1228)
    tuple_166708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1228)
    # Adding element type (line 1228)
    # Getting the type of 'ideg' (line 1228)
    ideg_166709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 12), 'ideg')
    int_166710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, 19), 'int')
    # Applying the binary operator '+' (line 1228)
    result_add_166711 = python_operator(stypy.reporting.localization.Localization(__file__, 1228, 12), '+', ideg_166709, int_166710)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1228, 12), tuple_166708, result_add_166711)
    
    # Getting the type of 'x' (line 1228)
    x_166712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1228)
    shape_166713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 25), x_166712, 'shape')
    # Applying the binary operator '+' (line 1228)
    result_add_166714 = python_operator(stypy.reporting.localization.Localization(__file__, 1228, 11), '+', tuple_166708, shape_166713)
    
    # Assigning a type to the variable 'dims' (line 1228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 4), 'dims', result_add_166714)
    
    # Assigning a Attribute to a Name (line 1229):
    
    # Assigning a Attribute to a Name (line 1229):
    # Getting the type of 'x' (line 1229)
    x_166715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1229)
    dtype_166716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1229, 11), x_166715, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1229, 4), 'dtyp', dtype_166716)
    
    # Assigning a Call to a Name (line 1230):
    
    # Assigning a Call to a Name (line 1230):
    
    # Call to empty(...): (line 1230)
    # Processing the call arguments (line 1230)
    # Getting the type of 'dims' (line 1230)
    dims_166719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 17), 'dims', False)
    # Processing the call keyword arguments (line 1230)
    # Getting the type of 'dtyp' (line 1230)
    dtyp_166720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 29), 'dtyp', False)
    keyword_166721 = dtyp_166720
    kwargs_166722 = {'dtype': keyword_166721}
    # Getting the type of 'np' (line 1230)
    np_166717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1230)
    empty_166718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1230, 8), np_166717, 'empty')
    # Calling empty(args, kwargs) (line 1230)
    empty_call_result_166723 = invoke(stypy.reporting.localization.Localization(__file__, 1230, 8), empty_166718, *[dims_166719], **kwargs_166722)
    
    # Assigning a type to the variable 'v' (line 1230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 4), 'v', empty_call_result_166723)
    
    # Assigning a BinOp to a Subscript (line 1231):
    
    # Assigning a BinOp to a Subscript (line 1231):
    # Getting the type of 'x' (line 1231)
    x_166724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 11), 'x')
    int_166725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 13), 'int')
    # Applying the binary operator '*' (line 1231)
    result_mul_166726 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '*', x_166724, int_166725)
    
    int_166727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 17), 'int')
    # Applying the binary operator '+' (line 1231)
    result_add_166728 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 11), '+', result_mul_166726, int_166727)
    
    # Getting the type of 'v' (line 1231)
    v_166729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 4), 'v')
    int_166730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 6), 'int')
    # Storing an element on a container (line 1231)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1231, 4), v_166729, (int_166730, result_add_166728))
    
    
    # Getting the type of 'ideg' (line 1232)
    ideg_166731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 7), 'ideg')
    int_166732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 14), 'int')
    # Applying the binary operator '>' (line 1232)
    result_gt_166733 = python_operator(stypy.reporting.localization.Localization(__file__, 1232, 7), '>', ideg_166731, int_166732)
    
    # Testing the type of an if condition (line 1232)
    if_condition_166734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1232, 4), result_gt_166733)
    # Assigning a type to the variable 'if_condition_166734' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 4), 'if_condition_166734', if_condition_166734)
    # SSA begins for if statement (line 1232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1233):
    
    # Assigning a BinOp to a Name (line 1233):
    # Getting the type of 'x' (line 1233)
    x_166735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 13), 'x')
    int_166736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 15), 'int')
    # Applying the binary operator '*' (line 1233)
    result_mul_166737 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 13), '*', x_166735, int_166736)
    
    # Assigning a type to the variable 'x2' (line 1233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1233, 8), 'x2', result_mul_166737)
    
    # Assigning a Name to a Subscript (line 1234):
    
    # Assigning a Name to a Subscript (line 1234):
    # Getting the type of 'x2' (line 1234)
    x2_166738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 15), 'x2')
    # Getting the type of 'v' (line 1234)
    v_166739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 8), 'v')
    int_166740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 10), 'int')
    # Storing an element on a container (line 1234)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1234, 8), v_166739, (int_166740, x2_166738))
    
    
    # Call to range(...): (line 1235)
    # Processing the call arguments (line 1235)
    int_166742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 23), 'int')
    # Getting the type of 'ideg' (line 1235)
    ideg_166743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 26), 'ideg', False)
    int_166744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 33), 'int')
    # Applying the binary operator '+' (line 1235)
    result_add_166745 = python_operator(stypy.reporting.localization.Localization(__file__, 1235, 26), '+', ideg_166743, int_166744)
    
    # Processing the call keyword arguments (line 1235)
    kwargs_166746 = {}
    # Getting the type of 'range' (line 1235)
    range_166741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 17), 'range', False)
    # Calling range(args, kwargs) (line 1235)
    range_call_result_166747 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 17), range_166741, *[int_166742, result_add_166745], **kwargs_166746)
    
    # Testing the type of a for loop iterable (line 1235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1235, 8), range_call_result_166747)
    # Getting the type of the for loop variable (line 1235)
    for_loop_var_166748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1235, 8), range_call_result_166747)
    # Assigning a type to the variable 'i' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 8), 'i', for_loop_var_166748)
    # SSA begins for a for statement (line 1235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1236):
    
    # Assigning a BinOp to a Subscript (line 1236):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1236)
    i_166749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 22), 'i')
    int_166750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 24), 'int')
    # Applying the binary operator '-' (line 1236)
    result_sub_166751 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 22), '-', i_166749, int_166750)
    
    # Getting the type of 'v' (line 1236)
    v_166752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 20), 'v')
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___166753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 20), v_166752, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_166754 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 20), getitem___166753, result_sub_166751)
    
    # Getting the type of 'x2' (line 1236)
    x2_166755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 27), 'x2')
    # Applying the binary operator '*' (line 1236)
    result_mul_166756 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 20), '*', subscript_call_result_166754, x2_166755)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1236)
    i_166757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 34), 'i')
    int_166758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 36), 'int')
    # Applying the binary operator '-' (line 1236)
    result_sub_166759 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 34), '-', i_166757, int_166758)
    
    # Getting the type of 'v' (line 1236)
    v_166760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 32), 'v')
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___166761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 32), v_166760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_166762 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 32), getitem___166761, result_sub_166759)
    
    int_166763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 40), 'int')
    # Getting the type of 'i' (line 1236)
    i_166764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 43), 'i')
    int_166765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 47), 'int')
    # Applying the binary operator '-' (line 1236)
    result_sub_166766 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 43), '-', i_166764, int_166765)
    
    # Applying the binary operator '*' (line 1236)
    result_mul_166767 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 40), '*', int_166763, result_sub_166766)
    
    # Applying the binary operator '*' (line 1236)
    result_mul_166768 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 32), '*', subscript_call_result_166762, result_mul_166767)
    
    # Applying the binary operator '-' (line 1236)
    result_sub_166769 = python_operator(stypy.reporting.localization.Localization(__file__, 1236, 20), '-', result_mul_166756, result_mul_166768)
    
    # Getting the type of 'v' (line 1236)
    v_166770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 12), 'v')
    # Getting the type of 'i' (line 1236)
    i_166771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 14), 'i')
    # Storing an element on a container (line 1236)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1236, 12), v_166770, (i_166771, result_sub_166769))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1237)
    # Processing the call arguments (line 1237)
    # Getting the type of 'v' (line 1237)
    v_166774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 23), 'v', False)
    int_166775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1237, 26), 'int')
    # Getting the type of 'v' (line 1237)
    v_166776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1237)
    ndim_166777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 29), v_166776, 'ndim')
    # Processing the call keyword arguments (line 1237)
    kwargs_166778 = {}
    # Getting the type of 'np' (line 1237)
    np_166772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1237)
    rollaxis_166773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1237, 11), np_166772, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1237)
    rollaxis_call_result_166779 = invoke(stypy.reporting.localization.Localization(__file__, 1237, 11), rollaxis_166773, *[v_166774, int_166775, ndim_166777], **kwargs_166778)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 4), 'stypy_return_type', rollaxis_call_result_166779)
    
    # ################# End of 'hermvander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermvander' in the type store
    # Getting the type of 'stypy_return_type' (line 1177)
    stypy_return_type_166780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermvander'
    return stypy_return_type_166780

# Assigning a type to the variable 'hermvander' (line 1177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 0), 'hermvander', hermvander)

@norecursion
def hermvander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermvander2d'
    module_type_store = module_type_store.open_function_context('hermvander2d', 1240, 0, False)
    
    # Passed parameters checking function
    hermvander2d.stypy_localization = localization
    hermvander2d.stypy_type_of_self = None
    hermvander2d.stypy_type_store = module_type_store
    hermvander2d.stypy_function_name = 'hermvander2d'
    hermvander2d.stypy_param_names_list = ['x', 'y', 'deg']
    hermvander2d.stypy_varargs_param_name = None
    hermvander2d.stypy_kwargs_param_name = None
    hermvander2d.stypy_call_defaults = defaults
    hermvander2d.stypy_call_varargs = varargs
    hermvander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermvander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermvander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermvander2d(...)' code ##################

    str_166781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = H_i(x) * H_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the Hermite polynomials.\n\n    If ``V = hermvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``hermval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D Hermite\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    hermvander, hermvander3d. hermval2d, hermval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1290):
    
    # Assigning a ListComp to a Name (line 1290):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1290)
    deg_166786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 28), 'deg')
    comprehension_166787 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 12), deg_166786)
    # Assigning a type to the variable 'd' (line 1290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 12), 'd', comprehension_166787)
    
    # Call to int(...): (line 1290)
    # Processing the call arguments (line 1290)
    # Getting the type of 'd' (line 1290)
    d_166783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 16), 'd', False)
    # Processing the call keyword arguments (line 1290)
    kwargs_166784 = {}
    # Getting the type of 'int' (line 1290)
    int_166782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 12), 'int', False)
    # Calling int(args, kwargs) (line 1290)
    int_call_result_166785 = invoke(stypy.reporting.localization.Localization(__file__, 1290, 12), int_166782, *[d_166783], **kwargs_166784)
    
    list_166788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1290, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1290, 12), list_166788, int_call_result_166785)
    # Assigning a type to the variable 'ideg' (line 1290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1290, 4), 'ideg', list_166788)
    
    # Assigning a ListComp to a Name (line 1291):
    
    # Assigning a ListComp to a Name (line 1291):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1291)
    # Processing the call arguments (line 1291)
    # Getting the type of 'ideg' (line 1291)
    ideg_166797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1291)
    deg_166798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 59), 'deg', False)
    # Processing the call keyword arguments (line 1291)
    kwargs_166799 = {}
    # Getting the type of 'zip' (line 1291)
    zip_166796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1291)
    zip_call_result_166800 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 49), zip_166796, *[ideg_166797, deg_166798], **kwargs_166799)
    
    comprehension_166801 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 16), zip_call_result_166800)
    # Assigning a type to the variable 'id' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 16), comprehension_166801))
    # Assigning a type to the variable 'd' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 16), comprehension_166801))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1291)
    id_166789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 16), 'id')
    # Getting the type of 'd' (line 1291)
    d_166790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 22), 'd')
    # Applying the binary operator '==' (line 1291)
    result_eq_166791 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 16), '==', id_166789, d_166790)
    
    
    # Getting the type of 'id' (line 1291)
    id_166792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 28), 'id')
    int_166793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 34), 'int')
    # Applying the binary operator '>=' (line 1291)
    result_ge_166794 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 28), '>=', id_166792, int_166793)
    
    # Applying the binary operator 'and' (line 1291)
    result_and_keyword_166795 = python_operator(stypy.reporting.localization.Localization(__file__, 1291, 16), 'and', result_eq_166791, result_ge_166794)
    
    list_166802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1291, 16), list_166802, result_and_keyword_166795)
    # Assigning a type to the variable 'is_valid' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'is_valid', list_166802)
    
    
    # Getting the type of 'is_valid' (line 1292)
    is_valid_166803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1292)
    list_166804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1292)
    # Adding element type (line 1292)
    int_166805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 19), list_166804, int_166805)
    # Adding element type (line 1292)
    int_166806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 19), list_166804, int_166806)
    
    # Applying the binary operator '!=' (line 1292)
    result_ne_166807 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 7), '!=', is_valid_166803, list_166804)
    
    # Testing the type of an if condition (line 1292)
    if_condition_166808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1292, 4), result_ne_166807)
    # Assigning a type to the variable 'if_condition_166808' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'if_condition_166808', if_condition_166808)
    # SSA begins for if statement (line 1292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1293)
    # Processing the call arguments (line 1293)
    str_166810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1293, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1293)
    kwargs_166811 = {}
    # Getting the type of 'ValueError' (line 1293)
    ValueError_166809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1293, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1293)
    ValueError_call_result_166812 = invoke(stypy.reporting.localization.Localization(__file__, 1293, 14), ValueError_166809, *[str_166810], **kwargs_166811)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1293, 8), ValueError_call_result_166812, 'raise parameter', BaseException)
    # SSA join for if statement (line 1292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1294):
    
    # Assigning a Subscript to a Name (line 1294):
    
    # Obtaining the type of the subscript
    int_166813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 4), 'int')
    # Getting the type of 'ideg' (line 1294)
    ideg_166814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1294)
    getitem___166815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 4), ideg_166814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1294)
    subscript_call_result_166816 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 4), getitem___166815, int_166813)
    
    # Assigning a type to the variable 'tuple_var_assignment_165041' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_165041', subscript_call_result_166816)
    
    # Assigning a Subscript to a Name (line 1294):
    
    # Obtaining the type of the subscript
    int_166817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1294, 4), 'int')
    # Getting the type of 'ideg' (line 1294)
    ideg_166818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1294)
    getitem___166819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1294, 4), ideg_166818, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1294)
    subscript_call_result_166820 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 4), getitem___166819, int_166817)
    
    # Assigning a type to the variable 'tuple_var_assignment_165042' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_165042', subscript_call_result_166820)
    
    # Assigning a Name to a Name (line 1294):
    # Getting the type of 'tuple_var_assignment_165041' (line 1294)
    tuple_var_assignment_165041_166821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_165041')
    # Assigning a type to the variable 'degx' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'degx', tuple_var_assignment_165041_166821)
    
    # Assigning a Name to a Name (line 1294):
    # Getting the type of 'tuple_var_assignment_165042' (line 1294)
    tuple_var_assignment_165042_166822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'tuple_var_assignment_165042')
    # Assigning a type to the variable 'degy' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 10), 'degy', tuple_var_assignment_165042_166822)
    
    # Assigning a BinOp to a Tuple (line 1295):
    
    # Assigning a Subscript to a Name (line 1295):
    
    # Obtaining the type of the subscript
    int_166823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 4), 'int')
    
    # Call to array(...): (line 1295)
    # Processing the call arguments (line 1295)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1295)
    tuple_166826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1295)
    # Adding element type (line 1295)
    # Getting the type of 'x' (line 1295)
    x_166827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1295, 21), tuple_166826, x_166827)
    # Adding element type (line 1295)
    # Getting the type of 'y' (line 1295)
    y_166828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1295, 21), tuple_166826, y_166828)
    
    # Processing the call keyword arguments (line 1295)
    int_166829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 33), 'int')
    keyword_166830 = int_166829
    kwargs_166831 = {'copy': keyword_166830}
    # Getting the type of 'np' (line 1295)
    np_166824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1295)
    array_166825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1295, 11), np_166824, 'array')
    # Calling array(args, kwargs) (line 1295)
    array_call_result_166832 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 11), array_166825, *[tuple_166826], **kwargs_166831)
    
    float_166833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 38), 'float')
    # Applying the binary operator '+' (line 1295)
    result_add_166834 = python_operator(stypy.reporting.localization.Localization(__file__, 1295, 11), '+', array_call_result_166832, float_166833)
    
    # Obtaining the member '__getitem__' of a type (line 1295)
    getitem___166835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1295, 4), result_add_166834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1295)
    subscript_call_result_166836 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 4), getitem___166835, int_166823)
    
    # Assigning a type to the variable 'tuple_var_assignment_165043' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'tuple_var_assignment_165043', subscript_call_result_166836)
    
    # Assigning a Subscript to a Name (line 1295):
    
    # Obtaining the type of the subscript
    int_166837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 4), 'int')
    
    # Call to array(...): (line 1295)
    # Processing the call arguments (line 1295)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1295)
    tuple_166840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1295)
    # Adding element type (line 1295)
    # Getting the type of 'x' (line 1295)
    x_166841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1295, 21), tuple_166840, x_166841)
    # Adding element type (line 1295)
    # Getting the type of 'y' (line 1295)
    y_166842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1295, 21), tuple_166840, y_166842)
    
    # Processing the call keyword arguments (line 1295)
    int_166843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 33), 'int')
    keyword_166844 = int_166843
    kwargs_166845 = {'copy': keyword_166844}
    # Getting the type of 'np' (line 1295)
    np_166838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1295)
    array_166839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1295, 11), np_166838, 'array')
    # Calling array(args, kwargs) (line 1295)
    array_call_result_166846 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 11), array_166839, *[tuple_166840], **kwargs_166845)
    
    float_166847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1295, 38), 'float')
    # Applying the binary operator '+' (line 1295)
    result_add_166848 = python_operator(stypy.reporting.localization.Localization(__file__, 1295, 11), '+', array_call_result_166846, float_166847)
    
    # Obtaining the member '__getitem__' of a type (line 1295)
    getitem___166849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1295, 4), result_add_166848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1295)
    subscript_call_result_166850 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 4), getitem___166849, int_166837)
    
    # Assigning a type to the variable 'tuple_var_assignment_165044' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'tuple_var_assignment_165044', subscript_call_result_166850)
    
    # Assigning a Name to a Name (line 1295):
    # Getting the type of 'tuple_var_assignment_165043' (line 1295)
    tuple_var_assignment_165043_166851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'tuple_var_assignment_165043')
    # Assigning a type to the variable 'x' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'x', tuple_var_assignment_165043_166851)
    
    # Assigning a Name to a Name (line 1295):
    # Getting the type of 'tuple_var_assignment_165044' (line 1295)
    tuple_var_assignment_165044_166852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'tuple_var_assignment_165044')
    # Assigning a type to the variable 'y' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 7), 'y', tuple_var_assignment_165044_166852)
    
    # Assigning a Call to a Name (line 1297):
    
    # Assigning a Call to a Name (line 1297):
    
    # Call to hermvander(...): (line 1297)
    # Processing the call arguments (line 1297)
    # Getting the type of 'x' (line 1297)
    x_166854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 20), 'x', False)
    # Getting the type of 'degx' (line 1297)
    degx_166855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 23), 'degx', False)
    # Processing the call keyword arguments (line 1297)
    kwargs_166856 = {}
    # Getting the type of 'hermvander' (line 1297)
    hermvander_166853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 9), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1297)
    hermvander_call_result_166857 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 9), hermvander_166853, *[x_166854, degx_166855], **kwargs_166856)
    
    # Assigning a type to the variable 'vx' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'vx', hermvander_call_result_166857)
    
    # Assigning a Call to a Name (line 1298):
    
    # Assigning a Call to a Name (line 1298):
    
    # Call to hermvander(...): (line 1298)
    # Processing the call arguments (line 1298)
    # Getting the type of 'y' (line 1298)
    y_166859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 20), 'y', False)
    # Getting the type of 'degy' (line 1298)
    degy_166860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 23), 'degy', False)
    # Processing the call keyword arguments (line 1298)
    kwargs_166861 = {}
    # Getting the type of 'hermvander' (line 1298)
    hermvander_166858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1298, 9), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1298)
    hermvander_call_result_166862 = invoke(stypy.reporting.localization.Localization(__file__, 1298, 9), hermvander_166858, *[y_166859, degy_166860], **kwargs_166861)
    
    # Assigning a type to the variable 'vy' (line 1298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1298, 4), 'vy', hermvander_call_result_166862)
    
    # Assigning a BinOp to a Name (line 1299):
    
    # Assigning a BinOp to a Name (line 1299):
    
    # Obtaining the type of the subscript
    Ellipsis_166863 = Ellipsis
    # Getting the type of 'None' (line 1299)
    None_166864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 16), 'None')
    # Getting the type of 'vx' (line 1299)
    vx_166865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1299)
    getitem___166866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 8), vx_166865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1299)
    subscript_call_result_166867 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 8), getitem___166866, (Ellipsis_166863, None_166864))
    
    
    # Obtaining the type of the subscript
    Ellipsis_166868 = Ellipsis
    # Getting the type of 'None' (line 1299)
    None_166869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 30), 'None')
    slice_166870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1299, 22), None, None, None)
    # Getting the type of 'vy' (line 1299)
    vy_166871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1299, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1299)
    getitem___166872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1299, 22), vy_166871, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1299)
    subscript_call_result_166873 = invoke(stypy.reporting.localization.Localization(__file__, 1299, 22), getitem___166872, (Ellipsis_166868, None_166869, slice_166870))
    
    # Applying the binary operator '*' (line 1299)
    result_mul_166874 = python_operator(stypy.reporting.localization.Localization(__file__, 1299, 8), '*', subscript_call_result_166867, subscript_call_result_166873)
    
    # Assigning a type to the variable 'v' (line 1299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1299, 4), 'v', result_mul_166874)
    
    # Call to reshape(...): (line 1300)
    # Processing the call arguments (line 1300)
    
    # Obtaining the type of the subscript
    int_166877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1300, 30), 'int')
    slice_166878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1300, 21), None, int_166877, None)
    # Getting the type of 'v' (line 1300)
    v_166879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1300)
    shape_166880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 21), v_166879, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1300)
    getitem___166881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 21), shape_166880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1300)
    subscript_call_result_166882 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 21), getitem___166881, slice_166878)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1300)
    tuple_166883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1300, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1300)
    # Adding element type (line 1300)
    int_166884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1300, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1300, 37), tuple_166883, int_166884)
    
    # Applying the binary operator '+' (line 1300)
    result_add_166885 = python_operator(stypy.reporting.localization.Localization(__file__, 1300, 21), '+', subscript_call_result_166882, tuple_166883)
    
    # Processing the call keyword arguments (line 1300)
    kwargs_166886 = {}
    # Getting the type of 'v' (line 1300)
    v_166875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1300)
    reshape_166876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 11), v_166875, 'reshape')
    # Calling reshape(args, kwargs) (line 1300)
    reshape_call_result_166887 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 11), reshape_166876, *[result_add_166885], **kwargs_166886)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 4), 'stypy_return_type', reshape_call_result_166887)
    
    # ################# End of 'hermvander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermvander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1240)
    stypy_return_type_166888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_166888)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermvander2d'
    return stypy_return_type_166888

# Assigning a type to the variable 'hermvander2d' (line 1240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 0), 'hermvander2d', hermvander2d)

@norecursion
def hermvander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermvander3d'
    module_type_store = module_type_store.open_function_context('hermvander3d', 1303, 0, False)
    
    # Passed parameters checking function
    hermvander3d.stypy_localization = localization
    hermvander3d.stypy_type_of_self = None
    hermvander3d.stypy_type_store = module_type_store
    hermvander3d.stypy_function_name = 'hermvander3d'
    hermvander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    hermvander3d.stypy_varargs_param_name = None
    hermvander3d.stypy_kwargs_param_name = None
    hermvander3d.stypy_call_defaults = defaults
    hermvander3d.stypy_call_varargs = varargs
    hermvander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermvander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermvander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermvander3d(...)' code ##################

    str_166889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = H_i(x)*H_j(y)*H_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the Hermite polynomials.\n\n    If ``V = hermvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and  ``np.dot(V, c.flat)`` and ``hermval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D Hermite\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    hermvander, hermvander3d. hermval2d, hermval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1354):
    
    # Assigning a ListComp to a Name (line 1354):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1354)
    deg_166894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 28), 'deg')
    comprehension_166895 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 12), deg_166894)
    # Assigning a type to the variable 'd' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 12), 'd', comprehension_166895)
    
    # Call to int(...): (line 1354)
    # Processing the call arguments (line 1354)
    # Getting the type of 'd' (line 1354)
    d_166891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 16), 'd', False)
    # Processing the call keyword arguments (line 1354)
    kwargs_166892 = {}
    # Getting the type of 'int' (line 1354)
    int_166890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 12), 'int', False)
    # Calling int(args, kwargs) (line 1354)
    int_call_result_166893 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 12), int_166890, *[d_166891], **kwargs_166892)
    
    list_166896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1354, 12), list_166896, int_call_result_166893)
    # Assigning a type to the variable 'ideg' (line 1354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1354, 4), 'ideg', list_166896)
    
    # Assigning a ListComp to a Name (line 1355):
    
    # Assigning a ListComp to a Name (line 1355):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1355)
    # Processing the call arguments (line 1355)
    # Getting the type of 'ideg' (line 1355)
    ideg_166905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1355)
    deg_166906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 59), 'deg', False)
    # Processing the call keyword arguments (line 1355)
    kwargs_166907 = {}
    # Getting the type of 'zip' (line 1355)
    zip_166904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1355)
    zip_call_result_166908 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 49), zip_166904, *[ideg_166905, deg_166906], **kwargs_166907)
    
    comprehension_166909 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 16), zip_call_result_166908)
    # Assigning a type to the variable 'id' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 16), comprehension_166909))
    # Assigning a type to the variable 'd' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 16), comprehension_166909))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1355)
    id_166897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 16), 'id')
    # Getting the type of 'd' (line 1355)
    d_166898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 22), 'd')
    # Applying the binary operator '==' (line 1355)
    result_eq_166899 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 16), '==', id_166897, d_166898)
    
    
    # Getting the type of 'id' (line 1355)
    id_166900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 28), 'id')
    int_166901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 34), 'int')
    # Applying the binary operator '>=' (line 1355)
    result_ge_166902 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 28), '>=', id_166900, int_166901)
    
    # Applying the binary operator 'and' (line 1355)
    result_and_keyword_166903 = python_operator(stypy.reporting.localization.Localization(__file__, 1355, 16), 'and', result_eq_166899, result_ge_166902)
    
    list_166910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1355, 16), list_166910, result_and_keyword_166903)
    # Assigning a type to the variable 'is_valid' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'is_valid', list_166910)
    
    
    # Getting the type of 'is_valid' (line 1356)
    is_valid_166911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1356)
    list_166912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1356)
    # Adding element type (line 1356)
    int_166913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 19), list_166912, int_166913)
    # Adding element type (line 1356)
    int_166914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 19), list_166912, int_166914)
    # Adding element type (line 1356)
    int_166915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 19), list_166912, int_166915)
    
    # Applying the binary operator '!=' (line 1356)
    result_ne_166916 = python_operator(stypy.reporting.localization.Localization(__file__, 1356, 7), '!=', is_valid_166911, list_166912)
    
    # Testing the type of an if condition (line 1356)
    if_condition_166917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1356, 4), result_ne_166916)
    # Assigning a type to the variable 'if_condition_166917' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'if_condition_166917', if_condition_166917)
    # SSA begins for if statement (line 1356)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1357)
    # Processing the call arguments (line 1357)
    str_166919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1357, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1357)
    kwargs_166920 = {}
    # Getting the type of 'ValueError' (line 1357)
    ValueError_166918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1357, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1357)
    ValueError_call_result_166921 = invoke(stypy.reporting.localization.Localization(__file__, 1357, 14), ValueError_166918, *[str_166919], **kwargs_166920)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1357, 8), ValueError_call_result_166921, 'raise parameter', BaseException)
    # SSA join for if statement (line 1356)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1358):
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_166922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    # Getting the type of 'ideg' (line 1358)
    ideg_166923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___166924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), ideg_166923, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_166925 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___166924, int_166922)
    
    # Assigning a type to the variable 'tuple_var_assignment_165045' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165045', subscript_call_result_166925)
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_166926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    # Getting the type of 'ideg' (line 1358)
    ideg_166927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___166928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), ideg_166927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_166929 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___166928, int_166926)
    
    # Assigning a type to the variable 'tuple_var_assignment_165046' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165046', subscript_call_result_166929)
    
    # Assigning a Subscript to a Name (line 1358):
    
    # Obtaining the type of the subscript
    int_166930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1358, 4), 'int')
    # Getting the type of 'ideg' (line 1358)
    ideg_166931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1358)
    getitem___166932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1358, 4), ideg_166931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1358)
    subscript_call_result_166933 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 4), getitem___166932, int_166930)
    
    # Assigning a type to the variable 'tuple_var_assignment_165047' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165047', subscript_call_result_166933)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_165045' (line 1358)
    tuple_var_assignment_165045_166934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165045')
    # Assigning a type to the variable 'degx' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'degx', tuple_var_assignment_165045_166934)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_165046' (line 1358)
    tuple_var_assignment_165046_166935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165046')
    # Assigning a type to the variable 'degy' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 10), 'degy', tuple_var_assignment_165046_166935)
    
    # Assigning a Name to a Name (line 1358):
    # Getting the type of 'tuple_var_assignment_165047' (line 1358)
    tuple_var_assignment_165047_166936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'tuple_var_assignment_165047')
    # Assigning a type to the variable 'degz' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 16), 'degz', tuple_var_assignment_165047_166936)
    
    # Assigning a BinOp to a Tuple (line 1359):
    
    # Assigning a Subscript to a Name (line 1359):
    
    # Obtaining the type of the subscript
    int_166937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 4), 'int')
    
    # Call to array(...): (line 1359)
    # Processing the call arguments (line 1359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1359)
    tuple_166940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1359)
    # Adding element type (line 1359)
    # Getting the type of 'x' (line 1359)
    x_166941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166940, x_166941)
    # Adding element type (line 1359)
    # Getting the type of 'y' (line 1359)
    y_166942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166940, y_166942)
    # Adding element type (line 1359)
    # Getting the type of 'z' (line 1359)
    z_166943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166940, z_166943)
    
    # Processing the call keyword arguments (line 1359)
    int_166944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 39), 'int')
    keyword_166945 = int_166944
    kwargs_166946 = {'copy': keyword_166945}
    # Getting the type of 'np' (line 1359)
    np_166938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1359)
    array_166939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 14), np_166938, 'array')
    # Calling array(args, kwargs) (line 1359)
    array_call_result_166947 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 14), array_166939, *[tuple_166940], **kwargs_166946)
    
    float_166948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 44), 'float')
    # Applying the binary operator '+' (line 1359)
    result_add_166949 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 14), '+', array_call_result_166947, float_166948)
    
    # Obtaining the member '__getitem__' of a type (line 1359)
    getitem___166950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 4), result_add_166949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1359)
    subscript_call_result_166951 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 4), getitem___166950, int_166937)
    
    # Assigning a type to the variable 'tuple_var_assignment_165048' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165048', subscript_call_result_166951)
    
    # Assigning a Subscript to a Name (line 1359):
    
    # Obtaining the type of the subscript
    int_166952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 4), 'int')
    
    # Call to array(...): (line 1359)
    # Processing the call arguments (line 1359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1359)
    tuple_166955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1359)
    # Adding element type (line 1359)
    # Getting the type of 'x' (line 1359)
    x_166956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166955, x_166956)
    # Adding element type (line 1359)
    # Getting the type of 'y' (line 1359)
    y_166957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166955, y_166957)
    # Adding element type (line 1359)
    # Getting the type of 'z' (line 1359)
    z_166958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166955, z_166958)
    
    # Processing the call keyword arguments (line 1359)
    int_166959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 39), 'int')
    keyword_166960 = int_166959
    kwargs_166961 = {'copy': keyword_166960}
    # Getting the type of 'np' (line 1359)
    np_166953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1359)
    array_166954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 14), np_166953, 'array')
    # Calling array(args, kwargs) (line 1359)
    array_call_result_166962 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 14), array_166954, *[tuple_166955], **kwargs_166961)
    
    float_166963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 44), 'float')
    # Applying the binary operator '+' (line 1359)
    result_add_166964 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 14), '+', array_call_result_166962, float_166963)
    
    # Obtaining the member '__getitem__' of a type (line 1359)
    getitem___166965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 4), result_add_166964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1359)
    subscript_call_result_166966 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 4), getitem___166965, int_166952)
    
    # Assigning a type to the variable 'tuple_var_assignment_165049' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165049', subscript_call_result_166966)
    
    # Assigning a Subscript to a Name (line 1359):
    
    # Obtaining the type of the subscript
    int_166967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 4), 'int')
    
    # Call to array(...): (line 1359)
    # Processing the call arguments (line 1359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1359)
    tuple_166970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1359)
    # Adding element type (line 1359)
    # Getting the type of 'x' (line 1359)
    x_166971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166970, x_166971)
    # Adding element type (line 1359)
    # Getting the type of 'y' (line 1359)
    y_166972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166970, y_166972)
    # Adding element type (line 1359)
    # Getting the type of 'z' (line 1359)
    z_166973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1359, 24), tuple_166970, z_166973)
    
    # Processing the call keyword arguments (line 1359)
    int_166974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 39), 'int')
    keyword_166975 = int_166974
    kwargs_166976 = {'copy': keyword_166975}
    # Getting the type of 'np' (line 1359)
    np_166968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1359)
    array_166969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 14), np_166968, 'array')
    # Calling array(args, kwargs) (line 1359)
    array_call_result_166977 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 14), array_166969, *[tuple_166970], **kwargs_166976)
    
    float_166978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1359, 44), 'float')
    # Applying the binary operator '+' (line 1359)
    result_add_166979 = python_operator(stypy.reporting.localization.Localization(__file__, 1359, 14), '+', array_call_result_166977, float_166978)
    
    # Obtaining the member '__getitem__' of a type (line 1359)
    getitem___166980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1359, 4), result_add_166979, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1359)
    subscript_call_result_166981 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 4), getitem___166980, int_166967)
    
    # Assigning a type to the variable 'tuple_var_assignment_165050' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165050', subscript_call_result_166981)
    
    # Assigning a Name to a Name (line 1359):
    # Getting the type of 'tuple_var_assignment_165048' (line 1359)
    tuple_var_assignment_165048_166982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165048')
    # Assigning a type to the variable 'x' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'x', tuple_var_assignment_165048_166982)
    
    # Assigning a Name to a Name (line 1359):
    # Getting the type of 'tuple_var_assignment_165049' (line 1359)
    tuple_var_assignment_165049_166983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165049')
    # Assigning a type to the variable 'y' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 7), 'y', tuple_var_assignment_165049_166983)
    
    # Assigning a Name to a Name (line 1359):
    # Getting the type of 'tuple_var_assignment_165050' (line 1359)
    tuple_var_assignment_165050_166984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'tuple_var_assignment_165050')
    # Assigning a type to the variable 'z' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 10), 'z', tuple_var_assignment_165050_166984)
    
    # Assigning a Call to a Name (line 1361):
    
    # Assigning a Call to a Name (line 1361):
    
    # Call to hermvander(...): (line 1361)
    # Processing the call arguments (line 1361)
    # Getting the type of 'x' (line 1361)
    x_166986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 20), 'x', False)
    # Getting the type of 'degx' (line 1361)
    degx_166987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 23), 'degx', False)
    # Processing the call keyword arguments (line 1361)
    kwargs_166988 = {}
    # Getting the type of 'hermvander' (line 1361)
    hermvander_166985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 9), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1361)
    hermvander_call_result_166989 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 9), hermvander_166985, *[x_166986, degx_166987], **kwargs_166988)
    
    # Assigning a type to the variable 'vx' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), 'vx', hermvander_call_result_166989)
    
    # Assigning a Call to a Name (line 1362):
    
    # Assigning a Call to a Name (line 1362):
    
    # Call to hermvander(...): (line 1362)
    # Processing the call arguments (line 1362)
    # Getting the type of 'y' (line 1362)
    y_166991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 20), 'y', False)
    # Getting the type of 'degy' (line 1362)
    degy_166992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 23), 'degy', False)
    # Processing the call keyword arguments (line 1362)
    kwargs_166993 = {}
    # Getting the type of 'hermvander' (line 1362)
    hermvander_166990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 9), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1362)
    hermvander_call_result_166994 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 9), hermvander_166990, *[y_166991, degy_166992], **kwargs_166993)
    
    # Assigning a type to the variable 'vy' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'vy', hermvander_call_result_166994)
    
    # Assigning a Call to a Name (line 1363):
    
    # Assigning a Call to a Name (line 1363):
    
    # Call to hermvander(...): (line 1363)
    # Processing the call arguments (line 1363)
    # Getting the type of 'z' (line 1363)
    z_166996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 20), 'z', False)
    # Getting the type of 'degz' (line 1363)
    degz_166997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 23), 'degz', False)
    # Processing the call keyword arguments (line 1363)
    kwargs_166998 = {}
    # Getting the type of 'hermvander' (line 1363)
    hermvander_166995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 9), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1363)
    hermvander_call_result_166999 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 9), hermvander_166995, *[z_166996, degz_166997], **kwargs_166998)
    
    # Assigning a type to the variable 'vz' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'vz', hermvander_call_result_166999)
    
    # Assigning a BinOp to a Name (line 1364):
    
    # Assigning a BinOp to a Name (line 1364):
    
    # Obtaining the type of the subscript
    Ellipsis_167000 = Ellipsis
    # Getting the type of 'None' (line 1364)
    None_167001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 16), 'None')
    # Getting the type of 'None' (line 1364)
    None_167002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 22), 'None')
    # Getting the type of 'vx' (line 1364)
    vx_167003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1364)
    getitem___167004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 8), vx_167003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
    subscript_call_result_167005 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 8), getitem___167004, (Ellipsis_167000, None_167001, None_167002))
    
    
    # Obtaining the type of the subscript
    Ellipsis_167006 = Ellipsis
    # Getting the type of 'None' (line 1364)
    None_167007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 36), 'None')
    slice_167008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1364, 28), None, None, None)
    # Getting the type of 'None' (line 1364)
    None_167009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 44), 'None')
    # Getting the type of 'vy' (line 1364)
    vy_167010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1364)
    getitem___167011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 28), vy_167010, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
    subscript_call_result_167012 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 28), getitem___167011, (Ellipsis_167006, None_167007, slice_167008, None_167009))
    
    # Applying the binary operator '*' (line 1364)
    result_mul_167013 = python_operator(stypy.reporting.localization.Localization(__file__, 1364, 8), '*', subscript_call_result_167005, subscript_call_result_167012)
    
    
    # Obtaining the type of the subscript
    Ellipsis_167014 = Ellipsis
    # Getting the type of 'None' (line 1364)
    None_167015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 58), 'None')
    # Getting the type of 'None' (line 1364)
    None_167016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 64), 'None')
    slice_167017 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1364, 50), None, None, None)
    # Getting the type of 'vz' (line 1364)
    vz_167018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1364)
    getitem___167019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 50), vz_167018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1364)
    subscript_call_result_167020 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 50), getitem___167019, (Ellipsis_167014, None_167015, None_167016, slice_167017))
    
    # Applying the binary operator '*' (line 1364)
    result_mul_167021 = python_operator(stypy.reporting.localization.Localization(__file__, 1364, 49), '*', result_mul_167013, subscript_call_result_167020)
    
    # Assigning a type to the variable 'v' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 4), 'v', result_mul_167021)
    
    # Call to reshape(...): (line 1365)
    # Processing the call arguments (line 1365)
    
    # Obtaining the type of the subscript
    int_167024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 30), 'int')
    slice_167025 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1365, 21), None, int_167024, None)
    # Getting the type of 'v' (line 1365)
    v_167026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1365)
    shape_167027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1365, 21), v_167026, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1365)
    getitem___167028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1365, 21), shape_167027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1365)
    subscript_call_result_167029 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 21), getitem___167028, slice_167025)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1365)
    tuple_167030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1365)
    # Adding element type (line 1365)
    int_167031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1365, 37), tuple_167030, int_167031)
    
    # Applying the binary operator '+' (line 1365)
    result_add_167032 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 21), '+', subscript_call_result_167029, tuple_167030)
    
    # Processing the call keyword arguments (line 1365)
    kwargs_167033 = {}
    # Getting the type of 'v' (line 1365)
    v_167022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1365)
    reshape_167023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1365, 11), v_167022, 'reshape')
    # Calling reshape(args, kwargs) (line 1365)
    reshape_call_result_167034 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 11), reshape_167023, *[result_add_167032], **kwargs_167033)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1365, 4), 'stypy_return_type', reshape_call_result_167034)
    
    # ################# End of 'hermvander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermvander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1303)
    stypy_return_type_167035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167035)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermvander3d'
    return stypy_return_type_167035

# Assigning a type to the variable 'hermvander3d' (line 1303)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 0), 'hermvander3d', hermvander3d)

@norecursion
def hermfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1368)
    None_167036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 29), 'None')
    # Getting the type of 'False' (line 1368)
    False_167037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 40), 'False')
    # Getting the type of 'None' (line 1368)
    None_167038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 49), 'None')
    defaults = [None_167036, False_167037, None_167038]
    # Create a new context for function 'hermfit'
    module_type_store = module_type_store.open_function_context('hermfit', 1368, 0, False)
    
    # Passed parameters checking function
    hermfit.stypy_localization = localization
    hermfit.stypy_type_of_self = None
    hermfit.stypy_type_store = module_type_store
    hermfit.stypy_function_name = 'hermfit'
    hermfit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    hermfit.stypy_varargs_param_name = None
    hermfit.stypy_kwargs_param_name = None
    hermfit.stypy_call_defaults = defaults
    hermfit.stypy_call_varargs = varargs
    hermfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermfit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermfit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermfit(...)' code ##################

    str_167039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1491, (-1)), 'str', '\n    Least squares fit of Hermite series to data.\n\n    Return the coefficients of a Hermite series of degree `deg` that is the\n    least squares fit to the data values `y` given at points `x`. If `y` is\n    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple\n    fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * H_1(x) + ... + c_n * H_n(x),\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Hermite coefficients ordered from low to high. If `y` was 2-D,\n        the coefficients for the data in column k  of `y` are in column\n        `k`.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    chebfit, legfit, lagfit, polyfit, hermefit\n    hermval : Evaluates a Hermite series.\n    hermvander : Vandermonde matrix of Hermite series.\n    hermweight : Hermite weight function\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the Hermite series `p` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where the :math:`w_j` are the weights. This problem is solved by\n    setting up the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the\n    coefficients to be solved for, `w` are the weights, `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using Hermite series are probably most useful when the data can be\n    approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the Hermite\n    weight. In that case the weight ``sqrt(w(x[i])`` should be used\n    together with data values ``y[i]/sqrt(w(x[i])``. The weight function is\n    available as `hermweight`.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermfit, hermval\n    >>> x = np.linspace(-10, 10)\n    >>> err = np.random.randn(len(x))/10\n    >>> y = hermval(x, [1, 2, 3]) + err\n    >>> hermfit(x, y, 2)\n    array([ 0.97902637,  1.99849131,  3.00006   ])\n\n    ')
    
    # Assigning a BinOp to a Name (line 1492):
    
    # Assigning a BinOp to a Name (line 1492):
    
    # Call to asarray(...): (line 1492)
    # Processing the call arguments (line 1492)
    # Getting the type of 'x' (line 1492)
    x_167042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 19), 'x', False)
    # Processing the call keyword arguments (line 1492)
    kwargs_167043 = {}
    # Getting the type of 'np' (line 1492)
    np_167040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1492)
    asarray_167041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1492, 8), np_167040, 'asarray')
    # Calling asarray(args, kwargs) (line 1492)
    asarray_call_result_167044 = invoke(stypy.reporting.localization.Localization(__file__, 1492, 8), asarray_167041, *[x_167042], **kwargs_167043)
    
    float_167045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1492, 24), 'float')
    # Applying the binary operator '+' (line 1492)
    result_add_167046 = python_operator(stypy.reporting.localization.Localization(__file__, 1492, 8), '+', asarray_call_result_167044, float_167045)
    
    # Assigning a type to the variable 'x' (line 1492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1492, 4), 'x', result_add_167046)
    
    # Assigning a BinOp to a Name (line 1493):
    
    # Assigning a BinOp to a Name (line 1493):
    
    # Call to asarray(...): (line 1493)
    # Processing the call arguments (line 1493)
    # Getting the type of 'y' (line 1493)
    y_167049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 19), 'y', False)
    # Processing the call keyword arguments (line 1493)
    kwargs_167050 = {}
    # Getting the type of 'np' (line 1493)
    np_167047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1493)
    asarray_167048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1493, 8), np_167047, 'asarray')
    # Calling asarray(args, kwargs) (line 1493)
    asarray_call_result_167051 = invoke(stypy.reporting.localization.Localization(__file__, 1493, 8), asarray_167048, *[y_167049], **kwargs_167050)
    
    float_167052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 24), 'float')
    # Applying the binary operator '+' (line 1493)
    result_add_167053 = python_operator(stypy.reporting.localization.Localization(__file__, 1493, 8), '+', asarray_call_result_167051, float_167052)
    
    # Assigning a type to the variable 'y' (line 1493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1493, 4), 'y', result_add_167053)
    
    # Assigning a Call to a Name (line 1494):
    
    # Assigning a Call to a Name (line 1494):
    
    # Call to asarray(...): (line 1494)
    # Processing the call arguments (line 1494)
    # Getting the type of 'deg' (line 1494)
    deg_167056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 21), 'deg', False)
    # Processing the call keyword arguments (line 1494)
    kwargs_167057 = {}
    # Getting the type of 'np' (line 1494)
    np_167054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1494)
    asarray_167055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 10), np_167054, 'asarray')
    # Calling asarray(args, kwargs) (line 1494)
    asarray_call_result_167058 = invoke(stypy.reporting.localization.Localization(__file__, 1494, 10), asarray_167055, *[deg_167056], **kwargs_167057)
    
    # Assigning a type to the variable 'deg' (line 1494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1494, 4), 'deg', asarray_call_result_167058)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1497)
    deg_167059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1497)
    ndim_167060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1497, 7), deg_167059, 'ndim')
    int_167061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 18), 'int')
    # Applying the binary operator '>' (line 1497)
    result_gt_167062 = python_operator(stypy.reporting.localization.Localization(__file__, 1497, 7), '>', ndim_167060, int_167061)
    
    
    # Getting the type of 'deg' (line 1497)
    deg_167063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1497)
    dtype_167064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1497, 23), deg_167063, 'dtype')
    # Obtaining the member 'kind' of a type (line 1497)
    kind_167065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1497, 23), dtype_167064, 'kind')
    str_167066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1497)
    result_contains_167067 = python_operator(stypy.reporting.localization.Localization(__file__, 1497, 23), 'notin', kind_167065, str_167066)
    
    # Applying the binary operator 'or' (line 1497)
    result_or_keyword_167068 = python_operator(stypy.reporting.localization.Localization(__file__, 1497, 7), 'or', result_gt_167062, result_contains_167067)
    
    # Getting the type of 'deg' (line 1497)
    deg_167069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1497)
    size_167070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1497, 53), deg_167069, 'size')
    int_167071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 65), 'int')
    # Applying the binary operator '==' (line 1497)
    result_eq_167072 = python_operator(stypy.reporting.localization.Localization(__file__, 1497, 53), '==', size_167070, int_167071)
    
    # Applying the binary operator 'or' (line 1497)
    result_or_keyword_167073 = python_operator(stypy.reporting.localization.Localization(__file__, 1497, 7), 'or', result_or_keyword_167068, result_eq_167072)
    
    # Testing the type of an if condition (line 1497)
    if_condition_167074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1497, 4), result_or_keyword_167073)
    # Assigning a type to the variable 'if_condition_167074' (line 1497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1497, 4), 'if_condition_167074', if_condition_167074)
    # SSA begins for if statement (line 1497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1498)
    # Processing the call arguments (line 1498)
    str_167076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1498, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1498)
    kwargs_167077 = {}
    # Getting the type of 'TypeError' (line 1498)
    TypeError_167075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1498)
    TypeError_call_result_167078 = invoke(stypy.reporting.localization.Localization(__file__, 1498, 14), TypeError_167075, *[str_167076], **kwargs_167077)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1498, 8), TypeError_call_result_167078, 'raise parameter', BaseException)
    # SSA join for if statement (line 1497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1499)
    # Processing the call keyword arguments (line 1499)
    kwargs_167081 = {}
    # Getting the type of 'deg' (line 1499)
    deg_167079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1499)
    min_167080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1499, 7), deg_167079, 'min')
    # Calling min(args, kwargs) (line 1499)
    min_call_result_167082 = invoke(stypy.reporting.localization.Localization(__file__, 1499, 7), min_167080, *[], **kwargs_167081)
    
    int_167083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, 19), 'int')
    # Applying the binary operator '<' (line 1499)
    result_lt_167084 = python_operator(stypy.reporting.localization.Localization(__file__, 1499, 7), '<', min_call_result_167082, int_167083)
    
    # Testing the type of an if condition (line 1499)
    if_condition_167085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1499, 4), result_lt_167084)
    # Assigning a type to the variable 'if_condition_167085' (line 1499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1499, 4), 'if_condition_167085', if_condition_167085)
    # SSA begins for if statement (line 1499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1500)
    # Processing the call arguments (line 1500)
    str_167087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1500, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1500)
    kwargs_167088 = {}
    # Getting the type of 'ValueError' (line 1500)
    ValueError_167086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1500, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1500)
    ValueError_call_result_167089 = invoke(stypy.reporting.localization.Localization(__file__, 1500, 14), ValueError_167086, *[str_167087], **kwargs_167088)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1500, 8), ValueError_call_result_167089, 'raise parameter', BaseException)
    # SSA join for if statement (line 1499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1501)
    x_167090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1501)
    ndim_167091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1501, 7), x_167090, 'ndim')
    int_167092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1501, 17), 'int')
    # Applying the binary operator '!=' (line 1501)
    result_ne_167093 = python_operator(stypy.reporting.localization.Localization(__file__, 1501, 7), '!=', ndim_167091, int_167092)
    
    # Testing the type of an if condition (line 1501)
    if_condition_167094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1501, 4), result_ne_167093)
    # Assigning a type to the variable 'if_condition_167094' (line 1501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1501, 4), 'if_condition_167094', if_condition_167094)
    # SSA begins for if statement (line 1501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1502)
    # Processing the call arguments (line 1502)
    str_167096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1502)
    kwargs_167097 = {}
    # Getting the type of 'TypeError' (line 1502)
    TypeError_167095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1502)
    TypeError_call_result_167098 = invoke(stypy.reporting.localization.Localization(__file__, 1502, 14), TypeError_167095, *[str_167096], **kwargs_167097)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1502, 8), TypeError_call_result_167098, 'raise parameter', BaseException)
    # SSA join for if statement (line 1501)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1503)
    x_167099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 7), 'x')
    # Obtaining the member 'size' of a type (line 1503)
    size_167100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1503, 7), x_167099, 'size')
    int_167101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1503, 17), 'int')
    # Applying the binary operator '==' (line 1503)
    result_eq_167102 = python_operator(stypy.reporting.localization.Localization(__file__, 1503, 7), '==', size_167100, int_167101)
    
    # Testing the type of an if condition (line 1503)
    if_condition_167103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1503, 4), result_eq_167102)
    # Assigning a type to the variable 'if_condition_167103' (line 1503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1503, 4), 'if_condition_167103', if_condition_167103)
    # SSA begins for if statement (line 1503)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1504)
    # Processing the call arguments (line 1504)
    str_167105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1504, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1504)
    kwargs_167106 = {}
    # Getting the type of 'TypeError' (line 1504)
    TypeError_167104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1504)
    TypeError_call_result_167107 = invoke(stypy.reporting.localization.Localization(__file__, 1504, 14), TypeError_167104, *[str_167105], **kwargs_167106)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1504, 8), TypeError_call_result_167107, 'raise parameter', BaseException)
    # SSA join for if statement (line 1503)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1505)
    y_167108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1505)
    ndim_167109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1505, 7), y_167108, 'ndim')
    int_167110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 16), 'int')
    # Applying the binary operator '<' (line 1505)
    result_lt_167111 = python_operator(stypy.reporting.localization.Localization(__file__, 1505, 7), '<', ndim_167109, int_167110)
    
    
    # Getting the type of 'y' (line 1505)
    y_167112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1505)
    ndim_167113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1505, 21), y_167112, 'ndim')
    int_167114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 30), 'int')
    # Applying the binary operator '>' (line 1505)
    result_gt_167115 = python_operator(stypy.reporting.localization.Localization(__file__, 1505, 21), '>', ndim_167113, int_167114)
    
    # Applying the binary operator 'or' (line 1505)
    result_or_keyword_167116 = python_operator(stypy.reporting.localization.Localization(__file__, 1505, 7), 'or', result_lt_167111, result_gt_167115)
    
    # Testing the type of an if condition (line 1505)
    if_condition_167117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1505, 4), result_or_keyword_167116)
    # Assigning a type to the variable 'if_condition_167117' (line 1505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1505, 4), 'if_condition_167117', if_condition_167117)
    # SSA begins for if statement (line 1505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1506)
    # Processing the call arguments (line 1506)
    str_167119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1506, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1506)
    kwargs_167120 = {}
    # Getting the type of 'TypeError' (line 1506)
    TypeError_167118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1506, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1506)
    TypeError_call_result_167121 = invoke(stypy.reporting.localization.Localization(__file__, 1506, 14), TypeError_167118, *[str_167119], **kwargs_167120)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1506, 8), TypeError_call_result_167121, 'raise parameter', BaseException)
    # SSA join for if statement (line 1505)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1507)
    # Processing the call arguments (line 1507)
    # Getting the type of 'x' (line 1507)
    x_167123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 11), 'x', False)
    # Processing the call keyword arguments (line 1507)
    kwargs_167124 = {}
    # Getting the type of 'len' (line 1507)
    len_167122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 7), 'len', False)
    # Calling len(args, kwargs) (line 1507)
    len_call_result_167125 = invoke(stypy.reporting.localization.Localization(__file__, 1507, 7), len_167122, *[x_167123], **kwargs_167124)
    
    
    # Call to len(...): (line 1507)
    # Processing the call arguments (line 1507)
    # Getting the type of 'y' (line 1507)
    y_167127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 21), 'y', False)
    # Processing the call keyword arguments (line 1507)
    kwargs_167128 = {}
    # Getting the type of 'len' (line 1507)
    len_167126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 17), 'len', False)
    # Calling len(args, kwargs) (line 1507)
    len_call_result_167129 = invoke(stypy.reporting.localization.Localization(__file__, 1507, 17), len_167126, *[y_167127], **kwargs_167128)
    
    # Applying the binary operator '!=' (line 1507)
    result_ne_167130 = python_operator(stypy.reporting.localization.Localization(__file__, 1507, 7), '!=', len_call_result_167125, len_call_result_167129)
    
    # Testing the type of an if condition (line 1507)
    if_condition_167131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1507, 4), result_ne_167130)
    # Assigning a type to the variable 'if_condition_167131' (line 1507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1507, 4), 'if_condition_167131', if_condition_167131)
    # SSA begins for if statement (line 1507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1508)
    # Processing the call arguments (line 1508)
    str_167133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1508, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1508)
    kwargs_167134 = {}
    # Getting the type of 'TypeError' (line 1508)
    TypeError_167132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1508)
    TypeError_call_result_167135 = invoke(stypy.reporting.localization.Localization(__file__, 1508, 14), TypeError_167132, *[str_167133], **kwargs_167134)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1508, 8), TypeError_call_result_167135, 'raise parameter', BaseException)
    # SSA join for if statement (line 1507)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1510)
    deg_167136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1510)
    ndim_167137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1510, 7), deg_167136, 'ndim')
    int_167138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1510, 19), 'int')
    # Applying the binary operator '==' (line 1510)
    result_eq_167139 = python_operator(stypy.reporting.localization.Localization(__file__, 1510, 7), '==', ndim_167137, int_167138)
    
    # Testing the type of an if condition (line 1510)
    if_condition_167140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1510, 4), result_eq_167139)
    # Assigning a type to the variable 'if_condition_167140' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 4), 'if_condition_167140', if_condition_167140)
    # SSA begins for if statement (line 1510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1511):
    
    # Assigning a Name to a Name (line 1511):
    # Getting the type of 'deg' (line 1511)
    deg_167141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1511, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1511, 8), 'lmax', deg_167141)
    
    # Assigning a BinOp to a Name (line 1512):
    
    # Assigning a BinOp to a Name (line 1512):
    # Getting the type of 'lmax' (line 1512)
    lmax_167142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 16), 'lmax')
    int_167143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1512, 23), 'int')
    # Applying the binary operator '+' (line 1512)
    result_add_167144 = python_operator(stypy.reporting.localization.Localization(__file__, 1512, 16), '+', lmax_167142, int_167143)
    
    # Assigning a type to the variable 'order' (line 1512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1512, 8), 'order', result_add_167144)
    
    # Assigning a Call to a Name (line 1513):
    
    # Assigning a Call to a Name (line 1513):
    
    # Call to hermvander(...): (line 1513)
    # Processing the call arguments (line 1513)
    # Getting the type of 'x' (line 1513)
    x_167146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 25), 'x', False)
    # Getting the type of 'lmax' (line 1513)
    lmax_167147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1513)
    kwargs_167148 = {}
    # Getting the type of 'hermvander' (line 1513)
    hermvander_167145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 14), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1513)
    hermvander_call_result_167149 = invoke(stypy.reporting.localization.Localization(__file__, 1513, 14), hermvander_167145, *[x_167146, lmax_167147], **kwargs_167148)
    
    # Assigning a type to the variable 'van' (line 1513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1513, 8), 'van', hermvander_call_result_167149)
    # SSA branch for the else part of an if statement (line 1510)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1515):
    
    # Assigning a Call to a Name (line 1515):
    
    # Call to sort(...): (line 1515)
    # Processing the call arguments (line 1515)
    # Getting the type of 'deg' (line 1515)
    deg_167152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 22), 'deg', False)
    # Processing the call keyword arguments (line 1515)
    kwargs_167153 = {}
    # Getting the type of 'np' (line 1515)
    np_167150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1515)
    sort_167151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1515, 14), np_167150, 'sort')
    # Calling sort(args, kwargs) (line 1515)
    sort_call_result_167154 = invoke(stypy.reporting.localization.Localization(__file__, 1515, 14), sort_167151, *[deg_167152], **kwargs_167153)
    
    # Assigning a type to the variable 'deg' (line 1515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1515, 8), 'deg', sort_call_result_167154)
    
    # Assigning a Subscript to a Name (line 1516):
    
    # Assigning a Subscript to a Name (line 1516):
    
    # Obtaining the type of the subscript
    int_167155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1516, 19), 'int')
    # Getting the type of 'deg' (line 1516)
    deg_167156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1516, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1516)
    getitem___167157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1516, 15), deg_167156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1516)
    subscript_call_result_167158 = invoke(stypy.reporting.localization.Localization(__file__, 1516, 15), getitem___167157, int_167155)
    
    # Assigning a type to the variable 'lmax' (line 1516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1516, 8), 'lmax', subscript_call_result_167158)
    
    # Assigning a Call to a Name (line 1517):
    
    # Assigning a Call to a Name (line 1517):
    
    # Call to len(...): (line 1517)
    # Processing the call arguments (line 1517)
    # Getting the type of 'deg' (line 1517)
    deg_167160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 20), 'deg', False)
    # Processing the call keyword arguments (line 1517)
    kwargs_167161 = {}
    # Getting the type of 'len' (line 1517)
    len_167159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1517, 16), 'len', False)
    # Calling len(args, kwargs) (line 1517)
    len_call_result_167162 = invoke(stypy.reporting.localization.Localization(__file__, 1517, 16), len_167159, *[deg_167160], **kwargs_167161)
    
    # Assigning a type to the variable 'order' (line 1517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1517, 8), 'order', len_call_result_167162)
    
    # Assigning a Subscript to a Name (line 1518):
    
    # Assigning a Subscript to a Name (line 1518):
    
    # Obtaining the type of the subscript
    slice_167163 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1518, 14), None, None, None)
    # Getting the type of 'deg' (line 1518)
    deg_167164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 37), 'deg')
    
    # Call to hermvander(...): (line 1518)
    # Processing the call arguments (line 1518)
    # Getting the type of 'x' (line 1518)
    x_167166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 25), 'x', False)
    # Getting the type of 'lmax' (line 1518)
    lmax_167167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 28), 'lmax', False)
    # Processing the call keyword arguments (line 1518)
    kwargs_167168 = {}
    # Getting the type of 'hermvander' (line 1518)
    hermvander_167165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 14), 'hermvander', False)
    # Calling hermvander(args, kwargs) (line 1518)
    hermvander_call_result_167169 = invoke(stypy.reporting.localization.Localization(__file__, 1518, 14), hermvander_167165, *[x_167166, lmax_167167], **kwargs_167168)
    
    # Obtaining the member '__getitem__' of a type (line 1518)
    getitem___167170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1518, 14), hermvander_call_result_167169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1518)
    subscript_call_result_167171 = invoke(stypy.reporting.localization.Localization(__file__, 1518, 14), getitem___167170, (slice_167163, deg_167164))
    
    # Assigning a type to the variable 'van' (line 1518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1518, 8), 'van', subscript_call_result_167171)
    # SSA join for if statement (line 1510)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1521):
    
    # Assigning a Attribute to a Name (line 1521):
    # Getting the type of 'van' (line 1521)
    van_167172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 10), 'van')
    # Obtaining the member 'T' of a type (line 1521)
    T_167173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1521, 10), van_167172, 'T')
    # Assigning a type to the variable 'lhs' (line 1521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 4), 'lhs', T_167173)
    
    # Assigning a Attribute to a Name (line 1522):
    
    # Assigning a Attribute to a Name (line 1522):
    # Getting the type of 'y' (line 1522)
    y_167174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 10), 'y')
    # Obtaining the member 'T' of a type (line 1522)
    T_167175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 10), y_167174, 'T')
    # Assigning a type to the variable 'rhs' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'rhs', T_167175)
    
    # Type idiom detected: calculating its left and rigth part (line 1523)
    # Getting the type of 'w' (line 1523)
    w_167176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'w')
    # Getting the type of 'None' (line 1523)
    None_167177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 16), 'None')
    
    (may_be_167178, more_types_in_union_167179) = may_not_be_none(w_167176, None_167177)

    if may_be_167178:

        if more_types_in_union_167179:
            # Runtime conditional SSA (line 1523)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1524):
        
        # Assigning a BinOp to a Name (line 1524):
        
        # Call to asarray(...): (line 1524)
        # Processing the call arguments (line 1524)
        # Getting the type of 'w' (line 1524)
        w_167182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 23), 'w', False)
        # Processing the call keyword arguments (line 1524)
        kwargs_167183 = {}
        # Getting the type of 'np' (line 1524)
        np_167180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1524)
        asarray_167181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 12), np_167180, 'asarray')
        # Calling asarray(args, kwargs) (line 1524)
        asarray_call_result_167184 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 12), asarray_167181, *[w_167182], **kwargs_167183)
        
        float_167185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 28), 'float')
        # Applying the binary operator '+' (line 1524)
        result_add_167186 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 12), '+', asarray_call_result_167184, float_167185)
        
        # Assigning a type to the variable 'w' (line 1524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 8), 'w', result_add_167186)
        
        
        # Getting the type of 'w' (line 1525)
        w_167187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1525)
        ndim_167188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1525, 11), w_167187, 'ndim')
        int_167189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1525, 21), 'int')
        # Applying the binary operator '!=' (line 1525)
        result_ne_167190 = python_operator(stypy.reporting.localization.Localization(__file__, 1525, 11), '!=', ndim_167188, int_167189)
        
        # Testing the type of an if condition (line 1525)
        if_condition_167191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1525, 8), result_ne_167190)
        # Assigning a type to the variable 'if_condition_167191' (line 1525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1525, 8), 'if_condition_167191', if_condition_167191)
        # SSA begins for if statement (line 1525)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1526)
        # Processing the call arguments (line 1526)
        str_167193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1526, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1526)
        kwargs_167194 = {}
        # Getting the type of 'TypeError' (line 1526)
        TypeError_167192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1526)
        TypeError_call_result_167195 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 18), TypeError_167192, *[str_167193], **kwargs_167194)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1526, 12), TypeError_call_result_167195, 'raise parameter', BaseException)
        # SSA join for if statement (line 1525)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1527)
        # Processing the call arguments (line 1527)
        # Getting the type of 'x' (line 1527)
        x_167197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 15), 'x', False)
        # Processing the call keyword arguments (line 1527)
        kwargs_167198 = {}
        # Getting the type of 'len' (line 1527)
        len_167196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 11), 'len', False)
        # Calling len(args, kwargs) (line 1527)
        len_call_result_167199 = invoke(stypy.reporting.localization.Localization(__file__, 1527, 11), len_167196, *[x_167197], **kwargs_167198)
        
        
        # Call to len(...): (line 1527)
        # Processing the call arguments (line 1527)
        # Getting the type of 'w' (line 1527)
        w_167201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 25), 'w', False)
        # Processing the call keyword arguments (line 1527)
        kwargs_167202 = {}
        # Getting the type of 'len' (line 1527)
        len_167200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1527, 21), 'len', False)
        # Calling len(args, kwargs) (line 1527)
        len_call_result_167203 = invoke(stypy.reporting.localization.Localization(__file__, 1527, 21), len_167200, *[w_167201], **kwargs_167202)
        
        # Applying the binary operator '!=' (line 1527)
        result_ne_167204 = python_operator(stypy.reporting.localization.Localization(__file__, 1527, 11), '!=', len_call_result_167199, len_call_result_167203)
        
        # Testing the type of an if condition (line 1527)
        if_condition_167205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1527, 8), result_ne_167204)
        # Assigning a type to the variable 'if_condition_167205' (line 1527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1527, 8), 'if_condition_167205', if_condition_167205)
        # SSA begins for if statement (line 1527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1528)
        # Processing the call arguments (line 1528)
        str_167207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1528, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1528)
        kwargs_167208 = {}
        # Getting the type of 'TypeError' (line 1528)
        TypeError_167206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1528)
        TypeError_call_result_167209 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 18), TypeError_167206, *[str_167207], **kwargs_167208)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1528, 12), TypeError_call_result_167209, 'raise parameter', BaseException)
        # SSA join for if statement (line 1527)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1531):
        
        # Assigning a BinOp to a Name (line 1531):
        # Getting the type of 'lhs' (line 1531)
        lhs_167210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 14), 'lhs')
        # Getting the type of 'w' (line 1531)
        w_167211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1531, 20), 'w')
        # Applying the binary operator '*' (line 1531)
        result_mul_167212 = python_operator(stypy.reporting.localization.Localization(__file__, 1531, 14), '*', lhs_167210, w_167211)
        
        # Assigning a type to the variable 'lhs' (line 1531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1531, 8), 'lhs', result_mul_167212)
        
        # Assigning a BinOp to a Name (line 1532):
        
        # Assigning a BinOp to a Name (line 1532):
        # Getting the type of 'rhs' (line 1532)
        rhs_167213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 14), 'rhs')
        # Getting the type of 'w' (line 1532)
        w_167214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 20), 'w')
        # Applying the binary operator '*' (line 1532)
        result_mul_167215 = python_operator(stypy.reporting.localization.Localization(__file__, 1532, 14), '*', rhs_167213, w_167214)
        
        # Assigning a type to the variable 'rhs' (line 1532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 8), 'rhs', result_mul_167215)

        if more_types_in_union_167179:
            # SSA join for if statement (line 1523)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1535)
    # Getting the type of 'rcond' (line 1535)
    rcond_167216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 7), 'rcond')
    # Getting the type of 'None' (line 1535)
    None_167217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1535, 16), 'None')
    
    (may_be_167218, more_types_in_union_167219) = may_be_none(rcond_167216, None_167217)

    if may_be_167218:

        if more_types_in_union_167219:
            # Runtime conditional SSA (line 1535)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1536):
        
        # Assigning a BinOp to a Name (line 1536):
        
        # Call to len(...): (line 1536)
        # Processing the call arguments (line 1536)
        # Getting the type of 'x' (line 1536)
        x_167221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 20), 'x', False)
        # Processing the call keyword arguments (line 1536)
        kwargs_167222 = {}
        # Getting the type of 'len' (line 1536)
        len_167220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 16), 'len', False)
        # Calling len(args, kwargs) (line 1536)
        len_call_result_167223 = invoke(stypy.reporting.localization.Localization(__file__, 1536, 16), len_167220, *[x_167221], **kwargs_167222)
        
        
        # Call to finfo(...): (line 1536)
        # Processing the call arguments (line 1536)
        # Getting the type of 'x' (line 1536)
        x_167226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1536)
        dtype_167227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 32), x_167226, 'dtype')
        # Processing the call keyword arguments (line 1536)
        kwargs_167228 = {}
        # Getting the type of 'np' (line 1536)
        np_167224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1536)
        finfo_167225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 23), np_167224, 'finfo')
        # Calling finfo(args, kwargs) (line 1536)
        finfo_call_result_167229 = invoke(stypy.reporting.localization.Localization(__file__, 1536, 23), finfo_167225, *[dtype_167227], **kwargs_167228)
        
        # Obtaining the member 'eps' of a type (line 1536)
        eps_167230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 23), finfo_call_result_167229, 'eps')
        # Applying the binary operator '*' (line 1536)
        result_mul_167231 = python_operator(stypy.reporting.localization.Localization(__file__, 1536, 16), '*', len_call_result_167223, eps_167230)
        
        # Assigning a type to the variable 'rcond' (line 1536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1536, 8), 'rcond', result_mul_167231)

        if more_types_in_union_167219:
            # SSA join for if statement (line 1535)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'lhs' (line 1539)
    lhs_167233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1539)
    dtype_167234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 18), lhs_167233, 'dtype')
    # Obtaining the member 'type' of a type (line 1539)
    type_167235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 18), dtype_167234, 'type')
    # Getting the type of 'np' (line 1539)
    np_167236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1539)
    complexfloating_167237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 34), np_167236, 'complexfloating')
    # Processing the call keyword arguments (line 1539)
    kwargs_167238 = {}
    # Getting the type of 'issubclass' (line 1539)
    issubclass_167232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1539)
    issubclass_call_result_167239 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 7), issubclass_167232, *[type_167235, complexfloating_167237], **kwargs_167238)
    
    # Testing the type of an if condition (line 1539)
    if_condition_167240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1539, 4), issubclass_call_result_167239)
    # Assigning a type to the variable 'if_condition_167240' (line 1539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1539, 4), 'if_condition_167240', if_condition_167240)
    # SSA begins for if statement (line 1539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1540):
    
    # Assigning a Call to a Name (line 1540):
    
    # Call to sqrt(...): (line 1540)
    # Processing the call arguments (line 1540)
    
    # Call to sum(...): (line 1540)
    # Processing the call arguments (line 1540)
    int_167257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1540, 70), 'int')
    # Processing the call keyword arguments (line 1540)
    kwargs_167258 = {}
    
    # Call to square(...): (line 1540)
    # Processing the call arguments (line 1540)
    # Getting the type of 'lhs' (line 1540)
    lhs_167245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1540)
    real_167246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 33), lhs_167245, 'real')
    # Processing the call keyword arguments (line 1540)
    kwargs_167247 = {}
    # Getting the type of 'np' (line 1540)
    np_167243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1540)
    square_167244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 23), np_167243, 'square')
    # Calling square(args, kwargs) (line 1540)
    square_call_result_167248 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 23), square_167244, *[real_167246], **kwargs_167247)
    
    
    # Call to square(...): (line 1540)
    # Processing the call arguments (line 1540)
    # Getting the type of 'lhs' (line 1540)
    lhs_167251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1540)
    imag_167252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 55), lhs_167251, 'imag')
    # Processing the call keyword arguments (line 1540)
    kwargs_167253 = {}
    # Getting the type of 'np' (line 1540)
    np_167249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1540)
    square_167250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 45), np_167249, 'square')
    # Calling square(args, kwargs) (line 1540)
    square_call_result_167254 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 45), square_167250, *[imag_167252], **kwargs_167253)
    
    # Applying the binary operator '+' (line 1540)
    result_add_167255 = python_operator(stypy.reporting.localization.Localization(__file__, 1540, 23), '+', square_call_result_167248, square_call_result_167254)
    
    # Obtaining the member 'sum' of a type (line 1540)
    sum_167256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 23), result_add_167255, 'sum')
    # Calling sum(args, kwargs) (line 1540)
    sum_call_result_167259 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 23), sum_167256, *[int_167257], **kwargs_167258)
    
    # Processing the call keyword arguments (line 1540)
    kwargs_167260 = {}
    # Getting the type of 'np' (line 1540)
    np_167241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1540)
    sqrt_167242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1540, 14), np_167241, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1540)
    sqrt_call_result_167261 = invoke(stypy.reporting.localization.Localization(__file__, 1540, 14), sqrt_167242, *[sum_call_result_167259], **kwargs_167260)
    
    # Assigning a type to the variable 'scl' (line 1540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1540, 8), 'scl', sqrt_call_result_167261)
    # SSA branch for the else part of an if statement (line 1539)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1542):
    
    # Assigning a Call to a Name (line 1542):
    
    # Call to sqrt(...): (line 1542)
    # Processing the call arguments (line 1542)
    
    # Call to sum(...): (line 1542)
    # Processing the call arguments (line 1542)
    int_167270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1542, 41), 'int')
    # Processing the call keyword arguments (line 1542)
    kwargs_167271 = {}
    
    # Call to square(...): (line 1542)
    # Processing the call arguments (line 1542)
    # Getting the type of 'lhs' (line 1542)
    lhs_167266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1542)
    kwargs_167267 = {}
    # Getting the type of 'np' (line 1542)
    np_167264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1542)
    square_167265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1542, 22), np_167264, 'square')
    # Calling square(args, kwargs) (line 1542)
    square_call_result_167268 = invoke(stypy.reporting.localization.Localization(__file__, 1542, 22), square_167265, *[lhs_167266], **kwargs_167267)
    
    # Obtaining the member 'sum' of a type (line 1542)
    sum_167269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1542, 22), square_call_result_167268, 'sum')
    # Calling sum(args, kwargs) (line 1542)
    sum_call_result_167272 = invoke(stypy.reporting.localization.Localization(__file__, 1542, 22), sum_167269, *[int_167270], **kwargs_167271)
    
    # Processing the call keyword arguments (line 1542)
    kwargs_167273 = {}
    # Getting the type of 'np' (line 1542)
    np_167262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1542, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1542)
    sqrt_167263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1542, 14), np_167262, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1542)
    sqrt_call_result_167274 = invoke(stypy.reporting.localization.Localization(__file__, 1542, 14), sqrt_167263, *[sum_call_result_167272], **kwargs_167273)
    
    # Assigning a type to the variable 'scl' (line 1542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1542, 8), 'scl', sqrt_call_result_167274)
    # SSA join for if statement (line 1539)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1543):
    
    # Assigning a Num to a Subscript (line 1543):
    int_167275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 20), 'int')
    # Getting the type of 'scl' (line 1543)
    scl_167276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'scl')
    
    # Getting the type of 'scl' (line 1543)
    scl_167277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 8), 'scl')
    int_167278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 15), 'int')
    # Applying the binary operator '==' (line 1543)
    result_eq_167279 = python_operator(stypy.reporting.localization.Localization(__file__, 1543, 8), '==', scl_167277, int_167278)
    
    # Storing an element on a container (line 1543)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1543, 4), scl_167276, (result_eq_167279, int_167275))
    
    # Assigning a Call to a Tuple (line 1546):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1546)
    # Processing the call arguments (line 1546)
    # Getting the type of 'lhs' (line 1546)
    lhs_167282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1546)
    T_167283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 34), lhs_167282, 'T')
    # Getting the type of 'scl' (line 1546)
    scl_167284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1546)
    result_div_167285 = python_operator(stypy.reporting.localization.Localization(__file__, 1546, 34), 'div', T_167283, scl_167284)
    
    # Getting the type of 'rhs' (line 1546)
    rhs_167286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1546)
    T_167287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 45), rhs_167286, 'T')
    # Getting the type of 'rcond' (line 1546)
    rcond_167288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1546)
    kwargs_167289 = {}
    # Getting the type of 'la' (line 1546)
    la_167280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1546)
    lstsq_167281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 25), la_167280, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1546)
    lstsq_call_result_167290 = invoke(stypy.reporting.localization.Localization(__file__, 1546, 25), lstsq_167281, *[result_div_167285, T_167287, rcond_167288], **kwargs_167289)
    
    # Assigning a type to the variable 'call_assignment_165051' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165051', lstsq_call_result_167290)
    
    # Assigning a Call to a Name (line 1546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167294 = {}
    # Getting the type of 'call_assignment_165051' (line 1546)
    call_assignment_165051_167291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165051', False)
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___167292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 4), call_assignment_165051_167291, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167295 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167292, *[int_167293], **kwargs_167294)
    
    # Assigning a type to the variable 'call_assignment_165052' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165052', getitem___call_result_167295)
    
    # Assigning a Name to a Name (line 1546):
    # Getting the type of 'call_assignment_165052' (line 1546)
    call_assignment_165052_167296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165052')
    # Assigning a type to the variable 'c' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'c', call_assignment_165052_167296)
    
    # Assigning a Call to a Name (line 1546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167300 = {}
    # Getting the type of 'call_assignment_165051' (line 1546)
    call_assignment_165051_167297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165051', False)
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___167298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 4), call_assignment_165051_167297, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167301 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167298, *[int_167299], **kwargs_167300)
    
    # Assigning a type to the variable 'call_assignment_165053' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165053', getitem___call_result_167301)
    
    # Assigning a Name to a Name (line 1546):
    # Getting the type of 'call_assignment_165053' (line 1546)
    call_assignment_165053_167302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165053')
    # Assigning a type to the variable 'resids' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 7), 'resids', call_assignment_165053_167302)
    
    # Assigning a Call to a Name (line 1546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167306 = {}
    # Getting the type of 'call_assignment_165051' (line 1546)
    call_assignment_165051_167303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165051', False)
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___167304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 4), call_assignment_165051_167303, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167307 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167304, *[int_167305], **kwargs_167306)
    
    # Assigning a type to the variable 'call_assignment_165054' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165054', getitem___call_result_167307)
    
    # Assigning a Name to a Name (line 1546):
    # Getting the type of 'call_assignment_165054' (line 1546)
    call_assignment_165054_167308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165054')
    # Assigning a type to the variable 'rank' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 15), 'rank', call_assignment_165054_167308)
    
    # Assigning a Call to a Name (line 1546):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1546, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167312 = {}
    # Getting the type of 'call_assignment_165051' (line 1546)
    call_assignment_165051_167309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165051', False)
    # Obtaining the member '__getitem__' of a type (line 1546)
    getitem___167310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1546, 4), call_assignment_165051_167309, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167313 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167310, *[int_167311], **kwargs_167312)
    
    # Assigning a type to the variable 'call_assignment_165055' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165055', getitem___call_result_167313)
    
    # Assigning a Name to a Name (line 1546):
    # Getting the type of 'call_assignment_165055' (line 1546)
    call_assignment_165055_167314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1546, 4), 'call_assignment_165055')
    # Assigning a type to the variable 's' (line 1546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1546, 21), 's', call_assignment_165055_167314)
    
    # Assigning a Attribute to a Name (line 1547):
    
    # Assigning a Attribute to a Name (line 1547):
    # Getting the type of 'c' (line 1547)
    c_167315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 9), 'c')
    # Obtaining the member 'T' of a type (line 1547)
    T_167316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 9), c_167315, 'T')
    # Getting the type of 'scl' (line 1547)
    scl_167317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 13), 'scl')
    # Applying the binary operator 'div' (line 1547)
    result_div_167318 = python_operator(stypy.reporting.localization.Localization(__file__, 1547, 9), 'div', T_167316, scl_167317)
    
    # Obtaining the member 'T' of a type (line 1547)
    T_167319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 9), result_div_167318, 'T')
    # Assigning a type to the variable 'c' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'c', T_167319)
    
    
    # Getting the type of 'deg' (line 1550)
    deg_167320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1550)
    ndim_167321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 7), deg_167320, 'ndim')
    int_167322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1550, 18), 'int')
    # Applying the binary operator '>' (line 1550)
    result_gt_167323 = python_operator(stypy.reporting.localization.Localization(__file__, 1550, 7), '>', ndim_167321, int_167322)
    
    # Testing the type of an if condition (line 1550)
    if_condition_167324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1550, 4), result_gt_167323)
    # Assigning a type to the variable 'if_condition_167324' (line 1550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1550, 4), 'if_condition_167324', if_condition_167324)
    # SSA begins for if statement (line 1550)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1551)
    c_167325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1551)
    ndim_167326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 11), c_167325, 'ndim')
    int_167327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 21), 'int')
    # Applying the binary operator '==' (line 1551)
    result_eq_167328 = python_operator(stypy.reporting.localization.Localization(__file__, 1551, 11), '==', ndim_167326, int_167327)
    
    # Testing the type of an if condition (line 1551)
    if_condition_167329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1551, 8), result_eq_167328)
    # Assigning a type to the variable 'if_condition_167329' (line 1551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 8), 'if_condition_167329', if_condition_167329)
    # SSA begins for if statement (line 1551)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1552):
    
    # Assigning a Call to a Name (line 1552):
    
    # Call to zeros(...): (line 1552)
    # Processing the call arguments (line 1552)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1552)
    tuple_167332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1552, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1552)
    # Adding element type (line 1552)
    # Getting the type of 'lmax' (line 1552)
    lmax_167333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 27), 'lmax', False)
    int_167334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1552, 32), 'int')
    # Applying the binary operator '+' (line 1552)
    result_add_167335 = python_operator(stypy.reporting.localization.Localization(__file__, 1552, 27), '+', lmax_167333, int_167334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1552, 27), tuple_167332, result_add_167335)
    # Adding element type (line 1552)
    
    # Obtaining the type of the subscript
    int_167336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1552, 43), 'int')
    # Getting the type of 'c' (line 1552)
    c_167337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 35), 'c', False)
    # Obtaining the member 'shape' of a type (line 1552)
    shape_167338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1552, 35), c_167337, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1552)
    getitem___167339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1552, 35), shape_167338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1552)
    subscript_call_result_167340 = invoke(stypy.reporting.localization.Localization(__file__, 1552, 35), getitem___167339, int_167336)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1552, 27), tuple_167332, subscript_call_result_167340)
    
    # Processing the call keyword arguments (line 1552)
    # Getting the type of 'c' (line 1552)
    c_167341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 54), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1552)
    dtype_167342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1552, 54), c_167341, 'dtype')
    keyword_167343 = dtype_167342
    kwargs_167344 = {'dtype': keyword_167343}
    # Getting the type of 'np' (line 1552)
    np_167330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1552)
    zeros_167331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1552, 17), np_167330, 'zeros')
    # Calling zeros(args, kwargs) (line 1552)
    zeros_call_result_167345 = invoke(stypy.reporting.localization.Localization(__file__, 1552, 17), zeros_167331, *[tuple_167332], **kwargs_167344)
    
    # Assigning a type to the variable 'cc' (line 1552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1552, 12), 'cc', zeros_call_result_167345)
    # SSA branch for the else part of an if statement (line 1551)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1554):
    
    # Assigning a Call to a Name (line 1554):
    
    # Call to zeros(...): (line 1554)
    # Processing the call arguments (line 1554)
    # Getting the type of 'lmax' (line 1554)
    lmax_167348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 26), 'lmax', False)
    int_167349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1554, 31), 'int')
    # Applying the binary operator '+' (line 1554)
    result_add_167350 = python_operator(stypy.reporting.localization.Localization(__file__, 1554, 26), '+', lmax_167348, int_167349)
    
    # Processing the call keyword arguments (line 1554)
    # Getting the type of 'c' (line 1554)
    c_167351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 40), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1554)
    dtype_167352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1554, 40), c_167351, 'dtype')
    keyword_167353 = dtype_167352
    kwargs_167354 = {'dtype': keyword_167353}
    # Getting the type of 'np' (line 1554)
    np_167346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1554, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1554)
    zeros_167347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1554, 17), np_167346, 'zeros')
    # Calling zeros(args, kwargs) (line 1554)
    zeros_call_result_167355 = invoke(stypy.reporting.localization.Localization(__file__, 1554, 17), zeros_167347, *[result_add_167350], **kwargs_167354)
    
    # Assigning a type to the variable 'cc' (line 1554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1554, 12), 'cc', zeros_call_result_167355)
    # SSA join for if statement (line 1551)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1555):
    
    # Assigning a Name to a Subscript (line 1555):
    # Getting the type of 'c' (line 1555)
    c_167356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 18), 'c')
    # Getting the type of 'cc' (line 1555)
    cc_167357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 8), 'cc')
    # Getting the type of 'deg' (line 1555)
    deg_167358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1555, 11), 'deg')
    # Storing an element on a container (line 1555)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1555, 8), cc_167357, (deg_167358, c_167356))
    
    # Assigning a Name to a Name (line 1556):
    
    # Assigning a Name to a Name (line 1556):
    # Getting the type of 'cc' (line 1556)
    cc_167359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1556, 8), 'c', cc_167359)
    # SSA join for if statement (line 1550)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1559)
    rank_167360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 7), 'rank')
    # Getting the type of 'order' (line 1559)
    order_167361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 15), 'order')
    # Applying the binary operator '!=' (line 1559)
    result_ne_167362 = python_operator(stypy.reporting.localization.Localization(__file__, 1559, 7), '!=', rank_167360, order_167361)
    
    
    # Getting the type of 'full' (line 1559)
    full_167363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1559, 29), 'full')
    # Applying the 'not' unary operator (line 1559)
    result_not__167364 = python_operator(stypy.reporting.localization.Localization(__file__, 1559, 25), 'not', full_167363)
    
    # Applying the binary operator 'and' (line 1559)
    result_and_keyword_167365 = python_operator(stypy.reporting.localization.Localization(__file__, 1559, 7), 'and', result_ne_167362, result_not__167364)
    
    # Testing the type of an if condition (line 1559)
    if_condition_167366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1559, 4), result_and_keyword_167365)
    # Assigning a type to the variable 'if_condition_167366' (line 1559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1559, 4), 'if_condition_167366', if_condition_167366)
    # SSA begins for if statement (line 1559)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1560):
    
    # Assigning a Str to a Name (line 1560):
    str_167367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1560, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1560, 8), 'msg', str_167367)
    
    # Call to warn(...): (line 1561)
    # Processing the call arguments (line 1561)
    # Getting the type of 'msg' (line 1561)
    msg_167370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 22), 'msg', False)
    # Getting the type of 'pu' (line 1561)
    pu_167371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1561)
    RankWarning_167372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1561, 27), pu_167371, 'RankWarning')
    # Processing the call keyword arguments (line 1561)
    kwargs_167373 = {}
    # Getting the type of 'warnings' (line 1561)
    warnings_167368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1561)
    warn_167369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1561, 8), warnings_167368, 'warn')
    # Calling warn(args, kwargs) (line 1561)
    warn_call_result_167374 = invoke(stypy.reporting.localization.Localization(__file__, 1561, 8), warn_167369, *[msg_167370, RankWarning_167372], **kwargs_167373)
    
    # SSA join for if statement (line 1559)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1563)
    full_167375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 7), 'full')
    # Testing the type of an if condition (line 1563)
    if_condition_167376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1563, 4), full_167375)
    # Assigning a type to the variable 'if_condition_167376' (line 1563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1563, 4), 'if_condition_167376', if_condition_167376)
    # SSA begins for if statement (line 1563)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1564)
    tuple_167377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1564, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1564)
    # Adding element type (line 1564)
    # Getting the type of 'c' (line 1564)
    c_167378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 15), tuple_167377, c_167378)
    # Adding element type (line 1564)
    
    # Obtaining an instance of the builtin type 'list' (line 1564)
    list_167379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1564, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1564)
    # Adding element type (line 1564)
    # Getting the type of 'resids' (line 1564)
    resids_167380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 18), list_167379, resids_167380)
    # Adding element type (line 1564)
    # Getting the type of 'rank' (line 1564)
    rank_167381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 18), list_167379, rank_167381)
    # Adding element type (line 1564)
    # Getting the type of 's' (line 1564)
    s_167382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 18), list_167379, s_167382)
    # Adding element type (line 1564)
    # Getting the type of 'rcond' (line 1564)
    rcond_167383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1564, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 18), list_167379, rcond_167383)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1564, 15), tuple_167377, list_167379)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1564, 8), 'stypy_return_type', tuple_167377)
    # SSA branch for the else part of an if statement (line 1563)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1566)
    c_167384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1566, 8), 'stypy_return_type', c_167384)
    # SSA join for if statement (line 1563)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermfit' in the type store
    # Getting the type of 'stypy_return_type' (line 1368)
    stypy_return_type_167385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167385)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermfit'
    return stypy_return_type_167385

# Assigning a type to the variable 'hermfit' (line 1368)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 0), 'hermfit', hermfit)

@norecursion
def hermcompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermcompanion'
    module_type_store = module_type_store.open_function_context('hermcompanion', 1569, 0, False)
    
    # Passed parameters checking function
    hermcompanion.stypy_localization = localization
    hermcompanion.stypy_type_of_self = None
    hermcompanion.stypy_type_store = module_type_store
    hermcompanion.stypy_function_name = 'hermcompanion'
    hermcompanion.stypy_param_names_list = ['c']
    hermcompanion.stypy_varargs_param_name = None
    hermcompanion.stypy_kwargs_param_name = None
    hermcompanion.stypy_call_defaults = defaults
    hermcompanion.stypy_call_varargs = varargs
    hermcompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermcompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermcompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermcompanion(...)' code ##################

    str_167386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, (-1)), 'str', 'Return the scaled companion matrix of c.\n\n    The basis polynomials are scaled so that the companion matrix is\n    symmetric when `c` is an Hermite basis polynomial. This provides\n    better eigenvalue estimates than the unscaled case and for basis\n    polynomials the eigenvalues are guaranteed to be real if\n    `numpy.linalg.eigvalsh` is used to obtain them.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Hermite series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Scaled companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1596):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1596)
    # Processing the call arguments (line 1596)
    
    # Obtaining an instance of the builtin type 'list' (line 1596)
    list_167389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1596)
    # Adding element type (line 1596)
    # Getting the type of 'c' (line 1596)
    c_167390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1596, 23), list_167389, c_167390)
    
    # Processing the call keyword arguments (line 1596)
    kwargs_167391 = {}
    # Getting the type of 'pu' (line 1596)
    pu_167387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1596)
    as_series_167388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 10), pu_167387, 'as_series')
    # Calling as_series(args, kwargs) (line 1596)
    as_series_call_result_167392 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 10), as_series_167388, *[list_167389], **kwargs_167391)
    
    # Assigning a type to the variable 'call_assignment_165056' (line 1596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 4), 'call_assignment_165056', as_series_call_result_167392)
    
    # Assigning a Call to a Name (line 1596):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167396 = {}
    # Getting the type of 'call_assignment_165056' (line 1596)
    call_assignment_165056_167393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 4), 'call_assignment_165056', False)
    # Obtaining the member '__getitem__' of a type (line 1596)
    getitem___167394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 4), call_assignment_165056_167393, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167397 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167394, *[int_167395], **kwargs_167396)
    
    # Assigning a type to the variable 'call_assignment_165057' (line 1596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 4), 'call_assignment_165057', getitem___call_result_167397)
    
    # Assigning a Name to a Name (line 1596):
    # Getting the type of 'call_assignment_165057' (line 1596)
    call_assignment_165057_167398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 4), 'call_assignment_165057')
    # Assigning a type to the variable 'c' (line 1596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 5), 'c', call_assignment_165057_167398)
    
    
    
    # Call to len(...): (line 1597)
    # Processing the call arguments (line 1597)
    # Getting the type of 'c' (line 1597)
    c_167400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 11), 'c', False)
    # Processing the call keyword arguments (line 1597)
    kwargs_167401 = {}
    # Getting the type of 'len' (line 1597)
    len_167399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 7), 'len', False)
    # Calling len(args, kwargs) (line 1597)
    len_call_result_167402 = invoke(stypy.reporting.localization.Localization(__file__, 1597, 7), len_167399, *[c_167400], **kwargs_167401)
    
    int_167403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1597, 16), 'int')
    # Applying the binary operator '<' (line 1597)
    result_lt_167404 = python_operator(stypy.reporting.localization.Localization(__file__, 1597, 7), '<', len_call_result_167402, int_167403)
    
    # Testing the type of an if condition (line 1597)
    if_condition_167405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1597, 4), result_lt_167404)
    # Assigning a type to the variable 'if_condition_167405' (line 1597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1597, 4), 'if_condition_167405', if_condition_167405)
    # SSA begins for if statement (line 1597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1598)
    # Processing the call arguments (line 1598)
    str_167407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1598)
    kwargs_167408 = {}
    # Getting the type of 'ValueError' (line 1598)
    ValueError_167406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1598)
    ValueError_call_result_167409 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 14), ValueError_167406, *[str_167407], **kwargs_167408)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1598, 8), ValueError_call_result_167409, 'raise parameter', BaseException)
    # SSA join for if statement (line 1597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1599)
    # Processing the call arguments (line 1599)
    # Getting the type of 'c' (line 1599)
    c_167411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1599, 11), 'c', False)
    # Processing the call keyword arguments (line 1599)
    kwargs_167412 = {}
    # Getting the type of 'len' (line 1599)
    len_167410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1599, 7), 'len', False)
    # Calling len(args, kwargs) (line 1599)
    len_call_result_167413 = invoke(stypy.reporting.localization.Localization(__file__, 1599, 7), len_167410, *[c_167411], **kwargs_167412)
    
    int_167414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1599, 17), 'int')
    # Applying the binary operator '==' (line 1599)
    result_eq_167415 = python_operator(stypy.reporting.localization.Localization(__file__, 1599, 7), '==', len_call_result_167413, int_167414)
    
    # Testing the type of an if condition (line 1599)
    if_condition_167416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1599, 4), result_eq_167415)
    # Assigning a type to the variable 'if_condition_167416' (line 1599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1599, 4), 'if_condition_167416', if_condition_167416)
    # SSA begins for if statement (line 1599)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1600)
    # Processing the call arguments (line 1600)
    
    # Obtaining an instance of the builtin type 'list' (line 1600)
    list_167419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1600)
    # Adding element type (line 1600)
    
    # Obtaining an instance of the builtin type 'list' (line 1600)
    list_167420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1600)
    # Adding element type (line 1600)
    float_167421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 26), 'float')
    
    # Obtaining the type of the subscript
    int_167422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 32), 'int')
    # Getting the type of 'c' (line 1600)
    c_167423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 30), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1600)
    getitem___167424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1600, 30), c_167423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1600)
    subscript_call_result_167425 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 30), getitem___167424, int_167422)
    
    # Applying the binary operator '*' (line 1600)
    result_mul_167426 = python_operator(stypy.reporting.localization.Localization(__file__, 1600, 26), '*', float_167421, subscript_call_result_167425)
    
    
    # Obtaining the type of the subscript
    int_167427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 37), 'int')
    # Getting the type of 'c' (line 1600)
    c_167428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 35), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1600)
    getitem___167429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1600, 35), c_167428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1600)
    subscript_call_result_167430 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 35), getitem___167429, int_167427)
    
    # Applying the binary operator 'div' (line 1600)
    result_div_167431 = python_operator(stypy.reporting.localization.Localization(__file__, 1600, 34), 'div', result_mul_167426, subscript_call_result_167430)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1600, 25), list_167420, result_div_167431)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1600, 24), list_167419, list_167420)
    
    # Processing the call keyword arguments (line 1600)
    kwargs_167432 = {}
    # Getting the type of 'np' (line 1600)
    np_167417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1600)
    array_167418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1600, 15), np_167417, 'array')
    # Calling array(args, kwargs) (line 1600)
    array_call_result_167433 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 15), array_167418, *[list_167419], **kwargs_167432)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 8), 'stypy_return_type', array_call_result_167433)
    # SSA join for if statement (line 1599)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1602):
    
    # Assigning a BinOp to a Name (line 1602):
    
    # Call to len(...): (line 1602)
    # Processing the call arguments (line 1602)
    # Getting the type of 'c' (line 1602)
    c_167435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 12), 'c', False)
    # Processing the call keyword arguments (line 1602)
    kwargs_167436 = {}
    # Getting the type of 'len' (line 1602)
    len_167434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 8), 'len', False)
    # Calling len(args, kwargs) (line 1602)
    len_call_result_167437 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 8), len_167434, *[c_167435], **kwargs_167436)
    
    int_167438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 17), 'int')
    # Applying the binary operator '-' (line 1602)
    result_sub_167439 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 8), '-', len_call_result_167437, int_167438)
    
    # Assigning a type to the variable 'n' (line 1602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 4), 'n', result_sub_167439)
    
    # Assigning a Call to a Name (line 1603):
    
    # Assigning a Call to a Name (line 1603):
    
    # Call to zeros(...): (line 1603)
    # Processing the call arguments (line 1603)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1603)
    tuple_167442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1603)
    # Adding element type (line 1603)
    # Getting the type of 'n' (line 1603)
    n_167443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1603, 20), tuple_167442, n_167443)
    # Adding element type (line 1603)
    # Getting the type of 'n' (line 1603)
    n_167444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1603, 20), tuple_167442, n_167444)
    
    # Processing the call keyword arguments (line 1603)
    # Getting the type of 'c' (line 1603)
    c_167445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1603)
    dtype_167446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 33), c_167445, 'dtype')
    keyword_167447 = dtype_167446
    kwargs_167448 = {'dtype': keyword_167447}
    # Getting the type of 'np' (line 1603)
    np_167440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1603)
    zeros_167441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), np_167440, 'zeros')
    # Calling zeros(args, kwargs) (line 1603)
    zeros_call_result_167449 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 10), zeros_167441, *[tuple_167442], **kwargs_167448)
    
    # Assigning a type to the variable 'mat' (line 1603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1603, 4), 'mat', zeros_call_result_167449)
    
    # Assigning a Call to a Name (line 1604):
    
    # Assigning a Call to a Name (line 1604):
    
    # Call to hstack(...): (line 1604)
    # Processing the call arguments (line 1604)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1604)
    tuple_167452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1604)
    # Adding element type (line 1604)
    float_167453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 21), tuple_167452, float_167453)
    # Adding element type (line 1604)
    float_167454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 25), 'float')
    
    # Call to sqrt(...): (line 1604)
    # Processing the call arguments (line 1604)
    float_167457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 36), 'float')
    
    # Call to arange(...): (line 1604)
    # Processing the call arguments (line 1604)
    # Getting the type of 'n' (line 1604)
    n_167460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 49), 'n', False)
    int_167461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 53), 'int')
    # Applying the binary operator '-' (line 1604)
    result_sub_167462 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 49), '-', n_167460, int_167461)
    
    int_167463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 56), 'int')
    int_167464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 59), 'int')
    # Processing the call keyword arguments (line 1604)
    kwargs_167465 = {}
    # Getting the type of 'np' (line 1604)
    np_167458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 39), 'np', False)
    # Obtaining the member 'arange' of a type (line 1604)
    arange_167459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 39), np_167458, 'arange')
    # Calling arange(args, kwargs) (line 1604)
    arange_call_result_167466 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 39), arange_167459, *[result_sub_167462, int_167463, int_167464], **kwargs_167465)
    
    # Applying the binary operator '*' (line 1604)
    result_mul_167467 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 36), '*', float_167457, arange_call_result_167466)
    
    # Processing the call keyword arguments (line 1604)
    kwargs_167468 = {}
    # Getting the type of 'np' (line 1604)
    np_167455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 28), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1604)
    sqrt_167456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 28), np_167455, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1604)
    sqrt_call_result_167469 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 28), sqrt_167456, *[result_mul_167467], **kwargs_167468)
    
    # Applying the binary operator 'div' (line 1604)
    result_div_167470 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 25), 'div', float_167454, sqrt_call_result_167469)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 21), tuple_167452, result_div_167470)
    
    # Processing the call keyword arguments (line 1604)
    kwargs_167471 = {}
    # Getting the type of 'np' (line 1604)
    np_167450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 10), 'np', False)
    # Obtaining the member 'hstack' of a type (line 1604)
    hstack_167451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 10), np_167450, 'hstack')
    # Calling hstack(args, kwargs) (line 1604)
    hstack_call_result_167472 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 10), hstack_167451, *[tuple_167452], **kwargs_167471)
    
    # Assigning a type to the variable 'scl' (line 1604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1604, 4), 'scl', hstack_call_result_167472)
    
    # Assigning a Subscript to a Name (line 1605):
    
    # Assigning a Subscript to a Name (line 1605):
    
    # Obtaining the type of the subscript
    int_167473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 40), 'int')
    slice_167474 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1605, 10), None, None, int_167473)
    
    # Call to accumulate(...): (line 1605)
    # Processing the call arguments (line 1605)
    # Getting the type of 'scl' (line 1605)
    scl_167478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 33), 'scl', False)
    # Processing the call keyword arguments (line 1605)
    kwargs_167479 = {}
    # Getting the type of 'np' (line 1605)
    np_167475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 10), 'np', False)
    # Obtaining the member 'multiply' of a type (line 1605)
    multiply_167476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 10), np_167475, 'multiply')
    # Obtaining the member 'accumulate' of a type (line 1605)
    accumulate_167477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 10), multiply_167476, 'accumulate')
    # Calling accumulate(args, kwargs) (line 1605)
    accumulate_call_result_167480 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 10), accumulate_167477, *[scl_167478], **kwargs_167479)
    
    # Obtaining the member '__getitem__' of a type (line 1605)
    getitem___167481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 10), accumulate_call_result_167480, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1605)
    subscript_call_result_167482 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 10), getitem___167481, slice_167474)
    
    # Assigning a type to the variable 'scl' (line 1605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1605, 4), 'scl', subscript_call_result_167482)
    
    # Assigning a Subscript to a Name (line 1606):
    
    # Assigning a Subscript to a Name (line 1606):
    
    # Obtaining the type of the subscript
    int_167483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 26), 'int')
    # Getting the type of 'n' (line 1606)
    n_167484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 29), 'n')
    int_167485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 31), 'int')
    # Applying the binary operator '+' (line 1606)
    result_add_167486 = python_operator(stypy.reporting.localization.Localization(__file__, 1606, 29), '+', n_167484, int_167485)
    
    slice_167487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1606, 10), int_167483, None, result_add_167486)
    
    # Call to reshape(...): (line 1606)
    # Processing the call arguments (line 1606)
    int_167490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 22), 'int')
    # Processing the call keyword arguments (line 1606)
    kwargs_167491 = {}
    # Getting the type of 'mat' (line 1606)
    mat_167488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1606)
    reshape_167489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1606, 10), mat_167488, 'reshape')
    # Calling reshape(args, kwargs) (line 1606)
    reshape_call_result_167492 = invoke(stypy.reporting.localization.Localization(__file__, 1606, 10), reshape_167489, *[int_167490], **kwargs_167491)
    
    # Obtaining the member '__getitem__' of a type (line 1606)
    getitem___167493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1606, 10), reshape_call_result_167492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1606)
    subscript_call_result_167494 = invoke(stypy.reporting.localization.Localization(__file__, 1606, 10), getitem___167493, slice_167487)
    
    # Assigning a type to the variable 'top' (line 1606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1606, 4), 'top', subscript_call_result_167494)
    
    # Assigning a Subscript to a Name (line 1607):
    
    # Assigning a Subscript to a Name (line 1607):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1607)
    n_167495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 26), 'n')
    # Getting the type of 'n' (line 1607)
    n_167496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 29), 'n')
    int_167497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1607, 31), 'int')
    # Applying the binary operator '+' (line 1607)
    result_add_167498 = python_operator(stypy.reporting.localization.Localization(__file__, 1607, 29), '+', n_167496, int_167497)
    
    slice_167499 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1607, 10), n_167495, None, result_add_167498)
    
    # Call to reshape(...): (line 1607)
    # Processing the call arguments (line 1607)
    int_167502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1607, 22), 'int')
    # Processing the call keyword arguments (line 1607)
    kwargs_167503 = {}
    # Getting the type of 'mat' (line 1607)
    mat_167500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1607)
    reshape_167501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1607, 10), mat_167500, 'reshape')
    # Calling reshape(args, kwargs) (line 1607)
    reshape_call_result_167504 = invoke(stypy.reporting.localization.Localization(__file__, 1607, 10), reshape_167501, *[int_167502], **kwargs_167503)
    
    # Obtaining the member '__getitem__' of a type (line 1607)
    getitem___167505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1607, 10), reshape_call_result_167504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1607)
    subscript_call_result_167506 = invoke(stypy.reporting.localization.Localization(__file__, 1607, 10), getitem___167505, slice_167499)
    
    # Assigning a type to the variable 'bot' (line 1607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1607, 4), 'bot', subscript_call_result_167506)
    
    # Assigning a Call to a Subscript (line 1608):
    
    # Assigning a Call to a Subscript (line 1608):
    
    # Call to sqrt(...): (line 1608)
    # Processing the call arguments (line 1608)
    float_167509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 23), 'float')
    
    # Call to arange(...): (line 1608)
    # Processing the call arguments (line 1608)
    int_167512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 36), 'int')
    # Getting the type of 'n' (line 1608)
    n_167513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 39), 'n', False)
    # Processing the call keyword arguments (line 1608)
    kwargs_167514 = {}
    # Getting the type of 'np' (line 1608)
    np_167510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 26), 'np', False)
    # Obtaining the member 'arange' of a type (line 1608)
    arange_167511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 26), np_167510, 'arange')
    # Calling arange(args, kwargs) (line 1608)
    arange_call_result_167515 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 26), arange_167511, *[int_167512, n_167513], **kwargs_167514)
    
    # Applying the binary operator '*' (line 1608)
    result_mul_167516 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 23), '*', float_167509, arange_call_result_167515)
    
    # Processing the call keyword arguments (line 1608)
    kwargs_167517 = {}
    # Getting the type of 'np' (line 1608)
    np_167507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 15), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1608)
    sqrt_167508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 15), np_167507, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1608)
    sqrt_call_result_167518 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 15), sqrt_167508, *[result_mul_167516], **kwargs_167517)
    
    # Getting the type of 'top' (line 1608)
    top_167519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'top')
    Ellipsis_167520 = Ellipsis
    # Storing an element on a container (line 1608)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1608, 4), top_167519, (Ellipsis_167520, sqrt_call_result_167518))
    
    # Assigning a Name to a Subscript (line 1609):
    
    # Assigning a Name to a Subscript (line 1609):
    # Getting the type of 'top' (line 1609)
    top_167521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 15), 'top')
    # Getting the type of 'bot' (line 1609)
    bot_167522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 4), 'bot')
    Ellipsis_167523 = Ellipsis
    # Storing an element on a container (line 1609)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1609, 4), bot_167522, (Ellipsis_167523, top_167521))
    
    # Getting the type of 'mat' (line 1610)
    mat_167524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_167525 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1610, 4), None, None, None)
    int_167526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 11), 'int')
    # Getting the type of 'mat' (line 1610)
    mat_167527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1610)
    getitem___167528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1610, 4), mat_167527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1610)
    subscript_call_result_167529 = invoke(stypy.reporting.localization.Localization(__file__, 1610, 4), getitem___167528, (slice_167525, int_167526))
    
    # Getting the type of 'scl' (line 1610)
    scl_167530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 18), 'scl')
    
    # Obtaining the type of the subscript
    int_167531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 25), 'int')
    slice_167532 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1610, 22), None, int_167531, None)
    # Getting the type of 'c' (line 1610)
    c_167533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 1610)
    getitem___167534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1610, 22), c_167533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1610)
    subscript_call_result_167535 = invoke(stypy.reporting.localization.Localization(__file__, 1610, 22), getitem___167534, slice_167532)
    
    # Applying the binary operator '*' (line 1610)
    result_mul_167536 = python_operator(stypy.reporting.localization.Localization(__file__, 1610, 18), '*', scl_167530, subscript_call_result_167535)
    
    float_167537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 30), 'float')
    
    # Obtaining the type of the subscript
    int_167538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 36), 'int')
    # Getting the type of 'c' (line 1610)
    c_167539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 34), 'c')
    # Obtaining the member '__getitem__' of a type (line 1610)
    getitem___167540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1610, 34), c_167539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1610)
    subscript_call_result_167541 = invoke(stypy.reporting.localization.Localization(__file__, 1610, 34), getitem___167540, int_167538)
    
    # Applying the binary operator '*' (line 1610)
    result_mul_167542 = python_operator(stypy.reporting.localization.Localization(__file__, 1610, 30), '*', float_167537, subscript_call_result_167541)
    
    # Applying the binary operator 'div' (line 1610)
    result_div_167543 = python_operator(stypy.reporting.localization.Localization(__file__, 1610, 28), 'div', result_mul_167536, result_mul_167542)
    
    # Applying the binary operator '-=' (line 1610)
    result_isub_167544 = python_operator(stypy.reporting.localization.Localization(__file__, 1610, 4), '-=', subscript_call_result_167529, result_div_167543)
    # Getting the type of 'mat' (line 1610)
    mat_167545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 4), 'mat')
    slice_167546 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1610, 4), None, None, None)
    int_167547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 11), 'int')
    # Storing an element on a container (line 1610)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 4), mat_167545, ((slice_167546, int_167547), result_isub_167544))
    
    # Getting the type of 'mat' (line 1611)
    mat_167548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1611, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1611, 4), 'stypy_return_type', mat_167548)
    
    # ################# End of 'hermcompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermcompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1569)
    stypy_return_type_167549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1569, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167549)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermcompanion'
    return stypy_return_type_167549

# Assigning a type to the variable 'hermcompanion' (line 1569)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1569, 0), 'hermcompanion', hermcompanion)

@norecursion
def hermroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermroots'
    module_type_store = module_type_store.open_function_context('hermroots', 1614, 0, False)
    
    # Passed parameters checking function
    hermroots.stypy_localization = localization
    hermroots.stypy_type_of_self = None
    hermroots.stypy_type_store = module_type_store
    hermroots.stypy_function_name = 'hermroots'
    hermroots.stypy_param_names_list = ['c']
    hermroots.stypy_varargs_param_name = None
    hermroots.stypy_kwargs_param_name = None
    hermroots.stypy_call_defaults = defaults
    hermroots.stypy_call_varargs = varargs
    hermroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermroots(...)' code ##################

    str_167550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, (-1)), 'str', '\n    Compute the roots of a Hermite series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * H_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    polyroots, legroots, lagroots, chebroots, hermeroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    The Hermite series basis polynomials aren\'t powers of `x` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite import hermroots, hermfromroots\n    >>> coef = hermfromroots([-1, 0, 1])\n    >>> coef\n    array([ 0.   ,  0.25 ,  0.   ,  0.125])\n    >>> hermroots(coef)\n    array([ -1.00000000e+00,  -1.38777878e-17,   1.00000000e+00])\n\n    ')
    
    # Assigning a Call to a List (line 1661):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1661)
    # Processing the call arguments (line 1661)
    
    # Obtaining an instance of the builtin type 'list' (line 1661)
    list_167553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1661)
    # Adding element type (line 1661)
    # Getting the type of 'c' (line 1661)
    c_167554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1661, 23), list_167553, c_167554)
    
    # Processing the call keyword arguments (line 1661)
    kwargs_167555 = {}
    # Getting the type of 'pu' (line 1661)
    pu_167551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1661)
    as_series_167552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 10), pu_167551, 'as_series')
    # Calling as_series(args, kwargs) (line 1661)
    as_series_call_result_167556 = invoke(stypy.reporting.localization.Localization(__file__, 1661, 10), as_series_167552, *[list_167553], **kwargs_167555)
    
    # Assigning a type to the variable 'call_assignment_165058' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 4), 'call_assignment_165058', as_series_call_result_167556)
    
    # Assigning a Call to a Name (line 1661):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 4), 'int')
    # Processing the call keyword arguments
    kwargs_167560 = {}
    # Getting the type of 'call_assignment_165058' (line 1661)
    call_assignment_165058_167557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 4), 'call_assignment_165058', False)
    # Obtaining the member '__getitem__' of a type (line 1661)
    getitem___167558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 4), call_assignment_165058_167557, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_167561 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___167558, *[int_167559], **kwargs_167560)
    
    # Assigning a type to the variable 'call_assignment_165059' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 4), 'call_assignment_165059', getitem___call_result_167561)
    
    # Assigning a Name to a Name (line 1661):
    # Getting the type of 'call_assignment_165059' (line 1661)
    call_assignment_165059_167562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 4), 'call_assignment_165059')
    # Assigning a type to the variable 'c' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 5), 'c', call_assignment_165059_167562)
    
    
    
    # Call to len(...): (line 1662)
    # Processing the call arguments (line 1662)
    # Getting the type of 'c' (line 1662)
    c_167564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 11), 'c', False)
    # Processing the call keyword arguments (line 1662)
    kwargs_167565 = {}
    # Getting the type of 'len' (line 1662)
    len_167563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 7), 'len', False)
    # Calling len(args, kwargs) (line 1662)
    len_call_result_167566 = invoke(stypy.reporting.localization.Localization(__file__, 1662, 7), len_167563, *[c_167564], **kwargs_167565)
    
    int_167567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 17), 'int')
    # Applying the binary operator '<=' (line 1662)
    result_le_167568 = python_operator(stypy.reporting.localization.Localization(__file__, 1662, 7), '<=', len_call_result_167566, int_167567)
    
    # Testing the type of an if condition (line 1662)
    if_condition_167569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1662, 4), result_le_167568)
    # Assigning a type to the variable 'if_condition_167569' (line 1662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1662, 4), 'if_condition_167569', if_condition_167569)
    # SSA begins for if statement (line 1662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1663)
    # Processing the call arguments (line 1663)
    
    # Obtaining an instance of the builtin type 'list' (line 1663)
    list_167572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1663)
    
    # Processing the call keyword arguments (line 1663)
    # Getting the type of 'c' (line 1663)
    c_167573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1663)
    dtype_167574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 34), c_167573, 'dtype')
    keyword_167575 = dtype_167574
    kwargs_167576 = {'dtype': keyword_167575}
    # Getting the type of 'np' (line 1663)
    np_167570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1663)
    array_167571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 15), np_167570, 'array')
    # Calling array(args, kwargs) (line 1663)
    array_call_result_167577 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 15), array_167571, *[list_167572], **kwargs_167576)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1663, 8), 'stypy_return_type', array_call_result_167577)
    # SSA join for if statement (line 1662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1664)
    # Processing the call arguments (line 1664)
    # Getting the type of 'c' (line 1664)
    c_167579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 11), 'c', False)
    # Processing the call keyword arguments (line 1664)
    kwargs_167580 = {}
    # Getting the type of 'len' (line 1664)
    len_167578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1664, 7), 'len', False)
    # Calling len(args, kwargs) (line 1664)
    len_call_result_167581 = invoke(stypy.reporting.localization.Localization(__file__, 1664, 7), len_167578, *[c_167579], **kwargs_167580)
    
    int_167582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1664, 17), 'int')
    # Applying the binary operator '==' (line 1664)
    result_eq_167583 = python_operator(stypy.reporting.localization.Localization(__file__, 1664, 7), '==', len_call_result_167581, int_167582)
    
    # Testing the type of an if condition (line 1664)
    if_condition_167584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1664, 4), result_eq_167583)
    # Assigning a type to the variable 'if_condition_167584' (line 1664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1664, 4), 'if_condition_167584', if_condition_167584)
    # SSA begins for if statement (line 1664)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1665)
    # Processing the call arguments (line 1665)
    
    # Obtaining an instance of the builtin type 'list' (line 1665)
    list_167587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1665)
    # Adding element type (line 1665)
    float_167588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 25), 'float')
    
    # Obtaining the type of the subscript
    int_167589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 31), 'int')
    # Getting the type of 'c' (line 1665)
    c_167590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1665)
    getitem___167591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1665, 29), c_167590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1665)
    subscript_call_result_167592 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 29), getitem___167591, int_167589)
    
    # Applying the binary operator '*' (line 1665)
    result_mul_167593 = python_operator(stypy.reporting.localization.Localization(__file__, 1665, 25), '*', float_167588, subscript_call_result_167592)
    
    
    # Obtaining the type of the subscript
    int_167594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, 36), 'int')
    # Getting the type of 'c' (line 1665)
    c_167595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 34), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1665)
    getitem___167596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1665, 34), c_167595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1665)
    subscript_call_result_167597 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 34), getitem___167596, int_167594)
    
    # Applying the binary operator 'div' (line 1665)
    result_div_167598 = python_operator(stypy.reporting.localization.Localization(__file__, 1665, 33), 'div', result_mul_167593, subscript_call_result_167597)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1665, 24), list_167587, result_div_167598)
    
    # Processing the call keyword arguments (line 1665)
    kwargs_167599 = {}
    # Getting the type of 'np' (line 1665)
    np_167585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1665)
    array_167586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1665, 15), np_167585, 'array')
    # Calling array(args, kwargs) (line 1665)
    array_call_result_167600 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 15), array_167586, *[list_167587], **kwargs_167599)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1665, 8), 'stypy_return_type', array_call_result_167600)
    # SSA join for if statement (line 1664)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1667):
    
    # Assigning a Call to a Name (line 1667):
    
    # Call to hermcompanion(...): (line 1667)
    # Processing the call arguments (line 1667)
    # Getting the type of 'c' (line 1667)
    c_167602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 22), 'c', False)
    # Processing the call keyword arguments (line 1667)
    kwargs_167603 = {}
    # Getting the type of 'hermcompanion' (line 1667)
    hermcompanion_167601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 8), 'hermcompanion', False)
    # Calling hermcompanion(args, kwargs) (line 1667)
    hermcompanion_call_result_167604 = invoke(stypy.reporting.localization.Localization(__file__, 1667, 8), hermcompanion_167601, *[c_167602], **kwargs_167603)
    
    # Assigning a type to the variable 'm' (line 1667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1667, 4), 'm', hermcompanion_call_result_167604)
    
    # Assigning a Call to a Name (line 1668):
    
    # Assigning a Call to a Name (line 1668):
    
    # Call to eigvals(...): (line 1668)
    # Processing the call arguments (line 1668)
    # Getting the type of 'm' (line 1668)
    m_167607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 19), 'm', False)
    # Processing the call keyword arguments (line 1668)
    kwargs_167608 = {}
    # Getting the type of 'la' (line 1668)
    la_167605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1668)
    eigvals_167606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1668, 8), la_167605, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1668)
    eigvals_call_result_167609 = invoke(stypy.reporting.localization.Localization(__file__, 1668, 8), eigvals_167606, *[m_167607], **kwargs_167608)
    
    # Assigning a type to the variable 'r' (line 1668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1668, 4), 'r', eigvals_call_result_167609)
    
    # Call to sort(...): (line 1669)
    # Processing the call keyword arguments (line 1669)
    kwargs_167612 = {}
    # Getting the type of 'r' (line 1669)
    r_167610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1669)
    sort_167611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1669, 4), r_167610, 'sort')
    # Calling sort(args, kwargs) (line 1669)
    sort_call_result_167613 = invoke(stypy.reporting.localization.Localization(__file__, 1669, 4), sort_167611, *[], **kwargs_167612)
    
    # Getting the type of 'r' (line 1670)
    r_167614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1670, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1670, 4), 'stypy_return_type', r_167614)
    
    # ################# End of 'hermroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1614)
    stypy_return_type_167615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1614, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermroots'
    return stypy_return_type_167615

# Assigning a type to the variable 'hermroots' (line 1614)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1614, 0), 'hermroots', hermroots)

@norecursion
def _normed_hermite_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_normed_hermite_n'
    module_type_store = module_type_store.open_function_context('_normed_hermite_n', 1673, 0, False)
    
    # Passed parameters checking function
    _normed_hermite_n.stypy_localization = localization
    _normed_hermite_n.stypy_type_of_self = None
    _normed_hermite_n.stypy_type_store = module_type_store
    _normed_hermite_n.stypy_function_name = '_normed_hermite_n'
    _normed_hermite_n.stypy_param_names_list = ['x', 'n']
    _normed_hermite_n.stypy_varargs_param_name = None
    _normed_hermite_n.stypy_kwargs_param_name = None
    _normed_hermite_n.stypy_call_defaults = defaults
    _normed_hermite_n.stypy_call_varargs = varargs
    _normed_hermite_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_normed_hermite_n', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_normed_hermite_n', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_normed_hermite_n(...)' code ##################

    str_167616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1701, (-1)), 'str', '\n    Evaluate a normalized Hermite polynomial.\n\n    Compute the value of the normalized Hermite polynomial of degree ``n``\n    at the points ``x``.\n\n\n    Parameters\n    ----------\n    x : ndarray of double.\n        Points at which to evaluate the function\n    n : int\n        Degree of the normalized Hermite function to be evaluated.\n\n    Returns\n    -------\n    values : ndarray\n        The shape of the return value is described above.\n\n    Notes\n    -----\n    .. versionadded:: 1.10.0\n\n    This function is needed for finding the Gauss points and integration\n    weights for high degrees. The values of the standard Hermite functions\n    overflow when n >= 207.\n\n    ')
    
    
    # Getting the type of 'n' (line 1702)
    n_167617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1702, 7), 'n')
    int_167618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1702, 12), 'int')
    # Applying the binary operator '==' (line 1702)
    result_eq_167619 = python_operator(stypy.reporting.localization.Localization(__file__, 1702, 7), '==', n_167617, int_167618)
    
    # Testing the type of an if condition (line 1702)
    if_condition_167620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1702, 4), result_eq_167619)
    # Assigning a type to the variable 'if_condition_167620' (line 1702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1702, 4), 'if_condition_167620', if_condition_167620)
    # SSA begins for if statement (line 1702)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1703)
    # Processing the call arguments (line 1703)
    # Getting the type of 'x' (line 1703)
    x_167623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1703, 23), 'x', False)
    # Obtaining the member 'shape' of a type (line 1703)
    shape_167624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1703, 23), x_167623, 'shape')
    # Processing the call keyword arguments (line 1703)
    kwargs_167625 = {}
    # Getting the type of 'np' (line 1703)
    np_167621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1703, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1703)
    ones_167622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1703, 15), np_167621, 'ones')
    # Calling ones(args, kwargs) (line 1703)
    ones_call_result_167626 = invoke(stypy.reporting.localization.Localization(__file__, 1703, 15), ones_167622, *[shape_167624], **kwargs_167625)
    
    
    # Call to sqrt(...): (line 1703)
    # Processing the call arguments (line 1703)
    
    # Call to sqrt(...): (line 1703)
    # Processing the call arguments (line 1703)
    # Getting the type of 'np' (line 1703)
    np_167631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1703, 48), 'np', False)
    # Obtaining the member 'pi' of a type (line 1703)
    pi_167632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1703, 48), np_167631, 'pi')
    # Processing the call keyword arguments (line 1703)
    kwargs_167633 = {}
    # Getting the type of 'np' (line 1703)
    np_167629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1703, 40), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1703)
    sqrt_167630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1703, 40), np_167629, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1703)
    sqrt_call_result_167634 = invoke(stypy.reporting.localization.Localization(__file__, 1703, 40), sqrt_167630, *[pi_167632], **kwargs_167633)
    
    # Processing the call keyword arguments (line 1703)
    kwargs_167635 = {}
    # Getting the type of 'np' (line 1703)
    np_167627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1703, 32), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1703)
    sqrt_167628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1703, 32), np_167627, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1703)
    sqrt_call_result_167636 = invoke(stypy.reporting.localization.Localization(__file__, 1703, 32), sqrt_167628, *[sqrt_call_result_167634], **kwargs_167635)
    
    # Applying the binary operator 'div' (line 1703)
    result_div_167637 = python_operator(stypy.reporting.localization.Localization(__file__, 1703, 15), 'div', ones_call_result_167626, sqrt_call_result_167636)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1703, 8), 'stypy_return_type', result_div_167637)
    # SSA join for if statement (line 1702)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 1705):
    
    # Assigning a Num to a Name (line 1705):
    float_167638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1705, 9), 'float')
    # Assigning a type to the variable 'c0' (line 1705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1705, 4), 'c0', float_167638)
    
    # Assigning a BinOp to a Name (line 1706):
    
    # Assigning a BinOp to a Name (line 1706):
    float_167639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1706, 9), 'float')
    
    # Call to sqrt(...): (line 1706)
    # Processing the call arguments (line 1706)
    
    # Call to sqrt(...): (line 1706)
    # Processing the call arguments (line 1706)
    # Getting the type of 'np' (line 1706)
    np_167644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1706, 28), 'np', False)
    # Obtaining the member 'pi' of a type (line 1706)
    pi_167645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1706, 28), np_167644, 'pi')
    # Processing the call keyword arguments (line 1706)
    kwargs_167646 = {}
    # Getting the type of 'np' (line 1706)
    np_167642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1706, 20), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1706)
    sqrt_167643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1706, 20), np_167642, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1706)
    sqrt_call_result_167647 = invoke(stypy.reporting.localization.Localization(__file__, 1706, 20), sqrt_167643, *[pi_167645], **kwargs_167646)
    
    # Processing the call keyword arguments (line 1706)
    kwargs_167648 = {}
    # Getting the type of 'np' (line 1706)
    np_167640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1706, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1706)
    sqrt_167641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1706, 12), np_167640, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1706)
    sqrt_call_result_167649 = invoke(stypy.reporting.localization.Localization(__file__, 1706, 12), sqrt_167641, *[sqrt_call_result_167647], **kwargs_167648)
    
    # Applying the binary operator 'div' (line 1706)
    result_div_167650 = python_operator(stypy.reporting.localization.Localization(__file__, 1706, 9), 'div', float_167639, sqrt_call_result_167649)
    
    # Assigning a type to the variable 'c1' (line 1706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1706, 4), 'c1', result_div_167650)
    
    # Assigning a Call to a Name (line 1707):
    
    # Assigning a Call to a Name (line 1707):
    
    # Call to float(...): (line 1707)
    # Processing the call arguments (line 1707)
    # Getting the type of 'n' (line 1707)
    n_167652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1707, 15), 'n', False)
    # Processing the call keyword arguments (line 1707)
    kwargs_167653 = {}
    # Getting the type of 'float' (line 1707)
    float_167651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1707, 9), 'float', False)
    # Calling float(args, kwargs) (line 1707)
    float_call_result_167654 = invoke(stypy.reporting.localization.Localization(__file__, 1707, 9), float_167651, *[n_167652], **kwargs_167653)
    
    # Assigning a type to the variable 'nd' (line 1707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1707, 4), 'nd', float_call_result_167654)
    
    
    # Call to range(...): (line 1708)
    # Processing the call arguments (line 1708)
    # Getting the type of 'n' (line 1708)
    n_167656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 19), 'n', False)
    int_167657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1708, 23), 'int')
    # Applying the binary operator '-' (line 1708)
    result_sub_167658 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 19), '-', n_167656, int_167657)
    
    # Processing the call keyword arguments (line 1708)
    kwargs_167659 = {}
    # Getting the type of 'range' (line 1708)
    range_167655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 13), 'range', False)
    # Calling range(args, kwargs) (line 1708)
    range_call_result_167660 = invoke(stypy.reporting.localization.Localization(__file__, 1708, 13), range_167655, *[result_sub_167658], **kwargs_167659)
    
    # Testing the type of a for loop iterable (line 1708)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1708, 4), range_call_result_167660)
    # Getting the type of the for loop variable (line 1708)
    for_loop_var_167661 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1708, 4), range_call_result_167660)
    # Assigning a type to the variable 'i' (line 1708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1708, 4), 'i', for_loop_var_167661)
    # SSA begins for a for statement (line 1708)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 1709):
    
    # Assigning a Name to a Name (line 1709):
    # Getting the type of 'c0' (line 1709)
    c0_167662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 14), 'c0')
    # Assigning a type to the variable 'tmp' (line 1709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1709, 8), 'tmp', c0_167662)
    
    # Assigning a BinOp to a Name (line 1710):
    
    # Assigning a BinOp to a Name (line 1710):
    
    # Getting the type of 'c1' (line 1710)
    c1_167663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1710, 14), 'c1')
    # Applying the 'usub' unary operator (line 1710)
    result___neg___167664 = python_operator(stypy.reporting.localization.Localization(__file__, 1710, 13), 'usub', c1_167663)
    
    
    # Call to sqrt(...): (line 1710)
    # Processing the call arguments (line 1710)
    # Getting the type of 'nd' (line 1710)
    nd_167667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1710, 26), 'nd', False)
    float_167668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1710, 31), 'float')
    # Applying the binary operator '-' (line 1710)
    result_sub_167669 = python_operator(stypy.reporting.localization.Localization(__file__, 1710, 26), '-', nd_167667, float_167668)
    
    # Getting the type of 'nd' (line 1710)
    nd_167670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1710, 35), 'nd', False)
    # Applying the binary operator 'div' (line 1710)
    result_div_167671 = python_operator(stypy.reporting.localization.Localization(__file__, 1710, 25), 'div', result_sub_167669, nd_167670)
    
    # Processing the call keyword arguments (line 1710)
    kwargs_167672 = {}
    # Getting the type of 'np' (line 1710)
    np_167665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1710, 17), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1710)
    sqrt_167666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1710, 17), np_167665, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1710)
    sqrt_call_result_167673 = invoke(stypy.reporting.localization.Localization(__file__, 1710, 17), sqrt_167666, *[result_div_167671], **kwargs_167672)
    
    # Applying the binary operator '*' (line 1710)
    result_mul_167674 = python_operator(stypy.reporting.localization.Localization(__file__, 1710, 13), '*', result___neg___167664, sqrt_call_result_167673)
    
    # Assigning a type to the variable 'c0' (line 1710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1710, 8), 'c0', result_mul_167674)
    
    # Assigning a BinOp to a Name (line 1711):
    
    # Assigning a BinOp to a Name (line 1711):
    # Getting the type of 'tmp' (line 1711)
    tmp_167675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 13), 'tmp')
    # Getting the type of 'c1' (line 1711)
    c1_167676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 19), 'c1')
    # Getting the type of 'x' (line 1711)
    x_167677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 22), 'x')
    # Applying the binary operator '*' (line 1711)
    result_mul_167678 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 19), '*', c1_167676, x_167677)
    
    
    # Call to sqrt(...): (line 1711)
    # Processing the call arguments (line 1711)
    float_167681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1711, 32), 'float')
    # Getting the type of 'nd' (line 1711)
    nd_167682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 35), 'nd', False)
    # Applying the binary operator 'div' (line 1711)
    result_div_167683 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 32), 'div', float_167681, nd_167682)
    
    # Processing the call keyword arguments (line 1711)
    kwargs_167684 = {}
    # Getting the type of 'np' (line 1711)
    np_167679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 24), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1711)
    sqrt_167680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1711, 24), np_167679, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1711)
    sqrt_call_result_167685 = invoke(stypy.reporting.localization.Localization(__file__, 1711, 24), sqrt_167680, *[result_div_167683], **kwargs_167684)
    
    # Applying the binary operator '*' (line 1711)
    result_mul_167686 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 23), '*', result_mul_167678, sqrt_call_result_167685)
    
    # Applying the binary operator '+' (line 1711)
    result_add_167687 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 13), '+', tmp_167675, result_mul_167686)
    
    # Assigning a type to the variable 'c1' (line 1711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1711, 8), 'c1', result_add_167687)
    
    # Assigning a BinOp to a Name (line 1712):
    
    # Assigning a BinOp to a Name (line 1712):
    # Getting the type of 'nd' (line 1712)
    nd_167688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1712, 13), 'nd')
    float_167689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1712, 18), 'float')
    # Applying the binary operator '-' (line 1712)
    result_sub_167690 = python_operator(stypy.reporting.localization.Localization(__file__, 1712, 13), '-', nd_167688, float_167689)
    
    # Assigning a type to the variable 'nd' (line 1712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1712, 8), 'nd', result_sub_167690)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 1713)
    c0_167691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 11), 'c0')
    # Getting the type of 'c1' (line 1713)
    c1_167692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 16), 'c1')
    # Getting the type of 'x' (line 1713)
    x_167693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 19), 'x')
    # Applying the binary operator '*' (line 1713)
    result_mul_167694 = python_operator(stypy.reporting.localization.Localization(__file__, 1713, 16), '*', c1_167692, x_167693)
    
    
    # Call to sqrt(...): (line 1713)
    # Processing the call arguments (line 1713)
    int_167697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1713, 29), 'int')
    # Processing the call keyword arguments (line 1713)
    kwargs_167698 = {}
    # Getting the type of 'np' (line 1713)
    np_167695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1713, 21), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1713)
    sqrt_167696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1713, 21), np_167695, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1713)
    sqrt_call_result_167699 = invoke(stypy.reporting.localization.Localization(__file__, 1713, 21), sqrt_167696, *[int_167697], **kwargs_167698)
    
    # Applying the binary operator '*' (line 1713)
    result_mul_167700 = python_operator(stypy.reporting.localization.Localization(__file__, 1713, 20), '*', result_mul_167694, sqrt_call_result_167699)
    
    # Applying the binary operator '+' (line 1713)
    result_add_167701 = python_operator(stypy.reporting.localization.Localization(__file__, 1713, 11), '+', c0_167691, result_mul_167700)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1713, 4), 'stypy_return_type', result_add_167701)
    
    # ################# End of '_normed_hermite_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_normed_hermite_n' in the type store
    # Getting the type of 'stypy_return_type' (line 1673)
    stypy_return_type_167702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1673, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167702)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_normed_hermite_n'
    return stypy_return_type_167702

# Assigning a type to the variable '_normed_hermite_n' (line 1673)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1673, 0), '_normed_hermite_n', _normed_hermite_n)

@norecursion
def hermgauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermgauss'
    module_type_store = module_type_store.open_function_context('hermgauss', 1716, 0, False)
    
    # Passed parameters checking function
    hermgauss.stypy_localization = localization
    hermgauss.stypy_type_of_self = None
    hermgauss.stypy_type_store = module_type_store
    hermgauss.stypy_function_name = 'hermgauss'
    hermgauss.stypy_param_names_list = ['deg']
    hermgauss.stypy_varargs_param_name = None
    hermgauss.stypy_kwargs_param_name = None
    hermgauss.stypy_call_defaults = defaults
    hermgauss.stypy_call_varargs = varargs
    hermgauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermgauss', ['deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermgauss', localization, ['deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermgauss(...)' code ##################

    str_167703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1751, (-1)), 'str', "\n    Gauss-Hermite quadrature.\n\n    Computes the sample points and weights for Gauss-Hermite quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`\n    with the weight function :math:`f(x) = \\exp(-x^2)`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    The results have only been tested up to degree 100, higher degrees may\n    be problematic. The weights are determined by using the fact that\n\n    .. math:: w_k = c / (H'_n(x_k) * H_{n-1}(x_k))\n\n    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`\n    is the k'th root of :math:`H_n`, and then scaling the results to get\n    the right value when integrating 1.\n\n    ")
    
    # Assigning a Call to a Name (line 1752):
    
    # Assigning a Call to a Name (line 1752):
    
    # Call to int(...): (line 1752)
    # Processing the call arguments (line 1752)
    # Getting the type of 'deg' (line 1752)
    deg_167705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 15), 'deg', False)
    # Processing the call keyword arguments (line 1752)
    kwargs_167706 = {}
    # Getting the type of 'int' (line 1752)
    int_167704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 11), 'int', False)
    # Calling int(args, kwargs) (line 1752)
    int_call_result_167707 = invoke(stypy.reporting.localization.Localization(__file__, 1752, 11), int_167704, *[deg_167705], **kwargs_167706)
    
    # Assigning a type to the variable 'ideg' (line 1752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1752, 4), 'ideg', int_call_result_167707)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ideg' (line 1753)
    ideg_167708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1753, 7), 'ideg')
    # Getting the type of 'deg' (line 1753)
    deg_167709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1753, 15), 'deg')
    # Applying the binary operator '!=' (line 1753)
    result_ne_167710 = python_operator(stypy.reporting.localization.Localization(__file__, 1753, 7), '!=', ideg_167708, deg_167709)
    
    
    # Getting the type of 'ideg' (line 1753)
    ideg_167711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1753, 22), 'ideg')
    int_167712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1753, 29), 'int')
    # Applying the binary operator '<' (line 1753)
    result_lt_167713 = python_operator(stypy.reporting.localization.Localization(__file__, 1753, 22), '<', ideg_167711, int_167712)
    
    # Applying the binary operator 'or' (line 1753)
    result_or_keyword_167714 = python_operator(stypy.reporting.localization.Localization(__file__, 1753, 7), 'or', result_ne_167710, result_lt_167713)
    
    # Testing the type of an if condition (line 1753)
    if_condition_167715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1753, 4), result_or_keyword_167714)
    # Assigning a type to the variable 'if_condition_167715' (line 1753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1753, 4), 'if_condition_167715', if_condition_167715)
    # SSA begins for if statement (line 1753)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1754)
    # Processing the call arguments (line 1754)
    str_167717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1754, 25), 'str', 'deg must be a non-negative integer')
    # Processing the call keyword arguments (line 1754)
    kwargs_167718 = {}
    # Getting the type of 'ValueError' (line 1754)
    ValueError_167716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1754, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1754)
    ValueError_call_result_167719 = invoke(stypy.reporting.localization.Localization(__file__, 1754, 14), ValueError_167716, *[str_167717], **kwargs_167718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1754, 8), ValueError_call_result_167719, 'raise parameter', BaseException)
    # SSA join for if statement (line 1753)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1758):
    
    # Assigning a Call to a Name (line 1758):
    
    # Call to array(...): (line 1758)
    # Processing the call arguments (line 1758)
    
    # Obtaining an instance of the builtin type 'list' (line 1758)
    list_167722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1758, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1758)
    # Adding element type (line 1758)
    int_167723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1758, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1758, 17), list_167722, int_167723)
    
    # Getting the type of 'deg' (line 1758)
    deg_167724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 21), 'deg', False)
    # Applying the binary operator '*' (line 1758)
    result_mul_167725 = python_operator(stypy.reporting.localization.Localization(__file__, 1758, 17), '*', list_167722, deg_167724)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1758)
    list_167726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1758, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1758)
    # Adding element type (line 1758)
    int_167727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1758, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1758, 27), list_167726, int_167727)
    
    # Applying the binary operator '+' (line 1758)
    result_add_167728 = python_operator(stypy.reporting.localization.Localization(__file__, 1758, 17), '+', result_mul_167725, list_167726)
    
    # Processing the call keyword arguments (line 1758)
    # Getting the type of 'np' (line 1758)
    np_167729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 38), 'np', False)
    # Obtaining the member 'float64' of a type (line 1758)
    float64_167730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1758, 38), np_167729, 'float64')
    keyword_167731 = float64_167730
    kwargs_167732 = {'dtype': keyword_167731}
    # Getting the type of 'np' (line 1758)
    np_167720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1758)
    array_167721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1758, 8), np_167720, 'array')
    # Calling array(args, kwargs) (line 1758)
    array_call_result_167733 = invoke(stypy.reporting.localization.Localization(__file__, 1758, 8), array_167721, *[result_add_167728], **kwargs_167732)
    
    # Assigning a type to the variable 'c' (line 1758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1758, 4), 'c', array_call_result_167733)
    
    # Assigning a Call to a Name (line 1759):
    
    # Assigning a Call to a Name (line 1759):
    
    # Call to hermcompanion(...): (line 1759)
    # Processing the call arguments (line 1759)
    # Getting the type of 'c' (line 1759)
    c_167735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 22), 'c', False)
    # Processing the call keyword arguments (line 1759)
    kwargs_167736 = {}
    # Getting the type of 'hermcompanion' (line 1759)
    hermcompanion_167734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1759, 8), 'hermcompanion', False)
    # Calling hermcompanion(args, kwargs) (line 1759)
    hermcompanion_call_result_167737 = invoke(stypy.reporting.localization.Localization(__file__, 1759, 8), hermcompanion_167734, *[c_167735], **kwargs_167736)
    
    # Assigning a type to the variable 'm' (line 1759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1759, 4), 'm', hermcompanion_call_result_167737)
    
    # Assigning a Call to a Name (line 1760):
    
    # Assigning a Call to a Name (line 1760):
    
    # Call to eigvalsh(...): (line 1760)
    # Processing the call arguments (line 1760)
    # Getting the type of 'm' (line 1760)
    m_167740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 20), 'm', False)
    # Processing the call keyword arguments (line 1760)
    kwargs_167741 = {}
    # Getting the type of 'la' (line 1760)
    la_167738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1760, 8), 'la', False)
    # Obtaining the member 'eigvalsh' of a type (line 1760)
    eigvalsh_167739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1760, 8), la_167738, 'eigvalsh')
    # Calling eigvalsh(args, kwargs) (line 1760)
    eigvalsh_call_result_167742 = invoke(stypy.reporting.localization.Localization(__file__, 1760, 8), eigvalsh_167739, *[m_167740], **kwargs_167741)
    
    # Assigning a type to the variable 'x' (line 1760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1760, 4), 'x', eigvalsh_call_result_167742)
    
    # Assigning a Call to a Name (line 1763):
    
    # Assigning a Call to a Name (line 1763):
    
    # Call to _normed_hermite_n(...): (line 1763)
    # Processing the call arguments (line 1763)
    # Getting the type of 'x' (line 1763)
    x_167744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 27), 'x', False)
    # Getting the type of 'ideg' (line 1763)
    ideg_167745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 30), 'ideg', False)
    # Processing the call keyword arguments (line 1763)
    kwargs_167746 = {}
    # Getting the type of '_normed_hermite_n' (line 1763)
    _normed_hermite_n_167743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 9), '_normed_hermite_n', False)
    # Calling _normed_hermite_n(args, kwargs) (line 1763)
    _normed_hermite_n_call_result_167747 = invoke(stypy.reporting.localization.Localization(__file__, 1763, 9), _normed_hermite_n_167743, *[x_167744, ideg_167745], **kwargs_167746)
    
    # Assigning a type to the variable 'dy' (line 1763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1763, 4), 'dy', _normed_hermite_n_call_result_167747)
    
    # Assigning a BinOp to a Name (line 1764):
    
    # Assigning a BinOp to a Name (line 1764):
    
    # Call to _normed_hermite_n(...): (line 1764)
    # Processing the call arguments (line 1764)
    # Getting the type of 'x' (line 1764)
    x_167749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 27), 'x', False)
    # Getting the type of 'ideg' (line 1764)
    ideg_167750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 30), 'ideg', False)
    int_167751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1764, 37), 'int')
    # Applying the binary operator '-' (line 1764)
    result_sub_167752 = python_operator(stypy.reporting.localization.Localization(__file__, 1764, 30), '-', ideg_167750, int_167751)
    
    # Processing the call keyword arguments (line 1764)
    kwargs_167753 = {}
    # Getting the type of '_normed_hermite_n' (line 1764)
    _normed_hermite_n_167748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 9), '_normed_hermite_n', False)
    # Calling _normed_hermite_n(args, kwargs) (line 1764)
    _normed_hermite_n_call_result_167754 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 9), _normed_hermite_n_167748, *[x_167749, result_sub_167752], **kwargs_167753)
    
    
    # Call to sqrt(...): (line 1764)
    # Processing the call arguments (line 1764)
    int_167757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1764, 50), 'int')
    # Getting the type of 'ideg' (line 1764)
    ideg_167758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 52), 'ideg', False)
    # Applying the binary operator '*' (line 1764)
    result_mul_167759 = python_operator(stypy.reporting.localization.Localization(__file__, 1764, 50), '*', int_167757, ideg_167758)
    
    # Processing the call keyword arguments (line 1764)
    kwargs_167760 = {}
    # Getting the type of 'np' (line 1764)
    np_167755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1764, 42), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1764)
    sqrt_167756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1764, 42), np_167755, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1764)
    sqrt_call_result_167761 = invoke(stypy.reporting.localization.Localization(__file__, 1764, 42), sqrt_167756, *[result_mul_167759], **kwargs_167760)
    
    # Applying the binary operator '*' (line 1764)
    result_mul_167762 = python_operator(stypy.reporting.localization.Localization(__file__, 1764, 9), '*', _normed_hermite_n_call_result_167754, sqrt_call_result_167761)
    
    # Assigning a type to the variable 'df' (line 1764)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1764, 4), 'df', result_mul_167762)
    
    # Getting the type of 'x' (line 1765)
    x_167763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 4), 'x')
    # Getting the type of 'dy' (line 1765)
    dy_167764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 9), 'dy')
    # Getting the type of 'df' (line 1765)
    df_167765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1765, 12), 'df')
    # Applying the binary operator 'div' (line 1765)
    result_div_167766 = python_operator(stypy.reporting.localization.Localization(__file__, 1765, 9), 'div', dy_167764, df_167765)
    
    # Applying the binary operator '-=' (line 1765)
    result_isub_167767 = python_operator(stypy.reporting.localization.Localization(__file__, 1765, 4), '-=', x_167763, result_div_167766)
    # Assigning a type to the variable 'x' (line 1765)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1765, 4), 'x', result_isub_167767)
    
    
    # Assigning a Call to a Name (line 1769):
    
    # Assigning a Call to a Name (line 1769):
    
    # Call to _normed_hermite_n(...): (line 1769)
    # Processing the call arguments (line 1769)
    # Getting the type of 'x' (line 1769)
    x_167769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1769, 27), 'x', False)
    # Getting the type of 'ideg' (line 1769)
    ideg_167770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1769, 30), 'ideg', False)
    int_167771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1769, 37), 'int')
    # Applying the binary operator '-' (line 1769)
    result_sub_167772 = python_operator(stypy.reporting.localization.Localization(__file__, 1769, 30), '-', ideg_167770, int_167771)
    
    # Processing the call keyword arguments (line 1769)
    kwargs_167773 = {}
    # Getting the type of '_normed_hermite_n' (line 1769)
    _normed_hermite_n_167768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1769, 9), '_normed_hermite_n', False)
    # Calling _normed_hermite_n(args, kwargs) (line 1769)
    _normed_hermite_n_call_result_167774 = invoke(stypy.reporting.localization.Localization(__file__, 1769, 9), _normed_hermite_n_167768, *[x_167769, result_sub_167772], **kwargs_167773)
    
    # Assigning a type to the variable 'fm' (line 1769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1769, 4), 'fm', _normed_hermite_n_call_result_167774)
    
    # Getting the type of 'fm' (line 1770)
    fm_167775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'fm')
    
    # Call to max(...): (line 1770)
    # Processing the call keyword arguments (line 1770)
    kwargs_167782 = {}
    
    # Call to abs(...): (line 1770)
    # Processing the call arguments (line 1770)
    # Getting the type of 'fm' (line 1770)
    fm_167778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 17), 'fm', False)
    # Processing the call keyword arguments (line 1770)
    kwargs_167779 = {}
    # Getting the type of 'np' (line 1770)
    np_167776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1770, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1770)
    abs_167777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 10), np_167776, 'abs')
    # Calling abs(args, kwargs) (line 1770)
    abs_call_result_167780 = invoke(stypy.reporting.localization.Localization(__file__, 1770, 10), abs_167777, *[fm_167778], **kwargs_167779)
    
    # Obtaining the member 'max' of a type (line 1770)
    max_167781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1770, 10), abs_call_result_167780, 'max')
    # Calling max(args, kwargs) (line 1770)
    max_call_result_167783 = invoke(stypy.reporting.localization.Localization(__file__, 1770, 10), max_167781, *[], **kwargs_167782)
    
    # Applying the binary operator 'div=' (line 1770)
    result_div_167784 = python_operator(stypy.reporting.localization.Localization(__file__, 1770, 4), 'div=', fm_167775, max_call_result_167783)
    # Assigning a type to the variable 'fm' (line 1770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1770, 4), 'fm', result_div_167784)
    
    
    # Assigning a BinOp to a Name (line 1771):
    
    # Assigning a BinOp to a Name (line 1771):
    int_167785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1771, 8), 'int')
    # Getting the type of 'fm' (line 1771)
    fm_167786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1771, 11), 'fm')
    # Getting the type of 'fm' (line 1771)
    fm_167787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1771, 16), 'fm')
    # Applying the binary operator '*' (line 1771)
    result_mul_167788 = python_operator(stypy.reporting.localization.Localization(__file__, 1771, 11), '*', fm_167786, fm_167787)
    
    # Applying the binary operator 'div' (line 1771)
    result_div_167789 = python_operator(stypy.reporting.localization.Localization(__file__, 1771, 8), 'div', int_167785, result_mul_167788)
    
    # Assigning a type to the variable 'w' (line 1771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1771, 4), 'w', result_div_167789)
    
    # Assigning a BinOp to a Name (line 1774):
    
    # Assigning a BinOp to a Name (line 1774):
    # Getting the type of 'w' (line 1774)
    w_167790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1774, 9), 'w')
    
    # Obtaining the type of the subscript
    int_167791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1774, 17), 'int')
    slice_167792 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1774, 13), None, None, int_167791)
    # Getting the type of 'w' (line 1774)
    w_167793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1774, 13), 'w')
    # Obtaining the member '__getitem__' of a type (line 1774)
    getitem___167794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1774, 13), w_167793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1774)
    subscript_call_result_167795 = invoke(stypy.reporting.localization.Localization(__file__, 1774, 13), getitem___167794, slice_167792)
    
    # Applying the binary operator '+' (line 1774)
    result_add_167796 = python_operator(stypy.reporting.localization.Localization(__file__, 1774, 9), '+', w_167790, subscript_call_result_167795)
    
    int_167797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1774, 22), 'int')
    # Applying the binary operator 'div' (line 1774)
    result_div_167798 = python_operator(stypy.reporting.localization.Localization(__file__, 1774, 8), 'div', result_add_167796, int_167797)
    
    # Assigning a type to the variable 'w' (line 1774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1774, 4), 'w', result_div_167798)
    
    # Assigning a BinOp to a Name (line 1775):
    
    # Assigning a BinOp to a Name (line 1775):
    # Getting the type of 'x' (line 1775)
    x_167799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1775, 9), 'x')
    
    # Obtaining the type of the subscript
    int_167800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1775, 17), 'int')
    slice_167801 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1775, 13), None, None, int_167800)
    # Getting the type of 'x' (line 1775)
    x_167802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1775, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 1775)
    getitem___167803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1775, 13), x_167802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1775)
    subscript_call_result_167804 = invoke(stypy.reporting.localization.Localization(__file__, 1775, 13), getitem___167803, slice_167801)
    
    # Applying the binary operator '-' (line 1775)
    result_sub_167805 = python_operator(stypy.reporting.localization.Localization(__file__, 1775, 9), '-', x_167799, subscript_call_result_167804)
    
    int_167806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1775, 22), 'int')
    # Applying the binary operator 'div' (line 1775)
    result_div_167807 = python_operator(stypy.reporting.localization.Localization(__file__, 1775, 8), 'div', result_sub_167805, int_167806)
    
    # Assigning a type to the variable 'x' (line 1775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1775, 4), 'x', result_div_167807)
    
    # Getting the type of 'w' (line 1778)
    w_167808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 4), 'w')
    
    # Call to sqrt(...): (line 1778)
    # Processing the call arguments (line 1778)
    # Getting the type of 'np' (line 1778)
    np_167811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 17), 'np', False)
    # Obtaining the member 'pi' of a type (line 1778)
    pi_167812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1778, 17), np_167811, 'pi')
    # Processing the call keyword arguments (line 1778)
    kwargs_167813 = {}
    # Getting the type of 'np' (line 1778)
    np_167809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 9), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1778)
    sqrt_167810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1778, 9), np_167809, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1778)
    sqrt_call_result_167814 = invoke(stypy.reporting.localization.Localization(__file__, 1778, 9), sqrt_167810, *[pi_167812], **kwargs_167813)
    
    
    # Call to sum(...): (line 1778)
    # Processing the call keyword arguments (line 1778)
    kwargs_167817 = {}
    # Getting the type of 'w' (line 1778)
    w_167815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 26), 'w', False)
    # Obtaining the member 'sum' of a type (line 1778)
    sum_167816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1778, 26), w_167815, 'sum')
    # Calling sum(args, kwargs) (line 1778)
    sum_call_result_167818 = invoke(stypy.reporting.localization.Localization(__file__, 1778, 26), sum_167816, *[], **kwargs_167817)
    
    # Applying the binary operator 'div' (line 1778)
    result_div_167819 = python_operator(stypy.reporting.localization.Localization(__file__, 1778, 9), 'div', sqrt_call_result_167814, sum_call_result_167818)
    
    # Applying the binary operator '*=' (line 1778)
    result_imul_167820 = python_operator(stypy.reporting.localization.Localization(__file__, 1778, 4), '*=', w_167808, result_div_167819)
    # Assigning a type to the variable 'w' (line 1778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1778, 4), 'w', result_imul_167820)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1780)
    tuple_167821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1780, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1780)
    # Adding element type (line 1780)
    # Getting the type of 'x' (line 1780)
    x_167822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1780, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1780, 11), tuple_167821, x_167822)
    # Adding element type (line 1780)
    # Getting the type of 'w' (line 1780)
    w_167823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1780, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1780, 11), tuple_167821, w_167823)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1780, 4), 'stypy_return_type', tuple_167821)
    
    # ################# End of 'hermgauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermgauss' in the type store
    # Getting the type of 'stypy_return_type' (line 1716)
    stypy_return_type_167824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1716, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167824)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermgauss'
    return stypy_return_type_167824

# Assigning a type to the variable 'hermgauss' (line 1716)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1716, 0), 'hermgauss', hermgauss)

@norecursion
def hermweight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermweight'
    module_type_store = module_type_store.open_function_context('hermweight', 1783, 0, False)
    
    # Passed parameters checking function
    hermweight.stypy_localization = localization
    hermweight.stypy_type_of_self = None
    hermweight.stypy_type_store = module_type_store
    hermweight.stypy_function_name = 'hermweight'
    hermweight.stypy_param_names_list = ['x']
    hermweight.stypy_varargs_param_name = None
    hermweight.stypy_kwargs_param_name = None
    hermweight.stypy_call_defaults = defaults
    hermweight.stypy_call_varargs = varargs
    hermweight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermweight', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermweight', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermweight(...)' code ##################

    str_167825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1806, (-1)), 'str', '\n    Weight function of the Hermite polynomials.\n\n    The weight function is :math:`\\exp(-x^2)` and the interval of\n    integration is :math:`[-\\inf, \\inf]`. the Hermite polynomials are\n    orthogonal, but not normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a Name (line 1807):
    
    # Assigning a Call to a Name (line 1807):
    
    # Call to exp(...): (line 1807)
    # Processing the call arguments (line 1807)
    
    # Getting the type of 'x' (line 1807)
    x_167828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1807, 16), 'x', False)
    int_167829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1807, 19), 'int')
    # Applying the binary operator '**' (line 1807)
    result_pow_167830 = python_operator(stypy.reporting.localization.Localization(__file__, 1807, 16), '**', x_167828, int_167829)
    
    # Applying the 'usub' unary operator (line 1807)
    result___neg___167831 = python_operator(stypy.reporting.localization.Localization(__file__, 1807, 15), 'usub', result_pow_167830)
    
    # Processing the call keyword arguments (line 1807)
    kwargs_167832 = {}
    # Getting the type of 'np' (line 1807)
    np_167826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1807, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1807)
    exp_167827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1807, 8), np_167826, 'exp')
    # Calling exp(args, kwargs) (line 1807)
    exp_call_result_167833 = invoke(stypy.reporting.localization.Localization(__file__, 1807, 8), exp_167827, *[result___neg___167831], **kwargs_167832)
    
    # Assigning a type to the variable 'w' (line 1807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1807, 4), 'w', exp_call_result_167833)
    # Getting the type of 'w' (line 1808)
    w_167834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1808, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1808, 4), 'stypy_return_type', w_167834)
    
    # ################# End of 'hermweight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermweight' in the type store
    # Getting the type of 'stypy_return_type' (line 1783)
    stypy_return_type_167835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1783, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_167835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermweight'
    return stypy_return_type_167835

# Assigning a type to the variable 'hermweight' (line 1783)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1783, 0), 'hermweight', hermweight)
# Declaration of the 'Hermite' class
# Getting the type of 'ABCPolyBase' (line 1815)
ABCPolyBase_167836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1815, 14), 'ABCPolyBase')

class Hermite(ABCPolyBase_167836, ):
    str_167837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1836, (-1)), 'str', "An Hermite series class.\n\n    The Hermite class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    attributes and methods listed in the `ABCPolyBase` documentation.\n\n    Parameters\n    ----------\n    coef : array_like\n        Hermite coefficients in order of increasing degree, i.e,\n        ``(1, 2, 3)`` gives ``1*H_0(x) + 2*H_1(X) + 3*H_2(x)``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [-1, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [-1, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 1838):
    
    # Assigning a Call to a Name (line 1839):
    
    # Assigning a Call to a Name (line 1840):
    
    # Assigning a Call to a Name (line 1841):
    
    # Assigning a Call to a Name (line 1842):
    
    # Assigning a Call to a Name (line 1843):
    
    # Assigning a Call to a Name (line 1844):
    
    # Assigning a Call to a Name (line 1845):
    
    # Assigning a Call to a Name (line 1846):
    
    # Assigning a Call to a Name (line 1847):
    
    # Assigning a Call to a Name (line 1848):
    
    # Assigning a Call to a Name (line 1849):
    
    # Assigning a Str to a Name (line 1852):
    
    # Assigning a Call to a Name (line 1853):
    
    # Assigning a Call to a Name (line 1854):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1815, 0, False)
        # Assigning a type to the variable 'self' (line 1816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1816, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Hermite.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Hermite' (line 1815)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1815, 0), 'Hermite', Hermite)

# Assigning a Call to a Name (line 1838):

# Call to staticmethod(...): (line 1838)
# Processing the call arguments (line 1838)
# Getting the type of 'hermadd' (line 1838)
hermadd_167839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 24), 'hermadd', False)
# Processing the call keyword arguments (line 1838)
kwargs_167840 = {}
# Getting the type of 'staticmethod' (line 1838)
staticmethod_167838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1838)
staticmethod_call_result_167841 = invoke(stypy.reporting.localization.Localization(__file__, 1838, 11), staticmethod_167838, *[hermadd_167839], **kwargs_167840)

# Getting the type of 'Hermite'
Hermite_167842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167842, '_add', staticmethod_call_result_167841)

# Assigning a Call to a Name (line 1839):

# Call to staticmethod(...): (line 1839)
# Processing the call arguments (line 1839)
# Getting the type of 'hermsub' (line 1839)
hermsub_167844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 24), 'hermsub', False)
# Processing the call keyword arguments (line 1839)
kwargs_167845 = {}
# Getting the type of 'staticmethod' (line 1839)
staticmethod_167843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1839)
staticmethod_call_result_167846 = invoke(stypy.reporting.localization.Localization(__file__, 1839, 11), staticmethod_167843, *[hermsub_167844], **kwargs_167845)

# Getting the type of 'Hermite'
Hermite_167847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167847, '_sub', staticmethod_call_result_167846)

# Assigning a Call to a Name (line 1840):

# Call to staticmethod(...): (line 1840)
# Processing the call arguments (line 1840)
# Getting the type of 'hermmul' (line 1840)
hermmul_167849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 24), 'hermmul', False)
# Processing the call keyword arguments (line 1840)
kwargs_167850 = {}
# Getting the type of 'staticmethod' (line 1840)
staticmethod_167848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1840)
staticmethod_call_result_167851 = invoke(stypy.reporting.localization.Localization(__file__, 1840, 11), staticmethod_167848, *[hermmul_167849], **kwargs_167850)

# Getting the type of 'Hermite'
Hermite_167852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167852, '_mul', staticmethod_call_result_167851)

# Assigning a Call to a Name (line 1841):

# Call to staticmethod(...): (line 1841)
# Processing the call arguments (line 1841)
# Getting the type of 'hermdiv' (line 1841)
hermdiv_167854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 24), 'hermdiv', False)
# Processing the call keyword arguments (line 1841)
kwargs_167855 = {}
# Getting the type of 'staticmethod' (line 1841)
staticmethod_167853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1841)
staticmethod_call_result_167856 = invoke(stypy.reporting.localization.Localization(__file__, 1841, 11), staticmethod_167853, *[hermdiv_167854], **kwargs_167855)

# Getting the type of 'Hermite'
Hermite_167857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167857, '_div', staticmethod_call_result_167856)

# Assigning a Call to a Name (line 1842):

# Call to staticmethod(...): (line 1842)
# Processing the call arguments (line 1842)
# Getting the type of 'hermpow' (line 1842)
hermpow_167859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 24), 'hermpow', False)
# Processing the call keyword arguments (line 1842)
kwargs_167860 = {}
# Getting the type of 'staticmethod' (line 1842)
staticmethod_167858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1842)
staticmethod_call_result_167861 = invoke(stypy.reporting.localization.Localization(__file__, 1842, 11), staticmethod_167858, *[hermpow_167859], **kwargs_167860)

# Getting the type of 'Hermite'
Hermite_167862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167862, '_pow', staticmethod_call_result_167861)

# Assigning a Call to a Name (line 1843):

# Call to staticmethod(...): (line 1843)
# Processing the call arguments (line 1843)
# Getting the type of 'hermval' (line 1843)
hermval_167864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 24), 'hermval', False)
# Processing the call keyword arguments (line 1843)
kwargs_167865 = {}
# Getting the type of 'staticmethod' (line 1843)
staticmethod_167863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1843)
staticmethod_call_result_167866 = invoke(stypy.reporting.localization.Localization(__file__, 1843, 11), staticmethod_167863, *[hermval_167864], **kwargs_167865)

# Getting the type of 'Hermite'
Hermite_167867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167867, '_val', staticmethod_call_result_167866)

# Assigning a Call to a Name (line 1844):

# Call to staticmethod(...): (line 1844)
# Processing the call arguments (line 1844)
# Getting the type of 'hermint' (line 1844)
hermint_167869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 24), 'hermint', False)
# Processing the call keyword arguments (line 1844)
kwargs_167870 = {}
# Getting the type of 'staticmethod' (line 1844)
staticmethod_167868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1844)
staticmethod_call_result_167871 = invoke(stypy.reporting.localization.Localization(__file__, 1844, 11), staticmethod_167868, *[hermint_167869], **kwargs_167870)

# Getting the type of 'Hermite'
Hermite_167872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167872, '_int', staticmethod_call_result_167871)

# Assigning a Call to a Name (line 1845):

# Call to staticmethod(...): (line 1845)
# Processing the call arguments (line 1845)
# Getting the type of 'hermder' (line 1845)
hermder_167874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1845, 24), 'hermder', False)
# Processing the call keyword arguments (line 1845)
kwargs_167875 = {}
# Getting the type of 'staticmethod' (line 1845)
staticmethod_167873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1845, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1845)
staticmethod_call_result_167876 = invoke(stypy.reporting.localization.Localization(__file__, 1845, 11), staticmethod_167873, *[hermder_167874], **kwargs_167875)

# Getting the type of 'Hermite'
Hermite_167877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167877, '_der', staticmethod_call_result_167876)

# Assigning a Call to a Name (line 1846):

# Call to staticmethod(...): (line 1846)
# Processing the call arguments (line 1846)
# Getting the type of 'hermfit' (line 1846)
hermfit_167879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1846, 24), 'hermfit', False)
# Processing the call keyword arguments (line 1846)
kwargs_167880 = {}
# Getting the type of 'staticmethod' (line 1846)
staticmethod_167878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1846, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1846)
staticmethod_call_result_167881 = invoke(stypy.reporting.localization.Localization(__file__, 1846, 11), staticmethod_167878, *[hermfit_167879], **kwargs_167880)

# Getting the type of 'Hermite'
Hermite_167882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167882, '_fit', staticmethod_call_result_167881)

# Assigning a Call to a Name (line 1847):

# Call to staticmethod(...): (line 1847)
# Processing the call arguments (line 1847)
# Getting the type of 'hermline' (line 1847)
hermline_167884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1847, 25), 'hermline', False)
# Processing the call keyword arguments (line 1847)
kwargs_167885 = {}
# Getting the type of 'staticmethod' (line 1847)
staticmethod_167883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1847, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1847)
staticmethod_call_result_167886 = invoke(stypy.reporting.localization.Localization(__file__, 1847, 12), staticmethod_167883, *[hermline_167884], **kwargs_167885)

# Getting the type of 'Hermite'
Hermite_167887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167887, '_line', staticmethod_call_result_167886)

# Assigning a Call to a Name (line 1848):

# Call to staticmethod(...): (line 1848)
# Processing the call arguments (line 1848)
# Getting the type of 'hermroots' (line 1848)
hermroots_167889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1848, 26), 'hermroots', False)
# Processing the call keyword arguments (line 1848)
kwargs_167890 = {}
# Getting the type of 'staticmethod' (line 1848)
staticmethod_167888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1848, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1848)
staticmethod_call_result_167891 = invoke(stypy.reporting.localization.Localization(__file__, 1848, 13), staticmethod_167888, *[hermroots_167889], **kwargs_167890)

# Getting the type of 'Hermite'
Hermite_167892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167892, '_roots', staticmethod_call_result_167891)

# Assigning a Call to a Name (line 1849):

# Call to staticmethod(...): (line 1849)
# Processing the call arguments (line 1849)
# Getting the type of 'hermfromroots' (line 1849)
hermfromroots_167894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1849, 30), 'hermfromroots', False)
# Processing the call keyword arguments (line 1849)
kwargs_167895 = {}
# Getting the type of 'staticmethod' (line 1849)
staticmethod_167893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1849, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1849)
staticmethod_call_result_167896 = invoke(stypy.reporting.localization.Localization(__file__, 1849, 17), staticmethod_167893, *[hermfromroots_167894], **kwargs_167895)

# Getting the type of 'Hermite'
Hermite_167897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167897, '_fromroots', staticmethod_call_result_167896)

# Assigning a Str to a Name (line 1852):
str_167898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1852, 15), 'str', 'herm')
# Getting the type of 'Hermite'
Hermite_167899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167899, 'nickname', str_167898)

# Assigning a Call to a Name (line 1853):

# Call to array(...): (line 1853)
# Processing the call arguments (line 1853)
# Getting the type of 'hermdomain' (line 1853)
hermdomain_167902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 22), 'hermdomain', False)
# Processing the call keyword arguments (line 1853)
kwargs_167903 = {}
# Getting the type of 'np' (line 1853)
np_167900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1853, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1853)
array_167901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1853, 13), np_167900, 'array')
# Calling array(args, kwargs) (line 1853)
array_call_result_167904 = invoke(stypy.reporting.localization.Localization(__file__, 1853, 13), array_167901, *[hermdomain_167902], **kwargs_167903)

# Getting the type of 'Hermite'
Hermite_167905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167905, 'domain', array_call_result_167904)

# Assigning a Call to a Name (line 1854):

# Call to array(...): (line 1854)
# Processing the call arguments (line 1854)
# Getting the type of 'hermdomain' (line 1854)
hermdomain_167908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 22), 'hermdomain', False)
# Processing the call keyword arguments (line 1854)
kwargs_167909 = {}
# Getting the type of 'np' (line 1854)
np_167906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1854, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1854)
array_167907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1854, 13), np_167906, 'array')
# Calling array(args, kwargs) (line 1854)
array_call_result_167910 = invoke(stypy.reporting.localization.Localization(__file__, 1854, 13), array_167907, *[hermdomain_167908], **kwargs_167909)

# Getting the type of 'Hermite'
Hermite_167911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Hermite')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Hermite_167911, 'window', array_call_result_167910)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
