
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Objects for dealing with Hermite_e series.
3: 
4: This module provides a number of objects (mostly functions) useful for
5: dealing with Hermite_e series, including a `HermiteE` class that
6: encapsulates the usual arithmetic operations.  (General information
7: on how this module represents and works with such polynomials is in the
8: docstring for its "parent" sub-package, `numpy.polynomial`).
9: 
10: Constants
11: ---------
12: - `hermedomain` -- Hermite_e series default domain, [-1,1].
13: - `hermezero` -- Hermite_e series that evaluates identically to 0.
14: - `hermeone` -- Hermite_e series that evaluates identically to 1.
15: - `hermex` -- Hermite_e series for the identity map, ``f(x) = x``.
16: 
17: Arithmetic
18: ----------
19: - `hermemulx` -- multiply a Hermite_e series in ``P_i(x)`` by ``x``.
20: - `hermeadd` -- add two Hermite_e series.
21: - `hermesub` -- subtract one Hermite_e series from another.
22: - `hermemul` -- multiply two Hermite_e series.
23: - `hermediv` -- divide one Hermite_e series by another.
24: - `hermeval` -- evaluate a Hermite_e series at given points.
25: - `hermeval2d` -- evaluate a 2D Hermite_e series at given points.
26: - `hermeval3d` -- evaluate a 3D Hermite_e series at given points.
27: - `hermegrid2d` -- evaluate a 2D Hermite_e series on a Cartesian product.
28: - `hermegrid3d` -- evaluate a 3D Hermite_e series on a Cartesian product.
29: 
30: Calculus
31: --------
32: - `hermeder` -- differentiate a Hermite_e series.
33: - `hermeint` -- integrate a Hermite_e series.
34: 
35: Misc Functions
36: --------------
37: - `hermefromroots` -- create a Hermite_e series with specified roots.
38: - `hermeroots` -- find the roots of a Hermite_e series.
39: - `hermevander` -- Vandermonde-like matrix for Hermite_e polynomials.
40: - `hermevander2d` -- Vandermonde-like matrix for 2D power series.
41: - `hermevander3d` -- Vandermonde-like matrix for 3D power series.
42: - `hermegauss` -- Gauss-Hermite_e quadrature, points and weights.
43: - `hermeweight` -- Hermite_e weight function.
44: - `hermecompanion` -- symmetrized companion matrix in Hermite_e form.
45: - `hermefit` -- least-squares fit returning a Hermite_e series.
46: - `hermetrim` -- trim leading coefficients from a Hermite_e series.
47: - `hermeline` -- Hermite_e series of given straight line.
48: - `herme2poly` -- convert a Hermite_e series to a polynomial.
49: - `poly2herme` -- convert a polynomial to a Hermite_e series.
50: 
51: Classes
52: -------
53: - `HermiteE` -- A Hermite_e series class.
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
70:     'hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline',
71:     'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv',
72:     'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly',
73:     'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim',
74:     'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d',
75:     'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion',
76:     'hermegauss', 'hermeweight']
77: 
78: hermetrim = pu.trimcoef
79: 
80: 
81: def poly2herme(pol):
82:     '''
83:     poly2herme(pol)
84: 
85:     Convert a polynomial to a Hermite series.
86: 
87:     Convert an array representing the coefficients of a polynomial (relative
88:     to the "standard" basis) ordered from lowest degree to highest, to an
89:     array of the coefficients of the equivalent Hermite series, ordered
90:     from lowest to highest degree.
91: 
92:     Parameters
93:     ----------
94:     pol : array_like
95:         1-D array containing the polynomial coefficients
96: 
97:     Returns
98:     -------
99:     c : ndarray
100:         1-D array containing the coefficients of the equivalent Hermite
101:         series.
102: 
103:     See Also
104:     --------
105:     herme2poly
106: 
107:     Notes
108:     -----
109:     The easy way to do conversions between polynomial basis sets
110:     is to use the convert method of a class instance.
111: 
112:     Examples
113:     --------
114:     >>> from numpy.polynomial.hermite_e import poly2herme
115:     >>> poly2herme(np.arange(4))
116:     array([  2.,  10.,   2.,   3.])
117: 
118:     '''
119:     [pol] = pu.as_series([pol])
120:     deg = len(pol) - 1
121:     res = 0
122:     for i in range(deg, -1, -1):
123:         res = hermeadd(hermemulx(res), pol[i])
124:     return res
125: 
126: 
127: def herme2poly(c):
128:     '''
129:     Convert a Hermite series to a polynomial.
130: 
131:     Convert an array representing the coefficients of a Hermite series,
132:     ordered from lowest degree to highest, to an array of the coefficients
133:     of the equivalent polynomial (relative to the "standard" basis) ordered
134:     from lowest to highest degree.
135: 
136:     Parameters
137:     ----------
138:     c : array_like
139:         1-D array containing the Hermite series coefficients, ordered
140:         from lowest order term to highest.
141: 
142:     Returns
143:     -------
144:     pol : ndarray
145:         1-D array containing the coefficients of the equivalent polynomial
146:         (relative to the "standard" basis) ordered from lowest order term
147:         to highest.
148: 
149:     See Also
150:     --------
151:     poly2herme
152: 
153:     Notes
154:     -----
155:     The easy way to do conversions between polynomial basis sets
156:     is to use the convert method of a class instance.
157: 
158:     Examples
159:     --------
160:     >>> from numpy.polynomial.hermite_e import herme2poly
161:     >>> herme2poly([  2.,  10.,   2.,   3.])
162:     array([ 0.,  1.,  2.,  3.])
163: 
164:     '''
165:     from .polynomial import polyadd, polysub, polymulx
166: 
167:     [c] = pu.as_series([c])
168:     n = len(c)
169:     if n == 1:
170:         return c
171:     if n == 2:
172:         return c
173:     else:
174:         c0 = c[-2]
175:         c1 = c[-1]
176:         # i is the current degree of c1
177:         for i in range(n - 1, 1, -1):
178:             tmp = c0
179:             c0 = polysub(c[i - 2], c1*(i - 1))
180:             c1 = polyadd(tmp, polymulx(c1))
181:         return polyadd(c0, polymulx(c1))
182: 
183: #
184: # These are constant arrays are of integer type so as to be compatible
185: # with the widest range of other types, such as Decimal.
186: #
187: 
188: # Hermite
189: hermedomain = np.array([-1, 1])
190: 
191: # Hermite coefficients representing zero.
192: hermezero = np.array([0])
193: 
194: # Hermite coefficients representing one.
195: hermeone = np.array([1])
196: 
197: # Hermite coefficients representing the identity x.
198: hermex = np.array([0, 1])
199: 
200: 
201: def hermeline(off, scl):
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
224:     >>> from numpy.polynomial.hermite_e import hermeline
225:     >>> from numpy.polynomial.hermite_e import hermeline, hermeval
226:     >>> hermeval(0,hermeline(3, 2))
227:     3.0
228:     >>> hermeval(1,hermeline(3, 2))
229:     5.0
230: 
231:     '''
232:     if scl != 0:
233:         return np.array([off, scl])
234:     else:
235:         return np.array([off])
236: 
237: 
238: def hermefromroots(roots):
239:     '''
240:     Generate a HermiteE series with given roots.
241: 
242:     The function returns the coefficients of the polynomial
243: 
244:     .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),
245: 
246:     in HermiteE form, where the `r_n` are the roots specified in `roots`.
247:     If a zero has multiplicity n, then it must appear in `roots` n times.
248:     For instance, if 2 is a root of multiplicity three and 3 is a root of
249:     multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
250:     roots can appear in any order.
251: 
252:     If the returned coefficients are `c`, then
253: 
254:     .. math:: p(x) = c_0 + c_1 * He_1(x) + ... +  c_n * He_n(x)
255: 
256:     The coefficient of the last term is not generally 1 for monic
257:     polynomials in HermiteE form.
258: 
259:     Parameters
260:     ----------
261:     roots : array_like
262:         Sequence containing the roots.
263: 
264:     Returns
265:     -------
266:     out : ndarray
267:         1-D array of coefficients.  If all roots are real then `out` is a
268:         real array, if some of the roots are complex, then `out` is complex
269:         even if all the coefficients in the result are real (see Examples
270:         below).
271: 
272:     See Also
273:     --------
274:     polyfromroots, legfromroots, lagfromroots, hermfromroots,
275:     chebfromroots.
276: 
277:     Examples
278:     --------
279:     >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval
280:     >>> coef = hermefromroots((-1, 0, 1))
281:     >>> hermeval((-1, 0, 1), coef)
282:     array([ 0.,  0.,  0.])
283:     >>> coef = hermefromroots((-1j, 1j))
284:     >>> hermeval((-1j, 1j), coef)
285:     array([ 0.+0.j,  0.+0.j])
286: 
287:     '''
288:     if len(roots) == 0:
289:         return np.ones(1)
290:     else:
291:         [roots] = pu.as_series([roots], trim=False)
292:         roots.sort()
293:         p = [hermeline(-r, 1) for r in roots]
294:         n = len(p)
295:         while n > 1:
296:             m, r = divmod(n, 2)
297:             tmp = [hermemul(p[i], p[i+m]) for i in range(m)]
298:             if r:
299:                 tmp[0] = hermemul(tmp[0], p[-1])
300:             p = tmp
301:             n = m
302:         return p[0]
303: 
304: 
305: def hermeadd(c1, c2):
306:     '''
307:     Add one Hermite series to another.
308: 
309:     Returns the sum of two Hermite series `c1` + `c2`.  The arguments
310:     are sequences of coefficients ordered from lowest order term to
311:     highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
312: 
313:     Parameters
314:     ----------
315:     c1, c2 : array_like
316:         1-D arrays of Hermite series coefficients ordered from low to
317:         high.
318: 
319:     Returns
320:     -------
321:     out : ndarray
322:         Array representing the Hermite series of their sum.
323: 
324:     See Also
325:     --------
326:     hermesub, hermemul, hermediv, hermepow
327: 
328:     Notes
329:     -----
330:     Unlike multiplication, division, etc., the sum of two Hermite series
331:     is a Hermite series (without having to "reproject" the result onto
332:     the basis set) so addition, just like that of "standard" polynomials,
333:     is simply "component-wise."
334: 
335:     Examples
336:     --------
337:     >>> from numpy.polynomial.hermite_e import hermeadd
338:     >>> hermeadd([1, 2, 3], [1, 2, 3, 4])
339:     array([ 2.,  4.,  6.,  4.])
340: 
341:     '''
342:     # c1, c2 are trimmed copies
343:     [c1, c2] = pu.as_series([c1, c2])
344:     if len(c1) > len(c2):
345:         c1[:c2.size] += c2
346:         ret = c1
347:     else:
348:         c2[:c1.size] += c1
349:         ret = c2
350:     return pu.trimseq(ret)
351: 
352: 
353: def hermesub(c1, c2):
354:     '''
355:     Subtract one Hermite series from another.
356: 
357:     Returns the difference of two Hermite series `c1` - `c2`.  The
358:     sequences of coefficients are from lowest order term to highest, i.e.,
359:     [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
360: 
361:     Parameters
362:     ----------
363:     c1, c2 : array_like
364:         1-D arrays of Hermite series coefficients ordered from low to
365:         high.
366: 
367:     Returns
368:     -------
369:     out : ndarray
370:         Of Hermite series coefficients representing their difference.
371: 
372:     See Also
373:     --------
374:     hermeadd, hermemul, hermediv, hermepow
375: 
376:     Notes
377:     -----
378:     Unlike multiplication, division, etc., the difference of two Hermite
379:     series is a Hermite series (without having to "reproject" the result
380:     onto the basis set) so subtraction, just like that of "standard"
381:     polynomials, is simply "component-wise."
382: 
383:     Examples
384:     --------
385:     >>> from numpy.polynomial.hermite_e import hermesub
386:     >>> hermesub([1, 2, 3, 4], [1, 2, 3])
387:     array([ 0.,  0.,  0.,  4.])
388: 
389:     '''
390:     # c1, c2 are trimmed copies
391:     [c1, c2] = pu.as_series([c1, c2])
392:     if len(c1) > len(c2):
393:         c1[:c2.size] -= c2
394:         ret = c1
395:     else:
396:         c2 = -c2
397:         c2[:c1.size] += c1
398:         ret = c2
399:     return pu.trimseq(ret)
400: 
401: 
402: def hermemulx(c):
403:     '''Multiply a Hermite series by x.
404: 
405:     Multiply the Hermite series `c` by x, where x is the independent
406:     variable.
407: 
408: 
409:     Parameters
410:     ----------
411:     c : array_like
412:         1-D array of Hermite series coefficients ordered from low to
413:         high.
414: 
415:     Returns
416:     -------
417:     out : ndarray
418:         Array representing the result of the multiplication.
419: 
420:     Notes
421:     -----
422:     The multiplication uses the recursion relationship for Hermite
423:     polynomials in the form
424: 
425:     .. math::
426: 
427:     xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))
428: 
429:     Examples
430:     --------
431:     >>> from numpy.polynomial.hermite_e import hermemulx
432:     >>> hermemulx([1, 2, 3])
433:     array([ 2.,  7.,  2.,  3.])
434: 
435:     '''
436:     # c is a trimmed copy
437:     [c] = pu.as_series([c])
438:     # The zero series needs special treatment
439:     if len(c) == 1 and c[0] == 0:
440:         return c
441: 
442:     prd = np.empty(len(c) + 1, dtype=c.dtype)
443:     prd[0] = c[0]*0
444:     prd[1] = c[0]
445:     for i in range(1, len(c)):
446:         prd[i + 1] = c[i]
447:         prd[i - 1] += c[i]*i
448:     return prd
449: 
450: 
451: def hermemul(c1, c2):
452:     '''
453:     Multiply one Hermite series by another.
454: 
455:     Returns the product of two Hermite series `c1` * `c2`.  The arguments
456:     are sequences of coefficients, from lowest order "term" to highest,
457:     e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.
458: 
459:     Parameters
460:     ----------
461:     c1, c2 : array_like
462:         1-D arrays of Hermite series coefficients ordered from low to
463:         high.
464: 
465:     Returns
466:     -------
467:     out : ndarray
468:         Of Hermite series coefficients representing their product.
469: 
470:     See Also
471:     --------
472:     hermeadd, hermesub, hermediv, hermepow
473: 
474:     Notes
475:     -----
476:     In general, the (polynomial) product of two C-series results in terms
477:     that are not in the Hermite polynomial basis set.  Thus, to express
478:     the product as a Hermite series, it is necessary to "reproject" the
479:     product onto said basis set, which may produce "unintuitive" (but
480:     correct) results; see Examples section below.
481: 
482:     Examples
483:     --------
484:     >>> from numpy.polynomial.hermite_e import hermemul
485:     >>> hermemul([1, 2, 3], [0, 1, 2])
486:     array([ 14.,  15.,  28.,   7.,   6.])
487: 
488:     '''
489:     # s1, s2 are trimmed copies
490:     [c1, c2] = pu.as_series([c1, c2])
491: 
492:     if len(c1) > len(c2):
493:         c = c2
494:         xs = c1
495:     else:
496:         c = c1
497:         xs = c2
498: 
499:     if len(c) == 1:
500:         c0 = c[0]*xs
501:         c1 = 0
502:     elif len(c) == 2:
503:         c0 = c[0]*xs
504:         c1 = c[1]*xs
505:     else:
506:         nd = len(c)
507:         c0 = c[-2]*xs
508:         c1 = c[-1]*xs
509:         for i in range(3, len(c) + 1):
510:             tmp = c0
511:             nd = nd - 1
512:             c0 = hermesub(c[-i]*xs, c1*(nd - 1))
513:             c1 = hermeadd(tmp, hermemulx(c1))
514:     return hermeadd(c0, hermemulx(c1))
515: 
516: 
517: def hermediv(c1, c2):
518:     '''
519:     Divide one Hermite series by another.
520: 
521:     Returns the quotient-with-remainder of two Hermite series
522:     `c1` / `c2`.  The arguments are sequences of coefficients from lowest
523:     order "term" to highest, e.g., [1,2,3] represents the series
524:     ``P_0 + 2*P_1 + 3*P_2``.
525: 
526:     Parameters
527:     ----------
528:     c1, c2 : array_like
529:         1-D arrays of Hermite series coefficients ordered from low to
530:         high.
531: 
532:     Returns
533:     -------
534:     [quo, rem] : ndarrays
535:         Of Hermite series coefficients representing the quotient and
536:         remainder.
537: 
538:     See Also
539:     --------
540:     hermeadd, hermesub, hermemul, hermepow
541: 
542:     Notes
543:     -----
544:     In general, the (polynomial) division of one Hermite series by another
545:     results in quotient and remainder terms that are not in the Hermite
546:     polynomial basis set.  Thus, to express these results as a Hermite
547:     series, it is necessary to "reproject" the results onto the Hermite
548:     basis set, which may produce "unintuitive" (but correct) results; see
549:     Examples section below.
550: 
551:     Examples
552:     --------
553:     >>> from numpy.polynomial.hermite_e import hermediv
554:     >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])
555:     (array([ 1.,  2.,  3.]), array([ 0.]))
556:     >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])
557:     (array([ 1.,  2.,  3.]), array([ 1.,  2.]))
558: 
559:     '''
560:     # c1, c2 are trimmed copies
561:     [c1, c2] = pu.as_series([c1, c2])
562:     if c2[-1] == 0:
563:         raise ZeroDivisionError()
564: 
565:     lc1 = len(c1)
566:     lc2 = len(c2)
567:     if lc1 < lc2:
568:         return c1[:1]*0, c1
569:     elif lc2 == 1:
570:         return c1/c2[-1], c1[:1]*0
571:     else:
572:         quo = np.empty(lc1 - lc2 + 1, dtype=c1.dtype)
573:         rem = c1
574:         for i in range(lc1 - lc2, - 1, -1):
575:             p = hermemul([0]*i + [1], c2)
576:             q = rem[-1]/p[-1]
577:             rem = rem[:-1] - q*p[:-1]
578:             quo[i] = q
579:         return quo, pu.trimseq(rem)
580: 
581: 
582: def hermepow(c, pow, maxpower=16):
583:     '''Raise a Hermite series to a power.
584: 
585:     Returns the Hermite series `c` raised to the power `pow`. The
586:     argument `c` is a sequence of coefficients ordered from low to high.
587:     i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``
588: 
589:     Parameters
590:     ----------
591:     c : array_like
592:         1-D array of Hermite series coefficients ordered from low to
593:         high.
594:     pow : integer
595:         Power to which the series will be raised
596:     maxpower : integer, optional
597:         Maximum power allowed. This is mainly to limit growth of the series
598:         to unmanageable size. Default is 16
599: 
600:     Returns
601:     -------
602:     coef : ndarray
603:         Hermite series of power.
604: 
605:     See Also
606:     --------
607:     hermeadd, hermesub, hermemul, hermediv
608: 
609:     Examples
610:     --------
611:     >>> from numpy.polynomial.hermite_e import hermepow
612:     >>> hermepow([1, 2, 3], 2)
613:     array([ 23.,  28.,  46.,  12.,   9.])
614: 
615:     '''
616:     # c is a trimmed copy
617:     [c] = pu.as_series([c])
618:     power = int(pow)
619:     if power != pow or power < 0:
620:         raise ValueError("Power must be a non-negative integer.")
621:     elif maxpower is not None and power > maxpower:
622:         raise ValueError("Power is too large")
623:     elif power == 0:
624:         return np.array([1], dtype=c.dtype)
625:     elif power == 1:
626:         return c
627:     else:
628:         # This can be made more efficient by using powers of two
629:         # in the usual way.
630:         prd = c
631:         for i in range(2, power + 1):
632:             prd = hermemul(prd, c)
633:         return prd
634: 
635: 
636: def hermeder(c, m=1, scl=1, axis=0):
637:     '''
638:     Differentiate a Hermite_e series.
639: 
640:     Returns the series coefficients `c` differentiated `m` times along
641:     `axis`.  At each iteration the result is multiplied by `scl` (the
642:     scaling factor is for use in a linear change of variable). The argument
643:     `c` is an array of coefficients from low to high degree along each
644:     axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``
645:     while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)
646:     + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1
647:     is ``y``.
648: 
649:     Parameters
650:     ----------
651:     c : array_like
652:         Array of Hermite_e series coefficients. If `c` is multidimensional
653:         the different axis correspond to different variables with the
654:         degree in each axis given by the corresponding index.
655:     m : int, optional
656:         Number of derivatives taken, must be non-negative. (Default: 1)
657:     scl : scalar, optional
658:         Each differentiation is multiplied by `scl`.  The end result is
659:         multiplication by ``scl**m``.  This is for use in a linear change of
660:         variable. (Default: 1)
661:     axis : int, optional
662:         Axis over which the derivative is taken. (Default: 0).
663: 
664:         .. versionadded:: 1.7.0
665: 
666:     Returns
667:     -------
668:     der : ndarray
669:         Hermite series of the derivative.
670: 
671:     See Also
672:     --------
673:     hermeint
674: 
675:     Notes
676:     -----
677:     In general, the result of differentiating a Hermite series does not
678:     resemble the same operation on a power series. Thus the result of this
679:     function may be "unintuitive," albeit correct; see Examples section
680:     below.
681: 
682:     Examples
683:     --------
684:     >>> from numpy.polynomial.hermite_e import hermeder
685:     >>> hermeder([ 1.,  1.,  1.,  1.])
686:     array([ 1.,  2.,  3.])
687:     >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)
688:     array([ 1.,  2.,  3.])
689: 
690:     '''
691:     c = np.array(c, ndmin=1, copy=1)
692:     if c.dtype.char in '?bBhHiIlLqQpP':
693:         c = c.astype(np.double)
694:     cnt, iaxis = [int(t) for t in [m, axis]]
695: 
696:     if cnt != m:
697:         raise ValueError("The order of derivation must be integer")
698:     if cnt < 0:
699:         raise ValueError("The order of derivation must be non-negative")
700:     if iaxis != axis:
701:         raise ValueError("The axis must be integer")
702:     if not -c.ndim <= iaxis < c.ndim:
703:         raise ValueError("The axis is out of range")
704:     if iaxis < 0:
705:         iaxis += c.ndim
706: 
707:     if cnt == 0:
708:         return c
709: 
710:     c = np.rollaxis(c, iaxis)
711:     n = len(c)
712:     if cnt >= n:
713:         return c[:1]*0
714:     else:
715:         for i in range(cnt):
716:             n = n - 1
717:             c *= scl
718:             der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
719:             for j in range(n, 0, -1):
720:                 der[j - 1] = j*c[j]
721:             c = der
722:     c = np.rollaxis(c, 0, iaxis + 1)
723:     return c
724: 
725: 
726: def hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
727:     '''
728:     Integrate a Hermite_e series.
729: 
730:     Returns the Hermite_e series coefficients `c` integrated `m` times from
731:     `lbnd` along `axis`. At each iteration the resulting series is
732:     **multiplied** by `scl` and an integration constant, `k`, is added.
733:     The scaling factor is for use in a linear change of variable.  ("Buyer
734:     beware": note that, depending on what one is doing, one may want `scl`
735:     to be the reciprocal of what one might expect; for more information,
736:     see the Notes section below.)  The argument `c` is an array of
737:     coefficients from low to high degree along each axis, e.g., [1,2,3]
738:     represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]
739:     represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +
740:     2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.
741: 
742:     Parameters
743:     ----------
744:     c : array_like
745:         Array of Hermite_e series coefficients. If c is multidimensional
746:         the different axis correspond to different variables with the
747:         degree in each axis given by the corresponding index.
748:     m : int, optional
749:         Order of integration, must be positive. (Default: 1)
750:     k : {[], list, scalar}, optional
751:         Integration constant(s).  The value of the first integral at
752:         ``lbnd`` is the first value in the list, the value of the second
753:         integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
754:         default), all constants are set to zero.  If ``m == 1``, a single
755:         scalar can be given instead of a list.
756:     lbnd : scalar, optional
757:         The lower bound of the integral. (Default: 0)
758:     scl : scalar, optional
759:         Following each integration the result is *multiplied* by `scl`
760:         before the integration constant is added. (Default: 1)
761:     axis : int, optional
762:         Axis over which the integral is taken. (Default: 0).
763: 
764:         .. versionadded:: 1.7.0
765: 
766:     Returns
767:     -------
768:     S : ndarray
769:         Hermite_e series coefficients of the integral.
770: 
771:     Raises
772:     ------
773:     ValueError
774:         If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or
775:         ``np.isscalar(scl) == False``.
776: 
777:     See Also
778:     --------
779:     hermeder
780: 
781:     Notes
782:     -----
783:     Note that the result of each integration is *multiplied* by `scl`.
784:     Why is this important to note?  Say one is making a linear change of
785:     variable :math:`u = ax + b` in an integral relative to `x`.  Then
786:     .. math::`dx = du/a`, so one will need to set `scl` equal to
787:     :math:`1/a` - perhaps not what one would have first thought.
788: 
789:     Also note that, in general, the result of integrating a C-series needs
790:     to be "reprojected" onto the C-series basis set.  Thus, typically,
791:     the result of this function is "unintuitive," albeit correct; see
792:     Examples section below.
793: 
794:     Examples
795:     --------
796:     >>> from numpy.polynomial.hermite_e import hermeint
797:     >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.
798:     array([ 1.,  1.,  1.,  1.])
799:     >>> hermeint([1, 2, 3], m=2) # integrate twice, value & deriv 0 at 0
800:     array([-0.25      ,  1.        ,  0.5       ,  0.33333333,  0.25      ])
801:     >>> hermeint([1, 2, 3], k=1) # integrate once, value 1 at 0.
802:     array([ 2.,  1.,  1.,  1.])
803:     >>> hermeint([1, 2, 3], lbnd=-1) # integrate once, value 0 at -1
804:     array([-1.,  1.,  1.,  1.])
805:     >>> hermeint([1, 2, 3], m=2, k=[1, 2], lbnd=-1)
806:     array([ 1.83333333,  0.        ,  0.5       ,  0.33333333,  0.25      ])
807: 
808:     '''
809:     c = np.array(c, ndmin=1, copy=1)
810:     if c.dtype.char in '?bBhHiIlLqQpP':
811:         c = c.astype(np.double)
812:     if not np.iterable(k):
813:         k = [k]
814:     cnt, iaxis = [int(t) for t in [m, axis]]
815: 
816:     if cnt != m:
817:         raise ValueError("The order of integration must be integer")
818:     if cnt < 0:
819:         raise ValueError("The order of integration must be non-negative")
820:     if len(k) > cnt:
821:         raise ValueError("Too many integration constants")
822:     if iaxis != axis:
823:         raise ValueError("The axis must be integer")
824:     if not -c.ndim <= iaxis < c.ndim:
825:         raise ValueError("The axis is out of range")
826:     if iaxis < 0:
827:         iaxis += c.ndim
828: 
829:     if cnt == 0:
830:         return c
831: 
832:     c = np.rollaxis(c, iaxis)
833:     k = list(k) + [0]*(cnt - len(k))
834:     for i in range(cnt):
835:         n = len(c)
836:         c *= scl
837:         if n == 1 and np.all(c[0] == 0):
838:             c[0] += k[i]
839:         else:
840:             tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
841:             tmp[0] = c[0]*0
842:             tmp[1] = c[0]
843:             for j in range(1, n):
844:                 tmp[j + 1] = c[j]/(j + 1)
845:             tmp[0] += k[i] - hermeval(lbnd, tmp)
846:             c = tmp
847:     c = np.rollaxis(c, 0, iaxis + 1)
848:     return c
849: 
850: 
851: def hermeval(x, c, tensor=True):
852:     '''
853:     Evaluate an HermiteE series at points x.
854: 
855:     If `c` is of length `n + 1`, this function returns the value:
856: 
857:     .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)
858: 
859:     The parameter `x` is converted to an array only if it is a tuple or a
860:     list, otherwise it is treated as a scalar. In either case, either `x`
861:     or its elements must support multiplication and addition both with
862:     themselves and with the elements of `c`.
863: 
864:     If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
865:     `c` is multidimensional, then the shape of the result depends on the
866:     value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
867:     x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
868:     scalars have shape (,).
869: 
870:     Trailing zeros in the coefficients will be used in the evaluation, so
871:     they should be avoided if efficiency is a concern.
872: 
873:     Parameters
874:     ----------
875:     x : array_like, compatible object
876:         If `x` is a list or tuple, it is converted to an ndarray, otherwise
877:         it is left unchanged and treated as a scalar. In either case, `x`
878:         or its elements must support addition and multiplication with
879:         with themselves and with the elements of `c`.
880:     c : array_like
881:         Array of coefficients ordered so that the coefficients for terms of
882:         degree n are contained in c[n]. If `c` is multidimensional the
883:         remaining indices enumerate multiple polynomials. In the two
884:         dimensional case the coefficients may be thought of as stored in
885:         the columns of `c`.
886:     tensor : boolean, optional
887:         If True, the shape of the coefficient array is extended with ones
888:         on the right, one for each dimension of `x`. Scalars have dimension 0
889:         for this action. The result is that every column of coefficients in
890:         `c` is evaluated for every element of `x`. If False, `x` is broadcast
891:         over the columns of `c` for the evaluation.  This keyword is useful
892:         when `c` is multidimensional. The default value is True.
893: 
894:         .. versionadded:: 1.7.0
895: 
896:     Returns
897:     -------
898:     values : ndarray, algebra_like
899:         The shape of the return value is described above.
900: 
901:     See Also
902:     --------
903:     hermeval2d, hermegrid2d, hermeval3d, hermegrid3d
904: 
905:     Notes
906:     -----
907:     The evaluation uses Clenshaw recursion, aka synthetic division.
908: 
909:     Examples
910:     --------
911:     >>> from numpy.polynomial.hermite_e import hermeval
912:     >>> coef = [1,2,3]
913:     >>> hermeval(1, coef)
914:     3.0
915:     >>> hermeval([[1,2],[3,4]], coef)
916:     array([[  3.,  14.],
917:            [ 31.,  54.]])
918: 
919:     '''
920:     c = np.array(c, ndmin=1, copy=0)
921:     if c.dtype.char in '?bBhHiIlLqQpP':
922:         c = c.astype(np.double)
923:     if isinstance(x, (tuple, list)):
924:         x = np.asarray(x)
925:     if isinstance(x, np.ndarray) and tensor:
926:         c = c.reshape(c.shape + (1,)*x.ndim)
927: 
928:     if len(c) == 1:
929:         c0 = c[0]
930:         c1 = 0
931:     elif len(c) == 2:
932:         c0 = c[0]
933:         c1 = c[1]
934:     else:
935:         nd = len(c)
936:         c0 = c[-2]
937:         c1 = c[-1]
938:         for i in range(3, len(c) + 1):
939:             tmp = c0
940:             nd = nd - 1
941:             c0 = c[-i] - c1*(nd - 1)
942:             c1 = tmp + c1*x
943:     return c0 + c1*x
944: 
945: 
946: def hermeval2d(x, y, c):
947:     '''
948:     Evaluate a 2-D HermiteE series at points (x, y).
949: 
950:     This function returns the values:
951: 
952:     .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * He_i(x) * He_j(y)
953: 
954:     The parameters `x` and `y` are converted to arrays only if they are
955:     tuples or a lists, otherwise they are treated as a scalars and they
956:     must have the same shape after conversion. In either case, either `x`
957:     and `y` or their elements must support multiplication and addition both
958:     with themselves and with the elements of `c`.
959: 
960:     If `c` is a 1-D array a one is implicitly appended to its shape to make
961:     it 2-D. The shape of the result will be c.shape[2:] + x.shape.
962: 
963:     Parameters
964:     ----------
965:     x, y : array_like, compatible objects
966:         The two dimensional series is evaluated at the points `(x, y)`,
967:         where `x` and `y` must have the same shape. If `x` or `y` is a list
968:         or tuple, it is first converted to an ndarray, otherwise it is left
969:         unchanged and if it isn't an ndarray it is treated as a scalar.
970:     c : array_like
971:         Array of coefficients ordered so that the coefficient of the term
972:         of multi-degree i,j is contained in ``c[i,j]``. If `c` has
973:         dimension greater than two the remaining indices enumerate multiple
974:         sets of coefficients.
975: 
976:     Returns
977:     -------
978:     values : ndarray, compatible object
979:         The values of the two dimensional polynomial at points formed with
980:         pairs of corresponding values from `x` and `y`.
981: 
982:     See Also
983:     --------
984:     hermeval, hermegrid2d, hermeval3d, hermegrid3d
985: 
986:     Notes
987:     -----
988: 
989:     .. versionadded::1.7.0
990: 
991:     '''
992:     try:
993:         x, y = np.array((x, y), copy=0)
994:     except:
995:         raise ValueError('x, y are incompatible')
996: 
997:     c = hermeval(x, c)
998:     c = hermeval(y, c, tensor=False)
999:     return c
1000: 
1001: 
1002: def hermegrid2d(x, y, c):
1003:     '''
1004:     Evaluate a 2-D HermiteE series on the Cartesian product of x and y.
1005: 
1006:     This function returns the values:
1007: 
1008:     .. math:: p(a,b) = \sum_{i,j} c_{i,j} * H_i(a) * H_j(b)
1009: 
1010:     where the points `(a, b)` consist of all pairs formed by taking
1011:     `a` from `x` and `b` from `y`. The resulting points form a grid with
1012:     `x` in the first dimension and `y` in the second.
1013: 
1014:     The parameters `x` and `y` are converted to arrays only if they are
1015:     tuples or a lists, otherwise they are treated as a scalars. In either
1016:     case, either `x` and `y` or their elements must support multiplication
1017:     and addition both with themselves and with the elements of `c`.
1018: 
1019:     If `c` has fewer than two dimensions, ones are implicitly appended to
1020:     its shape to make it 2-D. The shape of the result will be c.shape[2:] +
1021:     x.shape.
1022: 
1023:     Parameters
1024:     ----------
1025:     x, y : array_like, compatible objects
1026:         The two dimensional series is evaluated at the points in the
1027:         Cartesian product of `x` and `y`.  If `x` or `y` is a list or
1028:         tuple, it is first converted to an ndarray, otherwise it is left
1029:         unchanged and, if it isn't an ndarray, it is treated as a scalar.
1030:     c : array_like
1031:         Array of coefficients ordered so that the coefficients for terms of
1032:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1033:         greater than two the remaining indices enumerate multiple sets of
1034:         coefficients.
1035: 
1036:     Returns
1037:     -------
1038:     values : ndarray, compatible object
1039:         The values of the two dimensional polynomial at points in the Cartesian
1040:         product of `x` and `y`.
1041: 
1042:     See Also
1043:     --------
1044:     hermeval, hermeval2d, hermeval3d, hermegrid3d
1045: 
1046:     Notes
1047:     -----
1048: 
1049:     .. versionadded::1.7.0
1050: 
1051:     '''
1052:     c = hermeval(x, c)
1053:     c = hermeval(y, c)
1054:     return c
1055: 
1056: 
1057: def hermeval3d(x, y, z, c):
1058:     '''
1059:     Evaluate a 3-D Hermite_e series at points (x, y, z).
1060: 
1061:     This function returns the values:
1062: 
1063:     .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * He_i(x) * He_j(y) * He_k(z)
1064: 
1065:     The parameters `x`, `y`, and `z` are converted to arrays only if
1066:     they are tuples or a lists, otherwise they are treated as a scalars and
1067:     they must have the same shape after conversion. In either case, either
1068:     `x`, `y`, and `z` or their elements must support multiplication and
1069:     addition both with themselves and with the elements of `c`.
1070: 
1071:     If `c` has fewer than 3 dimensions, ones are implicitly appended to its
1072:     shape to make it 3-D. The shape of the result will be c.shape[3:] +
1073:     x.shape.
1074: 
1075:     Parameters
1076:     ----------
1077:     x, y, z : array_like, compatible object
1078:         The three dimensional series is evaluated at the points
1079:         `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
1080:         any of `x`, `y`, or `z` is a list or tuple, it is first converted
1081:         to an ndarray, otherwise it is left unchanged and if it isn't an
1082:         ndarray it is  treated as a scalar.
1083:     c : array_like
1084:         Array of coefficients ordered so that the coefficient of the term of
1085:         multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
1086:         greater than 3 the remaining indices enumerate multiple sets of
1087:         coefficients.
1088: 
1089:     Returns
1090:     -------
1091:     values : ndarray, compatible object
1092:         The values of the multidimensional polynomial on points formed with
1093:         triples of corresponding values from `x`, `y`, and `z`.
1094: 
1095:     See Also
1096:     --------
1097:     hermeval, hermeval2d, hermegrid2d, hermegrid3d
1098: 
1099:     Notes
1100:     -----
1101: 
1102:     .. versionadded::1.7.0
1103: 
1104:     '''
1105:     try:
1106:         x, y, z = np.array((x, y, z), copy=0)
1107:     except:
1108:         raise ValueError('x, y, z are incompatible')
1109: 
1110:     c = hermeval(x, c)
1111:     c = hermeval(y, c, tensor=False)
1112:     c = hermeval(z, c, tensor=False)
1113:     return c
1114: 
1115: 
1116: def hermegrid3d(x, y, z, c):
1117:     '''
1118:     Evaluate a 3-D HermiteE series on the Cartesian product of x, y, and z.
1119: 
1120:     This function returns the values:
1121: 
1122:     .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c)
1123: 
1124:     where the points `(a, b, c)` consist of all triples formed by taking
1125:     `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
1126:     a grid with `x` in the first dimension, `y` in the second, and `z` in
1127:     the third.
1128: 
1129:     The parameters `x`, `y`, and `z` are converted to arrays only if they
1130:     are tuples or a lists, otherwise they are treated as a scalars. In
1131:     either case, either `x`, `y`, and `z` or their elements must support
1132:     multiplication and addition both with themselves and with the elements
1133:     of `c`.
1134: 
1135:     If `c` has fewer than three dimensions, ones are implicitly appended to
1136:     its shape to make it 3-D. The shape of the result will be c.shape[3:] +
1137:     x.shape + y.shape + z.shape.
1138: 
1139:     Parameters
1140:     ----------
1141:     x, y, z : array_like, compatible objects
1142:         The three dimensional series is evaluated at the points in the
1143:         Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
1144:         list or tuple, it is first converted to an ndarray, otherwise it is
1145:         left unchanged and, if it isn't an ndarray, it is treated as a
1146:         scalar.
1147:     c : array_like
1148:         Array of coefficients ordered so that the coefficients for terms of
1149:         degree i,j are contained in ``c[i,j]``. If `c` has dimension
1150:         greater than two the remaining indices enumerate multiple sets of
1151:         coefficients.
1152: 
1153:     Returns
1154:     -------
1155:     values : ndarray, compatible object
1156:         The values of the two dimensional polynomial at points in the Cartesian
1157:         product of `x` and `y`.
1158: 
1159:     See Also
1160:     --------
1161:     hermeval, hermeval2d, hermegrid2d, hermeval3d
1162: 
1163:     Notes
1164:     -----
1165: 
1166:     .. versionadded::1.7.0
1167: 
1168:     '''
1169:     c = hermeval(x, c)
1170:     c = hermeval(y, c)
1171:     c = hermeval(z, c)
1172:     return c
1173: 
1174: 
1175: def hermevander(x, deg):
1176:     '''Pseudo-Vandermonde matrix of given degree.
1177: 
1178:     Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
1179:     `x`. The pseudo-Vandermonde matrix is defined by
1180: 
1181:     .. math:: V[..., i] = He_i(x),
1182: 
1183:     where `0 <= i <= deg`. The leading indices of `V` index the elements of
1184:     `x` and the last index is the degree of the HermiteE polynomial.
1185: 
1186:     If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
1187:     array ``V = hermevander(x, n)``, then ``np.dot(V, c)`` and
1188:     ``hermeval(x, c)`` are the same up to roundoff. This equivalence is
1189:     useful both for least squares fitting and for the evaluation of a large
1190:     number of HermiteE series of the same degree and sample points.
1191: 
1192:     Parameters
1193:     ----------
1194:     x : array_like
1195:         Array of points. The dtype is converted to float64 or complex128
1196:         depending on whether any of the elements are complex. If `x` is
1197:         scalar it is converted to a 1-D array.
1198:     deg : int
1199:         Degree of the resulting matrix.
1200: 
1201:     Returns
1202:     -------
1203:     vander : ndarray
1204:         The pseudo-Vandermonde matrix. The shape of the returned matrix is
1205:         ``x.shape + (deg + 1,)``, where The last index is the degree of the
1206:         corresponding HermiteE polynomial.  The dtype will be the same as
1207:         the converted `x`.
1208: 
1209:     Examples
1210:     --------
1211:     >>> from numpy.polynomial.hermite_e import hermevander
1212:     >>> x = np.array([-1, 0, 1])
1213:     >>> hermevander(x, 3)
1214:     array([[ 1., -1.,  0.,  2.],
1215:            [ 1.,  0., -1., -0.],
1216:            [ 1.,  1.,  0., -2.]])
1217: 
1218:     '''
1219:     ideg = int(deg)
1220:     if ideg != deg:
1221:         raise ValueError("deg must be integer")
1222:     if ideg < 0:
1223:         raise ValueError("deg must be non-negative")
1224: 
1225:     x = np.array(x, copy=0, ndmin=1) + 0.0
1226:     dims = (ideg + 1,) + x.shape
1227:     dtyp = x.dtype
1228:     v = np.empty(dims, dtype=dtyp)
1229:     v[0] = x*0 + 1
1230:     if ideg > 0:
1231:         v[1] = x
1232:         for i in range(2, ideg + 1):
1233:             v[i] = (v[i-1]*x - v[i-2]*(i - 1))
1234:     return np.rollaxis(v, 0, v.ndim)
1235: 
1236: 
1237: def hermevander2d(x, y, deg):
1238:     '''Pseudo-Vandermonde matrix of given degrees.
1239: 
1240:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1241:     points `(x, y)`. The pseudo-Vandermonde matrix is defined by
1242: 
1243:     .. math:: V[..., deg[1]*i + j] = He_i(x) * He_j(y),
1244: 
1245:     where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
1246:     `V` index the points `(x, y)` and the last index encodes the degrees of
1247:     the HermiteE polynomials.
1248: 
1249:     If ``V = hermevander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
1250:     correspond to the elements of a 2-D coefficient array `c` of shape
1251:     (xdeg + 1, ydeg + 1) in the order
1252: 
1253:     .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...
1254: 
1255:     and ``np.dot(V, c.flat)`` and ``hermeval2d(x, y, c)`` will be the same
1256:     up to roundoff. This equivalence is useful both for least squares
1257:     fitting and for the evaluation of a large number of 2-D HermiteE
1258:     series of the same degrees and sample points.
1259: 
1260:     Parameters
1261:     ----------
1262:     x, y : array_like
1263:         Arrays of point coordinates, all of the same shape. The dtypes
1264:         will be converted to either float64 or complex128 depending on
1265:         whether any of the elements are complex. Scalars are converted to
1266:         1-D arrays.
1267:     deg : list of ints
1268:         List of maximum degrees of the form [x_deg, y_deg].
1269: 
1270:     Returns
1271:     -------
1272:     vander2d : ndarray
1273:         The shape of the returned matrix is ``x.shape + (order,)``, where
1274:         :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
1275:         as the converted `x` and `y`.
1276: 
1277:     See Also
1278:     --------
1279:     hermevander, hermevander3d. hermeval2d, hermeval3d
1280: 
1281:     Notes
1282:     -----
1283: 
1284:     .. versionadded::1.7.0
1285: 
1286:     '''
1287:     ideg = [int(d) for d in deg]
1288:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1289:     if is_valid != [1, 1]:
1290:         raise ValueError("degrees must be non-negative integers")
1291:     degx, degy = ideg
1292:     x, y = np.array((x, y), copy=0) + 0.0
1293: 
1294:     vx = hermevander(x, degx)
1295:     vy = hermevander(y, degy)
1296:     v = vx[..., None]*vy[..., None,:]
1297:     return v.reshape(v.shape[:-2] + (-1,))
1298: 
1299: 
1300: def hermevander3d(x, y, z, deg):
1301:     '''Pseudo-Vandermonde matrix of given degrees.
1302: 
1303:     Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
1304:     points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
1305:     then Hehe pseudo-Vandermonde matrix is defined by
1306: 
1307:     .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = He_i(x)*He_j(y)*He_k(z),
1308: 
1309:     where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
1310:     indices of `V` index the points `(x, y, z)` and the last index encodes
1311:     the degrees of the HermiteE polynomials.
1312: 
1313:     If ``V = hermevander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
1314:     of `V` correspond to the elements of a 3-D coefficient array `c` of
1315:     shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order
1316: 
1317:     .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...
1318: 
1319:     and  ``np.dot(V, c.flat)`` and ``hermeval3d(x, y, z, c)`` will be the
1320:     same up to roundoff. This equivalence is useful both for least squares
1321:     fitting and for the evaluation of a large number of 3-D HermiteE
1322:     series of the same degrees and sample points.
1323: 
1324:     Parameters
1325:     ----------
1326:     x, y, z : array_like
1327:         Arrays of point coordinates, all of the same shape. The dtypes will
1328:         be converted to either float64 or complex128 depending on whether
1329:         any of the elements are complex. Scalars are converted to 1-D
1330:         arrays.
1331:     deg : list of ints
1332:         List of maximum degrees of the form [x_deg, y_deg, z_deg].
1333: 
1334:     Returns
1335:     -------
1336:     vander3d : ndarray
1337:         The shape of the returned matrix is ``x.shape + (order,)``, where
1338:         :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
1339:         be the same as the converted `x`, `y`, and `z`.
1340: 
1341:     See Also
1342:     --------
1343:     hermevander, hermevander3d. hermeval2d, hermeval3d
1344: 
1345:     Notes
1346:     -----
1347: 
1348:     .. versionadded::1.7.0
1349: 
1350:     '''
1351:     ideg = [int(d) for d in deg]
1352:     is_valid = [id == d and id >= 0 for id, d in zip(ideg, deg)]
1353:     if is_valid != [1, 1, 1]:
1354:         raise ValueError("degrees must be non-negative integers")
1355:     degx, degy, degz = ideg
1356:     x, y, z = np.array((x, y, z), copy=0) + 0.0
1357: 
1358:     vx = hermevander(x, degx)
1359:     vy = hermevander(y, degy)
1360:     vz = hermevander(z, degz)
1361:     v = vx[..., None, None]*vy[..., None,:, None]*vz[..., None, None,:]
1362:     return v.reshape(v.shape[:-3] + (-1,))
1363: 
1364: 
1365: def hermefit(x, y, deg, rcond=None, full=False, w=None):
1366:     '''
1367:     Least squares fit of Hermite series to data.
1368: 
1369:     Return the coefficients of a HermiteE series of degree `deg` that is
1370:     the least squares fit to the data values `y` given at points `x`. If
1371:     `y` is 1-D the returned coefficients will also be 1-D. If `y` is 2-D
1372:     multiple fits are done, one for each column of `y`, and the resulting
1373:     coefficients are stored in the corresponding columns of a 2-D return.
1374:     The fitted polynomial(s) are in the form
1375: 
1376:     .. math::  p(x) = c_0 + c_1 * He_1(x) + ... + c_n * He_n(x),
1377: 
1378:     where `n` is `deg`.
1379: 
1380:     Parameters
1381:     ----------
1382:     x : array_like, shape (M,)
1383:         x-coordinates of the M sample points ``(x[i], y[i])``.
1384:     y : array_like, shape (M,) or (M, K)
1385:         y-coordinates of the sample points. Several data sets of sample
1386:         points sharing the same x-coordinates can be fitted at once by
1387:         passing in a 2D-array that contains one dataset per column.
1388:     deg : int or 1-D array_like
1389:         Degree(s) of the fitting polynomials. If `deg` is a single integer
1390:         all terms up to and including the `deg`'th term are included in the
1391:         fit. For Numpy versions >= 1.11 a list of integers specifying the
1392:         degrees of the terms to include may be used instead.
1393:     rcond : float, optional
1394:         Relative condition number of the fit. Singular values smaller than
1395:         this relative to the largest singular value will be ignored. The
1396:         default value is len(x)*eps, where eps is the relative precision of
1397:         the float type, about 2e-16 in most cases.
1398:     full : bool, optional
1399:         Switch determining nature of return value. When it is False (the
1400:         default) just the coefficients are returned, when True diagnostic
1401:         information from the singular value decomposition is also returned.
1402:     w : array_like, shape (`M`,), optional
1403:         Weights. If not None, the contribution of each point
1404:         ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the
1405:         weights are chosen so that the errors of the products ``w[i]*y[i]``
1406:         all have the same variance.  The default value is None.
1407: 
1408:     Returns
1409:     -------
1410:     coef : ndarray, shape (M,) or (M, K)
1411:         Hermite coefficients ordered from low to high. If `y` was 2-D,
1412:         the coefficients for the data in column k  of `y` are in column
1413:         `k`.
1414: 
1415:     [residuals, rank, singular_values, rcond] : list
1416:         These values are only returned if `full` = True
1417: 
1418:         resid -- sum of squared residuals of the least squares fit
1419:         rank -- the numerical rank of the scaled Vandermonde matrix
1420:         sv -- singular values of the scaled Vandermonde matrix
1421:         rcond -- value of `rcond`.
1422: 
1423:         For more details, see `linalg.lstsq`.
1424: 
1425:     Warns
1426:     -----
1427:     RankWarning
1428:         The rank of the coefficient matrix in the least-squares fit is
1429:         deficient. The warning is only raised if `full` = False.  The
1430:         warnings can be turned off by
1431: 
1432:         >>> import warnings
1433:         >>> warnings.simplefilter('ignore', RankWarning)
1434: 
1435:     See Also
1436:     --------
1437:     chebfit, legfit, polyfit, hermfit, polyfit
1438:     hermeval : Evaluates a Hermite series.
1439:     hermevander : pseudo Vandermonde matrix of Hermite series.
1440:     hermeweight : HermiteE weight function.
1441:     linalg.lstsq : Computes a least-squares fit from the matrix.
1442:     scipy.interpolate.UnivariateSpline : Computes spline fits.
1443: 
1444:     Notes
1445:     -----
1446:     The solution is the coefficients of the HermiteE series `p` that
1447:     minimizes the sum of the weighted squared errors
1448: 
1449:     .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,
1450: 
1451:     where the :math:`w_j` are the weights. This problem is solved by
1452:     setting up the (typically) overdetermined matrix equation
1453: 
1454:     .. math:: V(x) * c = w * y,
1455: 
1456:     where `V` is the pseudo Vandermonde matrix of `x`, the elements of `c`
1457:     are the coefficients to be solved for, and the elements of `y` are the
1458:     observed values.  This equation is then solved using the singular value
1459:     decomposition of `V`.
1460: 
1461:     If some of the singular values of `V` are so small that they are
1462:     neglected, then a `RankWarning` will be issued. This means that the
1463:     coefficient values may be poorly determined. Using a lower order fit
1464:     will usually get rid of the warning.  The `rcond` parameter can also be
1465:     set to a value smaller than its default, but the resulting fit may be
1466:     spurious and have large contributions from roundoff error.
1467: 
1468:     Fits using HermiteE series are probably most useful when the data can
1469:     be approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the HermiteE
1470:     weight. In that case the weight ``sqrt(w(x[i])`` should be used
1471:     together with data values ``y[i]/sqrt(w(x[i])``. The weight function is
1472:     available as `hermeweight`.
1473: 
1474:     References
1475:     ----------
1476:     .. [1] Wikipedia, "Curve fitting",
1477:            http://en.wikipedia.org/wiki/Curve_fitting
1478: 
1479:     Examples
1480:     --------
1481:     >>> from numpy.polynomial.hermite_e import hermefik, hermeval
1482:     >>> x = np.linspace(-10, 10)
1483:     >>> err = np.random.randn(len(x))/10
1484:     >>> y = hermeval(x, [1, 2, 3]) + err
1485:     >>> hermefit(x, y, 2)
1486:     array([ 1.01690445,  1.99951418,  2.99948696])
1487: 
1488:     '''
1489:     x = np.asarray(x) + 0.0
1490:     y = np.asarray(y) + 0.0
1491:     deg = np.asarray(deg)
1492: 
1493:     # check arguments.
1494:     if deg.ndim > 1 or deg.dtype.kind not in 'iu' or deg.size == 0:
1495:         raise TypeError("deg must be an int or non-empty 1-D array of int")
1496:     if deg.min() < 0:
1497:         raise ValueError("expected deg >= 0")
1498:     if x.ndim != 1:
1499:         raise TypeError("expected 1D vector for x")
1500:     if x.size == 0:
1501:         raise TypeError("expected non-empty vector for x")
1502:     if y.ndim < 1 or y.ndim > 2:
1503:         raise TypeError("expected 1D or 2D array for y")
1504:     if len(x) != len(y):
1505:         raise TypeError("expected x and y to have same length")
1506: 
1507:     if deg.ndim == 0:
1508:         lmax = deg
1509:         order = lmax + 1
1510:         van = hermevander(x, lmax)
1511:     else:
1512:         deg = np.sort(deg)
1513:         lmax = deg[-1]
1514:         order = len(deg)
1515:         van = hermevander(x, lmax)[:, deg]
1516: 
1517:     # set up the least squares matrices in transposed form
1518:     lhs = van.T
1519:     rhs = y.T
1520:     if w is not None:
1521:         w = np.asarray(w) + 0.0
1522:         if w.ndim != 1:
1523:             raise TypeError("expected 1D vector for w")
1524:         if len(x) != len(w):
1525:             raise TypeError("expected x and w to have same length")
1526:         # apply weights. Don't use inplace operations as they
1527:         # can cause problems with NA.
1528:         lhs = lhs * w
1529:         rhs = rhs * w
1530: 
1531:     # set rcond
1532:     if rcond is None:
1533:         rcond = len(x)*np.finfo(x.dtype).eps
1534: 
1535:     # Determine the norms of the design matrix columns.
1536:     if issubclass(lhs.dtype.type, np.complexfloating):
1537:         scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
1538:     else:
1539:         scl = np.sqrt(np.square(lhs).sum(1))
1540:     scl[scl == 0] = 1
1541: 
1542:     # Solve the least squares problem.
1543:     c, resids, rank, s = la.lstsq(lhs.T/scl, rhs.T, rcond)
1544:     c = (c.T/scl).T
1545: 
1546:     # Expand c to include non-fitted coefficients which are set to zero
1547:     if deg.ndim > 0:
1548:         if c.ndim == 2:
1549:             cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
1550:         else:
1551:             cc = np.zeros(lmax+1, dtype=c.dtype)
1552:         cc[deg] = c
1553:         c = cc
1554: 
1555:     # warn on rank reduction
1556:     if rank != order and not full:
1557:         msg = "The fit may be poorly conditioned"
1558:         warnings.warn(msg, pu.RankWarning)
1559: 
1560:     if full:
1561:         return c, [resids, rank, s, rcond]
1562:     else:
1563:         return c
1564: 
1565: 
1566: def hermecompanion(c):
1567:     '''
1568:     Return the scaled companion matrix of c.
1569: 
1570:     The basis polynomials are scaled so that the companion matrix is
1571:     symmetric when `c` is an HermiteE basis polynomial. This provides
1572:     better eigenvalue estimates than the unscaled case and for basis
1573:     polynomials the eigenvalues are guaranteed to be real if
1574:     `numpy.linalg.eigvalsh` is used to obtain them.
1575: 
1576:     Parameters
1577:     ----------
1578:     c : array_like
1579:         1-D array of HermiteE series coefficients ordered from low to high
1580:         degree.
1581: 
1582:     Returns
1583:     -------
1584:     mat : ndarray
1585:         Scaled companion matrix of dimensions (deg, deg).
1586: 
1587:     Notes
1588:     -----
1589: 
1590:     .. versionadded::1.7.0
1591: 
1592:     '''
1593:     # c is a trimmed copy
1594:     [c] = pu.as_series([c])
1595:     if len(c) < 2:
1596:         raise ValueError('Series must have maximum degree of at least 1.')
1597:     if len(c) == 2:
1598:         return np.array([[-c[0]/c[1]]])
1599: 
1600:     n = len(c) - 1
1601:     mat = np.zeros((n, n), dtype=c.dtype)
1602:     scl = np.hstack((1., 1./np.sqrt(np.arange(n - 1, 0, -1))))
1603:     scl = np.multiply.accumulate(scl)[::-1]
1604:     top = mat.reshape(-1)[1::n+1]
1605:     bot = mat.reshape(-1)[n::n+1]
1606:     top[...] = np.sqrt(np.arange(1, n))
1607:     bot[...] = top
1608:     mat[:, -1] -= scl*c[:-1]/c[-1]
1609:     return mat
1610: 
1611: 
1612: def hermeroots(c):
1613:     '''
1614:     Compute the roots of a HermiteE series.
1615: 
1616:     Return the roots (a.k.a. "zeros") of the polynomial
1617: 
1618:     .. math:: p(x) = \\sum_i c[i] * He_i(x).
1619: 
1620:     Parameters
1621:     ----------
1622:     c : 1-D array_like
1623:         1-D array of coefficients.
1624: 
1625:     Returns
1626:     -------
1627:     out : ndarray
1628:         Array of the roots of the series. If all the roots are real,
1629:         then `out` is also real, otherwise it is complex.
1630: 
1631:     See Also
1632:     --------
1633:     polyroots, legroots, lagroots, hermroots, chebroots
1634: 
1635:     Notes
1636:     -----
1637:     The root estimates are obtained as the eigenvalues of the companion
1638:     matrix, Roots far from the origin of the complex plane may have large
1639:     errors due to the numerical instability of the series for such
1640:     values. Roots with multiplicity greater than 1 will also show larger
1641:     errors as the value of the series near such points is relatively
1642:     insensitive to errors in the roots. Isolated roots near the origin can
1643:     be improved by a few iterations of Newton's method.
1644: 
1645:     The HermiteE series basis polynomials aren't powers of `x` so the
1646:     results of this function may seem unintuitive.
1647: 
1648:     Examples
1649:     --------
1650:     >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots
1651:     >>> coef = hermefromroots([-1, 0, 1])
1652:     >>> coef
1653:     array([ 0.,  2.,  0.,  1.])
1654:     >>> hermeroots(coef)
1655:     array([-1.,  0.,  1.])
1656: 
1657:     '''
1658:     # c is a trimmed copy
1659:     [c] = pu.as_series([c])
1660:     if len(c) <= 1:
1661:         return np.array([], dtype=c.dtype)
1662:     if len(c) == 2:
1663:         return np.array([-c[0]/c[1]])
1664: 
1665:     m = hermecompanion(c)
1666:     r = la.eigvals(m)
1667:     r.sort()
1668:     return r
1669: 
1670: 
1671: def _normed_hermite_e_n(x, n):
1672:     '''
1673:     Evaluate a normalized HermiteE polynomial.
1674: 
1675:     Compute the value of the normalized HermiteE polynomial of degree ``n``
1676:     at the points ``x``.
1677: 
1678: 
1679:     Parameters
1680:     ----------
1681:     x : ndarray of double.
1682:         Points at which to evaluate the function
1683:     n : int
1684:         Degree of the normalized HermiteE function to be evaluated.
1685: 
1686:     Returns
1687:     -------
1688:     values : ndarray
1689:         The shape of the return value is described above.
1690: 
1691:     Notes
1692:     -----
1693:     .. versionadded:: 1.10.0
1694: 
1695:     This function is needed for finding the Gauss points and integration
1696:     weights for high degrees. The values of the standard HermiteE functions
1697:     overflow when n >= 207.
1698: 
1699:     '''
1700:     if n == 0:
1701:         return np.ones(x.shape)/np.sqrt(np.sqrt(2*np.pi))
1702: 
1703:     c0 = 0.
1704:     c1 = 1./np.sqrt(np.sqrt(2*np.pi))
1705:     nd = float(n)
1706:     for i in range(n - 1):
1707:         tmp = c0
1708:         c0 = -c1*np.sqrt((nd - 1.)/nd)
1709:         c1 = tmp + c1*x*np.sqrt(1./nd)
1710:         nd = nd - 1.0
1711:     return c0 + c1*x
1712: 
1713: 
1714: def hermegauss(deg):
1715:     '''
1716:     Gauss-HermiteE quadrature.
1717: 
1718:     Computes the sample points and weights for Gauss-HermiteE quadrature.
1719:     These sample points and weights will correctly integrate polynomials of
1720:     degree :math:`2*deg - 1` or less over the interval :math:`[-\inf, \inf]`
1721:     with the weight function :math:`f(x) = \exp(-x^2/2)`.
1722: 
1723:     Parameters
1724:     ----------
1725:     deg : int
1726:         Number of sample points and weights. It must be >= 1.
1727: 
1728:     Returns
1729:     -------
1730:     x : ndarray
1731:         1-D ndarray containing the sample points.
1732:     y : ndarray
1733:         1-D ndarray containing the weights.
1734: 
1735:     Notes
1736:     -----
1737: 
1738:     .. versionadded::1.7.0
1739: 
1740:     The results have only been tested up to degree 100, higher degrees may
1741:     be problematic. The weights are determined by using the fact that
1742: 
1743:     .. math:: w_k = c / (He'_n(x_k) * He_{n-1}(x_k))
1744: 
1745:     where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
1746:     is the k'th root of :math:`He_n`, and then scaling the results to get
1747:     the right value when integrating 1.
1748: 
1749:     '''
1750:     ideg = int(deg)
1751:     if ideg != deg or ideg < 1:
1752:         raise ValueError("deg must be a non-negative integer")
1753: 
1754:     # first approximation of roots. We use the fact that the companion
1755:     # matrix is symmetric in this case in order to obtain better zeros.
1756:     c = np.array([0]*deg + [1])
1757:     m = hermecompanion(c)
1758:     x = la.eigvalsh(m)
1759: 
1760:     # improve roots by one application of Newton
1761:     dy = _normed_hermite_e_n(x, ideg)
1762:     df = _normed_hermite_e_n(x, ideg - 1) * np.sqrt(ideg)
1763:     x -= dy/df
1764: 
1765:     # compute the weights. We scale the factor to avoid possible numerical
1766:     # overflow.
1767:     fm = _normed_hermite_e_n(x, ideg - 1)
1768:     fm /= np.abs(fm).max()
1769:     w = 1/(fm * fm)
1770: 
1771:     # for Hermite_e we can also symmetrize
1772:     w = (w + w[::-1])/2
1773:     x = (x - x[::-1])/2
1774: 
1775:     # scale w to get the right value
1776:     w *= np.sqrt(2*np.pi) / w.sum()
1777: 
1778:     return x, w
1779: 
1780: 
1781: def hermeweight(x):
1782:     '''Weight function of the Hermite_e polynomials.
1783: 
1784:     The weight function is :math:`\exp(-x^2/2)` and the interval of
1785:     integration is :math:`[-\inf, \inf]`. the HermiteE polynomials are
1786:     orthogonal, but not normalized, with respect to this weight function.
1787: 
1788:     Parameters
1789:     ----------
1790:     x : array_like
1791:        Values at which the weight function will be computed.
1792: 
1793:     Returns
1794:     -------
1795:     w : ndarray
1796:        The weight function at `x`.
1797: 
1798:     Notes
1799:     -----
1800: 
1801:     .. versionadded::1.7.0
1802: 
1803:     '''
1804:     w = np.exp(-.5*x**2)
1805:     return w
1806: 
1807: 
1808: #
1809: # HermiteE series class
1810: #
1811: 
1812: class HermiteE(ABCPolyBase):
1813:     '''An HermiteE series class.
1814: 
1815:     The HermiteE class provides the standard Python numerical methods
1816:     '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
1817:     attributes and methods listed in the `ABCPolyBase` documentation.
1818: 
1819:     Parameters
1820:     ----------
1821:     coef : array_like
1822:         HermiteE coefficients in order of increasing degree, i.e,
1823:         ``(1, 2, 3)`` gives ``1*He_0(x) + 2*He_1(X) + 3*He_2(x)``.
1824:     domain : (2,) array_like, optional
1825:         Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
1826:         to the interval ``[window[0], window[1]]`` by shifting and scaling.
1827:         The default value is [-1, 1].
1828:     window : (2,) array_like, optional
1829:         Window, see `domain` for its use. The default value is [-1, 1].
1830: 
1831:         .. versionadded:: 1.6.0
1832: 
1833:     '''
1834:     # Virtual Functions
1835:     _add = staticmethod(hermeadd)
1836:     _sub = staticmethod(hermesub)
1837:     _mul = staticmethod(hermemul)
1838:     _div = staticmethod(hermediv)
1839:     _pow = staticmethod(hermepow)
1840:     _val = staticmethod(hermeval)
1841:     _int = staticmethod(hermeint)
1842:     _der = staticmethod(hermeder)
1843:     _fit = staticmethod(hermefit)
1844:     _line = staticmethod(hermeline)
1845:     _roots = staticmethod(hermeroots)
1846:     _fromroots = staticmethod(hermefromroots)
1847: 
1848:     # Virtual properties
1849:     nickname = 'herme'
1850:     domain = np.array(hermedomain)
1851:     window = np.array(hermedomain)
1852: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_167967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\nObjects for dealing with Hermite_e series.\n\nThis module provides a number of objects (mostly functions) useful for\ndealing with Hermite_e series, including a `HermiteE` class that\nencapsulates the usual arithmetic operations.  (General information\non how this module represents and works with such polynomials is in the\ndocstring for its "parent" sub-package, `numpy.polynomial`).\n\nConstants\n---------\n- `hermedomain` -- Hermite_e series default domain, [-1,1].\n- `hermezero` -- Hermite_e series that evaluates identically to 0.\n- `hermeone` -- Hermite_e series that evaluates identically to 1.\n- `hermex` -- Hermite_e series for the identity map, ``f(x) = x``.\n\nArithmetic\n----------\n- `hermemulx` -- multiply a Hermite_e series in ``P_i(x)`` by ``x``.\n- `hermeadd` -- add two Hermite_e series.\n- `hermesub` -- subtract one Hermite_e series from another.\n- `hermemul` -- multiply two Hermite_e series.\n- `hermediv` -- divide one Hermite_e series by another.\n- `hermeval` -- evaluate a Hermite_e series at given points.\n- `hermeval2d` -- evaluate a 2D Hermite_e series at given points.\n- `hermeval3d` -- evaluate a 3D Hermite_e series at given points.\n- `hermegrid2d` -- evaluate a 2D Hermite_e series on a Cartesian product.\n- `hermegrid3d` -- evaluate a 3D Hermite_e series on a Cartesian product.\n\nCalculus\n--------\n- `hermeder` -- differentiate a Hermite_e series.\n- `hermeint` -- integrate a Hermite_e series.\n\nMisc Functions\n--------------\n- `hermefromroots` -- create a Hermite_e series with specified roots.\n- `hermeroots` -- find the roots of a Hermite_e series.\n- `hermevander` -- Vandermonde-like matrix for Hermite_e polynomials.\n- `hermevander2d` -- Vandermonde-like matrix for 2D power series.\n- `hermevander3d` -- Vandermonde-like matrix for 3D power series.\n- `hermegauss` -- Gauss-Hermite_e quadrature, points and weights.\n- `hermeweight` -- Hermite_e weight function.\n- `hermecompanion` -- symmetrized companion matrix in Hermite_e form.\n- `hermefit` -- least-squares fit returning a Hermite_e series.\n- `hermetrim` -- trim leading coefficients from a Hermite_e series.\n- `hermeline` -- Hermite_e series of given straight line.\n- `herme2poly` -- convert a Hermite_e series to a polynomial.\n- `poly2herme` -- convert a polynomial to a Hermite_e series.\n\nClasses\n-------\n- `HermiteE` -- A Hermite_e series class.\n\nSee also\n--------\n`numpy.polynomial`\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 0))

# 'import warnings' statement (line 62)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 62, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 63, 0))

# 'import numpy' statement (line 63)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_167968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy')

if (type(import_167968) is not StypyTypeError):

    if (import_167968 != 'pyd_module'):
        __import__(import_167968)
        sys_modules_167969 = sys.modules[import_167968]
        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', sys_modules_167969.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 63, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'numpy', import_167968)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 0))

# 'import numpy.linalg' statement (line 64)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_167970 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg')

if (type(import_167970) is not StypyTypeError):

    if (import_167970 != 'pyd_module'):
        __import__(import_167970)
        sys_modules_167971 = sys.modules[import_167970]
        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', sys_modules_167971.module_type_store, module_type_store)
    else:
        import numpy.linalg as la

        import_module(stypy.reporting.localization.Localization(__file__, 64, 0), 'la', numpy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'numpy.linalg' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'numpy.linalg', import_167970)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 66, 0))

# 'from numpy.polynomial import pu' statement (line 66)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_167972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial')

if (type(import_167972) is not StypyTypeError):

    if (import_167972 != 'pyd_module'):
        __import__(import_167972)
        sys_modules_167973 = sys.modules[import_167972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', sys_modules_167973.module_type_store, module_type_store, ['polyutils'])
        nest_module(stypy.reporting.localization.Localization(__file__, 66, 0), __file__, sys_modules_167973, sys_modules_167973.module_type_store, module_type_store)
    else:
        from numpy.polynomial import polyutils as pu

        import_from_module(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', None, module_type_store, ['polyutils'], [pu])

else:
    # Assigning a type to the variable 'numpy.polynomial' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'numpy.polynomial', import_167972)

# Adding an alias
module_type_store.add_alias('pu', 'polyutils')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 67, 0))

# 'from numpy.polynomial._polybase import ABCPolyBase' statement (line 67)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
import_167974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase')

if (type(import_167974) is not StypyTypeError):

    if (import_167974 != 'pyd_module'):
        __import__(import_167974)
        sys_modules_167975 = sys.modules[import_167974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', sys_modules_167975.module_type_store, module_type_store, ['ABCPolyBase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 67, 0), __file__, sys_modules_167975, sys_modules_167975.module_type_store, module_type_store)
    else:
        from numpy.polynomial._polybase import ABCPolyBase

        import_from_module(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', None, module_type_store, ['ABCPolyBase'], [ABCPolyBase])

else:
    # Assigning a type to the variable 'numpy.polynomial._polybase' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'numpy.polynomial._polybase', import_167974)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')


# Assigning a List to a Name (line 69):

# Assigning a List to a Name (line 69):
__all__ = ['hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline', 'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv', 'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly', 'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim', 'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d', 'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion', 'hermegauss', 'hermeweight']
module_type_store.set_exportable_members(['hermezero', 'hermeone', 'hermex', 'hermedomain', 'hermeline', 'hermeadd', 'hermesub', 'hermemulx', 'hermemul', 'hermediv', 'hermepow', 'hermeval', 'hermeder', 'hermeint', 'herme2poly', 'poly2herme', 'hermefromroots', 'hermevander', 'hermefit', 'hermetrim', 'hermeroots', 'HermiteE', 'hermeval2d', 'hermeval3d', 'hermegrid2d', 'hermegrid3d', 'hermevander2d', 'hermevander3d', 'hermecompanion', 'hermegauss', 'hermeweight'])

# Obtaining an instance of the builtin type 'list' (line 69)
list_167976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
str_167977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'hermezero')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167977)
# Adding element type (line 69)
str_167978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'str', 'hermeone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167978)
# Adding element type (line 69)
str_167979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'str', 'hermex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167979)
# Adding element type (line 69)
str_167980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'str', 'hermedomain')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167980)
# Adding element type (line 69)
str_167981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 54), 'str', 'hermeline')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167981)
# Adding element type (line 69)
str_167982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'hermeadd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167982)
# Adding element type (line 69)
str_167983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'str', 'hermesub')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167983)
# Adding element type (line 69)
str_167984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', 'hermemulx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167984)
# Adding element type (line 69)
str_167985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'str', 'hermemul')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167985)
# Adding element type (line 69)
str_167986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 53), 'str', 'hermediv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167986)
# Adding element type (line 69)
str_167987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'hermepow')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167987)
# Adding element type (line 69)
str_167988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 16), 'str', 'hermeval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167988)
# Adding element type (line 69)
str_167989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'str', 'hermeder')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167989)
# Adding element type (line 69)
str_167990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 40), 'str', 'hermeint')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167990)
# Adding element type (line 69)
str_167991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'str', 'herme2poly')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167991)
# Adding element type (line 69)
str_167992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'poly2herme')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167992)
# Adding element type (line 69)
str_167993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'str', 'hermefromroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167993)
# Adding element type (line 69)
str_167994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'str', 'hermevander')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167994)
# Adding element type (line 69)
str_167995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 51), 'str', 'hermefit')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167995)
# Adding element type (line 69)
str_167996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 63), 'str', 'hermetrim')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167996)
# Adding element type (line 69)
str_167997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'hermeroots')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167997)
# Adding element type (line 69)
str_167998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'str', 'HermiteE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167998)
# Adding element type (line 69)
str_167999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', 'hermeval2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_167999)
# Adding element type (line 69)
str_168000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 44), 'str', 'hermeval3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168000)
# Adding element type (line 69)
str_168001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 58), 'str', 'hermegrid2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168001)
# Adding element type (line 69)
str_168002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'hermegrid3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168002)
# Adding element type (line 69)
str_168003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'str', 'hermevander2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168003)
# Adding element type (line 69)
str_168004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'str', 'hermevander3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168004)
# Adding element type (line 69)
str_168005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 53), 'str', 'hermecompanion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168005)
# Adding element type (line 69)
str_168006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'hermegauss')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168006)
# Adding element type (line 69)
str_168007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'str', 'hermeweight')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 10), list_167976, str_168007)

# Assigning a type to the variable '__all__' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), '__all__', list_167976)

# Assigning a Attribute to a Name (line 78):

# Assigning a Attribute to a Name (line 78):
# Getting the type of 'pu' (line 78)
pu_168008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'pu')
# Obtaining the member 'trimcoef' of a type (line 78)
trimcoef_168009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), pu_168008, 'trimcoef')
# Assigning a type to the variable 'hermetrim' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'hermetrim', trimcoef_168009)

@norecursion
def poly2herme(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'poly2herme'
    module_type_store = module_type_store.open_function_context('poly2herme', 81, 0, False)
    
    # Passed parameters checking function
    poly2herme.stypy_localization = localization
    poly2herme.stypy_type_of_self = None
    poly2herme.stypy_type_store = module_type_store
    poly2herme.stypy_function_name = 'poly2herme'
    poly2herme.stypy_param_names_list = ['pol']
    poly2herme.stypy_varargs_param_name = None
    poly2herme.stypy_kwargs_param_name = None
    poly2herme.stypy_call_defaults = defaults
    poly2herme.stypy_call_varargs = varargs
    poly2herme.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'poly2herme', ['pol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'poly2herme', localization, ['pol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'poly2herme(...)' code ##################

    str_168010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', '\n    poly2herme(pol)\n\n    Convert a polynomial to a Hermite series.\n\n    Convert an array representing the coefficients of a polynomial (relative\n    to the "standard" basis) ordered from lowest degree to highest, to an\n    array of the coefficients of the equivalent Hermite series, ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    pol : array_like\n        1-D array containing the polynomial coefficients\n\n    Returns\n    -------\n    c : ndarray\n        1-D array containing the coefficients of the equivalent Hermite\n        series.\n\n    See Also\n    --------\n    herme2poly\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import poly2herme\n    >>> poly2herme(np.arange(4))\n    array([  2.,  10.,   2.,   3.])\n\n    ')
    
    # Assigning a Call to a List (line 119):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_168013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    # Getting the type of 'pol' (line 119)
    pol_168014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 26), 'pol', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 25), list_168013, pol_168014)
    
    # Processing the call keyword arguments (line 119)
    kwargs_168015 = {}
    # Getting the type of 'pu' (line 119)
    pu_168011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 119)
    as_series_168012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), pu_168011, 'as_series')
    # Calling as_series(args, kwargs) (line 119)
    as_series_call_result_168016 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), as_series_168012, *[list_168013], **kwargs_168015)
    
    # Assigning a type to the variable 'call_assignment_167912' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'call_assignment_167912', as_series_call_result_168016)
    
    # Assigning a Call to a Name (line 119):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168020 = {}
    # Getting the type of 'call_assignment_167912' (line 119)
    call_assignment_167912_168017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'call_assignment_167912', False)
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___168018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), call_assignment_167912_168017, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168021 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168018, *[int_168019], **kwargs_168020)
    
    # Assigning a type to the variable 'call_assignment_167913' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'call_assignment_167913', getitem___call_result_168021)
    
    # Assigning a Name to a Name (line 119):
    # Getting the type of 'call_assignment_167913' (line 119)
    call_assignment_167913_168022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'call_assignment_167913')
    # Assigning a type to the variable 'pol' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 5), 'pol', call_assignment_167913_168022)
    
    # Assigning a BinOp to a Name (line 120):
    
    # Assigning a BinOp to a Name (line 120):
    
    # Call to len(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'pol' (line 120)
    pol_168024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'pol', False)
    # Processing the call keyword arguments (line 120)
    kwargs_168025 = {}
    # Getting the type of 'len' (line 120)
    len_168023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 10), 'len', False)
    # Calling len(args, kwargs) (line 120)
    len_call_result_168026 = invoke(stypy.reporting.localization.Localization(__file__, 120, 10), len_168023, *[pol_168024], **kwargs_168025)
    
    int_168027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
    # Applying the binary operator '-' (line 120)
    result_sub_168028 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 10), '-', len_call_result_168026, int_168027)
    
    # Assigning a type to the variable 'deg' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'deg', result_sub_168028)
    
    # Assigning a Num to a Name (line 121):
    
    # Assigning a Num to a Name (line 121):
    int_168029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 10), 'int')
    # Assigning a type to the variable 'res' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'res', int_168029)
    
    
    # Call to range(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'deg' (line 122)
    deg_168031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'deg', False)
    int_168032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 24), 'int')
    int_168033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'int')
    # Processing the call keyword arguments (line 122)
    kwargs_168034 = {}
    # Getting the type of 'range' (line 122)
    range_168030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 13), 'range', False)
    # Calling range(args, kwargs) (line 122)
    range_call_result_168035 = invoke(stypy.reporting.localization.Localization(__file__, 122, 13), range_168030, *[deg_168031, int_168032, int_168033], **kwargs_168034)
    
    # Testing the type of a for loop iterable (line 122)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 4), range_call_result_168035)
    # Getting the type of the for loop variable (line 122)
    for_loop_var_168036 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 4), range_call_result_168035)
    # Assigning a type to the variable 'i' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'i', for_loop_var_168036)
    # SSA begins for a for statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to hermeadd(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Call to hermemulx(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'res' (line 123)
    res_168039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 33), 'res', False)
    # Processing the call keyword arguments (line 123)
    kwargs_168040 = {}
    # Getting the type of 'hermemulx' (line 123)
    hermemulx_168038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'hermemulx', False)
    # Calling hermemulx(args, kwargs) (line 123)
    hermemulx_call_result_168041 = invoke(stypy.reporting.localization.Localization(__file__, 123, 23), hermemulx_168038, *[res_168039], **kwargs_168040)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 123)
    i_168042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'i', False)
    # Getting the type of 'pol' (line 123)
    pol_168043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'pol', False)
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___168044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 39), pol_168043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_168045 = invoke(stypy.reporting.localization.Localization(__file__, 123, 39), getitem___168044, i_168042)
    
    # Processing the call keyword arguments (line 123)
    kwargs_168046 = {}
    # Getting the type of 'hermeadd' (line 123)
    hermeadd_168037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'hermeadd', False)
    # Calling hermeadd(args, kwargs) (line 123)
    hermeadd_call_result_168047 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), hermeadd_168037, *[hermemulx_call_result_168041, subscript_call_result_168045], **kwargs_168046)
    
    # Assigning a type to the variable 'res' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'res', hermeadd_call_result_168047)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 124)
    res_168048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type', res_168048)
    
    # ################# End of 'poly2herme(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'poly2herme' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_168049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'poly2herme'
    return stypy_return_type_168049

# Assigning a type to the variable 'poly2herme' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'poly2herme', poly2herme)

@norecursion
def herme2poly(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'herme2poly'
    module_type_store = module_type_store.open_function_context('herme2poly', 127, 0, False)
    
    # Passed parameters checking function
    herme2poly.stypy_localization = localization
    herme2poly.stypy_type_of_self = None
    herme2poly.stypy_type_store = module_type_store
    herme2poly.stypy_function_name = 'herme2poly'
    herme2poly.stypy_param_names_list = ['c']
    herme2poly.stypy_varargs_param_name = None
    herme2poly.stypy_kwargs_param_name = None
    herme2poly.stypy_call_defaults = defaults
    herme2poly.stypy_call_varargs = varargs
    herme2poly.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'herme2poly', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'herme2poly', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'herme2poly(...)' code ##################

    str_168050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', '\n    Convert a Hermite series to a polynomial.\n\n    Convert an array representing the coefficients of a Hermite series,\n    ordered from lowest degree to highest, to an array of the coefficients\n    of the equivalent polynomial (relative to the "standard" basis) ordered\n    from lowest to highest degree.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array containing the Hermite series coefficients, ordered\n        from lowest order term to highest.\n\n    Returns\n    -------\n    pol : ndarray\n        1-D array containing the coefficients of the equivalent polynomial\n        (relative to the "standard" basis) ordered from lowest order term\n        to highest.\n\n    See Also\n    --------\n    poly2herme\n\n    Notes\n    -----\n    The easy way to do conversions between polynomial basis sets\n    is to use the convert method of a class instance.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import herme2poly\n    >>> herme2poly([  2.,  10.,   2.,   3.])\n    array([ 0.,  1.,  2.,  3.])\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 165, 4))
    
    # 'from numpy.polynomial.polynomial import polyadd, polysub, polymulx' statement (line 165)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/polynomial/')
    import_168051 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 165, 4), 'numpy.polynomial.polynomial')

    if (type(import_168051) is not StypyTypeError):

        if (import_168051 != 'pyd_module'):
            __import__(import_168051)
            sys_modules_168052 = sys.modules[import_168051]
            import_from_module(stypy.reporting.localization.Localization(__file__, 165, 4), 'numpy.polynomial.polynomial', sys_modules_168052.module_type_store, module_type_store, ['polyadd', 'polysub', 'polymulx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 165, 4), __file__, sys_modules_168052, sys_modules_168052.module_type_store, module_type_store)
        else:
            from numpy.polynomial.polynomial import polyadd, polysub, polymulx

            import_from_module(stypy.reporting.localization.Localization(__file__, 165, 4), 'numpy.polynomial.polynomial', None, module_type_store, ['polyadd', 'polysub', 'polymulx'], [polyadd, polysub, polymulx])

    else:
        # Assigning a type to the variable 'numpy.polynomial.polynomial' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'numpy.polynomial.polynomial', import_168051)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/polynomial/')
    
    
    # Assigning a Call to a List (line 167):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_168055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    # Getting the type of 'c' (line 167)
    c_168056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), list_168055, c_168056)
    
    # Processing the call keyword arguments (line 167)
    kwargs_168057 = {}
    # Getting the type of 'pu' (line 167)
    pu_168053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 167)
    as_series_168054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 10), pu_168053, 'as_series')
    # Calling as_series(args, kwargs) (line 167)
    as_series_call_result_168058 = invoke(stypy.reporting.localization.Localization(__file__, 167, 10), as_series_168054, *[list_168055], **kwargs_168057)
    
    # Assigning a type to the variable 'call_assignment_167914' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'call_assignment_167914', as_series_call_result_168058)
    
    # Assigning a Call to a Name (line 167):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168062 = {}
    # Getting the type of 'call_assignment_167914' (line 167)
    call_assignment_167914_168059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'call_assignment_167914', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___168060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 4), call_assignment_167914_168059, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168063 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168060, *[int_168061], **kwargs_168062)
    
    # Assigning a type to the variable 'call_assignment_167915' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'call_assignment_167915', getitem___call_result_168063)
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'call_assignment_167915' (line 167)
    call_assignment_167915_168064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'call_assignment_167915')
    # Assigning a type to the variable 'c' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 5), 'c', call_assignment_167915_168064)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to len(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'c' (line 168)
    c_168066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'c', False)
    # Processing the call keyword arguments (line 168)
    kwargs_168067 = {}
    # Getting the type of 'len' (line 168)
    len_168065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'len', False)
    # Calling len(args, kwargs) (line 168)
    len_call_result_168068 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), len_168065, *[c_168066], **kwargs_168067)
    
    # Assigning a type to the variable 'n' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'n', len_call_result_168068)
    
    
    # Getting the type of 'n' (line 169)
    n_168069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'n')
    int_168070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_168071 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), '==', n_168069, int_168070)
    
    # Testing the type of an if condition (line 169)
    if_condition_168072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_168071)
    # Assigning a type to the variable 'if_condition_168072' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_168072', if_condition_168072)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 170)
    c_168073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', c_168073)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 171)
    n_168074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'n')
    int_168075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 12), 'int')
    # Applying the binary operator '==' (line 171)
    result_eq_168076 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 7), '==', n_168074, int_168075)
    
    # Testing the type of an if condition (line 171)
    if_condition_168077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 4), result_eq_168076)
    # Assigning a type to the variable 'if_condition_168077' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'if_condition_168077', if_condition_168077)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 172)
    c_168078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', c_168078)
    # SSA branch for the else part of an if statement (line 171)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 174):
    
    # Assigning a Subscript to a Name (line 174):
    
    # Obtaining the type of the subscript
    int_168079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'int')
    # Getting the type of 'c' (line 174)
    c_168080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___168081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), c_168080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_168082 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), getitem___168081, int_168079)
    
    # Assigning a type to the variable 'c0' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'c0', subscript_call_result_168082)
    
    # Assigning a Subscript to a Name (line 175):
    
    # Assigning a Subscript to a Name (line 175):
    
    # Obtaining the type of the subscript
    int_168083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 15), 'int')
    # Getting the type of 'c' (line 175)
    c_168084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___168085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), c_168084, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_168086 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), getitem___168085, int_168083)
    
    # Assigning a type to the variable 'c1' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'c1', subscript_call_result_168086)
    
    
    # Call to range(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'n' (line 177)
    n_168088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'n', False)
    int_168089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'int')
    # Applying the binary operator '-' (line 177)
    result_sub_168090 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 23), '-', n_168088, int_168089)
    
    int_168091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'int')
    int_168092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 33), 'int')
    # Processing the call keyword arguments (line 177)
    kwargs_168093 = {}
    # Getting the type of 'range' (line 177)
    range_168087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'range', False)
    # Calling range(args, kwargs) (line 177)
    range_call_result_168094 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), range_168087, *[result_sub_168090, int_168091, int_168092], **kwargs_168093)
    
    # Testing the type of a for loop iterable (line 177)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_168094)
    # Getting the type of the for loop variable (line 177)
    for_loop_var_168095 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_168094)
    # Assigning a type to the variable 'i' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'i', for_loop_var_168095)
    # SSA begins for a for statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 178):
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'c0' (line 178)
    c0_168096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'tmp', c0_168096)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to polysub(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 179)
    i_168098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 27), 'i', False)
    int_168099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 31), 'int')
    # Applying the binary operator '-' (line 179)
    result_sub_168100 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 27), '-', i_168098, int_168099)
    
    # Getting the type of 'c' (line 179)
    c_168101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___168102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 25), c_168101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_168103 = invoke(stypy.reporting.localization.Localization(__file__, 179, 25), getitem___168102, result_sub_168100)
    
    # Getting the type of 'c1' (line 179)
    c1_168104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 35), 'c1', False)
    # Getting the type of 'i' (line 179)
    i_168105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 39), 'i', False)
    int_168106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 43), 'int')
    # Applying the binary operator '-' (line 179)
    result_sub_168107 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 39), '-', i_168105, int_168106)
    
    # Applying the binary operator '*' (line 179)
    result_mul_168108 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 35), '*', c1_168104, result_sub_168107)
    
    # Processing the call keyword arguments (line 179)
    kwargs_168109 = {}
    # Getting the type of 'polysub' (line 179)
    polysub_168097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'polysub', False)
    # Calling polysub(args, kwargs) (line 179)
    polysub_call_result_168110 = invoke(stypy.reporting.localization.Localization(__file__, 179, 17), polysub_168097, *[subscript_call_result_168103, result_mul_168108], **kwargs_168109)
    
    # Assigning a type to the variable 'c0' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'c0', polysub_call_result_168110)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to polyadd(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'tmp' (line 180)
    tmp_168112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'tmp', False)
    
    # Call to polymulx(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'c1' (line 180)
    c1_168114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'c1', False)
    # Processing the call keyword arguments (line 180)
    kwargs_168115 = {}
    # Getting the type of 'polymulx' (line 180)
    polymulx_168113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 180)
    polymulx_call_result_168116 = invoke(stypy.reporting.localization.Localization(__file__, 180, 30), polymulx_168113, *[c1_168114], **kwargs_168115)
    
    # Processing the call keyword arguments (line 180)
    kwargs_168117 = {}
    # Getting the type of 'polyadd' (line 180)
    polyadd_168111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 180)
    polyadd_call_result_168118 = invoke(stypy.reporting.localization.Localization(__file__, 180, 17), polyadd_168111, *[tmp_168112, polymulx_call_result_168116], **kwargs_168117)
    
    # Assigning a type to the variable 'c1' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'c1', polyadd_call_result_168118)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to polyadd(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c0' (line 181)
    c0_168120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 23), 'c0', False)
    
    # Call to polymulx(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'c1' (line 181)
    c1_168122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'c1', False)
    # Processing the call keyword arguments (line 181)
    kwargs_168123 = {}
    # Getting the type of 'polymulx' (line 181)
    polymulx_168121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 27), 'polymulx', False)
    # Calling polymulx(args, kwargs) (line 181)
    polymulx_call_result_168124 = invoke(stypy.reporting.localization.Localization(__file__, 181, 27), polymulx_168121, *[c1_168122], **kwargs_168123)
    
    # Processing the call keyword arguments (line 181)
    kwargs_168125 = {}
    # Getting the type of 'polyadd' (line 181)
    polyadd_168119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'polyadd', False)
    # Calling polyadd(args, kwargs) (line 181)
    polyadd_call_result_168126 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), polyadd_168119, *[c0_168120, polymulx_call_result_168124], **kwargs_168125)
    
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', polyadd_call_result_168126)
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'herme2poly(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'herme2poly' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_168127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168127)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'herme2poly'
    return stypy_return_type_168127

# Assigning a type to the variable 'herme2poly' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'herme2poly', herme2poly)

# Assigning a Call to a Name (line 189):

# Assigning a Call to a Name (line 189):

# Call to array(...): (line 189)
# Processing the call arguments (line 189)

# Obtaining an instance of the builtin type 'list' (line 189)
list_168130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 189)
# Adding element type (line 189)
int_168131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 23), list_168130, int_168131)
# Adding element type (line 189)
int_168132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 23), list_168130, int_168132)

# Processing the call keyword arguments (line 189)
kwargs_168133 = {}
# Getting the type of 'np' (line 189)
np_168128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), 'np', False)
# Obtaining the member 'array' of a type (line 189)
array_168129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 14), np_168128, 'array')
# Calling array(args, kwargs) (line 189)
array_call_result_168134 = invoke(stypy.reporting.localization.Localization(__file__, 189, 14), array_168129, *[list_168130], **kwargs_168133)

# Assigning a type to the variable 'hermedomain' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'hermedomain', array_call_result_168134)

# Assigning a Call to a Name (line 192):

# Assigning a Call to a Name (line 192):

# Call to array(...): (line 192)
# Processing the call arguments (line 192)

# Obtaining an instance of the builtin type 'list' (line 192)
list_168137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 192)
# Adding element type (line 192)
int_168138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 21), list_168137, int_168138)

# Processing the call keyword arguments (line 192)
kwargs_168139 = {}
# Getting the type of 'np' (line 192)
np_168135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'np', False)
# Obtaining the member 'array' of a type (line 192)
array_168136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), np_168135, 'array')
# Calling array(args, kwargs) (line 192)
array_call_result_168140 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), array_168136, *[list_168137], **kwargs_168139)

# Assigning a type to the variable 'hermezero' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'hermezero', array_call_result_168140)

# Assigning a Call to a Name (line 195):

# Assigning a Call to a Name (line 195):

# Call to array(...): (line 195)
# Processing the call arguments (line 195)

# Obtaining an instance of the builtin type 'list' (line 195)
list_168143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 195)
# Adding element type (line 195)
int_168144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 20), list_168143, int_168144)

# Processing the call keyword arguments (line 195)
kwargs_168145 = {}
# Getting the type of 'np' (line 195)
np_168141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'np', False)
# Obtaining the member 'array' of a type (line 195)
array_168142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 11), np_168141, 'array')
# Calling array(args, kwargs) (line 195)
array_call_result_168146 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), array_168142, *[list_168143], **kwargs_168145)

# Assigning a type to the variable 'hermeone' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'hermeone', array_call_result_168146)

# Assigning a Call to a Name (line 198):

# Assigning a Call to a Name (line 198):

# Call to array(...): (line 198)
# Processing the call arguments (line 198)

# Obtaining an instance of the builtin type 'list' (line 198)
list_168149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 198)
# Adding element type (line 198)
int_168150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_168149, int_168150)
# Adding element type (line 198)
int_168151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 18), list_168149, int_168151)

# Processing the call keyword arguments (line 198)
kwargs_168152 = {}
# Getting the type of 'np' (line 198)
np_168147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 9), 'np', False)
# Obtaining the member 'array' of a type (line 198)
array_168148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 9), np_168147, 'array')
# Calling array(args, kwargs) (line 198)
array_call_result_168153 = invoke(stypy.reporting.localization.Localization(__file__, 198, 9), array_168148, *[list_168149], **kwargs_168152)

# Assigning a type to the variable 'hermex' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'hermex', array_call_result_168153)

@norecursion
def hermeline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeline'
    module_type_store = module_type_store.open_function_context('hermeline', 201, 0, False)
    
    # Passed parameters checking function
    hermeline.stypy_localization = localization
    hermeline.stypy_type_of_self = None
    hermeline.stypy_type_store = module_type_store
    hermeline.stypy_function_name = 'hermeline'
    hermeline.stypy_param_names_list = ['off', 'scl']
    hermeline.stypy_varargs_param_name = None
    hermeline.stypy_kwargs_param_name = None
    hermeline.stypy_call_defaults = defaults
    hermeline.stypy_call_varargs = varargs
    hermeline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeline', ['off', 'scl'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeline', localization, ['off', 'scl'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeline(...)' code ##################

    str_168154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'str', "\n    Hermite series whose graph is a straight line.\n\n\n\n    Parameters\n    ----------\n    off, scl : scalars\n        The specified line is given by ``off + scl*x``.\n\n    Returns\n    -------\n    y : ndarray\n        This module's representation of the Hermite series for\n        ``off + scl*x``.\n\n    See Also\n    --------\n    polyline, chebline\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeline\n    >>> from numpy.polynomial.hermite_e import hermeline, hermeval\n    >>> hermeval(0,hermeline(3, 2))\n    3.0\n    >>> hermeval(1,hermeline(3, 2))\n    5.0\n\n    ")
    
    
    # Getting the type of 'scl' (line 232)
    scl_168155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 7), 'scl')
    int_168156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 14), 'int')
    # Applying the binary operator '!=' (line 232)
    result_ne_168157 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 7), '!=', scl_168155, int_168156)
    
    # Testing the type of an if condition (line 232)
    if_condition_168158 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 4), result_ne_168157)
    # Assigning a type to the variable 'if_condition_168158' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'if_condition_168158', if_condition_168158)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 233)
    # Processing the call arguments (line 233)
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_168161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    # Getting the type of 'off' (line 233)
    off_168162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 24), list_168161, off_168162)
    # Adding element type (line 233)
    # Getting the type of 'scl' (line 233)
    scl_168163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'scl', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 24), list_168161, scl_168163)
    
    # Processing the call keyword arguments (line 233)
    kwargs_168164 = {}
    # Getting the type of 'np' (line 233)
    np_168159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 233)
    array_168160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 15), np_168159, 'array')
    # Calling array(args, kwargs) (line 233)
    array_call_result_168165 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), array_168160, *[list_168161], **kwargs_168164)
    
    # Assigning a type to the variable 'stypy_return_type' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'stypy_return_type', array_call_result_168165)
    # SSA branch for the else part of an if statement (line 232)
    module_type_store.open_ssa_branch('else')
    
    # Call to array(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'list' (line 235)
    list_168168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'off' (line 235)
    off_168169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 25), 'off', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 24), list_168168, off_168169)
    
    # Processing the call keyword arguments (line 235)
    kwargs_168170 = {}
    # Getting the type of 'np' (line 235)
    np_168166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 235)
    array_168167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), np_168166, 'array')
    # Calling array(args, kwargs) (line 235)
    array_call_result_168171 = invoke(stypy.reporting.localization.Localization(__file__, 235, 15), array_168167, *[list_168168], **kwargs_168170)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'stypy_return_type', array_call_result_168171)
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermeline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeline' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_168172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168172)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeline'
    return stypy_return_type_168172

# Assigning a type to the variable 'hermeline' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'hermeline', hermeline)

@norecursion
def hermefromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermefromroots'
    module_type_store = module_type_store.open_function_context('hermefromroots', 238, 0, False)
    
    # Passed parameters checking function
    hermefromroots.stypy_localization = localization
    hermefromroots.stypy_type_of_self = None
    hermefromroots.stypy_type_store = module_type_store
    hermefromroots.stypy_function_name = 'hermefromroots'
    hermefromroots.stypy_param_names_list = ['roots']
    hermefromroots.stypy_varargs_param_name = None
    hermefromroots.stypy_kwargs_param_name = None
    hermefromroots.stypy_call_defaults = defaults
    hermefromroots.stypy_call_varargs = varargs
    hermefromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermefromroots', ['roots'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermefromroots', localization, ['roots'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermefromroots(...)' code ##################

    str_168173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, (-1)), 'str', '\n    Generate a HermiteE series with given roots.\n\n    The function returns the coefficients of the polynomial\n\n    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),\n\n    in HermiteE form, where the `r_n` are the roots specified in `roots`.\n    If a zero has multiplicity n, then it must appear in `roots` n times.\n    For instance, if 2 is a root of multiplicity three and 3 is a root of\n    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The\n    roots can appear in any order.\n\n    If the returned coefficients are `c`, then\n\n    .. math:: p(x) = c_0 + c_1 * He_1(x) + ... +  c_n * He_n(x)\n\n    The coefficient of the last term is not generally 1 for monic\n    polynomials in HermiteE form.\n\n    Parameters\n    ----------\n    roots : array_like\n        Sequence containing the roots.\n\n    Returns\n    -------\n    out : ndarray\n        1-D array of coefficients.  If all roots are real then `out` is a\n        real array, if some of the roots are complex, then `out` is complex\n        even if all the coefficients in the result are real (see Examples\n        below).\n\n    See Also\n    --------\n    polyfromroots, legfromroots, lagfromroots, hermfromroots,\n    chebfromroots.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermefromroots, hermeval\n    >>> coef = hermefromroots((-1, 0, 1))\n    >>> hermeval((-1, 0, 1), coef)\n    array([ 0.,  0.,  0.])\n    >>> coef = hermefromroots((-1j, 1j))\n    >>> hermeval((-1j, 1j), coef)\n    array([ 0.+0.j,  0.+0.j])\n\n    ')
    
    
    
    # Call to len(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 'roots' (line 288)
    roots_168175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'roots', False)
    # Processing the call keyword arguments (line 288)
    kwargs_168176 = {}
    # Getting the type of 'len' (line 288)
    len_168174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'len', False)
    # Calling len(args, kwargs) (line 288)
    len_call_result_168177 = invoke(stypy.reporting.localization.Localization(__file__, 288, 7), len_168174, *[roots_168175], **kwargs_168176)
    
    int_168178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'int')
    # Applying the binary operator '==' (line 288)
    result_eq_168179 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 7), '==', len_call_result_168177, int_168178)
    
    # Testing the type of an if condition (line 288)
    if_condition_168180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 4), result_eq_168179)
    # Assigning a type to the variable 'if_condition_168180' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'if_condition_168180', if_condition_168180)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 289)
    # Processing the call arguments (line 289)
    int_168183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 23), 'int')
    # Processing the call keyword arguments (line 289)
    kwargs_168184 = {}
    # Getting the type of 'np' (line 289)
    np_168181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 289)
    ones_168182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 15), np_168181, 'ones')
    # Calling ones(args, kwargs) (line 289)
    ones_call_result_168185 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), ones_168182, *[int_168183], **kwargs_168184)
    
    # Assigning a type to the variable 'stypy_return_type' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', ones_call_result_168185)
    # SSA branch for the else part of an if statement (line 288)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a List (line 291):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Obtaining an instance of the builtin type 'list' (line 291)
    list_168188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'roots' (line 291)
    roots_168189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 32), 'roots', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 31), list_168188, roots_168189)
    
    # Processing the call keyword arguments (line 291)
    # Getting the type of 'False' (line 291)
    False_168190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 45), 'False', False)
    keyword_168191 = False_168190
    kwargs_168192 = {'trim': keyword_168191}
    # Getting the type of 'pu' (line 291)
    pu_168186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 18), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 291)
    as_series_168187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 18), pu_168186, 'as_series')
    # Calling as_series(args, kwargs) (line 291)
    as_series_call_result_168193 = invoke(stypy.reporting.localization.Localization(__file__, 291, 18), as_series_168187, *[list_168188], **kwargs_168192)
    
    # Assigning a type to the variable 'call_assignment_167916' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_167916', as_series_call_result_168193)
    
    # Assigning a Call to a Name (line 291):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 8), 'int')
    # Processing the call keyword arguments
    kwargs_168197 = {}
    # Getting the type of 'call_assignment_167916' (line 291)
    call_assignment_167916_168194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_167916', False)
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___168195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), call_assignment_167916_168194, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168198 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168195, *[int_168196], **kwargs_168197)
    
    # Assigning a type to the variable 'call_assignment_167917' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_167917', getitem___call_result_168198)
    
    # Assigning a Name to a Name (line 291):
    # Getting the type of 'call_assignment_167917' (line 291)
    call_assignment_167917_168199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'call_assignment_167917')
    # Assigning a type to the variable 'roots' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 9), 'roots', call_assignment_167917_168199)
    
    # Call to sort(...): (line 292)
    # Processing the call keyword arguments (line 292)
    kwargs_168202 = {}
    # Getting the type of 'roots' (line 292)
    roots_168200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'roots', False)
    # Obtaining the member 'sort' of a type (line 292)
    sort_168201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), roots_168200, 'sort')
    # Calling sort(args, kwargs) (line 292)
    sort_call_result_168203 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), sort_168201, *[], **kwargs_168202)
    
    
    # Assigning a ListComp to a Name (line 293):
    
    # Assigning a ListComp to a Name (line 293):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'roots' (line 293)
    roots_168210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 39), 'roots')
    comprehension_168211 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 13), roots_168210)
    # Assigning a type to the variable 'r' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'r', comprehension_168211)
    
    # Call to hermeline(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Getting the type of 'r' (line 293)
    r_168205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'r', False)
    # Applying the 'usub' unary operator (line 293)
    result___neg___168206 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 23), 'usub', r_168205)
    
    int_168207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 27), 'int')
    # Processing the call keyword arguments (line 293)
    kwargs_168208 = {}
    # Getting the type of 'hermeline' (line 293)
    hermeline_168204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'hermeline', False)
    # Calling hermeline(args, kwargs) (line 293)
    hermeline_call_result_168209 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), hermeline_168204, *[result___neg___168206, int_168207], **kwargs_168208)
    
    list_168212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 13), list_168212, hermeline_call_result_168209)
    # Assigning a type to the variable 'p' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'p', list_168212)
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to len(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'p' (line 294)
    p_168214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'p', False)
    # Processing the call keyword arguments (line 294)
    kwargs_168215 = {}
    # Getting the type of 'len' (line 294)
    len_168213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'len', False)
    # Calling len(args, kwargs) (line 294)
    len_call_result_168216 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), len_168213, *[p_168214], **kwargs_168215)
    
    # Assigning a type to the variable 'n' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'n', len_call_result_168216)
    
    
    # Getting the type of 'n' (line 295)
    n_168217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'n')
    int_168218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 18), 'int')
    # Applying the binary operator '>' (line 295)
    result_gt_168219 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 14), '>', n_168217, int_168218)
    
    # Testing the type of an if condition (line 295)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_gt_168219)
    # SSA begins for while statement (line 295)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 296):
    
    # Assigning a Call to a Name:
    
    # Call to divmod(...): (line 296)
    # Processing the call arguments (line 296)
    # Getting the type of 'n' (line 296)
    n_168221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'n', False)
    int_168222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 29), 'int')
    # Processing the call keyword arguments (line 296)
    kwargs_168223 = {}
    # Getting the type of 'divmod' (line 296)
    divmod_168220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 296)
    divmod_call_result_168224 = invoke(stypy.reporting.localization.Localization(__file__, 296, 19), divmod_168220, *[n_168221, int_168222], **kwargs_168223)
    
    # Assigning a type to the variable 'call_assignment_167918' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167918', divmod_call_result_168224)
    
    # Assigning a Call to a Name (line 296):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'int')
    # Processing the call keyword arguments
    kwargs_168228 = {}
    # Getting the type of 'call_assignment_167918' (line 296)
    call_assignment_167918_168225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167918', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___168226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), call_assignment_167918_168225, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168229 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168226, *[int_168227], **kwargs_168228)
    
    # Assigning a type to the variable 'call_assignment_167919' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167919', getitem___call_result_168229)
    
    # Assigning a Name to a Name (line 296):
    # Getting the type of 'call_assignment_167919' (line 296)
    call_assignment_167919_168230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167919')
    # Assigning a type to the variable 'm' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'm', call_assignment_167919_168230)
    
    # Assigning a Call to a Name (line 296):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'int')
    # Processing the call keyword arguments
    kwargs_168234 = {}
    # Getting the type of 'call_assignment_167918' (line 296)
    call_assignment_167918_168231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167918', False)
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___168232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), call_assignment_167918_168231, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168235 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168232, *[int_168233], **kwargs_168234)
    
    # Assigning a type to the variable 'call_assignment_167920' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167920', getitem___call_result_168235)
    
    # Assigning a Name to a Name (line 296):
    # Getting the type of 'call_assignment_167920' (line 296)
    call_assignment_167920_168236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'call_assignment_167920')
    # Assigning a type to the variable 'r' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'r', call_assignment_167920_168236)
    
    # Assigning a ListComp to a Name (line 297):
    
    # Assigning a ListComp to a Name (line 297):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'm' (line 297)
    m_168251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 57), 'm', False)
    # Processing the call keyword arguments (line 297)
    kwargs_168252 = {}
    # Getting the type of 'range' (line 297)
    range_168250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 51), 'range', False)
    # Calling range(args, kwargs) (line 297)
    range_call_result_168253 = invoke(stypy.reporting.localization.Localization(__file__, 297, 51), range_168250, *[m_168251], **kwargs_168252)
    
    comprehension_168254 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 19), range_call_result_168253)
    # Assigning a type to the variable 'i' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'i', comprehension_168254)
    
    # Call to hermemul(...): (line 297)
    # Processing the call arguments (line 297)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 297)
    i_168238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 30), 'i', False)
    # Getting the type of 'p' (line 297)
    p_168239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 28), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___168240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 28), p_168239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_168241 = invoke(stypy.reporting.localization.Localization(__file__, 297, 28), getitem___168240, i_168238)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 297)
    i_168242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'i', False)
    # Getting the type of 'm' (line 297)
    m_168243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 38), 'm', False)
    # Applying the binary operator '+' (line 297)
    result_add_168244 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 36), '+', i_168242, m_168243)
    
    # Getting the type of 'p' (line 297)
    p_168245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 34), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___168246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 34), p_168245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_168247 = invoke(stypy.reporting.localization.Localization(__file__, 297, 34), getitem___168246, result_add_168244)
    
    # Processing the call keyword arguments (line 297)
    kwargs_168248 = {}
    # Getting the type of 'hermemul' (line 297)
    hermemul_168237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'hermemul', False)
    # Calling hermemul(args, kwargs) (line 297)
    hermemul_call_result_168249 = invoke(stypy.reporting.localization.Localization(__file__, 297, 19), hermemul_168237, *[subscript_call_result_168241, subscript_call_result_168247], **kwargs_168248)
    
    list_168255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 19), list_168255, hermemul_call_result_168249)
    # Assigning a type to the variable 'tmp' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'tmp', list_168255)
    
    # Getting the type of 'r' (line 298)
    r_168256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'r')
    # Testing the type of an if condition (line 298)
    if_condition_168257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 12), r_168256)
    # Assigning a type to the variable 'if_condition_168257' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'if_condition_168257', if_condition_168257)
    # SSA begins for if statement (line 298)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 299):
    
    # Assigning a Call to a Subscript (line 299):
    
    # Call to hermemul(...): (line 299)
    # Processing the call arguments (line 299)
    
    # Obtaining the type of the subscript
    int_168259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 38), 'int')
    # Getting the type of 'tmp' (line 299)
    tmp_168260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'tmp', False)
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___168261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 34), tmp_168260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_168262 = invoke(stypy.reporting.localization.Localization(__file__, 299, 34), getitem___168261, int_168259)
    
    
    # Obtaining the type of the subscript
    int_168263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 44), 'int')
    # Getting the type of 'p' (line 299)
    p_168264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___168265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 42), p_168264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_168266 = invoke(stypy.reporting.localization.Localization(__file__, 299, 42), getitem___168265, int_168263)
    
    # Processing the call keyword arguments (line 299)
    kwargs_168267 = {}
    # Getting the type of 'hermemul' (line 299)
    hermemul_168258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'hermemul', False)
    # Calling hermemul(args, kwargs) (line 299)
    hermemul_call_result_168268 = invoke(stypy.reporting.localization.Localization(__file__, 299, 25), hermemul_168258, *[subscript_call_result_168262, subscript_call_result_168266], **kwargs_168267)
    
    # Getting the type of 'tmp' (line 299)
    tmp_168269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'tmp')
    int_168270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'int')
    # Storing an element on a container (line 299)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 16), tmp_168269, (int_168270, hermemul_call_result_168268))
    # SSA join for if statement (line 298)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 300):
    
    # Assigning a Name to a Name (line 300):
    # Getting the type of 'tmp' (line 300)
    tmp_168271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'tmp')
    # Assigning a type to the variable 'p' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'p', tmp_168271)
    
    # Assigning a Name to a Name (line 301):
    
    # Assigning a Name to a Name (line 301):
    # Getting the type of 'm' (line 301)
    m_168272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'm')
    # Assigning a type to the variable 'n' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'n', m_168272)
    # SSA join for while statement (line 295)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_168273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 17), 'int')
    # Getting the type of 'p' (line 302)
    p_168274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'p')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___168275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), p_168274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_168276 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), getitem___168275, int_168273)
    
    # Assigning a type to the variable 'stypy_return_type' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', subscript_call_result_168276)
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermefromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermefromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_168277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168277)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermefromroots'
    return stypy_return_type_168277

# Assigning a type to the variable 'hermefromroots' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'hermefromroots', hermefromroots)

@norecursion
def hermeadd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeadd'
    module_type_store = module_type_store.open_function_context('hermeadd', 305, 0, False)
    
    # Passed parameters checking function
    hermeadd.stypy_localization = localization
    hermeadd.stypy_type_of_self = None
    hermeadd.stypy_type_store = module_type_store
    hermeadd.stypy_function_name = 'hermeadd'
    hermeadd.stypy_param_names_list = ['c1', 'c2']
    hermeadd.stypy_varargs_param_name = None
    hermeadd.stypy_kwargs_param_name = None
    hermeadd.stypy_call_defaults = defaults
    hermeadd.stypy_call_varargs = varargs
    hermeadd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeadd', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeadd', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeadd(...)' code ##################

    str_168278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Add one Hermite series to another.\n\n    Returns the sum of two Hermite series `c1` + `c2`.  The arguments\n    are sequences of coefficients ordered from lowest order term to\n    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the Hermite series of their sum.\n\n    See Also\n    --------\n    hermesub, hermemul, hermediv, hermepow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the sum of two Hermite series\n    is a Hermite series (without having to "reproject" the result onto\n    the basis set) so addition, just like that of "standard" polynomials,\n    is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeadd\n    >>> hermeadd([1, 2, 3], [1, 2, 3, 4])\n    array([ 2.,  4.,  6.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 343):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 343)
    # Processing the call arguments (line 343)
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_168281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    # Adding element type (line 343)
    # Getting the type of 'c1' (line 343)
    c1_168282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 28), list_168281, c1_168282)
    # Adding element type (line 343)
    # Getting the type of 'c2' (line 343)
    c2_168283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 28), list_168281, c2_168283)
    
    # Processing the call keyword arguments (line 343)
    kwargs_168284 = {}
    # Getting the type of 'pu' (line 343)
    pu_168279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 343)
    as_series_168280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 15), pu_168279, 'as_series')
    # Calling as_series(args, kwargs) (line 343)
    as_series_call_result_168285 = invoke(stypy.reporting.localization.Localization(__file__, 343, 15), as_series_168280, *[list_168281], **kwargs_168284)
    
    # Assigning a type to the variable 'call_assignment_167921' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167921', as_series_call_result_168285)
    
    # Assigning a Call to a Name (line 343):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168289 = {}
    # Getting the type of 'call_assignment_167921' (line 343)
    call_assignment_167921_168286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167921', False)
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___168287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 4), call_assignment_167921_168286, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168290 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168287, *[int_168288], **kwargs_168289)
    
    # Assigning a type to the variable 'call_assignment_167922' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167922', getitem___call_result_168290)
    
    # Assigning a Name to a Name (line 343):
    # Getting the type of 'call_assignment_167922' (line 343)
    call_assignment_167922_168291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167922')
    # Assigning a type to the variable 'c1' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 5), 'c1', call_assignment_167922_168291)
    
    # Assigning a Call to a Name (line 343):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168295 = {}
    # Getting the type of 'call_assignment_167921' (line 343)
    call_assignment_167921_168292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167921', False)
    # Obtaining the member '__getitem__' of a type (line 343)
    getitem___168293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 4), call_assignment_167921_168292, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168296 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168293, *[int_168294], **kwargs_168295)
    
    # Assigning a type to the variable 'call_assignment_167923' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167923', getitem___call_result_168296)
    
    # Assigning a Name to a Name (line 343):
    # Getting the type of 'call_assignment_167923' (line 343)
    call_assignment_167923_168297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'call_assignment_167923')
    # Assigning a type to the variable 'c2' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 9), 'c2', call_assignment_167923_168297)
    
    
    
    # Call to len(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'c1' (line 344)
    c1_168299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 11), 'c1', False)
    # Processing the call keyword arguments (line 344)
    kwargs_168300 = {}
    # Getting the type of 'len' (line 344)
    len_168298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 7), 'len', False)
    # Calling len(args, kwargs) (line 344)
    len_call_result_168301 = invoke(stypy.reporting.localization.Localization(__file__, 344, 7), len_168298, *[c1_168299], **kwargs_168300)
    
    
    # Call to len(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'c2' (line 344)
    c2_168303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'c2', False)
    # Processing the call keyword arguments (line 344)
    kwargs_168304 = {}
    # Getting the type of 'len' (line 344)
    len_168302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 17), 'len', False)
    # Calling len(args, kwargs) (line 344)
    len_call_result_168305 = invoke(stypy.reporting.localization.Localization(__file__, 344, 17), len_168302, *[c2_168303], **kwargs_168304)
    
    # Applying the binary operator '>' (line 344)
    result_gt_168306 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 7), '>', len_call_result_168301, len_call_result_168305)
    
    # Testing the type of an if condition (line 344)
    if_condition_168307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 4), result_gt_168306)
    # Assigning a type to the variable 'if_condition_168307' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'if_condition_168307', if_condition_168307)
    # SSA begins for if statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 345)
    c1_168308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 345)
    c2_168309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'c2')
    # Obtaining the member 'size' of a type (line 345)
    size_168310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), c2_168309, 'size')
    slice_168311 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 8), None, size_168310, None)
    # Getting the type of 'c1' (line 345)
    c1_168312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 345)
    getitem___168313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), c1_168312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 345)
    subscript_call_result_168314 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), getitem___168313, slice_168311)
    
    # Getting the type of 'c2' (line 345)
    c2_168315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'c2')
    # Applying the binary operator '+=' (line 345)
    result_iadd_168316 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 8), '+=', subscript_call_result_168314, c2_168315)
    # Getting the type of 'c1' (line 345)
    c1_168317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'c1')
    # Getting the type of 'c2' (line 345)
    c2_168318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'c2')
    # Obtaining the member 'size' of a type (line 345)
    size_168319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), c2_168318, 'size')
    slice_168320 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 345, 8), None, size_168319, None)
    # Storing an element on a container (line 345)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 8), c1_168317, (slice_168320, result_iadd_168316))
    
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'c1' (line 346)
    c1_168321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'ret', c1_168321)
    # SSA branch for the else part of an if statement (line 344)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'c2' (line 348)
    c2_168322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 348)
    c1_168323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'c1')
    # Obtaining the member 'size' of a type (line 348)
    size_168324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), c1_168323, 'size')
    slice_168325 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 8), None, size_168324, None)
    # Getting the type of 'c2' (line 348)
    c2_168326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 348)
    getitem___168327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), c2_168326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 348)
    subscript_call_result_168328 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), getitem___168327, slice_168325)
    
    # Getting the type of 'c1' (line 348)
    c1_168329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 24), 'c1')
    # Applying the binary operator '+=' (line 348)
    result_iadd_168330 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 8), '+=', subscript_call_result_168328, c1_168329)
    # Getting the type of 'c2' (line 348)
    c2_168331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'c2')
    # Getting the type of 'c1' (line 348)
    c1_168332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'c1')
    # Obtaining the member 'size' of a type (line 348)
    size_168333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 12), c1_168332, 'size')
    slice_168334 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 348, 8), None, size_168333, None)
    # Storing an element on a container (line 348)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 8), c2_168331, (slice_168334, result_iadd_168330))
    
    
    # Assigning a Name to a Name (line 349):
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'c2' (line 349)
    c2_168335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'ret', c2_168335)
    # SSA join for if statement (line 344)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'ret' (line 350)
    ret_168338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 22), 'ret', False)
    # Processing the call keyword arguments (line 350)
    kwargs_168339 = {}
    # Getting the type of 'pu' (line 350)
    pu_168336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 350)
    trimseq_168337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 11), pu_168336, 'trimseq')
    # Calling trimseq(args, kwargs) (line 350)
    trimseq_call_result_168340 = invoke(stypy.reporting.localization.Localization(__file__, 350, 11), trimseq_168337, *[ret_168338], **kwargs_168339)
    
    # Assigning a type to the variable 'stypy_return_type' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type', trimseq_call_result_168340)
    
    # ################# End of 'hermeadd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeadd' in the type store
    # Getting the type of 'stypy_return_type' (line 305)
    stypy_return_type_168341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168341)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeadd'
    return stypy_return_type_168341

# Assigning a type to the variable 'hermeadd' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'hermeadd', hermeadd)

@norecursion
def hermesub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermesub'
    module_type_store = module_type_store.open_function_context('hermesub', 353, 0, False)
    
    # Passed parameters checking function
    hermesub.stypy_localization = localization
    hermesub.stypy_type_of_self = None
    hermesub.stypy_type_store = module_type_store
    hermesub.stypy_function_name = 'hermesub'
    hermesub.stypy_param_names_list = ['c1', 'c2']
    hermesub.stypy_varargs_param_name = None
    hermesub.stypy_kwargs_param_name = None
    hermesub.stypy_call_defaults = defaults
    hermesub.stypy_call_varargs = varargs
    hermesub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermesub', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermesub', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermesub(...)' code ##################

    str_168342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, (-1)), 'str', '\n    Subtract one Hermite series from another.\n\n    Returns the difference of two Hermite series `c1` - `c2`.  The\n    sequences of coefficients are from lowest order term to highest, i.e.,\n    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Hermite series coefficients representing their difference.\n\n    See Also\n    --------\n    hermeadd, hermemul, hermediv, hermepow\n\n    Notes\n    -----\n    Unlike multiplication, division, etc., the difference of two Hermite\n    series is a Hermite series (without having to "reproject" the result\n    onto the basis set) so subtraction, just like that of "standard"\n    polynomials, is simply "component-wise."\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermesub\n    >>> hermesub([1, 2, 3, 4], [1, 2, 3])\n    array([ 0.,  0.,  0.,  4.])\n\n    ')
    
    # Assigning a Call to a List (line 391):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 391)
    # Processing the call arguments (line 391)
    
    # Obtaining an instance of the builtin type 'list' (line 391)
    list_168345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 391)
    # Adding element type (line 391)
    # Getting the type of 'c1' (line 391)
    c1_168346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 28), list_168345, c1_168346)
    # Adding element type (line 391)
    # Getting the type of 'c2' (line 391)
    c2_168347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 28), list_168345, c2_168347)
    
    # Processing the call keyword arguments (line 391)
    kwargs_168348 = {}
    # Getting the type of 'pu' (line 391)
    pu_168343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 391)
    as_series_168344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 15), pu_168343, 'as_series')
    # Calling as_series(args, kwargs) (line 391)
    as_series_call_result_168349 = invoke(stypy.reporting.localization.Localization(__file__, 391, 15), as_series_168344, *[list_168345], **kwargs_168348)
    
    # Assigning a type to the variable 'call_assignment_167924' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167924', as_series_call_result_168349)
    
    # Assigning a Call to a Name (line 391):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168353 = {}
    # Getting the type of 'call_assignment_167924' (line 391)
    call_assignment_167924_168350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167924', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___168351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 4), call_assignment_167924_168350, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168354 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168351, *[int_168352], **kwargs_168353)
    
    # Assigning a type to the variable 'call_assignment_167925' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167925', getitem___call_result_168354)
    
    # Assigning a Name to a Name (line 391):
    # Getting the type of 'call_assignment_167925' (line 391)
    call_assignment_167925_168355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167925')
    # Assigning a type to the variable 'c1' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 5), 'c1', call_assignment_167925_168355)
    
    # Assigning a Call to a Name (line 391):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168359 = {}
    # Getting the type of 'call_assignment_167924' (line 391)
    call_assignment_167924_168356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167924', False)
    # Obtaining the member '__getitem__' of a type (line 391)
    getitem___168357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 4), call_assignment_167924_168356, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168360 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168357, *[int_168358], **kwargs_168359)
    
    # Assigning a type to the variable 'call_assignment_167926' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167926', getitem___call_result_168360)
    
    # Assigning a Name to a Name (line 391):
    # Getting the type of 'call_assignment_167926' (line 391)
    call_assignment_167926_168361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'call_assignment_167926')
    # Assigning a type to the variable 'c2' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 9), 'c2', call_assignment_167926_168361)
    
    
    
    # Call to len(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'c1' (line 392)
    c1_168363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'c1', False)
    # Processing the call keyword arguments (line 392)
    kwargs_168364 = {}
    # Getting the type of 'len' (line 392)
    len_168362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 7), 'len', False)
    # Calling len(args, kwargs) (line 392)
    len_call_result_168365 = invoke(stypy.reporting.localization.Localization(__file__, 392, 7), len_168362, *[c1_168363], **kwargs_168364)
    
    
    # Call to len(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'c2' (line 392)
    c2_168367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'c2', False)
    # Processing the call keyword arguments (line 392)
    kwargs_168368 = {}
    # Getting the type of 'len' (line 392)
    len_168366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 17), 'len', False)
    # Calling len(args, kwargs) (line 392)
    len_call_result_168369 = invoke(stypy.reporting.localization.Localization(__file__, 392, 17), len_168366, *[c2_168367], **kwargs_168368)
    
    # Applying the binary operator '>' (line 392)
    result_gt_168370 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 7), '>', len_call_result_168365, len_call_result_168369)
    
    # Testing the type of an if condition (line 392)
    if_condition_168371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 4), result_gt_168370)
    # Assigning a type to the variable 'if_condition_168371' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'if_condition_168371', if_condition_168371)
    # SSA begins for if statement (line 392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c1' (line 393)
    c1_168372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'c1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c2' (line 393)
    c2_168373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'c2')
    # Obtaining the member 'size' of a type (line 393)
    size_168374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), c2_168373, 'size')
    slice_168375 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 8), None, size_168374, None)
    # Getting the type of 'c1' (line 393)
    c1_168376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'c1')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___168377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), c1_168376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_168378 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___168377, slice_168375)
    
    # Getting the type of 'c2' (line 393)
    c2_168379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'c2')
    # Applying the binary operator '-=' (line 393)
    result_isub_168380 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 8), '-=', subscript_call_result_168378, c2_168379)
    # Getting the type of 'c1' (line 393)
    c1_168381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'c1')
    # Getting the type of 'c2' (line 393)
    c2_168382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'c2')
    # Obtaining the member 'size' of a type (line 393)
    size_168383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), c2_168382, 'size')
    slice_168384 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 393, 8), None, size_168383, None)
    # Storing an element on a container (line 393)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 8), c1_168381, (slice_168384, result_isub_168380))
    
    
    # Assigning a Name to a Name (line 394):
    
    # Assigning a Name to a Name (line 394):
    # Getting the type of 'c1' (line 394)
    c1_168385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 14), 'c1')
    # Assigning a type to the variable 'ret' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'ret', c1_168385)
    # SSA branch for the else part of an if statement (line 392)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a UnaryOp to a Name (line 396):
    
    # Assigning a UnaryOp to a Name (line 396):
    
    # Getting the type of 'c2' (line 396)
    c2_168386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 14), 'c2')
    # Applying the 'usub' unary operator (line 396)
    result___neg___168387 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 13), 'usub', c2_168386)
    
    # Assigning a type to the variable 'c2' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'c2', result___neg___168387)
    
    # Getting the type of 'c2' (line 397)
    c2_168388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'c2')
    
    # Obtaining the type of the subscript
    # Getting the type of 'c1' (line 397)
    c1_168389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'c1')
    # Obtaining the member 'size' of a type (line 397)
    size_168390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), c1_168389, 'size')
    slice_168391 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 397, 8), None, size_168390, None)
    # Getting the type of 'c2' (line 397)
    c2_168392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'c2')
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___168393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), c2_168392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_168394 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), getitem___168393, slice_168391)
    
    # Getting the type of 'c1' (line 397)
    c1_168395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'c1')
    # Applying the binary operator '+=' (line 397)
    result_iadd_168396 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 8), '+=', subscript_call_result_168394, c1_168395)
    # Getting the type of 'c2' (line 397)
    c2_168397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'c2')
    # Getting the type of 'c1' (line 397)
    c1_168398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'c1')
    # Obtaining the member 'size' of a type (line 397)
    size_168399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), c1_168398, 'size')
    slice_168400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 397, 8), None, size_168399, None)
    # Storing an element on a container (line 397)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 8), c2_168397, (slice_168400, result_iadd_168396))
    
    
    # Assigning a Name to a Name (line 398):
    
    # Assigning a Name to a Name (line 398):
    # Getting the type of 'c2' (line 398)
    c2_168401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 14), 'c2')
    # Assigning a type to the variable 'ret' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'ret', c2_168401)
    # SSA join for if statement (line 392)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to trimseq(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'ret' (line 399)
    ret_168404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 22), 'ret', False)
    # Processing the call keyword arguments (line 399)
    kwargs_168405 = {}
    # Getting the type of 'pu' (line 399)
    pu_168402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 399)
    trimseq_168403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 11), pu_168402, 'trimseq')
    # Calling trimseq(args, kwargs) (line 399)
    trimseq_call_result_168406 = invoke(stypy.reporting.localization.Localization(__file__, 399, 11), trimseq_168403, *[ret_168404], **kwargs_168405)
    
    # Assigning a type to the variable 'stypy_return_type' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type', trimseq_call_result_168406)
    
    # ################# End of 'hermesub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermesub' in the type store
    # Getting the type of 'stypy_return_type' (line 353)
    stypy_return_type_168407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermesub'
    return stypy_return_type_168407

# Assigning a type to the variable 'hermesub' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'hermesub', hermesub)

@norecursion
def hermemulx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermemulx'
    module_type_store = module_type_store.open_function_context('hermemulx', 402, 0, False)
    
    # Passed parameters checking function
    hermemulx.stypy_localization = localization
    hermemulx.stypy_type_of_self = None
    hermemulx.stypy_type_store = module_type_store
    hermemulx.stypy_function_name = 'hermemulx'
    hermemulx.stypy_param_names_list = ['c']
    hermemulx.stypy_varargs_param_name = None
    hermemulx.stypy_kwargs_param_name = None
    hermemulx.stypy_call_defaults = defaults
    hermemulx.stypy_call_varargs = varargs
    hermemulx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermemulx', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermemulx', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermemulx(...)' code ##################

    str_168408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, (-1)), 'str', 'Multiply a Hermite series by x.\n\n    Multiply the Hermite series `c` by x, where x is the independent\n    variable.\n\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Array representing the result of the multiplication.\n\n    Notes\n    -----\n    The multiplication uses the recursion relationship for Hermite\n    polynomials in the form\n\n    .. math::\n\n    xP_i(x) = (P_{i + 1}(x) + iP_{i - 1}(x)))\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermemulx\n    >>> hermemulx([1, 2, 3])\n    array([ 2.,  7.,  2.,  3.])\n\n    ')
    
    # Assigning a Call to a List (line 437):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Obtaining an instance of the builtin type 'list' (line 437)
    list_168411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 437)
    # Adding element type (line 437)
    # Getting the type of 'c' (line 437)
    c_168412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 23), list_168411, c_168412)
    
    # Processing the call keyword arguments (line 437)
    kwargs_168413 = {}
    # Getting the type of 'pu' (line 437)
    pu_168409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 437)
    as_series_168410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 10), pu_168409, 'as_series')
    # Calling as_series(args, kwargs) (line 437)
    as_series_call_result_168414 = invoke(stypy.reporting.localization.Localization(__file__, 437, 10), as_series_168410, *[list_168411], **kwargs_168413)
    
    # Assigning a type to the variable 'call_assignment_167927' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'call_assignment_167927', as_series_call_result_168414)
    
    # Assigning a Call to a Name (line 437):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168418 = {}
    # Getting the type of 'call_assignment_167927' (line 437)
    call_assignment_167927_168415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'call_assignment_167927', False)
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___168416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 4), call_assignment_167927_168415, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168419 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168416, *[int_168417], **kwargs_168418)
    
    # Assigning a type to the variable 'call_assignment_167928' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'call_assignment_167928', getitem___call_result_168419)
    
    # Assigning a Name to a Name (line 437):
    # Getting the type of 'call_assignment_167928' (line 437)
    call_assignment_167928_168420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'call_assignment_167928')
    # Assigning a type to the variable 'c' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 5), 'c', call_assignment_167928_168420)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'c' (line 439)
    c_168422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'c', False)
    # Processing the call keyword arguments (line 439)
    kwargs_168423 = {}
    # Getting the type of 'len' (line 439)
    len_168421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 7), 'len', False)
    # Calling len(args, kwargs) (line 439)
    len_call_result_168424 = invoke(stypy.reporting.localization.Localization(__file__, 439, 7), len_168421, *[c_168422], **kwargs_168423)
    
    int_168425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 17), 'int')
    # Applying the binary operator '==' (line 439)
    result_eq_168426 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 7), '==', len_call_result_168424, int_168425)
    
    
    
    # Obtaining the type of the subscript
    int_168427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 25), 'int')
    # Getting the type of 'c' (line 439)
    c_168428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'c')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___168429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), c_168428, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_168430 = invoke(stypy.reporting.localization.Localization(__file__, 439, 23), getitem___168429, int_168427)
    
    int_168431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 31), 'int')
    # Applying the binary operator '==' (line 439)
    result_eq_168432 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 23), '==', subscript_call_result_168430, int_168431)
    
    # Applying the binary operator 'and' (line 439)
    result_and_keyword_168433 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 7), 'and', result_eq_168426, result_eq_168432)
    
    # Testing the type of an if condition (line 439)
    if_condition_168434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 4), result_and_keyword_168433)
    # Assigning a type to the variable 'if_condition_168434' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'if_condition_168434', if_condition_168434)
    # SSA begins for if statement (line 439)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 440)
    c_168435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'stypy_return_type', c_168435)
    # SSA join for if statement (line 439)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to empty(...): (line 442)
    # Processing the call arguments (line 442)
    
    # Call to len(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'c' (line 442)
    c_168439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'c', False)
    # Processing the call keyword arguments (line 442)
    kwargs_168440 = {}
    # Getting the type of 'len' (line 442)
    len_168438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'len', False)
    # Calling len(args, kwargs) (line 442)
    len_call_result_168441 = invoke(stypy.reporting.localization.Localization(__file__, 442, 19), len_168438, *[c_168439], **kwargs_168440)
    
    int_168442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 28), 'int')
    # Applying the binary operator '+' (line 442)
    result_add_168443 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 19), '+', len_call_result_168441, int_168442)
    
    # Processing the call keyword arguments (line 442)
    # Getting the type of 'c' (line 442)
    c_168444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 37), 'c', False)
    # Obtaining the member 'dtype' of a type (line 442)
    dtype_168445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 37), c_168444, 'dtype')
    keyword_168446 = dtype_168445
    kwargs_168447 = {'dtype': keyword_168446}
    # Getting the type of 'np' (line 442)
    np_168436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 442)
    empty_168437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 10), np_168436, 'empty')
    # Calling empty(args, kwargs) (line 442)
    empty_call_result_168448 = invoke(stypy.reporting.localization.Localization(__file__, 442, 10), empty_168437, *[result_add_168443], **kwargs_168447)
    
    # Assigning a type to the variable 'prd' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'prd', empty_call_result_168448)
    
    # Assigning a BinOp to a Subscript (line 443):
    
    # Assigning a BinOp to a Subscript (line 443):
    
    # Obtaining the type of the subscript
    int_168449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 15), 'int')
    # Getting the type of 'c' (line 443)
    c_168450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___168451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 13), c_168450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_168452 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), getitem___168451, int_168449)
    
    int_168453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 18), 'int')
    # Applying the binary operator '*' (line 443)
    result_mul_168454 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 13), '*', subscript_call_result_168452, int_168453)
    
    # Getting the type of 'prd' (line 443)
    prd_168455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'prd')
    int_168456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 8), 'int')
    # Storing an element on a container (line 443)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 4), prd_168455, (int_168456, result_mul_168454))
    
    # Assigning a Subscript to a Subscript (line 444):
    
    # Assigning a Subscript to a Subscript (line 444):
    
    # Obtaining the type of the subscript
    int_168457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 15), 'int')
    # Getting the type of 'c' (line 444)
    c_168458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 444)
    getitem___168459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 13), c_168458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 444)
    subscript_call_result_168460 = invoke(stypy.reporting.localization.Localization(__file__, 444, 13), getitem___168459, int_168457)
    
    # Getting the type of 'prd' (line 444)
    prd_168461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'prd')
    int_168462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 8), 'int')
    # Storing an element on a container (line 444)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 4), prd_168461, (int_168462, subscript_call_result_168460))
    
    
    # Call to range(...): (line 445)
    # Processing the call arguments (line 445)
    int_168464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'int')
    
    # Call to len(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'c' (line 445)
    c_168466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'c', False)
    # Processing the call keyword arguments (line 445)
    kwargs_168467 = {}
    # Getting the type of 'len' (line 445)
    len_168465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 22), 'len', False)
    # Calling len(args, kwargs) (line 445)
    len_call_result_168468 = invoke(stypy.reporting.localization.Localization(__file__, 445, 22), len_168465, *[c_168466], **kwargs_168467)
    
    # Processing the call keyword arguments (line 445)
    kwargs_168469 = {}
    # Getting the type of 'range' (line 445)
    range_168463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 13), 'range', False)
    # Calling range(args, kwargs) (line 445)
    range_call_result_168470 = invoke(stypy.reporting.localization.Localization(__file__, 445, 13), range_168463, *[int_168464, len_call_result_168468], **kwargs_168469)
    
    # Testing the type of a for loop iterable (line 445)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 4), range_call_result_168470)
    # Getting the type of the for loop variable (line 445)
    for_loop_var_168471 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 4), range_call_result_168470)
    # Assigning a type to the variable 'i' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'i', for_loop_var_168471)
    # SSA begins for a for statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 446):
    
    # Assigning a Subscript to a Subscript (line 446):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 446)
    i_168472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'i')
    # Getting the type of 'c' (line 446)
    c_168473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___168474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 21), c_168473, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_168475 = invoke(stypy.reporting.localization.Localization(__file__, 446, 21), getitem___168474, i_168472)
    
    # Getting the type of 'prd' (line 446)
    prd_168476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'prd')
    # Getting the type of 'i' (line 446)
    i_168477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'i')
    int_168478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 16), 'int')
    # Applying the binary operator '+' (line 446)
    result_add_168479 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 12), '+', i_168477, int_168478)
    
    # Storing an element on a container (line 446)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 8), prd_168476, (result_add_168479, subscript_call_result_168475))
    
    # Getting the type of 'prd' (line 447)
    prd_168480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'prd')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 447)
    i_168481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'i')
    int_168482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 16), 'int')
    # Applying the binary operator '-' (line 447)
    result_sub_168483 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 12), '-', i_168481, int_168482)
    
    # Getting the type of 'prd' (line 447)
    prd_168484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'prd')
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___168485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), prd_168484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_168486 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), getitem___168485, result_sub_168483)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 447)
    i_168487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 24), 'i')
    # Getting the type of 'c' (line 447)
    c_168488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 447)
    getitem___168489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 22), c_168488, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 447)
    subscript_call_result_168490 = invoke(stypy.reporting.localization.Localization(__file__, 447, 22), getitem___168489, i_168487)
    
    # Getting the type of 'i' (line 447)
    i_168491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), 'i')
    # Applying the binary operator '*' (line 447)
    result_mul_168492 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 22), '*', subscript_call_result_168490, i_168491)
    
    # Applying the binary operator '+=' (line 447)
    result_iadd_168493 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 8), '+=', subscript_call_result_168486, result_mul_168492)
    # Getting the type of 'prd' (line 447)
    prd_168494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'prd')
    # Getting the type of 'i' (line 447)
    i_168495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'i')
    int_168496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 16), 'int')
    # Applying the binary operator '-' (line 447)
    result_sub_168497 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 12), '-', i_168495, int_168496)
    
    # Storing an element on a container (line 447)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 8), prd_168494, (result_sub_168497, result_iadd_168493))
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 448)
    prd_168498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'stypy_return_type', prd_168498)
    
    # ################# End of 'hermemulx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermemulx' in the type store
    # Getting the type of 'stypy_return_type' (line 402)
    stypy_return_type_168499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168499)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermemulx'
    return stypy_return_type_168499

# Assigning a type to the variable 'hermemulx' (line 402)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 0), 'hermemulx', hermemulx)

@norecursion
def hermemul(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermemul'
    module_type_store = module_type_store.open_function_context('hermemul', 451, 0, False)
    
    # Passed parameters checking function
    hermemul.stypy_localization = localization
    hermemul.stypy_type_of_self = None
    hermemul.stypy_type_store = module_type_store
    hermemul.stypy_function_name = 'hermemul'
    hermemul.stypy_param_names_list = ['c1', 'c2']
    hermemul.stypy_varargs_param_name = None
    hermemul.stypy_kwargs_param_name = None
    hermemul.stypy_call_defaults = defaults
    hermemul.stypy_call_varargs = varargs
    hermemul.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermemul', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermemul', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermemul(...)' code ##################

    str_168500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, (-1)), 'str', '\n    Multiply one Hermite series by another.\n\n    Returns the product of two Hermite series `c1` * `c2`.  The arguments\n    are sequences of coefficients, from lowest order "term" to highest,\n    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    out : ndarray\n        Of Hermite series coefficients representing their product.\n\n    See Also\n    --------\n    hermeadd, hermesub, hermediv, hermepow\n\n    Notes\n    -----\n    In general, the (polynomial) product of two C-series results in terms\n    that are not in the Hermite polynomial basis set.  Thus, to express\n    the product as a Hermite series, it is necessary to "reproject" the\n    product onto said basis set, which may produce "unintuitive" (but\n    correct) results; see Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermemul\n    >>> hermemul([1, 2, 3], [0, 1, 2])\n    array([ 14.,  15.,  28.,   7.,   6.])\n\n    ')
    
    # Assigning a Call to a List (line 490):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 490)
    # Processing the call arguments (line 490)
    
    # Obtaining an instance of the builtin type 'list' (line 490)
    list_168503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 490)
    # Adding element type (line 490)
    # Getting the type of 'c1' (line 490)
    c1_168504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 28), list_168503, c1_168504)
    # Adding element type (line 490)
    # Getting the type of 'c2' (line 490)
    c2_168505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 28), list_168503, c2_168505)
    
    # Processing the call keyword arguments (line 490)
    kwargs_168506 = {}
    # Getting the type of 'pu' (line 490)
    pu_168501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 490)
    as_series_168502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 15), pu_168501, 'as_series')
    # Calling as_series(args, kwargs) (line 490)
    as_series_call_result_168507 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), as_series_168502, *[list_168503], **kwargs_168506)
    
    # Assigning a type to the variable 'call_assignment_167929' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167929', as_series_call_result_168507)
    
    # Assigning a Call to a Name (line 490):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168511 = {}
    # Getting the type of 'call_assignment_167929' (line 490)
    call_assignment_167929_168508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167929', False)
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___168509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), call_assignment_167929_168508, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168512 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168509, *[int_168510], **kwargs_168511)
    
    # Assigning a type to the variable 'call_assignment_167930' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167930', getitem___call_result_168512)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'call_assignment_167930' (line 490)
    call_assignment_167930_168513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167930')
    # Assigning a type to the variable 'c1' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 5), 'c1', call_assignment_167930_168513)
    
    # Assigning a Call to a Name (line 490):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168517 = {}
    # Getting the type of 'call_assignment_167929' (line 490)
    call_assignment_167929_168514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167929', False)
    # Obtaining the member '__getitem__' of a type (line 490)
    getitem___168515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 4), call_assignment_167929_168514, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168518 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168515, *[int_168516], **kwargs_168517)
    
    # Assigning a type to the variable 'call_assignment_167931' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167931', getitem___call_result_168518)
    
    # Assigning a Name to a Name (line 490):
    # Getting the type of 'call_assignment_167931' (line 490)
    call_assignment_167931_168519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 4), 'call_assignment_167931')
    # Assigning a type to the variable 'c2' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 9), 'c2', call_assignment_167931_168519)
    
    
    
    # Call to len(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'c1' (line 492)
    c1_168521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 11), 'c1', False)
    # Processing the call keyword arguments (line 492)
    kwargs_168522 = {}
    # Getting the type of 'len' (line 492)
    len_168520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 7), 'len', False)
    # Calling len(args, kwargs) (line 492)
    len_call_result_168523 = invoke(stypy.reporting.localization.Localization(__file__, 492, 7), len_168520, *[c1_168521], **kwargs_168522)
    
    
    # Call to len(...): (line 492)
    # Processing the call arguments (line 492)
    # Getting the type of 'c2' (line 492)
    c2_168525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 21), 'c2', False)
    # Processing the call keyword arguments (line 492)
    kwargs_168526 = {}
    # Getting the type of 'len' (line 492)
    len_168524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 17), 'len', False)
    # Calling len(args, kwargs) (line 492)
    len_call_result_168527 = invoke(stypy.reporting.localization.Localization(__file__, 492, 17), len_168524, *[c2_168525], **kwargs_168526)
    
    # Applying the binary operator '>' (line 492)
    result_gt_168528 = python_operator(stypy.reporting.localization.Localization(__file__, 492, 7), '>', len_call_result_168523, len_call_result_168527)
    
    # Testing the type of an if condition (line 492)
    if_condition_168529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 492, 4), result_gt_168528)
    # Assigning a type to the variable 'if_condition_168529' (line 492)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'if_condition_168529', if_condition_168529)
    # SSA begins for if statement (line 492)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 493):
    
    # Assigning a Name to a Name (line 493):
    # Getting the type of 'c2' (line 493)
    c2_168530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'c2')
    # Assigning a type to the variable 'c' (line 493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'c', c2_168530)
    
    # Assigning a Name to a Name (line 494):
    
    # Assigning a Name to a Name (line 494):
    # Getting the type of 'c1' (line 494)
    c1_168531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 13), 'c1')
    # Assigning a type to the variable 'xs' (line 494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'xs', c1_168531)
    # SSA branch for the else part of an if statement (line 492)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 496):
    
    # Assigning a Name to a Name (line 496):
    # Getting the type of 'c1' (line 496)
    c1_168532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'c1')
    # Assigning a type to the variable 'c' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'c', c1_168532)
    
    # Assigning a Name to a Name (line 497):
    
    # Assigning a Name to a Name (line 497):
    # Getting the type of 'c2' (line 497)
    c2_168533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 13), 'c2')
    # Assigning a type to the variable 'xs' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'xs', c2_168533)
    # SSA join for if statement (line 492)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'c' (line 499)
    c_168535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'c', False)
    # Processing the call keyword arguments (line 499)
    kwargs_168536 = {}
    # Getting the type of 'len' (line 499)
    len_168534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 7), 'len', False)
    # Calling len(args, kwargs) (line 499)
    len_call_result_168537 = invoke(stypy.reporting.localization.Localization(__file__, 499, 7), len_168534, *[c_168535], **kwargs_168536)
    
    int_168538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 17), 'int')
    # Applying the binary operator '==' (line 499)
    result_eq_168539 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 7), '==', len_call_result_168537, int_168538)
    
    # Testing the type of an if condition (line 499)
    if_condition_168540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), result_eq_168539)
    # Assigning a type to the variable 'if_condition_168540' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_168540', if_condition_168540)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 500):
    
    # Assigning a BinOp to a Name (line 500):
    
    # Obtaining the type of the subscript
    int_168541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 15), 'int')
    # Getting the type of 'c' (line 500)
    c_168542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 500)
    getitem___168543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 13), c_168542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 500)
    subscript_call_result_168544 = invoke(stypy.reporting.localization.Localization(__file__, 500, 13), getitem___168543, int_168541)
    
    # Getting the type of 'xs' (line 500)
    xs_168545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 18), 'xs')
    # Applying the binary operator '*' (line 500)
    result_mul_168546 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 13), '*', subscript_call_result_168544, xs_168545)
    
    # Assigning a type to the variable 'c0' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'c0', result_mul_168546)
    
    # Assigning a Num to a Name (line 501):
    
    # Assigning a Num to a Name (line 501):
    int_168547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 13), 'int')
    # Assigning a type to the variable 'c1' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'c1', int_168547)
    # SSA branch for the else part of an if statement (line 499)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 502)
    # Processing the call arguments (line 502)
    # Getting the type of 'c' (line 502)
    c_168549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 13), 'c', False)
    # Processing the call keyword arguments (line 502)
    kwargs_168550 = {}
    # Getting the type of 'len' (line 502)
    len_168548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 9), 'len', False)
    # Calling len(args, kwargs) (line 502)
    len_call_result_168551 = invoke(stypy.reporting.localization.Localization(__file__, 502, 9), len_168548, *[c_168549], **kwargs_168550)
    
    int_168552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 19), 'int')
    # Applying the binary operator '==' (line 502)
    result_eq_168553 = python_operator(stypy.reporting.localization.Localization(__file__, 502, 9), '==', len_call_result_168551, int_168552)
    
    # Testing the type of an if condition (line 502)
    if_condition_168554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 502, 9), result_eq_168553)
    # Assigning a type to the variable 'if_condition_168554' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 9), 'if_condition_168554', if_condition_168554)
    # SSA begins for if statement (line 502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 503):
    
    # Assigning a BinOp to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_168555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 15), 'int')
    # Getting the type of 'c' (line 503)
    c_168556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___168557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 13), c_168556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_168558 = invoke(stypy.reporting.localization.Localization(__file__, 503, 13), getitem___168557, int_168555)
    
    # Getting the type of 'xs' (line 503)
    xs_168559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 18), 'xs')
    # Applying the binary operator '*' (line 503)
    result_mul_168560 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 13), '*', subscript_call_result_168558, xs_168559)
    
    # Assigning a type to the variable 'c0' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'c0', result_mul_168560)
    
    # Assigning a BinOp to a Name (line 504):
    
    # Assigning a BinOp to a Name (line 504):
    
    # Obtaining the type of the subscript
    int_168561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 15), 'int')
    # Getting the type of 'c' (line 504)
    c_168562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___168563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 13), c_168562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_168564 = invoke(stypy.reporting.localization.Localization(__file__, 504, 13), getitem___168563, int_168561)
    
    # Getting the type of 'xs' (line 504)
    xs_168565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 18), 'xs')
    # Applying the binary operator '*' (line 504)
    result_mul_168566 = python_operator(stypy.reporting.localization.Localization(__file__, 504, 13), '*', subscript_call_result_168564, xs_168565)
    
    # Assigning a type to the variable 'c1' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'c1', result_mul_168566)
    # SSA branch for the else part of an if statement (line 502)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to len(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'c' (line 506)
    c_168568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 17), 'c', False)
    # Processing the call keyword arguments (line 506)
    kwargs_168569 = {}
    # Getting the type of 'len' (line 506)
    len_168567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 13), 'len', False)
    # Calling len(args, kwargs) (line 506)
    len_call_result_168570 = invoke(stypy.reporting.localization.Localization(__file__, 506, 13), len_168567, *[c_168568], **kwargs_168569)
    
    # Assigning a type to the variable 'nd' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'nd', len_call_result_168570)
    
    # Assigning a BinOp to a Name (line 507):
    
    # Assigning a BinOp to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_168571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 15), 'int')
    # Getting the type of 'c' (line 507)
    c_168572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___168573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 13), c_168572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_168574 = invoke(stypy.reporting.localization.Localization(__file__, 507, 13), getitem___168573, int_168571)
    
    # Getting the type of 'xs' (line 507)
    xs_168575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 19), 'xs')
    # Applying the binary operator '*' (line 507)
    result_mul_168576 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 13), '*', subscript_call_result_168574, xs_168575)
    
    # Assigning a type to the variable 'c0' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'c0', result_mul_168576)
    
    # Assigning a BinOp to a Name (line 508):
    
    # Assigning a BinOp to a Name (line 508):
    
    # Obtaining the type of the subscript
    int_168577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 15), 'int')
    # Getting the type of 'c' (line 508)
    c_168578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 508)
    getitem___168579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 13), c_168578, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 508)
    subscript_call_result_168580 = invoke(stypy.reporting.localization.Localization(__file__, 508, 13), getitem___168579, int_168577)
    
    # Getting the type of 'xs' (line 508)
    xs_168581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 19), 'xs')
    # Applying the binary operator '*' (line 508)
    result_mul_168582 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 13), '*', subscript_call_result_168580, xs_168581)
    
    # Assigning a type to the variable 'c1' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'c1', result_mul_168582)
    
    
    # Call to range(...): (line 509)
    # Processing the call arguments (line 509)
    int_168584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 23), 'int')
    
    # Call to len(...): (line 509)
    # Processing the call arguments (line 509)
    # Getting the type of 'c' (line 509)
    c_168586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 30), 'c', False)
    # Processing the call keyword arguments (line 509)
    kwargs_168587 = {}
    # Getting the type of 'len' (line 509)
    len_168585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 26), 'len', False)
    # Calling len(args, kwargs) (line 509)
    len_call_result_168588 = invoke(stypy.reporting.localization.Localization(__file__, 509, 26), len_168585, *[c_168586], **kwargs_168587)
    
    int_168589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 35), 'int')
    # Applying the binary operator '+' (line 509)
    result_add_168590 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 26), '+', len_call_result_168588, int_168589)
    
    # Processing the call keyword arguments (line 509)
    kwargs_168591 = {}
    # Getting the type of 'range' (line 509)
    range_168583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 17), 'range', False)
    # Calling range(args, kwargs) (line 509)
    range_call_result_168592 = invoke(stypy.reporting.localization.Localization(__file__, 509, 17), range_168583, *[int_168584, result_add_168590], **kwargs_168591)
    
    # Testing the type of a for loop iterable (line 509)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 509, 8), range_call_result_168592)
    # Getting the type of the for loop variable (line 509)
    for_loop_var_168593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 509, 8), range_call_result_168592)
    # Assigning a type to the variable 'i' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'i', for_loop_var_168593)
    # SSA begins for a for statement (line 509)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 510):
    
    # Assigning a Name to a Name (line 510):
    # Getting the type of 'c0' (line 510)
    c0_168594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'tmp', c0_168594)
    
    # Assigning a BinOp to a Name (line 511):
    
    # Assigning a BinOp to a Name (line 511):
    # Getting the type of 'nd' (line 511)
    nd_168595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'nd')
    int_168596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 22), 'int')
    # Applying the binary operator '-' (line 511)
    result_sub_168597 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 17), '-', nd_168595, int_168596)
    
    # Assigning a type to the variable 'nd' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'nd', result_sub_168597)
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to hermesub(...): (line 512)
    # Processing the call arguments (line 512)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 512)
    i_168599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 29), 'i', False)
    # Applying the 'usub' unary operator (line 512)
    result___neg___168600 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 28), 'usub', i_168599)
    
    # Getting the type of 'c' (line 512)
    c_168601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 512)
    getitem___168602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 26), c_168601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 512)
    subscript_call_result_168603 = invoke(stypy.reporting.localization.Localization(__file__, 512, 26), getitem___168602, result___neg___168600)
    
    # Getting the type of 'xs' (line 512)
    xs_168604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 32), 'xs', False)
    # Applying the binary operator '*' (line 512)
    result_mul_168605 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 26), '*', subscript_call_result_168603, xs_168604)
    
    # Getting the type of 'c1' (line 512)
    c1_168606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 36), 'c1', False)
    # Getting the type of 'nd' (line 512)
    nd_168607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 40), 'nd', False)
    int_168608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 45), 'int')
    # Applying the binary operator '-' (line 512)
    result_sub_168609 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 40), '-', nd_168607, int_168608)
    
    # Applying the binary operator '*' (line 512)
    result_mul_168610 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 36), '*', c1_168606, result_sub_168609)
    
    # Processing the call keyword arguments (line 512)
    kwargs_168611 = {}
    # Getting the type of 'hermesub' (line 512)
    hermesub_168598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 17), 'hermesub', False)
    # Calling hermesub(args, kwargs) (line 512)
    hermesub_call_result_168612 = invoke(stypy.reporting.localization.Localization(__file__, 512, 17), hermesub_168598, *[result_mul_168605, result_mul_168610], **kwargs_168611)
    
    # Assigning a type to the variable 'c0' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'c0', hermesub_call_result_168612)
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to hermeadd(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'tmp' (line 513)
    tmp_168614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 26), 'tmp', False)
    
    # Call to hermemulx(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'c1' (line 513)
    c1_168616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 41), 'c1', False)
    # Processing the call keyword arguments (line 513)
    kwargs_168617 = {}
    # Getting the type of 'hermemulx' (line 513)
    hermemulx_168615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 'hermemulx', False)
    # Calling hermemulx(args, kwargs) (line 513)
    hermemulx_call_result_168618 = invoke(stypy.reporting.localization.Localization(__file__, 513, 31), hermemulx_168615, *[c1_168616], **kwargs_168617)
    
    # Processing the call keyword arguments (line 513)
    kwargs_168619 = {}
    # Getting the type of 'hermeadd' (line 513)
    hermeadd_168613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 17), 'hermeadd', False)
    # Calling hermeadd(args, kwargs) (line 513)
    hermeadd_call_result_168620 = invoke(stypy.reporting.localization.Localization(__file__, 513, 17), hermeadd_168613, *[tmp_168614, hermemulx_call_result_168618], **kwargs_168619)
    
    # Assigning a type to the variable 'c1' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'c1', hermeadd_call_result_168620)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 502)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to hermeadd(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'c0' (line 514)
    c0_168622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'c0', False)
    
    # Call to hermemulx(...): (line 514)
    # Processing the call arguments (line 514)
    # Getting the type of 'c1' (line 514)
    c1_168624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 34), 'c1', False)
    # Processing the call keyword arguments (line 514)
    kwargs_168625 = {}
    # Getting the type of 'hermemulx' (line 514)
    hermemulx_168623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 24), 'hermemulx', False)
    # Calling hermemulx(args, kwargs) (line 514)
    hermemulx_call_result_168626 = invoke(stypy.reporting.localization.Localization(__file__, 514, 24), hermemulx_168623, *[c1_168624], **kwargs_168625)
    
    # Processing the call keyword arguments (line 514)
    kwargs_168627 = {}
    # Getting the type of 'hermeadd' (line 514)
    hermeadd_168621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'hermeadd', False)
    # Calling hermeadd(args, kwargs) (line 514)
    hermeadd_call_result_168628 = invoke(stypy.reporting.localization.Localization(__file__, 514, 11), hermeadd_168621, *[c0_168622, hermemulx_call_result_168626], **kwargs_168627)
    
    # Assigning a type to the variable 'stypy_return_type' (line 514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 4), 'stypy_return_type', hermeadd_call_result_168628)
    
    # ################# End of 'hermemul(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermemul' in the type store
    # Getting the type of 'stypy_return_type' (line 451)
    stypy_return_type_168629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermemul'
    return stypy_return_type_168629

# Assigning a type to the variable 'hermemul' (line 451)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'hermemul', hermemul)

@norecursion
def hermediv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermediv'
    module_type_store = module_type_store.open_function_context('hermediv', 517, 0, False)
    
    # Passed parameters checking function
    hermediv.stypy_localization = localization
    hermediv.stypy_type_of_self = None
    hermediv.stypy_type_store = module_type_store
    hermediv.stypy_function_name = 'hermediv'
    hermediv.stypy_param_names_list = ['c1', 'c2']
    hermediv.stypy_varargs_param_name = None
    hermediv.stypy_kwargs_param_name = None
    hermediv.stypy_call_defaults = defaults
    hermediv.stypy_call_varargs = varargs
    hermediv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermediv', ['c1', 'c2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermediv', localization, ['c1', 'c2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermediv(...)' code ##################

    str_168630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, (-1)), 'str', '\n    Divide one Hermite series by another.\n\n    Returns the quotient-with-remainder of two Hermite series\n    `c1` / `c2`.  The arguments are sequences of coefficients from lowest\n    order "term" to highest, e.g., [1,2,3] represents the series\n    ``P_0 + 2*P_1 + 3*P_2``.\n\n    Parameters\n    ----------\n    c1, c2 : array_like\n        1-D arrays of Hermite series coefficients ordered from low to\n        high.\n\n    Returns\n    -------\n    [quo, rem] : ndarrays\n        Of Hermite series coefficients representing the quotient and\n        remainder.\n\n    See Also\n    --------\n    hermeadd, hermesub, hermemul, hermepow\n\n    Notes\n    -----\n    In general, the (polynomial) division of one Hermite series by another\n    results in quotient and remainder terms that are not in the Hermite\n    polynomial basis set.  Thus, to express these results as a Hermite\n    series, it is necessary to "reproject" the results onto the Hermite\n    basis set, which may produce "unintuitive" (but correct) results; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermediv\n    >>> hermediv([ 14.,  15.,  28.,   7.,   6.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 0.]))\n    >>> hermediv([ 15.,  17.,  28.,   7.,   6.], [0, 1, 2])\n    (array([ 1.,  2.,  3.]), array([ 1.,  2.]))\n\n    ')
    
    # Assigning a Call to a List (line 561):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 561)
    # Processing the call arguments (line 561)
    
    # Obtaining an instance of the builtin type 'list' (line 561)
    list_168633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 561)
    # Adding element type (line 561)
    # Getting the type of 'c1' (line 561)
    c1_168634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'c1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 28), list_168633, c1_168634)
    # Adding element type (line 561)
    # Getting the type of 'c2' (line 561)
    c2_168635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 33), 'c2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 28), list_168633, c2_168635)
    
    # Processing the call keyword arguments (line 561)
    kwargs_168636 = {}
    # Getting the type of 'pu' (line 561)
    pu_168631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 15), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 561)
    as_series_168632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 15), pu_168631, 'as_series')
    # Calling as_series(args, kwargs) (line 561)
    as_series_call_result_168637 = invoke(stypy.reporting.localization.Localization(__file__, 561, 15), as_series_168632, *[list_168633], **kwargs_168636)
    
    # Assigning a type to the variable 'call_assignment_167932' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167932', as_series_call_result_168637)
    
    # Assigning a Call to a Name (line 561):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168641 = {}
    # Getting the type of 'call_assignment_167932' (line 561)
    call_assignment_167932_168638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167932', False)
    # Obtaining the member '__getitem__' of a type (line 561)
    getitem___168639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 4), call_assignment_167932_168638, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168642 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168639, *[int_168640], **kwargs_168641)
    
    # Assigning a type to the variable 'call_assignment_167933' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167933', getitem___call_result_168642)
    
    # Assigning a Name to a Name (line 561):
    # Getting the type of 'call_assignment_167933' (line 561)
    call_assignment_167933_168643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167933')
    # Assigning a type to the variable 'c1' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 5), 'c1', call_assignment_167933_168643)
    
    # Assigning a Call to a Name (line 561):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168647 = {}
    # Getting the type of 'call_assignment_167932' (line 561)
    call_assignment_167932_168644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167932', False)
    # Obtaining the member '__getitem__' of a type (line 561)
    getitem___168645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 4), call_assignment_167932_168644, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168648 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168645, *[int_168646], **kwargs_168647)
    
    # Assigning a type to the variable 'call_assignment_167934' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167934', getitem___call_result_168648)
    
    # Assigning a Name to a Name (line 561):
    # Getting the type of 'call_assignment_167934' (line 561)
    call_assignment_167934_168649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'call_assignment_167934')
    # Assigning a type to the variable 'c2' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 9), 'c2', call_assignment_167934_168649)
    
    
    
    # Obtaining the type of the subscript
    int_168650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 10), 'int')
    # Getting the type of 'c2' (line 562)
    c2_168651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 7), 'c2')
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___168652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 7), c2_168651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_168653 = invoke(stypy.reporting.localization.Localization(__file__, 562, 7), getitem___168652, int_168650)
    
    int_168654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 17), 'int')
    # Applying the binary operator '==' (line 562)
    result_eq_168655 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 7), '==', subscript_call_result_168653, int_168654)
    
    # Testing the type of an if condition (line 562)
    if_condition_168656 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 4), result_eq_168655)
    # Assigning a type to the variable 'if_condition_168656' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'if_condition_168656', if_condition_168656)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ZeroDivisionError(...): (line 563)
    # Processing the call keyword arguments (line 563)
    kwargs_168658 = {}
    # Getting the type of 'ZeroDivisionError' (line 563)
    ZeroDivisionError_168657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 14), 'ZeroDivisionError', False)
    # Calling ZeroDivisionError(args, kwargs) (line 563)
    ZeroDivisionError_call_result_168659 = invoke(stypy.reporting.localization.Localization(__file__, 563, 14), ZeroDivisionError_168657, *[], **kwargs_168658)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 563, 8), ZeroDivisionError_call_result_168659, 'raise parameter', BaseException)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to len(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'c1' (line 565)
    c1_168661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 14), 'c1', False)
    # Processing the call keyword arguments (line 565)
    kwargs_168662 = {}
    # Getting the type of 'len' (line 565)
    len_168660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 10), 'len', False)
    # Calling len(args, kwargs) (line 565)
    len_call_result_168663 = invoke(stypy.reporting.localization.Localization(__file__, 565, 10), len_168660, *[c1_168661], **kwargs_168662)
    
    # Assigning a type to the variable 'lc1' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'lc1', len_call_result_168663)
    
    # Assigning a Call to a Name (line 566):
    
    # Assigning a Call to a Name (line 566):
    
    # Call to len(...): (line 566)
    # Processing the call arguments (line 566)
    # Getting the type of 'c2' (line 566)
    c2_168665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 14), 'c2', False)
    # Processing the call keyword arguments (line 566)
    kwargs_168666 = {}
    # Getting the type of 'len' (line 566)
    len_168664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 10), 'len', False)
    # Calling len(args, kwargs) (line 566)
    len_call_result_168667 = invoke(stypy.reporting.localization.Localization(__file__, 566, 10), len_168664, *[c2_168665], **kwargs_168666)
    
    # Assigning a type to the variable 'lc2' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'lc2', len_call_result_168667)
    
    
    # Getting the type of 'lc1' (line 567)
    lc1_168668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 7), 'lc1')
    # Getting the type of 'lc2' (line 567)
    lc2_168669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 13), 'lc2')
    # Applying the binary operator '<' (line 567)
    result_lt_168670 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 7), '<', lc1_168668, lc2_168669)
    
    # Testing the type of an if condition (line 567)
    if_condition_168671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 4), result_lt_168670)
    # Assigning a type to the variable 'if_condition_168671' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'if_condition_168671', if_condition_168671)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_168672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    
    # Obtaining the type of the subscript
    int_168673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 19), 'int')
    slice_168674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 568, 15), None, int_168673, None)
    # Getting the type of 'c1' (line 568)
    c1_168675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'c1')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___168676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 15), c1_168675, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_168677 = invoke(stypy.reporting.localization.Localization(__file__, 568, 15), getitem___168676, slice_168674)
    
    int_168678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 22), 'int')
    # Applying the binary operator '*' (line 568)
    result_mul_168679 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 15), '*', subscript_call_result_168677, int_168678)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_168672, result_mul_168679)
    # Adding element type (line 568)
    # Getting the type of 'c1' (line 568)
    c1_168680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 25), 'c1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 15), tuple_168672, c1_168680)
    
    # Assigning a type to the variable 'stypy_return_type' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'stypy_return_type', tuple_168672)
    # SSA branch for the else part of an if statement (line 567)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lc2' (line 569)
    lc2_168681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 9), 'lc2')
    int_168682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 16), 'int')
    # Applying the binary operator '==' (line 569)
    result_eq_168683 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 9), '==', lc2_168681, int_168682)
    
    # Testing the type of an if condition (line 569)
    if_condition_168684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 569, 9), result_eq_168683)
    # Assigning a type to the variable 'if_condition_168684' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 9), 'if_condition_168684', if_condition_168684)
    # SSA begins for if statement (line 569)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 570)
    tuple_168685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 570)
    # Adding element type (line 570)
    # Getting the type of 'c1' (line 570)
    c1_168686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 15), 'c1')
    
    # Obtaining the type of the subscript
    int_168687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 21), 'int')
    # Getting the type of 'c2' (line 570)
    c2_168688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 18), 'c2')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___168689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 18), c2_168688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_168690 = invoke(stypy.reporting.localization.Localization(__file__, 570, 18), getitem___168689, int_168687)
    
    # Applying the binary operator 'div' (line 570)
    result_div_168691 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 15), 'div', c1_168686, subscript_call_result_168690)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 15), tuple_168685, result_div_168691)
    # Adding element type (line 570)
    
    # Obtaining the type of the subscript
    int_168692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 30), 'int')
    slice_168693 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 570, 26), None, int_168692, None)
    # Getting the type of 'c1' (line 570)
    c1_168694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 26), 'c1')
    # Obtaining the member '__getitem__' of a type (line 570)
    getitem___168695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 26), c1_168694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 570)
    subscript_call_result_168696 = invoke(stypy.reporting.localization.Localization(__file__, 570, 26), getitem___168695, slice_168693)
    
    int_168697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 33), 'int')
    # Applying the binary operator '*' (line 570)
    result_mul_168698 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 26), '*', subscript_call_result_168696, int_168697)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 15), tuple_168685, result_mul_168698)
    
    # Assigning a type to the variable 'stypy_return_type' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'stypy_return_type', tuple_168685)
    # SSA branch for the else part of an if statement (line 569)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 572):
    
    # Assigning a Call to a Name (line 572):
    
    # Call to empty(...): (line 572)
    # Processing the call arguments (line 572)
    # Getting the type of 'lc1' (line 572)
    lc1_168701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 572)
    lc2_168702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 29), 'lc2', False)
    # Applying the binary operator '-' (line 572)
    result_sub_168703 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 23), '-', lc1_168701, lc2_168702)
    
    int_168704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 35), 'int')
    # Applying the binary operator '+' (line 572)
    result_add_168705 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 33), '+', result_sub_168703, int_168704)
    
    # Processing the call keyword arguments (line 572)
    # Getting the type of 'c1' (line 572)
    c1_168706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 44), 'c1', False)
    # Obtaining the member 'dtype' of a type (line 572)
    dtype_168707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 44), c1_168706, 'dtype')
    keyword_168708 = dtype_168707
    kwargs_168709 = {'dtype': keyword_168708}
    # Getting the type of 'np' (line 572)
    np_168699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 14), 'np', False)
    # Obtaining the member 'empty' of a type (line 572)
    empty_168700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 14), np_168699, 'empty')
    # Calling empty(args, kwargs) (line 572)
    empty_call_result_168710 = invoke(stypy.reporting.localization.Localization(__file__, 572, 14), empty_168700, *[result_add_168705], **kwargs_168709)
    
    # Assigning a type to the variable 'quo' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'quo', empty_call_result_168710)
    
    # Assigning a Name to a Name (line 573):
    
    # Assigning a Name to a Name (line 573):
    # Getting the type of 'c1' (line 573)
    c1_168711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'c1')
    # Assigning a type to the variable 'rem' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'rem', c1_168711)
    
    
    # Call to range(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'lc1' (line 574)
    lc1_168713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 23), 'lc1', False)
    # Getting the type of 'lc2' (line 574)
    lc2_168714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 29), 'lc2', False)
    # Applying the binary operator '-' (line 574)
    result_sub_168715 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 23), '-', lc1_168713, lc2_168714)
    
    int_168716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 34), 'int')
    int_168717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 39), 'int')
    # Processing the call keyword arguments (line 574)
    kwargs_168718 = {}
    # Getting the type of 'range' (line 574)
    range_168712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 17), 'range', False)
    # Calling range(args, kwargs) (line 574)
    range_call_result_168719 = invoke(stypy.reporting.localization.Localization(__file__, 574, 17), range_168712, *[result_sub_168715, int_168716, int_168717], **kwargs_168718)
    
    # Testing the type of a for loop iterable (line 574)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 574, 8), range_call_result_168719)
    # Getting the type of the for loop variable (line 574)
    for_loop_var_168720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 574, 8), range_call_result_168719)
    # Assigning a type to the variable 'i' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'i', for_loop_var_168720)
    # SSA begins for a for statement (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 575):
    
    # Assigning a Call to a Name (line 575):
    
    # Call to hermemul(...): (line 575)
    # Processing the call arguments (line 575)
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_168722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    # Adding element type (line 575)
    int_168723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), list_168722, int_168723)
    
    # Getting the type of 'i' (line 575)
    i_168724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 29), 'i', False)
    # Applying the binary operator '*' (line 575)
    result_mul_168725 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 25), '*', list_168722, i_168724)
    
    
    # Obtaining an instance of the builtin type 'list' (line 575)
    list_168726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 575)
    # Adding element type (line 575)
    int_168727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 33), list_168726, int_168727)
    
    # Applying the binary operator '+' (line 575)
    result_add_168728 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 25), '+', result_mul_168725, list_168726)
    
    # Getting the type of 'c2' (line 575)
    c2_168729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 38), 'c2', False)
    # Processing the call keyword arguments (line 575)
    kwargs_168730 = {}
    # Getting the type of 'hermemul' (line 575)
    hermemul_168721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 16), 'hermemul', False)
    # Calling hermemul(args, kwargs) (line 575)
    hermemul_call_result_168731 = invoke(stypy.reporting.localization.Localization(__file__, 575, 16), hermemul_168721, *[result_add_168728, c2_168729], **kwargs_168730)
    
    # Assigning a type to the variable 'p' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'p', hermemul_call_result_168731)
    
    # Assigning a BinOp to a Name (line 576):
    
    # Assigning a BinOp to a Name (line 576):
    
    # Obtaining the type of the subscript
    int_168732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 20), 'int')
    # Getting the type of 'rem' (line 576)
    rem_168733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'rem')
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___168734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), rem_168733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 576)
    subscript_call_result_168735 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), getitem___168734, int_168732)
    
    
    # Obtaining the type of the subscript
    int_168736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 26), 'int')
    # Getting the type of 'p' (line 576)
    p_168737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 24), 'p')
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___168738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 24), p_168737, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 576)
    subscript_call_result_168739 = invoke(stypy.reporting.localization.Localization(__file__, 576, 24), getitem___168738, int_168736)
    
    # Applying the binary operator 'div' (line 576)
    result_div_168740 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 16), 'div', subscript_call_result_168735, subscript_call_result_168739)
    
    # Assigning a type to the variable 'q' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'q', result_div_168740)
    
    # Assigning a BinOp to a Name (line 577):
    
    # Assigning a BinOp to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_168741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 23), 'int')
    slice_168742 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 18), None, int_168741, None)
    # Getting the type of 'rem' (line 577)
    rem_168743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 18), 'rem')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___168744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 18), rem_168743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_168745 = invoke(stypy.reporting.localization.Localization(__file__, 577, 18), getitem___168744, slice_168742)
    
    # Getting the type of 'q' (line 577)
    q_168746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 29), 'q')
    
    # Obtaining the type of the subscript
    int_168747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 34), 'int')
    slice_168748 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 31), None, int_168747, None)
    # Getting the type of 'p' (line 577)
    p_168749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'p')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___168750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 31), p_168749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_168751 = invoke(stypy.reporting.localization.Localization(__file__, 577, 31), getitem___168750, slice_168748)
    
    # Applying the binary operator '*' (line 577)
    result_mul_168752 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 29), '*', q_168746, subscript_call_result_168751)
    
    # Applying the binary operator '-' (line 577)
    result_sub_168753 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 18), '-', subscript_call_result_168745, result_mul_168752)
    
    # Assigning a type to the variable 'rem' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'rem', result_sub_168753)
    
    # Assigning a Name to a Subscript (line 578):
    
    # Assigning a Name to a Subscript (line 578):
    # Getting the type of 'q' (line 578)
    q_168754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 21), 'q')
    # Getting the type of 'quo' (line 578)
    quo_168755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'quo')
    # Getting the type of 'i' (line 578)
    i_168756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 16), 'i')
    # Storing an element on a container (line 578)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 12), quo_168755, (i_168756, q_168754))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 579)
    tuple_168757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 579)
    # Adding element type (line 579)
    # Getting the type of 'quo' (line 579)
    quo_168758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 15), 'quo')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 15), tuple_168757, quo_168758)
    # Adding element type (line 579)
    
    # Call to trimseq(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'rem' (line 579)
    rem_168761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'rem', False)
    # Processing the call keyword arguments (line 579)
    kwargs_168762 = {}
    # Getting the type of 'pu' (line 579)
    pu_168759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 20), 'pu', False)
    # Obtaining the member 'trimseq' of a type (line 579)
    trimseq_168760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 20), pu_168759, 'trimseq')
    # Calling trimseq(args, kwargs) (line 579)
    trimseq_call_result_168763 = invoke(stypy.reporting.localization.Localization(__file__, 579, 20), trimseq_168760, *[rem_168761], **kwargs_168762)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 15), tuple_168757, trimseq_call_result_168763)
    
    # Assigning a type to the variable 'stypy_return_type' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'stypy_return_type', tuple_168757)
    # SSA join for if statement (line 569)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermediv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermediv' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_168764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermediv'
    return stypy_return_type_168764

# Assigning a type to the variable 'hermediv' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'hermediv', hermediv)

@norecursion
def hermepow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_168765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 30), 'int')
    defaults = [int_168765]
    # Create a new context for function 'hermepow'
    module_type_store = module_type_store.open_function_context('hermepow', 582, 0, False)
    
    # Passed parameters checking function
    hermepow.stypy_localization = localization
    hermepow.stypy_type_of_self = None
    hermepow.stypy_type_store = module_type_store
    hermepow.stypy_function_name = 'hermepow'
    hermepow.stypy_param_names_list = ['c', 'pow', 'maxpower']
    hermepow.stypy_varargs_param_name = None
    hermepow.stypy_kwargs_param_name = None
    hermepow.stypy_call_defaults = defaults
    hermepow.stypy_call_varargs = varargs
    hermepow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermepow', ['c', 'pow', 'maxpower'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermepow', localization, ['c', 'pow', 'maxpower'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermepow(...)' code ##################

    str_168766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, (-1)), 'str', 'Raise a Hermite series to a power.\n\n    Returns the Hermite series `c` raised to the power `pow`. The\n    argument `c` is a sequence of coefficients ordered from low to high.\n    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of Hermite series coefficients ordered from low to\n        high.\n    pow : integer\n        Power to which the series will be raised\n    maxpower : integer, optional\n        Maximum power allowed. This is mainly to limit growth of the series\n        to unmanageable size. Default is 16\n\n    Returns\n    -------\n    coef : ndarray\n        Hermite series of power.\n\n    See Also\n    --------\n    hermeadd, hermesub, hermemul, hermediv\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermepow\n    >>> hermepow([1, 2, 3], 2)\n    array([ 23.,  28.,  46.,  12.,   9.])\n\n    ')
    
    # Assigning a Call to a List (line 617):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Obtaining an instance of the builtin type 'list' (line 617)
    list_168769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 617)
    # Adding element type (line 617)
    # Getting the type of 'c' (line 617)
    c_168770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 23), list_168769, c_168770)
    
    # Processing the call keyword arguments (line 617)
    kwargs_168771 = {}
    # Getting the type of 'pu' (line 617)
    pu_168767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 617)
    as_series_168768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 10), pu_168767, 'as_series')
    # Calling as_series(args, kwargs) (line 617)
    as_series_call_result_168772 = invoke(stypy.reporting.localization.Localization(__file__, 617, 10), as_series_168768, *[list_168769], **kwargs_168771)
    
    # Assigning a type to the variable 'call_assignment_167935' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'call_assignment_167935', as_series_call_result_168772)
    
    # Assigning a Call to a Name (line 617):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_168775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168776 = {}
    # Getting the type of 'call_assignment_167935' (line 617)
    call_assignment_167935_168773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'call_assignment_167935', False)
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___168774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 4), call_assignment_167935_168773, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_168777 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___168774, *[int_168775], **kwargs_168776)
    
    # Assigning a type to the variable 'call_assignment_167936' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'call_assignment_167936', getitem___call_result_168777)
    
    # Assigning a Name to a Name (line 617):
    # Getting the type of 'call_assignment_167936' (line 617)
    call_assignment_167936_168778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'call_assignment_167936')
    # Assigning a type to the variable 'c' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 5), 'c', call_assignment_167936_168778)
    
    # Assigning a Call to a Name (line 618):
    
    # Assigning a Call to a Name (line 618):
    
    # Call to int(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'pow' (line 618)
    pow_168780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'pow', False)
    # Processing the call keyword arguments (line 618)
    kwargs_168781 = {}
    # Getting the type of 'int' (line 618)
    int_168779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'int', False)
    # Calling int(args, kwargs) (line 618)
    int_call_result_168782 = invoke(stypy.reporting.localization.Localization(__file__, 618, 12), int_168779, *[pow_168780], **kwargs_168781)
    
    # Assigning a type to the variable 'power' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'power', int_call_result_168782)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'power' (line 619)
    power_168783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 7), 'power')
    # Getting the type of 'pow' (line 619)
    pow_168784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'pow')
    # Applying the binary operator '!=' (line 619)
    result_ne_168785 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 7), '!=', power_168783, pow_168784)
    
    
    # Getting the type of 'power' (line 619)
    power_168786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 23), 'power')
    int_168787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 31), 'int')
    # Applying the binary operator '<' (line 619)
    result_lt_168788 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 23), '<', power_168786, int_168787)
    
    # Applying the binary operator 'or' (line 619)
    result_or_keyword_168789 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 7), 'or', result_ne_168785, result_lt_168788)
    
    # Testing the type of an if condition (line 619)
    if_condition_168790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 4), result_or_keyword_168789)
    # Assigning a type to the variable 'if_condition_168790' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'if_condition_168790', if_condition_168790)
    # SSA begins for if statement (line 619)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 620)
    # Processing the call arguments (line 620)
    str_168792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 25), 'str', 'Power must be a non-negative integer.')
    # Processing the call keyword arguments (line 620)
    kwargs_168793 = {}
    # Getting the type of 'ValueError' (line 620)
    ValueError_168791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 620)
    ValueError_call_result_168794 = invoke(stypy.reporting.localization.Localization(__file__, 620, 14), ValueError_168791, *[str_168792], **kwargs_168793)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 620, 8), ValueError_call_result_168794, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 619)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxpower' (line 621)
    maxpower_168795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 9), 'maxpower')
    # Getting the type of 'None' (line 621)
    None_168796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 25), 'None')
    # Applying the binary operator 'isnot' (line 621)
    result_is_not_168797 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 9), 'isnot', maxpower_168795, None_168796)
    
    
    # Getting the type of 'power' (line 621)
    power_168798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 34), 'power')
    # Getting the type of 'maxpower' (line 621)
    maxpower_168799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 42), 'maxpower')
    # Applying the binary operator '>' (line 621)
    result_gt_168800 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 34), '>', power_168798, maxpower_168799)
    
    # Applying the binary operator 'and' (line 621)
    result_and_keyword_168801 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 9), 'and', result_is_not_168797, result_gt_168800)
    
    # Testing the type of an if condition (line 621)
    if_condition_168802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 9), result_and_keyword_168801)
    # Assigning a type to the variable 'if_condition_168802' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 9), 'if_condition_168802', if_condition_168802)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 622)
    # Processing the call arguments (line 622)
    str_168804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 25), 'str', 'Power is too large')
    # Processing the call keyword arguments (line 622)
    kwargs_168805 = {}
    # Getting the type of 'ValueError' (line 622)
    ValueError_168803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 622)
    ValueError_call_result_168806 = invoke(stypy.reporting.localization.Localization(__file__, 622, 14), ValueError_168803, *[str_168804], **kwargs_168805)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 622, 8), ValueError_call_result_168806, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 621)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 623)
    power_168807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'power')
    int_168808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 18), 'int')
    # Applying the binary operator '==' (line 623)
    result_eq_168809 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 9), '==', power_168807, int_168808)
    
    # Testing the type of an if condition (line 623)
    if_condition_168810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 9), result_eq_168809)
    # Assigning a type to the variable 'if_condition_168810' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'if_condition_168810', if_condition_168810)
    # SSA begins for if statement (line 623)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 624)
    # Processing the call arguments (line 624)
    
    # Obtaining an instance of the builtin type 'list' (line 624)
    list_168813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 624)
    # Adding element type (line 624)
    int_168814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 24), list_168813, int_168814)
    
    # Processing the call keyword arguments (line 624)
    # Getting the type of 'c' (line 624)
    c_168815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 35), 'c', False)
    # Obtaining the member 'dtype' of a type (line 624)
    dtype_168816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 35), c_168815, 'dtype')
    keyword_168817 = dtype_168816
    kwargs_168818 = {'dtype': keyword_168817}
    # Getting the type of 'np' (line 624)
    np_168811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 624)
    array_168812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 15), np_168811, 'array')
    # Calling array(args, kwargs) (line 624)
    array_call_result_168819 = invoke(stypy.reporting.localization.Localization(__file__, 624, 15), array_168812, *[list_168813], **kwargs_168818)
    
    # Assigning a type to the variable 'stypy_return_type' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'stypy_return_type', array_call_result_168819)
    # SSA branch for the else part of an if statement (line 623)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'power' (line 625)
    power_168820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 9), 'power')
    int_168821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 18), 'int')
    # Applying the binary operator '==' (line 625)
    result_eq_168822 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 9), '==', power_168820, int_168821)
    
    # Testing the type of an if condition (line 625)
    if_condition_168823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 9), result_eq_168822)
    # Assigning a type to the variable 'if_condition_168823' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 9), 'if_condition_168823', if_condition_168823)
    # SSA begins for if statement (line 625)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 626)
    c_168824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'stypy_return_type', c_168824)
    # SSA branch for the else part of an if statement (line 625)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 630):
    
    # Assigning a Name to a Name (line 630):
    # Getting the type of 'c' (line 630)
    c_168825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 14), 'c')
    # Assigning a type to the variable 'prd' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 8), 'prd', c_168825)
    
    
    # Call to range(...): (line 631)
    # Processing the call arguments (line 631)
    int_168827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 23), 'int')
    # Getting the type of 'power' (line 631)
    power_168828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 26), 'power', False)
    int_168829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 34), 'int')
    # Applying the binary operator '+' (line 631)
    result_add_168830 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 26), '+', power_168828, int_168829)
    
    # Processing the call keyword arguments (line 631)
    kwargs_168831 = {}
    # Getting the type of 'range' (line 631)
    range_168826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 17), 'range', False)
    # Calling range(args, kwargs) (line 631)
    range_call_result_168832 = invoke(stypy.reporting.localization.Localization(__file__, 631, 17), range_168826, *[int_168827, result_add_168830], **kwargs_168831)
    
    # Testing the type of a for loop iterable (line 631)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 631, 8), range_call_result_168832)
    # Getting the type of the for loop variable (line 631)
    for_loop_var_168833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 631, 8), range_call_result_168832)
    # Assigning a type to the variable 'i' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'i', for_loop_var_168833)
    # SSA begins for a for statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 632):
    
    # Assigning a Call to a Name (line 632):
    
    # Call to hermemul(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'prd' (line 632)
    prd_168835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 27), 'prd', False)
    # Getting the type of 'c' (line 632)
    c_168836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 32), 'c', False)
    # Processing the call keyword arguments (line 632)
    kwargs_168837 = {}
    # Getting the type of 'hermemul' (line 632)
    hermemul_168834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 18), 'hermemul', False)
    # Calling hermemul(args, kwargs) (line 632)
    hermemul_call_result_168838 = invoke(stypy.reporting.localization.Localization(__file__, 632, 18), hermemul_168834, *[prd_168835, c_168836], **kwargs_168837)
    
    # Assigning a type to the variable 'prd' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'prd', hermemul_call_result_168838)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'prd' (line 633)
    prd_168839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 15), 'prd')
    # Assigning a type to the variable 'stypy_return_type' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'stypy_return_type', prd_168839)
    # SSA join for if statement (line 625)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 623)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 619)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermepow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermepow' in the type store
    # Getting the type of 'stypy_return_type' (line 582)
    stypy_return_type_168840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168840)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermepow'
    return stypy_return_type_168840

# Assigning a type to the variable 'hermepow' (line 582)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 0), 'hermepow', hermepow)

@norecursion
def hermeder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_168841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 18), 'int')
    int_168842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 25), 'int')
    int_168843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 33), 'int')
    defaults = [int_168841, int_168842, int_168843]
    # Create a new context for function 'hermeder'
    module_type_store = module_type_store.open_function_context('hermeder', 636, 0, False)
    
    # Passed parameters checking function
    hermeder.stypy_localization = localization
    hermeder.stypy_type_of_self = None
    hermeder.stypy_type_store = module_type_store
    hermeder.stypy_function_name = 'hermeder'
    hermeder.stypy_param_names_list = ['c', 'm', 'scl', 'axis']
    hermeder.stypy_varargs_param_name = None
    hermeder.stypy_kwargs_param_name = None
    hermeder.stypy_call_defaults = defaults
    hermeder.stypy_call_varargs = varargs
    hermeder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeder', ['c', 'm', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeder', localization, ['c', 'm', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeder(...)' code ##################

    str_168844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, (-1)), 'str', '\n    Differentiate a Hermite_e series.\n\n    Returns the series coefficients `c` differentiated `m` times along\n    `axis`.  At each iteration the result is multiplied by `scl` (the\n    scaling factor is for use in a linear change of variable). The argument\n    `c` is an array of coefficients from low to high degree along each\n    axis, e.g., [1,2,3] represents the series ``1*He_0 + 2*He_1 + 3*He_2``\n    while [[1,2],[1,2]] represents ``1*He_0(x)*He_0(y) + 1*He_1(x)*He_0(y)\n    + 2*He_0(x)*He_1(y) + 2*He_1(x)*He_1(y)`` if axis=0 is ``x`` and axis=1\n    is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Hermite_e series coefficients. If `c` is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Number of derivatives taken, must be non-negative. (Default: 1)\n    scl : scalar, optional\n        Each differentiation is multiplied by `scl`.  The end result is\n        multiplication by ``scl**m``.  This is for use in a linear change of\n        variable. (Default: 1)\n    axis : int, optional\n        Axis over which the derivative is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    der : ndarray\n        Hermite series of the derivative.\n\n    See Also\n    --------\n    hermeint\n\n    Notes\n    -----\n    In general, the result of differentiating a Hermite series does not\n    resemble the same operation on a power series. Thus the result of this\n    function may be "unintuitive," albeit correct; see Examples section\n    below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeder\n    >>> hermeder([ 1.,  1.,  1.,  1.])\n    array([ 1.,  2.,  3.])\n    >>> hermeder([-0.25,  1.,  1./2.,  1./3.,  1./4 ], m=2)\n    array([ 1.,  2.,  3.])\n\n    ')
    
    # Assigning a Call to a Name (line 691):
    
    # Assigning a Call to a Name (line 691):
    
    # Call to array(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'c' (line 691)
    c_168847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 17), 'c', False)
    # Processing the call keyword arguments (line 691)
    int_168848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 26), 'int')
    keyword_168849 = int_168848
    int_168850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 34), 'int')
    keyword_168851 = int_168850
    kwargs_168852 = {'copy': keyword_168851, 'ndmin': keyword_168849}
    # Getting the type of 'np' (line 691)
    np_168845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 691)
    array_168846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 8), np_168845, 'array')
    # Calling array(args, kwargs) (line 691)
    array_call_result_168853 = invoke(stypy.reporting.localization.Localization(__file__, 691, 8), array_168846, *[c_168847], **kwargs_168852)
    
    # Assigning a type to the variable 'c' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 4), 'c', array_call_result_168853)
    
    
    # Getting the type of 'c' (line 692)
    c_168854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 692)
    dtype_168855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 7), c_168854, 'dtype')
    # Obtaining the member 'char' of a type (line 692)
    char_168856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 7), dtype_168855, 'char')
    str_168857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 692)
    result_contains_168858 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 7), 'in', char_168856, str_168857)
    
    # Testing the type of an if condition (line 692)
    if_condition_168859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 4), result_contains_168858)
    # Assigning a type to the variable 'if_condition_168859' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'if_condition_168859', if_condition_168859)
    # SSA begins for if statement (line 692)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 693):
    
    # Assigning a Call to a Name (line 693):
    
    # Call to astype(...): (line 693)
    # Processing the call arguments (line 693)
    # Getting the type of 'np' (line 693)
    np_168862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 693)
    double_168863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 21), np_168862, 'double')
    # Processing the call keyword arguments (line 693)
    kwargs_168864 = {}
    # Getting the type of 'c' (line 693)
    c_168860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 693)
    astype_168861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 12), c_168860, 'astype')
    # Calling astype(args, kwargs) (line 693)
    astype_call_result_168865 = invoke(stypy.reporting.localization.Localization(__file__, 693, 12), astype_168861, *[double_168863], **kwargs_168864)
    
    # Assigning a type to the variable 'c' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'c', astype_call_result_168865)
    # SSA join for if statement (line 692)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_168866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 694)
    list_168871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 694)
    # Adding element type (line 694)
    # Getting the type of 'm' (line 694)
    m_168872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 34), list_168871, m_168872)
    # Adding element type (line 694)
    # Getting the type of 'axis' (line 694)
    axis_168873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 34), list_168871, axis_168873)
    
    comprehension_168874 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 18), list_168871)
    # Assigning a type to the variable 't' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 18), 't', comprehension_168874)
    
    # Call to int(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 't' (line 694)
    t_168868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 22), 't', False)
    # Processing the call keyword arguments (line 694)
    kwargs_168869 = {}
    # Getting the type of 'int' (line 694)
    int_168867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 18), 'int', False)
    # Calling int(args, kwargs) (line 694)
    int_call_result_168870 = invoke(stypy.reporting.localization.Localization(__file__, 694, 18), int_168867, *[t_168868], **kwargs_168869)
    
    list_168875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 18), list_168875, int_call_result_168870)
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___168876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), list_168875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_168877 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___168876, int_168866)
    
    # Assigning a type to the variable 'tuple_var_assignment_167937' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_167937', subscript_call_result_168877)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_168878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 694)
    list_168883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 694)
    # Adding element type (line 694)
    # Getting the type of 'm' (line 694)
    m_168884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 34), list_168883, m_168884)
    # Adding element type (line 694)
    # Getting the type of 'axis' (line 694)
    axis_168885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 34), list_168883, axis_168885)
    
    comprehension_168886 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 18), list_168883)
    # Assigning a type to the variable 't' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 18), 't', comprehension_168886)
    
    # Call to int(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 't' (line 694)
    t_168880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 22), 't', False)
    # Processing the call keyword arguments (line 694)
    kwargs_168881 = {}
    # Getting the type of 'int' (line 694)
    int_168879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 18), 'int', False)
    # Calling int(args, kwargs) (line 694)
    int_call_result_168882 = invoke(stypy.reporting.localization.Localization(__file__, 694, 18), int_168879, *[t_168880], **kwargs_168881)
    
    list_168887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 18), list_168887, int_call_result_168882)
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___168888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), list_168887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_168889 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___168888, int_168878)
    
    # Assigning a type to the variable 'tuple_var_assignment_167938' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_167938', subscript_call_result_168889)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_167937' (line 694)
    tuple_var_assignment_167937_168890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_167937')
    # Assigning a type to the variable 'cnt' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'cnt', tuple_var_assignment_167937_168890)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_167938' (line 694)
    tuple_var_assignment_167938_168891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_167938')
    # Assigning a type to the variable 'iaxis' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 9), 'iaxis', tuple_var_assignment_167938_168891)
    
    
    # Getting the type of 'cnt' (line 696)
    cnt_168892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 7), 'cnt')
    # Getting the type of 'm' (line 696)
    m_168893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 14), 'm')
    # Applying the binary operator '!=' (line 696)
    result_ne_168894 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 7), '!=', cnt_168892, m_168893)
    
    # Testing the type of an if condition (line 696)
    if_condition_168895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 696, 4), result_ne_168894)
    # Assigning a type to the variable 'if_condition_168895' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 4), 'if_condition_168895', if_condition_168895)
    # SSA begins for if statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 697)
    # Processing the call arguments (line 697)
    str_168897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 25), 'str', 'The order of derivation must be integer')
    # Processing the call keyword arguments (line 697)
    kwargs_168898 = {}
    # Getting the type of 'ValueError' (line 697)
    ValueError_168896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 697)
    ValueError_call_result_168899 = invoke(stypy.reporting.localization.Localization(__file__, 697, 14), ValueError_168896, *[str_168897], **kwargs_168898)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 697, 8), ValueError_call_result_168899, 'raise parameter', BaseException)
    # SSA join for if statement (line 696)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 698)
    cnt_168900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 7), 'cnt')
    int_168901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 13), 'int')
    # Applying the binary operator '<' (line 698)
    result_lt_168902 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 7), '<', cnt_168900, int_168901)
    
    # Testing the type of an if condition (line 698)
    if_condition_168903 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 698, 4), result_lt_168902)
    # Assigning a type to the variable 'if_condition_168903' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'if_condition_168903', if_condition_168903)
    # SSA begins for if statement (line 698)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 699)
    # Processing the call arguments (line 699)
    str_168905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 25), 'str', 'The order of derivation must be non-negative')
    # Processing the call keyword arguments (line 699)
    kwargs_168906 = {}
    # Getting the type of 'ValueError' (line 699)
    ValueError_168904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 699)
    ValueError_call_result_168907 = invoke(stypy.reporting.localization.Localization(__file__, 699, 14), ValueError_168904, *[str_168905], **kwargs_168906)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 699, 8), ValueError_call_result_168907, 'raise parameter', BaseException)
    # SSA join for if statement (line 698)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 700)
    iaxis_168908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 7), 'iaxis')
    # Getting the type of 'axis' (line 700)
    axis_168909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'axis')
    # Applying the binary operator '!=' (line 700)
    result_ne_168910 = python_operator(stypy.reporting.localization.Localization(__file__, 700, 7), '!=', iaxis_168908, axis_168909)
    
    # Testing the type of an if condition (line 700)
    if_condition_168911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 700, 4), result_ne_168910)
    # Assigning a type to the variable 'if_condition_168911' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'if_condition_168911', if_condition_168911)
    # SSA begins for if statement (line 700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 701)
    # Processing the call arguments (line 701)
    str_168913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 701)
    kwargs_168914 = {}
    # Getting the type of 'ValueError' (line 701)
    ValueError_168912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 701)
    ValueError_call_result_168915 = invoke(stypy.reporting.localization.Localization(__file__, 701, 14), ValueError_168912, *[str_168913], **kwargs_168914)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 701, 8), ValueError_call_result_168915, 'raise parameter', BaseException)
    # SSA join for if statement (line 700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 702)
    c_168916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 702)
    ndim_168917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 12), c_168916, 'ndim')
    # Applying the 'usub' unary operator (line 702)
    result___neg___168918 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 11), 'usub', ndim_168917)
    
    # Getting the type of 'iaxis' (line 702)
    iaxis_168919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 22), 'iaxis')
    # Applying the binary operator '<=' (line 702)
    result_le_168920 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 11), '<=', result___neg___168918, iaxis_168919)
    # Getting the type of 'c' (line 702)
    c_168921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 702)
    ndim_168922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 30), c_168921, 'ndim')
    # Applying the binary operator '<' (line 702)
    result_lt_168923 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 11), '<', iaxis_168919, ndim_168922)
    # Applying the binary operator '&' (line 702)
    result_and__168924 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 11), '&', result_le_168920, result_lt_168923)
    
    # Applying the 'not' unary operator (line 702)
    result_not__168925 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 7), 'not', result_and__168924)
    
    # Testing the type of an if condition (line 702)
    if_condition_168926 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 702, 4), result_not__168925)
    # Assigning a type to the variable 'if_condition_168926' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 4), 'if_condition_168926', if_condition_168926)
    # SSA begins for if statement (line 702)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 703)
    # Processing the call arguments (line 703)
    str_168928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 703, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 703)
    kwargs_168929 = {}
    # Getting the type of 'ValueError' (line 703)
    ValueError_168927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 703)
    ValueError_call_result_168930 = invoke(stypy.reporting.localization.Localization(__file__, 703, 14), ValueError_168927, *[str_168928], **kwargs_168929)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 703, 8), ValueError_call_result_168930, 'raise parameter', BaseException)
    # SSA join for if statement (line 702)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 704)
    iaxis_168931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 7), 'iaxis')
    int_168932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 15), 'int')
    # Applying the binary operator '<' (line 704)
    result_lt_168933 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 7), '<', iaxis_168931, int_168932)
    
    # Testing the type of an if condition (line 704)
    if_condition_168934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 704, 4), result_lt_168933)
    # Assigning a type to the variable 'if_condition_168934' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'if_condition_168934', if_condition_168934)
    # SSA begins for if statement (line 704)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 705)
    iaxis_168935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'iaxis')
    # Getting the type of 'c' (line 705)
    c_168936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 705)
    ndim_168937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 17), c_168936, 'ndim')
    # Applying the binary operator '+=' (line 705)
    result_iadd_168938 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 8), '+=', iaxis_168935, ndim_168937)
    # Assigning a type to the variable 'iaxis' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'iaxis', result_iadd_168938)
    
    # SSA join for if statement (line 704)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 707)
    cnt_168939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 7), 'cnt')
    int_168940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 707, 14), 'int')
    # Applying the binary operator '==' (line 707)
    result_eq_168941 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 7), '==', cnt_168939, int_168940)
    
    # Testing the type of an if condition (line 707)
    if_condition_168942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 707, 4), result_eq_168941)
    # Assigning a type to the variable 'if_condition_168942' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 4), 'if_condition_168942', if_condition_168942)
    # SSA begins for if statement (line 707)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 708)
    c_168943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'stypy_return_type', c_168943)
    # SSA join for if statement (line 707)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 710):
    
    # Assigning a Call to a Name (line 710):
    
    # Call to rollaxis(...): (line 710)
    # Processing the call arguments (line 710)
    # Getting the type of 'c' (line 710)
    c_168946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 20), 'c', False)
    # Getting the type of 'iaxis' (line 710)
    iaxis_168947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 710)
    kwargs_168948 = {}
    # Getting the type of 'np' (line 710)
    np_168944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 710)
    rollaxis_168945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 8), np_168944, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 710)
    rollaxis_call_result_168949 = invoke(stypy.reporting.localization.Localization(__file__, 710, 8), rollaxis_168945, *[c_168946, iaxis_168947], **kwargs_168948)
    
    # Assigning a type to the variable 'c' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 4), 'c', rollaxis_call_result_168949)
    
    # Assigning a Call to a Name (line 711):
    
    # Assigning a Call to a Name (line 711):
    
    # Call to len(...): (line 711)
    # Processing the call arguments (line 711)
    # Getting the type of 'c' (line 711)
    c_168951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'c', False)
    # Processing the call keyword arguments (line 711)
    kwargs_168952 = {}
    # Getting the type of 'len' (line 711)
    len_168950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'len', False)
    # Calling len(args, kwargs) (line 711)
    len_call_result_168953 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), len_168950, *[c_168951], **kwargs_168952)
    
    # Assigning a type to the variable 'n' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 4), 'n', len_call_result_168953)
    
    
    # Getting the type of 'cnt' (line 712)
    cnt_168954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 7), 'cnt')
    # Getting the type of 'n' (line 712)
    n_168955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 14), 'n')
    # Applying the binary operator '>=' (line 712)
    result_ge_168956 = python_operator(stypy.reporting.localization.Localization(__file__, 712, 7), '>=', cnt_168954, n_168955)
    
    # Testing the type of an if condition (line 712)
    if_condition_168957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 712, 4), result_ge_168956)
    # Assigning a type to the variable 'if_condition_168957' (line 712)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 712, 4), 'if_condition_168957', if_condition_168957)
    # SSA begins for if statement (line 712)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_168958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 18), 'int')
    slice_168959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 713, 15), None, int_168958, None)
    # Getting the type of 'c' (line 713)
    c_168960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 15), 'c')
    # Obtaining the member '__getitem__' of a type (line 713)
    getitem___168961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 713, 15), c_168960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 713)
    subscript_call_result_168962 = invoke(stypy.reporting.localization.Localization(__file__, 713, 15), getitem___168961, slice_168959)
    
    int_168963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 21), 'int')
    # Applying the binary operator '*' (line 713)
    result_mul_168964 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 15), '*', subscript_call_result_168962, int_168963)
    
    # Assigning a type to the variable 'stypy_return_type' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'stypy_return_type', result_mul_168964)
    # SSA branch for the else part of an if statement (line 712)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to range(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 'cnt' (line 715)
    cnt_168966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 23), 'cnt', False)
    # Processing the call keyword arguments (line 715)
    kwargs_168967 = {}
    # Getting the type of 'range' (line 715)
    range_168965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 17), 'range', False)
    # Calling range(args, kwargs) (line 715)
    range_call_result_168968 = invoke(stypy.reporting.localization.Localization(__file__, 715, 17), range_168965, *[cnt_168966], **kwargs_168967)
    
    # Testing the type of a for loop iterable (line 715)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 715, 8), range_call_result_168968)
    # Getting the type of the for loop variable (line 715)
    for_loop_var_168969 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 715, 8), range_call_result_168968)
    # Assigning a type to the variable 'i' (line 715)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'i', for_loop_var_168969)
    # SSA begins for a for statement (line 715)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 716):
    
    # Assigning a BinOp to a Name (line 716):
    # Getting the type of 'n' (line 716)
    n_168970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 16), 'n')
    int_168971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 20), 'int')
    # Applying the binary operator '-' (line 716)
    result_sub_168972 = python_operator(stypy.reporting.localization.Localization(__file__, 716, 16), '-', n_168970, int_168971)
    
    # Assigning a type to the variable 'n' (line 716)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 716, 12), 'n', result_sub_168972)
    
    # Getting the type of 'c' (line 717)
    c_168973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'c')
    # Getting the type of 'scl' (line 717)
    scl_168974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 17), 'scl')
    # Applying the binary operator '*=' (line 717)
    result_imul_168975 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 12), '*=', c_168973, scl_168974)
    # Assigning a type to the variable 'c' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 12), 'c', result_imul_168975)
    
    
    # Assigning a Call to a Name (line 718):
    
    # Assigning a Call to a Name (line 718):
    
    # Call to empty(...): (line 718)
    # Processing the call arguments (line 718)
    
    # Obtaining an instance of the builtin type 'tuple' (line 718)
    tuple_168978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 718)
    # Adding element type (line 718)
    # Getting the type of 'n' (line 718)
    n_168979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 28), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 28), tuple_168978, n_168979)
    
    
    # Obtaining the type of the subscript
    int_168980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 42), 'int')
    slice_168981 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 718, 34), int_168980, None, None)
    # Getting the type of 'c' (line 718)
    c_168982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 34), 'c', False)
    # Obtaining the member 'shape' of a type (line 718)
    shape_168983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 34), c_168982, 'shape')
    # Obtaining the member '__getitem__' of a type (line 718)
    getitem___168984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 34), shape_168983, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 718)
    subscript_call_result_168985 = invoke(stypy.reporting.localization.Localization(__file__, 718, 34), getitem___168984, slice_168981)
    
    # Applying the binary operator '+' (line 718)
    result_add_168986 = python_operator(stypy.reporting.localization.Localization(__file__, 718, 27), '+', tuple_168978, subscript_call_result_168985)
    
    # Processing the call keyword arguments (line 718)
    # Getting the type of 'c' (line 718)
    c_168987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 53), 'c', False)
    # Obtaining the member 'dtype' of a type (line 718)
    dtype_168988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 53), c_168987, 'dtype')
    keyword_168989 = dtype_168988
    kwargs_168990 = {'dtype': keyword_168989}
    # Getting the type of 'np' (line 718)
    np_168976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 718)
    empty_168977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 718, 18), np_168976, 'empty')
    # Calling empty(args, kwargs) (line 718)
    empty_call_result_168991 = invoke(stypy.reporting.localization.Localization(__file__, 718, 18), empty_168977, *[result_add_168986], **kwargs_168990)
    
    # Assigning a type to the variable 'der' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 12), 'der', empty_call_result_168991)
    
    
    # Call to range(...): (line 719)
    # Processing the call arguments (line 719)
    # Getting the type of 'n' (line 719)
    n_168993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 27), 'n', False)
    int_168994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 30), 'int')
    int_168995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 33), 'int')
    # Processing the call keyword arguments (line 719)
    kwargs_168996 = {}
    # Getting the type of 'range' (line 719)
    range_168992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 21), 'range', False)
    # Calling range(args, kwargs) (line 719)
    range_call_result_168997 = invoke(stypy.reporting.localization.Localization(__file__, 719, 21), range_168992, *[n_168993, int_168994, int_168995], **kwargs_168996)
    
    # Testing the type of a for loop iterable (line 719)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 719, 12), range_call_result_168997)
    # Getting the type of the for loop variable (line 719)
    for_loop_var_168998 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 719, 12), range_call_result_168997)
    # Assigning a type to the variable 'j' (line 719)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 719, 12), 'j', for_loop_var_168998)
    # SSA begins for a for statement (line 719)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 720):
    
    # Assigning a BinOp to a Subscript (line 720):
    # Getting the type of 'j' (line 720)
    j_168999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 29), 'j')
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 720)
    j_169000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 33), 'j')
    # Getting the type of 'c' (line 720)
    c_169001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 31), 'c')
    # Obtaining the member '__getitem__' of a type (line 720)
    getitem___169002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 720, 31), c_169001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 720)
    subscript_call_result_169003 = invoke(stypy.reporting.localization.Localization(__file__, 720, 31), getitem___169002, j_169000)
    
    # Applying the binary operator '*' (line 720)
    result_mul_169004 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 29), '*', j_168999, subscript_call_result_169003)
    
    # Getting the type of 'der' (line 720)
    der_169005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 16), 'der')
    # Getting the type of 'j' (line 720)
    j_169006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 20), 'j')
    int_169007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 24), 'int')
    # Applying the binary operator '-' (line 720)
    result_sub_169008 = python_operator(stypy.reporting.localization.Localization(__file__, 720, 20), '-', j_169006, int_169007)
    
    # Storing an element on a container (line 720)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 16), der_169005, (result_sub_169008, result_mul_169004))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 721):
    
    # Assigning a Name to a Name (line 721):
    # Getting the type of 'der' (line 721)
    der_169009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 16), 'der')
    # Assigning a type to the variable 'c' (line 721)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 12), 'c', der_169009)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 712)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 722):
    
    # Assigning a Call to a Name (line 722):
    
    # Call to rollaxis(...): (line 722)
    # Processing the call arguments (line 722)
    # Getting the type of 'c' (line 722)
    c_169012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 20), 'c', False)
    int_169013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 23), 'int')
    # Getting the type of 'iaxis' (line 722)
    iaxis_169014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 26), 'iaxis', False)
    int_169015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 34), 'int')
    # Applying the binary operator '+' (line 722)
    result_add_169016 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 26), '+', iaxis_169014, int_169015)
    
    # Processing the call keyword arguments (line 722)
    kwargs_169017 = {}
    # Getting the type of 'np' (line 722)
    np_169010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 722)
    rollaxis_169011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 722, 8), np_169010, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 722)
    rollaxis_call_result_169018 = invoke(stypy.reporting.localization.Localization(__file__, 722, 8), rollaxis_169011, *[c_169012, int_169013, result_add_169016], **kwargs_169017)
    
    # Assigning a type to the variable 'c' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'c', rollaxis_call_result_169018)
    # Getting the type of 'c' (line 723)
    c_169019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 4), 'stypy_return_type', c_169019)
    
    # ################# End of 'hermeder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeder' in the type store
    # Getting the type of 'stypy_return_type' (line 636)
    stypy_return_type_169020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169020)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeder'
    return stypy_return_type_169020

# Assigning a type to the variable 'hermeder' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'hermeder', hermeder)

@norecursion
def hermeint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_169021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 18), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 726)
    list_169022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 726)
    
    int_169023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 32), 'int')
    int_169024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 39), 'int')
    int_169025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 47), 'int')
    defaults = [int_169021, list_169022, int_169023, int_169024, int_169025]
    # Create a new context for function 'hermeint'
    module_type_store = module_type_store.open_function_context('hermeint', 726, 0, False)
    
    # Passed parameters checking function
    hermeint.stypy_localization = localization
    hermeint.stypy_type_of_self = None
    hermeint.stypy_type_store = module_type_store
    hermeint.stypy_function_name = 'hermeint'
    hermeint.stypy_param_names_list = ['c', 'm', 'k', 'lbnd', 'scl', 'axis']
    hermeint.stypy_varargs_param_name = None
    hermeint.stypy_kwargs_param_name = None
    hermeint.stypy_call_defaults = defaults
    hermeint.stypy_call_varargs = varargs
    hermeint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeint', ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeint', localization, ['c', 'm', 'k', 'lbnd', 'scl', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeint(...)' code ##################

    str_169026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, (-1)), 'str', '\n    Integrate a Hermite_e series.\n\n    Returns the Hermite_e series coefficients `c` integrated `m` times from\n    `lbnd` along `axis`. At each iteration the resulting series is\n    **multiplied** by `scl` and an integration constant, `k`, is added.\n    The scaling factor is for use in a linear change of variable.  ("Buyer\n    beware": note that, depending on what one is doing, one may want `scl`\n    to be the reciprocal of what one might expect; for more information,\n    see the Notes section below.)  The argument `c` is an array of\n    coefficients from low to high degree along each axis, e.g., [1,2,3]\n    represents the series ``H_0 + 2*H_1 + 3*H_2`` while [[1,2],[1,2]]\n    represents ``1*H_0(x)*H_0(y) + 1*H_1(x)*H_0(y) + 2*H_0(x)*H_1(y) +\n    2*H_1(x)*H_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.\n\n    Parameters\n    ----------\n    c : array_like\n        Array of Hermite_e series coefficients. If c is multidimensional\n        the different axis correspond to different variables with the\n        degree in each axis given by the corresponding index.\n    m : int, optional\n        Order of integration, must be positive. (Default: 1)\n    k : {[], list, scalar}, optional\n        Integration constant(s).  The value of the first integral at\n        ``lbnd`` is the first value in the list, the value of the second\n        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the\n        default), all constants are set to zero.  If ``m == 1``, a single\n        scalar can be given instead of a list.\n    lbnd : scalar, optional\n        The lower bound of the integral. (Default: 0)\n    scl : scalar, optional\n        Following each integration the result is *multiplied* by `scl`\n        before the integration constant is added. (Default: 1)\n    axis : int, optional\n        Axis over which the integral is taken. (Default: 0).\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    S : ndarray\n        Hermite_e series coefficients of the integral.\n\n    Raises\n    ------\n    ValueError\n        If ``m < 0``, ``len(k) > m``, ``np.isscalar(lbnd) == False``, or\n        ``np.isscalar(scl) == False``.\n\n    See Also\n    --------\n    hermeder\n\n    Notes\n    -----\n    Note that the result of each integration is *multiplied* by `scl`.\n    Why is this important to note?  Say one is making a linear change of\n    variable :math:`u = ax + b` in an integral relative to `x`.  Then\n    .. math::`dx = du/a`, so one will need to set `scl` equal to\n    :math:`1/a` - perhaps not what one would have first thought.\n\n    Also note that, in general, the result of integrating a C-series needs\n    to be "reprojected" onto the C-series basis set.  Thus, typically,\n    the result of this function is "unintuitive," albeit correct; see\n    Examples section below.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeint\n    >>> hermeint([1, 2, 3]) # integrate once, value 0 at 0.\n    array([ 1.,  1.,  1.,  1.])\n    >>> hermeint([1, 2, 3], m=2) # integrate twice, value & deriv 0 at 0\n    array([-0.25      ,  1.        ,  0.5       ,  0.33333333,  0.25      ])\n    >>> hermeint([1, 2, 3], k=1) # integrate once, value 1 at 0.\n    array([ 2.,  1.,  1.,  1.])\n    >>> hermeint([1, 2, 3], lbnd=-1) # integrate once, value 0 at -1\n    array([-1.,  1.,  1.,  1.])\n    >>> hermeint([1, 2, 3], m=2, k=[1, 2], lbnd=-1)\n    array([ 1.83333333,  0.        ,  0.5       ,  0.33333333,  0.25      ])\n\n    ')
    
    # Assigning a Call to a Name (line 809):
    
    # Assigning a Call to a Name (line 809):
    
    # Call to array(...): (line 809)
    # Processing the call arguments (line 809)
    # Getting the type of 'c' (line 809)
    c_169029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 17), 'c', False)
    # Processing the call keyword arguments (line 809)
    int_169030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 26), 'int')
    keyword_169031 = int_169030
    int_169032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 34), 'int')
    keyword_169033 = int_169032
    kwargs_169034 = {'copy': keyword_169033, 'ndmin': keyword_169031}
    # Getting the type of 'np' (line 809)
    np_169027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 809)
    array_169028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 8), np_169027, 'array')
    # Calling array(args, kwargs) (line 809)
    array_call_result_169035 = invoke(stypy.reporting.localization.Localization(__file__, 809, 8), array_169028, *[c_169029], **kwargs_169034)
    
    # Assigning a type to the variable 'c' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'c', array_call_result_169035)
    
    
    # Getting the type of 'c' (line 810)
    c_169036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 810)
    dtype_169037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 7), c_169036, 'dtype')
    # Obtaining the member 'char' of a type (line 810)
    char_169038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 7), dtype_169037, 'char')
    str_169039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 810, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 810)
    result_contains_169040 = python_operator(stypy.reporting.localization.Localization(__file__, 810, 7), 'in', char_169038, str_169039)
    
    # Testing the type of an if condition (line 810)
    if_condition_169041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 810, 4), result_contains_169040)
    # Assigning a type to the variable 'if_condition_169041' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'if_condition_169041', if_condition_169041)
    # SSA begins for if statement (line 810)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 811):
    
    # Assigning a Call to a Name (line 811):
    
    # Call to astype(...): (line 811)
    # Processing the call arguments (line 811)
    # Getting the type of 'np' (line 811)
    np_169044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 811)
    double_169045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 21), np_169044, 'double')
    # Processing the call keyword arguments (line 811)
    kwargs_169046 = {}
    # Getting the type of 'c' (line 811)
    c_169042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 811)
    astype_169043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 12), c_169042, 'astype')
    # Calling astype(args, kwargs) (line 811)
    astype_call_result_169047 = invoke(stypy.reporting.localization.Localization(__file__, 811, 12), astype_169043, *[double_169045], **kwargs_169046)
    
    # Assigning a type to the variable 'c' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 8), 'c', astype_call_result_169047)
    # SSA join for if statement (line 810)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to iterable(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'k' (line 812)
    k_169050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 23), 'k', False)
    # Processing the call keyword arguments (line 812)
    kwargs_169051 = {}
    # Getting the type of 'np' (line 812)
    np_169048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 11), 'np', False)
    # Obtaining the member 'iterable' of a type (line 812)
    iterable_169049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 11), np_169048, 'iterable')
    # Calling iterable(args, kwargs) (line 812)
    iterable_call_result_169052 = invoke(stypy.reporting.localization.Localization(__file__, 812, 11), iterable_169049, *[k_169050], **kwargs_169051)
    
    # Applying the 'not' unary operator (line 812)
    result_not__169053 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 7), 'not', iterable_call_result_169052)
    
    # Testing the type of an if condition (line 812)
    if_condition_169054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 812, 4), result_not__169053)
    # Assigning a type to the variable 'if_condition_169054' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'if_condition_169054', if_condition_169054)
    # SSA begins for if statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 813):
    
    # Assigning a List to a Name (line 813):
    
    # Obtaining an instance of the builtin type 'list' (line 813)
    list_169055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 813)
    # Adding element type (line 813)
    # Getting the type of 'k' (line 813)
    k_169056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 13), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 12), list_169055, k_169056)
    
    # Assigning a type to the variable 'k' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 8), 'k', list_169055)
    # SSA join for if statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Tuple (line 814):
    
    # Assigning a Subscript to a Name (line 814):
    
    # Obtaining the type of the subscript
    int_169057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 814)
    list_169062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 814)
    # Adding element type (line 814)
    # Getting the type of 'm' (line 814)
    m_169063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 34), list_169062, m_169063)
    # Adding element type (line 814)
    # Getting the type of 'axis' (line 814)
    axis_169064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 34), list_169062, axis_169064)
    
    comprehension_169065 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 18), list_169062)
    # Assigning a type to the variable 't' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 18), 't', comprehension_169065)
    
    # Call to int(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 't' (line 814)
    t_169059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 22), 't', False)
    # Processing the call keyword arguments (line 814)
    kwargs_169060 = {}
    # Getting the type of 'int' (line 814)
    int_169058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 18), 'int', False)
    # Calling int(args, kwargs) (line 814)
    int_call_result_169061 = invoke(stypy.reporting.localization.Localization(__file__, 814, 18), int_169058, *[t_169059], **kwargs_169060)
    
    list_169066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 18), list_169066, int_call_result_169061)
    # Obtaining the member '__getitem__' of a type (line 814)
    getitem___169067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 4), list_169066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 814)
    subscript_call_result_169068 = invoke(stypy.reporting.localization.Localization(__file__, 814, 4), getitem___169067, int_169057)
    
    # Assigning a type to the variable 'tuple_var_assignment_167939' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'tuple_var_assignment_167939', subscript_call_result_169068)
    
    # Assigning a Subscript to a Name (line 814):
    
    # Obtaining the type of the subscript
    int_169069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 4), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 814)
    list_169074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 814)
    # Adding element type (line 814)
    # Getting the type of 'm' (line 814)
    m_169075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 35), 'm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 34), list_169074, m_169075)
    # Adding element type (line 814)
    # Getting the type of 'axis' (line 814)
    axis_169076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 38), 'axis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 34), list_169074, axis_169076)
    
    comprehension_169077 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 18), list_169074)
    # Assigning a type to the variable 't' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 18), 't', comprehension_169077)
    
    # Call to int(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 't' (line 814)
    t_169071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 22), 't', False)
    # Processing the call keyword arguments (line 814)
    kwargs_169072 = {}
    # Getting the type of 'int' (line 814)
    int_169070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 18), 'int', False)
    # Calling int(args, kwargs) (line 814)
    int_call_result_169073 = invoke(stypy.reporting.localization.Localization(__file__, 814, 18), int_169070, *[t_169071], **kwargs_169072)
    
    list_169078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 814, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 814, 18), list_169078, int_call_result_169073)
    # Obtaining the member '__getitem__' of a type (line 814)
    getitem___169079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 4), list_169078, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 814)
    subscript_call_result_169080 = invoke(stypy.reporting.localization.Localization(__file__, 814, 4), getitem___169079, int_169069)
    
    # Assigning a type to the variable 'tuple_var_assignment_167940' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'tuple_var_assignment_167940', subscript_call_result_169080)
    
    # Assigning a Name to a Name (line 814):
    # Getting the type of 'tuple_var_assignment_167939' (line 814)
    tuple_var_assignment_167939_169081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'tuple_var_assignment_167939')
    # Assigning a type to the variable 'cnt' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'cnt', tuple_var_assignment_167939_169081)
    
    # Assigning a Name to a Name (line 814):
    # Getting the type of 'tuple_var_assignment_167940' (line 814)
    tuple_var_assignment_167940_169082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'tuple_var_assignment_167940')
    # Assigning a type to the variable 'iaxis' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 9), 'iaxis', tuple_var_assignment_167940_169082)
    
    
    # Getting the type of 'cnt' (line 816)
    cnt_169083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 7), 'cnt')
    # Getting the type of 'm' (line 816)
    m_169084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 14), 'm')
    # Applying the binary operator '!=' (line 816)
    result_ne_169085 = python_operator(stypy.reporting.localization.Localization(__file__, 816, 7), '!=', cnt_169083, m_169084)
    
    # Testing the type of an if condition (line 816)
    if_condition_169086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 816, 4), result_ne_169085)
    # Assigning a type to the variable 'if_condition_169086' (line 816)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 4), 'if_condition_169086', if_condition_169086)
    # SSA begins for if statement (line 816)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 817)
    # Processing the call arguments (line 817)
    str_169088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 25), 'str', 'The order of integration must be integer')
    # Processing the call keyword arguments (line 817)
    kwargs_169089 = {}
    # Getting the type of 'ValueError' (line 817)
    ValueError_169087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 817)
    ValueError_call_result_169090 = invoke(stypy.reporting.localization.Localization(__file__, 817, 14), ValueError_169087, *[str_169088], **kwargs_169089)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 817, 8), ValueError_call_result_169090, 'raise parameter', BaseException)
    # SSA join for if statement (line 816)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 818)
    cnt_169091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 7), 'cnt')
    int_169092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 13), 'int')
    # Applying the binary operator '<' (line 818)
    result_lt_169093 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 7), '<', cnt_169091, int_169092)
    
    # Testing the type of an if condition (line 818)
    if_condition_169094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 818, 4), result_lt_169093)
    # Assigning a type to the variable 'if_condition_169094' (line 818)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 4), 'if_condition_169094', if_condition_169094)
    # SSA begins for if statement (line 818)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 819)
    # Processing the call arguments (line 819)
    str_169096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 25), 'str', 'The order of integration must be non-negative')
    # Processing the call keyword arguments (line 819)
    kwargs_169097 = {}
    # Getting the type of 'ValueError' (line 819)
    ValueError_169095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 819)
    ValueError_call_result_169098 = invoke(stypy.reporting.localization.Localization(__file__, 819, 14), ValueError_169095, *[str_169096], **kwargs_169097)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 819, 8), ValueError_call_result_169098, 'raise parameter', BaseException)
    # SSA join for if statement (line 818)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 820)
    # Processing the call arguments (line 820)
    # Getting the type of 'k' (line 820)
    k_169100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 11), 'k', False)
    # Processing the call keyword arguments (line 820)
    kwargs_169101 = {}
    # Getting the type of 'len' (line 820)
    len_169099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 7), 'len', False)
    # Calling len(args, kwargs) (line 820)
    len_call_result_169102 = invoke(stypy.reporting.localization.Localization(__file__, 820, 7), len_169099, *[k_169100], **kwargs_169101)
    
    # Getting the type of 'cnt' (line 820)
    cnt_169103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 16), 'cnt')
    # Applying the binary operator '>' (line 820)
    result_gt_169104 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 7), '>', len_call_result_169102, cnt_169103)
    
    # Testing the type of an if condition (line 820)
    if_condition_169105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 820, 4), result_gt_169104)
    # Assigning a type to the variable 'if_condition_169105' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 4), 'if_condition_169105', if_condition_169105)
    # SSA begins for if statement (line 820)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 821)
    # Processing the call arguments (line 821)
    str_169107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 25), 'str', 'Too many integration constants')
    # Processing the call keyword arguments (line 821)
    kwargs_169108 = {}
    # Getting the type of 'ValueError' (line 821)
    ValueError_169106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 821)
    ValueError_call_result_169109 = invoke(stypy.reporting.localization.Localization(__file__, 821, 14), ValueError_169106, *[str_169107], **kwargs_169108)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 821, 8), ValueError_call_result_169109, 'raise parameter', BaseException)
    # SSA join for if statement (line 820)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 822)
    iaxis_169110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 7), 'iaxis')
    # Getting the type of 'axis' (line 822)
    axis_169111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 16), 'axis')
    # Applying the binary operator '!=' (line 822)
    result_ne_169112 = python_operator(stypy.reporting.localization.Localization(__file__, 822, 7), '!=', iaxis_169110, axis_169111)
    
    # Testing the type of an if condition (line 822)
    if_condition_169113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 822, 4), result_ne_169112)
    # Assigning a type to the variable 'if_condition_169113' (line 822)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 4), 'if_condition_169113', if_condition_169113)
    # SSA begins for if statement (line 822)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 823)
    # Processing the call arguments (line 823)
    str_169115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 25), 'str', 'The axis must be integer')
    # Processing the call keyword arguments (line 823)
    kwargs_169116 = {}
    # Getting the type of 'ValueError' (line 823)
    ValueError_169114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 823)
    ValueError_call_result_169117 = invoke(stypy.reporting.localization.Localization(__file__, 823, 14), ValueError_169114, *[str_169115], **kwargs_169116)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 823, 8), ValueError_call_result_169117, 'raise parameter', BaseException)
    # SSA join for if statement (line 822)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Getting the type of 'c' (line 824)
    c_169118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'c')
    # Obtaining the member 'ndim' of a type (line 824)
    ndim_169119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 12), c_169118, 'ndim')
    # Applying the 'usub' unary operator (line 824)
    result___neg___169120 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 11), 'usub', ndim_169119)
    
    # Getting the type of 'iaxis' (line 824)
    iaxis_169121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 22), 'iaxis')
    # Applying the binary operator '<=' (line 824)
    result_le_169122 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 11), '<=', result___neg___169120, iaxis_169121)
    # Getting the type of 'c' (line 824)
    c_169123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 30), 'c')
    # Obtaining the member 'ndim' of a type (line 824)
    ndim_169124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 30), c_169123, 'ndim')
    # Applying the binary operator '<' (line 824)
    result_lt_169125 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 11), '<', iaxis_169121, ndim_169124)
    # Applying the binary operator '&' (line 824)
    result_and__169126 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 11), '&', result_le_169122, result_lt_169125)
    
    # Applying the 'not' unary operator (line 824)
    result_not__169127 = python_operator(stypy.reporting.localization.Localization(__file__, 824, 7), 'not', result_and__169126)
    
    # Testing the type of an if condition (line 824)
    if_condition_169128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 4), result_not__169127)
    # Assigning a type to the variable 'if_condition_169128' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'if_condition_169128', if_condition_169128)
    # SSA begins for if statement (line 824)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 825)
    # Processing the call arguments (line 825)
    str_169130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 25), 'str', 'The axis is out of range')
    # Processing the call keyword arguments (line 825)
    kwargs_169131 = {}
    # Getting the type of 'ValueError' (line 825)
    ValueError_169129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 825)
    ValueError_call_result_169132 = invoke(stypy.reporting.localization.Localization(__file__, 825, 14), ValueError_169129, *[str_169130], **kwargs_169131)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 825, 8), ValueError_call_result_169132, 'raise parameter', BaseException)
    # SSA join for if statement (line 824)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'iaxis' (line 826)
    iaxis_169133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 7), 'iaxis')
    int_169134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 15), 'int')
    # Applying the binary operator '<' (line 826)
    result_lt_169135 = python_operator(stypy.reporting.localization.Localization(__file__, 826, 7), '<', iaxis_169133, int_169134)
    
    # Testing the type of an if condition (line 826)
    if_condition_169136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 4), result_lt_169135)
    # Assigning a type to the variable 'if_condition_169136' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'if_condition_169136', if_condition_169136)
    # SSA begins for if statement (line 826)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'iaxis' (line 827)
    iaxis_169137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'iaxis')
    # Getting the type of 'c' (line 827)
    c_169138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 17), 'c')
    # Obtaining the member 'ndim' of a type (line 827)
    ndim_169139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 17), c_169138, 'ndim')
    # Applying the binary operator '+=' (line 827)
    result_iadd_169140 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 8), '+=', iaxis_169137, ndim_169139)
    # Assigning a type to the variable 'iaxis' (line 827)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'iaxis', result_iadd_169140)
    
    # SSA join for if statement (line 826)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt' (line 829)
    cnt_169141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 7), 'cnt')
    int_169142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 829, 14), 'int')
    # Applying the binary operator '==' (line 829)
    result_eq_169143 = python_operator(stypy.reporting.localization.Localization(__file__, 829, 7), '==', cnt_169141, int_169142)
    
    # Testing the type of an if condition (line 829)
    if_condition_169144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 829, 4), result_eq_169143)
    # Assigning a type to the variable 'if_condition_169144' (line 829)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 829, 4), 'if_condition_169144', if_condition_169144)
    # SSA begins for if statement (line 829)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'c' (line 830)
    c_169145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 830)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'stypy_return_type', c_169145)
    # SSA join for if statement (line 829)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 832):
    
    # Assigning a Call to a Name (line 832):
    
    # Call to rollaxis(...): (line 832)
    # Processing the call arguments (line 832)
    # Getting the type of 'c' (line 832)
    c_169148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 20), 'c', False)
    # Getting the type of 'iaxis' (line 832)
    iaxis_169149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 23), 'iaxis', False)
    # Processing the call keyword arguments (line 832)
    kwargs_169150 = {}
    # Getting the type of 'np' (line 832)
    np_169146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 832)
    rollaxis_169147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 832, 8), np_169146, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 832)
    rollaxis_call_result_169151 = invoke(stypy.reporting.localization.Localization(__file__, 832, 8), rollaxis_169147, *[c_169148, iaxis_169149], **kwargs_169150)
    
    # Assigning a type to the variable 'c' (line 832)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 4), 'c', rollaxis_call_result_169151)
    
    # Assigning a BinOp to a Name (line 833):
    
    # Assigning a BinOp to a Name (line 833):
    
    # Call to list(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'k' (line 833)
    k_169153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 13), 'k', False)
    # Processing the call keyword arguments (line 833)
    kwargs_169154 = {}
    # Getting the type of 'list' (line 833)
    list_169152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 8), 'list', False)
    # Calling list(args, kwargs) (line 833)
    list_call_result_169155 = invoke(stypy.reporting.localization.Localization(__file__, 833, 8), list_169152, *[k_169153], **kwargs_169154)
    
    
    # Obtaining an instance of the builtin type 'list' (line 833)
    list_169156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 833)
    # Adding element type (line 833)
    int_169157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 833, 18), list_169156, int_169157)
    
    # Getting the type of 'cnt' (line 833)
    cnt_169158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'cnt')
    
    # Call to len(...): (line 833)
    # Processing the call arguments (line 833)
    # Getting the type of 'k' (line 833)
    k_169160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 33), 'k', False)
    # Processing the call keyword arguments (line 833)
    kwargs_169161 = {}
    # Getting the type of 'len' (line 833)
    len_169159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 29), 'len', False)
    # Calling len(args, kwargs) (line 833)
    len_call_result_169162 = invoke(stypy.reporting.localization.Localization(__file__, 833, 29), len_169159, *[k_169160], **kwargs_169161)
    
    # Applying the binary operator '-' (line 833)
    result_sub_169163 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 23), '-', cnt_169158, len_call_result_169162)
    
    # Applying the binary operator '*' (line 833)
    result_mul_169164 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 18), '*', list_169156, result_sub_169163)
    
    # Applying the binary operator '+' (line 833)
    result_add_169165 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 8), '+', list_call_result_169155, result_mul_169164)
    
    # Assigning a type to the variable 'k' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'k', result_add_169165)
    
    
    # Call to range(...): (line 834)
    # Processing the call arguments (line 834)
    # Getting the type of 'cnt' (line 834)
    cnt_169167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 19), 'cnt', False)
    # Processing the call keyword arguments (line 834)
    kwargs_169168 = {}
    # Getting the type of 'range' (line 834)
    range_169166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 13), 'range', False)
    # Calling range(args, kwargs) (line 834)
    range_call_result_169169 = invoke(stypy.reporting.localization.Localization(__file__, 834, 13), range_169166, *[cnt_169167], **kwargs_169168)
    
    # Testing the type of a for loop iterable (line 834)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 834, 4), range_call_result_169169)
    # Getting the type of the for loop variable (line 834)
    for_loop_var_169170 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 834, 4), range_call_result_169169)
    # Assigning a type to the variable 'i' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'i', for_loop_var_169170)
    # SSA begins for a for statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 835):
    
    # Assigning a Call to a Name (line 835):
    
    # Call to len(...): (line 835)
    # Processing the call arguments (line 835)
    # Getting the type of 'c' (line 835)
    c_169172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 16), 'c', False)
    # Processing the call keyword arguments (line 835)
    kwargs_169173 = {}
    # Getting the type of 'len' (line 835)
    len_169171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 12), 'len', False)
    # Calling len(args, kwargs) (line 835)
    len_call_result_169174 = invoke(stypy.reporting.localization.Localization(__file__, 835, 12), len_169171, *[c_169172], **kwargs_169173)
    
    # Assigning a type to the variable 'n' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'n', len_call_result_169174)
    
    # Getting the type of 'c' (line 836)
    c_169175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'c')
    # Getting the type of 'scl' (line 836)
    scl_169176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 13), 'scl')
    # Applying the binary operator '*=' (line 836)
    result_imul_169177 = python_operator(stypy.reporting.localization.Localization(__file__, 836, 8), '*=', c_169175, scl_169176)
    # Assigning a type to the variable 'c' (line 836)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 8), 'c', result_imul_169177)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 837)
    n_169178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 11), 'n')
    int_169179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 16), 'int')
    # Applying the binary operator '==' (line 837)
    result_eq_169180 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 11), '==', n_169178, int_169179)
    
    
    # Call to all(...): (line 837)
    # Processing the call arguments (line 837)
    
    
    # Obtaining the type of the subscript
    int_169183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 31), 'int')
    # Getting the type of 'c' (line 837)
    c_169184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 29), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 837)
    getitem___169185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 29), c_169184, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 837)
    subscript_call_result_169186 = invoke(stypy.reporting.localization.Localization(__file__, 837, 29), getitem___169185, int_169183)
    
    int_169187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 837, 37), 'int')
    # Applying the binary operator '==' (line 837)
    result_eq_169188 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 29), '==', subscript_call_result_169186, int_169187)
    
    # Processing the call keyword arguments (line 837)
    kwargs_169189 = {}
    # Getting the type of 'np' (line 837)
    np_169181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 22), 'np', False)
    # Obtaining the member 'all' of a type (line 837)
    all_169182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 22), np_169181, 'all')
    # Calling all(args, kwargs) (line 837)
    all_call_result_169190 = invoke(stypy.reporting.localization.Localization(__file__, 837, 22), all_169182, *[result_eq_169188], **kwargs_169189)
    
    # Applying the binary operator 'and' (line 837)
    result_and_keyword_169191 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 11), 'and', result_eq_169180, all_call_result_169190)
    
    # Testing the type of an if condition (line 837)
    if_condition_169192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 837, 8), result_and_keyword_169191)
    # Assigning a type to the variable 'if_condition_169192' (line 837)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 8), 'if_condition_169192', if_condition_169192)
    # SSA begins for if statement (line 837)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'c' (line 838)
    c_169193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'c')
    
    # Obtaining the type of the subscript
    int_169194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 14), 'int')
    # Getting the type of 'c' (line 838)
    c_169195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'c')
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___169196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 12), c_169195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_169197 = invoke(stypy.reporting.localization.Localization(__file__, 838, 12), getitem___169196, int_169194)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 838)
    i_169198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 22), 'i')
    # Getting the type of 'k' (line 838)
    k_169199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 20), 'k')
    # Obtaining the member '__getitem__' of a type (line 838)
    getitem___169200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 838, 20), k_169199, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 838)
    subscript_call_result_169201 = invoke(stypy.reporting.localization.Localization(__file__, 838, 20), getitem___169200, i_169198)
    
    # Applying the binary operator '+=' (line 838)
    result_iadd_169202 = python_operator(stypy.reporting.localization.Localization(__file__, 838, 12), '+=', subscript_call_result_169197, subscript_call_result_169201)
    # Getting the type of 'c' (line 838)
    c_169203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 12), 'c')
    int_169204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 838, 14), 'int')
    # Storing an element on a container (line 838)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 838, 12), c_169203, (int_169204, result_iadd_169202))
    
    # SSA branch for the else part of an if statement (line 837)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 840):
    
    # Assigning a Call to a Name (line 840):
    
    # Call to empty(...): (line 840)
    # Processing the call arguments (line 840)
    
    # Obtaining an instance of the builtin type 'tuple' (line 840)
    tuple_169207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 840)
    # Adding element type (line 840)
    # Getting the type of 'n' (line 840)
    n_169208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 28), 'n', False)
    int_169209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 32), 'int')
    # Applying the binary operator '+' (line 840)
    result_add_169210 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 28), '+', n_169208, int_169209)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 840, 28), tuple_169207, result_add_169210)
    
    
    # Obtaining the type of the subscript
    int_169211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 46), 'int')
    slice_169212 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 840, 38), int_169211, None, None)
    # Getting the type of 'c' (line 840)
    c_169213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 38), 'c', False)
    # Obtaining the member 'shape' of a type (line 840)
    shape_169214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 38), c_169213, 'shape')
    # Obtaining the member '__getitem__' of a type (line 840)
    getitem___169215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 38), shape_169214, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 840)
    subscript_call_result_169216 = invoke(stypy.reporting.localization.Localization(__file__, 840, 38), getitem___169215, slice_169212)
    
    # Applying the binary operator '+' (line 840)
    result_add_169217 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 27), '+', tuple_169207, subscript_call_result_169216)
    
    # Processing the call keyword arguments (line 840)
    # Getting the type of 'c' (line 840)
    c_169218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 57), 'c', False)
    # Obtaining the member 'dtype' of a type (line 840)
    dtype_169219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 57), c_169218, 'dtype')
    keyword_169220 = dtype_169219
    kwargs_169221 = {'dtype': keyword_169220}
    # Getting the type of 'np' (line 840)
    np_169205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 18), 'np', False)
    # Obtaining the member 'empty' of a type (line 840)
    empty_169206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 18), np_169205, 'empty')
    # Calling empty(args, kwargs) (line 840)
    empty_call_result_169222 = invoke(stypy.reporting.localization.Localization(__file__, 840, 18), empty_169206, *[result_add_169217], **kwargs_169221)
    
    # Assigning a type to the variable 'tmp' (line 840)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 12), 'tmp', empty_call_result_169222)
    
    # Assigning a BinOp to a Subscript (line 841):
    
    # Assigning a BinOp to a Subscript (line 841):
    
    # Obtaining the type of the subscript
    int_169223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 23), 'int')
    # Getting the type of 'c' (line 841)
    c_169224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 841)
    getitem___169225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 841, 21), c_169224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 841)
    subscript_call_result_169226 = invoke(stypy.reporting.localization.Localization(__file__, 841, 21), getitem___169225, int_169223)
    
    int_169227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 26), 'int')
    # Applying the binary operator '*' (line 841)
    result_mul_169228 = python_operator(stypy.reporting.localization.Localization(__file__, 841, 21), '*', subscript_call_result_169226, int_169227)
    
    # Getting the type of 'tmp' (line 841)
    tmp_169229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 12), 'tmp')
    int_169230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 16), 'int')
    # Storing an element on a container (line 841)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 12), tmp_169229, (int_169230, result_mul_169228))
    
    # Assigning a Subscript to a Subscript (line 842):
    
    # Assigning a Subscript to a Subscript (line 842):
    
    # Obtaining the type of the subscript
    int_169231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 23), 'int')
    # Getting the type of 'c' (line 842)
    c_169232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 21), 'c')
    # Obtaining the member '__getitem__' of a type (line 842)
    getitem___169233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 842, 21), c_169232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 842)
    subscript_call_result_169234 = invoke(stypy.reporting.localization.Localization(__file__, 842, 21), getitem___169233, int_169231)
    
    # Getting the type of 'tmp' (line 842)
    tmp_169235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 12), 'tmp')
    int_169236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 16), 'int')
    # Storing an element on a container (line 842)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 842, 12), tmp_169235, (int_169236, subscript_call_result_169234))
    
    
    # Call to range(...): (line 843)
    # Processing the call arguments (line 843)
    int_169238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 27), 'int')
    # Getting the type of 'n' (line 843)
    n_169239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 30), 'n', False)
    # Processing the call keyword arguments (line 843)
    kwargs_169240 = {}
    # Getting the type of 'range' (line 843)
    range_169237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 21), 'range', False)
    # Calling range(args, kwargs) (line 843)
    range_call_result_169241 = invoke(stypy.reporting.localization.Localization(__file__, 843, 21), range_169237, *[int_169238, n_169239], **kwargs_169240)
    
    # Testing the type of a for loop iterable (line 843)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 843, 12), range_call_result_169241)
    # Getting the type of the for loop variable (line 843)
    for_loop_var_169242 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 843, 12), range_call_result_169241)
    # Assigning a type to the variable 'j' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 12), 'j', for_loop_var_169242)
    # SSA begins for a for statement (line 843)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 844):
    
    # Assigning a BinOp to a Subscript (line 844):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 844)
    j_169243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 31), 'j')
    # Getting the type of 'c' (line 844)
    c_169244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___169245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 29), c_169244, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_169246 = invoke(stypy.reporting.localization.Localization(__file__, 844, 29), getitem___169245, j_169243)
    
    # Getting the type of 'j' (line 844)
    j_169247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 35), 'j')
    int_169248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 39), 'int')
    # Applying the binary operator '+' (line 844)
    result_add_169249 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 35), '+', j_169247, int_169248)
    
    # Applying the binary operator 'div' (line 844)
    result_div_169250 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 29), 'div', subscript_call_result_169246, result_add_169249)
    
    # Getting the type of 'tmp' (line 844)
    tmp_169251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 16), 'tmp')
    # Getting the type of 'j' (line 844)
    j_169252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 20), 'j')
    int_169253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 24), 'int')
    # Applying the binary operator '+' (line 844)
    result_add_169254 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 20), '+', j_169252, int_169253)
    
    # Storing an element on a container (line 844)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 844, 16), tmp_169251, (result_add_169254, result_div_169250))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tmp' (line 845)
    tmp_169255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'tmp')
    
    # Obtaining the type of the subscript
    int_169256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 16), 'int')
    # Getting the type of 'tmp' (line 845)
    tmp_169257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'tmp')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___169258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 12), tmp_169257, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_169259 = invoke(stypy.reporting.localization.Localization(__file__, 845, 12), getitem___169258, int_169256)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 845)
    i_169260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 24), 'i')
    # Getting the type of 'k' (line 845)
    k_169261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 22), 'k')
    # Obtaining the member '__getitem__' of a type (line 845)
    getitem___169262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 845, 22), k_169261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 845)
    subscript_call_result_169263 = invoke(stypy.reporting.localization.Localization(__file__, 845, 22), getitem___169262, i_169260)
    
    
    # Call to hermeval(...): (line 845)
    # Processing the call arguments (line 845)
    # Getting the type of 'lbnd' (line 845)
    lbnd_169265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 38), 'lbnd', False)
    # Getting the type of 'tmp' (line 845)
    tmp_169266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 44), 'tmp', False)
    # Processing the call keyword arguments (line 845)
    kwargs_169267 = {}
    # Getting the type of 'hermeval' (line 845)
    hermeval_169264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 29), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 845)
    hermeval_call_result_169268 = invoke(stypy.reporting.localization.Localization(__file__, 845, 29), hermeval_169264, *[lbnd_169265, tmp_169266], **kwargs_169267)
    
    # Applying the binary operator '-' (line 845)
    result_sub_169269 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 22), '-', subscript_call_result_169263, hermeval_call_result_169268)
    
    # Applying the binary operator '+=' (line 845)
    result_iadd_169270 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 12), '+=', subscript_call_result_169259, result_sub_169269)
    # Getting the type of 'tmp' (line 845)
    tmp_169271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 12), 'tmp')
    int_169272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 16), 'int')
    # Storing an element on a container (line 845)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 12), tmp_169271, (int_169272, result_iadd_169270))
    
    
    # Assigning a Name to a Name (line 846):
    
    # Assigning a Name to a Name (line 846):
    # Getting the type of 'tmp' (line 846)
    tmp_169273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 16), 'tmp')
    # Assigning a type to the variable 'c' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 12), 'c', tmp_169273)
    # SSA join for if statement (line 837)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 847):
    
    # Assigning a Call to a Name (line 847):
    
    # Call to rollaxis(...): (line 847)
    # Processing the call arguments (line 847)
    # Getting the type of 'c' (line 847)
    c_169276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 20), 'c', False)
    int_169277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 23), 'int')
    # Getting the type of 'iaxis' (line 847)
    iaxis_169278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 26), 'iaxis', False)
    int_169279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 847, 34), 'int')
    # Applying the binary operator '+' (line 847)
    result_add_169280 = python_operator(stypy.reporting.localization.Localization(__file__, 847, 26), '+', iaxis_169278, int_169279)
    
    # Processing the call keyword arguments (line 847)
    kwargs_169281 = {}
    # Getting the type of 'np' (line 847)
    np_169274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 8), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 847)
    rollaxis_169275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 847, 8), np_169274, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 847)
    rollaxis_call_result_169282 = invoke(stypy.reporting.localization.Localization(__file__, 847, 8), rollaxis_169275, *[c_169276, int_169277, result_add_169280], **kwargs_169281)
    
    # Assigning a type to the variable 'c' (line 847)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 847, 4), 'c', rollaxis_call_result_169282)
    # Getting the type of 'c' (line 848)
    c_169283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 4), 'stypy_return_type', c_169283)
    
    # ################# End of 'hermeint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeint' in the type store
    # Getting the type of 'stypy_return_type' (line 726)
    stypy_return_type_169284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169284)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeint'
    return stypy_return_type_169284

# Assigning a type to the variable 'hermeint' (line 726)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'hermeint', hermeint)

@norecursion
def hermeval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 851)
    True_169285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 26), 'True')
    defaults = [True_169285]
    # Create a new context for function 'hermeval'
    module_type_store = module_type_store.open_function_context('hermeval', 851, 0, False)
    
    # Passed parameters checking function
    hermeval.stypy_localization = localization
    hermeval.stypy_type_of_self = None
    hermeval.stypy_type_store = module_type_store
    hermeval.stypy_function_name = 'hermeval'
    hermeval.stypy_param_names_list = ['x', 'c', 'tensor']
    hermeval.stypy_varargs_param_name = None
    hermeval.stypy_kwargs_param_name = None
    hermeval.stypy_call_defaults = defaults
    hermeval.stypy_call_varargs = varargs
    hermeval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeval', ['x', 'c', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeval', localization, ['x', 'c', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeval(...)' code ##################

    str_169286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, (-1)), 'str', '\n    Evaluate an HermiteE series at points x.\n\n    If `c` is of length `n + 1`, this function returns the value:\n\n    .. math:: p(x) = c_0 * He_0(x) + c_1 * He_1(x) + ... + c_n * He_n(x)\n\n    The parameter `x` is converted to an array only if it is a tuple or a\n    list, otherwise it is treated as a scalar. In either case, either `x`\n    or its elements must support multiplication and addition both with\n    themselves and with the elements of `c`.\n\n    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n    `c` is multidimensional, then the shape of the result depends on the\n    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +\n    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that\n    scalars have shape (,).\n\n    Trailing zeros in the coefficients will be used in the evaluation, so\n    they should be avoided if efficiency is a concern.\n\n    Parameters\n    ----------\n    x : array_like, compatible object\n        If `x` is a list or tuple, it is converted to an ndarray, otherwise\n        it is left unchanged and treated as a scalar. In either case, `x`\n        or its elements must support addition and multiplication with\n        with themselves and with the elements of `c`.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree n are contained in c[n]. If `c` is multidimensional the\n        remaining indices enumerate multiple polynomials. In the two\n        dimensional case the coefficients may be thought of as stored in\n        the columns of `c`.\n    tensor : boolean, optional\n        If True, the shape of the coefficient array is extended with ones\n        on the right, one for each dimension of `x`. Scalars have dimension 0\n        for this action. The result is that every column of coefficients in\n        `c` is evaluated for every element of `x`. If False, `x` is broadcast\n        over the columns of `c` for the evaluation.  This keyword is useful\n        when `c` is multidimensional. The default value is True.\n\n        .. versionadded:: 1.7.0\n\n    Returns\n    -------\n    values : ndarray, algebra_like\n        The shape of the return value is described above.\n\n    See Also\n    --------\n    hermeval2d, hermegrid2d, hermeval3d, hermegrid3d\n\n    Notes\n    -----\n    The evaluation uses Clenshaw recursion, aka synthetic division.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeval\n    >>> coef = [1,2,3]\n    >>> hermeval(1, coef)\n    3.0\n    >>> hermeval([[1,2],[3,4]], coef)\n    array([[  3.,  14.],\n           [ 31.,  54.]])\n\n    ')
    
    # Assigning a Call to a Name (line 920):
    
    # Assigning a Call to a Name (line 920):
    
    # Call to array(...): (line 920)
    # Processing the call arguments (line 920)
    # Getting the type of 'c' (line 920)
    c_169289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 17), 'c', False)
    # Processing the call keyword arguments (line 920)
    int_169290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 26), 'int')
    keyword_169291 = int_169290
    int_169292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 34), 'int')
    keyword_169293 = int_169292
    kwargs_169294 = {'copy': keyword_169293, 'ndmin': keyword_169291}
    # Getting the type of 'np' (line 920)
    np_169287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 920)
    array_169288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 920, 8), np_169287, 'array')
    # Calling array(args, kwargs) (line 920)
    array_call_result_169295 = invoke(stypy.reporting.localization.Localization(__file__, 920, 8), array_169288, *[c_169289], **kwargs_169294)
    
    # Assigning a type to the variable 'c' (line 920)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 920, 4), 'c', array_call_result_169295)
    
    
    # Getting the type of 'c' (line 921)
    c_169296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 7), 'c')
    # Obtaining the member 'dtype' of a type (line 921)
    dtype_169297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 7), c_169296, 'dtype')
    # Obtaining the member 'char' of a type (line 921)
    char_169298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 7), dtype_169297, 'char')
    str_169299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 921, 23), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 921)
    result_contains_169300 = python_operator(stypy.reporting.localization.Localization(__file__, 921, 7), 'in', char_169298, str_169299)
    
    # Testing the type of an if condition (line 921)
    if_condition_169301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 921, 4), result_contains_169300)
    # Assigning a type to the variable 'if_condition_169301' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 4), 'if_condition_169301', if_condition_169301)
    # SSA begins for if statement (line 921)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 922):
    
    # Assigning a Call to a Name (line 922):
    
    # Call to astype(...): (line 922)
    # Processing the call arguments (line 922)
    # Getting the type of 'np' (line 922)
    np_169304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 21), 'np', False)
    # Obtaining the member 'double' of a type (line 922)
    double_169305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 21), np_169304, 'double')
    # Processing the call keyword arguments (line 922)
    kwargs_169306 = {}
    # Getting the type of 'c' (line 922)
    c_169302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 12), 'c', False)
    # Obtaining the member 'astype' of a type (line 922)
    astype_169303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 922, 12), c_169302, 'astype')
    # Calling astype(args, kwargs) (line 922)
    astype_call_result_169307 = invoke(stypy.reporting.localization.Localization(__file__, 922, 12), astype_169303, *[double_169305], **kwargs_169306)
    
    # Assigning a type to the variable 'c' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 8), 'c', astype_call_result_169307)
    # SSA join for if statement (line 921)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 923)
    # Processing the call arguments (line 923)
    # Getting the type of 'x' (line 923)
    x_169309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 18), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 923)
    tuple_169310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 923)
    # Adding element type (line 923)
    # Getting the type of 'tuple' (line 923)
    tuple_169311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 22), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 923, 22), tuple_169310, tuple_169311)
    # Adding element type (line 923)
    # Getting the type of 'list' (line 923)
    list_169312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 29), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 923, 22), tuple_169310, list_169312)
    
    # Processing the call keyword arguments (line 923)
    kwargs_169313 = {}
    # Getting the type of 'isinstance' (line 923)
    isinstance_169308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 923)
    isinstance_call_result_169314 = invoke(stypy.reporting.localization.Localization(__file__, 923, 7), isinstance_169308, *[x_169309, tuple_169310], **kwargs_169313)
    
    # Testing the type of an if condition (line 923)
    if_condition_169315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 4), isinstance_call_result_169314)
    # Assigning a type to the variable 'if_condition_169315' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'if_condition_169315', if_condition_169315)
    # SSA begins for if statement (line 923)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 924):
    
    # Assigning a Call to a Name (line 924):
    
    # Call to asarray(...): (line 924)
    # Processing the call arguments (line 924)
    # Getting the type of 'x' (line 924)
    x_169318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 23), 'x', False)
    # Processing the call keyword arguments (line 924)
    kwargs_169319 = {}
    # Getting the type of 'np' (line 924)
    np_169316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 924)
    asarray_169317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 12), np_169316, 'asarray')
    # Calling asarray(args, kwargs) (line 924)
    asarray_call_result_169320 = invoke(stypy.reporting.localization.Localization(__file__, 924, 12), asarray_169317, *[x_169318], **kwargs_169319)
    
    # Assigning a type to the variable 'x' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 8), 'x', asarray_call_result_169320)
    # SSA join for if statement (line 923)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 925)
    # Processing the call arguments (line 925)
    # Getting the type of 'x' (line 925)
    x_169322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 18), 'x', False)
    # Getting the type of 'np' (line 925)
    np_169323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 925)
    ndarray_169324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 21), np_169323, 'ndarray')
    # Processing the call keyword arguments (line 925)
    kwargs_169325 = {}
    # Getting the type of 'isinstance' (line 925)
    isinstance_169321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 925)
    isinstance_call_result_169326 = invoke(stypy.reporting.localization.Localization(__file__, 925, 7), isinstance_169321, *[x_169322, ndarray_169324], **kwargs_169325)
    
    # Getting the type of 'tensor' (line 925)
    tensor_169327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 37), 'tensor')
    # Applying the binary operator 'and' (line 925)
    result_and_keyword_169328 = python_operator(stypy.reporting.localization.Localization(__file__, 925, 7), 'and', isinstance_call_result_169326, tensor_169327)
    
    # Testing the type of an if condition (line 925)
    if_condition_169329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 925, 4), result_and_keyword_169328)
    # Assigning a type to the variable 'if_condition_169329' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 4), 'if_condition_169329', if_condition_169329)
    # SSA begins for if statement (line 925)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 926):
    
    # Assigning a Call to a Name (line 926):
    
    # Call to reshape(...): (line 926)
    # Processing the call arguments (line 926)
    # Getting the type of 'c' (line 926)
    c_169332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 22), 'c', False)
    # Obtaining the member 'shape' of a type (line 926)
    shape_169333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 22), c_169332, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 926)
    tuple_169334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 926)
    # Adding element type (line 926)
    int_169335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 926, 33), tuple_169334, int_169335)
    
    # Getting the type of 'x' (line 926)
    x_169336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 37), 'x', False)
    # Obtaining the member 'ndim' of a type (line 926)
    ndim_169337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 37), x_169336, 'ndim')
    # Applying the binary operator '*' (line 926)
    result_mul_169338 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 32), '*', tuple_169334, ndim_169337)
    
    # Applying the binary operator '+' (line 926)
    result_add_169339 = python_operator(stypy.reporting.localization.Localization(__file__, 926, 22), '+', shape_169333, result_mul_169338)
    
    # Processing the call keyword arguments (line 926)
    kwargs_169340 = {}
    # Getting the type of 'c' (line 926)
    c_169330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 926)
    reshape_169331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 12), c_169330, 'reshape')
    # Calling reshape(args, kwargs) (line 926)
    reshape_call_result_169341 = invoke(stypy.reporting.localization.Localization(__file__, 926, 12), reshape_169331, *[result_add_169339], **kwargs_169340)
    
    # Assigning a type to the variable 'c' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 8), 'c', reshape_call_result_169341)
    # SSA join for if statement (line 925)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 928)
    # Processing the call arguments (line 928)
    # Getting the type of 'c' (line 928)
    c_169343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 11), 'c', False)
    # Processing the call keyword arguments (line 928)
    kwargs_169344 = {}
    # Getting the type of 'len' (line 928)
    len_169342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 7), 'len', False)
    # Calling len(args, kwargs) (line 928)
    len_call_result_169345 = invoke(stypy.reporting.localization.Localization(__file__, 928, 7), len_169342, *[c_169343], **kwargs_169344)
    
    int_169346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 928, 17), 'int')
    # Applying the binary operator '==' (line 928)
    result_eq_169347 = python_operator(stypy.reporting.localization.Localization(__file__, 928, 7), '==', len_call_result_169345, int_169346)
    
    # Testing the type of an if condition (line 928)
    if_condition_169348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 928, 4), result_eq_169347)
    # Assigning a type to the variable 'if_condition_169348' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 4), 'if_condition_169348', if_condition_169348)
    # SSA begins for if statement (line 928)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 929):
    
    # Assigning a Subscript to a Name (line 929):
    
    # Obtaining the type of the subscript
    int_169349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 15), 'int')
    # Getting the type of 'c' (line 929)
    c_169350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 929)
    getitem___169351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 13), c_169350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 929)
    subscript_call_result_169352 = invoke(stypy.reporting.localization.Localization(__file__, 929, 13), getitem___169351, int_169349)
    
    # Assigning a type to the variable 'c0' (line 929)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 8), 'c0', subscript_call_result_169352)
    
    # Assigning a Num to a Name (line 930):
    
    # Assigning a Num to a Name (line 930):
    int_169353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 13), 'int')
    # Assigning a type to the variable 'c1' (line 930)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 8), 'c1', int_169353)
    # SSA branch for the else part of an if statement (line 928)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 931)
    # Processing the call arguments (line 931)
    # Getting the type of 'c' (line 931)
    c_169355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 13), 'c', False)
    # Processing the call keyword arguments (line 931)
    kwargs_169356 = {}
    # Getting the type of 'len' (line 931)
    len_169354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 9), 'len', False)
    # Calling len(args, kwargs) (line 931)
    len_call_result_169357 = invoke(stypy.reporting.localization.Localization(__file__, 931, 9), len_169354, *[c_169355], **kwargs_169356)
    
    int_169358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 19), 'int')
    # Applying the binary operator '==' (line 931)
    result_eq_169359 = python_operator(stypy.reporting.localization.Localization(__file__, 931, 9), '==', len_call_result_169357, int_169358)
    
    # Testing the type of an if condition (line 931)
    if_condition_169360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 931, 9), result_eq_169359)
    # Assigning a type to the variable 'if_condition_169360' (line 931)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 9), 'if_condition_169360', if_condition_169360)
    # SSA begins for if statement (line 931)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 932):
    
    # Assigning a Subscript to a Name (line 932):
    
    # Obtaining the type of the subscript
    int_169361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 15), 'int')
    # Getting the type of 'c' (line 932)
    c_169362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 932)
    getitem___169363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 13), c_169362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 932)
    subscript_call_result_169364 = invoke(stypy.reporting.localization.Localization(__file__, 932, 13), getitem___169363, int_169361)
    
    # Assigning a type to the variable 'c0' (line 932)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 8), 'c0', subscript_call_result_169364)
    
    # Assigning a Subscript to a Name (line 933):
    
    # Assigning a Subscript to a Name (line 933):
    
    # Obtaining the type of the subscript
    int_169365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 15), 'int')
    # Getting the type of 'c' (line 933)
    c_169366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 933)
    getitem___169367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 13), c_169366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 933)
    subscript_call_result_169368 = invoke(stypy.reporting.localization.Localization(__file__, 933, 13), getitem___169367, int_169365)
    
    # Assigning a type to the variable 'c1' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 8), 'c1', subscript_call_result_169368)
    # SSA branch for the else part of an if statement (line 931)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 935):
    
    # Assigning a Call to a Name (line 935):
    
    # Call to len(...): (line 935)
    # Processing the call arguments (line 935)
    # Getting the type of 'c' (line 935)
    c_169370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 17), 'c', False)
    # Processing the call keyword arguments (line 935)
    kwargs_169371 = {}
    # Getting the type of 'len' (line 935)
    len_169369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 13), 'len', False)
    # Calling len(args, kwargs) (line 935)
    len_call_result_169372 = invoke(stypy.reporting.localization.Localization(__file__, 935, 13), len_169369, *[c_169370], **kwargs_169371)
    
    # Assigning a type to the variable 'nd' (line 935)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 8), 'nd', len_call_result_169372)
    
    # Assigning a Subscript to a Name (line 936):
    
    # Assigning a Subscript to a Name (line 936):
    
    # Obtaining the type of the subscript
    int_169373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 15), 'int')
    # Getting the type of 'c' (line 936)
    c_169374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 936)
    getitem___169375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 936, 13), c_169374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 936)
    subscript_call_result_169376 = invoke(stypy.reporting.localization.Localization(__file__, 936, 13), getitem___169375, int_169373)
    
    # Assigning a type to the variable 'c0' (line 936)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 8), 'c0', subscript_call_result_169376)
    
    # Assigning a Subscript to a Name (line 937):
    
    # Assigning a Subscript to a Name (line 937):
    
    # Obtaining the type of the subscript
    int_169377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 15), 'int')
    # Getting the type of 'c' (line 937)
    c_169378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 937, 13), 'c')
    # Obtaining the member '__getitem__' of a type (line 937)
    getitem___169379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 937, 13), c_169378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 937)
    subscript_call_result_169380 = invoke(stypy.reporting.localization.Localization(__file__, 937, 13), getitem___169379, int_169377)
    
    # Assigning a type to the variable 'c1' (line 937)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 937, 8), 'c1', subscript_call_result_169380)
    
    
    # Call to range(...): (line 938)
    # Processing the call arguments (line 938)
    int_169382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 23), 'int')
    
    # Call to len(...): (line 938)
    # Processing the call arguments (line 938)
    # Getting the type of 'c' (line 938)
    c_169384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 30), 'c', False)
    # Processing the call keyword arguments (line 938)
    kwargs_169385 = {}
    # Getting the type of 'len' (line 938)
    len_169383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 26), 'len', False)
    # Calling len(args, kwargs) (line 938)
    len_call_result_169386 = invoke(stypy.reporting.localization.Localization(__file__, 938, 26), len_169383, *[c_169384], **kwargs_169385)
    
    int_169387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 35), 'int')
    # Applying the binary operator '+' (line 938)
    result_add_169388 = python_operator(stypy.reporting.localization.Localization(__file__, 938, 26), '+', len_call_result_169386, int_169387)
    
    # Processing the call keyword arguments (line 938)
    kwargs_169389 = {}
    # Getting the type of 'range' (line 938)
    range_169381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 938, 17), 'range', False)
    # Calling range(args, kwargs) (line 938)
    range_call_result_169390 = invoke(stypy.reporting.localization.Localization(__file__, 938, 17), range_169381, *[int_169382, result_add_169388], **kwargs_169389)
    
    # Testing the type of a for loop iterable (line 938)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 938, 8), range_call_result_169390)
    # Getting the type of the for loop variable (line 938)
    for_loop_var_169391 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 938, 8), range_call_result_169390)
    # Assigning a type to the variable 'i' (line 938)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 938, 8), 'i', for_loop_var_169391)
    # SSA begins for a for statement (line 938)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 939):
    
    # Assigning a Name to a Name (line 939):
    # Getting the type of 'c0' (line 939)
    c0_169392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 18), 'c0')
    # Assigning a type to the variable 'tmp' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 12), 'tmp', c0_169392)
    
    # Assigning a BinOp to a Name (line 940):
    
    # Assigning a BinOp to a Name (line 940):
    # Getting the type of 'nd' (line 940)
    nd_169393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 17), 'nd')
    int_169394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 22), 'int')
    # Applying the binary operator '-' (line 940)
    result_sub_169395 = python_operator(stypy.reporting.localization.Localization(__file__, 940, 17), '-', nd_169393, int_169394)
    
    # Assigning a type to the variable 'nd' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 12), 'nd', result_sub_169395)
    
    # Assigning a BinOp to a Name (line 941):
    
    # Assigning a BinOp to a Name (line 941):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'i' (line 941)
    i_169396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 20), 'i')
    # Applying the 'usub' unary operator (line 941)
    result___neg___169397 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 19), 'usub', i_169396)
    
    # Getting the type of 'c' (line 941)
    c_169398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 17), 'c')
    # Obtaining the member '__getitem__' of a type (line 941)
    getitem___169399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 941, 17), c_169398, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 941)
    subscript_call_result_169400 = invoke(stypy.reporting.localization.Localization(__file__, 941, 17), getitem___169399, result___neg___169397)
    
    # Getting the type of 'c1' (line 941)
    c1_169401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 25), 'c1')
    # Getting the type of 'nd' (line 941)
    nd_169402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 29), 'nd')
    int_169403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 34), 'int')
    # Applying the binary operator '-' (line 941)
    result_sub_169404 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 29), '-', nd_169402, int_169403)
    
    # Applying the binary operator '*' (line 941)
    result_mul_169405 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 25), '*', c1_169401, result_sub_169404)
    
    # Applying the binary operator '-' (line 941)
    result_sub_169406 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 17), '-', subscript_call_result_169400, result_mul_169405)
    
    # Assigning a type to the variable 'c0' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 12), 'c0', result_sub_169406)
    
    # Assigning a BinOp to a Name (line 942):
    
    # Assigning a BinOp to a Name (line 942):
    # Getting the type of 'tmp' (line 942)
    tmp_169407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 17), 'tmp')
    # Getting the type of 'c1' (line 942)
    c1_169408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 23), 'c1')
    # Getting the type of 'x' (line 942)
    x_169409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 26), 'x')
    # Applying the binary operator '*' (line 942)
    result_mul_169410 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 23), '*', c1_169408, x_169409)
    
    # Applying the binary operator '+' (line 942)
    result_add_169411 = python_operator(stypy.reporting.localization.Localization(__file__, 942, 17), '+', tmp_169407, result_mul_169410)
    
    # Assigning a type to the variable 'c1' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 12), 'c1', result_add_169411)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 931)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 928)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 943)
    c0_169412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 11), 'c0')
    # Getting the type of 'c1' (line 943)
    c1_169413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 16), 'c1')
    # Getting the type of 'x' (line 943)
    x_169414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 19), 'x')
    # Applying the binary operator '*' (line 943)
    result_mul_169415 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 16), '*', c1_169413, x_169414)
    
    # Applying the binary operator '+' (line 943)
    result_add_169416 = python_operator(stypy.reporting.localization.Localization(__file__, 943, 11), '+', c0_169412, result_mul_169415)
    
    # Assigning a type to the variable 'stypy_return_type' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 4), 'stypy_return_type', result_add_169416)
    
    # ################# End of 'hermeval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeval' in the type store
    # Getting the type of 'stypy_return_type' (line 851)
    stypy_return_type_169417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169417)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeval'
    return stypy_return_type_169417

# Assigning a type to the variable 'hermeval' (line 851)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 0), 'hermeval', hermeval)

@norecursion
def hermeval2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeval2d'
    module_type_store = module_type_store.open_function_context('hermeval2d', 946, 0, False)
    
    # Passed parameters checking function
    hermeval2d.stypy_localization = localization
    hermeval2d.stypy_type_of_self = None
    hermeval2d.stypy_type_store = module_type_store
    hermeval2d.stypy_function_name = 'hermeval2d'
    hermeval2d.stypy_param_names_list = ['x', 'y', 'c']
    hermeval2d.stypy_varargs_param_name = None
    hermeval2d.stypy_kwargs_param_name = None
    hermeval2d.stypy_call_defaults = defaults
    hermeval2d.stypy_call_varargs = varargs
    hermeval2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeval2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeval2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeval2d(...)' code ##################

    str_169418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 991, (-1)), 'str', "\n    Evaluate a 2-D HermiteE series at points (x, y).\n\n    This function returns the values:\n\n    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * He_i(x) * He_j(y)\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars and they\n    must have the same shape after conversion. In either case, either `x`\n    and `y` or their elements must support multiplication and addition both\n    with themselves and with the elements of `c`.\n\n    If `c` is a 1-D array a one is implicitly appended to its shape to make\n    it 2-D. The shape of the result will be c.shape[2:] + x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points `(x, y)`,\n        where `x` and `y` must have the same shape. If `x` or `y` is a list\n        or tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and if it isn't an ndarray it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term\n        of multi-degree i,j is contained in ``c[i,j]``. If `c` has\n        dimension greater than two the remaining indices enumerate multiple\n        sets of coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points formed with\n        pairs of corresponding values from `x` and `y`.\n\n    See Also\n    --------\n    hermeval, hermegrid2d, hermeval3d, hermegrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 992)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 993):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 993)
    # Processing the call arguments (line 993)
    
    # Obtaining an instance of the builtin type 'tuple' (line 993)
    tuple_169421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 993)
    # Adding element type (line 993)
    # Getting the type of 'x' (line 993)
    x_169422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 993, 25), tuple_169421, x_169422)
    # Adding element type (line 993)
    # Getting the type of 'y' (line 993)
    y_169423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 28), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 993, 25), tuple_169421, y_169423)
    
    # Processing the call keyword arguments (line 993)
    int_169424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 37), 'int')
    keyword_169425 = int_169424
    kwargs_169426 = {'copy': keyword_169425}
    # Getting the type of 'np' (line 993)
    np_169419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 993)
    array_169420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 15), np_169419, 'array')
    # Calling array(args, kwargs) (line 993)
    array_call_result_169427 = invoke(stypy.reporting.localization.Localization(__file__, 993, 15), array_169420, *[tuple_169421], **kwargs_169426)
    
    # Assigning a type to the variable 'call_assignment_167941' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167941', array_call_result_169427)
    
    # Assigning a Call to a Name (line 993):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_169430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 8), 'int')
    # Processing the call keyword arguments
    kwargs_169431 = {}
    # Getting the type of 'call_assignment_167941' (line 993)
    call_assignment_167941_169428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167941', False)
    # Obtaining the member '__getitem__' of a type (line 993)
    getitem___169429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 8), call_assignment_167941_169428, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169432 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___169429, *[int_169430], **kwargs_169431)
    
    # Assigning a type to the variable 'call_assignment_167942' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167942', getitem___call_result_169432)
    
    # Assigning a Name to a Name (line 993):
    # Getting the type of 'call_assignment_167942' (line 993)
    call_assignment_167942_169433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167942')
    # Assigning a type to the variable 'x' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'x', call_assignment_167942_169433)
    
    # Assigning a Call to a Name (line 993):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_169436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 8), 'int')
    # Processing the call keyword arguments
    kwargs_169437 = {}
    # Getting the type of 'call_assignment_167941' (line 993)
    call_assignment_167941_169434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167941', False)
    # Obtaining the member '__getitem__' of a type (line 993)
    getitem___169435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 8), call_assignment_167941_169434, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169438 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___169435, *[int_169436], **kwargs_169437)
    
    # Assigning a type to the variable 'call_assignment_167943' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167943', getitem___call_result_169438)
    
    # Assigning a Name to a Name (line 993):
    # Getting the type of 'call_assignment_167943' (line 993)
    call_assignment_167943_169439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'call_assignment_167943')
    # Assigning a type to the variable 'y' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 11), 'y', call_assignment_167943_169439)
    # SSA branch for the except part of a try statement (line 992)
    # SSA branch for the except '<any exception>' branch of a try statement (line 992)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 995)
    # Processing the call arguments (line 995)
    str_169441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 995, 25), 'str', 'x, y are incompatible')
    # Processing the call keyword arguments (line 995)
    kwargs_169442 = {}
    # Getting the type of 'ValueError' (line 995)
    ValueError_169440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 995, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 995)
    ValueError_call_result_169443 = invoke(stypy.reporting.localization.Localization(__file__, 995, 14), ValueError_169440, *[str_169441], **kwargs_169442)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 995, 8), ValueError_call_result_169443, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 992)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 997):
    
    # Assigning a Call to a Name (line 997):
    
    # Call to hermeval(...): (line 997)
    # Processing the call arguments (line 997)
    # Getting the type of 'x' (line 997)
    x_169445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 17), 'x', False)
    # Getting the type of 'c' (line 997)
    c_169446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 20), 'c', False)
    # Processing the call keyword arguments (line 997)
    kwargs_169447 = {}
    # Getting the type of 'hermeval' (line 997)
    hermeval_169444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 997)
    hermeval_call_result_169448 = invoke(stypy.reporting.localization.Localization(__file__, 997, 8), hermeval_169444, *[x_169445, c_169446], **kwargs_169447)
    
    # Assigning a type to the variable 'c' (line 997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'c', hermeval_call_result_169448)
    
    # Assigning a Call to a Name (line 998):
    
    # Assigning a Call to a Name (line 998):
    
    # Call to hermeval(...): (line 998)
    # Processing the call arguments (line 998)
    # Getting the type of 'y' (line 998)
    y_169450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 17), 'y', False)
    # Getting the type of 'c' (line 998)
    c_169451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 20), 'c', False)
    # Processing the call keyword arguments (line 998)
    # Getting the type of 'False' (line 998)
    False_169452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 30), 'False', False)
    keyword_169453 = False_169452
    kwargs_169454 = {'tensor': keyword_169453}
    # Getting the type of 'hermeval' (line 998)
    hermeval_169449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 998)
    hermeval_call_result_169455 = invoke(stypy.reporting.localization.Localization(__file__, 998, 8), hermeval_169449, *[y_169450, c_169451], **kwargs_169454)
    
    # Assigning a type to the variable 'c' (line 998)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'c', hermeval_call_result_169455)
    # Getting the type of 'c' (line 999)
    c_169456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'stypy_return_type', c_169456)
    
    # ################# End of 'hermeval2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeval2d' in the type store
    # Getting the type of 'stypy_return_type' (line 946)
    stypy_return_type_169457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169457)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeval2d'
    return stypy_return_type_169457

# Assigning a type to the variable 'hermeval2d' (line 946)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 0), 'hermeval2d', hermeval2d)

@norecursion
def hermegrid2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermegrid2d'
    module_type_store = module_type_store.open_function_context('hermegrid2d', 1002, 0, False)
    
    # Passed parameters checking function
    hermegrid2d.stypy_localization = localization
    hermegrid2d.stypy_type_of_self = None
    hermegrid2d.stypy_type_store = module_type_store
    hermegrid2d.stypy_function_name = 'hermegrid2d'
    hermegrid2d.stypy_param_names_list = ['x', 'y', 'c']
    hermegrid2d.stypy_varargs_param_name = None
    hermegrid2d.stypy_kwargs_param_name = None
    hermegrid2d.stypy_call_defaults = defaults
    hermegrid2d.stypy_call_varargs = varargs
    hermegrid2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermegrid2d', ['x', 'y', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermegrid2d', localization, ['x', 'y', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermegrid2d(...)' code ##################

    str_169458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, (-1)), 'str', "\n    Evaluate a 2-D HermiteE series on the Cartesian product of x and y.\n\n    This function returns the values:\n\n    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * H_i(a) * H_j(b)\n\n    where the points `(a, b)` consist of all pairs formed by taking\n    `a` from `x` and `b` from `y`. The resulting points form a grid with\n    `x` in the first dimension and `y` in the second.\n\n    The parameters `x` and `y` are converted to arrays only if they are\n    tuples or a lists, otherwise they are treated as a scalars. In either\n    case, either `x` and `y` or their elements must support multiplication\n    and addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than two dimensions, ones are implicitly appended to\n    its shape to make it 2-D. The shape of the result will be c.shape[2:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y : array_like, compatible objects\n        The two dimensional series is evaluated at the points in the\n        Cartesian product of `x` and `y`.  If `x` or `y` is a list or\n        tuple, it is first converted to an ndarray, otherwise it is left\n        unchanged and, if it isn't an ndarray, it is treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    hermeval, hermeval2d, hermeval3d, hermegrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1052):
    
    # Assigning a Call to a Name (line 1052):
    
    # Call to hermeval(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'x' (line 1052)
    x_169460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 17), 'x', False)
    # Getting the type of 'c' (line 1052)
    c_169461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 20), 'c', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_169462 = {}
    # Getting the type of 'hermeval' (line 1052)
    hermeval_169459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1052)
    hermeval_call_result_169463 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 8), hermeval_169459, *[x_169460, c_169461], **kwargs_169462)
    
    # Assigning a type to the variable 'c' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'c', hermeval_call_result_169463)
    
    # Assigning a Call to a Name (line 1053):
    
    # Assigning a Call to a Name (line 1053):
    
    # Call to hermeval(...): (line 1053)
    # Processing the call arguments (line 1053)
    # Getting the type of 'y' (line 1053)
    y_169465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 17), 'y', False)
    # Getting the type of 'c' (line 1053)
    c_169466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 20), 'c', False)
    # Processing the call keyword arguments (line 1053)
    kwargs_169467 = {}
    # Getting the type of 'hermeval' (line 1053)
    hermeval_169464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1053)
    hermeval_call_result_169468 = invoke(stypy.reporting.localization.Localization(__file__, 1053, 8), hermeval_169464, *[y_169465, c_169466], **kwargs_169467)
    
    # Assigning a type to the variable 'c' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 4), 'c', hermeval_call_result_169468)
    # Getting the type of 'c' (line 1054)
    c_169469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 4), 'stypy_return_type', c_169469)
    
    # ################# End of 'hermegrid2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermegrid2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1002)
    stypy_return_type_169470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169470)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermegrid2d'
    return stypy_return_type_169470

# Assigning a type to the variable 'hermegrid2d' (line 1002)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 0), 'hermegrid2d', hermegrid2d)

@norecursion
def hermeval3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeval3d'
    module_type_store = module_type_store.open_function_context('hermeval3d', 1057, 0, False)
    
    # Passed parameters checking function
    hermeval3d.stypy_localization = localization
    hermeval3d.stypy_type_of_self = None
    hermeval3d.stypy_type_store = module_type_store
    hermeval3d.stypy_function_name = 'hermeval3d'
    hermeval3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    hermeval3d.stypy_varargs_param_name = None
    hermeval3d.stypy_kwargs_param_name = None
    hermeval3d.stypy_call_defaults = defaults
    hermeval3d.stypy_call_varargs = varargs
    hermeval3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeval3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeval3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeval3d(...)' code ##################

    str_169471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1104, (-1)), 'str', "\n    Evaluate a 3-D Hermite_e series at points (x, y, z).\n\n    This function returns the values:\n\n    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * He_i(x) * He_j(y) * He_k(z)\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if\n    they are tuples or a lists, otherwise they are treated as a scalars and\n    they must have the same shape after conversion. In either case, either\n    `x`, `y`, and `z` or their elements must support multiplication and\n    addition both with themselves and with the elements of `c`.\n\n    If `c` has fewer than 3 dimensions, ones are implicitly appended to its\n    shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible object\n        The three dimensional series is evaluated at the points\n        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If\n        any of `x`, `y`, or `z` is a list or tuple, it is first converted\n        to an ndarray, otherwise it is left unchanged and if it isn't an\n        ndarray it is  treated as a scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficient of the term of\n        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension\n        greater than 3 the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the multidimensional polynomial on points formed with\n        triples of corresponding values from `x`, `y`, and `z`.\n\n    See Also\n    --------\n    hermeval, hermeval2d, hermegrid2d, hermegrid3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    
    # SSA begins for try-except statement (line 1105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 1106):
    
    # Assigning a Call to a Name:
    
    # Call to array(...): (line 1106)
    # Processing the call arguments (line 1106)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1106)
    tuple_169474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1106)
    # Adding element type (line 1106)
    # Getting the type of 'x' (line 1106)
    x_169475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 28), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 28), tuple_169474, x_169475)
    # Adding element type (line 1106)
    # Getting the type of 'y' (line 1106)
    y_169476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 31), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 28), tuple_169474, y_169476)
    # Adding element type (line 1106)
    # Getting the type of 'z' (line 1106)
    z_169477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 34), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1106, 28), tuple_169474, z_169477)
    
    # Processing the call keyword arguments (line 1106)
    int_169478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 43), 'int')
    keyword_169479 = int_169478
    kwargs_169480 = {'copy': keyword_169479}
    # Getting the type of 'np' (line 1106)
    np_169472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1106)
    array_169473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 18), np_169472, 'array')
    # Calling array(args, kwargs) (line 1106)
    array_call_result_169481 = invoke(stypy.reporting.localization.Localization(__file__, 1106, 18), array_169473, *[tuple_169474], **kwargs_169480)
    
    # Assigning a type to the variable 'call_assignment_167944' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167944', array_call_result_169481)
    
    # Assigning a Call to a Name (line 1106):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_169484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 8), 'int')
    # Processing the call keyword arguments
    kwargs_169485 = {}
    # Getting the type of 'call_assignment_167944' (line 1106)
    call_assignment_167944_169482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167944', False)
    # Obtaining the member '__getitem__' of a type (line 1106)
    getitem___169483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 8), call_assignment_167944_169482, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169486 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___169483, *[int_169484], **kwargs_169485)
    
    # Assigning a type to the variable 'call_assignment_167945' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167945', getitem___call_result_169486)
    
    # Assigning a Name to a Name (line 1106):
    # Getting the type of 'call_assignment_167945' (line 1106)
    call_assignment_167945_169487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167945')
    # Assigning a type to the variable 'x' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'x', call_assignment_167945_169487)
    
    # Assigning a Call to a Name (line 1106):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_169490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 8), 'int')
    # Processing the call keyword arguments
    kwargs_169491 = {}
    # Getting the type of 'call_assignment_167944' (line 1106)
    call_assignment_167944_169488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167944', False)
    # Obtaining the member '__getitem__' of a type (line 1106)
    getitem___169489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 8), call_assignment_167944_169488, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169492 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___169489, *[int_169490], **kwargs_169491)
    
    # Assigning a type to the variable 'call_assignment_167946' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167946', getitem___call_result_169492)
    
    # Assigning a Name to a Name (line 1106):
    # Getting the type of 'call_assignment_167946' (line 1106)
    call_assignment_167946_169493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167946')
    # Assigning a type to the variable 'y' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 11), 'y', call_assignment_167946_169493)
    
    # Assigning a Call to a Name (line 1106):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_169496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1106, 8), 'int')
    # Processing the call keyword arguments
    kwargs_169497 = {}
    # Getting the type of 'call_assignment_167944' (line 1106)
    call_assignment_167944_169494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167944', False)
    # Obtaining the member '__getitem__' of a type (line 1106)
    getitem___169495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1106, 8), call_assignment_167944_169494, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169498 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___169495, *[int_169496], **kwargs_169497)
    
    # Assigning a type to the variable 'call_assignment_167947' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167947', getitem___call_result_169498)
    
    # Assigning a Name to a Name (line 1106):
    # Getting the type of 'call_assignment_167947' (line 1106)
    call_assignment_167947_169499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1106, 8), 'call_assignment_167947')
    # Assigning a type to the variable 'z' (line 1106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1106, 14), 'z', call_assignment_167947_169499)
    # SSA branch for the except part of a try statement (line 1105)
    # SSA branch for the except '<any exception>' branch of a try statement (line 1105)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 1108)
    # Processing the call arguments (line 1108)
    str_169501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1108, 25), 'str', 'x, y, z are incompatible')
    # Processing the call keyword arguments (line 1108)
    kwargs_169502 = {}
    # Getting the type of 'ValueError' (line 1108)
    ValueError_169500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1108, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1108)
    ValueError_call_result_169503 = invoke(stypy.reporting.localization.Localization(__file__, 1108, 14), ValueError_169500, *[str_169501], **kwargs_169502)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1108, 8), ValueError_call_result_169503, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 1105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1110):
    
    # Assigning a Call to a Name (line 1110):
    
    # Call to hermeval(...): (line 1110)
    # Processing the call arguments (line 1110)
    # Getting the type of 'x' (line 1110)
    x_169505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 17), 'x', False)
    # Getting the type of 'c' (line 1110)
    c_169506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 20), 'c', False)
    # Processing the call keyword arguments (line 1110)
    kwargs_169507 = {}
    # Getting the type of 'hermeval' (line 1110)
    hermeval_169504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1110, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1110)
    hermeval_call_result_169508 = invoke(stypy.reporting.localization.Localization(__file__, 1110, 8), hermeval_169504, *[x_169505, c_169506], **kwargs_169507)
    
    # Assigning a type to the variable 'c' (line 1110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1110, 4), 'c', hermeval_call_result_169508)
    
    # Assigning a Call to a Name (line 1111):
    
    # Assigning a Call to a Name (line 1111):
    
    # Call to hermeval(...): (line 1111)
    # Processing the call arguments (line 1111)
    # Getting the type of 'y' (line 1111)
    y_169510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 17), 'y', False)
    # Getting the type of 'c' (line 1111)
    c_169511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 20), 'c', False)
    # Processing the call keyword arguments (line 1111)
    # Getting the type of 'False' (line 1111)
    False_169512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 30), 'False', False)
    keyword_169513 = False_169512
    kwargs_169514 = {'tensor': keyword_169513}
    # Getting the type of 'hermeval' (line 1111)
    hermeval_169509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1111, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1111)
    hermeval_call_result_169515 = invoke(stypy.reporting.localization.Localization(__file__, 1111, 8), hermeval_169509, *[y_169510, c_169511], **kwargs_169514)
    
    # Assigning a type to the variable 'c' (line 1111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1111, 4), 'c', hermeval_call_result_169515)
    
    # Assigning a Call to a Name (line 1112):
    
    # Assigning a Call to a Name (line 1112):
    
    # Call to hermeval(...): (line 1112)
    # Processing the call arguments (line 1112)
    # Getting the type of 'z' (line 1112)
    z_169517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 17), 'z', False)
    # Getting the type of 'c' (line 1112)
    c_169518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 20), 'c', False)
    # Processing the call keyword arguments (line 1112)
    # Getting the type of 'False' (line 1112)
    False_169519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 30), 'False', False)
    keyword_169520 = False_169519
    kwargs_169521 = {'tensor': keyword_169520}
    # Getting the type of 'hermeval' (line 1112)
    hermeval_169516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1112, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1112)
    hermeval_call_result_169522 = invoke(stypy.reporting.localization.Localization(__file__, 1112, 8), hermeval_169516, *[z_169517, c_169518], **kwargs_169521)
    
    # Assigning a type to the variable 'c' (line 1112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1112, 4), 'c', hermeval_call_result_169522)
    # Getting the type of 'c' (line 1113)
    c_169523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 4), 'stypy_return_type', c_169523)
    
    # ################# End of 'hermeval3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeval3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1057)
    stypy_return_type_169524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169524)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeval3d'
    return stypy_return_type_169524

# Assigning a type to the variable 'hermeval3d' (line 1057)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'hermeval3d', hermeval3d)

@norecursion
def hermegrid3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermegrid3d'
    module_type_store = module_type_store.open_function_context('hermegrid3d', 1116, 0, False)
    
    # Passed parameters checking function
    hermegrid3d.stypy_localization = localization
    hermegrid3d.stypy_type_of_self = None
    hermegrid3d.stypy_type_store = module_type_store
    hermegrid3d.stypy_function_name = 'hermegrid3d'
    hermegrid3d.stypy_param_names_list = ['x', 'y', 'z', 'c']
    hermegrid3d.stypy_varargs_param_name = None
    hermegrid3d.stypy_kwargs_param_name = None
    hermegrid3d.stypy_call_defaults = defaults
    hermegrid3d.stypy_call_varargs = varargs
    hermegrid3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermegrid3d', ['x', 'y', 'z', 'c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermegrid3d', localization, ['x', 'y', 'z', 'c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermegrid3d(...)' code ##################

    str_169525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, (-1)), 'str', "\n    Evaluate a 3-D HermiteE series on the Cartesian product of x, y, and z.\n\n    This function returns the values:\n\n    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * He_i(a) * He_j(b) * He_k(c)\n\n    where the points `(a, b, c)` consist of all triples formed by taking\n    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form\n    a grid with `x` in the first dimension, `y` in the second, and `z` in\n    the third.\n\n    The parameters `x`, `y`, and `z` are converted to arrays only if they\n    are tuples or a lists, otherwise they are treated as a scalars. In\n    either case, either `x`, `y`, and `z` or their elements must support\n    multiplication and addition both with themselves and with the elements\n    of `c`.\n\n    If `c` has fewer than three dimensions, ones are implicitly appended to\n    its shape to make it 3-D. The shape of the result will be c.shape[3:] +\n    x.shape + y.shape + z.shape.\n\n    Parameters\n    ----------\n    x, y, z : array_like, compatible objects\n        The three dimensional series is evaluated at the points in the\n        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a\n        list or tuple, it is first converted to an ndarray, otherwise it is\n        left unchanged and, if it isn't an ndarray, it is treated as a\n        scalar.\n    c : array_like\n        Array of coefficients ordered so that the coefficients for terms of\n        degree i,j are contained in ``c[i,j]``. If `c` has dimension\n        greater than two the remaining indices enumerate multiple sets of\n        coefficients.\n\n    Returns\n    -------\n    values : ndarray, compatible object\n        The values of the two dimensional polynomial at points in the Cartesian\n        product of `x` and `y`.\n\n    See Also\n    --------\n    hermeval, hermeval2d, hermegrid2d, hermeval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ")
    
    # Assigning a Call to a Name (line 1169):
    
    # Assigning a Call to a Name (line 1169):
    
    # Call to hermeval(...): (line 1169)
    # Processing the call arguments (line 1169)
    # Getting the type of 'x' (line 1169)
    x_169527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 17), 'x', False)
    # Getting the type of 'c' (line 1169)
    c_169528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 20), 'c', False)
    # Processing the call keyword arguments (line 1169)
    kwargs_169529 = {}
    # Getting the type of 'hermeval' (line 1169)
    hermeval_169526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1169)
    hermeval_call_result_169530 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 8), hermeval_169526, *[x_169527, c_169528], **kwargs_169529)
    
    # Assigning a type to the variable 'c' (line 1169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 4), 'c', hermeval_call_result_169530)
    
    # Assigning a Call to a Name (line 1170):
    
    # Assigning a Call to a Name (line 1170):
    
    # Call to hermeval(...): (line 1170)
    # Processing the call arguments (line 1170)
    # Getting the type of 'y' (line 1170)
    y_169532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 17), 'y', False)
    # Getting the type of 'c' (line 1170)
    c_169533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 20), 'c', False)
    # Processing the call keyword arguments (line 1170)
    kwargs_169534 = {}
    # Getting the type of 'hermeval' (line 1170)
    hermeval_169531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1170, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1170)
    hermeval_call_result_169535 = invoke(stypy.reporting.localization.Localization(__file__, 1170, 8), hermeval_169531, *[y_169532, c_169533], **kwargs_169534)
    
    # Assigning a type to the variable 'c' (line 1170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1170, 4), 'c', hermeval_call_result_169535)
    
    # Assigning a Call to a Name (line 1171):
    
    # Assigning a Call to a Name (line 1171):
    
    # Call to hermeval(...): (line 1171)
    # Processing the call arguments (line 1171)
    # Getting the type of 'z' (line 1171)
    z_169537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 17), 'z', False)
    # Getting the type of 'c' (line 1171)
    c_169538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 20), 'c', False)
    # Processing the call keyword arguments (line 1171)
    kwargs_169539 = {}
    # Getting the type of 'hermeval' (line 1171)
    hermeval_169536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 8), 'hermeval', False)
    # Calling hermeval(args, kwargs) (line 1171)
    hermeval_call_result_169540 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 8), hermeval_169536, *[z_169537, c_169538], **kwargs_169539)
    
    # Assigning a type to the variable 'c' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 4), 'c', hermeval_call_result_169540)
    # Getting the type of 'c' (line 1172)
    c_169541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 4), 'stypy_return_type', c_169541)
    
    # ################# End of 'hermegrid3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermegrid3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1116)
    stypy_return_type_169542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169542)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermegrid3d'
    return stypy_return_type_169542

# Assigning a type to the variable 'hermegrid3d' (line 1116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 0), 'hermegrid3d', hermegrid3d)

@norecursion
def hermevander(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermevander'
    module_type_store = module_type_store.open_function_context('hermevander', 1175, 0, False)
    
    # Passed parameters checking function
    hermevander.stypy_localization = localization
    hermevander.stypy_type_of_self = None
    hermevander.stypy_type_store = module_type_store
    hermevander.stypy_function_name = 'hermevander'
    hermevander.stypy_param_names_list = ['x', 'deg']
    hermevander.stypy_varargs_param_name = None
    hermevander.stypy_kwargs_param_name = None
    hermevander.stypy_call_defaults = defaults
    hermevander.stypy_call_varargs = varargs
    hermevander.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermevander', ['x', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermevander', localization, ['x', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermevander(...)' code ##################

    str_169543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1218, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degree.\n\n    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points\n    `x`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., i] = He_i(x),\n\n    where `0 <= i <= deg`. The leading indices of `V` index the elements of\n    `x` and the last index is the degree of the HermiteE polynomial.\n\n    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the\n    array ``V = hermevander(x, n)``, then ``np.dot(V, c)`` and\n    ``hermeval(x, c)`` are the same up to roundoff. This equivalence is\n    useful both for least squares fitting and for the evaluation of a large\n    number of HermiteE series of the same degree and sample points.\n\n    Parameters\n    ----------\n    x : array_like\n        Array of points. The dtype is converted to float64 or complex128\n        depending on whether any of the elements are complex. If `x` is\n        scalar it is converted to a 1-D array.\n    deg : int\n        Degree of the resulting matrix.\n\n    Returns\n    -------\n    vander : ndarray\n        The pseudo-Vandermonde matrix. The shape of the returned matrix is\n        ``x.shape + (deg + 1,)``, where The last index is the degree of the\n        corresponding HermiteE polynomial.  The dtype will be the same as\n        the converted `x`.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermevander\n    >>> x = np.array([-1, 0, 1])\n    >>> hermevander(x, 3)\n    array([[ 1., -1.,  0.,  2.],\n           [ 1.,  0., -1., -0.],\n           [ 1.,  1.,  0., -2.]])\n\n    ')
    
    # Assigning a Call to a Name (line 1219):
    
    # Assigning a Call to a Name (line 1219):
    
    # Call to int(...): (line 1219)
    # Processing the call arguments (line 1219)
    # Getting the type of 'deg' (line 1219)
    deg_169545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 15), 'deg', False)
    # Processing the call keyword arguments (line 1219)
    kwargs_169546 = {}
    # Getting the type of 'int' (line 1219)
    int_169544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1219, 11), 'int', False)
    # Calling int(args, kwargs) (line 1219)
    int_call_result_169547 = invoke(stypy.reporting.localization.Localization(__file__, 1219, 11), int_169544, *[deg_169545], **kwargs_169546)
    
    # Assigning a type to the variable 'ideg' (line 1219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1219, 4), 'ideg', int_call_result_169547)
    
    
    # Getting the type of 'ideg' (line 1220)
    ideg_169548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 7), 'ideg')
    # Getting the type of 'deg' (line 1220)
    deg_169549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1220, 15), 'deg')
    # Applying the binary operator '!=' (line 1220)
    result_ne_169550 = python_operator(stypy.reporting.localization.Localization(__file__, 1220, 7), '!=', ideg_169548, deg_169549)
    
    # Testing the type of an if condition (line 1220)
    if_condition_169551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1220, 4), result_ne_169550)
    # Assigning a type to the variable 'if_condition_169551' (line 1220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1220, 4), 'if_condition_169551', if_condition_169551)
    # SSA begins for if statement (line 1220)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1221)
    # Processing the call arguments (line 1221)
    str_169553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1221, 25), 'str', 'deg must be integer')
    # Processing the call keyword arguments (line 1221)
    kwargs_169554 = {}
    # Getting the type of 'ValueError' (line 1221)
    ValueError_169552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1221, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1221)
    ValueError_call_result_169555 = invoke(stypy.reporting.localization.Localization(__file__, 1221, 14), ValueError_169552, *[str_169553], **kwargs_169554)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1221, 8), ValueError_call_result_169555, 'raise parameter', BaseException)
    # SSA join for if statement (line 1220)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ideg' (line 1222)
    ideg_169556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 7), 'ideg')
    int_169557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1222, 14), 'int')
    # Applying the binary operator '<' (line 1222)
    result_lt_169558 = python_operator(stypy.reporting.localization.Localization(__file__, 1222, 7), '<', ideg_169556, int_169557)
    
    # Testing the type of an if condition (line 1222)
    if_condition_169559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1222, 4), result_lt_169558)
    # Assigning a type to the variable 'if_condition_169559' (line 1222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1222, 4), 'if_condition_169559', if_condition_169559)
    # SSA begins for if statement (line 1222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1223)
    # Processing the call arguments (line 1223)
    str_169561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1223, 25), 'str', 'deg must be non-negative')
    # Processing the call keyword arguments (line 1223)
    kwargs_169562 = {}
    # Getting the type of 'ValueError' (line 1223)
    ValueError_169560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1223, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1223)
    ValueError_call_result_169563 = invoke(stypy.reporting.localization.Localization(__file__, 1223, 14), ValueError_169560, *[str_169561], **kwargs_169562)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1223, 8), ValueError_call_result_169563, 'raise parameter', BaseException)
    # SSA join for if statement (line 1222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1225):
    
    # Assigning a BinOp to a Name (line 1225):
    
    # Call to array(...): (line 1225)
    # Processing the call arguments (line 1225)
    # Getting the type of 'x' (line 1225)
    x_169566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 17), 'x', False)
    # Processing the call keyword arguments (line 1225)
    int_169567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 25), 'int')
    keyword_169568 = int_169567
    int_169569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 34), 'int')
    keyword_169570 = int_169569
    kwargs_169571 = {'copy': keyword_169568, 'ndmin': keyword_169570}
    # Getting the type of 'np' (line 1225)
    np_169564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1225)
    array_169565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1225, 8), np_169564, 'array')
    # Calling array(args, kwargs) (line 1225)
    array_call_result_169572 = invoke(stypy.reporting.localization.Localization(__file__, 1225, 8), array_169565, *[x_169566], **kwargs_169571)
    
    float_169573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 39), 'float')
    # Applying the binary operator '+' (line 1225)
    result_add_169574 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 8), '+', array_call_result_169572, float_169573)
    
    # Assigning a type to the variable 'x' (line 1225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1225, 4), 'x', result_add_169574)
    
    # Assigning a BinOp to a Name (line 1226):
    
    # Assigning a BinOp to a Name (line 1226):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1226)
    tuple_169575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1226, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1226)
    # Adding element type (line 1226)
    # Getting the type of 'ideg' (line 1226)
    ideg_169576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 12), 'ideg')
    int_169577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1226, 19), 'int')
    # Applying the binary operator '+' (line 1226)
    result_add_169578 = python_operator(stypy.reporting.localization.Localization(__file__, 1226, 12), '+', ideg_169576, int_169577)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1226, 12), tuple_169575, result_add_169578)
    
    # Getting the type of 'x' (line 1226)
    x_169579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 25), 'x')
    # Obtaining the member 'shape' of a type (line 1226)
    shape_169580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1226, 25), x_169579, 'shape')
    # Applying the binary operator '+' (line 1226)
    result_add_169581 = python_operator(stypy.reporting.localization.Localization(__file__, 1226, 11), '+', tuple_169575, shape_169580)
    
    # Assigning a type to the variable 'dims' (line 1226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1226, 4), 'dims', result_add_169581)
    
    # Assigning a Attribute to a Name (line 1227):
    
    # Assigning a Attribute to a Name (line 1227):
    # Getting the type of 'x' (line 1227)
    x_169582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 11), 'x')
    # Obtaining the member 'dtype' of a type (line 1227)
    dtype_169583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1227, 11), x_169582, 'dtype')
    # Assigning a type to the variable 'dtyp' (line 1227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 4), 'dtyp', dtype_169583)
    
    # Assigning a Call to a Name (line 1228):
    
    # Assigning a Call to a Name (line 1228):
    
    # Call to empty(...): (line 1228)
    # Processing the call arguments (line 1228)
    # Getting the type of 'dims' (line 1228)
    dims_169586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 17), 'dims', False)
    # Processing the call keyword arguments (line 1228)
    # Getting the type of 'dtyp' (line 1228)
    dtyp_169587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 29), 'dtyp', False)
    keyword_169588 = dtyp_169587
    kwargs_169589 = {'dtype': keyword_169588}
    # Getting the type of 'np' (line 1228)
    np_169584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1228)
    empty_169585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1228, 8), np_169584, 'empty')
    # Calling empty(args, kwargs) (line 1228)
    empty_call_result_169590 = invoke(stypy.reporting.localization.Localization(__file__, 1228, 8), empty_169585, *[dims_169586], **kwargs_169589)
    
    # Assigning a type to the variable 'v' (line 1228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1228, 4), 'v', empty_call_result_169590)
    
    # Assigning a BinOp to a Subscript (line 1229):
    
    # Assigning a BinOp to a Subscript (line 1229):
    # Getting the type of 'x' (line 1229)
    x_169591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 11), 'x')
    int_169592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 13), 'int')
    # Applying the binary operator '*' (line 1229)
    result_mul_169593 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 11), '*', x_169591, int_169592)
    
    int_169594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 17), 'int')
    # Applying the binary operator '+' (line 1229)
    result_add_169595 = python_operator(stypy.reporting.localization.Localization(__file__, 1229, 11), '+', result_mul_169593, int_169594)
    
    # Getting the type of 'v' (line 1229)
    v_169596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1229, 4), 'v')
    int_169597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1229, 6), 'int')
    # Storing an element on a container (line 1229)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1229, 4), v_169596, (int_169597, result_add_169595))
    
    
    # Getting the type of 'ideg' (line 1230)
    ideg_169598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1230, 7), 'ideg')
    int_169599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1230, 14), 'int')
    # Applying the binary operator '>' (line 1230)
    result_gt_169600 = python_operator(stypy.reporting.localization.Localization(__file__, 1230, 7), '>', ideg_169598, int_169599)
    
    # Testing the type of an if condition (line 1230)
    if_condition_169601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1230, 4), result_gt_169600)
    # Assigning a type to the variable 'if_condition_169601' (line 1230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1230, 4), 'if_condition_169601', if_condition_169601)
    # SSA begins for if statement (line 1230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 1231):
    
    # Assigning a Name to a Subscript (line 1231):
    # Getting the type of 'x' (line 1231)
    x_169602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 15), 'x')
    # Getting the type of 'v' (line 1231)
    v_169603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 8), 'v')
    int_169604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 10), 'int')
    # Storing an element on a container (line 1231)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1231, 8), v_169603, (int_169604, x_169602))
    
    
    # Call to range(...): (line 1232)
    # Processing the call arguments (line 1232)
    int_169606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 23), 'int')
    # Getting the type of 'ideg' (line 1232)
    ideg_169607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 26), 'ideg', False)
    int_169608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1232, 33), 'int')
    # Applying the binary operator '+' (line 1232)
    result_add_169609 = python_operator(stypy.reporting.localization.Localization(__file__, 1232, 26), '+', ideg_169607, int_169608)
    
    # Processing the call keyword arguments (line 1232)
    kwargs_169610 = {}
    # Getting the type of 'range' (line 1232)
    range_169605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1232, 17), 'range', False)
    # Calling range(args, kwargs) (line 1232)
    range_call_result_169611 = invoke(stypy.reporting.localization.Localization(__file__, 1232, 17), range_169605, *[int_169606, result_add_169609], **kwargs_169610)
    
    # Testing the type of a for loop iterable (line 1232)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1232, 8), range_call_result_169611)
    # Getting the type of the for loop variable (line 1232)
    for_loop_var_169612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1232, 8), range_call_result_169611)
    # Assigning a type to the variable 'i' (line 1232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1232, 8), 'i', for_loop_var_169612)
    # SSA begins for a for statement (line 1232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 1233):
    
    # Assigning a BinOp to a Subscript (line 1233):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1233)
    i_169613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 22), 'i')
    int_169614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 24), 'int')
    # Applying the binary operator '-' (line 1233)
    result_sub_169615 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 22), '-', i_169613, int_169614)
    
    # Getting the type of 'v' (line 1233)
    v_169616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 20), 'v')
    # Obtaining the member '__getitem__' of a type (line 1233)
    getitem___169617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 20), v_169616, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1233)
    subscript_call_result_169618 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 20), getitem___169617, result_sub_169615)
    
    # Getting the type of 'x' (line 1233)
    x_169619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 27), 'x')
    # Applying the binary operator '*' (line 1233)
    result_mul_169620 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 20), '*', subscript_call_result_169618, x_169619)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 1233)
    i_169621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 33), 'i')
    int_169622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 35), 'int')
    # Applying the binary operator '-' (line 1233)
    result_sub_169623 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 33), '-', i_169621, int_169622)
    
    # Getting the type of 'v' (line 1233)
    v_169624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 31), 'v')
    # Obtaining the member '__getitem__' of a type (line 1233)
    getitem___169625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1233, 31), v_169624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1233)
    subscript_call_result_169626 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 31), getitem___169625, result_sub_169623)
    
    # Getting the type of 'i' (line 1233)
    i_169627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 39), 'i')
    int_169628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1233, 43), 'int')
    # Applying the binary operator '-' (line 1233)
    result_sub_169629 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 39), '-', i_169627, int_169628)
    
    # Applying the binary operator '*' (line 1233)
    result_mul_169630 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 31), '*', subscript_call_result_169626, result_sub_169629)
    
    # Applying the binary operator '-' (line 1233)
    result_sub_169631 = python_operator(stypy.reporting.localization.Localization(__file__, 1233, 20), '-', result_mul_169620, result_mul_169630)
    
    # Getting the type of 'v' (line 1233)
    v_169632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 12), 'v')
    # Getting the type of 'i' (line 1233)
    i_169633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 14), 'i')
    # Storing an element on a container (line 1233)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1233, 12), v_169632, (i_169633, result_sub_169631))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1230)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to rollaxis(...): (line 1234)
    # Processing the call arguments (line 1234)
    # Getting the type of 'v' (line 1234)
    v_169636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 23), 'v', False)
    int_169637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, 26), 'int')
    # Getting the type of 'v' (line 1234)
    v_169638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 29), 'v', False)
    # Obtaining the member 'ndim' of a type (line 1234)
    ndim_169639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 29), v_169638, 'ndim')
    # Processing the call keyword arguments (line 1234)
    kwargs_169640 = {}
    # Getting the type of 'np' (line 1234)
    np_169634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1234, 11), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 1234)
    rollaxis_169635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1234, 11), np_169634, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 1234)
    rollaxis_call_result_169641 = invoke(stypy.reporting.localization.Localization(__file__, 1234, 11), rollaxis_169635, *[v_169636, int_169637, ndim_169639], **kwargs_169640)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1234, 4), 'stypy_return_type', rollaxis_call_result_169641)
    
    # ################# End of 'hermevander(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermevander' in the type store
    # Getting the type of 'stypy_return_type' (line 1175)
    stypy_return_type_169642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169642)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermevander'
    return stypy_return_type_169642

# Assigning a type to the variable 'hermevander' (line 1175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1175, 0), 'hermevander', hermevander)

@norecursion
def hermevander2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermevander2d'
    module_type_store = module_type_store.open_function_context('hermevander2d', 1237, 0, False)
    
    # Passed parameters checking function
    hermevander2d.stypy_localization = localization
    hermevander2d.stypy_type_of_self = None
    hermevander2d.stypy_type_store = module_type_store
    hermevander2d.stypy_function_name = 'hermevander2d'
    hermevander2d.stypy_param_names_list = ['x', 'y', 'deg']
    hermevander2d.stypy_varargs_param_name = None
    hermevander2d.stypy_kwargs_param_name = None
    hermevander2d.stypy_call_defaults = defaults
    hermevander2d.stypy_call_varargs = varargs
    hermevander2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermevander2d', ['x', 'y', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermevander2d', localization, ['x', 'y', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermevander2d(...)' code ##################

    str_169643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1286, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y)`. The pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., deg[1]*i + j] = He_i(x) * He_j(y),\n\n    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of\n    `V` index the points `(x, y)` and the last index encodes the degrees of\n    the HermiteE polynomials.\n\n    If ``V = hermevander2d(x, y, [xdeg, ydeg])``, then the columns of `V`\n    correspond to the elements of a 2-D coefficient array `c` of shape\n    (xdeg + 1, ydeg + 1) in the order\n\n    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...\n\n    and ``np.dot(V, c.flat)`` and ``hermeval2d(x, y, c)`` will be the same\n    up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 2-D HermiteE\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes\n        will be converted to either float64 or complex128 depending on\n        whether any of the elements are complex. Scalars are converted to\n        1-D arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg].\n\n    Returns\n    -------\n    vander2d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same\n        as the converted `x` and `y`.\n\n    See Also\n    --------\n    hermevander, hermevander3d. hermeval2d, hermeval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1287):
    
    # Assigning a ListComp to a Name (line 1287):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1287)
    deg_169648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 28), 'deg')
    comprehension_169649 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1287, 12), deg_169648)
    # Assigning a type to the variable 'd' (line 1287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 12), 'd', comprehension_169649)
    
    # Call to int(...): (line 1287)
    # Processing the call arguments (line 1287)
    # Getting the type of 'd' (line 1287)
    d_169645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 16), 'd', False)
    # Processing the call keyword arguments (line 1287)
    kwargs_169646 = {}
    # Getting the type of 'int' (line 1287)
    int_169644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1287, 12), 'int', False)
    # Calling int(args, kwargs) (line 1287)
    int_call_result_169647 = invoke(stypy.reporting.localization.Localization(__file__, 1287, 12), int_169644, *[d_169645], **kwargs_169646)
    
    list_169650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1287, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1287, 12), list_169650, int_call_result_169647)
    # Assigning a type to the variable 'ideg' (line 1287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1287, 4), 'ideg', list_169650)
    
    # Assigning a ListComp to a Name (line 1288):
    
    # Assigning a ListComp to a Name (line 1288):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1288)
    # Processing the call arguments (line 1288)
    # Getting the type of 'ideg' (line 1288)
    ideg_169659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1288)
    deg_169660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 59), 'deg', False)
    # Processing the call keyword arguments (line 1288)
    kwargs_169661 = {}
    # Getting the type of 'zip' (line 1288)
    zip_169658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1288)
    zip_call_result_169662 = invoke(stypy.reporting.localization.Localization(__file__, 1288, 49), zip_169658, *[ideg_169659, deg_169660], **kwargs_169661)
    
    comprehension_169663 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1288, 16), zip_call_result_169662)
    # Assigning a type to the variable 'id' (line 1288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1288, 16), comprehension_169663))
    # Assigning a type to the variable 'd' (line 1288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1288, 16), comprehension_169663))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1288)
    id_169651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 16), 'id')
    # Getting the type of 'd' (line 1288)
    d_169652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 22), 'd')
    # Applying the binary operator '==' (line 1288)
    result_eq_169653 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 16), '==', id_169651, d_169652)
    
    
    # Getting the type of 'id' (line 1288)
    id_169654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1288, 28), 'id')
    int_169655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 34), 'int')
    # Applying the binary operator '>=' (line 1288)
    result_ge_169656 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 28), '>=', id_169654, int_169655)
    
    # Applying the binary operator 'and' (line 1288)
    result_and_keyword_169657 = python_operator(stypy.reporting.localization.Localization(__file__, 1288, 16), 'and', result_eq_169653, result_ge_169656)
    
    list_169664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1288, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1288, 16), list_169664, result_and_keyword_169657)
    # Assigning a type to the variable 'is_valid' (line 1288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1288, 4), 'is_valid', list_169664)
    
    
    # Getting the type of 'is_valid' (line 1289)
    is_valid_169665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1289, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1289)
    list_169666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1289)
    # Adding element type (line 1289)
    int_169667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1289, 19), list_169666, int_169667)
    # Adding element type (line 1289)
    int_169668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1289, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1289, 19), list_169666, int_169668)
    
    # Applying the binary operator '!=' (line 1289)
    result_ne_169669 = python_operator(stypy.reporting.localization.Localization(__file__, 1289, 7), '!=', is_valid_169665, list_169666)
    
    # Testing the type of an if condition (line 1289)
    if_condition_169670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1289, 4), result_ne_169669)
    # Assigning a type to the variable 'if_condition_169670' (line 1289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1289, 4), 'if_condition_169670', if_condition_169670)
    # SSA begins for if statement (line 1289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1290)
    # Processing the call arguments (line 1290)
    str_169672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1290, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1290)
    kwargs_169673 = {}
    # Getting the type of 'ValueError' (line 1290)
    ValueError_169671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1290, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1290)
    ValueError_call_result_169674 = invoke(stypy.reporting.localization.Localization(__file__, 1290, 14), ValueError_169671, *[str_169672], **kwargs_169673)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1290, 8), ValueError_call_result_169674, 'raise parameter', BaseException)
    # SSA join for if statement (line 1289)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1291):
    
    # Assigning a Subscript to a Name (line 1291):
    
    # Obtaining the type of the subscript
    int_169675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 4), 'int')
    # Getting the type of 'ideg' (line 1291)
    ideg_169676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1291)
    getitem___169677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1291, 4), ideg_169676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1291)
    subscript_call_result_169678 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 4), getitem___169677, int_169675)
    
    # Assigning a type to the variable 'tuple_var_assignment_167948' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'tuple_var_assignment_167948', subscript_call_result_169678)
    
    # Assigning a Subscript to a Name (line 1291):
    
    # Obtaining the type of the subscript
    int_169679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1291, 4), 'int')
    # Getting the type of 'ideg' (line 1291)
    ideg_169680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 17), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1291)
    getitem___169681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1291, 4), ideg_169680, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1291)
    subscript_call_result_169682 = invoke(stypy.reporting.localization.Localization(__file__, 1291, 4), getitem___169681, int_169679)
    
    # Assigning a type to the variable 'tuple_var_assignment_167949' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'tuple_var_assignment_167949', subscript_call_result_169682)
    
    # Assigning a Name to a Name (line 1291):
    # Getting the type of 'tuple_var_assignment_167948' (line 1291)
    tuple_var_assignment_167948_169683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'tuple_var_assignment_167948')
    # Assigning a type to the variable 'degx' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'degx', tuple_var_assignment_167948_169683)
    
    # Assigning a Name to a Name (line 1291):
    # Getting the type of 'tuple_var_assignment_167949' (line 1291)
    tuple_var_assignment_167949_169684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1291, 4), 'tuple_var_assignment_167949')
    # Assigning a type to the variable 'degy' (line 1291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1291, 10), 'degy', tuple_var_assignment_167949_169684)
    
    # Assigning a BinOp to a Tuple (line 1292):
    
    # Assigning a Subscript to a Name (line 1292):
    
    # Obtaining the type of the subscript
    int_169685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 4), 'int')
    
    # Call to array(...): (line 1292)
    # Processing the call arguments (line 1292)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1292)
    tuple_169688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1292)
    # Adding element type (line 1292)
    # Getting the type of 'x' (line 1292)
    x_169689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 21), tuple_169688, x_169689)
    # Adding element type (line 1292)
    # Getting the type of 'y' (line 1292)
    y_169690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 21), tuple_169688, y_169690)
    
    # Processing the call keyword arguments (line 1292)
    int_169691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 33), 'int')
    keyword_169692 = int_169691
    kwargs_169693 = {'copy': keyword_169692}
    # Getting the type of 'np' (line 1292)
    np_169686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1292)
    array_169687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 11), np_169686, 'array')
    # Calling array(args, kwargs) (line 1292)
    array_call_result_169694 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 11), array_169687, *[tuple_169688], **kwargs_169693)
    
    float_169695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 38), 'float')
    # Applying the binary operator '+' (line 1292)
    result_add_169696 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 11), '+', array_call_result_169694, float_169695)
    
    # Obtaining the member '__getitem__' of a type (line 1292)
    getitem___169697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 4), result_add_169696, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1292)
    subscript_call_result_169698 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 4), getitem___169697, int_169685)
    
    # Assigning a type to the variable 'tuple_var_assignment_167950' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'tuple_var_assignment_167950', subscript_call_result_169698)
    
    # Assigning a Subscript to a Name (line 1292):
    
    # Obtaining the type of the subscript
    int_169699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 4), 'int')
    
    # Call to array(...): (line 1292)
    # Processing the call arguments (line 1292)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1292)
    tuple_169702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1292)
    # Adding element type (line 1292)
    # Getting the type of 'x' (line 1292)
    x_169703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 21), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 21), tuple_169702, x_169703)
    # Adding element type (line 1292)
    # Getting the type of 'y' (line 1292)
    y_169704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 24), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1292, 21), tuple_169702, y_169704)
    
    # Processing the call keyword arguments (line 1292)
    int_169705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 33), 'int')
    keyword_169706 = int_169705
    kwargs_169707 = {'copy': keyword_169706}
    # Getting the type of 'np' (line 1292)
    np_169700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 1292)
    array_169701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 11), np_169700, 'array')
    # Calling array(args, kwargs) (line 1292)
    array_call_result_169708 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 11), array_169701, *[tuple_169702], **kwargs_169707)
    
    float_169709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1292, 38), 'float')
    # Applying the binary operator '+' (line 1292)
    result_add_169710 = python_operator(stypy.reporting.localization.Localization(__file__, 1292, 11), '+', array_call_result_169708, float_169709)
    
    # Obtaining the member '__getitem__' of a type (line 1292)
    getitem___169711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1292, 4), result_add_169710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1292)
    subscript_call_result_169712 = invoke(stypy.reporting.localization.Localization(__file__, 1292, 4), getitem___169711, int_169699)
    
    # Assigning a type to the variable 'tuple_var_assignment_167951' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'tuple_var_assignment_167951', subscript_call_result_169712)
    
    # Assigning a Name to a Name (line 1292):
    # Getting the type of 'tuple_var_assignment_167950' (line 1292)
    tuple_var_assignment_167950_169713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'tuple_var_assignment_167950')
    # Assigning a type to the variable 'x' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'x', tuple_var_assignment_167950_169713)
    
    # Assigning a Name to a Name (line 1292):
    # Getting the type of 'tuple_var_assignment_167951' (line 1292)
    tuple_var_assignment_167951_169714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1292, 4), 'tuple_var_assignment_167951')
    # Assigning a type to the variable 'y' (line 1292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1292, 7), 'y', tuple_var_assignment_167951_169714)
    
    # Assigning a Call to a Name (line 1294):
    
    # Assigning a Call to a Name (line 1294):
    
    # Call to hermevander(...): (line 1294)
    # Processing the call arguments (line 1294)
    # Getting the type of 'x' (line 1294)
    x_169716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 21), 'x', False)
    # Getting the type of 'degx' (line 1294)
    degx_169717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 24), 'degx', False)
    # Processing the call keyword arguments (line 1294)
    kwargs_169718 = {}
    # Getting the type of 'hermevander' (line 1294)
    hermevander_169715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1294, 9), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1294)
    hermevander_call_result_169719 = invoke(stypy.reporting.localization.Localization(__file__, 1294, 9), hermevander_169715, *[x_169716, degx_169717], **kwargs_169718)
    
    # Assigning a type to the variable 'vx' (line 1294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1294, 4), 'vx', hermevander_call_result_169719)
    
    # Assigning a Call to a Name (line 1295):
    
    # Assigning a Call to a Name (line 1295):
    
    # Call to hermevander(...): (line 1295)
    # Processing the call arguments (line 1295)
    # Getting the type of 'y' (line 1295)
    y_169721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 21), 'y', False)
    # Getting the type of 'degy' (line 1295)
    degy_169722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 24), 'degy', False)
    # Processing the call keyword arguments (line 1295)
    kwargs_169723 = {}
    # Getting the type of 'hermevander' (line 1295)
    hermevander_169720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1295, 9), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1295)
    hermevander_call_result_169724 = invoke(stypy.reporting.localization.Localization(__file__, 1295, 9), hermevander_169720, *[y_169721, degy_169722], **kwargs_169723)
    
    # Assigning a type to the variable 'vy' (line 1295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1295, 4), 'vy', hermevander_call_result_169724)
    
    # Assigning a BinOp to a Name (line 1296):
    
    # Assigning a BinOp to a Name (line 1296):
    
    # Obtaining the type of the subscript
    Ellipsis_169725 = Ellipsis
    # Getting the type of 'None' (line 1296)
    None_169726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 16), 'None')
    # Getting the type of 'vx' (line 1296)
    vx_169727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1296)
    getitem___169728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1296, 8), vx_169727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1296)
    subscript_call_result_169729 = invoke(stypy.reporting.localization.Localization(__file__, 1296, 8), getitem___169728, (Ellipsis_169725, None_169726))
    
    
    # Obtaining the type of the subscript
    Ellipsis_169730 = Ellipsis
    # Getting the type of 'None' (line 1296)
    None_169731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 30), 'None')
    slice_169732 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1296, 22), None, None, None)
    # Getting the type of 'vy' (line 1296)
    vy_169733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1296, 22), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1296)
    getitem___169734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1296, 22), vy_169733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1296)
    subscript_call_result_169735 = invoke(stypy.reporting.localization.Localization(__file__, 1296, 22), getitem___169734, (Ellipsis_169730, None_169731, slice_169732))
    
    # Applying the binary operator '*' (line 1296)
    result_mul_169736 = python_operator(stypy.reporting.localization.Localization(__file__, 1296, 8), '*', subscript_call_result_169729, subscript_call_result_169735)
    
    # Assigning a type to the variable 'v' (line 1296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1296, 4), 'v', result_mul_169736)
    
    # Call to reshape(...): (line 1297)
    # Processing the call arguments (line 1297)
    
    # Obtaining the type of the subscript
    int_169739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 30), 'int')
    slice_169740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1297, 21), None, int_169739, None)
    # Getting the type of 'v' (line 1297)
    v_169741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1297)
    shape_169742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1297, 21), v_169741, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1297)
    getitem___169743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1297, 21), shape_169742, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1297)
    subscript_call_result_169744 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 21), getitem___169743, slice_169740)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1297)
    tuple_169745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1297)
    # Adding element type (line 1297)
    int_169746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1297, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1297, 37), tuple_169745, int_169746)
    
    # Applying the binary operator '+' (line 1297)
    result_add_169747 = python_operator(stypy.reporting.localization.Localization(__file__, 1297, 21), '+', subscript_call_result_169744, tuple_169745)
    
    # Processing the call keyword arguments (line 1297)
    kwargs_169748 = {}
    # Getting the type of 'v' (line 1297)
    v_169737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1297, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1297)
    reshape_169738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1297, 11), v_169737, 'reshape')
    # Calling reshape(args, kwargs) (line 1297)
    reshape_call_result_169749 = invoke(stypy.reporting.localization.Localization(__file__, 1297, 11), reshape_169738, *[result_add_169747], **kwargs_169748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1297, 4), 'stypy_return_type', reshape_call_result_169749)
    
    # ################# End of 'hermevander2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermevander2d' in the type store
    # Getting the type of 'stypy_return_type' (line 1237)
    stypy_return_type_169750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169750)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermevander2d'
    return stypy_return_type_169750

# Assigning a type to the variable 'hermevander2d' (line 1237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 0), 'hermevander2d', hermevander2d)

@norecursion
def hermevander3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermevander3d'
    module_type_store = module_type_store.open_function_context('hermevander3d', 1300, 0, False)
    
    # Passed parameters checking function
    hermevander3d.stypy_localization = localization
    hermevander3d.stypy_type_of_self = None
    hermevander3d.stypy_type_store = module_type_store
    hermevander3d.stypy_function_name = 'hermevander3d'
    hermevander3d.stypy_param_names_list = ['x', 'y', 'z', 'deg']
    hermevander3d.stypy_varargs_param_name = None
    hermevander3d.stypy_kwargs_param_name = None
    hermevander3d.stypy_call_defaults = defaults
    hermevander3d.stypy_call_varargs = varargs
    hermevander3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermevander3d', ['x', 'y', 'z', 'deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermevander3d', localization, ['x', 'y', 'z', 'deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermevander3d(...)' code ##################

    str_169751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1350, (-1)), 'str', 'Pseudo-Vandermonde matrix of given degrees.\n\n    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample\n    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,\n    then Hehe pseudo-Vandermonde matrix is defined by\n\n    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = He_i(x)*He_j(y)*He_k(z),\n\n    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading\n    indices of `V` index the points `(x, y, z)` and the last index encodes\n    the degrees of the HermiteE polynomials.\n\n    If ``V = hermevander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns\n    of `V` correspond to the elements of a 3-D coefficient array `c` of\n    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order\n\n    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...\n\n    and  ``np.dot(V, c.flat)`` and ``hermeval3d(x, y, z, c)`` will be the\n    same up to roundoff. This equivalence is useful both for least squares\n    fitting and for the evaluation of a large number of 3-D HermiteE\n    series of the same degrees and sample points.\n\n    Parameters\n    ----------\n    x, y, z : array_like\n        Arrays of point coordinates, all of the same shape. The dtypes will\n        be converted to either float64 or complex128 depending on whether\n        any of the elements are complex. Scalars are converted to 1-D\n        arrays.\n    deg : list of ints\n        List of maximum degrees of the form [x_deg, y_deg, z_deg].\n\n    Returns\n    -------\n    vander3d : ndarray\n        The shape of the returned matrix is ``x.shape + (order,)``, where\n        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will\n        be the same as the converted `x`, `y`, and `z`.\n\n    See Also\n    --------\n    hermevander, hermevander3d. hermeval2d, hermeval3d\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a ListComp to a Name (line 1351):
    
    # Assigning a ListComp to a Name (line 1351):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'deg' (line 1351)
    deg_169756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 28), 'deg')
    comprehension_169757 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1351, 12), deg_169756)
    # Assigning a type to the variable 'd' (line 1351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1351, 12), 'd', comprehension_169757)
    
    # Call to int(...): (line 1351)
    # Processing the call arguments (line 1351)
    # Getting the type of 'd' (line 1351)
    d_169753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 16), 'd', False)
    # Processing the call keyword arguments (line 1351)
    kwargs_169754 = {}
    # Getting the type of 'int' (line 1351)
    int_169752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1351, 12), 'int', False)
    # Calling int(args, kwargs) (line 1351)
    int_call_result_169755 = invoke(stypy.reporting.localization.Localization(__file__, 1351, 12), int_169752, *[d_169753], **kwargs_169754)
    
    list_169758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1351, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1351, 12), list_169758, int_call_result_169755)
    # Assigning a type to the variable 'ideg' (line 1351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1351, 4), 'ideg', list_169758)
    
    # Assigning a ListComp to a Name (line 1352):
    
    # Assigning a ListComp to a Name (line 1352):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 1352)
    # Processing the call arguments (line 1352)
    # Getting the type of 'ideg' (line 1352)
    ideg_169767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 53), 'ideg', False)
    # Getting the type of 'deg' (line 1352)
    deg_169768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 59), 'deg', False)
    # Processing the call keyword arguments (line 1352)
    kwargs_169769 = {}
    # Getting the type of 'zip' (line 1352)
    zip_169766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 49), 'zip', False)
    # Calling zip(args, kwargs) (line 1352)
    zip_call_result_169770 = invoke(stypy.reporting.localization.Localization(__file__, 1352, 49), zip_169766, *[ideg_169767, deg_169768], **kwargs_169769)
    
    comprehension_169771 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1352, 16), zip_call_result_169770)
    # Assigning a type to the variable 'id' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 16), 'id', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1352, 16), comprehension_169771))
    # Assigning a type to the variable 'd' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 16), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1352, 16), comprehension_169771))
    
    # Evaluating a boolean operation
    
    # Getting the type of 'id' (line 1352)
    id_169759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 16), 'id')
    # Getting the type of 'd' (line 1352)
    d_169760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 22), 'd')
    # Applying the binary operator '==' (line 1352)
    result_eq_169761 = python_operator(stypy.reporting.localization.Localization(__file__, 1352, 16), '==', id_169759, d_169760)
    
    
    # Getting the type of 'id' (line 1352)
    id_169762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1352, 28), 'id')
    int_169763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1352, 34), 'int')
    # Applying the binary operator '>=' (line 1352)
    result_ge_169764 = python_operator(stypy.reporting.localization.Localization(__file__, 1352, 28), '>=', id_169762, int_169763)
    
    # Applying the binary operator 'and' (line 1352)
    result_and_keyword_169765 = python_operator(stypy.reporting.localization.Localization(__file__, 1352, 16), 'and', result_eq_169761, result_ge_169764)
    
    list_169772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1352, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1352, 16), list_169772, result_and_keyword_169765)
    # Assigning a type to the variable 'is_valid' (line 1352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1352, 4), 'is_valid', list_169772)
    
    
    # Getting the type of 'is_valid' (line 1353)
    is_valid_169773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1353, 7), 'is_valid')
    
    # Obtaining an instance of the builtin type 'list' (line 1353)
    list_169774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1353)
    # Adding element type (line 1353)
    int_169775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1353, 19), list_169774, int_169775)
    # Adding element type (line 1353)
    int_169776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1353, 19), list_169774, int_169776)
    # Adding element type (line 1353)
    int_169777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1353, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1353, 19), list_169774, int_169777)
    
    # Applying the binary operator '!=' (line 1353)
    result_ne_169778 = python_operator(stypy.reporting.localization.Localization(__file__, 1353, 7), '!=', is_valid_169773, list_169774)
    
    # Testing the type of an if condition (line 1353)
    if_condition_169779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1353, 4), result_ne_169778)
    # Assigning a type to the variable 'if_condition_169779' (line 1353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1353, 4), 'if_condition_169779', if_condition_169779)
    # SSA begins for if statement (line 1353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1354)
    # Processing the call arguments (line 1354)
    str_169781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1354, 25), 'str', 'degrees must be non-negative integers')
    # Processing the call keyword arguments (line 1354)
    kwargs_169782 = {}
    # Getting the type of 'ValueError' (line 1354)
    ValueError_169780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1354, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1354)
    ValueError_call_result_169783 = invoke(stypy.reporting.localization.Localization(__file__, 1354, 14), ValueError_169780, *[str_169781], **kwargs_169782)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1354, 8), ValueError_call_result_169783, 'raise parameter', BaseException)
    # SSA join for if statement (line 1353)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 1355):
    
    # Assigning a Subscript to a Name (line 1355):
    
    # Obtaining the type of the subscript
    int_169784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 4), 'int')
    # Getting the type of 'ideg' (line 1355)
    ideg_169785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1355)
    getitem___169786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 4), ideg_169785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1355)
    subscript_call_result_169787 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 4), getitem___169786, int_169784)
    
    # Assigning a type to the variable 'tuple_var_assignment_167952' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167952', subscript_call_result_169787)
    
    # Assigning a Subscript to a Name (line 1355):
    
    # Obtaining the type of the subscript
    int_169788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 4), 'int')
    # Getting the type of 'ideg' (line 1355)
    ideg_169789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1355)
    getitem___169790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 4), ideg_169789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1355)
    subscript_call_result_169791 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 4), getitem___169790, int_169788)
    
    # Assigning a type to the variable 'tuple_var_assignment_167953' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167953', subscript_call_result_169791)
    
    # Assigning a Subscript to a Name (line 1355):
    
    # Obtaining the type of the subscript
    int_169792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1355, 4), 'int')
    # Getting the type of 'ideg' (line 1355)
    ideg_169793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 23), 'ideg')
    # Obtaining the member '__getitem__' of a type (line 1355)
    getitem___169794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1355, 4), ideg_169793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1355)
    subscript_call_result_169795 = invoke(stypy.reporting.localization.Localization(__file__, 1355, 4), getitem___169794, int_169792)
    
    # Assigning a type to the variable 'tuple_var_assignment_167954' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167954', subscript_call_result_169795)
    
    # Assigning a Name to a Name (line 1355):
    # Getting the type of 'tuple_var_assignment_167952' (line 1355)
    tuple_var_assignment_167952_169796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167952')
    # Assigning a type to the variable 'degx' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'degx', tuple_var_assignment_167952_169796)
    
    # Assigning a Name to a Name (line 1355):
    # Getting the type of 'tuple_var_assignment_167953' (line 1355)
    tuple_var_assignment_167953_169797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167953')
    # Assigning a type to the variable 'degy' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 10), 'degy', tuple_var_assignment_167953_169797)
    
    # Assigning a Name to a Name (line 1355):
    # Getting the type of 'tuple_var_assignment_167954' (line 1355)
    tuple_var_assignment_167954_169798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1355, 4), 'tuple_var_assignment_167954')
    # Assigning a type to the variable 'degz' (line 1355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1355, 16), 'degz', tuple_var_assignment_167954_169798)
    
    # Assigning a BinOp to a Tuple (line 1356):
    
    # Assigning a Subscript to a Name (line 1356):
    
    # Obtaining the type of the subscript
    int_169799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 4), 'int')
    
    # Call to array(...): (line 1356)
    # Processing the call arguments (line 1356)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1356)
    tuple_169802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1356)
    # Adding element type (line 1356)
    # Getting the type of 'x' (line 1356)
    x_169803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169802, x_169803)
    # Adding element type (line 1356)
    # Getting the type of 'y' (line 1356)
    y_169804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169802, y_169804)
    # Adding element type (line 1356)
    # Getting the type of 'z' (line 1356)
    z_169805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169802, z_169805)
    
    # Processing the call keyword arguments (line 1356)
    int_169806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 39), 'int')
    keyword_169807 = int_169806
    kwargs_169808 = {'copy': keyword_169807}
    # Getting the type of 'np' (line 1356)
    np_169800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1356)
    array_169801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 14), np_169800, 'array')
    # Calling array(args, kwargs) (line 1356)
    array_call_result_169809 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 14), array_169801, *[tuple_169802], **kwargs_169808)
    
    float_169810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 44), 'float')
    # Applying the binary operator '+' (line 1356)
    result_add_169811 = python_operator(stypy.reporting.localization.Localization(__file__, 1356, 14), '+', array_call_result_169809, float_169810)
    
    # Obtaining the member '__getitem__' of a type (line 1356)
    getitem___169812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 4), result_add_169811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1356)
    subscript_call_result_169813 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 4), getitem___169812, int_169799)
    
    # Assigning a type to the variable 'tuple_var_assignment_167955' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167955', subscript_call_result_169813)
    
    # Assigning a Subscript to a Name (line 1356):
    
    # Obtaining the type of the subscript
    int_169814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 4), 'int')
    
    # Call to array(...): (line 1356)
    # Processing the call arguments (line 1356)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1356)
    tuple_169817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1356)
    # Adding element type (line 1356)
    # Getting the type of 'x' (line 1356)
    x_169818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169817, x_169818)
    # Adding element type (line 1356)
    # Getting the type of 'y' (line 1356)
    y_169819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169817, y_169819)
    # Adding element type (line 1356)
    # Getting the type of 'z' (line 1356)
    z_169820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169817, z_169820)
    
    # Processing the call keyword arguments (line 1356)
    int_169821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 39), 'int')
    keyword_169822 = int_169821
    kwargs_169823 = {'copy': keyword_169822}
    # Getting the type of 'np' (line 1356)
    np_169815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1356)
    array_169816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 14), np_169815, 'array')
    # Calling array(args, kwargs) (line 1356)
    array_call_result_169824 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 14), array_169816, *[tuple_169817], **kwargs_169823)
    
    float_169825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 44), 'float')
    # Applying the binary operator '+' (line 1356)
    result_add_169826 = python_operator(stypy.reporting.localization.Localization(__file__, 1356, 14), '+', array_call_result_169824, float_169825)
    
    # Obtaining the member '__getitem__' of a type (line 1356)
    getitem___169827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 4), result_add_169826, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1356)
    subscript_call_result_169828 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 4), getitem___169827, int_169814)
    
    # Assigning a type to the variable 'tuple_var_assignment_167956' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167956', subscript_call_result_169828)
    
    # Assigning a Subscript to a Name (line 1356):
    
    # Obtaining the type of the subscript
    int_169829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 4), 'int')
    
    # Call to array(...): (line 1356)
    # Processing the call arguments (line 1356)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1356)
    tuple_169832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1356)
    # Adding element type (line 1356)
    # Getting the type of 'x' (line 1356)
    x_169833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 24), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169832, x_169833)
    # Adding element type (line 1356)
    # Getting the type of 'y' (line 1356)
    y_169834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 27), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169832, y_169834)
    # Adding element type (line 1356)
    # Getting the type of 'z' (line 1356)
    z_169835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 30), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1356, 24), tuple_169832, z_169835)
    
    # Processing the call keyword arguments (line 1356)
    int_169836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 39), 'int')
    keyword_169837 = int_169836
    kwargs_169838 = {'copy': keyword_169837}
    # Getting the type of 'np' (line 1356)
    np_169830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 14), 'np', False)
    # Obtaining the member 'array' of a type (line 1356)
    array_169831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 14), np_169830, 'array')
    # Calling array(args, kwargs) (line 1356)
    array_call_result_169839 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 14), array_169831, *[tuple_169832], **kwargs_169838)
    
    float_169840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1356, 44), 'float')
    # Applying the binary operator '+' (line 1356)
    result_add_169841 = python_operator(stypy.reporting.localization.Localization(__file__, 1356, 14), '+', array_call_result_169839, float_169840)
    
    # Obtaining the member '__getitem__' of a type (line 1356)
    getitem___169842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1356, 4), result_add_169841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1356)
    subscript_call_result_169843 = invoke(stypy.reporting.localization.Localization(__file__, 1356, 4), getitem___169842, int_169829)
    
    # Assigning a type to the variable 'tuple_var_assignment_167957' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167957', subscript_call_result_169843)
    
    # Assigning a Name to a Name (line 1356):
    # Getting the type of 'tuple_var_assignment_167955' (line 1356)
    tuple_var_assignment_167955_169844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167955')
    # Assigning a type to the variable 'x' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'x', tuple_var_assignment_167955_169844)
    
    # Assigning a Name to a Name (line 1356):
    # Getting the type of 'tuple_var_assignment_167956' (line 1356)
    tuple_var_assignment_167956_169845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167956')
    # Assigning a type to the variable 'y' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 7), 'y', tuple_var_assignment_167956_169845)
    
    # Assigning a Name to a Name (line 1356):
    # Getting the type of 'tuple_var_assignment_167957' (line 1356)
    tuple_var_assignment_167957_169846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1356, 4), 'tuple_var_assignment_167957')
    # Assigning a type to the variable 'z' (line 1356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1356, 10), 'z', tuple_var_assignment_167957_169846)
    
    # Assigning a Call to a Name (line 1358):
    
    # Assigning a Call to a Name (line 1358):
    
    # Call to hermevander(...): (line 1358)
    # Processing the call arguments (line 1358)
    # Getting the type of 'x' (line 1358)
    x_169848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 21), 'x', False)
    # Getting the type of 'degx' (line 1358)
    degx_169849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 24), 'degx', False)
    # Processing the call keyword arguments (line 1358)
    kwargs_169850 = {}
    # Getting the type of 'hermevander' (line 1358)
    hermevander_169847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1358, 9), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1358)
    hermevander_call_result_169851 = invoke(stypy.reporting.localization.Localization(__file__, 1358, 9), hermevander_169847, *[x_169848, degx_169849], **kwargs_169850)
    
    # Assigning a type to the variable 'vx' (line 1358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1358, 4), 'vx', hermevander_call_result_169851)
    
    # Assigning a Call to a Name (line 1359):
    
    # Assigning a Call to a Name (line 1359):
    
    # Call to hermevander(...): (line 1359)
    # Processing the call arguments (line 1359)
    # Getting the type of 'y' (line 1359)
    y_169853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 21), 'y', False)
    # Getting the type of 'degy' (line 1359)
    degy_169854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 24), 'degy', False)
    # Processing the call keyword arguments (line 1359)
    kwargs_169855 = {}
    # Getting the type of 'hermevander' (line 1359)
    hermevander_169852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1359, 9), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1359)
    hermevander_call_result_169856 = invoke(stypy.reporting.localization.Localization(__file__, 1359, 9), hermevander_169852, *[y_169853, degy_169854], **kwargs_169855)
    
    # Assigning a type to the variable 'vy' (line 1359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1359, 4), 'vy', hermevander_call_result_169856)
    
    # Assigning a Call to a Name (line 1360):
    
    # Assigning a Call to a Name (line 1360):
    
    # Call to hermevander(...): (line 1360)
    # Processing the call arguments (line 1360)
    # Getting the type of 'z' (line 1360)
    z_169858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 21), 'z', False)
    # Getting the type of 'degz' (line 1360)
    degz_169859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 24), 'degz', False)
    # Processing the call keyword arguments (line 1360)
    kwargs_169860 = {}
    # Getting the type of 'hermevander' (line 1360)
    hermevander_169857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1360, 9), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1360)
    hermevander_call_result_169861 = invoke(stypy.reporting.localization.Localization(__file__, 1360, 9), hermevander_169857, *[z_169858, degz_169859], **kwargs_169860)
    
    # Assigning a type to the variable 'vz' (line 1360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1360, 4), 'vz', hermevander_call_result_169861)
    
    # Assigning a BinOp to a Name (line 1361):
    
    # Assigning a BinOp to a Name (line 1361):
    
    # Obtaining the type of the subscript
    Ellipsis_169862 = Ellipsis
    # Getting the type of 'None' (line 1361)
    None_169863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 16), 'None')
    # Getting the type of 'None' (line 1361)
    None_169864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 22), 'None')
    # Getting the type of 'vx' (line 1361)
    vx_169865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 8), 'vx')
    # Obtaining the member '__getitem__' of a type (line 1361)
    getitem___169866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 8), vx_169865, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1361)
    subscript_call_result_169867 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 8), getitem___169866, (Ellipsis_169862, None_169863, None_169864))
    
    
    # Obtaining the type of the subscript
    Ellipsis_169868 = Ellipsis
    # Getting the type of 'None' (line 1361)
    None_169869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 36), 'None')
    slice_169870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1361, 28), None, None, None)
    # Getting the type of 'None' (line 1361)
    None_169871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 44), 'None')
    # Getting the type of 'vy' (line 1361)
    vy_169872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 28), 'vy')
    # Obtaining the member '__getitem__' of a type (line 1361)
    getitem___169873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 28), vy_169872, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1361)
    subscript_call_result_169874 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 28), getitem___169873, (Ellipsis_169868, None_169869, slice_169870, None_169871))
    
    # Applying the binary operator '*' (line 1361)
    result_mul_169875 = python_operator(stypy.reporting.localization.Localization(__file__, 1361, 8), '*', subscript_call_result_169867, subscript_call_result_169874)
    
    
    # Obtaining the type of the subscript
    Ellipsis_169876 = Ellipsis
    # Getting the type of 'None' (line 1361)
    None_169877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 58), 'None')
    # Getting the type of 'None' (line 1361)
    None_169878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 64), 'None')
    slice_169879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1361, 50), None, None, None)
    # Getting the type of 'vz' (line 1361)
    vz_169880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 50), 'vz')
    # Obtaining the member '__getitem__' of a type (line 1361)
    getitem___169881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 50), vz_169880, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1361)
    subscript_call_result_169882 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 50), getitem___169881, (Ellipsis_169876, None_169877, None_169878, slice_169879))
    
    # Applying the binary operator '*' (line 1361)
    result_mul_169883 = python_operator(stypy.reporting.localization.Localization(__file__, 1361, 49), '*', result_mul_169875, subscript_call_result_169882)
    
    # Assigning a type to the variable 'v' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), 'v', result_mul_169883)
    
    # Call to reshape(...): (line 1362)
    # Processing the call arguments (line 1362)
    
    # Obtaining the type of the subscript
    int_169886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 30), 'int')
    slice_169887 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1362, 21), None, int_169886, None)
    # Getting the type of 'v' (line 1362)
    v_169888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 21), 'v', False)
    # Obtaining the member 'shape' of a type (line 1362)
    shape_169889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 21), v_169888, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1362)
    getitem___169890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 21), shape_169889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1362)
    subscript_call_result_169891 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 21), getitem___169890, slice_169887)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1362)
    tuple_169892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1362)
    # Adding element type (line 1362)
    int_169893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1362, 37), tuple_169892, int_169893)
    
    # Applying the binary operator '+' (line 1362)
    result_add_169894 = python_operator(stypy.reporting.localization.Localization(__file__, 1362, 21), '+', subscript_call_result_169891, tuple_169892)
    
    # Processing the call keyword arguments (line 1362)
    kwargs_169895 = {}
    # Getting the type of 'v' (line 1362)
    v_169884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 11), 'v', False)
    # Obtaining the member 'reshape' of a type (line 1362)
    reshape_169885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 11), v_169884, 'reshape')
    # Calling reshape(args, kwargs) (line 1362)
    reshape_call_result_169896 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 11), reshape_169885, *[result_add_169894], **kwargs_169895)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'stypy_return_type', reshape_call_result_169896)
    
    # ################# End of 'hermevander3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermevander3d' in the type store
    # Getting the type of 'stypy_return_type' (line 1300)
    stypy_return_type_169897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_169897)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermevander3d'
    return stypy_return_type_169897

# Assigning a type to the variable 'hermevander3d' (line 1300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 0), 'hermevander3d', hermevander3d)

@norecursion
def hermefit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1365)
    None_169898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 30), 'None')
    # Getting the type of 'False' (line 1365)
    False_169899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 41), 'False')
    # Getting the type of 'None' (line 1365)
    None_169900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 50), 'None')
    defaults = [None_169898, False_169899, None_169900]
    # Create a new context for function 'hermefit'
    module_type_store = module_type_store.open_function_context('hermefit', 1365, 0, False)
    
    # Passed parameters checking function
    hermefit.stypy_localization = localization
    hermefit.stypy_type_of_self = None
    hermefit.stypy_type_store = module_type_store
    hermefit.stypy_function_name = 'hermefit'
    hermefit.stypy_param_names_list = ['x', 'y', 'deg', 'rcond', 'full', 'w']
    hermefit.stypy_varargs_param_name = None
    hermefit.stypy_kwargs_param_name = None
    hermefit.stypy_call_defaults = defaults
    hermefit.stypy_call_varargs = varargs
    hermefit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermefit', ['x', 'y', 'deg', 'rcond', 'full', 'w'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermefit', localization, ['x', 'y', 'deg', 'rcond', 'full', 'w'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermefit(...)' code ##################

    str_169901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1488, (-1)), 'str', '\n    Least squares fit of Hermite series to data.\n\n    Return the coefficients of a HermiteE series of degree `deg` that is\n    the least squares fit to the data values `y` given at points `x`. If\n    `y` is 1-D the returned coefficients will also be 1-D. If `y` is 2-D\n    multiple fits are done, one for each column of `y`, and the resulting\n    coefficients are stored in the corresponding columns of a 2-D return.\n    The fitted polynomial(s) are in the form\n\n    .. math::  p(x) = c_0 + c_1 * He_1(x) + ... + c_n * He_n(x),\n\n    where `n` is `deg`.\n\n    Parameters\n    ----------\n    x : array_like, shape (M,)\n        x-coordinates of the M sample points ``(x[i], y[i])``.\n    y : array_like, shape (M,) or (M, K)\n        y-coordinates of the sample points. Several data sets of sample\n        points sharing the same x-coordinates can be fitted at once by\n        passing in a 2D-array that contains one dataset per column.\n    deg : int or 1-D array_like\n        Degree(s) of the fitting polynomials. If `deg` is a single integer\n        all terms up to and including the `deg`\'th term are included in the\n        fit. For Numpy versions >= 1.11 a list of integers specifying the\n        degrees of the terms to include may be used instead.\n    rcond : float, optional\n        Relative condition number of the fit. Singular values smaller than\n        this relative to the largest singular value will be ignored. The\n        default value is len(x)*eps, where eps is the relative precision of\n        the float type, about 2e-16 in most cases.\n    full : bool, optional\n        Switch determining nature of return value. When it is False (the\n        default) just the coefficients are returned, when True diagnostic\n        information from the singular value decomposition is also returned.\n    w : array_like, shape (`M`,), optional\n        Weights. If not None, the contribution of each point\n        ``(x[i],y[i])`` to the fit is weighted by `w[i]`. Ideally the\n        weights are chosen so that the errors of the products ``w[i]*y[i]``\n        all have the same variance.  The default value is None.\n\n    Returns\n    -------\n    coef : ndarray, shape (M,) or (M, K)\n        Hermite coefficients ordered from low to high. If `y` was 2-D,\n        the coefficients for the data in column k  of `y` are in column\n        `k`.\n\n    [residuals, rank, singular_values, rcond] : list\n        These values are only returned if `full` = True\n\n        resid -- sum of squared residuals of the least squares fit\n        rank -- the numerical rank of the scaled Vandermonde matrix\n        sv -- singular values of the scaled Vandermonde matrix\n        rcond -- value of `rcond`.\n\n        For more details, see `linalg.lstsq`.\n\n    Warns\n    -----\n    RankWarning\n        The rank of the coefficient matrix in the least-squares fit is\n        deficient. The warning is only raised if `full` = False.  The\n        warnings can be turned off by\n\n        >>> import warnings\n        >>> warnings.simplefilter(\'ignore\', RankWarning)\n\n    See Also\n    --------\n    chebfit, legfit, polyfit, hermfit, polyfit\n    hermeval : Evaluates a Hermite series.\n    hermevander : pseudo Vandermonde matrix of Hermite series.\n    hermeweight : HermiteE weight function.\n    linalg.lstsq : Computes a least-squares fit from the matrix.\n    scipy.interpolate.UnivariateSpline : Computes spline fits.\n\n    Notes\n    -----\n    The solution is the coefficients of the HermiteE series `p` that\n    minimizes the sum of the weighted squared errors\n\n    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,\n\n    where the :math:`w_j` are the weights. This problem is solved by\n    setting up the (typically) overdetermined matrix equation\n\n    .. math:: V(x) * c = w * y,\n\n    where `V` is the pseudo Vandermonde matrix of `x`, the elements of `c`\n    are the coefficients to be solved for, and the elements of `y` are the\n    observed values.  This equation is then solved using the singular value\n    decomposition of `V`.\n\n    If some of the singular values of `V` are so small that they are\n    neglected, then a `RankWarning` will be issued. This means that the\n    coefficient values may be poorly determined. Using a lower order fit\n    will usually get rid of the warning.  The `rcond` parameter can also be\n    set to a value smaller than its default, but the resulting fit may be\n    spurious and have large contributions from roundoff error.\n\n    Fits using HermiteE series are probably most useful when the data can\n    be approximated by ``sqrt(w(x)) * p(x)``, where `w(x)` is the HermiteE\n    weight. In that case the weight ``sqrt(w(x[i])`` should be used\n    together with data values ``y[i]/sqrt(w(x[i])``. The weight function is\n    available as `hermeweight`.\n\n    References\n    ----------\n    .. [1] Wikipedia, "Curve fitting",\n           http://en.wikipedia.org/wiki/Curve_fitting\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermefik, hermeval\n    >>> x = np.linspace(-10, 10)\n    >>> err = np.random.randn(len(x))/10\n    >>> y = hermeval(x, [1, 2, 3]) + err\n    >>> hermefit(x, y, 2)\n    array([ 1.01690445,  1.99951418,  2.99948696])\n\n    ')
    
    # Assigning a BinOp to a Name (line 1489):
    
    # Assigning a BinOp to a Name (line 1489):
    
    # Call to asarray(...): (line 1489)
    # Processing the call arguments (line 1489)
    # Getting the type of 'x' (line 1489)
    x_169904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 19), 'x', False)
    # Processing the call keyword arguments (line 1489)
    kwargs_169905 = {}
    # Getting the type of 'np' (line 1489)
    np_169902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1489, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1489)
    asarray_169903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1489, 8), np_169902, 'asarray')
    # Calling asarray(args, kwargs) (line 1489)
    asarray_call_result_169906 = invoke(stypy.reporting.localization.Localization(__file__, 1489, 8), asarray_169903, *[x_169904], **kwargs_169905)
    
    float_169907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, 24), 'float')
    # Applying the binary operator '+' (line 1489)
    result_add_169908 = python_operator(stypy.reporting.localization.Localization(__file__, 1489, 8), '+', asarray_call_result_169906, float_169907)
    
    # Assigning a type to the variable 'x' (line 1489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1489, 4), 'x', result_add_169908)
    
    # Assigning a BinOp to a Name (line 1490):
    
    # Assigning a BinOp to a Name (line 1490):
    
    # Call to asarray(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'y' (line 1490)
    y_169911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 19), 'y', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_169912 = {}
    # Getting the type of 'np' (line 1490)
    np_169909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1490)
    asarray_169910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 8), np_169909, 'asarray')
    # Calling asarray(args, kwargs) (line 1490)
    asarray_call_result_169913 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 8), asarray_169910, *[y_169911], **kwargs_169912)
    
    float_169914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 24), 'float')
    # Applying the binary operator '+' (line 1490)
    result_add_169915 = python_operator(stypy.reporting.localization.Localization(__file__, 1490, 8), '+', asarray_call_result_169913, float_169914)
    
    # Assigning a type to the variable 'y' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'y', result_add_169915)
    
    # Assigning a Call to a Name (line 1491):
    
    # Assigning a Call to a Name (line 1491):
    
    # Call to asarray(...): (line 1491)
    # Processing the call arguments (line 1491)
    # Getting the type of 'deg' (line 1491)
    deg_169918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 21), 'deg', False)
    # Processing the call keyword arguments (line 1491)
    kwargs_169919 = {}
    # Getting the type of 'np' (line 1491)
    np_169916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1491)
    asarray_169917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1491, 10), np_169916, 'asarray')
    # Calling asarray(args, kwargs) (line 1491)
    asarray_call_result_169920 = invoke(stypy.reporting.localization.Localization(__file__, 1491, 10), asarray_169917, *[deg_169918], **kwargs_169919)
    
    # Assigning a type to the variable 'deg' (line 1491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1491, 4), 'deg', asarray_call_result_169920)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'deg' (line 1494)
    deg_169921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1494)
    ndim_169922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 7), deg_169921, 'ndim')
    int_169923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 18), 'int')
    # Applying the binary operator '>' (line 1494)
    result_gt_169924 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 7), '>', ndim_169922, int_169923)
    
    
    # Getting the type of 'deg' (line 1494)
    deg_169925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 23), 'deg')
    # Obtaining the member 'dtype' of a type (line 1494)
    dtype_169926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 23), deg_169925, 'dtype')
    # Obtaining the member 'kind' of a type (line 1494)
    kind_169927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 23), dtype_169926, 'kind')
    str_169928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 45), 'str', 'iu')
    # Applying the binary operator 'notin' (line 1494)
    result_contains_169929 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 23), 'notin', kind_169927, str_169928)
    
    # Applying the binary operator 'or' (line 1494)
    result_or_keyword_169930 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 7), 'or', result_gt_169924, result_contains_169929)
    
    # Getting the type of 'deg' (line 1494)
    deg_169931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 53), 'deg')
    # Obtaining the member 'size' of a type (line 1494)
    size_169932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 53), deg_169931, 'size')
    int_169933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 65), 'int')
    # Applying the binary operator '==' (line 1494)
    result_eq_169934 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 53), '==', size_169932, int_169933)
    
    # Applying the binary operator 'or' (line 1494)
    result_or_keyword_169935 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 7), 'or', result_or_keyword_169930, result_eq_169934)
    
    # Testing the type of an if condition (line 1494)
    if_condition_169936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1494, 4), result_or_keyword_169935)
    # Assigning a type to the variable 'if_condition_169936' (line 1494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1494, 4), 'if_condition_169936', if_condition_169936)
    # SSA begins for if statement (line 1494)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1495)
    # Processing the call arguments (line 1495)
    str_169938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1495, 24), 'str', 'deg must be an int or non-empty 1-D array of int')
    # Processing the call keyword arguments (line 1495)
    kwargs_169939 = {}
    # Getting the type of 'TypeError' (line 1495)
    TypeError_169937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1495)
    TypeError_call_result_169940 = invoke(stypy.reporting.localization.Localization(__file__, 1495, 14), TypeError_169937, *[str_169938], **kwargs_169939)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1495, 8), TypeError_call_result_169940, 'raise parameter', BaseException)
    # SSA join for if statement (line 1494)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to min(...): (line 1496)
    # Processing the call keyword arguments (line 1496)
    kwargs_169943 = {}
    # Getting the type of 'deg' (line 1496)
    deg_169941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 7), 'deg', False)
    # Obtaining the member 'min' of a type (line 1496)
    min_169942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1496, 7), deg_169941, 'min')
    # Calling min(args, kwargs) (line 1496)
    min_call_result_169944 = invoke(stypy.reporting.localization.Localization(__file__, 1496, 7), min_169942, *[], **kwargs_169943)
    
    int_169945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 19), 'int')
    # Applying the binary operator '<' (line 1496)
    result_lt_169946 = python_operator(stypy.reporting.localization.Localization(__file__, 1496, 7), '<', min_call_result_169944, int_169945)
    
    # Testing the type of an if condition (line 1496)
    if_condition_169947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1496, 4), result_lt_169946)
    # Assigning a type to the variable 'if_condition_169947' (line 1496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1496, 4), 'if_condition_169947', if_condition_169947)
    # SSA begins for if statement (line 1496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1497)
    # Processing the call arguments (line 1497)
    str_169949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1497, 25), 'str', 'expected deg >= 0')
    # Processing the call keyword arguments (line 1497)
    kwargs_169950 = {}
    # Getting the type of 'ValueError' (line 1497)
    ValueError_169948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1497, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1497)
    ValueError_call_result_169951 = invoke(stypy.reporting.localization.Localization(__file__, 1497, 14), ValueError_169948, *[str_169949], **kwargs_169950)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1497, 8), ValueError_call_result_169951, 'raise parameter', BaseException)
    # SSA join for if statement (line 1496)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1498)
    x_169952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1498, 7), 'x')
    # Obtaining the member 'ndim' of a type (line 1498)
    ndim_169953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1498, 7), x_169952, 'ndim')
    int_169954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1498, 17), 'int')
    # Applying the binary operator '!=' (line 1498)
    result_ne_169955 = python_operator(stypy.reporting.localization.Localization(__file__, 1498, 7), '!=', ndim_169953, int_169954)
    
    # Testing the type of an if condition (line 1498)
    if_condition_169956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1498, 4), result_ne_169955)
    # Assigning a type to the variable 'if_condition_169956' (line 1498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1498, 4), 'if_condition_169956', if_condition_169956)
    # SSA begins for if statement (line 1498)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1499)
    # Processing the call arguments (line 1499)
    str_169958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1499, 24), 'str', 'expected 1D vector for x')
    # Processing the call keyword arguments (line 1499)
    kwargs_169959 = {}
    # Getting the type of 'TypeError' (line 1499)
    TypeError_169957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1499, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1499)
    TypeError_call_result_169960 = invoke(stypy.reporting.localization.Localization(__file__, 1499, 14), TypeError_169957, *[str_169958], **kwargs_169959)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1499, 8), TypeError_call_result_169960, 'raise parameter', BaseException)
    # SSA join for if statement (line 1498)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'x' (line 1500)
    x_169961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1500, 7), 'x')
    # Obtaining the member 'size' of a type (line 1500)
    size_169962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1500, 7), x_169961, 'size')
    int_169963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1500, 17), 'int')
    # Applying the binary operator '==' (line 1500)
    result_eq_169964 = python_operator(stypy.reporting.localization.Localization(__file__, 1500, 7), '==', size_169962, int_169963)
    
    # Testing the type of an if condition (line 1500)
    if_condition_169965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1500, 4), result_eq_169964)
    # Assigning a type to the variable 'if_condition_169965' (line 1500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1500, 4), 'if_condition_169965', if_condition_169965)
    # SSA begins for if statement (line 1500)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1501)
    # Processing the call arguments (line 1501)
    str_169967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1501, 24), 'str', 'expected non-empty vector for x')
    # Processing the call keyword arguments (line 1501)
    kwargs_169968 = {}
    # Getting the type of 'TypeError' (line 1501)
    TypeError_169966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1501, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1501)
    TypeError_call_result_169969 = invoke(stypy.reporting.localization.Localization(__file__, 1501, 14), TypeError_169966, *[str_169967], **kwargs_169968)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1501, 8), TypeError_call_result_169969, 'raise parameter', BaseException)
    # SSA join for if statement (line 1500)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'y' (line 1502)
    y_169970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 7), 'y')
    # Obtaining the member 'ndim' of a type (line 1502)
    ndim_169971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1502, 7), y_169970, 'ndim')
    int_169972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 16), 'int')
    # Applying the binary operator '<' (line 1502)
    result_lt_169973 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), '<', ndim_169971, int_169972)
    
    
    # Getting the type of 'y' (line 1502)
    y_169974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1502, 21), 'y')
    # Obtaining the member 'ndim' of a type (line 1502)
    ndim_169975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1502, 21), y_169974, 'ndim')
    int_169976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1502, 30), 'int')
    # Applying the binary operator '>' (line 1502)
    result_gt_169977 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 21), '>', ndim_169975, int_169976)
    
    # Applying the binary operator 'or' (line 1502)
    result_or_keyword_169978 = python_operator(stypy.reporting.localization.Localization(__file__, 1502, 7), 'or', result_lt_169973, result_gt_169977)
    
    # Testing the type of an if condition (line 1502)
    if_condition_169979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1502, 4), result_or_keyword_169978)
    # Assigning a type to the variable 'if_condition_169979' (line 1502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1502, 4), 'if_condition_169979', if_condition_169979)
    # SSA begins for if statement (line 1502)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1503)
    # Processing the call arguments (line 1503)
    str_169981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1503, 24), 'str', 'expected 1D or 2D array for y')
    # Processing the call keyword arguments (line 1503)
    kwargs_169982 = {}
    # Getting the type of 'TypeError' (line 1503)
    TypeError_169980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1503)
    TypeError_call_result_169983 = invoke(stypy.reporting.localization.Localization(__file__, 1503, 14), TypeError_169980, *[str_169981], **kwargs_169982)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1503, 8), TypeError_call_result_169983, 'raise parameter', BaseException)
    # SSA join for if statement (line 1502)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1504)
    # Processing the call arguments (line 1504)
    # Getting the type of 'x' (line 1504)
    x_169985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 11), 'x', False)
    # Processing the call keyword arguments (line 1504)
    kwargs_169986 = {}
    # Getting the type of 'len' (line 1504)
    len_169984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 7), 'len', False)
    # Calling len(args, kwargs) (line 1504)
    len_call_result_169987 = invoke(stypy.reporting.localization.Localization(__file__, 1504, 7), len_169984, *[x_169985], **kwargs_169986)
    
    
    # Call to len(...): (line 1504)
    # Processing the call arguments (line 1504)
    # Getting the type of 'y' (line 1504)
    y_169989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 21), 'y', False)
    # Processing the call keyword arguments (line 1504)
    kwargs_169990 = {}
    # Getting the type of 'len' (line 1504)
    len_169988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1504, 17), 'len', False)
    # Calling len(args, kwargs) (line 1504)
    len_call_result_169991 = invoke(stypy.reporting.localization.Localization(__file__, 1504, 17), len_169988, *[y_169989], **kwargs_169990)
    
    # Applying the binary operator '!=' (line 1504)
    result_ne_169992 = python_operator(stypy.reporting.localization.Localization(__file__, 1504, 7), '!=', len_call_result_169987, len_call_result_169991)
    
    # Testing the type of an if condition (line 1504)
    if_condition_169993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1504, 4), result_ne_169992)
    # Assigning a type to the variable 'if_condition_169993' (line 1504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1504, 4), 'if_condition_169993', if_condition_169993)
    # SSA begins for if statement (line 1504)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1505)
    # Processing the call arguments (line 1505)
    str_169995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1505, 24), 'str', 'expected x and y to have same length')
    # Processing the call keyword arguments (line 1505)
    kwargs_169996 = {}
    # Getting the type of 'TypeError' (line 1505)
    TypeError_169994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1505, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1505)
    TypeError_call_result_169997 = invoke(stypy.reporting.localization.Localization(__file__, 1505, 14), TypeError_169994, *[str_169995], **kwargs_169996)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1505, 8), TypeError_call_result_169997, 'raise parameter', BaseException)
    # SSA join for if statement (line 1504)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'deg' (line 1507)
    deg_169998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1507, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1507)
    ndim_169999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1507, 7), deg_169998, 'ndim')
    int_170000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1507, 19), 'int')
    # Applying the binary operator '==' (line 1507)
    result_eq_170001 = python_operator(stypy.reporting.localization.Localization(__file__, 1507, 7), '==', ndim_169999, int_170000)
    
    # Testing the type of an if condition (line 1507)
    if_condition_170002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1507, 4), result_eq_170001)
    # Assigning a type to the variable 'if_condition_170002' (line 1507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1507, 4), 'if_condition_170002', if_condition_170002)
    # SSA begins for if statement (line 1507)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 1508):
    
    # Assigning a Name to a Name (line 1508):
    # Getting the type of 'deg' (line 1508)
    deg_170003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1508, 15), 'deg')
    # Assigning a type to the variable 'lmax' (line 1508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1508, 8), 'lmax', deg_170003)
    
    # Assigning a BinOp to a Name (line 1509):
    
    # Assigning a BinOp to a Name (line 1509):
    # Getting the type of 'lmax' (line 1509)
    lmax_170004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1509, 16), 'lmax')
    int_170005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1509, 23), 'int')
    # Applying the binary operator '+' (line 1509)
    result_add_170006 = python_operator(stypy.reporting.localization.Localization(__file__, 1509, 16), '+', lmax_170004, int_170005)
    
    # Assigning a type to the variable 'order' (line 1509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1509, 8), 'order', result_add_170006)
    
    # Assigning a Call to a Name (line 1510):
    
    # Assigning a Call to a Name (line 1510):
    
    # Call to hermevander(...): (line 1510)
    # Processing the call arguments (line 1510)
    # Getting the type of 'x' (line 1510)
    x_170008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 26), 'x', False)
    # Getting the type of 'lmax' (line 1510)
    lmax_170009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 29), 'lmax', False)
    # Processing the call keyword arguments (line 1510)
    kwargs_170010 = {}
    # Getting the type of 'hermevander' (line 1510)
    hermevander_170007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1510, 14), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1510)
    hermevander_call_result_170011 = invoke(stypy.reporting.localization.Localization(__file__, 1510, 14), hermevander_170007, *[x_170008, lmax_170009], **kwargs_170010)
    
    # Assigning a type to the variable 'van' (line 1510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1510, 8), 'van', hermevander_call_result_170011)
    # SSA branch for the else part of an if statement (line 1507)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1512):
    
    # Assigning a Call to a Name (line 1512):
    
    # Call to sort(...): (line 1512)
    # Processing the call arguments (line 1512)
    # Getting the type of 'deg' (line 1512)
    deg_170014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 22), 'deg', False)
    # Processing the call keyword arguments (line 1512)
    kwargs_170015 = {}
    # Getting the type of 'np' (line 1512)
    np_170012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1512, 14), 'np', False)
    # Obtaining the member 'sort' of a type (line 1512)
    sort_170013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1512, 14), np_170012, 'sort')
    # Calling sort(args, kwargs) (line 1512)
    sort_call_result_170016 = invoke(stypy.reporting.localization.Localization(__file__, 1512, 14), sort_170013, *[deg_170014], **kwargs_170015)
    
    # Assigning a type to the variable 'deg' (line 1512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1512, 8), 'deg', sort_call_result_170016)
    
    # Assigning a Subscript to a Name (line 1513):
    
    # Assigning a Subscript to a Name (line 1513):
    
    # Obtaining the type of the subscript
    int_170017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1513, 19), 'int')
    # Getting the type of 'deg' (line 1513)
    deg_170018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1513, 15), 'deg')
    # Obtaining the member '__getitem__' of a type (line 1513)
    getitem___170019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1513, 15), deg_170018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1513)
    subscript_call_result_170020 = invoke(stypy.reporting.localization.Localization(__file__, 1513, 15), getitem___170019, int_170017)
    
    # Assigning a type to the variable 'lmax' (line 1513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1513, 8), 'lmax', subscript_call_result_170020)
    
    # Assigning a Call to a Name (line 1514):
    
    # Assigning a Call to a Name (line 1514):
    
    # Call to len(...): (line 1514)
    # Processing the call arguments (line 1514)
    # Getting the type of 'deg' (line 1514)
    deg_170022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 20), 'deg', False)
    # Processing the call keyword arguments (line 1514)
    kwargs_170023 = {}
    # Getting the type of 'len' (line 1514)
    len_170021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1514, 16), 'len', False)
    # Calling len(args, kwargs) (line 1514)
    len_call_result_170024 = invoke(stypy.reporting.localization.Localization(__file__, 1514, 16), len_170021, *[deg_170022], **kwargs_170023)
    
    # Assigning a type to the variable 'order' (line 1514)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1514, 8), 'order', len_call_result_170024)
    
    # Assigning a Subscript to a Name (line 1515):
    
    # Assigning a Subscript to a Name (line 1515):
    
    # Obtaining the type of the subscript
    slice_170025 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1515, 14), None, None, None)
    # Getting the type of 'deg' (line 1515)
    deg_170026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 38), 'deg')
    
    # Call to hermevander(...): (line 1515)
    # Processing the call arguments (line 1515)
    # Getting the type of 'x' (line 1515)
    x_170028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 26), 'x', False)
    # Getting the type of 'lmax' (line 1515)
    lmax_170029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 29), 'lmax', False)
    # Processing the call keyword arguments (line 1515)
    kwargs_170030 = {}
    # Getting the type of 'hermevander' (line 1515)
    hermevander_170027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1515, 14), 'hermevander', False)
    # Calling hermevander(args, kwargs) (line 1515)
    hermevander_call_result_170031 = invoke(stypy.reporting.localization.Localization(__file__, 1515, 14), hermevander_170027, *[x_170028, lmax_170029], **kwargs_170030)
    
    # Obtaining the member '__getitem__' of a type (line 1515)
    getitem___170032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1515, 14), hermevander_call_result_170031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1515)
    subscript_call_result_170033 = invoke(stypy.reporting.localization.Localization(__file__, 1515, 14), getitem___170032, (slice_170025, deg_170026))
    
    # Assigning a type to the variable 'van' (line 1515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1515, 8), 'van', subscript_call_result_170033)
    # SSA join for if statement (line 1507)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 1518):
    
    # Assigning a Attribute to a Name (line 1518):
    # Getting the type of 'van' (line 1518)
    van_170034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1518, 10), 'van')
    # Obtaining the member 'T' of a type (line 1518)
    T_170035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1518, 10), van_170034, 'T')
    # Assigning a type to the variable 'lhs' (line 1518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1518, 4), 'lhs', T_170035)
    
    # Assigning a Attribute to a Name (line 1519):
    
    # Assigning a Attribute to a Name (line 1519):
    # Getting the type of 'y' (line 1519)
    y_170036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1519, 10), 'y')
    # Obtaining the member 'T' of a type (line 1519)
    T_170037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1519, 10), y_170036, 'T')
    # Assigning a type to the variable 'rhs' (line 1519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1519, 4), 'rhs', T_170037)
    
    # Type idiom detected: calculating its left and rigth part (line 1520)
    # Getting the type of 'w' (line 1520)
    w_170038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 4), 'w')
    # Getting the type of 'None' (line 1520)
    None_170039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1520, 16), 'None')
    
    (may_be_170040, more_types_in_union_170041) = may_not_be_none(w_170038, None_170039)

    if may_be_170040:

        if more_types_in_union_170041:
            # Runtime conditional SSA (line 1520)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1521):
        
        # Assigning a BinOp to a Name (line 1521):
        
        # Call to asarray(...): (line 1521)
        # Processing the call arguments (line 1521)
        # Getting the type of 'w' (line 1521)
        w_170044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 23), 'w', False)
        # Processing the call keyword arguments (line 1521)
        kwargs_170045 = {}
        # Getting the type of 'np' (line 1521)
        np_170042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 1521)
        asarray_170043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1521, 12), np_170042, 'asarray')
        # Calling asarray(args, kwargs) (line 1521)
        asarray_call_result_170046 = invoke(stypy.reporting.localization.Localization(__file__, 1521, 12), asarray_170043, *[w_170044], **kwargs_170045)
        
        float_170047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1521, 28), 'float')
        # Applying the binary operator '+' (line 1521)
        result_add_170048 = python_operator(stypy.reporting.localization.Localization(__file__, 1521, 12), '+', asarray_call_result_170046, float_170047)
        
        # Assigning a type to the variable 'w' (line 1521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 8), 'w', result_add_170048)
        
        
        # Getting the type of 'w' (line 1522)
        w_170049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 11), 'w')
        # Obtaining the member 'ndim' of a type (line 1522)
        ndim_170050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 11), w_170049, 'ndim')
        int_170051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1522, 21), 'int')
        # Applying the binary operator '!=' (line 1522)
        result_ne_170052 = python_operator(stypy.reporting.localization.Localization(__file__, 1522, 11), '!=', ndim_170050, int_170051)
        
        # Testing the type of an if condition (line 1522)
        if_condition_170053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1522, 8), result_ne_170052)
        # Assigning a type to the variable 'if_condition_170053' (line 1522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 8), 'if_condition_170053', if_condition_170053)
        # SSA begins for if statement (line 1522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1523)
        # Processing the call arguments (line 1523)
        str_170055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1523, 28), 'str', 'expected 1D vector for w')
        # Processing the call keyword arguments (line 1523)
        kwargs_170056 = {}
        # Getting the type of 'TypeError' (line 1523)
        TypeError_170054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1523)
        TypeError_call_result_170057 = invoke(stypy.reporting.localization.Localization(__file__, 1523, 18), TypeError_170054, *[str_170055], **kwargs_170056)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1523, 12), TypeError_call_result_170057, 'raise parameter', BaseException)
        # SSA join for if statement (line 1522)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 1524)
        # Processing the call arguments (line 1524)
        # Getting the type of 'x' (line 1524)
        x_170059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 15), 'x', False)
        # Processing the call keyword arguments (line 1524)
        kwargs_170060 = {}
        # Getting the type of 'len' (line 1524)
        len_170058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 11), 'len', False)
        # Calling len(args, kwargs) (line 1524)
        len_call_result_170061 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 11), len_170058, *[x_170059], **kwargs_170060)
        
        
        # Call to len(...): (line 1524)
        # Processing the call arguments (line 1524)
        # Getting the type of 'w' (line 1524)
        w_170063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 25), 'w', False)
        # Processing the call keyword arguments (line 1524)
        kwargs_170064 = {}
        # Getting the type of 'len' (line 1524)
        len_170062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 21), 'len', False)
        # Calling len(args, kwargs) (line 1524)
        len_call_result_170065 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 21), len_170062, *[w_170063], **kwargs_170064)
        
        # Applying the binary operator '!=' (line 1524)
        result_ne_170066 = python_operator(stypy.reporting.localization.Localization(__file__, 1524, 11), '!=', len_call_result_170061, len_call_result_170065)
        
        # Testing the type of an if condition (line 1524)
        if_condition_170067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1524, 8), result_ne_170066)
        # Assigning a type to the variable 'if_condition_170067' (line 1524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 8), 'if_condition_170067', if_condition_170067)
        # SSA begins for if statement (line 1524)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 1525)
        # Processing the call arguments (line 1525)
        str_170069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1525, 28), 'str', 'expected x and w to have same length')
        # Processing the call keyword arguments (line 1525)
        kwargs_170070 = {}
        # Getting the type of 'TypeError' (line 1525)
        TypeError_170068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 1525)
        TypeError_call_result_170071 = invoke(stypy.reporting.localization.Localization(__file__, 1525, 18), TypeError_170068, *[str_170069], **kwargs_170070)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1525, 12), TypeError_call_result_170071, 'raise parameter', BaseException)
        # SSA join for if statement (line 1524)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 1528):
        
        # Assigning a BinOp to a Name (line 1528):
        # Getting the type of 'lhs' (line 1528)
        lhs_170072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 14), 'lhs')
        # Getting the type of 'w' (line 1528)
        w_170073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 20), 'w')
        # Applying the binary operator '*' (line 1528)
        result_mul_170074 = python_operator(stypy.reporting.localization.Localization(__file__, 1528, 14), '*', lhs_170072, w_170073)
        
        # Assigning a type to the variable 'lhs' (line 1528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1528, 8), 'lhs', result_mul_170074)
        
        # Assigning a BinOp to a Name (line 1529):
        
        # Assigning a BinOp to a Name (line 1529):
        # Getting the type of 'rhs' (line 1529)
        rhs_170075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 14), 'rhs')
        # Getting the type of 'w' (line 1529)
        w_170076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 20), 'w')
        # Applying the binary operator '*' (line 1529)
        result_mul_170077 = python_operator(stypy.reporting.localization.Localization(__file__, 1529, 14), '*', rhs_170075, w_170076)
        
        # Assigning a type to the variable 'rhs' (line 1529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1529, 8), 'rhs', result_mul_170077)

        if more_types_in_union_170041:
            # SSA join for if statement (line 1520)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 1532)
    # Getting the type of 'rcond' (line 1532)
    rcond_170078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 7), 'rcond')
    # Getting the type of 'None' (line 1532)
    None_170079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 16), 'None')
    
    (may_be_170080, more_types_in_union_170081) = may_be_none(rcond_170078, None_170079)

    if may_be_170080:

        if more_types_in_union_170081:
            # Runtime conditional SSA (line 1532)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 1533):
        
        # Assigning a BinOp to a Name (line 1533):
        
        # Call to len(...): (line 1533)
        # Processing the call arguments (line 1533)
        # Getting the type of 'x' (line 1533)
        x_170083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 20), 'x', False)
        # Processing the call keyword arguments (line 1533)
        kwargs_170084 = {}
        # Getting the type of 'len' (line 1533)
        len_170082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 16), 'len', False)
        # Calling len(args, kwargs) (line 1533)
        len_call_result_170085 = invoke(stypy.reporting.localization.Localization(__file__, 1533, 16), len_170082, *[x_170083], **kwargs_170084)
        
        
        # Call to finfo(...): (line 1533)
        # Processing the call arguments (line 1533)
        # Getting the type of 'x' (line 1533)
        x_170088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 32), 'x', False)
        # Obtaining the member 'dtype' of a type (line 1533)
        dtype_170089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1533, 32), x_170088, 'dtype')
        # Processing the call keyword arguments (line 1533)
        kwargs_170090 = {}
        # Getting the type of 'np' (line 1533)
        np_170086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1533, 23), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1533)
        finfo_170087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1533, 23), np_170086, 'finfo')
        # Calling finfo(args, kwargs) (line 1533)
        finfo_call_result_170091 = invoke(stypy.reporting.localization.Localization(__file__, 1533, 23), finfo_170087, *[dtype_170089], **kwargs_170090)
        
        # Obtaining the member 'eps' of a type (line 1533)
        eps_170092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1533, 23), finfo_call_result_170091, 'eps')
        # Applying the binary operator '*' (line 1533)
        result_mul_170093 = python_operator(stypy.reporting.localization.Localization(__file__, 1533, 16), '*', len_call_result_170085, eps_170092)
        
        # Assigning a type to the variable 'rcond' (line 1533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1533, 8), 'rcond', result_mul_170093)

        if more_types_in_union_170081:
            # SSA join for if statement (line 1532)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to issubclass(...): (line 1536)
    # Processing the call arguments (line 1536)
    # Getting the type of 'lhs' (line 1536)
    lhs_170095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 18), 'lhs', False)
    # Obtaining the member 'dtype' of a type (line 1536)
    dtype_170096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 18), lhs_170095, 'dtype')
    # Obtaining the member 'type' of a type (line 1536)
    type_170097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 18), dtype_170096, 'type')
    # Getting the type of 'np' (line 1536)
    np_170098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 34), 'np', False)
    # Obtaining the member 'complexfloating' of a type (line 1536)
    complexfloating_170099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1536, 34), np_170098, 'complexfloating')
    # Processing the call keyword arguments (line 1536)
    kwargs_170100 = {}
    # Getting the type of 'issubclass' (line 1536)
    issubclass_170094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1536, 7), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 1536)
    issubclass_call_result_170101 = invoke(stypy.reporting.localization.Localization(__file__, 1536, 7), issubclass_170094, *[type_170097, complexfloating_170099], **kwargs_170100)
    
    # Testing the type of an if condition (line 1536)
    if_condition_170102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1536, 4), issubclass_call_result_170101)
    # Assigning a type to the variable 'if_condition_170102' (line 1536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1536, 4), 'if_condition_170102', if_condition_170102)
    # SSA begins for if statement (line 1536)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1537):
    
    # Assigning a Call to a Name (line 1537):
    
    # Call to sqrt(...): (line 1537)
    # Processing the call arguments (line 1537)
    
    # Call to sum(...): (line 1537)
    # Processing the call arguments (line 1537)
    int_170119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1537, 70), 'int')
    # Processing the call keyword arguments (line 1537)
    kwargs_170120 = {}
    
    # Call to square(...): (line 1537)
    # Processing the call arguments (line 1537)
    # Getting the type of 'lhs' (line 1537)
    lhs_170107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 33), 'lhs', False)
    # Obtaining the member 'real' of a type (line 1537)
    real_170108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 33), lhs_170107, 'real')
    # Processing the call keyword arguments (line 1537)
    kwargs_170109 = {}
    # Getting the type of 'np' (line 1537)
    np_170105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 23), 'np', False)
    # Obtaining the member 'square' of a type (line 1537)
    square_170106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 23), np_170105, 'square')
    # Calling square(args, kwargs) (line 1537)
    square_call_result_170110 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 23), square_170106, *[real_170108], **kwargs_170109)
    
    
    # Call to square(...): (line 1537)
    # Processing the call arguments (line 1537)
    # Getting the type of 'lhs' (line 1537)
    lhs_170113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 55), 'lhs', False)
    # Obtaining the member 'imag' of a type (line 1537)
    imag_170114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 55), lhs_170113, 'imag')
    # Processing the call keyword arguments (line 1537)
    kwargs_170115 = {}
    # Getting the type of 'np' (line 1537)
    np_170111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 45), 'np', False)
    # Obtaining the member 'square' of a type (line 1537)
    square_170112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 45), np_170111, 'square')
    # Calling square(args, kwargs) (line 1537)
    square_call_result_170116 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 45), square_170112, *[imag_170114], **kwargs_170115)
    
    # Applying the binary operator '+' (line 1537)
    result_add_170117 = python_operator(stypy.reporting.localization.Localization(__file__, 1537, 23), '+', square_call_result_170110, square_call_result_170116)
    
    # Obtaining the member 'sum' of a type (line 1537)
    sum_170118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 23), result_add_170117, 'sum')
    # Calling sum(args, kwargs) (line 1537)
    sum_call_result_170121 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 23), sum_170118, *[int_170119], **kwargs_170120)
    
    # Processing the call keyword arguments (line 1537)
    kwargs_170122 = {}
    # Getting the type of 'np' (line 1537)
    np_170103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1537, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1537)
    sqrt_170104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1537, 14), np_170103, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1537)
    sqrt_call_result_170123 = invoke(stypy.reporting.localization.Localization(__file__, 1537, 14), sqrt_170104, *[sum_call_result_170121], **kwargs_170122)
    
    # Assigning a type to the variable 'scl' (line 1537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1537, 8), 'scl', sqrt_call_result_170123)
    # SSA branch for the else part of an if statement (line 1536)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1539):
    
    # Assigning a Call to a Name (line 1539):
    
    # Call to sqrt(...): (line 1539)
    # Processing the call arguments (line 1539)
    
    # Call to sum(...): (line 1539)
    # Processing the call arguments (line 1539)
    int_170132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1539, 41), 'int')
    # Processing the call keyword arguments (line 1539)
    kwargs_170133 = {}
    
    # Call to square(...): (line 1539)
    # Processing the call arguments (line 1539)
    # Getting the type of 'lhs' (line 1539)
    lhs_170128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 32), 'lhs', False)
    # Processing the call keyword arguments (line 1539)
    kwargs_170129 = {}
    # Getting the type of 'np' (line 1539)
    np_170126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 22), 'np', False)
    # Obtaining the member 'square' of a type (line 1539)
    square_170127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 22), np_170126, 'square')
    # Calling square(args, kwargs) (line 1539)
    square_call_result_170130 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 22), square_170127, *[lhs_170128], **kwargs_170129)
    
    # Obtaining the member 'sum' of a type (line 1539)
    sum_170131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 22), square_call_result_170130, 'sum')
    # Calling sum(args, kwargs) (line 1539)
    sum_call_result_170134 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 22), sum_170131, *[int_170132], **kwargs_170133)
    
    # Processing the call keyword arguments (line 1539)
    kwargs_170135 = {}
    # Getting the type of 'np' (line 1539)
    np_170124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1539, 14), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1539)
    sqrt_170125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1539, 14), np_170124, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1539)
    sqrt_call_result_170136 = invoke(stypy.reporting.localization.Localization(__file__, 1539, 14), sqrt_170125, *[sum_call_result_170134], **kwargs_170135)
    
    # Assigning a type to the variable 'scl' (line 1539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1539, 8), 'scl', sqrt_call_result_170136)
    # SSA join for if statement (line 1536)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 1540):
    
    # Assigning a Num to a Subscript (line 1540):
    int_170137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1540, 20), 'int')
    # Getting the type of 'scl' (line 1540)
    scl_170138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 4), 'scl')
    
    # Getting the type of 'scl' (line 1540)
    scl_170139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1540, 8), 'scl')
    int_170140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1540, 15), 'int')
    # Applying the binary operator '==' (line 1540)
    result_eq_170141 = python_operator(stypy.reporting.localization.Localization(__file__, 1540, 8), '==', scl_170139, int_170140)
    
    # Storing an element on a container (line 1540)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1540, 4), scl_170138, (result_eq_170141, int_170137))
    
    # Assigning a Call to a Tuple (line 1543):
    
    # Assigning a Call to a Name:
    
    # Call to lstsq(...): (line 1543)
    # Processing the call arguments (line 1543)
    # Getting the type of 'lhs' (line 1543)
    lhs_170144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 34), 'lhs', False)
    # Obtaining the member 'T' of a type (line 1543)
    T_170145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 34), lhs_170144, 'T')
    # Getting the type of 'scl' (line 1543)
    scl_170146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 40), 'scl', False)
    # Applying the binary operator 'div' (line 1543)
    result_div_170147 = python_operator(stypy.reporting.localization.Localization(__file__, 1543, 34), 'div', T_170145, scl_170146)
    
    # Getting the type of 'rhs' (line 1543)
    rhs_170148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 45), 'rhs', False)
    # Obtaining the member 'T' of a type (line 1543)
    T_170149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 45), rhs_170148, 'T')
    # Getting the type of 'rcond' (line 1543)
    rcond_170150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 52), 'rcond', False)
    # Processing the call keyword arguments (line 1543)
    kwargs_170151 = {}
    # Getting the type of 'la' (line 1543)
    la_170142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 25), 'la', False)
    # Obtaining the member 'lstsq' of a type (line 1543)
    lstsq_170143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 25), la_170142, 'lstsq')
    # Calling lstsq(args, kwargs) (line 1543)
    lstsq_call_result_170152 = invoke(stypy.reporting.localization.Localization(__file__, 1543, 25), lstsq_170143, *[result_div_170147, T_170149, rcond_170150], **kwargs_170151)
    
    # Assigning a type to the variable 'call_assignment_167958' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167958', lstsq_call_result_170152)
    
    # Assigning a Call to a Name (line 1543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170156 = {}
    # Getting the type of 'call_assignment_167958' (line 1543)
    call_assignment_167958_170153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167958', False)
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___170154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 4), call_assignment_167958_170153, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170157 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170154, *[int_170155], **kwargs_170156)
    
    # Assigning a type to the variable 'call_assignment_167959' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167959', getitem___call_result_170157)
    
    # Assigning a Name to a Name (line 1543):
    # Getting the type of 'call_assignment_167959' (line 1543)
    call_assignment_167959_170158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167959')
    # Assigning a type to the variable 'c' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'c', call_assignment_167959_170158)
    
    # Assigning a Call to a Name (line 1543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170162 = {}
    # Getting the type of 'call_assignment_167958' (line 1543)
    call_assignment_167958_170159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167958', False)
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___170160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 4), call_assignment_167958_170159, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170163 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170160, *[int_170161], **kwargs_170162)
    
    # Assigning a type to the variable 'call_assignment_167960' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167960', getitem___call_result_170163)
    
    # Assigning a Name to a Name (line 1543):
    # Getting the type of 'call_assignment_167960' (line 1543)
    call_assignment_167960_170164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167960')
    # Assigning a type to the variable 'resids' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 7), 'resids', call_assignment_167960_170164)
    
    # Assigning a Call to a Name (line 1543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170168 = {}
    # Getting the type of 'call_assignment_167958' (line 1543)
    call_assignment_167958_170165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167958', False)
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___170166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 4), call_assignment_167958_170165, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170169 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170166, *[int_170167], **kwargs_170168)
    
    # Assigning a type to the variable 'call_assignment_167961' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167961', getitem___call_result_170169)
    
    # Assigning a Name to a Name (line 1543):
    # Getting the type of 'call_assignment_167961' (line 1543)
    call_assignment_167961_170170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167961')
    # Assigning a type to the variable 'rank' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 15), 'rank', call_assignment_167961_170170)
    
    # Assigning a Call to a Name (line 1543):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1543, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170174 = {}
    # Getting the type of 'call_assignment_167958' (line 1543)
    call_assignment_167958_170171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167958', False)
    # Obtaining the member '__getitem__' of a type (line 1543)
    getitem___170172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1543, 4), call_assignment_167958_170171, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170175 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170172, *[int_170173], **kwargs_170174)
    
    # Assigning a type to the variable 'call_assignment_167962' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167962', getitem___call_result_170175)
    
    # Assigning a Name to a Name (line 1543):
    # Getting the type of 'call_assignment_167962' (line 1543)
    call_assignment_167962_170176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1543, 4), 'call_assignment_167962')
    # Assigning a type to the variable 's' (line 1543)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1543, 21), 's', call_assignment_167962_170176)
    
    # Assigning a Attribute to a Name (line 1544):
    
    # Assigning a Attribute to a Name (line 1544):
    # Getting the type of 'c' (line 1544)
    c_170177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 9), 'c')
    # Obtaining the member 'T' of a type (line 1544)
    T_170178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 9), c_170177, 'T')
    # Getting the type of 'scl' (line 1544)
    scl_170179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1544, 13), 'scl')
    # Applying the binary operator 'div' (line 1544)
    result_div_170180 = python_operator(stypy.reporting.localization.Localization(__file__, 1544, 9), 'div', T_170178, scl_170179)
    
    # Obtaining the member 'T' of a type (line 1544)
    T_170181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1544, 9), result_div_170180, 'T')
    # Assigning a type to the variable 'c' (line 1544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1544, 4), 'c', T_170181)
    
    
    # Getting the type of 'deg' (line 1547)
    deg_170182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1547, 7), 'deg')
    # Obtaining the member 'ndim' of a type (line 1547)
    ndim_170183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1547, 7), deg_170182, 'ndim')
    int_170184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1547, 18), 'int')
    # Applying the binary operator '>' (line 1547)
    result_gt_170185 = python_operator(stypy.reporting.localization.Localization(__file__, 1547, 7), '>', ndim_170183, int_170184)
    
    # Testing the type of an if condition (line 1547)
    if_condition_170186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1547, 4), result_gt_170185)
    # Assigning a type to the variable 'if_condition_170186' (line 1547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1547, 4), 'if_condition_170186', if_condition_170186)
    # SSA begins for if statement (line 1547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'c' (line 1548)
    c_170187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1548, 11), 'c')
    # Obtaining the member 'ndim' of a type (line 1548)
    ndim_170188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1548, 11), c_170187, 'ndim')
    int_170189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1548, 21), 'int')
    # Applying the binary operator '==' (line 1548)
    result_eq_170190 = python_operator(stypy.reporting.localization.Localization(__file__, 1548, 11), '==', ndim_170188, int_170189)
    
    # Testing the type of an if condition (line 1548)
    if_condition_170191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1548, 8), result_eq_170190)
    # Assigning a type to the variable 'if_condition_170191' (line 1548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1548, 8), 'if_condition_170191', if_condition_170191)
    # SSA begins for if statement (line 1548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1549):
    
    # Assigning a Call to a Name (line 1549):
    
    # Call to zeros(...): (line 1549)
    # Processing the call arguments (line 1549)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1549)
    tuple_170194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1549)
    # Adding element type (line 1549)
    # Getting the type of 'lmax' (line 1549)
    lmax_170195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 27), 'lmax', False)
    int_170196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 32), 'int')
    # Applying the binary operator '+' (line 1549)
    result_add_170197 = python_operator(stypy.reporting.localization.Localization(__file__, 1549, 27), '+', lmax_170195, int_170196)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1549, 27), tuple_170194, result_add_170197)
    # Adding element type (line 1549)
    
    # Obtaining the type of the subscript
    int_170198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, 43), 'int')
    # Getting the type of 'c' (line 1549)
    c_170199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 35), 'c', False)
    # Obtaining the member 'shape' of a type (line 1549)
    shape_170200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 35), c_170199, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1549)
    getitem___170201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 35), shape_170200, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1549)
    subscript_call_result_170202 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 35), getitem___170201, int_170198)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1549, 27), tuple_170194, subscript_call_result_170202)
    
    # Processing the call keyword arguments (line 1549)
    # Getting the type of 'c' (line 1549)
    c_170203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 54), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1549)
    dtype_170204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 54), c_170203, 'dtype')
    keyword_170205 = dtype_170204
    kwargs_170206 = {'dtype': keyword_170205}
    # Getting the type of 'np' (line 1549)
    np_170192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1549, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1549)
    zeros_170193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1549, 17), np_170192, 'zeros')
    # Calling zeros(args, kwargs) (line 1549)
    zeros_call_result_170207 = invoke(stypy.reporting.localization.Localization(__file__, 1549, 17), zeros_170193, *[tuple_170194], **kwargs_170206)
    
    # Assigning a type to the variable 'cc' (line 1549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1549, 12), 'cc', zeros_call_result_170207)
    # SSA branch for the else part of an if statement (line 1548)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1551):
    
    # Assigning a Call to a Name (line 1551):
    
    # Call to zeros(...): (line 1551)
    # Processing the call arguments (line 1551)
    # Getting the type of 'lmax' (line 1551)
    lmax_170210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 26), 'lmax', False)
    int_170211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1551, 31), 'int')
    # Applying the binary operator '+' (line 1551)
    result_add_170212 = python_operator(stypy.reporting.localization.Localization(__file__, 1551, 26), '+', lmax_170210, int_170211)
    
    # Processing the call keyword arguments (line 1551)
    # Getting the type of 'c' (line 1551)
    c_170213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 40), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1551)
    dtype_170214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 40), c_170213, 'dtype')
    keyword_170215 = dtype_170214
    kwargs_170216 = {'dtype': keyword_170215}
    # Getting the type of 'np' (line 1551)
    np_170208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1551, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1551)
    zeros_170209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1551, 17), np_170208, 'zeros')
    # Calling zeros(args, kwargs) (line 1551)
    zeros_call_result_170217 = invoke(stypy.reporting.localization.Localization(__file__, 1551, 17), zeros_170209, *[result_add_170212], **kwargs_170216)
    
    # Assigning a type to the variable 'cc' (line 1551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1551, 12), 'cc', zeros_call_result_170217)
    # SSA join for if statement (line 1548)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 1552):
    
    # Assigning a Name to a Subscript (line 1552):
    # Getting the type of 'c' (line 1552)
    c_170218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 18), 'c')
    # Getting the type of 'cc' (line 1552)
    cc_170219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 8), 'cc')
    # Getting the type of 'deg' (line 1552)
    deg_170220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1552, 11), 'deg')
    # Storing an element on a container (line 1552)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1552, 8), cc_170219, (deg_170220, c_170218))
    
    # Assigning a Name to a Name (line 1553):
    
    # Assigning a Name to a Name (line 1553):
    # Getting the type of 'cc' (line 1553)
    cc_170221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1553, 12), 'cc')
    # Assigning a type to the variable 'c' (line 1553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1553, 8), 'c', cc_170221)
    # SSA join for if statement (line 1547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'rank' (line 1556)
    rank_170222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 7), 'rank')
    # Getting the type of 'order' (line 1556)
    order_170223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 15), 'order')
    # Applying the binary operator '!=' (line 1556)
    result_ne_170224 = python_operator(stypy.reporting.localization.Localization(__file__, 1556, 7), '!=', rank_170222, order_170223)
    
    
    # Getting the type of 'full' (line 1556)
    full_170225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1556, 29), 'full')
    # Applying the 'not' unary operator (line 1556)
    result_not__170226 = python_operator(stypy.reporting.localization.Localization(__file__, 1556, 25), 'not', full_170225)
    
    # Applying the binary operator 'and' (line 1556)
    result_and_keyword_170227 = python_operator(stypy.reporting.localization.Localization(__file__, 1556, 7), 'and', result_ne_170224, result_not__170226)
    
    # Testing the type of an if condition (line 1556)
    if_condition_170228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1556, 4), result_and_keyword_170227)
    # Assigning a type to the variable 'if_condition_170228' (line 1556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1556, 4), 'if_condition_170228', if_condition_170228)
    # SSA begins for if statement (line 1556)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1557):
    
    # Assigning a Str to a Name (line 1557):
    str_170229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1557, 14), 'str', 'The fit may be poorly conditioned')
    # Assigning a type to the variable 'msg' (line 1557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1557, 8), 'msg', str_170229)
    
    # Call to warn(...): (line 1558)
    # Processing the call arguments (line 1558)
    # Getting the type of 'msg' (line 1558)
    msg_170232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 22), 'msg', False)
    # Getting the type of 'pu' (line 1558)
    pu_170233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 27), 'pu', False)
    # Obtaining the member 'RankWarning' of a type (line 1558)
    RankWarning_170234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1558, 27), pu_170233, 'RankWarning')
    # Processing the call keyword arguments (line 1558)
    kwargs_170235 = {}
    # Getting the type of 'warnings' (line 1558)
    warnings_170230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1558, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 1558)
    warn_170231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1558, 8), warnings_170230, 'warn')
    # Calling warn(args, kwargs) (line 1558)
    warn_call_result_170236 = invoke(stypy.reporting.localization.Localization(__file__, 1558, 8), warn_170231, *[msg_170232, RankWarning_170234], **kwargs_170235)
    
    # SSA join for if statement (line 1556)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full' (line 1560)
    full_170237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1560, 7), 'full')
    # Testing the type of an if condition (line 1560)
    if_condition_170238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1560, 4), full_170237)
    # Assigning a type to the variable 'if_condition_170238' (line 1560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1560, 4), 'if_condition_170238', if_condition_170238)
    # SSA begins for if statement (line 1560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1561)
    tuple_170239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1561, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1561)
    # Adding element type (line 1561)
    # Getting the type of 'c' (line 1561)
    c_170240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 15), 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 15), tuple_170239, c_170240)
    # Adding element type (line 1561)
    
    # Obtaining an instance of the builtin type 'list' (line 1561)
    list_170241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1561, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1561)
    # Adding element type (line 1561)
    # Getting the type of 'resids' (line 1561)
    resids_170242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 19), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 18), list_170241, resids_170242)
    # Adding element type (line 1561)
    # Getting the type of 'rank' (line 1561)
    rank_170243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 27), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 18), list_170241, rank_170243)
    # Adding element type (line 1561)
    # Getting the type of 's' (line 1561)
    s_170244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 33), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 18), list_170241, s_170244)
    # Adding element type (line 1561)
    # Getting the type of 'rcond' (line 1561)
    rcond_170245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1561, 36), 'rcond')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 18), list_170241, rcond_170245)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1561, 15), tuple_170239, list_170241)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1561, 8), 'stypy_return_type', tuple_170239)
    # SSA branch for the else part of an if statement (line 1560)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'c' (line 1563)
    c_170246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1563, 15), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 1563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1563, 8), 'stypy_return_type', c_170246)
    # SSA join for if statement (line 1560)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hermefit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermefit' in the type store
    # Getting the type of 'stypy_return_type' (line 1365)
    stypy_return_type_170247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170247)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermefit'
    return stypy_return_type_170247

# Assigning a type to the variable 'hermefit' (line 1365)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1365, 0), 'hermefit', hermefit)

@norecursion
def hermecompanion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermecompanion'
    module_type_store = module_type_store.open_function_context('hermecompanion', 1566, 0, False)
    
    # Passed parameters checking function
    hermecompanion.stypy_localization = localization
    hermecompanion.stypy_type_of_self = None
    hermecompanion.stypy_type_store = module_type_store
    hermecompanion.stypy_function_name = 'hermecompanion'
    hermecompanion.stypy_param_names_list = ['c']
    hermecompanion.stypy_varargs_param_name = None
    hermecompanion.stypy_kwargs_param_name = None
    hermecompanion.stypy_call_defaults = defaults
    hermecompanion.stypy_call_varargs = varargs
    hermecompanion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermecompanion', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermecompanion', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermecompanion(...)' code ##################

    str_170248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, (-1)), 'str', '\n    Return the scaled companion matrix of c.\n\n    The basis polynomials are scaled so that the companion matrix is\n    symmetric when `c` is an HermiteE basis polynomial. This provides\n    better eigenvalue estimates than the unscaled case and for basis\n    polynomials the eigenvalues are guaranteed to be real if\n    `numpy.linalg.eigvalsh` is used to obtain them.\n\n    Parameters\n    ----------\n    c : array_like\n        1-D array of HermiteE series coefficients ordered from low to high\n        degree.\n\n    Returns\n    -------\n    mat : ndarray\n        Scaled companion matrix of dimensions (deg, deg).\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a List (line 1594):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1594)
    # Processing the call arguments (line 1594)
    
    # Obtaining an instance of the builtin type 'list' (line 1594)
    list_170251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1594)
    # Adding element type (line 1594)
    # Getting the type of 'c' (line 1594)
    c_170252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1594, 23), list_170251, c_170252)
    
    # Processing the call keyword arguments (line 1594)
    kwargs_170253 = {}
    # Getting the type of 'pu' (line 1594)
    pu_170249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1594)
    as_series_170250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 10), pu_170249, 'as_series')
    # Calling as_series(args, kwargs) (line 1594)
    as_series_call_result_170254 = invoke(stypy.reporting.localization.Localization(__file__, 1594, 10), as_series_170250, *[list_170251], **kwargs_170253)
    
    # Assigning a type to the variable 'call_assignment_167963' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_167963', as_series_call_result_170254)
    
    # Assigning a Call to a Name (line 1594):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1594, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170258 = {}
    # Getting the type of 'call_assignment_167963' (line 1594)
    call_assignment_167963_170255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_167963', False)
    # Obtaining the member '__getitem__' of a type (line 1594)
    getitem___170256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1594, 4), call_assignment_167963_170255, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170259 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170256, *[int_170257], **kwargs_170258)
    
    # Assigning a type to the variable 'call_assignment_167964' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_167964', getitem___call_result_170259)
    
    # Assigning a Name to a Name (line 1594):
    # Getting the type of 'call_assignment_167964' (line 1594)
    call_assignment_167964_170260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1594, 4), 'call_assignment_167964')
    # Assigning a type to the variable 'c' (line 1594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1594, 5), 'c', call_assignment_167964_170260)
    
    
    
    # Call to len(...): (line 1595)
    # Processing the call arguments (line 1595)
    # Getting the type of 'c' (line 1595)
    c_170262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 11), 'c', False)
    # Processing the call keyword arguments (line 1595)
    kwargs_170263 = {}
    # Getting the type of 'len' (line 1595)
    len_170261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 7), 'len', False)
    # Calling len(args, kwargs) (line 1595)
    len_call_result_170264 = invoke(stypy.reporting.localization.Localization(__file__, 1595, 7), len_170261, *[c_170262], **kwargs_170263)
    
    int_170265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1595, 16), 'int')
    # Applying the binary operator '<' (line 1595)
    result_lt_170266 = python_operator(stypy.reporting.localization.Localization(__file__, 1595, 7), '<', len_call_result_170264, int_170265)
    
    # Testing the type of an if condition (line 1595)
    if_condition_170267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1595, 4), result_lt_170266)
    # Assigning a type to the variable 'if_condition_170267' (line 1595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1595, 4), 'if_condition_170267', if_condition_170267)
    # SSA begins for if statement (line 1595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1596)
    # Processing the call arguments (line 1596)
    str_170269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 25), 'str', 'Series must have maximum degree of at least 1.')
    # Processing the call keyword arguments (line 1596)
    kwargs_170270 = {}
    # Getting the type of 'ValueError' (line 1596)
    ValueError_170268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1596)
    ValueError_call_result_170271 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 14), ValueError_170268, *[str_170269], **kwargs_170270)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1596, 8), ValueError_call_result_170271, 'raise parameter', BaseException)
    # SSA join for if statement (line 1595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1597)
    # Processing the call arguments (line 1597)
    # Getting the type of 'c' (line 1597)
    c_170273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 11), 'c', False)
    # Processing the call keyword arguments (line 1597)
    kwargs_170274 = {}
    # Getting the type of 'len' (line 1597)
    len_170272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 7), 'len', False)
    # Calling len(args, kwargs) (line 1597)
    len_call_result_170275 = invoke(stypy.reporting.localization.Localization(__file__, 1597, 7), len_170272, *[c_170273], **kwargs_170274)
    
    int_170276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1597, 17), 'int')
    # Applying the binary operator '==' (line 1597)
    result_eq_170277 = python_operator(stypy.reporting.localization.Localization(__file__, 1597, 7), '==', len_call_result_170275, int_170276)
    
    # Testing the type of an if condition (line 1597)
    if_condition_170278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1597, 4), result_eq_170277)
    # Assigning a type to the variable 'if_condition_170278' (line 1597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1597, 4), 'if_condition_170278', if_condition_170278)
    # SSA begins for if statement (line 1597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1598)
    # Processing the call arguments (line 1598)
    
    # Obtaining an instance of the builtin type 'list' (line 1598)
    list_170281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1598)
    # Adding element type (line 1598)
    
    # Obtaining an instance of the builtin type 'list' (line 1598)
    list_170282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1598)
    # Adding element type (line 1598)
    
    
    # Obtaining the type of the subscript
    int_170283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 29), 'int')
    # Getting the type of 'c' (line 1598)
    c_170284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 27), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1598)
    getitem___170285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 27), c_170284, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1598)
    subscript_call_result_170286 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 27), getitem___170285, int_170283)
    
    # Applying the 'usub' unary operator (line 1598)
    result___neg___170287 = python_operator(stypy.reporting.localization.Localization(__file__, 1598, 26), 'usub', subscript_call_result_170286)
    
    
    # Obtaining the type of the subscript
    int_170288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1598, 34), 'int')
    # Getting the type of 'c' (line 1598)
    c_170289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 32), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1598)
    getitem___170290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 32), c_170289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1598)
    subscript_call_result_170291 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 32), getitem___170290, int_170288)
    
    # Applying the binary operator 'div' (line 1598)
    result_div_170292 = python_operator(stypy.reporting.localization.Localization(__file__, 1598, 26), 'div', result___neg___170287, subscript_call_result_170291)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1598, 25), list_170282, result_div_170292)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1598, 24), list_170281, list_170282)
    
    # Processing the call keyword arguments (line 1598)
    kwargs_170293 = {}
    # Getting the type of 'np' (line 1598)
    np_170279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1598, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1598)
    array_170280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1598, 15), np_170279, 'array')
    # Calling array(args, kwargs) (line 1598)
    array_call_result_170294 = invoke(stypy.reporting.localization.Localization(__file__, 1598, 15), array_170280, *[list_170281], **kwargs_170293)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1598, 8), 'stypy_return_type', array_call_result_170294)
    # SSA join for if statement (line 1597)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 1600):
    
    # Assigning a BinOp to a Name (line 1600):
    
    # Call to len(...): (line 1600)
    # Processing the call arguments (line 1600)
    # Getting the type of 'c' (line 1600)
    c_170296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 12), 'c', False)
    # Processing the call keyword arguments (line 1600)
    kwargs_170297 = {}
    # Getting the type of 'len' (line 1600)
    len_170295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 8), 'len', False)
    # Calling len(args, kwargs) (line 1600)
    len_call_result_170298 = invoke(stypy.reporting.localization.Localization(__file__, 1600, 8), len_170295, *[c_170296], **kwargs_170297)
    
    int_170299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1600, 17), 'int')
    # Applying the binary operator '-' (line 1600)
    result_sub_170300 = python_operator(stypy.reporting.localization.Localization(__file__, 1600, 8), '-', len_call_result_170298, int_170299)
    
    # Assigning a type to the variable 'n' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 4), 'n', result_sub_170300)
    
    # Assigning a Call to a Name (line 1601):
    
    # Assigning a Call to a Name (line 1601):
    
    # Call to zeros(...): (line 1601)
    # Processing the call arguments (line 1601)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1601)
    tuple_170303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1601, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1601)
    # Adding element type (line 1601)
    # Getting the type of 'n' (line 1601)
    n_170304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 20), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 20), tuple_170303, n_170304)
    # Adding element type (line 1601)
    # Getting the type of 'n' (line 1601)
    n_170305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 23), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 20), tuple_170303, n_170305)
    
    # Processing the call keyword arguments (line 1601)
    # Getting the type of 'c' (line 1601)
    c_170306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 33), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1601)
    dtype_170307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 33), c_170306, 'dtype')
    keyword_170308 = dtype_170307
    kwargs_170309 = {'dtype': keyword_170308}
    # Getting the type of 'np' (line 1601)
    np_170301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 10), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1601)
    zeros_170302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 10), np_170301, 'zeros')
    # Calling zeros(args, kwargs) (line 1601)
    zeros_call_result_170310 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 10), zeros_170302, *[tuple_170303], **kwargs_170309)
    
    # Assigning a type to the variable 'mat' (line 1601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 4), 'mat', zeros_call_result_170310)
    
    # Assigning a Call to a Name (line 1602):
    
    # Assigning a Call to a Name (line 1602):
    
    # Call to hstack(...): (line 1602)
    # Processing the call arguments (line 1602)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1602)
    tuple_170313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1602)
    # Adding element type (line 1602)
    float_170314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1602, 21), tuple_170313, float_170314)
    # Adding element type (line 1602)
    float_170315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 25), 'float')
    
    # Call to sqrt(...): (line 1602)
    # Processing the call arguments (line 1602)
    
    # Call to arange(...): (line 1602)
    # Processing the call arguments (line 1602)
    # Getting the type of 'n' (line 1602)
    n_170320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 46), 'n', False)
    int_170321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 50), 'int')
    # Applying the binary operator '-' (line 1602)
    result_sub_170322 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 46), '-', n_170320, int_170321)
    
    int_170323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 53), 'int')
    int_170324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1602, 56), 'int')
    # Processing the call keyword arguments (line 1602)
    kwargs_170325 = {}
    # Getting the type of 'np' (line 1602)
    np_170318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 36), 'np', False)
    # Obtaining the member 'arange' of a type (line 1602)
    arange_170319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 36), np_170318, 'arange')
    # Calling arange(args, kwargs) (line 1602)
    arange_call_result_170326 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 36), arange_170319, *[result_sub_170322, int_170323, int_170324], **kwargs_170325)
    
    # Processing the call keyword arguments (line 1602)
    kwargs_170327 = {}
    # Getting the type of 'np' (line 1602)
    np_170316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 28), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1602)
    sqrt_170317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 28), np_170316, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1602)
    sqrt_call_result_170328 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 28), sqrt_170317, *[arange_call_result_170326], **kwargs_170327)
    
    # Applying the binary operator 'div' (line 1602)
    result_div_170329 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 25), 'div', float_170315, sqrt_call_result_170328)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1602, 21), tuple_170313, result_div_170329)
    
    # Processing the call keyword arguments (line 1602)
    kwargs_170330 = {}
    # Getting the type of 'np' (line 1602)
    np_170311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 10), 'np', False)
    # Obtaining the member 'hstack' of a type (line 1602)
    hstack_170312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1602, 10), np_170311, 'hstack')
    # Calling hstack(args, kwargs) (line 1602)
    hstack_call_result_170331 = invoke(stypy.reporting.localization.Localization(__file__, 1602, 10), hstack_170312, *[tuple_170313], **kwargs_170330)
    
    # Assigning a type to the variable 'scl' (line 1602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 4), 'scl', hstack_call_result_170331)
    
    # Assigning a Subscript to a Name (line 1603):
    
    # Assigning a Subscript to a Name (line 1603):
    
    # Obtaining the type of the subscript
    int_170332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1603, 40), 'int')
    slice_170333 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1603, 10), None, None, int_170332)
    
    # Call to accumulate(...): (line 1603)
    # Processing the call arguments (line 1603)
    # Getting the type of 'scl' (line 1603)
    scl_170337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 33), 'scl', False)
    # Processing the call keyword arguments (line 1603)
    kwargs_170338 = {}
    # Getting the type of 'np' (line 1603)
    np_170334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1603, 10), 'np', False)
    # Obtaining the member 'multiply' of a type (line 1603)
    multiply_170335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), np_170334, 'multiply')
    # Obtaining the member 'accumulate' of a type (line 1603)
    accumulate_170336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), multiply_170335, 'accumulate')
    # Calling accumulate(args, kwargs) (line 1603)
    accumulate_call_result_170339 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 10), accumulate_170336, *[scl_170337], **kwargs_170338)
    
    # Obtaining the member '__getitem__' of a type (line 1603)
    getitem___170340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1603, 10), accumulate_call_result_170339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1603)
    subscript_call_result_170341 = invoke(stypy.reporting.localization.Localization(__file__, 1603, 10), getitem___170340, slice_170333)
    
    # Assigning a type to the variable 'scl' (line 1603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1603, 4), 'scl', subscript_call_result_170341)
    
    # Assigning a Subscript to a Name (line 1604):
    
    # Assigning a Subscript to a Name (line 1604):
    
    # Obtaining the type of the subscript
    int_170342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 26), 'int')
    # Getting the type of 'n' (line 1604)
    n_170343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 29), 'n')
    int_170344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 31), 'int')
    # Applying the binary operator '+' (line 1604)
    result_add_170345 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 29), '+', n_170343, int_170344)
    
    slice_170346 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1604, 10), int_170342, None, result_add_170345)
    
    # Call to reshape(...): (line 1604)
    # Processing the call arguments (line 1604)
    int_170349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 22), 'int')
    # Processing the call keyword arguments (line 1604)
    kwargs_170350 = {}
    # Getting the type of 'mat' (line 1604)
    mat_170347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1604)
    reshape_170348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 10), mat_170347, 'reshape')
    # Calling reshape(args, kwargs) (line 1604)
    reshape_call_result_170351 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 10), reshape_170348, *[int_170349], **kwargs_170350)
    
    # Obtaining the member '__getitem__' of a type (line 1604)
    getitem___170352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 10), reshape_call_result_170351, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1604)
    subscript_call_result_170353 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 10), getitem___170352, slice_170346)
    
    # Assigning a type to the variable 'top' (line 1604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1604, 4), 'top', subscript_call_result_170353)
    
    # Assigning a Subscript to a Name (line 1605):
    
    # Assigning a Subscript to a Name (line 1605):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1605)
    n_170354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 26), 'n')
    # Getting the type of 'n' (line 1605)
    n_170355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 29), 'n')
    int_170356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 31), 'int')
    # Applying the binary operator '+' (line 1605)
    result_add_170357 = python_operator(stypy.reporting.localization.Localization(__file__, 1605, 29), '+', n_170355, int_170356)
    
    slice_170358 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1605, 10), n_170354, None, result_add_170357)
    
    # Call to reshape(...): (line 1605)
    # Processing the call arguments (line 1605)
    int_170361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1605, 22), 'int')
    # Processing the call keyword arguments (line 1605)
    kwargs_170362 = {}
    # Getting the type of 'mat' (line 1605)
    mat_170359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1605, 10), 'mat', False)
    # Obtaining the member 'reshape' of a type (line 1605)
    reshape_170360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 10), mat_170359, 'reshape')
    # Calling reshape(args, kwargs) (line 1605)
    reshape_call_result_170363 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 10), reshape_170360, *[int_170361], **kwargs_170362)
    
    # Obtaining the member '__getitem__' of a type (line 1605)
    getitem___170364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1605, 10), reshape_call_result_170363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1605)
    subscript_call_result_170365 = invoke(stypy.reporting.localization.Localization(__file__, 1605, 10), getitem___170364, slice_170358)
    
    # Assigning a type to the variable 'bot' (line 1605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1605, 4), 'bot', subscript_call_result_170365)
    
    # Assigning a Call to a Subscript (line 1606):
    
    # Assigning a Call to a Subscript (line 1606):
    
    # Call to sqrt(...): (line 1606)
    # Processing the call arguments (line 1606)
    
    # Call to arange(...): (line 1606)
    # Processing the call arguments (line 1606)
    int_170370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 33), 'int')
    # Getting the type of 'n' (line 1606)
    n_170371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 36), 'n', False)
    # Processing the call keyword arguments (line 1606)
    kwargs_170372 = {}
    # Getting the type of 'np' (line 1606)
    np_170368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 23), 'np', False)
    # Obtaining the member 'arange' of a type (line 1606)
    arange_170369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1606, 23), np_170368, 'arange')
    # Calling arange(args, kwargs) (line 1606)
    arange_call_result_170373 = invoke(stypy.reporting.localization.Localization(__file__, 1606, 23), arange_170369, *[int_170370, n_170371], **kwargs_170372)
    
    # Processing the call keyword arguments (line 1606)
    kwargs_170374 = {}
    # Getting the type of 'np' (line 1606)
    np_170366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 15), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1606)
    sqrt_170367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1606, 15), np_170366, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1606)
    sqrt_call_result_170375 = invoke(stypy.reporting.localization.Localization(__file__, 1606, 15), sqrt_170367, *[arange_call_result_170373], **kwargs_170374)
    
    # Getting the type of 'top' (line 1606)
    top_170376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 4), 'top')
    Ellipsis_170377 = Ellipsis
    # Storing an element on a container (line 1606)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1606, 4), top_170376, (Ellipsis_170377, sqrt_call_result_170375))
    
    # Assigning a Name to a Subscript (line 1607):
    
    # Assigning a Name to a Subscript (line 1607):
    # Getting the type of 'top' (line 1607)
    top_170378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 15), 'top')
    # Getting the type of 'bot' (line 1607)
    bot_170379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 4), 'bot')
    Ellipsis_170380 = Ellipsis
    # Storing an element on a container (line 1607)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1607, 4), bot_170379, (Ellipsis_170380, top_170378))
    
    # Getting the type of 'mat' (line 1608)
    mat_170381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    
    # Obtaining the type of the subscript
    slice_170382 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 4), None, None, None)
    int_170383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 11), 'int')
    # Getting the type of 'mat' (line 1608)
    mat_170384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___170385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 4), mat_170384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_170386 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 4), getitem___170385, (slice_170382, int_170383))
    
    # Getting the type of 'scl' (line 1608)
    scl_170387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 18), 'scl')
    
    # Obtaining the type of the subscript
    int_170388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 25), 'int')
    slice_170389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 22), None, int_170388, None)
    # Getting the type of 'c' (line 1608)
    c_170390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 22), 'c')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___170391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 22), c_170390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_170392 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 22), getitem___170391, slice_170389)
    
    # Applying the binary operator '*' (line 1608)
    result_mul_170393 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 18), '*', scl_170387, subscript_call_result_170392)
    
    
    # Obtaining the type of the subscript
    int_170394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 31), 'int')
    # Getting the type of 'c' (line 1608)
    c_170395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 1608)
    getitem___170396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1608, 29), c_170395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1608)
    subscript_call_result_170397 = invoke(stypy.reporting.localization.Localization(__file__, 1608, 29), getitem___170396, int_170394)
    
    # Applying the binary operator 'div' (line 1608)
    result_div_170398 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 28), 'div', result_mul_170393, subscript_call_result_170397)
    
    # Applying the binary operator '-=' (line 1608)
    result_isub_170399 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 4), '-=', subscript_call_result_170386, result_div_170398)
    # Getting the type of 'mat' (line 1608)
    mat_170400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 4), 'mat')
    slice_170401 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1608, 4), None, None, None)
    int_170402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1608, 11), 'int')
    # Storing an element on a container (line 1608)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1608, 4), mat_170400, ((slice_170401, int_170402), result_isub_170399))
    
    # Getting the type of 'mat' (line 1609)
    mat_170403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1609, 11), 'mat')
    # Assigning a type to the variable 'stypy_return_type' (line 1609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1609, 4), 'stypy_return_type', mat_170403)
    
    # ################# End of 'hermecompanion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermecompanion' in the type store
    # Getting the type of 'stypy_return_type' (line 1566)
    stypy_return_type_170404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1566, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170404)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermecompanion'
    return stypy_return_type_170404

# Assigning a type to the variable 'hermecompanion' (line 1566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1566, 0), 'hermecompanion', hermecompanion)

@norecursion
def hermeroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeroots'
    module_type_store = module_type_store.open_function_context('hermeroots', 1612, 0, False)
    
    # Passed parameters checking function
    hermeroots.stypy_localization = localization
    hermeroots.stypy_type_of_self = None
    hermeroots.stypy_type_store = module_type_store
    hermeroots.stypy_function_name = 'hermeroots'
    hermeroots.stypy_param_names_list = ['c']
    hermeroots.stypy_varargs_param_name = None
    hermeroots.stypy_kwargs_param_name = None
    hermeroots.stypy_call_defaults = defaults
    hermeroots.stypy_call_varargs = varargs
    hermeroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeroots', ['c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeroots', localization, ['c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeroots(...)' code ##################

    str_170405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1657, (-1)), 'str', '\n    Compute the roots of a HermiteE series.\n\n    Return the roots (a.k.a. "zeros") of the polynomial\n\n    .. math:: p(x) = \\sum_i c[i] * He_i(x).\n\n    Parameters\n    ----------\n    c : 1-D array_like\n        1-D array of coefficients.\n\n    Returns\n    -------\n    out : ndarray\n        Array of the roots of the series. If all the roots are real,\n        then `out` is also real, otherwise it is complex.\n\n    See Also\n    --------\n    polyroots, legroots, lagroots, hermroots, chebroots\n\n    Notes\n    -----\n    The root estimates are obtained as the eigenvalues of the companion\n    matrix, Roots far from the origin of the complex plane may have large\n    errors due to the numerical instability of the series for such\n    values. Roots with multiplicity greater than 1 will also show larger\n    errors as the value of the series near such points is relatively\n    insensitive to errors in the roots. Isolated roots near the origin can\n    be improved by a few iterations of Newton\'s method.\n\n    The HermiteE series basis polynomials aren\'t powers of `x` so the\n    results of this function may seem unintuitive.\n\n    Examples\n    --------\n    >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots\n    >>> coef = hermefromroots([-1, 0, 1])\n    >>> coef\n    array([ 0.,  2.,  0.,  1.])\n    >>> hermeroots(coef)\n    array([-1.,  0.,  1.])\n\n    ')
    
    # Assigning a Call to a List (line 1659):
    
    # Assigning a Call to a Name:
    
    # Call to as_series(...): (line 1659)
    # Processing the call arguments (line 1659)
    
    # Obtaining an instance of the builtin type 'list' (line 1659)
    list_170408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1659)
    # Adding element type (line 1659)
    # Getting the type of 'c' (line 1659)
    c_170409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 24), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1659, 23), list_170408, c_170409)
    
    # Processing the call keyword arguments (line 1659)
    kwargs_170410 = {}
    # Getting the type of 'pu' (line 1659)
    pu_170406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 10), 'pu', False)
    # Obtaining the member 'as_series' of a type (line 1659)
    as_series_170407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1659, 10), pu_170406, 'as_series')
    # Calling as_series(args, kwargs) (line 1659)
    as_series_call_result_170411 = invoke(stypy.reporting.localization.Localization(__file__, 1659, 10), as_series_170407, *[list_170408], **kwargs_170410)
    
    # Assigning a type to the variable 'call_assignment_167965' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_167965', as_series_call_result_170411)
    
    # Assigning a Call to a Name (line 1659):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_170414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1659, 4), 'int')
    # Processing the call keyword arguments
    kwargs_170415 = {}
    # Getting the type of 'call_assignment_167965' (line 1659)
    call_assignment_167965_170412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_167965', False)
    # Obtaining the member '__getitem__' of a type (line 1659)
    getitem___170413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1659, 4), call_assignment_167965_170412, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_170416 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___170413, *[int_170414], **kwargs_170415)
    
    # Assigning a type to the variable 'call_assignment_167966' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_167966', getitem___call_result_170416)
    
    # Assigning a Name to a Name (line 1659):
    # Getting the type of 'call_assignment_167966' (line 1659)
    call_assignment_167966_170417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1659, 4), 'call_assignment_167966')
    # Assigning a type to the variable 'c' (line 1659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1659, 5), 'c', call_assignment_167966_170417)
    
    
    
    # Call to len(...): (line 1660)
    # Processing the call arguments (line 1660)
    # Getting the type of 'c' (line 1660)
    c_170419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 11), 'c', False)
    # Processing the call keyword arguments (line 1660)
    kwargs_170420 = {}
    # Getting the type of 'len' (line 1660)
    len_170418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1660, 7), 'len', False)
    # Calling len(args, kwargs) (line 1660)
    len_call_result_170421 = invoke(stypy.reporting.localization.Localization(__file__, 1660, 7), len_170418, *[c_170419], **kwargs_170420)
    
    int_170422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1660, 17), 'int')
    # Applying the binary operator '<=' (line 1660)
    result_le_170423 = python_operator(stypy.reporting.localization.Localization(__file__, 1660, 7), '<=', len_call_result_170421, int_170422)
    
    # Testing the type of an if condition (line 1660)
    if_condition_170424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1660, 4), result_le_170423)
    # Assigning a type to the variable 'if_condition_170424' (line 1660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1660, 4), 'if_condition_170424', if_condition_170424)
    # SSA begins for if statement (line 1660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1661)
    # Processing the call arguments (line 1661)
    
    # Obtaining an instance of the builtin type 'list' (line 1661)
    list_170427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1661, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1661)
    
    # Processing the call keyword arguments (line 1661)
    # Getting the type of 'c' (line 1661)
    c_170428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 34), 'c', False)
    # Obtaining the member 'dtype' of a type (line 1661)
    dtype_170429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 34), c_170428, 'dtype')
    keyword_170430 = dtype_170429
    kwargs_170431 = {'dtype': keyword_170430}
    # Getting the type of 'np' (line 1661)
    np_170425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1661, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1661)
    array_170426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1661, 15), np_170425, 'array')
    # Calling array(args, kwargs) (line 1661)
    array_call_result_170432 = invoke(stypy.reporting.localization.Localization(__file__, 1661, 15), array_170426, *[list_170427], **kwargs_170431)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1661, 8), 'stypy_return_type', array_call_result_170432)
    # SSA join for if statement (line 1660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1662)
    # Processing the call arguments (line 1662)
    # Getting the type of 'c' (line 1662)
    c_170434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 11), 'c', False)
    # Processing the call keyword arguments (line 1662)
    kwargs_170435 = {}
    # Getting the type of 'len' (line 1662)
    len_170433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1662, 7), 'len', False)
    # Calling len(args, kwargs) (line 1662)
    len_call_result_170436 = invoke(stypy.reporting.localization.Localization(__file__, 1662, 7), len_170433, *[c_170434], **kwargs_170435)
    
    int_170437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1662, 17), 'int')
    # Applying the binary operator '==' (line 1662)
    result_eq_170438 = python_operator(stypy.reporting.localization.Localization(__file__, 1662, 7), '==', len_call_result_170436, int_170437)
    
    # Testing the type of an if condition (line 1662)
    if_condition_170439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1662, 4), result_eq_170438)
    # Assigning a type to the variable 'if_condition_170439' (line 1662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1662, 4), 'if_condition_170439', if_condition_170439)
    # SSA begins for if statement (line 1662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to array(...): (line 1663)
    # Processing the call arguments (line 1663)
    
    # Obtaining an instance of the builtin type 'list' (line 1663)
    list_170442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1663)
    # Adding element type (line 1663)
    
    
    # Obtaining the type of the subscript
    int_170443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 28), 'int')
    # Getting the type of 'c' (line 1663)
    c_170444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 26), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1663)
    getitem___170445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 26), c_170444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1663)
    subscript_call_result_170446 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 26), getitem___170445, int_170443)
    
    # Applying the 'usub' unary operator (line 1663)
    result___neg___170447 = python_operator(stypy.reporting.localization.Localization(__file__, 1663, 25), 'usub', subscript_call_result_170446)
    
    
    # Obtaining the type of the subscript
    int_170448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1663, 33), 'int')
    # Getting the type of 'c' (line 1663)
    c_170449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 31), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 1663)
    getitem___170450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 31), c_170449, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1663)
    subscript_call_result_170451 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 31), getitem___170450, int_170448)
    
    # Applying the binary operator 'div' (line 1663)
    result_div_170452 = python_operator(stypy.reporting.localization.Localization(__file__, 1663, 25), 'div', result___neg___170447, subscript_call_result_170451)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1663, 24), list_170442, result_div_170452)
    
    # Processing the call keyword arguments (line 1663)
    kwargs_170453 = {}
    # Getting the type of 'np' (line 1663)
    np_170440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1663, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1663)
    array_170441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1663, 15), np_170440, 'array')
    # Calling array(args, kwargs) (line 1663)
    array_call_result_170454 = invoke(stypy.reporting.localization.Localization(__file__, 1663, 15), array_170441, *[list_170442], **kwargs_170453)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1663)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1663, 8), 'stypy_return_type', array_call_result_170454)
    # SSA join for if statement (line 1662)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1665):
    
    # Assigning a Call to a Name (line 1665):
    
    # Call to hermecompanion(...): (line 1665)
    # Processing the call arguments (line 1665)
    # Getting the type of 'c' (line 1665)
    c_170456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 23), 'c', False)
    # Processing the call keyword arguments (line 1665)
    kwargs_170457 = {}
    # Getting the type of 'hermecompanion' (line 1665)
    hermecompanion_170455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1665, 8), 'hermecompanion', False)
    # Calling hermecompanion(args, kwargs) (line 1665)
    hermecompanion_call_result_170458 = invoke(stypy.reporting.localization.Localization(__file__, 1665, 8), hermecompanion_170455, *[c_170456], **kwargs_170457)
    
    # Assigning a type to the variable 'm' (line 1665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1665, 4), 'm', hermecompanion_call_result_170458)
    
    # Assigning a Call to a Name (line 1666):
    
    # Assigning a Call to a Name (line 1666):
    
    # Call to eigvals(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_170461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), 'm', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_170462 = {}
    # Getting the type of 'la' (line 1666)
    la_170459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 8), 'la', False)
    # Obtaining the member 'eigvals' of a type (line 1666)
    eigvals_170460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 8), la_170459, 'eigvals')
    # Calling eigvals(args, kwargs) (line 1666)
    eigvals_call_result_170463 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 8), eigvals_170460, *[m_170461], **kwargs_170462)
    
    # Assigning a type to the variable 'r' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'r', eigvals_call_result_170463)
    
    # Call to sort(...): (line 1667)
    # Processing the call keyword arguments (line 1667)
    kwargs_170466 = {}
    # Getting the type of 'r' (line 1667)
    r_170464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 4), 'r', False)
    # Obtaining the member 'sort' of a type (line 1667)
    sort_170465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1667, 4), r_170464, 'sort')
    # Calling sort(args, kwargs) (line 1667)
    sort_call_result_170467 = invoke(stypy.reporting.localization.Localization(__file__, 1667, 4), sort_170465, *[], **kwargs_170466)
    
    # Getting the type of 'r' (line 1668)
    r_170468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 1668)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1668, 4), 'stypy_return_type', r_170468)
    
    # ################# End of 'hermeroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeroots' in the type store
    # Getting the type of 'stypy_return_type' (line 1612)
    stypy_return_type_170469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1612, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170469)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeroots'
    return stypy_return_type_170469

# Assigning a type to the variable 'hermeroots' (line 1612)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1612, 0), 'hermeroots', hermeroots)

@norecursion
def _normed_hermite_e_n(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_normed_hermite_e_n'
    module_type_store = module_type_store.open_function_context('_normed_hermite_e_n', 1671, 0, False)
    
    # Passed parameters checking function
    _normed_hermite_e_n.stypy_localization = localization
    _normed_hermite_e_n.stypy_type_of_self = None
    _normed_hermite_e_n.stypy_type_store = module_type_store
    _normed_hermite_e_n.stypy_function_name = '_normed_hermite_e_n'
    _normed_hermite_e_n.stypy_param_names_list = ['x', 'n']
    _normed_hermite_e_n.stypy_varargs_param_name = None
    _normed_hermite_e_n.stypy_kwargs_param_name = None
    _normed_hermite_e_n.stypy_call_defaults = defaults
    _normed_hermite_e_n.stypy_call_varargs = varargs
    _normed_hermite_e_n.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_normed_hermite_e_n', ['x', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_normed_hermite_e_n', localization, ['x', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_normed_hermite_e_n(...)' code ##################

    str_170470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1699, (-1)), 'str', '\n    Evaluate a normalized HermiteE polynomial.\n\n    Compute the value of the normalized HermiteE polynomial of degree ``n``\n    at the points ``x``.\n\n\n    Parameters\n    ----------\n    x : ndarray of double.\n        Points at which to evaluate the function\n    n : int\n        Degree of the normalized HermiteE function to be evaluated.\n\n    Returns\n    -------\n    values : ndarray\n        The shape of the return value is described above.\n\n    Notes\n    -----\n    .. versionadded:: 1.10.0\n\n    This function is needed for finding the Gauss points and integration\n    weights for high degrees. The values of the standard HermiteE functions\n    overflow when n >= 207.\n\n    ')
    
    
    # Getting the type of 'n' (line 1700)
    n_170471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1700, 7), 'n')
    int_170472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1700, 12), 'int')
    # Applying the binary operator '==' (line 1700)
    result_eq_170473 = python_operator(stypy.reporting.localization.Localization(__file__, 1700, 7), '==', n_170471, int_170472)
    
    # Testing the type of an if condition (line 1700)
    if_condition_170474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1700, 4), result_eq_170473)
    # Assigning a type to the variable 'if_condition_170474' (line 1700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1700, 4), 'if_condition_170474', if_condition_170474)
    # SSA begins for if statement (line 1700)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ones(...): (line 1701)
    # Processing the call arguments (line 1701)
    # Getting the type of 'x' (line 1701)
    x_170477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1701, 23), 'x', False)
    # Obtaining the member 'shape' of a type (line 1701)
    shape_170478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1701, 23), x_170477, 'shape')
    # Processing the call keyword arguments (line 1701)
    kwargs_170479 = {}
    # Getting the type of 'np' (line 1701)
    np_170475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1701, 15), 'np', False)
    # Obtaining the member 'ones' of a type (line 1701)
    ones_170476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1701, 15), np_170475, 'ones')
    # Calling ones(args, kwargs) (line 1701)
    ones_call_result_170480 = invoke(stypy.reporting.localization.Localization(__file__, 1701, 15), ones_170476, *[shape_170478], **kwargs_170479)
    
    
    # Call to sqrt(...): (line 1701)
    # Processing the call arguments (line 1701)
    
    # Call to sqrt(...): (line 1701)
    # Processing the call arguments (line 1701)
    int_170485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1701, 48), 'int')
    # Getting the type of 'np' (line 1701)
    np_170486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1701, 50), 'np', False)
    # Obtaining the member 'pi' of a type (line 1701)
    pi_170487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1701, 50), np_170486, 'pi')
    # Applying the binary operator '*' (line 1701)
    result_mul_170488 = python_operator(stypy.reporting.localization.Localization(__file__, 1701, 48), '*', int_170485, pi_170487)
    
    # Processing the call keyword arguments (line 1701)
    kwargs_170489 = {}
    # Getting the type of 'np' (line 1701)
    np_170483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1701, 40), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1701)
    sqrt_170484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1701, 40), np_170483, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1701)
    sqrt_call_result_170490 = invoke(stypy.reporting.localization.Localization(__file__, 1701, 40), sqrt_170484, *[result_mul_170488], **kwargs_170489)
    
    # Processing the call keyword arguments (line 1701)
    kwargs_170491 = {}
    # Getting the type of 'np' (line 1701)
    np_170481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1701, 32), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1701)
    sqrt_170482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1701, 32), np_170481, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1701)
    sqrt_call_result_170492 = invoke(stypy.reporting.localization.Localization(__file__, 1701, 32), sqrt_170482, *[sqrt_call_result_170490], **kwargs_170491)
    
    # Applying the binary operator 'div' (line 1701)
    result_div_170493 = python_operator(stypy.reporting.localization.Localization(__file__, 1701, 15), 'div', ones_call_result_170480, sqrt_call_result_170492)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1701, 8), 'stypy_return_type', result_div_170493)
    # SSA join for if statement (line 1700)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 1703):
    
    # Assigning a Num to a Name (line 1703):
    float_170494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1703, 9), 'float')
    # Assigning a type to the variable 'c0' (line 1703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1703, 4), 'c0', float_170494)
    
    # Assigning a BinOp to a Name (line 1704):
    
    # Assigning a BinOp to a Name (line 1704):
    float_170495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1704, 9), 'float')
    
    # Call to sqrt(...): (line 1704)
    # Processing the call arguments (line 1704)
    
    # Call to sqrt(...): (line 1704)
    # Processing the call arguments (line 1704)
    int_170500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1704, 28), 'int')
    # Getting the type of 'np' (line 1704)
    np_170501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1704, 30), 'np', False)
    # Obtaining the member 'pi' of a type (line 1704)
    pi_170502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1704, 30), np_170501, 'pi')
    # Applying the binary operator '*' (line 1704)
    result_mul_170503 = python_operator(stypy.reporting.localization.Localization(__file__, 1704, 28), '*', int_170500, pi_170502)
    
    # Processing the call keyword arguments (line 1704)
    kwargs_170504 = {}
    # Getting the type of 'np' (line 1704)
    np_170498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1704, 20), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1704)
    sqrt_170499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1704, 20), np_170498, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1704)
    sqrt_call_result_170505 = invoke(stypy.reporting.localization.Localization(__file__, 1704, 20), sqrt_170499, *[result_mul_170503], **kwargs_170504)
    
    # Processing the call keyword arguments (line 1704)
    kwargs_170506 = {}
    # Getting the type of 'np' (line 1704)
    np_170496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1704, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1704)
    sqrt_170497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1704, 12), np_170496, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1704)
    sqrt_call_result_170507 = invoke(stypy.reporting.localization.Localization(__file__, 1704, 12), sqrt_170497, *[sqrt_call_result_170505], **kwargs_170506)
    
    # Applying the binary operator 'div' (line 1704)
    result_div_170508 = python_operator(stypy.reporting.localization.Localization(__file__, 1704, 9), 'div', float_170495, sqrt_call_result_170507)
    
    # Assigning a type to the variable 'c1' (line 1704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1704, 4), 'c1', result_div_170508)
    
    # Assigning a Call to a Name (line 1705):
    
    # Assigning a Call to a Name (line 1705):
    
    # Call to float(...): (line 1705)
    # Processing the call arguments (line 1705)
    # Getting the type of 'n' (line 1705)
    n_170510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1705, 15), 'n', False)
    # Processing the call keyword arguments (line 1705)
    kwargs_170511 = {}
    # Getting the type of 'float' (line 1705)
    float_170509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1705, 9), 'float', False)
    # Calling float(args, kwargs) (line 1705)
    float_call_result_170512 = invoke(stypy.reporting.localization.Localization(__file__, 1705, 9), float_170509, *[n_170510], **kwargs_170511)
    
    # Assigning a type to the variable 'nd' (line 1705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1705, 4), 'nd', float_call_result_170512)
    
    
    # Call to range(...): (line 1706)
    # Processing the call arguments (line 1706)
    # Getting the type of 'n' (line 1706)
    n_170514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1706, 19), 'n', False)
    int_170515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1706, 23), 'int')
    # Applying the binary operator '-' (line 1706)
    result_sub_170516 = python_operator(stypy.reporting.localization.Localization(__file__, 1706, 19), '-', n_170514, int_170515)
    
    # Processing the call keyword arguments (line 1706)
    kwargs_170517 = {}
    # Getting the type of 'range' (line 1706)
    range_170513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1706, 13), 'range', False)
    # Calling range(args, kwargs) (line 1706)
    range_call_result_170518 = invoke(stypy.reporting.localization.Localization(__file__, 1706, 13), range_170513, *[result_sub_170516], **kwargs_170517)
    
    # Testing the type of a for loop iterable (line 1706)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1706, 4), range_call_result_170518)
    # Getting the type of the for loop variable (line 1706)
    for_loop_var_170519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1706, 4), range_call_result_170518)
    # Assigning a type to the variable 'i' (line 1706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1706, 4), 'i', for_loop_var_170519)
    # SSA begins for a for statement (line 1706)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 1707):
    
    # Assigning a Name to a Name (line 1707):
    # Getting the type of 'c0' (line 1707)
    c0_170520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1707, 14), 'c0')
    # Assigning a type to the variable 'tmp' (line 1707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1707, 8), 'tmp', c0_170520)
    
    # Assigning a BinOp to a Name (line 1708):
    
    # Assigning a BinOp to a Name (line 1708):
    
    # Getting the type of 'c1' (line 1708)
    c1_170521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 14), 'c1')
    # Applying the 'usub' unary operator (line 1708)
    result___neg___170522 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 13), 'usub', c1_170521)
    
    
    # Call to sqrt(...): (line 1708)
    # Processing the call arguments (line 1708)
    # Getting the type of 'nd' (line 1708)
    nd_170525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 26), 'nd', False)
    float_170526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1708, 31), 'float')
    # Applying the binary operator '-' (line 1708)
    result_sub_170527 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 26), '-', nd_170525, float_170526)
    
    # Getting the type of 'nd' (line 1708)
    nd_170528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 35), 'nd', False)
    # Applying the binary operator 'div' (line 1708)
    result_div_170529 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 25), 'div', result_sub_170527, nd_170528)
    
    # Processing the call keyword arguments (line 1708)
    kwargs_170530 = {}
    # Getting the type of 'np' (line 1708)
    np_170523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1708, 17), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1708)
    sqrt_170524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1708, 17), np_170523, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1708)
    sqrt_call_result_170531 = invoke(stypy.reporting.localization.Localization(__file__, 1708, 17), sqrt_170524, *[result_div_170529], **kwargs_170530)
    
    # Applying the binary operator '*' (line 1708)
    result_mul_170532 = python_operator(stypy.reporting.localization.Localization(__file__, 1708, 13), '*', result___neg___170522, sqrt_call_result_170531)
    
    # Assigning a type to the variable 'c0' (line 1708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1708, 8), 'c0', result_mul_170532)
    
    # Assigning a BinOp to a Name (line 1709):
    
    # Assigning a BinOp to a Name (line 1709):
    # Getting the type of 'tmp' (line 1709)
    tmp_170533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 13), 'tmp')
    # Getting the type of 'c1' (line 1709)
    c1_170534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 19), 'c1')
    # Getting the type of 'x' (line 1709)
    x_170535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 22), 'x')
    # Applying the binary operator '*' (line 1709)
    result_mul_170536 = python_operator(stypy.reporting.localization.Localization(__file__, 1709, 19), '*', c1_170534, x_170535)
    
    
    # Call to sqrt(...): (line 1709)
    # Processing the call arguments (line 1709)
    float_170539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1709, 32), 'float')
    # Getting the type of 'nd' (line 1709)
    nd_170540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 35), 'nd', False)
    # Applying the binary operator 'div' (line 1709)
    result_div_170541 = python_operator(stypy.reporting.localization.Localization(__file__, 1709, 32), 'div', float_170539, nd_170540)
    
    # Processing the call keyword arguments (line 1709)
    kwargs_170542 = {}
    # Getting the type of 'np' (line 1709)
    np_170537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1709, 24), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1709)
    sqrt_170538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1709, 24), np_170537, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1709)
    sqrt_call_result_170543 = invoke(stypy.reporting.localization.Localization(__file__, 1709, 24), sqrt_170538, *[result_div_170541], **kwargs_170542)
    
    # Applying the binary operator '*' (line 1709)
    result_mul_170544 = python_operator(stypy.reporting.localization.Localization(__file__, 1709, 23), '*', result_mul_170536, sqrt_call_result_170543)
    
    # Applying the binary operator '+' (line 1709)
    result_add_170545 = python_operator(stypy.reporting.localization.Localization(__file__, 1709, 13), '+', tmp_170533, result_mul_170544)
    
    # Assigning a type to the variable 'c1' (line 1709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1709, 8), 'c1', result_add_170545)
    
    # Assigning a BinOp to a Name (line 1710):
    
    # Assigning a BinOp to a Name (line 1710):
    # Getting the type of 'nd' (line 1710)
    nd_170546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1710, 13), 'nd')
    float_170547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1710, 18), 'float')
    # Applying the binary operator '-' (line 1710)
    result_sub_170548 = python_operator(stypy.reporting.localization.Localization(__file__, 1710, 13), '-', nd_170546, float_170547)
    
    # Assigning a type to the variable 'nd' (line 1710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1710, 8), 'nd', result_sub_170548)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c0' (line 1711)
    c0_170549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 11), 'c0')
    # Getting the type of 'c1' (line 1711)
    c1_170550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 16), 'c1')
    # Getting the type of 'x' (line 1711)
    x_170551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1711, 19), 'x')
    # Applying the binary operator '*' (line 1711)
    result_mul_170552 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 16), '*', c1_170550, x_170551)
    
    # Applying the binary operator '+' (line 1711)
    result_add_170553 = python_operator(stypy.reporting.localization.Localization(__file__, 1711, 11), '+', c0_170549, result_mul_170552)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1711, 4), 'stypy_return_type', result_add_170553)
    
    # ################# End of '_normed_hermite_e_n(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_normed_hermite_e_n' in the type store
    # Getting the type of 'stypy_return_type' (line 1671)
    stypy_return_type_170554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1671, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170554)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_normed_hermite_e_n'
    return stypy_return_type_170554

# Assigning a type to the variable '_normed_hermite_e_n' (line 1671)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1671, 0), '_normed_hermite_e_n', _normed_hermite_e_n)

@norecursion
def hermegauss(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermegauss'
    module_type_store = module_type_store.open_function_context('hermegauss', 1714, 0, False)
    
    # Passed parameters checking function
    hermegauss.stypy_localization = localization
    hermegauss.stypy_type_of_self = None
    hermegauss.stypy_type_store = module_type_store
    hermegauss.stypy_function_name = 'hermegauss'
    hermegauss.stypy_param_names_list = ['deg']
    hermegauss.stypy_varargs_param_name = None
    hermegauss.stypy_kwargs_param_name = None
    hermegauss.stypy_call_defaults = defaults
    hermegauss.stypy_call_varargs = varargs
    hermegauss.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermegauss', ['deg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermegauss', localization, ['deg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermegauss(...)' code ##################

    str_170555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1749, (-1)), 'str', "\n    Gauss-HermiteE quadrature.\n\n    Computes the sample points and weights for Gauss-HermiteE quadrature.\n    These sample points and weights will correctly integrate polynomials of\n    degree :math:`2*deg - 1` or less over the interval :math:`[-\\inf, \\inf]`\n    with the weight function :math:`f(x) = \\exp(-x^2/2)`.\n\n    Parameters\n    ----------\n    deg : int\n        Number of sample points and weights. It must be >= 1.\n\n    Returns\n    -------\n    x : ndarray\n        1-D ndarray containing the sample points.\n    y : ndarray\n        1-D ndarray containing the weights.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    The results have only been tested up to degree 100, higher degrees may\n    be problematic. The weights are determined by using the fact that\n\n    .. math:: w_k = c / (He'_n(x_k) * He_{n-1}(x_k))\n\n    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`\n    is the k'th root of :math:`He_n`, and then scaling the results to get\n    the right value when integrating 1.\n\n    ")
    
    # Assigning a Call to a Name (line 1750):
    
    # Assigning a Call to a Name (line 1750):
    
    # Call to int(...): (line 1750)
    # Processing the call arguments (line 1750)
    # Getting the type of 'deg' (line 1750)
    deg_170557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 15), 'deg', False)
    # Processing the call keyword arguments (line 1750)
    kwargs_170558 = {}
    # Getting the type of 'int' (line 1750)
    int_170556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1750, 11), 'int', False)
    # Calling int(args, kwargs) (line 1750)
    int_call_result_170559 = invoke(stypy.reporting.localization.Localization(__file__, 1750, 11), int_170556, *[deg_170557], **kwargs_170558)
    
    # Assigning a type to the variable 'ideg' (line 1750)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1750, 4), 'ideg', int_call_result_170559)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ideg' (line 1751)
    ideg_170560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 7), 'ideg')
    # Getting the type of 'deg' (line 1751)
    deg_170561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 15), 'deg')
    # Applying the binary operator '!=' (line 1751)
    result_ne_170562 = python_operator(stypy.reporting.localization.Localization(__file__, 1751, 7), '!=', ideg_170560, deg_170561)
    
    
    # Getting the type of 'ideg' (line 1751)
    ideg_170563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1751, 22), 'ideg')
    int_170564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1751, 29), 'int')
    # Applying the binary operator '<' (line 1751)
    result_lt_170565 = python_operator(stypy.reporting.localization.Localization(__file__, 1751, 22), '<', ideg_170563, int_170564)
    
    # Applying the binary operator 'or' (line 1751)
    result_or_keyword_170566 = python_operator(stypy.reporting.localization.Localization(__file__, 1751, 7), 'or', result_ne_170562, result_lt_170565)
    
    # Testing the type of an if condition (line 1751)
    if_condition_170567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1751, 4), result_or_keyword_170566)
    # Assigning a type to the variable 'if_condition_170567' (line 1751)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1751, 4), 'if_condition_170567', if_condition_170567)
    # SSA begins for if statement (line 1751)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1752)
    # Processing the call arguments (line 1752)
    str_170569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1752, 25), 'str', 'deg must be a non-negative integer')
    # Processing the call keyword arguments (line 1752)
    kwargs_170570 = {}
    # Getting the type of 'ValueError' (line 1752)
    ValueError_170568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1752, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1752)
    ValueError_call_result_170571 = invoke(stypy.reporting.localization.Localization(__file__, 1752, 14), ValueError_170568, *[str_170569], **kwargs_170570)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1752, 8), ValueError_call_result_170571, 'raise parameter', BaseException)
    # SSA join for if statement (line 1751)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1756):
    
    # Assigning a Call to a Name (line 1756):
    
    # Call to array(...): (line 1756)
    # Processing the call arguments (line 1756)
    
    # Obtaining an instance of the builtin type 'list' (line 1756)
    list_170574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1756)
    # Adding element type (line 1756)
    int_170575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1756, 17), list_170574, int_170575)
    
    # Getting the type of 'deg' (line 1756)
    deg_170576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 21), 'deg', False)
    # Applying the binary operator '*' (line 1756)
    result_mul_170577 = python_operator(stypy.reporting.localization.Localization(__file__, 1756, 17), '*', list_170574, deg_170576)
    
    
    # Obtaining an instance of the builtin type 'list' (line 1756)
    list_170578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1756)
    # Adding element type (line 1756)
    int_170579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1756, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1756, 27), list_170578, int_170579)
    
    # Applying the binary operator '+' (line 1756)
    result_add_170580 = python_operator(stypy.reporting.localization.Localization(__file__, 1756, 17), '+', result_mul_170577, list_170578)
    
    # Processing the call keyword arguments (line 1756)
    kwargs_170581 = {}
    # Getting the type of 'np' (line 1756)
    np_170572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1756, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 1756)
    array_170573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1756, 8), np_170572, 'array')
    # Calling array(args, kwargs) (line 1756)
    array_call_result_170582 = invoke(stypy.reporting.localization.Localization(__file__, 1756, 8), array_170573, *[result_add_170580], **kwargs_170581)
    
    # Assigning a type to the variable 'c' (line 1756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1756, 4), 'c', array_call_result_170582)
    
    # Assigning a Call to a Name (line 1757):
    
    # Assigning a Call to a Name (line 1757):
    
    # Call to hermecompanion(...): (line 1757)
    # Processing the call arguments (line 1757)
    # Getting the type of 'c' (line 1757)
    c_170584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1757, 23), 'c', False)
    # Processing the call keyword arguments (line 1757)
    kwargs_170585 = {}
    # Getting the type of 'hermecompanion' (line 1757)
    hermecompanion_170583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1757, 8), 'hermecompanion', False)
    # Calling hermecompanion(args, kwargs) (line 1757)
    hermecompanion_call_result_170586 = invoke(stypy.reporting.localization.Localization(__file__, 1757, 8), hermecompanion_170583, *[c_170584], **kwargs_170585)
    
    # Assigning a type to the variable 'm' (line 1757)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1757, 4), 'm', hermecompanion_call_result_170586)
    
    # Assigning a Call to a Name (line 1758):
    
    # Assigning a Call to a Name (line 1758):
    
    # Call to eigvalsh(...): (line 1758)
    # Processing the call arguments (line 1758)
    # Getting the type of 'm' (line 1758)
    m_170589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 20), 'm', False)
    # Processing the call keyword arguments (line 1758)
    kwargs_170590 = {}
    # Getting the type of 'la' (line 1758)
    la_170587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1758, 8), 'la', False)
    # Obtaining the member 'eigvalsh' of a type (line 1758)
    eigvalsh_170588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1758, 8), la_170587, 'eigvalsh')
    # Calling eigvalsh(args, kwargs) (line 1758)
    eigvalsh_call_result_170591 = invoke(stypy.reporting.localization.Localization(__file__, 1758, 8), eigvalsh_170588, *[m_170589], **kwargs_170590)
    
    # Assigning a type to the variable 'x' (line 1758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1758, 4), 'x', eigvalsh_call_result_170591)
    
    # Assigning a Call to a Name (line 1761):
    
    # Assigning a Call to a Name (line 1761):
    
    # Call to _normed_hermite_e_n(...): (line 1761)
    # Processing the call arguments (line 1761)
    # Getting the type of 'x' (line 1761)
    x_170593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1761, 29), 'x', False)
    # Getting the type of 'ideg' (line 1761)
    ideg_170594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1761, 32), 'ideg', False)
    # Processing the call keyword arguments (line 1761)
    kwargs_170595 = {}
    # Getting the type of '_normed_hermite_e_n' (line 1761)
    _normed_hermite_e_n_170592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1761, 9), '_normed_hermite_e_n', False)
    # Calling _normed_hermite_e_n(args, kwargs) (line 1761)
    _normed_hermite_e_n_call_result_170596 = invoke(stypy.reporting.localization.Localization(__file__, 1761, 9), _normed_hermite_e_n_170592, *[x_170593, ideg_170594], **kwargs_170595)
    
    # Assigning a type to the variable 'dy' (line 1761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1761, 4), 'dy', _normed_hermite_e_n_call_result_170596)
    
    # Assigning a BinOp to a Name (line 1762):
    
    # Assigning a BinOp to a Name (line 1762):
    
    # Call to _normed_hermite_e_n(...): (line 1762)
    # Processing the call arguments (line 1762)
    # Getting the type of 'x' (line 1762)
    x_170598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 29), 'x', False)
    # Getting the type of 'ideg' (line 1762)
    ideg_170599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 32), 'ideg', False)
    int_170600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1762, 39), 'int')
    # Applying the binary operator '-' (line 1762)
    result_sub_170601 = python_operator(stypy.reporting.localization.Localization(__file__, 1762, 32), '-', ideg_170599, int_170600)
    
    # Processing the call keyword arguments (line 1762)
    kwargs_170602 = {}
    # Getting the type of '_normed_hermite_e_n' (line 1762)
    _normed_hermite_e_n_170597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 9), '_normed_hermite_e_n', False)
    # Calling _normed_hermite_e_n(args, kwargs) (line 1762)
    _normed_hermite_e_n_call_result_170603 = invoke(stypy.reporting.localization.Localization(__file__, 1762, 9), _normed_hermite_e_n_170597, *[x_170598, result_sub_170601], **kwargs_170602)
    
    
    # Call to sqrt(...): (line 1762)
    # Processing the call arguments (line 1762)
    # Getting the type of 'ideg' (line 1762)
    ideg_170606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 52), 'ideg', False)
    # Processing the call keyword arguments (line 1762)
    kwargs_170607 = {}
    # Getting the type of 'np' (line 1762)
    np_170604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1762, 44), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1762)
    sqrt_170605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1762, 44), np_170604, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1762)
    sqrt_call_result_170608 = invoke(stypy.reporting.localization.Localization(__file__, 1762, 44), sqrt_170605, *[ideg_170606], **kwargs_170607)
    
    # Applying the binary operator '*' (line 1762)
    result_mul_170609 = python_operator(stypy.reporting.localization.Localization(__file__, 1762, 9), '*', _normed_hermite_e_n_call_result_170603, sqrt_call_result_170608)
    
    # Assigning a type to the variable 'df' (line 1762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1762, 4), 'df', result_mul_170609)
    
    # Getting the type of 'x' (line 1763)
    x_170610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 4), 'x')
    # Getting the type of 'dy' (line 1763)
    dy_170611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 9), 'dy')
    # Getting the type of 'df' (line 1763)
    df_170612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1763, 12), 'df')
    # Applying the binary operator 'div' (line 1763)
    result_div_170613 = python_operator(stypy.reporting.localization.Localization(__file__, 1763, 9), 'div', dy_170611, df_170612)
    
    # Applying the binary operator '-=' (line 1763)
    result_isub_170614 = python_operator(stypy.reporting.localization.Localization(__file__, 1763, 4), '-=', x_170610, result_div_170613)
    # Assigning a type to the variable 'x' (line 1763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1763, 4), 'x', result_isub_170614)
    
    
    # Assigning a Call to a Name (line 1767):
    
    # Assigning a Call to a Name (line 1767):
    
    # Call to _normed_hermite_e_n(...): (line 1767)
    # Processing the call arguments (line 1767)
    # Getting the type of 'x' (line 1767)
    x_170616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 29), 'x', False)
    # Getting the type of 'ideg' (line 1767)
    ideg_170617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 32), 'ideg', False)
    int_170618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1767, 39), 'int')
    # Applying the binary operator '-' (line 1767)
    result_sub_170619 = python_operator(stypy.reporting.localization.Localization(__file__, 1767, 32), '-', ideg_170617, int_170618)
    
    # Processing the call keyword arguments (line 1767)
    kwargs_170620 = {}
    # Getting the type of '_normed_hermite_e_n' (line 1767)
    _normed_hermite_e_n_170615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1767, 9), '_normed_hermite_e_n', False)
    # Calling _normed_hermite_e_n(args, kwargs) (line 1767)
    _normed_hermite_e_n_call_result_170621 = invoke(stypy.reporting.localization.Localization(__file__, 1767, 9), _normed_hermite_e_n_170615, *[x_170616, result_sub_170619], **kwargs_170620)
    
    # Assigning a type to the variable 'fm' (line 1767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1767, 4), 'fm', _normed_hermite_e_n_call_result_170621)
    
    # Getting the type of 'fm' (line 1768)
    fm_170622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1768, 4), 'fm')
    
    # Call to max(...): (line 1768)
    # Processing the call keyword arguments (line 1768)
    kwargs_170629 = {}
    
    # Call to abs(...): (line 1768)
    # Processing the call arguments (line 1768)
    # Getting the type of 'fm' (line 1768)
    fm_170625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1768, 17), 'fm', False)
    # Processing the call keyword arguments (line 1768)
    kwargs_170626 = {}
    # Getting the type of 'np' (line 1768)
    np_170623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1768, 10), 'np', False)
    # Obtaining the member 'abs' of a type (line 1768)
    abs_170624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1768, 10), np_170623, 'abs')
    # Calling abs(args, kwargs) (line 1768)
    abs_call_result_170627 = invoke(stypy.reporting.localization.Localization(__file__, 1768, 10), abs_170624, *[fm_170625], **kwargs_170626)
    
    # Obtaining the member 'max' of a type (line 1768)
    max_170628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1768, 10), abs_call_result_170627, 'max')
    # Calling max(args, kwargs) (line 1768)
    max_call_result_170630 = invoke(stypy.reporting.localization.Localization(__file__, 1768, 10), max_170628, *[], **kwargs_170629)
    
    # Applying the binary operator 'div=' (line 1768)
    result_div_170631 = python_operator(stypy.reporting.localization.Localization(__file__, 1768, 4), 'div=', fm_170622, max_call_result_170630)
    # Assigning a type to the variable 'fm' (line 1768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1768, 4), 'fm', result_div_170631)
    
    
    # Assigning a BinOp to a Name (line 1769):
    
    # Assigning a BinOp to a Name (line 1769):
    int_170632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1769, 8), 'int')
    # Getting the type of 'fm' (line 1769)
    fm_170633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1769, 11), 'fm')
    # Getting the type of 'fm' (line 1769)
    fm_170634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1769, 16), 'fm')
    # Applying the binary operator '*' (line 1769)
    result_mul_170635 = python_operator(stypy.reporting.localization.Localization(__file__, 1769, 11), '*', fm_170633, fm_170634)
    
    # Applying the binary operator 'div' (line 1769)
    result_div_170636 = python_operator(stypy.reporting.localization.Localization(__file__, 1769, 8), 'div', int_170632, result_mul_170635)
    
    # Assigning a type to the variable 'w' (line 1769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1769, 4), 'w', result_div_170636)
    
    # Assigning a BinOp to a Name (line 1772):
    
    # Assigning a BinOp to a Name (line 1772):
    # Getting the type of 'w' (line 1772)
    w_170637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1772, 9), 'w')
    
    # Obtaining the type of the subscript
    int_170638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1772, 17), 'int')
    slice_170639 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1772, 13), None, None, int_170638)
    # Getting the type of 'w' (line 1772)
    w_170640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1772, 13), 'w')
    # Obtaining the member '__getitem__' of a type (line 1772)
    getitem___170641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1772, 13), w_170640, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1772)
    subscript_call_result_170642 = invoke(stypy.reporting.localization.Localization(__file__, 1772, 13), getitem___170641, slice_170639)
    
    # Applying the binary operator '+' (line 1772)
    result_add_170643 = python_operator(stypy.reporting.localization.Localization(__file__, 1772, 9), '+', w_170637, subscript_call_result_170642)
    
    int_170644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1772, 22), 'int')
    # Applying the binary operator 'div' (line 1772)
    result_div_170645 = python_operator(stypy.reporting.localization.Localization(__file__, 1772, 8), 'div', result_add_170643, int_170644)
    
    # Assigning a type to the variable 'w' (line 1772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1772, 4), 'w', result_div_170645)
    
    # Assigning a BinOp to a Name (line 1773):
    
    # Assigning a BinOp to a Name (line 1773):
    # Getting the type of 'x' (line 1773)
    x_170646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1773, 9), 'x')
    
    # Obtaining the type of the subscript
    int_170647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1773, 17), 'int')
    slice_170648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1773, 13), None, None, int_170647)
    # Getting the type of 'x' (line 1773)
    x_170649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1773, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 1773)
    getitem___170650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1773, 13), x_170649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1773)
    subscript_call_result_170651 = invoke(stypy.reporting.localization.Localization(__file__, 1773, 13), getitem___170650, slice_170648)
    
    # Applying the binary operator '-' (line 1773)
    result_sub_170652 = python_operator(stypy.reporting.localization.Localization(__file__, 1773, 9), '-', x_170646, subscript_call_result_170651)
    
    int_170653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1773, 22), 'int')
    # Applying the binary operator 'div' (line 1773)
    result_div_170654 = python_operator(stypy.reporting.localization.Localization(__file__, 1773, 8), 'div', result_sub_170652, int_170653)
    
    # Assigning a type to the variable 'x' (line 1773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1773, 4), 'x', result_div_170654)
    
    # Getting the type of 'w' (line 1776)
    w_170655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 4), 'w')
    
    # Call to sqrt(...): (line 1776)
    # Processing the call arguments (line 1776)
    int_170658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1776, 17), 'int')
    # Getting the type of 'np' (line 1776)
    np_170659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 19), 'np', False)
    # Obtaining the member 'pi' of a type (line 1776)
    pi_170660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 19), np_170659, 'pi')
    # Applying the binary operator '*' (line 1776)
    result_mul_170661 = python_operator(stypy.reporting.localization.Localization(__file__, 1776, 17), '*', int_170658, pi_170660)
    
    # Processing the call keyword arguments (line 1776)
    kwargs_170662 = {}
    # Getting the type of 'np' (line 1776)
    np_170656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 9), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 1776)
    sqrt_170657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 9), np_170656, 'sqrt')
    # Calling sqrt(args, kwargs) (line 1776)
    sqrt_call_result_170663 = invoke(stypy.reporting.localization.Localization(__file__, 1776, 9), sqrt_170657, *[result_mul_170661], **kwargs_170662)
    
    
    # Call to sum(...): (line 1776)
    # Processing the call keyword arguments (line 1776)
    kwargs_170666 = {}
    # Getting the type of 'w' (line 1776)
    w_170664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1776, 28), 'w', False)
    # Obtaining the member 'sum' of a type (line 1776)
    sum_170665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1776, 28), w_170664, 'sum')
    # Calling sum(args, kwargs) (line 1776)
    sum_call_result_170667 = invoke(stypy.reporting.localization.Localization(__file__, 1776, 28), sum_170665, *[], **kwargs_170666)
    
    # Applying the binary operator 'div' (line 1776)
    result_div_170668 = python_operator(stypy.reporting.localization.Localization(__file__, 1776, 9), 'div', sqrt_call_result_170663, sum_call_result_170667)
    
    # Applying the binary operator '*=' (line 1776)
    result_imul_170669 = python_operator(stypy.reporting.localization.Localization(__file__, 1776, 4), '*=', w_170655, result_div_170668)
    # Assigning a type to the variable 'w' (line 1776)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1776, 4), 'w', result_imul_170669)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1778)
    tuple_170670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1778, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1778)
    # Adding element type (line 1778)
    # Getting the type of 'x' (line 1778)
    x_170671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 11), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1778, 11), tuple_170670, x_170671)
    # Adding element type (line 1778)
    # Getting the type of 'w' (line 1778)
    w_170672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1778, 14), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1778, 11), tuple_170670, w_170672)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1778)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1778, 4), 'stypy_return_type', tuple_170670)
    
    # ################# End of 'hermegauss(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermegauss' in the type store
    # Getting the type of 'stypy_return_type' (line 1714)
    stypy_return_type_170673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1714, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170673)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermegauss'
    return stypy_return_type_170673

# Assigning a type to the variable 'hermegauss' (line 1714)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1714, 0), 'hermegauss', hermegauss)

@norecursion
def hermeweight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hermeweight'
    module_type_store = module_type_store.open_function_context('hermeweight', 1781, 0, False)
    
    # Passed parameters checking function
    hermeweight.stypy_localization = localization
    hermeweight.stypy_type_of_self = None
    hermeweight.stypy_type_store = module_type_store
    hermeweight.stypy_function_name = 'hermeweight'
    hermeweight.stypy_param_names_list = ['x']
    hermeweight.stypy_varargs_param_name = None
    hermeweight.stypy_kwargs_param_name = None
    hermeweight.stypy_call_defaults = defaults
    hermeweight.stypy_call_varargs = varargs
    hermeweight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hermeweight', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hermeweight', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hermeweight(...)' code ##################

    str_170674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1803, (-1)), 'str', 'Weight function of the Hermite_e polynomials.\n\n    The weight function is :math:`\\exp(-x^2/2)` and the interval of\n    integration is :math:`[-\\inf, \\inf]`. the HermiteE polynomials are\n    orthogonal, but not normalized, with respect to this weight function.\n\n    Parameters\n    ----------\n    x : array_like\n       Values at which the weight function will be computed.\n\n    Returns\n    -------\n    w : ndarray\n       The weight function at `x`.\n\n    Notes\n    -----\n\n    .. versionadded::1.7.0\n\n    ')
    
    # Assigning a Call to a Name (line 1804):
    
    # Assigning a Call to a Name (line 1804):
    
    # Call to exp(...): (line 1804)
    # Processing the call arguments (line 1804)
    float_170677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1804, 15), 'float')
    # Getting the type of 'x' (line 1804)
    x_170678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1804, 19), 'x', False)
    int_170679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1804, 22), 'int')
    # Applying the binary operator '**' (line 1804)
    result_pow_170680 = python_operator(stypy.reporting.localization.Localization(__file__, 1804, 19), '**', x_170678, int_170679)
    
    # Applying the binary operator '*' (line 1804)
    result_mul_170681 = python_operator(stypy.reporting.localization.Localization(__file__, 1804, 15), '*', float_170677, result_pow_170680)
    
    # Processing the call keyword arguments (line 1804)
    kwargs_170682 = {}
    # Getting the type of 'np' (line 1804)
    np_170675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1804, 8), 'np', False)
    # Obtaining the member 'exp' of a type (line 1804)
    exp_170676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1804, 8), np_170675, 'exp')
    # Calling exp(args, kwargs) (line 1804)
    exp_call_result_170683 = invoke(stypy.reporting.localization.Localization(__file__, 1804, 8), exp_170676, *[result_mul_170681], **kwargs_170682)
    
    # Assigning a type to the variable 'w' (line 1804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1804, 4), 'w', exp_call_result_170683)
    # Getting the type of 'w' (line 1805)
    w_170684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1805, 11), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1805, 4), 'stypy_return_type', w_170684)
    
    # ################# End of 'hermeweight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hermeweight' in the type store
    # Getting the type of 'stypy_return_type' (line 1781)
    stypy_return_type_170685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1781, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170685)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hermeweight'
    return stypy_return_type_170685

# Assigning a type to the variable 'hermeweight' (line 1781)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1781, 0), 'hermeweight', hermeweight)
# Declaration of the 'HermiteE' class
# Getting the type of 'ABCPolyBase' (line 1812)
ABCPolyBase_170686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1812, 15), 'ABCPolyBase')

class HermiteE(ABCPolyBase_170686, ):
    str_170687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1833, (-1)), 'str', "An HermiteE series class.\n\n    The HermiteE class provides the standard Python numerical methods\n    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the\n    attributes and methods listed in the `ABCPolyBase` documentation.\n\n    Parameters\n    ----------\n    coef : array_like\n        HermiteE coefficients in order of increasing degree, i.e,\n        ``(1, 2, 3)`` gives ``1*He_0(x) + 2*He_1(X) + 3*He_2(x)``.\n    domain : (2,) array_like, optional\n        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped\n        to the interval ``[window[0], window[1]]`` by shifting and scaling.\n        The default value is [-1, 1].\n    window : (2,) array_like, optional\n        Window, see `domain` for its use. The default value is [-1, 1].\n\n        .. versionadded:: 1.6.0\n\n    ")
    
    # Assigning a Call to a Name (line 1835):
    
    # Assigning a Call to a Name (line 1836):
    
    # Assigning a Call to a Name (line 1837):
    
    # Assigning a Call to a Name (line 1838):
    
    # Assigning a Call to a Name (line 1839):
    
    # Assigning a Call to a Name (line 1840):
    
    # Assigning a Call to a Name (line 1841):
    
    # Assigning a Call to a Name (line 1842):
    
    # Assigning a Call to a Name (line 1843):
    
    # Assigning a Call to a Name (line 1844):
    
    # Assigning a Call to a Name (line 1845):
    
    # Assigning a Call to a Name (line 1846):
    
    # Assigning a Str to a Name (line 1849):
    
    # Assigning a Call to a Name (line 1850):
    
    # Assigning a Call to a Name (line 1851):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1812, 0, False)
        # Assigning a type to the variable 'self' (line 1813)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1813, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HermiteE.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HermiteE' (line 1812)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1812, 0), 'HermiteE', HermiteE)

# Assigning a Call to a Name (line 1835):

# Call to staticmethod(...): (line 1835)
# Processing the call arguments (line 1835)
# Getting the type of 'hermeadd' (line 1835)
hermeadd_170689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 24), 'hermeadd', False)
# Processing the call keyword arguments (line 1835)
kwargs_170690 = {}
# Getting the type of 'staticmethod' (line 1835)
staticmethod_170688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1835, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1835)
staticmethod_call_result_170691 = invoke(stypy.reporting.localization.Localization(__file__, 1835, 11), staticmethod_170688, *[hermeadd_170689], **kwargs_170690)

# Getting the type of 'HermiteE'
HermiteE_170692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_add' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170692, '_add', staticmethod_call_result_170691)

# Assigning a Call to a Name (line 1836):

# Call to staticmethod(...): (line 1836)
# Processing the call arguments (line 1836)
# Getting the type of 'hermesub' (line 1836)
hermesub_170694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1836, 24), 'hermesub', False)
# Processing the call keyword arguments (line 1836)
kwargs_170695 = {}
# Getting the type of 'staticmethod' (line 1836)
staticmethod_170693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1836, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1836)
staticmethod_call_result_170696 = invoke(stypy.reporting.localization.Localization(__file__, 1836, 11), staticmethod_170693, *[hermesub_170694], **kwargs_170695)

# Getting the type of 'HermiteE'
HermiteE_170697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_sub' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170697, '_sub', staticmethod_call_result_170696)

# Assigning a Call to a Name (line 1837):

# Call to staticmethod(...): (line 1837)
# Processing the call arguments (line 1837)
# Getting the type of 'hermemul' (line 1837)
hermemul_170699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 24), 'hermemul', False)
# Processing the call keyword arguments (line 1837)
kwargs_170700 = {}
# Getting the type of 'staticmethod' (line 1837)
staticmethod_170698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1837, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1837)
staticmethod_call_result_170701 = invoke(stypy.reporting.localization.Localization(__file__, 1837, 11), staticmethod_170698, *[hermemul_170699], **kwargs_170700)

# Getting the type of 'HermiteE'
HermiteE_170702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_mul' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170702, '_mul', staticmethod_call_result_170701)

# Assigning a Call to a Name (line 1838):

# Call to staticmethod(...): (line 1838)
# Processing the call arguments (line 1838)
# Getting the type of 'hermediv' (line 1838)
hermediv_170704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 24), 'hermediv', False)
# Processing the call keyword arguments (line 1838)
kwargs_170705 = {}
# Getting the type of 'staticmethod' (line 1838)
staticmethod_170703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1838, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1838)
staticmethod_call_result_170706 = invoke(stypy.reporting.localization.Localization(__file__, 1838, 11), staticmethod_170703, *[hermediv_170704], **kwargs_170705)

# Getting the type of 'HermiteE'
HermiteE_170707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_div' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170707, '_div', staticmethod_call_result_170706)

# Assigning a Call to a Name (line 1839):

# Call to staticmethod(...): (line 1839)
# Processing the call arguments (line 1839)
# Getting the type of 'hermepow' (line 1839)
hermepow_170709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 24), 'hermepow', False)
# Processing the call keyword arguments (line 1839)
kwargs_170710 = {}
# Getting the type of 'staticmethod' (line 1839)
staticmethod_170708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1839, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1839)
staticmethod_call_result_170711 = invoke(stypy.reporting.localization.Localization(__file__, 1839, 11), staticmethod_170708, *[hermepow_170709], **kwargs_170710)

# Getting the type of 'HermiteE'
HermiteE_170712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_pow' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170712, '_pow', staticmethod_call_result_170711)

# Assigning a Call to a Name (line 1840):

# Call to staticmethod(...): (line 1840)
# Processing the call arguments (line 1840)
# Getting the type of 'hermeval' (line 1840)
hermeval_170714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 24), 'hermeval', False)
# Processing the call keyword arguments (line 1840)
kwargs_170715 = {}
# Getting the type of 'staticmethod' (line 1840)
staticmethod_170713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1840, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1840)
staticmethod_call_result_170716 = invoke(stypy.reporting.localization.Localization(__file__, 1840, 11), staticmethod_170713, *[hermeval_170714], **kwargs_170715)

# Getting the type of 'HermiteE'
HermiteE_170717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_val' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170717, '_val', staticmethod_call_result_170716)

# Assigning a Call to a Name (line 1841):

# Call to staticmethod(...): (line 1841)
# Processing the call arguments (line 1841)
# Getting the type of 'hermeint' (line 1841)
hermeint_170719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 24), 'hermeint', False)
# Processing the call keyword arguments (line 1841)
kwargs_170720 = {}
# Getting the type of 'staticmethod' (line 1841)
staticmethod_170718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1841, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1841)
staticmethod_call_result_170721 = invoke(stypy.reporting.localization.Localization(__file__, 1841, 11), staticmethod_170718, *[hermeint_170719], **kwargs_170720)

# Getting the type of 'HermiteE'
HermiteE_170722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_int' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170722, '_int', staticmethod_call_result_170721)

# Assigning a Call to a Name (line 1842):

# Call to staticmethod(...): (line 1842)
# Processing the call arguments (line 1842)
# Getting the type of 'hermeder' (line 1842)
hermeder_170724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 24), 'hermeder', False)
# Processing the call keyword arguments (line 1842)
kwargs_170725 = {}
# Getting the type of 'staticmethod' (line 1842)
staticmethod_170723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1842, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1842)
staticmethod_call_result_170726 = invoke(stypy.reporting.localization.Localization(__file__, 1842, 11), staticmethod_170723, *[hermeder_170724], **kwargs_170725)

# Getting the type of 'HermiteE'
HermiteE_170727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_der' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170727, '_der', staticmethod_call_result_170726)

# Assigning a Call to a Name (line 1843):

# Call to staticmethod(...): (line 1843)
# Processing the call arguments (line 1843)
# Getting the type of 'hermefit' (line 1843)
hermefit_170729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 24), 'hermefit', False)
# Processing the call keyword arguments (line 1843)
kwargs_170730 = {}
# Getting the type of 'staticmethod' (line 1843)
staticmethod_170728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1843, 11), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1843)
staticmethod_call_result_170731 = invoke(stypy.reporting.localization.Localization(__file__, 1843, 11), staticmethod_170728, *[hermefit_170729], **kwargs_170730)

# Getting the type of 'HermiteE'
HermiteE_170732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_fit' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170732, '_fit', staticmethod_call_result_170731)

# Assigning a Call to a Name (line 1844):

# Call to staticmethod(...): (line 1844)
# Processing the call arguments (line 1844)
# Getting the type of 'hermeline' (line 1844)
hermeline_170734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 25), 'hermeline', False)
# Processing the call keyword arguments (line 1844)
kwargs_170735 = {}
# Getting the type of 'staticmethod' (line 1844)
staticmethod_170733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1844, 12), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1844)
staticmethod_call_result_170736 = invoke(stypy.reporting.localization.Localization(__file__, 1844, 12), staticmethod_170733, *[hermeline_170734], **kwargs_170735)

# Getting the type of 'HermiteE'
HermiteE_170737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_line' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170737, '_line', staticmethod_call_result_170736)

# Assigning a Call to a Name (line 1845):

# Call to staticmethod(...): (line 1845)
# Processing the call arguments (line 1845)
# Getting the type of 'hermeroots' (line 1845)
hermeroots_170739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1845, 26), 'hermeroots', False)
# Processing the call keyword arguments (line 1845)
kwargs_170740 = {}
# Getting the type of 'staticmethod' (line 1845)
staticmethod_170738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1845, 13), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1845)
staticmethod_call_result_170741 = invoke(stypy.reporting.localization.Localization(__file__, 1845, 13), staticmethod_170738, *[hermeroots_170739], **kwargs_170740)

# Getting the type of 'HermiteE'
HermiteE_170742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_roots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170742, '_roots', staticmethod_call_result_170741)

# Assigning a Call to a Name (line 1846):

# Call to staticmethod(...): (line 1846)
# Processing the call arguments (line 1846)
# Getting the type of 'hermefromroots' (line 1846)
hermefromroots_170744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1846, 30), 'hermefromroots', False)
# Processing the call keyword arguments (line 1846)
kwargs_170745 = {}
# Getting the type of 'staticmethod' (line 1846)
staticmethod_170743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1846, 17), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 1846)
staticmethod_call_result_170746 = invoke(stypy.reporting.localization.Localization(__file__, 1846, 17), staticmethod_170743, *[hermefromroots_170744], **kwargs_170745)

# Getting the type of 'HermiteE'
HermiteE_170747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member '_fromroots' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170747, '_fromroots', staticmethod_call_result_170746)

# Assigning a Str to a Name (line 1849):
str_170748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1849, 15), 'str', 'herme')
# Getting the type of 'HermiteE'
HermiteE_170749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member 'nickname' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170749, 'nickname', str_170748)

# Assigning a Call to a Name (line 1850):

# Call to array(...): (line 1850)
# Processing the call arguments (line 1850)
# Getting the type of 'hermedomain' (line 1850)
hermedomain_170752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1850, 22), 'hermedomain', False)
# Processing the call keyword arguments (line 1850)
kwargs_170753 = {}
# Getting the type of 'np' (line 1850)
np_170750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1850, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1850)
array_170751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1850, 13), np_170750, 'array')
# Calling array(args, kwargs) (line 1850)
array_call_result_170754 = invoke(stypy.reporting.localization.Localization(__file__, 1850, 13), array_170751, *[hermedomain_170752], **kwargs_170753)

# Getting the type of 'HermiteE'
HermiteE_170755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member 'domain' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170755, 'domain', array_call_result_170754)

# Assigning a Call to a Name (line 1851):

# Call to array(...): (line 1851)
# Processing the call arguments (line 1851)
# Getting the type of 'hermedomain' (line 1851)
hermedomain_170758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1851, 22), 'hermedomain', False)
# Processing the call keyword arguments (line 1851)
kwargs_170759 = {}
# Getting the type of 'np' (line 1851)
np_170756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1851, 13), 'np', False)
# Obtaining the member 'array' of a type (line 1851)
array_170757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1851, 13), np_170756, 'array')
# Calling array(args, kwargs) (line 1851)
array_call_result_170760 = invoke(stypy.reporting.localization.Localization(__file__, 1851, 13), array_170757, *[hermedomain_170758], **kwargs_170759)

# Getting the type of 'HermiteE'
HermiteE_170761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HermiteE')
# Setting the type of the member 'window' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HermiteE_170761, 'window', array_call_result_170760)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
